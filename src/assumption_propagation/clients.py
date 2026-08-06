"""Model clients: Azure OpenAI (production) and a mock (tests/dev).

Both expose the same async interface:

    raw_text = await client.generate(system_prompt, user_prompt)

so the orchestrator never knows which one it is talking to. All JSON
parsing/repair lives here too: strict parse -> one repair attempt ->
recorded failure. Failures are never silently discarded (they would bias
the sample); they are returned as ParsedResponse with parse_failed=True.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .schemas import AgentOutput

load_dotenv()

RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


@dataclass(frozen=True)
class CompletionCallTrace:
    """Auditable metadata for one logical Chat Completions request.

    ``attempts`` includes transport-level retries. The response body is not
    duplicated here; callers retain the final model text separately.
    """

    response_id: Optional[str]
    model: Optional[str]
    finish_reason: Optional[str]
    attempts: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass(frozen=True)
class ToolExecutionResult:
    """Application-side tool output and JSON-serializable receipt metadata."""

    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForcedToolTrace:
    """Complete provenance for a forced function-tool exchange."""

    tool_name: str
    tool_call_id: str
    tool_definition_sha256: str
    arguments_json: str
    arguments_sha256: str
    tool_result: str
    tool_result_sha256: str
    execution_metadata: Mapping[str, Any]
    request: CompletionCallTrace
    final: CompletionCallTrace

    @property
    def api_call_attempts(self) -> int:
        """Total HTTP attempts, including retries, across both completions."""
        return self.request.attempts + self.final.attempts


@dataclass(frozen=True)
class ToolTurnResult:
    """Final model text plus the tool exchange needed to audit its origin."""

    final_text: str
    trace: ForcedToolTrace


class ForcedToolProtocolError(ValueError):
    """The model or caller violated the required one-tool protocol."""


@dataclass(frozen=True)
class _CompletionResult:
    response: Any
    attempts: int

_CONCLUSION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:correct|best|final|recommended|lowest-cost)\s+"
        r"(?:answer|choice|option|selection)?\s*(?:is|:)\s*(?:option\s*)?([A-E])\b",
        r"\b(?:therefore|thus|hence|so)[,:\s]+(?:the\s+)?(?:correct\s+)?"
        r"(?:answer|choice|option|selection)\s*(?:is|:)?\s*(?:option\s*)?([A-E])\b",
        r"\b(?:choose|select)\s+(?:option\s+)?([A-E])\b",
        r"\b([A-E])\s+is\s+(?:actually\s+)?(?:the\s+)?"
        r"(?:correct|best|lowest-cost|recommended)\b",
    )
)

RATIONALE_AUDIT_VERSION = "v2.0"

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_ENTITY_STOPWORDS = {
    "a", "an", "and", "at", "book", "choose", "company", "for", "from",
    "in", "is", "itinerary", "non", "of", "on", "option", "per", "preferred",
    "provider", "provision", "rate", "select", "service", "the", "tier", "to",
    "use", "vendor", "via", "with",
}
_ENTITY_CONCLUSION_CUE = re.compile(
    r"\b(?:best|book|choose|correct|lowest-cost|recommend(?:ed)?|right|select|"
    r"should\s+govern|qualif(?:y|ies))\b",
    re.IGNORECASE,
)


def _match_is_nonconclusive(text: str, start: int, end: int) -> bool:
    """Filter explicit-letter mentions that are negated or conditional."""
    prefix = text[max(0, start - 55):start]
    suffix = text[end:end + 35]
    if re.search(
        r"(?:\b(?:do|would|should|could)\s+not|\bnot|\bnever|\brather\s+than)\s*$",
        prefix,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:if|unless|whether)(?:\s+\w+){0,4}\s*$",
        prefix,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.match(r"\s*(?:only\s+if|if|provided\s+that|assuming)\b", suffix, re.IGNORECASE)
    )


def rationale_conclusion_letters(rationale: str | None) -> list[str]:
    """Extract option letters explicitly presented as the conclusion.

    This is a deliberately conservative audit heuristic, not a semantic
    repair. It catches direct answer-letter contradictions while leaving the
    original model response untouched.
    """
    if not rationale:
        return []
    matches: list[tuple[int, str]] = []
    for pattern in _CONCLUSION_PATTERNS:
        for match in pattern.finditer(rationale):
            if not _match_is_nonconclusive(rationale, match.start(), match.end()):
                matches.append((match.start(), match.group(1).upper()))
    found: list[str] = []
    for _, letter in sorted(matches):
        if not found or found[-1] != letter:
            found.append(letter)
    return found


@dataclass(frozen=True)
class RationaleConsistencyAudit:
    conclusion_letters: list[str]
    conclusion_sources: list[str]
    inconsistent: Optional[bool]


def _option_entity_conclusions(
    rationale: str,
    options: Mapping[str, str],
) -> list[tuple[int, str]]:
    option_tokens = {
        letter: {
            token.lower()
            for token in _TOKEN_PATTERN.findall(text)
            if token.lower() not in _ENTITY_STOPWORDS
        }
        for letter, text in options.items()
    }
    frequencies: dict[str, int] = {}
    for tokens in option_tokens.values():
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
    distinctive = {
        letter: {token for token in tokens if frequencies[token] == 1}
        for letter, tokens in option_tokens.items()
    }

    conclusions: list[tuple[int, str]] = []
    for clause in re.finditer(r"[^.!?;]+", rationale):
        text = clause.group(0)
        if not _ENTITY_CONCLUSION_CUE.search(text):
            continue
        if re.search(r"\b(?:if|unless|whether)\b", text, re.IGNORECASE):
            # A named option inside a hypothetical is not the writer's final
            # selection, even when the clause contains a choice verb.
            continue
        if re.search(
            r"\b(?:do(?:es)?|would|should|could|can)?\s*not\s+"
            r"(?:book|choose|recommend|select)\b|"
            r"\bnot\b.{0,18}\b(?:best|correct|lowest-cost|recommended|"
            r"qualif(?:y|ies)|should\s+govern)\b",
            text,
            re.IGNORECASE,
        ):
            continue
        tokens = {token.lower() for token in _TOKEN_PATTERN.findall(text)}
        scores = {
            letter: len(tokens & unique_tokens)
            for letter, unique_tokens in distinctive.items()
        }
        best = max(scores.values(), default=0)
        winners = [letter for letter, score in scores.items() if score == best and score > 0]
        if len(winners) == 1:
            conclusions.append((clause.start(), winners[0]))
    return conclusions


def audit_answer_rationale(
    answer: str | None,
    rationale: str | None,
    options: Mapping[str, str] | None = None,
) -> RationaleConsistencyAudit:
    """Conservatively audit the final unnegated letter/entity conclusion."""
    if answer is None or not rationale:
        return RationaleConsistencyAudit([], [], None)

    events: list[tuple[int, str, str]] = []
    for pattern in _CONCLUSION_PATTERNS:
        for match in pattern.finditer(rationale):
            if not _match_is_nonconclusive(rationale, match.start(), match.end()):
                events.append((match.start(), match.group(1).upper(), "explicit_letter"))
    if options:
        events.extend(
            (position, letter, "option_entity")
            for position, letter in _option_entity_conclusions(rationale, options)
        )
    events.sort(key=lambda item: item[0])

    letters: list[str] = []
    sources: list[str] = []
    for _, letter, source in events:
        if letters and letters[-1] == letter:
            if source not in sources[-1].split("+"):
                sources[-1] += "+" + source
            continue
        letters.append(letter)
        sources.append(source)
    inconsistent = (letters[-1] != answer) if letters else False
    return RationaleConsistencyAudit(letters, sources, inconsistent)


# ---------------------------------------------------------------------------
# Parsing (shared by all clients)
# ---------------------------------------------------------------------------

@dataclass
class ParsedResponse:
    """Outcome of parsing one raw model response. Nothing is discarded."""

    raw_response: str
    output: Optional[AgentOutput] = None
    parse_failed: bool = False
    parse_repaired: bool = False
    generation_failed: bool = False
    generation_error_type: Optional[str] = None
    generation_error_message: Optional[str] = None

    @property
    def answer(self) -> Optional[str]:
        return self.output.answer if self.output else None

    @property
    def rationale(self) -> Optional[str]:
        return self.output.rationale if self.output else None

    @property
    def rationale_conclusion_letters(self) -> list[str]:
        return audit_answer_rationale(self.answer, self.rationale).conclusion_letters

    @property
    def answer_rationale_inconsistent(self) -> bool:
        return bool(audit_answer_rationale(self.answer, self.rationale).inconsistent)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences models often wrap JSON in."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def _extract_json_object(text: str) -> Optional[str]:
    """Repair step: find the first balanced {...} block in the text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_or_repair(raw: str) -> ParsedResponse:
    """Strict parse, then ONE syntax-only repair attempt, then recorded failure.

    The repair never invents content: it only strips fences / extracts the
    JSON object from surrounding prose. Semantic problems (invalid answer
    letter, missing keys) are failures, not repair targets.
    """
    # Attempt 1: strict (after fence stripping, which is cosmetic).
    candidate = _strip_code_fences(raw)
    try:
        return ParsedResponse(
            raw_response=raw,
            output=AgentOutput.model_validate(json.loads(candidate)),
        )
    except (json.JSONDecodeError, ValidationError):
        pass

    # Attempt 2 (repair): extract first balanced JSON object.
    extracted = _extract_json_object(candidate)
    if extracted is not None and extracted != candidate:
        try:
            output = AgentOutput.model_validate(json.loads(extracted))
            return ParsedResponse(raw_response=raw, output=output, parse_repaired=True)
        except (json.JSONDecodeError, ValidationError):
            pass

    return ParsedResponse(raw_response=raw, parse_failed=True)


# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    """Stable bytes for hashing tool definitions in the audit trace."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _completion_call_trace(result: _CompletionResult) -> CompletionCallTrace:
    response = result.response
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return CompletionCallTrace(
        response_id=getattr(response, "id", None),
        model=getattr(response, "model", None),
        finish_reason=getattr(choice, "finish_reason", None),
        attempts=result.attempts,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=(
            getattr(usage, "completion_tokens", None) if usage else None
        ),
        reasoning_tokens=(
            getattr(details, "reasoning_tokens", None) if details else None
        ),
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
    )


def _normalise_function_tool(tool: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate and detach one OpenAI-format function-tool definition."""
    if tool.get("type") != "function":
        raise ForcedToolProtocolError("forced tool must have type='function'")
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ForcedToolProtocolError("forced tool requires a function definition")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ForcedToolProtocolError("forced tool requires a non-empty function name")
    # A JSON round-trip both detaches nested caller-owned mappings and rejects
    # definitions the SDK could not serialize in a request.
    try:
        detached = json.loads(json.dumps(tool, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ForcedToolProtocolError("forced tool definition is not JSON serializable") from exc
    return detached, name


def _normalise_tool_execution(
    value: str | ToolExecutionResult,
) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, ToolExecutionResult):
        raise ForcedToolProtocolError(
            "forced tool callback must return a string or ToolExecutionResult"
        )
    if not isinstance(value.content, str):
        raise ForcedToolProtocolError("forced tool result content must be a string")
    if not isinstance(value.metadata, Mapping):
        raise ForcedToolProtocolError("tool execution metadata must be a mapping")
    try:
        metadata = json.loads(json.dumps(value.metadata, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ForcedToolProtocolError(
            "tool execution metadata is not JSON serializable"
        ) from exc
    if not isinstance(metadata, dict):  # defensive: Mapping serialized unusually
        raise ForcedToolProtocolError("tool execution metadata must encode an object")
    return value.content, metadata

class AzureChatClient:
    """Async client for the team's Azure OpenAI deployment.

    Credentials come from environment:
      AZURE_OPENAI_API_KEY   — the resource key
      AZURE_OPENAI_ENDPOINT  — the v1 base URL, ending in /openai/v1
                               e.g. https://<resource>.services.ai.azure.com/openai/v1
      AZURE_OPENAI_DEPLOYMENT — the deployment name (used as the model id)

    Uses the OpenAI-compatible v1 endpoint (Azure AI Foundry) rather than the
    classic azure_endpoint + api_version routing, which 404s on Foundry
    resources.
    """

    _retry_wait = wait_exponential(multiplier=2, min=2, max=60)
    REQUEST_TIMEOUT_SECONDS = 120.0
    SDK_MAX_RETRIES = 0
    TENACITY_MAX_ATTEMPTS = 6

    def __init__(
        self,
        deployment: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        max_tokens: int = 1000,
    ) -> None:
        self.deployment = deployment or os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout_seconds = self.REQUEST_TIMEOUT_SECONDS
        self.sdk_max_retries = self.SDK_MAX_RETRIES
        self._client = AsyncOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
            timeout=self.request_timeout_seconds,
            # Tenacity below is the single visible retry layer. Disabling the
            # SDK's internal retries keeps attempt counts and wall-clock stalls
            # auditable in long confirmatory runs.
            max_retries=self.sdk_max_retries,
        )

    @property
    def model_identifier(self) -> str:
        return f"azure:{self.deployment}"

    async def _create_completion(self, **kwargs: Any) -> _CompletionResult:
        """Run one API request with isolated retries and report attempt count."""
        response = None
        attempts = 0
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(RETRYABLE),
            wait=self._retry_wait,
            stop=stop_after_attempt(self.TENACITY_MAX_ATTEMPTS),
            reraise=True,
        ):
            with attempt:
                attempts = attempt.retry_state.attempt_number
                response = await self._client.chat.completions.create(**kwargs)
        if response is None:  # pragma: no cover - AsyncRetrying either returns or raises
            raise RuntimeError("completion retry loop ended without a response")
        return _CompletionResult(response=response, attempts=attempts)

    def _base_completion_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "max_completion_tokens": self.max_tokens,
        }
        # Newer reasoning-capable deployments reject custom temperatures;
        # pass None to omit and record the deployment default in logs.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """One completion. Retries transient errors with exponential backoff."""
        kwargs = self._base_completion_kwargs()
        kwargs.update({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        })
        result = await self._create_completion(**kwargs)
        return result.response.choices[0].message.content or ""

    async def generate_with_forced_tool(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tool: Mapping[str, Any],
        execute: Callable[[dict[str, Any]], str | ToolExecutionResult],
    ) -> ToolTurnResult:
        """Force one function call, execute it once, then obtain final JSON.

        API retries are scoped to each completion. In particular, once the
        synchronous ``execute`` callback has returned, retrying the final
        completion cannot resample the first call or execute the callback
        again.
        """
        tool_definition, expected_name = _normalise_function_tool(tool)
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request_kwargs = self._base_completion_kwargs()
        request_kwargs.update(
            {
                "messages": initial_messages,
                "tools": [tool_definition],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": expected_name},
                },
                "parallel_tool_calls": False,
            }
        )
        request_result = await self._create_completion(**request_kwargs)
        request_choice = request_result.response.choices[0]
        message = request_choice.message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if getattr(request_choice, "finish_reason", None) != "tool_calls":
            raise ForcedToolProtocolError(
                "forced-tool completion must finish with finish_reason='tool_calls'"
            )
        if len(tool_calls) != 1:
            raise ForcedToolProtocolError(
                f"forced-tool completion must return exactly one call; got {len(tool_calls)}"
            )
        tool_call = tool_calls[0]
        if getattr(tool_call, "type", None) != "function":
            raise ForcedToolProtocolError("forced tool call must have type='function'")
        call_id = getattr(tool_call, "id", None)
        if not isinstance(call_id, str) or not call_id:
            raise ForcedToolProtocolError("forced tool call requires a non-empty id")
        function_call = getattr(tool_call, "function", None)
        actual_name = getattr(function_call, "name", None)
        if actual_name != expected_name:
            raise ForcedToolProtocolError(
                f"forced tool called {actual_name!r}; expected {expected_name!r}"
            )
        arguments_json = getattr(function_call, "arguments", None)
        if not isinstance(arguments_json, str):
            raise ForcedToolProtocolError("forced tool arguments must be a JSON string")
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            raise ForcedToolProtocolError("forced tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ForcedToolProtocolError("forced tool arguments must decode to an object")

        # The callback is deliberately outside either retry loop.
        tool_result, execution_metadata = _normalise_tool_execution(
            execute(arguments)
        )

        assistant_tool_message = {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": actual_name,
                        "arguments": arguments_json,
                    },
                }
            ],
        }
        tool_result_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_result,
        }
        final_kwargs = self._base_completion_kwargs()
        final_kwargs.update(
            {
                "messages": [
                    *initial_messages,
                    assistant_tool_message,
                    tool_result_message,
                ],
                "tools": [tool_definition],
                "tool_choice": "none",
                "parallel_tool_calls": False,
                "response_format": {"type": "json_object"},
            }
        )
        final_result = await self._create_completion(**final_kwargs)
        final_text = final_result.response.choices[0].message.content or ""
        trace = ForcedToolTrace(
            tool_name=expected_name,
            tool_call_id=call_id,
            tool_definition_sha256=_sha256_text(_canonical_json(tool_definition)),
            arguments_json=arguments_json,
            arguments_sha256=_sha256_text(arguments_json),
            tool_result=tool_result,
            tool_result_sha256=_sha256_text(tool_result),
            execution_metadata=execution_metadata,
            request=_completion_call_trace(request_result),
            final=_completion_call_trace(final_result),
        )
        return ToolTurnResult(final_text=final_text, trace=trace)


# ---------------------------------------------------------------------------
# Mock client (tests / free development)
# ---------------------------------------------------------------------------

@dataclass
class MockClient:
    """Scriptable stand-in with the same interface as AzureChatClient.

    Supply `responses` (returned in order, then the last one repeats), or
    leave empty for a fixed valid default. Records every call for
    assertions in tests.
    """

    responses: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    forced_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_identifier: str = "mock:test"
    temperature: Optional[float] = None  # mock samples nothing; log honestly

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            return json.dumps({"answer": "B", "rationale": "Mock default rationale."})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    async def generate_with_forced_tool(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tool: Mapping[str, Any],
        execute: Callable[[dict[str, Any]], str | ToolExecutionResult],
    ) -> ToolTurnResult:
        """Deterministic stand-in for the Azure two-completion tool exchange."""
        tool_definition, tool_name = _normalise_function_tool(tool)
        arguments_json = "{}"
        tool_result, execution_metadata = _normalise_tool_execution(execute({}))
        call_index = len(self.forced_tool_calls)
        call_id = f"mock_call_{call_index:04d}"
        self.forced_tool_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "tool": tool_definition,
                "tool_name": tool_name,
                "tool_call_id": call_id,
                "arguments": {},
                "tool_result": tool_result,
            }
        )
        if not self.responses:
            final_text = json.dumps(
                {"answer": "B", "rationale": "Mock default rationale."}
            )
        else:
            index = min(
                len(self.calls) + len(self.forced_tool_calls) - 1,
                len(self.responses) - 1,
            )
            final_text = self.responses[index]
        request_trace = CompletionCallTrace(
            response_id=f"mock_tool_request_{call_index:04d}",
            model=self.model_identifier,
            finish_reason="tool_calls",
            attempts=1,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            total_tokens=None,
        )
        final_trace = CompletionCallTrace(
            response_id=f"mock_tool_final_{call_index:04d}",
            model=self.model_identifier,
            finish_reason="stop",
            attempts=1,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            total_tokens=None,
        )
        return ToolTurnResult(
            final_text=final_text,
            trace=ForcedToolTrace(
                tool_name=tool_name,
                tool_call_id=call_id,
                tool_definition_sha256=_sha256_text(
                    _canonical_json(tool_definition)
                ),
                arguments_json=arguments_json,
                arguments_sha256=_sha256_text(arguments_json),
                tool_result=tool_result,
                tool_result_sha256=_sha256_text(tool_result),
                execution_metadata=execution_metadata,
                request=request_trace,
                final=final_trace,
            ),
        )
