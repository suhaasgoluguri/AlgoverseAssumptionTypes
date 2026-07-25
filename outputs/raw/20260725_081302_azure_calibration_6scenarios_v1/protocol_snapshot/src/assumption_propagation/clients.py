"""Model clients: Azure OpenAI (production) and a mock (tests/dev).

Both expose the same async interface:

    raw_text = await client.generate(system_prompt, user_prompt)

so the orchestrator never knows which one it is talking to. All JSON
parsing/repair lives here too: strict parse -> one repair attempt ->
recorded failure. Failures are never silently discarded (they would bias
the sample); they are returned as ParsedResponse with parse_failed=True.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .schemas import AgentOutput

load_dotenv()

RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)

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

class AzureChatClient:
    """Async client for the team's Azure OpenAI deployment (GPT-5.4 mini).

    Credentials come from environment:
      AZURE_OPENAI_API_KEY   — the resource key
      AZURE_OPENAI_ENDPOINT  — the v1 base URL, ending in /openai/v1
                               e.g. https://<resource>.services.ai.azure.com/openai/v1
      AZURE_OPENAI_DEPLOYMENT — the deployment name (used as the model id)

    Uses the OpenAI-compatible v1 endpoint (Azure AI Foundry) rather than the
    classic azure_endpoint + api_version routing, which 404s on Foundry
    resources.
    """

    def __init__(
        self,
        deployment: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        max_tokens: int = 400,
    ) -> None:
        self.deployment = deployment or os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

    @property
    def model_identifier(self) -> str:
        return f"azure:{self.deployment}"

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """One completion. Retries transient errors with exponential backoff."""
        kwargs: dict = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        # Newer reasoning-capable deployments reject custom temperatures;
        # pass None to omit and record the deployment default in logs.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


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
    model_identifier: str = "mock:test"
    temperature: Optional[float] = None  # mock samples nothing; log honestly

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            return json.dumps({"answer": "B", "rationale": "Mock default rationale."})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]
