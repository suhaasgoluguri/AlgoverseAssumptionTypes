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
from typing import Optional

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

    @property
    def answer(self) -> Optional[str]:
        return self.output.answer if self.output else None

    @property
    def rationale(self) -> Optional[str]:
        return self.output.rationale if self.output else None


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
        return ParsedResponse(raw_response=raw, output=AgentOutput.model_validate(json.loads(candidate)))
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
