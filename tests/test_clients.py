"""Tests for JSON parse-and-repair and the mock client.

The parser is part of the scientific protocol: exactly one syntax-only
repair attempt, and no response is ever silently discarded.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import APIConnectionError
from tenacity import wait_none

from assumption_propagation.clients import (
    AzureChatClient,
    ForcedToolProtocolError,
    MockClient,
    ToolExecutionResult,
    audit_answer_rationale,
    parse_or_repair,
)


CASE_RECORD_TOOL = {
    "type": "function",
    "function": {
        "name": "read_case_record",
        "description": "Read the case record available to this agent.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _tool_call(
    *,
    call_id: str = "call_record_123",
    name: str = "read_case_record",
    arguments: str = "{}",
):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(
    *,
    content: str | None,
    finish_reason: str,
    tool_calls=None,
    response_id: str = "completion_1",
):
    return SimpleNamespace(
        id=response_id,
        model="gpt-5-mini",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
        ),
    )


class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _azure_client(outcomes) -> tuple[AzureChatClient, _FakeCompletions]:
    completions = _FakeCompletions(outcomes)
    client = AzureChatClient.__new__(AzureChatClient)
    client.deployment = "gpt-5.4-mini"
    client.temperature = None
    client.max_tokens = 1000
    client._retry_wait = wait_none()
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def test_azure_client_has_one_bounded_visible_retry_layer(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.test/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

    with patch("assumption_propagation.clients.AsyncOpenAI") as async_openai:
        client = AzureChatClient(temperature=None)

    assert client.request_timeout_seconds == 120.0
    assert client.sdk_max_retries == 0
    assert client.TENACITY_MAX_ATTEMPTS == 6
    async_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.test/openai/v1",
        timeout=120.0,
        max_retries=0,
    )


def test_clean_json_parses():
    r = parse_or_repair('{"answer": "B", "rationale": "H3 is confirmed accessible."}')
    assert not r.parse_failed and not r.parse_repaired
    assert r.answer == "B"


def test_lowercase_answer_normalized():
    r = parse_or_repair('{"answer": "b", "rationale": "ok reasoning here"}')
    assert r.answer == "B"


def test_markdown_fences_stripped_without_counting_as_repair():
    raw = '```json\n{"answer": "A", "rationale": "Fenced response."}\n```'
    r = parse_or_repair(raw)
    assert r.answer == "A"
    assert not r.parse_repaired  # cosmetic, not a repair


def test_json_embedded_in_prose_is_repaired():
    raw = 'Sure! Here is my decision:\n{"answer": "C", "rationale": "Embedded."}\nHope that helps.'
    r = parse_or_repair(raw)
    assert r.answer == "C"
    assert r.parse_repaired


def test_invalid_answer_letter_is_failure_not_repair():
    r = parse_or_repair('{"answer": "F", "rationale": "No such option."}')
    assert r.parse_failed and r.answer is None
    assert r.raw_response  # raw text preserved for the audit trail


def test_missing_rationale_is_failure():
    r = parse_or_repair('{"answer": "A"}')
    assert r.parse_failed


def test_unparseable_text_is_recorded_failure():
    r = parse_or_repair("I think option B is best because it satisfies the policy.")
    assert r.parse_failed
    assert r.raw_response.startswith("I think")


def test_extra_keys_rejected():
    """AgentOutput forbids extra keys — agents must not smuggle fields."""
    r = parse_or_repair('{"answer": "A", "rationale": "ok", "confidence": 0.9}')
    assert r.parse_failed


def test_blank_rationale_is_failure():
    r = parse_or_repair('{"answer": "A", "rationale": "   "}')
    assert r.parse_failed


def test_answer_rationale_contradiction_is_flagged_without_repair():
    raw = '{"answer": "A", "rationale": "Therefore, the correct answer is option B."}'
    r = parse_or_repair(raw)
    assert r.answer == "A"
    assert r.rationale_conclusion_letters == ["B"]
    assert r.answer_rationale_inconsistent is True
    assert r.raw_response == raw
    assert not r.parse_repaired and not r.parse_failed


def test_matching_rationale_conclusion_is_not_flagged():
    r = parse_or_repair(
        '{"answer": "C", "rationale": "I choose option C because it meets both rules."}'
    )
    assert r.rationale_conclusion_letters == ["C"]
    assert r.answer_rationale_inconsistent is False


def test_option_mention_without_conclusion_is_not_flagged():
    r = parse_or_repair(
        '{"answer": "D", "rationale": "Option B exceeds the cap, so D remains feasible."}'
    )
    assert r.rationale_conclusion_letters == []
    assert r.answer_rationale_inconsistent is False


def test_final_revised_conclusion_governs_consistency_audit():
    audit = audit_answer_rationale(
        "B",
        "Option A is the best. After applying the cap, I choose option B.",
    )
    assert audit.conclusion_letters == ["A", "B"]
    assert audit.inconsistent is False


def test_negated_and_conditional_choices_are_not_treated_as_conclusions():
    negated = audit_answer_rationale(
        "B", "Option A is not the best; option B is the best."
    )
    conditional = audit_answer_rationale(
        "B", "Choose option A only if the cap is waived. I choose option B."
    )
    assert negated.conclusion_letters == ["B"]
    assert conditional.conclusion_letters == ["B"]
    assert negated.inconsistent is False
    assert conditional.inconsistent is False


def test_named_option_conclusion_is_mapped_to_stable_option_letter():
    options = {
        "A": "Company H, preferred, $110/day",
        "B": "Company G, preferred, $80/day",
        "C": "Company K, non-preferred, $90/day",
        "D": "Company J, preferred, $95/day",
        "E": "Company L, preferred, $130/day",
    }
    audit = audit_answer_rationale(
        "A",
        "Company J is the lowest-cost compliant choice, so I recommend Company J.",
        options,
    )
    assert audit.conclusion_letters[-1] == "D"
    assert audit.conclusion_sources[-1] == "option_entity"
    assert audit.inconsistent is True


def test_negated_named_option_is_not_mapped_as_a_selection():
    options = {
        "A": "Company H, preferred, $110/day",
        "B": "Company G, preferred, $80/day",
        "C": "Company K, non-preferred, $90/day",
        "D": "Company J, preferred, $95/day",
        "E": "Company L, preferred, $130/day",
    }
    audit = audit_answer_rationale(
        "A", "I do not recommend Company J. Company H is the best choice.", options
    )
    assert audit.conclusion_letters == ["A"]
    assert audit.inconsistent is False


async def test_mock_client_returns_scripted_responses_in_order():
    mock = MockClient(responses=[
        json.dumps({"answer": "A", "rationale": "first"}),
        json.dumps({"answer": "B", "rationale": "second"}),
    ])
    first = await mock.generate("sys", "user1")
    second = await mock.generate("sys", "user2")
    third = await mock.generate("sys", "user3")  # last response repeats
    assert '"A"' in first and '"B"' in second and '"B"' in third
    assert len(mock.calls) == 3
    assert mock.calls[1] == ("sys", "user2")


async def test_mock_client_default_is_valid():
    mock = MockClient()
    r = parse_or_repair(await mock.generate("s", "u"))
    assert not r.parse_failed


async def test_forced_tool_runs_two_call_protocol_and_returns_auditable_trace():
    tool_request = _completion(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[_tool_call()],
        response_id="request_completion",
    )
    final_text = '{"answer":"B","rationale":"The record disqualifies A."}'
    final_response = _completion(
        content=final_text,
        finish_reason="stop",
        response_id="final_completion",
    )
    client, completions = _azure_client([tool_request, final_response])
    executor_arguments = []

    def execute(arguments):
        executor_arguments.append(arguments)
        return ToolExecutionResult(
            content='{"record_id":"REC-123","content":"A is disqualified."}',
            metadata={"recipient_agent_id": 2, "retrieval_round": 2},
        )

    result = await client.generate_with_forced_tool(
        "system prompt",
        "base user prompt",
        tool=CASE_RECORD_TOOL,
        execute=execute,
    )

    assert result.final_text == final_text
    assert executor_arguments == [{}]
    assert len(completions.calls) == 2

    request = completions.calls[0]
    assert request["tool_choice"] == {
        "type": "function",
        "function": {"name": "read_case_record"},
    }
    assert request["parallel_tool_calls"] is False
    assert "response_format" not in request

    final = completions.calls[1]
    assert final["tool_choice"] == "none"
    assert final["response_format"] == {"type": "json_object"}
    assert [message["role"] for message in final["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assistant_message = final["messages"][2]
    tool_message = final["messages"][3]
    assert assistant_message["tool_calls"][0]["id"] == "call_record_123"
    assert tool_message["tool_call_id"] == "call_record_123"
    assert tool_message["content"] == result.trace.tool_result

    assert result.trace.tool_name == "read_case_record"
    assert result.trace.arguments_json == "{}"
    assert result.trace.request.finish_reason == "tool_calls"
    assert result.trace.final.finish_reason == "stop"
    assert result.trace.request.reasoning_tokens == 10
    assert result.trace.api_call_attempts == 2
    assert result.trace.execution_metadata == {
        "recipient_agent_id": 2,
        "retrieval_round": 2,
    }
    assert len(result.trace.tool_definition_sha256) == 64
    assert len(result.trace.arguments_sha256) == 64
    assert len(result.trace.tool_result_sha256) == 64


@pytest.mark.parametrize(
    ("tool_calls", "match"),
    [
        (None, "exactly one call; got 0"),
        ([_tool_call(call_id="one"), _tool_call(call_id="two")], "got 2"),
        ([_tool_call(name="wrong_tool")], "expected 'read_case_record'"),
        ([_tool_call(arguments="not-json")], "arguments are not valid JSON"),
        ([_tool_call(arguments="[]")], "arguments must decode to an object"),
    ],
)
async def test_forced_tool_rejects_missing_multiple_wrong_or_malformed_calls(
    tool_calls, match
):
    client, completions = _azure_client(
        [
            _completion(
                content=None,
                finish_reason="tool_calls",
                tool_calls=tool_calls,
            )
        ]
    )
    executor_calls = []

    with pytest.raises(ForcedToolProtocolError, match=match):
        await client.generate_with_forced_tool(
            "system",
            "user",
            tool=CASE_RECORD_TOOL,
            execute=lambda arguments: executor_calls.append(arguments) or "result",
        )

    assert executor_calls == []
    assert len(completions.calls) == 1


async def test_forced_tool_rejects_non_tool_finish_before_execution():
    client, _ = _azure_client(
        [
            _completion(
                content="I answered instead.",
                finish_reason="stop",
                tool_calls=[_tool_call()],
            )
        ]
    )
    executor_calls = []
    with pytest.raises(ForcedToolProtocolError, match="finish_reason='tool_calls'"):
        await client.generate_with_forced_tool(
            "system",
            "user",
            tool=CASE_RECORD_TOOL,
            execute=lambda arguments: executor_calls.append(arguments) or "result",
        )
    assert executor_calls == []


async def test_final_completion_retry_does_not_repeat_tool_or_callback():
    tool_request = _completion(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[_tool_call()],
    )
    transient = APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid/chat/completions")
    )
    final_response = _completion(
        content='{"answer":"B","rationale":"Recovered after retry."}',
        finish_reason="stop",
    )
    client, completions = _azure_client([tool_request, transient, final_response])
    executor_calls = []

    result = await client.generate_with_forced_tool(
        "system",
        "user",
        tool=CASE_RECORD_TOOL,
        execute=lambda arguments: executor_calls.append(arguments) or "record-result",
    )

    assert executor_calls == [{}]
    assert len(completions.calls) == 3
    assert completions.calls[1] == completions.calls[2]
    assert result.trace.request.attempts == 1
    assert result.trace.final.attempts == 2
    assert result.trace.api_call_attempts == 3


async def test_initial_completion_retry_executes_callback_only_after_success():
    transient = APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid/chat/completions")
    )
    tool_request = _completion(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[_tool_call()],
    )
    final_response = _completion(
        content='{"answer":"B","rationale":"Successful."}',
        finish_reason="stop",
    )
    client, completions = _azure_client([transient, tool_request, final_response])
    executor_calls = []

    result = await client.generate_with_forced_tool(
        "system",
        "user",
        tool=CASE_RECORD_TOOL,
        execute=lambda arguments: executor_calls.append(arguments) or "record-result",
    )

    assert executor_calls == [{}]
    assert len(completions.calls) == 3
    assert result.trace.request.attempts == 2
    assert result.trace.final.attempts == 1


async def test_mock_client_exposes_matching_forced_tool_interface():
    mock = MockClient()
    executor_calls = []
    result = await mock.generate_with_forced_tool(
        "system",
        "user",
        tool=CASE_RECORD_TOOL,
        execute=lambda arguments: executor_calls.append(arguments) or "mock-result",
    )
    assert executor_calls == [{}]
    assert parse_or_repair(result.final_text).answer == "B"
    assert result.trace.tool_result == "mock-result"
    assert result.trace.api_call_attempts == 2
    assert mock.forced_tool_calls[0]["tool_name"] == "read_case_record"
