"""Tests for JSON parse-and-repair and the mock client.

The parser is part of the scientific protocol: exactly one syntax-only
repair attempt, and no response is ever silently discarded.
"""

import json

from assumption_propagation.clients import (
    MockClient,
    audit_answer_rationale,
    parse_or_repair,
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
