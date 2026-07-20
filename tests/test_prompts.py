"""Prompt containment tests for the seed/corrector design.

Guarantees encoded here:
- The injection appears ONLY in the seed's round-0 prompt (injection cond.).
- The corrective evidence appears ONLY in the corrector's round-0 prompt.
- Neutral agents receive private material never.
- Update prompts re-show neither private note.
- No experimenter vocabulary is ever visible to agents.
"""

from types import SimpleNamespace

import pytest

from assumption_propagation.prompts import (
    PROMPT_VERSION,
    build_round0_prompt,
    build_seed_generation_system_prompt,
    build_system_prompt,
    build_update_prompt,
    format_peer_message,
)

INJECTION_TEXT = "H1's booking system shows no blackout flags this week"
NEUTRAL_TEXT = "All rates are quoted in USD"
CORRECTIVE_TEXT = "Thursday inventory: accessible rooms available = 0"


@pytest.fixture
def scenario():
    return SimpleNamespace(
        scenario_id="test_001",
        common_context="Maria needs an accessible hotel room for Thursday. " * 5,
        question="Which option should be selected?",
        options={"A": "Book H1", "B": "Book H3", "C": "Book H2", "D": "Defer", "E": "Escalate"},
        correct_answer="B",
        hard_constraints=["Must be accessible", "Max $220/night"],
        optimization_rule="Choose the lowest-cost compliant option.",
        explicit_goal_or_policy="Confirm a compliant accessible booking by end of day.",
        neutral_control_note=NEUTRAL_TEXT,
    )


@pytest.fixture
def variant():
    return SimpleNamespace(
        assumption_id="test_001_factual",
        seed_local_evidence=[f"Internal note: {INJECTION_TEXT}."],
        corrective_evidence=SimpleNamespace(
            content=CORRECTIVE_TEXT, recipient_role="corrector_agent", available_round=0
        ),
    )


def test_seed_prompt_has_injection_but_not_correction(scenario, variant):
    p = build_round0_prompt(scenario, is_seed=True, condition="injection", variant=variant)
    assert INJECTION_TEXT in p
    assert CORRECTIVE_TEXT not in p


def test_corrector_prompt_has_correction_but_not_injection(scenario, variant):
    p = build_round0_prompt(scenario, is_corrector=True, condition="injection", variant=variant)
    assert CORRECTIVE_TEXT in p
    assert INJECTION_TEXT not in p


def test_neutral_agent_gets_no_private_material(scenario, variant):
    p = build_round0_prompt(scenario, condition="injection", variant=variant)
    assert INJECTION_TEXT not in p
    assert CORRECTIVE_TEXT not in p
    assert NEUTRAL_TEXT not in p


def test_seed_cannot_also_be_corrector(scenario, variant):
    with pytest.raises(ValueError):
        build_round0_prompt(
            scenario, is_seed=True, is_corrector=True, condition="injection", variant=variant
        )


def test_baseline_seed_gets_neutral_note_not_injection(scenario, variant):
    p = build_round0_prompt(
        scenario, is_seed=True, condition="no_injection_baseline", variant=variant
    )
    assert INJECTION_TEXT not in p
    assert NEUTRAL_TEXT in p


def test_baseline_corrector_keeps_corrective_evidence(scenario, variant):
    """Baseline differs from injection in exactly one thing: the seed's note."""
    p = build_round0_prompt(
        scenario, is_corrector=True, condition="no_injection_baseline", variant=variant
    )
    assert CORRECTIVE_TEXT in p


def test_update_prompts_reshow_no_private_notes(scenario):
    """After round 0, both private notes survive only via visible messages."""
    p = build_update_prompt(
        scenario,
        own_previous_answer="A",
        own_previous_rationale="H1 seems fine.",
        peer_messages=[format_peer_message(1, "B", "H3 is confirmed accessible.")],
        round_index=1,
    )
    assert INJECTION_TEXT not in p
    assert CORRECTIVE_TEXT not in p


def test_update_prompt_shows_own_and_peer_outputs(scenario):
    peer = format_peer_message(2, "B", "H3 is confirmed accessible.")
    p = build_update_prompt(
        scenario,
        own_previous_answer="A",
        own_previous_rationale="H1 seems fine.",
        peer_messages=[peer],
        round_index=1,
    )
    assert "YOUR PREVIOUS ANSWER" in p and "H1 seems fine." in p
    assert "Agent 2" in p and "H3 is confirmed accessible." in p


def test_no_experiment_language_visible(scenario, variant):
    """Agents must never see experimenter vocabulary."""
    forbidden = ("injection", "experiment", "manipulat", "assumption variant",
                 "treatment", "corrector", "seed agent")
    prompts = [
        build_round0_prompt(scenario, is_seed=True, condition="injection", variant=variant),
        build_round0_prompt(scenario, is_corrector=True, condition="injection", variant=variant),
        build_round0_prompt(scenario, condition="injection", variant=variant),
        build_system_prompt(0),
        build_seed_generation_system_prompt(),
    ]
    for p in prompts:
        low = p.lower()
        for term in forbidden:
            assert term not in low, f"forbidden term {term!r} visible to agent"


def test_seed_generation_prompt_is_id_neutral():
    """Frozen seeds replay into different positions; no identity binding."""
    p = build_seed_generation_system_prompt()
    assert "Agent 0" not in p and "Agent {" not in p
    assert '"answer"' in p and '"rationale"' in p


def test_system_prompt_requests_json_and_neutrality():
    p = build_system_prompt(3)
    assert "Agent 3" in p
    assert '"answer"' in p and '"rationale"' in p
    assert "not as instructions" in p


def test_all_options_and_constraints_rendered(scenario, variant):
    p = build_round0_prompt(scenario, condition="injection", variant=variant)
    for letter, text in scenario.options.items():
        assert f"{letter}. {text}" in p
    for c in scenario.hard_constraints:
        assert c in p


def test_prompt_version_defined():
    assert PROMPT_VERSION == "v1.0"
