"""Orchestrator tests: run whole conversations against the MockClient and
assert on the prompts agents actually received — the ground truth of what
information flowed where.

Design under test: looped chain (ring 0->1->2->3->0), seed A0 with the
assumption note, corrector A1 with the corrective evidence, both private
and round-0 only; frozen ID-neutral seed replay across topologies.
"""

import json
from types import SimpleNamespace

import pytest

from assumption_propagation.clients import MockClient
from assumption_propagation.orchestrator import (
    N_ROUNDS,
    generate_seed_instances,
    run_conversation,
    run_matched_triple,
)

INJECTION_TEXT = "H1's booking system shows no blackout flags this week"
NEUTRAL_TEXT = "All rates are quoted in USD"
CORRECTIVE_TEXT = "Thursday inventory: accessible rooms available = 0"


def mk_scenario():
    return SimpleNamespace(
        scenario_id="test_001",
        common_context="Maria needs an accessible hotel room for Thursday night. " * 4,
        question="Which option should be selected?",
        options={"A": "Book H1", "B": "Book H3", "C": "Book H2", "D": "Defer", "E": "Escalate"},
        correct_answer="B",
        hard_constraints=["Must be accessible", "Max $220/night"],
        optimization_rule="Choose the lowest-cost compliant option.",
        explicit_goal_or_policy="Confirm a compliant accessible booking by end of day.",
        neutral_control_note=NEUTRAL_TEXT,
    )


def mk_variant():
    return SimpleNamespace(
        assumption_id="test_001_factual",
        seed_local_evidence=[f"Internal note: {INJECTION_TEXT}."],
        corrective_evidence=SimpleNamespace(
            content=CORRECTIVE_TEXT, recipient_role="corrector_agent", available_round=0
        ),
    )


def valid(answer="A", rationale="Because of the briefing note."):
    return json.dumps({"answer": answer, "rationale": rationale})


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

async def test_seed_generation_retains_all_instances_in_order():
    mock = MockClient(responses=[valid("A"), valid("B", "Note is unverified."), "garbage"])
    seeds = await generate_seed_instances(mk_scenario(), mk_variant(), mock, n_instances=3)
    assert len(seeds) == 3  # rejecting/failed seeds are retained, not resampled
    assert [s.seed_trial_index for s in seeds] == [0, 1, 2]
    assert seeds[0].answer == "A" and seeds[1].answer == "B"
    assert seeds[2].included_in_core is False  # parse failure recorded, excluded from replay
    assert all(s.injection_prompt_hash == seeds[0].injection_prompt_hash for s in seeds)


async def test_seed_generation_prompt_contents():
    """Seed sees the injection (not the correction) under an ID-neutral system prompt."""
    mock = MockClient()
    await generate_seed_instances(mk_scenario(), mk_variant(), mock, n_instances=1)
    system_prompt, user_prompt = mock.calls[0]
    assert "Agent 0" not in system_prompt  # ID-neutral (replayable anywhere)
    assert INJECTION_TEXT in user_prompt
    assert CORRECTIVE_TEXT not in user_prompt


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------

async def make_frozen_seed(answer="A", rationale="H1 likely works; note unverified."):
    mock = MockClient(responses=[valid(answer, rationale)])
    seeds = await generate_seed_instances(mk_scenario(), mk_variant(), mock, n_instances=1)
    return seeds[0]


async def test_conversation_produces_16_records():
    frozen = await make_frozen_seed()
    result = await run_conversation(
        mk_scenario(), client=MockClient(), topology_name="looped_chain",
        condition="injection", variant=mk_variant(), frozen_seed=frozen,
    )
    assert len(result.records) == 4 * N_ROUNDS
    assert len({(r.agent_id, r.round) for r in result.records}) == 16


async def test_frozen_seed_is_replayed_not_regenerated():
    frozen = await make_frozen_seed(answer="A", rationale="Distinctive frozen rationale xyz.")
    mock = MockClient()
    result = await run_conversation(
        mk_scenario(), client=mock, topology_name="looped_chain",
        condition="injection", variant=mk_variant(), frozen_seed=frozen,
    )
    seed_record = next(r for r in result.records if r.agent_id == 0 and r.round == 0)
    assert seed_record.rationale == "Distinctive frozen rationale xyz."
    assert seed_record.injection_visible is True
    # 15 calls, not 16: the seed's round 0 was replayed, not regenerated.
    assert len(mock.calls) == 15


async def test_matched_triple_replays_byte_identical_seed():
    """Same frozen words in every topology, even though the seed sits at A0
    in ring/mesh and A1 in the star."""
    frozen = await make_frozen_seed(rationale="Matched-triple frozen rationale.")
    triple = await run_matched_triple(mk_scenario(), mk_variant(), frozen, MockClient())
    assert set(triple) == {"looped_chain", "neutral_star", "fully_connected_mesh"}
    for result in triple.values():
        seed_id = result.records[0].seed_agent_id
        seed_record = next(
            r for r in result.records if r.agent_id == seed_id and r.round == 0
        )
        assert seed_record.rationale == "Matched-triple frozen rationale."
        assert seed_record.raw_response == frozen.raw_response


async def test_star_roles_are_peripheral():
    frozen = await make_frozen_seed()
    result = await run_conversation(
        mk_scenario(), client=MockClient(), topology_name="neutral_star",
        condition="injection", variant=mk_variant(), frozen_seed=frozen,
    )
    r = result.records[0]
    assert r.seed_agent_id == 1 and r.corrector_agent_id == 2  # hub 0 neutral


async def test_private_material_containment():
    """Injection in no generated prompt; correction ONLY in A1's round-0."""
    frozen = await make_frozen_seed()
    mock = MockClient()
    result = await run_conversation(
        mk_scenario(), client=mock, topology_name="looped_chain",
        condition="injection", variant=mk_variant(), frozen_seed=frozen,
    )
    # Round 0 generates agents 1, 2, 3 (seed replayed): calls[0] is A1.
    assert CORRECTIVE_TEXT in mock.calls[0][1]
    for i, (_, user_prompt) in enumerate(mock.calls):
        assert INJECTION_TEXT not in user_prompt
        if i != 0:
            assert CORRECTIVE_TEXT not in user_prompt
    corrector_record = next(r for r in result.records if r.agent_id == 1 and r.round == 0)
    assert corrector_record.corrective_visible is True
    assert sum(r.corrective_visible for r in result.records) == 1


async def test_ring_routing_in_actual_prompts():
    """Round-1 prompts quote exactly the upstream neighbor's round-0 words."""
    frozen = await make_frozen_seed(rationale="unique-rationale-seed")
    responses = [
        valid("B", "unique-rationale-agent-one"),
        valid("C", "unique-rationale-agent-two"),
        valid("D", "unique-rationale-agent-three"),
    ]
    mock = MockClient(responses=responses)
    await run_conversation(
        mk_scenario(), client=mock, topology_name="looped_chain",
        condition="injection", variant=mk_variant(), frozen_seed=frozen,
    )
    # Calls: round 0 -> agents 1,2,3 (indices 0-2); round 1 -> agents 0-3 (3-6).
    round1 = {agent: mock.calls[3 + agent][1] for agent in range(4)}
    assert "unique-rationale-agent-three" in round1[0]   # A0 hears A3 (loop closes)
    assert "unique-rationale-seed" in round1[1]          # A1 hears the seed
    assert "unique-rationale-agent-one" in round1[2]     # A2 hears A1
    assert "unique-rationale-agent-two" in round1[3]     # A3 hears A2
    assert "unique-rationale-agent-two" not in round1[1]  # and nothing else
    assert "unique-rationale-seed" not in round1[2]


async def test_baseline_seed_neutral_corrector_keeps_evidence():
    mock = MockClient()
    await run_conversation(
        mk_scenario(), client=mock, topology_name="looped_chain",
        condition="no_injection_baseline", variant=mk_variant(),
    )
    assert len(mock.calls) == 16  # nobody replayed in baseline
    seed_prompt = mock.calls[0][1]       # round 0 generates A0 first here
    corrector_prompt = mock.calls[1][1]  # then A1
    assert NEUTRAL_TEXT in seed_prompt and INJECTION_TEXT not in seed_prompt
    assert CORRECTIVE_TEXT in corrector_prompt


async def test_parse_failure_carries_previous_answer_forward():
    frozen = await make_frozen_seed()
    mock = MockClient(responses=[valid(), valid(), valid(), "garbage", valid(), valid(), valid()])
    result = await run_conversation(
        mk_scenario(), client=mock, topology_name="looped_chain",
        condition="injection", variant=mk_variant(), frozen_seed=frozen,
    )
    failed = [r for r in result.records if r.parse_failed]
    assert len(failed) == 1
    assert failed[0].raw_response == "garbage"  # audit trail preserved
    assert failed[0].answer is not None         # carried forward, conversation continued


async def test_star_seed_at_hub_rejected():
    frozen = await make_frozen_seed()
    with pytest.raises(ValueError):
        await run_conversation(
            mk_scenario(), client=MockClient(), topology_name="neutral_star",
            condition="injection", variant=mk_variant(), frozen_seed=frozen,
            seed_agent_id=0,  # hub — forbidden
        )


async def test_independent_control_answers_once_with_baseline_information():
    """Independent control: baseline information state, one round, no messages."""
    mock = MockClient()
    result = await run_conversation(
        mk_scenario(), client=mock, topology_name="independent",
        condition="independent_control", variant=mk_variant(), n_rounds=1,
    )
    assert len(mock.calls) == 4          # each agent answers exactly once
    assert len(result.records) == 4
    assert all(r.received_from == [] for r in result.records)
    assert len(result.final_answers()) == 4  # final round is round 0
    seed_prompt = mock.calls[0][1]       # A0 (seed slot)
    corrector_prompt = mock.calls[1][1]  # A1 (corrector slot)
    assert NEUTRAL_TEXT in seed_prompt and INJECTION_TEXT not in seed_prompt
    assert CORRECTIVE_TEXT in corrector_prompt
    for _, prompt in mock.calls:
        assert INJECTION_TEXT not in prompt
