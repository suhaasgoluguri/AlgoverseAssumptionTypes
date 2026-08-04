"""The four corrective-evidence scope arms must differ in exactly one way each.

Scope is the primary experimental variable, so a silent mismatch here would not
show up as an error — it would show up as a null result. These tests pin the
distinctions:

    none       nobody holds the record
    verifier   corrector holds it; peers hear only its own words
    relay      corrector holds it; the record travels in its outgoing message
    broadcast  every agent receives it directly from the record service

The relay/broadcast distinction is the easy one to get wrong: relay must NOT put
the record in anyone else's dossier, and broadcast must be a separate direct
receipt rather than either a private dossier item or a peer attachment.
"""

import json
from pathlib import Path

import pytest

from assumption_propagation.clients import MockClient
from assumption_propagation.evidence import CorrectiveRecord, canonical_placebo_json
from assumption_propagation.orchestrator import run_tool_correction_scope_ladder
from assumption_propagation.prompts import build_private_dossier, format_peer_message
from assumption_propagation.schemas import (
    CORRECTION_SCOPE,
    TOOL_CORRECTION_CONTROLS,
    TOOL_CORRECTION_SCOPE,
    CorrectionDelivery,
    ResponseOrigin,
    Scenario,
    SeedInstance,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "scenarios_hidden_profile_candidates.json"


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return Scenario.model_validate(json.loads(DATA.read_text(encoding="utf-8"))[0])


def dossier(scenario, condition, *, is_corrector, round_index=2, withhold=False):
    return build_private_dossier(
        scenario,
        is_seed=False,
        is_corrector=is_corrector,
        condition=condition,
        variant=scenario.assumption_variants[0],
        round_index=round_index,
        neutral_ordinal=None if is_corrector else 0,
        withhold_corrective=withhold,
    ) or ""


def correction_text(scenario) -> str:
    return scenario.assumption_variants[0].corrective_evidence.content


def test_scope_ladder_is_ordered_and_complete():
    assert CORRECTION_SCOPE == (
        "injection_no_correction",
        "injection",
        "injection_relay",
        "injection_broadcast",
    )


def test_tool_scope_ladder_names_do_not_alias_legacy_conditions():
    assert TOOL_CORRECTION_SCOPE == (
        "injection_tool_none",
        "injection_tool_testimony",
        "injection_tool_relay",
        "injection_tool_direct",
    )
    assert TOOL_CORRECTION_CONTROLS == (
        "injection_prompt_receipt_control",
        "injection_tool_placebo",
    )
    assert set(TOOL_CORRECTION_SCOPE).isdisjoint(CORRECTION_SCOPE)


def test_corrector_holds_the_record_in_every_arm_except_none(scenario):
    text = correction_text(scenario)
    assert text not in dossier(scenario, "injection_no_correction",
                               is_corrector=True, withhold=True)
    for condition in ("injection", "injection_relay", "injection_broadcast"):
        assert text in dossier(scenario, condition, is_corrector=True), condition


def test_no_arm_puts_the_record_in_a_peer_private_dossier(scenario):
    """Relay and direct receipts must stay out of role-private dossiers."""
    text = correction_text(scenario)
    available = scenario.assumption_variants[0].corrective_evidence.available_round
    for condition in CORRECTION_SCOPE:
        assert text not in dossier(scenario, condition, is_corrector=False,
                                   round_index=available + 1), condition


def test_corrector_private_record_respects_available_round(scenario):
    """The corrector's private source appears on schedule and persists."""
    text = correction_text(scenario)
    available = scenario.assumption_variants[0].corrective_evidence.available_round

    # corrector: visible from available_round onwards
    assert text not in dossier(scenario, "injection_broadcast",
                               is_corrector=True, round_index=available - 1)
    assert text in dossier(scenario, "injection_broadcast",
                           is_corrector=True, round_index=available)

    # Direct receipts are separate from role-private dossier content.
    assert text not in dossier(scenario, "injection_broadcast",
                               is_corrector=False, round_index=available)
    assert text not in dossier(scenario, "injection_broadcast",
                               is_corrector=False, round_index=available + 1)


@pytest.mark.asyncio
async def test_broadcast_and_relay_reach_peers_in_the_same_round(scenario):
    """The two wide-scope arms must differ in scope alone, not in timing.

    Regression test for a defect found on 2026-08-03: broadcast delivered to
    non-correctors in `available_round` while relay could not reach them until
    the round after, so every level-4 estimate confounded "everyone holds it"
    with "everyone holds it sooner".
    """
    text = correction_text(scenario)

    def count_by_round(prompts: list[str]) -> dict[int, int]:
        """How many prompts contain the record, keyed by round."""
        counts: dict[int, int] = {}
        for prompt in prompts:
            if text not in prompt:
                continue
            if "YOUR PREVIOUS ANSWER (round " in prompt:
                previous = prompt.split("YOUR PREVIOUS ANSWER (round ")[1].split(")")[0]
                round_index = int(previous) + 1
            else:
                round_index = 0
            counts[round_index] = counts.get(round_index, 0) + 1
        return counts

    available = scenario.assumption_variants[0].corrective_evidence.available_round
    relay = count_by_round(await _prompts_for(scenario, "injection_relay"))
    broadcast = count_by_round(await _prompts_for(scenario, "injection_broadcast"))

    # In `available_round` exactly one agent — the corrector — sees the record,
    # in both arms. Counting prompts avoids needing to know which agent each
    # prompt belongs to: 1 means corrector only, 4 means everyone.
    assert relay[available] == 1
    assert broadcast[available] == 1, (
        "broadcast leaked the record to peers in the same round the corrector "
        "received it; level 4 would gain a round of exposure over level 3"
    )
    # In the round after, all four agents have it: the corrector still holds it,
    # and peers receive it — relayed in one arm, directly in the other.
    assert relay[available + 1] == 4
    assert broadcast[available + 1] == 4


def test_relay_attaches_the_source_to_the_message():
    plain = format_peer_message(1, "B", "Apex is not compliant.")
    relayed = format_peer_message(1, "B", "Apex is not compliant.",
                                  relayed_source="The registry shows it expired.")
    assert "The registry shows it expired." not in plain
    assert "The registry shows it expired." in relayed
    # same answer and rationale — only provenance differs
    assert plain in relayed


def test_relay_is_inert_without_a_source():
    """None and empty must both leave the message byte-identical."""
    plain = format_peer_message(2, "C", "Cirrus fails residency.")
    assert format_peer_message(2, "C", "Cirrus fails residency.",
                               relayed_source=None) == plain
    assert format_peer_message(2, "C", "Cirrus fails residency.",
                               relayed_source="") == plain


# --------------------------------------------------------------------------
# Integration: the orchestrator must actually pass the source through.
# A break here would make injection_relay silently identical to injection —
# no error, just a null result on the primary variable.
# --------------------------------------------------------------------------

class PromptCapturingClient:
    """Mock client that records every prompt it is asked to complete."""

    model_identifier = "capture:test"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return '{"answer": "B", "rationale": "test"}'


async def _prompts_for(scenario, condition):
    from assumption_propagation.orchestrator import run_conversation
    from assumption_propagation.schemas import SeedInstance

    variant = scenario.assumption_variants[0]
    client = PromptCapturingClient()
    await run_conversation(
        scenario,
        variant=variant,
        client=client,
        topology_name="fully_connected_mesh",
        n_rounds=4,
        condition=condition,
        seed_agent_id=0,
        corrector_agent_id=1,
        frozen_seed=SeedInstance(
            seed_instance_id=f"{scenario.scenario_id}_test_seed_00",
            scenario_id=scenario.scenario_id,
            assumption_id=variant.assumption_id,
            seed_trial_index=0,
            answer=variant.favored_option,
            rationale="replayed seed",
            raw_response='{"answer": "%s", "rationale": "replayed seed"}'
                         % variant.favored_option,
        ),
    )
    return client.prompts


@pytest.mark.asyncio
async def test_relay_source_reaches_peer_prompts(scenario):
    text = correction_text(scenario)
    relay = "\n".join(await _prompts_for(scenario, "injection_relay"))
    plain = "\n".join(await _prompts_for(scenario, "injection"))

    # relay: the record shows up somewhere the corrector is NOT the reader
    assert "FORWARDED CASE-RECORD COPY" in relay
    assert text in relay
    # injection: the record appears only in the corrector's own dossier,
    # never as a relayed message
    assert "FORWARDED CASE-RECORD COPY" not in plain


def _frozen_seed(scenario: Scenario) -> SeedInstance:
    variant = scenario.assumption_variants[0]
    return SeedInstance(
        seed_instance_id=f"{scenario.scenario_id}_tool_test_seed_00",
        scenario_id=scenario.scenario_id,
        assumption_id=variant.assumption_id,
        seed_trial_index=0,
        answer=variant.favored_option,
        rationale="replayed tool-protocol seed",
        raw_response=(
            '{"answer": "%s", "rationale": "replayed tool-protocol seed"}'
            % variant.favored_option
        ),
    )


async def _tool_ladder(scenario: Scenario, *, include_controls: bool = True):
    client = MockClient()
    outputs = await run_tool_correction_scope_ladder(
        scenario,
        scenario.assumption_variants[0],
        _frozen_seed(scenario),
        client,
        include_controls=include_controls,
    )
    return outputs, client


@pytest.mark.asyncio
async def test_tool_ladder_matches_release_history_and_replays_one_corrector_trace(
    scenario,
):
    outputs, _ = await _tool_ladder(scenario)
    available = scenario.assumption_variants[0].corrective_evidence.available_round
    all_arms = (*TOOL_CORRECTION_SCOPE, *TOOL_CORRECTION_CONTROLS)

    # Round 0 and every non-corrector release-round response are one matched
    # history, not six fresh samples.
    for round_index in range(available + 1):
        for agent_id in range(4):
            if round_index == available and agent_id == 1:
                continue
            values = {
                (
                    outputs[arm].outputs[(agent_id, round_index)].answer,
                    outputs[arm].outputs[(agent_id, round_index)].rationale,
                )
                for arm in all_arms
            }
            assert len(values) == 1, (agent_id, round_index, values)

    # The corrected release response/tool trace was actually sampled in one
    # hidden anchor. Every corrected measured arm identifies it as a replay and
    # claims zero API work while retaining the same source call/result hashes.
    corrected = [arm for arm in all_arms if arm != "injection_tool_none"]
    release_records = [
        next(
            record
            for record in outputs[arm].records
            if record.agent_id == 1 and record.round == available
        )
        for arm in corrected
    ]
    assert {record.response_origin for record in release_records} == {
        ResponseOrigin.MATCHED_REPLAY
    }
    assert all(record.tool_call_executed is False for record in release_records)
    assert all(record.tool_trace_replayed is True for record in release_records)
    assert all(record.tool_trace.api_call_count == 0 for record in release_records)
    assert len({record.tool_trace.tool_call_id for record in release_records}) == 1
    assert len({record.tool_trace.result_sha256 for record in release_records}) == 1

    none_release = next(
        record
        for record in outputs["injection_tool_none"].records
        if record.agent_id == 1 and record.round == available
    )
    assert none_release.tool_trace is None
    assert none_release.corrective_record_delivery == CorrectionDelivery.NONE


@pytest.mark.asyncio
async def test_tool_ladder_delivery_routes_are_persistent_and_operational(scenario):
    outputs, _ = await _tool_ladder(scenario)
    available = scenario.assumption_variants[0].corrective_evidence.available_round
    peer_rounds = range(available + 1, 4)

    expected = {
        "injection_tool_testimony": CorrectionDelivery.NONE,
        "injection_tool_relay": CorrectionDelivery.RELAYED_RECORD,
        "injection_tool_direct": CorrectionDelivery.DIRECT_TOOL_RECORD,
        "injection_prompt_receipt_control": CorrectionDelivery.PROMPT_RECEIPT_CONTROL,
        "injection_tool_placebo": CorrectionDelivery.TOOL_PLACEBO,
    }
    for arm, delivery in expected.items():
        for round_index in peer_rounds:
            for agent_id in (0, 2, 3):
                record = next(
                    row
                    for row in outputs[arm].records
                    if row.agent_id == agent_id and row.round == round_index
                )
                assert record.corrective_record_delivery == delivery
                if delivery in (
                    CorrectionDelivery.DIRECT_TOOL_RECORD,
                    CorrectionDelivery.TOOL_PLACEBO,
                ):
                    assert record.tool_call_executed is True
                    assert record.tool_trace_replayed is False
                    assert record.tool_trace.api_call_count == 2
                else:
                    assert record.tool_trace is None


@pytest.mark.asyncio
async def test_tool_relay_direct_and_placebo_use_canonical_payload_bytes(scenario):
    _, client = await _tool_ladder(scenario)
    record = CorrectiveRecord.from_variant(scenario.assumption_variants[0])
    payload = record.canonical_json()
    placebo = canonical_placebo_json()

    results = [call["tool_result"] for call in client.forced_tool_calls]
    assert payload in results and placebo in results
    assert set(results) == {payload, placebo}

    ordinary_prompts = [user for _, user in client.calls]
    relay_prompts = [p for p in ordinary_prompts if "BEGIN TOOL OUTPUT" in p]
    prompt_controls = [p for p in ordinary_prompts if "BEGIN RECORD PAYLOAD" in p]
    assert len(relay_prompts) == 6
    assert len(prompt_controls) == 6
    assert all(p.count(payload) == 1 for p in relay_prompts)
    assert all(p.count(payload) == 1 for p in prompt_controls)
    assert all(
        payload not in call["user_prompt"] for call in client.forced_tool_calls
    ), "genuine tool results must not be preinserted in the base user prompt"
