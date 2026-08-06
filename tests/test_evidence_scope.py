"""Fail-closed tests for the corrective-evidence delivery experiment.

These checks focus on *delivery mechanics*, not model behaviour.  A model may
ignore a record, but the protocol must still make it possible to establish from
the prompt and raw metadata exactly which record reached which agent, by which
route, and in which round.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from assumption_propagation.evidence import (
    AUTHORITATIVE_RECORD_SOURCE,
    PLACEBO_RECORD_STATUS,
    CorrectiveRecord,
    CorrectiveRecordStore,
    canonical_placebo_json,
    canonical_placebo_json_bytes,
    canonical_placebo_sha256,
    sha256_bytes,
    sha256_text,
)
from assumption_propagation.orchestrator import (
    run_conversation,
    run_correction_scope_ladder,
)
from assumption_propagation.prompts import (
    DOSSIER_HEADER,
    build_private_dossier,
    build_update_prompt,
    format_direct_record_receipt,
    format_peer_message,
)
from assumption_propagation.schemas import (
    CORRECTION_SCOPE,
    TOOL_CORRECTION_CONTROLS,
    TOOL_CORRECTION_SCOPE,
    TOOL_CORRECTION_SCOPE_WITH_CONTROLS,
    AgentRoundRecord,
    Condition,
    CorrectionDelivery,
    ResponseOrigin,
    Scenario,
    SeedInstance,
    ToolCallTrace,
)


DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "scenarios_hidden_profile_candidates.json"
)


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return Scenario.model_validate(json.loads(DATA.read_text(encoding="utf-8"))[0])


@pytest.fixture(scope="module")
def variant(scenario):
    return scenario.assumption_variants[0]


@pytest.fixture(scope="module")
def canonical_record(variant) -> CorrectiveRecord:
    return CorrectiveRecord.from_variant(variant)


def _frozen_seed(scenario, variant) -> SeedInstance:
    return SeedInstance(
        seed_instance_id=f"{variant.assumption_id}_scope_contract_seed_00",
        scenario_id=scenario.scenario_id,
        assumption_id=variant.assumption_id,
        seed_trial_index=0,
        answer=variant.favored_option,
        rationale="The private assessment supports the selected option.",
        raw_response=json.dumps(
            {
                "answer": variant.favored_option,
                "rationale": "The private assessment supports the selected option.",
            }
        ),
    )


class RouteCapturingClient:
    """Deterministic client that indexes prompts by their recipient and round."""

    model_identifier = "capture:evidence-scope"
    temperature = None

    def __init__(self) -> None:
        self.prompts: dict[tuple[int, int], str] = {}

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        # The frozen seed is never generated here.  Every other production
        # system prompt includes the recipient's stable agent id.
        marker = "You are Agent "
        assert marker in system_prompt
        agent_id = int(system_prompt.split(marker, 1)[1].split(" ", 1)[0])
        if "YOUR PREVIOUS ANSWER (round " in user_prompt:
            previous = int(
                user_prompt.split("YOUR PREVIOUS ANSWER (round ", 1)[1]
                .split(")", 1)[0]
            )
            round_index = previous + 1
        else:
            round_index = 0
        assert (agent_id, round_index) not in self.prompts
        self.prompts[(agent_id, round_index)] = user_prompt
        return '{"answer":"B","rationale":"Deterministic test response."}'


class UniqueResponseClient:
    """Every generated call is unique, making an accidental resample visible."""

    model_identifier = "capture:unique-evidence-scope"
    temperature = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return json.dumps(
            {
                "answer": "B",
                "rationale": f"Unique generated response {len(self.calls)}.",
            }
        )


async def _run_scope_arm(scenario, variant, condition):
    client = RouteCapturingClient()
    result = await run_conversation(
        scenario,
        variant=variant,
        client=client,
        topology_name="fully_connected_mesh",
        n_rounds=4,
        condition=condition,
        seed_agent_id=0,
        corrector_agent_id=1,
        frozen_seed=_frozen_seed(scenario, variant),
    )
    return result, client


# ---------------------------------------------------------------------------
# Canonical record and bounded record service
# ---------------------------------------------------------------------------


def test_canonical_record_identity_is_deterministic(variant, canonical_record):
    duplicate = CorrectiveRecord.from_variant(variant)
    assert duplicate == canonical_record
    assert canonical_record.content == variant.corrective_evidence.content
    assert canonical_record.source == AUTHORITATIVE_RECORD_SOURCE
    assert canonical_record.sha256 == sha256_text(canonical_record.content)
    assert len(canonical_record.sha256) == 64
    assert canonical_record.record_id == (
        f"REC-{canonical_record.sha256[:12].upper()}"
    )


def test_canonical_record_json_is_sorted_minified_unicode_utf8():
    content = "Résumé Δ: record is current."
    digest = sha256_text(content)
    record = CorrectiveRecord(
        record_id=f"REC-{digest[:12].upper()}",
        source=AUTHORITATIVE_RECORD_SOURCE,
        content=content,
        sha256=digest,
    )
    expected = (
        '{"content":"Résumé Δ: record is current.",'
        f'"record_id":"{record.record_id}",'
        f'"sha256":"{digest}",'
        f'"source":"{AUTHORITATIVE_RECORD_SOURCE}"}}'
    )

    assert record.canonical_json() == expected
    assert record.canonical_json_bytes() == expected.encode("utf-8")
    assert "\\u00e9" not in record.canonical_json()
    assert '\": ' not in record.canonical_json()
    assert '\", \"' not in record.canonical_json()
    assert record.canonical_json_sha256() == sha256_bytes(expected.encode("utf-8"))


def test_relay_and_tool_use_byte_identical_canonical_record(canonical_record):
    relay_payload = canonical_record.canonical_json()
    tool_payload = CorrectiveRecordStore(canonical_record).retrieve_canonical_json(
        canonical_record.record_id,
        recipient_agent_id=2,
        round_index=3,
    )

    assert tool_payload == relay_payload
    assert tool_payload.encode("utf-8") == canonical_record.canonical_json_bytes()
    assert json.loads(tool_payload) == canonical_record.as_payload()


def test_placebo_payload_is_canonical_stable_and_non_diagnostic(canonical_record):
    expected = (
        '{"content":null,"record_id":null,"sha256":null,'
        f'"source":"{AUTHORITATIVE_RECORD_SOURCE}",'
        f'"status":"{PLACEBO_RECORD_STATUS}"}}'
    )

    assert canonical_placebo_json() == expected
    assert canonical_placebo_json_bytes() == expected.encode("utf-8")
    assert canonical_placebo_sha256() == sha256_bytes(expected.encode("utf-8"))
    assert canonical_record.content not in expected
    payload = json.loads(expected)
    assert payload["status"] == "no_additional_record"
    assert payload["record_id"] is None
    assert payload["content"] is None


def test_protocol_v2_conditions_are_additive_and_legacy_scope_is_unchanged():
    assert CORRECTION_SCOPE == (
        "injection_no_correction",
        "injection",
        "injection_relay",
        "injection_broadcast",
    )
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
    assert TOOL_CORRECTION_SCOPE_WITH_CONTROLS == (
        *TOOL_CORRECTION_SCOPE,
        *TOOL_CORRECTION_CONTROLS,
    )
    for legacy in CORRECTION_SCOPE:
        assert Condition(legacy).value == legacy
    for condition in TOOL_CORRECTION_SCOPE_WITH_CONTROLS:
        assert Condition(condition).value == condition


def test_protocol_v2_delivery_values_are_additive():
    for legacy in (
        "none",
        "corrector_private_record",
        "relayed_record",
        "direct_system_record",
    ):
        assert CorrectionDelivery(legacy).value == legacy
    for delivery in (
        "corrector_tool_record",
        "direct_tool_record",
        "prompt_receipt_control",
        "tool_placebo",
    ):
        assert CorrectionDelivery(delivery).value == delivery


def test_canonical_record_is_immutable(canonical_record):
    with pytest.raises(FrozenInstanceError):
        canonical_record.content = "replacement"  # type: ignore[misc]


def test_record_store_exposes_only_the_designated_correction(canonical_record):
    store = CorrectiveRecordStore(canonical_record)
    receipt = store.retrieve(
        canonical_record.record_id, recipient_agent_id=3, round_index=3
    )
    assert receipt.record is canonical_record
    assert receipt.recipient_agent_id == 3
    assert receipt.round_index == 3

    with pytest.raises(KeyError, match="not available in this case"):
        store.retrieve("REC-ANOTHER-CANDIDATE", recipient_agent_id=3, round_index=3)
    with pytest.raises(ValueError, match="between 0 and 3"):
        store.retrieve(canonical_record.record_id, recipient_agent_id=4, round_index=3)
    with pytest.raises(ValueError, match="non-negative"):
        store.retrieve(canonical_record.record_id, recipient_agent_id=3, round_index=-1)


def test_relay_and_direct_receipt_carry_the_same_record(canonical_record):
    relay = format_peer_message(
        1,
        "B",
        "The trap is disqualified.",
        relayed_record=canonical_record,
    )
    receipt = CorrectiveRecordStore(canonical_record).retrieve(
        canonical_record.record_id, recipient_agent_id=2, round_index=3
    )
    direct = format_direct_record_receipt(receipt)

    for value in (
        canonical_record.record_id,
        canonical_record.sha256,
        canonical_record.content,
    ):
        assert value in relay
        assert value in direct

    assert "FORWARDED CASE-RECORD COPY" in relay
    assert "not retrieved directly" in relay
    assert "DIRECT CASE-RECORD RECEIPT" not in relay
    assert "DIRECT CASE-RECORD RECEIPT" in direct
    assert "FORWARDED CASE-RECORD COPY" not in direct


def test_verifier_receives_the_same_canonical_record_in_its_private_dossier(
    scenario, variant, canonical_record
):
    dossier = build_private_dossier(
        scenario,
        is_seed=False,
        is_corrector=True,
        condition="injection",
        variant=variant,
        round_index=variant.corrective_evidence.available_round,
    )
    assert dossier is not None
    assert "AUTHORITATIVE CASE RECORD" in dossier
    for value in (
        canonical_record.source,
        canonical_record.record_id,
        canonical_record.sha256,
        canonical_record.content,
    ):
        assert value in dossier


# ---------------------------------------------------------------------------
# Prompt containment and persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", CORRECTION_SCOPE)
@pytest.mark.parametrize("round_index", range(4))
def test_injection_persists_in_seed_dossier_in_every_scope_arm(
    scenario, variant, condition, round_index
):
    dossier = build_private_dossier(
        scenario,
        is_seed=True,
        is_corrector=False,
        condition=condition,
        variant=variant,
        round_index=round_index,
    )
    assert dossier is not None
    assert variant.seed_private_assessment.content in dossier


def test_direct_receipt_is_a_separate_prompt_block_not_private_dossier(
    scenario, variant, canonical_record
):
    round_index = variant.corrective_evidence.available_round + 1
    private_dossier = build_private_dossier(
        scenario,
        is_seed=False,
        is_corrector=False,
        condition="injection_broadcast",
        variant=variant,
        round_index=round_index,
        neutral_ordinal=0,
    )
    assert private_dossier is not None  # this neutral still has its own brief
    assert canonical_record.content not in private_dossier
    assert canonical_record.record_id not in private_dossier

    receipt = CorrectiveRecordStore(canonical_record).retrieve(
        canonical_record.record_id,
        recipient_agent_id=2,
        round_index=round_index,
    )
    direct_block = format_direct_record_receipt(receipt)
    prompt = build_update_prompt(
        scenario,
        own_previous_answer="E",
        own_previous_rationale="Status is unresolved.",
        peer_messages=[],
        round_index=round_index,
        private_note=private_dossier,
        direct_record_receipt=direct_block,
    )

    assert prompt.count(DOSSIER_HEADER) == 1
    assert prompt.index(DOSSIER_HEADER) < prompt.index("DIRECT CASE-RECORD RECEIPT")
    assert prompt.index("DIRECT CASE-RECORD RECEIPT") < prompt.index(
        "YOUR PREVIOUS ANSWER"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", CORRECTION_SCOPE)
async def test_scope_prompt_routes_are_mutually_exclusive(
    scenario, variant, canonical_record, condition
):
    _, client = await _run_scope_arm(scenario, variant, condition)
    available = variant.corrective_evidence.available_round

    # Before the corrector has spoken with the record, neither wide-delivery
    # marker may appear in any prompt.
    for (_, round_index), prompt in client.prompts.items():
        if round_index <= available:
            assert "FORWARDED CASE-RECORD COPY" not in prompt
            assert "DIRECT CASE-RECORD RECEIPT" not in prompt

    final_peer_prompts = {
        agent_id: client.prompts[(agent_id, available + 1)]
        for agent_id in (0, 2, 3)
    }
    for agent_id, prompt in final_peer_prompts.items():
        if condition == "injection_relay":
            assert "FORWARDED CASE-RECORD COPY" in prompt
            assert "forwarded by: Agent 1" in prompt
            assert "DIRECT CASE-RECORD RECEIPT" not in prompt
        elif condition == "injection_broadcast":
            assert "DIRECT CASE-RECORD RECEIPT" in prompt
            assert f"retrieved directly for: Agent {agent_id}" in prompt
            assert f"retrieval round: {available + 1}" in prompt
            assert "FORWARDED CASE-RECORD COPY" not in prompt
        else:
            assert "FORWARDED CASE-RECORD COPY" not in prompt
            assert "DIRECT CASE-RECORD RECEIPT" not in prompt

    if condition in {"injection_relay", "injection_broadcast"}:
        for prompt in final_peer_prompts.values():
            assert canonical_record.record_id in prompt
            assert canonical_record.sha256 in prompt
            assert canonical_record.content in prompt


@pytest.mark.asyncio
async def test_direct_service_receipt_does_not_expose_other_private_briefs(
    scenario, variant
):
    _, client = await _run_scope_arm(scenario, variant, "injection_broadcast")
    prompt = client.prompts[(2, variant.corrective_evidence.available_round + 1)]

    own_brief = scenario.hidden_profile.neutral_briefs[0]
    other_briefs = (
        scenario.hidden_profile.seed_brief,
        scenario.hidden_profile.corrector_brief,
        scenario.hidden_profile.neutral_briefs[1],
    )
    assert own_brief in prompt
    for brief in other_briefs:
        assert brief not in prompt


# ---------------------------------------------------------------------------
# Serialized route metadata
# ---------------------------------------------------------------------------


def _expected_delivery(condition: str, agent_id: int, round_index: int, available: int):
    if condition == "injection_no_correction" or round_index < available:
        return CorrectionDelivery.NONE
    if agent_id == 1:
        return CorrectionDelivery.CORRECTOR_PRIVATE
    if round_index == available:
        return CorrectionDelivery.NONE
    if condition == "injection_relay":
        return CorrectionDelivery.RELAYED_RECORD
    if condition == "injection_broadcast":
        return CorrectionDelivery.DIRECT_SYSTEM_RECORD
    return CorrectionDelivery.NONE


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", CORRECTION_SCOPE)
async def test_serialized_delivery_metadata_matches_actual_route(
    scenario, variant, canonical_record, condition
):
    result, _ = await _run_scope_arm(scenario, variant, condition)
    available = variant.corrective_evidence.available_round

    for record in result.records:
        expected = _expected_delivery(
            condition, record.agent_id, record.round, available
        )
        assert record.corrective_record_delivery == expected
        assert record.corrective_visible is (expected != CorrectionDelivery.NONE)

        if expected == CorrectionDelivery.NONE:
            assert record.corrective_record_id is None
            assert record.corrective_record_sha256 is None
            assert not record.corrective_record_sender_ids
            continue

        assert record.corrective_record_id == canonical_record.record_id
        assert record.corrective_record_sha256 == canonical_record.sha256

        if expected == CorrectionDelivery.RELAYED_RECORD:
            assert record.corrective_record_in_peer_message is True
            assert record.corrective_record_in_dossier is False
            assert record.corrective_record_sender_ids == [1]
        elif expected == CorrectionDelivery.DIRECT_SYSTEM_RECORD:
            assert record.corrective_record_in_peer_message is False
            assert record.corrective_record_in_direct_receipt is True
            assert not record.corrective_record_sender_ids
            # A direct service receipt is a separate transport block, not part
            # of the recipient's role-private dossier.
            assert record.corrective_record_in_dossier is False
        else:
            assert expected == CorrectionDelivery.CORRECTOR_PRIVATE
            assert record.corrective_record_in_dossier is True
            assert record.corrective_record_in_peer_message is False
            assert not record.corrective_record_sender_ids


def test_agent_round_schema_declares_direct_receipt_separately_from_dossier():
    assert "corrective_record_in_direct_receipt" in AgentRoundRecord.model_fields


def _record_payload(**updates):
    payload = {
        "run_id": "route-schema-test",
        "scenario_id": "vendor_001",
        "assumption_id": "vendor_001_factual_negative",
        "seed_instance_id": "seed-00",
        "condition": "injection_relay",
        "topology": "fully_connected_mesh",
        "model": "mock:test",
        "seed_agent_id": 0,
        "corrector_agent_id": 1,
        "agent_id": 2,
        "round": 3,
        "answer": "B",
        "rationale": "The record disqualifies the trap.",
        "corrective_visible": True,
    }
    payload.update(updates)
    return payload


def _tool_trace_payload(**updates):
    payload = {
        "tool_name": "get_corrective_record",
        "tool_call_id": "call_scope_001",
        "arguments_sha256": "1" * 64,
        "result_sha256": "2" * 64,
        "success": True,
        "error": None,
        "api_call_count": 2,
        "finish_reasons": ["tool_calls", "stop"],
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "transcript_sha256": "3" * 64,
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    "route_fields",
    [
        {
            "condition": "injection_tool_testimony",
            "corrective_record_delivery": "corrector_tool_record",
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "agent_id": 1,
            "tool_trace": _tool_trace_payload(),
            "tool_call_executed": True,
            "tool_trace_replayed": False,
        },
        {
            # A matched arm retains source/call/result hashes but does not
            # claim another backend call, finish reason, or token usage.
            "condition": "injection_tool_relay",
            "corrective_record_delivery": "corrector_tool_record",
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "agent_id": 1,
            "tool_trace": _tool_trace_payload(
                api_call_count=0,
                finish_reasons=[],
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
            "tool_call_executed": False,
            "tool_trace_replayed": True,
        },
        {
            "condition": "injection_tool_direct",
            "corrective_record_delivery": "direct_tool_record",
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "tool_trace": _tool_trace_payload(tool_call_id="call_scope_direct_002"),
            "tool_call_executed": True,
            "tool_trace_replayed": False,
        },
        {
            "condition": "injection_prompt_receipt_control",
            "corrective_record_delivery": "prompt_receipt_control",
            "corrective_record_in_direct_receipt": True,
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
        },
        {
            "condition": "injection_tool_placebo",
            "corrective_record_delivery": "tool_placebo",
            "corrective_visible": False,
            "tool_trace": _tool_trace_payload(
                tool_name="check_for_additional_record",
                tool_call_id="call_scope_placebo_003",
                result_sha256=canonical_placebo_sha256(),
            ),
            "tool_call_executed": True,
            "tool_trace_replayed": False,
        },
        {
            # A failed genuine call is observable even though no record was
            # delivered and corrective visibility remains false.
            "condition": "injection_tool_direct",
            "corrective_record_delivery": "none",
            "corrective_visible": False,
            "tool_trace": _tool_trace_payload(
                result_sha256=None,
                success=False,
                error="record service unavailable",
            ),
            "tool_call_executed": True,
            "tool_trace_replayed": False,
        },
        {
            # The placebo arm still gives the verifier the real record; only
            # ordinary recipients receive the non-diagnostic tool result.
            "condition": "injection_tool_placebo",
            "corrective_record_delivery": "corrector_tool_record",
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "agent_id": 1,
            "tool_trace": _tool_trace_payload(tool_call_id="call_verifier_placebo_arm"),
            "tool_call_executed": True,
            "tool_trace_replayed": False,
        },
    ],
)
def test_protocol_v2_tool_routes_round_trip(route_fields):
    record = AgentRoundRecord.model_validate(_record_payload(**route_fields))
    restored = AgentRoundRecord.model_validate_json(record.model_dump_json())
    assert restored == record


def test_legacy_record_without_tool_instrumentation_still_loads():
    record = AgentRoundRecord.model_validate(
        _record_payload(
            corrective_record_delivery="relayed_record",
            corrective_record_in_peer_message=True,
            corrective_record_sender_ids=[1],
            corrective_record_id="REC-ABC",
            corrective_record_sha256="a" * 64,
            received_from=[0, 1, 3],
        )
    )
    assert record.tool_trace is None
    assert record.tool_call_executed is None
    assert record.tool_trace_replayed is None


@pytest.mark.parametrize(
    ("route_fields", "message"),
    [
        (
            {
                "condition": "injection_tool_direct",
                "corrective_record_delivery": "direct_tool_record",
                "corrective_record_id": "REC-ABC",
                "corrective_record_sha256": "a" * 64,
            },
            "requires tool_trace",
        ),
        (
            {
                "condition": "injection_tool_relay",
                "corrective_record_delivery": "corrector_tool_record",
                "corrective_record_id": "REC-ABC",
                "corrective_record_sha256": "a" * 64,
                "agent_id": 1,
                "tool_trace": _tool_trace_payload(),
                "tool_call_executed": False,
                "tool_trace_replayed": True,
            },
            "api_call_count=0",
        ),
        (
            {
                "condition": "injection_tool_direct",
                "corrective_record_delivery": "direct_tool_record",
                "corrective_record_id": "REC-ABC",
                "corrective_record_sha256": "a" * 64,
                "tool_trace": _tool_trace_payload(api_call_count=0, finish_reasons=[]),
                "tool_call_executed": True,
                "tool_trace_replayed": False,
            },
            "at least one API call",
        ),
        (
            {
                "condition": "injection_tool_placebo",
                "corrective_record_delivery": "tool_placebo",
                "corrective_visible": True,
                "tool_trace": _tool_trace_payload(
                    tool_name="check_for_additional_record",
                    result_sha256=canonical_placebo_sha256(),
                ),
                "tool_call_executed": True,
                "tool_trace_replayed": False,
            },
            "corrective_visible=false",
        ),
        (
            {
                "condition": "injection_relay",
                "corrective_record_delivery": "none",
                "corrective_visible": False,
                "tool_trace": _tool_trace_payload(),
                "tool_call_executed": True,
                "tool_trace_replayed": False,
            },
            "protocol-v2 tool condition",
        ),
        (
            {
                "condition": "injection_prompt_receipt_control",
                "corrective_record_delivery": "prompt_receipt_control",
                "corrective_record_in_direct_receipt": True,
                "corrective_record_id": "REC-ABC",
                "corrective_record_sha256": "a" * 64,
                "tool_trace": _tool_trace_payload(),
                "tool_call_executed": True,
                "tool_trace_replayed": False,
            },
            "cannot claim a tool execution",
        ),
    ],
)
def test_protocol_v2_tool_routes_reject_false_instrumentation(route_fields, message):
    with pytest.raises(ValueError, match=message):
        AgentRoundRecord.model_validate(_record_payload(**route_fields))


@pytest.mark.parametrize(
    ("trace_fields", "message"),
    [
        ({"result_sha256": None}, "requires result_sha256"),
        ({"error": "unexpected error"}, "cannot carry an error"),
        (
            {"success": False, "result_sha256": None, "error": None},
            "requires an error",
        ),
        ({"total_tokens": 999}, "must equal"),
        ({"api_call_count": 1}, "cannot outnumber API calls"),
    ],
)
def test_tool_trace_rejects_internally_inconsistent_audit_data(trace_fields, message):
    with pytest.raises(ValueError, match=message):
        ToolCallTrace.model_validate(_tool_trace_payload(**trace_fields))


@pytest.mark.parametrize(
    "route_fields",
    [
        {
            "corrective_record_delivery": "none",
            "corrective_visible": False,
        },
        {
            "corrective_record_delivery": "corrector_private_record",
            "corrective_record_in_dossier": True,
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "agent_id": 1,
            "received_from": [0, 2, 3],
        },
        {
            "corrective_record_delivery": "relayed_record",
            "corrective_record_in_peer_message": True,
            "corrective_record_sender_ids": [1],
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "received_from": [0, 1, 3],
        },
        {
            "corrective_record_delivery": "direct_system_record",
            "corrective_record_in_direct_receipt": True,
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "received_from": [0, 1, 3],
        },
    ],
)
def test_route_metadata_valid_shapes_round_trip(route_fields):
    record = AgentRoundRecord.model_validate(_record_payload(**route_fields))
    restored = AgentRoundRecord.model_validate_json(record.model_dump_json())
    assert restored == record


@pytest.mark.parametrize(
    ("route_fields", "message"),
    [
        (
            {
                "corrective_record_delivery": "none",
                "corrective_visible": False,
                "corrective_record_id": "REC-ABC",
                "corrective_record_sha256": "a" * 64,
            },
            "delivery=none",
        ),
        (
            {
                "corrective_record_delivery": "relayed_record",
                "corrective_record_in_peer_message": True,
                "corrective_record_id": "REC-ABC",
                "corrective_record_sha256": "a" * 64,
            },
            "peer-message source",
        ),
        (
            {
                "corrective_record_delivery": "direct_system_record",
                "corrective_record_in_dossier": True,
                "corrective_record_id": "REC-ABC",
                "corrective_record_sha256": "a" * 64,
            },
            "receipt-only",
        ),
        (
            {
                "corrective_record_delivery": "corrector_private_record",
                "corrective_record_in_dossier": True,
                "corrective_record_in_direct_receipt": True,
                "corrective_record_id": "REC-ABC",
                "corrective_record_sha256": "a" * 64,
                "agent_id": 1,
            },
            "dossier-only",
        ),
    ],
)
def test_route_metadata_rejects_impossible_combinations(route_fields, message):
    with pytest.raises(ValueError, match=message):
        AgentRoundRecord.model_validate(_record_payload(**route_fields))


# ---------------------------------------------------------------------------
# Matched branching: scope, not resampling, must create arm differences
# ---------------------------------------------------------------------------


def _output_signature(result, agent_id: int, round_index: int):
    parsed = result.outputs[(agent_id, round_index)]
    return parsed.answer, parsed.rationale, parsed.raw_response


@pytest.mark.asyncio
async def test_scope_ladder_replays_one_shared_pre_correction_history(
    scenario, variant
):
    client = UniqueResponseClient()
    results = await run_correction_scope_ladder(
        scenario,
        variant,
        _frozen_seed(scenario, variant),
        client,
        topology_name="fully_connected_mesh",
        n_rounds=4,
    )
    arms = tuple(CORRECTION_SCOPE)
    available = variant.corrective_evidence.available_round

    # All agents share the exact sampled history before correction release.
    for round_index in range(available):
        for agent_id in range(4):
            signatures = {
                _output_signature(results[arm], agent_id, round_index)
                for arm in arms
            }
            assert len(signatures) == 1, (agent_id, round_index, signatures)

    # In the release round, agents who cannot yet see the record are still a
    # single replayed sample across all four arms.
    for agent_id in (0, 2, 3):
        signatures = {
            _output_signature(results[arm], agent_id, available)
            for arm in arms
        }
        assert len(signatures) == 1, (agent_id, available, signatures)

    # The corrector's sampled statement is held byte-identical in every live
    # correction arm.  Relay/direct may only alter how peers receive evidence.
    live = ("injection", "injection_relay", "injection_broadcast")
    corrector_signatures = {
        _output_signature(results[arm], 1, available) for arm in live
    }
    assert len(corrector_signatures) == 1


@pytest.mark.asyncio
async def test_scope_ladder_records_replay_origin_and_prompt_hash_invariants(
    scenario, variant
):
    results = await run_correction_scope_ladder(
        scenario,
        variant,
        _frozen_seed(scenario, variant),
        UniqueResponseClient(),
    )
    by_arm = {
        arm: {(r.agent_id, r.round): r for r in result.records}
        for arm, result in results.items()
    }
    available = variant.corrective_evidence.available_round

    for arm, grid in by_arm.items():
        seed = grid[(0, 0)]
        assert seed.response_origin == ResponseOrigin.FROZEN_SEED
        assert seed.system_prompt_sha256 and len(seed.system_prompt_sha256) == 64
        assert seed.user_prompt_sha256 and len(seed.user_prompt_sha256) == 64
        assert all(record.arm_sequence_position is not None for record in grid.values())

    # Identical histories must also have identical prompts, not merely happen
    # to produce the same answer letter.
    for round_index in range(available):
        for agent_id in range(4):
            hashes = {
                by_arm[arm][(agent_id, round_index)].user_prompt_sha256
                for arm in CORRECTION_SCOPE
            }
            assert len(hashes) == 1

    for agent_id in (0, 2, 3):
        hashes = {
            by_arm[arm][(agent_id, available)].user_prompt_sha256
            for arm in CORRECTION_SCOPE
        }
        assert len(hashes) == 1

    live = ("injection", "injection_relay", "injection_broadcast")
    corrector_hashes = {
        by_arm[arm][(1, available)].user_prompt_sha256 for arm in live
    }
    assert len(corrector_hashes) == 1

    for arm in ("injection_relay", "injection_broadcast"):
        assert by_arm[arm][(1, available)].response_origin == ResponseOrigin.MATCHED_REPLAY
        assert by_arm[arm][(1, available)].replay_source_condition.value == "injection"


@pytest.mark.asyncio
@pytest.mark.parametrize("trial_index", [0, 1, 2, 3])
async def test_scope_ladder_counterbalances_execution_order(
    scenario, variant, trial_index
):
    seed = _frozen_seed(scenario, variant).model_copy(
        update={
            "seed_instance_id": f"{variant.assumption_id}_order_seed_{trial_index:02d}",
            "seed_trial_index": trial_index,
        }
    )
    results = await run_correction_scope_ladder(
        scenario, variant, seed, UniqueResponseClient()
    )
    base_order = list(CORRECTION_SCOPE)
    rotation_key = (
        f"prompt-v1:{variant.assumption_id}:{trial_index}:"
        "fully_connected_mesh:anonymous"
    )
    rotation = int(sha256_text(rotation_key)[:8], 16) % len(base_order)
    expected_order = tuple(base_order[rotation:] + base_order[:rotation])
    assert tuple(results) == expected_order
    positions = {
        arm: result.records[0].arm_sequence_position
        for arm, result in results.items()
    }
    assert positions == {arm: i for i, arm in enumerate(expected_order, 1)}


@pytest.mark.asyncio
async def test_scope_ladder_supports_sparse_topologies_and_rejects_short_timing(
    scenario, variant
):
    seed = _frozen_seed(scenario, variant)
    for topology_name in ("looped_chain", "neutral_star"):
        results = await run_correction_scope_ladder(
            scenario,
            variant,
            seed,
            UniqueResponseClient(),
            topology_name=topology_name,
        )
        assert set(results) == set(CORRECTION_SCOPE)
    for insufficient_rounds in (
        variant.corrective_evidence.available_round + 1,
        variant.corrective_evidence.available_round + 2,
    ):
        with pytest.raises(ValueError, match="peer-response rounds"):
            await run_correction_scope_ladder(
                scenario,
                variant,
                seed,
                UniqueResponseClient(),
                n_rounds=insufficient_rounds,
            )


@pytest.mark.parametrize(
    "route_fields",
    [
        {
            "corrective_record_delivery": "none",
            "corrective_visible": True,
        },
        {
            "corrective_record_delivery": "relayed_record",
            "corrective_visible": False,
            "corrective_record_in_peer_message": True,
            "corrective_record_sender_ids": [1],
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "received_from": [0, 1, 3],
        },
        {
            "corrective_record_delivery": "corrector_private_record",
            "corrective_record_in_dossier": True,
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "agent_id": 2,
        },
        {
            "corrective_record_delivery": "relayed_record",
            "corrective_record_in_peer_message": True,
            "corrective_record_sender_ids": [3],
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "received_from": [0, 1, 3],
        },
        {
            "corrective_record_delivery": "relayed_record",
            "corrective_record_in_peer_message": True,
            "corrective_record_sender_ids": [1],
            "corrective_record_id": "REC-ABC",
            "corrective_record_sha256": "a" * 64,
            "received_from": [0, 3],
        },
    ],
)
def test_route_metadata_rejects_visibility_role_and_routing_contradictions(
    route_fields,
):
    with pytest.raises(ValueError):
        AgentRoundRecord.model_validate(_record_payload(**route_fields))
