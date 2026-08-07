"""Orchestrator: Stage A seed generation and Stage B matched propagation.

Invariants enforced here (and verified in tests/test_orchestrator.py):
1. FROZEN SEED REPLAY — within a matched triple, the seed's round-0 output
   is byte-identical across chain, star, and mesh; it is never regenerated.
2. SYNCHRONOUS ROUNDS — a round's outputs are all committed before any
   agent can see them; agents receive only previous-round messages, routed
   exclusively through topologies.senders_for().
3. NOTHING DISCARDED — parse failures are logged records, not exceptions;
   every valid seed instance is retained regardless of its stance.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .clients import (
    ForcedToolTrace,
    RATIONALE_AUDIT_VERSION,
    ParsedResponse,
    ToolExecutionResult,
    ToolTurnResult,
    audit_answer_rationale,
    parse_or_repair,
)
from .evidence import CorrectiveRecordStore, sha256_text
from .prompts import (
    ATTRIBUTION_MODES,
    assignment_for_agent,
    build_private_dossier,
    build_round0_prompt,
    build_seed_generation_system_prompt,
    build_system_prompt,
    build_update_prompt,
    canonical_tool_placebo_payload,
    canonical_tool_record_payload,
    format_direct_record_receipt,
    format_peer_message,
    format_prompt_record_receipt_control,
    READ_CASE_RECORD_TOOL,
)
from .schemas import (
    CORRECTION_SCOPE,
    INJECTION_CONDITIONS,
    TOOL_CORRECTION_CONTROLS,
    TOOL_CORRECTION_SCOPE,
    AgentOutput,
    AgentRoundRecord,
    Condition,
    CorrectionDelivery,
    ResponseOrigin,
    SeedInstance,
    ToolCallTrace,
    Topology,
)
from .topologies import (
    AGENT_IDS,
    DEFAULT_ROLES,
    corrector_for_seed,
    senders_for,
    validate_roles,
    validate_topology,
)

N_ROUNDS = 4  # round 0 + 3 synchronous updates


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

def append_jsonl(path: str | Path, record) -> None:
    """Append one pydantic record to a JSONL file (auditable raw log)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Stage A — seed generation
# ---------------------------------------------------------------------------

async def generate_seed_instances(
    scenario,
    variant,
    client,
    *,
    n_instances: int = 3,
    log_path: Optional[str | Path] = None,
) -> list[SeedInstance]:
    """Generate n independent seed responses. EVERY valid response is
    retained in fixed run order — no acceptance-based filtering.

    Uses an ID-neutral system prompt: the frozen response is replayed into
    different agent positions per topology, so it must not be authored as
    any particular agent.
    """
    system_prompt = build_seed_generation_system_prompt()
    user_prompt = build_round0_prompt(
        scenario, is_seed=True, condition="injection", variant=variant
    )
    prompt_hash = _hash(system_prompt + "\n" + user_prompt)

    instances: list[SeedInstance] = []
    for trial in range(n_instances):
        try:
            raw = await client.generate(system_prompt, user_prompt)
            parsed = parse_or_repair(raw)
        except Exception as exc:
            parsed = ParsedResponse(
                raw_response="",
                parse_failed=True,
                generation_failed=True,
                generation_error_type=type(exc).__name__,
                generation_error_message=str(exc)[:1200],
            )
        rationale_audit = (
            audit_answer_rationale(parsed.answer, parsed.rationale, scenario.options)
            if not parsed.parse_failed
            else None
        )
        if parsed.parse_failed:
            # Recorded, not silently dropped; excluded from replay because
            # there is no message to propagate.
            instance = SeedInstance(
                seed_instance_id=f"{variant.assumption_id}_seed_{trial:02d}",
                scenario_id=scenario.scenario_id,
                assumption_id=variant.assumption_id,
                seed_trial_index=trial,
                injection_prompt_hash=prompt_hash,
                answer="",
                rationale="",
                raw_response=parsed.raw_response,
                generation_failed=parsed.generation_failed,
                generation_error_type=parsed.generation_error_type,
                generation_error_message=parsed.generation_error_message,
                answer_rationale_inconsistent=None,
                answer_rationale_audit_version=None,
                rationale_conclusion_letters=[],
                included_in_core=False,
            )
        else:
            instance = SeedInstance(
                seed_instance_id=f"{variant.assumption_id}_seed_{trial:02d}",
                scenario_id=scenario.scenario_id,
                assumption_id=variant.assumption_id,
                seed_trial_index=trial,
                injection_prompt_hash=prompt_hash,
                answer=parsed.answer,
                rationale=parsed.rationale,
                raw_response=parsed.raw_response,
                generation_failed=parsed.generation_failed,
                generation_error_type=parsed.generation_error_type,
                generation_error_message=parsed.generation_error_message,
                answer_rationale_inconsistent=rationale_audit.inconsistent,
                answer_rationale_audit_version=RATIONALE_AUDIT_VERSION,
                rationale_conclusion_letters=rationale_audit.conclusion_letters,
            )
        instances.append(instance)
        if log_path:
            append_jsonl(log_path, instance)
    return instances


# ---------------------------------------------------------------------------
# Stage B — one conversation (propagation or baseline)
# ---------------------------------------------------------------------------

@dataclass
class ConversationOutputs:
    """Everything one conversation produced."""

    run_id: str
    records: list[AgentRoundRecord]
    # (agent_id, round) -> ParsedResponse; the full message history.
    outputs: dict[tuple[int, int], ParsedResponse]
    n_rounds: int = N_ROUNDS
    # Genuine tool traces are separate from model text so a hidden matched
    # anchor can replay one sampled tool-backed response without pretending
    # that each measured arm issued another backend request.
    tool_traces: dict[tuple[int, int], ToolCallTrace] = field(default_factory=dict)

    def final_answers(self) -> dict[int, Optional[str]]:
        final_round = self.n_rounds - 1
        final_records = {
            record.agent_id: record
            for record in self.records
            if record.round == final_round
        }
        return {
            agent_id: (
                None
                if final_records[agent_id].parse_failed
                else self.outputs[(agent_id, final_round)].answer
            )
            for agent_id in AGENT_IDS
        }


def _optional_sum(*values: Optional[int]) -> Optional[int]:
    """Sum usage fields only when the provider supplied every component."""
    return sum(values) if all(value is not None for value in values) else None


def _tool_trace_model(
    trace: ForcedToolTrace,
    *,
    replayed: bool,
) -> ToolCallTrace:
    """Convert a client trace into the stable raw-log schema.

    Replayed cells preserve the source call/result/transcript hashes and call
    id but record zero API calls and zero token use in the measured arm.  This
    makes the one hidden-anchor execution distinguishable from copies of its
    response and provenance.
    """
    transcript = json.dumps(
        {
            "arguments_json": trace.arguments_json,
            "final_response_id": trace.final.response_id,
            "request_response_id": trace.request.response_id,
            "tool_call_id": trace.tool_call_id,
            "tool_definition_sha256": trace.tool_definition_sha256,
            "tool_name": trace.tool_name,
            "tool_result_sha256": trace.tool_result_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if replayed:
        api_call_count = 0
        input_tokens = output_tokens = total_tokens = 0
        finish_reasons: list[str] = []
    else:
        api_call_count = trace.api_call_attempts
        input_tokens = _optional_sum(
            trace.request.prompt_tokens, trace.final.prompt_tokens
        )
        output_tokens = _optional_sum(
            trace.request.completion_tokens, trace.final.completion_tokens
        )
        total_tokens = _optional_sum(trace.request.total_tokens, trace.final.total_tokens)
        finish_reasons = [
            reason
            for reason in (trace.request.finish_reason, trace.final.finish_reason)
            if reason
        ]
    return ToolCallTrace(
        tool_name=trace.tool_name,
        tool_call_id=trace.tool_call_id,
        arguments_sha256=trace.arguments_sha256,
        result_sha256=trace.tool_result_sha256,
        success=True,
        api_call_count=api_call_count,
        finish_reasons=finish_reasons,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        transcript_sha256=sha256_text(transcript),
    )


def _replayed_tool_trace(trace: ToolCallTrace) -> ToolCallTrace:
    """Copy source hashes while recording zero work in this measured cell."""
    return trace.model_copy(
        update={
            "api_call_count": 0,
            "finish_reasons": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )


async def run_conversation(
    scenario,
    *,
    client,
    topology_name: str,
    condition: str,
    variant=None,
    frozen_seed: Optional[SeedInstance] = None,
    seed_agent_id: Optional[int] = None,
    corrector_agent_id: Optional[int] = None,
    n_rounds: int = N_ROUNDS,
    withhold_corrective: bool = False,
    attribution: str = "anonymous",
    log_path: Optional[str | Path] = None,
    replay_outputs: Optional[dict[tuple[int, int], ParsedResponse]] = None,
    replay_source_conditions: Optional[dict[tuple[int, int], str]] = None,
    replay_tool_traces: Optional[dict[tuple[int, int], ToolCallTrace]] = None,
    arm_sequence_position: Optional[int] = None,
) -> ConversationOutputs:
    """Run one synchronous four-agent conversation.

    ``replay_outputs`` is used only by matched scope-ladder drivers. It reuses
    responses from conversation cells whose prompts are intentionally
    identical across arms, preventing model resampling from masquerading as a
    treatment effect. Every replay must name its source condition. Protocol-v2
    tool cells additionally copy their source trace through
    ``replay_tool_traces`` while recording that no new tool/API call executed.
    """
    try:
        condition_value = Condition(condition)
    except ValueError as exc:
        raise ValueError(f"unknown experimental condition: {condition!r}") from exc
    validate_topology(topology_name)
    if not 1 <= n_rounds <= N_ROUNDS:
        raise ValueError(f"n_rounds must be between 1 and {N_ROUNDS}, got {n_rounds}")
    default_seed, _ = DEFAULT_ROLES[topology_name]
    if seed_agent_id is None:
        seed_agent_id = default_seed
    if corrector_agent_id is None:
        corrector_agent_id = corrector_for_seed(topology_name, seed_agent_id)
    validate_roles(topology_name, seed_agent_id, corrector_agent_id)

    is_injection = condition in INJECTION_CONDITIONS
    is_tool_protocol = condition in (*TOOL_CORRECTION_SCOPE, *TOOL_CORRECTION_CONTROLS)
    if is_tool_protocol and not callable(
        getattr(client, "generate_with_forced_tool", None)
    ):
        raise TypeError(
            "protocol-v2 conditions require client.generate_with_forced_tool"
        )
    if condition in ("injection_no_correction", "injection_tool_none"):
        withhold_corrective = True
    if is_injection and frozen_seed is None:
        raise ValueError("injection conversations require a frozen_seed to replay")
    if not is_injection and frozen_seed is not None:
        raise ValueError("frozen_seed is only valid for the injection conditions")
    if is_injection:
        if variant is None:
            raise ValueError("injection conversations require an assumption variant")
        if frozen_seed.scenario_id != scenario.scenario_id:
            raise ValueError(
                "frozen seed scenario mismatch: "
                f"seed={frozen_seed.scenario_id!r}, active={scenario.scenario_id!r}"
            )
        if frozen_seed.assumption_id != variant.assumption_id:
            raise ValueError(
                "frozen seed assumption mismatch: "
                f"seed={frozen_seed.assumption_id!r}, active={variant.assumption_id!r}"
            )
        if not frozen_seed.answer or not frozen_seed.included_in_core:
            raise ValueError("frozen seed must contain a valid replayable answer")

    replay_outputs = dict(replay_outputs or {})
    replay_source_conditions = dict(replay_source_conditions or {})
    replay_tool_traces = dict(replay_tool_traces or {})
    if set(replay_outputs) != set(replay_source_conditions):
        raise ValueError(
            "replay_outputs and replay_source_conditions must have identical keys"
        )
    if not set(replay_tool_traces).issubset(replay_outputs):
        raise ValueError("replay_tool_traces keys must be a subset of replay_outputs")
    for (agent_id, round_index), source_condition in replay_source_conditions.items():
        if agent_id not in AGENT_IDS or not 0 <= round_index < n_rounds:
            raise ValueError(
                f"replay cell {(agent_id, round_index)!r} is outside the conversation grid"
            )
        try:
            Condition(source_condition)
        except ValueError as exc:
            raise ValueError(
                f"unknown replay source condition: {source_condition!r}"
            ) from exc
    if arm_sequence_position is not None and arm_sequence_position < 1:
        raise ValueError("arm_sequence_position must be positive")

    neutral_ids = [a for a in AGENT_IDS if a not in (seed_agent_id, corrector_agent_id)]

    def neutral_ordinal_of(agent_id: int) -> Optional[int]:
        return neutral_ids.index(agent_id) if agent_id in neutral_ids else None

    # Speaker attribution. Orthogonal to record provenance: this names WHO is
    # speaking, the case-record blocks name WHERE the evidence came from.
    if attribution not in ATTRIBUTION_MODES:
        raise ValueError(
            f"unknown attribution mode {attribution!r}; expected one of "
            f"{ATTRIBUTION_MODES}"
        )
    assignments: dict[int, Optional[str]] = {}
    if attribution == "assigned":
        for a in AGENT_IDS:
            assignments[a] = assignment_for_agent(
                scenario,
                is_seed=(a == seed_agent_id),
                is_corrector=(a == corrector_agent_id),
                neutral_ordinal=neutral_ordinal_of(a),
            )
        # Track B only. Failing loudly beats rendering "Agent 2 ()" for half the
        # group, which would make the factor mean different things per scenario.
        missing = [a for a, label in assignments.items() if not label]
        if missing:
            raise ValueError(
                f"attribution='assigned' but {scenario.scenario_id} has no parsable "
                f"assignment for agent(s) {missing}; hidden-profile briefs must open "
                f"'Your <role> assignment was <Entity>.'"
            )

    run_id = new_run_id()
    outputs: dict[tuple[int, int], ParsedResponse] = {}
    tool_traces: dict[tuple[int, int], ToolCallTrace] = {}
    records: list[AgentRoundRecord] = []
    record_store = CorrectiveRecordStore.from_variant(variant) if variant else None

    def tool_executor(
        *, agent_id: int, round_index: int, placebo: bool
    ):
        """Bind the active case/recipient outside model-controlled arguments."""
        if record_store is None:
            raise RuntimeError("tool access requires a corrective record store")

        def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
            if arguments:
                raise ValueError("read_case_record accepts no model-supplied arguments")
            if placebo:
                return ToolExecutionResult(
                    content=canonical_tool_placebo_payload(record_store.record),
                    metadata={
                        "delivery": CorrectionDelivery.TOOL_PLACEBO.value,
                        "recipient_agent_id": agent_id,
                        "round_index": round_index,
                        "record_id": None,
                        "record_sha256": None,
                    },
                )
            receipt = record_store.retrieve(
                record_store.record.record_id,
                recipient_agent_id=agent_id,
                round_index=round_index,
            )
            return ToolExecutionResult(
                content=canonical_tool_record_payload(receipt.record),
                metadata={
                    "delivery": (
                        CorrectionDelivery.CORRECTOR_TOOL_RECORD.value
                        if agent_id == corrector_agent_id
                        else CorrectionDelivery.DIRECT_TOOL_RECORD.value
                    ),
                    "recipient_agent_id": receipt.recipient_agent_id,
                    "round_index": receipt.round_index,
                    "record_id": receipt.record.record_id,
                    "record_sha256": receipt.record.sha256,
                },
            )

        return execute

    def make_record(
        agent_id: int,
        round_index: int,
        parsed: ParsedResponse,
        received_from: list[int],
        *,
        injection_visible: bool,
        neutral_control_visible: bool,
        corrective_visible: bool,
        delivery: CorrectionDelivery,
        record_in_dossier: bool,
        record_in_peer_message: bool,
        record_in_direct_receipt: bool,
        record_sender_ids: list[int],
        corrector_message_received: bool,
        corrector_held_record_when_speaking: bool,
        system_prompt: str,
        user_prompt: str,
        response_origin: ResponseOrigin,
        replay_source_condition: Optional[str],
        tool_trace: Optional[ToolCallTrace],
        tool_call_executed: Optional[bool],
        tool_trace_replayed: Optional[bool],
    ) -> AgentRoundRecord:
        rationale_audit = (
            audit_answer_rationale(parsed.answer, parsed.rationale, scenario.options)
            if not parsed.parse_failed
            else None
        )
        record_id = None
        record_sha256 = None
        real_record_deliveries = {
            CorrectionDelivery.CORRECTOR_PRIVATE,
            CorrectionDelivery.RELAYED_RECORD,
            CorrectionDelivery.DIRECT_SYSTEM_RECORD,
            CorrectionDelivery.CORRECTOR_TOOL_RECORD,
            CorrectionDelivery.DIRECT_TOOL_RECORD,
            CorrectionDelivery.PROMPT_RECEIPT_CONTROL,
        }
        if delivery in real_record_deliveries:
            if record_store is None:
                raise RuntimeError("correction delivery requires a record store")
            record_id = record_store.record.record_id
            record_sha256 = record_store.record.sha256
        return AgentRoundRecord(
            run_id=run_id,
            stage="propagation",
            scenario_id=scenario.scenario_id,
            assumption_id=variant.assumption_id if variant is not None else None,
            seed_instance_id=frozen_seed.seed_instance_id if frozen_seed else None,
            condition=condition_value,
            topology=Topology(topology_name),
            model=client.model_identifier,
            # Provenance must reflect what the client actually sampled with;
            # None means the deployment default was used.
            temperature=getattr(client, "temperature", None),
            seed_agent_id=seed_agent_id,
            corrector_agent_id=corrector_agent_id,
            agent_id=agent_id,
            round=round_index,
            attribution=attribution,
            injection_visible=injection_visible,
            neutral_control_visible=neutral_control_visible,
            corrective_visible=corrective_visible,
            corrective_record_delivery=delivery,
            corrective_record_in_dossier=record_in_dossier,
            corrective_record_in_peer_message=record_in_peer_message,
            corrective_record_in_direct_receipt=record_in_direct_receipt,
            corrective_record_sender_ids=record_sender_ids,
            corrective_record_id=record_id,
            corrective_record_sha256=record_sha256,
            corrector_message_received=corrector_message_received,
            corrector_held_record_when_speaking=corrector_held_record_when_speaking,
            system_prompt_sha256=sha256_text(system_prompt),
            user_prompt_sha256=sha256_text(user_prompt),
            response_origin=response_origin,
            replay_source_condition=(
                Condition(replay_source_condition)
                if replay_source_condition is not None
                else None
            ),
            arm_sequence_position=arm_sequence_position,
            tool_trace=tool_trace,
            tool_call_executed=tool_call_executed,
            tool_trace_replayed=tool_trace_replayed,
            received_from=received_from,
            messages_received=len(received_from),
            answer=parsed.answer,
            rationale=parsed.rationale,
            raw_response=parsed.raw_response,
            parse_failed=parsed.parse_failed,
            parse_repaired=parsed.parse_repaired,
            # On an update failure, the orchestrator deliberately replays the
            # previous valid output so the conversation can continue.  Mark
            # that imputation explicitly so expression-level analyses do not
            # mistake it for a newly generated rationale.
            carried_forward=(parsed.parse_failed and round_index > 0),
            generation_failed=parsed.generation_failed,
            generation_error_type=parsed.generation_error_type,
            generation_error_message=parsed.generation_error_message,
            answer_rationale_inconsistent=(
                rationale_audit.inconsistent if rationale_audit is not None else None
            ),
            answer_rationale_audit_version=(
                RATIONALE_AUDIT_VERSION if rationale_audit is not None else None
            ),
            rationale_conclusion_letters=(
                rationale_audit.conclusion_letters if rationale_audit is not None else []
            ),
        )

    for round_index in range(n_rounds):
        round_record_start = len(records)
        plans: dict[int, dict] = {}
        tasks = []
        generating_agents: list[int] = []

        for agent_id in AGENT_IDS:
            injection_visible = is_injection and agent_id == seed_agent_id
            neutral_control_visible = (
                condition in ("no_injection_baseline", "independent_control")
                and agent_id == seed_agent_id
            )
            received_from = (
                [] if round_index == 0
                else senders_for(topology_name, agent_id, round_index)
            )
            available_round = (
                variant.corrective_evidence.available_round if variant is not None else None
            )
            legacy_corrector_record = bool(
                available_round is not None
                and not withhold_corrective
                and agent_id == corrector_agent_id
                and round_index >= available_round
                and not is_tool_protocol
            )
            corrector_tool = bool(
                is_tool_protocol
                and available_round is not None
                and not withhold_corrective
                and agent_id == corrector_agent_id
                and round_index >= available_round
            )
            record_sender_ids = []
            if (
                condition in ("injection_relay", "injection_tool_relay")
                and available_round is not None
                and not withhold_corrective
                and round_index >= available_round + 1
                and corrector_agent_id in received_from
            ):
                record_sender_ids = [corrector_agent_id]
            legacy_direct_record = bool(
                condition == "injection_broadcast"
                and available_round is not None
                and not withhold_corrective
                and agent_id != corrector_agent_id
                and round_index >= available_round + 1
            )
            direct_tool = bool(
                condition == "injection_tool_direct"
                and available_round is not None
                and not withhold_corrective
                and agent_id != corrector_agent_id
                and round_index >= available_round + 1
            )
            placebo_tool = bool(
                condition == "injection_tool_placebo"
                and available_round is not None
                and not withhold_corrective
                and agent_id != corrector_agent_id
                and round_index >= available_round + 1
            )
            prompt_receipt_control = bool(
                condition == "injection_prompt_receipt_control"
                and available_round is not None
                and not withhold_corrective
                and agent_id != corrector_agent_id
                and round_index >= available_round + 1
            )
            if corrector_tool:
                delivery = CorrectionDelivery.CORRECTOR_TOOL_RECORD
            elif legacy_corrector_record:
                delivery = CorrectionDelivery.CORRECTOR_PRIVATE
            elif record_sender_ids:
                delivery = CorrectionDelivery.RELAYED_RECORD
            elif direct_tool:
                delivery = CorrectionDelivery.DIRECT_TOOL_RECORD
            elif placebo_tool:
                delivery = CorrectionDelivery.TOOL_PLACEBO
            elif prompt_receipt_control:
                delivery = CorrectionDelivery.PROMPT_RECEIPT_CONTROL
            elif legacy_direct_record:
                delivery = CorrectionDelivery.DIRECT_SYSTEM_RECORD
            else:
                delivery = CorrectionDelivery.NONE
            record_in_dossier = delivery == CorrectionDelivery.CORRECTOR_PRIVATE
            record_in_peer_message = delivery == CorrectionDelivery.RELAYED_RECORD
            record_in_direct_receipt = delivery in (
                CorrectionDelivery.DIRECT_SYSTEM_RECORD,
                CorrectionDelivery.PROMPT_RECEIPT_CONTROL,
            )
            corrective_visible = delivery not in (
                CorrectionDelivery.NONE,
                CorrectionDelivery.TOOL_PLACEBO,
            )
            corrector_message_received = (
                round_index > 0 and corrector_agent_id in received_from
            )
            corrector_held_when_speaking = bool(
                corrector_message_received
                and available_round is not None
                and not withhold_corrective
                and round_index - 1 >= available_round
            )

            if round_index == 0:
                prompt = build_round0_prompt(
                    scenario,
                    is_seed=(agent_id == seed_agent_id),
                    is_corrector=(agent_id == corrector_agent_id),
                    condition=condition,
                    variant=variant,
                    round_index=round_index,
                    neutral_ordinal=neutral_ordinal_of(agent_id),
                    withhold_corrective=withhold_corrective,
                )
            else:
                own_prev = outputs[(agent_id, round_index - 1)]
                peer_messages = [
                    format_peer_message(
                        sender,
                        outputs[(sender, round_index - 1)].answer or "(no valid answer)",
                        outputs[(sender, round_index - 1)].rationale or "(no rationale)",
                        relayed_record=(
                            record_store.record
                            if (
                                condition == "injection_relay"
                                and sender in record_sender_ids
                                and record_store is not None
                            )
                            else None
                        ),
                        relayed_tool_payload=(
                            canonical_tool_record_payload(record_store.record)
                            if (
                                condition == "injection_tool_relay"
                                and sender in record_sender_ids
                                and record_store is not None
                            )
                            else None
                        ),
                        assignment=assignments.get(sender),
                    )
                    for sender in received_from
                ]
                private_dossier = build_private_dossier(
                    scenario,
                    is_seed=(agent_id == seed_agent_id),
                    is_corrector=(agent_id == corrector_agent_id),
                    condition=condition,
                    variant=variant,
                    round_index=round_index,
                    neutral_ordinal=neutral_ordinal_of(agent_id),
                    withhold_corrective=withhold_corrective,
                )
                direct_receipt = None
                if legacy_direct_record:
                    if record_store is None:
                        raise RuntimeError("direct delivery requires a record store")
                    direct_receipt = format_direct_record_receipt(
                        record_store.retrieve(
                            record_store.record.record_id,
                            recipient_agent_id=agent_id,
                            round_index=round_index,
                        )
                    )
                elif prompt_receipt_control:
                    if record_store is None:
                        raise RuntimeError("prompt receipt requires a record store")
                    direct_receipt = format_prompt_record_receipt_control(
                        canonical_tool_record_payload(record_store.record)
                    )
                prompt = build_update_prompt(
                    scenario,
                    own_previous_answer=own_prev.answer or "(no valid answer)",
                    own_previous_rationale=own_prev.rationale or "(no rationale)",
                    peer_messages=peer_messages,
                    round_index=round_index,
                    private_note=private_dossier,
                    direct_record_receipt=direct_receipt,
                )

            system_prompt = (
                build_seed_generation_system_prompt()
                if round_index == 0 and agent_id == seed_agent_id
                else build_system_prompt(agent_id)
            )
            plan = {
                "received_from": received_from,
                "injection_visible": injection_visible,
                "neutral_control_visible": neutral_control_visible,
                "corrective_visible": corrective_visible,
                "delivery": delivery,
                "record_in_dossier": record_in_dossier,
                "record_in_peer_message": record_in_peer_message,
                "record_in_direct_receipt": record_in_direct_receipt,
                "record_sender_ids": record_sender_ids,
                "corrector_message_received": corrector_message_received,
                "corrector_held_record_when_speaking": corrector_held_when_speaking,
                "system_prompt": system_prompt,
                "user_prompt": prompt,
                "parsed": None,
                "response_origin": ResponseOrigin.GENERATED,
                "replay_source_condition": None,
                "tool_trace": None,
                "tool_call_executed": None,
                "tool_trace_replayed": None,
            }
            replay_key = (agent_id, round_index)
            if round_index == 0 and is_injection and agent_id == seed_agent_id:
                plan["parsed"] = ParsedResponse(
                    raw_response=frozen_seed.raw_response,
                    output=AgentOutput(
                        answer=frozen_seed.answer, rationale=frozen_seed.rationale
                    ),
                )
                plan["response_origin"] = ResponseOrigin.FROZEN_SEED
            elif replay_key in replay_outputs:
                plan["parsed"] = replay_outputs[replay_key]
                plan["response_origin"] = ResponseOrigin.MATCHED_REPLAY
                plan["replay_source_condition"] = replay_source_conditions[replay_key]
                if replay_key in replay_tool_traces:
                    replayed_trace = _replayed_tool_trace(
                        replay_tool_traces[replay_key]
                    )
                    plan["tool_trace"] = replayed_trace
                    plan["tool_call_executed"] = False
                    plan["tool_trace_replayed"] = True
                    tool_traces[replay_key] = replayed_trace
                elif plan["delivery"] in (
                    CorrectionDelivery.CORRECTOR_TOOL_RECORD,
                    CorrectionDelivery.DIRECT_TOOL_RECORD,
                    CorrectionDelivery.TOOL_PLACEBO,
                ):
                    # The anchor's own forced tool call failed and demoted to
                    # NONE with no trace; replaying arms must inherit that
                    # demotion rather than keep a tool-delivery label with no
                    # tool_trace to back it.
                    plan.update(
                        corrective_visible=False,
                        delivery=CorrectionDelivery.NONE,
                        record_in_dossier=False,
                        record_in_peer_message=False,
                        record_in_direct_receipt=False,
                        record_sender_ids=[],
                        corrector_held_record_when_speaking=False,
                    )
            else:
                if corrector_tool or direct_tool or placebo_tool:
                    tasks.append(
                        client.generate_with_forced_tool(
                            system_prompt,
                            prompt,
                            tool=READ_CASE_RECORD_TOOL,
                            execute=tool_executor(
                                agent_id=agent_id,
                                round_index=round_index,
                                placebo=placebo_tool,
                            ),
                        )
                    )
                else:
                    tasks.append(client.generate(system_prompt, prompt))
                generating_agents.append(agent_id)
            plans[agent_id] = plan

        # SYNCHRONOUS COMMIT: all generated cells finish before any round-r
        # output becomes visible to a round-r peer.
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        for agent_id, raw in zip(generating_agents, raw_results):
            if isinstance(raw, Exception):
                # A tool protocol failure is a failed generation, not evidence
                # delivery.  Log the cell with an explicit NONE route; run
                # integrity checks can then reject a supposedly complete arm
                # that lacks its required tool interaction.
                if plans[agent_id]["delivery"] in (
                    CorrectionDelivery.CORRECTOR_TOOL_RECORD,
                    CorrectionDelivery.DIRECT_TOOL_RECORD,
                    CorrectionDelivery.TOOL_PLACEBO,
                ):
                    plans[agent_id].update(
                        corrective_visible=False,
                        delivery=CorrectionDelivery.NONE,
                        record_in_dossier=False,
                        record_in_peer_message=False,
                        record_in_direct_receipt=False,
                        record_sender_ids=[],
                        corrector_held_record_when_speaking=False,
                        tool_trace=None,
                        tool_call_executed=None,
                        tool_trace_replayed=None,
                    )
                parsed = ParsedResponse(
                    raw_response="",
                    parse_failed=True,
                    generation_failed=True,
                    generation_error_type=type(raw).__name__,
                    generation_error_message=str(raw)[:1200],
                )
            else:
                if isinstance(raw, ToolTurnResult):
                    trace = _tool_trace_model(raw.trace, replayed=False)
                    plans[agent_id]["tool_trace"] = trace
                    plans[agent_id]["tool_call_executed"] = True
                    plans[agent_id]["tool_trace_replayed"] = False
                    tool_traces[(agent_id, round_index)] = trace
                    raw_text = raw.final_text
                else:
                    raw_text = raw
                parsed = parse_or_repair(raw_text)
            plans[agent_id]["parsed"] = parsed

        for agent_id in AGENT_IDS:
            plan = plans[agent_id]
            parsed = plan["parsed"]
            if parsed is None:
                raise RuntimeError("turn plan was not resolved")
            if parsed.parse_failed and round_index > 0:
                prev = outputs[(agent_id, round_index - 1)]
                parsed = ParsedResponse(
                    raw_response=parsed.raw_response,
                    output=prev.output,
                    parse_failed=True,
                    generation_failed=parsed.generation_failed,
                    generation_error_type=parsed.generation_error_type,
                    generation_error_message=parsed.generation_error_message,
                )
            outputs[(agent_id, round_index)] = parsed
            records.append(
                make_record(
                    agent_id,
                    round_index,
                    parsed,
                    plan["received_from"],
                    injection_visible=plan["injection_visible"],
                    neutral_control_visible=plan["neutral_control_visible"],
                    corrective_visible=plan["corrective_visible"],
                    delivery=plan["delivery"],
                    record_in_dossier=plan["record_in_dossier"],
                    record_in_peer_message=plan["record_in_peer_message"],
                    record_in_direct_receipt=plan["record_in_direct_receipt"],
                    record_sender_ids=plan["record_sender_ids"],
                    corrector_message_received=plan["corrector_message_received"],
                    corrector_held_record_when_speaking=(
                        plan["corrector_held_record_when_speaking"]
                    ),
                    system_prompt=plan["system_prompt"],
                    user_prompt=plan["user_prompt"],
                    response_origin=plan["response_origin"],
                    replay_source_condition=plan["replay_source_condition"],
                    tool_trace=plan["tool_trace"],
                    tool_call_executed=plan["tool_call_executed"],
                    tool_trace_replayed=plan["tool_trace_replayed"],
                )
            )

        # Commit each complete synchronous round immediately. If a later API
        # call exhausts retries or the process stops, earlier paid responses
        # remain auditable and analysis will flag the incomplete round grid.
        if log_path:
            for record in records[round_record_start:]:
                append_jsonl(log_path, record)

    return ConversationOutputs(
        run_id=run_id,
        records=records,
        outputs=outputs,
        n_rounds=n_rounds,
        tool_traces=tool_traces,
    )


async def run_correction_scope_ladder(
    scenario,
    variant,
    frozen_seed: SeedInstance,
    client,
    *,
    topology_name: str = "fully_connected_mesh",
    n_rounds: int = N_ROUNDS,
    attribution: str = "anonymous",
    log_path: Optional[str | Path] = None,
) -> dict[str, ConversationOutputs]:
    """Run four matched correction-scope arms with a shared prefix.

    A common pre-correction history is sampled once and replayed into every arm
    within one topology/attribution coordinate. At the correction-release round,
    every non-corrector response is also replayed because those agents have
    identical prompts. Hidden matched anchors sample the verifier once with and
    once without the record; measured arms replay those anchors and may run in
    rotated order. Thus the first point at which arms can differ is the intended
    evidence-delivery channel. Sparse topologies intentionally limit which
    recipients hear conversation-carried testimony or relay payloads; direct
    delivery remains topology-independent.
    """
    validate_topology(topology_name)
    available = variant.corrective_evidence.available_round
    if n_rounds < available + 3:
        raise ValueError(
            "scope ladder requires two peer-response rounds after correction: "
            f"available_round={available}, n_rounds={n_rounds}"
        )
    seed_id, _ = DEFAULT_ROLES[topology_name]
    corrector_id = corrector_for_seed(topology_name, seed_id)

    prefix_outputs: dict[tuple[int, int], ParsedResponse] = {}
    if available > 0:
        prefix = await run_conversation(
            scenario,
            client=client,
            topology_name=topology_name,
            condition="injection_no_correction",
            variant=variant,
            frozen_seed=frozen_seed,
            n_rounds=available,
            withhold_corrective=True,
            attribution=attribution,
        )
        prefix_outputs = {
            key: value
            for key, value in prefix.outputs.items()
            if key != (seed_id, 0)
        }

    # Build two hidden anchors. They sample the verifier once without the
    # record and once with it, while replaying every unaffected release-round
    # response. Actual measured arms branch from these anchors, so none of the
    # four must run first merely to supply history to another.
    anchor_key = (
        f"prompt-v1-anchor:{variant.assumption_id}:"
        f"{frozen_seed.seed_trial_index}:{topology_name}:{attribution}"
    )
    parity = int(sha256_text(anchor_key)[:8], 16) % 2
    anchor_order = (
        ["injection_no_correction", "injection"]
        if parity == 0
        else ["injection", "injection_no_correction"]
    )
    anchors: dict[str, ConversationOutputs] = {}
    for arm in anchor_order:
        replay = dict(prefix_outputs)
        sources = {key: "injection_no_correction" for key in replay}
        if anchors:
            first_arm = anchor_order[0]
            first_result = anchors[first_arm]
            for agent_id in AGENT_IDS:
                if agent_id == corrector_id or (agent_id, available) == (seed_id, 0):
                    continue
                key = (agent_id, available)
                replay[key] = first_result.outputs[key]
                sources[key] = first_arm
        anchors[arm] = await run_conversation(
            scenario,
            client=client,
            topology_name=topology_name,
            condition=arm,
            variant=variant,
            frozen_seed=frozen_seed,
            n_rounds=available + 2,
            withhold_corrective=(arm == "injection_no_correction"),
            replay_outputs=replay,
            replay_source_conditions=sources,
            attribution=attribution,
        )

    # Rotate all four measured arms, rather than always collecting the wider
    # routes last. The stable assumption/seed hash spreads rotations across the
    # corpus while preserving exact reproducibility.
    base_order = list(CORRECTION_SCOPE)
    rotation_key = (
        f"prompt-v1:{variant.assumption_id}:{frozen_seed.seed_trial_index}:"
        f"{topology_name}:{attribution}"
    )
    rotation = int(sha256_text(rotation_key)[:8], 16) % len(base_order)
    execution_order = base_order[rotation:] + base_order[:rotation]
    results: dict[str, ConversationOutputs] = {}
    for position, arm in enumerate(execution_order, 1):
        anchor_condition = (
            "injection_no_correction"
            if arm == "injection_no_correction"
            else "injection"
        )
        anchor = anchors[anchor_condition]
        replay: dict[tuple[int, int], ParsedResponse] = {}
        for key, value in anchor.outputs.items():
            agent_id, round_index = key
            if round_index <= available or (
                agent_id == corrector_id and round_index == available + 1
            ):
                if key != (seed_id, 0):
                    replay[key] = value
        sources = {key: anchor_condition for key in replay}
        results[arm] = await run_conversation(
            scenario,
            client=client,
            topology_name=topology_name,
            condition=arm,
            variant=variant,
            frozen_seed=frozen_seed,
            n_rounds=n_rounds,
            withhold_corrective=(arm == "injection_no_correction"),
            attribution=attribution,
            log_path=log_path,
            replay_outputs=replay,
            replay_source_conditions=sources,
            arm_sequence_position=position,
        )

    return results


async def run_tool_correction_scope_ladder(
    scenario,
    variant,
    frozen_seed: SeedInstance,
    client,
    *,
    topology_name: str = "fully_connected_mesh",
    n_rounds: int = N_ROUNDS,
    attribution: str = "anonymous",
    log_path: Optional[str | Path] = None,
    include_controls: bool = False,
) -> dict[str, ConversationOutputs]:
    """Run protocol-v2 matched arms with genuine forced tool access.

    The seed and every pre-release response are sampled once.  At release, the
    corrector makes one real backend tool exchange in a hidden corrected anchor;
    its response and trace are replayed across every corrected measured arm.
    Its next response is likewise sampled once against the shared release-round
    history, ensuring that testimony content does not vary with delivery route.

    In the first peer-response round, the four primary arms differ only as
    follows: no corrective tool; matched natural testimony; that testimony plus
    the exact canonical tool-result bytes; or a forced direct tool exchange for
    each recipient. Optional controls add the canonical payload as ordinary
    prompt text or make the same tool turn with a non-diagnostic result. In
    sparse topologies, testimony and exact relay follow graph edges; direct
    recipient delivery and its controls remain available to every non-verifier.
    """
    validate_topology(topology_name)
    if not callable(getattr(client, "generate_with_forced_tool", None)):
        raise TypeError(
            "tool correction-scope ladder requires "
            "client.generate_with_forced_tool"
        )
    available = variant.corrective_evidence.available_round
    if n_rounds < available + 3:
        raise ValueError(
            "tool scope ladder requires two peer-response rounds after correction: "
            f"available_round={available}, n_rounds={n_rounds}"
        )
    seed_id, _ = DEFAULT_ROLES[topology_name]
    corrector_id = corrector_for_seed(topology_name, seed_id)

    prefix_outputs: dict[tuple[int, int], ParsedResponse] = {}
    if available > 0:
        prefix = await run_conversation(
            scenario,
            client=client,
            topology_name=topology_name,
            condition="injection_tool_none",
            variant=variant,
            frozen_seed=frozen_seed,
            n_rounds=available,
            withhold_corrective=True,
            attribution=attribution,
        )
        prefix_outputs = {
            key: value
            for key, value in prefix.outputs.items()
            if key != (seed_id, 0)
        }

    # Counterbalance which hidden anchor samples first. Every unaffected
    # release-round response is copied from the first anchor to the second.
    anchor_key = (
        f"tool-v2-anchor:{variant.assumption_id}:"
        f"{frozen_seed.seed_trial_index}:{topology_name}:{attribution}"
    )
    parity = int(sha256_text(anchor_key)[:8], 16) % 2
    anchor_order = (
        ["injection_tool_none", "injection_tool_testimony"]
        if parity == 0
        else ["injection_tool_testimony", "injection_tool_none"]
    )
    anchors: dict[str, ConversationOutputs] = {}
    for arm in anchor_order:
        replay = dict(prefix_outputs)
        sources = {key: "injection_tool_none" for key in replay}
        if anchors:
            first_arm = anchor_order[0]
            first_result = anchors[first_arm]
            for agent_id in AGENT_IDS:
                if agent_id == corrector_id or (agent_id, available) == (seed_id, 0):
                    continue
                key = (agent_id, available)
                replay[key] = first_result.outputs[key]
                sources[key] = first_arm
        anchors[arm] = await run_conversation(
            scenario,
            client=client,
            topology_name=topology_name,
            condition=arm,
            variant=variant,
            frozen_seed=frozen_seed,
            n_rounds=available + 2,
            withhold_corrective=(arm == "injection_tool_none"),
            replay_outputs=replay,
            replay_source_conditions=sources,
            attribution=attribution,
        )

    base_order = list(TOOL_CORRECTION_SCOPE)
    if include_controls:
        base_order.extend(TOOL_CORRECTION_CONTROLS)
    rotation_key = (
        f"tool-v2:{variant.assumption_id}:{frozen_seed.seed_trial_index}:"
        f"{topology_name}:{attribution}:controls={include_controls}"
    )
    rotation = int(sha256_text(rotation_key)[:8], 16) % len(base_order)
    execution_order = base_order[rotation:] + base_order[:rotation]

    results: dict[str, ConversationOutputs] = {}
    for position, arm in enumerate(execution_order, 1):
        anchor_condition = (
            "injection_tool_none"
            if arm == "injection_tool_none"
            else "injection_tool_testimony"
        )
        anchor = anchors[anchor_condition]
        replay: dict[tuple[int, int], ParsedResponse] = {}
        replay_traces: dict[tuple[int, int], ToolCallTrace] = {}
        for key, value in anchor.outputs.items():
            agent_id, round_index = key
            if round_index <= available or (
                agent_id == corrector_id and round_index == available + 1
            ):
                if key != (seed_id, 0):
                    replay[key] = value
                    if key in anchor.tool_traces:
                        replay_traces[key] = anchor.tool_traces[key]
        sources = {key: anchor_condition for key in replay}
        results[arm] = await run_conversation(
            scenario,
            client=client,
            topology_name=topology_name,
            condition=arm,
            variant=variant,
            frozen_seed=frozen_seed,
            n_rounds=n_rounds,
            withhold_corrective=(arm == "injection_tool_none"),
            attribution=attribution,
            log_path=log_path,
            replay_outputs=replay,
            replay_source_conditions=sources,
            replay_tool_traces=replay_traces,
            arm_sequence_position=position,
        )

    return results


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

async def run_matched_triple(
    scenario,
    variant,
    frozen_seed: SeedInstance,
    client,
    *,
    topologies: tuple[str, ...] = ("looped_chain", "neutral_star", "fully_connected_mesh"),
    n_rounds: int = N_ROUNDS,
    condition: str = "injection",
    withhold_corrective: bool = False,
    attribution: str = "anonymous",
    log_path: Optional[str | Path] = None,
) -> dict[str, ConversationOutputs]:
    """Replay ONE frozen seed across all three topologies (the matched triple).

    Seed/corrector placement comes from DEFAULT_ROLES per topology; the
    seed response itself is ID-neutral, so replaying it into different
    positions is well-defined.
    """
    if len(set(topologies)) != len(topologies):
        raise ValueError(f"topologies must be unique, got {topologies}")
    results: dict[str, ConversationOutputs] = {}
    for topology_name in topologies:
        results[topology_name] = await run_conversation(
            scenario,
            client=client,
            topology_name=topology_name,
            condition=condition,
            variant=variant,
            frozen_seed=frozen_seed,
            n_rounds=n_rounds,
            withhold_corrective=withhold_corrective,
            attribution=attribution,
            log_path=log_path,
        )
    return results
