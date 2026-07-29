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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .clients import (
    RATIONALE_AUDIT_VERSION,
    ParsedResponse,
    audit_answer_rationale,
    parse_or_repair,
)
from .prompts import (
    build_private_dossier,
    build_round0_prompt,
    build_seed_generation_system_prompt,
    build_system_prompt,
    build_update_prompt,
    format_peer_message,
)
from .schemas import (
    INJECTION_CONDITIONS,
    AgentOutput,
    AgentRoundRecord,
    Condition,
    SeedInstance,
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

    def final_answers(self) -> dict[int, Optional[str]]:
        final_round = self.n_rounds - 1
        return {a: self.outputs[(a, final_round)].answer for a in AGENT_IDS}


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
    log_path: Optional[str | Path] = None,
) -> ConversationOutputs:
    """Run one 4-agent, 4-round conversation.

    Roles default to the team-decided placement (DEFAULT_ROLES): the seed's
    private dossier persists from round 0, while the corrector's dossier
    appears at ``corrective_evidence.available_round`` and persists thereafter.
    condition="injection": `frozen_seed` is
    REPLAYED as the seed's round-0 output (never regenerated).
    condition="no_injection_baseline": all four agents generate normally;
    the seed slot receives the neutral note; the corrector keeps its
    evidence so the two conditions differ only in the injection.
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
    # injection_no_correction is the ablation arm: identical delivery, the
    # corrector simply never receives the disproof. It is validated exactly
    # like the injection arm so the two remain matched.
    is_injection = condition in INJECTION_CONDITIONS
    if condition == "injection_no_correction":
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

    # Track B hidden-profile briefs are role-keyed, and the two neutral slots
    # are distinguished by ascending agent id. That ordering is stable across
    # topologies (ring neutrals 2,3; star neutrals 0,3), so "the first neutral"
    # names the same experimental position everywhere.
    neutral_ids = [a for a in AGENT_IDS if a not in (seed_agent_id, corrector_agent_id)]

    def neutral_ordinal_of(agent_id: int) -> Optional[int]:
        return neutral_ids.index(agent_id) if agent_id in neutral_ids else None

    run_id = new_run_id()
    outputs: dict[tuple[int, int], ParsedResponse] = {}
    records: list[AgentRoundRecord] = []

    def make_record(
        agent_id: int,
        round_index: int,
        parsed: ParsedResponse,
        received_from: list[int],
        *,
        injection_visible: bool,
        neutral_control_visible: bool,
        corrective_visible: bool,
    ) -> AgentRoundRecord:
        rationale_audit = (
            audit_answer_rationale(parsed.answer, parsed.rationale, scenario.options)
            if not parsed.parse_failed
            else None
        )
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
            injection_visible=injection_visible,
            neutral_control_visible=neutral_control_visible,
            corrective_visible=corrective_visible,
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
        tasks = []
        generating: list[tuple[int, list[int], bool, bool, bool]] = []

        for agent_id in AGENT_IDS:
            injection_visible = (
                is_injection and agent_id == seed_agent_id
            )
            neutral_control_visible = (
                condition in ("no_injection_baseline", "independent_control")
                and agent_id == seed_agent_id
            )
            corrective_visible = (
                variant is not None
                and not withhold_corrective
                and agent_id == corrector_agent_id
                and round_index >= variant.corrective_evidence.available_round
            )

            # --- Frozen seed replay (injection, round 0 only) ---
            if round_index == 0 and is_injection and agent_id == seed_agent_id:
                parsed = ParsedResponse(
                    raw_response=frozen_seed.raw_response,
                    output=AgentOutput(
                        answer=frozen_seed.answer, rationale=frozen_seed.rationale
                    ) if frozen_seed.answer else None,
                )
                outputs[(agent_id, 0)] = parsed
                records.append(
                    make_record(
                        agent_id,
                        0,
                        parsed,
                        [],
                        injection_visible=True,
                        neutral_control_visible=False,
                        corrective_visible=False,
                    )
                )
                continue

            received_from = senders_for(topology_name, agent_id, round_index)

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
                prompt = build_update_prompt(
                    scenario,
                    own_previous_answer=own_prev.answer or "(no valid answer)",
                    own_previous_rationale=own_prev.rationale or "(no rationale)",
                    peer_messages=peer_messages,
                    round_index=round_index,
                    private_note=private_dossier,
                )

            system_prompt = (
                build_seed_generation_system_prompt()
                if round_index == 0 and agent_id == seed_agent_id
                else build_system_prompt(agent_id)
            )
            tasks.append(client.generate(system_prompt, prompt))
            generating.append(
                (
                    agent_id,
                    received_from,
                    injection_visible,
                    neutral_control_visible,
                    corrective_visible,
                )
            )

        # SYNCHRONOUS COMMIT: gather the whole round before anyone sees it.
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        for (
            agent_id,
            received_from,
            injection_visible,
            neutral_control_visible,
            corrective_visible,
        ), raw in zip(generating, raw_results):
            if isinstance(raw, Exception):
                parsed = ParsedResponse(
                    raw_response="",
                    parse_failed=True,
                    generation_failed=True,
                    generation_error_type=type(raw).__name__,
                    generation_error_message=str(raw)[:1200],
                )
            else:
                parsed = parse_or_repair(raw)
            if parsed.parse_failed and round_index > 0:
                # Carry forward the previous valid output so the
                # conversation continues; the failure itself is logged.
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
                    received_from,
                    injection_visible=injection_visible,
                    neutral_control_visible=neutral_control_visible,
                    corrective_visible=corrective_visible,
                )
            )

        # Commit each complete synchronous round immediately. If a later API
        # call exhausts retries or the process stops, earlier paid responses
        # remain auditable and analysis will flag the incomplete round grid.
        if log_path:
            for record in records[round_record_start:]:
                append_jsonl(log_path, record)

    return ConversationOutputs(
        run_id=run_id, records=records, outputs=outputs, n_rounds=n_rounds
    )


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
            log_path=log_path,
        )
    return results
