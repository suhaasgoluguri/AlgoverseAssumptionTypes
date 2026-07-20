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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .clients import ParsedResponse, parse_or_repair
from .prompts import (
    PROMPT_VERSION,
    build_round0_prompt,
    build_seed_generation_system_prompt,
    build_system_prompt,
    build_update_prompt,
    format_peer_message,
)
from .schemas import AgentOutput, AgentRoundRecord, Condition, SeedInstance, Topology
from .topologies import (
    AGENT_IDS,
    DEFAULT_ROLES,
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
        raw = await client.generate(system_prompt, user_prompt)
        parsed = parse_or_repair(raw)
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
    log_path: Optional[str | Path] = None,
) -> ConversationOutputs:
    """Run one 4-agent, 4-round conversation.

    Roles default to the team-decided placement (DEFAULT_ROLES): seed holds
    the assumption note, corrector holds the corrective evidence, both as
    round-0 private additions. condition="injection": `frozen_seed` is
    REPLAYED as the seed's round-0 output (never regenerated).
    condition="no_injection_baseline": all four agents generate normally;
    the seed slot receives the neutral note; the corrector keeps its
    evidence so the two conditions differ only in the injection.
    """
    validate_topology(topology_name)
    default_seed, default_corrector = DEFAULT_ROLES[topology_name]
    if seed_agent_id is None:
        seed_agent_id = default_seed
    if corrector_agent_id is None:
        corrector_agent_id = default_corrector
        if corrector_agent_id == seed_agent_id:  # explicit seed override collided
            corrector_agent_id = next(a for a in AGENT_IDS if a != seed_agent_id)
    validate_roles(topology_name, seed_agent_id, corrector_agent_id)
    if condition == "injection" and frozen_seed is None:
        raise ValueError("injection conversations require a frozen_seed to replay")

    run_id = new_run_id()
    outputs: dict[tuple[int, int], ParsedResponse] = {}
    records: list[AgentRoundRecord] = []

    def make_record(agent_id: int, round_index: int, parsed: ParsedResponse,
                    received_from: list[int], injection_visible: bool) -> AgentRoundRecord:
        return AgentRoundRecord(
            run_id=run_id,
            stage="propagation",
            scenario_id=scenario.scenario_id,
            assumption_id=variant.assumption_id if variant is not None else None,
            seed_instance_id=frozen_seed.seed_instance_id if frozen_seed else None,
            condition=Condition(condition),
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
            corrective_visible=(agent_id == corrector_agent_id and round_index == 0
                                and variant is not None),
            received_from=received_from,
            messages_received=len(received_from),
            answer=parsed.answer,
            rationale=parsed.rationale,
            raw_response=parsed.raw_response,
            parse_failed=parsed.parse_failed,
            parse_repaired=parsed.parse_repaired,
        )

    for round_index in range(n_rounds):
        tasks = []
        generating: list[tuple[int, list[int]]] = []  # (agent_id, received_from)

        for agent_id in AGENT_IDS:
            # --- Frozen seed replay (injection, round 0 only) ---
            if round_index == 0 and condition == "injection" and agent_id == seed_agent_id:
                parsed = ParsedResponse(
                    raw_response=frozen_seed.raw_response,
                    output=AgentOutput(
                        answer=frozen_seed.answer, rationale=frozen_seed.rationale
                    ) if frozen_seed.answer else None,
                )
                outputs[(agent_id, 0)] = parsed
                records.append(make_record(agent_id, 0, parsed, [], injection_visible=True))
                continue

            received_from = senders_for(topology_name, agent_id, round_index)

            if round_index == 0:
                prompt = build_round0_prompt(
                    scenario,
                    is_seed=(agent_id == seed_agent_id),
                    is_corrector=(agent_id == corrector_agent_id),
                    condition=condition,
                    variant=variant,
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
                prompt = build_update_prompt(
                    scenario,
                    own_previous_answer=own_prev.answer or "(no valid answer)",
                    own_previous_rationale=own_prev.rationale or "(no rationale)",
                    peer_messages=peer_messages,
                    round_index=round_index,
                )

            tasks.append(client.generate(build_system_prompt(agent_id), prompt))
            generating.append((agent_id, received_from))

        # SYNCHRONOUS COMMIT: gather the whole round before anyone sees it.
        raw_results = await asyncio.gather(*tasks)
        for (agent_id, received_from), raw in zip(generating, raw_results):
            parsed = parse_or_repair(raw)
            if parsed.parse_failed and round_index > 0:
                # Carry forward the previous valid output so the
                # conversation continues; the failure itself is logged.
                prev = outputs[(agent_id, round_index - 1)]
                parsed = ParsedResponse(
                    raw_response=parsed.raw_response,
                    output=prev.output,
                    parse_failed=True,
                )
            outputs[(agent_id, round_index)] = parsed
            records.append(
                make_record(agent_id, round_index, parsed, received_from,
                            injection_visible=False)
            )

    if log_path:
        for record in records:
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
    log_path: Optional[str | Path] = None,
) -> dict[str, ConversationOutputs]:
    """Replay ONE frozen seed across all three topologies (the matched triple).

    Seed/corrector placement comes from DEFAULT_ROLES per topology; the
    seed response itself is ID-neutral, so replaying it into different
    positions is well-defined.
    """
    results: dict[str, ConversationOutputs] = {}
    for topology_name in topologies:
        results[topology_name] = await run_conversation(
            scenario,
            client=client,
            topology_name=topology_name,
            condition="injection",
            variant=variant,
            frozen_seed=frozen_seed,
            log_path=log_path,
        )
    return results
