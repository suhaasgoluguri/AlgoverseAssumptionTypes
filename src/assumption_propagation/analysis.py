"""Cross-conversation analysis over raw run logs.

metrics.py computes quantities within ONE conversation; this module works
across a whole run: loading the JSONL logs, joining scenario metadata,
per-conversation summaries, condition/topology aggregates, and merging the
post-hoc judge's labels back onto our records.

The judge runner is isolated from experiment orchestration. Here we validate
its output against the JudgedAgentRound contract in
schemas.py and join it to AgentRoundRecords — invalid judge rows are
reported, never silently dropped, mirroring the retention rule for agent
parse failures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

from pydantic import ValidationError

from . import metrics
from .schemas import (
    AgentRoundRecord,
    AssumptionState,
    AssumptionVariant,
    JudgedAgentRound,
    Scenario,
)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_agent_records(path: Path) -> list[AgentRoundRecord]:
    return [
        AgentRoundRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_judged_rounds(path: Path) -> tuple[list[JudgedAgentRound], list[tuple[int, str]]]:
    """Parse judge output; returns (valid rows, [(1-based line, error)]).

    Contract violations are collected, not raised: the merge report must
    show exactly which lines failed and why.
    """
    rows: list[JudgedAgentRound] = []
    errors: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(JudgedAgentRound.model_validate_json(line))
        except ValidationError as exc:
            errors.append((line_no, str(exc)))
    return rows, errors


def variant_lookup(scenarios: Sequence[Scenario]) -> dict[str, tuple[Scenario, AssumptionVariant]]:
    """assumption_id -> (scenario, variant), for joining metadata onto logs."""
    table: dict[str, tuple[Scenario, AssumptionVariant]] = {}
    for scenario in scenarios:
        for variant in scenario.assumption_variants:
            table[variant.assumption_id] = (scenario, variant)
    return table


def group_by_run(records: Sequence[AgentRoundRecord]) -> dict[str, list[AgentRoundRecord]]:
    """One conversation == one run_id."""
    grouped: dict[str, list[AgentRoundRecord]] = defaultdict(list)
    for record in records:
        grouped[record.run_id].append(record)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Per-conversation answer-level summary
# ---------------------------------------------------------------------------

def summarize_conversation(
    records: Sequence[AgentRoundRecord],
    *,
    correct_answer: str,
    favored_option: Optional[str] = None,
    semantic_type: Optional[str] = None,
) -> dict:
    """Answer-level summary for one conversation (no judge needed).

    Works for both 4-round conversations and the 1-round independent
    control: the "final" round is the last round actually present, so the
    control's round-0 answers are its finals.
    """
    first = records[0]
    final_round = max(r.round for r in records)
    finals = [r for r in records if r.round == final_round]
    agent_ids = sorted({r.agent_id for r in records})

    summary: dict = {
        "run_id": first.run_id,
        "scenario_id": first.scenario_id,
        "assumption_id": first.assumption_id,
        "seed_instance_id": first.seed_instance_id,
        "condition": first.condition.value,
        "topology": first.topology.value,
        "semantic_type": semantic_type,
        "final_round": final_round,
        "final_individual_accuracy": (
            sum(r.answer == correct_answer for r in finals) / len(finals)
        ),
        "parse_failure_count": sum(r.parse_failed for r in records),
        "trajectories": {
            agent_id: metrics.answer_trajectory(records, agent_id)
            for agent_id in agent_ids
        },
        "flip_counts": {
            agent_id: metrics.flip_count(records, agent_id) for agent_id in agent_ids
        },
    }

    if favored_option is not None:
        summary["favored_adoption_final"] = (
            sum(r.answer == favored_option for r in finals) / len(finals)
        )
        if final_round == metrics.FINAL_ROUND:
            summary["favored_adoption_by_round"] = metrics.favored_adoption_by_round(
                records, favored_option
            )
        if first.seed_agent_id is not None:
            summary["seed_recanted"] = metrics.seed_recanted(
                records, first.seed_agent_id, favored_option
            )
    return summary


def summarize_judged_conversation(
    judged: Sequence[JudgedAgentRound],
    *,
    seed_agent_id: int,
    n_records: int,
) -> dict:
    """Judge-level summary for one conversation.

    Substantive values are withheld unless coverage is complete.  Treating a
    missing label as non-acceptance or no correction would bias exactly the
    outcomes this experiment is designed to estimate.
    """
    unique_by_key = {}
    for row in judged:
        unique_by_key.setdefault((row.run_id, row.agent_id, row.round), row)
    unique_rows = list(unique_by_key.values())
    duplicate_key_count = len(judged) - len(unique_rows)
    coverage = len(unique_rows) / n_records if n_records else 0.0
    unclear_count = sum(
        row.assumption_state == AssumptionState.UNCLEAR for row in unique_rows
    )
    uncertain_count = sum(row.judge_uncertain for row in unique_rows)
    usable_count = sum(
        row.assumption_state != AssumptionState.UNCLEAR and not row.judge_uncertain
        for row in unique_rows
    )
    summary = {
        "judge_coverage": coverage,
        "judge_duplicate_key_count": duplicate_key_count,
        "judge_unclear_count": unclear_count,
        "judge_unclear_rate": unclear_count / len(unique_rows) if unique_rows else None,
        "judge_uncertain_count": uncertain_count,
        "judge_uncertain_rate": uncertain_count / len(unique_rows) if unique_rows else None,
        "judge_usable_coverage": usable_count / n_records if n_records else 0.0,
        "judge_adjudication_required": bool(unclear_count or uncertain_count),
        "final_acceptance_rate": None,
        "false_consensus": None,
        "first_adoption_round": None,
        "first_resistance_round": None,
        "first_correction_round": None,
        "correction_occurred": None,
        "correction_latency": None,
    }
    if (
        coverage != 1.0
        or duplicate_key_count
        or summary["judge_adjudication_required"]
    ):
        return summary

    final_round = max(label.round for label in unique_rows)
    summary.update({
        "final_acceptance_rate": metrics.final_acceptance_rate(
            unique_rows, seed_agent_id, final_round
        ),
        "false_consensus": metrics.false_consensus(unique_rows, final_round),
        "first_adoption_round": metrics.first_adoption_round(unique_rows, seed_agent_id),
        "first_resistance_round": metrics.first_resistance_round(unique_rows),
        "first_correction_round": metrics.first_correction_round(unique_rows),
        "correction_occurred": metrics.correction_occurred(unique_rows),
        "correction_latency": metrics.correction_latency(unique_rows, seed_agent_id),
    })
    return summary


# ---------------------------------------------------------------------------
# Judge merge
# ---------------------------------------------------------------------------

JoinKey = tuple[str, int, int]  # (run_id, agent_id, round)


def merge_judged(
    records: Sequence[AgentRoundRecord],
    judged: Sequence[JudgedAgentRound],
    *,
    verification_gap_defined: Mapping[str, bool] | None = None,
) -> tuple[dict[JoinKey, tuple[AgentRoundRecord, JudgedAgentRound]], list[JudgedAgentRound], list[AgentRoundRecord], list[str]]:
    """Join judge labels onto records by (run_id, agent_id, round).

    Returns (matched, orphan_judged, unjudged_records, conflicts):
    - matched: join key -> (record, judged label)
    - orphan_judged: labels with no matching record (judge hallucinated a
      key, or was run against different logs)
    - unjudged_records: records still awaiting a label
    - conflicts: human-readable inconsistencies (duplicate records or labels
      for one key, or assumption_id disagreement between record and label)
    """
    by_key: dict[JoinKey, AgentRoundRecord] = {}
    ambiguous_record_keys: set[JoinKey] = set()
    conflicts: list[str] = []
    judge_configurations = {
        (row.judge_model, row.judge_codebook_version, row.judge_prompt_version)
        for row in judged
    }
    if len(judge_configurations) > 1:
        conflicts.append(
            "mixed judge configurations: "
            + ", ".join(repr(config) for config in sorted(judge_configurations))
        )
    for record in records:
        key = (record.run_id, record.agent_id, record.round)
        if key in by_key:
            conflicts.append(f"duplicate raw record for {key}")
            ambiguous_record_keys.add(key)
        else:
            by_key[key] = record

    matched: dict[JoinKey, tuple[AgentRoundRecord, JudgedAgentRound]] = {}
    orphans: list[JudgedAgentRound] = []

    for label in judged:
        key = (label.run_id, label.agent_id, label.round)
        record = by_key.get(key)
        if record is None:
            orphans.append(label)
            continue
        if key in ambiguous_record_keys:
            continue
        if key in matched:
            conflicts.append(f"duplicate judge label for {key}")
            continue
        if record.assumption_id != label.assumption_id:
            conflicts.append(
                f"assumption_id mismatch at {key}: "
                f"record={record.assumption_id!r} judge={label.assumption_id!r}"
            )
            continue
        if verification_gap_defined is not None:
            if label.assumption_id not in verification_gap_defined:
                conflicts.append(
                    f"no scenario reference for assumption_id "
                    f"{label.assumption_id!r} at {key}"
                )
                continue
            expected_gap = verification_gap_defined[label.assumption_id]
            observed_gap = label.verification_gap_retained is not None
            if expected_gap != observed_gap:
                conflicts.append(
                    f"verification_gap_retained availability mismatch at {key}: "
                    f"reference_defined={expected_gap} label_is_null={not observed_gap}"
                )
                continue
        judged_text = record.rationale if record.rationale is not None else ""
        expected_source = (
            "carried_forward_rationale"
            if record.rationale is not None and record.carried_forward
            else "rationale"
            if record.rationale is not None
            else "missing_rationale"
        )
        if label.judged_text_source != expected_source:
            conflicts.append(
                f"judged_text_source mismatch at {key}: "
                f"record implies {expected_source!r}, judge={label.judged_text_source!r}"
            )
            continue
        try:
            label.validate_evidence_span(judged_text or "")
        except ValueError as exc:
            conflicts.append(f"invalid evidence span at {key}: {exc}")
            continue
        matched[key] = (record, label)

    unjudged = [
        record
        for record in records
        if (record.run_id, record.agent_id, record.round) not in matched
    ]
    return matched, orphans, unjudged, conflicts


def derive_protocol_metadata(
    records: Sequence[AgentRoundRecord],
    judged: Sequence[JudgedAgentRound],
) -> dict[JoinKey, dict]:
    """Separate direct-note visibility from routed semantic message content.

    The protocol flags establish only that a private note was visible; they do
    not establish that the note explicitly stated the canonical target.  Peer
    channels are derived from preceding-round semantic labels and remain
    unknown whenever a needed label is missing, unclear, or judge-uncertain.
    """
    labels = {(row.run_id, row.agent_id, row.round): row for row in judged}
    result: dict[JoinKey, dict] = {}
    injection_note_seen: dict[tuple[str, int], bool] = defaultdict(bool)
    correction_note_seen: dict[tuple[str, int], bool] = defaultdict(bool)
    target_routed_seen: dict[tuple[str, int], bool] = defaultdict(bool)
    acceptance_routed_seen: dict[tuple[str, int], bool] = defaultdict(bool)
    correction_routed_seen: dict[tuple[str, int], bool] = defaultdict(bool)
    routing_uncertain_seen: dict[tuple[str, int], bool] = defaultdict(bool)

    ordered = sorted(records, key=lambda row: (row.run_id, row.round, row.agent_id))
    for record in ordered:
        key = (record.run_id, record.agent_id, record.round)
        target_senders: list[int] = []
        acceptance_senders: list[int] = []
        correction_senders: list[int] = []
        uncertain_senders: list[int] = []
        if record.round > 0:
            for sender in record.received_from:
                prior = labels.get((record.run_id, sender, record.round - 1))
                if (
                    prior is None
                    or prior.assumption_state == AssumptionState.UNCLEAR
                    or prior.judge_uncertain
                ):
                    uncertain_senders.append(sender)
                    continue
                if prior.target_engaged is True:
                    target_senders.append(sender)
                if (
                    prior.assumption_state in metrics.ACCEPTING_STATES
                    or prior.decision_reliance
                ):
                    acceptance_senders.append(sender)
                if prior.correction_present:
                    correction_senders.append(sender)

        agent_key = (record.run_id, record.agent_id)
        injection_note_seen[agent_key] |= record.injection_visible
        correction_note_seen[agent_key] |= record.corrective_visible
        target_routed_seen[agent_key] |= bool(target_senders)
        acceptance_routed_seen[agent_key] |= bool(acceptance_senders)
        correction_routed_seen[agent_key] |= bool(correction_senders)
        routing_uncertain_seen[agent_key] |= bool(uncertain_senders)

        result[key] = {
            "injection_note_visible_this_round": record.injection_visible,
            "injection_note_visible_by_round": injection_note_seen[agent_key],
            "correction_note_visible_this_round": record.corrective_visible,
            "correction_note_visible_by_round": correction_note_seen[agent_key],
            "target_engagement_routed_this_round": bool(target_senders),
            "target_engagement_routed_by_round": target_routed_seen[agent_key],
            "acceptance_routed_this_round": bool(acceptance_senders),
            "acceptance_routed_by_round": acceptance_routed_seen[agent_key],
            "correction_routed_this_round": bool(correction_senders),
            "correction_routed_by_round": correction_routed_seen[agent_key],
            "target_engaging_sender_ids": sorted(target_senders),
            "acceptance_bearing_sender_ids": sorted(acceptance_senders),
            "correction_bearing_sender_ids": sorted(correction_senders),
            "uncertain_sender_ids": sorted(uncertain_senders),
            "routing_derivation_uncertain_this_round": bool(uncertain_senders),
            "routing_derivation_uncertain_by_round": routing_uncertain_seen[agent_key],
        }
    return result


# ---------------------------------------------------------------------------
# Aggregation across conversations
# ---------------------------------------------------------------------------

AGGREGATE_KEYS = ("condition", "topology", "semantic_type")

# Answer fields skip structurally inapplicable values (for example,
# seed_recanted when the seed never held the favored option). Judge quality
# fields report available coverage. Headline judge outcomes use a stricter
# all-conversations rule so unresolved runs cannot disappear from denominators.
_ANSWER_MEAN_FIELDS = (
    "final_individual_accuracy",
    "favored_adoption_final",
    "seed_recanted",
)
_JUDGE_QUALITY_FIELDS = (
    "judge_coverage",
    "judge_usable_coverage",
    "judge_unclear_rate",
    "judge_uncertain_rate",
)
_JUDGE_HEADLINE_FIELDS = (
    "final_acceptance_rate",
    "false_consensus",
    "correction_occurred",
)


def aggregate_summaries(summaries: Sequence[dict]) -> list[dict]:
    """Group per-conversation summaries by (condition, topology, semantic_type)
    and average comparable fields. Booleans average into rates; judge headline
    fields remain null if any member conversation is unresolved."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for summary in summaries:
        groups[tuple(summary.get(k) for k in AGGREGATE_KEYS)].append(summary)

    rows = []
    for key, members in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row: dict = dict(zip(AGGREGATE_KEYS, key))
        row["n_conversations"] = len(members)
        row["n_judged_conversations"] = sum(
            "judge_coverage" in member for member in members
        )
        for field in _ANSWER_MEAN_FIELDS + _JUDGE_QUALITY_FIELDS:
            values = [m[field] for m in members if m.get(field) is not None]
            row[field] = sum(values) / len(values) if values else None
        for field in _JUDGE_HEADLINE_FIELDS:
            values = [member.get(field) for member in members]
            row[field] = (
                sum(values) / len(values)
                if values and all(value is not None for value in values)
                else None
            )
        by_round = [m["favored_adoption_by_round"] for m in members
                    if m.get("favored_adoption_by_round") is not None]
        row["favored_adoption_by_round"] = (
            [sum(col) / len(col) for col in zip(*by_round)] if by_round else None
        )
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
