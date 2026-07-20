"""Cross-conversation analysis over raw run logs.

metrics.py computes quantities within ONE conversation; this module works
across a whole run: loading the JSONL logs, joining scenario metadata,
per-conversation summaries, condition/topology aggregates, and merging the
teammate judge's labels back onto our records.

The judge itself is owned by a teammate (Sonnet 5, Anthropic API). Here we
only validate its output against the JudgedAgentRound contract in
schemas.py and join it to AgentRoundRecords — invalid judge rows are
reported, never silently dropped, mirroring the retention rule for agent
parse failures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from pydantic import ValidationError

from . import metrics
from .schemas import (
    AgentRoundRecord,
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
    show the teammate exactly which lines failed and why.
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

    judge_coverage < 1.0 means some agent-rounds are still unjudged; the
    first_* metrics are then lower bounds and the rates provisional.
    """
    return {
        "judge_coverage": len(judged) / n_records if n_records else 0.0,
        "final_acceptance_rate": metrics.final_acceptance_rate(judged, seed_agent_id),
        "false_consensus": metrics.false_consensus(judged),
        "first_adoption_round": metrics.first_adoption_round(judged, seed_agent_id),
        "first_correction_round": metrics.first_correction_round(judged),
        "correction_occurred": metrics.correction_occurred(judged),
        "correction_latency": metrics.correction_latency(judged, seed_agent_id),
    }


# ---------------------------------------------------------------------------
# Judge merge
# ---------------------------------------------------------------------------

JoinKey = tuple[str, int, int]  # (run_id, agent_id, round)


def merge_judged(
    records: Sequence[AgentRoundRecord],
    judged: Sequence[JudgedAgentRound],
) -> tuple[dict[JoinKey, tuple[AgentRoundRecord, JudgedAgentRound]], list[JudgedAgentRound], list[AgentRoundRecord], list[str]]:
    """Join judge labels onto records by (run_id, agent_id, round).

    Returns (matched, orphan_judged, unjudged_records, conflicts):
    - matched: join key -> (record, judged label)
    - orphan_judged: labels with no matching record (judge hallucinated a
      key, or was run against different logs)
    - unjudged_records: records still awaiting a label
    - conflicts: human-readable inconsistencies (duplicate labels for one
      key, or assumption_id disagreement between record and label)
    """
    by_key = {(r.run_id, r.agent_id, r.round): r for r in records}
    matched: dict[JoinKey, tuple[AgentRoundRecord, JudgedAgentRound]] = {}
    orphans: list[JudgedAgentRound] = []
    conflicts: list[str] = []

    for label in judged:
        key = (label.run_id, label.agent_id, label.round)
        record = by_key.get(key)
        if record is None:
            orphans.append(label)
            continue
        if key in matched:
            conflicts.append(f"duplicate judge label for {key}")
            continue
        if record.assumption_id is not None and record.assumption_id != label.assumption_id:
            conflicts.append(
                f"assumption_id mismatch at {key}: "
                f"record={record.assumption_id!r} judge={label.assumption_id!r}"
            )
            continue
        matched[key] = (record, label)

    unjudged = [r for key, r in by_key.items() if key not in matched]
    return matched, orphans, unjudged, conflicts


# ---------------------------------------------------------------------------
# Aggregation across conversations
# ---------------------------------------------------------------------------

AGGREGATE_KEYS = ("condition", "topology", "semantic_type")

# Mean-aggregated per-conversation fields; None values are skipped (e.g.
# seed_recanted is None when the seed never held the favored option).
_MEAN_FIELDS = (
    "final_individual_accuracy",
    "favored_adoption_final",
    "seed_recanted",
    "final_acceptance_rate",
    "false_consensus",
    "correction_occurred",
)


def aggregate_summaries(summaries: Sequence[dict]) -> list[dict]:
    """Group per-conversation summaries by (condition, topology, semantic_type)
    and average the comparable fields. Booleans average into rates."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for summary in summaries:
        groups[tuple(summary.get(k) for k in AGGREGATE_KEYS)].append(summary)

    rows = []
    for key, members in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row: dict = dict(zip(AGGREGATE_KEYS, key))
        row["n_conversations"] = len(members)
        for field in _MEAN_FIELDS:
            values = [m[field] for m in members if m.get(field) is not None]
            row[field] = sum(values) / len(values) if values else None
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
