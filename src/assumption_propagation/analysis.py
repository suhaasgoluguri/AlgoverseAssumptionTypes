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
from .aggregation import summarize_final_round
from .topologies import AGENT_IDS, senders_for
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


def validate_conversation_records(
    records: Sequence[AgentRoundRecord],
    *,
    expected_n_rounds: int | None = None,
) -> None:
    """Reject malformed or internally mixed conversations before scoring."""
    if not records:
        raise ValueError("conversation has no records")

    first = records[0]
    invariant_fields = (
        "run_id",
        "stage",
        "scenario_id",
        "assumption_id",
        "condition",
        "topology",
        "model",
        "temperature",
        "seed_instance_id",
        "seed_agent_id",
        "corrector_agent_id",
    )
    for field in invariant_fields:
        expected = getattr(first, field)
        observed = {getattr(row, field) for row in records}
        if observed != {expected}:
            raise ValueError(
                f"{first.run_id}: mixed {field} values: {sorted(map(str, observed))}"
            )

    rounds = sorted({row.round for row in records})
    if rounds != list(range(rounds[-1] + 1)):
        raise ValueError(f"{first.run_id}: non-contiguous rounds {rounds}")
    if expected_n_rounds is not None and len(rounds) != expected_n_rounds:
        raise ValueError(
            f"{first.run_id}: expected {expected_n_rounds} rounds, found {len(rounds)}"
        )

    by_key: dict[tuple[int, int], AgentRoundRecord] = {}
    for row in records:
        key = (row.agent_id, row.round)
        if key in by_key:
            raise ValueError(f"{first.run_id}: duplicate agent-round record {key}")
        by_key[key] = row
        if row.messages_received != len(row.received_from):
            raise ValueError(
                f"{first.run_id}: messages_received mismatch at agent-round {key}"
            )
        expected_senders = senders_for(first.topology.value, row.agent_id, row.round)
        if row.received_from != expected_senders:
            raise ValueError(
                f"{first.run_id}: routing mismatch at agent-round {key}: "
                f"expected {expected_senders}, found {row.received_from}"
            )

    expected_keys = {
        (agent_id, round_index)
        for round_index in rounds
        for agent_id in AGENT_IDS
    }
    missing = expected_keys - set(by_key)
    extra = set(by_key) - expected_keys
    if missing or extra:
        raise ValueError(
            f"{first.run_id}: incomplete agent-round grid; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


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
    validate_conversation_records(records)
    first = records[0]
    final_round = max(r.round for r in records)
    finals = [r for r in records if r.round == final_round]
    agent_ids = sorted({r.agent_id for r in records})
    vote_summary = summarize_final_round(
        {r.agent_id: r.answer for r in finals},
        correct_answer=correct_answer,
        favored_option=favored_option,
    )
    rationale_audited = [
        record
        for record in records
        if record.answer_rationale_inconsistent is not None
    ]
    rationale_inconsistencies = sum(
        record.answer_rationale_inconsistent is True
        for record in rationale_audited
    )

    summary: dict = {
        "run_id": first.run_id,
        "scenario_id": first.scenario_id,
        "assumption_id": first.assumption_id,
        "seed_instance_id": first.seed_instance_id,
        "condition": first.condition.value,
        "topology": first.topology.value,
        "semantic_type": semantic_type,
        "correct_answer": correct_answer,
        "favored_option": favored_option,
        "final_round": final_round,
        "final_individual_accuracy": (
            sum(r.answer == correct_answer for r in finals) / len(finals)
        ),
        # A tie/no-consensus is not a correct MAS decision. Keep the separate
        # no-consensus flag so this convention remains transparent.
        "final_plurality_accuracy": float(
            vote_summary["final_answer_correct"] is True
        ),
        "final_plurality_answer": vote_summary["plurality_answer"],
        "final_no_consensus": vote_summary["no_consensus"],
        "final_consensus_strength": vote_summary["consensus_strength"],
        "final_votes": vote_summary["final_votes"],
        "record_count": len(records),
        "parse_failure_count": sum(r.parse_failed for r in records),
        "generation_failure_count": sum(r.generation_failed for r in records),
        "answer_rationale_inconsistency_count": rationale_inconsistencies,
        "answer_rationale_audit_n": len(rationale_audited),
        "answer_rationale_audit_coverage": len(rationale_audited) / len(records),
        "answer_rationale_inconsistency_rate": (
            rationale_inconsistencies / len(rationale_audited)
            if rationale_audited
            else None
        ),
        "answer_rationale_audit_versions": sorted({
            record.answer_rationale_audit_version
            for record in rationale_audited
            if record.answer_rationale_audit_version is not None
        }),
        "accuracy_by_round": [
            sum(r.answer == correct_answer for r in records if r.round == round_index)
            / sum(r.round == round_index for r in records)
            for round_index in range(final_round + 1)
        ],
        "trajectories": {
            agent_id: metrics.answer_trajectory(records, agent_id)
            for agent_id in agent_ids
        },
        "flip_counts": {
            agent_id: metrics.flip_count(records, agent_id) for agent_id in agent_ids
        },
    }

    if favored_option is not None:
        favored_by_round = metrics.favored_adoption_by_round(records, favored_option)
        summary["favored_adoption_final"] = (
            sum(r.answer == favored_option for r in finals) / len(finals)
        )
        summary["favored_adoption_by_round"] = favored_by_round
        summary["favored_share_auc"] = metrics.favored_share_auc(favored_by_round)
        summary["unrelated_option_rate_final"] = metrics.unrelated_option_rate(
            records,
            correct_answer=correct_answer,
            favored_option=favored_option,
            round_index=final_round,
        )
        summary["unrelated_option_rate_by_round"] = [
            metrics.unrelated_option_rate(
                records,
                correct_answer=correct_answer,
                favored_option=favored_option,
                round_index=round_index,
            )
            for round_index in range(final_round + 1)
        ]
        direct_rate, direct_count, direct_opportunities = (
            metrics.direct_exposure_adoption(records, favored_option)
        )
        summary["direct_exposure_adoption_rate"] = direct_rate
        summary["direct_exposure_adoption_count"] = direct_count
        summary["direct_exposure_opportunity_count"] = direct_opportunities
        summary["first_answer_recovery_round"] = metrics.first_answer_recovery_round(
            records, correct_answer, favored_option
        )
        if first.seed_agent_id is not None:
            summary["seed_trap_uptake"] = metrics.seed_trap_uptake(
                records, first.seed_agent_id, favored_option
            )
            summary["seed_recanted"] = metrics.seed_recanted(
                records, first.seed_agent_id, favored_option
            )
            summary["first_downstream_answer_adoption_round"] = (
                metrics.first_downstream_answer_adoption_round(
                    records, first.seed_agent_id, favored_option
                )
            )
            if first.corrector_agent_id is not None:
                summary["neutral_spontaneous_trap_rate"] = (
                    metrics.neutral_spontaneous_trap_rate(
                        records,
                        seed_agent_id=first.seed_agent_id,
                        corrector_agent_id=first.corrector_agent_id,
                        favored_option=favored_option,
                    )
                )
        correction_rounds = [r.round for r in records if r.corrective_visible]
        summary["correction_release_round"] = (
            min(correction_rounds) if correction_rounds else None
        )
    return summary


def summarize_judged_conversation(
    judged: Sequence[JudgedAgentRound],
    *,
    seed_agent_id: int,
    n_records: int,
    records: Sequence[AgentRoundRecord] | None = None,
) -> dict:
    """Judge-level summary for one conversation.

    Headline conversation-wide outcomes are withheld unless coverage is
    complete. Treating a missing label as non-acceptance or no correction
    would bias exactly the outcomes this experiment is designed to estimate.
    The seed's round-0 manipulation check and direct-exposure transition
    diagnostic are independently usable when their required labels are
    unique and adjudication-ready; the latter reports both exposed and
    evaluable denominators explicitly.
    """
    if records is not None and len(records) != n_records:
        raise ValueError(
            "records length must equal n_records for judged conversation summary"
        )

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
    seed_initial_candidates = [
        row
        for row in judged
        if row.agent_id == seed_agent_id and row.round == 0
    ]
    seed_initial = (
        seed_initial_candidates[0]
        if len(seed_initial_candidates) == 1
        and seed_initial_candidates[0].assumption_state != AssumptionState.UNCLEAR
        and not seed_initial_candidates[0].judge_uncertain
        else None
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
        "semantic_direct_exposure_adoption_rate": None,
        "semantic_direct_exposure_adoption_count": None,
        "semantic_direct_exposure_evaluable_opportunity_count": None,
        "semantic_direct_exposure_exposed_opportunity_count": None,
        "seed_initial_assumption_acceptance": (
            seed_initial.assumption_state in metrics.ACCEPTING_STATES
            if seed_initial is not None
            else None
        ),
        "seed_initial_decision_reliance": (
            seed_initial.decision_reliance if seed_initial is not None else None
        ),
    }
    if records is not None and not duplicate_key_count:
        (
            semantic_rate,
            semantic_adoptions,
            semantic_evaluable,
            semantic_exposed,
        ) = metrics.semantic_direct_exposure_adoption(records, unique_rows)
        summary.update({
            "semantic_direct_exposure_adoption_rate": semantic_rate,
            "semantic_direct_exposure_adoption_count": semantic_adoptions,
            "semantic_direct_exposure_evaluable_opportunity_count": (
                semantic_evaluable
            ),
            "semantic_direct_exposure_exposed_opportunity_count": semantic_exposed,
        })
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
) -> tuple[
    dict[JoinKey, tuple[AgentRoundRecord, JudgedAgentRound]],
    list[JudgedAgentRound],
    list[AgentRoundRecord],
    list[str],
]:
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
    neutral_control_note_history: dict[
        tuple[str, int], list[Optional[bool]]
    ] = defaultdict(list)
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
        neutral_control_note_history[agent_key].append(record.neutral_control_visible)
        neutral_history = neutral_control_note_history[agent_key]
        # A historical missing flag is unknown. It must never be collapsed to
        # false, because that would turn absent instrumentation into evidence
        # that the neutral dossier was not shown.
        if any(value is True for value in neutral_history):
            neutral_visible_by_round: Optional[bool] = True
        elif any(value is None for value in neutral_history):
            neutral_visible_by_round = None
        else:
            neutral_visible_by_round = False
        correction_note_seen[agent_key] |= record.corrective_visible
        target_routed_seen[agent_key] |= bool(target_senders)
        acceptance_routed_seen[agent_key] |= bool(acceptance_senders)
        correction_routed_seen[agent_key] |= bool(correction_senders)
        routing_uncertain_seen[agent_key] |= bool(uncertain_senders)

        result[key] = {
            "injection_note_visible_this_round": record.injection_visible,
            "injection_note_visible_by_round": injection_note_seen[agent_key],
            "neutral_control_note_visible_this_round": record.neutral_control_visible,
            "neutral_control_visibility_observed_this_round": (
                record.neutral_control_visible is not None
            ),
            "neutral_control_note_visible_by_round": neutral_visible_by_round,
            "neutral_control_visibility_observed_round_count": sum(
                value is not None for value in neutral_history
            ),
            "neutral_control_visibility_round_count": len(neutral_history),
            "neutral_control_visibility_complete_by_round": all(
                value is not None for value in neutral_history
            ),
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
    "final_plurality_accuracy",
    "final_no_consensus",
    "final_consensus_strength",
    "favored_adoption_final",
    "favored_share_auc",
    "seed_trap_uptake",
    "seed_recanted",
    "neutral_spontaneous_trap_rate",
    "direct_exposure_adoption_rate",
    "unrelated_option_rate_final",
    "answer_rationale_inconsistency_rate",
)
_JUDGE_QUALITY_FIELDS = (
    "judge_coverage",
    "judge_usable_coverage",
    "judge_unclear_rate",
    "judge_uncertain_rate",
)
_JUDGE_OPTIONAL_MEAN_FIELDS = (
    # Null when a conversation has no evaluable exposure opportunity, so
    # average only across conversations where the transition rate applies.
    "semantic_direct_exposure_adoption_rate",
)
_JUDGE_HEADLINE_FIELDS = (
    "final_acceptance_rate",
    "false_consensus",
    "correction_occurred",
    "seed_initial_assumption_acceptance",
    "seed_initial_decision_reliance",
)


def _roundwise_mean(series: Sequence[Sequence[Optional[float]]]) -> list[Optional[float]]:
    """Mean each observed round without truncating or treating null as zero."""
    if not series:
        return []
    result: list[Optional[float]] = []
    for round_index in range(max(len(values) for values in series)):
        observed = [
            values[round_index]
            for values in series
            if round_index < len(values) and values[round_index] is not None
        ]
        result.append(sum(observed) / len(observed) if observed else None)
    return result


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
        for field in (
            _ANSWER_MEAN_FIELDS
            + _JUDGE_QUALITY_FIELDS
            + _JUDGE_OPTIONAL_MEAN_FIELDS
        ):
            values = [m[field] for m in members if m.get(field) is not None]
            row[field] = sum(values) / len(values) if values else None
        for field in _JUDGE_HEADLINE_FIELDS:
            values = [member.get(field) for member in members]
            row[field] = (
                sum(values) / len(values)
                if values and all(value is not None for value in values)
                else None
            )

        # Applicability and exposure denominators must remain visible. A
        # macro-average of per-conversation rates alone can obscure both how
        # few runs were eligible and how unequal their opportunity counts are.
        row["seed_recanted_applicable_n"] = sum(
            member.get("seed_recanted") is not None for member in members
        )
        direct_adoptions = sum(
            int(member.get("direct_exposure_adoption_count") or 0)
            for member in members
        )
        direct_opportunities = sum(
            int(member.get("direct_exposure_opportunity_count") or 0)
            for member in members
        )
        row["direct_exposure_adoption_count"] = direct_adoptions
        row["direct_exposure_opportunity_count"] = direct_opportunities
        row["direct_exposure_adoption_rate_micro"] = (
            direct_adoptions / direct_opportunities
            if direct_opportunities
            else None
        )
        row["direct_exposure_opportunity_conversation_n"] = sum(
            int(member.get("direct_exposure_opportunity_count") or 0) > 0
            for member in members
        )

        # Judge-based exposure is separate from the answer-level routing
        # proxy above. Preserve unmeasured (no complete labels/records) as
        # null, while a measured conversation with zero opportunities has
        # explicit zero counts and a null rate. "Measured" here means the
        # raw records were supplied and judge keys were unambiguous; individual
        # unusable labels remain visible in exposed-versus-evaluable counts.
        semantic_measured = [
            member
            for member in members
            if member.get(
                "semantic_direct_exposure_evaluable_opportunity_count"
            ) is not None
        ]
        semantic_adoptions = (
            sum(
                int(member["semantic_direct_exposure_adoption_count"])
                for member in semantic_measured
            )
            if semantic_measured
            else None
        )
        semantic_evaluable = (
            sum(
                int(
                    member[
                        "semantic_direct_exposure_evaluable_opportunity_count"
                    ]
                )
                for member in semantic_measured
            )
            if semantic_measured
            else None
        )
        semantic_exposed = (
            sum(
                int(
                    member[
                        "semantic_direct_exposure_exposed_opportunity_count"
                    ]
                )
                for member in semantic_measured
            )
            if semantic_measured
            else None
        )
        row["semantic_direct_exposure_adoption_count"] = semantic_adoptions
        row["semantic_direct_exposure_evaluable_opportunity_count"] = (
            semantic_evaluable
        )
        row["semantic_direct_exposure_exposed_opportunity_count"] = semantic_exposed
        row["semantic_direct_exposure_adoption_rate_micro"] = (
            semantic_adoptions / semantic_evaluable
            if semantic_evaluable
            else None
        )
        row["semantic_direct_exposure_measured_conversation_n"] = len(
            semantic_measured
        )
        row["semantic_direct_exposure_evaluable_conversation_n"] = sum(
            int(
                member[
                    "semantic_direct_exposure_evaluable_opportunity_count"
                ]
            ) > 0
            for member in semantic_measured
        )

        record_count = sum(int(member.get("record_count") or 0) for member in members)
        parse_failure_count = sum(
            int(member.get("parse_failure_count") or 0) for member in members
        )
        generation_failure_count = sum(
            int(member.get("generation_failure_count") or 0) for member in members
        )
        audit_n = sum(
            int(member.get("answer_rationale_audit_n") or 0) for member in members
        )
        inconsistency_count = sum(
            int(member.get("answer_rationale_inconsistency_count") or 0)
            for member in members
        )
        row["record_count"] = record_count
        row["parse_failure_count"] = parse_failure_count
        row["generation_failure_count"] = generation_failure_count
        row["parse_failure_rate"] = (
            parse_failure_count / record_count if record_count else None
        )
        row["generation_failure_rate"] = (
            generation_failure_count / record_count if record_count else None
        )
        row["answer_rationale_inconsistency_count"] = inconsistency_count
        row["answer_rationale_audit_n"] = audit_n
        row["answer_rationale_audit_coverage"] = (
            audit_n / record_count if record_count else None
        )
        row["answer_rationale_inconsistency_rate_micro"] = (
            inconsistency_count / audit_n if audit_n else None
        )
        row["answer_rationale_audit_versions"] = sorted({
            version
            for member in members
            for version in member.get("answer_rationale_audit_versions", [])
        })
        by_round = [m["favored_adoption_by_round"] for m in members
                    if m.get("favored_adoption_by_round") is not None]
        row["favored_adoption_by_round"] = (
            _roundwise_mean(by_round) if by_round else None
        )
        unrelated_by_round = [
            m["unrelated_option_rate_by_round"]
            for m in members
            if m.get("unrelated_option_rate_by_round") is not None
        ]
        row["unrelated_option_rate_by_round"] = (
            _roundwise_mean(unrelated_by_round)
            if unrelated_by_round
            else None
        )
        rows.append(row)
    return rows


CONDITIONING_FIELDS = (
    "seed_trap_uptake",
    "seed_initial_decision_reliance",
)


def aggregate_conditional_summaries(summaries: Sequence[dict]) -> list[dict]:
    """Aggregate outcomes conditional on an observed seed manipulation check.

    These are secondary, post-treatment summaries. The ordinary
    :func:`aggregate_summaries` output is the unconditional summary of
    completed conversations, but it is not the all-attempt seed ITT because a
    seed generation without a replayable answer cannot start a conversation.
    ``scripts/analyze_run.py`` reports that broader population separately in
    ``seed_trial_itt.jsonl``. Rows here are emitted only where the conditioning
    variable was observed, and report both its observed denominator and the
    subset meeting the condition.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for summary in summaries:
        groups[tuple(summary.get(k) for k in AGGREGATE_KEYS)].append(summary)

    rows: list[dict] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        for field in CONDITIONING_FIELDS:
            observed = [member for member in members if member.get(field) is not None]
            if not observed:
                continue
            selected = [member for member in observed if member[field] is True]
            if selected:
                row = aggregate_summaries(selected)[0]
            else:
                row = dict(zip(AGGREGATE_KEYS, key))
                row["n_conversations"] = 0
            row.update({
                "estimand": "conditional",
                "conditioning_variable": field,
                "n_condition_observed": len(observed),
                "n_condition_met": len(selected),
                "condition_met_rate": len(selected) / len(observed),
            })
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
