"""Answer-level (and optionally judge-level) analysis of one run directory.

Reads outputs/raw/<stamp>/records.jsonl, joins scenario metadata from the
run's frozen scenario snapshot (or an explicit --data override), and writes
into <run_dir>/analysis/:

  conversations.jsonl   one summary per conversation (run_id)
  aggregates.jsonl      complete/replayable-conversation means grouped by
                        condition x topology x semantic type
  conditional_aggregates.jsonl
                        secondary summaries conditional on seed trap uptake
                        and, with judge labels, seed decision reliance
  seed_trials.jsonl     every attempted seed generation, including failures
  seed_trial_itt.jsonl  true all-attempt seed ITT by condition and topology,
                        including seed generation failures and missing-outcome
                        bounds
  analysis_manifest.json hashes and provenance for this analysis pass

With --judged pointing at a merge_judged.py output (or raw judge JSONL),
judge-level metrics are added to each covered conversation.

Usage:
  uv run python scripts/analyze_run.py outputs/raw/<stamp>
  uv run python scripts/analyze_run.py <run-dir> --judged <merged.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from assumption_propagation.analysis import (
    aggregate_conditional_summaries,
    aggregate_summaries,
    group_by_run,
    load_agent_records,
    merge_judged,
    summarize_conversation,
    summarize_judged_conversation,
    validate_conversation_records,
    variant_lookup,
    write_jsonl,
)
from assumption_propagation.schemas import (
    CORRECTION_SCOPE,
    INJECTION_CONDITIONS,
    JudgedAgentRound,
    Scenario,
    SeedInstance,
    Topology,
)
from assumption_propagation.judge import load_assumption_references
from assumption_propagation.reproducibility import (
    RUN_MANIFEST_NAME,
    resolve_scenario_data,
    sha256_file,
)


def load_scenarios(path: Path) -> list[Scenario]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    malformed = [
        entry for entry in entries
        if "scenario_id" not in entry and entry.get("legacy") is not True
    ]
    if malformed:
        raise ValueError(
            "scenario data contains undeclared malformed entries: "
            + ", ".join(str(entry.get("id", "?")) for entry in malformed)
        )
    scenarios = [
        Scenario.model_validate(entry)
        for entry in entries
        if entry.get("legacy") is not True
    ]
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    duplicate_scenarios = sorted(
        key for key, count in Counter(scenario_ids).items() if count > 1
    )
    assumption_ids = [
        variant.assumption_id
        for scenario in scenarios
        for variant in scenario.assumption_variants
    ]
    duplicate_assumptions = sorted(
        key for key, count in Counter(assumption_ids).items() if count > 1
    )
    if duplicate_scenarios or duplicate_assumptions:
        raise ValueError(
            "scenario data contains duplicate identifiers: "
            f"scenario_id={duplicate_scenarios}, "
            f"assumption_id={duplicate_assumptions}"
        )
    return scenarios


def _manifest_version_at_least(manifest: dict, minimum: tuple[int, int]) -> bool:
    """Return whether a simple major.minor manifest version meets a minimum."""
    value = manifest.get("manifest_version")
    if not isinstance(value, str):
        return False
    try:
        parts = tuple(int(part) for part in value.split(".")[:2])
    except ValueError:
        return False
    return len(parts) == 2 and parts >= minimum


def _validated_planned_topologies(
    run_manifest: dict | None,
    injection_summaries: list[dict],
) -> list[str]:
    """Resolve the topology trial set without silently accepting bad plans."""
    if run_manifest is None:
        raw = sorted({summary["topology"] for summary in injection_summaries})
        if not raw:
            return []
    else:
        try:
            raw = run_manifest["protocol"]["topologies"]
        except (KeyError, TypeError) as exc:
            raise ValueError("manifest is missing protocol.topologies") from exc
        if not isinstance(raw, list) or not raw:
            raise ValueError("manifest protocol.topologies must be a non-empty list")
        if any(not isinstance(value, str) for value in raw):
            raise ValueError("manifest protocol.topologies must contain only strings")

    duplicates = sorted(key for key, count in Counter(raw).items() if count > 1)
    allowed = {
        topology.value
        for topology in Topology
        if topology != Topology.INDEPENDENT
    }
    unknown = sorted(set(raw) - allowed)
    if duplicates or unknown:
        raise ValueError(
            "invalid planned topologies: "
            f"duplicates={duplicates}, unknown_or_control={unknown}; "
            f"allowed={sorted(allowed)}"
        )
    return list(raw)


def _index_injection_summaries(
    summaries: list[dict],
    trial_by_id: dict[str, dict],
    planned_topologies: list[str],
) -> dict[tuple[str, str, str], list[dict]]:
    """Index seeded outcomes by arm, seed trial, and topology.

    Every corrective-scope arm replays a frozen seed.  Keeping ``condition``
    in the key prevents the four matched outcomes from being mistaken for
    duplicate runs of the legacy single-injection condition.
    """
    indexed: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for summary in summaries:
        # Legacy callers and hand-built diagnostic summaries predate the
        # condition field; their only possible seeded arm was ``injection``.
        condition = summary.get("condition") or "injection"
        if condition not in INJECTION_CONDITIONS:
            raise ValueError(
                f"seeded summary {summary['run_id']} has non-injection condition "
                f"{condition!r}"
            )
        seed_id = summary.get("seed_instance_id")
        if not seed_id:
            raise ValueError(
                f"injection summary {summary['run_id']} is missing seed_instance_id"
            )
        trial = trial_by_id.get(seed_id)
        if trial is None:
            raise ValueError(
                f"injection summary {summary['run_id']} references unknown "
                f"seed_instance_id {seed_id!r}"
            )
        if summary["topology"] not in planned_topologies:
            raise ValueError(
                f"injection summary {summary['run_id']} uses topology "
                f"{summary['topology']!r}, which was not in the run plan"
            )
        if (
            summary["scenario_id"] != trial["scenario_id"]
            or summary["assumption_id"] != trial["assumption_id"]
        ):
            raise ValueError(
                f"injection summary {summary['run_id']} does not match seed "
                f"{seed_id!r}'s scenario/assumption"
            )
        indexed[(condition, seed_id, summary["topology"])].append(summary)
    return dict(indexed)


def _validated_expected_injection_conditions(
    run_manifest: dict | None,
    injection_summaries: list[dict],
) -> list[str]:
    """Resolve the planned seeded arms without hiding missing conditions.

    New manifests declare ``protocol.injection_conditions`` explicitly.
    Older manifests are reconstructed from their scope/ablation switches;
    runs with no manifest fall back to the conditions actually observed.  The
    final default preserves legacy runs, whose sole seeded arm was
    ``injection``.
    """
    observed = sorted({
        summary.get("condition") or "injection"
        for summary in injection_summaries
    })
    unknown_observed = sorted(set(observed) - set(INJECTION_CONDITIONS), key=str)
    if unknown_observed:
        raise ValueError(
            f"unknown observed injection conditions: {unknown_observed}; "
            f"allowed={list(INJECTION_CONDITIONS)}"
        )

    if run_manifest is None:
        return observed

    protocol = run_manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("manifest is missing a valid protocol object")

    raw = protocol.get("injection_conditions")
    if raw is None:
        # Legacy manifests recorded switches rather than the resulting arm
        # list. Prefer those declared switches over observations so a missing
        # arm in a completed run cannot validate itself away.
        if protocol.get("scope_ladder") is True:
            raw = list(CORRECTION_SCOPE)
        elif protocol.get("ablation_arm") is True:
            raw = ["injection", "injection_no_correction"]
        elif protocol.get("withhold_corrective") is True:
            raw = ["injection_no_correction"]
        elif protocol.get("skip_injection") is True:
            raw = []
        elif observed:
            # Manifest versions predating the switches can still be analyzed,
            # but their arm plan is necessarily recoverable only from output.
            raw = observed
        else:
            raw = ["injection"]

    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ValueError(
            "manifest protocol.injection_conditions must be a list of strings"
        )
    duplicates = sorted(key for key, count in Counter(raw).items() if count > 1)
    unknown = sorted(set(raw) - set(INJECTION_CONDITIONS))
    if duplicates or unknown:
        raise ValueError(
            "invalid planned injection conditions: "
            f"duplicates={duplicates}, unknown={unknown}; "
            f"allowed={list(INJECTION_CONDITIONS)}"
        )
    if not raw and protocol.get("skip_injection") is not True:
        raise ValueError(
            "manifest plans no injection conditions but skip_injection is not true"
        )
    return list(raw)


def _validate_injection_outcome_grid(
    indexed: dict[tuple[str, str, str], list[dict]],
    seed_trials: list[dict],
    planned_conditions: list[str],
    planned_topologies: list[str],
    *,
    require_complete: bool,
) -> None:
    """Validate one matched outcome per replayable seed/arm/topology cell.

    Incomplete runs may omit cells for diagnostics, but duplicate cells and
    outcomes attached to unreplayable seeds are invalid in every run state.
    """
    planned_keys = {
        (condition, trial["seed_instance_id"], topology)
        for trial in seed_trials
        for condition in planned_conditions
        for topology in planned_topologies
    }
    extra = sorted(set(indexed) - planned_keys)
    if extra:
        raise ValueError(f"seeded outcomes outside the planned grid: {extra}")

    for trial in seed_trials:
        for condition in planned_conditions:
            for topology in planned_topologies:
                key = (condition, trial["seed_instance_id"], topology)
                count = len(indexed.get(key, []))
                if count > 1:
                    raise ValueError(
                        f"duplicate {condition} outcomes for "
                        f"{trial['seed_instance_id']} at {topology}"
                    )
                if count and not trial["replayable"]:
                    raise ValueError(
                        f"non-replayable seed {trial['seed_instance_id']} has a "
                        f"{condition} outcome at {topology}"
                    )
                expected = int(trial["replayable"])
                if require_complete and count != expected:
                    raise ValueError(
                        f"completed run has {count} {condition} outcomes for seed "
                        f"{trial['seed_instance_id']} at {topology}; expected {expected}"
                    )


def _summarize_seed_trial_itt(
    seed_outcome_rows: list[dict],
    *,
    complete_run: bool,
) -> list[dict]:
    """Aggregate all attempted seeds separately for every scope arm."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_outcome_rows:
        groups[
            (
                row["condition"],
                row["scenario_id"],
                row["assumption_id"],
                row["semantic_type"],
                row["correct_answer"],
                row["favored_option"],
                row["topology"],
            )
        ].append(row)

    result: list[dict] = []
    for key, members in sorted(groups.items()):
        attempted = len(members)
        replayable = sum(member["replayable"] for member in members)
        observed = [member for member in members if member["outcome_observed"]]
        missing = attempted - len(observed)
        correct = sum(
            member["final_plurality_accuracy"] == 1.0 for member in observed
        )
        favored = sum(
            member["final_plurality_favored"] is True for member in observed
        )
        result.append({
            "condition": key[0],
            "scenario_id": key[1],
            "assumption_id": key[2],
            "semantic_type": key[3],
            "correct_answer": key[4],
            "favored_option": key[5],
            "topology": key[6],
            "estimand": "all_attempted_seed_trials",
            "complete_run": complete_run,
            "n_seed_trials": attempted,
            "n_replayable": replayable,
            "n_unreplayable": attempted - replayable,
            "seed_replayability_rate": replayable / attempted if attempted else None,
            "n_outcomes_observed": len(observed),
            "n_missing_outcomes": missing,
            "n_replayable_without_outcome": sum(
                member["replayable"] and not member["outcome_observed"]
                for member in members
            ),
            "outcome_coverage": len(observed) / attempted if attempted else None,
            "seed_trap_uptake_rate_replayable": (
                sum(member["seed_trap_uptake"] is True for member in members)
                / replayable
                if replayable
                else None
            ),
            "final_plurality_accuracy_observed": (
                correct / len(observed) if observed else None
            ),
            "n_final_plurality_correct_observed": correct,
            "final_plurality_accuracy_all_trials_missing_as_incorrect": (
                correct / attempted if attempted else None
            ),
            "final_plurality_accuracy_all_trials_failure_as_incorrect": (
                correct / attempted if attempted and complete_run else None
            ),
            "final_plurality_accuracy_missing_bounds": (
                [correct / attempted, (correct + missing) / attempted]
                if attempted
                else None
            ),
            "favored_plurality_rate_observed": (
                favored / len(observed) if observed else None
            ),
        })
    return result


def _validate_completed_output_files(
    run_dir: Path,
    manifest_path: Path,
    observed_files: object,
) -> None:
    """Verify the row count and digest of every raw output in a complete run."""
    if not isinstance(observed_files, dict):
        raise ValueError(
            f"invalid {manifest_path}: status.observed_files must be an object"
        )
    for name, path in (
        ("seeds", run_dir / "seeds.jsonl"),
        ("records", run_dir / "records.jsonl"),
        ("results", run_dir / "results.jsonl"),
    ):
        expected = observed_files.get(name)
        if not isinstance(expected, dict):
            raise ValueError(
                f"invalid {manifest_path}: no valid observed_files.{name}"
            )
        expected_rows = expected.get("rows")
        expected_sha256 = expected.get("sha256")
        if (
            not isinstance(expected_rows, int)
            or isinstance(expected_rows, bool)
            or expected_rows < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ValueError(
                f"invalid {manifest_path}: malformed observed_files.{name}"
            )
        if not path.is_file():
            raise ValueError(f"completed run is missing recorded output: {path}")
        actual_rows = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if actual_rows != expected_rows:
            raise ValueError(
                f"completed-run row-count mismatch for {path}: "
                f"manifest={expected_rows}, actual={actual_rows}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"completed-run hash mismatch for {path}: "
                f"manifest={expected_sha256}, actual={actual_sha256}"
            )


def load_judged_any(path: Path) -> list[JudgedAgentRound]:
    """Accept either raw judge JSONL or merge_judged.py merged.jsonl rows."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if "judge_assumption_state" in data:  # merged row: un-prefix label fields
            label_data = {
                "run_id": data["run_id"],
                "agent_id": data["agent_id"],
                "round": data["round"],
                "assumption_id": data["assumption_id"],
            }
            for field in JudgedAgentRound.model_fields:
                if field in label_data:
                    continue
                merged_name = (
                    "judge_text_source"
                    if field == "judged_text_source"
                    else field
                    if field.startswith("judge_")
                    else f"judge_{field}"
                )
                if merged_name in data:
                    label_data[field] = data[merged_name]
            data = label_data
        rows.append(JudgedAgentRound.model_validate(data))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="scenario JSON override; default <run_dir>/scenario_snapshot.json, "
             "legacy runs without a snapshot require this override",
    )
    parser.add_argument("--judged", type=Path, default=None)
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="permit development-only mock labels (never use for research results)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="analyze an explicitly incomplete/interrupted run for diagnostics only",
    )
    args = parser.parse_args()

    records_path = args.run_dir / "records.jsonl"
    if not records_path.exists():
        sys.exit(f"no records.jsonl in {args.run_dir}")

    try:
        data_path, data_source = resolve_scenario_data(args.run_dir, args.data)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    try:
        records = load_agent_records(records_path)
        scenarios = load_scenarios(data_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        sys.exit(f"invalid analysis input: {exc}")
    variants = variant_lookup(scenarios)
    correct_by_scenario = {s.scenario_id: s.correct_answer for s in scenarios}
    seeds_path = args.run_dir / "seeds.jsonl"
    seeds: list[SeedInstance] = []
    if seeds_path.is_file():
        for line_number, line in enumerate(
            seeds_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                seeds.append(SeedInstance.model_validate_json(line))
            except ValidationError as exc:
                sys.exit(f"invalid seed row {seeds_path}:{line_number}: {exc}")
    seed_ids = [seed.seed_instance_id for seed in seeds]
    duplicate_seed_ids = sorted(
        key for key, count in Counter(seed_ids).items() if count > 1
    )
    seed_coordinates = [
        (seed.assumption_id, seed.seed_trial_index) for seed in seeds
    ]
    duplicate_seed_coordinates = sorted(
        key for key, count in Counter(seed_coordinates).items() if count > 1
    )
    if duplicate_seed_ids or duplicate_seed_coordinates:
        sys.exit(
            "refusing duplicate seed rows: "
            f"seed_instance_id={duplicate_seed_ids}, "
            f"(assumption_id, seed_trial_index)={duplicate_seed_coordinates}"
        )
    if any(not seed.seed_instance_id.strip() for seed in seeds):
        sys.exit("refusing seed row with blank seed_instance_id")
    run_manifest_path = args.run_dir / RUN_MANIFEST_NAME
    try:
        run_manifest = (
            json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest_path.is_file()
            else None
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        sys.exit(f"invalid {run_manifest_path}: {exc}")
    run_state = None
    if run_manifest is not None:
        status = run_manifest.get("status")
        if not isinstance(status, dict):
            sys.exit(f"invalid {run_manifest_path}: status must be an object")
        run_state = status.get("state")
        if _manifest_version_at_least(run_manifest, (1, 2)) and not isinstance(
            run_state, str
        ):
            sys.exit(f"invalid {run_manifest_path}: status.state is required")
        if run_state != "complete" and not args.allow_incomplete:
            sys.exit(
                f"refusing run with manifest status {run_state!r}; rerun with "
                "--allow-incomplete only for diagnostic analysis"
            )
        if run_state == "complete":
            observed_files = status.get("observed_files")
            if observed_files is not None or _manifest_version_at_least(
                run_manifest, (1, 2)
            ):
                try:
                    _validate_completed_output_files(
                        args.run_dir, run_manifest_path, observed_files
                    )
                except (OSError, UnicodeError, ValueError) as exc:
                    sys.exit(str(exc))

            planned_seed_generations = status.get("planned_seed_generations")
            if _manifest_version_at_least(run_manifest, (1, 2)):
                if (
                    not isinstance(planned_seed_generations, int)
                    or isinstance(planned_seed_generations, bool)
                    or planned_seed_generations < 1
                ):
                    sys.exit(
                        f"invalid {run_manifest_path}: "
                        "status.planned_seed_generations must be a positive integer"
                    )
                if len(seeds) != planned_seed_generations:
                    sys.exit(
                        "completed-run seed-count mismatch: "
                        f"manifest={planned_seed_generations}, actual={len(seeds)}"
                    )

    judged_by_run: dict[str, list[JudgedAgentRound]] = defaultdict(list)
    if args.judged:
        try:
            rows = load_judged_any(args.judged)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            sys.exit(f"invalid judged data {args.judged}: {exc}")
        mock_rows = [row for row in rows if row.judge_model.startswith("mock:")]
        if mock_rows and not args.allow_mock:
            sys.exit(
                f"refusing {len(mock_rows)} development-only mock judge rows; "
                "use --allow-mock only for plumbing tests"
            )
        references = load_assumption_references(data_path)
        matched, orphans, unjudged, conflicts = merge_judged(
            records,
            rows,
            verification_gap_defined={
                assumption_id: reference.verification_gap is not None
                for assumption_id, reference in references.items()
            },
        )
        if orphans or unjudged or conflicts:
            sys.exit(
                "judge labels do not form a complete, conflict-free join: "
                f"orphan={len(orphans)} "
                f"unjudged={len(unjudged)} conflicts={len(conflicts)}; "
                "run scripts/merge_judged.py for the detailed report"
            )
        for _, row in matched.values():
            judged_by_run[row.run_id].append(row)

    summaries = []
    for run_id, conversation in group_by_run(records).items():
        first = conversation[0]
        expected_n_rounds = None
        if run_manifest is not None and not (
            args.allow_incomplete and run_state != "complete"
        ):
            try:
                protocol = run_manifest["protocol"]
                expected_n_rounds = (
                    protocol["independent_control_n_rounds"]
                    if first.condition.value == "independent_control"
                    else protocol["n_rounds"]
                )
            except (KeyError, TypeError) as exc:
                sys.exit(
                    f"invalid {run_manifest_path}: missing protocol round count ({exc})"
                )
        try:
            validate_conversation_records(
                conversation, expected_n_rounds=expected_n_rounds
            )
        except ValueError as exc:
            sys.exit(f"refusing malformed conversation: {exc}")
        correct = correct_by_scenario.get(first.scenario_id)
        if correct is None:
            sys.exit(
                f"refusing {run_id}: scenario {first.scenario_id!r} is absent from "
                f"the resolved scenario data {data_path}"
            )
        favored = semantic = None
        if first.assumption_id:
            if first.assumption_id not in variants:
                sys.exit(
                    f"refusing {run_id}: assumption {first.assumption_id!r} is absent "
                    f"from the resolved scenario data {data_path}"
                )
            variant_scenario, variant = variants[first.assumption_id]
            if variant_scenario.scenario_id != first.scenario_id:
                sys.exit(
                    f"refusing {run_id}: assumption {first.assumption_id!r} belongs to "
                    f"{variant_scenario.scenario_id!r}, not {first.scenario_id!r}"
                )
            favored = variant.favored_option
            semantic = variant.semantic_type.value
        summary = summarize_conversation(
            conversation, correct_answer=correct,
            favored_option=favored, semantic_type=semantic,
        )
        labels = judged_by_run.get(run_id)
        if labels and first.seed_agent_id is not None:
            summary.update(summarize_judged_conversation(
                labels,
                seed_agent_id=first.seed_agent_id,
                n_records=len(conversation),
                records=conversation,
            ))
        summaries.append(summary)

    seed_trial_rows: list[dict] = []
    for seed in seeds:
        if seed.assumption_id not in variants:
            sys.exit(
                f"refusing seed {seed.seed_instance_id}: unknown assumption "
                f"{seed.assumption_id!r}"
            )
        seed_scenario, seed_variant = variants[seed.assumption_id]
        if seed_scenario.scenario_id != seed.scenario_id:
            sys.exit(
                f"refusing seed {seed.seed_instance_id}: assumption/scenario mismatch"
            )
        replayable = bool(seed.answer)
        if seed.included_in_core != replayable:
            sys.exit(
                f"refusing seed {seed.seed_instance_id}: included_in_core="
                f"{seed.included_in_core} disagrees with replayability={replayable}; "
                "valid seed outputs must never be stance-filtered"
            )
        seed_trial_rows.append({
            "seed_instance_id": seed.seed_instance_id,
            "scenario_id": seed.scenario_id,
            "assumption_id": seed.assumption_id,
            "semantic_type": seed_variant.semantic_type.value,
            "seed_trial_index": seed.seed_trial_index,
            "injection_prompt_hash": seed.injection_prompt_hash or None,
            "correct_answer": seed_scenario.correct_answer,
            "favored_option": seed_variant.favored_option,
            "answer": seed.answer or None,
            "replayable": replayable,
            "generation_failed": seed.generation_failed,
            "generation_error_type": seed.generation_error_type,
            "seed_answer_correct": (
                seed.answer == seed_scenario.correct_answer if replayable else None
            ),
            "seed_trap_uptake": (
                seed.answer == seed_variant.favored_option if replayable else None
            ),
            "answer_rationale_inconsistent": seed.answer_rationale_inconsistent,
            "answer_rationale_audit_version": seed.answer_rationale_audit_version,
        })

    trial_by_id = {trial["seed_instance_id"]: trial for trial in seed_trial_rows}
    raw_injection_summaries = [
        summary
        for summary in summaries
        if summary["condition"] in INJECTION_CONDITIONS
    ]
    invalid_control_seed_ids = [
        summary["run_id"]
        for summary in summaries
        if summary["condition"] not in INJECTION_CONDITIONS
        and summary.get("seed_instance_id") is not None
    ]
    if invalid_control_seed_ids:
        sys.exit(
            "refusing control conversations carrying a seed_instance_id: "
            + ", ".join(sorted(invalid_control_seed_ids))
        )
    try:
        planned_topologies = _validated_planned_topologies(
            run_manifest, raw_injection_summaries
        )
        planned_injection_conditions = _validated_expected_injection_conditions(
            run_manifest, raw_injection_summaries
        )
    except ValueError as exc:
        sys.exit(f"invalid {run_manifest_path}: {exc}")

    try:
        injection_summaries = _index_injection_summaries(
            raw_injection_summaries, trial_by_id, planned_topologies
        )
    except ValueError as exc:
        sys.exit(f"refusing malformed injection outcome: {exc}")

    observed_injection_conditions = {
        summary["condition"] for summary in raw_injection_summaries
    }
    unexpected_injection_conditions = (
        observed_injection_conditions - set(planned_injection_conditions)
    )
    if unexpected_injection_conditions:
        sys.exit(
            "refusing injection outcomes outside the planned arm set: "
            + ", ".join(sorted(unexpected_injection_conditions))
        )
    try:
        _validate_injection_outcome_grid(
            injection_summaries,
            seed_trial_rows,
            planned_injection_conditions,
            planned_topologies,
            require_complete=(run_state == "complete"),
        )
    except ValueError as exc:
        sys.exit(f"refusing malformed injection outcome grid: {exc}")

    if run_manifest is not None:
        scenario_manifest = run_manifest.get("scenario_data")
        if not isinstance(scenario_manifest, dict):
            sys.exit(f"invalid {run_manifest_path}: scenario_data must be an object")
        selected_assumption_ids = scenario_manifest.get("selected_assumption_ids")
        if selected_assumption_ids is None and _manifest_version_at_least(
            run_manifest, (1, 2)
        ):
            sys.exit(
                f"invalid {run_manifest_path}: "
                "scenario_data.selected_assumption_ids is required"
            )
        if selected_assumption_ids is not None:
            if (
                not isinstance(selected_assumption_ids, list)
                or any(not isinstance(value, str) for value in selected_assumption_ids)
                or len(selected_assumption_ids) != len(set(selected_assumption_ids))
            ):
                sys.exit(
                    f"invalid {run_manifest_path}: "
                    "scenario_data.selected_assumption_ids must be a unique string list"
                )
            observed_assumption_ids = {
                trial["assumption_id"] for trial in seed_trial_rows
            }
            if run_state == "complete" and observed_assumption_ids != set(
                selected_assumption_ids
            ):
                sys.exit(
                    "completed-run seed assumptions disagree with the manifest: "
                    f"manifest={sorted(selected_assumption_ids)}, "
                    f"observed={sorted(observed_assumption_ids)}"
                )
            if run_state == "complete":
                planned_n_seeds = run_manifest.get("protocol", {}).get("n_seeds")
                if (
                    not isinstance(planned_n_seeds, int)
                    or isinstance(planned_n_seeds, bool)
                    or planned_n_seeds < 1
                ):
                    sys.exit(
                        f"invalid {run_manifest_path}: protocol.n_seeds must be "
                        "a positive integer"
                    )
                expected_coordinates = {
                    (assumption_id, trial_index)
                    for assumption_id in selected_assumption_ids
                    for trial_index in range(planned_n_seeds)
                }
                actual_coordinates = {
                    (trial["assumption_id"], trial["seed_trial_index"])
                    for trial in seed_trial_rows
                }
                if actual_coordinates != expected_coordinates:
                    sys.exit(
                        "completed-run seed trial grid disagrees with the manifest: "
                        f"missing={sorted(expected_coordinates - actual_coordinates)}, "
                        f"extra={sorted(actual_coordinates - expected_coordinates)}"
                    )

    seed_outcome_rows: list[dict] = []
    for trial in seed_trial_rows:
        for condition in planned_injection_conditions:
            for topology in planned_topologies:
                matches = injection_summaries.get(
                    (condition, trial["seed_instance_id"], topology), []
                )
                outcome = matches[0] if matches else None
                seed_outcome_rows.append({
                    **trial,
                    "condition": condition,
                    "topology": topology,
                    "outcome_observed": outcome is not None,
                    "final_plurality_accuracy": (
                        outcome["final_plurality_accuracy"] if outcome else None
                    ),
                    "final_plurality_favored": (
                        outcome["final_plurality_answer"] == outcome["favored_option"]
                        if outcome
                        else None
                    ),
                })

    seed_trial_itt = _summarize_seed_trial_itt(
        seed_outcome_rows,
        complete_run=(run_state == "complete"),
    )

    out_dir = args.run_dir / "analysis"
    write_jsonl(out_dir / "seed_trials.jsonl", seed_trial_rows)
    write_jsonl(out_dir / "seed_trial_itt.jsonl", seed_trial_itt)
    write_jsonl(out_dir / "conversations.jsonl", summaries)
    aggregates = aggregate_summaries(summaries)
    write_jsonl(out_dir / "aggregates.jsonl", aggregates)
    conditional_aggregates = aggregate_conditional_summaries(summaries)
    write_jsonl(out_dir / "conditional_aggregates.jsonl", conditional_aggregates)
    analysis_manifest = {
        "analysis_manifest_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": sys.argv,
        "records": {
            "path": str(records_path),
            "sha256": sha256_file(records_path),
            "n_rows": len(records),
        },
        "seeds": (
            {
                "path": str(seeds_path),
                "sha256": sha256_file(seeds_path),
                "n_rows": len(seeds),
            }
            if seeds_path.is_file()
            else None
        ),
        "scenario_data": {
            "path": str(data_path),
            "source": data_source,
            "sha256": sha256_file(data_path),
        },
        "judged_data": (
            {
                "path": str(args.judged),
                "sha256": sha256_file(args.judged),
            }
            if args.judged
            else None
        ),
        "analysis_sources": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1]
                / "src"
                / "assumption_propagation"
                / "analysis.py",
                Path(__file__).resolve().parents[1]
                / "src"
                / "assumption_propagation"
                / "metrics.py",
                Path(__file__).resolve().parents[1]
                / "src"
                / "assumption_propagation"
                / "aggregation.py",
                Path(__file__).resolve().parents[1]
                / "src"
                / "assumption_propagation"
                / "schemas.py",
                Path(__file__).resolve().parents[1]
                / "src"
                / "assumption_propagation"
                / "topologies.py",
                Path(__file__).resolve().parents[1]
                / "src"
                / "assumption_propagation"
                / "judge.py",
            )
        },
        "n_conversations": len(summaries),
        "planned_injection_conditions": planned_injection_conditions,
        "n_replayable_conversation_aggregate_rows": len(aggregates),
        "n_conditional_aggregate_rows": len(conditional_aggregates),
        "n_seed_trials": len(seed_trial_rows),
        "n_seed_trial_itt_rows": len(seed_trial_itt),
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"{len(summaries)} conversations -> {out_dir} "
        f"(scenario_data={data_path}, source={data_source})\n"
    )
    header = (
        f"{'condition':26s} {'topology':22s} {'semantic':9s} {'n':>3s} "
        f"{'ind':>6s} {'MAS':>6s} {'favored':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in aggregates:
        acc = row["final_individual_accuracy"]
        mas_acc = row["final_plurality_accuracy"]
        fav = row["favored_adoption_final"]
        print(f"{row['condition'] or '-':26s} {row['topology'] or '-':22s} "
              f"{row['semantic_type'] or '-':9s} {row['n_conversations']:>3d} "
              f"{acc:>6.2f} {mas_acc:>6.2f} "
              f"{'-' if fav is None else format(fav, '>8.2f')}")


if __name__ == "__main__":
    main()
