"""Answer-level (and optionally judge-level) analysis of one run directory.

Reads outputs/raw/<stamp>/records.jsonl, joins scenario metadata from
data/scenarios.json, and writes into <run_dir>/analysis/:

  conversations.jsonl   one summary per conversation (run_id)
  aggregates.jsonl      means grouped by condition x topology x semantic type

With --judged pointing at a merge_judged.py output (or raw judge JSONL),
judge-level metrics are added to each covered conversation.

Usage:
  uv run python scripts/analyze_run.py outputs/raw/<stamp>
  uv run python scripts/analyze_run.py outputs/raw/<stamp> --judged outputs/judged/<stamp>/merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from assumption_propagation.analysis import (
    aggregate_summaries,
    group_by_run,
    load_agent_records,
    load_judged_rounds,
    merge_judged,
    summarize_conversation,
    summarize_judged_conversation,
    variant_lookup,
    write_jsonl,
)
from assumption_propagation.schemas import JudgedAgentRound, Scenario
from assumption_propagation.judge import load_assumption_references


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
    return [
        Scenario.model_validate(entry)
        for entry in entries
        if entry.get("legacy") is not True
    ]


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
    parser.add_argument("--data", type=Path, default=Path("data/scenarios.json"))
    parser.add_argument("--judged", type=Path, default=None)
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="permit development-only mock labels (never use for research results)",
    )
    args = parser.parse_args()

    records_path = args.run_dir / "records.jsonl"
    if not records_path.exists():
        sys.exit(f"no records.jsonl in {args.run_dir}")

    records = load_agent_records(records_path)
    scenarios = load_scenarios(args.data)
    variants = variant_lookup(scenarios)
    correct_by_scenario = {s.scenario_id: s.correct_answer for s in scenarios}

    judged_by_run: dict[str, list[JudgedAgentRound]] = defaultdict(list)
    if args.judged:
        invalid: list[tuple[int, str]] = []
        if args.judged.suffix == ".jsonl" and "merged" in args.judged.name:
            rows = load_judged_any(args.judged)
        else:
            rows, invalid = load_judged_rounds(args.judged)
        mock_rows = [row for row in rows if row.judge_model.startswith("mock:")]
        if mock_rows and not args.allow_mock:
            sys.exit(
                f"refusing {len(mock_rows)} development-only mock judge rows; "
                "use --allow-mock only for plumbing tests"
            )
        references = load_assumption_references(args.data)
        matched, orphans, unjudged, conflicts = merge_judged(
            records,
            rows,
            verification_gap_defined={
                assumption_id: reference.verification_gap is not None
                for assumption_id, reference in references.items()
            },
        )
        if invalid or orphans or unjudged or conflicts:
            sys.exit(
                "judge labels do not form a complete, conflict-free join: "
                f"invalid={len(invalid)} orphan={len(orphans)} "
                f"unjudged={len(unjudged)} conflicts={len(conflicts)}; "
                "run scripts/merge_judged.py for the detailed report"
            )
        for _, row in matched.values():
            judged_by_run[row.run_id].append(row)

    summaries = []
    for run_id, conversation in group_by_run(records).items():
        first = conversation[0]
        correct = correct_by_scenario.get(first.scenario_id)
        if correct is None:
            print(f"! skipping {run_id}: scenario {first.scenario_id} not in {args.data}",
                  file=sys.stderr)
            continue
        favored = semantic = None
        if first.assumption_id and first.assumption_id in variants:
            _, variant = variants[first.assumption_id]
            favored = variant.favored_option
            semantic = variant.semantic_type.value
        summary = summarize_conversation(
            conversation, correct_answer=correct,
            favored_option=favored, semantic_type=semantic,
        )
        labels = judged_by_run.get(run_id)
        if labels and first.seed_agent_id is not None:
            summary.update(summarize_judged_conversation(
                labels, seed_agent_id=first.seed_agent_id, n_records=len(conversation),
            ))
        summaries.append(summary)

    out_dir = args.run_dir / "analysis"
    write_jsonl(out_dir / "conversations.jsonl", summaries)
    aggregates = aggregate_summaries(summaries)
    write_jsonl(out_dir / "aggregates.jsonl", aggregates)

    print(f"{len(summaries)} conversations -> {out_dir}\n")
    header = f"{'condition':26s} {'topology':22s} {'semantic':9s} {'n':>3s} {'acc':>6s} {'favored':>8s}"
    print(header)
    print("-" * len(header))
    for row in aggregates:
        acc = row["final_individual_accuracy"]
        fav = row["favored_adoption_final"]
        print(f"{row['condition'] or '-':26s} {row['topology'] or '-':22s} "
              f"{row['semantic_type'] or '-':9s} {row['n_conversations']:>3d} "
              f"{acc:>6.2f} {'-' if fav is None else format(fav, '>8.2f')}")


if __name__ == "__main__":
    main()
