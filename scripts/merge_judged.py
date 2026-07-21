"""Merge validated judge labels onto a run's agent-round records.

The judge runner emits JSONL rows conforming to schemas.JudgedAgentRound.
This script validates those rows, joins them to
records.jsonl by (run_id, agent_id, round), and writes:

  outputs/judged/<run-stamp>/merged.jsonl    record fields + judge_* label fields
  outputs/judged/<run-stamp>/report.json     coverage + every rejected/orphan row

Usage:
  uv run python scripts/merge_judged.py outputs/raw/<stamp> path/to/judged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assumption_propagation.analysis import (
    derive_protocol_metadata,
    load_agent_records,
    load_judged_rounds,
    merge_judged,
    write_jsonl,
)
from assumption_propagation.judge import load_assumption_references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="outputs/raw/<stamp> directory")
    parser.add_argument("judged", type=Path, help="judge output JSONL (JudgedAgentRound rows)")
    parser.add_argument("--data", type=Path, default=Path("data/scenarios.json"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory; default outputs/judged/<run-stamp>")
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
    judged, invalid = load_judged_rounds(args.judged)
    mock_rows = [row for row in judged if row.judge_model.startswith("mock:")]
    if mock_rows and not args.allow_mock:
        sys.exit(
            f"refusing {len(mock_rows)} development-only mock judge rows; "
            "use --allow-mock only for plumbing tests"
        )
    references = load_assumption_references(args.data)
    matched, orphans, unjudged, conflicts = merge_judged(
        records,
        judged,
        verification_gap_defined={
            assumption_id: reference.verification_gap is not None
            for assumption_id, reference in references.items()
        },
    )
    protocol = derive_protocol_metadata(
        records, [label for _, label in matched.values()]
    )

    out_dir = args.out or Path("outputs/judged") / args.run_dir.name
    merged_rows = []
    for key, (record, label) in matched.items():
        label_fields = {}
        for field, value in label.model_dump(mode="json").items():
            if field in ("run_id", "agent_id", "round", "assumption_id"):
                continue
            merged_name = (
                "judge_text_source"
                if field == "judged_text_source"
                else field
                if field.startswith("judge_")
                else f"judge_{field}"
            )
            label_fields[merged_name] = value
        merged_rows.append({
            **record.model_dump(mode="json"),
            **label_fields,
            "judge_target_engaged": label.target_engaged,
            **protocol.get(key, {}),
        })
    write_jsonl(out_dir / "merged.jsonl", merged_rows)

    report = {
        "records_total": len(records),
        "judge_rows_valid": len(judged),
        "judge_rows_invalid": len(invalid),
        "matched": len(matched),
        "coverage": len(matched) / len(records) if records else 0.0,
        "orphan_judge_rows": [o.model_dump(mode="json") for o in orphans],
        "unjudged_record_keys": [
            {"run_id": r.run_id, "agent_id": r.agent_id, "round": r.round}
            for r in unjudged
        ],
        "conflicts": conflicts,
        "mock_judge_rows": len(mock_rows),
        "judge_configurations": [
            {
                "judge_model": model,
                "judge_codebook_version": codebook,
                "judge_prompt_version": prompt,
            }
            for model, codebook, prompt in sorted({
                (row.judge_model, row.judge_codebook_version, row.judge_prompt_version)
                for row in judged
            })
        ],
        "judge_unclear_rows": sum(
            row.assumption_state.value == "unclear" for row in judged
        ),
        "judge_uncertain_rows": sum(row.judge_uncertain for row in judged),
        "invalid_lines": [{"line": n, "error": e} for n, e in invalid],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"records={len(records)}  judged valid={len(judged)} invalid={len(invalid)}")
    print(f"matched={len(matched)} ({report['coverage']:.1%} coverage)  "
          f"orphans={len(orphans)}  unjudged={len(unjudged)}  conflicts={len(conflicts)}")
    print(f"wrote {out_dir / 'merged.jsonl'} and report.json")
    if invalid or conflicts or orphans or unjudged:
        print("\n!! judge output problems — see report.json", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
