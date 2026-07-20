"""Merge the teammate judge's labels onto a run's agent-round records.

The judge (Sonnet 5, teammate-owned) emits JSONL rows conforming to
schemas.JudgedAgentRound. This script validates those rows, joins them to
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
    load_agent_records,
    load_judged_rounds,
    merge_judged,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="outputs/raw/<stamp> directory")
    parser.add_argument("judged", type=Path, help="judge output JSONL (JudgedAgentRound rows)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory; default outputs/judged/<run-stamp>")
    args = parser.parse_args()

    records_path = args.run_dir / "records.jsonl"
    if not records_path.exists():
        sys.exit(f"no records.jsonl in {args.run_dir}")

    records = load_agent_records(records_path)
    judged, invalid = load_judged_rounds(args.judged)
    matched, orphans, unjudged, conflicts = merge_judged(records, judged)

    out_dir = args.out or Path("outputs/judged") / args.run_dir.name
    merged_rows = [
        {
            **record.model_dump(mode="json"),
            **{f"judge_{k}" if not k.startswith("judge_") else k: v
               for k, v in label.model_dump(mode="json").items()
               if k not in ("run_id", "agent_id", "round", "assumption_id")},
        }
        for record, label in matched.values()
    ]
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
        "invalid_lines": [{"line": n, "error": e} for n, e in invalid],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"records={len(records)}  judged valid={len(judged)} invalid={len(invalid)}")
    print(f"matched={len(matched)} ({report['coverage']:.1%} coverage)  "
          f"orphans={len(orphans)}  unjudged={len(unjudged)}  conflicts={len(conflicts)}")
    print(f"wrote {out_dir / 'merged.jsonl'} and report.json")
    if invalid or conflicts:
        print("\n!! judge output problems — see report.json", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
