"""Replace whole scenarios in a run with re-collected ones.

When a run loses records to transient API failures, the affected scenarios can
be re-collected and spliced in. Replacement is by whole scenario because
scenario is the bootstrap cluster, so no paired contrast straddles the join.

Both runs must share a corpus; the merged manifest records each source commit
and is marked complete.

Usage:
  uv run python scripts/merge_scenarios.py <main-run> <recovery-run> <out-dir> \
      --scenarios foodsafety_001
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

ROW_FILES = ("records.jsonl", "results.jsonl", "seeds.jsonl")
CARRY_FILES = ("scenario_snapshot.json", "git_tracked_changes.patch")


def split_rows(path: Path, replace: set[str]) -> tuple[list[str], int]:
    """Return (rows to keep, count dropped) for one JSONL file."""
    keep, dropped = [], 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if json.loads(line)["scenario_id"] in replace:
                dropped += 1
            else:
                keep.append(line)
    return keep, dropped


def read_rows(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line + "\n" for line in lines if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("main_run", type=Path)
    parser.add_argument("recovery_run", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    replace = set(args.scenarios)
    if args.out_dir.exists():
        sys.exit(f"{args.out_dir} already exists")

    main_manifest = json.loads((args.main_run / "run_manifest.json").read_text())
    reco_manifest = json.loads((args.recovery_run / "run_manifest.json").read_text())
    main_sha = main_manifest["scenario_data"]["sha256"]
    reco_sha = reco_manifest["scenario_data"]["sha256"]
    if main_sha != reco_sha:
        sys.exit(f"corpus mismatch: main {main_sha[:12]} != recovery {reco_sha[:12]}")

    args.out_dir.mkdir(parents=True)
    created = datetime.datetime.now(datetime.UTC).isoformat()
    counts: dict[str, dict[str, int]] = {}

    for name in ROW_FILES:
        kept, dropped = split_rows(args.main_run / name, replace)
        added = read_rows(args.recovery_run / name)
        (args.out_dir / name).write_text("".join(kept + added), encoding="utf-8")
        counts[name] = {
            "kept_from_main": len(kept),
            "dropped_from_main": dropped,
            "added_from_recovery": len(added),
            "total": len(kept) + len(added),
        }
        if dropped == 0:
            sys.exit(f"{name}: no rows matched {sorted(replace)}")

    for extra in CARRY_FILES:
        if (args.main_run / extra).exists():
            shutil.copy2(args.main_run / extra, args.out_dir / extra)
    if (args.main_run / "protocol_snapshot").exists():
        shutil.copytree(
            args.main_run / "protocol_snapshot", args.out_dir / "protocol_snapshot"
        )

    main_manifest["status"]["state"] = "complete"
    main_manifest["status"]["completed_at_utc"] = created
    main_manifest["merge"] = {
        "kind": "merge_scenarios_v1",
        "main_run": args.main_run.name,
        "recovery_run": args.recovery_run.name,
        "replaced_scenarios": sorted(replace),
        "reason": args.reason,
        "main_git_commit": main_manifest["git"]["commit"],
        "recovery_git_commit": reco_manifest["git"]["commit"],
        "corpus_sha256": main_sha,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(main_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "merge_report.json").write_text(
        json.dumps(
            {
                "created_at_utc": created,
                "main_run": args.main_run.name,
                "recovery_run": args.recovery_run.name,
                "replaced_scenarios": sorted(replace),
                "reason": args.reason,
                "counts": counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"merged -> {args.out_dir}")
    for name, c in counts.items():
        print(
            f"  {name:<16} kept {c['kept_from_main']:>6}  "
            f"dropped {c['dropped_from_main']:>4}  "
            f"added {c['added_from_recovery']:>4}  total {c['total']:>6}"
        )


if __name__ == "__main__":
    main()
