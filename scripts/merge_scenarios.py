"""Replace whole scenarios in a run with re-collected ones.

When a run loses records to transient API failures, the affected scenarios can
be re-collected and spliced in. Replacement is by whole scenario because
scenario is the bootstrap cluster, so no paired contrast straddles the join.

Every recovery run contributes all of its scenarios, and each of those replaces
the main run's copy wholesale. Recovery runs must not overlap each other, and
all runs must share a corpus. The merged manifest records each source commit
and is marked complete.

Usage:
  uv run python scripts/merge_scenarios.py <main-run> <out-dir> \
      --recovery <run-a> <run-b> --reason "..."
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROW_FILES = ("records.jsonl", "results.jsonl", "seeds.jsonl")
CARRY_FILES = ("scenario_snapshot.json", "git_tracked_changes.patch")


def scenarios_in(run: Path) -> set[str]:
    """Scenario ids present in a run's results."""
    with (run / "results.jsonl").open(encoding="utf-8") as handle:
        return {json.loads(line)["scenario_id"] for line in handle}


def read_rows(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line + "\n" for line in lines if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("main_run", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--recovery", nargs="+", required=True, type=Path)
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    if args.out_dir.exists():
        sys.exit(f"{args.out_dir} already exists")

    main_manifest = json.loads((args.main_run / "run_manifest.json").read_text())
    corpus = main_manifest["scenario_data"]["sha256"]

    recovery: dict[Path, set[str]] = {}
    for run in args.recovery:
        manifest = json.loads((run / "run_manifest.json").read_text())
        if manifest["scenario_data"]["sha256"] != corpus:
            sys.exit(f"{run.name}: corpus {manifest['scenario_data']['sha256'][:12]} "
                     f"!= main {corpus[:12]}")
        recovery[run] = scenarios_in(run)

    overlap = [s for s, n in Counter(
        s for ids in recovery.values() for s in ids
    ).items() if n > 1]
    if overlap:
        sys.exit(f"recovery runs overlap on {sorted(overlap)}")

    replace = set().union(*recovery.values())
    missing = replace - scenarios_in(args.main_run)
    if missing:
        sys.exit(f"not present in main run: {sorted(missing)}")

    args.out_dir.mkdir(parents=True)
    created = datetime.datetime.now(datetime.UTC).isoformat()
    counts: dict[str, dict[str, int]] = {}

    for name in ROW_FILES:
        kept, dropped = [], 0
        with (args.main_run / name).open(encoding="utf-8") as handle:
            for line in handle:
                if json.loads(line)["scenario_id"] in replace:
                    dropped += 1
                else:
                    kept.append(line)
        added: list[str] = []
        for run in recovery:
            added.extend(read_rows(run / name))
        (args.out_dir / name).write_text("".join(kept + added), encoding="utf-8")
        counts[name] = {
            "kept_from_main": len(kept),
            "dropped_from_main": dropped,
            "added_from_recovery": len(added),
            "total": len(kept) + len(added),
        }

    for extra in CARRY_FILES:
        if (args.main_run / extra).exists():
            shutil.copy2(args.main_run / extra, args.out_dir / extra)
    if (args.main_run / "protocol_snapshot").exists():
        shutil.copytree(
            args.main_run / "protocol_snapshot", args.out_dir / "protocol_snapshot"
        )

    sources = [
        {
            "run": run.name,
            "scenarios": sorted(ids),
            "git_commit": json.loads(
                (run / "run_manifest.json").read_text()
            )["git"]["commit"],
        }
        for run, ids in recovery.items()
    ]
    main_manifest["status"]["state"] = "complete"
    main_manifest["status"]["completed_at_utc"] = created
    main_manifest["merge"] = {
        "kind": "merge_scenarios_v1",
        "main_run": args.main_run.name,
        "main_git_commit": main_manifest["git"]["commit"],
        "recovery_runs": sources,
        "replaced_scenarios": sorted(replace),
        "reason": args.reason,
        "corpus_sha256": corpus,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(main_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "merge_report.json").write_text(
        json.dumps(
            {
                "created_at_utc": created,
                "main_run": args.main_run.name,
                "recovery_runs": sources,
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
    print(f"  replaced {len(replace)} scenario(s): {', '.join(sorted(replace))}")
    for name, c in counts.items():
        print(
            f"  {name:<16} kept {c['kept_from_main']:>6}  "
            f"dropped {c['dropped_from_main']:>4}  "
            f"added {c['added_from_recovery']:>4}  total {c['total']:>6}"
        )


if __name__ == "__main__":
    main()
