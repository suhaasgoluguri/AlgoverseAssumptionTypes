"""Join per-scenario runs into one run that analysis treats like a single run.

Built for run_parallel.py, which collects each scenario in its own run folder,
but any runs can be joined if they were produced under the same protocol. Every
source must match the first on the corpus (canonical hash), client settings,
protocol settings, prompt version and the hash of every protocol source file.
Git commits may differ (committing the launcher itself changes the commit) and
are recorded per source. No scenario may come from two sources, and every
scenario must have the same semantic types selected (--variants is not stored
in the protocol settings, so it is checked from the selected variants).

The joined rows are then put through run_pilot.py's own completion checks (full
seed grid, one result and a complete round grid per conversation) before the
manifest is marked complete, so analyze_run.py accepts it without
--allow-incomplete.

A source contributes all of its scenarios, which requires a completed run, or
only the scenarios named after a colon. That form may take finished scenarios
from a run that was stopped part-way; they still have to pass the checks.

Usage:
  uv run python scripts/combine_runs.py outputs/raw/<name> \\
      --parallel outputs/parallel/<stamp>_<tag> \\
      --source outputs/raw/<earlier-run>:vendor_001 \\
      --reason "scenario-parallel collection"
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import run_pilot  # noqa: E402  (reuse the runner's own validator and loader)
from assumption_propagation.reproducibility import sha256_file  # noqa: E402

ROW_FILES = {"seeds": "seeds.jsonl", "records": "records.jsonl", "results": "results.jsonl"}
# Settings that must be identical in every source. Everything that defines what
# a conversation is lives in one of these.
MATCHED_FIELDS = ("manifest_version", "prompt_version", "client")
# Varies with which scenarios a run selected; merged rather than matched.
PER_SCENARIO_PROTOCOL_FIELDS = ("correction_available_round_by_assumption",)


class CombineError(ValueError):
    pass


def _load_manifest(run: Path) -> dict:
    path = run / run_pilot.RUN_MANIFEST_NAME
    if not path.is_file():
        raise CombineError(f"{run}: no {run_pilot.RUN_MANIFEST_NAME}")
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus_digest(manifest: dict) -> str:
    data = manifest["scenario_data"]
    return data.get("canonical_sha256") or data["sha256"]


def _protocol_without_per_scenario(manifest: dict) -> dict:
    return {
        key: value
        for key, value in manifest["protocol"].items()
        if key not in PER_SCENARIO_PROTOCOL_FIELDS
    }


def _source_hashes(manifest: dict) -> dict[str, str]:
    return {path: entry["sha256"] for path, entry in manifest["protocol_sources"].items()}


def parse_source(spec: str) -> tuple[Path, list[str] | None]:
    """'RUN' or 'RUN:scenario_a,scenario_b'."""
    if ":" in spec:
        run, scenarios = spec.rsplit(":", 1)
        names = [name for name in scenarios.split(",") if name]
        if not names:
            raise CombineError(f"no scenarios named after ':' in {spec!r}")
        return Path(run), names
    return Path(spec), None


def sources_from_parallel(group: Path) -> list[tuple[Path, list[str] | None]]:
    """Every completed scenario run listed by a run_parallel.py manifest.

    Failed or stopped scenarios are skipped with a notice, so a rerun of them
    in a second group can be joined alongside. The corpus-coverage check in
    combine() still catches any scenario that ends up missing.
    """
    manifest = json.loads((group / "parallel_manifest.json").read_text(encoding="utf-8"))
    sources = []
    for scenario_id, entry in manifest["scenarios"].items():
        if entry.get("state") != "complete" or "run_dir" not in entry:
            print(f"  skipping {scenario_id} from {group.name}: {entry.get('state')}")
            continue
        run = Path(entry["run_dir"])
        sources.append((run if run.is_absolute() else REPO_ROOT / run, [scenario_id]))
    return sources


def check_and_select(
    sources: list[tuple[Path, list[str] | None]],
) -> tuple[list[tuple[Path, dict, list[str]]], list[str]]:
    """Validate sources against each other; return (run, manifest, taken) rows."""
    if not sources:
        raise CombineError("no sources given")
    resolved: list[tuple[Path, dict, list[str]]] = []
    first_run, first = None, None
    seen: dict[str, str] = {}
    for run, named in sources:
        manifest = _load_manifest(run)
        selected = manifest["scenario_data"]["selected_scenario_ids"]
        if named is None:
            if manifest["status"].get("state") != "complete":
                raise CombineError(
                    f"{run.name} is {manifest['status'].get('state')!r}; name the "
                    "finished scenarios to take from it as RUN:scenario_a,scenario_b"
                )
            taken = list(selected)
        else:
            not_in_run = sorted(set(named) - set(selected))
            if not_in_run:
                raise CombineError(f"{run.name} did not select {not_in_run}")
            taken = list(named)
        if first is None:
            first_run, first = run, manifest
        else:
            problems = []
            if _corpus_digest(manifest) != _corpus_digest(first):
                problems.append("scenario corpus")
            for field in MATCHED_FIELDS:
                if manifest.get(field) != first.get(field):
                    problems.append(field)
            if _protocol_without_per_scenario(manifest) != _protocol_without_per_scenario(first):
                problems.append("protocol")
            mine, theirs = _source_hashes(manifest), _source_hashes(first)
            changed = sorted(p for p in set(mine) | set(theirs) if mine.get(p) != theirs.get(p))
            if changed:
                problems.append(f"protocol source files {changed}")
            if problems:
                raise CombineError(
                    f"{run.name} differs from {first_run.name} in: {'; '.join(problems)}"
                )
        for scenario_id in taken:
            if scenario_id in seen:
                raise CombineError(
                    f"{scenario_id} comes from both {seen[scenario_id]} and {run.name}"
                )
            seen[scenario_id] = run.name
        resolved.append((run, manifest, taken))
    return resolved, list(seen)


def _rows(path: Path, keep: set[str]) -> list[str]:
    """Lines of a JSONL output whose scenario is in ``keep``.

    A run stopped mid-write can leave a partial final line. It belongs to
    whatever was running at the time and is skipped; if that was a taken
    scenario, its missing row fails the completion checks. An unreadable line
    anywhere else is an error.
    """
    if not path.is_file():
        return []
    rows = []
    unreadable: int | None = None
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if unreadable is not None:
                raise CombineError(f"{path}: unreadable line {unreadable}")
            try:
                scenario_id = json.loads(line)["scenario_id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                unreadable = number
                continue
            if scenario_id in keep:
                rows.append(line if line.endswith("\n") else line + "\n")
    if unreadable is not None:
        print(f"  ignoring a partial final line in {path} (line {unreadable})")
    return rows


def _check_variant_selection(
    resolved: list[tuple[Path, dict, list[str]]], by_id: dict
) -> None:
    """Every taken scenario must have the same semantic types selected."""
    types: set[str] = set()
    for _, manifest, taken in resolved:
        chosen = set(manifest["scenario_data"]["selected_assumption_ids"])
        for scenario_id in taken:
            types |= {
                variant.semantic_type.value
                for variant in by_id[scenario_id].assumption_variants
                if variant.assumption_id in chosen
            }
    for run, manifest, taken in resolved:
        chosen = set(manifest["scenario_data"]["selected_assumption_ids"])
        for scenario_id in taken:
            variants = by_id[scenario_id].assumption_variants
            got = sorted(v.semantic_type.value for v in variants if v.assumption_id in chosen)
            expected = sorted(v.semantic_type.value for v in variants if v.semantic_type.value in types)
            if got != expected:
                raise CombineError(
                    f"{run.name} selected variant types {got} for {scenario_id}; "
                    f"the other sources imply {expected}"
                )


def combine(
    out_dir: Path,
    sources: list[tuple[Path, list[str] | None]],
    *,
    reason: str,
    argv: list[str],
    require_all: bool = True,
) -> dict:
    if out_dir.exists():
        raise CombineError(f"{out_dir} already exists")
    resolved, taken_ids = check_and_select(sources)
    first_run, first, _ = resolved[0]

    corpus = run_pilot.load_scenarios(first_run / first["scenario_data"]["snapshot_path"], None)
    order = {scenario.scenario_id: index for index, scenario in enumerate(corpus)}
    missing = sorted(set(taken_ids) - set(order))
    if missing:
        raise CombineError(f"not in the corpus snapshot: {missing}")
    absent = [s for s in order if s not in set(taken_ids)]
    if require_all and absent:
        raise CombineError(
            f"{len(absent)} corpus scenario(s) missing: {', '.join(absent)} "
            "(pass --allow-partial to join a subset on purpose)"
        )
    _check_variant_selection(resolved, {scenario.scenario_id: scenario for scenario in corpus})
    resolved.sort(key=lambda item: min(order[s] for s in item[2]))
    scenario_ids = sorted(taken_ids, key=order.__getitem__)
    scenarios = [s for s in corpus if s.scenario_id in set(scenario_ids)]

    planned_assumptions = {
        assumption_id
        for _, manifest, _ in resolved
        for assumption_id in manifest["scenario_data"]["selected_assumption_ids"]
    }
    assumption_ids = [
        variant.assumption_id
        for scenario in scenarios
        for variant in scenario.assumption_variants
        if variant.assumption_id in planned_assumptions
    ]

    out_dir.mkdir(parents=True)
    try:
        paths = {name: out_dir / filename for name, filename in ROW_FILES.items()}
        counts: dict[str, dict[str, int]] = {name: {} for name in ROW_FILES}
        for name, filename in ROW_FILES.items():
            rows: list[str] = []
            for run, _, taken in resolved:
                chunk = _rows(run / filename, set(taken))
                counts[name][run.name] = len(chunk)
                rows.extend(chunk)
            paths[name].write_text("".join(rows), encoding="utf-8")

        protocol = first["protocol"]
        attribution = protocol["attribution"]
        attribution_modes = (
            ("anonymous", "assigned") if attribution == "both" else (attribution,)
        )
        try:
            integrity = run_pilot._validate_run_outputs(
                paths,
                scenarios,
                assumption_ids,
                n_seeds=protocol["n_seeds"],
                topologies=list(protocol["topologies"]),
                n_rounds=protocol["n_rounds"],
                baseline_reps=protocol["baseline_reps"],
                skip_baselines=protocol["skip_baselines"],
                skip_injection=protocol["skip_injection"],
                independent_reps=protocol["independent_reps"],
                skip_independent=protocol["skip_independent"],
                injection_conditions=tuple(protocol["injection_conditions"]),
                attribution_modes=attribution_modes,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise CombineError(f"joined rows fail the completion checks: {exc}") from exc

        created = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest = copy.deepcopy(first)
        manifest["created_at_utc"] = created
        manifest["runner"] = "scripts/combine_runs.py"
        manifest["command_argv"] = argv
        manifest["scenario_data"]["selected_scenario_ids"] = scenario_ids
        manifest["scenario_data"]["selected_assumption_ids"] = assumption_ids
        rounds_by_assumption: dict[str, object] = {}
        for _, source, _ in resolved:
            rounds_by_assumption.update(
                source["protocol"].get("correction_available_round_by_assumption") or {}
            )
        manifest["protocol"]["correction_available_round_by_assumption"] = {
            assumption_id: rounds_by_assumption[assumption_id]
            for assumption_id in assumption_ids
            if assumption_id in rounds_by_assumption
        }

        n_attr = len(attribution_modes)
        n_topologies = len(protocol["topologies"])
        planned_injection = (
            len(assumption_ids) * protocol["n_seeds"] * n_topologies * n_attr
            * len(protocol["injection_conditions"])
        )
        planned_baseline = (
            0 if protocol["skip_baselines"]
            else len(assumption_ids) * protocol["baseline_reps"] * n_topologies * n_attr
        )
        planned_independent = (
            0 if protocol["skip_independent"]
            else len(assumption_ids) * protocol["independent_reps"]
        )
        observed_files = {
            name: {
                "path": path.name,
                "rows": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        }
        manifest["status"] = {
            "state": "complete",
            "completed_at_utc": created,
            "planned_seed_generations": len(assumption_ids) * protocol["n_seeds"],
            "planned_conversation_attempts": {
                "injection_max_if_all_seeds_replayable": planned_injection,
                "no_injection_baseline": planned_baseline,
                "independent_control": planned_independent,
                "total_max_if_all_seeds_replayable": (
                    planned_injection + planned_baseline + planned_independent
                ),
            },
            "observed_rows": {
                "seeds": observed_files["seeds"]["rows"],
                "agent_round_records": observed_files["records"]["rows"],
                "conversation_results": observed_files["results"]["rows"],
            },
            "observed_files": observed_files,
            **integrity,
        }
        source_rows = [
            {
                "run": run.name,
                "scenarios": sorted(taken, key=order.__getitem__),
                "source_state": source["status"].get("state"),
                "created_at_utc": source.get("created_at_utc"),
                "git_commit": (source.get("git") or {}).get("commit"),
                "git_dirty": (source.get("git") or {}).get("dirty"),
                "command_argv": source.get("command_argv"),
                "rows": {name: counts[name][run.name] for name in ROW_FILES},
            }
            for run, source, taken in resolved
        ]
        manifest["combine"] = {
            "kind": "combine_runs_v1",
            "reason": reason,
            "matched": [
                "scenario corpus", *MATCHED_FIELDS, "protocol",
                "protocol source file hashes",
            ],
            "sources": source_rows,
        }
        (out_dir / run_pilot.RUN_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        shutil.copy2(first_run / first["scenario_data"]["snapshot_path"],
                     out_dir / first["scenario_data"]["snapshot_path"])
        for extra in ("git_tracked_changes.patch",):
            if (first_run / extra).exists():
                shutil.copy2(first_run / extra, out_dir / extra)
        if (first_run / "protocol_snapshot").is_dir():
            shutil.copytree(first_run / "protocol_snapshot", out_dir / "protocol_snapshot")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--parallel", nargs="+", type=Path, default=[],
                        help="run_parallel.py group folders; each contributes its scenarios")
    parser.add_argument("--source", nargs="+", default=[],
                        help="RUN (a completed run) or RUN:scenario_a,scenario_b")
    parser.add_argument("--reason", default="")
    parser.add_argument("--allow-partial", action="store_true",
                        help="join fewer scenarios than the corpus holds")
    args = parser.parse_args()
    try:
        sources = [parse_source(spec) for spec in args.source]
        for group in args.parallel:
            sources.extend(sources_from_parallel(group))
        manifest = combine(
            args.out_dir, sources, reason=args.reason, argv=sys.argv,
            require_all=not args.allow_partial,
        )
    except (CombineError, OSError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(f"combine failed: {type(exc).__name__}: {exc}")
    status = manifest["status"]
    print(f"combined -> {args.out_dir}")
    print(f"  {len(manifest['scenario_data']['selected_scenario_ids'])} scenarios from "
          f"{len(manifest['combine']['sources'])} runs")
    print(f"  seeds {status['observed_rows']['seeds']}  "
          f"conversations {status['observed_rows']['conversation_results']}  "
          f"records {status['observed_rows']['agent_round_records']}")


if __name__ == "__main__":
    main()
