"""Pilot runner: Stage A seed generation + Stage B matched propagation.

Usage (from the repo root):

  # Free dry run with the mock client (no API key needed):
  uv run python scripts/run_pilot.py --client mock --scenario travel_001

  # Live run against the Azure deployment in .env:
  uv run python scripts/run_pilot.py --client azure --scenario travel_001 --variants factual

  # Everything in the file, all variants, all topologies:
  uv run python scripts/run_pilot.py --client azure

Outputs (append-only JSONL, one directory per run timestamp):
  outputs/raw/<stamp>/scenario_snapshot.json exact input scenario-file bytes
  outputs/raw/<stamp>/run_manifest.json      frozen run/protocol provenance
  outputs/raw/<stamp>/seeds.jsonl     frozen seed instances
  outputs/raw/<stamp>/records.jsonl   every agent-round record
  outputs/raw/<stamp>/results.jsonl   per-conversation vote summaries
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from assumption_propagation.aggregation import summarize_final_round
from assumption_propagation.clients import AzureChatClient, MockClient
from assumption_propagation.orchestrator import (
    N_ROUNDS,
    generate_seed_instances,
    run_conversation,
    run_matched_triple,
)
from assumption_propagation.prompts import PROMPT_VERSION
from assumption_propagation.reproducibility import (
    RUN_MANIFEST_NAME,
    git_provenance,
    sha256_file,
    snapshot_scenario_data,
    snapshot_protocol_sources,
    snapshot_tracked_git_diff,
)
from assumption_propagation.schemas import (
    INJECTION_CONDITIONS,
    condition_label,
    AgentRoundRecord,
    Scenario,
    SeedInstance,
)

TOPOLOGIES = ("looped_chain", "neutral_star", "fully_connected_mesh")
PROTOCOL_SOURCE_PATHS = tuple(
    Path(path)
    for path in (
        "pyproject.toml",
        "uv.lock",
        "scripts/run_pilot.py",
        "src/assumption_propagation/prompts.py",
        "src/assumption_propagation/orchestrator.py",
        "src/assumption_propagation/topologies.py",
        "src/assumption_propagation/schemas.py",
        "src/assumption_propagation/clients.py",
        "src/assumption_propagation/aggregation.py",
        "src/assumption_propagation/reproducibility.py",
    )
)


def load_scenarios(path: Path, only: str | None) -> list[Scenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = []
    for entry in raw:
        if entry.get("legacy") is True:
            print(
                f"  ! skipping declared legacy scenario {entry.get('id', '?')}: "
                f"{entry.get('legacy_reason', 'not executable')}"
            )
            continue
        if "scenario_id" not in entry:
            raise ValueError(
                f"malformed scenario entry has no scenario_id: {entry.get('id', '?')}"
            )
        if only and entry["scenario_id"] != only:
            continue
        scenarios.append(Scenario.model_validate(entry))
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    assumption_ids = [
        variant.assumption_id
        for scenario in scenarios
        for variant in scenario.assumption_variants
    ]
    duplicate_scenarios = sorted(
        key for key, count in Counter(scenario_ids).items() if count > 1
    )
    duplicate_assumptions = sorted(
        key for key, count in Counter(assumption_ids).items() if count > 1
    )
    if duplicate_scenarios or duplicate_assumptions:
        raise ValueError(
            "selected scenario data contains duplicate identifiers: "
            f"scenario_id={duplicate_scenarios}, "
            f"assumption_id={duplicate_assumptions}"
        )
    return scenarios


def _load_jsonl_objects(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _validate_run_outputs(
    paths: dict[str, Path],
    scenarios: list[Scenario],
    selected_assumption_ids: list[str],
    *,
    n_seeds: int,
    topologies: list[str],
    n_rounds: int,
    baseline_reps: int,
    skip_baselines: bool,
    skip_injection: bool = False,
    independent_reps: int,
    skip_independent: bool,
) -> dict[str, object]:
    """Fail closed before a raw run is marked complete.

    The manifest hashes preserve exact bytes; this check additionally proves
    that those bytes contain the planned seed grid and exactly one result plus
    a complete four-agent round grid for every completed conversation.
    """
    seed_payloads = _load_jsonl_objects(paths["seeds"])
    record_payloads = _load_jsonl_objects(paths["records"])
    result_rows = _load_jsonl_objects(paths["results"])
    try:
        seeds = [SeedInstance.model_validate(row) for row in seed_payloads]
        records = [AgentRoundRecord.model_validate(row) for row in record_payloads]
    except ValidationError as exc:
        raise ValueError(f"raw output schema violation: {exc}") from exc

    assumption_to_scenario = {
        variant.assumption_id: scenario.scenario_id
        for scenario in scenarios
        for variant in scenario.assumption_variants
        if variant.assumption_id in selected_assumption_ids
    }
    expected_seed_coordinates = {
        (assumption_id, trial_index)
        for assumption_id in selected_assumption_ids
        for trial_index in range(n_seeds)
    }
    actual_seed_coordinates = [
        (seed.assumption_id, seed.seed_trial_index) for seed in seeds
    ]
    duplicate_seed_ids = sorted(
        key
        for key, count in Counter(seed.seed_instance_id for seed in seeds).items()
        if count > 1
    )
    duplicate_seed_coordinates = sorted(
        key for key, count in Counter(actual_seed_coordinates).items() if count > 1
    )
    if duplicate_seed_ids or duplicate_seed_coordinates:
        raise ValueError(
            "duplicate generated seed rows: "
            f"ids={duplicate_seed_ids}, coordinates={duplicate_seed_coordinates}"
        )
    if skip_injection:
        expected_seed_coordinates = set()
    if set(actual_seed_coordinates) != expected_seed_coordinates:
        raise ValueError(
            "generated seed grid differs from the plan: "
            f"missing={sorted(expected_seed_coordinates - set(actual_seed_coordinates))}, "
            f"extra={sorted(set(actual_seed_coordinates) - expected_seed_coordinates)}"
        )
    for seed in seeds:
        expected_scenario = assumption_to_scenario.get(seed.assumption_id)
        if expected_scenario is None or seed.scenario_id != expected_scenario:
            raise ValueError(
                f"seed {seed.seed_instance_id!r} has an unplanned "
                "scenario/assumption identity"
            )
        if seed.included_in_core != bool(seed.answer):
            raise ValueError(
                f"seed {seed.seed_instance_id!r} replayability metadata is inconsistent"
            )

    replayable_seed_ids = {
        seed.seed_instance_id for seed in seeds if seed.answer
    }
    seed_by_id = {seed.seed_instance_id: seed for seed in seeds}
    result_run_ids = [row.get("run_id") for row in result_rows]
    if any(not isinstance(run_id, str) or not run_id for run_id in result_run_ids):
        raise ValueError("every result row must have a nonblank string run_id")
    duplicate_result_runs = sorted(
        key for key, count in Counter(result_run_ids).items() if count > 1
    )
    if duplicate_result_runs:
        raise ValueError(f"duplicate result run_ids: {duplicate_result_runs}")

    # Keyed by condition: the injection and ablation arms each form their own
    # complete (seed x topology) grid when both are collected in one run.
    injection_coordinates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    baseline_coordinates: Counter = Counter()
    independent_coordinates: Counter = Counter()
    result_by_run: dict[str, dict] = {}
    for row in result_rows:
        run_id = row["run_id"]
        result_by_run[run_id] = row
        assumption_id = row.get("assumption_id")
        scenario_id = row.get("scenario_id")
        if assumption_to_scenario.get(assumption_id) != scenario_id:
            raise ValueError(f"result {run_id!r} has an unplanned scenario/assumption")
        condition = row.get("condition")
        topology = row.get("topology")
        seed_id = row.get("seed_instance_id")
        if condition in INJECTION_CONDITIONS:
            if seed_id not in replayable_seed_ids or topology not in topologies:
                raise ValueError(
                    f"injection result {run_id!r} has an invalid seed/topology"
                )
            frozen_seed = seed_by_id[seed_id]
            if (
                frozen_seed.scenario_id != scenario_id
                or frozen_seed.assumption_id != assumption_id
            ):
                raise ValueError(
                    f"injection result {run_id!r} does not match frozen seed "
                    f"{seed_id!r}'s scenario/assumption"
                )
            injection_coordinates[condition].append((seed_id, topology))
        elif condition == "no_injection_baseline":
            if seed_id is not None or topology not in topologies:
                raise ValueError(f"baseline result {run_id!r} has invalid metadata")
            baseline_coordinates[(assumption_id, topology)] += 1
        elif condition == "independent_control":
            if seed_id is not None or topology != "independent":
                raise ValueError(
                    f"independent result {run_id!r} has invalid metadata"
                )
            independent_coordinates[assumption_id] += 1
        else:
            raise ValueError(f"result {run_id!r} has unknown condition {condition!r}")

    expected_injection_coordinates = {
        (seed_id, topology)
        for seed_id in replayable_seed_ids
        for topology in topologies
    }
    if not injection_coordinates and not skip_injection:
        raise ValueError("no injection results found")
    if injection_coordinates and skip_injection:
        raise ValueError("injection results present despite --skip-injection")
    for arm, coords in sorted(injection_coordinates.items()):
        if len(coords) != len(set(coords)) or set(coords) != expected_injection_coordinates:
            raise ValueError(
                f"{arm} result grid differs from the replayable-seed plan"
            )
    expected_baseline_coordinates = Counter()
    if not skip_baselines:
        for assumption_id in selected_assumption_ids:
            for topology in topologies:
                expected_baseline_coordinates[(assumption_id, topology)] = baseline_reps
    if baseline_coordinates != expected_baseline_coordinates:
        raise ValueError("baseline result grid differs from the replicate plan")
    expected_independent_coordinates = Counter()
    if not skip_independent:
        for assumption_id in selected_assumption_ids:
            expected_independent_coordinates[assumption_id] = independent_reps
    if independent_coordinates != expected_independent_coordinates:
        raise ValueError("independent result grid differs from the replicate plan")

    records_by_run: dict[str, list[AgentRoundRecord]] = defaultdict(list)
    for record in records:
        records_by_run[record.run_id].append(record)
    if set(records_by_run) != set(result_by_run):
        raise ValueError(
            "record/result conversation IDs differ: "
            f"records_only={sorted(set(records_by_run) - set(result_by_run))}, "
            f"results_only={sorted(set(result_by_run) - set(records_by_run))}"
        )
    for run_id, conversation in records_by_run.items():
        result = result_by_run[run_id]
        expected_rounds = (
            1 if result["condition"] == "independent_control" else n_rounds
        )
        expected_grid = {
            (agent_id, round_index)
            for round_index in range(expected_rounds)
            for agent_id in range(4)
        }
        actual_grid = [(record.agent_id, record.round) for record in conversation]
        if len(actual_grid) != len(set(actual_grid)) or set(actual_grid) != expected_grid:
            raise ValueError(f"conversation {run_id!r} has an incomplete record grid")
        for record in conversation:
            if (
                record.condition.value != result["condition"]
                or record.topology.value != result["topology"]
                or record.scenario_id != result["scenario_id"]
                or record.assumption_id != result["assumption_id"]
                or record.seed_instance_id != result["seed_instance_id"]
            ):
                raise ValueError(
                    f"conversation {run_id!r} disagrees with its result metadata"
                )

    injection_count = len(expected_injection_coordinates) * len(injection_coordinates)
    baseline_count = sum(expected_baseline_coordinates.values())
    independent_count = sum(expected_independent_coordinates.values())
    return {
        "output_integrity_check_version": "1.0",
        "replayable_seed_generations": len(replayable_seed_ids),
        "unreplayable_seed_generations": len(seeds) - len(replayable_seed_ids),
        "reconciled_expected_rows": {
            "seeds": len(expected_seed_coordinates),
            "agent_round_records": (
                4 * n_rounds * (injection_count + baseline_count)
                + 4 * independent_count
            ),
            "conversation_results": (
                injection_count + baseline_count + independent_count
            ),
        },
        "observed_conversations": {
            "injection": injection_count,
            "no_injection_baseline": baseline_count,
            "independent_control": independent_count,
            "total": len(result_rows),
        },
    }


async def run_one_variant(scenario, variant, client, args, paths) -> None:
    label = f"{scenario.scenario_id}/{variant.semantic_type.value}"
    print(f"\n=== {label} ===")

    # Stage A: frozen seeds (every instance retained).
    #
    # The pooling-gap contrast is oracle-vs-baseline and carries no injection,
    # so seed generation is skipped entirely rather than generated and unused.
    if args.skip_injection:
        seeds = []
        print("  (skipping seed generation and the injection arm)")
    else:
        seeds = await generate_seed_instances(
            scenario, variant, client, n_instances=args.n_seeds, log_path=paths["seeds"]
        )
        for seed in seeds:
            state = "PARSE-FAILED" if not seed.answer else f"answer={seed.answer}"
            print(f"  seed {seed.seed_trial_index}: {state}")

    # Stage B: matched replay of every replayable seed.
    #
    # When the ablation arm is requested, BOTH arms run against the SAME frozen
    # seed inside this run. Round-0 behaviour varies enough between runs that a
    # between-run ablation contrast is not identifiable; replaying one seed into
    # both arms makes the comparison paired and within-run.
    arms = [("injection", False)]
    if args.scope_ladder:
        # All four correction-scope levels against the SAME frozen seed, so the
        # ordered contrast is paired and within-run.
        arms = [
            ("injection_no_correction", True),
            ("injection", False),
            ("injection_relay", False),
            ("injection_broadcast", False),
        ]
    elif args.ablation_arm:
        arms.append(("injection_no_correction", True))
    elif args.withhold_corrective:
        arms = [("injection_no_correction", True)]

    for seed in seeds:
        if not seed.answer:
            continue  # nothing to replay; the failure is already logged
        for arm_condition, arm_withhold in arms:
            triple = await run_matched_triple(
                scenario, variant, seed, client,
                topologies=tuple(args.topologies), n_rounds=args.rounds,
                condition=arm_condition,
                withhold_corrective=arm_withhold,
                log_path=paths["records"],
            )
            for topology_name, result in triple.items():
                summary = summarize_final_round(
                    result.final_answers(),
                    correct_answer=scenario.correct_answer,
                    favored_option=variant.favored_option,
                )
                row = {
                    "run_id": result.run_id,
                    "scenario_id": scenario.scenario_id,
                    "assumption_id": variant.assumption_id,
                    "seed_instance_id": seed.seed_instance_id,
                    "condition": arm_condition,
                    "topology": topology_name,
                    **summary,
                }
                with paths["results"].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                verdict = summary["plurality_answer"] or "TIE"
                flag = ("correct" if summary["final_answer_correct"]
                        else "no consensus" if summary["no_consensus"] else "WRONG")
                print(f"    [{condition_label(arm_condition):19s}] {seed.seed_instance_id}"
                      f" @ {topology_name:22s}"
                      f" votes={summary['final_votes']} -> {verdict} ({flag})")
                if args.verbose or args.show_rationales:
                    seed_id = result.records[0].seed_agent_id
                    corr_id = result.records[0].corrector_agent_id
                    def _tag(a):
                        return "*" if a == seed_id else "^" if a == corr_id else " "
                    for rnd in range(result.n_rounds):
                        cells = "  ".join(
                            f"A{a}={result.outputs[(a, rnd)].answer or '-'}{_tag(a)}"
                            for a in range(4)
                        )
                        print(f"        round {rnd}: {cells}")
                        if args.show_rationales:
                            for a in range(4):
                                r = (result.outputs[(a, rnd)].rationale or "")
                                r = r.strip().replace("\n", " ")
                                if len(r) > 140:
                                    r = r[:137] + "..."
                                print(f"            A{a}{_tag(a)}: {r}")
                    print("        (*=seed  ^=corrector)")

    # Independent no-communication control: the baseline round-0 information
    # state, with the neutral seed note and only correction scheduled for round
    # 0 (a delayed round-2 correction is not yet available). Agents answer once.
    if not args.skip_independent:
        for rep in range(args.independent_reps):
            result = await run_conversation(
                scenario, client=client, topology_name="independent",
                condition="independent_control", variant=variant,
                n_rounds=1, withhold_corrective=args.withhold_corrective,
                log_path=paths["records"],
            )
            summary = summarize_final_round(
                result.final_answers(),
                correct_answer=scenario.correct_answer,
                favored_option=variant.favored_option,
            )
            row = {
                "run_id": result.run_id,
                "scenario_id": scenario.scenario_id,
                "assumption_id": variant.assumption_id,
                "seed_instance_id": None,
                "condition": "independent_control",
                "topology": "independent",
                **summary,
            }
            with paths["results"].open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"    [{condition_label('independent_control'):19s}] rep {rep}:"
                  f" votes={summary['final_votes']}"
                  f" -> {summary['plurality_answer'] or 'TIE'}")

    # Topology-matched no-injection baselines.
    if not args.skip_baselines:
        for topology_name in args.topologies:
            for rep in range(args.baseline_reps):
                result = await run_conversation(
                    scenario, client=client, topology_name=topology_name,
                    condition="no_injection_baseline", variant=variant,
                    n_rounds=args.rounds,
                    withhold_corrective=args.withhold_corrective,
                    log_path=paths["records"],
                )
                summary = summarize_final_round(
                    result.final_answers(),
                    correct_answer=scenario.correct_answer,
                    favored_option=variant.favored_option,
                )
                row = {
                    "run_id": result.run_id,
                    "scenario_id": scenario.scenario_id,
                    "assumption_id": variant.assumption_id,
                    "seed_instance_id": None,
                    "condition": "no_injection_baseline",
                    "topology": topology_name,
                    **summary,
                }
                with paths["results"].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"    [{condition_label('no_injection_baseline'):19s}] rep {rep}"
                      f" @ {topology_name:22s}"
                      f" -> {summary['plurality_answer'] or 'TIE'}"
                      f" ({'correct' if summary['final_answer_correct'] else 'not correct'})")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--client", choices=("mock", "azure"), default="mock")
    parser.add_argument(
        "--data",
        default="data/scenarios_hidden_profile_candidates.json",
        help="scenario package. Defaults to the 33-scenario hidden-profile corpus "
             "(the one the no-pooling gate is enforced on). Superseded corpora live in "
             "data/archive/ and must be named explicitly.",
    )
    parser.add_argument("--scenario", default=None, help="single scenario_id; default all")
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=("factual", "causal", "framing"),
                        help="semantic types to run; default all present")
    parser.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES),
                        choices=TOPOLOGIES)
    parser.add_argument(
        "--rounds",
        type=int,
        default=N_ROUNDS,
        choices=range(1, N_ROUNDS + 1),
        help=f"rounds for communicating injection/baseline runs (1-{N_ROUNDS}); "
             "independent controls remain one round",
    )
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--baseline-reps", type=int, default=1)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument(
        "--skip-injection",
        action="store_true",
        help="collect no seeds and no injection arm. The pooling-gap contrast "
             "(oracle ceiling vs distributed group) needs only baselines, so this "
             "runs the primary outcome at a fraction of the cost.",
    )
    parser.add_argument(
        "--scope-ladder",
        action="store_true",
        help="run all four corrective-evidence scope levels against the same frozen "
             "seeds inside one run: none / one verifier / relayed with its source / "
             "broadcast to all. Supersedes --ablation-arm, which is the first two "
             "levels only.",
    )
    parser.add_argument(
        "--ablation-arm",
        action="store_true",
        help="run BOTH the injection and injection_no_correction arms against the "
             "same frozen seeds inside this run, giving a paired within-run "
             "ablation contrast. Preferred over --withhold-corrective.",
    )
    parser.add_argument(
        "--withhold-corrective",
        action="store_true",
        help="Track A ablation: the corrector keeps its role but never receives the "
             "decisive corrective evidence, so the group must resolve the injected "
             "claim unaided. Recorded in the run manifest.",
    )
    parser.add_argument("--independent-reps", type=int, default=1)
    parser.add_argument("--skip-independent", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print each agent's answer per round for injection conversations")
    parser.add_argument("--show-rationales", action="store_true",
                        help="also print each agent's rationale per round (implies --verbose)")
    parser.add_argument("--out", default="outputs/raw")
    parser.add_argument("--tag", default=None,
                        help="readable label appended to the run folder name, "
                             "e.g. --tag travel001_fix -> <stamp>_azure_travel001_fix")
    parser.add_argument("--temperature", default="0.2",
                        help="sampling temperature for the azure client, or 'none' to "
                             "omit the parameter (deployments that reject custom "
                             "temperatures). Ignored by the mock client.")
    args = parser.parse_args()
    if len(set(args.topologies)) != len(args.topologies):
        parser.error(f"--topologies values must be unique, got {args.topologies}")
    if args.n_seeds < 1:
        parser.error("--n-seeds must be at least 1")
    if args.baseline_reps < 0 or args.independent_reps < 0:
        parser.error("replicate counts must be non-negative")
    temperature = None if args.temperature.lower() == "none" else float(args.temperature)

    data_path = Path(args.data)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.client}"
    if args.tag:
        safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.tag).strip("_")
        if safe_tag:
            stamp = f"{stamp}_{safe_tag}"
    repo_root = Path(__file__).resolve().parents[1]
    # Capture Git state before creating the run directory, since outputs are
    # intentionally retained in this repository and would otherwise make an
    # initially clean worktree appear dirty.
    git_state = git_provenance(repo_root)
    out_dir = Path(args.out) / stamp
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        sys.exit(f"refusing to append a new run into existing directory: {out_dir}")
    paths = {name: out_dir / f"{name}.jsonl" for name in ("seeds", "records", "results")}
    for path in paths.values():
        path.touch(exist_ok=False)

    # Freeze scenario bytes before parsing them, then build every prompt from
    # the snapshot. This closes the audit gap where an actively edited source
    # file could otherwise change between validation and copying.
    try:
        snapshot_path, scenario_sha256 = snapshot_scenario_data(data_path, out_dir)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    try:
        scenarios = load_scenarios(snapshot_path, args.scenario)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        sys.exit(f"invalid scenario data {snapshot_path}: {exc}")
    if not scenarios:
        sys.exit(f"No scenarios matched (data={args.data}, scenario={args.scenario})")
    selected_variants = [
        variant
        for scenario in scenarios
        for variant in scenario.assumption_variants
        if not args.variants or variant.semantic_type.value in args.variants
    ]
    if not selected_variants:
        sys.exit("No assumption variants matched the requested semantic types")
    selected_assumption_ids = [
        variant.assumption_id for variant in selected_variants
    ]

    client = (MockClient() if args.client == "mock"
              else AzureChatClient(temperature=temperature))

    try:
        protocol_sources = snapshot_protocol_sources(
            repo_root, out_dir, PROTOCOL_SOURCE_PATHS
        )
        tracked_changes = snapshot_tracked_git_diff(repo_root, out_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    # Freeze all remaining run-defining provenance before the first request.
    planned_injection_conversations = (
        len(selected_variants) * args.n_seeds * len(args.topologies)
    )
    planned_baseline_conversations = (
        0
        if args.skip_baselines
        else len(selected_variants) * args.baseline_reps * len(args.topologies)
    )
    planned_independent_conversations = (
        0
        if args.skip_independent
        else len(selected_variants) * args.independent_reps
    )
    manifest = {
        "manifest_version": "1.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "scripts/run_pilot.py",
        "command_argv": sys.argv,
        "git": git_state,
        "git_tracked_changes": tracked_changes,
        "scenario_data": {
            "source_path": str(data_path.resolve()),
            "snapshot_path": snapshot_path.name,
            "sha256": scenario_sha256,
            "selected_scenario_ids": [scenario.scenario_id for scenario in scenarios],
            "selected_assumption_ids": selected_assumption_ids,
        },
        "prompt_version": PROMPT_VERSION,
        "protocol_sources": protocol_sources,
        "client": {
            "kind": args.client,
            "model_identifier": client.model_identifier,
            "deployment": getattr(client, "deployment", None),
            "temperature": getattr(client, "temperature", None),
            "max_tokens": getattr(client, "max_tokens", None),
        },
        "protocol": {
            "n_rounds": args.rounds,
            "withhold_corrective": args.withhold_corrective,
            "ablation_arm": args.ablation_arm,
            "scope_ladder": args.scope_ladder,
            "independent_control_n_rounds": 1,
            "topologies": list(args.topologies),
            "n_seeds": args.n_seeds,
            "baseline_reps": args.baseline_reps,
            "skip_baselines": args.skip_baselines,
            "skip_injection": args.skip_injection,
            "independent_reps": args.independent_reps,
            "skip_independent": args.skip_independent,
            "correction_available_round_by_assumption": {
                variant.assumption_id: variant.corrective_evidence.available_round
                for variant in selected_variants
            },
            "seed_private_dossier_persistent": True,
            "neutral_control_dossier_persistent": True,
            "corrective_dossier_persistent_from_available_round": True,
        },
        "status": {
            "state": "running",
            "completed_at_utc": None,
            "planned_seed_generations": len(selected_variants) * args.n_seeds,
            "planned_conversation_attempts": {
                "injection_max_if_all_seeds_replayable": planned_injection_conversations,
                "no_injection_baseline": planned_baseline_conversations,
                "independent_control": planned_independent_conversations,
                "total_max_if_all_seeds_replayable": (
                    planned_injection_conversations
                    + planned_baseline_conversations
                    + planned_independent_conversations
                ),
            },
        },
    }
    (out_dir / RUN_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"client={client.model_identifier}  scenarios={len(scenarios)}  out={out_dir}")

    for scenario in scenarios:
        for variant in scenario.assumption_variants:
            if args.variants and variant.semantic_type.value not in args.variants:
                continue
            await run_one_variant(scenario, variant, client, args, paths)

    try:
        integrity_summary = _validate_run_outputs(
            paths,
            scenarios,
            selected_assumption_ids,
            n_seeds=args.n_seeds,
            topologies=list(args.topologies),
            n_rounds=args.rounds,
            baseline_reps=args.baseline_reps,
            skip_baselines=args.skip_baselines,
            skip_injection=args.skip_injection,
            independent_reps=args.independent_reps,
            skip_independent=args.skip_independent,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        sys.exit(
            "run finished generating output but failed completion-integrity checks; "
            f"manifest remains 'running': {exc}"
        )

    def row_count(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def observed_file(path: Path) -> dict[str, object]:
        return {
            "path": path.name,
            "rows": row_count(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    observed_files = {
        name: observed_file(path) for name, path in paths.items()
    }
    manifest["status"].update({
        "state": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "observed_rows": {
            "seeds": observed_files["seeds"]["rows"],
            "agent_round_records": observed_files["records"]["rows"],
            "conversation_results": observed_files["results"]["rows"],
        },
        "observed_files": observed_files,
        **integrity_summary,
    })
    (out_dir / RUN_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nDone. Raw logs in {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
