"""Role x round x scenario propagation breakdown for one run directory.

analyze_run.py summarizes conversations and final votes. This script answers a
different, diagnostic question: WHO holds the assumption-favored option, and
WHEN. That is the calibration signal — seed uptake, neutral temptability, and
corrector capitulation are three different numbers, and a run can look healthy
on final accuracy while having no propagation to measure at all.

Corrective evidence is scheduled at ``corrective_evidence.available_round``,
and the verifier obtains it through the record tool. Rounds are marked
accordingly, including explicit no-correction arms, so the report never
confuses a scenario schedule with a delivery that actually occurred.

Usage:
  uv run python scripts/propagation_report.py outputs/raw/<stamp>
  uv run python scripts/propagation_report.py <run-dir> \\
      --data data/scenarios_hidden_profile_candidates.json
  uv run python scripts/propagation_report.py <run-dir> --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from assumption_propagation.aggregation import N_AGENTS
from assumption_propagation.analysis import load_agent_records
from assumption_propagation.schemas import INJECTION_CONDITIONS, AgentRoundRecord
from assumption_propagation.reproducibility import resolve_scenario_data, sha256_file

ROLES = ("seed", "corrector", "neutral")


def verify_answer_key(run_dir: Path, correct, favored) -> int:
    """Cross-check the supplied answer key against the run's own results.

    results.jsonl stores final_answer_correct and favored_vote_share as scored
    at run time with the scenario file the run actually used. Legacy runs have
    no scenario snapshot, so an explicit --data path can silently disagree with
    that file — different answer keys across scenario versions turn every trap
    count into a zero rather than an error. Recomputing those two fields from
    the supplied key and diffing them makes the disagreement loud.

    Returns the number of conversations checked; raises on any mismatch.
    """
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return 0
    checked = 0
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scenario_id = row["scenario_id"]
        if scenario_id not in correct:
            continue
        winner = row.get("plurality_answer")
        expected_correct = (winner == correct[scenario_id]) if winner else None
        if row.get("final_answer_correct") != expected_correct:
            raise ValueError(
                f"answer-key mismatch on {row['run_id']} ({scenario_id}): the run scored "
                f"final_answer_correct={row.get('final_answer_correct')}, this scenario "
                f"file implies {expected_correct} (correct_answer={correct[scenario_id]!r}, "
                f"plurality={winner!r}). The --data file is not the one this run used."
            )
        assumption_id = row.get("assumption_id")
        recorded_share = row.get("favored_vote_share")
        if assumption_id in favored and recorded_share is not None:
            expected_share = row["final_votes"].get(favored[assumption_id], 0) / N_AGENTS
            if abs(expected_share - recorded_share) > 1e-9:
                raise ValueError(
                    f"favored-option mismatch on {row['run_id']} ({assumption_id}): the run "
                    f"scored favored_vote_share={recorded_share}, this scenario file implies "
                    f"{expected_share} (favored_option={favored[assumption_id]!r}). "
                    "The --data file is not the one this run used."
                )
        checked += 1
    return checked


def role_of(record) -> str:
    """Resolve an agent's experimental role from the record's own metadata."""
    if record.seed_agent_id is not None and record.agent_id == record.seed_agent_id:
        return "seed"
    if record.corrector_agent_id is not None and record.agent_id == record.corrector_agent_id:
        return "corrector"
    return "neutral"


def load_keys(data_path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """Return (correct answer by scenario, favored option and corrective
    available_round by assumption_id) from the frozen scenario data."""
    entries = json.loads(data_path.read_text(encoding="utf-8"))
    correct: dict[str, str] = {}
    favored: dict[str, str] = {}
    available: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "scenario_id" not in entry:
            continue
        correct[entry["scenario_id"]] = entry["correct_answer"]
        for variant in entry.get("assumption_variants", []):
            favored[variant["assumption_id"]] = variant["favored_option"]
            corrective = variant.get("corrective_evidence") or {}
            if "available_round" in corrective:
                available[variant["assumption_id"]] = corrective["available_round"]
    return correct, favored, available


def tabulate(records, correct, favored) -> dict:
    """Count trap and correct answers per (condition, scenario, round, role)."""
    counts: dict = defaultdict(lambda: {"trap": 0, "correct": 0, "answered": 0})
    for record in records:
        if record.answer is None:
            continue
        key = (record.condition.value, record.scenario_id, record.round, role_of(record))
        cell = counts[key]
        cell["answered"] += 1
        if record.answer == correct.get(record.scenario_id):
            cell["correct"] += 1
        if record.assumption_id and record.answer == favored.get(record.assumption_id):
            cell["trap"] += 1
    return counts


def corrective_withheld(run_dir: Path) -> bool:
    """Whether this run was executed with the corrective-evidence ablation.

    available_round lives in the scenario file and describes the *schedule*;
    --withhold-corrective overrides it at run time, so the schedule alone
    would mislabel an ablation run as having live corrective evidence.
    """
    manifest = run_dir / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data.get("protocol", {}).get("withhold_corrective"))


def print_report(counts, available, conditions, scenarios, max_round, withheld=False) -> None:
    for condition in conditions:
        print(f"\n{'=' * 78}\n{condition}\n{'=' * 78}")
        for scenario in scenarios:
            rows = [
                (rnd, role, counts[(condition, scenario, rnd, role)])
                for rnd in range(max_round + 1)
                for role in ROLES
                if counts[(condition, scenario, rnd, role)]["answered"]
            ]
            if not rows:
                continue
            print(f"\n  {scenario}")
            for rnd in range(max_round + 1):
                cells = [
                    (role, cell) for r, role, cell in rows if r == rnd
                ]
                if not cells:
                    continue
                parts = [
                    f"{role[:4]} trap {cell['trap']:>2}/{cell['answered']:<2}"
                    f" correct {cell['correct']:>2}/{cell['answered']:<2}"
                    for role, cell in cells
                ]
                # Corrective evidence is per-assumption; report the window only
                # when every assumption in the run agrees, so the marker can
                # never imply a precision the data does not have.
                rounds_live = set(available.values())
                arm_withholds = condition in {
                    "injection_no_correction",
                    "injection_tool_none",
                }
                if withheld or arm_withholds:
                    marker = "  [corrective WITHHELD]"
                elif len(rounds_live) == 1:
                    live = rnd >= next(iter(rounds_live))
                    marker = (
                        "  [corrector record released]"
                        if live
                        else "  [no corrective evidence yet]"
                    )
                else:
                    marker = ""
                print(f"    r{rnd}: " + "  ".join(parts) + marker)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", type=Path, default=None,
                        help="scenario JSON override; default <run_dir>/scenario_snapshot.json "
                             "(legacy runs without a snapshot require this)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the tabulated counts to this path")
    args = parser.parse_args()

    records_path = args.run_dir / "records.jsonl"
    if not records_path.exists():
        sys.exit(f"no records.jsonl in {args.run_dir}")
    try:
        data_path, data_source = resolve_scenario_data(args.run_dir, args.data)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    records = load_agent_records(records_path)
    correct, favored, available = load_keys(data_path)

    unknown = sorted({r.scenario_id for r in records} - set(correct))
    if unknown:
        sys.exit(
            f"scenario data {data_path} has no answer key for {unknown}; "
            "pass the --data file this run actually used"
        )

    try:
        verified = verify_answer_key(args.run_dir, correct, favored)
    except ValueError as exc:
        sys.exit(str(exc))

    counts = tabulate(records, correct, favored)
    conditions = sorted({r.condition.value for r in records})
    scenarios = sorted({r.scenario_id for r in records})
    max_round = max(r.round for r in records)

    print(f"run={args.run_dir}")
    print(f"scenario_data={data_path} ({data_source}, sha256={sha256_file(data_path)[:16]})")
    print(f"records={len(records)}  scenarios={len(scenarios)}  rounds=0..{max_round}")
    print(
        f"answer key cross-checked against {verified} scored conversations"
        if verified
        else "answer key NOT cross-checked (no results.jsonl in this run)"
    )
    withheld = corrective_withheld(args.run_dir)
    if withheld:
        print("protocol: corrective evidence WITHHELD (ablation)")
    print_report(counts, available, conditions, scenarios, max_round, withheld=withheld)

    # Propagation must be measured as CONVERSION, not as presence.
    #
    # Counting every non-seed agent-round on the favored option after round 0
    # conflates two different things: an agent persuaded by the seed, and an
    # agent that independently chose that option at round 0 and never moved.
    # Round 0 precedes all messages, so the latter cannot be contagion. In this
    # project the naive count reported 15 "adoptions" in a run whose true
    # conversion count was zero — every one was a round-0 prior being carried
    # forward, at the same rate as in the no-injection baseline.
    #
    # An agent counts as converted only if it did NOT hold the favored option
    # at round 0 and did hold it at some later round.
    trajectories: dict[tuple[str, int], dict[int, Optional[str]]] = defaultdict(dict)
    context: dict[tuple[str, int], AgentRoundRecord] = {}
    for record in records:
        key = (record.run_id, record.agent_id)
        trajectories[key][record.round] = record.answer
        context[key] = record

    converted: dict[tuple[str, str, str], int] = defaultdict(int)
    held: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str, str], int] = defaultdict(int)
    for key, rounds in trajectories.items():
        record = context[key]
        if (
            record.condition.value not in INJECTION_CONDITIONS
            or role_of(record) != "neutral"
        ):
            continue
        fav = favored.get(record.assumption_id)
        if fav is None:
            continue
        cell = (
            record.condition.value,
            record.scenario_id,
            record.topology.value,
        )
        totals[cell] += 1
        if rounds.get(0) == fav:
            held[cell] += 1
        elif any(rounds.get(r) == fav for r in range(1, max_round + 1)):
            converted[cell] += 1

    print(
        f"\n{'=' * 78}\n"
        "ordinary-neutral CONVERSIONS after round 0 (all injection arms)\n"
        f"{'=' * 78}"
    )
    print("  converted = did not hold the trap at round 0, held it later")
    print("  held-r0   = independent round-0 prior, NOT propagation\n")
    total_conv = sum(converted.values())
    for cell in sorted(totals):
        condition, scenario, topology = cell
        print(
            f"  {condition:25s} {scenario:16s} {topology:22s} "
            f"converted {converted[cell]:>3d}/{totals[cell]:<3d}   "
            f"held-r0 {held[cell]:>3d}"
        )
    if not total_conv:
        print("\n  NO CONVERSIONS — the assumption never propagated beyond the seed.")
        print("  Compare the converted rate against the matched baseline before")
        print("  interpreting any non-zero value as propagation.")

    if args.json:
        payload = {
            "run_dir": str(args.run_dir),
            "scenario_data": str(data_path),
            "scenario_data_sha256": sha256_file(data_path),
            "corrective_available_round": available,
            "cells": [
                {"condition": c, "scenario_id": s, "round": r, "role": role, **cell}
                for (c, s, r, role), cell in sorted(counts.items(), key=lambda kv: tuple(map(str, kv[0])))
            ],
            "ordinary_neutral_post_round0_conversions": [
                {
                    "condition": condition,
                    "scenario_id": scenario,
                    "topology": topology,
                    "converted": converted[(condition, scenario, topology)],
                    "held_round0": held[(condition, scenario, topology)],
                    "total": total,
                }
                for (condition, scenario, topology), total in sorted(totals.items())
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
