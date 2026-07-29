"""Detect endpoint drift between runs by comparing round-0 behaviour.

Round 0 is the experiment's natural control surface: no messages have been
exchanged, no corrective evidence is visible to anyone, and neutral agents
have seen nothing but the shared task. Two runs of the same scenario file
must therefore produce statistically indistinguishable round-0 behaviour. If
they do not, the deployment changed between them, and any cross-run
comparison of downstream effects is confounded.

This matters because hosted deployments are versioned by the provider, not by
the experimenter: the same deployment name can serve a different model on a
different day, silently invalidating a control collected the previous week.

Two probes are reported:
  neutral round-0 trap rate  -- neutral agents see only the shared task, so
                                this is pure task-plus-model behaviour
  seed round-0 uptake rate   -- the seed additionally sees the injection, so
                                this tracks susceptibility to the treatment

Usage:
  uv run python scripts/check_drift.py outputs/raw/<run_a> outputs/raw/<run_b>
  uv run python scripts/check_drift.py <a> <b> <c> --data data/scenarios_v3.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from assumption_propagation.analysis import load_agent_records
from assumption_propagation.reproducibility import resolve_scenario_data, sha256_file

ALPHA = 0.05


def favored_by_assumption(data_path: Path) -> dict[str, str]:
    entries = json.loads(data_path.read_text(encoding="utf-8"))
    out = {}
    for entry in entries:
        if not isinstance(entry, dict) or "scenario_id" not in entry:
            continue
        for variant in entry.get("assumption_variants", []):
            out[variant["assumption_id"]] = variant["favored_option"]
    return out


def round0_rates(run_dir: Path, data_path: Path) -> dict:
    """Round-0 trap rates for neutral and seed agents in the injection arm."""
    records = load_agent_records(run_dir / "records.jsonl")
    favored = favored_by_assumption(data_path)

    neutral_hits = neutral_n = seed_hits = seed_n = 0
    scenarios = set()
    for r in records:
        if r.round != 0 or r.condition.value != "injection" or r.answer is None:
            continue
        fav = favored.get(r.assumption_id)
        if fav is None:
            continue
        scenarios.add(r.scenario_id)
        is_seed = r.seed_agent_id is not None and r.agent_id == r.seed_agent_id
        is_corrector = (r.corrector_agent_id is not None
                        and r.agent_id == r.corrector_agent_id)
        if is_seed:
            seed_n += 1
            seed_hits += int(r.answer == fav)
        elif not is_corrector:
            neutral_n += 1
            neutral_hits += int(r.answer == fav)

    return {
        "run": run_dir.name,
        "scenarios": sorted(scenarios),
        "scenario_data_sha256": sha256_file(data_path),
        "models": sorted({r.model for r in records}),
        "temperatures": sorted({str(r.temperature) for r in records}),
        "neutral_trap": (neutral_hits, neutral_n),
        "seed_trap": (seed_hits, seed_n),
    }


def joint_homogeneity(summaries: list[dict], key: str) -> dict:
    """Chi-square test that all runs share one underlying rate.

    This is the headline statistic, not the pairwise tests. Pairwise Fisher
    comparisons across k runs are k-choose-2 tests without multiplicity
    control, and reading a single significant pair as "the deployment changed
    on date X" over-reads the data: three runs can be jointly heterogeneous
    while no individual pair clears alpha, and two runs collected minutes
    apart can differ as much as two collected days apart.
    """
    from scipy.stats import chi2_contingency

    table = []
    for s in summaries:
        hits, n = s[key]
        if n == 0:
            return {"error": f"no {key} observations in {s['run']}"}
        table.append([hits, n - hits])
    if len(table) < 2:
        return {"error": "need at least two runs"}
    rates = [h / n for h, n in (s[key] for s in summaries)]
    # Every run at the same ceiling (or floor) leaves an all-zero column, for
    # which chi-square is undefined. That is a degenerate agreement, not
    # evidence of exchangeability: the measure simply has no resolution here.
    if all(c[0] == 0 for c in table) or all(c[1] == 0 for c in table):
        return {
            "degenerate": True,
            "rates": rates,
            "spread": float(max(rates) - min(rates)),
            "note": "all runs at the same ceiling/floor; test has no resolution",
            "heterogeneous": False,
        }
    chi2, p, dof, _ = chi2_contingency(table)
    return {
        "chi2": float(chi2), "dof": int(dof), "p_value": float(p),
        "heterogeneous": bool(p < ALPHA),
        "rates": rates,
        "spread": float(max(rates) - min(rates)),
    }


def compare(a: dict, b: dict, key: str) -> dict:
    from scipy.stats import fisher_exact

    ha, na = a[key]
    hb, nb = b[key]
    if min(na, nb) == 0:
        return {"error": f"no {key} observations in one run"}
    _, p = fisher_exact([[ha, na - ha], [hb, nb - hb]])
    return {
        "a_rate": ha / na, "a": f"{ha}/{na}",
        "b_rate": hb / nb, "b": f"{hb}/{nb}",
        "p_value": float(p),
        "drift_detected": bool(p < ALPHA),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=None,
                        help="scenario JSON override applied to every run "
                             "(legacy runs without a snapshot need this)")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        sys.exit("need at least two run directories to compare")

    summaries = []
    for run_dir in args.run_dirs:
        if not (run_dir / "records.jsonl").exists():
            sys.exit(f"no records.jsonl in {run_dir}")
        try:
            data_path, _ = resolve_scenario_data(run_dir, args.data)
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(str(exc))
        summaries.append(round0_rates(run_dir, data_path))

    print("Round-0 probes (injection arm; identical prompts across runs by construction)\n")
    for s in summaries:
        nh, nn = s["neutral_trap"]
        sh, sn = s["seed_trap"]
        print(f"  {s['run']}")
        print(f"    scenarios={','.join(s['scenarios'])}  model={','.join(s['models'])}"
              f"  temp={','.join(s['temperatures'])}")
        print(f"    data sha256={s['scenario_data_sha256'][:16]}")
        print(f"    neutral r0 trap {nh}/{nn}" + (f" = {nh/nn:.0%}" if nn else ""))
        print(f"    seed    r0 trap {sh}/{sn}" + (f" = {sh/sn:.0%}" if sn else ""))

    # Comparability preconditions: a differing scenario hash or model name is a
    # bookkeeping error, not drift, and must be ruled out before interpreting p.
    hashes = {s["scenario_data_sha256"] for s in summaries}
    scenario_sets = {tuple(s["scenarios"]) for s in summaries}
    if len(hashes) > 1:
        print("\n!! runs used DIFFERENT scenario files; round-0 is not comparable.")
    if len(scenario_sets) > 1:
        print("!! runs cover different scenarios; restrict to a shared subset first.")

    print(f"\n{'=' * 70}\nJOINT test: do all runs share one underlying rate?\n{'=' * 70}")
    joint = {}
    heterogeneous = False
    for key, name in (("neutral_trap", "neutral r0"), ("seed_trap", "seed r0")):
        res = joint_homogeneity(summaries, key)
        joint[key] = res
        if "error" in res:
            print(f"  {name}: {res['error']}")
            continue
        if res.get("degenerate"):
            rates = ", ".join(f"{r:.0%}" for r in res["rates"])
            print(f"  {name}: rates [{rates}]  -> {res['note']}")
            continue
        rates = ", ".join(f"{r:.0%}" for r in res["rates"])
        print(f"  {name}: rates [{rates}]  spread {res['spread']:.0%}  "
              f"chi2={res['chi2']:.2f} df={res['dof']} p={res['p_value']:.4f}"
              f"  -> {'HETEROGENEOUS' if res['heterogeneous'] else 'consistent'}")
        heterogeneous |= res["heterogeneous"]

    print(f"\n{'=' * 70}\npairwise comparisons (exploratory; unadjusted for multiplicity)\n{'=' * 70}")
    any_drift = False
    results = {}
    for a, b in itertools.combinations(summaries, 2):
        label = f"{a['run']}  vs  {b['run']}"
        print(f"\n{label}")
        for key, name in (("neutral_trap", "neutral r0"), ("seed_trap", "seed r0")):
            res = compare(a, b, key)
            results[f"{label}|{key}"] = res
            if "error" in res:
                print(f"  {name}: {res['error']}")
                continue
            verdict = "DRIFT" if res["drift_detected"] else "consistent"
            print(f"  {name}: {res['a']} ({res['a_rate']:.0%}) vs "
                  f"{res['b']} ({res['b_rate']:.0%})  p={res['p_value']:.2e}  -> {verdict}")
            any_drift |= res["drift_detected"]

    print(f"\n{'=' * 70}")
    if heterogeneous:
        print("VERDICT: these runs do NOT share one round-0 rate, at a point in the")
        print("protocol where the manipulation cannot act. Between-run variation")
        print("exceeds sampling error, so BETWEEN-run contrasts are unreliable.")
        print("Note this says nothing about the CAUSE: check whether the spread")
        print("tracks collection date before attributing it to deployment change.")
        print("Use within-run contrasts (--ablation-arm) instead.")
    elif any_drift:
        print("VERDICT: the joint test does not reject homogeneity, but at least one")
        print("pairwise comparison is nominally significant. With no multiplicity")
        print("control that is weak evidence; collect more replicates before")
        print("treating these runs as non-comparable.")
    else:
        print("VERDICT: round-0 behaviour is consistent across these runs.")
        print("Note that with small n this test has low power; consistency here is")
        print("not positive evidence that the runs are exchangeable.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"summaries": summaries, "joint": joint, "comparisons": results},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
