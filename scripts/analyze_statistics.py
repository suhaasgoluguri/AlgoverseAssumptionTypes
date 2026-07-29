"""Inferential statistics for the confirmatory run.

Every number the project has produced so far is a raw count. Counts cannot
support a published claim here for two reasons:

1. CLUSTERING. Agent-rounds are not independent draws. Four agents share a
   conversation, three topologies share a frozen seed, and every conversation
   for one scenario shares its wording. Treating 36 agent-rounds as 36
   independent observations inflates significance badly. Scenario is the unit
   that generalizes, so it is the cluster and the bootstrap resampling unit.

2. NO INTERVAL. A point estimate with no CI cannot be compared across
   conditions, and the run-to-run drift documented in this project means
   overlapping intervals are the norm rather than the exception.

PRIMARY OUTCOME (preregister before running):
  Non-seed trap adoption at the final round, injection vs matched
  no-injection baseline, with scenario as the cluster.

The seed is excluded from the outcome: seed adoption is the treatment being
delivered, not propagation. A conversation whose seed never adopted the
assumption cannot exhibit propagation, so the seed-uptake-conditioned
estimate is reported alongside the intention-to-treat estimate; they answer
different questions and both belong in the paper.

Usage:
  uv run python scripts/analyze_statistics.py outputs/raw/<stamp>
  uv run python scripts/analyze_statistics.py <run-dir> --by-topology --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from assumption_propagation.analysis import load_agent_records
from assumption_propagation.reproducibility import resolve_scenario_data, sha256_file

N_BOOTSTRAP = 10_000
RNG_SEED = 20260826  # fixed so the reported intervals are reproducible

# A cluster bootstrap resamples clusters. With one cluster every resample is
# identical, so the interval collapses to a point and reports certainty it has
# not earned. Below this many scenarios there is no inference to be had, and
# the script refuses rather than emitting a publishable-looking number.
MIN_CLUSTERS = 3
ADVISORY_CLUSTERS = 8


def load_keys(data_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """(correct answer by scenario, favored option by assumption_id)."""
    entries = json.loads(data_path.read_text(encoding="utf-8"))
    correct, favored = {}, {}
    for entry in entries:
        if not isinstance(entry, dict) or "scenario_id" not in entry:
            continue
        correct[entry["scenario_id"]] = entry["correct_answer"]
        for variant in entry.get("assumption_variants", []):
            favored[variant["assumption_id"]] = variant["favored_option"]
    return correct, favored


def build_frame(run_dir: Path, data_path: Path) -> pd.DataFrame:
    """One row per non-seed agent at the final round of each conversation."""
    records = load_agent_records(run_dir / "records.jsonl")
    correct, favored = load_keys(data_path)

    # Seed round-0 answer per conversation drives the uptake conditioning.
    seed_uptake: dict[str, bool] = {}
    for r in records:
        if r.round == 0 and r.seed_agent_id is not None and r.agent_id == r.seed_agent_id:
            fav = favored.get(r.assumption_id)
            seed_uptake[r.run_id] = bool(fav and r.answer == fav)

    final_round = {}
    for r in records:
        final_round[r.run_id] = max(final_round.get(r.run_id, 0), r.round)

    rows = []
    for r in records:
        if r.round != final_round[r.run_id]:
            continue
        if r.condition.value not in ("injection", "no_injection_baseline"):
            continue
        if r.seed_agent_id is not None and r.agent_id == r.seed_agent_id:
            continue  # seed delivers the treatment; it is not the outcome
        fav = favored.get(r.assumption_id)
        if fav is None or r.answer is None:
            continue
        rows.append({
            "run_id": r.run_id,
            "scenario_id": r.scenario_id,
            "assumption_id": r.assumption_id,
            "semantic_type": (r.assumption_id or "").split("_")[-2]
                             if r.assumption_id else None,
            "topology": r.topology.value,
            "condition": r.condition.value,
            "injection": int(r.condition.value == "injection"),
            "agent_id": r.agent_id,
            "trap": int(r.answer == fav),
            "correct": int(r.answer == correct.get(r.scenario_id)),
            "seed_took_trap": seed_uptake.get(r.run_id, False),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap_diff(
    frame: pd.DataFrame, outcome: str = "trap", cluster: str = "scenario_id"
) -> dict:
    """Injection-minus-baseline rate difference with a cluster bootstrap CI.

    Scenarios are resampled with replacement; all rows for a drawn scenario
    travel together, which is what keeps the interval honest under the
    within-scenario correlation this design creates by construction.
    """
    inj = frame[frame.injection == 1]
    base = frame[frame.injection == 0]
    if inj.empty or base.empty:
        return {"error": "need both injection and baseline rows"}

    observed = inj[outcome].mean() - base[outcome].mean()
    clusters = frame[cluster].unique()
    if len(clusters) < MIN_CLUSTERS:
        return {
            "error": (
                f"only {len(clusters)} scenario cluster(s); a cluster bootstrap needs "
                f"at least {MIN_CLUSTERS}. The point estimate below is descriptive "
                f"only and no interval or p-value can be computed from this run."
            ),
            "injection_rate": float(inj[outcome].mean()),
            "injection_n": int(len(inj)),
            "baseline_rate": float(base[outcome].mean()),
            "baseline_n": int(len(base)),
            "difference": float(observed),
            "n_clusters": int(len(clusters)),
        }
    rng = np.random.default_rng(RNG_SEED)
    by_cluster = {c: frame[frame[cluster] == c] for c in clusters}

    diffs = []
    for _ in range(N_BOOTSTRAP):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([by_cluster[c] for c in drawn], ignore_index=True)
        si, sb = sample[sample.injection == 1], sample[sample.injection == 0]
        if si.empty or sb.empty:
            continue
        diffs.append(si[outcome].mean() - sb[outcome].mean())

    diffs = np.array(diffs)
    return {
        "injection_rate": float(inj[outcome].mean()),
        "injection_n": int(len(inj)),
        "baseline_rate": float(base[outcome].mean()),
        "baseline_n": int(len(base)),
        "difference": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_clusters": int(len(clusters)),
        "bootstrap_draws": int(len(diffs)),
    }


def gee_logistic(frame: pd.DataFrame) -> dict:
    """Population-averaged logistic model, working correlation within scenario.

    GEE rather than a mixed model: with ~10 scenario clusters the random-effect
    variance is poorly identified, whereas GEE with a robust sandwich estimator
    targets exactly the marginal contrast the primary outcome is stated in.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
    except ImportError:  # pragma: no cover
        return {"error": "statsmodels not available"}

    if frame.injection.nunique() < 2:
        return {"error": "need both conditions"}
    groups = pd.Categorical(frame.scenario_id).codes
    if len(set(groups)) < MIN_CLUSTERS:
        return {"error": f"only {len(set(groups))} cluster(s); GEE not estimable"}
    # Complete separation (a condition with a 0% or 100% rate) drives the
    # coefficient to +/-infinity and yields a meaningless odds ratio.
    rates = frame.groupby("injection")["trap"].mean()
    if (rates <= 0).any() or (rates >= 1).any():
        return {
            "error": "complete separation (a condition rate is exactly 0 or 1); "
                     "odds ratio not estimable — report the rate difference instead"
        }
    design = sm.add_constant(frame[["injection"]].astype(float))
    try:
        model = sm.GEE(
            frame["trap"].astype(float), design, groups=groups,
            family=Binomial(), cov_struct=Exchangeable(),
        ).fit()
    except Exception as exc:  # numerical failure is reportable, not fatal
        return {"error": f"GEE failed: {exc}"}

    coef = float(model.params["injection"])
    se = float(model.bse["injection"])
    return {
        "log_odds": coef,
        "odds_ratio": float(np.exp(coef)),
        "or_ci_low": float(np.exp(coef - 1.96 * se)),
        "or_ci_high": float(np.exp(coef + 1.96 * se)),
        "p_value": float(model.pvalues["injection"]),
        "n_obs": int(len(frame)),
        "n_clusters": int(len(set(groups))),
    }


def report_block(title: str, frame: pd.DataFrame) -> dict:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    if frame.empty:
        print("  no rows")
        return {}
    boot = cluster_bootstrap_diff(frame)
    if "error" in boot:
        if "injection_rate" in boot:
            print(f"  injection  {boot['injection_rate']:.3f}  (n={boot['injection_n']})")
            print(f"  baseline   {boot['baseline_rate']:.3f}  (n={boot['baseline_n']})")
            print(f"  difference {boot['difference']:+.3f}  (descriptive only)")
        print(f"  !! {boot['error']}")
        return boot
    print(f"  injection  {boot['injection_rate']:.3f}  (n={boot['injection_n']})")
    print(f"  baseline   {boot['baseline_rate']:.3f}  (n={boot['baseline_n']})")
    print(f"  difference {boot['difference']:+.3f}  "
          f"95% CI [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]  "
          f"({boot['n_clusters']} scenario clusters)")
    crosses = boot["ci_low"] <= 0 <= boot["ci_high"]
    print(f"  -> {'NOT significant (CI includes 0)' if crosses else 'significant at 95%'}")

    model = gee_logistic(frame)
    if "error" in model:
        print(f"  GEE: {model['error']}")
    else:
        print(f"  GEE odds ratio {model['odds_ratio']:.2f} "
              f"[{model['or_ci_low']:.2f}, {model['or_ci_high']:.2f}], "
              f"p = {model['p_value']:.4f}")
    return {"bootstrap": boot, "gee": model}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--by-topology", action="store_true")
    parser.add_argument("--by-semantic-type", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not (args.run_dir / "records.jsonl").exists():
        sys.exit(f"no records.jsonl in {args.run_dir}")
    try:
        data_path, source = resolve_scenario_data(args.run_dir, args.data)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    frame = build_frame(args.run_dir, data_path)
    if frame.empty:
        sys.exit("no usable final-round non-seed rows (need injection AND baseline)")

    print(f"run={args.run_dir}")
    print(f"scenario_data={data_path} ({source}, sha256={sha256_file(data_path)[:16]})")
    print(f"rows={len(frame)}  scenarios={frame.scenario_id.nunique()}  "
          f"conversations={frame.run_id.nunique()}")
    print("outcome: non-seed trap adoption at final round; cluster = scenario")
    n_clusters = frame.scenario_id.nunique()
    if n_clusters < MIN_CLUSTERS:
        print(f"\n!! {n_clusters} scenario(s): this run CANNOT support an inferential "
              f"claim. Descriptive rates only.")
    elif n_clusters < ADVISORY_CLUSTERS:
        print(f"\n!  {n_clusters} scenarios: intervals will be wide and the GEE "
              f"sandwich estimator is anti-conservative below ~{ADVISORY_CLUSTERS} "
              f"clusters. Treat as provisional.")

    out: dict = {
        "run_dir": str(args.run_dir),
        "scenario_data_sha256": sha256_file(data_path),
        "n_rows": int(len(frame)),
        "n_scenarios": int(frame.scenario_id.nunique()),
    }

    out["primary_itt"] = report_block(
        "PRIMARY (intention-to-treat: all injection conversations)", frame
    )

    conditioned = pd.concat([
        frame[(frame.injection == 1) & (frame.seed_took_trap)],
        frame[frame.injection == 0],
    ], ignore_index=True)
    out["primary_conditioned"] = report_block(
        "PRIMARY (conditioned on seed uptake: seed actually adopted the assumption)",
        conditioned,
    )

    if args.by_topology:
        out["by_topology"] = {}
        for topology in sorted(frame.topology.unique()):
            out["by_topology"][topology] = report_block(
                f"SECONDARY by topology: {topology}",
                frame[frame.topology == topology],
            )

    if args.by_semantic_type:
        out["by_semantic_type"] = {}
        for semantic in sorted(t for t in frame.semantic_type.unique() if t):
            out["by_semantic_type"][semantic] = report_block(
                f"SECONDARY by semantic type: {semantic}",
                frame[frame.semantic_type == semantic],
            )

    if args.by_topology or args.by_semantic_type:
        print("\nNOTE: secondary contrasts are exploratory and unadjusted for "
              "multiplicity. Report them as such.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
