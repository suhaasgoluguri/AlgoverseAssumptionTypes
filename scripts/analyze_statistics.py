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
from assumption_propagation.schemas import (
    CORRECTION_SCOPE,
    INJECTION_CONDITIONS,
    condition_label,
)

N_BOOTSTRAP = 10_000
RNG_SEED = 20260826  # fixed so the reported intervals are reproducible

# Corrective-evidence scope, level 1 (nobody holds the record) to level 4
# (everyone holds it). Ordering is the experimental variable, so it lives in
# schemas.py and is imported rather than restated here.
SCOPE_LEVEL = {condition: i + 1 for i, condition in enumerate(CORRECTION_SCOPE)}
DEFER_OPTION = "E"  # same meaning in every scenario by construction

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
        if r.condition.value not in INJECTION_CONDITIONS + ("no_injection_baseline",):
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
            "scope_level": SCOPE_LEVEL.get(r.condition.value, 0),
            # `injection` is the two-arm contrast indicator. On a scope-ladder
            # run it would pool four different treatments into one arm, so the
            # ladder path filters to a single condition before using it.
            "injection": int(r.condition.value in INJECTION_CONDITIONS),
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


# ---------------------------------------------------------------------------
# Corrective-evidence scope ladder
#
# H1 conversion falls as scope widens, H2 deferral rises, H3 accuracy is NOT
# predicted to rise. The three can come apart, which is the point of the
# design, so each is computed separately and none is inferred from another.
# ---------------------------------------------------------------------------

def build_conversation_frame(run_dir: Path) -> pd.DataFrame:
    """One row per conversation, for the decision-level outcomes (H2, H3).

    Conversion is an agent-round outcome and comes from records.jsonl; deferral
    and accuracy are properties of the group's decision and come from
    results.jsonl, which already holds the plurality tally.
    """
    path = run_dir / "results.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        condition = r.get("condition")
        if condition not in INJECTION_CONDITIONS + ("no_injection_baseline",):
            continue
        plurality = r.get("plurality_answer")
        rows.append({
            "run_id": r.get("run_id"),
            "scenario_id": r.get("scenario_id"),
            "assumption_id": r.get("assumption_id"),
            "topology": r.get("topology"),
            "condition": condition,
            "scope_level": SCOPE_LEVEL.get(condition, 0),
            # A tie has no plurality answer. It is neither correct nor a
            # deferral — counting it as either would manufacture an effect.
            "tie": int(bool(r.get("no_consensus"))),
            "defer": int(plurality == DEFER_OPTION) if plurality else 0,
            "correct": int(bool(r.get("final_answer_correct"))) if plurality else 0,
        })
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    outcome: str,
    arm_a: str,
    arm_b: str,
    cluster: str = "scenario_id",
) -> dict:
    """rate(arm_a) - rate(arm_b), scenarios resampled with replacement.

    Paired: the two arms share scenarios by construction (all ladder levels run
    against the same frozen seeds inside one run), so a drawn scenario
    contributes its rows from BOTH arms. Resampling the arms independently
    would discard that pairing and widen the interval for no reason.
    """
    a = frame[frame.condition == arm_a]
    b = frame[frame.condition == arm_b]
    if a.empty or b.empty:
        return {"error": f"missing rows for {arm_a} or {arm_b}"}

    observed = a[outcome].mean() - b[outcome].mean()
    shared = sorted(set(a[cluster].unique()) & set(b[cluster].unique()))
    if len(shared) < MIN_CLUSTERS:
        return {
            "error": f"only {len(shared)} shared scenario cluster(s); "
                     f"need at least {MIN_CLUSTERS}",
            "rate_a": float(a[outcome].mean()),
            "rate_b": float(b[outcome].mean()),
            "difference": float(observed),
            "n_clusters": len(shared),
        }

    by_cluster = {
        c: (a[a[cluster] == c][outcome].to_numpy(),
            b[b[cluster] == c][outcome].to_numpy())
        for c in shared
    }
    rng = np.random.default_rng(RNG_SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        drawn = rng.choice(shared, size=len(shared), replace=True)
        va = np.concatenate([by_cluster[c][0] for c in drawn])
        vb = np.concatenate([by_cluster[c][1] for c in drawn])
        if va.size and vb.size:
            diffs.append(va.mean() - vb.mean())
    diffs = np.array(diffs)

    # Bootstrap two-sided p: the smaller tail mass on either side of zero,
    # doubled. Used only to feed the Holm correction across the three primary
    # tests; the CI remains the reported quantity.
    if diffs.size:
        tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
        p_value = float(min(1.0, 2 * tail))
    else:
        p_value = float("nan")

    return {
        "rate_a": float(a[outcome].mean()),
        "n_a": int(len(a)),
        "rate_b": float(b[outcome].mean()),
        "n_b": int(len(b)),
        "difference": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": p_value,
        "n_clusters": len(shared),
        "bootstrap_draws": int(diffs.size),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, preregistered across the 3 primaries."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (n - i) * p))
        adjusted[key] = running
    return adjusted


def scope_ladder_report(
    agent_frame: pd.DataFrame, conv_frame: pd.DataFrame
) -> dict:
    """Per-level rates plus the three preregistered primary contrasts."""
    out: dict = {}
    present = [c for c in CORRECTION_SCOPE if c in set(agent_frame.condition)]
    if len(present) < 2:
        return {"error": "fewer than two ladder levels present"}

    # --- descriptive ladder ------------------------------------------------
    print(f"\n{'=' * 74}\nCORRECTIVE-EVIDENCE SCOPE LADDER\n{'=' * 74}")
    baseline = agent_frame[agent_frame.condition == "no_injection_baseline"]
    base_rate = baseline["trap"].mean() if not baseline.empty else float("nan")
    print(f"  matched no-injection baseline conversion: {base_rate:.3f} "
          f"(n={len(baseline)})")
    print(f"\n  {'level':22s} {'conversion':>11s} {'vs base':>9s} "
          f"{'deferral':>9s} {'accuracy':>9s} {'ties':>6s}")
    levels: dict[str, dict] = {}
    for condition in present:
        a = agent_frame[agent_frame.condition == condition]
        c = conv_frame[conv_frame.condition == condition]
        row = {
            "scope_level": SCOPE_LEVEL[condition],
            "conversion": float(a["trap"].mean()),
            "conversion_vs_baseline": float(a["trap"].mean() - base_rate),
            "deferral": float(c["defer"].mean()) if not c.empty else None,
            "accuracy": float(c["correct"].mean()) if not c.empty else None,
            "tie_rate": float(c["tie"].mean()) if not c.empty else None,
            "n_agent_rows": int(len(a)),
            "n_conversations": int(len(c)),
        }
        levels[condition] = row
        print(f"  {condition_label(condition):22s} {row['conversion']:11.3f} "
              f"{row['conversion_vs_baseline']:+9.3f} "
              f"{(row['deferral'] if row['deferral'] is not None else float('nan')):9.3f} "
              f"{(row['accuracy'] if row['accuracy'] is not None else float('nan')):9.3f} "
              f"{(row['tie_rate'] if row['tie_rate'] is not None else float('nan')):6.3f}")
    out["levels"] = levels

    # --- monotonicity (descriptive, per preregistration) -------------------
    conv_seq = [levels[c]["conversion"] for c in present]
    defer_seq = [levels[c]["deferral"] for c in present
                 if levels[c]["deferral"] is not None]
    out["conversion_non_increasing"] = all(
        b <= a + 1e-12 for a, b in zip(conv_seq, conv_seq[1:]))
    out["deferral_non_decreasing"] = all(
        b >= a - 1e-12 for a, b in zip(defer_seq, defer_seq[1:]))
    print(f"\n  H1 ordering (conversion non-increasing): "
          f"{'holds' if out['conversion_non_increasing'] else 'VIOLATED'}")
    print(f"  H2 ordering (deferral non-decreasing):   "
          f"{'holds' if out['deferral_non_decreasing'] else 'VIOLATED'}")
    print("  (orderings are descriptive; the tests below are the preregistered ones)")

    # --- the three preregistered primary contrasts -------------------------
    lo, hi = CORRECTION_SCOPE[0], CORRECTION_SCOPE[-1]
    third = CORRECTION_SCOPE[2]
    tests = {
        "H1_conversion_L1_minus_L4": (agent_frame, "trap", lo, hi, "decrease"),
        "H2_deferral_L4_minus_L1": (conv_frame, "defer", hi, lo, "increase"),
        "H3_accuracy_L4_minus_L3": (conv_frame, "correct", hi, third, "no increase"),
    }
    results, p_values = {}, {}
    for name, (frame, outcome, arm_a, arm_b, direction) in tests.items():
        if frame.empty or arm_a not in set(frame.condition) or arm_b not in set(frame.condition):
            results[name] = {"error": f"arms not present for {name}"}
            continue
        res = paired_cluster_bootstrap(frame, outcome, arm_a, arm_b)
        res["predicted_direction"] = direction
        results[name] = res
        if "p_value" in res:
            p_values[name] = res["p_value"]

    adjusted = holm(p_values) if p_values else {}
    print(f"\n{'-' * 74}\n  PRIMARY CONTRASTS (Holm-corrected across the three)\n{'-' * 74}")
    for name, res in results.items():
        if "error" in res:
            print(f"  {name}: {res['error']}")
            continue
        adj = adjusted.get(name, float("nan"))
        res["p_holm"] = float(adj)
        crosses = res["ci_low"] <= 0 <= res["ci_high"]
        verdict = "NOT significant" if (crosses or adj > 0.05) else "significant at 95%"
        print(f"  {name}")
        print(f"    {res['rate_a']:.3f} - {res['rate_b']:.3f} = "
              f"{res['difference']:+.3f}  95% CI "
              f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]  "
              f"({res['n_clusters']} clusters)")
        print(f"    p={res['p_value']:.4f}  p_holm={adj:.4f}  "
              f"predicted: {res['predicted_direction']}  -> {verdict}")
    out["primary_contrasts"] = results

    # --- provenance: level 2 vs level 3, no direction predicted ------------
    if all(c in set(agent_frame.condition) for c in (CORRECTION_SCOPE[1], third)):
        prov = paired_cluster_bootstrap(
            agent_frame, "trap", CORRECTION_SCOPE[2], CORRECTION_SCOPE[1])
        out["provenance_L3_minus_L2"] = prov
        if "error" not in prov:
            print(f"\n  SECONDARY provenance (L3 - L2, exploratory, no direction "
                  f"predicted):\n    {prov['difference']:+.3f} 95% CI "
                  f"[{prov['ci_low']:+.3f}, {prov['ci_high']:+.3f}]")
            print("    NOTE: an interval containing zero is not equivalence. "
                  "Use TOST if you want to claim provenance does not matter.")
    return out


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

    # A scope-ladder run holds four different treatments. The two-arm path
    # below sets injection=1 for all of them, which would silently average the
    # ladder into a single meaningless arm. Refuse instead.
    arms = [c for c in CORRECTION_SCOPE if c in set(frame.condition)]
    out["ladder_arms_present"] = arms
    if len(arms) > 1:
        conv_frame = build_conversation_frame(args.run_dir)
        out["scope_ladder"] = scope_ladder_report(frame, conv_frame)
        print("\nNOTE: this run contains %d corrective-evidence scope arms (%s). "
              "The pooled two-arm contrast is NOT reported, because pooling the "
              "ladder would average four different treatments into one."
              % (len(arms), ", ".join(arms)))
        two_arm = frame[frame.condition.isin(("injection", "no_injection_baseline"))]
    else:
        two_arm = frame

    out["primary_itt"] = report_block(
        "PRIMARY (intention-to-treat: all injection conversations)"
        if len(arms) <= 1 else
        "REFERENCE two-arm contrast (level 2 'one verifier' vs baseline only)",
        two_arm,
    )

    conditioned = pd.concat([
        two_arm[(two_arm.injection == 1) & (two_arm.seed_took_trap)],
        two_arm[two_arm.injection == 0],
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
