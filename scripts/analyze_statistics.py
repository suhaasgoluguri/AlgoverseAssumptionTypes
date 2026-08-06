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

PRIMARY OUTCOME (frozen before the confirmatory run):
  Ordinary-neutral final accuracy for three paired adjacent delivery-route
  contrasts: testimony minus none, exact relay minus testimony, and direct
  recipient delivery minus exact relay, restricted to the fully connected mesh
  with assigned speaker attribution. Scenario is the cluster. Raw bootstrap
  p-values for these three tests receive Holm correction.

The seed and designated verifier are excluded from the primary population:
seed adoption is treatment delivery and verifier behavior is mechanically
affected by holding the corrective record. Group accuracy, trap outcomes,
seed-uptake-conditioned estimates, semantic-type splits, and mechanism-control
contrasts are secondary or exploratory.

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
    TOOL_CORRECTION_CONTROLS,
    TOOL_CORRECTION_SCOPE,
    condition_label,
)

N_BOOTSTRAP = 10_000
RNG_SEED = 20260826  # fixed so the reported intervals are reproducible

# Corrective-evidence scope, level 1 (nobody holds the record) to level 4
# (everyone holds it). Ordering is the experimental variable, so it lives in
# schemas.py and is imported rather than restated here.
SCOPE_LEVEL = {
    condition: i + 1
    for scope in (CORRECTION_SCOPE, TOOL_CORRECTION_SCOPE)
    for i, condition in enumerate(scope)
}
# Mechanism controls are deliberately not interpreted as higher doses.  The
# numeric value exists only for stable sorting in flat analysis frames.
SCOPE_LEVEL.update(
    {
        condition: len(TOOL_CORRECTION_SCOPE) + i + 1
        for i, condition in enumerate(TOOL_CORRECTION_CONTROLS)
    }
)
DEFER_OPTION = "E"  # same meaning in every scenario by construction

# A cluster bootstrap resamples clusters. With one cluster every resample is
# identical, so the interval collapses to a point and reports certainty it has
# not earned. Below this many scenarios there is no inference to be had, and
# the script refuses rather than emitting a publishable-looking number.
MIN_CLUSTERS = 3
ADVISORY_CLUSTERS = 8
PRIMARY_TOPOLOGY = "fully_connected_mesh"
PRIMARY_ATTRIBUTION = "assigned"
EXPOSURE_COLUMNS = (
    "heard_record_backed_corrector_by_final",
    "exact_relay_received_by_final",
    "direct_tool_record_received_by_final",
    "prompt_record_received_by_final",
    "correction_reached_by_final",
    "correction_unreached_by_final",
)


def load_keys(
    data_path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """Return answer and timing keys needed for agent-level endpoints.

    The correction checkpoint is variant-specific.  It is the round stored in
    ``corrective_evidence.available_round``: ordinary neutrals have not yet
    received peer-delivered evidence at that point, so their answer there is
    the appropriate pre-scope state for a recovery-to-final endpoint.
    """
    entries = json.loads(data_path.read_text(encoding="utf-8"))
    correct, favored, correction_available_round = {}, {}, {}
    for entry in entries:
        if not isinstance(entry, dict) or "scenario_id" not in entry:
            continue
        correct[entry["scenario_id"]] = entry["correct_answer"]
        for variant in entry.get("assumption_variants", []):
            assumption_id = variant["assumption_id"]
            favored[assumption_id] = variant["favored_option"]
            correction = variant.get("corrective_evidence") or {}
            available_round = correction.get("available_round")
            if isinstance(available_round, int):
                correction_available_round[assumption_id] = available_round
    return correct, favored, correction_available_round


def build_frame(run_dir: Path, data_path: Path) -> pd.DataFrame:
    """One row per non-seed agent at the final round of each conversation.

    Roles remain explicit. Scope-ladder estimates filter this frame to the two
    ordinary neutrals; the designated verifier is a manipulation carrier, not
    an independent recovery opportunity.
    """
    records = load_agent_records(run_dir / "records.jsonl")
    correct, favored, correction_available_round = load_keys(data_path)

    # Seed round-0 answer per conversation drives the uptake conditioning.
    seed_uptake: dict[str, bool] = {}
    for r in records:
        if r.round == 0 and r.seed_agent_id is not None and r.agent_id == r.seed_agent_id:
            fav = favored.get(r.assumption_id)
            seed_uptake[r.run_id] = bool(fav and r.answer == fav)

    final_round = {}
    answer_at: dict[tuple[str, int, int], str | None] = {}
    records_by_agent: dict[tuple[str, int], list] = {}
    for r in records:
        final_round[r.run_id] = max(final_round.get(r.run_id, 0), r.round)
        answer_at[(r.run_id, r.agent_id, r.round)] = r.answer
        records_by_agent.setdefault((r.run_id, r.agent_id), []).append(r)

    rows = []
    for r in records:
        if r.round != final_round[r.run_id]:
            continue
        if r.condition.value not in INJECTION_CONDITIONS + ("no_injection_baseline",):
            continue
        if r.seed_agent_id is not None and r.agent_id == r.seed_agent_id:
            continue  # seed delivers the treatment; it is not the outcome
        if r.parse_failed:
            continue  # carried-forward text is not a newly observed final answer
        fav = favored.get(r.assumption_id)
        if fav is None or r.answer is None:
            continue
        correct_answer = correct.get(r.scenario_id)
        if r.corrector_agent_id is not None and r.agent_id == r.corrector_agent_id:
            role = "corrector"
        elif r.seed_agent_id is None or r.corrector_agent_id is None:
            role = "unknown_non_seed"
        else:
            role = "ordinary_neutral"
        initial_answer = answer_at.get((r.run_id, r.agent_id, 0))
        pre_final_answer = answer_at.get(
            (r.run_id, r.agent_id, final_round[r.run_id] - 1)
        ) if final_round[r.run_id] > 0 else None
        pre_scope_round = correction_available_round.get(r.assumption_id)
        pre_scope_answer = (
            answer_at.get((r.run_id, r.agent_id, pre_scope_round))
            if pre_scope_round is not None
            and final_round[r.run_id] > pre_scope_round
            else None
        )
        agent_history = records_by_agent[(r.run_id, r.agent_id)]
        heard_record_backed_corrector = any(
            record.corrector_held_record_when_speaking is True
            for record in agent_history
        )
        exact_relay_received = any(
            record.corrective_record_in_peer_message is True
            for record in agent_history
        )
        direct_tool_received = any(
            getattr(record.corrective_record_delivery, "value", None)
            == "direct_tool_record"
            for record in agent_history
        )
        prompt_record_received = any(
            getattr(record.corrective_record_delivery, "value", None)
            in ("prompt_receipt_control", "direct_system_record")
            for record in agent_history
        )
        correction_reached = (
            heard_record_backed_corrector
            or exact_relay_received
            or direct_tool_received
            or prompt_record_received
        )
        rows.append({
            "run_id": r.run_id,
            "scenario_id": r.scenario_id,
            "assumption_id": r.assumption_id,
            "semantic_type": (r.assumption_id or "").split("_")[-2]
                             if r.assumption_id else None,
            "topology": r.topology.value,
            "attribution": r.attribution or "anonymous",
            "condition": r.condition.value,
            "scope_level": SCOPE_LEVEL.get(r.condition.value, 0),
            # `injection` is the two-arm contrast indicator. On a scope-ladder
            # run it would pool four different treatments into one arm, so the
            # ladder path filters to a single condition before using it.
            "injection": int(r.condition.value in INJECTION_CONDITIONS),
            "agent_id": r.agent_id,
            "role": role,
            "initial_answer": initial_answer,
            "pre_final_answer": pre_final_answer,
            "pre_scope_round": pre_scope_round,
            "pre_scope_answer": pre_scope_answer,
            "trap": int(r.answer == fav),
            "correct": int(r.answer == correct_answer),
            "heard_record_backed_corrector_by_final": int(
                heard_record_backed_corrector
            ),
            "exact_relay_received_by_final": int(exact_relay_received),
            "direct_tool_record_received_by_final": int(direct_tool_received),
            "prompt_record_received_by_final": int(prompt_record_received),
            "correction_reached_by_final": int(correction_reached),
            "correction_unreached_by_final": int(not correction_reached),
            # True conversion requires a valid non-trap starting answer. Final
            # trap share remains a separate descriptive endpoint.
            "trap_conversion": (
                int(r.answer == fav)
                if initial_answer is not None and initial_answer != fav
                else None
            ),
            # Compatibility endpoint: only detects a trap in the immediately
            # pre-final round.  It misses an earlier correction that is then
            # retained through the final round.
            "trap_to_correct_recovery": (
                int(r.answer == correct_answer)
                if pre_final_answer == fav
                else None
            ),
            # Primary scope-recovery endpoint. At ``available_round`` the
            # ordinary neutral has not yet had peer evidence access; count a
            # recovery if a trap held at that checkpoint is correct at final.
            "pre_scope_trap_to_final_correct_recovery": (
                int(r.answer == correct_answer)
                if pre_scope_answer == fav
                else None
            ),
            "correct_answer_retained": (
                int(r.answer == correct_answer)
                if pre_final_answer == correct_answer
                else None
            ),
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
# Corrective-evidence arm comparison. These are mechanistically different
# delivery modes, not assumed to be an ordinal dose. Agent-level outcomes use
# ordinary neutrals only; group outcomes retain the complete team decision.
# ---------------------------------------------------------------------------

def build_conversation_frame(run_dir: Path) -> pd.DataFrame:
    """One row per conversation for team-level decision outcomes.

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
            "semantic_type": (
                (r.get("assumption_id") or "").split("_")[-2]
                if r.get("assumption_id")
                else ""
            ),
            "topology": r.get("topology"),
            "attribution": r.get("attribution") or "anonymous",
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

    Paired: the two arms share scenarios by construction (all delivery arms run
    against the same frozen seeds inside one run), so a drawn scenario
    contributes its rows from BOTH arms. Resampling the arms independently
    would discard that pairing and widen the interval for no reason.
    """
    a = frame[(frame.condition == arm_a) & frame[outcome].notna()]
    b = frame[(frame.condition == arm_b) & frame[outcome].notna()]
    if a.empty or b.empty:
        return {
            "error": (
                f"no eligible {outcome} opportunities in {arm_a} or {arm_b}"
            )
        }

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
    agent_frame: pd.DataFrame,
    conv_frame: pd.DataFrame,
    *,
    scope_order: tuple[str, ...] = CORRECTION_SCOPE,
    control_conditions: tuple[str, ...] = (),
    apply_primary_holm: bool = True,
) -> dict:
    """Descriptive arm rates and paired adjacent contrasts.

    Agent-level scope outcomes include only ordinary neutrals. The designated
    verifier is the manipulation carrier and would mechanically improve every
    arm in which it receives the record. The arms also change different
    mechanisms, so adjacent comparisons are more defensible than one ordinal
    dose claim.
    """
    if scope_order not in (CORRECTION_SCOPE, TOOL_CORRECTION_SCOPE):
        raise ValueError(f"unknown corrective-evidence scope order: {scope_order!r}")
    out: dict = {
        "analysis_population": "ordinary_neutral",
        "scope_protocol": (
            "tool_v2" if scope_order == TOOL_CORRECTION_SCOPE else "prompt_v1"
        ),
    }
    if "role" not in agent_frame:
        return {**out, "error": "agent frame lacks role metadata"}
    ordinary = agent_frame[agent_frame.role == "ordinary_neutral"].copy()
    out["excluded_corrector_rows"] = int((agent_frame.role == "corrector").sum())
    out["excluded_unknown_role_rows"] = int(
        (agent_frame.role == "unknown_non_seed").sum()
    )
    if ordinary.empty:
        return {
            **out,
            "error": "no ordinary-neutral rows; role identifiers are required",
        }
    observed_conditions = set(ordinary.condition)
    primary_present = [c for c in scope_order if c in observed_conditions]
    controls_present = [c for c in control_conditions if c in observed_conditions]
    present = [*primary_present, *controls_present]
    if len(primary_present) < 2:
        return {**out, "error": "fewer than two delivery arms present"}

    # --- descriptive arm rates ---------------------------------------------
    print(f"\n{'=' * 74}\nCORRECTIVE-EVIDENCE SCOPE ARMS\n{'=' * 74}")
    print("  agent-level population: ordinary neutrals only (seed and verifier excluded)")
    baseline = ordinary[ordinary.condition == "no_injection_baseline"]
    baseline_conversion_rows = (
        baseline["trap_conversion"].dropna() if not baseline.empty else pd.Series(dtype=float)
    )
    base_conversion = (
        float(baseline_conversion_rows.mean())
        if not baseline_conversion_rows.empty
        else float("nan")
    )
    print(f"  matched uncorrupted-reference trap conversion: {base_conversion:.3f}")
    print(
        f"\n  {'arm':28s} {'final trap':>10s} {'new trap':>9s} "
        f"{'scope->final':>12s} {'neutral acc':>11s} {'group acc':>10s}"
    )
    levels: dict[str, dict] = {}
    for condition in present:
        a = ordinary[ordinary.condition == condition]
        c = conv_frame[conv_frame.condition == condition]
        conversion = a["trap_conversion"].dropna()
        pre_scope_recovery = a[
            "pre_scope_trap_to_final_correct_recovery"
        ].dropna()
        immediate_recovery = a["trap_to_correct_recovery"].dropna()
        retention = a["correct_answer_retained"].dropna()
        row = {
            "scope_level": SCOPE_LEVEL[condition],
            "final_trap_share": float(a["trap"].mean()),
            "trap_conversion_rate": (
                float(conversion.mean()) if not conversion.empty else None
            ),
            # Compatibility key, now correctly defined as a transition rather
            # than final trap prevalence.
            "conversion": float(conversion.mean()) if not conversion.empty else None,
            "conversion_vs_baseline": (
                float(conversion.mean() - base_conversion)
                if not conversion.empty and not np.isnan(base_conversion)
                else None
            ),
            "pre_scope_trap_to_final_correct_recovery_rate": (
                float(pre_scope_recovery.mean())
                if not pre_scope_recovery.empty
                else None
            ),
            "n_pre_scope_recovery_opportunities": int(len(pre_scope_recovery)),
            # Legacy endpoint retained for compatibility. Unlike the primary
            # scope endpoint, this only recognizes a trap at final_round - 1.
            "trap_to_correct_recovery_rate": (
                float(immediate_recovery.mean())
                if not immediate_recovery.empty
                else None
            ),
            "correct_answer_retention_rate": (
                float(retention.mean()) if not retention.empty else None
            ),
            "ordinary_neutral_accuracy": float(a["correct"].mean()),
            "deferral": float(c["defer"].mean()) if not c.empty else None,
            "group_accuracy": float(c["correct"].mean()) if not c.empty else None,
            "accuracy": float(c["correct"].mean()) if not c.empty else None,
            "tie_rate": float(c["tie"].mean()) if not c.empty else None,
            "n_ordinary_neutral_rows": int(len(a)),
            "n_conversion_opportunities": int(len(conversion)),
            "n_recovery_opportunities": int(len(immediate_recovery)),
            "n_retention_opportunities": int(len(retention)),
            "n_conversations": int(len(c)),
        }
        for exposure_column in EXPOSURE_COLUMNS:
            if exposure_column in a:
                row[f"{exposure_column}_rate"] = float(
                    a[exposure_column].mean()
                )
        levels[condition] = row
        print(
            f"  {condition_label(condition):28s} {row['final_trap_share']:10.3f} "
            f"{(row['trap_conversion_rate'] if row['trap_conversion_rate'] is not None else float('nan')):9.3f} "
            f"{(row['pre_scope_trap_to_final_correct_recovery_rate'] if row['pre_scope_trap_to_final_correct_recovery_rate'] is not None else float('nan')):12.3f} "
            f"{row['ordinary_neutral_accuracy']:11.3f} "
            f"{(row['group_accuracy'] if row['group_accuracy'] is not None else float('nan')):10.3f}"
        )
    out["levels"] = levels

    # --- adjacent paired contrasts -----------------------------------------
    adjacent: dict[str, dict] = {}
    if scope_order == TOOL_CORRECTION_SCOPE:
        primary_pairs = [
            ("testimony_minus_none", scope_order[0], scope_order[1]),
            ("relay_minus_testimony", scope_order[1], scope_order[2]),
            ("direct_tool_minus_relay", scope_order[2], scope_order[3]),
        ]
        pairs = list(primary_pairs)
        prompt_control, placebo_control = TOOL_CORRECTION_CONTROLS
        if prompt_control in controls_present:
            pairs.append(
                ("direct_tool_minus_prompt_receipt", prompt_control, scope_order[3])
            )
        if placebo_control in controls_present:
            pairs.extend(
                [
                    ("direct_tool_minus_placebo", placebo_control, scope_order[3]),
                    ("placebo_minus_testimony", scope_order[1], placebo_control),
                ]
            )
    else:
        primary_pairs = [
            ("verifier_minus_none", scope_order[0], scope_order[1]),
            ("relay_minus_verifier", scope_order[1], scope_order[2]),
            ("direct_minus_relay", scope_order[2], scope_order[3]),
        ]
        pairs = list(primary_pairs)
    print(
        f"\n{'-' * 74}\n"
        "  ADJACENT PAIRED CONTRASTS (to arm minus from arm; descriptive)\n"
        f"{'-' * 74}"
    )
    for name, arm_from, arm_to in pairs:
        if arm_from not in present or arm_to not in present:
            adjacent[name] = {"error": "one or both adjacent arms are absent"}
            continue
        comparisons = {
            "ordinary_neutral_final_trap_share": paired_cluster_bootstrap(
                ordinary, "trap", arm_to, arm_from
            ),
            "ordinary_neutral_trap_conversion": paired_cluster_bootstrap(
                ordinary, "trap_conversion", arm_to, arm_from
            ),
            "ordinary_neutral_pre_scope_trap_to_final_correct_recovery": (
                paired_cluster_bootstrap(
                    ordinary,
                    "pre_scope_trap_to_final_correct_recovery",
                    arm_to,
                    arm_from,
                )
            ),
            # Compatibility comparison for the old immediate pre-final
            # endpoint. It is no longer the displayed scope-recovery result.
            "ordinary_neutral_trap_to_correct_recovery": paired_cluster_bootstrap(
                ordinary, "trap_to_correct_recovery", arm_to, arm_from
            ),
            "ordinary_neutral_accuracy": paired_cluster_bootstrap(
                ordinary, "correct", arm_to, arm_from
            ),
            "group_accuracy": paired_cluster_bootstrap(
                conv_frame, "correct", arm_to, arm_from
            ),
        }
        adjacent[name] = {
            "from_condition": arm_from,
            "to_condition": arm_to,
            **comparisons,
        }
        print(f"  {name}")
        for outcome in (
            "ordinary_neutral_pre_scope_trap_to_final_correct_recovery",
            "ordinary_neutral_accuracy",
            "group_accuracy",
        ):
            result = comparisons[outcome]
            if "error" in result:
                print(f"    {outcome}: {result['error']}")
            else:
                print(
                    f"    {outcome}: {result['difference']:+.3f} "
                    f"95% CI [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]"
                )
    out["adjacent_contrasts"] = adjacent

    # Holm correction is applied only to the three preregistered adjacent-arm
    # tests for ordinary-neutral final accuracy. Mechanism controls, group
    # accuracy, and mechanism outcomes stay outside this family. The 95%
    # bootstrap intervals above remain unadjusted and are labelled as such in
    # the preregistration.
    primary_names = [name for name, _arm_from, _arm_to in primary_pairs]
    if not apply_primary_holm:
        out["primary_inference"] = {
            "applied": False,
            "reason": "secondary/exploratory subgroup; outside primary Holm family",
            "family": primary_names,
        }
        return out

    raw_p_values: dict[str, float] = {}
    inference_errors: dict[str, str] = {}
    for name in primary_names:
        result = adjacent.get(name, {}).get("ordinary_neutral_accuracy", {})
        if "error" in result:
            inference_errors[name] = str(result["error"])
            continue
        p_value = result.get("p_value")
        if p_value is None or not np.isfinite(p_value):
            inference_errors[name] = "finite bootstrap p-value unavailable"
            continue
        raw_p_values[name] = float(p_value)

    primary_inference: dict[str, object] = {
        "outcome": "ordinary_neutral_accuracy",
        "family": primary_names,
        "method": "Holm-Bonferroni across the three primary paired contrasts",
        "alpha": 0.05,
        "raw_p_values": raw_p_values,
    }
    if inference_errors:
        primary_inference["errors"] = inference_errors
        primary_inference["holm_adjusted_p_values"] = None
        print("\n  PRIMARY HOLM INFERENCE: unavailable")
        for name, error in inference_errors.items():
            print(f"    {name}: {error}")
    else:
        adjusted = holm(raw_p_values)
        primary_inference["holm_adjusted_p_values"] = adjusted
        primary_inference["reject_at_alpha_0_05"] = {
            name: bool(adjusted[name] < 0.05) for name in primary_names
        }
        print("\n  PRIMARY HOLM INFERENCE (ordinary-neutral final accuracy)")
        for name in primary_names:
            result = adjacent[name]["ordinary_neutral_accuracy"]
            result["p_value_holm"] = adjusted[name]
            result["holm_reject_alpha_0_05"] = bool(adjusted[name] < 0.05)
            print(
                f"    {name}: raw p={raw_p_values[name]:.4g}, "
                f"Holm p={adjusted[name]:.4g}, "
                f"reject={adjusted[name] < 0.05}"
            )
    out["primary_inference"] = primary_inference

    # Backward-compatible alias for the formerly separate provenance output.
    provenance_key = (
        "relay_minus_testimony"
        if scope_order == TOOL_CORRECTION_SCOPE
        else "relay_minus_verifier"
    )
    if provenance_key in adjacent:
        out["provenance_L3_minus_L2"] = adjacent[provenance_key].get(
            "ordinary_neutral_final_trap_share"
        )
    return out


def attribution_effect_report(
    agent_frame: pd.DataFrame,
    conv_frame: pd.DataFrame,
    conditions: list[str],
) -> dict:
    """Assigned-minus-anonymous effects within mesh, separately by arm."""
    ordinary = agent_frame[agent_frame.role == "ordinary_neutral"].copy()
    out: dict[str, object] = {
        "population": "ordinary_neutral",
        "topology": PRIMARY_TOPOLOGY,
        "comparison": "assigned_minus_anonymous",
        "contrasts": {},
    }
    for condition in conditions:
        agent_cell = ordinary[ordinary.condition == condition].copy()
        conv_cell = conv_frame[conv_frame.condition == condition].copy()
        if agent_cell.empty or conv_cell.empty:
            continue
        agent_cell["condition"] = agent_cell["attribution"]
        conv_cell["condition"] = conv_cell["attribution"]
        out["contrasts"][condition] = {
            "label": condition_label(condition),
            "ordinary_neutral_accuracy": paired_cluster_bootstrap(
                agent_cell, "correct", "assigned", "anonymous"
            ),
            "ordinary_neutral_final_trap_share": paired_cluster_bootstrap(
                agent_cell, "trap", "assigned", "anonymous"
            ),
            "group_accuracy": paired_cluster_bootstrap(
                conv_cell, "correct", "assigned", "anonymous"
            ),
        }
    return out


def factorial_cell_report(
    agent_frame: pd.DataFrame,
    conv_frame: pd.DataFrame,
) -> list[dict]:
    """Descriptive cells that never average topology or attribution together."""
    ordinary = agent_frame[agent_frame.role == "ordinary_neutral"]
    keys = ["condition", "topology", "attribution"]
    rows: list[dict] = []
    for key, agents in ordinary.groupby(keys, dropna=False):
        condition, topology, attribution = key
        conversations = conv_frame[
            (conv_frame.condition == condition)
            & (conv_frame.topology == topology)
            & (conv_frame.attribution == attribution)
        ]
        row = {
            "condition": condition,
            "topology": topology,
            "attribution": attribution,
            "n_scenarios": int(agents.scenario_id.nunique()),
            "n_ordinary_neutral_rows": int(len(agents)),
            "n_conversations": int(len(conversations)),
            "ordinary_neutral_accuracy": float(agents.correct.mean()),
            "ordinary_neutral_final_trap_share": float(agents.trap.mean()),
            "group_accuracy": (
                float(conversations.correct.mean())
                if not conversations.empty
                else None
            ),
            "group_deferral_rate": (
                float(conversations.defer.mean())
                if not conversations.empty
                else None
            ),
            "group_tie_rate": (
                float(conversations.tie.mean())
                if not conversations.empty
                else None
            ),
        }
        for exposure_column in EXPOSURE_COLUMNS:
            if exposure_column in agents:
                row[f"{exposure_column}_rate"] = float(
                    agents[exposure_column].mean()
                )
        rows.append(row)
    return rows


def factorial_interaction_report(
    agent_frame: pd.DataFrame,
    conv_frame: pd.DataFrame,
    scope_order: tuple[str, ...],
) -> dict:
    """Descriptive arm x topology x attribution interaction decomposition.

    These are transparent difference-in-differences, not additional
    confirmatory tests. Inferential claims remain restricted to the primary
    mesh/assigned family.
    """
    ordinary = agent_frame[agent_frame.role == "ordinary_neutral"]
    if scope_order == TOOL_CORRECTION_SCOPE:
        pairs = (
            ("testimony_minus_none", scope_order[0], scope_order[1]),
            ("relay_minus_testimony", scope_order[1], scope_order[2]),
            ("direct_tool_minus_relay", scope_order[2], scope_order[3]),
        )
    else:
        pairs = (
            ("verifier_minus_none", scope_order[0], scope_order[1]),
            ("relay_minus_verifier", scope_order[1], scope_order[2]),
            ("direct_minus_relay", scope_order[2], scope_order[3]),
        )

    def rates(frame: pd.DataFrame, outcome: str) -> dict[tuple[str, str, str], float]:
        return {
            tuple(key): float(value)
            for key, value in frame.groupby(
                ["condition", "topology", "attribution"]
            )[outcome].mean().items()
        }

    outcomes = {
        "ordinary_neutral_accuracy": rates(ordinary, "correct"),
        "group_accuracy": rates(conv_frame, "correct"),
    }
    report: dict[str, object] = {
        "status": "exploratory_descriptive",
        "primary_reference_cell": {
            "topology": PRIMARY_TOPOLOGY,
            "attribution": PRIMARY_ATTRIBUTION,
        },
        "outcomes": {},
    }
    for outcome_name, rate_map in outcomes.items():
        outcome_report: dict[str, object] = {}
        for contrast_name, arm_from, arm_to in pairs:
            effects: dict[tuple[str, str], float] = {}
            for topology in sorted(agent_frame.topology.unique()):
                for attribution in sorted(agent_frame.attribution.unique()):
                    from_key = (arm_from, topology, attribution)
                    to_key = (arm_to, topology, attribution)
                    if from_key in rate_map and to_key in rate_map:
                        effects[(topology, attribution)] = (
                            rate_map[to_key] - rate_map[from_key]
                        )
            attribution_modification = {}
            for topology in sorted(agent_frame.topology.unique()):
                assigned = effects.get((topology, "assigned"))
                anonymous = effects.get((topology, "anonymous"))
                if assigned is not None and anonymous is not None:
                    attribution_modification[topology] = assigned - anonymous
            topology_modification = {}
            for attribution in ("anonymous", "assigned"):
                mesh = effects.get((PRIMARY_TOPOLOGY, attribution))
                if mesh is None:
                    continue
                topology_modification[attribution] = {
                    topology: effect - mesh
                    for (topology, active_attribution), effect in effects.items()
                    if active_attribution == attribution
                    and topology != PRIMARY_TOPOLOGY
                }
            mesh_attr_modification = attribution_modification.get(PRIMARY_TOPOLOGY)
            three_way = (
                {
                    topology: value - mesh_attr_modification
                    for topology, value in attribution_modification.items()
                    if topology != PRIMARY_TOPOLOGY
                }
                if mesh_attr_modification is not None
                else {}
            )
            outcome_report[contrast_name] = {
                "from_condition": arm_from,
                "to_condition": arm_to,
                "scope_effect_by_cell": {
                    f"{topology}|{attribution}": effect
                    for (topology, attribution), effect in sorted(effects.items())
                },
                "attribution_modification_by_topology": attribution_modification,
                "topology_modification_vs_mesh_by_attribution": topology_modification,
                "three_way_modification_vs_mesh": three_way,
            }
        report["outcomes"][outcome_name] = outcome_report
    return report


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
    print(
        "  -> unadjusted 95% interval "
        f"{'includes' if crosses else 'excludes'} 0"
    )

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
    parser.add_argument("--by-attribution", action="store_true")
    parser.add_argument("--by-semantic-type", action="store_true")
    parser.add_argument(
        "--factorial",
        action="store_true",
        help="emit every arm/topology/attribution cell and descriptive interactions",
    )
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

    all_records = load_agent_records(args.run_dir / "records.jsonl")
    parse_failures = sum(bool(record.parse_failed) for record in all_records)
    carried_forward = sum(bool(record.carried_forward) for record in all_records)

    print(f"run={args.run_dir}")
    print(f"scenario_data={data_path} ({source}, sha256={sha256_file(data_path)[:16]})")
    print(f"rows={len(frame)}  scenarios={frame.scenario_id.nunique()}  "
          f"conversations={frame.run_id.nunique()}")
    print(
        "base frame: non-seed final outcomes; scope-arm agent metrics use "
        "ordinary neutrals only; cluster = scenario"
    )
    if parse_failures:
        print(
            f"\n!! RESPONSE-QUALITY WARNING: {parse_failures}/{len(all_records)} "
            "agent-round responses were unparsable; "
            f"{carried_forward} answers were carried forward. Treat trajectory and "
            "arm-effect magnitudes as quality-compromised."
        )
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
        "response_quality": {
            "agent_round_records": len(all_records),
            "parse_failed": parse_failures,
            "carried_forward": carried_forward,
            "zero_parse_failures": parse_failures == 0,
        },
        "analysis_hierarchy": {
            "primary": {
                "topology": PRIMARY_TOPOLOGY,
                "attribution": PRIMARY_ATTRIBUTION,
                "family": "three adjacent correction-scope contrasts",
            },
            "secondary": [
                "topology within assigned attribution",
                "attribution within fully connected mesh",
            ],
            "exploratory": [
                "semantic-type subgroups",
                "arm x topology x attribution interactions",
            ],
        },
    }

    # A scope run holds four different treatments. The legacy two-arm path
    # below sets injection=1 for all of them, which would silently average the
    # would collapse them into a single meaningless arm. Refuse instead.
    observed_conditions = set(frame.condition)
    tool_arms = [c for c in TOOL_CORRECTION_SCOPE if c in observed_conditions]
    legacy_arms = [c for c in CORRECTION_SCOPE if c in observed_conditions]
    if len(tool_arms) > 1 and len(legacy_arms) > 1:
        sys.exit("run mixes prompt-v1 and tool-v2 scope ladders; refusing to pool them")
    if len(tool_arms) > 1:
        scope_order = TOOL_CORRECTION_SCOPE
        scope_controls = tuple(
            c for c in TOOL_CORRECTION_CONTROLS if c in observed_conditions
        )
        arms = [*tool_arms, *scope_controls]
        reference_condition = TOOL_CORRECTION_SCOPE[1]
        out["scope_protocol"] = "tool_v2"
    else:
        scope_order = CORRECTION_SCOPE
        scope_controls = ()
        arms = legacy_arms
        reference_condition = CORRECTION_SCOPE[1]
        if len(legacy_arms) > 1:
            out["scope_protocol"] = "prompt_v1"
    out["ladder_arms_present"] = arms
    conv_frame = build_conversation_frame(args.run_dir)
    if len(arms) > 1:
        primary_frame = frame[
            (frame.topology == PRIMARY_TOPOLOGY)
            & (frame.attribution == PRIMARY_ATTRIBUTION)
        ]
        primary_conv_frame = conv_frame[
            (conv_frame.topology == PRIMARY_TOPOLOGY)
            & (conv_frame.attribution == PRIMARY_ATTRIBUTION)
        ]
        if primary_frame.empty or primary_conv_frame.empty:
            if frame.topology.nunique() == 1 and frame.attribution.nunique() == 1:
                # Historical single-cell runs remain analyzable, but are never
                # mislabeled as the new confirmatory cell.
                primary_frame = frame
                primary_conv_frame = conv_frame
                out["analysis_hierarchy"]["primary"]["status"] = (
                    "historical_single_cell_fallback"
                )
                out["analysis_hierarchy"]["primary"]["observed_topology"] = str(
                    frame.topology.iloc[0]
                )
                out["analysis_hierarchy"]["primary"]["observed_attribution"] = str(
                    frame.attribution.iloc[0]
                )
            else:
                sys.exit(
                    "factorial run lacks the preregistered primary mesh/assigned cell"
                )
        else:
            out["analysis_hierarchy"]["primary"]["status"] = "exact_cell"
        out["scope_ladder"] = scope_ladder_report(
            primary_frame,
            primary_conv_frame,
            scope_order=scope_order,
            control_conditions=scope_controls,
        )
        print(
            "\nNOTE: this run contains %d corrective-evidence delivery arms (%s). "
            "The pooled two-arm contrast is NOT reported, because pooling the "
            "arms would average mechanistically different treatments."
            % (len(arms), ", ".join(arms))
        )
        two_arm = primary_frame[
            primary_frame.condition.isin(
                (reference_condition, "no_injection_baseline")
            )
        ]
    else:
        primary_frame = frame
        primary_conv_frame = conv_frame
        two_arm = frame

    report_key = "primary_itt" if len(arms) <= 1 else "reference_two_arm"
    out[report_key] = report_block(
        "PRIMARY (intention-to-treat: all injection conversations)"
        if len(arms) <= 1
        else (
            "REFERENCE (non-primary; all non-seed agents): verifier-testimony "
            "arm versus uncorrupted reference"
        ),
        two_arm,
    )

    conditioned = pd.concat([
        two_arm[(two_arm.injection == 1) & (two_arm.seed_took_trap)],
        two_arm[two_arm.injection == 0],
    ], ignore_index=True)
    conditioned_key = (
        "primary_conditioned"
        if len(arms) <= 1
        else "exploratory_seed_uptake_conditioned"
    )
    out[conditioned_key] = report_block(
        "PRIMARY (conditioned on seed uptake: seed actually adopted the assumption)"
        if len(arms) <= 1
        else (
            "EXPLORATORY REFERENCE (all non-seed agents): verifier testimony "
            "versus uncorrupted reference, conditioned on seed uptake"
        ),
        conditioned,
    )

    if args.by_topology:
        out["by_topology"] = {}
        for topology in sorted(frame.topology.unique()):
            topology_frame = frame[
                (frame.topology == topology)
                & (frame.attribution == PRIMARY_ATTRIBUTION)
            ]
            topology_conv = conv_frame[
                (conv_frame.topology == topology)
                & (conv_frame.attribution == PRIMARY_ATTRIBUTION)
            ]
            if topology_frame.empty:
                continue
            if len(arms) > 1:
                print(
                    f"\n{'=' * 74}\n"
                    f"SECONDARY SCOPE ANALYSIS by topology: {topology}\n"
                    f"{'=' * 74}"
                )
                out["by_topology"][topology] = scope_ladder_report(
                    topology_frame,
                    topology_conv,
                    scope_order=scope_order,
                    control_conditions=scope_controls,
                    apply_primary_holm=False,
                )
            else:
                out["by_topology"][topology] = report_block(
                    f"SECONDARY by topology: {topology}",
                    topology_frame,
                )

    if args.by_attribution:
        mesh_frame = frame[frame.topology == PRIMARY_TOPOLOGY]
        mesh_conv = conv_frame[conv_frame.topology == PRIMARY_TOPOLOGY]
        out["by_attribution"] = {}
        for attribution in sorted(mesh_frame.attribution.unique()):
            attribution_frame = mesh_frame[
                mesh_frame.attribution == attribution
            ]
            attribution_conv = mesh_conv[
                mesh_conv.attribution == attribution
            ]
            if len(arms) > 1:
                out["by_attribution"][attribution] = scope_ladder_report(
                    attribution_frame,
                    attribution_conv,
                    scope_order=scope_order,
                    control_conditions=scope_controls,
                    apply_primary_holm=False,
                )
            else:
                out["by_attribution"][attribution] = report_block(
                    f"SECONDARY by attribution: {attribution}",
                    attribution_frame,
                )
        if len(arms) > 1:
            out["attribution_effects_within_mesh"] = attribution_effect_report(
                mesh_frame,
                mesh_conv,
                arms,
            )

    if args.by_semantic_type:
        out["by_semantic_type"] = {}
        for semantic in sorted(t for t in primary_frame.semantic_type.unique() if t):
            if len(arms) > 1:
                print(
                    f"\n{'=' * 74}\n"
                    f"SECONDARY SCOPE ANALYSIS by semantic type: {semantic}\n"
                    f"{'=' * 74}"
                )
                out["by_semantic_type"][semantic] = scope_ladder_report(
                    primary_frame[primary_frame.semantic_type == semantic],
                    primary_conv_frame[
                        primary_conv_frame.semantic_type == semantic
                    ],
                    scope_order=scope_order,
                    control_conditions=scope_controls,
                    apply_primary_holm=False,
                )
            else:
                out["by_semantic_type"][semantic] = report_block(
                    f"SECONDARY by semantic type: {semantic}",
                    primary_frame[primary_frame.semantic_type == semantic],
                )

    if args.factorial:
        out["factorial_cells"] = factorial_cell_report(frame, conv_frame)
        if len(arms) > 1:
            out["factorial_interactions"] = factorial_interaction_report(
                frame,
                conv_frame,
                scope_order,
            )

    if args.by_topology or args.by_attribution or args.by_semantic_type or args.factorial:
        print("\nNOTE: secondary contrasts are exploratory and unadjusted for "
              "multiplicity. Report them as such.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
