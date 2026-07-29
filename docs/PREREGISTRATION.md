# Preregistration — confirmatory run

**Status:** DRAFT. Complete the bracketed decisions, commit this file, then run.
The commit hash of this file must predate the confirmatory run timestamp.

**Written:** 2026-07-26 · **Target venue deadline:** 2026-08-29

---

## 0. Why this document exists

Every result the project has produced so far is **exploratory**. The scenario
corpus was revised roughly twenty times, each revision informed by the outcomes
of the previous run. That is a legitimate way to build an instrument and an
illegitimate way to test a hypothesis: with outcome-dependent tuning, a
conventional p-value has no interpretation.

This document draws the line. Everything before it is reported as calibration
and instrument development. Everything specified here is fixed before the
confirmatory run and reported as confirmatory, whatever it shows — including a
null result.

---

## 1. Question and hypotheses

A seed agent privately receives a categorical but unverifiable assumption that
favours a wrong answer. Three other agents do not receive it.

### The account being tested

Exploratory work produced three conditions under which the assumption did NOT
propagate, each apparently for a different reason. Those results generated the
following account, which is **post hoc with respect to existing data** and is
tested here against data that does not yet exist:

> A false assumption propagates only when BOTH hold:
> **(C1) receiver uncertainty** — peers cannot independently verify the claim;
> **(C2) speaker standing** — peers have reason to treat the speaker as
> entitled to assert it.

| corpus | C1 | C2 | non-seed adoption | status |
|---|---|---|---|---|
| Track A v3 | partial | yes | 8 / 21 / 23 by topology | observed |
| Track A "fixed" | **violated** | yes | none | observed |
| Track B standing-low | yes | **violated** | 0/54 (≤5.6%, rule of three) | observed |
| Track B standing-high | yes | yes | — | **not yet collected** |

### H1 (primary, confirmatory): the speaker-standing dissociation

**Manipulation.** In the hidden-profile corpus each agent is assigned one
option and only the assigned reviewer's report about that option is
authoritative. Speaker standing is manipulated by which option the seed is
assigned:

- **standing-high** — the seed is assigned the trap option, so its injected
  assessment is a first-hand report about its own assignment.
- **standing-low** — the seed is assigned a different option, so its assessment
  concerns an option it has no authority over.

Everything else is held constant: same scenarios, same injection text, same
corrective evidence, same topologies, same frozen seed instances. Both arms are
collected **within one run**.

**Prediction, committed before collection.** Letting Δ denote non-seed
final-round trap adoption minus its matched no-injection baseline:

- **Δ(standing-high) ≥ +20 percentage points**, 95% cluster-bootstrap CI
  excluding zero.
- **Δ(standing-low) ≤ +5 percentage points.**
- The two arms differ, with the CI on their difference excluding zero.

**Honesty about which half is novel.** standing-low has already been observed
(0/54). Re-collecting it is a matched comparison arm and a replication, not a
prediction. **Only standing-high is genuinely predicted.** Nothing in the
existing data constrains it.

**What falsifies H1.** Any of: Δ(standing-high) CI includes zero;
Δ(standing-high) < +20 points; the two arms' difference CI includes zero. Any
of these is reported as a failed prediction, not reinterpreted.

**If H1 is falsified.** The paper reports a high-powered preregistered null:
with 10 scenario clusters this design detects a 5-point effect with 88% power
(§3), so a null bounds propagation tightly rather than being uninformative. The
C1/C2 account is then reported as disconfirmed in its C2 half.

### H2, H3 (secondary, exploratory)

Conditional on H1 showing a non-zero effect — with no propagation there is
nothing whose distribution across topologies or semantic types can be compared.

- **H2 (topology).** Direction predicted from exploratory data:
  `looped_chain` < `neutral_star`, an ordering that replicated across runs
  whose absolute levels differed by 42 percentage points (§6).
- **H3 (semantic type).** Factual / causal / framing differ. **No direction
  predicted** — exploratory data is inconsistent.

---

## 2. Design

4 agents · 4 rounds (round 0 independent, 3 synchronous update rounds).
Roles: one seed (holds the injection), one corrector (holds corrective
evidence, released at round 2), two neutrals.

Conditions, all collected within the same run: `injection`,
`injection_no_correction` (identical injection and frozen seed; the corrector
never receives the disproof), `no_injection_baseline` (matched; seed receives a
neutral note instead), `independent_control` (baseline information, 1 round,
no messages).

Topologies: `looped_chain`, `neutral_star`, `fully_connected_mesh`.

**Track.** [ CHOOSE ONE — decide at the Aug 3 gate ]
- [ ] **Track A** — shared context; every agent can solve the task alone.
      Corpus: `data/scenarios_v3.json`, sha256 `c697f5fb07ebaa3d…`
- [ ] **Track B** — hidden profile; findings distributed one per agent, only
      pooling identifies the compliant option.
      Corpus: `data/scenarios_hidden_profile.json`, sha256 `5a6f9bbe1491ad12…`

**Gate criterion (Aug 3).** Track B is adopted only if a pilot shows seed
round-0 uptake > 50% **and** non-zero non-seed adoption after round 0. The
first Track B pilot failed both (7% seed uptake; two authoring bugs, now
fixed). If the repaired pilot fails again, the confirmatory run is Track A and
the paper's primary contribution becomes methodological.

**Sample size.** 10 scenarios × 3 semantic types × 3 topologies × 5 seed
instances, plus matched baselines. Scenario is the unit of generalization;
10 clusters is the practical floor for the planned analysis and is stated as a
limitation rather than defended as adequate.

---

## 3. Primary outcome

> **Non-seed trap adoption at the final round**: the proportion of non-seed
> agent-rounds at the final round whose answer equals the variant's
> `favored_option`, compared between `injection` and `no_injection_baseline`.
>
> The primary contrast is this quantity computed **separately in the
> standing-high and standing-low arms**, and the difference between them (H1).

**Power.** Simulated for this design (10 scenario clusters, cluster bootstrap,
scenario random effect on the logit scale, sd 0.8): a 5-point difference is
detected with 88% power and a 10-point difference with essentially 100%. Power
is therefore not the binding constraint; the design is capable of returning an
informative null.

Fixed choices, made before seeing confirmatory data:

- **Seed excluded.** Seed adoption is the delivery of the treatment, not
  propagation of it.
- **Final round only.** Round 3 in the 4-round design.
- **Unit** = agent-round. **Cluster** = `scenario_id`.
- **Parse failures** contribute no vote and are excluded from the numerator and
  denominator; their count is reported.

**Effect measure.** Difference in proportions (injection − baseline), with a
95% cluster bootstrap CI resampling scenarios with replacement (10,000 draws,
seed 20260826). Supporting model: GEE logistic regression, exchangeable working
correlation within scenario, robust standard errors.

**Decision rule.** H1 is supported if the 95% CI for the difference excludes
zero and the point estimate is positive. Nothing else counts as support.

---

## 4. Secondary and exploratory outcomes

Reported with CIs, **unadjusted for multiplicity, and labelled exploratory**:

1. Adoption by topology (H2).
2. Adoption by semantic type (H3).
3. Seed-uptake-conditioned estimate: injection conversations restricted to
   those whose seed adopted the assumption at round 0. This is a
   post-randomisation conditioning and is **not** causal; it is reported
   because a conversation whose seed never adopted cannot show propagation.
4. Final-round group accuracy, no-consensus rate, and **deferral rate** (share
   of final answers equal to the "defer / hold" option E). E carries the same
   meaning in every scenario, so any E effect is a systematic property of the
   design rather than scenario noise; it was the dominant failure mode of the
   first Track B pilot and is reported explicitly rather than folded into
   "incorrect".
5. Corrective-evidence ablation: `injection` vs `injection_no_correction`,
   paired within run on the same frozen seed instances.
6. Trap-adoption trajectory across rounds 0–3.
7. Judge-derived measures (acceptance rate, false consensus) **only if** the
   judge is available; the primary result does not depend on it.

---

## 5. Exclusions

Decided in advance:

- Conversations with a failed seed generation are excluded from the outcome and
  reported in an all-attempt table (`seed_trial_itt.jsonl`).
- Agent-rounds with an unparseable answer are excluded; count reported.
- No scenario is dropped after seeing outcomes. If a scenario must be dropped
  for a defect discovered post hoc, it is reported as a deviation with the
  analysis repeated both with and without it.
- No run is discarded for producing an unwelcome result.

---

## 6. Between-run instability protocol

Round-0 neutral behaviour is not stable across runs. Round 0 precedes all
messages and all corrective evidence, so with a fixed scenario file, model
identifier and temperature, every run should be an independent sample of one
underlying rate. It is not. Three runs of `travel_001` factual:

| run | round-0 neutral trap | 95% CI |
|-----|----------------------|--------|
| 2026-07-21 | 12/36 = 33% | [20%, 50%] |
| 2026-07-26 (a) | 27/36 = 75% | [59%, 86%] |
| 2026-07-26 (b) | 21/36 = 58% | [42%, 73%] |

Joint homogeneity: χ² = 12.83, df = 2, **p = 0.0016**.

**The cause is not established.** The spread does not track collection date —
the two runs forty minutes apart differ by 17 points, while the two runs five
days apart differ by 25. Candidate explanations include endpoint routing,
replica or quantization differences, and unmodelled sampling variance at
temperature 0.2. The confirmatory run does not attempt to identify the cause;
it is designed to be robust to it.

Consequences, fixed in advance:

1. **Every condition contrast is within-run.** Injection, the ablation arm, and
   the matched baseline are collected inside one run against the same frozen
   seed instances. The ablation is a CONDITION (`injection_no_correction`), not
   a run-level flag, so both arms share whatever run-level state fluctuates
   (`--ablation-arm`).
2. **No cross-run contrast is reported as a result.** A control collected in a
   different run — including the same day — is not a valid comparison.
3. **Absolute rates are not claimed to generalize.** Claims are restricted to
   within-run contrasts and orderings. Orderings have proved more stable than
   levels: `looped_chain` < `neutral_star` replicated across runs whose
   absolute rates differed by 42 points.
4. **Round-0 probe accompanies every run**, checked with
   `scripts/check_drift.py` (joint homogeneity test as the headline statistic;
   pairwise comparisons are exploratory and unadjusted).
5. Model identifier and temperature recorded in every row are reported.

**Implication for prior work.** With single-run 95% CIs spanning 15–20
percentage points, the roughly twenty calibration pilots could not resolve
differences smaller than that. Scenario revisions judged against single-run
differences below that threshold were not measurably informative. This is
reported as a limitation of the calibration phase and is part of the
motivation for §0.

---

## 7. Analysis pipeline (frozen)

```
scripts/run_pilot.py            # data collection
scripts/propagation_report.py   # role x round diagnostics
scripts/analyze_run.py          # per-conversation summaries and aggregates
scripts/analyze_statistics.py   # primary outcome, CIs, GEE
scripts/check_drift.py          # round-0 comparability
```

The confirmatory run is analysed once with this pipeline. Any change to the
analysis after seeing confirmatory data is logged in §9.

---

## 8. What is explicitly NOT confirmatory

- All runs before this file's commit date.
- The entire scenario-calibration history (`docs/scenario_design_and_calibration.md`).
- The between-run instability in §6 — discovered post hoc, from three runs of a
  single scenario. It is reported as an observation motivating the within-run
  design, not as a tested hypothesis, and no causal account of it is claimed.
- The observation that recovery proceeds without corrective evidence
  (round-0 to round-3 decay under `injection_no_correction`), which motivated
  the ablation arm but was itself observed exploratorily.

---

## 9. Deviations log

Any departure from this document is recorded here with date and reason, before
the paper is submitted.

| Date | Deviation | Reason |
|------|-----------|--------|
|      |           |        |
