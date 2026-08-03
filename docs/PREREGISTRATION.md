# Preregistration — confirmatory run

**Status:** DRAFT. Bracketed magnitudes marked `[ FILL FROM EXPLORATORY ]` are
filled from the 2026-08-03 exploratory run, then this file is committed and the
confirmatory run is started. The commit hash of this file must predate the
confirmatory run timestamp.

**Written:** 2026-07-26 · **Rewritten for the scope ladder:** 2026-08-03 ·
**Target venue deadline:** 2026-08-29

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
favours a wrong answer. Three other agents do not receive it. One agent (the
corrector) receives an external record refuting the assumption at round 2.

**The independent variable is the scope of that corrective record** — how many
agents hold it, and whether its provenance survives the hop between agents.

### The ladder

| level | condition | who holds the record | how it moves |
|---|---|---|---|
| 1 | `injection_no_correction` | nobody | — |
| 2 | `injection` | corrector only | corrector's own words only |
| 3 | `injection_relay` | corrector only | attached verbatim to the corrector's outgoing message |
| 4 | `injection_broadcast` | every agent | directly, no hop |

Levels 2 and 3 hold *who knows* constant and vary *whether the source travels
with the claim*. Levels 3 and 4 hold *the record reaching everyone* constant and
vary *whether it arrives first-hand or relayed*. Held constant across all four:
scenarios, injection text, corrective text, `available_round`, topologies, and
the frozen seed instances. All four arms are collected **within one run**
(`--scope-ladder`).

### The account being tested

Corrective evidence in this design is **negative information**. It establishes
that the trap option is disqualified; it does not establish which option is
compliant. Only the neutral holding the correct option's clearance can license
a positive choice, and under the scenario's pooling rule that clearance is
authoritative secondhand as well as first-hand.

The account, generated post hoc from `vendor_001` and tested here against data
that does not yet exist:

> Widening the scope of a disqualifying record **suppresses the false belief**
> and simultaneously **crowds out the positive testimony** needed to replace it.
> Agents act on the eliminating fact and discount the vouching fact, even where
> the task states the vouching fact is sufficient. Suppression and correctness
> therefore dissociate: the scope that best removes a false belief is not the
> scope that best produces a right decision.

### H1 (primary, confirmatory): conversion falls monotonically with scope

Non-seed final-round trap adoption, minus its matched no-injection baseline,
is non-increasing across levels 1 → 4.

**Committed prediction.**

- Δ(level 1) − Δ(level 4) ≥ `[ FILL FROM EXPLORATORY ]` percentage points, 95%
  cluster-bootstrap CI excluding zero.
- The four-level ordering is non-increasing at every step.

**What falsifies H1.** The level 1 − level 4 CI includes zero, or the ordering
inverts at any step by more than sampling error. Either is reported as a failed
prediction, not reinterpreted.

**Prior evidence.** The two-level version of this contrast has been observed
exploratorily: neutral conversion 10.2% without correction vs 5.7% with a single
verifier, difference +4.9 points [+1.7, +8.8]. Levels 1 and 2 are therefore a
replication. **Only levels 3 and 4 are genuinely predicted.**

### H2 (primary, confirmatory): deferral rises with scope

Final-round group deferral — the share of conversations whose plurality answer
is the "defer / hold" option E — is non-decreasing across levels 1 → 4.

**Committed prediction.** Deferral(level 4) − deferral(level 1) ≥
`[ FILL FROM EXPLORATORY ]` percentage points, 95% CI excluding zero.

E carries the same meaning in every scenario, so an E effect is a systematic
property of the design rather than scenario noise.

### H3 (primary, confirmatory): accuracy is NOT predicted monotonic

Final-round group accuracy is **not** predicted to increase with scope. The
account predicts that level 4 converts wrong answers into abstentions rather
than into right answers, so accuracy may fall between level 3 and level 4 even
as conversion falls.

**Committed prediction.** Accuracy(level 4) ≤ accuracy(level 3). This is a
directional prediction of *non-improvement*, and it is the hypothesis that
distinguishes the account from the ordinary "more correction is better" view.

**What falsifies H3.** Accuracy(level 4) > accuracy(level 3) with the CI on the
difference excluding zero. That result supports the ordinary view and is
reported as such.

H1, H2 and H3 are three separate predictions about the same manipulation. They
can come apart, and the interest of the design is precisely that they may.

### H4 (secondary, exploratory)

- **Provenance.** Level 2 vs level 3 isolates whether a source travelling with a
  claim changes anything. **No direction predicted** — the smoke test showed the
  two levels near-identical, which is uninformative at n=2.
- **Semantic type.** Factual vs causal vs framing. **No direction predicted**;
  exploratory data showed factual − causal = +0.0 points [0, 0].
- **Topology.** `looped_chain` < `neutral_star`, an ordering that replicated
  across runs whose absolute levels differed by 42 points (§6).

---

## 2. Design

4 agents · 4 rounds (round 0 independent, 3 synchronous update rounds).
Roles: one seed (holds the injection), one corrector (holds corrective
evidence, released at round 2), two neutrals — one holding the correct option's
clearance, one holding a different disqualification.

Conditions, all collected within the same run: the four ladder levels above,
plus `no_injection_baseline` (matched; seed receives a neutral note instead of
the injection) and `independent_control` (baseline information, 1 round, no
messages).

Topologies: `looped_chain`, `neutral_star`, `fully_connected_mesh`.

**Track.** Track B (hidden profile) — resolved. Findings are distributed one per
agent and only pooling identifies the compliant option.

- Corpus: `data/scenarios_hidden_profile_candidates.json`,
  sha256 `c0dc5ac515c70287bff650e28535f4e2a35fcd9b43b0432674709fb9ca1559d7`
- 33 scenarios (34 authored, `custodian_001` excluded — see §5)
- Oracle validity: 164/165 = 99.4% with full information
- No-pooling gate: 0/33 scenarios are solvable by dropping the trap and taking
  the metric optimum, enforced by Rule 3 in
  `scripts/build_hidden_profile_scenarios.py`. Lone-agent accuracy is 10%,
  below the 25% chance rate.

**Models.** Two families, same tier, to prevent a single-model result:
`azure:gpt-5.4-mini` and `claude-haiku-4-5-20251001`. The semantic judge is
`claude-sonnet-5` and is deliberately a different model from either subject, so
that no subject model judges its own output.

**Sample size.** 33 scenarios × 3 semantic types × 3 topologies × 3 seed
instances × 4 ladder levels, plus matched baselines and controls, per model
family. Scenario is the unit of generalization; 33 clusters.

---

## 3. Primary outcomes

> **1. Non-seed trap adoption at the final round** — the proportion of non-seed
> agent-rounds at the final round whose answer equals the variant's
> `favored_option`, minus its matched `no_injection_baseline`, computed at each
> of the four ladder levels (H1).
>
> **2. Group deferral at the final round** — the share of conversations whose
> plurality answer is option E, at each level (H2).
>
> **3. Group accuracy at the final round** — the share of conversations whose
> plurality answer equals `correct_answer`, at each level (H3).

**Power.** `[ RECOMPUTE — the existing simulation assumed 10 scenario clusters
and a two-arm contrast; this design has 33 clusters and a four-level ordered
factor. Recompute before committing this file. ]`

Fixed choices, made before seeing confirmatory data:

- **Seed excluded** from the conversion outcome. Seed adoption is the delivery
  of the treatment, not propagation of it.
- **Final round only.** Round 3 in the 4-round design. An any-later-round
  measure counts transient adoption and scores a working correction as a
  failure; it is not used.
- **Unit** = agent-round (conversion) or conversation (deferral, accuracy).
  **Cluster** = `scenario_id` throughout.
- **Parse failures** contribute no vote and are excluded from numerator and
  denominator; their count is reported.
- **Ties** are not counted as correct, and the tie rate is reported separately.

**Effect measure.** Difference in proportions with a 95% cluster bootstrap CI
resampling scenarios with replacement (10,000 draws, seed 20260826), paired
where arms share scenarios. Supporting model: GEE logistic regression,
exchangeable working correlation within scenario, robust standard errors.

**Multiplicity.** H1, H2 and H3 are three primary tests. Holm correction across
the three, applied to the level 1 vs level 4 contrast for H1 and H2 and the
level 3 vs level 4 contrast for H3. Per-step ordering checks are descriptive.

**Decision rule.** Each hypothesis is supported only if its 95% CI excludes zero
in the predicted direction after correction. Nothing else counts as support.

---

## 4. Secondary and exploratory outcomes

Reported with CIs, **unadjusted for multiplicity, and labelled exploratory**:

1. Level 2 vs level 3 (provenance, H4).
2. Adoption by topology and by semantic type (H4).
3. Seed-uptake-conditioned estimate: injection conversations restricted to those
   whose seed adopted the assumption at round 0. This is a post-randomisation
   conditioning and is **not** causal; it is reported because a conversation
   whose seed never adopted cannot show propagation.
4. No-consensus rate and uncertainty-language rate by level.
5. Resistance by neutral role — the neutral holding the correct option's
   clearance vs the neutral holding a different disqualification. Exploratory
   rates were 3.6% and 8.4%. This is the direct test of the crowding-out
   mechanism: the account predicts the clearance-holder resists more.
6. Trap-adoption trajectory across rounds 0–3.
7. Self-correction rate without external evidence (exploratory: 0/60).
8. Judge-derived measures (acceptance rate, false consensus) **only if** the
   judge is available; no primary result depends on it.

---

## 5. Exclusions

Decided in advance:

- `custodian_001` is excluded from the corpus and never collected. Its oracle
  scored 0/5 then 3/5 after two revisions against a 0.8 threshold; a third
  revision would be tuning a scenario to its outcome. The exclusion is recorded
  in `EXCLUDED` in `scripts/build_hidden_profile_scenarios.py` with this reason.
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

1. **Every condition contrast is within-run.** All four ladder levels, the
   matched baseline and the independent control are collected inside one run
   against the same frozen seed instances. Scope is a CONDITION, not a run-level
   flag, so all four arms share whatever run-level state fluctuates.
2. **No cross-run contrast is reported as a result.** A control collected in a
   different run — including the same day — is not a valid comparison. This
   applies to the two model families: they are reported as two separate
   within-run ladders, never as a cross-run difference.
3. **Absolute rates are not claimed to generalize.** Claims are restricted to
   within-run contrasts and orderings. Orderings have proved more stable than
   levels.
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
scripts/analyze_statistics.py   # primary outcomes, CIs, GEE
scripts/check_drift.py          # round-0 comparability
```

The confirmatory run is analysed once with this pipeline. Any change to the
analysis after seeing confirmatory data is logged in §9.

**Measurement validity precondition.** The answer/rationale incoherence rate
must be established by hand-labelling before this file is committed. Two
regexes disagree (5% vs 30%) and neither is trusted. A hand-labelled sample of
~100 corrector rationales at round 2 goes in `data/human_labels/`, labelled by
someone not working on the analysis code. If the rate is high, the coding of
the primary outcome is in question and that must be known in advance, not
discovered afterwards.

---

## 8. What is explicitly NOT confirmatory

- All runs before this file's commit date, including the 2026-08-03 exploratory
  scope-ladder run used to fill the magnitudes in §1.
- The entire scenario-calibration history (`docs/scenario_design_and_calibration.md`).
- The crowding-out account in §1 — generated from two seeds of one scenario
  (`vendor_001`) and then widened by an exploratory run. It is the hypothesis
  under test here, not a finding.
- **The speaker-standing dissociation**, which was the previous version's H1. It
  is not tested and cannot be tested with this corpus: the builder *enforces*
  that the seed owns the trap option, so every scenario is standing-high by
  construction. Testing it would require a second corpus and is out of scope.
- The between-run instability in §6 — discovered post hoc, from three runs of a
  single scenario. It is reported as an observation motivating the within-run
  design, not as a tested hypothesis, and no causal account of it is claimed.
- The observation that recovery proceeds without corrective evidence, which
  motivated the ablation arm but was itself observed exploratorily.

---

## 9. Deviations log

Any departure from this document is recorded here with date and reason, before
the paper is submitted.

| Date | Deviation | Reason |
|------|-----------|--------|
| 2026-08-03 | Primary hypothesis replaced. The previous H1 (speaker-standing dissociation) is dropped and replaced by the three-part scope-ladder prediction. | The standing manipulation is not implemented and is blocked by a corpus invariant (§8). The scope ladder is the design that was actually built. |
| 2026-08-03 | Exploratory run drops the `framing` semantic type (`--variants factual causal`). | Framing asks the agent to override a hard constraint and cannot succeed by construction, so it carries no information about scope. It is **restored for the confirmatory run**, because "cannot succeed by construction" is an argument and the arm is the demonstration. |
