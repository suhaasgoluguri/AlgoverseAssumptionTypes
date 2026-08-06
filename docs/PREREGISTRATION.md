# Preregistration — confirmatory run

**Status:** READY TO FREEZE. All design and sample-size fields are resolved.
Commit this file together with the frozen analysis code before starting the
confirmatory run; that commit hash must predate the first confirmatory API call.

**Written:** 2026-07-26 · **Rewritten for the scope ladder:** 2026-08-03 ·
**Finalized for `azure:gpt-5.4-mini`, `tool_v2`, and the factorial design:** 2026-08-05 ·
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
verifier) receives an external record refuting the assumption at round 1,
after one clean propagation round.

**The independent variable is the delivery route and recipient scope of the
same corrective record.** The question is whether ordinary team members
recover because a verifier can testify, because an exact source record
survives the communication hop, or because the recipient receives that record
directly rather than through another agent. The primary construct is direct
recipient delivery, not the mechanical act of invoking a tool.

### The primary `tool_v2` arms

| arm | condition | backend access | what ordinary agents receive from round 2 onward |
|---|---|---|---|
| 1 | `injection_tool_none` | no corrective tool call | verifier output produced without the corrective record |
| 2 | `injection_tool_testimony` | verifier is forced to call the real record tool | the verifier's matched free-text answer and rationale only |
| 3 | `injection_tool_relay` | verifier is forced to call the real record tool | that same free text plus the exact canonical tool-result payload carried inside the verifier's message; peers do not call the tool |
| 4 | `injection_tool_direct` | verifier and every non-verifier are forced to call the real record tool when eligible | the matched verifier testimony plus each recipient's own `role=tool` result |

Every real access uses a forced two-completion exchange. The first completion
must call the zero-argument `read_case_record` function. The backend binds the
active scenario and recipient, returns the canonical JSON record as a
`role=tool` message, and the second completion produces the agent's answer.
The function cannot be used to search other candidates or request another
record, so verification-seeking and search ability are not part of the
treatment.

Arms 2 and 3 hold the verifier response fixed and vary whether the exact source
payload travels with it. Arms 3 and 4 hold canonical payload bytes, record ID,
record hash, and first ordinary-agent exposure round fixed and vary whether the
recipient sees a copy inside peer communication or receives the record directly
as a first-party tool result. Tool access is forced rather than optional. The
separately prespecified prompt-receipt control determines whether an Arm 4
advantage is specific to the tool channel or instead follows from direct record
presentation.

The arms are **not treated as an ordinal dose**. Three adjacent comparisons
identify different bottlenecks:

1. verifier testimony minus no correction: does merely putting truth in the
   conversation help?
2. verbatim relay minus verifier testimony: does preserving provenance help
   beyond the verifier's identical words?
3. direct recipient delivery minus exact relay: does bypassing the
   agent-mediated carrier help beyond receiving identical record bytes through
   another agent?

Held constant within every arm/topology/attribution coordinate: scenario,
injected assessment, canonical corrective record wherever it is delivered,
timing, frozen seed, and pre-correction conversation. Speaker attribution is
crossed within-run: anonymous headers identify only `Agent 2`, while assigned
headers identify `Agent 2 (assigned: Borealis)`. Assignment labels reveal the
reviewer-to-candidate mapping but not the reviewer's private finding. The
primary correction-scope estimates use mesh + assigned attribution; attribution
itself is a prespecified secondary comparison. A hidden no-correction anchor and a hidden
corrected anchor sample the release history. Every corrected measured arm
replays the same verifier response at the release round and the following
round, while unaffected pre-release cells are also matched. Replayed tool
traces preserve call/result/transcript hashes but explicitly record zero new
API calls and token use; `tool_call_executed=false` and
`tool_trace_replayed=true` distinguish them from genuine measured-arm tool
executions. This matching is part of the protocol, not an outcome exclusion.

All three topologies are collected. The fully connected mesh is the primary
topology because every ordinary agent receives the verifier's next message.
Chain and star are prespecified secondary finite-horizon conditions: testimony
and exact relay follow graph edges, whereas direct recipient delivery bypasses
the graph. Measured coordinate and arm execution orders are counterbalanced by
stable hashes. All arms, topologies, and attribution modes are collected within
one run (`--scope-ladder --scope-protocol tool_v2 --attribution both`).

### Prespecified mechanism controls

The confirmatory run includes the following conditions via `--scope-controls`.
They are not fifth and sixth levels of an ordinal ladder and are not included
in the three primary contrasts.

| control | condition | ordinary-agent treatment from round 2 onward | purpose |
|---|---|---|---|
| prompt receipt | `injection_prompt_receipt_control` | exact canonical payload inserted as neutral user-prompt text; no peer tool call | tests whether the direct arm requires a personal tool exchange or follows from direct matched presentation |
| tool placebo | `injection_tool_placebo` | forced `read_case_record` interaction returning a canonical non-diagnostic payload; matched verifier testimony is still visible | tests whether an extra forced tool turn changes behavior without corrective content |

The prompt-receipt payload is content-matched to the direct tool result. The
placebo is interaction-matched but intentionally contains no candidate status,
constraint finding, corrective record ID, or corrective record hash. The
verifier still uses the genuine tool in both corrected controls, so these
controls do not remove testimony from the conversation.

### Protocol versioning

The earlier prompt-only implementation remains available as
`--scope-protocol prompt_v1` under the historical condition IDs
`injection_no_correction`, `injection`, `injection_relay`, and
`injection_broadcast`. It used prompt-rendered private/relay/direct receipts;
it did not implement a model function call. Those runs remain valid historical
exploration but are never pooled with `tool_v2`.

### The mechanism being tested

Corrective evidence here is negative information: it disqualifies the trap but
does not itself identify the correct option. Recovery still requires pooling
the neutral agent's positive clearance with the disqualification. This lets the
experiment separate *removing the false option* from *reaching the correct
group decision*.

### Primary estimands

The primary report gives three paired adjacent-arm contrasts rather than one
"broadcast is best" comparison:

- **C1 — testimony effect:** `injection_tool_testimony` minus
  `injection_tool_none`.
- **C2 — exact-relay effect:** `injection_tool_relay` minus
  `injection_tool_testimony`.
- **C3 — direct-recipient-delivery effect:** `injection_tool_direct` minus
  `injection_tool_relay`.

Each contrast is estimated for ordinary-neutral final accuracy and group
accuracy. C3 is the main direct-delivery test. C1 and C2 explain *why* C3 is or
is not different: a C1 effect localizes benefit to testimony, a C2 effect to
source preservation, and a C3 effect to bypassing the conversational carrier.
A null C3 with a positive C2 would mean provenance survives relay adequately;
a positive C3 after a null C2 would identify the agent-mediated channel as the
bottleneck. C3 alone is not interpreted as evidence that tool invocation is
causal; that narrower claim requires the prompt-receipt control.

Three additional prespecified contrasts are mechanism checks rather than
primary tests: direct tool minus prompt receipt,
placebo tool minus testimony, and direct tool minus placebo tool. Respectively,
they probe the tool channel beyond matched direct presentation, tool-turn
mechanics without evidence, and diagnostic versus non-diagnostic tool output.

The directional expectation is non-negative accuracy change for each adjacent
contrast, but no monotonic ordering is required and any inversion is reported.
Final trap share, true round-0-to-final trap conversion, pre-scope-to-final
recovery, correct-answer retention, deferral, and rationale-coded evidence use
are mechanism outcomes. Pre-scope recovery is eligible when an ordinary
neutral holds the trap at `corrective_evidence.available_round` (the last
matched checkpoint before peer access) and succeeds when its final answer is
correct. This captures an immediate round-2 correction that remains correct in
round 3. The older pre-final-to-final recovery measure is retained as a legacy
diagnostic, but is not the scope table's displayed recovery endpoint. Both
recovery measures are conditional descriptive outcomes rather than causal
primary endpoints.

### Secondary, exploratory comparisons

- Factual and causal variants are reported separately. Semantic type is not the
  randomized primary variable. Framing has a different uptake precondition and
  is excluded from the primary scope analysis; any framing run is a separately
  labelled follow-up and is not pooled with the primary estimand.
- Seed uptake is a manipulation check, not a propagation outcome.
- Sparse topologies are collected in the same factorial run but reported only
  as prespecified secondary finite-horizon system-effectiveness analyses. They
  are never pooled with the primary mesh estimand.

---

## 2. Design

4 agents · 4 rounds (round 0 independent, 3 synchronous update rounds).
Roles: one seed (holds the injection), one verifier (receives corrective
evidence through a forced tool call in round 1 in every corrected `tool_v2`
arm), and two ordinary neutrals — one holding the correct option's clearance
and one holding a different disqualification. The verifier speaks with the
record in round 1; peers first respond to the resulting testimony, relay, or
direct tool result in round 2 and receive the assigned route again in round 3.

Conditions, all collected within the same run: the four primary `tool_v2`
delivery arms and both mechanism controls above. `no_injection_baseline` (seed
receives a neutral note instead of the injection and the verifier receives the
standard corroborating record) is an uncorrupted reference condition, not a
scope arm and not a pure no-record control. `independent_control` is omitted
from the confirmatory run.

Collected topologies: `looped_chain`, `neutral_star`, and
`fully_connected_mesh`. Primary analysis cell: `fully_connected_mesh` with
`assigned` attribution.

**Track.** Track B (hidden profile) — resolved. Findings are distributed one per
agent and only pooling identifies the compliant option.

- Corpus: `data/scenarios_hidden_profile_candidates.json`,
  sha256 `7f0f31556b2b9543815eac77272ab00f91804deb3128d33b5c7c12d48ac07fbf`
- 33 scenarios (34 authored, `custodian_001` excluded — see §5)
- Oracle validity: 164/165 = 99.4% with full information
- No-pooling gate: 0/33 scenarios are solvable by dropping the trap and taking
  the metric optimum, enforced by Rule 3 in
  `scripts/build_hidden_profile_scenarios.py`. Lone-agent accuracy is 10%,
  below the 25% chance rate.

**Subject model.** One frozen Azure deployment, recorded in the run manifest as
`azure:gpt-5.4-mini`. The request omits a custom temperature
(`--temperature none`) because the deployment rejects it. No second subject
model is part of the confirmatory claim. A separately labelled replication on
another model may be reported later but is never pooled with this run. A
semantic judge, if used, is post-hoc and cannot affect any primary outcome.

**Maximum planned sample size.** 33 scenarios × 2 primary semantic types
(factual and causal) × 3 topologies × 2 attribution modes × 3 frozen seed
instances × 6 collected
injection conditions (4 primary arms + 2 mechanism controls), plus one
uncorrupted reference conversation per scenario/type/topology/attribution
coordinate. This gives 198 seed generations shared across the factorial grid,
up to 7,128 injection conversations, 396 reference conversations, 7,524
conversations in total, and 120,384 planned agent-round records when every seed
is replayable. Scenario remains the unit of
generalization; there are 33 clusters, not 120,384 independent observations.
Framing and independent controls are not collected.

**Frozen collection command.** The run starts from a clean committed worktree
with the deployment environment variable set to `gpt-5.4-mini`:

```powershell
uv run python scripts/run_pilot.py `
  --client azure `
  --data data/scenarios_hidden_profile_candidates.json `
  --variants factual causal `
  --scope-ladder `
  --scope-protocol tool_v2 `
  --scope-controls `
  --topologies looped_chain neutral_star fully_connected_mesh `
  --attribution both `
  --n-seeds 3 `
  --baseline-reps 1 `
  --skip-independent `
  --rounds 4 `
  --temperature none `
  --max-completion-tokens 8000 `
  --verbose `
  --tag confirmatory_scope33_gpt54mini
```

---

## 3. Primary outcomes

> **Primary outcome: ordinary-neutral final accuracy in the mesh + assigned
> attribution cell.** The proportion of the two ordinary agents' round-3
> answers equal to `correct_answer`. The seed is
> the treatment recipient and the verifier is the correction carrier; including
> either would mechanically contaminate the estimate of downstream recovery.

The three primary estimates are C1, C2, and C3 above, each expressed as the
paired difference in ordinary-neutral final accuracy (to arm minus from arm).
Group accuracy is the key team-level secondary endpoint.

**Sample-size justification.** The 33 scenarios are the complete fixed corpus
that passed the oracle and no-pooling gates, so the number of independent
clusters is a corpus census rather than an effect-size-optimized choice. The
8-scenario `tool_v2` mechanism pilot estimated ordinary-neutral differences of
+0.156 for C1, +0.219 for C2, and +0.562 for C3. Under the prespecified Holm
family, the corresponding exploratory bootstrap p-values imply approximately
0.0544, 0.0544, and <0.0001 after adjustment. Those tuned-pilot values are not
treated as prospective power guarantees. Expanding to all 33 scenario clusters
and three seed instances is intended to improve cluster-level precision and
stability; inference is based on the confirmatory effect estimates and
intervals, not on post-hoc observed power.

Fixed choices, made before seeing confirmatory data:

- **Seed and verifier excluded** from ordinary-neutral outcomes. Seed uptake is
  treatment delivery; verifier behavior is mechanically affected by holding
  the corrective record.
- **Final round only.** Round 3 in the 4-round design. An any-later-round
  measure counts transient adoption and scores a working correction as a
  failure; it is not used.
- **Unit** = ordinary agent at round 3 for the primary endpoint; conversation
  for group accuracy. **Cluster** = `scenario_id` throughout.
- **Parse failures** contribute no vote and are excluded from numerator and
  denominator; their count is reported.
- **Ties** are not counted as correct, and the tie rate is reported separately.

**Effect measure.** Difference in proportions with a 95% cluster bootstrap CI
resampling scenarios with replacement (10,000 draws, seed 20260826), paired
where arms share scenarios. Supporting model: GEE logistic regression,
exchangeable working correlation within scenario, robust standard errors.

**Multiplicity.** Holm correction across C1, C2, and C3 for the primary
ordinary-neutral accuracy endpoint. Raw bootstrap p-values, Holm-adjusted
p-values, and unadjusted 95% cluster-bootstrap intervals are all reported. No
endpoint is interpreted through a single assumed monotonic trend.

**Decision rule.** An adjacent-route benefit is supported only when its point
estimate is positive and its Holm-adjusted two-sided bootstrap p-value is below
0.05. The accompanying 95% intervals are unadjusted and descriptive of
precision; Holm is applied to p-values, not retroactively to those intervals.
Null and negative contrasts are retained and interpreted as evidence about
where the delivery bottleneck is not located.

---

## 4. Secondary and exploratory outcomes

Reported with CIs, **unadjusted for multiplicity, and labelled exploratory**:

1. Group accuracy for C1, C2, and C3.
2. Final trap share and true non-trap-at-round-0 to trap-at-round-3 conversion.
3. Trap-to-correct recovery and correct-answer retention. These condition on a
   prior answer state and are mechanism descriptions, not causal endpoints.
4. Outcomes by semantic type within the primary mesh + assigned cell.
5. Seed-uptake-conditioned estimate: injection conversations restricted to those
   whose seed adopted the assumption at round 0. This is a post-randomisation
   conditioning and is **not** causal; it is reported because a conversation
   whose seed never adopted cannot show propagation.
6. No-consensus, explicit option-E deferral, and uncertainty-language rates by
   arm, reported separately. A tie is not relabelled as deferral, and no
   combined `defer_or_tie` endpoint is confirmatory.
7. Resistance by ordinary-neutral role: correct-option clearance holder versus
   other-disqualification holder.
8. Trap-adoption trajectory across rounds 0–3.
9. Judge-derived evidence citation/use and assumption-state measures **only if** the
   judge is available; no primary result depends on it.
10. **Topology analysis under assigned attribution:** the four scope arms are
    reported separately for mesh, chain, and star. Sparse results measure
    four-round system effectiveness, including limited network reach; they are
    not relabelled as pure evidence-format effects.
11. **Attribution analysis within mesh:** assigned-minus-anonymous differences
    are reported separately for each arm. Anonymous and assigned rows are never
    pooled into one arm rate.
12. **Exposure diagnostics:** whether an ordinary agent heard a record-backed
    verifier message, received the exact relay, received a direct tool record,
    or remained unreached by the final round.
13. Arm × topology × attribution difference-in-differences are descriptive and
    explicitly exploratory.

---

## 5. Exclusions

Decided in advance:

- `custodian_001` is excluded from the corpus and never collected. Its oracle
  scored 0/5 then 3/5 after two revisions against a 0.8 threshold; a third
  revision would be tuning a scenario to its outcome. The exclusion is recorded
  in `EXCLUDED` in `scripts/build_hidden_profile_scenarios.py` with this reason.
- Conversations with a failed seed generation are excluded from the outcome and
  reported in an all-attempt table (`seed_trial_itt.jsonl`).
- Agent-rounds with an unparseable answer are flagged and counted. A failed
  final response contributes no individual vote and is excluded from the
  corresponding agent-level denominator; it is never silently replaced by the
  carried-forward answer for scoring. Intermediate failures may carry prior
  text forward so later rounds can continue, but the failure remains logged.
  Parse or non-tool generation failures do not invalidate an otherwise complete
  run; required tool-delivery failures still fail the protocol-integrity gate.
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

1. **Every delivery contrast is within-run.** All four delivery arms and both
   mechanism controls are collected inside one run against the same frozen
   seed instances. The uncorrupted reference is collected in that run but does
   not replay an injected seed. Delivery route is a condition, not a run-level
   flag, so all primary arms share run-level state.
2. **No cross-run contrast is reported as a result.** A control collected in a
   different run — including the same day — is not a valid comparison. Any
   later model replication is reported separately and never pooled with the
   `gpt-5.4-mini` confirmatory run.
3. **Absolute rates are not claimed to generalize.** Claims are restricted to
   paired within-run contrasts.
4. **Round-0 probe accompanies every run**, checked with
   `scripts/check_drift.py` (joint homogeneity test as the headline statistic;
   pairwise comparisons are exploratory and unadjusted).
5. Model identifier, omitted custom temperature, completion-token limit, SDK
   timeout, and retry policy are frozen in the run provenance and reported.

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
scripts/analyze_statistics.py   # primary outcomes, CIs, Holm p-values, GEE
scripts/check_drift.py          # round-0 comparability
```

The statistics command uses `--by-topology --by-attribution
--by-semantic-type --factorial`. The primary Holm family is computed only from
the mesh + assigned cell. Secondary topology reports use assigned attribution;
secondary attribution reports use mesh. Factorial cells and three-way
interactions are never silently pooled and remain exploratory.

The confirmatory run is analysed once with this pipeline. Any change to the
analysis after seeing confirmatory data is logged in §9.

**Answer/rationale scope.** The primary outcome is the parsed answer letter,
not a rationale-derived label. Answer/rationale consistency flags and any
judge-derived evidence-use labels are secondary quality diagnostics. They are
reported with coverage and uncertainty and cannot override or exclude a valid
primary answer. Any hand-labelled rationale sample is therefore useful
measurement validation for qualitative claims, but is not a precondition for
the answer-based confirmatory analysis.

---

## 8. What is explicitly NOT confirmatory

- All runs before this file's commit date, including the 2026-08-03 prompt-v1
  scope run and the 2026-08-04 eight-scenario `tool_v2` mechanism run used to
  choose controls and finalize the estimands.
- The entire scenario-calibration history (`docs/scenario_design_and_calibration.md`).
- All exploratory magnitudes used in the sample-size justification, including
  the observed equality of direct tool and prompt receipt. Those observations
  motivated the mechanism controls but are not confirmatory findings.
- **The speaker-standing dissociation**, which was the previous version's H1. It
  is not tested and cannot be tested with this corpus: the builder *enforces*
  that the seed owns the trap option, so every scenario is standing-high by
  construction. Testing it would require a second corpus and is out of scope.
- The between-run instability in §6 — discovered post hoc, from three runs of a
  single scenario. It is reported as an observation motivating the within-run
  design, not as a tested hypothesis, and no causal account of it is claimed.
- Any prompt-v1 crowding-out, deferral, or no-decision account. Those historical
  observations use a different delivery protocol and are not pooled with
  `tool_v2`.

---

## 9. Deviations log

Any departure from this document is recorded here with date and reason, before
the paper is submitted.

| Date | Deviation | Reason |
|------|-----------|--------|
| 2026-08-03 | Primary hypothesis replaced. The previous H1 (speaker-standing dissociation) is dropped and replaced by the three-part scope-ladder prediction. | The standing manipulation is not implemented and is blocked by a corpus invariant (§8). The scope ladder is the design that was actually built. |
| 2026-08-03 | Primary evidence-scope runs drop the `framing` semantic type (`--variants factual causal`); framing is retained only for separately labelled follow-up work. | Framing asks the agent to override a hard constraint and has a different uptake precondition, so pooling it with factual and causal variants would confound evidence delivery with manipulation validity. |
| 2026-08-04 | The current primary scope implementation changes from prompt-rendered receipts (`prompt_v1`) to forced function calls (`tool_v2`), with new condition IDs. | The v2 flow separately identifies testimony, exact relay, direct recipient delivery, and tool mechanics while preserving legacy runs as a separate protocol. |
| 2026-08-04 | Prompt receipt and tool placebo are included as prespecified mechanism controls across the full corpus. | The exploratory eight-scenario run found direct tool and matched prompt receipt both at ceiling while placebo remained near no correction. Full-corpus controls are necessary to distinguish evidence content and delivery from tool-turn mechanics; they remain outside the Holm-adjusted primary family. |
| 2026-08-04 | The confirmatory subject model is the single Azure deployment `azure:gpt-5-mini`; the earlier two-family plan is removed. | The implemented runner and completed compatibility gates use this deployment. A hypothetical unimplemented model family cannot be preregistered as collected data. |
| 2026-08-04 | C3 is interpreted as direct recipient delivery minus agent-mediated relay, not as a tool-invocation effect. | The prompt-receipt control gives recipients identical evidence without their own tool call, so tool-specific attribution requires that separate control contrast. |
| 2026-08-05 | Speaker attribution is fixed to `assigned` in every confirmatory condition rather than crossed as a second factor. | The pooling rule credits a report from a candidate's assigned reviewer. Anonymous headers make that standing unverifiable, introducing an avoidable authentication failure unrelated to corrective-evidence scope. Fixing attribution preserves the scope contrast and sample size. |
| 2026-08-05 | Before any confirmatory API call, collection was expanded to the complete 4-arm × 3-topology × 2-attribution factorial, with both mechanism controls in every coordinate. The primary scope claim remains mesh + assigned. | Complete cells prevent selective topology omission and allow prespecified secondary network and attribution analyses. Sparse testimony/relay coverage is treated as part of finite-horizon system effectiveness, not pooled into the primary delivery-format contrast. This supersedes the same-day fixed-attribution collection decision above. |
| 2026-08-05 | Before any confirmatory API call, the frozen subject deployment was changed from `azure:gpt-5-mini` to `azure:gpt-5.4-mini`. | The team selected the stronger deployment for the final confirmatory study. The deployment is pinned in the launcher and manifest and no results are pooled across models. This supersedes the 2026-08-04 model-selection row above. |
