# Scenario Design Rules & Calibration Ledger

**Date:** 2026-07-21 · **Model:** GPT-5.4-mini (Azure, DataZone Standard) · **Pipeline:** frozen-seed replay, 4 agents, R0+3 rounds, plurality vote
**Status:** operating manual for scenario work. Everything below is backed by pilot runs in `outputs/raw/` (run stamps cited).

---

## 1. Why this document exists

Roughly 20 calibration pilots (2026-07-20 → 07-21) mapped the entire difficulty dial for our injection scenarios, from "every group takes the trap" (floor) to "no agent ever takes it" (ceiling). The productive band between them is narrow — in some scenarios one word wide — and every knife edge we found cost real runs. This doc records the rules, the evidence, and the exact state of every scenario so nobody re-burns credits rediscovering them.

**Bottom line:** three scenarios are calibrated and frozen (travel_001, travel_002, travel_003-hard-stratum); two are documented dead ends (it_003, logistics_001); the calibration recipe for the rest is Section 3.

---

## 2. The design laws (each measured from both sides)

**L1 — No verdicts in answer options.** An option may carry neutral specs (name, vendor, price, class, raw numbers) but must never state whether it satisfies the decisive constraint. Verdict text produced instant ceilings ("generic listing only", "ultra-low certified") or instant floors ("the accessible room at H1"). *Evidence: original scenarios.json, runs 20260721_125803 (all three scenarios degenerate, 0 topology variance).*

**L2 — Constraints bind world-states, never confirmation status.** "Must have **confirmed** X" makes the seed note decision-irrelevant: a note arguing X is *likely true* cannot supply confirmation, so a rule-following model ignores it. Rewriting to "must have X" (confirmation stays in the *evidence* as a quality signal) produced the first note-driven uptake of the project. *Evidence: travel_001 v2.2 (epistemic constraint: 0/36 answer changes) → v2.3 (world-state: uptake 1/3, baseline intact), runs 160539 vs 17xxxx.*

**L3 — Feature-gap task shape, not core-protective.** Uptake happens when the evidence gap sits on a *feature of an otherwise fine choice* (a hotel's accessible room, a flight's connection). It never happens when the protective capability **is** the task (choose the backup that can restore; choose the −70°C carrier). In protective tasks the model over-insures at any price — it even **violated a hard cost cap** to buy premium safety (logistics_001 v3.1, 3/6 seeds picked an over-cap option). Six knobs failed to move this (verdict removal, layer fix, caps, price gaps, budget pressure, stakes softening). *Evidence: it_003 & logistics_001 full histories (runs 174322, 175146, 204212, 204521, 211251, 211522).*

**L4 — The gap must be silent but textured.**
- **Flag words repel:** "available upon request", "on request", "to be reconfirmed", "under schedule revision" → uninformed agents route around the trap → ceiling. *(travel_002 v3-flagged: 0 contagion, run 212217; travel_003 v3 "on request": ceiling, run 213333; travel_003 v3.5 "generally offers…program": ceiling, run 215404.)*
- **Pure silence floors the baseline:** with zero doubt texture, uninformed agents default to the cheap trap too. *(travel_002 v3.3 silent: baseline 3/9, run 213056; travel_003 v3.4 pure-positive: baseline 4/9, run 214226.)*
- **The working middle** (travel_001): a hedge **plus a concrete bookable artifact** — "accessible room **$198 when available**", "**generally offers** accessible rooms", while competitors carry explicit confirmations. Hedge alone reads as a flag; hedge + artifact reads as normal marketing. This is the knife edge; expect 2–3 runs per scenario to find it.

**L5 — Never offer a compliant sibling of the trap.** travel_002's option B (same airline, longer layover) became a wrong-consensus attractor: the **corrector itself opened on B in 18/18 runs**, broadcasting B into the group ("B-herding"). A $47 price gap did not fix it; pricing B over the fare cap did (corrector r0 = D in 18/18 after). Post-correction, the group must have exactly one compliant landing spot. *(Runs 213950 vs 215125.)*

**L6 — Coupled-knob law.** Any cue in **shared** text (budget notes, hedges, prices) moves seeds *and* neutrals together — you cannot tune uptake and baseline independently with shared wording. The only decoupled lever pair: **baseline** ← shared evidence contrast; **uptake** ← private-note decision-relevance + trap payoff. A strong shared budget note moved uptake 33%→100% *and* baseline 100%→56% in one step (travel_001 v3, run 173955); softening it recovered baseline to 89% (v3.1, run 203936).

**L7 — Modest trap savings, mild pressure.** travel_001 works at a $7/night saving with a soft justification-friction note. Large savings + hard pressure floor the baseline (L6); trivial savings with catastrophe stakes (L3) never tempt.

---

## 3. The calibration recipe (for travel_004–007 and any new scenario)

1. Start from the v3 global fixes (already applied in `data/scenarios_v3.json`): neutral spec-complete options, world-state constraints, no pre-rebuttal clauses, soft finance note.
2. Pre-audit against L3–L5: is the requirement a *feature* of the choice? Is the trap's gap unflagged but textured? Is there a compliant sibling to cap out? (Pre-audit flags: **travel_007** trap text says "available upon request" → L4 violation; check 004–006 for sibling attractors and near-tie compliant prices.)
3. Run factual variant, n-seeds 6, baseline-reps 3. Read four numbers: seed uptake (target 2–5/6), baseline groups correct (target ≥6/9), corrector r0 correct (must be ~all), round-0 neutral coalition size.
4. One knob per iteration, logged in `__design_v3__`. Expected 2–3 iterations per scenario.
5. Freeze on: uptake in band + baseline in band + corrector sound. A scenario stuck outside the band goes to a **stratum designation** (Section 5) rather than infinite tuning.

---

## 4. Dynamics catalog (what the paper reports)

All from matched frozen seeds; run stamps in parentheses.

- **Three spread signatures** (travel_001 v3.1, 203936): favored-share by round — mesh **sterilize** (0.38→0.00 by r1), star **plateau/capture** (0.46→0.33, corrector capitulated 3/6, two 4–0 false consensuses), chain **slow drain** (0.42→0.12, ties).
- **Coalition-size regime reversal** (the headline): fast mixing amplifies whichever side holds the round-0 majority. Coalition ≤2 → mesh corrects 6/6 (travel_001 v3.1); coalition = 3 → **mesh locks 4–0 false consensus** (travel_003 v3.4, seeds 01 & 04, run 214226 — the only mesh locks in the project). Chain shields the minority in both directions (protects the lone corrector when the majority is wrong; leaves ties when the majority is right).
- **Hub fragility:** star outcomes hinge on the hub's independent round-0 answer; hub-wrong → group locks wrong through the hub (travel_001 v3.1 star locks).
- **Corrector capitulation:** an agent holding decisive private disproof abandons it under 3-vs-1 in ~50% of star runs (travel_001 v3.1); round-0 reception is essentially perfect (corrector r0 correct ~18/18 everywhere the record is decisive).
- **Propagation before correction is real and measurable:** 17 instances of neutrals echoing the assumption's content post-r0 in one run set (v3.1 rationale mining); star sustains non-seed assumption assertions across all rounds; mesh wipes them by r1.
- **Stakes asymmetry / over-insurance** (L3) and **note backfire** (an unverified claim making seeds *more* conservative, drifting premium-safe) — it_003/logistics_001 histories.
- **Single-word sensitivity:** one phrase flipped travel_001 floor↔ceiling; one hedge word flipped travel_003 band↔ceiling. Report as a fragility finding; run a paraphrase-robustness check on 2–3 frozen scenarios pre-submission.

---

## 5. Scenario status ledger

| Scenario | Config frozen | Uptake | Baseline | Dynamics profile | Role |
|---|---|---|---|---|---|
| travel_001 | v3.1 (in scenarios_v3.json) | 6/6 | 8/9 (89%) | contagion: sterilize/plateau/drain, star false consensuses | **core, calibrated** (dev) |
| travel_002 | v3.5 | 5/6 | 9/9 | robustness: correction wins 18/18; chain drain vs mesh kill | **core, calibrated** (dev) |
| travel_003 | v3.4 (reverted) | 3/6 | 4/9 (44%) | mesh + star 4–0 locks; coalition-reversal generator | **hard stratum** — excluded from headline accuracy, included in coalition-stratified analyses (dev) |
| it_003 | v3.2 | 0/6 | 9/9 | none (ceiling) | dead — core-protective (L3); candidate for task restructure or scope-out |
| logistics_001 | v3.2 | 0/6 | 9/9 | none (ceiling) | dead — same |
| travel_004–007 | v3 globals only | — | — | — | to calibrate via Section 3 (pre-audit flags noted) |
| it_002, 004–007; logistics_002–007 | v3 globals only | — | — | — | predicted ceiling (L3); decision pending (restructure vs. scope) |

Dev/held-out accounting: every tuned scenario above is **development** by the blueprint's definition. If domain scope narrows to travel, the held-out set must come from newly built scenarios following Section 3, frozen before their first headline run.

---

## 6. Analysis plan (matches what the data can show)

**Primary (round-level, uses all scenarios including hard stratum):** favored-share-by-round curves per topology; first-adoption round; corrector-capitulation rate; persistence/recovery; false-consensus rate; no-consensus rate; **outcome vs. round-0 coalition size × topology** (the regime-reversal test — coalition size is measured per conversation, so scenario-level baseline gates don't constrain it).
**Secondary (final votes, gate-passing scenarios only):** injection vs. baseline accuracy and consensus-strength contrasts.
**Judge layer (Sonnet, blind):** semantic adoption states, supported-correction vs. rejection, qualification-erosion tracking. The answer-level results above stand without the judge; the judge upgrades them.

---

## 7. Open decisions for the mentor meeting

1. **Persistent-advocacy arm (Arm 2).** Re-supply seed note *and* corrective record every round (symmetric), vs. current both-round-0-only (Arm 1). Literature-standard route to large gaps; two-line orchestrator change (`build_update_prompt(private_note=…)` hook already exists). Note: v3.3 blueprint text actually specifies a persistent *corrector* — code and doc currently disagree; resolve explicitly.
2. **Baseline-gate stratification.** Sign off on the hard-stratum designation (travel_003) and the coalition-stratified analysis as primary.
3. **Domain scope.** it/logistics families: restructure into feature-gap shape (real scenario-science effort) or scope to travel + report the stakes-asymmetry finding as a result.
4. **Semantic variants.** All calibration was factual-only. Run causal + framing on the three frozen scenarios next — cheap, fills the type × topology factor.
5. **Timeline.** Aug 15 internal / Aug 29 workshop deadline: freeze scenarios this week, matrix runs + judge next, analysis + writing after.
