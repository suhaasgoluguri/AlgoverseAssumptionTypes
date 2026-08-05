# Assumption propagation in multi-agent LLMs — working context

Research project for the **Verify-Agents** workshop, NeurIPS 2026. Deadline
**29 August 2026**, 4–9 pages, non-archival, double-blind.

## The experiment

Four LLM agents solve a **hidden-profile** task (Stasser & Titus 1985): decisive
facts are split one per agent, and only pooling identifies the correct option.
One agent (the **seed**) is privately given a false but plausible claim. Another
(the **corrector** / **verifier**) receives an external record refuting it.

Roles, fixed per topology:

| role | holds |
|---|---|
| seed | the injected false claim about its own assigned option |
| corrector | a disqualification about a different option, plus the corrective record |
| **clearance-holder** (`neutral_briefs[0]`) | "the correct option is clean" — starts at ~100% correct **by construction** |
| ordinary neutral (`neutral_briefs[1]`) | a disqualification about a losing option — **the propagation measure** |

Always split the two neutrals in analysis. Pooling them hides everything: the
clearance-holder sits at ~95–100% throughout and dominates any average.

## Branches

| branch | commit | contents |
|---|---|---|
| `attribution` | `f3ccba9` | our code — 4 prompt-based arms + attribution factor. **Produced the 33-scenario run.** Tagged `full-exploratory-2026-08-04`. |
| `hidden-profile-v2` | `da4a3b2` | Richard's — tool-call protocol, `evidence.py`, placebo + prompt-receipt controls |
| `merged-design` | `9271c31`+ | both, plus corpus article fix. **Current working branch.** 365 tests pass. |

Local `hidden-profile-v2` is stale at `f3ccba9` and never received his work —
harmless, but the name lies.

## Corpus lineage

`data/scenarios_hidden_profile_candidates.json`, built by
`scripts/build_hidden_profile_scenarios.py` (run with no args to reproduce).

| sha | wording | `available_round` | notes |
|---|---|---|---|
| `c0dc5ac5` | "A reviewer's" | 2 | **the 33-scenario run used this**; ungrammatical article in 20/33 |
| `929c0e3d` | "Each reviewer's" | 1 | Richard's — the "Each" was a fix for that article bug |
| `7f0f3155` | "A reviewer's" / **"An engineer's"** | 1 | **current** — article agreement, his timing |

`available_round: 1` is better and was kept: peers get rounds 2 *and* 3 to
respond, rather than only the final round.

## Runs

| run | corpus | what |
|---|---|---|
| `20260803_204252_azure_full_exploratory` | `c0dc5ac5` | **the main dataset** — 33 scenarios × 3 types × 3 topologies × 4 arms × 2 attribution × 3 seeds, 123,948 turns |
| `20260805_105749_azure_each_reviewer_test` | `7b01d70d` | one-word wording test, 10 scenarios |
| `20260805_112426` / `133319_reproduce_*` | `d42a097c` | Richard's ladder + corpus reproduced here |

## Findings (33-scenario run unless noted)

**Speaker attribution is the largest effect.** Naming what a speaker was
assigned — `Agent 3` → `Agent 3 (assigned: Borealis)` — adds **no new
information** and:

- group accuracy 32.5% → **64.0%** (+31.5, CI [+25.9, +37.0])
- can't-decide 58.3% → 22.6% (−35.7)
- honest baseline 24.9% → **82.5%** (97.0% in mesh)
- **but spread of the false claim 6.1% → 14.3%** (+8.2)

It credentials true and false reports equally. That trade-off is the finding.

**Why agents defer.** Of 1,131 deferring non-seed agents: 48.8% call the trap
option **unknown**, 2.4% call it disqualified, 5.2% mention a peer at all, and
**0%** say a report arrived but didn't cover the other requirements. They aren't
disbelieving the claim — unattributed speech never registers as a *report*, so
the option stays "unknown", and the brief forbids picking an unknown. Holds in
all three topologies.

**Correction-scope ladder.** Conversion falls monotonically (L1−L4 = +0.086,
p_holm <0.0001). Accuracy: 40.2 / 51.7 / 54.6 / 46.6 — broadcast is
significantly *worse* than relay (−0.080, [−0.109, −0.047]). Deferral is flat
across the ladder (n.s.) — the H2 prediction in the prereg is wrong.

**Framing never starts**: seed uptake 0.0%, so nothing can propagate.

## Known confounds and limitations

**Conversation-carried arms are confounded with topology.** The corrector's
message reaches **3 of 3 peers in mesh but 1 of 3 in chain and star**
(`senders_for`: chain is A0→A1→A2→A3→A0, star is spokes→hub). So any treatment
riding on that message lands on a different number of agents.

| arm | delivery | topology-invariant? |
|---|---|---|
| `injection_no_correction` | nothing delivered | ✅ |
| `injection` (testimony) | corrector speaks | ❌ conversation-carried |
| `injection_relay` | record attached to that message | ❌ conversation-carried |
| `injection_broadcast` / `tool_direct` | system writes to every dossier | ✅ |

Measured, relay − testimony on spread:

| | anonymous | assigned |
|---|---|---|
| mesh | −0.019 [−0.045, +0.007] | **−0.095 [−0.129, −0.062]** |
| chain | +0.010 | +0.016 |
| star | −0.007 | −0.007 |

**The rule is one rule for all arms:** contrasts are valid *within* a topology;
effect *magnitudes* must not be compared *across* topologies. Singling out relay
was wrong — testimony is affected identically. Testimony still works everywhere
(chain −0.140 under attribution, its largest cell anywhere); relay only works in
mesh, because in sparse topologies its increment lands on 1 peer instead of 3.

Any pooled relay number — e.g. the old `L3 − L2 = −0.017` "provenance null" —
averages one working cell with two underpowered ones and should not be reported.

**Open design question.** Two defensible options: (a) collect all arms in all
topologies and restrict *cross-topology* claims to the topology-invariant arms,
stating the rest in limitations; or (b) collect only topology-invariant arms
outside mesh. (a) is cheaper (~76k turns vs ~87k, since (b) must re-collect mesh
to keep the topology contrast within-run) and discards no valid data, but needs
Richard's guards relaxed and would be `prompt_v1` only.

**Richard's mesh-only guards** live in four places: the `--scope-ladder` parser
check, both ladder drivers in `orchestrator.py`, and
`test_scope_ladder_rejects_topology_or_timing_confounds`. `--scope-ladder` also
forces `--variants factual causal`. Don't route around these without asking him.

**Corpus grammar.** The main run used `c0dc5ac5`, which has "A engineer's" /
"A analyst's" in 20 of 33 scenarios, in the pooling rule itself. Fixed in
`7f0f3155`. Belongs in limitations.

**Richard's 8-scenario run doesn't reproduce — because of the model.** His table
shows 0% deferral in every arm and 100% for direct access. Running his code, his
corpus, his scenarios, clean tree, twice: deferral 25–59%, direct 40.6%. Corpus
wording, `available_round`, tool exposure and scenario set were all ruled out —
then he mentioned he had switched to **5-mini** while every one of our 218k turns
is `azure:gpt-5.4-mini` at temp 0.2. That is the remaining difference.

Two consequences. Nothing of his and ours can be pooled until one deployment is
agreed — **this is unresolved and blocks any confirmatory run.** And a minor
version bump moving deferral from 0% to 40% is itself evidence for why the second
model family matters.

**He may also have got attribution for free.** He reports rationales like "my
assigned review of Delta confirmed this". Our agents state their own assignment
in **0.2%** of rationales, so nothing self-attributes on our side. Worth checking
his rate — if it is high, his prompt supplies what our header does.

## Design constraints

- **Scenario is the bootstrap cluster.** 33 clusters sets interval width; seeds
  add within-cluster precision only. 2 seeds is fine for exploratory work.
- **All condition contrasts must be within-run** (prereg §6). Run-to-run drift
  has spanned 42 points. Attribution is crossed *inside* a run for this reason.
- **Ladder runs are single-topology** by design (see the relay confound above).

## Planned runs (verified on mock, not yet collected)

`--arms` was added for cross-topology work: it selects a subset of `prompt_v1`
arms and routes them through `run_matched_triple`, which covers every requested
topology. Mutually exclusive with `--scope-ladder`.

```
# ladder, mesh only, 4 arms x 2 attribution, factual+causal   ~30k turns, ~3.5 h
uv run python -u scripts/run_pilot.py --client azure --scope-ladder \
  --scope-protocol tool_v2 --attribution both --n-seeds 2 --tag preconf_ladder

# topology, 2 topology-invariant arms x 3 topologies x 3 types  ~57k turns, ~6.5 h
uv run python -u scripts/run_pilot.py --client azure \
  --arms injection_no_correction injection_broadcast \
  --attribution both --n-seeds 2 --tag preconf_topology
```

Mesh appears in both deliberately: run 2 needs it so the topology contrast is
within-run. Run sequentially — concurrent runs split the rate limit and caused a
600 s stall on 3 Aug.

## Open — none of these have moved

1. **Which model.** Richard is on 5-mini, we are on 5.4-mini. Blocks everything;
   nine hours of collection on the wrong one is unrecoverable.
2. **Second model family.** `clients.py` is still Azure + Mock only. Single-model
   is the exact criticism we levelled at Becker et al.
3. **Preregistration.** Now Richard's `tool_v2` rewrite with C1/C2/C3 contrasts —
   and it **does not mention attribution at all**, though attribution is the
   largest effect in the project (+31.5 vs the ladder's ~8-point spread).
   Freezing it as-is preregisters the smaller finding.
4. **Hand-labelling** ~100 corrector rationales (`data/human_labels/` is empty)
   and the 15-minute conversion spot-check. Both need someone outside the
   analysis code. Both gate the prereg freeze. Four days overdue.
5. **The paper's spine** — attribution or the ladder. Everything above waits on
   this. Recommendation on file: attribution primary, ladder secondary, tool
   protocol as converging evidence.

## Attribution is threaded by hand — check new drivers

Three silent bugs surfaced on 5 Aug, all in `run_pilot.py`: the `prompt_v1`
ladder, the matched-triple path, and the independent control each failed to pass
`attribution` down. Each would have run twice and produced anonymous
conversations both times. Two were caught only by the completion-integrity
check. If a new driver is added, verify it forwards `attribution`.

Verified inert when anonymous: 1,539 generated prompt strings hashed across
scenarios × variants × conditions × roles × rounds are **byte-identical** to
Richard's untouched commit. An anonymous run on `merged-design` is his
experiment exactly.

## Practical notes

- **PowerShell 5.1** — no `&&`. Separate lines, or `;`.
- **Cost is higher than token estimates suggest.** 218k turns consumed a monthly
  Azure allowance. Get the real per-token rate before sizing a run, and check
  whether the deployment bills reasoning tokens.
- **Runs are self-describing**: every run folder has `run_manifest.json` (git
  commit, `dirty` flag, model id, temperature, corpus sha) and
  `scenario_snapshot.json` (verbatim corpus). Trust those over the command line.
- **Launching twice is easy to do by accident** — check for a second run folder
  with 0 records.
- `uv run python -u` for live progress; piping into `Tee-Object` block-buffers.
