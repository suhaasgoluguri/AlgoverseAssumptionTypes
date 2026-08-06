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
| `merged-design` | `9271c31`+ | both, plus corpus article fix |
| `hidden-profile-v2` | `c987054`+ | **current working branch**; merged design plus the frozen full-factorial protocol |

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

**Relay is confounded with topology.** The relay arm attaches the record to the
corrector's outgoing message, so it reaches only that agent's direct listeners:
**3 of 3 peers in mesh, 1 of 3 in chain and star**. Measured consequence
(relay − testimony on spread):

| | anonymous | assigned |
|---|---|---|
| mesh | −0.019 [−0.045, +0.007] | **−0.095 [−0.129, −0.062]** |
| chain | +0.010 | +0.016 |
| star | −0.007 | −0.007 |

Relay exposure is topology-dependent by design: the record travels only with the
corrector's outgoing message. Never pool these topology cells. The confirmatory
scope contrast is estimated in mesh + assigned attribution, where all three
peers receive the corrector's message. Chain and star are collected as
prespecified secondary finite-horizon system-effectiveness analyses; their
partial relay reach is measured explicitly rather than treated as missing data.

The official run therefore crosses all six arms with all three topologies and
both attribution modes while restricting variants to factual and causal. The
analysis must retain the hierarchy above rather than combine all cells.

**Corpus grammar.** The main run used `c0dc5ac5`, which has "A engineer's" /
"A analyst's" in 20 of 33 scenarios, in the pooling rule itself. Fixed in
`7f0f3155`. Belongs in limitations.

**Richard's 8-scenario run doesn't reproduce.** His table shows 0% deferral in
every arm and 100% accuracy for direct access. Running his code, his corpus, his
scenarios, same model and temperature, clean tree: deferral 25–59%, direct 40.6%.
Corpus wording, `available_round`, tool exposure and scenario set are all ruled
out. Cause unknown; needs his `run_manifest.json`.

## Design constraints

- **Scenario is the bootstrap cluster.** 33 clusters sets interval width; seeds
  add within-cluster precision only. 2 seeds is fine for exploratory work.
- **All condition contrasts must be within-run** (prereg §6). Run-to-run drift
  has spanned 42 points. The confirmatory run crosses topology and attribution
  inside one run; the primary scope claim is restricted to mesh + assigned.
- **Sparse ladder cells measure finite-horizon system effectiveness.** Relay and
  testimony follow graph edges, whereas direct delivery bypasses the graph.

## Remaining follow-up work

1. **Second model family.** `clients.py` is still Azure + Mock only. Single-model
   is the exact criticism we levelled at Becker et al.
2. **Human rationale validation.** A hand-labelled rationale sample would
   strengthen qualitative mechanism claims, but it does not gate the frozen
   answer-based confirmatory outcomes.
3. **Optional replication.** A second model family would test generality, but it
   must remain separately labelled and cannot be pooled with the confirmatory
   `azure:gpt-5.4-mini` run.

The confirmatory corpus, six arms, three topologies, two attribution modes,
primary cell, contrasts, and failure policy are frozen in
`docs/PREREGISTRATION.md` and enforced by the official launcher.

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
