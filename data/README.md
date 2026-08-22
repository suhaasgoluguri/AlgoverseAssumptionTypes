# Scenario corpora

## Active

| File | Role |
|---|---|
| `scenarios_hidden_profile_candidates.json` | The 33-scenario corpus. Every current result comes from this, and it is the default for `run_pilot.py`. |
| `scenarios_no_authority.json` | Ablation over the same 33: the pooling rule's authority sentence is removed, so no clause tells an agent that an assigned reviewer's report is authoritative and sufficient. |
| `scenarios_no_transfer_clause.json` | Ablation over the same 33: the authority sentence is kept but its transfer clause ("whether you hold it directly or hear it from that reviewer") is dropped, so authority no longer visibly survives relay. |
| `archive/scenarios_fixed.json` | Superseded 20-scenario Track A set. Tracked only because `scripts/audit_scenarios.py` and `tests/test_scenarios_fixed.py` audit it; its rules describe the Track A design and do not apply to the active corpus. |

## Superseded corpora are not tracked

The rest of `data/archive/` is git-ignored. No reproduction path reads it: the
confirmatory launcher runs against `scenarios_hidden_profile_candidates.json`
and the analysis reads a run snapshot. The corpora themselves are kept outside
the repository; what mattered about them is the account below.

Read that account rather than the files. Nothing here should be traced to a
superseded corpus without saying so explicitly.

## What the superseded corpora were, and why they were replaced


### Track A, pre-hidden-profile
- `scenarios_v2.json`, `scenarios_v3.json`, `scenarios_v4.json`

All four agents read the same case and one additionally received a private false
assumption. Every agent could identify the correct answer alone, so no agent had
reason to rely on a peer and nothing propagated. Replaced by the hidden-profile
design, in which the decisive facts are split one per agent.

### Presentation conditions, built on the 10-scenario corpus
- `scenarios_additive_all.json` — factual and causal made additive, so the seed
  holds its honest finding alongside every injected claim. Holds the seed's
  knowledge state constant across semantic types.
- `scenarios_labels_swapped.json` — the labels on the seed's honest finding and
  its injected claim exchanged.
- `scenarios_injection_first.json` — the injected claim stated before the honest
  finding.
- `scenarios_neutral_auth.json` — the word "authoritative" removed from the
  pooling rule.

These four measured how presentation affects which of two contradictory
statements an agent credits. The results are superseded: they predate the
28 July metric reorder, and the 10% baseline they were anchored to no longer
exists.

They are **not** regenerable. The builder flags still exist
(`--additive-all`, `--swap-labels`, `--injection-first`, `--neutral-authority`)
but would now build against the 33-scenario corpus rather than the 10 these were
collected on. That is why they are archived rather than deleted.

## Why there are so many versions

The corpus was revised repeatedly, and several of those revisions were prompted
by outcomes rather than by prior reasoning. That is recorded rather than hidden
because it is the reason every result to date is **exploratory**, and the reason
a preregistration precedes the confirmatory collection.

The revisions that changed the design substantively, in order:

1. **Seed role assignment** — three iterations. The seed was first given the
   correct option's clearance (its own verified knowledge outranked the
   injection, 7% uptake), then a third option (peers correctly discounted a
   claim about something the seed had no standing on, 0/54 conversions), and
   finally the trap itself, so that the claim is a first-hand report about its
   own assignment.
2. **Scoping clause** (`trap_other_constraints_ok`) — the injected claim
   originally cleared the trap on one of four constraints, so seeds rationally
   deferred. Adding an explicit statement that the contested item was the only
   open one is what made the manipulation sufficient. This is an outcome-driven
   revision and is disclosed as such: seed adoption of 100% is a property of
   this construction, not a measure of model credulity.
3. **Metric reorder, 28 July** — in 31 of 34 scenarios the correct answer was
   the best remaining option once the trap was dropped, so a single agent could
   solve the task without pooling. Now enforced by Rule 3 in `check_specs`.
4. **Causal injections, 28 July** — three conceded the deficiency while
   explaining its cause ("revoked *because of* a paperwork lapse"), which still
   fails the constraint. Rewritten to entail compliance.

## Archived 3 August 2026 — superseded working sets

Moved to `archive/` during a repo tidy. None was referenced by any script,
test or doc at the time; all three are calibration-era artefacts.

- `scenarios_candidates_additive.json` — the additive-framing candidate set from
  the 34-scenario build. Superseded by `scenarios_hidden_profile_candidates.json`,
  which carries the same variants after the metric reorder.
- `scenarios_remaining_v2b.json` — the partial v2b batch, folded into v3.
- `scenarios_six_calibration.json` — the six-scenario calibration subset used
  before the corpus reached 10.

What is still live in `data/` is listed in the table at the top of this file.

## Archived 3 August 2026 — the last two superseded corpora

This left `data/` holding only live corpora, so there is no way to run against
a superseded one by accident.

- `scenarios_fixed.json` — the 20-scenario Track A set. Every agent could solve
  these alone; Track B replaced them. Still audited by
  `scripts/audit_scenarios.py`, and the one archived corpus still tracked.
- `scenarios_hidden_profile.json` — the pre-28-July hidden-profile corpus, the
  version the July pilots were collected on. Superseded by the metric reorder
  that closed the no-pooling shortcut and by the causal-injection rewrite.

**One footgun this closed.** `build_hidden_profile_scenarios.py` had
`DEFAULT_OUT` pointing at `data/scenarios_hidden_profile.json`, so running the
builder with no arguments wrote the *current* 33-scenario corpus into the file
named after the *superseded* one. `DEFAULT_OUT` is now
`data/scenarios_hidden_profile_candidates.json`, and running the builder with no
arguments reproduces the live corpus byte-for-byte (sha256 `c0dc5ac5…`,
33 scenarios), which is what the builder's own comments say it should do.
