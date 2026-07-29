# Superseded scenario corpora

Kept for provenance. Nothing in the current analysis reads these files, and no
reported result should be traced to them without saying so explicitly.

The active corpora live in `data/`:

| File | Role |
|---|---|
| `scenarios_hidden_profile_candidates.json` | The 33-scenario corpus. Every current result comes from this. |
| `scenarios_remaining_v2b.json` | The 21-scenario subset used for the second collection of 28 July; needed as `--data` to re-run that report. |
| `scenarios_candidates_additive.json` | Knowledge-constant build of the 33 (`--additive-all`), used by `adoption_check.py`. |
| `scenarios_hidden_profile.json` | The frozen 10-scenario corpus the July pilots ran on. Retained because it can no longer be regenerated from source. |

## What is in here, and why it was superseded

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
