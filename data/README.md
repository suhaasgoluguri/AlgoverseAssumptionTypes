# Scenario corpora

The scenarios here are authored, not model-generated. Model-generated text does
appear elsewhere in this repository: each run's
`analysis/disclosure_slice.jsonl.gz` carries the agents' rationales verbatim as
the models produced them, served through Azure. Those are AI-generated
throughout and are published as measurement data for replication, not as
human-written text.

## What is here

| File | Scenarios | sha256 | Role |
|---|---|---|---|
| `scenarios_hidden_profile_candidates.json` | 33 | `880cf414d5d6` | The corpus every reported result comes from, and the default for `run_pilot.py`. |
| `scenarios_no_authority.json` | 33 | `af6ce27f6b1e` | Ablation over the same 33: the pooling rule's authority sentence is removed, so no clause tells an agent that an assigned reviewer's report is authoritative and sufficient. Behind the with/without-authority comparison. |
| `scenarios_no_transfer_clause.json` | 33 | `e8398c545c51` | Ablation over the same 33: the authority sentence is kept but its transfer clause ("whether you hold it directly or hear it from that reviewer") is dropped, so authority no longer visibly survives relay. Prepared but never run — no reported result uses it and no run manifest names it. |
| `archive/scenarios_fixed.json` | 20 | — | An earlier corpus, kept only because `scripts/audit_scenarios.py` and `tests/test_scenarios_fixed.py` audit it. Its rules do not apply to the 33-scenario corpus, which fails them by construction. Nothing reported uses it. |

Each run additionally freezes the exact corpus bytes it used in its own
`scenario_snapshot.json`, and records the hash in `run_manifest.json`. Analysis
reads that snapshot, not these files, so a reported number is tied to the bytes
that produced it.

## Rebuilding the corpus

`scripts/build_hidden_profile_scenarios.py` regenerates the live corpus
byte-for-byte with no arguments:

```bash
uv run python scripts/build_hidden_profile_scenarios.py
shasum -a 256 data/scenarios_hidden_profile_candidates.json   # 880cf414d5d6...
```

The builder also carries flags for presentation variants (`--additive-all`,
`--swap-labels`, `--injection-first`, `--neutral-authority`). No reported
result uses them; they build against the current 33 scenarios.

## How the corpus constrains a scenario

`check_specs` in the builder enforces the properties the design depends on.
The one worth knowing when reading a scenario: no single agent may be able to
identify the correct option alone. Dropping the trapped option must not leave
the correct answer as the best remaining choice, or an agent could solve the
task without pooling and the hidden-profile structure would not bind.

## A construction property to read results against

The injected claim states that the contested item is the only open constraint
on the option it concerns. Without that clause a seed can rationally defer, and
the manipulation does not take hold. It is what makes seed adoption near-total,
so **seed adoption is a property of this construction rather than a measure of
model credulity**, and should be read that way wherever it appears.
