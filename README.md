# When Testimony Is Not Enough: Corrective-Evidence Delivery in Multi-Agent LLM Teams

This repository contains the code and analysis for *When Testimony Is Not
Enough: Corrective-Evidence Delivery in Multi-Agent LLM Teams*. Four LLM
agents solve a hidden-profile task in which one agent's private dossier
carries an unsupported false assessment and a second agent receives an
external record refuting it. We vary how that record reaches the two
remaining agents across six delivery arms, and independently whether peer
message headers name each speaker's assigned option. The primary model is
`azure:gpt-5.4-mini`; `azure:deepseek-v4-pro` is a cross-model check reported
separately and never pooled with the primary run.

## Vocabulary

The code predates the paper's terminology. The mapping:

| Repository | Paper |
|---|---|
| `corrector_agent` (schemas, records) | the verifier |
| ordinary-neutral | ordinary agents |
| scope ladder / correction scope | delivery arms |
| hidden corrected anchor | the shared verifier response replayed across arms |

`tool_v2` and `prompt_v1` are internal protocol names. Everything reported in
the paper is `tool_v2`; `prompt_v1` is an earlier prompt-only protocol kept
only so historical runs remain analyzable, and it is never pooled with
`tool_v2`. See *Reproducibility notes* at the end.

All reported runs use `PROMPT_VERSION = v3.2` (rationale emitted before
answer), recorded in each run's `run_manifest.json`. This is unrelated to the
`prompt_v1` protocol name despite the shared word.

## Protocol

The primary comparison uses a fully connected mesh and four synchronous
rounds. The seed's assessment persists from round 0. After one clean
propagation round, a distinct verifier is forced to call the zero-argument
`read_case_record` tool in round 1. Ordinary peers first respond to the
verifier's testimony, an exact relayed tool payload, or their own direct tool
result in round 2; those routes persist through round 3. A hidden corrected
anchor samples the verifier once, and all corrected measured arms replay that
same verifier response. The no-correction arm shares the matched pre-release
history but not the corrected response. This prevents route effects from being
confounded with resampling the conversation carrier.

Confirmatory collection crosses anonymous peer headers (`Agent 2`) with assigned
headers (`Agent 2 (assigned: Borealis)`) across mesh, chain, and star. Assigned
headers expose which candidate the speaker reviewed, but not the private
finding. The primary correction-scope claim remains the clean mesh + assigned
cell; topology and attribution comparisons are secondary, and their interaction
is exploratory.

The direct arm is an actual two-step model/tool exchange: the first completion
is forced to request `read_case_record`, the backend returns the canonical JSON
as a `role=tool` message, and a second completion produces the answer. Prompt
hashes, canonical record and tool-result hashes, delivery route, execution
versus replay status, API attempts, finish reasons, token usage, and model
outputs are logged. A copied hidden-anchor trace has zero new calls/tokens and
is explicitly marked as replayed; it is never counted as a fresh measured-arm
tool execution.

## Repository map

- `data/scenarios_hidden_profile_candidates.json`: executable hidden-profile
  scenario packages and matched
  assumption variants.
- `src/assumption_propagation/orchestrator.py`: frozen-seed generation and
  synchronous propagation.
- `src/assumption_propagation/evidence.py`: deterministic canonical record and
  placebo payloads.
- `src/assumption_propagation/clients.py`: ordinary completion and forced-tool
  client flows.
- `src/assumption_propagation/topologies.py`: graph routing and role placement.
- `src/assumption_propagation/metrics.py` and `analysis.py`: answer-level
  outcomes, within one conversation and across a run.

There are 33 executable scenarios with 66 confirmatory variants (factual and
causal), plus 33 framing variants retained for labelled follow-up work and
excluded from the confirmatory run.
The primary evidence-scope protocol uses factual and causal variants only;
`--scope-ladder` therefore excludes framing by default. Framing remains in the
corpus for explicitly labelled follow-up work and can be requested with
`--variants framing`.

## Quick start

Install the locked environment:

```bash
uv sync
```

Run a no-cost ordinary pilot:

```bash
uv run python scripts/run_pilot.py --client mock --scenario vendor_001
```

Run the four primary `tool_v2` arms:

```bash
uv run python scripts/run_pilot.py --client mock --scenario vendor_001 \
  --variants factual --scope-ladder --scope-protocol tool_v2 --n-seeds 1 \
  --topologies looped_chain neutral_star fully_connected_mesh \
  --attribution both --skip-baselines --skip-independent --rounds 4 \
  --tag scope_smoke
```

Add the two mechanism controls. They are included in the confirmatory run but
remain outside the four-arm primary contrast family:

```bash
uv run python scripts/run_pilot.py --client mock --scenario vendor_001 \
  --variants factual --scope-ladder --scope-protocol tool_v2 --scope-controls \
  --topologies looped_chain neutral_star fully_connected_mesh \
  --attribution both --n-seeds 1 --skip-baselines --skip-independent --rounds 4 \
  --tag scope_smoke_controls
```

The four primary arms are kept separate from the controls:

| role in analysis | condition | ordinary agents receive from round 2 onward |
|---|---|---|
| primary | `injection_tool_none` | no corrective evidence |
| primary | `injection_tool_testimony` | the matched verifier answer/rationale only |
| primary | `injection_tool_relay` | that testimony plus the exact canonical tool-result payload inside the verifier message |
| primary | `injection_tool_direct` | that testimony plus a forced personal `read_case_record` call and `role=tool` result |
| mechanism control | `injection_prompt_receipt_control` | the canonical payload inserted as ordinary prompt text, with no tool call |
| mechanism control | `injection_tool_placebo` | the same forced tool interaction but a non-diagnostic payload; verifier testimony remains present |

The primary paired contrasts are testimony minus none, relay minus testimony,
and direct recipient delivery minus relay. Direct tool minus prompt receipt
tests whether the tool channel adds anything beyond matched direct content
presentation; placebo minus testimony tests the extra tool-turn mechanics; and
direct minus placebo tests receipt of diagnostic rather than non-diagnostic
tool output. A direct-versus-relay effect alone is not called a tool-invocation
effect.

The frozen 33-scenario collection command, exact sample size, exclusions, and
analysis decisions are fixed by `scripts/run_tool_v2_azure_33.ps1`, which
refuses to start unless the corpus hash, deployment, and worktree all match.
Commit the protocol and analysis code before the first confirmatory API
request.

`--scope-ladder` supports all three topologies. Mesh gives conversation-carried
correction equal immediate coverage and defines the primary scope comparison.
Chain and star intentionally limit who hears testimony or an exact relay within
the four-round horizon; direct delivery still reaches every non-verifier. The
analysis reports actual exposure and does not interpret sparse cells as pure
delivery-format effects.

The run launchers are **PowerShell** (`.ps1`) and assume a Windows-style
`.venv\Scripts\python.exe`. On Linux or macOS either run them under
`pwsh`, or read the argument list out of the file and invoke
`scripts/run_pilot.py` directly -- the launchers add preflight and
post-run gates, not behaviour.

Run launchers live in `scripts/`. Each one wraps `run_pilot.py` with a fixed
argument set, then gates the output it produced and chains the two analysis
scripts, so a malformed run fails loudly instead of being analyzed:

```powershell
# Full protocol against the mock client: no API key, no cost. Verifies the
# forced-tool path executes and its traces replay, then analyzes the result.
# Add -IncludeControls to exercise the prompt-receipt and placebo controls.
.\scripts\run_tool_v2_mock_smoke.ps1

# Frozen 33-scenario gpt-5.4-mini confirmatory run. Refuses to start on a
# dirty tracked worktree, a deployment other than gpt-5.4-mini, or a corpus
# whose canonical hash has moved.
.\scripts\run_tool_v2_azure_33.ps1

# No-authority ablation: same protocol and scenarios, against the corpus whose
# pooling rule drops the authority sentence. Mesh only, and the deployment is a
# parameter rather than a pin. Reported separately, never pooled with the
# confirmatory run -- it is a different corpus with a different hash.
.\scripts\run_tool_v2_azure_noauthority.ps1
```

Each raw run freezes the exact scenario bytes and protocol source files in its
run directory, records their hashes and Git/model settings in
`run_manifest.json`, and refuses to append to an existing run directory.
Analysis uses the verified run snapshot by default; legacy runs without one
require an explicit `--data` path.

Run data under `outputs/raw/` is not tracked in this repository: it is large
and machine-specific, and every run is reproducible from the corpus in `data/`
plus its run manifest. It is distributed alongside the paper instead.
`scripts/run_tool_v2_mock_smoke.ps1` produces a full run locally at no cost if
you want to see the record format.

Analysis reports two different populations explicitly. `aggregates.jsonl`
summarizes completed conversations, so injection rows include only seed trials
that produced a replayable round-0 answer. `seed_trials.jsonl` retains every
attempted seed generation, including parse/generation failures, and
`seed_trial_itt.jsonl` is the all-attempt seed-level ITT report: it includes
topology-specific outcome coverage, a missing-as-incorrect point estimate, and
missing-outcome bounds. The conditional aggregates are secondary
post-treatment summaries and must not replace that all-attempt report.

## Reproducing the paper's tables and figures

The analysis artifacts behind every number in Section 4 are committed under
`outputs/raw/`. Three runs are used:

| Directory | Role |
|---|---|
| `20260810_merged_confirmatory_scope33_gpt54mini_v32` | primary, `azure:gpt-5.4-mini` |
| `20260807_193819_azure_confirmatory_scope33_deepseekv4pro_v32` | cross-model check, `azure:deepseek-v4-pro` |
| `20260817_merge_scenarios_noauthority_gpt54mini` | no-authority ablation |

```bash
uv sync
uv run python docs/figures/make_tables.py          # all nine tab_*.tex
uv run python docs/figures/make_fig_results.py     # fig_results.{pdf,png}
uv run python docs/figures/make_fig_attribution.py # fig_attribution.{pdf,png}
uv run python docs/figures/make_fig_setup_rounds.py
```

**Read this before running `make_tables.py`: one input is not shipped.** It
reads `records.jsonl` from the primary and cross-model runs -- 310 MB and
297 MB of full per-agent-round transcripts -- and opens them at module scope,
so without those two files it raises `FileNotFoundError` immediately and
writes no tables at all. Request them if you need to rerun the table
generator; they are the only missing input.

Everything else the generators read **is** committed: `results.jsonl`,
`scenario_snapshot.json`, `analysis/conversations.jsonl`,
`analysis/aggregates.jsonl`, and the `statistics_*.json` files. The three
`make_fig_*.py` scripts do not touch `records.jsonl` and run against the
committed artifacts as-is.

With `records.jsonl` present, all nine `.tex` tables regenerate
byte-identically to the committed copies. Outputs are written next to the
generators in `docs/figures/`, overwriting them.

A further audit covers the manipulation check and the response dynamics that
the statistics files do not:

```bash
uv run python scripts/audit_disclosure_and_dynamics.py <run-dir> \
    --json <run-dir>/analysis/audit_disclosure_and_dynamics.json
```

It reports the anonymity-leak rate (whether an `anonymous`-condition message
discloses the speaker's own assignment), deadlock rate by arm, accuracy by
round, seed recantation by arm, parse failures by cell, and the per-scenario
spread of the ordinary-agent outcome. Committed output for both runs is under
each run's `analysis/`.

This audit does **not** need `records.jsonl`. It falls back to
`analysis/disclosure_slice.jsonl.gz`, which is committed for both confirmatory
runs: a verbatim projection of the transcripts down to the ten fields the
disclosure analyses read, ~5 MB gz against ~300 MB. The numbers are identical
either way, and where the transcripts are present the projection can be
re-derived and diffed:

```bash
uv run python scripts/make_disclosure_slice.py <run-dir> --verify
```

The finer-grained leak measurement runs off the same slice:

```bash
uv run python scripts/measure_anonymity_leakage.py <run-dir> \
    --json <run-dir>/analysis/anonymity_leakage.json
uv run python scripts/measure_anonymity_leakage.py --selftest   # 21 cases
```

It adds disclosure-by-elimination, assignment attributed to another agent,
per-round and per-topology splits, and `--sample N` to draw flagged clauses
for hand-labelling. Its headline measure scopes a first-person standing claim
to the speaker's own entity by clause, requiring no other candidate to
intervene; precision is 99.0% (Wilson 95% CI 96.4-99.7) on 200 sampled
clauses labelled by hand. Read the `assigned` column as a reference rather
than a comparable rate -- that mode's header names the assignment, so
effective disclosure there is 100% by construction.

`analysis/statistics_secondary.json` is regenerated rather than original; it
is produced by

```bash
uv run python scripts/analyze_statistics.py <run-dir> \
    --by-attribution --by-topology --json <run-dir>/analysis/statistics_secondary.json
```

Each run's `run_manifest.json` records the sha256 of `results.jsonl` under
`status.observed_files`, and `.gitattributes` pins `outputs/raw/**` to `-text`
so those hashes survive checkout on any platform.

## Verification

```bash
uv run pytest -q
uv run ruff check src scripts tests
```

## Reproducibility notes

`tool_v2` is the default when `--scope-ladder` is supplied, and is the only
protocol any reported result uses. The older prompt-only conditions remain
runnable with `--scope-protocol prompt_v1` so historical runs stay
analyzable; prompt-v1 and tool-v2 use distinct condition IDs and must never
be pooled.

For backward compatibility, schemas still call the designated verifier
`corrector_agent`; analysis reports it as the verifier and keeps it out of
ordinary-agent outcomes.

## License

MIT, in `LICENSE`, covering the code and the data alike. The run artifacts
under `outputs/raw/` include model-generated text: each run's
`analysis/disclosure_slice.jsonl.gz` carries the agents' rationales verbatim
as the models produced them, served through Azure. They are published as
measurement data for replication and are AI-generated throughout.

Both providers disclaim ownership of that output. The Microsoft Product Terms
state that "Output Content is Customer Data. Microsoft does not own Customer's
Output Content", and the DeepSeek model license that "DeepSeek claims no
rights in the Output You generate using the Model." Neither imposes an
attribution or pass-through requirement on output, so no third-party terms
travel with this corpus.
