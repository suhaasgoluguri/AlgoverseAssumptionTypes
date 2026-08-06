# Assumption Propagation in Multi-Agent LLM Systems

This repository studies how corrective evidence reaches a four-agent team
after one agent receives a categorical, unsupported private assessment. The
current primary protocol is `tool_v2`: no correction, verifier testimony,
the verifier's testimony plus an exact relay of the tool result, or direct
recipient delivery operationalized as a forced tool result for each
non-verifier. The confirmatory subject model is the frozen Azure deployment
`azure:gpt-5.4-mini`.

Confirmatory collection crosses anonymous peer headers (`Agent 2`) with assigned
headers (`Agent 2 (assigned: Borealis)`) across mesh, chain, and star. Assigned
headers expose which candidate the speaker reviewed, but not the private
finding. The primary correction-scope claim remains the clean mesh + assigned
cell; topology and attribution comparisons are secondary, and their interaction
is exploratory.

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

The direct arm is an actual two-step model/tool exchange: the first completion
is forced to request `read_case_record`, the backend returns the canonical JSON
as a `role=tool` message, and a second completion produces the answer. Prompt
hashes, canonical record and tool-result hashes, delivery route, execution
versus replay status, API attempts, finish reasons, token usage, and model
outputs are logged. A copied hidden-anchor trace has zero new calls/tokens and
is explicitly marked as replayed; it is never counted as a fresh measured-arm
tool execution.
For backward compatibility, schemas still call the designated verifier
`corrector_agent`; analysis reports it as the verifier/correction carrier and
keeps it out of ordinary-neutral outcomes.

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
- `src/assumption_propagation/judge.py`: blinded semantic Judge LLM pipeline.
- `src/assumption_propagation/metrics.py` and `analysis.py`: answer- and
  judge-level outcomes.
- `docs/JUDGE_RUBRIC.md`: shared Judge LLM/human annotation codebook.
- `docs/JUDGE_INTERFACE.md`: exact judge input, output, and merge boundary.

There are currently 33 executable scenarios with 99 matched variants: one
factual, causal, and framing treatment for every base scenario.
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
analysis decisions are in `docs/PREREGISTRATION.md`. Commit the preregistration
and analysis code before the first confirmatory API request.

`--scope-ladder` supports all three topologies. Mesh gives conversation-carried
correction equal immediate coverage and defines the primary scope comparison.
Chain and star intentionally limit who hears testimony or an exact relay within
the four-round horizon; direct delivery still reaches every non-verifier. The
analysis reports actual exposure and does not interpret sparse cells as pure
delivery-format effects.

`tool_v2` is the default when `--scope-ladder` is supplied. The older
prompt-only conditions remain reproducible with `--scope-protocol prompt_v1`
for historical comparisons. Prompt-v1 and tool-v2 runs use distinct condition
IDs and must never be pooled.

Convenience runners are available under `outputs/validation/`:

```powershell
# One scenario, no API cost. Add -IncludeControls to exercise both controls.
.\outputs\validation\run_tool_v2_mock_smoke.ps1

# One-scenario Azure compatibility gate; run this before the larger pilot.
.\outputs\validation\run_tool_v2_azure_smoke.ps1

# Six scenarios on Azure. Controls remain opt-in.
.\outputs\validation\run_tool_v2_azure_six.ps1
.\outputs\validation\run_tool_v2_azure_six.ps1 -IncludeControls

# Frozen 33-scenario gpt-5.4-mini confirmatory run. This refuses a dirty
# tracked worktree, a different deployment, inconsistent corpus bytes, or an
# incomplete arm x topology x attribution grid.
.\outputs\validation\run_tool_v2_azure_33.ps1
```

Each raw run freezes the exact scenario bytes and protocol source files in its
run directory, records their hashes and Git/model settings in
`run_manifest.json`, and refuses to append to an existing run directory.
Downstream judging and analysis use the verified run snapshot by default;
legacy runs without one require an explicit `--data` path.

Analysis reports two different populations explicitly. `aggregates.jsonl`
summarizes completed conversations, so injection rows include only seed trials
that produced a replayable round-0 answer. `seed_trials.jsonl` retains every
attempted seed generation, including parse/generation failures, and
`seed_trial_itt.jsonl` is the all-attempt seed-level ITT report: it includes
topology-specific outcome coverage, a missing-as-incorrect point estimate, and
missing-outcome bounds. The conditional aggregates are secondary
post-treatment summaries and must not replace that all-attempt report.

Run the judge plumbing without an API call, then merge and analyze it:

```bash
uv run python scripts/run_judge.py outputs/raw/<stamp> --client mock
uv run python scripts/merge_judged.py \
  outputs/raw/<stamp> outputs/judged/<stamp>/judged.jsonl --allow-mock
uv run python scripts/analyze_run.py \
  outputs/raw/<stamp> --judged outputs/judged/<stamp>/merged.jsonl --allow-mock
```

For a production Claude judge, copy `.env.example` to `.env`, configure
`ANTHROPIC_API_KEY` and the exact `ANTHROPIC_JUDGE_MODEL`, then use
`--client anthropic`. The default model identifier is `claude-sonnet-5`.

## Judge design

The Judge LLM is post-hoc and blinded. It receives one response, one canonical
target, and the semantic references needed to assess qualification and
correction. It never receives topology, condition, role, answer correctness,
or join identifiers. The runner attaches trusted keys after validation.

The current eight-state taxonomy is:

```text
absent | mention_only | qualified_acceptance | categorical_acceptance
challenge | rejection | supported_correction | unclear
```

Exact evidence spans are required for determinate non-absent labels. Exposure
is not guessed by the LLM: direct note visibility and routed target engagement,
acceptance, and correction are kept as distinct post-hoc fields. Partial row
coverage, unclear labels, and judge-uncertain labels are reported separately;
substantive judge-level metrics are withheld until coverage is complete and
uncertain cases are adjudicated.

The judge client must be selected explicitly. `mock` is a no-cost plumbing
client whose all-absent labels are rejected by merge and analysis unless
`--allow-mock` is deliberately supplied; never use them as research results.

`JUDGE_RUBRIC.md` is currently `v1.2-draft`: calibrate it on development data,
then freeze the rubric, prompt, schema, model snapshot, and adjudication plan
before opening the held-out validation set.

## Verification

```bash
uv run pytest -q
uv run ruff check src scripts tests
```
