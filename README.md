# Assumption Propagation in Multi-Agent LLM Systems

This repository studies how explicitly introduced factual, causal, and
framing assumptions are adopted, qualified, transmitted, resisted, and
corrected under different four-agent communication topologies.

The core comparison replays the same frozen seed response through a directed
loop, a neutral star, and a fully connected mesh. Each conversation has round
0 plus three synchronous update rounds. A designated seed receives the target
injection; a distinct corrector receives decisive counterevidence; all message
routing and model outputs are logged for post-hoc analysis.

## Repository map

- `data/scenarios.json`: executable scenario packages and matched assumption
  variants.
- `src/assumption_propagation/orchestrator.py`: frozen-seed generation and
  synchronous propagation.
- `src/assumption_propagation/topologies.py`: graph routing and role placement.
- `src/assumption_propagation/judge.py`: blinded semantic Judge LLM pipeline.
- `src/assumption_propagation/metrics.py` and `analysis.py`: answer- and
  judge-level outcomes.
- `docs/JUDGE_RUBRIC.md`: shared Judge LLM/human annotation codebook.
- `docs/JUDGE_INTERFACE.md`: exact judge input, output, and merge boundary.

There are currently 19 executable scenarios with 57 matched variants. The
older `log_001` authoring draft is explicitly marked `legacy` and is retained
for reference rather than silently treated as executable data.

## Quick start

Install the locked environment:

```bash
uv sync
```

Run a no-cost mock pilot:

```bash
uv run python scripts/run_pilot.py --client mock --scenario travel_001
```

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

`JUDGE_RUBRIC.md` is currently `v1.1-draft`: calibrate it on development data,
then freeze the rubric, prompt, schema, model snapshot, and adjudication plan
before opening the held-out validation set.

## Verification

```bash
uv run pytest -q
uv run ruff check src scripts tests
```
