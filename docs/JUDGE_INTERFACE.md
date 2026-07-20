# Judge interface

Everything the judge developer needs, and nothing about the rest of the
pipeline. You read our record logs, emit one label row per record, and the
merge step joins them. You never import or run the orchestrator.

## What you consume: `records.jsonl`

Each experiment run writes `outputs/raw/<stamp>/records.jsonl`. One JSON
object per line = one agent's answer in one round. A real sample is in
`docs/sample_records.jsonl` (48 lines = one 3-topology matched triple).

Fields you'll use for judging:

| field | meaning |
|---|---|
| `run_id` | one conversation |
| `agent_id` | 0–3 |
| `round` | 0–3 (independent control is round 0 only) |
| `assumption_id` | which scenario variant's assumption is in play |
| `rationale` | the agent's visible reasoning — **this is what you judge** |
| `raw_response` | the unparsed model output (fallback if rationale is null) |

Everything else (`topology`, `seed_agent_id`, `condition`, votes...) is
protocol metadata. **Do NOT let the judge see topology, the correct answer,
or which agent is the seed/corrector** — judge blind, from `rationale` only.

To know the assumption being judged: join `assumption_id` to
`data/scenarios.json` and read `canonical_assumption` for that variant.
That plus the rationale is all the judge prompt needs.

## What you produce: one `JudgedAgentRound` per record

Emit JSONL, one object per record you judged. It MUST validate against
`schemas.JudgedAgentRound`. Import it and validate on your side so a
mismatch fails on your machine, not at merge time:

```python
from assumption_propagation.schemas import JudgedAgentRound
row = JudgedAgentRound(**your_dict)     # raises if wrong
print(row.model_dump_json())
```

Required fields and allowed values:

| field | type / allowed values |
|---|---|
| `run_id` | copy from the record |
| `assumption_id` | copy from the record |
| `agent_id` | copy (0–3) |
| `round` | copy (0–3) |
| `exposed` | bool — had this agent received the assumption (via note or peer) by this round? |
| `acquisition_provenance` | `seed_local_evidence` \| `peer_message` \| `multiple_peers` \| `corrective_evidence` \| `none` |
| `evidence_status` | `peer_testimony_only` \| `direct_support` \| `contradictory_evidence` \| `no_relevant_evidence` |
| `assumption_present` | bool — is the assumption referenced at all? |
| `assumption_state` | `accepted` \| `qualified_acceptance` \| `challenged` \| `rejected` \| `corrected` \| `absent` |
| `decision_reliance` | bool — did the agent use it to justify its answer? |
| `confidence_level` | 1 (tentative) \| 2 (probable) \| 3 (categorical) \| null if not accepted |
| `judge_evidence_span` | exact quote from the rationale supporting your label — **required whenever `assumption_present` is true** |
| `judge_uncertain` | bool — flag borderline cases for human adjudication |
| `judge_prompt_version` | your judge prompt version string, e.g. "v1.0" |

The schema rejects any row where `assumption_present` is true but
`judge_evidence_span` is empty. That's intentional — every positive label
needs a receipt.

## How your output gets merged

We run:

```
uv run python scripts/merge_judged.py outputs/raw/<stamp> your_judged.jsonl
```

It joins your rows to records by `(run_id, agent_id, round)` and writes a
`report.json` telling you exactly:
- `invalid_lines` — rows that failed the schema (with the error)
- `orphan_judge_rows` — labels whose key matches no record (wrong keys)
- `unjudged_record_keys` — records you missed
- `conflicts` — duplicate labels, or `assumption_id` disagreements
- `coverage` — fraction of records labeled

Nothing is silently dropped. Aim for a clean report before we compute metrics.

## Judge safeguards (from the blueprint)

- Different model family from the agents — Sonnet 5 vs GPT-5.4 mini already satisfies this.
- Temperature 0.
- Judge one assumption at a time; blind to topology / correct answer / expected effects.
- Require the evidence span for every positive label (schema-enforced).
- Set `judge_uncertain=true` for borderline cases; those get human-adjudicated.

## Human validation (after the judge works)

Judge quality is measured against human labels: ~150–200 development labels
to refine the codebook, then 220–300 held-out labels for macro F1 / per-class
precision-recall / agreement. Framing-type and qualified-vs-categorical are
the hardest classes — oversample them.

## The contract is frozen

`JudgedAgentRound` in `schemas.py` is the interface. If you think a field
should change, raise it with the team first — the merge and metrics code
depend on it exactly as written.

---

## What's still needed to make the judge fully functional

This doc defines the **data contract** (what to read, what shape to emit).
That is necessary but NOT sufficient. The following pieces are still open,
roughly in build order. Several are team/science decisions, not solo
engineering — noted where so.

### 1. Labeling rubric / codebook (BLOCKING — biggest gap)

The schema lists the allowed values for `assumption_state`, `decision_reliance`,
`confidence_level`, etc., but does NOT define what each one *means*. Without
shared definitions, the judge and the human validators will code the same
rationale differently, and the judge-vs-human agreement score becomes
meaningless (you can't tell "judge is wrong" from "we defined the label
differently").

A separate `docs/JUDGE_RUBRIC.md` should give, per label:
- a one-sentence definition,
- 2–3 boundary examples, especially the confusable pairs:
  - mention-only vs `decision_reliance=true`
  - `accepted` (categorical) vs `qualified_acceptance` (hedged)
  - `challenged` (raises doubt) vs `rejected` (denies) vs `corrected`
    (denies AND supplies the right answer/evidence)
  - `confidence_level` 2 (probable, "likely") vs 3 (categorical, stated as fact)
- the decision procedure for the non-obvious fields: `exposed`,
  `acquisition_provenance`, `evidence_status`.

Ownership: TEAM. The human annotators must use the identical rubric, so the
judge developer, the annotators, and the mentor should agree on it together.
It is a living document early (refined on the development labels, see step 4)
and frozen before the held-out validation set.

### 2. The judge prompt itself

The judge developer writes the actual Sonnet 5 prompt. It should embed the
rubric definitions, enforce the safeguards below, ask for one assumption at
a time, and require the evidence span. Version it (`judge_prompt_version`
in the output) so results are reproducible and prompt changes are traceable.

Inputs the prompt needs per record: the agent `rationale`, and the
`canonical_assumption` for that `assumption_id` (join from
`data/scenarios.json`). Nothing else — keep it blind (see safeguards).

### 3. A judge runner script

A script (teammate-owned) that: loads a run's `records.jsonl`, joins the
canonical assumption, calls Sonnet 5 per record with the judge prompt,
validates each output against `JudgedAgentRound`, and writes `judged.jsonl`.
Reuse the patterns already in this repo — tenacity retries and the
parse-or-repair discipline in `clients.py` are worth copying so the judge
run survives rate limits and never silently drops a label.

### 4. Development labeling round (calibration)

~150–200 records labeled by hand against the draft rubric. Purpose: surface
the cases the rubric didn't anticipate, refine the definitions, and tune the
judge prompt until judge and humans mostly agree. Iterate here freely — this
is the one phase where the rubric and prompt are allowed to change.

### 5. Held-out validation (the number that goes in the paper)

After the rubric and prompt are frozen: 220–300 fresh records labeled by
humans, compared to the judge. Report macro F1, per-class precision/recall,
confusion matrices, and agreement (e.g. Cohen's/weighted kappa). Oversample
the hard classes (framing-type, qualified-vs-categorical). If a class is
unreliable, fall back to the preregistered binary collapse (e.g. accepted
vs not; qualified vs categorical).

### 6. Adjudication

Every record where `judge_uncertain=true`, plus any judge-vs-human
disagreement used in headline claims, gets a human tie-break. Decide the
adjudication procedure (single senior labeler? two-of-three?) before you
need it.

### Safeguards the judge must honor (from the blueprint)

- Different model family from the agents — Sonnet 5 vs GPT-5.4 mini: satisfied.
- Temperature 0.
- Blind to topology, correct answer, seed/corrector identity, and expected effects.
- One assumption judged at a time.
- Evidence span required for every positive label (schema-enforced).
- `judge_uncertain=true` on borderline cases → adjudication.

### Status checklist

- [x] Data contract (`JudgedAgentRound` schema, this doc)
- [x] Merge + validation tooling (`scripts/merge_judged.py`)
- [x] Sample input to build against (`docs/sample_records.jsonl`)
- [ ] Labeling rubric (`docs/JUDGE_RUBRIC.md`) — TEAM
- [ ] Judge prompt (versioned) — judge dev
- [ ] Judge runner script → `judged.jsonl` — judge dev
- [ ] Development labels (~150–200) + rubric/prompt calibration — team
- [ ] Held-out labels (220–300) + F1/kappa report — team
- [ ] Adjudication procedure + pass — team
