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
