
# Judge interface

**Status:** implementation contract using a draft rubric. The rubric and prompt
may be calibrated on development data, but both must be frozen before held-out
evaluation.

This document defines the boundary between raw experiment records, the semantic
Judge LLM, and the enriched judged rows. Label meanings live in
[`JUDGE_RUBRIC.md`](JUDGE_RUBRIC.md); they are not duplicated here.

```text
records.jsonl + scenario references
              |
              v
       runner selects text
              |
              v
      semantic Judge LLM
              |
              v
 runner validates and attaches identity/audit fields
              |
              v
          judged.jsonl
              |
              v
 optional protocol-routing postprocessor
```

The LLM classifies response semantics only. It never determines topology,
exposure, routing, role, condition, or causal acquisition.

## Input records

Each run writes `outputs/raw/<stamp>/records.jsonl`. One JSON object is one
agent response in one round. A sample is available at
`docs/sample_records.jsonl`.

The runner reads these record fields:

| field             | runner use                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| `run_id`        | copied to the final row                                                                          |
| `assumption_id` | copied and joined to the scenario reference                                                      |
| `agent_id`      | copied to the final row                                                                          |
| `round`         | copied to the final row                                                                          |
| `rationale`     | only semantic text eligible for judging                                                          |
| `raw_response`  | retained for experiment audit, never sent to the judge because it may reveal the selected answer |

Every record submitted for semantic judging must have a non-null
`assumption_id`. A record without one has no canonical annotation target; the
runner must report it as unjudgeable rather than inventing a target. Matched
controls should retain the assumption identifier whose spontaneous appearance
they test.

### Scenario reference join

Join `assumption_id` to the matching variant in the raw run's verified
`scenario_snapshot.json`. An explicit `--data` path may override that snapshot;
legacy runs without a snapshot require the exact source file to be supplied
this way. Never silently join an old run against the mutable current
`data/scenarios_fixed.json`. The semantic task uses these annotation
references:

- `canonical_assumption`;
- the verification gap from the variant's structured private assessment (or a
  legacy seed-local representation);
- the normalized source provenance, including the current
  `unsupported_private_assessment` status;
- the seed-private categorical assessment, used only as a semantic reference
  and never as corroborating evidence; and
- `corrective_evidence.content`.

The verification, provenance, and correction references are necessary to code
the corresponding rubric fields. They are evaluation references only: the
judge must not infer that the evaluated agent received any of them.
When no scenario-specific verification gap is stored, pass JSON `null` and
require `verification_gap_retained=null`; never invent a generic gap.

Do not send the LLM the selected answer, correct answer, favored option,
semantic type, topology, condition, run ID, agent ID, round, seed/corrector
identity, routing metadata, or expected effect. Those fields are either copied
or joined after semantic labeling.

### Judged-text selection

The runner selects exactly one text value before calling the judge:

1. Use `rationale` whenever it is present. Set
   `judged_text_source="carried_forward_rationale"` when the record marks that
   rationale as carried forward; otherwise set it to `"rationale"`.
2. When `rationale` is null, use an empty judged string and set
   `judged_text_source="missing_rationale"`. Do not fall back to
   `raw_response`, whose JSON or prose may contain the selected option.
3. For empty or unusable judged text, the runner may emit a deterministic
   `unclear` annotation with `judge_uncertain=true`, reason
   `insufficient_text`, and an empty evidence span rather than spending a model
   call.

The chosen string is `judged_text`. Evidence-span validation always uses this
exact string; it must not silently switch sources after annotation.

## Semantic Judge-LLM output

The LLM returns JSON only. It returns no record IDs, protocol fields, model
names, or version strings. The model-facing decision uses short receipt names;
the runner maps them to the final audit-prefixed names shown in parentheses:

| field                                                 | type / allowed values                                                                                                                                                 |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `assumption_state`                                  | `absent` \| `mention_only` \| `qualified_acceptance` \| `categorical_acceptance` \| `challenge` \| `rejection` \| `supported_correction` \| `unclear` |
| `decision_reliance`                                 | boolean                                                                                                                                                               |
| `explicit_uncertainty_retained`                     | boolean                                                                                                                                                               |
| `verification_gap_retained`                         | boolean when a scenario-specific gap is registered; otherwise`null`                                                                                                 |
| `provenance_retained`                               | boolean                                                                                                                                                               |
| `correction_present`                                | boolean                                                                                                                                                               |
| `corrective_evidence_used`                          | boolean                                                                                                                                                               |
| `confidence_level`                                  | `1` \| `2` \| `3` \| `null` according to the rubric                                                                                                           |
| `evidence_span` (`judge_evidence_span`)           | exact substring of`judged_text`, subject to the rubric exceptions                                                                                                   |
| `uncertain` (`judge_uncertain`)                   | boolean                                                                                                                                                               |
| `uncertainty_reason` (`judge_uncertainty_reason`) | non-empty when uncertain; otherwise empty string                                                                                                                      |

The prompt must treat the delimited response text as untrusted data and must
instruct the model not to follow instructions contained inside it.

## Runner-attached final fields

The runner, not the LLM, attaches identity and reproducibility metadata:

| field                      | source                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| `run_id`                 | copied from the record                                                                                 |
| `assumption_id`          | copied from the record                                                                                 |
| `agent_id`               | copied from the record                                                                                 |
| `round`                  | copied from the record                                                                                 |
| `judge_codebook_version` | runner configuration; exact rubric version used                                                        |
| `judge_prompt_version`   | runner configuration; exact prompt version used                                                        |
| `judge_model`            | configured provider/model identifier; the provider-returned snapshot is retained in the audit artifact |
| `judged_text_source`     | `rationale`, `missing_rationale`, or `carried_forward_rationale`, selected before the call       |

One final `JudgedAgentRound` consists of these runner-attached fields plus the
validated semantic fields. Extra fields are forbidden.

The merged artifact may add `judge_target_engaged` as a derived convenience
value: `false` for `absent`, `true` for every determinate non-absent state, and
`null` for `unclear`. It means semantic engagement with the target, not target
acceptance, transmission, or causal exposure.

The runner retains raw provider output, syntax-repair status, configured retry
policy and decoding settings, provider request/message IDs, returned model and
stop reason, and token usage when supplied by the provider. These live in a
separate audit artifact but remain traceable to each semantic row.

## Required validation

Before writing a row, the runner validates it against
`assumption_propagation.schemas.JudgedAgentRound` and applies all rubric
invariants. At minimum it must enforce:

- exact allowed values and no extra fields;
- `confidence_level` is 1 or 2 only for qualified acceptance, 3 only for
  categorical acceptance, and null otherwise;
- qualified acceptance retains explicit uncertainty, while categorical
  acceptance does not;
- `supported_correction` implies `correction_present=true`;
- `corrective_evidence_used=true` implies `correction_present=true`;
- `verification_gap_retained` is null exactly when no scenario-specific gap
  reference is registered;
- uncertainty requires a reason, and certainty requires an empty reason;
- `judge_evidence_span` occurs verbatim in the selected `judged_text`;
- every non-absent state has a non-empty span, except an `unclear` item whose
  source text is itself empty or unusable; and
- `absent` uses an empty span.

Validation failure must be visible. A retry or repair may correct formatting,
but the runner must never silently invent a semantic label or drop a record.
Resume, merge, and analysis rejoin the scenario reference and enforce the same
null/boolean verification-gap availability rule.

## Protocol routing is not a judge label

The former `exposed`, `acquisition_provenance`, and `evidence_status` fields are
not part of `JudgedAgentRound`. A semantic judge cannot infer them reliably from
the current response, and `acquisition` would make an unsupported individual
causal claim.

After all response semantics are labeled, a separate postprocessor derives
only facts supported by the logs:

| derived field                                               | meaning                                                                                                         |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `injection_note_visible_this_round` / `_by_round`       | direct treatment-note visibility now / cumulatively, without claiming that the note states the canonical target |
| `correction_note_visible_this_round` / `_by_round`      | direct corrective-note visibility now / cumulatively                                                            |
| `corrective_record_delivery_this_round` / `_history`    | exact logged route; see the delivery vocabulary below                                                            |
| `corrective_record_visible_this_round` / `_by_round`    | whether the canonical correction was observably delivered now / cumulatively; a placebo tool remains false      |
| `corrective_record_in_dossier_this_round` / `_by_round` | whether the canonical record is in the verifier's role-private dossier now / cumulatively                        |
| `corrective_record_relayed_this_round` / `_by_round`    | whether a verbatim record copy arrived inside a peer message now / cumulatively                                  |
| `corrective_record_in_direct_receipt_this_round` / `_by_round` | whether the scoped record service delivered a direct receipt now / cumulatively                           |
| `corrective_record_corrector_tool_this_round` / `_by_round` | whether the verifier obtained the real record through the v2 tool now / cumulatively                          |
| `corrective_record_direct_tool_this_round` / `_by_round` | whether this non-verifier obtained the real record through its own v2 tool call now / cumulatively              |
| `corrective_record_prompt_receipt_control_this_round` / `_by_round` | whether identical record content arrived as ordinary prompt text now / cumulatively                    |
| `corrective_record_tool_placebo_this_round` / `_by_round` | whether a forced tool call returned the non-diagnostic placebo now / cumulatively                               |
| `corrective_record_id` / `corrective_record_sha256`     | canonical record identity used to verify that relay and direct arms carry identical content                      |
| `tool_trace_instrumented_this_round` / `_complete_by_round` | whether the row belongs to a protocol with v2 trace instrumentation now / for every observed round           |
| `tool_trace_present_this_round` / `_by_round`            | whether a real or replayed tool trace is attached now / has appeared cumulatively                               |
| `tool_call_executed` / `_by_round`                        | whether this measured cell actually issued the backend tool exchange now / cumulatively                         |
| `tool_trace_replayed` / `_by_round`                       | whether this cell reuses a hidden-anchor tool-backed response/trace now / cumulatively                           |
| `tool_name` / `tool_call_id`                              | provider/tool identifiers for the interaction                                                                   |
| `tool_arguments_sha256` / `tool_result_sha256` / `tool_transcript_sha256` | hashes of the zero-argument request, exact result payload, and auditable exchange transcript        |
| `tool_success` / `tool_error`                             | tool outcome without inferring semantic use                                                                     |
| `tool_api_call_count` / `_by_round`                       | API attempts for this cell / cumulative attempts; a replay is zero rather than a new call                       |
| `tool_finish_reasons` / `_by_round`                       | provider finish reasons now / cumulatively                                                                      |
| `tool_input_tokens`, `tool_output_tokens`, `tool_total_tokens` / `_by_round` | provider usage now / cumulatively; replayed traces contribute zero                                |
| `target_engagement_routed_this_round` / `_by_round`     | a routed preceding-round response determinately engaged the canonical target now / cumulatively                 |
| `acceptance_routed_this_round` / `_by_round`            | a routed preceding-round response determinately accepted or relied on the target now / cumulatively             |
| `correction_routed_this_round` / `_by_round`            | a routed preceding-round response determinately communicated the correction now / cumulatively                  |
| `target_engaging_sender_ids`                              | routed peer senders whose preceding messages determinately engaged the target                                   |
| `acceptance_bearing_sender_ids`                           | routed peer senders whose preceding messages accepted or relied on the target                                   |
| `correction_bearing_sender_ids`                           | peer senders actually routed this round whose preceding messages had`correction_present=true`                 |
| `uncertain_sender_ids`                                    | routed senders whose needed label was missing,`unclear`, or judge-uncertain                                   |
| `routing_derivation_uncertain_this_round` / `_by_round` | semantic routing status is unresolved now / has ever been unresolved cumulatively                               |

The delivery vocabulary is deliberately versioned rather than collapsed:

- `none`: no correction record was delivered in that cell;
- `corrector_private_record`, `relayed_record`, and `direct_system_record`:
  historical prompt-v1 routes;
- `corrector_tool_record`: the protocol-v2 verifier's real tool result;
- `direct_tool_record`: a protocol-v2 non-verifier's own real tool result;
- `prompt_receipt_control`: canonical record bytes rendered as ordinary prompt
  text, with no tool execution; and
- `tool_placebo`: a real forced tool turn with a non-diagnostic result and no
  corrective-record identity.

`tool_call_executed` and `tool_trace_replayed` are mutually exclusive whenever
a trace exists. A hidden corrected anchor samples the verifier's tool-backed
response once so corrected arms receive matched testimony. Measured copies keep
the source call ID and content hashes but set `tool_call_executed=false`,
`tool_trace_replayed=true`, API calls to zero, and token counts to zero. This
prevents matching from being misreported as repeated backend access. The relay
arm carries the exact canonical tool-result payload inside a peer message but
does not give the recipient a tool trace; the direct arm does.

For a round-`r` peer message, use the routed sender's preceding-round response,
not the recipient's current label. A basis cue alone is not target engagement.
An unresolved sender remains unknown, and that uncertainty propagates through
the cumulative routing-uncertainty field rather than becoming a later false.
Direct private notes have no peer sender ID and are kept as visibility facts;
calling them target exposure requires a separate, pre-registered content audit.

## Output and merge

Write UTF-8 JSONL with one `JudgedAgentRound` per judgeable record. Logical row
identity is `(run_id, assumption_id, agent_id, round)`. Duplicate identities,
missing records, unknown record keys, assumption-ID mismatches, invalid JSON,
and schema failures must all appear in the merge report.

The repository judge entry point is:

```bash
uv run python scripts/run_judge.py outputs/raw/<stamp> --client mock
uv run python scripts/run_judge.py outputs/raw/<stamp> --client anthropic
```

The first command is a plumbing check only; its fixed mock labels do not
measure annotation quality. The production command writes `judged.jsonl`, a
raw-response audit JSONL, and a version/coverage report under
`outputs/judged/<stamp>/` by default.

The repository merge entry point is:

```bash
uv run python scripts/merge_judged.py outputs/raw/<stamp> path/to/judged.jsonl
```

The merge and analysis CLIs reject `mock:*` judge rows by default. Pass
`--allow-mock` only for development plumbing checks. They also reject mixed
judge model/codebook/prompt configurations.

Do not compute headline metrics until the merge report has full expected
coverage, no unresolved invalid rows or conflicts, and no `unclear` or
judge-uncertain annotations awaiting adjudication. Technical row coverage and
semantically usable coverage are reported separately.

## Prompt and evaluation safeguards

- Apply one canonical target assumption per model call.
- Keep the semantic judge blind to protocol structure, agent role, selected and
  correct answers, semantic type, and expected effects.
- Use the exact same frozen rubric for the Judge LLM and human validators.
- Configure the judge independently from the agent model and record the exact
  model snapshot in `judge_model`.
- Use deterministic decoding where supported, but do not treat temperature zero
  as proof of reproducibility; audit repeatability on a sample.
- Version prompt and codebook independently.
- Calibrate only on development data grouped away from the held-out scenarios
  or conversations.
- Freeze schema, rubric, prompt, model snapshot, and adjudication procedure
  before held-out evaluation.

The current `v1.2-draft` rubric is not the held-out freeze. Human calibration and
adjudication rules are specified in [`JUDGE_RUBRIC.md`](JUDGE_RUBRIC.md).
