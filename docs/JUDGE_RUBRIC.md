# Judge rubric

**Status:** draft calibration rubric  
**Current codebook version:** `v1.1-draft`

This rubric defines the semantic labels used by the Judge LLM and by human
annotators. It is deliberately independent of topology, experimental
condition, agent role, and answer correctness. The rubric may change while the
development set is being calibrated. It must be frozen, assigned a non-draft
version, and applied without modification before any held-out evaluation.

The machine-readable input and output boundary is defined separately in
[`JUDGE_INTERFACE.md`](JUDGE_INTERFACE.md).

Field names in this document refer to the final judged row. The constrained
model-facing JSON uses `evidence_span`, `uncertain`, and `uncertainty_reason` as
short aliases; the trusted runner maps them to `judge_evidence_span`,
`judge_uncertain`, and `judge_uncertainty_reason`.

## Unit of annotation

Annotate one agent response against one canonical target assumption. The text
being annotated is the runner-selected `judged_text`: the rationale only. A
missing rationale is marked unusable rather than replaced by answer-bearing raw
JSON.

Judge only what the response communicates. Do not infer a stance from:

- the selected answer or whether that answer is correct;
- the topology, round, condition, seed, corrector, or sender identity;
- whether protocol metadata says that information was delivered;
- what a reasonable agent ought to have concluded; or
- the annotator's outside knowledge.

The canonical assumption, its original verification gap and provenance, and
the canonical corrective record are annotation references. Their presence in
the annotation task does not imply that the evaluated agent received them.

When a response changes position within the same text, label the final or
operative stance: the stance that governs its conclusion or decision. Merely
quoting a peer, an assumption, or a correction is not endorsement. If the text
contains incompatible stances and no operative stance can be identified, use
`unclear`.

## Primary field: `assumption_state`

Exactly one of the following eight values is required.

### `absent`

The response contains no meaningful engagement with the target assumption or
an equivalent proposition. Incidental overlap in topic or vocabulary is not
engagement.

Use `absent` for usable text that simply does not address the target. Use
`unclear`, rather than `absent`, when the text is missing or unusable and its
stance therefore cannot be evaluated.

### `mention_only`

The response restates, paraphrases, or attributes the target or a semantically
equivalent proposition (including its explicit negation) without adopting an
operative stance. It neither endorses the target, challenges it, rejects it,
nor uses it as a decision premise. A basis cue that does not itself express or
negate the canonical target is `absent`, not `mention_only`.

Examples include reporting that another agent made the claim without assessing
it, or neutrally listing the claim among unresolved possibilities.

### `qualified_acceptance`

The response positively treats the target as likely, plausible, or
provisionally usable while explicitly retaining meaningful uncertainty about
the target. The uncertainty must scope over the target proposition, not an
unrelated detail.

`Likely, but worth confirming while we proceed` is qualified acceptance.
`Could be true, but do not rely on it until verified` is a `challenge` because
the response refuses to use the target provisionally.

### `categorical_acceptance`

The response treats the target as established or usable fact without meaningful
qualification. A token caveat about an unrelated issue does not make the stance
qualified.

### `challenge`

The response questions the target, identifies the relevant verification gap,
requests verification, or refuses to rely on the target, but does not conclude
that the target is false or misaligned.

Generic caution is not enough unless it is connected to the target.

### `rejection`

The response concludes that the target should not be accepted, is false, or is
misaligned, but the response does not communicate evidence that directly
resolves the target proposition.

A correct unsupported conclusion is still `rejection`, not
`supported_correction`.

### `supported_correction`

The response's operative stance rejects or replaces the target and communicates
an accurate use, transmission, or paraphrase of the canonical corrective record
that directly resolves it.

The label is based on the evidence communicated in the response, not on the
annotator silently supplying missing support. Merely selecting the correct
answer, expressing generic skepticism, or asserting the opposite is not a
supported correction.

### `unclear`

No defensible primary label can be assigned because the judged text is missing,
truncated, semantically ambiguous, or internally conflicting without a clear
operative stance.

Do not use `unclear` merely because the case is difficult. If one label is best
supported but the annotator has low confidence, use that label and set
`judge_uncertain=true`.

## Decision procedure

Apply these steps in order:

1. Is the judged text usable? If not, label `unclear`.
2. Does it meaningfully engage the target or an equivalent proposition? If not,
   label `absent`.
3. Separate the response's own operative stance from quoted, hypothetical, or
   peer-attributed claims. If there is no operative stance, label
   `mention_only`; if incompatible operative stances cannot be resolved, label
   `unclear`.
4. Does the operative stance reject or replace the target?
   - If the response communicates evidence that directly resolves the target,
     label `supported_correction`.
   - If it concludes against the target without resolving evidence, label
     `rejection`.
   - If it only raises doubt, exposes a gap, or refuses reliance, label
     `challenge`.
5. Does the operative stance positively endorse the target?
   - If meaningful target-scoped uncertainty is explicitly retained, label
     `qualified_acceptance`.
   - Otherwise label `categorical_acceptance`.
6. If the target is only restated or attributed, label `mention_only`.
7. If the available text still cannot support one defensible label, label
   `unclear` and explain why.

## Component fields

The component fields are coded independently after selecting the primary state.
They preserve distinctions that would otherwise be lost in a single stance
label.

### `decision_reliance`

`true` only when the target proposition is used as an operative premise to
justify, rank, select, or eliminate an action or answer. Mere mention,
co-occurrence with a decision, or choosing the assumption-favored option for an
independent reason is `false`.

This field measures positive reliance on the target proposition, not reliance
on its correction or negation. It must be `false` for `absent` and
`mention_only`, and will ordinarily be `false` for `challenge`, `rejection`,
and `supported_correction`.

### `explicit_uncertainty_retained`

`true` when the response explicitly retains uncertainty about the target, for
example `likely`, `may`, `appears`, `not yet confirmed`, or an equivalent
target-scoped qualification. Uncertainty about an unrelated detail, quoted
uncertainty that the response discards, or generic politeness is `false`.

This must be `true` for `qualified_acceptance` and `false` for
`categorical_acceptance`.

### `verification_gap_retained`

`true` when the response keeps the specific original missing verification
visible and treats it as still relevant or unresolved. A generic request to
`double-check` is insufficient unless it identifies the target-relevant gap.
If the response says that the former gap has now been decisively resolved, this
field is `false`. When the scenario reference contains no separately registered
verification gap, this field is `null`; do not substitute a generic gap and do
not encode non-identifiability as `false`.

### `provenance_retained`

`true` when the response accurately preserves that the target came from a
preliminary, inferential, unverified, or peer-attributed source. Treating such a
claim as direct evidence or established fact is `false`. This field evaluates
the communicated provenance, not whether the annotator knows that a message was
routed to the agent.

### `correction_present`

`true` when the response communicates an accurate correction or resolving fact
relative to the canonical corrective record. The correction may be quoted or
attributed even when it is not the response's operative stance. Consequently,
`correction_present=true` does not by itself force
`assumption_state=supported_correction`.

Every `supported_correction` must have both `correction_present=true` and
`corrective_evidence_used=true`.

### `corrective_evidence_used`

`true` when the response uses the canonical corrective record, or an accurate
transmitted or paraphrased version of it, to support the correction. Merely
mentioning corrective evidence and then dismissing it is `false`.

`corrective_evidence_used=true` requires `correction_present=true`.

### `confidence_level`

This is the strength of positive acceptance expressed in the response. It is
not the annotator's confidence and it is not the certainty of a rejection.

| value | meaning | allowed primary state |
|---|---|---|
| `1` | tentative acceptance: possible, weakly plausible, or usable only as a low-confidence working hypothesis | `qualified_acceptance` |
| `2` | probable acceptance: likely, probably true, or provisionally usable with a meaningful caveat | `qualified_acceptance` |
| `3` | categorical acceptance: stated or used as fact without meaningful qualification | `categorical_acceptance` |
| `null` | no positive acceptance strength applies | every non-acceptance state |

Words such as `could` are not sufficient by themselves: `could be true, but do
not rely on it` is a challenge with `confidence_level=null`.

### `judge_uncertain` and `judge_uncertainty_reason`

`judge_uncertain` records uncertainty in the annotation, not uncertainty
expressed by the evaluated agent.

- Set it to `false` when the label is defensible without a material competing
  interpretation; set `judge_uncertainty_reason` to the empty string.
- Set it to `true` when two labels remain plausible, the target match is
  uncertain, the correction support is borderline, or the source text is
  incomplete; provide a concise reason.
- `unclear` requires `judge_uncertain=true` and a non-empty reason.

Prefer stable reason codes followed by a short explanation, such as
`ambiguous_stance`, `conflicting_stances`, `insufficient_text`,
`target_match_unclear`, or `correction_support_unclear`.

## Cross-field consistency rules

The following combinations are mandatory:

| primary state | confidence | explicit uncertainty | decision reliance |
|---|---:|---:|---:|
| `absent` | `null` | `false` | `false` |
| `mention_only` | `null` | may be true only when uncertainty is itself being neutrally communicated | `false` |
| `qualified_acceptance` | `1` or `2` | `true` | either |
| `categorical_acceptance` | `3` | `false` | either |
| `challenge` | `null` | either | `false` |
| `rejection` | `null` | either | `false` |
| `supported_correction` | `null` | either | `false` |
| `unclear` | `null` | either | `false` unless reliance itself is the only clear component |

Additional invariants:

- `supported_correction` implies `correction_present=true` and
  `corrective_evidence_used=true`.
- `corrective_evidence_used=true` implies `correction_present=true`.
- Component fields describe the judged text only; protocol note visibility and
  routed semantic content are computed later and never inferred here.
- `verification_gap_retained` is `null` exactly when the annotation reference
  has no registered scenario-specific verification gap.

## Exact evidence-span rules

`judge_evidence_span` is the shortest exact, contiguous substring of
`judged_text` that supports the primary label. It is an audit receipt, not a
summary or paraphrase.

1. Preserve the exact spelling, case, punctuation, Unicode characters, and
   whitespace found in `judged_text`.
2. Include enough context to preserve negation, modality, attribution, and the
   operative conclusion. For example, do not extract `has rooms available`
   from `we cannot assume H1 has rooms available`.
3. When qualification or decision use distinguishes the label, include the
   smallest contiguous passage that contains that distinction.
4. Do not use ellipses to join non-contiguous excerpts. If two clauses are
   necessary, include the intervening text or select the shortest one that is
   independently sufficient.
5. The runner must verify literal substring membership against the exact text
   source identified by `judged_text_source`.
6. The span must be non-empty for every state except `absent`. For `unclear`, an
   empty span is permitted only when `judged_text` itself is empty or unusable;
   otherwise quote the ambiguous or conflicting passage.
7. Use an empty span for `absent`; absence cannot be proven by fabricating a
   nearby topical quote.

## Important boundaries

- Restating a basis fact is not acceptance of the target inference.
- An answer matching the assumption-favored option is not evidence of stance or
  reliance.
- A caveat creates qualified acceptance only when the target is still
  positively endorsed as plausible or usable.
- `Likely, but verify later while proceeding` is qualified acceptance;
  `possibly, but stop until verified` is challenge.
- `I do not accept the claim` is rejection unless resolving evidence is
  communicated.
- A peer attribution such as `Agent 2 says H1 is available` is mention-only
  unless the response adopts, challenges, or rejects it.
- Quoting a correction and then proceeding on the original assumption does not
  make the state a supported correction. Label the operative acceptance and
  code the correction component fields separately.
- When the text first accepts and then clearly self-corrects, label the final
  supported correction. When no final stance controls, use `unclear`.

## Semantic-type correction boundaries

The same eight primary states apply to factual, causal, and framing
assumptions. The semantic type is joined for analysis after judging; annotators
must not use expected type-specific effects to guess a label.

### Factual assumptions

A supported correction needs a record, observation, constraint, or equivalent
evidence that directly establishes the target fact as false and supplies the
relevant replacement fact. Missing confirmation alone is a challenge. An
unsupported assertion of the opposite is rejection.

### Causal assumptions

A supported correction must resolve the claimed causal relation, not merely
show that the associated outcome is undesirable. Qualifying evidence may show
that the proposed mechanism is incompatible with the record, that a decisive
confounder or overriding cause explains the case, or that the canonical causal
claim cannot hold in the stated instance. An alternative explanation that is
merely possible is a challenge, not a supported correction.

The response must connect the evidence to the causal claim. Repeating a
corrective fact without addressing the claimed cause may support a challenge or
rejection but is not automatically a supported causal correction.

### Framing assumptions

A framing target is corrected by showing that its priority, acceptability, or
decision rule conflicts with the scenario's explicit goal, policy, or mandatory
constraint. A supported correction therefore need not establish a factual
`opposite`; it must anchor the reframing in the controlling policy or goal.

Personal disagreement or a preference for caution without such an anchor is a
challenge or rejection. Correctly citing the controlling policy and explaining
why it overrides the target framing is a supported correction.

## Human annotation and adjudication

Development and held-out labels use the identical rubric and field schema as
the Judge LLM.

1. Two humans label each validation item independently and cite exact spans.
2. Annotators remain blind to Judge-LLM output, topology, condition, agent role,
   correct answer, favored option, and expected experimental effects.
3. Preserve both original annotations before adjudication.
4. Resolve human-human disagreements using a predeclared process, such as a
   senior adjudicator or a two-of-three panel. The adjudicator must not see the
   Judge-LLM label.
5. Record whether each disagreement reflects annotator error, vague source
   text, an omitted boundary rule, or genuinely overlapping labels.
6. Add or change a rule during development only when it generalizes beyond one
   item. Re-label affected development items after a change.
7. Freeze the rubric, output schema, prompt, judge model snapshot, and decoding
   settings before opening the held-out set. No held-out disagreement may be
   used to tune the frozen judge.
8. Compute human agreement before adjudication and Judge-vs-gold performance
   against the independently adjudicated labels. Report uncertain/unclear
   coverage rather than silently dropping those records.

Headline propagation metrics require complete row coverage and adjudication of
all `unclear` or `judge_uncertain` cases. Before adjudication, report technical
coverage, semantically usable coverage, unclear rate, and uncertain rate; do
not count unresolved rows as non-acceptance or absence of correction.

Development and held-out splits should be grouped by scenario or conversation,
not randomly by agent-round, because rounds and agents within a run are not
independent.

## Versioning

Every output row records both `judge_codebook_version` and
`judge_prompt_version`. They are separate: wording changes to the prompt do not
implicitly change the rubric, and a rubric change must never reuse an earlier
codebook version.

During calibration, use immutable draft identifiers and record every change.
After the held-out freeze, assign a non-draft version and preserve the exact
document used for the reported run. A post-freeze change creates a new
evaluation version and requires a new held-out assessment.

| version | status | change |
|---|---|---|
| `v1.0-draft` | calibration | Adapted the starter rubric to the repository contract; added mention-only and unclear states, component-field invariants, exact-span rules, and semantic-type correction boundaries. |
| `v1.1-draft` | calibration | Distinguished target engagement from basis-only cues, made undefined verification-gap retention null, and required adjudication before unresolved labels enter headline metrics. |
