"""Pydantic schemas for the assumption-propagation experiment.

The executable source of truth is ``data/scenarios_fixed.json``. The current
injection treatment is a categorical, unsupported seed-private assessment
(``seed_private_assessment``), not a fabricated record or confirmation. Its
canonical assumption records the exact target proposition. The legacy
``seed_local_evidence`` shape remains readable for old artifacts. Run-time
record schemas (agent rounds, judged rounds, conversation results) follow the
implementation blueprint.

Controlled vocabularies:
- SemanticType: factual / causal / framing.
- GlobalValidity: contradicted (factual/causal) or misaligned (framing) in
  the core design; supported/underdetermined reserved for diagnostics.
- AssumptionState: eight-way stance coding used by the judge and humans.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ANSWER_LETTERS = ("A", "B", "C", "D", "E")


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

class SemanticType(str, Enum):
    FACTUAL = "factual"
    CAUSAL = "causal"
    FRAMING = "framing"


class GlobalValidity(str, Enum):
    CONTRADICTED = "contradicted"      # core factual / causal variants
    MISALIGNED = "misaligned"          # core framing variants
    SUPPORTED = "supported"            # diagnostic only
    UNDERDETERMINED = "underdetermined"  # diagnostic only


class Topology(str, Enum):
    INDEPENDENT = "independent"
    LOOPED_CHAIN = "looped_chain"
    NEUTRAL_STAR = "neutral_star"
    FULLY_CONNECTED_MESH = "fully_connected_mesh"


class Condition(str, Enum):
    INJECTION = "injection"
    # Same injection, same frozen seed, but the corrector never receives the
    # decisive disproof. This is a CONDITION rather than a run-level switch so
    # that both arms are collected inside one run against the same seed
    # instances: round-0 behaviour varies enough between runs that a
    # between-run ablation contrast cannot be distinguished from run-to-run
    # variation.
    INJECTION_NO_CORRECTION = "injection_no_correction"
    NO_INJECTION_BASELINE = "no_injection_baseline"
    INDEPENDENT_CONTROL = "independent_control"


# Conditions that deliver the injection and therefore require a frozen seed.
INJECTION_CONDITIONS = ("injection", "injection_no_correction")


class AssumptionState(str, Enum):
    """Primary stance toward one canonical target assumption.

    The values intentionally distinguish neutral mention from endorsement
    and permit abstention when the response is genuinely ambiguous.  These
    are semantic labels; protocol exposure is derived from the run logs.
    """

    ABSENT = "absent"
    MENTION_ONLY = "mention_only"
    QUALIFIED_ACCEPTANCE = "qualified_acceptance"
    CATEGORICAL_ACCEPTANCE = "categorical_acceptance"
    CHALLENGE = "challenge"
    REJECTION = "rejection"
    SUPPORTED_CORRECTION = "supported_correction"
    UNCLEAR = "unclear"


# ---------------------------------------------------------------------------
# Scenario package (matches data/scenarios_fixed.json)
# ---------------------------------------------------------------------------

class CorrectiveEvidence(BaseModel):
    """Decisive correction with a predeclared staged-release round."""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=10)
    recipient_role: str = Field(description="e.g. 'corrector_agent' — the predeclared recipient.")
    available_round: int = Field(ge=0, le=3)


class SeedPrivateAssessment(BaseModel):
    """Seed-private assumption stated as an assessment, not fabricated evidence."""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=10)
    provenance: str = ""
    epistemic_marker: str = ""
    verification_gap: str = ""


class AssumptionVariant(BaseModel):
    """One semantic variant (factual/causal/framing) of a base scenario.

    The seed injection may be provided in one of two formats:
    - current: `seed_private_assessment` (a categorical private assumption), or
    - legacy: `seed_local_evidence` (a list of note strings).
    Use `seed_note_lines()` to read whichever is present.
    """

    model_config = ConfigDict(extra="ignore")  # tolerate __comment__ keys

    assumption_id: str
    semantic_type: SemanticType
    canonical_assumption: str = Field(
        min_length=10,
        description="The target inference the seed is invited to make.",
    )
    initial_provenance: str = ""
    explicitness: str = ""
    seed_local_evidence: Optional[list[str]] = None
    seed_private_assessment: Optional[SeedPrivateAssessment] = None
    # Framing assumptions do not deny a fact; they reweight its significance.
    # A factual or causal injection REPLACES the seed's honest finding, which
    # is a clean substitution. A framing injection that replaced it would
    # instead delete the fact from the group's information, confounding
    # semantic type with information loss. When this is set the seed retains
    # its honest finding AND receives the reframing, so the injection is
    # strictly additive and the group's available information is unchanged.
    reframes_baseline_finding: bool = False
    # Diagnostic control. When the seed holds both its honest finding and a
    # contradicting injected claim, the honest finding is normally stated
    # first. Position may carry weight independently of content or label, so
    # this reverses the order to measure it.
    injection_precedes_finding: bool = False
    corrective_evidence: CorrectiveEvidence
    global_validity: GlobalValidity
    favored_option: str

    @field_validator("favored_option")
    @classmethod
    def _valid_option(cls, v: str) -> str:
        if v not in ANSWER_LETTERS:
            raise ValueError(f"favored_option must be one of {ANSWER_LETTERS}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _has_a_seed_note(self) -> "AssumptionVariant":
        if not self.seed_local_evidence and self.seed_private_assessment is None:
            raise ValueError(
                f"{self.assumption_id}: needs seed_local_evidence or seed_private_assessment"
            )
        return self

    def seed_note_lines(self) -> list[str]:
        """The seed injection text, from whichever format is present."""
        if self.seed_private_assessment is not None:
            return [self.seed_private_assessment.content]
        return self.seed_local_evidence or []


class HiddenProfile(BaseModel):
    """Track B: decision-relevant facts split across the four agents.

    In the Track A design every agent can solve the task from
    ``common_context`` alone, so no agent ever has an epistemic reason to
    defer to a peer's unverifiable claim — which is why an injected
    assumption stops at the seed. A hidden profile removes that shortcut:
    each agent privately holds one fact, and only pooling them identifies
    the compliant option. Deference becomes rational, so a false claim in
    the pool is load-bearing rather than ignorable.

    Briefs are keyed by ROLE, never by agent id. Seed and corrector sit at
    different agent indices per topology (A0/A1 in ring and mesh, A1/A2 in
    the star), so agent-keyed briefs would hand the seed different private
    information in each topology and break frozen-seed replay.

    ``neutral_briefs`` is ordered: the nth entry goes to the nth neutral by
    ascending agent id, which is position-stable across topologies.
    """

    model_config = ConfigDict(extra="ignore")

    seed_brief: str = Field(min_length=10)
    corrector_brief: str = Field(min_length=10)
    neutral_briefs: list[str] = Field(min_length=2, max_length=2)
    shared_context_sufficient: bool = Field(
        description="Must be False: the hidden-profile property is that the shared "
        "context alone cannot identify the correct option. Authoring gate."
    )

    @field_validator("neutral_briefs")
    @classmethod
    def _neutral_briefs_nonempty(cls, v: list[str]) -> list[str]:
        for index, brief in enumerate(v):
            if len(brief.strip()) < 10:
                raise ValueError(f"neutral_briefs[{index}] is empty or too short")
        return v

    @model_validator(mode="after")
    def _must_be_a_hidden_profile(self) -> "HiddenProfile":
        if self.shared_context_sufficient:
            raise ValueError(
                "shared_context_sufficient must be False; a scenario solvable from the "
                "shared context alone is not a hidden profile and gives agents no "
                "reason to rely on peer claims"
            )
        return self

    def brief_for(
        self,
        *,
        is_seed: bool,
        is_corrector: bool,
        neutral_ordinal: Optional[int] = None,
    ) -> str:
        """The private brief for one role. Neutrals are ordered by agent id."""
        if is_seed and is_corrector:
            raise ValueError("seed and corrector must be different agents")
        if is_seed:
            return self.seed_brief
        if is_corrector:
            return self.corrector_brief
        if neutral_ordinal is None:
            raise ValueError(
                "neutral agents in a hidden-profile scenario require a neutral_ordinal"
            )
        if not 0 <= neutral_ordinal < len(self.neutral_briefs):
            raise ValueError(
                f"neutral_ordinal {neutral_ordinal} outside "
                f"0..{len(self.neutral_briefs) - 1}"
            )
        return self.neutral_briefs[neutral_ordinal]


class QualityChecklist(BaseModel):
    """Authoring-time quality gates recorded per scenario."""

    model_config = ConfigDict(extra="ignore")

    objective_ground_truth_exists: bool
    meaningful_causal_relation_supportable: bool
    explicit_goal_or_policy_anchors_framing: bool
    all_variants_affect_same_answer: bool
    baseline_solvable_without_specialist_knowledge: bool
    zero_verbatim_leakage_checked: bool


class Scenario(BaseModel):
    """One base scenario with its matched semantic variants."""

    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    domain: str
    benchmark_source: str = ""
    common_context: str = Field(min_length=100)
    question: str
    options: dict[str, str]
    correct_answer: str
    hard_constraints: list[str] = Field(min_length=1)
    optimization_rule: str
    explicit_goal_or_policy: str = Field(
        description="The authorized governing objective framing variants must violate."
    )
    neutral_control_note: str = Field(
        description="Innocuous seed-private note for the no-injection baseline, "
        "so baseline seeds also receive *a* private note (controls for having one)."
    )
    seed_confidence_level: Optional[int] = Field(default=None, ge=1, le=3)
    injected_epistemic_level: Optional[str] = None
    # Track B only. Absent => Track A: every agent solves from common_context.
    hidden_profile: Optional[HiddenProfile] = None
    assumption_variants: list[AssumptionVariant] = Field(min_length=1, max_length=3)
    quality_checklist: Optional[QualityChecklist] = None

    @property
    def track(self) -> str:
        """'hidden_profile' (Track B) or 'shared_context' (Track A)."""
        return "hidden_profile" if self.hidden_profile is not None else "shared_context"

    @field_validator("options")
    @classmethod
    def _five_options(cls, v: dict[str, str]) -> dict[str, str]:
        if tuple(sorted(v)) != ANSWER_LETTERS:
            raise ValueError(f"options must have exactly keys {ANSWER_LETTERS}, got {sorted(v)}")
        return v

    @field_validator("correct_answer")
    @classmethod
    def _valid_answer(cls, v: str) -> str:
        if v not in ANSWER_LETTERS:
            raise ValueError(f"correct_answer must be one of {ANSWER_LETTERS}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _favored_differs_from_correct(self) -> "Scenario":
        for variant in self.assumption_variants:
            if variant.favored_option == self.correct_answer:
                raise ValueError(
                    f"{variant.assumption_id}: favored_option equals correct_answer "
                    f"({self.correct_answer}); the assumption-favored distractor must be wrong."
                )
        return self


# ---------------------------------------------------------------------------
# Run-time records (experiment outputs)
# ---------------------------------------------------------------------------

class AgentOutput(BaseModel):
    """The only structured output agents produce: answer + short rationale."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    rationale: str = Field(min_length=1, max_length=1200)  # ~90 words plus slack

    @field_validator("answer")
    @classmethod
    def _valid_answer(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in ANSWER_LETTERS:
            raise ValueError(f"answer must be one of {ANSWER_LETTERS}, got {v!r}")
        return v

    @field_validator("rationale")
    @classmethod
    def _nonblank_rationale(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("rationale must contain non-whitespace text")
        return v


class SeedInstance(BaseModel):
    """One frozen seed response (Stage A output), replayed across topologies."""

    model_config = ConfigDict(extra="ignore")

    seed_instance_id: str
    scenario_id: str
    assumption_id: str
    seed_trial_index: int = Field(ge=0)
    injection_prompt_hash: str = ""
    answer: str
    rationale: str
    raw_response: str = ""
    generation_failed: bool = False
    generation_error_type: Optional[str] = None
    generation_error_message: Optional[str] = None
    # Nullable for backward compatibility: legacy seed logs predate the
    # rationale audit and must not be interpreted as audited-consistent.
    answer_rationale_inconsistent: Optional[bool] = None
    answer_rationale_audit_version: Optional[str] = None
    rationale_conclusion_letters: list[str] = Field(default_factory=list)
    # Labels below are filled by the judge/human pass, not at generation time.
    seed_assumption_state: Optional[AssumptionState] = None
    seed_decision_reliance: Optional[bool] = None
    seed_confidence_level: Optional[int] = Field(default=None, ge=1, le=3)
    included_in_core: bool = True  # every valid seed is retained (no filtering)

    @field_validator("answer")
    @classmethod
    def _valid_seed_answer(cls, v: str) -> str:
        v = v.strip().upper()
        if v and v not in ANSWER_LETTERS:
            raise ValueError(f"answer must be blank or one of {ANSWER_LETTERS}, got {v!r}")
        return v


class AgentRoundRecord(BaseModel):
    """Raw log for one agent in one round."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    stage: str = "propagation"
    scenario_id: str
    # Set whenever a variant is in play, including matched baselines where it
    # identifies the target/correction package; None only for variant-free runs.
    assumption_id: Optional[str] = None
    seed_instance_id: Optional[str] = None
    condition: Condition
    topology: Topology
    model: str
    temperature: Optional[float] = None
    seed_agent_id: Optional[int] = None
    corrector_agent_id: Optional[int] = None
    agent_id: int = Field(ge=0, le=3)
    round: int = Field(ge=0, le=3)
    injection_visible: bool = False
    # Nullable because this protocol flag was added after the first runs.
    # Missing legacy values are unknown, not evidence that no dossier was
    # visible.
    neutral_control_visible: Optional[bool] = None
    corrective_visible: bool = False
    received_from: list[int] = Field(default_factory=list)
    messages_received: int = 0
    peer_input_tokens: int = 0
    answer: Optional[str] = None  # None when parsing failed irrecoverably
    rationale: Optional[str] = None
    raw_response: str = ""
    parse_failed: bool = False
    parse_repaired: bool = False
    carried_forward: bool = False
    generation_failed: bool = False
    generation_error_type: Optional[str] = None
    generation_error_message: Optional[str] = None
    # Nullable for the same reason as on SeedInstance: only records carrying
    # an audit result belong in the consistency denominator.
    answer_rationale_inconsistent: Optional[bool] = None
    answer_rationale_audit_version: Optional[str] = None
    rationale_conclusion_letters: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def _valid_optional_answer(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        if v not in ANSWER_LETTERS:
            raise ValueError(f"answer must be one of {ANSWER_LETTERS}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _valid_failure_state(self) -> "AgentRoundRecord":
        if not self.parse_failed and self.answer is None:
            raise ValueError("a successfully parsed record requires an answer")
        if not self.parse_failed and not (self.rationale or "").strip():
            raise ValueError("a successfully parsed record requires a rationale")
        if self.generation_failed and not self.parse_failed:
            raise ValueError("generation_failed requires parse_failed=true")
        if self.parse_repaired and self.parse_failed:
            raise ValueError("parse_repaired and parse_failed cannot both be true")
        if self.carried_forward and (not self.parse_failed or self.round == 0):
            raise ValueError(
                "carried_forward requires a failed update in a round after round 0"
            )
        if self.parse_failed and self.answer is not None and not self.carried_forward:
            raise ValueError("a failed response may retain an answer only when carried forward")
        if self.parse_failed and self.answer_rationale_inconsistent is not None:
            raise ValueError("failed/carried responses are outside the rationale-audit denominator")
        return self


class JudgeDecision(BaseModel):
    """Semantic fields returned by the Judge LLM.

    Record identifiers, protocol metadata, and judge provenance are omitted
    deliberately.  The runner supplies those fields after validating this
    decision, which prevents the model from hallucinating join keys or using
    topology/role information while labeling a response.
    """

    model_config = ConfigDict(extra="forbid")

    assumption_state: AssumptionState
    decision_reliance: bool
    explicit_uncertainty_retained: bool
    verification_gap_retained: Optional[bool]
    provenance_retained: bool
    correction_present: bool
    corrective_evidence_used: bool
    confidence_level: Optional[int] = Field(ge=1, le=3)
    evidence_span: str
    uncertain: bool
    uncertainty_reason: str

    @model_validator(mode="after")
    def _semantic_consistency(self) -> "JudgeDecision":
        state = self.assumption_state
        accepting = {
            AssumptionState.QUALIFIED_ACCEPTANCE,
            AssumptionState.CATEGORICAL_ACCEPTANCE,
        }

        if state not in {AssumptionState.ABSENT, AssumptionState.UNCLEAR}:
            if not self.evidence_span.strip():
                raise ValueError("evidence_span is required for every determinate non-absent state")
        if state == AssumptionState.ABSENT:
            if self.evidence_span.strip():
                raise ValueError("absent requires an empty evidence_span")
            semantic_flags = (
                self.decision_reliance,
                self.explicit_uncertainty_retained,
                self.verification_gap_retained,
                self.provenance_retained,
                self.correction_present,
                self.corrective_evidence_used,
            )
            if any(semantic_flags):
                raise ValueError("absent requires all semantic component flags to be false")
        if state == AssumptionState.MENTION_ONLY and self.decision_reliance:
            raise ValueError("mention_only cannot have decision_reliance=true")
        if self.decision_reliance and state not in accepting | {AssumptionState.UNCLEAR}:
            raise ValueError(
                "decision_reliance=true requires an acceptance state or unclear"
            )
        if state == AssumptionState.UNCLEAR and not self.uncertain:
            raise ValueError("unclear requires uncertain=true")
        if self.uncertain and not self.uncertainty_reason.strip():
            raise ValueError("uncertain=true requires uncertainty_reason")
        if not self.uncertain and self.uncertainty_reason.strip():
            raise ValueError("uncertainty_reason must be empty when uncertain=false")

        if state in accepting:
            if self.confidence_level is None:
                raise ValueError("acceptance states require confidence_level")
        elif self.confidence_level is not None:
            raise ValueError("confidence_level is only defined for acceptance states")
        if state == AssumptionState.QUALIFIED_ACCEPTANCE:
            if not self.explicit_uncertainty_retained:
                raise ValueError(
                    "qualified_acceptance requires explicit_uncertainty_retained=true"
                )
            if self.confidence_level not in {1, 2}:
                raise ValueError("qualified_acceptance confidence_level must be 1 or 2")
        if state == AssumptionState.CATEGORICAL_ACCEPTANCE:
            if self.explicit_uncertainty_retained:
                raise ValueError(
                    "categorical_acceptance requires explicit_uncertainty_retained=false"
                )
            if self.confidence_level != 3:
                raise ValueError("categorical_acceptance confidence_level must be 3")
        if state == AssumptionState.SUPPORTED_CORRECTION:
            if not self.correction_present or not self.corrective_evidence_used:
                raise ValueError(
                    "supported_correction requires correction_present=true and "
                    "corrective_evidence_used=true"
                )
        if self.corrective_evidence_used and not self.correction_present:
            raise ValueError("corrective_evidence_used requires correction_present=true")
        return self

    def validate_evidence_span(self, judged_text: str) -> None:
        """Require the receipt to occur verbatim in the exact judged text."""
        if (
            self.assumption_state == AssumptionState.UNCLEAR
            and judged_text.strip()
            and not self.evidence_span.strip()
        ):
            raise ValueError(
                "unclear requires an evidence_span when the judged text is non-empty"
            )
        if self.evidence_span and self.evidence_span not in judged_text:
            raise ValueError("evidence_span must occur verbatim in the judged text")


class JudgedAgentRound(BaseModel):
    """Validated semantic annotation plus runner-attached provenance.

    The LLM emits :class:`JudgeDecision`; the runner copies the join keys and
    records the frozen codebook, prompt, model, and text source here.  Exposure
    and routing remain derived protocol data and are not part of this model.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    assumption_id: str
    agent_id: int = Field(ge=0, le=3)
    round: int = Field(ge=0, le=3)
    assumption_state: AssumptionState
    decision_reliance: bool
    explicit_uncertainty_retained: bool
    verification_gap_retained: Optional[bool]
    provenance_retained: bool
    correction_present: bool
    corrective_evidence_used: bool
    confidence_level: Optional[int] = Field(ge=1, le=3)
    judge_evidence_span: str
    judge_uncertain: bool
    judge_uncertainty_reason: str
    judge_codebook_version: str = Field(min_length=1)
    judge_prompt_version: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    judged_text_source: Literal[
        "rationale", "missing_rationale", "carried_forward_rationale"
    ]

    @model_validator(mode="after")
    def _semantic_consistency(self) -> "JudgedAgentRound":
        # Reuse the exact same invariants as the model-facing decision schema.
        JudgeDecision(
            assumption_state=self.assumption_state,
            decision_reliance=self.decision_reliance,
            explicit_uncertainty_retained=self.explicit_uncertainty_retained,
            verification_gap_retained=self.verification_gap_retained,
            provenance_retained=self.provenance_retained,
            correction_present=self.correction_present,
            corrective_evidence_used=self.corrective_evidence_used,
            confidence_level=self.confidence_level,
            evidence_span=self.judge_evidence_span,
            uncertain=self.judge_uncertain,
            uncertainty_reason=self.judge_uncertainty_reason,
        )
        return self

    @property
    def target_engaged(self) -> Optional[bool]:
        """Whether the text engages the target; unknown for ``unclear``.

        This is not acceptance, transmission, or causal exposure.  Challenge,
        rejection, and supported correction all engage the target while taking
        a non-accepting stance.
        """
        if self.assumption_state == AssumptionState.UNCLEAR:
            return None
        return self.assumption_state != AssumptionState.ABSENT

    def validate_evidence_span(self, judged_text: str) -> None:
        decision = JudgeDecision(
            assumption_state=self.assumption_state,
            decision_reliance=self.decision_reliance,
            explicit_uncertainty_retained=self.explicit_uncertainty_retained,
            verification_gap_retained=self.verification_gap_retained,
            provenance_retained=self.provenance_retained,
            correction_present=self.correction_present,
            corrective_evidence_used=self.corrective_evidence_used,
            confidence_level=self.confidence_level,
            evidence_span=self.judge_evidence_span,
            uncertain=self.judge_uncertain,
            uncertainty_reason=self.judge_uncertainty_reason,
        )
        decision.validate_evidence_span(judged_text)


class ConversationResult(BaseModel):
    """Conversation-level rollup after the final round."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    seed_instance_id: Optional[str] = None
    seed_assumption_state: Optional[AssumptionState] = None
    final_votes: dict[str, int]
    plurality_answer: Optional[str] = None  # None on tie
    no_consensus: bool
    consensus_strength: float = Field(ge=0.0, le=1.0)
    final_answer_correct: Optional[bool] = None  # None when no consensus
    final_acceptance_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    false_consensus: Optional[bool] = None
    correction_occurred: Optional[bool] = None
    first_adoption_round: Optional[int] = None
    first_correction_round: Optional[int] = None
