"""Pydantic schemas for the assumption-propagation experiment.

The source of truth for the scenario format is data/scenarios.json (team
design). The injection treatment is a seed-private evidence note
(`seed_local_evidence`): the seed agent privately receives evidence that
invites the target inference, and the canonical assumption records what
that inference is. Run-time record schemas (agent rounds, judged rounds,
conversation results) follow the implementation blueprint.

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
    NO_INJECTION_BASELINE = "no_injection_baseline"
    INDEPENDENT_CONTROL = "independent_control"


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
# Scenario package (matches data/scenarios.json)
# ---------------------------------------------------------------------------

class CorrectiveEvidence(BaseModel):
    """Decisive correction item with predeclared placement."""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=10)
    recipient_role: str = Field(description="e.g. 'corrector_agent' — the predeclared recipient.")
    available_round: int = Field(ge=0, le=3)


class AssumptionVariant(BaseModel):
    """One semantic variant (factual/causal/framing) of a base scenario.

    `seed_local_evidence` is the injection: seed-private note(s) that invite
    the target inference recorded in `canonical_assumption`.
    """

    model_config = ConfigDict(extra="ignore")  # tolerate __comment__ keys

    assumption_id: str
    semantic_type: SemanticType
    canonical_assumption: str = Field(
        min_length=10,
        description="The target inference the seed is invited to make.",
    )
    initial_provenance: str = Field(
        description="e.g. 'inferential_evidence_gap'.",
    )
    explicitness: str = Field(
        description="e.g. 'explicit_presentation_inferential_origin'.",
    )
    seed_local_evidence: list[str] = Field(
        min_length=1,
        description="Seed-private note(s) shown only to the seed agent (the injection).",
    )
    corrective_evidence: CorrectiveEvidence
    global_validity: GlobalValidity
    favored_option: str

    @field_validator("favored_option")
    @classmethod
    def _valid_option(cls, v: str) -> str:
        if v not in ANSWER_LETTERS:
            raise ValueError(f"favored_option must be one of {ANSWER_LETTERS}, got {v!r}")
        return v


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
    seed_confidence_level: int = Field(ge=1, le=3)
    assumption_variants: list[AssumptionVariant] = Field(min_length=1, max_length=3)
    quality_checklist: Optional[QualityChecklist] = None

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
    rationale: str = Field(max_length=1200)  # ~90 words plus slack

    @field_validator("answer")
    @classmethod
    def _valid_answer(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in ANSWER_LETTERS:
            raise ValueError(f"answer must be one of {ANSWER_LETTERS}, got {v!r}")
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
    # Labels below are filled by the judge/human pass, not at generation time.
    seed_assumption_state: Optional[AssumptionState] = None
    seed_decision_reliance: Optional[bool] = None
    seed_confidence_level: Optional[int] = Field(default=None, ge=1, le=3)
    included_in_core: bool = True  # every valid seed is retained (no filtering)


class AgentRoundRecord(BaseModel):
    """Raw log for one agent in one round."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    stage: str = "propagation"
    scenario_id: str
    assumption_id: Optional[str] = None  # set whenever a variant is in play (incl. baselines, where it identifies whose corrective evidence was present); None only if no variant
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
