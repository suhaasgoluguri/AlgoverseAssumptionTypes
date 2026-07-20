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
- AssumptionState: six-way stance coding used by the judge.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

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
    """Six-way stance coding for seed and downstream responses."""

    ACCEPTED = "accepted"
    QUALIFIED_ACCEPTANCE = "qualified_acceptance"
    CHALLENGED = "challenged"
    REJECTED = "rejected"
    CORRECTED = "corrected"
    ABSENT = "absent"  # absent / ignored


class AcquisitionProvenance(str, Enum):
    SEED_LOCAL_EVIDENCE = "seed_local_evidence"
    PEER_MESSAGE = "peer_message"
    MULTIPLE_PEERS = "multiple_peers"
    CORRECTIVE_EVIDENCE = "corrective_evidence"
    NONE = "none"


class EvidenceStatus(str, Enum):
    PEER_TESTIMONY_ONLY = "peer_testimony_only"
    DIRECT_SUPPORT = "direct_support"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    NO_RELEVANT_EVIDENCE = "no_relevant_evidence"


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


class JudgedAgentRound(BaseModel):
    """CONTRACT WITH THE JUDGE (teammate's Sonnet 5 judge must emit this).

    The judge labels only what cannot be read off the logs: stance, decision
    reliance, expressed confidence, plus an exact supporting evidence span.
    Protocol facts (topology, exposure routing, correct answer) come from
    experiment metadata, never from the judge.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    assumption_id: str
    agent_id: int = Field(ge=0, le=3)
    round: int = Field(ge=0, le=3)
    exposed: bool
    acquisition_provenance: AcquisitionProvenance
    evidence_status: EvidenceStatus
    assumption_present: bool
    assumption_state: AssumptionState
    decision_reliance: bool
    confidence_level: Optional[int] = Field(default=None, ge=1, le=3)  # null when not accepted
    judge_evidence_span: str = ""
    judge_uncertain: bool = False
    judge_prompt_version: str

    @model_validator(mode="after")
    def _span_required_for_positive_labels(self) -> "JudgedAgentRound":
        if self.assumption_present and not self.judge_evidence_span.strip():
            raise ValueError(
                "judge_evidence_span is required whenever assumption_present is true."
            )
        return self


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
