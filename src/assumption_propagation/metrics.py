"""Outcome metrics.

Two layers, matching what data exists when:

1. ANSWER-LEVEL metrics (available immediately from AgentRoundRecords):
   accuracy, favored-distractor adoption, answer trajectories, flips.
   These use only the multiple-choice answers — no judge required.

2. JUDGE-LEVEL metrics (available after the post-hoc semantic judge has
   labeled records as JudgedAgentRound): acceptance/correction rates,
   false consensus, first-adoption and first-correction rounds.

Both operate on records for ONE conversation unless stated otherwise;
cross-conversation analysis (regressions, clustering by scenario) lives in
analysis/, not here.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .schemas import AgentRoundRecord, AssumptionState, JudgedAgentRound

FINAL_ROUND = 3
ACCEPTING_STATES = {
    AssumptionState.QUALIFIED_ACCEPTANCE,
    AssumptionState.CATEGORICAL_ACCEPTANCE,
}
RESISTANCE_STATES = {
    AssumptionState.CHALLENGE,
    AssumptionState.REJECTION,
    AssumptionState.SUPPORTED_CORRECTION,
}


# ---------------------------------------------------------------------------
# Answer-level metrics (no judge needed)
# ---------------------------------------------------------------------------

def answer_trajectory(records: Sequence[AgentRoundRecord], agent_id: int) -> list[Optional[str]]:
    """[round0, round1, round2, round3] answers for one agent."""
    by_round = {r.round: r.answer for r in records if r.agent_id == agent_id}
    return [by_round.get(i) for i in range(FINAL_ROUND + 1)]


def final_individual_accuracy(records: Sequence[AgentRoundRecord], correct_answer: str) -> float:
    """Share of agents whose round-3 answer is correct."""
    finals = [r for r in records if r.round == FINAL_ROUND]
    return sum(r.answer == correct_answer for r in finals) / len(finals)


def favored_adoption_by_round(
    records: Sequence[AgentRoundRecord], favored_option: str
) -> list[float]:
    """Per round: share of agents answering the assumption-favored option.

    The cheapest propagation signal — visible before any judging.
    """
    shares = []
    for round_index in range(FINAL_ROUND + 1):
        round_records = [r for r in records if r.round == round_index]
        shares.append(sum(r.answer == favored_option for r in round_records) / len(round_records))
    return shares


def flip_count(records: Sequence[AgentRoundRecord], agent_id: int) -> int:
    """How many times one agent changed its answer across rounds."""
    answers = [a for a in answer_trajectory(records, agent_id) if a is not None]
    return sum(1 for prev, cur in zip(answers, answers[1:]) if prev != cur)


def seed_recanted(
    records: Sequence[AgentRoundRecord], seed_agent_id: int, favored_option: str
) -> Optional[bool]:
    """Did the seed start on the favored option and abandon it by round 3?

    None when the seed never started on the favored option (not applicable).
    In the ring, abandonment at round 3 is the loop-closure self-correction.
    """
    trajectory = answer_trajectory(records, seed_agent_id)
    if trajectory[0] != favored_option:
        return None
    return trajectory[FINAL_ROUND] != favored_option


# ---------------------------------------------------------------------------
# Judge-level metrics (require JudgedAgentRound labels)
# ---------------------------------------------------------------------------

def _downstream(judged: Sequence[JudgedAgentRound], seed_agent_id: int):
    return [j for j in judged if j.agent_id != seed_agent_id]


def final_acceptance_rate(
    judged: Sequence[JudgedAgentRound], seed_agent_id: int, final_round: int | None = None
) -> Optional[float]:
    """Share of non-seed agents accepting in the conversation's final round."""
    if not judged:
        return None
    final_round = max(j.round for j in judged) if final_round is None else final_round
    finals = [j for j in _downstream(judged, seed_agent_id) if j.round == final_round]
    if not finals:
        return None
    return sum(j.assumption_state in ACCEPTING_STATES for j in finals) / len(finals)


def false_consensus(
    judged: Sequence[JudgedAgentRound], final_round: int | None = None
) -> Optional[bool]:
    """At least 3 of 4 agents accept/rely on the (contradicted) assumption
    in the final round."""
    if not judged:
        return None
    final_round = max(j.round for j in judged) if final_round is None else final_round
    finals = [j for j in judged if j.round == final_round]
    return sum(
        j.assumption_state in ACCEPTING_STATES or j.decision_reliance for j in finals
    ) >= 3


def first_adoption_round(
    judged: Sequence[JudgedAgentRound], seed_agent_id: int
) -> Optional[int]:
    """Earliest round any non-seed agent accepts. None if never."""
    rounds = [
        j.round for j in _downstream(judged, seed_agent_id)
        if j.assumption_state in ACCEPTING_STATES
    ]
    return min(rounds) if rounds else None


def first_correction_round(judged: Sequence[JudgedAgentRound]) -> Optional[int]:
    """Earliest supported correction, distinct from challenge or rejection."""
    rounds = [
        j.round
        for j in judged
        if j.assumption_state == AssumptionState.SUPPORTED_CORRECTION
    ]
    return min(rounds) if rounds else None


def first_resistance_round(judged: Sequence[JudgedAgentRound]) -> Optional[int]:
    """Earliest challenge, rejection, or supported correction."""
    rounds = [j.round for j in judged if j.assumption_state in RESISTANCE_STATES]
    return min(rounds) if rounds else None


def correction_occurred(judged: Sequence[JudgedAgentRound]) -> bool:
    """Anyone ever supplied a supported correction."""
    return any(
        j.assumption_state == AssumptionState.SUPPORTED_CORRECTION for j in judged
    )


def correction_latency(
    judged: Sequence[JudgedAgentRound], seed_agent_id: int
) -> Optional[int]:
    """Rounds between first downstream adoption and first correction.

    Negative means correction beat adoption. None when either never occurs.
    """
    adopted = first_adoption_round(judged, seed_agent_id)
    corrected = first_correction_round(judged)
    if adopted is None or corrected is None:
        return None
    return corrected - adopted
