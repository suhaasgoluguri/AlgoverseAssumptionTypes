"""Outcome metrics.

Answer-level metrics, read straight off AgentRoundRecords: accuracy,
favored-distractor adoption, answer trajectories, flips. They use only the
multiple-choice answers, so every one of them is available the moment a run
finishes.

These operate on records for ONE conversation unless stated otherwise;
cross-conversation analysis (regressions, clustering by scenario) lives in
analysis/, not here.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .schemas import AgentRoundRecord

FINAL_ROUND = 3
# ---------------------------------------------------------------------------
# Answer-level metrics
# ---------------------------------------------------------------------------

def answer_trajectory(records: Sequence[AgentRoundRecord], agent_id: int) -> list[Optional[str]]:
    """[round0, round1, round2, round3] answers for one agent."""
    by_round = {r.round: r.answer for r in records if r.agent_id == agent_id}
    return [by_round.get(i) for i in range(FINAL_ROUND + 1)]


def final_individual_accuracy(records: Sequence[AgentRoundRecord], correct_answer: str) -> float:
    """Share of agents whose last observed answer is correct."""
    if not records:
        raise ValueError("final_individual_accuracy requires at least one record")
    final_round = max(r.round for r in records)
    finals = [r for r in records if r.round == final_round]
    return sum(r.answer == correct_answer for r in finals) / len(finals)


def favored_adoption_by_round(
    records: Sequence[AgentRoundRecord], favored_option: str
) -> list[float]:
    """Per round: share of agents answering the assumption-favored option.

    The cheapest propagation signal — visible directly in the answers.
    """
    if not records:
        return []
    shares = []
    for round_index in range(max(r.round for r in records) + 1):
        round_records = [r for r in records if r.round == round_index]
        shares.append(sum(r.answer == favored_option for r in round_records) / len(round_records))
    return shares


def favored_share_auc(shares: Sequence[float]) -> Optional[float]:
    """Normalized trapezoidal area under the favored-option share curve.

    The result remains on the 0--1 share scale and therefore compares runs
    with different round counts. A one-round control has AUC equal to its
    only observed share.
    """
    if not shares:
        return None
    if len(shares) == 1:
        return float(shares[0])
    area = sum((left + right) / 2 for left, right in zip(shares, shares[1:]))
    return area / (len(shares) - 1)


def seed_trap_uptake(
    records: Sequence[AgentRoundRecord], seed_agent_id: int, favored_option: str
) -> Optional[bool]:
    """Whether the seed selected the assumption-favored option in round 0."""
    row = next(
        (r for r in records if r.agent_id == seed_agent_id and r.round == 0),
        None,
    )
    if row is None or row.answer is None or row.parse_failed:
        return None
    return row.answer == favored_option


def neutral_spontaneous_trap_rate(
    records: Sequence[AgentRoundRecord],
    *,
    seed_agent_id: int,
    corrector_agent_id: int,
    favored_option: str,
) -> Optional[float]:
    """Round-0 trap share among the two neutral roles.

    Excluding both private-information roles keeps this diagnostic separate
    from seed uptake and from any early correction schedule.
    """
    rows = [
        r for r in records
        if r.round == 0 and r.agent_id not in (seed_agent_id, corrector_agent_id)
        and r.answer is not None and not r.parse_failed
    ]
    if not rows:
        return None
    return sum(r.answer == favored_option for r in rows) / len(rows)


def direct_exposure_adoption(
    records: Sequence[AgentRoundRecord], favored_option: str
) -> tuple[Optional[float], int, int]:
    """Adoption after direct prior-round exposure to a favored answer.

    An opportunity is a receiver that did not previously hold the favored
    answer and received at least one message from a sender who did. Returns
    ``(rate, adopted_count, opportunity_count)``. This is an answer-level
    routing proxy: it counts answer changes, not agreement in prose.
    """
    by_key = {(r.agent_id, r.round): r for r in records}
    opportunities = 0
    adopted = 0
    for row in records:
        if row.round == 0:
            continue
        if row.parse_failed or row.carried_forward or row.answer is None:
            # No new interpretable receiver decision was produced.
            continue
        prior_self = by_key.get((row.agent_id, row.round - 1))
        if prior_self is None or prior_self.answer == favored_option:
            continue
        exposed = any(
            (sender_row := by_key.get((sender, row.round - 1))) is not None
            and sender_row.answer == favored_option
            for sender in row.received_from
        )
        if exposed:
            opportunities += 1
            adopted += row.answer == favored_option
    rate = adopted / opportunities if opportunities else None
    return rate, adopted, opportunities


def first_downstream_answer_adoption_round(
    records: Sequence[AgentRoundRecord], seed_agent_id: int, favored_option: str
) -> Optional[int]:
    """Earliest post-round-0 switch to the trap by a non-seed agent.

    A neutral agent already on the trap in round 0 is counted by the separate
    spontaneous-trap metric, not relabeled as downstream adoption in round 1.
    Failed/carried updates are likewise not new adoption decisions.
    """
    by_key = {(r.agent_id, r.round): r for r in records}
    rounds = [
        r.round
        for r in records
        if r.round > 0
        and r.agent_id != seed_agent_id
        and r.answer == favored_option
        and not r.parse_failed
        and not r.carried_forward
        and (prior := by_key.get((r.agent_id, r.round - 1))) is not None
        and prior.answer != favored_option
    ]
    return min(rounds) if rounds else None


def first_answer_recovery_round(
    records: Sequence[AgentRoundRecord], correct_answer: str, favored_option: str
) -> Optional[int]:
    """Earliest switch from the favored trap to the correct answer."""
    by_key = {(r.agent_id, r.round): r for r in records}
    rounds = [
        row.round
        for row in records
        if row.round > 0
        and row.answer == correct_answer
        and (prior := by_key.get((row.agent_id, row.round - 1))) is not None
        and prior.answer == favored_option
    ]
    return min(rounds) if rounds else None


def unrelated_option_rate(
    records: Sequence[AgentRoundRecord],
    *,
    correct_answer: str,
    favored_option: str,
    round_index: int | None = None,
) -> Optional[float]:
    """Share choosing neither the correct answer nor the target trap."""
    rows = [
        r
        for r in records
        if (round_index is None or r.round == round_index)
        and r.answer is not None
        and not r.parse_failed
        and not r.carried_forward
    ]
    if not rows:
        return None
    return sum(r.answer not in (correct_answer, favored_option) for r in rows) / len(rows)


def flip_count(records: Sequence[AgentRoundRecord], agent_id: int) -> int:
    """How many times one agent changed its answer across rounds."""
    answers = [a for a in answer_trajectory(records, agent_id) if a is not None]
    return sum(1 for prev, cur in zip(answers, answers[1:]) if prev != cur)


def seed_recanted(
    records: Sequence[AgentRoundRecord], seed_agent_id: int, favored_option: str
) -> Optional[bool]:
    """Did the seed start on the favored option and abandon it by its final round?

    None when the seed never started on the favored option (not applicable).
    This answer-level diagnostic does not attribute the change to a particular
    message or to corrective evidence; those mechanisms are measured separately.
    """
    seed_rows = sorted(
        (r for r in records if r.agent_id == seed_agent_id),
        key=lambda row: row.round,
    )
    if not seed_rows or seed_rows[0].round != 0:
        return None
    if seed_rows[0].answer != favored_option:
        return None
    return seed_rows[-1].answer != favored_option
