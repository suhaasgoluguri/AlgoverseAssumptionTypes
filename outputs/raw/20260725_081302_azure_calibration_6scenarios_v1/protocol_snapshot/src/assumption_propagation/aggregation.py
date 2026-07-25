"""Final-answer aggregation: equal-weight plurality voting.

Design rules:
- Every agent's round-3 answer counts once (the star hub gets no extra
  decision authority).
- A tie for first place is recorded as NO CONSENSUS (plurality_answer=None);
  it is a first-class outcome, not an error.
- Agents whose final answer is unparseable contribute no vote; the
  denominator for consensus_strength stays 4 (absent votes weaken
  consensus, they don't shrink the group).
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from .schemas import ANSWER_LETTERS

N_AGENTS = 4


def tally_votes(final_answers: dict[int, Optional[str]]) -> dict[str, int]:
    """Count round-3 votes per option letter. None answers are not counted."""
    counts = Counter(a for a in final_answers.values() if a in ANSWER_LETTERS)
    return {letter: counts.get(letter, 0) for letter in ANSWER_LETTERS}


def plurality(votes: dict[str, int]) -> tuple[Optional[str], bool]:
    """Return (plurality_answer, no_consensus).

    no_consensus is True when the top vote count is shared by two or more
    options (e.g. 2-2, or 1-1-1-1), or when nobody cast a valid vote.
    """
    top = max(votes.values())
    if top == 0:
        return None, True
    leaders = [letter for letter, count in votes.items() if count == top]
    if len(leaders) > 1:
        return None, True
    return leaders[0], False


def consensus_strength(votes: dict[str, int]) -> float:
    """Share of the group (out of N_AGENTS) behind the leading option."""
    return max(votes.values()) / N_AGENTS


def summarize_final_round(
    final_answers: dict[int, Optional[str]],
    *,
    correct_answer: str,
    favored_option: Optional[str] = None,
) -> dict:
    """One conversation's decision-level summary.

    Returns a plain dict (merged into ConversationResult by the caller):
    final_votes, plurality_answer, no_consensus, consensus_strength,
    final_answer_correct (None when no consensus), favored_vote_share.
    """
    votes = tally_votes(final_answers)
    winner, no_consensus = plurality(votes)
    return {
        "final_votes": votes,
        "plurality_answer": winner,
        "no_consensus": no_consensus,
        "consensus_strength": consensus_strength(votes),
        "final_answer_correct": (winner == correct_answer) if winner else None,
        "favored_vote_share": (
            votes[favored_option] / N_AGENTS if favored_option else None
        ),
    }
