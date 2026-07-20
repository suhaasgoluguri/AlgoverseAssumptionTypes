"""Tests for voting and metrics — computed by hand, then asserted."""

from assumption_propagation.aggregation import (
    consensus_strength,
    plurality,
    summarize_final_round,
    tally_votes,
)
from assumption_propagation.metrics import (
    answer_trajectory,
    favored_adoption_by_round,
    final_individual_accuracy,
    flip_count,
    seed_recanted,
)
from assumption_propagation.schemas import AgentRoundRecord


def test_tally_ignores_invalid_and_missing():
    votes = tally_votes({0: "A", 1: "A", 2: None, 3: "B"})
    assert votes == {"A": 2, "B": 1, "C": 0, "D": 0, "E": 0}


def test_clear_plurality():
    winner, no_consensus = plurality({"A": 3, "B": 1, "C": 0, "D": 0, "E": 0})
    assert winner == "A" and not no_consensus


def test_two_two_tie_is_no_consensus():
    winner, no_consensus = plurality({"A": 2, "B": 2, "C": 0, "D": 0, "E": 0})
    assert winner is None and no_consensus


def test_no_valid_votes_is_no_consensus():
    winner, no_consensus = plurality({l: 0 for l in "ABCDE"})
    assert winner is None and no_consensus


def test_consensus_strength_uses_group_size_not_valid_votes():
    # 2 votes A, 1 vote B, 1 unparseable -> strength 0.5, not 2/3.
    votes = tally_votes({0: "A", 1: "A", 2: "B", 3: None})
    assert consensus_strength(votes) == 0.5


def test_summarize_final_round_correct_and_favored():
    s = summarize_final_round(
        {0: "A", 1: "D", 2: "A", 3: "A"}, correct_answer="D", favored_option="A"
    )
    assert s["plurality_answer"] == "A"
    assert s["final_answer_correct"] is False
    assert s["favored_vote_share"] == 0.75
    assert s["consensus_strength"] == 0.75


def test_summarize_tie_yields_none_correctness():
    s = summarize_final_round(
        {0: "A", 1: "D", 2: "A", 3: "D"}, correct_answer="D", favored_option="A"
    )
    assert s["no_consensus"] and s["final_answer_correct"] is None


# ---------------------------------------------------------------------------
# Answer-level metrics
# ---------------------------------------------------------------------------

def rec(agent, rnd, answer):
    return AgentRoundRecord(
        run_id="r1", scenario_id="s1", condition="injection",
        topology="looped_chain", model="mock", agent_id=agent, round=rnd,
        answer=answer,
    )


def make_records(trajs: dict[int, list[str]]):
    return [rec(a, i, ans) for a, t in trajs.items() for i, ans in enumerate(t)]


TRAJS = {
    0: ["A", "A", "A", "D"],  # seed: favored until the loop closes, then recants
    1: ["D", "D", "D", "D"],  # corrector holds
    2: ["D", "A", "D", "D"],  # briefly swayed, recovers
    3: ["C", "C", "A", "A"],  # adopts late, stays
}
RECORDS = make_records(TRAJS)


def test_answer_trajectory():
    assert answer_trajectory(RECORDS, 2) == ["D", "A", "D", "D"]


def test_final_accuracy():
    assert final_individual_accuracy(RECORDS, correct_answer="D") == 0.75


def test_favored_adoption_by_round():
    assert favored_adoption_by_round(RECORDS, favored_option="A") == [0.25, 0.5, 0.5, 0.25]


def test_flip_count():
    assert flip_count(RECORDS, 1) == 0
    assert flip_count(RECORDS, 2) == 2
    assert flip_count(RECORDS, 0) == 1


def test_seed_recanted_true():
    assert seed_recanted(RECORDS, seed_agent_id=0, favored_option="A") is True


def test_seed_recanted_not_applicable():
    # Seed never started on the favored option -> None, not False.
    records = make_records({0: ["D", "D", "D", "D"]})
    assert seed_recanted(records, seed_agent_id=0, favored_option="A") is None
