"""Tests for voting and metrics — computed by hand, then asserted."""

import pytest

from assumption_propagation.aggregation import (
    consensus_strength,
    plurality,
    summarize_final_round,
    tally_votes,
)
from assumption_propagation.metrics import (
    answer_trajectory,
    direct_exposure_adoption,
    favored_adoption_by_round,
    favored_share_auc,
    final_individual_accuracy,
    first_answer_recovery_round,
    first_downstream_answer_adoption_round,
    flip_count,
    neutral_spontaneous_trap_rate,
    seed_recanted,
    seed_trap_uptake,
    unrelated_option_rate,
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
    winner, no_consensus = plurality({letter: 0 for letter in "ABCDE"})
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

def rec(
    agent,
    rnd,
    answer,
    *,
    received_from=None,
    parse_failed=False,
    carried_forward=False,
):
    return AgentRoundRecord(
        run_id="r1", scenario_id="s1", condition="injection",
        topology="looped_chain", model="mock", agent_id=agent, round=rnd,
        answer=answer, rationale="test rationale", received_from=received_from or [],
        parse_failed=parse_failed, carried_forward=carried_forward,
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


def test_favored_share_auc_is_normalized_trapezoid():
    assert favored_share_auc([0.25, 0.5, 0.5, 0.25]) == pytest.approx(5 / 12)
    assert favored_share_auc([0.75]) == 0.75
    assert favored_share_auc([]) is None


def test_seed_trap_uptake_and_first_downstream_adoption():
    assert seed_trap_uptake(RECORDS, seed_agent_id=0, favored_option="A") is True
    assert first_downstream_answer_adoption_round(
        RECORDS, seed_agent_id=0, favored_option="A"
    ) == 1


def test_round_zero_spontaneous_trap_is_not_downstream_adoption_timing():
    records = make_records({0: ["A", "A"], 1: ["A", "A"], 2: ["D", "D"], 3: ["D", "D"]})
    assert first_downstream_answer_adoption_round(
        records, seed_agent_id=0, favored_option="A"
    ) is None


def test_neutral_spontaneous_trap_rate_excludes_private_roles():
    assert neutral_spontaneous_trap_rate(
        RECORDS,
        seed_agent_id=0,
        corrector_agent_id=1,
        favored_option="A",
    ) == 0.0


def test_seed_and_neutral_uptake_exclude_unparseable_round_zero_outputs():
    rows = [
        rec(0, 0, None, parse_failed=True),
        rec(1, 0, "D"),
        rec(2, 0, "A"),
        rec(3, 0, None, parse_failed=True),
    ]
    assert seed_trap_uptake(rows, seed_agent_id=0, favored_option="A") is None
    assert neutral_spontaneous_trap_rate(
        rows,
        seed_agent_id=0,
        corrector_agent_id=1,
        favored_option="A",
    ) == 1.0


def test_direct_exposure_adoption_uses_prior_round_sender_answers():
    records = make_records({0: ["A"], 1: ["D"], 2: ["D"], 3: ["D"]})
    records.extend(
        [
            rec(0, 1, "A"),
            rec(1, 1, "A", received_from=[0]),  # opportunity and adoption
            rec(2, 1, "D", received_from=[1]),  # sender was D: no opportunity
            rec(3, 1, "C", received_from=[0]),  # opportunity, no adoption
        ]
    )
    rate, adopted, opportunities = direct_exposure_adoption(records, favored_option="A")
    assert (adopted, opportunities) == (1, 2)
    assert rate == 0.5


def test_direct_exposure_rate_is_null_without_an_opportunity():
    rate, adopted, opportunities = direct_exposure_adoption(RECORDS, favored_option="A")
    assert rate is None and adopted == opportunities == 0


def test_direct_exposure_excludes_failed_receiver_update():
    records = [
        rec(0, 0, "A"),
        rec(1, 0, "D"),
        rec(
            1,
            1,
            "D",
            received_from=[0],
            parse_failed=True,
            carried_forward=True,
        ),
    ]
    rate, adopted, opportunities = direct_exposure_adoption(records, favored_option="A")
    assert rate is None and adopted == opportunities == 0


def test_first_answer_recovery_and_unrelated_error_rate():
    assert first_answer_recovery_round(
        RECORDS, correct_answer="D", favored_option="A"
    ) == 2
    assert unrelated_option_rate(
        RECORDS,
        correct_answer="D",
        favored_option="A",
        round_index=0,
    ) == 0.25
    assert unrelated_option_rate(
        RECORDS,
        correct_answer="D",
        favored_option="A",
        round_index=3,
    ) == 0.0


def test_unrelated_error_rate_excludes_parse_failures():
    rows = [rec(0, 0, "C"), rec(1, 0, None, parse_failed=True)]
    assert unrelated_option_rate(
        rows, correct_answer="D", favored_option="A", round_index=0
    ) == 1.0


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


def test_seed_recanted_uses_last_observed_round_for_short_runs():
    assert seed_recanted(
        make_records({0: ["A", "A"]}), seed_agent_id=0, favored_option="A"
    ) is False
    assert seed_recanted(
        make_records({0: ["A", "D"]}), seed_agent_id=0, favored_option="A"
    ) is True


def test_final_accuracy_uses_last_observed_round_for_short_runs():
    records = make_records({0: ["A", "D"], 1: ["D", "D"]})
    assert final_individual_accuracy(records, correct_answer="D") == 1.0
