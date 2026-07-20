"""Tests for cross-conversation analysis and the judged-records merge."""

from assumption_propagation.analysis import (
    aggregate_summaries,
    group_by_run,
    merge_judged,
    summarize_conversation,
    summarize_judged_conversation,
)
from assumption_propagation.schemas import AgentRoundRecord, JudgedAgentRound


def rec(run_id, agent, rnd, answer, *, condition="injection",
        topology="looped_chain", assumption_id="s1_factual_negative",
        seed_agent_id=0):
    return AgentRoundRecord(
        run_id=run_id, scenario_id="s1", assumption_id=assumption_id,
        condition=condition, topology=topology, model="mock",
        agent_id=agent, round=rnd, answer=answer, seed_agent_id=seed_agent_id,
    )


def conversation(run_id, trajs, **kwargs):
    return [rec(run_id, a, i, ans, **kwargs)
            for a, t in trajs.items() for i, ans in enumerate(t)]


FOUR_ROUND = {
    0: ["A", "A", "A", "D"],  # seed recants at loop closure
    1: ["D", "D", "D", "D"],
    2: ["D", "A", "D", "D"],
    3: ["D", "D", "A", "A"],
}


def test_group_by_run_splits_conversations():
    records = conversation("r1", FOUR_ROUND) + conversation("r2", FOUR_ROUND)
    grouped = group_by_run(records)
    assert set(grouped) == {"r1", "r2"}
    assert len(grouped["r1"]) == 16


def test_summarize_four_round_conversation():
    s = summarize_conversation(
        conversation("r1", FOUR_ROUND),
        correct_answer="D", favored_option="A", semantic_type="factual",
    )
    assert s["final_round"] == 3
    assert s["final_individual_accuracy"] == 0.75  # A3 ends on A
    assert s["favored_adoption_final"] == 0.25
    assert s["favored_adoption_by_round"] == [0.25, 0.5, 0.5, 0.25]
    assert s["seed_recanted"] is True
    assert s["flip_counts"][1] == 0 and s["flip_counts"][3] == 1


def test_summarize_independent_control_uses_round_zero_as_final():
    records = [rec("r3", a, 0, ans, condition="independent_control",
                   topology="independent", assumption_id=None, seed_agent_id=None)
               for a, ans in enumerate(["D", "D", "A", "D"])]
    s = summarize_conversation(records, correct_answer="D")
    assert s["final_round"] == 0
    assert s["final_individual_accuracy"] == 0.75
    assert "favored_adoption_by_round" not in s


def test_summarize_without_favored_option_skips_favored_fields():
    s = summarize_conversation(conversation("r1", FOUR_ROUND), correct_answer="D")
    assert "favored_adoption_final" not in s and "seed_recanted" not in s


# ---------------------------------------------------------------------------
# Judge merge
# ---------------------------------------------------------------------------

def label(run_id, agent, rnd, state, *, assumption_id="s1_factual_negative"):
    return JudgedAgentRound(
        run_id=run_id, assumption_id=assumption_id, agent_id=agent, round=rnd,
        exposed=True, acquisition_provenance="peer_message",
        evidence_status="peer_testimony_only",
        assumption_present=state != "absent", assumption_state=state,
        decision_reliance=state in ("accepted", "qualified_acceptance"),
        judge_evidence_span="" if state == "absent" else "span",
        judge_prompt_version="j1",
    )


def test_merge_joins_on_run_agent_round():
    records = conversation("r1", FOUR_ROUND)
    labels = [label("r1", 0, 0, "accepted"), label("r1", 1, 0, "corrected")]
    matched, orphans, unjudged, conflicts = merge_judged(records, labels)
    assert len(matched) == 2 and not orphans and not conflicts
    assert len(unjudged) == 14
    record, lab = matched[("r1", 0, 0)]
    assert record.answer == "A" and lab.assumption_state == "accepted"


def test_merge_reports_orphans_duplicates_and_mismatches():
    records = conversation("r1", FOUR_ROUND)
    labels = [
        label("r1", 0, 0, "accepted"),
        label("r1", 0, 0, "rejected"),                      # duplicate key
        label("r_ghost", 2, 1, "accepted"),                 # no such record
        label("r1", 1, 1, "accepted", assumption_id="other"),  # id mismatch
    ]
    matched, orphans, unjudged, conflicts = merge_judged(records, labels)
    assert len(matched) == 1
    assert len(orphans) == 1 and orphans[0].run_id == "r_ghost"
    assert len(conflicts) == 2
    assert any("duplicate" in c for c in conflicts)
    assert any("mismatch" in c for c in conflicts)


def test_judged_summary_metrics_and_coverage():
    labels = (
        [label("r1", 0, r, "accepted") for r in range(4)]      # seed
        + [label("r1", 1, r, "corrected") for r in range(4)]   # corrector
        + [label("r1", 2, 1, "accepted"), label("r1", 2, 3, "accepted")]
        + [label("r1", 3, 3, "absent")]
    )
    s = summarize_judged_conversation(labels, seed_agent_id=0, n_records=16)
    assert s["judge_coverage"] == 11 / 16
    # Round-3 non-seed labels: A1 corrected, A2 accepted, A3 absent.
    assert s["final_acceptance_rate"] == 1 / 3
    assert s["false_consensus"] is False  # A0+A2 accept at round 3: 2 < 3
    assert s["first_adoption_round"] == 1  # A2 accepts at round 1
    assert s["first_correction_round"] == 0
    assert s["correction_occurred"] is True
    assert s["correction_latency"] == -1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_groups_and_averages():
    def summary(topology, acc, favored):
        return {
            "condition": "injection", "topology": topology,
            "semantic_type": "factual", "final_individual_accuracy": acc,
            "favored_adoption_final": favored, "seed_recanted": None,
            "favored_adoption_by_round": [0.25, 0.5, 0.5, favored],
        }

    rows = aggregate_summaries([
        summary("looped_chain", 0.75, 0.25),
        summary("looped_chain", 0.25, 0.75),
        summary("neutral_star", 1.0, 0.0),
    ])
    assert len(rows) == 2
    chain = next(r for r in rows if r["topology"] == "looped_chain")
    assert chain["n_conversations"] == 2
    assert chain["final_individual_accuracy"] == 0.5
    assert chain["favored_adoption_final"] == 0.5
    assert chain["favored_adoption_by_round"] == [0.25, 0.5, 0.5, 0.5]
    assert chain["seed_recanted"] is None  # all-None field stays None


def test_aggregate_handles_missing_optional_fields():
    rows = aggregate_summaries([{
        "condition": "independent_control", "topology": "independent",
        "semantic_type": None, "final_individual_accuracy": 1.0,
    }])
    assert rows[0]["favored_adoption_final"] is None
    assert rows[0]["favored_adoption_by_round"] is None
