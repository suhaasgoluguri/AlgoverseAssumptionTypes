"""Tests for cross-conversation analysis and the judged-records merge."""

from assumption_propagation.analysis import (
    aggregate_summaries,
    derive_protocol_metadata,
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
        rationale="span",
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
    accepting = state in ("qualified_acceptance", "categorical_acceptance")
    supported_correction = state == "supported_correction"
    unclear = state == "unclear"
    return JudgedAgentRound(
        run_id=run_id, assumption_id=assumption_id, agent_id=agent, round=rnd,
        assumption_state=state,
        decision_reliance=accepting,
        explicit_uncertainty_retained=state == "qualified_acceptance",
        verification_gap_retained=state == "challenge",
        provenance_retained=False,
        correction_present=supported_correction,
        corrective_evidence_used=supported_correction,
        confidence_level=(
            2 if state == "qualified_acceptance"
            else 3 if state == "categorical_acceptance"
            else None
        ),
        judge_evidence_span="" if state in ("absent", "unclear") else "span",
        judge_uncertain=unclear,
        judge_uncertainty_reason="ambiguous rationale" if unclear else "",
        judge_codebook_version="v1.0",
        judge_prompt_version="j1",
        judge_model="mock:judge",
        judged_text_source="rationale",
    )


def test_merge_joins_on_run_agent_round():
    records = conversation("r1", FOUR_ROUND)
    labels = [
        label("r1", 0, 0, "categorical_acceptance"),
        label("r1", 1, 0, "supported_correction"),
    ]
    matched, orphans, unjudged, conflicts = merge_judged(records, labels)
    assert len(matched) == 2 and not orphans and not conflicts
    assert len(unjudged) == 14
    record, lab = matched[("r1", 0, 0)]
    assert record.answer == "A" and lab.assumption_state == "categorical_acceptance"


def test_merge_reports_orphans_duplicates_and_mismatches():
    records = conversation("r1", FOUR_ROUND)
    labels = [
        label("r1", 0, 0, "categorical_acceptance"),
        label("r1", 0, 0, "rejection"),                      # duplicate key
        label("r_ghost", 2, 1, "categorical_acceptance"),     # no such record
        label("r1", 1, 1, "categorical_acceptance", assumption_id="other"),
    ]
    matched, orphans, unjudged, conflicts = merge_judged(records, labels)
    assert len(matched) == 1
    assert len(orphans) == 1 and orphans[0].run_id == "r_ghost"
    assert len(conflicts) == 2
    assert any("duplicate" in c for c in conflicts)
    assert any("mismatch" in c for c in conflicts)


def test_merge_rejects_ambiguous_duplicate_raw_records():
    records = conversation("r1", FOUR_ROUND)
    records.append(records[0].model_copy())
    labels = [label("r1", 0, 0, "categorical_acceptance")]

    matched, orphans, unjudged, conflicts = merge_judged(records, labels)

    assert not matched and not orphans
    assert len(unjudged) == len(records)
    assert conflicts == ["duplicate raw record for ('r1', 0, 0)"]


def test_merge_reports_mixed_judge_measurement_instruments():
    records = conversation("r1", FOUR_ROUND)
    labels = [
        label("r1", 0, 0, "absent"),
        label("r1", 1, 0, "absent").model_copy(
            update={"judge_model": "different:judge"}
        ),
    ]

    matched, _, _, conflicts = merge_judged(records, labels)

    assert len(matched) == 2
    assert any("mixed judge configurations" in conflict for conflict in conflicts)


def test_merge_preserves_undefined_verification_gap_as_null():
    records = conversation("r1", FOUR_ROUND)
    non_null = label("r1", 0, 0, "absent")
    matched, _, unjudged, conflicts = merge_judged(
        records,
        [non_null],
        verification_gap_defined={"s1_factual_negative": False},
    )
    assert not matched
    assert len(unjudged) == len(records)
    assert any("availability mismatch" in conflict for conflict in conflicts)

    null_label = non_null.model_copy(update={"verification_gap_retained": None})
    matched, _, _, conflicts = merge_judged(
        records,
        [null_label],
        verification_gap_defined={"s1_factual_negative": False},
    )
    assert len(matched) == 1
    assert not conflicts


def test_protocol_exposure_uses_prior_message_content_not_routing_alone():
    records = [
        AgentRoundRecord(
            run_id="r1", scenario_id="s1", assumption_id="s1_factual_negative",
            condition="injection", topology="looped_chain", model="mock",
            agent_id=0, round=0, rationale="span", injection_visible=True,
        ),
        AgentRoundRecord(
            run_id="r1", scenario_id="s1", assumption_id="s1_factual_negative",
            condition="injection", topology="looped_chain", model="mock",
            agent_id=1, round=0, rationale="span", corrective_visible=True,
        ),
        AgentRoundRecord(
            run_id="r1", scenario_id="s1", assumption_id="s1_factual_negative",
            condition="injection", topology="looped_chain", model="mock",
            agent_id=1, round=1, rationale="span", received_from=[0],
        ),
        AgentRoundRecord(
            run_id="r1", scenario_id="s1", assumption_id="s1_factual_negative",
            condition="injection", topology="looped_chain", model="mock",
            agent_id=2, round=1, rationale="span", received_from=[1],
        ),
        AgentRoundRecord(
            run_id="r1", scenario_id="s1", assumption_id="s1_factual_negative",
            condition="injection", topology="looped_chain", model="mock",
            agent_id=1, round=2, rationale="span", received_from=[],
        ),
    ]
    labels = [
        label("r1", 0, 0, "categorical_acceptance"),
        label("r1", 1, 0, "supported_correction"),
        label("r1", 1, 1, "absent"),
        label("r1", 2, 1, "absent"),
        label("r1", 1, 2, "absent"),
    ]

    derived = derive_protocol_metadata(records, labels)

    seed = derived[("r1", 0, 0)]
    assert seed["injection_note_visible_this_round"] is True
    assert seed["target_engagement_routed_this_round"] is False

    downstream = derived[("r1", 1, 1)]
    assert downstream["target_engaging_sender_ids"] == [0]
    assert downstream["acceptance_bearing_sender_ids"] == [0]
    assert downstream["target_engagement_routed_this_round"] is True

    correction_recipient = derived[("r1", 2, 1)]
    assert correction_recipient["correction_bearing_sender_ids"] == [1]
    assert correction_recipient["correction_routed_this_round"] is True

    # Routed engagement remains cumulative even when no new such message arrives.
    later = derived[("r1", 1, 2)]
    assert later["target_engagement_routed_this_round"] is False
    assert later["target_engagement_routed_by_round"] is True


def test_protocol_routing_uncertainty_remains_cumulative():
    records = [
        AgentRoundRecord(
            run_id="r1", scenario_id="s1", assumption_id="s1_factual_negative",
            condition="injection", topology="looped_chain", model="mock",
            agent_id=0, round=0, rationale="ambiguous",
        ),
        AgentRoundRecord(
            run_id="r1", scenario_id="s1", assumption_id="s1_factual_negative",
            condition="injection", topology="looped_chain", model="mock",
            agent_id=1, round=1, rationale="span", received_from=[0],
        ),
        AgentRoundRecord(
            run_id="r1", scenario_id="s1", assumption_id="s1_factual_negative",
            condition="injection", topology="looped_chain", model="mock",
            agent_id=1, round=2, rationale="span", received_from=[],
        ),
    ]
    labels = [
        label("r1", 0, 0, "unclear"),
        label("r1", 1, 1, "absent"),
        label("r1", 1, 2, "absent"),
    ]

    derived = derive_protocol_metadata(records, labels)

    round_one = derived[("r1", 1, 1)]
    assert round_one["routing_derivation_uncertain_this_round"] is True
    assert round_one["routing_derivation_uncertain_by_round"] is True
    round_two = derived[("r1", 1, 2)]
    assert round_two["routing_derivation_uncertain_this_round"] is False
    assert round_two["routing_derivation_uncertain_by_round"] is True


def test_judged_summary_withholds_metrics_at_partial_coverage():
    labels = (
        [label("r1", 0, r, "categorical_acceptance") for r in range(4)]
        + [label("r1", 1, r, "supported_correction") for r in range(4)]
        + [
            label("r1", 2, 1, "categorical_acceptance"),
            label("r1", 2, 3, "categorical_acceptance"),
        ]
        + [label("r1", 3, 3, "absent")]
    )
    s = summarize_judged_conversation(labels, seed_agent_id=0, n_records=16)
    assert s["judge_coverage"] == 11 / 16
    for field in (
        "final_acceptance_rate", "false_consensus", "first_adoption_round",
        "first_resistance_round", "first_correction_round",
        "correction_occurred", "correction_latency",
    ):
        assert s[field] is None


def test_judged_summary_withholds_metrics_for_duplicate_keys():
    labels = [
        label("r1", agent, rnd, "absent")
        for agent in range(4)
        for rnd in range(4)
    ]
    labels.append(label("r1", 0, 0, "categorical_acceptance"))

    summary = summarize_judged_conversation(
        labels, seed_agent_id=0, n_records=16
    )

    assert summary["judge_coverage"] == 1.0
    assert summary["judge_duplicate_key_count"] == 1
    assert summary["final_acceptance_rate"] is None


def test_judged_summary_requires_adjudication_for_uncertain_labels():
    labels = [
        label("r1", agent, rnd, "absent")
        for agent in range(4)
        for rnd in range(4)
    ]
    labels[0] = label("r1", 0, 0, "unclear")

    summary = summarize_judged_conversation(
        labels, seed_agent_id=0, n_records=16
    )

    assert summary["judge_coverage"] == 1.0
    assert summary["judge_usable_coverage"] == 15 / 16
    assert summary["judge_unclear_count"] == 1
    assert summary["judge_uncertain_count"] == 1
    assert summary["judge_adjudication_required"] is True
    assert summary["final_acceptance_rate"] is None


def test_judged_summary_metrics_require_complete_coverage():
    labels = (
        [label("r1", 0, r, "categorical_acceptance") for r in range(4)]
        + [label("r1", 1, r, "supported_correction") for r in range(4)]
        + [label("r1", 2, 0, "absent")]
        + [label("r1", 2, r, "categorical_acceptance") for r in range(1, 4)]
        + [label("r1", 3, r, "absent") for r in range(4)]
    )
    s = summarize_judged_conversation(labels, seed_agent_id=0, n_records=16)
    assert s["judge_coverage"] == 1.0
    # Round-3 non-seed labels: A1 supplies a supported correction, A2 accepts, A3 absent.
    assert s["final_acceptance_rate"] == 1 / 3
    assert s["false_consensus"] is False  # A0+A2 accept at round 3: 2 < 3
    assert s["first_adoption_round"] == 1  # A2 accepts at round 1
    assert s["first_resistance_round"] == 0
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


def test_aggregate_withholds_headline_judge_metric_if_any_run_is_unresolved():
    base = {
        "condition": "injection",
        "topology": "looped_chain",
        "semantic_type": "factual",
        "final_individual_accuracy": 1.0,
        "judge_coverage": 1.0,
    }
    rows = aggregate_summaries([
        {**base, "final_acceptance_rate": 0.5},
        {**base, "final_acceptance_rate": None},
    ])

    assert rows[0]["n_judged_conversations"] == 2
    assert rows[0]["judge_coverage"] == 1.0
    assert rows[0]["final_acceptance_rate"] is None
