"""Focused tests for correction-scope statistical populations and contrasts."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from assumption_propagation.schemas import (
    TOOL_CORRECTION_CONTROLS,
    TOOL_CORRECTION_SCOPE,
    AgentRoundRecord,
)
from scripts.analyze_statistics import build_frame, scope_ladder_report


def _record(run_id, agent, round_index, answer, *, condition="injection"):
    return AgentRoundRecord(
        run_id=run_id,
        scenario_id="s1",
        assumption_id="s1_factual_negative",
        seed_instance_id="seed-1",
        condition=condition,
        topology="fully_connected_mesh",
        model="mock",
        seed_agent_id=0,
        corrector_agent_id=1,
        agent_id=agent,
        round=round_index,
        answer=answer,
        rationale="test",
        received_from=[] if round_index == 0 else [other for other in range(4) if other != agent],
        messages_received=0 if round_index == 0 else 3,
    )


def test_build_frame_labels_roles_and_true_transitions():
    records = []
    initial = {0: "A", 1: "A", 2: "A", 3: "C"}
    final = {0: "A", 1: "D", 2: "D", 3: "A"}
    for agent in range(4):
        records.append(_record("run-1", agent, 0, initial[agent]))
        records.append(_record("run-1", agent, 1, final[agent]))
    with (
        patch("scripts.analyze_statistics.load_agent_records", return_value=records),
        patch(
            "scripts.analyze_statistics.load_keys",
            return_value=(
                {"s1": "D"},
                {"s1_factual_negative": "A"},
                {"s1_factual_negative": 0},
            ),
        ),
    ):
        frame = build_frame(Path("unused-run"), Path("unused-data"))

    assert len(frame) == 3  # seed excluded; verifier retained but labelled
    assert frame.groupby("role").size().to_dict() == {
        "corrector": 1,
        "ordinary_neutral": 2,
    }
    neutral = frame[frame.role == "ordinary_neutral"].set_index("agent_id")
    assert neutral.loc[2, "trap_to_correct_recovery"] == 1
    assert pd.isna(neutral.loc[2, "trap_conversion"])
    assert neutral.loc[3, "trap_conversion"] == 1
    assert pd.isna(neutral.loc[3, "trap_to_correct_recovery"])


def test_pre_scope_recovery_retains_early_correction_through_final_round():
    records = []
    trajectories = {
        0: ["A", "A", "A", "A"],
        1: ["C", "D", "D", "D"],
        # Trap at the available-round checkpoint, corrected at round 2,
        # and retained at final round 3.
        2: ["C", "A", "D", "D"],
        # Never trapped at the checkpoint, so there is no opportunity.
        3: ["C", "C", "D", "D"],
    }
    for agent, answers in trajectories.items():
        for round_index, answer in enumerate(answers):
            records.append(_record("run-early", agent, round_index, answer))

    with (
        patch("scripts.analyze_statistics.load_agent_records", return_value=records),
        patch(
            "scripts.analyze_statistics.load_keys",
            return_value=(
                {"s1": "D"},
                {"s1_factual_negative": "A"},
                {"s1_factual_negative": 1},
            ),
        ),
    ):
        frame = build_frame(Path("unused-run"), Path("unused-data"))

    neutral = frame[frame.role == "ordinary_neutral"].set_index("agent_id")
    assert neutral.loc[2, "pre_scope_round"] == 1
    assert neutral.loc[2, "pre_scope_answer"] == "A"
    assert neutral.loc[2, "pre_scope_trap_to_final_correct_recovery"] == 1
    # The old immediate endpoint misses this recovery because round 2 was
    # already correct immediately before final round 3.
    assert pd.isna(neutral.loc[2, "trap_to_correct_recovery"])
    assert pd.isna(
        neutral.loc[3, "pre_scope_trap_to_final_correct_recovery"]
    )


def test_scope_report_excludes_corrector_and_uses_adjacent_contrasts():
    conditions = (
        "injection_no_correction",
        "injection",
        "injection_relay",
        "injection_broadcast",
    )
    agent_rows = []
    conversation_rows = []
    for scenario in ("s1", "s2", "s3"):
        for condition in conditions:
            agent_rows.append(
                {
                    "scenario_id": scenario,
                    "condition": condition,
                    "role": "corrector",
                    "trap": 1,
                    "correct": 0,
                    "trap_conversion": 1,
                    "pre_scope_trap_to_final_correct_recovery": 0,
                    "trap_to_correct_recovery": 0,
                    "correct_answer_retained": None,
                }
            )
            for _ in range(2):
                agent_rows.append(
                    {
                        "scenario_id": scenario,
                        "condition": condition,
                        "role": "ordinary_neutral",
                        "trap": 0,
                        "correct": 1,
                        "trap_conversion": 0,
                        "pre_scope_trap_to_final_correct_recovery": 1,
                        "trap_to_correct_recovery": 1,
                        "correct_answer_retained": 1,
                    }
                )
            conversation_rows.append(
                {
                    "scenario_id": scenario,
                    "condition": condition,
                    "correct": 1,
                    "defer": 0,
                    "tie": 0,
                }
            )

    report = scope_ladder_report(
        pd.DataFrame(agent_rows), pd.DataFrame(conversation_rows)
    )

    assert report["analysis_population"] == "ordinary_neutral"
    assert report["excluded_corrector_rows"] == 12
    for condition in conditions:
        level = report["levels"][condition]
        assert level["n_ordinary_neutral_rows"] == 6
        assert level["ordinary_neutral_accuracy"] == 1.0
        assert level["final_trap_share"] == 0.0
    assert set(report["adjacent_contrasts"]) == {
        "verifier_minus_none",
        "relay_minus_verifier",
        "direct_minus_relay",
    }
    assert "primary_contrasts" not in report
    assert "conversion_non_increasing" not in report


def test_tool_scope_report_keeps_controls_out_of_the_primary_order():
    conditions = (*TOOL_CORRECTION_SCOPE, *TOOL_CORRECTION_CONTROLS)
    agent_rows = []
    conversation_rows = []
    for scenario in ("s1", "s2", "s3"):
        for condition in conditions:
            for role in ("corrector", "ordinary_neutral", "ordinary_neutral"):
                agent_rows.append(
                    {
                        "scenario_id": scenario,
                        "condition": condition,
                        "role": role,
                        "trap": 0,
                        "correct": 1,
                        "trap_conversion": 0,
                        "pre_scope_trap_to_final_correct_recovery": 1,
                        "trap_to_correct_recovery": 1,
                        "correct_answer_retained": 1,
                    }
                )
            conversation_rows.append(
                {
                    "scenario_id": scenario,
                    "condition": condition,
                    "correct": 1,
                    "defer": 0,
                    "tie": 0,
                }
            )

    report = scope_ladder_report(
        pd.DataFrame(agent_rows),
        pd.DataFrame(conversation_rows),
        scope_order=TOOL_CORRECTION_SCOPE,
        control_conditions=TOOL_CORRECTION_CONTROLS,
    )

    assert report["scope_protocol"] == "tool_v2"
    assert set(report["levels"]) == set(conditions)
    assert set(report["adjacent_contrasts"]) == {
        "testimony_minus_none",
        "relay_minus_testimony",
        "direct_tool_minus_relay",
        "direct_tool_minus_prompt_receipt",
        "direct_tool_minus_placebo",
        "placebo_minus_testimony",
    }


def test_scope_report_marks_absent_pre_scope_opportunities_as_ineligible():
    agent_rows = []
    conversation_rows = []
    for scenario in ("s1", "s2", "s3"):
        for condition in TOOL_CORRECTION_SCOPE:
            for _ in range(2):
                agent_rows.append(
                    {
                        "scenario_id": scenario,
                        "condition": condition,
                        "role": "ordinary_neutral",
                        "trap": 0,
                        "correct": 1,
                        "trap_conversion": 0,
                        "pre_scope_trap_to_final_correct_recovery": None,
                        # Deliberately eligible under the old immediate metric:
                        # the new endpoint must not silently fall back to it.
                        "trap_to_correct_recovery": 1,
                        "correct_answer_retained": 1,
                    }
                )
            conversation_rows.append(
                {
                    "scenario_id": scenario,
                    "condition": condition,
                    "correct": 1,
                    "defer": 0,
                    "tie": 0,
                }
            )

    report = scope_ladder_report(
        pd.DataFrame(agent_rows),
        pd.DataFrame(conversation_rows),
        scope_order=TOOL_CORRECTION_SCOPE,
    )

    for level in report["levels"].values():
        assert level["pre_scope_trap_to_final_correct_recovery_rate"] is None
        assert level["n_pre_scope_recovery_opportunities"] == 0
        assert level["trap_to_correct_recovery_rate"] == 1.0
    result = report["adjacent_contrasts"]["direct_tool_minus_relay"]
    assert "error" in result[
        "ordinary_neutral_pre_scope_trap_to_final_correct_recovery"
    ]
