"""Correction-scope handling in ``scripts/analyze_run.py``."""

import pytest

from assumption_propagation.schemas import (
    CORRECTION_SCOPE,
    TOOL_CORRECTION_SCOPE,
    TOOL_CORRECTION_SCOPE_WITH_CONTROLS,
)
from scripts.analyze_run import (
    _index_injection_summaries,
    _summarize_seed_trial_itt,
    _validate_injection_outcome_grid,
    _validated_expected_injection_conditions,
)


def _summary(condition: str, *, run_id: str | None = None) -> dict:
    return {
        "run_id": run_id or f"run-{condition}",
        "condition": condition,
        "scenario_id": "vendor_001",
        "assumption_id": "vendor_001_factual_negative",
        "seed_instance_id": "seed-00",
        "topology": "fully_connected_mesh",
    }


def _trial(*, replayable: bool = True) -> dict:
    return {
        "seed_instance_id": "seed-00",
        "scenario_id": "vendor_001",
        "assumption_id": "vendor_001_factual_negative",
        "semantic_type": "factual",
        "correct_answer": "B",
        "favored_option": "A",
        "seed_trap_uptake": True if replayable else None,
        "replayable": replayable,
    }


def test_explicit_manifest_injection_conditions_are_authoritative():
    manifest = {
        "protocol": {
            "injection_conditions": [
                "injection_no_correction",
                "injection",
                "injection_relay",
                "injection_broadcast",
            ]
        }
    }

    assert _validated_expected_injection_conditions(manifest, []) == list(
        CORRECTION_SCOPE
    )


@pytest.mark.parametrize(
    "conditions",
    [TOOL_CORRECTION_SCOPE, TOOL_CORRECTION_SCOPE_WITH_CONTROLS],
)
def test_tool_protocol_manifest_keeps_new_arms_separate(conditions):
    manifest = {
        "protocol": {
            "scope_ladder": True,
            "scope_protocol": "tool_v2",
            "injection_conditions": list(conditions),
        }
    }

    assert _validated_expected_injection_conditions(manifest, []) == list(conditions)


@pytest.mark.parametrize(
    "protocol, expected",
    [
        ({"scope_ladder": True}, list(CORRECTION_SCOPE)),
        (
            {"ablation_arm": True},
            ["injection", "injection_no_correction"],
        ),
        ({"withhold_corrective": True}, ["injection_no_correction"]),
        ({}, ["injection"]),
    ],
)
def test_legacy_manifest_switches_reconstruct_the_arm_plan(protocol, expected):
    observed = [_summary("injection")] if not protocol else []
    assert _validated_expected_injection_conditions(
        {"protocol": protocol}, observed
    ) == expected


def test_manifestless_run_falls_back_to_observed_scope_arms():
    observed = [_summary(condition) for condition in CORRECTION_SCOPE]
    assert _validated_expected_injection_conditions(None, observed) == sorted(
        CORRECTION_SCOPE
    )


def test_scope_outcomes_are_indexed_separately_by_condition():
    trial = _trial()
    trial_by_id = {trial["seed_instance_id"]: trial}
    summaries = [_summary(condition) for condition in CORRECTION_SCOPE]

    indexed = _index_injection_summaries(
        summaries, trial_by_id, ["fully_connected_mesh"]
    )

    assert set(indexed) == {
        (condition, "seed-00", "fully_connected_mesh")
        for condition in CORRECTION_SCOPE
    }
    assert all(len(rows) == 1 for rows in indexed.values())


def test_complete_scope_grid_rejects_one_missing_arm():
    trial = _trial()
    trial_by_id = {trial["seed_instance_id"]: trial}
    summaries = [
        _summary(condition)
        for condition in CORRECTION_SCOPE
        if condition != "injection_broadcast"
    ]
    indexed = _index_injection_summaries(
        summaries, trial_by_id, ["fully_connected_mesh"]
    )

    with pytest.raises(ValueError, match="0 injection_broadcast outcomes"):
        _validate_injection_outcome_grid(
            indexed,
            [trial],
            list(CORRECTION_SCOPE),
            ["fully_connected_mesh"],
            require_complete=True,
        )

    # An explicitly incomplete diagnostic run may retain the partial grid.
    _validate_injection_outcome_grid(
        indexed,
        [trial],
        list(CORRECTION_SCOPE),
        ["fully_connected_mesh"],
        require_complete=False,
    )


def test_seed_trial_itt_keeps_scope_arms_in_separate_rows():
    trial = _trial()
    outcome_rows = []
    for index, condition in enumerate(CORRECTION_SCOPE):
        outcome_rows.append({
            **trial,
            "condition": condition,
            "topology": "fully_connected_mesh",
            "outcome_observed": True,
            "final_plurality_accuracy": float(index >= 2),
            "final_plurality_favored": index < 2,
        })

    rows = _summarize_seed_trial_itt(outcome_rows, complete_run=True)

    assert {row["condition"] for row in rows} == set(CORRECTION_SCOPE)
    assert len(rows) == 4
    by_condition = {row["condition"]: row for row in rows}
    assert by_condition["injection"]["final_plurality_accuracy_observed"] == 0.0
    assert by_condition["injection_broadcast"][
        "final_plurality_accuracy_observed"
    ] == 1.0
