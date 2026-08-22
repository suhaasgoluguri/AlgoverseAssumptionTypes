"""Completion-integrity and all-seed analysis guardrails."""

import json
from pathlib import Path

import pytest

from scripts.analyze_run import (
    _index_injection_summaries,
    _validate_completed_output_files,
    _validated_planned_topologies,
)
from scripts.run_pilot import _validate_run_outputs, load_scenarios
from assumption_propagation.reproducibility import sha256_file
from assumption_propagation.schemas import AgentRoundRecord, SeedInstance


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _minimal_run_files(tmp_path: Path) -> tuple[dict[str, Path], list, str]:
    # Any active scenario with at least two variants works: this fixture needs a
    # well-formed scenario, not a particular one.
    scenarios = load_scenarios(
        Path("data/scenarios_hidden_profile_candidates.json"), "vendor_001"
    )
    variant = scenarios[0].assumption_variants[0]
    seed = SeedInstance(
        seed_instance_id=f"{variant.assumption_id}_seed_00",
        scenario_id=scenarios[0].scenario_id,
        assumption_id=variant.assumption_id,
        seed_trial_index=0,
        answer=variant.favored_option,
        rationale="The private assessment favors this option.",
    )
    records = [
        AgentRoundRecord(
            run_id="run-1",
            scenario_id=scenarios[0].scenario_id,
            assumption_id=variant.assumption_id,
            seed_instance_id=seed.seed_instance_id,
            condition="injection",
            topology="looped_chain",
            model="test:model",
            seed_agent_id=0,
            corrector_agent_id=1,
            agent_id=agent_id,
            round=0,
            answer=variant.favored_option,
            rationale="The selected option appears best.",
        )
        for agent_id in range(4)
    ]
    result = {
        "run_id": "run-1",
        "scenario_id": scenarios[0].scenario_id,
        "assumption_id": variant.assumption_id,
        "seed_instance_id": seed.seed_instance_id,
        "condition": "injection",
        "topology": "looped_chain",
    }
    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("seeds", "records", "results")
    }
    _write_jsonl(paths["seeds"], [seed.model_dump(mode="json")])
    _write_jsonl(
        paths["records"],
        [record.model_dump(mode="json") for record in records],
    )
    _write_jsonl(paths["results"], [result])
    return paths, scenarios, variant.assumption_id


def test_completion_integrity_reconciles_a_minimal_run(tmp_path: Path):
    paths, scenarios, assumption_id = _minimal_run_files(tmp_path)

    summary = _validate_run_outputs(
        paths,
        scenarios,
        [assumption_id],
        n_seeds=1,
        topologies=["looped_chain"],
        n_rounds=1,
        baseline_reps=0,
        skip_baselines=True,
        independent_reps=0,
        skip_independent=True,
    )

    assert summary["replayable_seed_generations"] == 1
    assert summary["reconciled_expected_rows"] == {
        "seeds": 1,
        "agent_round_records": 4,
        "conversation_results": 1,
    }
    assert summary["response_quality"] == {
        "agent_round_records": 4,
        "parse_failed": 0,
        "parse_repaired": 0,
        "generation_failed": 0,
        "carried_forward": 0,
        "tool_trace_records": 0,
        "tool_calls_executed": 0,
        "tool_traces_replayed": 0,
        "tool_api_call_attempts": 0,
        "zero_parse_failures": True,
    }


def test_completion_integrity_reports_parse_failures(tmp_path: Path):
    paths, scenarios, assumption_id = _minimal_run_files(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["records"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0].update(
        {
            "answer": None,
            "rationale": None,
            "raw_response": "unparseable response",
            "parse_failed": True,
            "carried_forward": False,
        }
    )
    _write_jsonl(paths["records"], rows)

    summary = _validate_run_outputs(
        paths,
        scenarios,
        [assumption_id],
        n_seeds=1,
        topologies=["looped_chain"],
        n_rounds=1,
        baseline_reps=0,
        skip_baselines=True,
        independent_reps=0,
        skip_independent=True,
    )

    assert summary["response_quality"]["parse_failed"] == 1
    assert summary["response_quality"]["carried_forward"] == 0
    assert summary["response_quality"]["zero_parse_failures"] is False


def test_completion_integrity_rejects_record_result_attribution_mismatch(
    tmp_path: Path,
):
    paths, scenarios, assumption_id = _minimal_run_files(tmp_path)
    result = json.loads(paths["results"].read_text(encoding="utf-8"))
    result["attribution"] = "assigned"
    _write_jsonl(paths["results"], [result])

    with pytest.raises(ValueError, match="disagrees with its result metadata"):
        _validate_run_outputs(
            paths,
            scenarios,
            [assumption_id],
            n_seeds=1,
            topologies=["looped_chain"],
            n_rounds=1,
            baseline_reps=0,
            skip_baselines=True,
            independent_reps=0,
            skip_independent=True,
            attribution_modes=("assigned",),
        )


def test_completion_integrity_rejects_duplicate_seed_ids(tmp_path: Path):
    paths, scenarios, assumption_id = _minimal_run_files(tmp_path)
    seed_row = json.loads(paths["seeds"].read_text(encoding="utf-8").splitlines()[0])
    _write_jsonl(paths["seeds"], [seed_row, seed_row])

    with pytest.raises(ValueError, match="duplicate generated seed"):
        _validate_run_outputs(
            paths,
            scenarios,
            [assumption_id],
            n_seeds=1,
            topologies=["looped_chain"],
            n_rounds=1,
            baseline_reps=0,
            skip_baselines=True,
            independent_reps=0,
            skip_independent=True,
        )


def test_completion_integrity_rejects_seed_swapped_across_assumptions(
    tmp_path: Path,
):
    paths, scenarios, factual_id = _minimal_run_files(tmp_path)
    causal = scenarios[0].assumption_variants[1]
    factual_seed = SeedInstance.model_validate_json(
        paths["seeds"].read_text(encoding="utf-8").splitlines()[0]
    )
    causal_seed = factual_seed.model_copy(update={
        "seed_instance_id": f"{causal.assumption_id}_seed_00",
        "assumption_id": causal.assumption_id,
    })
    _write_jsonl(
        paths["seeds"],
        [
            factual_seed.model_dump(mode="json"),
            causal_seed.model_dump(mode="json"),
        ],
    )

    factual_result = json.loads(paths["results"].read_text(encoding="utf-8"))
    causal_result = {
        **factual_result,
        "run_id": "run-2",
        "assumption_id": causal.assumption_id,
    }
    # Keep the result/record joins internally consistent while swapping which
    # frozen seed is claimed by each assumption.
    factual_result["seed_instance_id"] = causal_seed.seed_instance_id
    causal_result["seed_instance_id"] = factual_seed.seed_instance_id
    _write_jsonl(paths["results"], [factual_result, causal_result])

    original_records = [
        AgentRoundRecord.model_validate_json(line)
        for line in paths["records"].read_text(encoding="utf-8").splitlines()
    ]
    swapped_records = [
        record.model_copy(update={
            "seed_instance_id": causal_seed.seed_instance_id,
        })
        for record in original_records
    ] + [
        record.model_copy(update={
            "run_id": "run-2",
            "assumption_id": causal.assumption_id,
            "seed_instance_id": factual_seed.seed_instance_id,
        })
        for record in original_records
    ]
    _write_jsonl(
        paths["records"],
        [record.model_dump(mode="json") for record in swapped_records],
    )

    with pytest.raises(ValueError, match="does not match frozen seed"):
        _validate_run_outputs(
            paths,
            scenarios,
            [factual_id, causal.assumption_id],
            n_seeds=1,
            topologies=["looped_chain"],
            n_rounds=1,
            baseline_reps=0,
            skip_baselines=True,
            independent_reps=0,
            skip_independent=True,
        )


@pytest.mark.parametrize(
    "topologies, message",
    [
        ([], "non-empty"),
        (["looped_chain", "looped_chain"], "duplicates"),
        (["independent"], "unknown_or_control"),
    ],
)
def test_analysis_rejects_invalid_manifest_topology_plans(topologies, message):
    manifest = {"protocol": {"topologies": topologies}}

    with pytest.raises(ValueError, match=message):
        _validated_planned_topologies(manifest, [])


def test_analysis_rejects_injection_outcome_for_unknown_seed():
    summary = {
        "run_id": "run-unknown",
        "scenario_id": "travel_001",
        "assumption_id": "travel_001_factual_negative",
        "seed_instance_id": "not-in-seeds-jsonl",
        "topology": "looped_chain",
    }

    with pytest.raises(ValueError, match="unknown seed_instance_id"):
        _index_injection_summaries(
            [summary], {}, ["looped_chain"], ["anonymous"]
        )


def test_complete_manifest_requires_all_observed_file_entries(tmp_path: Path):
    for name in ("seeds", "records", "results"):
        (tmp_path / f"{name}.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="observed_files.seeds"):
        _validate_completed_output_files(
            tmp_path,
            tmp_path / "run_manifest.json",
            {},
        )

    observed = {
        name: {
            "rows": 0,
            "sha256": sha256_file(tmp_path / f"{name}.jsonl"),
        }
        for name in ("seeds", "records", "results")
    }
    _validate_completed_output_files(
        tmp_path,
        tmp_path / "run_manifest.json",
        observed,
    )
