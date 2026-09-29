"""Scenario-parallel collection: run_parallel.py launches, combine_runs.py joins.

Real mock-client runs, so the joined output goes through run_pilot.py's own
completion checks exactly as a live collection would.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import combine_runs  # noqa: E402
from combine_runs import CombineError, combine  # noqa: E402

DATA = "data/scenarios_hidden_profile_candidates.json"
PILOT_ARGS = [
    "--client", "mock", "--data", DATA, "--variants", "factual",
    "--scope-ladder", "--scope-protocol", "tool_v2", "--n-seeds", "1",
    "--topologies", "fully_connected_mesh", "--attribution", "assigned",
    "--skip-independent", "--rounds", "4",
]


def _launch(tmp: Path, scenarios: list[str], *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "scripts/run_parallel.py", "--workers", "2",
            "--stagger-seconds", "0",
            "--tag", "t", "--scenarios", *scenarios, "--out", str(tmp / "parallel"),
            "--", *PILOT_ARGS, "--out", str(tmp / "raw"), *extra,
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parallel")
    result = _launch(tmp, ["vendor_001", "incident_001"])
    assert result.returncode == 0, result.stdout + result.stderr
    group = next((tmp / "parallel").iterdir())
    manifest = json.loads((group / "parallel_manifest.json").read_text())
    runs = {s: Path(e["run_dir"]) for s, e in manifest["scenarios"].items()}
    return tmp, group, runs


def test_launcher_runs_each_scenario_in_its_own_complete_run(collected):
    _, group, runs = collected
    manifest = json.loads((group / "parallel_manifest.json").read_text())
    assert manifest["state"] == "complete"
    for scenario_id, run in runs.items():
        run_manifest = json.loads((run / "run_manifest.json").read_text())
        assert run_manifest["status"]["state"] == "complete"
        assert run_manifest["scenario_data"]["selected_scenario_ids"] == [scenario_id]
        assert (group / f"{scenario_id}.log").is_file()


def test_combined_run_is_complete_and_matches_the_plan(collected, tmp_path):
    _, group, runs = collected
    manifest = combine(
        tmp_path / "combined", combine_runs.sources_from_parallel(group),
        reason="test", argv=["combine_runs.py"], require_all=False,
    )
    status = manifest["status"]
    assert status["state"] == "complete"
    assert manifest["scenario_data"]["selected_scenario_ids"] == ["vendor_001", "incident_001"]
    # One factual variant x one seed per scenario.
    assert status["planned_seed_generations"] == 2
    assert status["observed_rows"]["seeds"] == 2
    per_run = [
        json.loads((run / "run_manifest.json").read_text())["status"]["observed_rows"]
        for run in runs.values()
    ]
    for key in ("seeds", "agent_round_records", "conversation_results"):
        assert status["observed_rows"][key] == sum(rows[key] for rows in per_run)
    assert [s["scenarios"] for s in manifest["combine"]["sources"]] == [
        ["vendor_001"], ["incident_001"],
    ]

    analysis = subprocess.run(
        [sys.executable, "scripts/analyze_run.py", str(tmp_path / "combined")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert analysis.returncode == 0, analysis.stdout + analysis.stderr


def test_same_scenario_from_two_sources_is_rejected(collected, tmp_path):
    _, _, runs = collected
    copy = tmp_path / "copy"
    shutil.copytree(runs["vendor_001"], copy)
    with pytest.raises(CombineError, match="vendor_001 comes from both"):
        combine(tmp_path / "out", [(runs["vendor_001"], None), (copy, None)],
                reason="", argv=[], require_all=False)
    assert not (tmp_path / "out").exists()


def test_different_settings_are_rejected(collected, tmp_path):
    _, _, runs = collected
    altered = tmp_path / "altered"
    shutil.copytree(runs["incident_001"], altered)
    manifest_path = altered / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["client"]["max_tokens"] = 6000
    manifest["protocol_sources"]["src/assumption_propagation/prompts.py"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CombineError, match="client.*protocol source files"):
        combine(tmp_path / "out", [(runs["vendor_001"], None), (altered, None)],
                reason="", argv=[], require_all=False)


def test_stopped_run_contributes_only_named_finished_scenarios(collected, tmp_path):
    _, _, runs = collected
    stopped = tmp_path / "stopped"
    shutil.copytree(runs["vendor_001"], stopped)
    manifest_path = stopped / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = {"state": "running"}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(CombineError, match="name the finished scenarios"):
        combine(tmp_path / "a", [(stopped, None)], reason="", argv=[], require_all=False)

    manifest = combine(
        tmp_path / "b", [(stopped, ["vendor_001"]), (runs["incident_001"], None)],
        reason="", argv=[], require_all=False,
    )
    assert manifest["combine"]["sources"][0]["source_state"] == "running"

    records = stopped / "records.jsonl"
    lines = records.read_text().splitlines(keepends=True)
    records.write_text("".join(lines[:-4]))
    with pytest.raises(CombineError, match="completion checks"):
        combine(tmp_path / "c", [(stopped, ["vendor_001"])], reason="", argv=[], require_all=False)
    assert not (tmp_path / "c").exists()


def test_launcher_refuses_per_scenario_arguments(tmp_path):
    result = _launch(tmp_path, ["vendor_001"], "--scenario", "incident_001")
    assert result.returncode != 0
    assert "--scenario is set per scenario" in result.stderr
    assert not (tmp_path / "parallel").exists()


def test_missing_corpus_scenarios_are_rejected_by_default(collected, tmp_path):
    _, group, _ = collected
    with pytest.raises(CombineError, match=r"corpus scenario\(s\) missing: shipment_001"):
        combine(tmp_path / "out", combine_runs.sources_from_parallel(group),
                reason="", argv=[])
    assert not (tmp_path / "out").exists()


def _pilot(tmp: Path, out: str, *args: str) -> Path:
    result = subprocess.run(
        [sys.executable, "scripts/run_pilot.py", *args, "--out", str(tmp / out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return next((tmp / out).iterdir())


def test_variant_types_must_match_across_sources(collected, tmp_path):
    _, _, runs = collected
    # Same protocol settings, but factual and causal instead of factual only.
    both_types = [*PILOT_ARGS]
    both_types.insert(both_types.index("factual") + 1, "causal")
    wider = _pilot(tmp_path, "raw", *both_types, "--scenario", "incident_001")
    with pytest.raises(CombineError, match="selected variant types"):
        combine(tmp_path / "out", [(runs["vendor_001"], None), (wider, None)],
                reason="", argv=[], require_all=False)


def test_partial_final_line_from_a_stopped_run_is_ignored(collected, tmp_path):
    _, _, runs = collected
    stopped = tmp_path / "stopped"
    shutil.copytree(runs["vendor_001"], stopped)
    with (stopped / "records.jsonl").open("a") as handle:
        handle.write('{"scenario_id": "incident_001", "run_id": "run_cut')
    manifest = combine(tmp_path / "out", [(stopped, ["vendor_001"])],
                       reason="", argv=[], require_all=False)
    assert manifest["status"]["state"] == "complete"

    with (stopped / "records.jsonl").open("a") as handle:
        handle.write('\n{"scenario_id": "vendor_001"}\n')
    with pytest.raises(CombineError, match="unreadable line"):
        combine(tmp_path / "out2", [(stopped, ["vendor_001"])],
                reason="", argv=[], require_all=False)


def test_parallel_collection_matches_a_sequential_run(tmp_path):
    """Same corpus and arguments: per-scenario runs joined == one sequential run.

    Only the random conversation ids and the mock client's per-process call
    counter (tool call ids, and the transcript hash that includes them) may
    differ; every prompt hash, answer, route and count must be identical.
    """
    corpus = json.loads((REPO_ROOT / DATA).read_text())
    subset = [e for e in corpus if isinstance(e, dict) and "scenario_id" in e][:2]
    data = tmp_path / "two_scenarios.json"
    data.write_text(json.dumps(subset))
    args = [a if a != DATA else str(data) for a in PILOT_ARGS]
    args += ["--scope-controls"]

    sequential = _pilot(tmp_path, "seq", *args)
    launched = subprocess.run(
        [sys.executable, "scripts/run_parallel.py", "--workers", "2",
         "--stagger-seconds", "0", "--tag", "p", "--out", str(tmp_path / "parallel"),
         "--", *args, "--out", str(tmp_path / "par")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert launched.returncode == 0, launched.stdout + launched.stderr
    group = next((tmp_path / "parallel").iterdir())
    combine(tmp_path / "combined", combine_runs.sources_from_parallel(group),
            reason="", argv=[])

    def canonical(run: Path, name: str) -> list[str]:
        rows = []
        for line in (run / name).read_text().splitlines():
            row = json.loads(line)
            row.pop("run_id", None)
            if row.get("tool_trace"):
                row["tool_trace"].pop("tool_call_id")
                row["tool_trace"].pop("transcript_sha256")
            rows.append(json.dumps(row, sort_keys=True))
        return sorted(rows)

    for name in ("seeds.jsonl", "records.jsonl", "results.jsonl"):
        assert canonical(sequential, name) == canonical(tmp_path / "combined", name), name
    seq_manifest = json.loads((sequential / "run_manifest.json").read_text())
    comb_manifest = json.loads((tmp_path / "combined" / "run_manifest.json").read_text())
    for field in ("client", "protocol", "protocol_sources", "prompt_version"):
        assert seq_manifest[field] == comb_manifest[field], field
    for field in ("planned_seed_generations", "planned_conversation_attempts", "observed_rows"):
        assert seq_manifest["status"][field] == comb_manifest["status"][field], field


SLEEPER = """
import os, sys, time, pathlib
pathlib.Path(os.environ["PID_DIR"], str(os.getpid())).touch()
print("client=mock:test  scenarios=1  out=/nonexistent/run", flush=True)
time.sleep(60)
"""
WRAPPER = """
import sys
sys.path.insert(0, {scripts!r})
import run_parallel
run_parallel.RUN_PILOT = {sleeper!r}
sys.argv = ["run_parallel.py", *{argv!r}]
run_parallel.main()
"""


@pytest.mark.parametrize("signal_to", ["whole_group", "launcher_only"])
def test_stopping_the_launcher_leaves_no_children_running(tmp_path, signal_to):
    import os
    import signal
    import time

    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text(SLEEPER)
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    argv = ["--workers", "3", "--stagger-seconds", "0", "--tag", "s",
            "--scenarios", "vendor_001", "incident_001", "shipment_001", "clinic_001",
            "--out", str(tmp_path / "parallel"), "--", "--data", DATA]
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(WRAPPER.format(
        scripts=str(REPO_ROOT / "scripts"), sleeper=str(sleeper), argv=argv))
    launcher = subprocess.Popen(
        [sys.executable, str(wrapper)], cwd=REPO_ROOT,
        env={**os.environ, "PID_DIR": str(pid_dir)},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,  # its own group, so the test can signal all of it
    )
    deadline = time.time() + 30
    while len(list(pid_dir.iterdir())) < 3 and time.time() < deadline:
        time.sleep(0.1)
    children = [int(p.name) for p in pid_dir.iterdir()]
    assert len(children) == 3

    if signal_to == "whole_group":  # Ctrl-C in a terminal
        os.killpg(launcher.pid, signal.SIGINT)
    else:  # kill <launcher pid>
        launcher.send_signal(signal.SIGTERM)
    output = launcher.communicate(timeout=30)[0].decode()

    for pid in children:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert launcher.returncode == 1, output
    group = next((tmp_path / "parallel").iterdir())
    manifest = json.loads((group / "parallel_manifest.json").read_text())
    assert manifest["state"] == "stopped"
    states = sorted(entry["state"] for entry in manifest["scenarios"].values())
    # Three were running and are stopped; the fourth was never started.
    assert states == ["queued", "stopped", "stopped", "stopped"]
