"""Focused tests for run scenario snapshots and downstream resolution."""

import json
from pathlib import Path

import pytest

from assumption_propagation.reproducibility import (
    SCENARIO_SNAPSHOT_NAME,
    RUN_MANIFEST_NAME,
    resolve_scenario_data,
    sha256_bytes,
    snapshot_protocol_sources,
    snapshot_scenario_data,
)


def test_scenario_snapshot_preserves_exact_bytes(tmp_path: Path):
    source = tmp_path / "source.json"
    payload = b'[\r\n  {"scenario_id": "s1", "note": "exact bytes"}\r\n]\r\n'
    source.write_bytes(payload)
    run_dir = tmp_path / "run"

    snapshot, digest = snapshot_scenario_data(source, run_dir)

    assert snapshot == run_dir / SCENARIO_SNAPSHOT_NAME
    assert snapshot.read_bytes() == payload
    assert digest == sha256_bytes(payload)


def test_explicit_data_override_has_highest_precedence(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / SCENARIO_SNAPSHOT_NAME).write_text("[]", encoding="utf-8")
    override = tmp_path / "override.json"
    override.write_text('[{"scenario_id":"override"}]', encoding="utf-8")

    resolved, source = resolve_scenario_data(run_dir, override)

    assert resolved == override
    assert source == "explicit_override"


def test_run_snapshot_is_default(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = run_dir / SCENARIO_SNAPSHOT_NAME
    payload = b"[]"
    snapshot.write_bytes(payload)
    (run_dir / RUN_MANIFEST_NAME).write_text(
        json.dumps({"scenario_data": {"sha256": sha256_bytes(payload)}}),
        encoding="utf-8",
    )

    resolved, source = resolve_scenario_data(run_dir, None)

    assert resolved == snapshot
    assert source == "run_snapshot"


def test_legacy_run_requires_explicit_data(tmp_path: Path):
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="require an explicit --data"):
        resolve_scenario_data(run_dir, None)


def test_missing_explicit_data_does_not_fall_back(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="scenario data not found"):
        resolve_scenario_data(run_dir, tmp_path / "missing.json")


def test_snapshot_hash_mismatch_fails_closed(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / SCENARIO_SNAPSHOT_NAME).write_text("[]", encoding="utf-8")
    (run_dir / RUN_MANIFEST_NAME).write_text(
        json.dumps({"scenario_data": {"sha256": "0" * 64}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_scenario_data(run_dir, None)


def test_snapshot_without_manifest_requires_explicit_override(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / SCENARIO_SNAPSHOT_NAME).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="missing run_manifest.json"):
        resolve_scenario_data(run_dir, None)


def test_protocol_source_snapshot_preserves_relative_paths(tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "src" / "prompts.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"PROMPT_VERSION = 'v-test'\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    manifest = snapshot_protocol_sources(repo, run_dir, [Path("src/prompts.py")])

    frozen = run_dir / "protocol_snapshot" / "src" / "prompts.py"
    assert frozen.read_bytes() == source.read_bytes()
    assert manifest["src/prompts.py"]["sha256"] == sha256_bytes(source.read_bytes())
