"""Small reproducibility helpers shared by experiment CLI scripts.

Raw runs retain the exact scenario-file bytes they were generated from.  Any
later analysis step therefore defaults to that immutable snapshot,
while legacy runs without a snapshot require an explicit path to the exact
scenario file that produced them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional


SCENARIO_SNAPSHOT_NAME = "scenario_snapshot.json"
RUN_MANIFEST_NAME = "run_manifest.json"


def sha256_bytes(payload: bytes) -> str:
    """Return the full hexadecimal SHA-256 digest for ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the full hexadecimal SHA-256 digest of a file's exact bytes."""
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(payload: bytes) -> bytes:
    """Return a formatting-independent serialization of a JSON ``payload``.

    The document is parsed and re-serialized with sorted keys, no optional
    whitespace, and an explicit UTF-8 encoding, so line endings, indentation,
    key order, and a leading byte-order mark cannot change the result. Only the
    parsed data survives.
    """
    document = json.loads(payload.decode("utf-8-sig"))
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_json_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a JSON file's *content*, not its bytes.

    Use alongside :func:`sha256_file`, never instead of it. The raw digest
    identifies an exact file and is what proves which bytes a run consumed; it
    is also formatting-sensitive, so a checkout that rewrites line endings
    changes it without changing a single scenario. This digest identifies the
    data and is stable across those serializations, which makes it the right
    thing to gate an experiment on.
    """
    return sha256_bytes(canonical_json_bytes(path.read_bytes()))


def snapshot_scenario_data(source: Path, run_dir: Path) -> tuple[Path, str]:
    """Copy the exact input scenario bytes into ``run_dir`` and hash them.

    The file is deliberately copied byte-for-byte rather than parsed and
    re-serialized: whitespace and authoring metadata are part of the frozen
    experimental material.
    """
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"scenario data not found: {source}")
    payload = source.read_bytes()
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / SCENARIO_SNAPSHOT_NAME
    snapshot.write_bytes(payload)
    return snapshot, sha256_bytes(payload)


def resolve_scenario_data(
    run_dir: Path,
    override: Optional[Path],
) -> tuple[Path, str]:
    """Resolve scenario data for a downstream run-processing command.

    Precedence is an explicit ``--data`` override, then the raw run's frozen
    snapshot. Legacy runs without a snapshot require an explicit override;
    silently falling back to mutable current data could change answer keys.
    """
    if override is not None:
        path = override
        source = "explicit_override"
    else:
        snapshot = run_dir / SCENARIO_SNAPSHOT_NAME
        if snapshot.is_file():
            path = snapshot
            source = "run_snapshot"
        else:
            raise FileNotFoundError(
                f"{run_dir} has no {SCENARIO_SNAPSHOT_NAME}; legacy runs require "
                "an explicit --data path to the exact scenario file used"
            )

    if not path.is_file():
        raise FileNotFoundError(f"scenario data not found: {path}")

    if source == "run_snapshot":
        manifest_path = run_dir / RUN_MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError(
                f"cannot verify {path}: missing {RUN_MANIFEST_NAME}; pass --data "
                "explicitly only after independently identifying the source file"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_sha256 = manifest["scenario_data"]["sha256"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"cannot verify {path}: invalid scenario hash in {manifest_path}"
            ) from exc
        actual_sha256 = sha256_file(path)
        if expected_sha256 != actual_sha256:
            raise ValueError(
                f"scenario snapshot hash mismatch for {path}: "
                f"manifest={expected_sha256}, actual={actual_sha256}"
            )
    return path, source


def snapshot_protocol_sources(
    repo_root: Path,
    run_dir: Path,
    relative_paths: Iterable[Path],
) -> dict[str, dict[str, str]]:
    """Freeze exact prompt/protocol source files and return their hashes."""
    destination_root = run_dir / "protocol_snapshot"
    result: dict[str, dict[str, str]] = {}
    for relative in relative_paths:
        source = (repo_root / relative).resolve()
        try:
            source.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(f"protocol source escapes repository: {relative}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"protocol source not found: {source}")
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        result[relative.as_posix()] = {
            "snapshot_path": destination.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(destination),
        }
    return result


def snapshot_tracked_git_diff(repo_root: Path, run_dir: Path) -> dict[str, object] | None:
    """Save the tracked binary Git diff so a dirty run is reconstructable."""
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    destination = run_dir / "git_tracked_changes.patch"
    destination.write_bytes(completed.stdout)
    return {
        "path": destination.name,
        "sha256": sha256_file(destination),
        "bytes": len(completed.stdout),
    }


def git_provenance(repo_root: Path) -> dict[str, object | None]:
    """Return best-effort Git commit/branch/dirty metadata without mutation."""

    def run_git(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    status = run_git("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "branch": branch or None,
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }
