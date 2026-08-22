"""Static guardrails for the frozen 33-scenario confirmatory launcher."""

from pathlib import Path

from assumption_propagation.reproducibility import canonical_json_sha256, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_tool_v2_azure_33.ps1"
SCENARIOS = REPO_ROOT / "data" / "scenarios_hidden_profile_candidates.json"
EXPECTED_SCENARIO_SHA256 = (
    "880cf414d5d67a9347d5ffb8af6537920ef3b18e775ec1a7e8141e6ce85b1917"
)
EXPECTED_SCENARIO_CANONICAL_SHA256 = (
    "653fe93f821846aca67748f0004dda3bb2c02ae9280382ed1c990d41ba15bea1"
)


def test_confirmatory_runner_collects_the_full_factorial() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '$attribution = "both"' in text
    assert '$deployment -ne "gpt-5.4-mini"' in text
    assert '$manifest.client.model_identifier -eq "azure:gpt-5.4-mini"' in text
    assert '"--max-completion-tokens", "1000"' in text
    assert '$attributionModes = @("anonymous", "assigned")' in text
    assert "$replayableSeeds" in text
    assert "$expectedInjectionPerCoordinate = $replayableSeeds * $nArms" in text
    assert (
        '$topologies = @("looped_chain", "neutral_star", "fully_connected_mesh")'
        in text
    )
    assert '"--attribution", $attribution' in text
    assert '$manifest.protocol.attribution -eq $attribution' in text
    assert '$manifest.protocol.scope_primary_attribution -eq "assigned"' in text
    assert '$manifest.protocol.scope_primary_topology -eq "fully_connected_mesh"' in text
    assert "$badRecordAttribution -eq 0" in text
    assert "$badResultAttribution -eq 0" in text
    assert "$coordinateMismatches -eq 0" in text
    assert "--by-attribution" in text
    assert "--factorial" in text


def test_confirmatory_runner_pins_the_active_scenario_bytes() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert EXPECTED_SCENARIO_SHA256 in text
    assert sha256_file(SCENARIOS) == EXPECTED_SCENARIO_SHA256


def test_confirmatory_runner_gates_on_canonical_scenario_content() -> None:
    """The gate must be the content hash, not the formatting-sensitive one.

    A raw byte hash cannot pass on both platforms at once: a Windows checkout
    with ``core.autocrlf=true`` rewrites line endings and changes it without
    changing a single scenario.
    """
    text = RUNNER.read_text(encoding="utf-8")

    assert EXPECTED_SCENARIO_CANONICAL_SHA256 in text
    assert canonical_json_sha256(SCENARIOS) == EXPECTED_SCENARIO_CANONICAL_SHA256
    assert "$manifest.scenario_data.canonical_sha256 -eq" in text
    # The raw hash is provenance, so a mismatch warns rather than throwing.
    assert "Write-Warning" in text
