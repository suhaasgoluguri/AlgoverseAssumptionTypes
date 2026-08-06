"""Static guardrails for the frozen 33-scenario confirmatory launcher."""

from pathlib import Path

from assumption_propagation.reproducibility import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "outputs" / "validation" / "run_tool_v2_azure_33.ps1"
SCENARIOS = REPO_ROOT / "data" / "scenarios_hidden_profile_candidates.json"
EXPECTED_SCENARIO_SHA256 = (
    "7f0f31556b2b9543815eac77272ab00f91804deb3128d33b5c7c12d48ac07fbf"
)


def test_confirmatory_runner_collects_the_full_factorial() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '$attribution = "both"' in text
    assert '$deployment -ne "gpt-5.4-mini"' in text
    assert '$manifest.client.model_identifier -eq "azure:gpt-5.4-mini"' in text
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
