"""Static authoring gates for the fixed scenario package."""

from __future__ import annotations

import json

from scripts.audit_scenarios import (
    DEFAULT_SCENARIO_PATH,
    _contains_explicit_unknown_rule,
    _match_option_to_bullet,
    audit_file,
    audit_scenarios,
)


def _fresh_fixed_payload() -> list[dict]:
    """Load an independent copy so negative fixtures cannot mutate each other."""
    return json.loads(DEFAULT_SCENARIO_PATH.read_text(encoding="utf-8"))


def _audit_codes(payload: list[dict]) -> set[str]:
    return {finding.code for finding in audit_scenarios(payload)}


def test_scenarios_fixed_passes_deterministic_authoring_audit() -> None:
    findings = audit_file(DEFAULT_SCENARIO_PATH)
    rendered = "\n".join(f"- {finding}" for finding in findings)
    assert not findings, f"scenario authoring audit failed:\n{rendered}"


def test_unknown_gap_rule_requires_both_omission_and_unknown_semantics() -> None:
    assert _contains_explicit_unknown_rule(
        "A missing status entry is unresolved and does not establish that the feature is absent."
    )
    assert not _contains_explicit_unknown_rule("The status entry is missing from the table.")
    assert not _contains_explicit_unknown_rule("Availability is unknown.")


def test_public_list_matcher_refuses_ambiguous_entity_matches() -> None:
    bullets = [
        "Northstar Hotel: accessible room at $210.",
        "Riverside Hotel: accessible room at $205.",
    ]
    assert _match_option_to_bullet("Book Riverside Hotel at $205.", bullets) == 1
    assert _match_option_to_bullet("Book an accessible hotel room.", bullets) is None


def test_audit_rejects_self_undermining_seed_language() -> None:
    payload = _fresh_fixed_payload()
    assessment = payload[0]["assumption_variants"][0]["seed_private_assessment"]
    assessment["content"] += " This is likely correct."
    assert "seed.weak_language" in _audit_codes(payload)


def test_audit_rejects_early_correction_release() -> None:
    payload = _fresh_fixed_payload()
    correction = payload[0]["assumption_variants"][0]["corrective_evidence"]
    correction["available_round"] = 0
    assert "correction.release_round" in _audit_codes(payload)


def test_audit_rejects_treatment_length_imbalance() -> None:
    payload = _fresh_fixed_payload()
    assessment = payload[0]["assumption_variants"][0]["seed_private_assessment"]
    assessment["content"] += (
        " This categorical assessment controls the complete option comparison and all "
        "stated operational requirements for today's final selection decision without "
        "any additional qualification."
    )
    assert "treatment.length_imbalance" in _audit_codes(payload)


def test_audit_rejects_fabricated_private_artifact() -> None:
    payload = _fresh_fixed_payload()
    assessment = payload[0]["assumption_variants"][0]["seed_private_assessment"]
    assessment["content"] += " A private email confirms this conclusion."
    assert "seed.fake_artifact" in _audit_codes(payload)
