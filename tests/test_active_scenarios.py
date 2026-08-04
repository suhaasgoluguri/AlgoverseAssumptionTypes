"""Regression gates for the active hidden-profile scenario corpus."""

from __future__ import annotations

import json

from scripts.build_hidden_profile_scenarios import (
    DEFAULT_OUT,
    EXCLUDED,
    SPECS,
    build_scenario,
    check_specs,
)


def _active_specs():
    return [spec for spec in SPECS if spec.scenario_id not in EXCLUDED]


def test_active_scenario_corpus_matches_validated_default_builder_output() -> None:
    specs = _active_specs()
    assert check_specs(specs) == []

    expected = [build_scenario(spec) for spec in specs]
    actual = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    assert actual == expected


def test_active_contexts_do_not_contain_bad_analyst_article() -> None:
    actual = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    offenders = [
        scenario["scenario_id"]
        for scenario in actual
        if "A analyst's" in scenario["common_context"]
    ]
    assert offenders == []
