"""Regression gates for the active hidden-profile scenario corpus."""

from __future__ import annotations

import json

from scripts.build_hidden_profile_scenarios import (
    DEFAULT_OUT,
    EXCLUDED,
    SPECS,
    build_scenario,
    check_specs,
    entity_for,
)
from scripts.corpus_balance import trap_is_metric_optimum
from assumption_propagation.prompts import assignment_label


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


def test_every_active_trap_is_the_public_metric_optimum() -> None:
    failures = []
    for spec in _active_specs():
        trap_entity = entity_for(spec, spec.favored_option)
        if trap_is_metric_optimum(spec, trap_entity) is not True:
            failures.append(spec.scenario_id)

    assert failures == []


def test_every_active_hidden_profile_assignment_has_an_attribution_label() -> None:
    actual = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    failures = []
    for scenario in actual:
        profile = scenario["hidden_profile"]
        briefs = [
            profile["seed_brief"],
            profile["corrector_brief"],
            *profile["neutral_briefs"],
        ]
        for role_index, brief in enumerate(briefs):
            if assignment_label(brief) is None:
                failures.append((scenario["scenario_id"], role_index))

    assert failures == []
