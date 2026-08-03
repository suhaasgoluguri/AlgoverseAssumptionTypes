"""Track B (hidden profile) tests.

The scientific claim Track B rests on is that no agent can identify the
compliant option alone: each holds one disqualifying finding, and only the
pooled set resolves the decision. These tests assert the mechanics that
claim depends on — every agent gets a DIFFERENT brief, briefs follow the
ROLE rather than the agent index, and Track A behaviour is untouched.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_propagation.clients import MockClient
from assumption_propagation.orchestrator import run_conversation
from assumption_propagation.prompts import build_private_dossier
from assumption_propagation.schemas import HiddenProfile, Scenario

DATA = Path("data/scenarios_hidden_profile_candidates.json")


def load_track_b() -> list[Scenario]:
    entries = json.loads(DATA.read_text(encoding="utf-8"))
    return [Scenario.model_validate(e) for e in entries]


@pytest.fixture
def scenario() -> Scenario:
    return next(s for s in load_track_b() if s.scenario_id == "vendor_001")


@pytest.fixture
def variant(scenario):
    return scenario.assumption_variants[0]


# ---------------------------------------------------------------------------
# Shipped Track B data
# ---------------------------------------------------------------------------

def test_shipped_scenarios_are_hidden_profiles():
    scenarios = load_track_b()
    assert scenarios, "Track B data file is empty"
    for s in scenarios:
        assert s.track == "hidden_profile"
        assert s.hidden_profile is not None


def test_every_agent_holds_a_distinct_brief():
    """The pooling premise: four agents, four different facts."""
    for s in load_track_b():
        profile = s.hidden_profile
        briefs = [profile.seed_brief, profile.corrector_brief, *profile.neutral_briefs]
        assert len(briefs) == 4
        assert len(set(briefs)) == 4, f"{s.scenario_id}: duplicated brief"


def test_favored_option_is_never_correct_and_letters_vary():
    """Letter rotation guards against confounding trap uptake with position."""
    scenarios = load_track_b()
    for s in scenarios:
        for v in s.assumption_variants:
            assert v.favored_option != s.correct_answer
    assert len({s.correct_answer for s in scenarios}) > 1
    assert len({v.favored_option for s in scenarios for v in s.assumption_variants}) > 1


# ---------------------------------------------------------------------------
# HiddenProfile model
# ---------------------------------------------------------------------------

def make_profile(**overrides) -> dict:
    base = {
        "seed_brief": "Seed reviewed vendor one and found nothing disqualifying.",
        "corrector_brief": "Corrector reviewed vendor two and found a lapsed certificate.",
        "neutral_briefs": [
            "First neutral reviewed vendor three and found a residency breach.",
            "Second neutral reviewed vendor four and found an excluded fee.",
        ],
        "shared_context_sufficient": False,
    }
    base.update(overrides)
    return base


def test_brief_for_maps_each_role():
    profile = HiddenProfile.model_validate(make_profile())
    assert profile.brief_for(is_seed=True, is_corrector=False) == profile.seed_brief
    assert profile.brief_for(is_seed=False, is_corrector=True) == profile.corrector_brief
    for ordinal in (0, 1):
        assert profile.brief_for(
            is_seed=False, is_corrector=False, neutral_ordinal=ordinal
        ) == profile.neutral_briefs[ordinal]


def test_neutral_without_ordinal_is_an_error():
    profile = HiddenProfile.model_validate(make_profile())
    with pytest.raises(ValueError):
        profile.brief_for(is_seed=False, is_corrector=False)


def test_out_of_range_ordinal_rejected():
    profile = HiddenProfile.model_validate(make_profile())
    with pytest.raises(ValueError):
        profile.brief_for(is_seed=False, is_corrector=False, neutral_ordinal=2)


def test_shared_context_sufficient_must_be_false():
    """A scenario solvable alone is not a hidden profile."""
    with pytest.raises(ValueError):
        HiddenProfile.model_validate(make_profile(shared_context_sufficient=True))


def test_requires_exactly_two_neutral_briefs():
    with pytest.raises(ValueError):
        HiddenProfile.model_validate(make_profile(neutral_briefs=["only one brief here"]))


# ---------------------------------------------------------------------------
# Dossier assembly
# ---------------------------------------------------------------------------

def dossier(scenario, variant, *, is_seed=False, is_corrector=False,
            neutral_ordinal=None, condition="injection", round_index=0):
    return build_private_dossier(
        scenario,
        is_seed=is_seed,
        is_corrector=is_corrector,
        condition=condition,
        variant=variant,
        round_index=round_index,
        neutral_ordinal=neutral_ordinal,
    )


def test_seed_dossier_carries_brief_and_injection_under_one_header(scenario, variant):
    text = dossier(scenario, variant, is_seed=True)
    assert scenario.hidden_profile.seed_brief in text
    assert variant.seed_private_assessment.content in text
    # One header only: an agent must not be able to infer its role from the
    # shape of its own prompt.
    assert text.count("PRIVATE DOSSIER") == 1


def test_neutral_receives_only_its_own_brief(scenario, variant):
    text = dossier(scenario, variant, neutral_ordinal=0)
    assert scenario.hidden_profile.neutral_briefs[0] in text
    assert scenario.hidden_profile.neutral_briefs[1] not in text
    assert scenario.hidden_profile.seed_brief not in text
    assert variant.seed_private_assessment.content not in text
    assert variant.corrective_evidence.content not in text


def test_corrector_gets_brief_immediately_and_correction_on_schedule(scenario, variant):
    available = variant.corrective_evidence.available_round
    assert available > 0, "this test needs delayed corrective evidence"

    early = dossier(scenario, variant, is_corrector=True, round_index=available - 1)
    assert scenario.hidden_profile.corrector_brief in early
    assert variant.corrective_evidence.content not in early

    late = dossier(scenario, variant, is_corrector=True, round_index=available)
    assert scenario.hidden_profile.corrector_brief in late
    assert variant.corrective_evidence.content in late


def test_baseline_seed_gets_brief_and_neutral_note_not_injection(scenario, variant):
    text = dossier(scenario, variant, is_seed=True, condition="no_injection_baseline")
    assert scenario.hidden_profile.seed_brief in text
    assert scenario.neutral_control_note in text
    assert variant.seed_private_assessment.content not in text


def test_track_a_scenario_leaves_neutrals_with_no_dossier():
    """Backward compatibility: no hidden profile => unchanged Track A behaviour."""
    track_a = SimpleNamespace(neutral_control_note="Quotes are in USD.")
    variant = SimpleNamespace(
        seed_private_assessment=SimpleNamespace(content="Private assessment: X holds."),
        seed_local_evidence=None,
        corrective_evidence=SimpleNamespace(content="X does not hold.", available_round=0),
    )
    assert build_private_dossier(
        track_a, is_seed=False, is_corrector=False,
        condition="injection", variant=variant, round_index=0,
    ) is None


# ---------------------------------------------------------------------------
# Role-keyed placement across topologies
# ---------------------------------------------------------------------------

async def collect_round0_prompts(scenario, variant, topology_name):
    """Round-0 user prompts by agent id, in the no-injection baseline where
    all four agents generate (nobody is replayed)."""
    mock = MockClient()
    result = await run_conversation(
        scenario, client=mock, topology_name=topology_name,
        condition="no_injection_baseline", variant=variant,
    )
    first = result.records[0]
    round0 = [r for r in result.records if r.round == 0]
    assert len(round0) == 4
    # Round 0 generates agents in ascending id order.
    prompts = {agent_id: mock.calls[agent_id][1] for agent_id in range(4)}
    return prompts, first.seed_agent_id, first.corrector_agent_id


@pytest.mark.parametrize("topology_name", ["looped_chain", "neutral_star", "fully_connected_mesh"])
async def test_briefs_follow_the_role_in_every_topology(scenario, variant, topology_name):
    """Seed and corrector sit at different agent ids per topology; the brief
    must follow the role, or the matched triple compares different setups."""
    prompts, seed_id, corrector_id = await collect_round0_prompts(
        scenario, variant, topology_name
    )
    profile = scenario.hidden_profile

    assert profile.seed_brief in prompts[seed_id]
    assert profile.corrector_brief in prompts[corrector_id]

    neutral_ids = [a for a in range(4) if a not in (seed_id, corrector_id)]
    for ordinal, agent_id in enumerate(neutral_ids):
        assert profile.neutral_briefs[ordinal] in prompts[agent_id]


# ---------------------------------------------------------------------------
# withhold_corrective ablation (Track A's manipulation, usable in both tracks)
# ---------------------------------------------------------------------------

def test_withholding_corrective_keeps_the_brief_but_drops_the_correction(scenario, variant):
    """The ablation must remove only the disproof, not the corrector's role."""
    final_round = 3
    kept = dossier(scenario, variant, is_corrector=True, round_index=final_round)
    ablated = build_private_dossier(
        scenario, is_seed=False, is_corrector=True, condition="injection",
        variant=variant, round_index=final_round, withhold_corrective=True,
    )
    assert variant.corrective_evidence.content in kept
    assert variant.corrective_evidence.content not in ablated
    # The corrector still holds its hidden-profile fact, so the group's pooled
    # information is unchanged apart from the decisive correction itself.
    assert scenario.hidden_profile.corrector_brief in ablated


async def test_withheld_correction_is_recorded_as_not_visible(scenario, variant):
    """corrective_visible is the audit trail; it must not claim a correction
    the corrector never saw."""
    result = await run_conversation(
        scenario, client=MockClient(), topology_name="looped_chain",
        condition="no_injection_baseline", variant=variant,
        withhold_corrective=True,
    )
    assert not any(r.corrective_visible for r in result.records)

    control = await run_conversation(
        scenario, client=MockClient(), topology_name="looped_chain",
        condition="no_injection_baseline", variant=variant,
    )
    assert any(r.corrective_visible for r in control.records)


@pytest.mark.parametrize("topology_name", ["looped_chain", "neutral_star", "fully_connected_mesh"])
async def test_all_four_briefs_are_distributed_exactly_once(scenario, variant, topology_name):
    """Pooling is only possible if the group collectively holds every fact."""
    prompts, _, _ = await collect_round0_prompts(scenario, variant, topology_name)
    profile = scenario.hidden_profile
    for brief in [profile.seed_brief, profile.corrector_brief, *profile.neutral_briefs]:
        holders = [a for a, p in prompts.items() if brief in p]
        assert len(holders) == 1, f"brief held by {holders}, expected exactly one agent"
