"""Routing tests: these encode the structural guarantees the paper relies on.

If any of these fail, topology comparisons are invalid — stop and fix
before running any experiment.
"""

import pytest

from assumption_propagation.topologies import (
    AGENT_IDS,
    ALLOWED_SEEDS,
    DEFAULT_ROLES,
    STAR_HUB,
    TOPOLOGIES,
    exposure_round,
    senders_for,
    validate_roles,
    validate_seed,
    validate_topology,
)


@pytest.mark.parametrize("name", list(TOPOLOGIES))
def test_structural_sanity(name):
    """All agents defined, no self-messages, no duplicate or unknown senders."""
    validate_topology(name)


@pytest.mark.parametrize("name", list(TOPOLOGIES))
def test_round_zero_receives_nothing(name):
    """Round 0 is independent generation in every topology."""
    for agent in AGENT_IDS:
        assert senders_for(name, agent, 0) == []


@pytest.mark.parametrize("name", list(TOPOLOGIES))
def test_default_roles_are_valid(name):
    """Team-decided placements satisfy the placement policy everywhere."""
    seed, corrector = DEFAULT_ROLES[name]
    validate_roles(name, seed, corrector)


def test_seed_and_corrector_must_differ():
    with pytest.raises(ValueError):
        validate_roles("looped_chain", 0, 0)


def test_independent_never_communicates():
    for agent in AGENT_IDS:
        for rnd in range(4):
            assert senders_for("independent", agent, rnd) == []


def test_ring_each_agent_hears_exactly_its_predecessor():
    """Looped chain 0->1->2->3->0: agent i receives only from (i-1) mod 4."""
    for agent in AGENT_IDS:
        assert senders_for("looped_chain", agent, 1) == [(agent - 1) % 4]


def test_ring_assumption_walks_one_hop_per_round():
    exposure = exposure_round("looped_chain", seed_agent_id=0)
    assert exposure == {0: 0, 1: 1, 2: 2, 3: 3}


def test_ring_correction_reaches_seed_at_round_3():
    """Corrector at A1: correction travels 1->2->3->0, closing the loop in
    exactly 3 hops — the seed confronts the correction in the final round."""
    exposure = exposure_round("looped_chain", seed_agent_id=1)
    assert exposure[0] == 3


def test_star_leaves_never_communicate_directly():
    """Peripheral agents only exchange messages via the hub."""
    for leaf in (1, 2, 3):
        assert senders_for("neutral_star", leaf, 1) == [STAR_HUB]
    assert sorted(senders_for("neutral_star", STAR_HUB, 1)) == [1, 2, 3]


def test_star_roles_never_the_hub():
    seed, corrector = DEFAULT_ROLES["neutral_star"]
    assert STAR_HUB not in (seed, corrector)
    assert STAR_HUB not in ALLOWED_SEEDS["neutral_star"]
    with pytest.raises(ValueError):
        validate_seed("neutral_star", STAR_HUB)


def test_star_leaf_to_leaf_takes_two_rounds():
    """Seed at leaf 1: hub exposed round 1, other leaves round 2."""
    exposure = exposure_round("neutral_star", seed_agent_id=1)
    assert exposure == {1: 0, 0: 1, 2: 2, 3: 2}


def test_mesh_full_but_no_self_loops():
    for agent in AGENT_IDS:
        senders = senders_for("fully_connected_mesh", agent, 1)
        assert agent not in senders
        assert sorted(senders) == sorted(set(AGENT_IDS) - {agent})


def test_mesh_everyone_exposed_after_one_round():
    for seed in AGENT_IDS:
        exposure = exposure_round("fully_connected_mesh", seed_agent_id=seed)
        assert all(r == 1 for a, r in exposure.items() if a != seed)


def test_unknown_topology_rejected():
    with pytest.raises(KeyError):
        senders_for("directed_chain", 0, 1)  # replaced by looped_chain
