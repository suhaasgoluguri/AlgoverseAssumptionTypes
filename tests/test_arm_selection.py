"""`--arms`: explicit arm subsets for cross-topology collection.

The corrector's message reaches 3 of 3 peers in the mesh but 1 of 3 in the chain
and star, so any treatment carried by that message lands on a different number of
agents. Two arms are unaffected because their delivery bypasses conversation
entirely: `injection_no_correction` delivers nothing, and `injection_broadcast`
writes the record into every dossier directly.

`--arms` exists so those two can be collected in every topology through the
matched-triple path, while the conversation-carried arms stay in the mesh-only
ladder. The planned grid feeds the completion-integrity check, so a mismatch here
silently changes what counts as a complete run.
"""

import argparse

import pytest

from scripts.run_pilot import planned_injection_conditions
from assumption_propagation.schemas import CORRECTION_SCOPE


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        skip_injection=False,
        scope_ladder=False,
        scope_protocol=None,
        scope_controls=False,
        ablation_arm=False,
        withhold_corrective=False,
        arms=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_default_is_the_single_injection_arm():
    assert planned_injection_conditions(_args()) == ("injection",)


def test_explicit_arms_become_the_planned_grid():
    """The planned grid drives the integrity check; it must match the request."""
    chosen = ("injection_no_correction", "injection_broadcast")
    assert planned_injection_conditions(_args(arms=list(chosen))) == chosen


def test_arms_takes_precedence_over_the_ablation_default():
    got = planned_injection_conditions(
        _args(arms=["injection_broadcast"], ablation_arm=True)
    )
    assert got == ("injection_broadcast",)


def test_arms_order_is_preserved():
    """Order is part of the plan: the grid is compared as a tuple."""
    a = planned_injection_conditions(_args(arms=["injection", "injection_relay"]))
    b = planned_injection_conditions(_args(arms=["injection_relay", "injection"]))
    assert a == ("injection", "injection_relay")
    assert b == ("injection_relay", "injection")


def test_ablation_and_withhold_still_work():
    assert planned_injection_conditions(_args(ablation_arm=True)) == (
        "injection", "injection_no_correction",
    )
    assert planned_injection_conditions(_args(withhold_corrective=True)) == (
        "injection_no_correction",
    )


def test_skip_injection_wins_over_everything():
    assert planned_injection_conditions(
        _args(skip_injection=True, arms=["injection_broadcast"])
    ) == ()


@pytest.mark.parametrize("arm", CORRECTION_SCOPE)
def test_every_prompt_v1_arm_is_selectable(arm):
    """`--arms` is restricted to prompt_v1 conditions; all four must be valid."""
    assert planned_injection_conditions(_args(arms=[arm])) == (arm,)


def test_topology_invariant_arms_are_the_intended_pair():
    """Documents *why* these two, so the rationale survives the next refactor.

    Delivery for these does not depend on who talks to whom: one delivers
    nothing, the other writes into every agent's dossier. The other two ride on
    the corrector's outgoing message and so vary with the network.
    """
    invariant = {"injection_no_correction", "injection_broadcast"}
    conversation_carried = {"injection", "injection_relay"}
    assert invariant | conversation_carried == set(CORRECTION_SCOPE)
    assert planned_injection_conditions(_args(arms=sorted(invariant))) == tuple(
        sorted(invariant)
    )
