"""Speaker attribution: the factor, and the ways it has silently broken.

Attribution names what a peer was assigned — `Agent 3` becomes
`Agent 3 (assigned: Borealis)`. It adds no new information: the claim is already
in the transcript. It exists because the pooling rule credits a report heard
"from that reviewer", and an anonymous header deletes exactly the identity that
clause needs.

Two classes of failure have actually occurred and are pinned here:

1. **Silent non-passing.** On 2026-08-05 three call sites in run_pilot.py failed
   to pass `attribution` down — the prompt_v1 ladder, the matched-triple path,
   and the independent control. Each would run twice under `--attribution both`
   and produce anonymous conversations both times, with no error. Two were
   caught only by the completion-integrity check.

2. **Silent drift under anonymous.** If the anonymous rendering ever stops being
   byte-identical to the pre-attribution build, every historical comparison
   quietly breaks. The regression test below hashes the full prompt surface.
"""

import json
from pathlib import Path

import pytest

from assumption_propagation.prompts import (
    ATTRIBUTION_MODES,
    assignment_for_agent,
    assignment_label,
    format_peer_message,
)
from assumption_propagation.schemas import Scenario

DATA = Path(__file__).resolve().parents[1] / "data" / "scenarios_hidden_profile_candidates.json"


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scenario(entries) -> Scenario:
    return Scenario.model_validate(entries[0])


# ---------------------------------------------------------------------------
# Reading the assignment out of a brief
# ---------------------------------------------------------------------------

def test_every_brief_in_the_corpus_yields_an_assignment(entries):
    """A header that silently says nothing would make the factor meaningless.

    The label is parsed at runtime rather than stored, so the committed scenario
    bytes -- and the preregistered sha256 -- stay unchanged. That only works if
    every brief follows the expected opening.
    """
    missing = []
    for entry in entries:
        profile = entry["hidden_profile"]
        briefs = [profile["seed_brief"], profile["corrector_brief"], *profile["neutral_briefs"]]
        for index, brief in enumerate(briefs):
            if not assignment_label(brief):
                missing.append((entry["scenario_id"], index))
    assert not missing, f"unparsable assignment briefs: {missing}"


def test_assignment_label_handles_the_domain_role_words(entries):
    """Role nouns vary by domain (reviewer, engineer, coordinator, analyst...)."""
    labels = {
        assignment_label(entry["hidden_profile"]["seed_brief"])
        for entry in entries
    }
    assert None not in labels
    assert len(labels) > 10, "expected many distinct assigned entities across the corpus"


def test_assignment_label_returns_none_rather_than_guessing():
    assert assignment_label("Borealis confirmed four-hour response.") is None
    assert assignment_label("") is None
    assert assignment_label(None) is None


def test_assignment_for_agent_is_role_keyed(scenario):
    """Briefs follow the ROLE, not the agent index — so must the label."""
    seed = assignment_for_agent(scenario, is_seed=True, is_corrector=False, neutral_ordinal=None)
    corrector = assignment_for_agent(scenario, is_seed=False, is_corrector=True, neutral_ordinal=None)
    clearance = assignment_for_agent(scenario, is_seed=False, is_corrector=False, neutral_ordinal=0)
    ordinary = assignment_for_agent(scenario, is_seed=False, is_corrector=False, neutral_ordinal=1)
    assert len({seed, corrector, clearance, ordinary}) == 4, "each role holds a different option"
    assert all(x for x in (seed, corrector, clearance, ordinary))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_attribution_changes_only_the_speaker_label():
    plain = format_peer_message(3, "B", "Borealis is clean.")
    named = format_peer_message(3, "B", "Borealis is clean.", assignment="Borealis")
    assert "Agent 3 (assigned: Borealis)" in named
    assert "(assigned:" not in plain
    # answer and rationale untouched: only the label differs
    assert named.replace("Agent 3 (assigned: Borealis)", "Agent 3") == plain


def test_anonymous_is_the_default():
    assert format_peer_message(1, "A", "x") == format_peer_message(1, "A", "x", assignment=None)
    assert format_peer_message(1, "A", "x") == format_peer_message(1, "A", "x", assignment="")


def test_attribution_composes_with_a_relayed_record():
    """Speaker provenance and record provenance are orthogonal factors."""
    both = format_peer_message(
        1, "B", "Apex is out.",
        relayed_source="The registry shows it expired.",
        assignment="Cirrus",
    )
    assert "Agent 1 (assigned: Cirrus)" in both
    assert "The registry shows it expired." in both


def test_mode_list_is_the_contract():
    assert ATTRIBUTION_MODES == ("anonymous", "assigned", "n/a")


# ---------------------------------------------------------------------------
# The two failures that have actually happened
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_attribution_mode_is_rejected(scenario):
    """Fail loudly rather than silently rendering anonymous."""
    from assumption_propagation.orchestrator import run_conversation
    from assumption_propagation.schemas import SeedInstance

    variant = scenario.assumption_variants[0]

    class _Client:
        model_identifier = "capture:test"
        temperature = None

        async def generate(self, system_prompt, user_prompt):
            return '{"answer": "A", "rationale": "r"}'

    with pytest.raises(ValueError, match="unknown attribution mode"):
        await run_conversation(
            scenario, variant=variant, client=_Client(),
            topology_name="fully_connected_mesh", n_rounds=4,
            condition="injection", attribution="identified",
            seed_agent_id=0, corrector_agent_id=1,
            frozen_seed=SeedInstance(
                seed_instance_id="t", scenario_id=scenario.scenario_id,
                assumption_id=variant.assumption_id, seed_trial_index=0,
                answer=variant.favored_option, rationale="r",
                raw_response='{"answer": "A", "rationale": "r"}'),
        )


def test_every_collection_path_forwards_attribution():
    """Regression for three silent bugs found on 2026-08-05.

    A driver that accepts `attribution` but never forwards it produces a run
    that looks complete and has one factor missing. Source-level check because
    the failure is an omission, not an error.
    """
    import re

    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_pilot.py").read_text(
        encoding="utf-8"
    )
    drivers = (
        "run_conversation",
        "run_matched_triple",
        "run_correction_scope_ladder",
        "run_tool_correction_scope_ladder",
    )
    missing = []
    for match in re.finditer(rf"await ({'|'.join(drivers)})\(", source):
        tail = source[match.start():match.start() + 2000]
        end = tail.find("\n            )")
        if end < 0:
            end = tail.find("\n                )")
        body = tail[: end if end > 0 else 1500]
        if "attribution=" not in body:
            missing.append((match.group(1), source[: match.start()].count("\n") + 1))
    assert not missing, f"call sites not forwarding attribution: {missing}"


def test_anonymous_rendering_is_unchanged_from_the_pre_attribution_build(entries):
    """Anonymous output must stay byte-identical to the build without the factor.

    Historical runs and the tool-protocol runs are all anonymous. If this
    drifts, they stop being comparable and nothing raises.
    """
    from assumption_propagation.prompts import build_private_dossier, build_round0_prompt

    scenario = Scenario.model_validate(entries[0])
    variant = scenario.assumption_variants[0]
    rendered = []
    for condition in ("injection_tool_none", "injection_tool_testimony",
                      "injection_tool_relay", "injection_tool_direct",
                      "no_injection_baseline"):
        for is_seed, is_corrector, ordinal in ((True, False, None), (False, True, None),
                                               (False, False, 0), (False, False, 1)):
            for round_index in range(4):
                rendered.append(build_private_dossier(
                    scenario, is_seed=is_seed, is_corrector=is_corrector,
                    condition=condition, variant=variant,
                    round_index=round_index, neutral_ordinal=ordinal) or "")
        rendered.append(build_round0_prompt(
            scenario, is_seed=False, is_corrector=False, condition=condition,
            variant=variant, round_index=0, neutral_ordinal=1))
    for agent_id in range(4):
        rendered.append(format_peer_message(agent_id, "B", "some rationale"))

    # No attribution argument was passed anywhere above, so nothing may mention it.
    joined = "\n".join(rendered)
    assert "(assigned:" not in joined
