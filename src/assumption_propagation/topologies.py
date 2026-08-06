"""Communication topologies for the 4-agent propagation experiment.

Each topology is an adjacency dict: ``{receiver_id: [sender_ids]}``.
``TOPOLOGIES[name][agent]`` lists the agents whose *previous-round* outputs
this agent receives in each update round. Round 0 is always independent
(no agent receives peer messages in round 0).

Role placement policy (the seed's private assessment persists from round 0;
the corrector's decisive private record is released at round 1 and persists):
- looped_chain (ring 0->1->2->3->0): rotationally symmetric, so position is
  arbitrary; the design constant is the offset: corrector immediately
  downstream of the seed (gatekeeper configuration).
- neutral_star: seed and corrector are peripheral (never the hub); the hub
  is a neutral relay, not an authority.
- fully_connected_mesh: structurally symmetric; placement arbitrary.
- independent: control condition; no communication at all.

With four total rounds (0-3), the corrector first uses its newly released record
in its round-1 response, and recipients first see that response in round 2.
Thus the mesh exposes every peer by round 2; sparse-topology coverage and later
retransmission depend on the communication path. The confirmatory scope design
uses only the mesh so evidence route is not confounded with recipient coverage.
"""

from __future__ import annotations

import networkx as nx

N_AGENTS = 4
AGENT_IDS = tuple(range(N_AGENTS))

# {receiver: [senders]}
TOPOLOGIES: dict[str, dict[int, list[int]]] = {
    "independent": {0: [], 1: [], 2: [], 3: []},
    "looped_chain": {0: [3], 1: [0], 2: [1], 3: [2]},
    "neutral_star": {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]},
    "fully_connected_mesh": {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2],
    },
}

STAR_HUB = 0

# Which agents may be the seed in each topology.
ALLOWED_SEEDS: dict[str, tuple[int, ...]] = {
    "independent": AGENT_IDS,           # placement irrelevant; no messages flow
    "looped_chain": AGENT_IDS,          # ring is rotationally symmetric
    "neutral_star": (1, 2, 3),          # peripheral only, never the hub
    "fully_connected_mesh": AGENT_IDS,  # symmetric
}

# Team-decided (seed, corrector) placement per topology. The design
# constant is "corrector immediately downstream of the seed"; in the star
# both roles are peripheral so the hub stays neutral.
DEFAULT_ROLES: dict[str, tuple[int, int]] = {
    "independent": (0, 1),
    "looped_chain": (0, 1),          # ring: corrector one step downstream
    "neutral_star": (1, 2),          # hub (0) neutral; both roles peripheral
    "fully_connected_mesh": (0, 1),  # symmetric
}


def corrector_for_seed(topology_name: str, seed_agent_id: int) -> int:
    """Return the policy-compliant default corrector for a seed placement."""
    validate_seed(topology_name, seed_agent_id)
    if topology_name == "looped_chain":
        # The corrector is the unique receiver immediately downstream.
        return next(
            receiver
            for receiver, senders in TOPOLOGIES[topology_name].items()
            if seed_agent_id in senders
        )
    if topology_name == "neutral_star":
        return next(agent for agent in (1, 2, 3) if agent != seed_agent_id)
    return next(agent for agent in AGENT_IDS if agent != seed_agent_id)


def validate_roles(topology_name: str, seed_agent_id: int, corrector_agent_id: int) -> None:
    """Seed and corrector must be distinct, valid, and policy-compliant."""
    validate_seed(topology_name, seed_agent_id)
    if corrector_agent_id not in AGENT_IDS:
        raise ValueError(f"corrector_agent_id must be in {AGENT_IDS}, got {corrector_agent_id}")
    if corrector_agent_id == seed_agent_id:
        raise ValueError("seed and corrector must be different agents")
    if topology_name == "looped_chain":
        expected = corrector_for_seed(topology_name, seed_agent_id)
        if corrector_agent_id != expected:
            raise ValueError(
                "looped_chain corrector must be immediately downstream of the seed: "
                f"expected {expected}, got {corrector_agent_id}"
            )
    if topology_name == "neutral_star" and corrector_agent_id == 0:
        raise ValueError("neutral_star hub agent 0 must remain neutral")


def senders_for(topology_name: str, agent_id: int, round_index: int) -> list[int]:
    """Return the agent ids whose previous-round outputs `agent_id` receives.

    This is the single routing entry point used by the orchestrator.
    Round 0 receives nothing by definition (there is no previous round).
    """
    if topology_name not in TOPOLOGIES:
        raise KeyError(f"Unknown topology {topology_name!r}; expected one of {list(TOPOLOGIES)}")
    if agent_id not in AGENT_IDS:
        raise ValueError(f"agent_id must be in {AGENT_IDS}, got {agent_id}")
    if round_index == 0:
        return []
    return list(TOPOLOGIES[topology_name][agent_id])


def validate_seed(topology_name: str, seed_agent_id: int) -> None:
    """Raise if the seed placement violates the design's placement policy."""
    allowed = ALLOWED_SEEDS[topology_name]
    if seed_agent_id not in allowed:
        raise ValueError(
            f"seed_agent_id={seed_agent_id} not allowed for {topology_name!r}; "
            f"allowed seeds: {allowed}"
        )


def as_digraph(topology_name: str) -> nx.DiGraph:
    """Directed graph with edges sender -> receiver, for structural checks."""
    g = nx.DiGraph()
    g.add_nodes_from(AGENT_IDS)
    for receiver, senders in TOPOLOGIES[topology_name].items():
        for sender in senders:
            g.add_edge(sender, receiver)
    return g


def validate_topology(topology_name: str) -> None:
    """Structural sanity checks; raise AssertionError on violation."""
    adj = TOPOLOGIES[topology_name]
    assert set(adj) == set(AGENT_IDS), f"{topology_name}: must define all agents"
    for receiver, senders in adj.items():
        assert receiver not in senders, f"{topology_name}: agent {receiver} receives from itself"
        assert len(senders) == len(set(senders)), (
            f"{topology_name}: duplicate senders for {receiver}"
        )
        assert all(s in AGENT_IDS for s in senders), f"{topology_name}: unknown sender id"


def exposure_round(topology_name: str, seed_agent_id: int) -> dict[int, int | None]:
    """Earliest round each agent can first *receive* the seed's message.

    Useful for analysis (first-exposure metrics) and for tests. The seed
    itself is exposed at round 0 (its own frozen output). Agents that can
    never receive it get None.
    """
    g = as_digraph(topology_name)
    result: dict[int, int | None] = {}
    for agent in AGENT_IDS:
        if agent == seed_agent_id:
            result[agent] = 0
            continue
        try:
            hops = nx.shortest_path_length(g, source=seed_agent_id, target=agent)
            result[agent] = hops  # one edge traversed per update round
        except nx.NetworkXNoPath:
            result[agent] = None
    return result
