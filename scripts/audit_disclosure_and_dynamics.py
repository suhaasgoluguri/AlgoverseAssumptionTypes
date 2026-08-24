"""Reviewer-requested audits over a completed run.

Six analyses that the shipped statistics files do not already cover:

  B1  anonymity leak      does an anonymous-condition message name the
                          speaker's own assigned candidate?
  B3  tie exposure        deadlock rate by arm
  C1  trajectory          accuracy by round, by arm
  C2  seed recantation    does the seed abandon its false claim by round 3?
  C4  parse failures      which cells they fall in
  spread                  per-scenario distribution of the C1/C2 effect sizes

B1 is a manipulation check, not an outcome. Under `anonymous` a speaker is
labelled `Agent N` and nothing in the header states what they reviewed.

Measuring this needs care. The assigned entities ARE the answer options, so
"message names its own assigned entity" is ~99.9% under BOTH conditions --
it measures "discusses the candidates", which is the task, not disclosure.
That figure is still reported, labelled uninformative, because its symmetry
across conditions is the evidence that the naive measure says nothing.

The measure that carries signal is a first-person claim of standing: "my
review", "I was assigned", "assigned to me". Those appear in a small minority
of messages and are what actually tells a reader which candidate the speaker
held. Two are reported: any standing claim, and a standing claim occurring
within 120 characters of the speaker's own assigned entity.

All three are LOWER BOUNDS -- "the vendor I looked at" discloses the same
thing and matches nothing here. A judge model would catch paraphrase at the
cost of a non-reproducible dependency; these counts can be re-derived by
anyone holding the run directory.

Usage:
  uv run python scripts/audit_disclosure_and_dynamics.py <run-dir> [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

# First-person claims of standing. A speaker under `anonymous` who says "my
# review" tells a reader which candidate they held, which the header withholds.
STANDING = re.compile(
    r"\bmy (?:assignment|review|assigned|finding|report|notes?)\b"
    r"|\bI (?:was |am )?assigned\b"
    r"|\bassigned to me\b"
    r"|\bI (?:reviewed|assessed|checked|examined|evaluated|verified)\b",
    re.IGNORECASE,
)
LINK_WINDOW = 120

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assumption_propagation.prompts import assignment_label  # noqa: E402

N_AGENTS = 4


def load_assignments(run_dir: Path) -> dict[str, dict[str, str]]:
    """scenario_id -> {"seed": Entity, "corrector": Entity, "neutral": [E, E]}.

    Parsed with the same helper the runtime used to render the headers, so the
    entity strings are exactly those a reader would have seen.
    """
    raw = json.loads((run_dir / "scenario_snapshot.json").read_text(encoding="utf-8"))
    scenarios = raw if isinstance(raw, list) else raw["scenarios"]
    out: dict[str, dict[str, object]] = {}
    for sc in scenarios:
        hp = sc.get("hidden_profile")
        if not hp:
            continue
        out[sc["scenario_id"]] = {
            "seed": assignment_label(hp.get("seed_brief")),
            "corrector": assignment_label(hp.get("corrector_brief")),
            "neutral": [assignment_label(b) for b in (hp.get("neutral_briefs") or [])],
        }
    return out


def role_of(rec) -> str:
    if rec["agent_id"] == rec.get("seed_agent_id"):
        return "seed"
    if rec["agent_id"] == rec.get("corrector_agent_id"):
        return "verifier"
    return "ordinary"


def own_assignment(rec, assign) -> str | None:
    a = assign.get(rec["scenario_id"])
    if not a:
        return None
    role = role_of(rec)
    if role == "seed":
        return a["seed"]
    if role == "verifier":
        return a["corrector"]
    neutral_ids = [
        i for i in range(N_AGENTS)
        if i not in (rec.get("seed_agent_id"), rec.get("corrector_agent_id"))
    ]
    try:
        return a["neutral"][neutral_ids.index(rec["agent_id"])]
    except (ValueError, IndexError):
        return None


def cluster_boot(by_scenario, n=2000, seed=0):
    """Scenario-clustered bootstrap CI over per-scenario means."""
    keys = sorted(by_scenario)
    if not keys:
        return None
    rng = random.Random(seed)
    point = statistics.fmean([statistics.fmean(by_scenario[k]) for k in keys])
    draws = []
    for _ in range(n):
        pick = [rng.choice(keys) for _ in keys]
        draws.append(statistics.fmean([statistics.fmean(by_scenario[k]) for k in pick]))
    draws.sort()
    return {
        "point": point,
        "ci_low": draws[int(0.025 * n)],
        "ci_high": draws[int(0.975 * n)],
        "n_scenarios": len(keys),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    run = args.run_dir
    if not (run / "records.jsonl").is_file():
        sys.exit(
            f"{run}/records.jsonl not found. This audit needs the full "
            "per-agent-round transcripts, which are not distributed with the "
            "repository; request them alongside the paper."
        )

    assign = load_assignments(run)
    out: dict[str, object] = {"run": run.name}

    # ---------------- B1: anonymity leak, and C4: parse failures -------------
    # each bucket: [names_own, standing, linked, n]
    leak = collections.defaultdict(lambda: [0, 0, 0, 0])     # (arm, role)
    leak_cond = collections.defaultdict(lambda: [0, 0, 0, 0])  # attribution
    verifier_msgs = [0, 0, 0, 0]                             # frozen verifier round-1
    parse_cells = collections.Counter()
    parse_total = 0
    examples: list[dict] = []

    for line in (run / "records.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("parse_failed"):
            parse_total += 1
            parse_cells[(r.get("condition"), r.get("topology"),
                         r.get("attribution"), r.get("round"))] += 1
        text = r.get("rationale") or ""
        if not text:
            continue
        mine = own_assignment(r, assign)
        if not mine:
            continue
        m = re.search(rf"\b{re.escape(mine)}\b", text, re.IGNORECASE)
        names_own = m is not None
        standing = STANDING.search(text)
        # "linked": a standing claim sitting next to the speaker's own entity.
        linked = False
        if standing and m:
            for hit in re.finditer(rf"\b{re.escape(mine)}\b", text, re.IGNORECASE):
                for sc in STANDING.finditer(text):
                    if abs(sc.start() - hit.start()) <= LINK_WINDOW:
                        linked = True
                        break
                if linked:
                    break

        def bump(bucket):
            bucket[0] += names_own
            bucket[1] += standing is not None
            bucket[2] += linked
            bucket[3] += 1

        attribution = r.get("attribution")
        bump(leak_cond[attribution])
        if attribution == "anonymous":
            bump(leak[(r.get("condition"), role_of(r))])
            if linked and len(examples) < 6:
                examples.append({
                    "scenario_id": r["scenario_id"], "condition": r.get("condition"),
                    "role": role_of(r), "round": r.get("round"),
                    "assigned": mine, "rationale": text[:240],
                })
            if role_of(r) == "verifier" and r.get("round") == 1:
                bump(verifier_msgs)

    def rates(v):
        n = v[3]
        return {
            "n": n,
            "names_own_assignment": v[0],
            "names_own_assignment_rate": (v[0] / n) if n else None,
            "standing_claim": v[1],
            "standing_claim_rate": (v[1] / n) if n else None,
            "standing_claim_linked_to_own": v[2],
            "standing_claim_linked_to_own_rate": (v[2] / n) if n else None,
        }

    out["B1_anonymity_leak"] = {
        "method": {
            "names_own_assignment": "speaker's own assigned entity appears in the "
                "message. UNINFORMATIVE: the assigned entities are the answer "
                "options, so this is ~1.0 under both conditions and measures "
                "'discusses the candidates', not disclosure. Reported because "
                "its symmetry across conditions is the evidence for that.",
            "standing_claim": "first-person claim of standing ('my review', 'I "
                "was assigned', 'assigned to me'). This is the disclosure "
                "measure.",
            "standing_claim_linked_to_own": "a standing claim within "
                f"{LINK_WINDOW} characters of the speaker's own assigned entity; "
                "the strongest literal evidence a reader could identify the "
                "speaker's assignment from content alone.",
            "bound": "all three are lower bounds; paraphrase is not matched",
        },
        "by_attribution": {
            k: rates(v) for k, v in sorted(leak_cond.items(), key=lambda kv: str(kv[0]))
        },
        "anonymous_by_arm_role": {
            f"{a}|{ro}": rates(v)
            for (a, ro), v in sorted(leak.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))
        },
        "frozen_verifier_round1_anonymous": rates(verifier_msgs),
        "examples_linked": examples,
    }
    out["C4_parse_failures"] = {
        "total": parse_total,
        "by_cell": [
            {"condition": c, "topology": t, "attribution": at, "round": rd, "n": n}
            for (c, t, at, rd), n in sorted(parse_cells.items(), key=lambda kv: -kv[1])
        ],
    }

    # -------- B3 ties, C1 trajectory, C2 recantation, per-scenario spread ----
    conv_path = run / "analysis" / "conversations.jsonl"
    if conv_path.is_file():
        ties = collections.defaultdict(lambda: [0, 0])
        recant = collections.defaultdict(lambda: [0, 0])
        traj = collections.defaultdict(list)
        acc_by_scen = collections.defaultdict(lambda: collections.defaultdict(list))
        for line in conv_path.open(encoding="utf-8"):
            c = json.loads(line)
            arm = c.get("condition")
            ties[arm][1] += 1
            ties[arm][0] += bool(c.get("final_no_consensus"))
            if c.get("seed_recanted") is not None:
                recant[arm][1] += 1
                recant[arm][0] += bool(c.get("seed_recanted"))
            r = c.get("accuracy_by_round")
            if isinstance(r, list) and r:
                traj[arm].append(r)
            acc = c.get("ordinary_neutral_final_accuracy")
            if acc is not None:
                acc_by_scen[arm][c["scenario_id"]].append(acc)

        out["B3_tie_rate_by_arm"] = {
            k: {"no_consensus": v[0], "n": v[1], "rate": v[0] / v[1] if v[1] else None}
            for k, v in sorted(ties.items(), key=lambda kv: str(kv[0]))
        }
        out["C2_seed_recantation_by_arm"] = {
            k: {"recanted": v[0], "n": v[1], "rate": v[0] / v[1] if v[1] else None}
            for k, v in sorted(recant.items(), key=lambda kv: str(kv[0]))
        }
        out["C1_accuracy_by_round"] = {
            k: {
                "n_conversations": len(rows),
                "mean_by_round": [
                    statistics.fmean([r[i] for r in rows if len(r) > i])
                    for i in range(max(len(r) for r in rows))
                ],
            }
            for k, rows in sorted(traj.items(), key=lambda kv: str(kv[0]))
        }
        out["per_scenario_spread"] = {
            k: {
                "metric": "ordinary_neutral_final_accuracy, per-scenario mean",
                "min": min(m), "median": statistics.median(m), "max": max(m),
                "n_scenarios": len(m),
                "clustered": cluster_boot(by_scen),
            }
            for k, by_scen in sorted(acc_by_scen.items(), key=lambda kv: str(kv[0]))
            for m in [[statistics.fmean(v) for v in by_scen.values()]]
        }

    text = json.dumps(out, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    else:
        print(text)


if __name__ == "__main__":
    main()
