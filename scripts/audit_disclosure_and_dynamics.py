"""Reviewer-requested audits over a completed run.

Six analyses the shipped statistics files do not cover:

  B1  anonymity leak      does an anonymous-condition message disclose the
                          speaker's own assigned candidate?
  B3  tie exposure        deadlock rate by arm
  C1  trajectory          accuracy by round, by arm
  C2  seed recantation    does the seed abandon its false claim by round 3?
  C4  parse failures      which cells they fall in
  spread                  per-scenario distribution of the C1/C2 effect sizes

B1 is a manipulation check, not an outcome, and needs care: the assigned
entities are the answer options, so "names its own entity" is ~99.9% under
both conditions and measures the task rather than disclosure. It is reported
as `names_own_assignment` and labelled uninformative. The measure with signal
is a first-person standing claim ("my dossier", "I was assigned") scoped by
clause to the speaker's own entity, with no other candidate intervening;
precision 99.0% (Wilson 95% CI 96.4-99.7) on 200 hand-labelled clauses. The
superseded character-window version is still emitted as
`standing_claim_linked_proximity` for comparison.

Rates are per message and are lower bounds, since paraphrase is not matched.
Read the `assigned` column as a reference, not a comparable rate: that header
names the assignment, so effective disclosure there is 100% by construction.

Usage:
  uv run python scripts/audit_disclosure_and_dynamics.py <run-dir> [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
import statistics
import sys
from pathlib import Path

# First-person claims of standing. A speaker under `anonymous` who says "my
# review" tells a reader which candidate they held, which the header withholds.
#
# The noun list must include `dossier`, and a bounded set of qualifiers may
# stand between "my" and the noun. An earlier revision required adjacency and
# omitted `dossier`, which missed the single most common construction in both
# corpora -- "My own dossier confirms X" -- and understated the leak roughly
# threefold. "I hold that X is compliant" is belief rather than possession and
# "I have no access" disclaims sight of ANOTHER agent's material, so both are
# excluded.
_QUAL = (r"(?:(?:own|private|assigned|authoritative|case|personal|"
         r"confidential|internal|direct)\s+){0,2}")
STANDING = re.compile(
    rf"\bmy {_QUAL}(?:assignment|review|assigned|finding|report|notes?|"
    rf"dossier|brief|record)\b"
    rf"|\bin my {_QUAL}(?:review|assessment|notes|dossier|brief)\b"
    r"|\bI (?:was |am |had been )?assigned\b"
    r"|\bassigned to me\b|\bmy assignment was\b"
    r"|\bI (?:reviewed|assessed|checked|examined|evaluated|verified|inspected)\b"
    r"|\bI (?:hold|have|received|was given|possess)\b"
    r"(?!\s+that\b)(?!\s+no (?:access|visibility|sight)\b)"
    r"|\bas the (?:assigned )?(?:reviewer|panellist|panelist|assessor|verifier)\b"
    r"|\bthe (?:one|vendor|supplier|candidate|option|site)\s+I\b",
    re.IGNORECASE,
)
LINK_WINDOW = 120

# Linking a standing claim to an entity by character distance crosses sentence
# and clause boundaries: in "Cirrus fails a control. My dossier shows..." the
# claim is not about Cirrus. Segment into clauses instead, and require that no
# OTHER candidate sit between the claim and the speaker's own entity -- raw
# distance is symmetric where syntax is not. A candidate merely conjoined to
# the speaker's own ("my report confirms Aldgate and Wren are compliant") is
# one object, not a separate predication, so it does not sever the link.
_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+|[;:]|[—–]|\s-\s|\n+"
    r"|,\s+(?:and|but|so|yet|while|whereas|however|although|though|because|"
    r"since|which|who)\b"
    r"|\s+(?:but|however|whereas|although|though)\b",
    re.IGNORECASE,
)
COORDINATION = re.compile(r"\s*(?:,\s*)?(?:and|or|&|,|nor|plus)?\s*", re.IGNORECASE)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assumption_propagation.prompts import assignment_label  # noqa: E402

N_AGENTS = 4


def open_records(run: Path):
    """Yield records from the full transcripts, or from the shipped slice.

    `records.jsonl` is ~300 MB per run and is not distributed. The slice
    written by scripts/make_disclosure_slice.py carries the same fields this
    script reads, verbatim, so the numbers are identical either way; it is what
    makes this measurement reproducible from a clone.
    """
    full = run / "records.jsonl"
    if full.is_file():
        with full.open(encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
        return
    sl = run / "analysis" / "disclosure_slice.jsonl.gz"
    if not sl.is_file():
        sys.exit(f"need {full} or {sl}; neither is present")
    with gzip.open(sl, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


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


def clause_spans(text: str, entities: tuple[str, ...] = ()) -> list[tuple[int, int, str]]:
    """(start, end, text) per clause. A bare comma introducing a candidate is
    also a break: "Ferrers is compliant (my record), Corwen fails deadline"."""
    pattern = _BOUNDARY
    if entities:
        alt = "|".join(re.escape(e) for e in sorted(entities, key=len, reverse=True))
        pattern = re.compile(_BOUNDARY.pattern + rf"|,\s+(?=(?:{alt})\b)", re.IGNORECASE)
    spans, pos = [], 0
    for mo in pattern.finditer(text):
        if mo.start() > pos:
            spans.append((pos, mo.start(), text[pos:mo.start()]))
        pos = max(pos, mo.end())
    if pos < len(text):
        spans.append((pos, len(text), text[pos:]))
    return spans


def standing_linked(text: str, mine: str, others: list[str]) -> bool:
    """A standing claim and the speaker's own entity in one clause, with no
    other candidate intervening."""
    own_re = re.compile(rf"\b{re.escape(mine)}\b", re.IGNORECASE)
    other_res = [re.compile(rf"\b{re.escape(e)}\b", re.IGNORECASE) for e in others if e]
    spans = clause_spans(text, tuple(e for e in (mine, *others) if e))
    for st in STANDING.finditer(text):
        for i, j, seg in spans:
            if not (i <= st.start() < j):
                continue
            own = list(own_re.finditer(seg))
            if not own:
                break
            a, b = st.start() - i, st.end() - i
            spots = [x.span() for p in other_res for x in p.finditer(seg)]

            def blocks(q, o):
                lo, hi = (q[1], o[0]) if q[1] <= o[0] else (o[1], q[0])
                return not COORDINATION.fullmatch(seg[lo:hi])

            if any(not any(blocks(q, o.span()) for q in spots
                           if (b <= q[0] < o.start() if o.start() >= b
                               else o.end() <= q[0] < a))
                   for o in own):
                return True
            break
    return False


def other_assignments(rec, assign) -> list[str]:
    """Every OTHER agent's assigned entity, for the no-crossing test."""
    a = assign.get(rec["scenario_id"])
    if not a:
        return []
    mine = own_assignment(rec, assign)
    all_ents = [a["seed"], a["corrector"], *(a["neutral"] or [])]
    out, dropped = [], False
    for e in all_ents:
        if e == mine and not dropped:
            dropped = True
            continue
        if e:
            out.append(e)
    return out


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
    if not ((run / "records.jsonl").is_file()
            or (run / "analysis" / "disclosure_slice.jsonl.gz").is_file()):
        sys.exit(
            f"{run}: need records.jsonl or analysis/disclosure_slice.jsonl.gz. "
            "The full per-agent-round transcripts are ~300 MB and are not "
            "distributed; the slice, which carries the fields this audit reads "
            "verbatim, is shipped in their place."
        )

    assign = load_assignments(run)
    out: dict[str, object] = {"run": run.name}

    # ---------------- B1: anonymity leak, and C4: parse failures -------------
    # each bucket: [names_own, standing, linked, proximity, n]
    leak = collections.defaultdict(lambda: [0, 0, 0, 0, 0])     # (arm, role)
    leak_cond = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # attribution
    verifier_msgs = [0, 0, 0, 0, 0]                          # frozen verifier round-1
    verifier_uniq = [0, 0, 0, 0, 0]                          # ... deduplicated
    seen_fv: set = set()
    parse_cells = collections.Counter()
    parse_total = 0
    examples: list[dict] = []

    for r in open_records(run):
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
        # "linked": a standing claim scoped to the speaker's own entity.
        linked = bool(standing and m) and standing_linked(
            text, mine, other_assignments(r, assign))
        # legacy character-window measure, retained for comparison only
        proximity = False
        if standing and m:
            for hit in re.finditer(rf"\b{re.escape(mine)}\b", text, re.IGNORECASE):
                for sc in STANDING.finditer(text):
                    if abs(sc.start() - hit.start()) <= LINK_WINDOW:
                        proximity = True
                        break
                if proximity:
                    break

        def bump(bucket):
            bucket[0] += names_own
            bucket[1] += standing is not None
            bucket[2] += linked
            bucket[3] += proximity
            bucket[4] += 1

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
                # One generation is replayed identically across the delivery
                # arms, so the raw bucket counts a single wording several times
                # and over-weights whatever that wording happens to say.
                key = (r["scenario_id"], " ".join(text.split()))
                if key not in seen_fv:
                    seen_fv.add(key)
                    bump(verifier_uniq)

    def rates(v):
        n = v[4]
        return {
            "n": n,
            "names_own_assignment": v[0],
            "names_own_assignment_rate": (v[0] / n) if n else None,
            "standing_claim": v[1],
            "standing_claim_rate": (v[1] / n) if n else None,
            "standing_claim_linked_to_own": v[2],
            "standing_claim_linked_to_own_rate": (v[2] / n) if n else None,
            "standing_claim_linked_proximity": v[3],
            "standing_claim_linked_proximity_rate": (v[3] / n) if n else None,
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
            "standing_claim_linked_to_own": "a standing claim in the SAME "
                "CLAUSE as the speaker's own assigned entity, with no other "
                "candidate intervening; the strongest literal evidence a reader "
                "could identify the speaker's assignment from content alone. "
                "Precision 99.0% (Wilson 95% CI 96.4-99.7) on 200 randomly "
                "sampled flagged clauses labelled by hand.",
            "standing_claim_linked_proximity": "superseded: the same claim "
                f"within {LINK_WINDOW} characters, which crosses sentence and "
                "clause boundaries. Retained only so the two can be compared.",
            "frozen_verifier_round1_anonymous_distinct_wordings": "the same "
                "rows deduplicated on wording. The verifier message is "
                "generated once and replayed across the delivery arms, so the "
                "raw bucket counts one wording several times; this is the rate "
                "per distinct message and the one to quote.",
            "bound": "all are lower bounds; paraphrase is not matched",
        },
        "by_attribution": {
            k: rates(v) for k, v in sorted(leak_cond.items(), key=lambda kv: str(kv[0]))
        },
        "anonymous_by_arm_role": {
            f"{a}|{ro}": rates(v)
            for (a, ro), v in sorted(leak.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))
        },
        "frozen_verifier_round1_anonymous": rates(verifier_msgs),
        "frozen_verifier_round1_anonymous_distinct_wordings": rates(verifier_uniq),
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
