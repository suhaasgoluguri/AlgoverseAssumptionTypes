"""Anonymity-leakage measurement for one run.

Under `anonymous` attribution a peer header names only `Agent N`. The factor
only means anything if the message body does not restore what the header
withholds; this measures how often it does.

The headline measure is a first-person standing claim ("my dossier", "I was
assigned") in the same clause as the speaker's own assigned entity, with no
other candidate between the two. Clause scoping rather than a character
window, because a window crosses sentence boundaries; no-crossing rather than
nearest-entity, because distance is symmetric where syntax is not. Precision
is 99.0% (Wilson 95% CI 96.4-99.7) on 200 sampled clauses labelled by hand.

Every rate is PER MESSAGE -- one agent's output in one round -- and is a lower
bound, since paraphrase is not matched. Read the `assigned` column as a
reference, not a comparable rate: that header names the assignment, so
effective disclosure there is 100% by construction.

Usage:
  uv run python scripts/measure_anonymity_leakage.py <run-dir> [--json OUT]
                                    [--examples N] [--audit N] [--sample N]
  uv run python scripts/measure_anonymity_leakage.py --selftest
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assumption_propagation.prompts import assignment_label  # noqa: E402

N_AGENTS = 4
LINK_WINDOW = 120

# ---------------------------------------------------------------- segmentation
# Clause breaks: sentence end, ";:", em/en dashes, newlines, and a comma
# followed by a conjunction. Not parentheses -- both models write "Ferrers is
# compliant (my record), Corwen fails deadline (Agent 3)", and splitting on
# the bracket strands the standing phrase in a clause holding no candidate.
_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"
    r"|[;:]"
    r"|[—–]|\s-\s"
    r"|\n+"
    r"|,\s+(?:and|but|so|yet|while|whereas|however|although|though|because|"
    r"since|which|who)\b"
    r"|\s+(?:but|however|whereas|although|though)\b",
    re.IGNORECASE,
)


def clause_spans(text: str, entities: tuple[str, ...] = ()) -> list[tuple[int, int, str]]:
    """(start, end, text) for each clause, in document order.

    ``entities`` additionally breaks a bare comma that introduces a candidate
    name, which is how both models write list-style attributions:
    "Ferrers is compliant (my record), Corwen fails deadline (Agent 3)".
    """
    pattern = _BOUNDARY
    if entities:
        alt = "|".join(re.escape(e) for e in sorted(entities, key=len, reverse=True))
        pattern = re.compile(
            _BOUNDARY.pattern + rf"|,\s+(?=(?:{alt})\b)", re.IGNORECASE
        )
    spans, pos = [], 0
    for mo in pattern.finditer(text):
        if mo.start() > pos:
            spans.append((pos, mo.start(), text[pos:mo.start()]))
        pos = max(pos, mo.end())
    if pos < len(text):
        spans.append((pos, len(text), text[pos:]))
    return spans

# ------------------------------------------------------------------ detectors
# Standing families. Calibrated against the confirmatory corpora; the observed
# anonymous-mode rate for each is in the comment, so a reader can see which
# families are load-bearing and which are near-dead.
STANDING_FAMILIES: dict[str, str] = {
    # 4.85% -- dominant family
    "possessive_role": r"\bmy (?:(?:own|private|assigned|authoritative|case|personal|confidential|internal|direct)\s+){0,2}(?:assignment|assigned|review|finding|report|notes?|dossier|brief|record)\b",
    # 2.23%. "I hold that X" is belief rather than possession, and "I have no
    # access" disclaims sight of another agent's material; both are excluded.
    "i_hold_have": r"\bI (?:hold|have|received|was given|possess)\b(?!\s+that\b)(?!\s+no (?:access|visibility|sight)\b)",
    # 1.10%
    "in_my_x": r"\bin my (?:(?:own|private|assigned|authoritative|case|personal|confidential|internal|direct)\s+){0,2}(?:review|assessment|notes|dossier|brief)\b",
    # 0.25%
    "assigned_to_me": r"\bassigned to me\b|\bmy assignment was\b",
    # 0.09%
    "first_person_verb": r"\bI (?:was |am |had been )?(?:assigned|reviewed|assessed|checked|examined|evaluated|verified|inspected)\b",
    # 0.00% here, kept so a corpus that uses it is not silently missed
    "as_the_reviewer": r"\bas the (?:assigned )?(?:reviewer|panellist|panelist|assessor|verifier)\b",
    "the_one_i": r"\bthe (?:one|vendor|supplier|candidate|option|site)\s+I\b",
}
STANDING_ANY = re.compile("|".join(STANDING_FAMILIES.values()), re.IGNORECASE)
# Two candidates joined only by coordination share one predication.
COORDINATION = re.compile(r"\s*(?:,\s*)?(?:and|or|&|,|nor|plus)?\s*", re.IGNORECASE)
BARE_FIRST_PERSON = re.compile(r"\bI\b|\bmy\b|\bmine\b", re.IGNORECASE)


# tier3: an assignment verb binding the speaker to a named entity.
EXPLICIT_TEMPLATE = (
    r"(?:my (?:assignment|review|assigned candidate) (?:was|is) {e}"
    r"|I (?:was )?(?:assigned|reviewed|assessed|checked|verified) {e}"
    r"|{e}[,]? (?:which|whom) I (?:reviewed|assessed|was assigned|checked))"
)

# Another agent bound to an entity by an ASSIGNMENT predicate, as against mere
# testimony relay. "(assigned Tavis)" and "Agent 3's assigned report" count;
# "per Agent 1" and "Agent 1 confirms" do not.
ASSIGN_PRED = (
    r"(?:assigned|assignment|reviewer|assessor|panellist|panelist|auditor|"
    r"engineer|dossier|brief|holds?|reviewed|assessed|audited|inspected)"
)


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


def load_assignments(run_dir: Path) -> dict[str, dict]:
    raw = json.loads((run_dir / "scenario_snapshot.json").read_text(encoding="utf-8"))
    scenarios = raw if isinstance(raw, list) else raw["scenarios"]
    out: dict[str, dict] = {}
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


def assignment_map(rec, assign) -> dict[int, str | None]:
    """agent_id -> assigned entity, using the runtime's own ordinal rule."""
    a = assign.get(rec["scenario_id"])
    if not a:
        return {}
    seed, corr = rec.get("seed_agent_id"), rec.get("corrector_agent_id")
    neutral_ids = [i for i in range(N_AGENTS) if i not in (seed, corr)]
    out: dict[int, str | None] = {seed: a["seed"], corr: a["corrector"]}
    for ordinal, aid in enumerate(neutral_ids):
        out[aid] = a["neutral"][ordinal] if ordinal < len(a["neutral"]) else None
    return out


def score_tier2(text, mine, others, guards=True, collect=None):
    """(hit, reject_reason). Clause-scoped standing claim about the speaker's
    own entity, with the no-crossing false-positive guard applied."""
    ents = tuple(e for e in (mine, *others) if e)
    own_re = re.compile(rf"\b{re.escape(mine)}\b", re.IGNORECASE)
    other_res = [(e, re.compile(rf"\b{re.escape(e)}\b", re.IGNORECASE)) for e in others if e]
    spans = clause_spans(text, ents if guards else ())
    reject = None
    for st in STANDING_ANY.finditer(text):
        for i, j, seg in spans:
            if not (i <= st.start() < j):
                continue
            off = st.start() - i
            own = list(own_re.finditer(seg))
            if not own:
                break
            if not guards:
                return True, None
            # No other candidate may sit between the standing phrase and the
            # speaker's own entity; one that does signals a separate
            # predication. Distance would be the wrong test -- it is symmetric
            # and syntax is not.
            a, b = st.start() - i, st.end() - i
            spots = [m.span() for _, p in other_res for m in p.finditer(seg)]

            def blocks(q, o):
                """Does other-entity span q sever the claim from own span o?"""
                lo, hi = (q[1], o[0]) if q[1] <= o[0] else (o[1], q[0])
                # A candidate conjoined to the speaker's own entity is part of
                # the same object, not a separate predication: "my own report
                # ... confirm Aldgate and Wren are compliant" discloses Wren.
                return not COORDINATION.fullmatch(seg[lo:hi])

            if any(
                not any(blocks(q, o.span()) for q in spots
                        if (b <= q[0] < o.start() if o.start() >= b
                            else o.end() <= q[0] < a))
                for o in own
            ):
                if collect is not None:
                    collect.append((mine, st.group(0), seg))
                return True, None
            reject = reject or "crossed_entity"
            break
        else:
            continue
    return False, reject


def score_exclusion(text, mine, others):
    """A standing clause naming every candidate EXCEPT the speaker's own.

    Scored in its own pass, on the coarse segmentation: a message can disclose
    directly AND by elimination, so this must not be short-circuited by a
    tier2 hit.
    """
    if not others:
        return False
    own_re = re.compile(rf"\b{re.escape(mine)}\b", re.IGNORECASE)
    other_res = [re.compile(rf"\b{re.escape(e)}\b", re.IGNORECASE) for e in others if e]
    # Deliberately the COARSE segmentation. The target construction is an
    # enumeration -- "no report for Bardsey, Ilkeston, or Thurso" -- and the
    # entity-aware comma rule would split it into one clause per candidate,
    # so no clause could ever hold all of them.
    spans = clause_spans(text)
    for st in STANDING_ANY.finditer(text):
        for i, j, seg in spans:
            if not (i <= st.start() < j):
                continue
            if not own_re.search(seg) and all(p.search(seg) for p in other_res):
                return True
            break
    return False


def score_other_agent(text, amap, self_id):
    """(assignment_disclosure, any_attribution) for OTHER agents."""
    strict = loose = False
    for aid, ent in amap.items():
        if aid == self_id or not ent:
            continue
        e = re.escape(ent)
        if re.search(
            rf"Agent {aid}\b[^.]{{0,40}}{ASSIGN_PRED}[^.]{{0,40}}\b{e}\b"
            rf"|\b{e}\b[^.]{{0,40}}{ASSIGN_PRED}[^.]{{0,40}}Agent {aid}\b"
            rf"|Agent {aid}(?:'s|’s)\s+{ASSIGN_PRED}[^.]{{0,40}}\b{e}\b",
            text, re.IGNORECASE,
        ):
            strict = True
        if re.search(
            rf"Agent {aid}\b[^.]{{0,80}}\b{e}\b|\b{e}\b[^.]{{0,80}}Agent {aid}\b",
            text, re.IGNORECASE,
        ):
            loose = True
    return strict, loose


# Calibration cases, each lifted verbatim from one of the confirmatory corpora
# and hand-labelled. Run with --selftest. They pin the segmentation and the
# no-crossing guard against the specific constructions that broke earlier
# revisions, so a later edit that reintroduces one fails loudly.
SELFTEST: tuple[tuple[str, str, list[str], bool], ...] = (
    ("Agent 2\u2019s report confirms Failover is compliant and my own dossier "
     "disqualifies Rebuild (exceeds window).", "Rebuild",
     ["Failover", "Rollback", "Hotfix"], True),
    ("Skerry fails on duration (Agent 3), Orme fails on rate (my assignment).",
     "Orme", ["Skerry", "Lorn", "Tavis"], True),
    ("Skerry fails on duration (Agent 3), Orme fails on rate (my assignment).",
     "Skerry", ["Orme", "Lorn", "Tavis"], False),
    ("Ferrers is compliant (my record), Corwen fails deadline (Agent 3), "
     "Wraye exceeds cost (Agent 2).", "Ferrers",
     ["Corwen", "Wraye", "Otley"], True),
    ("Ferrers is compliant (my record), Corwen fails deadline (Agent 3), "
     "Wraye exceeds cost (Agent 2).", "Corwen",
     ["Ferrers", "Wraye", "Otley"], False),
    ("I have no direct knowledge of Orbis, only my own report confirming "
     "Tessel meets all constraints.", "Tessel", ["Orbis", "Vane", "Nyx"], True),
    ("Thurso exceeds the $250,000 ceiling from my dossier.", "Thurso",
     ["Skerry", "Lorn", "Tavis"], True),
    ("My dossier confirms Ashby fails the LEZ standard, excluding Ashby.",
     "Ashby", ["Colm", "Fenn", "Deene"], True),
    ("Naver is disqualified by my dossier, Talla exceeds the budget cap",
     "Talla", ["Naver", "Culzean", "Harrow"], False),
    ("Naver is disqualified by my dossier, Talla exceeds the budget cap",
     "Naver", ["Talla", "Culzean", "Harrow"], True),
    ("Agent 1's claim of a lapsed certification contradicts my dossier and is "
     "not authoritative for Apex.", "Apex",
     ["Cirrus", "Delta", "Borealis"], True),
    ("I am the assigned reviewer for Verity. My private dossier confirms "
     "Verity meets all hard constraints.", "Verity",
     ["Pallas", "Colm", "Naver"], True),
    ("However, I was assigned Cirrus, not Apex.", "Apex",
     ["Cirrus", "Delta", "Borealis"], False),
    ("Grimsel fails restore (my report), Penrhos fails cost.", "Penrhos",
     ["Grimsel", "Openpit", "Insert"], False),
    ("Agent 0, Agent 3, and my own report confirm Aldgate and Wren are "
     "compliant.", "Wren", ["Aldgate", "Otley", "Braid"], True),
    ("My own report and Agent 1 both confirm Fenn and Deene satisfy all "
     "constraints.", "Deene", ["Fenn", "Colm", "Ashby"], True),
    ("I hold evidence that Aster is disqualified and that Kestrel exceeds "
     "the rate cap.", "Kestrel", ["Aster", "Meridian", "Lumen"], False),
    ("My case record confirms Orbis has a fatal traceability defect. Vane "
     "fails the cap.", "Vane", ["Orbis", "Tessel", "Nyx"], False),
    # The two false positives surfaced by the 200-clause precision audit.
    ("I hold that Enderby is fully compliant.", "Enderby",
     ["Lambourn", "Pellow", "Ovett"], False),
    ("Since I have no access to the other auditors' reports and no "
     "disqualifying findings have been reported for Ilkeston, it satisfies "
     "all hard constraints.", "Ilkeston",
     ["Bardsey", "Thurso", "Cranmore"], False),
    # ...but possession of one's own record, negated finding included, stands.
    ("I have no disqualifying finding for Marden.", "Marden",
     ["Talos", "Girvan", "Verity"], True),
)


def selftest() -> int:
    bad = 0
    for text, mine, others, want in SELFTEST:
        got = score_tier2(text, mine, others)[0]
        if got != want:
            bad += 1
            print(f"FAIL want={want} got={got} mine={mine}: {text[:80]}")
    print(f"{len(SELFTEST) - bad}/{len(SELFTEST)} calibration cases pass")
    return bad


class Bucket:
    __slots__ = ("n", "names_own", "tier1", "tier2", "tier2l", "tier2p", "tier3",
                 "other", "other_loose", "excl", "bare", "fam", "rej")

    def __init__(self):
        self.n = self.names_own = self.tier1 = self.tier2 = self.tier2l = 0
        self.tier2p = self.tier3 = self.other = self.other_loose = 0
        self.excl = self.bare = 0
        self.fam = collections.Counter()
        self.rej = collections.Counter()

    def add(self, **kw):
        for k, v in kw.items():
            setattr(self, k, getattr(self, k) + v)

    def as_dict(self):
        def rate(x):
            return (x / self.n) if self.n else None
        return {
            "n": self.n,
            "names_own": self.names_own, "names_own_rate": rate(self.names_own),
            "tier1_any_standing": self.tier1, "tier1_rate": rate(self.tier1),
            "tier2_clause": self.tier2, "tier2_rate": rate(self.tier2),
            "tier2_lenient": self.tier2l, "tier2_lenient_rate": rate(self.tier2l),
            "tier2_proximity": self.tier2p, "tier2_proximity_rate": rate(self.tier2p),
            "exclusion_disclosure": self.excl, "exclusion_rate": rate(self.excl),
            "tier3_explicit": self.tier3, "tier3_rate": rate(self.tier3),
            "other_agent_assignment": self.other, "other_assignment_rate": rate(self.other),
            "other_agent_attribution": self.other_loose,
            "other_attribution_rate": rate(self.other_loose),
            "bare_first_person": self.bare, "bare_rate": rate(self.bare),
            "guard_rejections": dict(sorted(self.rej.items())),
            "by_family": {k: {"n": v, "rate": rate(v)} for k, v in sorted(self.fam.items())},
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path, nargs="?")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--audit", type=int, default=0,
                    help="print N clauses rejected by the guard, then exit")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="emit N randomly sampled flagged clauses (seed 11) for "
                         "hand-labelling, one per line, then exit")
    ap.add_argument("--selftest", action="store_true",
                    help="run the hand-labelled calibration cases and exit")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(1 if selftest() else 0)
    run = args.run_dir
    if not ((run / "records.jsonl").is_file()
            or (run / "analysis" / "disclosure_slice.jsonl.gz").is_file()):
        sys.exit(
            f"{run}: need records.jsonl or analysis/disclosure_slice.jsonl.gz. "
            "The transcripts are ~300 MB and are not distributed; the slice, "
            "which carries the fields this measurement reads verbatim, is "
            "shipped in their place."
        )

    assign = load_assignments(run)
    overall = collections.defaultdict(Bucket)        # attribution
    by_arm_role = collections.defaultdict(Bucket)    # (attribution, arm, role)
    by_round = collections.defaultdict(Bucket)       # (attribution, round)
    by_topology = collections.defaultdict(Bucket)    # (attribution, topology)
    by_scenario = collections.defaultdict(Bucket)    # (attribution, scenario)
    verifier_r1 = collections.defaultdict(Bucket)    # attribution, frozen message
    verifier_r1_uniq = collections.defaultdict(Bucket)
    seen_fv: set = set()                             # (scenario, att, wording)
    examples: list[dict] = []
    audit: dict[str, list[str]] = collections.defaultdict(list)
    flagged_clauses: list[tuple[str, str, str]] = []
    skipped = 0

    for r in open_records(run):
        text = r.get("rationale") or ""
        if not text:
            skipped += 1
            continue
        amap = assignment_map(r, assign)
        mine = amap.get(r["agent_id"])
        if not mine:
            skipped += 1
            continue

        att, arm, role = r.get("attribution"), r.get("condition"), role_of(r)
        others = [e for aid, e in amap.items() if e and aid != r["agent_id"]]

        own_hits = list(re.finditer(rf"\b{re.escape(mine)}\b", text, re.IGNORECASE))
        names_own = bool(own_hits)
        stand_hits = list(STANDING_ANY.finditer(text))
        tier1 = bool(stand_hits)
        tier2p = any(
            abs(s.start() - o.start()) <= LINK_WINDOW
            for o in own_hits for s in stand_hits
        )
        tier2, reject = score_tier2(
            text, mine, others, guards=True,
            collect=flagged_clauses if args.sample else None)
        tier2l, _ = score_tier2(text, mine, others, guards=False)
        excl = score_exclusion(text, mine, others)
        if reject and len(audit[reject]) < args.audit:
            audit[reject].append(f"[{att}|mine={mine}] {' '.join(text.split())[:220]}")
        tier3 = re.search(
            EXPLICIT_TEMPLATE.format(e=re.escape(mine)), text, re.IGNORECASE
        ) is not None
        other, other_loose = score_other_agent(text, amap, r["agent_id"])
        bare = BARE_FIRST_PERSON.search(text) is not None
        fams = [k for k, p in STANDING_FAMILIES.items()
                if re.search(p, text, re.IGNORECASE)]

        targets = [overall[att], by_arm_role[(att, arm, role)],
                   by_round[(att, r.get("round"))], by_topology[(att, r.get("topology"))],
                   by_scenario[(att, r["scenario_id"])]]
        if role == "verifier" and r.get("round") == 1:
            targets.append(verifier_r1[att])
            # One generation is replayed identically across the delivery arms,
            # so the raw bucket counts the same wording several times over and
            # over-weights whatever that wording happens to say. Deduplicate on
            # the wording itself to get the rate per distinct message.
            key = (r["scenario_id"], att, " ".join(text.split()))
            if key not in seen_fv:
                seen_fv.add(key)
                targets.append(verifier_r1_uniq[att])
        for b in targets:
            b.add(n=1, names_own=names_own, tier1=tier1, tier2=tier2, tier2l=tier2l,
                  tier2p=tier2p, tier3=tier3, other=other, other_loose=other_loose,
                  excl=excl, bare=bare)
            if reject:
                b.rej[reject] += 1
            for f in fams:
                b.fam[f] += 1
        if att == "anonymous" and tier2 and len(examples) < args.examples:
            examples.append({
                "scenario_id": r["scenario_id"], "condition": arm, "role": role,
                "round": r.get("round"), "topology": r.get("topology"),
                "assigned": mine, "tier3": tier3, "families": fams,
                "rationale": text[:300],
            })

    if args.sample:
        rows = [e for e in flagged_clauses]
        random.Random(11).shuffle(rows)
        for mine, sp, seg in rows[:args.sample]:
            print(f"{mine}\t{sp}\t{' '.join(seg.split())}")
        print(f"# {len(rows)} flagged clauses; {min(args.sample, len(rows))} sampled",
              file=sys.stderr)
        return
    if args.audit:
        for name in ("crossed_entity",):
            rows = audit.get(name, [])
            print(f"\n===== rejected by {name}: {len(rows)} shown =====")
            for s in rows:
                print(f"  {s}")
        return

    def spread(prefix):
        """Per-scenario tier2 rate distribution, so a single scenario cannot
        carry the headline number unnoticed."""
        vals = [b.tier2 / b.n for (att, _), b in by_scenario.items()
                if att == prefix and b.n]
        if not vals:
            return None
        return {"min": min(vals), "median": statistics.median(vals),
                "max": max(vals), "n_scenarios": len(vals)}

    out = {
        "run": run.name,
        "records_scored": sum(b.n for k, b in overall.items()),
        "records_skipped": skipped,
        "link_window_chars": LINK_WINDOW,
        "tiers": {
            "names_own": "UNINFORMATIVE -- assigned entities are the answer "
                         "options; ~1.0 under both modes; reported only so its "
                         "symmetry is visible",
            "tier1_any_standing": "any first-person standing family fires",
            "tier2_clause": "HEADLINE. Standing claim and the speaker's own "
                            "entity in the same clause, no other candidate "
                            "intervening",
            "tier2_lenient": "same, but with the guard off AND the coarser "
                             "conjunction-only segmentation. The gap to "
                             "tier2_clause is the whole correction; "
                             "guard_rejections isolates the guard's share of "
                             "it, the remainder being finer segmentation",
            "tier2_proximity": f"legacy: standing within {LINK_WINDOW} chars of "
                               "the entity, no clause scoping at all",
            "exclusion_disclosure": "standing clause names every candidate "
                                    "except the speaker's own -- discloses by "
                                    "elimination",
            "tier3_explicit": "assignment verb binding speaker to the entity",
            "other_agent_assignment": "another agent bound to an entity by an "
                                      "assignment predicate -- a real leak",
            "other_agent_attribution": "NOT A LEAK. Any claim about an entity "
                                       "credited to a named agent; this is "
                                       "testimony relay, i.e. the task itself",
            "bare_first_person": "ceiling, not a leak",
            "frozen_verifier_round1": "raw rows -- one generation replayed "
                                      "across arms, so a single wording is "
                                      "counted several times",
            "frozen_verifier_round1_distinct_wordings": "deduplicated on the "
                                      "wording; this is the rate per distinct "
                                      "verifier message and the one to quote",
            "bound": "all rates are lower bounds; paraphrase is not matched",
        },
        "overall_by_attribution": {k: v.as_dict() for k, v in sorted(overall.items(), key=lambda kv: str(kv[0]))},
        "frozen_verifier_round1": {k: v.as_dict() for k, v in sorted(verifier_r1.items(), key=lambda kv: str(kv[0]))},
        "frozen_verifier_round1_distinct_wordings": {
            k: v.as_dict() for k, v in sorted(verifier_r1_uniq.items(), key=lambda kv: str(kv[0]))
        },
        "by_attribution_arm_role": {
            f"{a}|{arm}|{ro}": b.as_dict()
            for (a, arm, ro), b in sorted(by_arm_role.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
        },
        "by_attribution_round": {
            f"{a}|r{rd}": b.as_dict()
            for (a, rd), b in sorted(by_round.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or -1))
        },
        "by_attribution_topology": {
            f"{a}|{t}": b.as_dict()
            for (a, t), b in sorted(by_topology.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
        },
        "per_scenario_tier2_spread": {
            k: spread(k) for k in sorted(overall, key=str)
        },
        "examples_tier2_anonymous": examples,
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
