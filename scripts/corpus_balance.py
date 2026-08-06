"""Audit corpus-level balance of the hidden-profile specs.

``check_specs`` in the builder enforces per-scenario structure: the seed owns
the trap, only one reviewer speaks to each option, corrections name the trap,
injections are length-matched. All of that is *within* a scenario.

This script checks the orthogonal thing: properties that are individually fine
in every scenario but become confounds if they correlate with scenario across
the corpus. Expanding from 10 to 30 hand-authored specs is exactly when these
drift, because the author has a house style and it shows up as a systematic
lean.

Reported, with a flag when the distribution is lopsided:

1. option letter of the correct answer, and of the trap — a model with a
   letter bias would otherwise score differently on old vs new scenarios;
2. position of the trap in the public list — recency/primacy in the shared
   brief;
3. which hard constraint the injection contests (heuristic, by content-word
   overlap with the seed's honest finding) — the first ten leaned 6/10 to
   constraint 1;
4. injection length by semantic type, corpus-wide;
5. whether the trap is the optimum on the public metric. This is the property
   that makes the injection *sufficient* to license choosing the trap; if it
   ever fails, the seed rationally defers and the scenario is dead weight.
   Parsed from the first number in each display value, with hours/minutes
   normalized to minutes. Other mixed-unit scenarios are reported as unparsed
   rather than as failures.

Advisory by design — it prints and returns 0 unless --strict is passed. Use it
to steer authoring, not to gate the build.

Usage:
  uv run python scripts/corpus_balance.py
  uv run python scripts/corpus_balance.py --strict
"""

from __future__ import annotations

import argparse
import re
import runpy
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_hidden_profile_scenarios.py"

# A letter or position is flagged when its share departs from uniform by more
# than this. With 30 scenarios over 4 cells, uniform is 7.5; 0.15 allows 3-12.
MAX_SHARE_DEVIATION = 0.15

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "is",
    "was", "were", "be", "been", "must", "not", "no", "any", "its", "it",
    "that", "this", "with", "by", "from", "your", "you", "assignment", "has",
    "have", "had", "which", "than", "so", "all", "every", "as", "are",
}


def load_specs() -> list:
    """The specs that will actually be COLLECTED, not the candidate pool.

    Gate-excluded scenarios are dropped here for the same reason the builder
    drops them: an audit of scenarios that will not be run tells you nothing
    about the corpus, and balance can shift when a scenario is cut.
    """
    sys.argv = [str(BUILDER)]
    namespace = runpy.run_path(str(BUILDER), run_name="_corpus_balance")
    excluded = namespace.get("EXCLUDED", {})
    if excluded:
        print(f"excluding {len(excluded)} gate-failed scenario(s): "
              f"{', '.join(sorted(excluded))}\n")
    return [s for s in namespace["SPECS"] if s.scenario_id not in excluded]


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


def contested_constraint_index(spec) -> int | None:
    """Which hard constraint the seed's honest finding speaks to.

    Heuristic: the constraint sharing the most content words with the finding.
    Returns None when nothing overlaps, which is itself worth seeing.
    """
    finding = content_words(spec.seed_true_finding)
    scores = [len(finding & content_words(c)) for c in spec.hard_constraints]
    best = max(scores)
    return scores.index(best) + 1 if best > 0 else None


def first_number(value: str) -> float | None:
    match = re.search(r"\d[\d,]*\.?\d*", value)
    if match is None:
        return None
    return float(match.group().replace(",", ""))


def duration_minutes(value: str) -> float | None:
    """Normalize a displayed hours/minutes duration, or return None."""
    lowered = value.lower()
    if "hour" not in lowered and "minute" not in lowered:
        return None
    hours = re.search(r"(\d+(?:\.\d+)?)\s*hours?", lowered)
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*minutes?", lowered)
    if hours is None and minutes is None:
        return None
    return (
        (float(hours.group(1)) * 60 if hours else 0.0)
        + (float(minutes.group(1)) if minutes else 0.0)
    )


def trap_is_metric_optimum(spec, trap_entity: str) -> bool | None:
    """True/False if the public values parse, None if they do not.

    None is not a pass. It means the check could not be run on that scenario
    and a human has to confirm it by reading.
    """
    duration_values = {
        entity: duration_minutes(display) for entity, display in spec.public_list
    }
    if all(value is not None for value in duration_values.values()):
        return duration_values[trap_entity] == min(duration_values.values())
    if any(value is not None for value in duration_values.values()):
        return None

    values = {}
    for entity, display in spec.public_list:
        number = first_number(display)
        if number is None:
            return None
        values[entity] = number
    # Mixed units defeat a numeric comparison; detect by unit tokens differing.
    units = {re.sub(r"[\d.,]", "", display).strip().lower()
             for _, display in spec.public_list}
    if len(units) > 1:
        return None
    return values[trap_entity] == min(values.values())


def report_distribution(title: str, counts: Counter, total: int,
                        cells: list) -> list[str]:
    warnings: list[str] = []
    shown = {c: counts.get(c, 0) for c in cells}
    print(f"  {title}: {shown}")
    uniform = 1 / len(cells)
    for cell in cells:
        share = counts.get(cell, 0) / total
        if abs(share - uniform) > MAX_SHARE_DEVIATION:
            warnings.append(
                f"{title}: {cell!r} holds {counts.get(cell, 0)}/{total} "
                f"({share:.0%}), uniform is {uniform:.0%}"
            )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any balance warning fires")
    args = parser.parse_args()

    specs = load_specs()
    total = len(specs)
    print(f"{total} scenarios\n")

    correct = Counter(s.correct_answer for s in specs)
    favored = Counter(s.favored_option for s in specs)
    trap_position: Counter = Counter()
    contested: Counter = Counter()
    lengths: dict[str, list[int]] = {"factual": [], "causal": [], "framing": []}
    not_optimum: list[str] = []
    unparsed: list[str] = []

    # entity_for lives in the builder; re-deriving it here would let the two
    # drift apart, so pull it from the module namespace instead.
    sys.argv = [str(BUILDER)]
    module = runpy.run_path(str(BUILDER), run_name="_corpus_balance")
    entity_for = module["entity_for"]

    for spec in specs:
        trap = entity_for(spec, spec.favored_option)
        names = [entity for entity, _ in spec.public_list]
        trap_position[names.index(trap) + 1] += 1
        contested[contested_constraint_index(spec)] += 1
        for semantic_type in ("factual", "causal", "framing"):
            lengths[semantic_type].append(len(getattr(spec, semantic_type)[1].split()))
        optimum = trap_is_metric_optimum(spec, trap)
        if optimum is None:
            unparsed.append(spec.scenario_id)
        elif not optimum:
            not_optimum.append(spec.scenario_id)

    warnings: list[str] = []
    print("balance")
    warnings += report_distribution("correct-answer letter", correct, total,
                                    ["A", "B", "C", "D"])
    warnings += report_distribution("trap option letter", favored, total,
                                    ["A", "B", "C", "D"])
    warnings += report_distribution("trap position in public list",
                                    trap_position, total, [1, 2, 3, 4])
    print(f"  contested constraint index: "
          f"{dict(sorted(contested.items(), key=lambda kv: (kv[0] is None, kv[0])))}")
    for index in (1, 2, 3, 4):
        share = contested.get(index, 0) / total
        if share > 1 / 4 + MAX_SHARE_DEVIATION:
            warnings.append(
                f"contested constraint: index {index} holds "
                f"{contested[index]}/{total} ({share:.0%})"
            )
    if contested.get(None):
        warnings.append(
            f"{contested[None]} scenario(s): could not match the seed's finding "
            f"to any hard constraint — read those by hand"
        )

    print("\ninjection length by type (words)")
    means = {}
    for semantic_type, values in lengths.items():
        means[semantic_type] = sum(values) / len(values)
        print(f"  {semantic_type:8s} mean {means[semantic_type]:5.2f}  "
              f"min {min(values)}  max {max(values)}")
    spread = max(means.values()) - min(means.values())
    print(f"  mean spread across types: {spread:.2f}")

    print("\ntrap is the optimum on the public metric")
    print(f"  confirmed: {total - len(not_optimum) - len(unparsed)}/{total}")
    if unparsed:
        print(f"  unparsed (mixed units — verify by reading): {', '.join(unparsed)}")
    if not_optimum:
        warnings.append(
            "trap is NOT the metric optimum in: " + ", ".join(not_optimum)
            + " — the injection cannot license choosing it and the seed will defer"
        )

    if warnings:
        print("\nwarnings")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nno balance warnings")

    return 1 if (warnings and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
