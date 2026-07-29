"""Ceiling test: is each scenario solvable at all?

Every propagation result sits downstream of one assumption — that a correct
answer is reachable from the information the design distributes. Nothing in
the pipeline currently establishes that. `audit_scenarios.py` checks structural
properties (letter balance, refutation markers, unknown-status rules), but
structure is not solvability, and Track B baseline group accuracy of 19% is
direct evidence that a well-formed scenario can still be unsolvable in practice.

This gives ONE agent every private finding at once, with no communication and
one round. That is the ceiling: the best any group could do if pooling worked
perfectly.

  oracle correct  -> the scenario is well posed, the information suffices, and
                     the answer key is right. Group failure is then a finding
                     about pooling rather than a broken task.
  oracle wrong    -> the scenario is broken. No downstream result from it means
                     anything, and it should be excluded before the
                     confirmatory run rather than after.

The oracle receives the BASELINE information state (every agent's honest
finding, including the seed's), so it tests the task, not the injection.
Prompts are built with the same builders the experiment uses, so this
exercises the actual materials rather than a paraphrase.

Usage:
  uv run python scripts/oracle_check.py --data data/scenarios_hidden_profile.json
  uv run python scripts/oracle_check.py --data <file> --client azure --n-samples 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from assumption_propagation.clients import AzureChatClient, MockClient, parse_or_repair
from assumption_propagation.prompts import (
    build_shared_task,
    build_seed_generation_system_prompt,
    render_dossier,
)
from assumption_propagation.reproducibility import sha256_file
from assumption_propagation.schemas import Scenario

# A scenario the oracle cannot solve is unusable; one it solves inconsistently
# is weak evidence for the answer key. Both thresholds are set here, before the
# numbers are seen, so exclusion is a rule rather than a judgement call.
PASS_THRESHOLD = 0.8
WEAK_THRESHOLD = 0.5


def oracle_prompt(scenario: Scenario) -> str:
    """Shared task plus EVERY private finding, in one dossier."""
    profile = scenario.hidden_profile
    if profile is None:
        raise ValueError(
            f"{scenario.scenario_id} has no hidden_profile; the ceiling test only "
            "applies to distributed-information scenarios"
        )
    # The seed's honest finding lives in neutral_control_note (the baseline
    # note); its brief carries only the assignment.
    findings = [
        profile.seed_brief,
        scenario.neutral_control_note,
        profile.corrector_brief,
        *profile.neutral_briefs,
    ]
    # Each finding is written in the second person for the reviewer who holds
    # it. Handing all four to one agent unchanged reads as though it were
    # assigned every option, which would make the oracle fail for a
    # comprehension reason rather than a design one. The header states the
    # consolidation explicitly instead of rewriting the authored text.
    dossier = render_dossier(findings).replace(
        "PRIVATE DOSSIER (retain for subsequent rounds)",
        "CONSOLIDATED REVIEW NOTES\nYou hold the notes of all four reviewers. "
        "Each entry below is one reviewer's own assignment and their finding "
        "for it, so together they cover every option.",
    )
    return "\n\n".join([
        build_shared_task(scenario),
        dossier,
        "Provide your answer now in the required JSON format.",
    ])


async def run_oracle(scenario: Scenario, client, n_samples: int) -> dict:
    prompt = oracle_prompt(scenario)
    # ID-neutral: the ceiling test is a single agent answering once, so the
    # standard prompt's agent identity and multi-round instructions do not apply.
    system = build_seed_generation_system_prompt()
    answers: list[str | None] = []
    for _ in range(n_samples):
        try:
            raw = await client.generate(system, prompt)
            answers.append(parse_or_repair(raw).answer)
        except Exception as exc:  # a failed call is data, not a crash
            answers.append(None)
            print(f"    ! generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    correct = sum(a == scenario.correct_answer for a in answers)
    return {
        "scenario_id": scenario.scenario_id,
        "correct_answer": scenario.correct_answer,
        "n": len(answers),
        "n_correct": correct,
        "accuracy": correct / len(answers) if answers else 0.0,
        "answers": dict(Counter(a or "PARSE_FAIL" for a in answers)),
    }


def verdict(accuracy: float) -> str:
    if accuracy >= PASS_THRESHOLD:
        return "PASS"
    if accuracy >= WEAK_THRESHOLD:
        return "WEAK"
    return "FAIL"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--client", choices=("mock", "azure"), default="mock")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--temperature", default="0.2")
    parser.add_argument("--scenario", default=None, help="single scenario_id")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    entries = json.loads(args.data.read_text(encoding="utf-8"))
    scenarios = [
        Scenario.model_validate(e)
        for e in entries
        if isinstance(e, dict) and "scenario_id" in e
        and (args.scenario is None or e["scenario_id"] == args.scenario)
    ]
    if not scenarios:
        sys.exit(f"no scenarios matched in {args.data}")

    temperature = None if args.temperature.lower() == "none" else float(args.temperature)
    client = MockClient() if args.client == "mock" else AzureChatClient(temperature=temperature)

    print(f"data={args.data} (sha256={sha256_file(args.data)[:16]})")
    print(f"client={client.model_identifier}  scenarios={len(scenarios)}  "
          f"samples each={args.n_samples}")
    print(f"pass >= {PASS_THRESHOLD:.0%}, weak >= {WEAK_THRESHOLD:.0%}, else fail\n")

    rows = []
    for scenario in scenarios:
        result = await run_oracle(scenario, client, args.n_samples)
        result["verdict"] = verdict(result["accuracy"])
        rows.append(result)
        print(f"  {result['scenario_id']:16s} key={result['correct_answer']}  "
              f"{result['n_correct']}/{result['n']} = {result['accuracy']:>4.0%}  "
              f"{result['verdict']:5s}  answers={result['answers']}")

    passed = [r for r in rows if r["verdict"] == "PASS"]
    weak = [r for r in rows if r["verdict"] == "WEAK"]
    failed = [r for r in rows if r["verdict"] == "FAIL"]

    print(f"\n{'=' * 70}")
    print(f"PASS {len(passed)}   WEAK {len(weak)}   FAIL {len(failed)}   of {len(rows)}")
    if failed:
        print("\nEXCLUDE from the confirmatory run (a single agent holding every "
              "finding cannot reach the key):")
        for r in failed:
            print(f"  - {r['scenario_id']}: {r['answers']}")
    if weak:
        print("\nINSPECT (solved inconsistently; the key may be defensible but the "
              "scenario is noisy):")
        for r in weak:
            print(f"  - {r['scenario_id']}: {r['answers']}")
    if not failed and not weak:
        print("\nAll scenarios solvable at ceiling. Group failure, where it occurs,")
        print("is a result about pooling rather than a broken task.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "data": str(args.data),
            "data_sha256": sha256_file(args.data),
            "model": client.model_identifier,
            "n_samples": args.n_samples,
            "pass_threshold": PASS_THRESHOLD,
            "weak_threshold": WEAK_THRESHOLD,
            "results": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
