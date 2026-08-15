"""Ceiling test: is each scenario solvable at all?

Gives one agent every private finding at once, with no communication — the best
any group could do if pooling worked perfectly. A correct oracle answer means
the task is well posed and the answer key is right, so group failure is a
finding about pooling; a wrong one means the scenario is broken and should be
cut before collection. `audit_scenarios.py` checks structure, which is not the
same as solvability.

Uses the baseline information state (every honest finding, including the
seed's), so it tests the task rather than the injection, and builds prompts with
the same builders the experiment uses.

Usage:
  uv run python scripts/oracle_check.py --data data/scenarios_hidden_profile_candidates.json
  uv run python scripts/oracle_check.py --data <file> --client azure --n-samples 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from assumption_propagation.clients import (
    AzureChatClient,
    MockClient,
    audit_answer_rationale,
    parse_or_repair,
)
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
    # Rationale-consistency audit (mirrors orchestrator.py's use of the same
    # helper on live records): the "answer" field and the letter the
    # rationale's own prose concludes with can disagree. That disagreement is
    # a binding/formatting defect, not evidence the scenario is unsolvable, so
    # it is tracked separately from n_correct/accuracy rather than folded in.
    inconsistent_count = 0
    rationale_answers: list[str | None] = []
    samples: list[dict] = []
    for sample_index in range(n_samples):
        try:
            raw = await client.generate(system, prompt)
            parsed = parse_or_repair(raw)
            answers.append(parsed.answer)
            audit = audit_answer_rationale(
                parsed.answer, parsed.rationale, scenario.options
            )
            rationale_letter = audit.conclusion_letters[-1] if audit.conclusion_letters else None
            rationale_answers.append(rationale_letter)
            samples.append({
                "sample_index": sample_index,
                "answer": parsed.answer,
                "rationale": parsed.rationale,
                "raw_response": parsed.raw_response,
                "parse_failed": parsed.parse_failed,
                "rationale_concludes": rationale_letter,
                "rationale_conclusion_letters": audit.conclusion_letters,
                "rationale_conclusion_sources": audit.conclusion_sources,
                "inconsistent": audit.inconsistent,
            })
            if audit.inconsistent:
                inconsistent_count += 1
                print(
                    f"    ! rationale/answer mismatch: {scenario.scenario_id} "
                    f"answer={parsed.answer} rationale_concludes={rationale_letter} "
                    f"rationale={parsed.rationale!r}",
                    file=sys.stderr,
                )
        except Exception as exc:  # a failed call is data, not a crash
            answers.append(None)
            rationale_answers.append(None)
            samples.append({
                "sample_index": sample_index,
                "answer": None,
                "rationale": None,
                "raw_response": None,
                "parse_failed": True,
                "rationale_concludes": None,
                "rationale_conclusion_letters": [],
                "rationale_conclusion_sources": [],
                "inconsistent": None,
                "generation_error_type": type(exc).__name__,
                "generation_error_message": str(exc)[:1200],
            })
            print(f"    ! generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    correct = sum(a == scenario.correct_answer for a in answers)
    rationale_correct = sum(a == scenario.correct_answer for a in rationale_answers)
    return {
        "scenario_id": scenario.scenario_id,
        "correct_answer": scenario.correct_answer,
        "n": len(answers),
        "n_correct": correct,
        "accuracy": correct / len(answers) if answers else 0.0,
        "answers": dict(Counter(a or "PARSE_FAIL" for a in answers)),
        "samples": samples,
        "rationale_inconsistent_count": inconsistent_count,
        "rationale_accuracy": rationale_correct / len(answers) if answers else 0.0,
        "rationale_answers": dict(Counter(a or "NONE" for a in rationale_answers)),
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
        mismatch_flag = (
            f"  [{result['rationale_inconsistent_count']}/{result['n']} rationale/answer mismatch, "
            f"rationale-accuracy={result['rationale_accuracy']:>4.0%}]"
            if result["rationale_inconsistent_count"]
            else ""
        )
        print(f"  {result['scenario_id']:16s} key={result['correct_answer']}  "
              f"{result['n_correct']}/{result['n']} = {result['accuracy']:>4.0%}  "
              f"{result['verdict']:5s}  answers={result['answers']}{mismatch_flag}")

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

    mismatched = [r for r in rows if r["rationale_inconsistent_count"]]
    if mismatched:
        total_mismatches = sum(r["rationale_inconsistent_count"] for r in mismatched)
        total_samples = sum(r["n"] for r in rows)
        print(f"\n{total_mismatches}/{total_samples} samples had a rationale/answer "
              f"letter mismatch (the model's prose names one option, the \"answer\" "
              f"field names another). This is a binding defect, not a scenario or "
              f"comprehension problem -- it inflates FAIL/WEAK verdicts for models "
              f"prone to it. rationale_accuracy in the JSON output rescores using "
              f"the rationale's own conclusion instead of the answer field:")
        for r in mismatched:
            print(f"  - {r['scenario_id']}: accuracy={r['accuracy']:>4.0%} vs "
                  f"rationale_accuracy={r['rationale_accuracy']:>4.0%} "
                  f"({r['rationale_inconsistent_count']}/{r['n']} mismatched)")

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
