"""Pilot runner: Stage A seed generation + Stage B matched propagation.

Usage (from the repo root):

  # Free dry run with the mock client (no API key needed):
  uv run python scripts/run_pilot.py --client mock --scenario travel_001

  # Live run against the Azure deployment in .env:
  uv run python scripts/run_pilot.py --client azure --scenario travel_001 --variants factual

  # Everything in the file, all variants, all topologies:
  uv run python scripts/run_pilot.py --client azure

Outputs (append-only JSONL, one directory per run timestamp):
  outputs/raw/<stamp>/seeds.jsonl     frozen seed instances
  outputs/raw/<stamp>/records.jsonl   every agent-round record
  outputs/raw/<stamp>/results.jsonl   per-conversation vote summaries
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from assumption_propagation.aggregation import summarize_final_round
from assumption_propagation.clients import AzureChatClient, MockClient
from assumption_propagation.orchestrator import (
    generate_seed_instances,
    run_conversation,
    run_matched_triple,
)
from assumption_propagation.schemas import Scenario

TOPOLOGIES = ("looped_chain", "neutral_star", "fully_connected_mesh")


def load_scenarios(path: Path, only: str | None) -> list[Scenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = []
    for entry in raw:
        if "scenario_id" not in entry:
            print(f"  ! skipping malformed entry (no scenario_id): {entry.get('id', '?')}")
            continue
        if only and entry["scenario_id"] != only:
            continue
        scenarios.append(Scenario.model_validate(entry))
    return scenarios


async def run_one_variant(scenario, variant, client, args, paths) -> None:
    label = f"{scenario.scenario_id}/{variant.semantic_type.value}"
    print(f"\n=== {label} ===")

    # Stage A: frozen seeds (every instance retained).
    seeds = await generate_seed_instances(
        scenario, variant, client, n_instances=args.n_seeds, log_path=paths["seeds"]
    )
    for seed in seeds:
        state = "PARSE-FAILED" if not seed.answer else f"answer={seed.answer}"
        print(f"  seed {seed.seed_trial_index}: {state}")

    # Stage B: matched replay of every replayable seed.
    for seed in seeds:
        if not seed.answer:
            continue  # nothing to replay; the failure is already logged
        triple = await run_matched_triple(
            scenario, variant, seed, client,
            topologies=tuple(args.topologies), log_path=paths["records"],
        )
        for topology_name, result in triple.items():
            summary = summarize_final_round(
                result.final_answers(),
                correct_answer=scenario.correct_answer,
                favored_option=variant.favored_option,
            )
            row = {
                "run_id": result.run_id,
                "scenario_id": scenario.scenario_id,
                "assumption_id": variant.assumption_id,
                "seed_instance_id": seed.seed_instance_id,
                "condition": "injection",
                "topology": topology_name,
                **summary,
            }
            with paths["results"].open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            verdict = summary["plurality_answer"] or "TIE"
            flag = ("correct" if summary["final_answer_correct"]
                    else "no consensus" if summary["no_consensus"] else "WRONG")
            print(f"    {seed.seed_instance_id} @ {topology_name:22s}"
                  f" votes={summary['final_votes']} -> {verdict} ({flag})")

    # Independent no-communication control: baseline information state
    # (neutral seed note, corrector evidence), agents answer once, no
    # messages. Separates communication effects from correlated reasoning.
    if not args.skip_independent:
        for rep in range(args.independent_reps):
            result = await run_conversation(
                scenario, client=client, topology_name="independent",
                condition="independent_control", variant=variant,
                n_rounds=1, log_path=paths["records"],
            )
            summary = summarize_final_round(
                result.final_answers(),
                correct_answer=scenario.correct_answer,
                favored_option=variant.favored_option,
            )
            row = {
                "run_id": result.run_id,
                "scenario_id": scenario.scenario_id,
                "assumption_id": variant.assumption_id,
                "seed_instance_id": None,
                "condition": "independent_control",
                "topology": "independent",
                **summary,
            }
            with paths["results"].open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"    independent rep {rep}: votes={summary['final_votes']}"
                  f" -> {summary['plurality_answer'] or 'TIE'}")

    # Topology-matched no-injection baselines.
    if not args.skip_baselines:
        for topology_name in args.topologies:
            for rep in range(args.baseline_reps):
                result = await run_conversation(
                    scenario, client=client, topology_name=topology_name,
                    condition="no_injection_baseline", variant=variant,
                    log_path=paths["records"],
                )
                summary = summarize_final_round(
                    result.final_answers(),
                    correct_answer=scenario.correct_answer,
                    favored_option=variant.favored_option,
                )
                row = {
                    "run_id": result.run_id,
                    "scenario_id": scenario.scenario_id,
                    "assumption_id": variant.assumption_id,
                    "seed_instance_id": None,
                    "condition": "no_injection_baseline",
                    "topology": topology_name,
                    **summary,
                }
                with paths["results"].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"    baseline rep {rep} @ {topology_name:22s}"
                      f" -> {summary['plurality_answer'] or 'TIE'}"
                      f" ({'correct' if summary['final_answer_correct'] else 'not correct'})")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--client", choices=("mock", "azure"), default="mock")
    parser.add_argument("--data", default="data/scenarios.json")
    parser.add_argument("--scenario", default=None, help="single scenario_id; default all")
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=("factual", "causal", "framing"),
                        help="semantic types to run; default all present")
    parser.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES),
                        choices=TOPOLOGIES)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--baseline-reps", type=int, default=1)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--independent-reps", type=int, default=1)
    parser.add_argument("--skip-independent", action="store_true")
    parser.add_argument("--out", default="outputs/raw")
    parser.add_argument("--temperature", default="0.2",
                        help="sampling temperature for the azure client, or 'none' to "
                             "omit the parameter (deployments that reject custom "
                             "temperatures). Ignored by the mock client.")
    args = parser.parse_args()
    temperature = None if args.temperature.lower() == "none" else float(args.temperature)

    scenarios = load_scenarios(Path(args.data), args.scenario)
    if not scenarios:
        sys.exit(f"No scenarios matched (data={args.data}, scenario={args.scenario})")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.client}"
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / f"{name}.jsonl" for name in ("seeds", "records", "results")}

    client = (MockClient() if args.client == "mock"
              else AzureChatClient(temperature=temperature))
    print(f"client={client.model_identifier}  scenarios={len(scenarios)}  out={out_dir}")

    for scenario in scenarios:
        for variant in scenario.assumption_variants:
            if args.variants and variant.semantic_type.value not in args.variants:
                continue
            await run_one_variant(scenario, variant, client, args, paths)

    print(f"\nDone. Raw logs in {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
