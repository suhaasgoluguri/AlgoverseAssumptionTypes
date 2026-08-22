"""Stage-A only: does the SEED act on the claim it was given?

Adoption and propagation are separate questions, and adoption needs no
conversations — only the seed's round-0 answer. Running the full pilot to
measure it wastes roughly 95% of the calls, so this generates seeds and stops.

The comparison this exists for: framing was adopted 0/50 in the main corpus,
but the framing seed is the only one that ALSO holds the true fact its claim
contradicts. That leaves two explanations — framing does not persuade, or
first-hand knowledge beats assertion — and the main corpus cannot separate
them. Run this against a corpus built with --additive-all, where every type
carries the honest finding alongside the injected claim, and the seed's
knowledge state is held constant across types:

  factual/causal adopted, framing not  -> a genuine semantic-type effect
  none adopted                         -> first-hand knowledge beats assertion,
                                          and the type finding dissolves

Usage:
  This wants the knowledge-constant build, which is not distributed with the
  repository. Regenerate it first, then point --data at it:

  uv run python scripts/build_hidden_profile_scenarios.py --additive-all \\
      --out data/scenarios_additive_all.json
  uv run python scripts/adoption_check.py --data data/scenarios_additive_all.json \\
      --client azure --n-seeds 5 --json outputs/adoption_check.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from assumption_propagation.clients import AzureChatClient, MockClient
from assumption_propagation.orchestrator import generate_seed_instances
from assumption_propagation.reproducibility import sha256_file
from assumption_propagation.schemas import Scenario

N_BOOTSTRAP = 10_000
RNG_SEED = 20260826


def cluster_ci(a: list[tuple[str, bool]], b: list[tuple[str, bool]]) -> tuple[float, float]:
    """Paired cluster bootstrap on a - b, resampling scenarios once per draw.

    Paired because the two types are measured on the same scenarios; drawing
    each arm independently would break that and inflate the interval.
    """
    scen = sorted({s for s, _ in a} | {s for s, _ in b})
    by_a = {s: [v for x, v in a if x == s] for s in scen}
    by_b = {s: [v for x, v in b if x == s] for s in scen}
    rng = np.random.default_rng(RNG_SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        drawn = rng.choice(scen, size=len(scen), replace=True)
        va = [v for s in drawn for v in by_a.get(s, [])]
        vb = [v for s in drawn for v in by_b.get(s, [])]
        if va and vb:
            diffs.append(np.mean(va) - np.mean(vb))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--client", choices=("mock", "azure"), default="mock")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--temperature", default="0.2")
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=("factual", "causal", "framing"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    entries = json.loads(args.data.read_text(encoding="utf-8"))
    scenarios = [Scenario.model_validate(e) for e in entries
                 if isinstance(e, dict) and "scenario_id" in e]
    if not scenarios:
        sys.exit(f"no scenarios in {args.data}")

    temperature = None if args.temperature.lower() == "none" else float(args.temperature)
    client = MockClient() if args.client == "mock" else AzureChatClient(temperature=temperature)

    planned = sum(
        len([v for v in s.assumption_variants
             if not args.variants or v.semantic_type.value in args.variants])
        for s in scenarios
    ) * args.n_seeds
    print(f"data={args.data} (sha256={sha256_file(args.data)[:16]})")
    print(f"client={client.model_identifier}  scenarios={len(scenarios)}  "
          f"seeds each={args.n_seeds}  calls={planned}\n")

    rows: list[dict] = []
    by_type: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for scenario in scenarios:
        for variant in scenario.assumption_variants:
            t = variant.semantic_type.value
            if args.variants and t not in args.variants:
                continue
            seeds = await generate_seed_instances(
                scenario, variant, client, n_instances=args.n_seeds
            )
            for seed in seeds:
                took = seed.answer == variant.favored_option
                by_type[t].append((scenario.scenario_id, took))
                rows.append({
                    "scenario_id": scenario.scenario_id, "semantic_type": t,
                    "trial": seed.seed_trial_index, "answer": seed.answer or None,
                    # An answer letter cannot say WHY a seed adopted or refused.
                    # Diagnosing outlier scenarios needs the reasoning text.
                    "rationale": seed.rationale or "",
                    "favored": variant.favored_option,
                    "correct": scenario.correct_answer, "adopted": took,
                })
            n = sum(v for _, v in by_type[t][-args.n_seeds:])
            print(f"  {scenario.scenario_id:16s} {t:8s} adopted {n}/{args.n_seeds}"
                  f"   answers={[s.answer for s in seeds]}")

    print(f"\n{'=' * 66}\nSEED ADOPTION by semantic type\n{'=' * 66}")
    for t in ("factual", "causal", "framing"):
        if not by_type[t]:
            continue
        vals = [v for _, v in by_type[t]]
        print(f"  {t:8s} {sum(vals):>3}/{len(vals):<4} = {np.mean(vals):>5.1%}")

    present = [t for t in ("factual", "causal", "framing") if by_type[t]]
    if len(present) > 1:
        print(f"\n{'pairwise (paired cluster bootstrap)':40s} {'difference':>11s}   95% CI")
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                da = np.mean([v for _, v in by_type[a]])
                db = np.mean([v for _, v in by_type[b]])
                lo, hi = cluster_ci(by_type[a], by_type[b])
                # Inclusive: when every observation is identical the bootstrap
                # has zero variance and lo == hi == 0, which is the clearest
                # possible NON-difference. A strict test mislabels it.
                flag = "" if lo <= 0 <= hi else "   <- distinguishable"
                label = f"{a} - {b}"
                print(f"  {label:38s} {da - db:>+10.1%}   "
                      f"[{lo:+.1%}, {hi:+.1%}]{flag}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "data": str(args.data), "data_sha256": sha256_file(args.data),
            "model": client.model_identifier, "n_seeds": args.n_seeds,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
