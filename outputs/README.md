# outputs/

Experiment output. Everything here except this README is git-ignored: run
data is large, machine-specific, and (for real runs) reproducible from the
code + scenarios, so it is NOT committed. Each person regenerates their own
by running the pipeline.

## Layout

```
outputs/
├── raw/<stamp>/          one folder per run (stamp = timestamp + client)
│   ├── seeds.jsonl       frozen seed responses (Stage A)
│   ├── records.jsonl     every agent-round: who said what, when, having heard whom
│   └── results.jsonl     per-conversation vote summary (final tally, winner, correct?)
└── judged/<stamp>/        produced AFTER the judge labels a run
    ├── merged.jsonl      records + judge labels, joined
    └── report.json       merge coverage + any invalid/orphan/unjudged/conflict rows
```

- `raw/` = straight from the experiment, before any judging. The `<stamp>`
  suffix (`_mock` / `_azure`) shows which client produced it. Mock runs are
  disposable smoke tests; real Azure runs are experimental data — keep and
  back those up outside the repo.
- `judged/` = written by `scripts/merge_judged.py` after the teammate judge
  labels a run. Same `<stamp>` so raw and judged pair up.

## Regenerate

```
uv run python scripts/run_pilot.py --client mock --scenario travel_001
```

See `docs/JUDGE_INTERFACE.md` for the records/judge formats.
