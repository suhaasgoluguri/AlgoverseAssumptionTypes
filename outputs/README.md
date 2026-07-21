# outputs/

Experiment output. Everything here except this README is git-ignored: run data
is large and machine-specific. Real experimental runs should also be backed up
outside the repository.

## Layout

```text
outputs/
|-- raw/<stamp>/
|   |-- seeds.jsonl          frozen seed responses (Stage A)
|   |-- records.jsonl        every agent-round response and routing envelope
|   `-- results.jsonl        per-conversation answer/vote summary
`-- judged/<stamp>/
    |-- judged.jsonl         validated semantic labels
    |-- judge_audit.jsonl    raw output, provider IDs/usage, cache/repair/failure audit
    |-- judge_report.json    coverage and a resumable configuration fingerprint
    |-- merged.jsonl         records + labels + direct-note/routed-content fields
    `-- report.json          invalid/orphan/unjudged/conflict merge report
```

- `raw/` is written directly by `scripts/run_pilot.py`, before judging.
- `judged/` is written by `scripts/run_judge.py` and
  `scripts/merge_judged.py`. The matching `<stamp>` keeps raw and judged
  artifacts paired.
- Mock runs are disposable plumbing checks. Their fixed labels do not measure
  annotation quality and merge/analysis reject them unless `--allow-mock` is
  explicitly supplied.

## Regenerate a mock run

```bash
uv run python scripts/run_pilot.py --client mock --scenario travel_001
```

See `docs/JUDGE_INTERFACE.md` for the record, judge, and merge contracts.
