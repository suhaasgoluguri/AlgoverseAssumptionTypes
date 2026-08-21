# outputs/

Experiment output.

`raw/` and `reports/` are git-ignored: run data is large and machine-specific,
and both are reproducible from the corpus plus a run manifest. Real
experimental runs should also be backed up outside the repository.

The small derived summaries in this directory (`adoption_*.json`,
`oracle_*.json`, `transit_diag.json`, and `validation/`) are tracked, because
they are what the write-up cites and they stay small.

## Layout

```Shell
outputs/
|-- raw/<stamp>/                 (git-ignored)
|   |-- seeds.jsonl              frozen seed responses (Stage A)
|   |-- records.jsonl            every agent-round response and routing envelope
|   |-- results.jsonl            per-conversation answer/vote summary
|   |-- run_manifest.json        corpus hash, protocol snapshot, provenance
|   `-- analysis/                written by scripts/analyze_run.py
|-- validation/                  oracle checks and run launchers
`-- *.json                       adoption and oracle summaries
```

- `raw/` is written directly by `scripts/run_pilot.py`.
- `analysis/` under a run dir is written by `scripts/analyze_run.py` and
  `scripts/analyze_statistics.py`.
- Mock runs are disposable plumbing checks and are not research results.

`docs/sample_records.jsonl` is the in-repo example of the `records.jsonl`
format, so the schema can be read without a run in hand.

## Regenerate a mock run

```bash
uv run python scripts/run_pilot.py --client mock --scenario travel_001
```
