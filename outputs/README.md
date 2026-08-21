# outputs/

Experiment output.

`raw/` and `reports/` are git-ignored: run data is large and machine-specific,
and both are reproducible from the corpus plus a run manifest. Real
experimental runs should also be backed up outside the repository.

Diagnostic artifacts (`adoption_*.json`, per-model oracle checks,
`transit_diag.json`) are git-ignored too: each is regenerable by re-running
the script that wrote it, and nothing in the repo cites them.

Two things here are tracked on purpose:

- `oracle_34*.json` -- the ceiling test for the active corpus, pinned to its
  `data_sha256`. It is the evidence that each scenario is solvable at all, so
  a group failure is a finding about pooling rather than a broken task.
- `validation/*.ps1` and `validation/scenarios_scope_live6.json` -- the
  reproduction launchers README.md documents, and the corpus one of them
  reads.

## Layout

```Shell
outputs/
|-- raw/<stamp>/                 (git-ignored)
|   |-- seeds.jsonl              frozen seed responses (Stage A)
|   |-- records.jsonl            every agent-round response and routing envelope
|   |-- results.jsonl            per-conversation answer/vote summary
|   |-- run_manifest.json        corpus hash, protocol snapshot, provenance
|   `-- analysis/                written by scripts/analyze_run.py
|-- validation/                  run launchers (README.md documents these)
`-- oracle_34*.json              corpus ceiling test, pinned to a corpus hash
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
