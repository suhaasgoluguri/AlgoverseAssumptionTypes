"""Run the blinded semantic Judge LLM over one experiment record log.

Examples (from the repository root):

  # Free end-to-end plumbing check:
  uv run python scripts/run_judge.py outputs/raw/<stamp> --client mock

  # Production judge (ANTHROPIC_API_KEY must be configured):
  uv run python scripts/run_judge.py outputs/raw/<stamp> --client anthropic

The runner writes ``judged.jsonl``, ``judge_audit.jsonl``, and
``judge_report.json`` under ``outputs/judged/<stamp>/`` by default.  It does
not mutate raw logs or run the experiment orchestrator.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from assumption_propagation.analysis import load_agent_records, load_judged_rounds
from assumption_propagation.judge import (
    AnthropicJudgeClient,
    JUDGE_CODEBOOK_VERSION,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    MockJudgeClient,
    build_judge_prompt,
    build_judge_items,
    judge_items,
    judge_output_schema,
    load_assumption_references,
    load_codebook,
)
from assumption_propagation.reproducibility import resolve_scenario_data


def _key(row) -> tuple[str, int, int]:
    return row.run_id, row.agent_id, row.round


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _ordered_text_sha256(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="outputs/raw/<stamp> directory")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="scenario JSON override; default <run_dir>/scenario_snapshot.json, "
             "legacy runs without a snapshot require this override",
    )
    parser.add_argument("--codebook", type=Path, default=Path("docs/JUDGE_RUBRIC.md"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory; default outputs/judged/<run-stamp>")
    parser.add_argument(
        "--client",
        choices=("mock", "anthropic"),
        required=True,
        help="explicitly select mock development labels or the production provider",
    )
    parser.add_argument("--model", default=None,
                        help="Anthropic model ID; default ANTHROPIC_JUDGE_MODEL or "
                             "claude-sonnet-5")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="judge only the first N pending rows (development only)")
    parser.add_argument("--resume", action="store_true",
                        help="keep valid existing judged.jsonl rows and fill missing keys")
    parser.add_argument("--allow-partial", action="store_true",
                        help="return success with less than 100%% coverage (development only)")
    args = parser.parse_args()

    records_path = args.run_dir / "records.jsonl"
    if not records_path.exists():
        sys.exit(f"no records.jsonl in {args.run_dir}")
    try:
        data_path, data_source = resolve_scenario_data(args.run_dir, args.data)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
    if not args.codebook.exists():
        sys.exit(f"judge codebook not found: {args.codebook}")

    out_dir = args.out or Path("outputs/judged") / args.run_dir.name
    judged_path = out_dir / "judged.jsonl"
    audit_path = out_dir / "judge_audit.jsonl"
    report_path = out_dir / "judge_report.json"
    existing_artifacts = [
        path for path in (judged_path, audit_path, report_path) if path.exists()
    ]
    if existing_artifacts and not args.resume:
        sys.exit(
            f"judge output already exists ({existing_artifacts[0]}); pass --resume "
            "or choose a new --out directory"
        )
    if args.resume and not judged_path.exists() and existing_artifacts:
        sys.exit("cannot resume safely: auxiliary judge artifacts exist without judged.jsonl")

    records = load_agent_records(records_path)
    references = load_assumption_references(data_path)
    codebook = load_codebook(args.codebook)
    all_items = build_judge_items(records, references)
    item_by_key = {_key(item): item for item in all_items}
    if len(item_by_key) != len(all_items):
        sys.exit("records contain duplicate (run_id, agent_id, round) judge keys")

    client = (
        MockJudgeClient()
        if args.client == "mock"
        else AnthropicJudgeClient(model=args.model)
    )
    agent_models = sorted({record.model for record in records})
    if client.model_identifier in agent_models:
        sys.exit(
            "judge model exactly matches an agent model; use a different model family "
            "or identifier"
        )

    codebook_sha256 = _sha256(codebook.encode("utf-8"))
    judge_configuration = {
        "records_sha256": _sha256(records_path.read_bytes()),
        "scenario_data_sha256": _sha256(data_path.read_bytes()),
        "codebook_sha256": codebook_sha256,
        "judge_output_schema_sha256": _stable_json_sha256(judge_output_schema()),
        "judge_system_prompt_sha256": _sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")),
        "rendered_judge_prompts_sha256": _ordered_text_sha256(
            build_judge_prompt(item, codebook) for item in all_items
        ),
        "judge_codebook_version": JUDGE_CODEBOOK_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_model": client.model_identifier,
        "judge_temperature": 0 if args.client == "anthropic" else None,
        "judge_max_tokens": getattr(client, "max_tokens", None),
        "judge_max_retries": getattr(client, "max_retries", None),
    }
    resume_fingerprint = _stable_json_sha256(judge_configuration)
    if args.resume and judged_path.exists():
        if not report_path.exists():
            sys.exit(
                "cannot resume safely: judge_report.json is required to verify the "
                "original judging configuration"
            )
        try:
            prior_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"cannot resume safely: invalid judge_report.json: {exc}")
        if prior_report.get("resume_fingerprint") != resume_fingerprint:
            sys.exit(
                "cannot resume: records, scenario data, rubric, schema, prompt, model, "
                "or decoding settings changed; choose a new --out directory"
            )
        if not audit_path.exists():
            sys.exit(
                "cannot resume safely: judge_audit.jsonl is missing for existing labels"
            )

    existing = []
    invalid_existing: list[tuple[int, str]] = []
    if args.resume and judged_path.exists():
        existing, invalid_existing = load_judged_rounds(judged_path)
        if invalid_existing:
            sys.exit(
                f"cannot resume: {judged_path} contains {len(invalid_existing)} invalid rows"
            )
        seen: set[tuple[str, int, int]] = set()
        for annotation in existing:
            key = _key(annotation)
            item = item_by_key.get(key)
            if item is None:
                sys.exit(f"cannot resume: existing judge row has no source record: {key}")
            if key in seen:
                sys.exit(f"cannot resume: duplicate existing judge key: {key}")
            if annotation.assumption_id != item.assumption_id:
                sys.exit(f"cannot resume: assumption_id mismatch at {key}")
            if annotation.judged_text_source != item.judged_text_source:
                sys.exit(f"cannot resume: judged_text_source mismatch at {key}")
            if annotation.judge_codebook_version != JUDGE_CODEBOOK_VERSION:
                sys.exit(f"cannot resume: judge codebook version mismatch at {key}")
            if annotation.judge_prompt_version != JUDGE_PROMPT_VERSION:
                sys.exit(f"cannot resume: judge prompt version mismatch at {key}")
            if annotation.judge_model != client.model_identifier:
                sys.exit(f"cannot resume: judge model mismatch at {key}")
            expected_gap = item.verification_gap is not None
            observed_gap = annotation.verification_gap_retained is not None
            if expected_gap != observed_gap:
                sys.exit(
                    f"cannot resume: verification_gap_retained availability "
                    f"mismatch at {key}"
                )
            try:
                annotation.validate_evidence_span(item.response_text)
            except ValueError as exc:
                sys.exit(f"cannot resume: invalid evidence span at {key}: {exc}")
            seen.add(key)

        valid_audit_keys: set[tuple[str, int, int]] = set()
        for line_no, line in enumerate(
            audit_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                audit_row = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"cannot resume: invalid audit JSON on line {line_no}: {exc}")
            if audit_row.get("valid") is not True:
                continue
            try:
                audit_key = (
                    audit_row["run_id"],
                    int(audit_row["agent_id"]),
                    int(audit_row["round"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                sys.exit(f"cannot resume: malformed valid audit row {line_no}: {exc}")
            if audit_row.get("judge_model") != client.model_identifier:
                sys.exit(f"cannot resume: judge model mismatch in audit line {line_no}")
            if audit_row.get("judge_codebook_version") != JUDGE_CODEBOOK_VERSION:
                sys.exit(f"cannot resume: codebook mismatch in audit line {line_no}")
            if audit_row.get("judge_prompt_version") != JUDGE_PROMPT_VERSION:
                sys.exit(f"cannot resume: prompt mismatch in audit line {line_no}")
            if audit_key in valid_audit_keys:
                sys.exit(f"cannot resume: duplicate valid audit key {audit_key}")
            valid_audit_keys.add(audit_key)
        if valid_audit_keys != seen:
            missing_audit = sorted(seen - valid_audit_keys)
            extra_audit = sorted(valid_audit_keys - seen)
            sys.exit(
                "cannot resume: judged/audit valid-key mismatch; "
                f"missing_audit={missing_audit[:5]} extra_audit={extra_audit[:5]}"
            )

    existing_keys = {_key(row) for row in existing}
    pending = [item for item in all_items if _key(item) not in existing_keys]
    if args.limit is not None:
        if args.limit < 0:
            sys.exit("--limit must be non-negative")
        pending = pending[: args.limit]

    print(
        f"judge={client.model_identifier} records={len(records)} "
        f"targeted={len(all_items)} pending={len(pending)} out={out_dir}"
    )
    if args.client == "mock":
        print(
            "! mock labels are development-only and must not be used as research data",
            file=sys.stderr,
        )
    results = await judge_items(
        pending,
        client=client,
        codebook=codebook,
        concurrency=args.concurrency,
    )
    valid = [result.annotation for result in results if result.annotation is not None]
    failures = [result for result in results if result.annotation is None]

    if valid:
        _write_jsonl(
            judged_path,
            [annotation.model_dump(mode="json") for annotation in valid],
            append=bool(existing),
        )
    elif not judged_path.exists():
        _write_jsonl(judged_path, [])

    audit_rows = [
        {
            "run_id": result.item.run_id,
            "assumption_id": result.item.assumption_id,
            "agent_id": result.item.agent_id,
            "round": result.item.round,
            "judge_model": client.model_identifier,
            "judge_codebook_version": JUDGE_CODEBOOK_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judged_text_source": result.item.judged_text_source,
            "cache_hit": result.cache_hit,
            "parse_repaired": result.parse_repaired,
            "valid": result.annotation is not None,
            "error": result.error,
            "raw_judge_response": result.raw_judge_response,
            "provider_request_id": result.provider_request_id,
            "provider_message_id": result.provider_message_id,
            "provider_model": result.provider_model,
            "provider_stop_reason": result.provider_stop_reason,
            "provider_input_tokens": result.provider_input_tokens,
            "provider_output_tokens": result.provider_output_tokens,
            "provider_cache_creation_input_tokens": (
                result.provider_cache_creation_input_tokens
            ),
            "provider_cache_read_input_tokens": result.provider_cache_read_input_tokens,
        }
        for result in results
    ]
    if audit_rows:
        _write_jsonl(audit_path, audit_rows, append=args.resume and audit_path.exists())
    elif not audit_path.exists():
        _write_jsonl(audit_path, [])

    covered_keys = existing_keys | {
        _key(annotation) for annotation in valid if annotation is not None
    }
    expected_keys = set(item_by_key)
    missing = sorted(expected_keys - covered_keys)
    missing_gap_ids = sorted(
        {
            item.assumption_id
            for item in all_items
            if not (item.verification_gap or "").strip()
        }
    )
    report = {
        "records_total": len(records),
        "targeted_records": len(all_items),
        "variant_free_records_skipped": len(records) - len(all_items),
        "preexisting_valid": len(existing),
        "attempted": len(results),
        "new_valid": len(valid),
        "new_failures": len(failures),
        "covered": len(covered_keys),
        "coverage": len(covered_keys) / len(expected_keys) if expected_keys else 1.0,
        "missing_record_keys": [
            {"run_id": run_id, "agent_id": agent_id, "round": round_index}
            for run_id, agent_id, round_index in missing
        ],
        "failure_keys": [
            {
                "run_id": result.item.run_id,
                "agent_id": result.item.agent_id,
                "round": result.item.round,
                "error": result.error,
            }
            for result in failures
        ],
        "agent_models": agent_models,
        "scenario_data_path": str(data_path),
        "scenario_data_source": data_source,
        "development_only": args.client == "mock",
        **judge_configuration,
        "resume_fingerprint": resume_fingerprint,
        "missing_verification_gap_assumption_ids": missing_gap_ids,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"valid={len(valid)} failures={len(failures)} "
        f"coverage={report['coverage']:.1%} cache_hits="
        f"{sum(result.cache_hit for result in results)}"
    )
    print(f"wrote {judged_path}, {audit_path}, and {report_path}")
    if missing_gap_ids:
        print(
            f"! {len(missing_gap_ids)} assumption variants lack a separately stored "
            "verification_gap; see judge_report.json",
            file=sys.stderr,
        )
    if failures or (missing and not args.allow_partial and args.limit is None):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
