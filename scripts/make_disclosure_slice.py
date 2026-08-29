"""Derive the shippable text slice a run's disclosure analyses need.

`records.jsonl` is ~295 MB per run and is not distributed, which would leave
the anonymity-leak and dynamics measurements unverifiable from a clone: every
other reported number rests on `results.jsonl`, which IS shipped, but anything
scored from message text would have to be taken on trust.

This writes `analysis/disclosure_slice.jsonl.gz`, holding only the ten fields
those two analyses read. It is a projection, not a summary -- the rationale
text is verbatim -- so `audit_disclosure_and_dynamics.py` and
`measure_anonymity_leakage.py` produce identical output from it, which the
--verify flag checks by re-deriving the slice and diffing field by field.

Nothing here is model-generated other than the rationale text the run already
produced; no prompt, system message, or corrective-record payload is copied,
so the slice cannot reconstruct the treatment a record received.

Usage:
  uv run python scripts/make_disclosure_slice.py <run-dir> [--verify]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

# Exactly the fields audit_disclosure_and_dynamics.py and
# measure_anonymity_leakage.py read off a record. Keep this list in sync with
# them; --verify fails loudly if a consumer starts reading something else.
FIELDS = (
    "scenario_id", "agent_id", "seed_agent_id", "corrector_agent_id",
    "round", "attribution", "condition", "topology", "parse_failed",
    "rationale",
)
SLICE_NAME = "disclosure_slice.jsonl.gz"


def project(rec: dict) -> dict:
    return {k: rec.get(k) for k in FIELDS}


def build(run: Path) -> Path:
    src = run / "records.jsonl"
    if not src.is_file():
        sys.exit(f"{src} not found; the slice can only be built where the "
                 "full transcripts are present")
    out = run / "analysis" / SLICE_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as fh:
        for line in src.open(encoding="utf-8"):
            fh.write(json.dumps(project(json.loads(line)),
                                ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    print(f"wrote {out}  ({n} records, "
          f"{out.stat().st_size / 1e6:.1f} MB gz, from "
          f"{src.stat().st_size / 1e6:.0f} MB)")
    return out


def verify(run: Path) -> int:
    """Re-project the transcripts and diff against the shipped slice."""
    src, sl = run / "records.jsonl", run / "analysis" / SLICE_NAME
    if not src.is_file():
        sys.exit(f"{src} not found; --verify needs the full transcripts")
    if not sl.is_file():
        sys.exit(f"{sl} not found; build it first")
    bad = n = 0
    with gzip.open(sl, "rt", encoding="utf-8") as fh:
        for line, want in zip(src.open(encoding="utf-8"), fh):
            n += 1
            if project(json.loads(line)) != json.loads(want):
                bad += 1
                if bad <= 3:
                    print(f"  mismatch at record {n}")
    extra = sum(1 for _ in src.open(encoding="utf-8")) - n
    print(f"{n} records compared, {bad} mismatched, {extra:+d} length difference")
    return 1 if (bad or extra) else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--verify", action="store_true",
                    help="diff the shipped slice against the transcripts")
    args = ap.parse_args()
    sys.exit(verify(args.run_dir) if args.verify else (build(args.run_dir) and 0))


if __name__ == "__main__":
    main()
