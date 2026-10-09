"""Run one run_pilot.py process per scenario, several at a time.

run_pilot.py works through scenarios one after another, so a long reasoning
model leaves most of the deployment's quota idle. Scenarios share nothing (each
has its own seeds, conversations and bootstrap cluster), so running them in
separate processes collects the same design faster. Each scenario gets its own
run folder, written by the unmodified runner with identical arguments apart
from --scenario and --tag. combine_runs.py then joins the folders into one run
and checks that they match.

Usage (arguments after -- go to run_pilot.py unchanged):
  uv run python scripts/run_parallel.py --workers 11 --tag scope33_luna -- \\
      --client azure --data data/scenarios_hidden_profile_candidates.json ...

  # Leave out scenarios that are already collected elsewhere:
  uv run python scripts/run_parallel.py --workers 11 --tag scope33_luna \\
      --skip-scenarios vendor_001 -- --client azure ...

Writes outputs/parallel/<stamp>_<tag>/parallel_manifest.json (scenario -> run
folder, exit code, timings) and one log per scenario next to it. A failed
scenario is reported, not retried. The children share this process group, so
Ctrl-C or closing the terminal stops all of them rather than orphaning any.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PILOT = REPO_ROOT / "scripts" / "run_pilot.py"
# run_pilot prints this once its run folder exists.
OUT_DIR_PATTERN = re.compile(r"^client=\S+\s+scenarios=\d+\s+out=(.+?)\s*$")
# Chosen per scenario by this launcher; passing them through would collide.
RESERVED_ARGS = ("--scenario", "--tag")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_relative(path: str | Path) -> str:
    # run_pilot records its argv in the committed manifest; an absolute script
    # path would carry the local home directory. Children run with
    # cwd=REPO_ROOT, so the relative form resolves the same.
    path = Path(path)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _option_value(argv: list[str], name: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == name and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def scenario_ids(data_path: Path) -> list[str]:
    """Executable scenario ids in corpus order (legacy entries excluded)."""
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return [
        entry["scenario_id"]
        for entry in raw
        if isinstance(entry, dict)
        and "scenario_id" in entry
        and entry.get("legacy") is not True
    ]


def run_dir_from_log(log_path: Path) -> str | None:
    with log_path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            match = OUT_DIR_PATTERN.match(line.strip())
            if match:
                return match.group(1)
    return None


class Launcher:
    def __init__(self, args: argparse.Namespace, pilot_args: list[str], group_dir: Path):
        self.args = args
        self.pilot_args = pilot_args
        self.group_dir = group_dir
        self.manifest_path = group_dir / "parallel_manifest.json"
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.entries: dict[str, dict] = {}
        self.stopping = False
        self.created = _now()

    def write_manifest(self, state: str) -> None:
        manifest = {
            "kind": "run_parallel_v1",
            "state": state,
            "created_at_utc": self.created,
            "updated_at_utc": _now(),
            "tag": self.args.tag,
            "workers": self.args.workers,
            "run_pilot_args": self.pilot_args,
            "scenarios": self.entries,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    async def run_scenario(self, scenario_id: str) -> None:
        tag = f"{self.args.tag}_{scenario_id}"
        log_path = self.group_dir / f"{scenario_id}.log"
        command = [
            sys.executable, _repo_relative(RUN_PILOT), *self.pilot_args,
            "--scenario", scenario_id, "--tag", tag,
        ]
        entry = self.entries[scenario_id]
        entry.update(state="running", started_at_utc=_now(), log=log_path.name)
        self.write_manifest("running")
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # Each run records `git status`; without this, concurrent starts
            # can contend for the index lock. The status output is unchanged.
            "GIT_OPTIONAL_LOCKS": "0",
        }
        # The child writes straight to its log file: no pipe that could fill
        # or break, and no line-length limit on what the runner prints.
        with log_path.open("w", encoding="utf-8") as log:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=REPO_ROOT, env=env, stdout=log, stderr=log,
            )
            self.processes[scenario_id] = process
            exit_code = await process.wait()
        self.processes.pop(scenario_id, None)
        run_dir = run_dir_from_log(log_path)
        if run_dir is not None:
            entry["run_dir"] = run_dir
        if self.stopping:
            state = "stopped"
        else:
            state = "complete" if exit_code == 0 else "failed"
        entry.update(state=state, exit_code=exit_code, finished_at_utc=_now())
        self.write_manifest("running")
        done = sum(e["state"] == "complete" for e in self.entries.values())
        print(
            f"[{datetime.now():%H:%M:%S}] {scenario_id}: {state} (exit {exit_code})  "
            f"{done}/{len(self.entries)} complete  run={run_dir or '?'}",
            flush=True,
        )

    async def worker(self, index: int, queue: asyncio.Queue[str]) -> None:
        # Stagger the first launches so runs do not all start in the same
        # second (git provenance checks, first round of requests).
        await asyncio.sleep(index * self.args.stagger_seconds)
        while not self.stopping:
            try:
                scenario_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            print(f"[{datetime.now():%H:%M:%S}] {scenario_id}: started", flush=True)
            await self.run_scenario(scenario_id)

    def stop(self) -> None:
        """Stop launching and end running scenarios.

        Ctrl-C in the terminal already reaches the children (same process
        group); this also covers a signal sent to the launcher alone.
        """
        if self.stopping:
            return
        self.stopping = True
        print("\nstopping: no new scenarios; ending running ones...", flush=True)
        for process in self.processes.values():
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

    async def run(self, selected: list[str]) -> int:
        self.entries = {scenario_id: {"state": "queued"} for scenario_id in selected}
        self.write_manifest("running")
        loop = asyncio.get_running_loop()
        # SIGHUP (terminal closed) keeps its default: every process in the
        # group, launcher and children alike, ends.
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)
        queue: asyncio.Queue[str] = asyncio.Queue()
        for scenario_id in selected:
            queue.put_nowait(scenario_id)
        await asyncio.gather(
            *(self.worker(index, queue) for index in range(self.args.workers))
        )
        states = [entry["state"] for entry in self.entries.values()]
        final = (
            "stopped" if self.stopping
            else "complete" if all(s == "complete" for s in states)
            else "failed"
        )
        self.write_manifest(final)
        failed = [s for s, e in self.entries.items() if e["state"] == "failed"]
        print(f"\n{final}: {states.count('complete')}/{len(states)} scenarios complete")
        if failed:
            print(f"failed (see logs in {self.group_dir}): {', '.join(failed)}")
        print(f"manifest: {self.manifest_path}")
        return 0 if final == "complete" else 1


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("pass run_pilot.py arguments after --")
    split = argv.index("--")
    own_argv, pilot_args = argv[:split], argv[split + 1:]

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, required=True,
                        help="scenarios to run at once; keep within the deployment quota")
    parser.add_argument("--tag", required=True,
                        help="base tag; each run folder is tagged <tag>_<scenario_id>")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="scenario ids to run (default: every scenario in --data)")
    parser.add_argument("--skip-scenarios", nargs="+", default=[],
                        help="scenario ids to leave out, e.g. ones already collected")
    parser.add_argument("--stagger-seconds", type=float, default=5.0,
                        help="delay between the first launches (default: 5)")
    parser.add_argument("--out", default="outputs/parallel",
                        help="where the launcher's manifest and logs go")
    args = parser.parse_args(own_argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.stagger_seconds < 0:
        parser.error("--stagger-seconds cannot be negative")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("--tag may contain only letters, digits, '_', '.', '-'")
    for name in RESERVED_ARGS:
        if _option_value(pilot_args, name) is not None or name in pilot_args:
            parser.error(f"{name} is set per scenario by this launcher; remove it after --")

    data = _option_value(pilot_args, "--data")
    if data is None:
        parser.error("pass --data explicitly after -- so every scenario uses the same corpus")
    data_path = Path(data) if Path(data).is_absolute() else REPO_ROOT / data
    available = scenario_ids(data_path)
    selected = args.scenarios or available
    unknown = sorted(set(selected) - set(available)) + sorted(
        set(args.skip_scenarios) - set(available)
    )
    if unknown:
        parser.error(f"not in {data}: {unknown}")
    selected = [s for s in selected if s not in set(args.skip_scenarios)]
    if not selected:
        parser.error("no scenarios left to run")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    group_dir = out_root / f"{stamp}_{args.tag}"
    group_dir.mkdir(parents=True, exist_ok=False)
    print(f"{len(selected)} scenarios, {args.workers} at a time -> {group_dir}", flush=True)
    launcher = Launcher(args, pilot_args, group_dir)
    sys.exit(asyncio.run(launcher.run(selected)))


if __name__ == "__main__":
    main()
