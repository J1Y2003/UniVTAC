#!/usr/bin/env python3
"""Triage running evaluation jobs by reading their logs.

    python scripts/eval_triage.py              # jobs currently in the queue
    python scripts/eval_triage.py --all        # every log under logs/
    python scripts/eval_triage.py --quiet      # only jobs worth acting on

An eval job that cannot start still holds its allocation for the full
``--startup-timeout 1800``: the GR00T server dies in its half of the ``.err``,
and the evaluator only notices half an hour later, reporting a ``TimeoutError``
that blames a gated-repo 401 whatever the real cause was. This reads the
server's half directly, so a doomed job is visible in seconds rather than 30
minutes.

Root causes are matched ahead of that downstream timeout, so a job that has
already timed out is still reported by whatever actually killed it.

Standard library only. ``squeue`` is the sole external call, and ``--all``
avoids even that.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# First match wins, so these run most-specific first. Each entry is
# (label, pattern, what to do about it).
_CAUSES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "bad-checkpoint-path",
        re.compile(r"HFValidationError|Repo id must be in the form"),
        "GROOT_MODEL does not exist on disk, so it was taken for a Hub repo id. "
        "Check it with `ls $GROOT_MODEL/config.json`.",
    ),
    (
        "gated-repo-401",
        re.compile(r"401 Client Error|GatedRepoError|Cannot access gated repo"),
        "HF auth. Export HF_TOKEN in the submitting shell -- never `hf auth login`, "
        "which writes into the shared account's token store.",
    ),
    (
        "gpu-oom",
        re.compile(
            r"CUDA error: out of memory|cudaErrorMemoryAllocation|"
            r"torch\.OutOfMemoryError|CUDA out of memory"
        ),
        "The GPU was already full when the server loaded. Check for a stray "
        "CUDA_VISIBLE_DEVICES in the submitting shell (--export=ALL carries it in and "
        "eval.sbatch does not scrub it), then check which node it landed on.",
    ),
    (
        "cudnn-pin",
        re.compile(r"CUDNN_STATUS_NOT_INITIALIZED"),
        "The GR00T venv's cuDNN is not 9.10.2.21. Run scripts/preflight.py --deep; "
        "`uv pip list` will lie about this, only cudnnGetVersion() catches it.",
    ),
    (
        "port-in-use",
        re.compile(r"Address already in use|EADDRINUSE|zmq\.error\.ZMQError"),
        "Another process on this node holds $PORT. Vary PORT per job.",
    ),
    (
        "missing-task",
        re.compile(r"could not load modality config|KeyError: 'TASK'"),
        "`export TASK=<task>` reaches the modality config at import; both halves need it.",
    ),
    (
        "import-error",
        re.compile(r"ModuleNotFoundError|ImportError: cannot import name"),
        "Wrong interpreter for one of the two processes -- check GROOT_PYTHON / UNIVTAC_PYTHON.",
    ),
]

# Reported only when nothing above matched: this is the symptom, not the cause.
_TIMEOUT = re.compile(r"was not ready within \d+s")
# `run_eval.py` prints this before anything expensive.
_OUTPUT_LINE = re.compile(r"^\[eval\] .*-> (.+)$")
# `run_server.py` prints this *before* the load, so it appears even when the
# path is wrong and the server then dies. Never read it as proof of anything.
_MODEL_LINE = re.compile(r"^\[server\] loading (\S+) ")


def parse_elapsed(text: str) -> int:
    """squeue's %M (``3:04``, ``1:16:53``, ``2-03:04:05``) in seconds; -1 if unparsable."""
    text = text.strip()
    days = 0
    if "-" in text:
        head, _, text = text.partition("-")
        try:
            days = int(head)
        except ValueError:
            return -1
    try:
        nums = [int(part) for part in text.split(":")]
    except ValueError:
        return -1
    while len(nums) < 3:
        nums.insert(0, 0)
    return days * 86400 + nums[0] * 3600 + nums[1] * 60 + nums[2]


def queued_jobs(pattern: str) -> list[dict]:
    """Jobs in the queue whose name contains `pattern`."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    cmd = ["squeue", "-h", "-o", "%i|%j|%T|%M|%P|%R"]
    if user:
        cmd[1:1] = ["-u", user]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"error: could not run squeue ({exc}). Use --all to read logs directly.")
    jobs = []
    for line in out.splitlines():
        fields = [f.strip() for f in line.split("|")]
        if len(fields) < 6 or pattern not in fields[1]:
            continue
        jobs.append({
            "jobid": fields[0], "name": fields[1], "state": fields[2],
            "elapsed": parse_elapsed(fields[3]), "partition": fields[4],
            "reason": fields[5],
        })
    return jobs


def logged_jobs(logs: Path, pattern: str) -> list[dict]:
    """Every ``<name>-<jobid>.err`` under `logs`, for when squeue is unavailable."""
    jobs = []
    for path in sorted(logs.glob("*.err")):
        name, _, jobid = path.stem.rpartition("-")
        if not jobid.isdigit() or pattern not in name:
            continue
        jobs.append({"jobid": jobid, "name": name, "state": "?", "elapsed": -1,
                     "partition": "?", "reason": ""})
    return jobs


def scan(path: Path) -> tuple[str, str, str] | None:
    """First root cause in `path`, as ``(label, offending line, advice)``."""
    timed_out = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for label, rx, advice in _CAUSES:
                    if rx.search(line):
                        return label, line.strip()[:160], advice
                if _TIMEOUT.search(line):
                    timed_out = True
    except OSError as exc:
        return "unreadable-log", str(exc), "Check the path and permissions."
    if timed_out:
        return (
            "startup-timeout",
            "server never became ready",
            "The server's own traceback is earlier in this same .err -- read that "
            "rather than the timeout message, which names a cause it did not verify.",
        )
    return None


def read_out(out_path: Path) -> tuple[str | None, Path | None]:
    """The model path and results JSONL a job announced in its ``.out``."""
    try:
        text = out_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    model, target = None, None
    for line in text.splitlines():
        if model is None:
            found = _MODEL_LINE.match(line)
            if found:
                model = found.group(1).strip()
        match = _OUTPUT_LINE.match(line)
        if match:
            target = match.group(1).strip()
    if target is None:
        return model, None
    jsonl = Path(target).expanduser()
    if not jsonl.is_absolute():
        jsonl = _REPO_ROOT / jsonl
    return model, jsonl


def count_seeds(jsonl: Path) -> int:
    """Distinct seeds recorded in a results file."""
    if not jsonl.exists():
        return 0
    seeds = set()
    with jsonl.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                seed = json.loads(line).get("seed")
            except json.JSONDecodeError:
                continue
            if isinstance(seed, int):
                seeds.add(seed)
    return len(seeds)


def episode_count(out_path: Path) -> int | None:
    """Distinct seeds written so far, via the JSONL `run_eval.py` announced."""
    _, jsonl = read_out(out_path)
    return None if jsonl is None else count_seeds(jsonl)


def audit(logs: Path, pattern: str) -> int:
    """Group every log by the results file it wrote, and name the model behind it.

    The question this answers is which checkpoint directory actually produced
    the episodes in a given JSONL. A job that died during startup contributed
    nothing -- `ResultWriter` is constructed after `build_env` in
    `run_eval.py`, so a failed job never opens the file -- and is listed as
    such rather than counted. A results file that two *surviving* jobs wrote
    from different model directories is flagged MIXED: that is the only case
    where the number in the table is not attributable to one checkpoint.
    """
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for path in sorted(logs.glob("*.err")):
        name, _, jobid = path.stem.rpartition("-")
        if not jobid.isdigit() or pattern not in name:
            continue
        model, jsonl = read_out(path.with_suffix(".out"))
        cause = scan(path)
        # Every cause we match kills the server before any episode is written.
        status = f"died: {cause[0]}" if cause else "ran"
        key = str(jsonl) if jsonl else "(no results file announced)"
        groups.setdefault(key, []).append((jobid, status, model or "(none logged)"))

    mixed = 0
    for key in sorted(groups):
        entries = groups[key]
        ran = {model for _, status, model in entries if status == "ran"}
        on_disk = 0 if key.startswith("(") else count_seeds(Path(key))
        flag = "  [MIXED]" if len(ran) > 1 else ""
        if flag:
            mixed += 1
        print(f"\n{key}  --  {on_disk} episodes on disk{flag}")
        for jobid, status, model in sorted(entries):
            short = model.split("/jaewon/", 1)[-1]
            print(f"  {jobid:>8}  {status:<28}  {short}")

    if mixed:
        print(f"\n{mixed} results file(s) were written by more than one model directory. "
              "Those numbers are not attributable to a single checkpoint.")
        return 1
    print("\nNo results file was written by more than one model directory.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--logs", type=Path, default=_REPO_ROOT / "logs")
    parser.add_argument("--all", action="store_true",
                        help="scan every log under --logs instead of asking squeue")
    parser.add_argument("--match", default="univtac-groot",
                        help="only jobs whose name contains this (default: univtac-groot)")
    parser.add_argument("--warn-after", type=int, default=20,
                        help="minutes a job may run with no episode before it is flagged "
                             "(default 20; --startup-timeout is 30)")
    parser.add_argument("--quiet", action="store_true", help="hide healthy jobs")
    parser.add_argument("--audit", action="store_true",
                        help="group every log by the results file it wrote and name the "
                             "model directory behind it; exit 1 if any file is MIXED")
    args = parser.parse_args(argv)

    if args.audit:
        return audit(args.logs, args.match)

    jobs = logged_jobs(args.logs, args.match) if args.all else queued_jobs(args.match)
    if not jobs:
        print("no matching jobs.")
        return 0

    doomed: list[str] = []
    rows: list[tuple[str, str, str, str]] = []
    notes: dict[str, str] = {}

    for job in jobs:
        errs = sorted(args.logs.glob(f"*-{job['jobid']}.err"))
        if not errs:
            rows.append((job["jobid"], job["name"], "NO LOG", "nothing written yet"))
            continue
        found = scan(errs[0])
        scored = episode_count(errs[0].with_suffix(".out"))

        if found:
            label, line, advice = found
            rows.append((job["jobid"], job["name"], f"FAILING/{label}", line))
            notes[label] = advice
            doomed.append(job["jobid"])
        elif scored:
            rows.append((job["jobid"], job["name"], "OK", f"{scored} episodes written"))
        elif job["elapsed"] > args.warn_after * 60:
            rows.append((job["jobid"], job["name"], "STALLED",
                         f"running {job['elapsed'] // 60} min with no episode yet"))
        else:
            rows.append((job["jobid"], job["name"], "starting", "no episode yet"))

    width = max(len(row[1]) for row in rows)
    for jobid, name, status, detail in rows:
        if args.quiet and status in {"OK", "starting"}:
            continue
        print(f"{jobid:>8}  {name:<{width}}  {status:<28}  {detail}")

    if notes:
        print()
        for label, advice in notes.items():
            print(f"{label}: {advice}")
    if doomed:
        print("\nThese cannot recover, and will hold their allocation until the "
              "startup timeout expires:")
        print(f"  scancel {' '.join(doomed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
