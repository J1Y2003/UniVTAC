#!/usr/bin/env python3
"""Cross-check `logs/*.out` against `eval_result/*/*/seed<N>-ckpt<step>.summary.json`.

Every `run_eval.py` invocation ends by printing `json.dumps(summary, ...)` to
stdout -- either the real end-of-run summary, or (if there was nothing left to
do) the same dict from `rewrite_summary()` in the early-return branch. So if a
`.out` log has no such JSON block in it, that job did not reach the end of
`main()`: it crashed, hit its walltime, or was killed -- regardless of how many
episodes it may have scored first.

This matters when a checkpoint was evaluated over several *manually*
resubmitted jobs (different job ids, not SLURM's own preempt/requeue): a
directory listing of `logs/` can't tell you whether the *last* job in that
chain actually finished, only that something with that name ran. This script
reads the last job for each (task, checkpoint, seed) group and tells you.

    python scripts/check_eval_logs.py
    python scripts/check_eval_logs.py --logs-dir logs --eval-result-dir eval_result/baseline_finetuned-50k

Standard library only -- run directly on the login node.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_LOG_RE = re.compile(
    r"^univtac-groot-evaluate-one-checkpoint-(?P<task>.+?)-ckpt(?P<step>\d+)"
    r"-seed(?P<seed>\d+)-(?P<jobid>\d+)\.out$"
)


def find_last_json_object(text: str) -> dict | None:
    """The last balanced top-level `{...}` in `text` that parses and looks like
    a run_eval.py summary (carries `episodes_scored`)."""
    spans = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, i + 1))
    for start, end in reversed(spans):
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "episodes_scored" in obj:
            return obj
    return None


def tail_lines(text: str, n: int = 8) -> list[str]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-n:]


def group_logs(logs_dir: Path) -> dict[tuple[str, int, int], list[tuple[int, Path]]]:
    groups: dict[tuple[str, int, int], list[tuple[int, Path]]] = {}
    for path in logs_dir.glob("*.out"):
        m = _LOG_RE.match(path.name)
        if not m:
            continue
        key = (m.group("task"), int(m.group("step")), int(m.group("seed")))
        groups.setdefault(key, []).append((int(m.group("jobid")), path))
    for key in groups:
        groups[key].sort(key=lambda t: t[0])
    return groups


def read_disk_summary(eval_result_dir: Path, task: str, step: int, seed: int) -> dict | None:
    path = eval_result_dir / task / f"seed{seed}-ckpt{step}.summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--eval-result-dir", default="eval_result/baseline_finetuned-50k",
                     help="directory holding <task>/seed<N>-ckpt<step>.summary.json")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir).expanduser()
    eval_result_dir = Path(args.eval_result_dir).expanduser()

    groups = group_logs(logs_dir)
    if not groups:
        print(f"no univtac-groot-evaluate-one-checkpoint-*.out under {logs_dir}", file=sys.stderr)
        return 1

    w_task = max(len(k[0]) for k in groups)
    rows_flagged = []
    print(f"{'task':<{w_task}} {'step':>7} {'seed':>4} {'jobs':>4} {'last jobid':>10} "
          f"{'log status':<28} {'log scored':>11} {'disk scored':>12}")

    for (task, step, seed), jobs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        last_jobid, last_path = jobs[-1]
        text = last_path.read_text(encoding="utf-8", errors="replace")
        summary = find_last_json_object(text)

        if summary is not None:
            log_scored = summary.get("episodes_scored")
            log_requested = summary.get("episodes_requested")
            log_status = "ok: reached end of main()"
            log_scored_str = f"{log_scored}/{log_requested}"
        else:
            log_scored = None
            log_status = "NO SUMMARY -- crashed/killed"
            log_scored_str = "-"

        disk = read_disk_summary(eval_result_dir, task, step, seed)
        if disk is not None:
            disk_scored_str = f"{disk.get('episodes_scored')}/{disk.get('episodes_requested')}"
        else:
            disk_scored_str = "(missing)"

        mismatch = (
            summary is not None and disk is not None
            and summary.get("episodes_scored") != disk.get("episodes_scored")
        )
        flag = " <-- MISMATCH" if mismatch else (" <-- CHECK" if summary is None else "")

        print(f"{task:<{w_task}} {step:>7} {seed:>4} {len(jobs):>4} {last_jobid:>10} "
              f"{log_status:<28} {log_scored_str:>11} {disk_scored_str:>12}{flag}")

        if summary is None:
            rows_flagged.append((task, step, seed, last_jobid, last_path))

    if rows_flagged:
        print("\nJobs whose last attempt never printed a summary (check the .err too):")
        for task, step, seed, jobid, out_path in rows_flagged:
            err_path = out_path.with_suffix(".err")
            print(f"\n  {task} ckpt{step} seed{seed}, job {jobid}: {out_path.name}")
            if err_path.exists():
                for line in tail_lines(err_path.read_text(encoding="utf-8", errors="replace")):
                    print(f"      {line}")
            else:
                print("      (no matching .err file)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
