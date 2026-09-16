#!/usr/bin/env python3
"""Eval progress across every task/checkpoint under an eval_result tree.

`results_table.py` only reads `*.summary.json`. `ResultWriter.close()` writes
that file in a `finally`, so a hard kill -- walltime, OOM, a preempted node --
can leave a `.jsonl` with episodes in it and no summary next to it. Those runs
are invisible to `results_table.py` and are exactly the ones worth finding: a
run still needs a job submitted to finish it. This script walks `*.jsonl`
instead and reports every one, counting scored/errored/skipped straight from
the file -- the same dedup-by-seed rule `run_eval.py`'s own resume logic uses
-- when there is no summary to read the authoritative numbers from.

    python scripts/eval_progress.py eval_result
    python scripts/eval_progress.py eval_result/baseline_finetuned-50k
    python scripts/eval_progress.py eval_result --json progress.json

Standard library only -- run directly on the login node, no venv needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_STEP_RE = re.compile(r"(?:ckpt|checkpoint[-_])(\d+)")
_SEED_RE = re.compile(r"seed(\d+)")

# run_eval.py's own default (docs/USAGE.md); used only to size the "requested"
# column for a run that has no summary yet to state it authoritatively.
DEFAULT_EPISODES_REQUESTED = 100


def scan_jsonl(path: Path) -> dict:
    """Count episodes the way `run_eval.py`'s resume logic does: parse every
    line, keep the last row per seed (an earlier pass may have retried it),
    and split into scored / errored / skipped."""
    by_seed: dict[int, dict] = {}
    malformed = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            seed = row.get("seed")
            if isinstance(seed, int):
                by_seed[seed] = row
    rows = list(by_seed.values())
    scored = [r for r in rows if not r.get("error") and not r.get("skipped")]
    return {
        "attempted": len(rows),
        "scored": len(scored),
        "errored": sum(1 for r in rows if r.get("error")),
        "skipped": sum(1 for r in rows if r.get("skipped")),
        "successes": sum(1 for r in scored if r.get("success")),
        "malformed_lines": malformed,
        "max_seed": max(by_seed) if by_seed else None,
    }


def collect(root: Path) -> list[dict]:
    """One row per `*.jsonl` under `root`, task/step/seed parsed from its path."""
    rows = []
    for jsonl in sorted(root.rglob("*.jsonl")):
        rel = jsonl.relative_to(root)
        run = str(rel)[: -len(".jsonl")]
        parts = rel.parts
        task = parts[-2] if len(parts) >= 2 else None
        variant = parts[-3] if len(parts) >= 3 else None
        step_match = _STEP_RE.search(jsonl.stem)
        seed_match = _SEED_RE.search(jsonl.stem)
        step = int(step_match.group(1)) if step_match else None
        seed_offset = int(seed_match.group(1)) if seed_match else None

        summary_path = jsonl.with_suffix(".summary.json")
        row = {
            "run": run,
            "task": task,
            "variant": variant,
            "step": step,
            "seed_offset": seed_offset,
            "has_summary": summary_path.exists(),
        }

        if row["has_summary"]:
            try:
                d = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"skipping unreadable {summary_path}: {exc}", file=sys.stderr)
                row["has_summary"] = False
                d = {}
            if row["has_summary"]:
                ci = d.get("success_rate_ci95") or [None, None]
                row.update({
                    "task": d.get("task", row["task"]),
                    "variant": d.get("variant", row["variant"]),
                    "scored": d.get("episodes_scored"),
                    "requested": d.get("episodes_requested"),
                    "sr_pct": d.get("success_rate_pct"),
                    "ci95_pct": [round(c * 100, 1) for c in ci] if ci[0] is not None else None,
                    "errored": d.get("episodes_errored"),
                    "skipped": d.get("episodes_skipped"),
                })
        if not row["has_summary"]:
            counts = scan_jsonl(jsonl)
            rate = (counts["successes"] / counts["scored"] * 100.0) if counts["scored"] else None
            row.update({
                "scored": counts["scored"],
                "requested": DEFAULT_EPISODES_REQUESTED,
                "sr_pct": round(rate, 1) if rate is not None else None,
                "ci95_pct": None,
                "errored": counts["errored"],
                "skipped": counts["skipped"],
                "malformed_lines": counts["malformed_lines"],
                "max_seed": counts["max_seed"],
            })
        rows.append(row)
    return rows


def status_of(row: dict) -> str:
    if not row["has_summary"]:
        return "INCOMPLETE"
    scored, requested = row.get("scored"), row.get("requested")
    if isinstance(scored, int) and isinstance(requested, int) and scored < requested:
        return "PARTIAL"
    return "done"


def print_table(rows: list[dict]) -> None:
    w_run = max(len(r["run"]) for r in rows)
    header = (f"{'run':<{w_run}} {'task':<14} {'step':>7} {'seed':>4} "
              f"{'scored':>9} {'SR%':>6} {'CI95%':>14} {'status':<11}")
    print(header)
    for r in rows:
        scored = f"{r['scored']}/{r['requested']}" if r.get("requested") is not None else str(r["scored"])
        sr = f"{r['sr_pct']:.1f}" if isinstance(r.get("sr_pct"), (int, float)) else "-"
        ci = f"[{r['ci95_pct'][0]}, {r['ci95_pct'][1]}]" if r.get("ci95_pct") else "-"
        status = status_of(r)
        mark = "*" if status != "done" else ""
        print(f"{r['run']:<{w_run}} {str(r['task']):<14} {str(r['step']):>7} "
              f"{str(r['seed_offset']):>4} {scored:>9} {sr:>6} {ci:>14} {status:<11}{mark}")

    incomplete = [r for r in rows if status_of(r) != "done"]
    print()
    print(f"{len(rows)} run(s) total, {len(rows) - len(incomplete)} done, "
          f"{len(incomplete)} not done.")
    if incomplete:
        print("\nnot done (no `.summary.json`, or fewer episodes scored than requested "
              "-- needs a job to finish it):")
        for r in incomplete:
            note = "no summary.json -- likely killed mid-run" if not r["has_summary"] else "partial summary"
            print(f"  {r['run']}: {r['scored']}/{r['requested']} scored, "
                  f"seed_offset={r['seed_offset']} ({note})")
        print("\n`requested` for a run with no summary.json is assumed "
              f"({DEFAULT_EPISODES_REQUESTED}, run_eval.py's own default) -- "
              "check the actual --episodes used if it differs.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="directory to search for *.jsonl")
    ap.add_argument("--json", help="also write the full rows here")
    args = ap.parse_args()

    root = Path(args.results_dir).expanduser()
    rows = collect(root)
    if not rows:
        print(f"no *.jsonl under {root}", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (str(r["variant"]), str(r["task"]), r["step"] or -1, r["seed_offset"] or -1))
    print_table(rows)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if all(status_of(r) == "done" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
