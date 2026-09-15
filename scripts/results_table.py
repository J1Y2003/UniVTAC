#!/usr/bin/env python3
"""Success rate per run, read from run_eval.py's own `*.summary.json` files.

    python scripts/results_table.py eval_result
    python scripts/results_table.py eval_result --pivot
    python scripts/results_table.py eval_result --json table.json

`--pivot` reshapes the same rows into tasks x checkpoint steps, which is the
view the step-count decision is made on (docs/BENCHMARK.md, "Choosing the step
count"). The step comes from `ckpt<N>` or `checkpoint-<N>` in the run's path,
the task from the summary's own `task` field.

Standard library only. The graph lives in `scripts/results_plot.py` because it
needs matplotlib.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# `seed1-ckpt20000.summary.json`, `insert_hole-ckpt20000-seed1.json`, and a
# path that names the checkpoint directory directly all resolve the same way.
_STEP_RE = re.compile(r"(?:ckpt|checkpoint[-_])(\d+)")


def parse_step(run: str) -> int | None:
    """Checkpoint step from a run path, or None when it carries no `ckpt<N>`."""
    match = _STEP_RE.search(run)
    return int(match.group(1)) if match else None


def collect(root: Path) -> list[dict]:
    """Read every `*.summary.json` under `root` into one row each."""
    rows: list[dict] = []
    for path in sorted(root.rglob("*.summary.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        ci = d.get("success_rate_ci95") or [None, None]
        run = str(path.relative_to(root)).removesuffix(".summary.json")
        rows.append({
            "run": run,
            "task": d.get("task"),
            "variant": d.get("variant"),
            "step": parse_step(run),
            "scored": d.get("episodes_scored"),
            "requested": d.get("episodes_requested"),
            "successes": d.get("successes"),
            "sr_pct": d.get("success_rate_pct"),
            "ci95_pct": [round(c * 100, 1) for c in ci] if ci[0] is not None else None,
            "errored": d.get("episodes_errored"),
            "skipped": d.get("episodes_skipped"),
            "mean_steps": d.get("mean_steps"),
        })
    return rows


def is_partial(row: dict) -> bool:
    """Fewer episodes scored than the run asked for -- it was cut short."""
    scored, requested = row.get("scored"), row.get("requested")
    return isinstance(scored, int) and isinstance(requested, int) and scored < requested


def pivot_cells(rows: list[dict]) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], list[dict]]]:
    """Index rows by ``(task, step)``.

    Two summaries can land on one cell -- pointing this at a parent directory
    that holds both a 30k and a 50k run gives two `insert_hole` / `ckpt10000`
    rows. The one with more scored episodes wins, deterministically, and every
    collision is returned so the caller can say so rather than silently
    picking.
    """
    buckets: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if row["task"] and row["step"] is not None:
            buckets.setdefault((row["task"], row["step"]), []).append(row)

    cells: dict[tuple[str, int], dict] = {}
    collisions: dict[tuple[str, int], list[dict]] = {}
    for key, candidates in buckets.items():
        ordered = sorted(candidates, key=lambda r: (-(r["scored"] or 0), r["run"]))
        cells[key] = ordered[0]
        if len(ordered) > 1:
            collisions[key] = ordered
    return cells, collisions


def print_flat(rows: list[dict]) -> None:
    w = max(len(r["run"]) for r in rows)
    print(f"{'run':<{w}} {'task':<14} {'scored':>6} {'SR%':>6} {'CI95%':>14} {'err':>4} {'skip':>4}")
    for r in rows:
        ci = f"[{r['ci95_pct'][0]}, {r['ci95_pct'][1]}]" if r["ci95_pct"] else "-"
        sr = f"{r['sr_pct']:.1f}" if isinstance(r["sr_pct"], (int, float)) else "-"
        print(f"{r['run']:<{w}} {str(r['task']):<14} {str(r['scored']):>6} {sr:>6} "
              f"{ci:>14} {str(r['errored']):>4} {str(r['skipped']):>4}")


def print_pivot(rows: list[dict]) -> int:
    cells, collisions = pivot_cells(rows)
    if not cells:
        print("no *.summary.json carries both a task and a ckpt<N> step; "
              "use the default flat table instead", file=sys.stderr)
        return 1

    tasks = sorted({task for task, _ in cells})
    steps = sorted({step for _, step in cells})
    label_w = max([len(t) for t in tasks] + [len("n tasks")])
    col_w = max(8, max(len(str(s)) for s in steps) + 2)

    def cell(key: tuple[str, int]) -> str:
        row = cells.get(key)
        if row is None or not isinstance(row["sr_pct"], (int, float)):
            return "-"
        text = f"{row['sr_pct']:.1f}"
        if is_partial(row):
            text += "~"
        if key in collisions:
            text += "*"
        return text

    print(f"{'task':<{label_w}}" + "".join(f"{s:>{col_w}}" for s in steps))
    for task in tasks:
        print(f"{task:<{label_w}}" + "".join(f"{cell((task, s)):>{col_w}}" for s in steps))

    # The mean is over steps where *every* task reported, so the line cannot
    # move just because the task set changed between columns.
    means, counts = [], []
    for step in steps:
        values = [cells[(t, step)]["sr_pct"] for t in tasks
                  if (t, step) in cells and isinstance(cells[(t, step)]["sr_pct"], (int, float))]
        counts.append(len(values))
        means.append(sum(values) / len(values) if len(values) == len(tasks) else None)

    print(f"{'-' * label_w}" + "".join(f"{'-' * (col_w - 1):>{col_w}}" for _ in steps))
    print(f"{'mean':<{label_w}}" + "".join(
        f"{(f'{m:.1f}' if m is not None else '-'):>{col_w}}" for m in means))
    print(f"{'n tasks':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in counts))

    notes = []
    if any(is_partial(r) for r in cells.values()):
        notes.append("~ = fewer episodes scored than requested (run was cut short)")
    if collisions:
        notes.append("* = more than one summary for this cell; showed the one with most scored")
    if any(m is None for m in means):
        notes.append("mean shown only where every task reported")
    if notes:
        print()
    for note in notes:
        print(note)
    for (task, step), candidates in sorted(collisions.items()):
        print(f"    {task} ckpt{step}: " + ", ".join(
            f"{c['run']} ({c['scored']} scored)" for c in candidates))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="directory to search for *.summary.json")
    ap.add_argument("--pivot", action="store_true",
                    help="tasks x checkpoint steps, SR%% in the cells")
    ap.add_argument("--json", help="also write the flat rows here")
    args = ap.parse_args()

    root = Path(args.results_dir).expanduser()
    rows = collect(root)
    if not rows:
        print(f"no *.summary.json under {root}", file=sys.stderr)
        return 1

    status = print_pivot(rows) if args.pivot else (print_flat(rows) or 0)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
