"""Aggregate the ablation: baseline vs tactile, per task and overall.

Reads the JSONL files ``scripts/run_eval.py`` writes and prints a table plus a
machine-readable JSON blob. Safe to run on a login node: it only reads scalar
per-episode records, never frames, and streams each file line by line.

Works on partial results, so it is useful while jobs are still running or after
one hit its walltime.

Example::

    python scripts/compare_ablation.py --results-dir eval_result \
        --baseline-arm baseline --tactile-arm tactile --json summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from univtac_groot.metrics import compare, read_jsonl, summarize  # noqa: E402


def collect(results_dir: Path, arm: str) -> dict[str, dict]:
    """Summarise every task under ``<results_dir>/<arm>/<task>/*.jsonl``.

    Multiple JSONL files for one task (e.g. seed-sharded array jobs) are pooled
    into a single summary.
    """
    arm_dir = results_dir / arm
    if not arm_dir.is_dir():
        return {}

    per_task: dict[str, dict] = {}
    for task_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
        records = []
        for jsonl in sorted(task_dir.glob("*.jsonl")):
            records.extend(read_jsonl(jsonl))
        if records:
            # Deduplicate by seed: re-runs of the same shard would otherwise
            # double-count. Last write wins.
            deduped = {r.seed: r for r in records}
            per_task[task_dir.name] = summarize(
                deduped.values(), metadata={"arm": arm, "task": task_dir.name}
            )
    return per_task


def pooled(per_task: dict[str, dict]) -> dict:
    """Pool per-task summaries into one arm-level summary.

    Tasks are pooled by raw episode counts, so a task evaluated with more
    episodes carries more weight. The macro (per-task unweighted) mean is
    reported alongside, since the UniVTAC paper reports average success rate
    across its task suite.
    """
    scored = sum(s["episodes_scored"] for s in per_task.values())
    successes = sum(s["successes"] for s in per_task.values())
    rate = successes / scored if scored else 0.0
    macro = (
        sum(s["success_rate"] for s in per_task.values()) / len(per_task)
        if per_task
        else 0.0
    )
    from univtac_groot.metrics import wilson_interval

    lo, hi = wilson_interval(successes, scored)
    return {
        "tasks": len(per_task),
        "episodes_scored": scored,
        "successes": successes,
        "success_rate": round(rate, 6),
        "success_rate_pct": round(rate * 100, 2),
        "success_rate_ci95": [round(lo, 6), round(hi, 6)],
        "macro_success_rate": round(macro, 6),
        "macro_success_rate_pct": round(macro * 100, 2),
        "episodes_errored": sum(s["episodes_errored"] for s in per_task.values()),
    }


def format_table(baseline: dict[str, dict], tactile: dict[str, dict]) -> str:
    """Render the per-task comparison as fixed-width text."""
    tasks = sorted(set(baseline) | set(tactile))
    width = max([len("task")] + [len(t) for t in tasks]) if tasks else 4

    lines = [
        f"{'task':<{width}}  {'baseline':>18}  {'tactile':>18}  {'delta':>8}",
        f"{'-' * width}  {'-' * 18}  {'-' * 18}  {'-' * 8}",
    ]
    for task in tasks:
        b, t = baseline.get(task), tactile.get(task)
        b_cell = f"{b['successes']}/{b['episodes_scored']} ({b['success_rate_pct']:.1f}%)" if b else "-"
        t_cell = f"{t['successes']}/{t['episodes_scored']} ({t['success_rate_pct']:.1f}%)" if t else "-"
        delta = (
            f"{(t['success_rate'] - b['success_rate']) * 100:+.1f}" if b and t else "-"
        )
        lines.append(f"{task:<{width}}  {b_cell:>18}  {t_cell:>18}  {delta:>8}")

    if baseline or tactile:
        pb, pt = pooled(baseline), pooled(tactile)
        lines.append(f"{'-' * width}  {'-' * 18}  {'-' * 18}  {'-' * 8}")
        b_cell = f"{pb['successes']}/{pb['episodes_scored']} ({pb['success_rate_pct']:.1f}%)"
        t_cell = f"{pt['successes']}/{pt['episodes_scored']} ({pt['success_rate_pct']:.1f}%)"
        delta = f"{(pt['success_rate'] - pb['success_rate']) * 100:+.1f}"
        lines.append(f"{'POOLED':<{width}}  {b_cell:>18}  {t_cell:>18}  {delta:>8}")
        lines.append(
            f"{'MACRO':<{width}}  {pb['macro_success_rate_pct']:>17.1f}%  "
            f"{pt['macro_success_rate_pct']:>17.1f}%  "
            f"{(pt['macro_success_rate'] - pb['macro_success_rate']) * 100:>+8.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the baseline and tactile ablation arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="eval_result")
    parser.add_argument("--baseline-arm", default="baseline")
    parser.add_argument("--tactile-arm", default="tactile")
    parser.add_argument("--json", default=None, help="also write the summary here")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"no results directory: {results_dir}")

    baseline = collect(results_dir, args.baseline_arm)
    tactile = collect(results_dir, args.tactile_arm)
    if not baseline and not tactile:
        raise SystemExit(
            f"no JSONL results under {results_dir}/{{{args.baseline_arm},{args.tactile_arm}}}/<task>/"
        )

    print(format_table(baseline, tactile))

    blob = {
        "baseline_arm": args.baseline_arm,
        "tactile_arm": args.tactile_arm,
        "per_task": {
            task: compare(baseline.get(task, {}), tactile.get(task, {}))
            for task in sorted(set(baseline) | set(tactile))
        },
        "pooled": compare(pooled(baseline), pooled(tactile)),
        "baseline_summary": pooled(baseline),
        "tactile_summary": pooled(tactile),
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
