"""Aggregate the ablation: baseline vs tactile, per task and overall.

Reads the JSONL files ``scripts/run_eval.py`` writes and prints a table plus a
machine-readable JSON blob. Safe to run on a login node: it only reads scalar
per-episode records, never frames, and streams each file line by line.

Works on partial results, so it is useful while jobs are still running or after
one hit its walltime.

Example::

    python scripts/compare_ablation.py --results-dir eval_result \
        --baseline-variant baseline --tactile-variant tactile --json summary.json
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

DEFAULT_RESULTS_DIR = _REPO_ROOT / "eval_result"
"""Where ``slurm/eval_ablation.sbatch`` writes: ``$REPO_ROOT/eval_result``.

Anchored to the repo rather than the working directory so this works from
anywhere -- notably from inside ``scripts/``, and from ``$UNIVTAC_ROOT``, which
is where the sbatch script leaves the evaluator's cwd.
"""


def collect(results_dir: Path, variant: str) -> dict[str, dict]:
    """Summarise every task under ``<results_dir>/<variant>/<task>/*.jsonl``.

    Multiple JSONL files for one task (e.g. seed-sharded array jobs) are pooled
    into a single summary.
    """
    arm_dir = results_dir / variant
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
                deduped.values(), metadata={"variant": variant, "task": task_dir.name}
            )
    return per_task


def pooled(per_task: dict[str, dict]) -> dict:
    """Pool per-task summaries into one variant-level summary.

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


def no_results_message(results_dir: Path, args, *, exists: bool) -> str:
    """Explain an empty result set and name the next action.

    Reaching here almost always means no evaluation has been run yet, so say
    that rather than reporting a missing path as if it were a misconfiguration.
    """
    lines = []
    if exists:
        found = sorted(p.name for p in results_dir.iterdir() if p.is_dir())
        lines.append(
            f"No episode results for variants {args.baseline_variant!r} / {args.tactile_variant!r} "
            f"under {results_dir}"
        )
        if found:
            lines.append(f"  variants present: {found}  (use --baseline-variant / --tactile-variant)")
        else:
            lines.append("  the directory is empty")
    else:
        lines.append(f"No results directory: {results_dir}")
        lines.append("  (this is where slurm/eval_ablation.sbatch writes its JSONL files)")

    lines += [
        "",
        "Nothing to compare yet -- run at least one evaluation first:",
        "",
        "  VARIANTS=baseline_finetuned bash slurm/submit_benchmark.sh --evals",
        "",
        "That needs UNIVTAC_ROOT, UNIVTAC_PYTHON and GROOT_PYTHON exported; see",
        "docs/SETUP.md. Evaluation is GPU work, so it must go to a compute node.",
        "",
        "If results live elsewhere, point at them:",
        "  python scripts/compare_ablation.py --results-dir /path/to/eval_result",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the baseline and tactile ablation variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="directory holding <variant>/<task>/*.jsonl result files",
    )
    parser.add_argument("--baseline-variant", default="baseline")
    parser.add_argument("--tactile-variant", default="tactile")
    parser.add_argument("--json", default=None, help="also write the summary here")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_dir():
        raise SystemExit(no_results_message(results_dir, args, exists=False))

    baseline = collect(results_dir, args.baseline_variant)
    tactile = collect(results_dir, args.tactile_variant)
    if not baseline and not tactile:
        raise SystemExit(no_results_message(results_dir, args, exists=True))

    print(format_table(baseline, tactile))

    blob = {
        "baseline_variant": args.baseline_variant,
        "tactile_variant": args.tactile_variant,
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
