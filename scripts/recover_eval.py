#!/usr/bin/env python3
"""Merge scattered eval JSONL fragments and recompute each run's true summary.

`ResultWriter` opens its JSONL in append mode and never truncates
(`univtac_groot/metrics.py:79`), so episodes can only accumulate -- a run's
lines are never lost, only *split* when a resubmission resolved a different
`--output` path (a relative path run from another cwd, say). This finds every
fragment for each (task, checkpoint, seed-offset), merges them by seed the way
`run_eval.py::rewrite_summary` does (last write per seed wins), and recomputes
the summary from the union.

Reports by default; `--write` is what actually touches your results.

    python3 scripts/recover_eval.py ../eval_result                    # report
    python3 scripts/recover_eval.py ../eval_result eval_result        # both trees
    python3 scripts/recover_eval.py ../eval_result --raw-dir eval_result/raw
    python3 scripts/recover_eval.py ../eval_result --write

Standard library only -- run directly on the login node, any python3.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_STEP_RE = re.compile(r"(?:ckpt|checkpoint[-_])(\d+)")
_SEED_RE = re.compile(r"seed(\d+)")

# Keys summarize() computes; everything else in an existing summary.json is
# run-level context worth carrying forward. Mirrors run_eval.py:390.
_AGGREGATE_KEYS = (
    "episodes_scored", "episodes_errored", "episodes_skipped", "successes",
    "success_rate", "success_rate_pct", "mean_reward", "mean_steps",
    "mean_steps_on_success", "truncated", "early_stopped", "total_inferences",
    "mean_inference_seconds", "success_rate_ci95", "wall_seconds",
    "skip_reasons", "errors",
)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Verbatim from univtac_groot/metrics.py, so numbers match exactly."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def read_rows(path: Path) -> tuple[list[dict], int]:
    """Episode rows from one JSONL, plus a count of unusable lines.

    Lenient on purpose: a job killed mid-write leaves one truncated line, and
    every episode before it is still valid data.
    """
    rows, bad = [], 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict) and isinstance(row.get("seed"), int):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def summarize_rows(rows: list[dict], metadata: dict, episodes_requested: int) -> dict:
    """Recompute summarize()'s output from merged rows (metrics.py:116)."""
    skipped = [r for r in rows if r.get("skipped") is not None]
    errored = [r for r in rows if r.get("error") is not None and r.get("skipped") is None]
    scored = [r for r in rows if r.get("error") is None and r.get("skipped") is None]
    successes = [r for r in scored if r.get("success")]

    n = len(scored)
    rate = len(successes) / n if n else 0.0
    inferences = sum(int(r.get("inferences") or 0) for r in scored)

    summary = {
        **metadata,
        "episodes_requested": episodes_requested,
        "episodes_scored": n,
        "episodes_errored": len(errored),
        "episodes_skipped": len(skipped),
        "successes": len(successes),
        "success_rate": round(rate, 6),
        "success_rate_pct": round(rate * 100.0, 2),
        "mean_reward": round(sum(float(r.get("reward") or 0) for r in scored) / n, 6) if n else 0.0,
        "mean_steps": round(sum(int(r.get("steps") or 0) for r in scored) / n, 2) if n else 0.0,
        "mean_steps_on_success": (
            round(sum(int(r.get("steps") or 0) for r in successes) / len(successes), 2)
            if successes else None
        ),
        "truncated": sum(1 for r in scored if r.get("truncated")),
        "early_stopped": sum(1 for r in scored if r.get("early_stop")),
        "total_inferences": inferences,
        "mean_inference_seconds": (
            round(sum(float(r.get("inference_seconds") or 0) for r in scored) / inferences, 4)
            if inferences else None
        ),
    }
    if n:
        lo, hi = wilson_interval(len(successes), n)
        summary["success_rate_ci95"] = [round(lo, 6), round(hi, 6)]
    if skipped:
        reasons: dict[str, int] = {}
        for r in skipped:
            reasons[str(r.get("skipped"))] = reasons.get(str(r.get("skipped")), 0) + 1
        summary["skip_reasons"] = reasons
    if errored:
        summary["errors"] = [
            {"seed": r.get("seed"), "error": (r.get("error") or "")[:400]} for r in errored
        ][:20]
    return summary


def find_fragments(roots: list[Path]) -> tuple[dict[tuple, list[Path]], list[Path]]:
    """Group every `*.jsonl` by (task, step, seed_offset); return unmatched too."""
    groups: dict[tuple, list[Path]] = {}
    unmatched: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.rglob("*.jsonl")):
            resolved = path.resolve()
            if resolved in seen:      # same file reached through two roots
                continue
            seen.add(resolved)
            step = _STEP_RE.search(path.stem)
            seed = _SEED_RE.search(path.stem)
            task = path.parent.name
            if not step or not seed:
                unmatched.append(path)
                continue
            groups.setdefault((task, int(step.group(1)), int(seed.group(1))), []).append(path)
    for key in groups:
        groups[key].sort(key=lambda p: p.stat().st_mtime)   # oldest first: later writes win
    return groups, unmatched


def video_seeds(raw_dir: Path, task: str) -> set[int]:
    """Seeds with a rendered video under `raw/<variant>/<task>/<stamp>/video/`.

    Each `.mp4` is named by its episode's seed, so these are seeds that
    provably ran -- independent of whether a result line survived. Note
    `clean.yml` sets `video_frequency: 2`, so a missing video does NOT mean a
    missing episode; this is a lower bound.
    """
    seeds: set[int] = set()
    if not raw_dir.exists():
        return seeds
    for mp4 in raw_dir.rglob(f"{task}/*/video/*.mp4"):
        if mp4.stem.isdigit():
            seeds.add(int(mp4.stem))
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="directories to scan for *.jsonl")
    ap.add_argument("--episodes", type=int, default=100, help="scored episodes a run targets")
    ap.add_argument("--raw-dir", help="eval_result/raw, to cross-check against rendered videos")
    ap.add_argument("--write", action="store_true",
                    help="write merged JSONL + regenerated summary.json (default: report only)")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.roots]
    for root in roots:
        if not root.exists():
            print(f"error: {root} does not exist", file=sys.stderr)
            return 1

    groups, unmatched = find_fragments(roots)
    if not groups:
        print(f"no *.jsonl with both ckpt<N> and seed<N> in the name under "
              f"{', '.join(str(r) for r in roots)}", file=sys.stderr)
        return 1

    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else None
    short = []

    for (task, step, seed_offset), fragments in sorted(groups.items()):
        merged: dict[int, dict] = {}
        total_bad = 0
        print(f"\n=== {task}  ckpt{step}  seed-offset {seed_offset} ===")
        for frag in fragments:
            rows, bad = read_rows(frag)
            total_bad += bad
            new = sum(1 for r in rows if r["seed"] not in merged)
            for row in rows:
                merged[row["seed"]] = row          # last write per seed wins
            note = f", {bad} unusable line(s)" if bad else ""
            print(f"  fragment {frag}: {len(rows)} row(s), {new} new seed(s){note}")

        rows = list(merged.values())
        summary = summarize_rows(rows, {"task": task}, args.episodes)
        scored = summary["episodes_scored"]
        print(f"  merged: {len(rows)} unique seed(s) -> {scored} scored, "
              f"{summary['episodes_errored']} errored, {summary['episodes_skipped']} skipped, "
              f"SR {summary['success_rate_pct']}%")

        if len(fragments) > 1:
            print(f"  NOTE: {len(fragments)} fragments merged -- these episodes were split "
                  f"across files and no single file held them all")

        if raw_dir is not None:
            vids = video_seeds(raw_dir, task)
            orphans = sorted(v for v in vids if v not in merged)
            if orphans:
                print(f"  {len(orphans)} seed(s) have a video but NO result row "
                      f"(episode ran, line lost): {orphans[:12]}"
                      f"{' ...' if len(orphans) > 12 else ''}")

        if scored < args.episodes:
            need = args.episodes - scored
            next_seed = (max(merged) + 1) if merged else 1_000_000 * (1 + seed_offset)
            print(f"  SHORT by {need} scored episode(s); a resume would start at seed {next_seed}")
            short.append((task, step, seed_offset, scored, need, next_seed))
        else:
            print("  COMPLETE -- no re-running needed")

        if args.write:
            target = fragments[-1]
            target.write_text(
                "".join(json.dumps(merged[s], ensure_ascii=False) + "\n" for s in sorted(merged)),
                encoding="utf-8")
            summary_path = target.with_suffix(".summary.json")
            existing = {}
            if summary_path.exists():
                try:
                    existing = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = {}
            for key in _AGGREGATE_KEYS:
                existing.pop(key, None)
            existing.setdefault("task", task)
            summary_path.write_text(
                json.dumps(summarize_rows(rows, existing, args.episodes),
                           indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            print(f"  wrote {target} ({len(merged)} rows) and {summary_path}")

    if unmatched:
        print("\nJSONL files skipped (no ckpt<N>/seed<N> in the filename):")
        for path in unmatched:
            print(f"  {path}")

    print(f"\n{len(groups)} run(s) scanned, {len(groups) - len(short)} complete, {len(short)} short.")
    for task, step, seed_offset, scored, need, next_seed in short:
        print(f"  {task} ckpt{step} seed{seed_offset}: {scored}/{args.episodes} "
              f"(need {need}, resume at seed {next_seed})")
    if not args.write:
        print("\nreport only -- pass --write to merge fragments and regenerate summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
