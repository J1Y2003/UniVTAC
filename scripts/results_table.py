#!/usr/bin/env python3
"""Success rate per run, read from run_eval.py's own `*.summary.json` files.

    python scripts/results_table.py eval_result
    python scripts/results_table.py eval_result --json table.json

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="directory to search for *.summary.json")
    ap.add_argument("--json", help="also write the rows here")
    args = ap.parse_args()

    root = Path(args.results_dir).expanduser()
    rows = []
    for path in sorted(root.rglob("*.summary.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        ci = d.get("success_rate_ci95") or [None, None]
        rows.append({
            "run": str(path.relative_to(root)).removesuffix(".summary.json"),
            "task": d.get("task"),
            "variant": d.get("variant"),
            "scored": d.get("episodes_scored"),
            "successes": d.get("successes"),
            "sr_pct": d.get("success_rate_pct"),
            "ci95_pct": [round(c * 100, 1) for c in ci] if ci[0] is not None else None,
            "errored": d.get("episodes_errored"),
            "skipped": d.get("episodes_skipped"),
            "mean_steps": d.get("mean_steps"),
        })

    if not rows:
        print(f"no *.summary.json under {root}", file=sys.stderr)
        return 1

    w = max(len(r["run"]) for r in rows)
    print(f"{'run':<{w}} {'task':<14} {'scored':>6} {'SR%':>6} {'CI95%':>14} {'err':>4} {'skip':>4}")
    for r in rows:
        ci = f"[{r['ci95_pct'][0]}, {r['ci95_pct'][1]}]" if r["ci95_pct"] else "-"
        sr = f"{r['sr_pct']:.1f}" if isinstance(r["sr_pct"], (int, float)) else "-"
        print(f"{r['run']:<{w}} {str(r['task']):<14} {str(r['scored']):>6} {sr:>6} "
              f"{ci:>14} {str(r['errored']):>4} {str(r['skipped']):>4}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
