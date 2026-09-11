#!/usr/bin/env python3
"""Summarise a wandb run from the command line, including system metrics.

Written to read back the 170 s/step run (SLURM job 164704) without opening a
browser, but it is general: pass any substring of a run's name or id.

The interesting series here is NOT the loss -- 140 optimisation steps says
nothing about policy quality. It is the system stream. The cuDNN diagnosis in
docs/STATUS.md rests on a GPU drawing 121 W of 400 W at 48% utilisation, which
was read from nvidia-smi snapshots after the fact; wandb recorded the same
counters continuously while the job ran, so it is an independent check on the
claim.

Usage:
    python scripts/wandb_report.py 164704
    python scripts/wandb_report.py 164704 --project univtac-groot
    python scripts/wandb_report.py 164704 --csv /tmp/run164704

Needs WANDB_API_KEY in the environment (never `wandb login` on the shared
account -- it writes a token into the account owner's ~/.netrc).
"""

from __future__ import annotations

import argparse
import os
import sys

# Keys differ between wandb versions and between the trainer's logging and
# wandb's system monitor, so each entry lists the aliases seen in practice and
# the first present wins.
TRAIN_SERIES = [
    ("loss", ["train/loss", "loss", "train_loss"]),
    ("learning rate", ["train/learning_rate", "learning_rate", "lr"]),
    ("grad norm", ["train/grad_norm", "grad_norm"]),
    ("epoch", ["train/epoch", "epoch"]),
    ("s/step", ["train/train_steps_per_second", "train/step_time", "step_time"]),
]

SYSTEM_SERIES = [
    ("gpu power (W)", ["system.gpu.0.powerWatts", "system/gpu.0.powerWatts"]),
    ("gpu power (% cap)", ["system.gpu.0.powerPercent", "system/gpu.0.powerPercent"]),
    ("gpu util (%)", ["system.gpu.0.gpu", "system/gpu.0.gpu"]),
    ("gpu mem alloc (%)", ["system.gpu.0.memoryAllocated", "system/gpu.0.memoryAllocated"]),
    ("gpu temp (C)", ["system.gpu.0.temp", "system/gpu.0.temp"]),
    ("cpu (% of one core)", ["system.cpu", "system/cpu"]),
    ("proc cpu threads", ["system.proc.cpu.threads", "system/proc.cpu.threads"]),
    ("disk read (MB)", ["system.disk.in", "system/disk.in"]),
    ("network recv (MB)", ["system.network.recv", "system/network.recv"]),
]


def stats(values: list[float]) -> dict[str, float] | None:
    vals = [v for v in values if isinstance(v, (int, float)) and v == v]
    if not vals:
        return None
    ordered = sorted(vals)
    n = len(ordered)
    return {
        "n": n,
        "first": vals[0],
        "last": vals[-1],
        "min": ordered[0],
        "median": ordered[n // 2],
        "max": ordered[-1],
        "mean": sum(vals) / n,
    }


def report(title: str, frame, series) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if frame is None or not len(frame):
        print("  (no rows -- the run may have been offline, or logged nothing here)")
        return
    columns = set(frame.columns)
    printed = False
    for label, aliases in series:
        key = next((a for a in aliases if a in columns), None)
        if key is None:
            continue
        s = stats(frame[key].tolist())
        if s is None:
            continue
        printed = True
        print(
            f"  {label:<22} n={s['n']:<5} first={s['first']:<12.4g} "
            f"last={s['last']:<12.4g} min={s['min']:<12.4g} "
            f"median={s['median']:<12.4g} max={s['max']:<12.4g}"
        )
    if not printed:
        print("  (none of the expected keys are present)")
        print(f"  available: {sorted(columns)[:40]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match", help="substring of the run name or id, e.g. 164704")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT",
                                                        "univtac-groot"))
    ap.add_argument("--csv", metavar="PREFIX",
                    help="also write PREFIX-train.csv and PREFIX-system.csv")
    args = ap.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        print("error: WANDB_API_KEY is not set. Export it -- do NOT run "
              "`wandb login`\n       on the shared account, it writes into the "
              "owner's ~/.netrc.", file=sys.stderr)
        return 2

    try:
        import wandb
    except ImportError:
        print("error: wandb is not importable. pip install wandb", file=sys.stderr)
        return 2

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    try:
        runs = list(api.runs(path))
    except Exception as exc:                                  # noqa: BLE001
        print(f"error: could not list runs in '{path}': {exc}", file=sys.stderr)
        print("       Pass --entity <your-wandb-entity> if the project is not "
              "under your default entity.", file=sys.stderr)
        return 2

    hits = [r for r in runs if args.match in r.name or args.match in r.id]
    if not hits:
        print(f"No run in '{path}' matches '{args.match}'. Runs present:",
              file=sys.stderr)
        for r in runs[:40]:
            print(f"  {r.id}  {r.name}  state={r.state}", file=sys.stderr)
        return 1

    for run in hits:
        print("=" * 72)
        print(f"{run.name}   id={run.id}   state={run.state}")
        print(f"  url      {run.url}")
        created = getattr(run, "created_at", "?")
        runtime = (run.summary.get("_runtime") if run.summary else None)
        print(f"  created  {created}")
        if runtime:
            print(f"  runtime  {runtime:.0f} s ({runtime / 3600:.2f} h)")
        for k in ("num_gpus", "max_steps", "batch_size", "learning_rate",
                  "weight_decay", "task", "variant", "action_horizon"):
            if k in run.config:
                print(f"  cfg      {k}={run.config[k]}")

        # history() paginates and returns a DataFrame; samples=None asks for
        # every row rather than wandb's default ~500-point downsample, which
        # matters at 140 steps but also for the 1 Hz system stream.
        train = run.history(samples=100000, pandas=True)
        report("training stream", train, TRAIN_SERIES)

        try:
            system = run.history(stream="system", samples=100000, pandas=True)
        except Exception as exc:                              # noqa: BLE001
            system = None
            print(f"\n(system stream unavailable: {exc})")
        report("system stream  <- the cuDNN signature lives here", system,
               SYSTEM_SERIES)

        if args.csv:
            if train is not None and len(train):
                train.to_csv(f"{args.csv}-train.csv", index=False)
                print(f"\nwrote {args.csv}-train.csv")
            if system is not None and len(system):
                system.to_csv(f"{args.csv}-system.csv", index=False)
                print(f"wrote {args.csv}-system.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
