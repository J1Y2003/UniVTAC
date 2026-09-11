"""Headless evaluation driver: one variant, one UniVTAC task, N episodes.

This is the standalone path — it constructs the Isaac Lab app itself and drives
:class:`~univtac_groot.env_wrapper.UniVTACGr00tEnv` through
:func:`univtac_groot.rollout.evaluate`. Use it when you want this repo's result
files (JSONL + summary JSON) and per-variant SLURM jobs. If you would rather stay
inside UniVTAC's own harness, use ``policy/GR00T`` with ``eval_policy.sh``
instead; both share the same adapters.

Must run inside UniVTAC's conda environment, with the GR00T inference server
already listening (``univtac_groot.server.run_server`` in the GR00T environment).

Example::

    python scripts/run_eval.py \
        --task insert_hole --task-config demo --variant baseline \
        --host 127.0.0.1 --port 5555 \
        --episodes 100 --execution-horizon 16 \
        --output eval_result/baseline/insert_hole.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import yaml
import argparse as _argparse
import importlib
from typing import NamedTuple

from univtac_groot.env_wrapper import UniVTACGr00tEnv
from univtac_groot.variants import build_spec
from univtac_groot.action_adapter import ActionAdapter, GripperConvention
from univtac_groot.client import Gr00tClient
from univtac_groot.metrics import ResultWriter
from univtac_groot.rollout import (
    BatchingPolicy,
    RolloutConfig,
    evaluate,
    resolve_spec_from_policy,
)

from isaaclab.app import AppLauncher


# Repo root on sys.path so ``univtac_groot`` imports when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate GR00T N1.7 on one UniVTAC task, headlessly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- what to run ------------------------------------------------------
    parser.add_argument("--task", required=True, help="UniVTAC task name, e.g. insert_hole")
    parser.add_argument(
        "--task-config",
        default="clean",
        help="stem of a file in UniVTAC/task_config (demo, contact, clean)",
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "baseline_finetuned"],
        help="'baseline_finetuned' is the reported model; 'baseline' is zero-shot",
    )
    parser.add_argument(
        "--univtac-root",
        default=os.environ.get("UNIVTAC_ROOT", "."),
        help="path to the UniVTAC checkout (must contain envs/ and task_config/)",
    )

    # -- server -----------------------------------------------------------
    parser.add_argument("--host", default=os.environ.get("GROOT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GROOT_PORT", 5555)))
    parser.add_argument("--api-token", default=os.environ.get("GROOT_API_TOKEN"))
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for the server to finish loading the checkpoint",
    )
    parser.add_argument("--request-timeout-ms", type=int, default=120_000)

    # -- control ----------------------------------------------------------
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=16,
        help="actions executed per chunk before re-planning. Must be identical "
             "across runs you compare: it sets how closed-loop the controller is",
    )
    parser.add_argument(
        "--action-type", default="qpos", choices=["qpos", "ee", "delta_ee"]
    )
    parser.add_argument("--gripper-max-qpos", type=float, default=0.039)
    parser.add_argument(
        "--gripper-invert",
        default=None,
        type=lambda v: str(v).lower() in ("1", "true", "yes"),
        help="policy gripper scalar means closed at 1.0; default: true for --variant baseline",
    )

    # -- episodes ---------------------------------------------------------
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--start-seed",
        type=int,
        default=None,
        help="first seed; default 1_000_000 * (1 + --seed-offset), as in UniVTAC",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--max-seed", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="per-episode action cap; default = the task's cfg.step_lim",
    )
    parser.add_argument("--max-consecutive-errors", type=int, default=5)

    # -- output -----------------------------------------------------------
    parser.add_argument(
        "--output",
        default=None,
        help="results JSONL path; default eval_result/<variant>/<task>/<timestamp>.jsonl",
    )
    parser.add_argument(
        "--device", default=None, help="Isaac Lab sim device, e.g. cuda:0"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="ignore any existing --output file and evaluate the full block again "
             "(results are appended, so the tally will double-count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the spec and talk to the server, but do not start Isaac Sim",
    )
    return parser


def resolve_output(args: argparse.Namespace) -> Path:
    """Where to write results.

    Defaults under ``$REPO_ROOT/eval_result`` rather than the working directory:
    the sbatch script runs this with cwd set to ``$UNIVTAC_ROOT``, so a
    cwd-relative default would scatter results into the UniVTAC checkout and
    hide them from ``scripts/results_table.py``.
    """
    if args.output:
        return Path(args.output).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return _REPO_ROOT / "eval_result" / args.variant / args.task / f"{stamp}.jsonl"


class ResumeState(NamedTuple):
    """What an existing results file says is still left to do."""

    start_seed: int      # first seed not yet attempted
    already_scored: int  # episodes that counted towards the budget


def read_resume_state(output: Path, base_seed: int) -> ResumeState:
    """Inspect an existing JSONL and work out where to continue.

    `background` preempts with `PreemptMode=REQUEUE`, so a preempted job re-runs
    this script with the same arguments and `ResultWriter` appends. Without this,
    every preemption would add a fresh pass from the first seed and inflate the
    tally.

    Errored and skipped episodes consumed a seed but not the episode budget, so
    the two are counted separately: a resumed run targets the same number of
    *scored* episodes a clean run would.

    Raises:
        ValueError: the file exists but holds no readable episode. Refusing to
            guess is deliberate -- appending a fresh pass would corrupt the tally.
    """
    max_seed, scored, seen = base_seed - 1, 0, set()
    with open(output, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # a kill mid-write leaves one truncated line
            seed = row.get("seed")
            if isinstance(seed, int):
                max_seed = max(max_seed, seed)
                if seed in seen:
                    continue      # an earlier pass already counted this seed
                seen.add(seed)
            if not row.get("error") and not row.get("skipped"):
                scored += 1
    if not seen:
        raise ValueError("no readable episodes in the file")
    return ResumeState(start_seed=max_seed + 1, already_scored=scored)


def apply_resume(args: argparse.Namespace, output: Path) -> bool:
    """Adjust `args.start_seed` and `args.episodes` to continue an interrupted run.

    Returns True when there is nothing left to do. Mutates `args` otherwise, so
    everything downstream sees the reduced workload. An explicit `--start-seed`
    from the caller wins.
    """
    if not args.resume or not output.exists() or output.stat().st_size == 0:
        return False

    base_seed = 1_000_000 * (1 + args.seed_offset)
    try:
        state = read_resume_state(output, base_seed)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"error: {output} exists but its resume state is unreadable ({exc}).\n"
            f"       Move it aside to start over, or pass --no-resume."
        ) from exc

    if state.already_scored >= args.episodes:
        print(f"[eval] {output} already holds {state.already_scored}/{args.episodes} "
              f"scored episodes -- nothing left to run.")
        return True

    remaining = args.episodes - state.already_scored
    if args.start_seed is None:
        args.start_seed = state.start_seed
    args.episodes = remaining
    print(f"[eval] RESUMING: {state.already_scored} already scored; running "
          f"{remaining} more from seed {args.start_seed}.")
    return False


def build_arm_spec(args: argparse.Namespace):
    """Build the observation spec for the requested variant."""

    return build_spec(args.variant, task=args.task)


def load_instructions(univtac_root: Path, task: str, kind: str = "seen") -> list[str] | None:
    """Read ``UniVTAC/instructions/<task>.json``.

    Returning ``None`` lets ``BaseTask.reset`` keep whatever instruction the
    task sets itself.
    """
    path = univtac_root / "instructions" / f"{task}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get(kind) or data.get("seen") or []
    return [str(v) for v in values] or None


def load_task_config(univtac_root: Path, name: str) -> dict:
    """Read a ``UniVTAC/task_config/<name>.yml`` file."""

    path = Path(name)
    if path.suffix not in (".yml", ".yaml"):
        path = univtac_root / "task_config" / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(
            f"task config not found: {path}. Pass --task-config with a stem that "
            f"exists in {univtac_root / 'task_config'} (demo, contact, clean)."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    univtac_root = Path(args.univtac_root).resolve()

    task_config = load_task_config(univtac_root, args.task_config)

    spec = build_arm_spec(args)
    output = resolve_output(args)
    print(f"[eval] variant={args.variant} task={args.task} -> {output}")

    # Before anything expensive: a requeued job may have nothing left to do.
    episodes_target = args.episodes
    if apply_resume(args, output):
        print(json.dumps(rewrite_summary(output, episodes_target), indent=2))
        return 0

    # -- connect to the model ------------------------------------------------
    client = Gr00tClient(
        host=args.host,
        port=args.port,
        timeout_ms=args.request_timeout_ms,
        api_token=args.api_token,
    )
    print(f"[eval] waiting for the GR00T server at {args.host}:{args.port} ...")
    client.wait_until_ready(timeout_s=args.startup_timeout)
    spec, horizons = resolve_spec_from_policy(
        client, spec, execution_horizon=args.execution_horizon
    )

    gripper_invert = (
        args.gripper_invert if args.gripper_invert is not None else args.variant == "baseline"
    )
    action_adapter = ActionAdapter(
        action_type=args.action_type,  # type: ignore[arg-type]
        gripper=GripperConvention(
            max_qpos=args.gripper_max_qpos, invert=gripper_invert
        ),
    )

    metadata = {
        "variant": args.variant,
        "task": args.task,
        "task_config": args.task_config,
        "state_dim": spec.state_dim,
        "state_keys": list(spec.state_keys),
        "video_keys": sorted(spec.all_video_keys),
        "action_type": args.action_type,
        "gripper_invert": gripper_invert,
        "episodes_requested": episodes_target,
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in horizons.items()},
    }

    if args.dry_run:
        print("[eval] dry run; observation contract:")
        print(json.dumps(metadata, indent=2))
        client.close()
        return 0

    # -- Isaac Lab must be launched before importing UniVTAC's envs ----------
    env = build_env(args, univtac_root, spec, action_adapter, task_config)

    writer = ResultWriter(output, metadata)
    config = RolloutConfig(
        num_episodes=args.episodes,
        start_seed=args.start_seed,
        seed_offset=args.seed_offset,
        max_seed=args.max_seed,
        execution_horizon=horizons["execution_horizon"],
        max_steps=args.max_steps,
        max_consecutive_errors=args.max_consecutive_errors,
    )

    try:
        summary = evaluate(
            env,
            BatchingPolicy(client, spec),
            config=config,
            writer=writer,
            action_adapter=action_adapter,
        )
    finally:
        writer.close()
        env.close()
        client.close()
        close_simulation_app()

    summary = rewrite_summary(output, episodes_target)
    print(json.dumps(summary, indent=2))
    return 0


def rewrite_summary(output: Path, episodes_target: int) -> dict:
    """Re-aggregate `<output>.summary.json` from the whole JSONL.

    `ResultWriter.close()` only sees the episodes *this process* added, so after
    a resume its summary would cover the final leg alone. Recomputing from the
    file, deduplicated by seed, is the only way the number reflects every pass.
    """
    from univtac_groot.metrics import read_jsonl, summarize

    rows = {r.seed: r for r in read_jsonl(output)}   # last write wins
    path = output.with_suffix(".summary.json")
    metadata: dict = {}
    if path.exists():
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    # Drop the aggregates; summarize() recomputes them. Everything else in the
    # file is run-level context worth keeping.
    for key in ("episodes_scored", "episodes_errored", "episodes_skipped",
                "successes", "success_rate", "success_rate_pct", "mean_reward",
                "mean_steps", "mean_steps_on_success", "truncated",
                "success_rate_ci95", "wall_seconds", "skip_reasons", "errors"):
        metadata.pop(key, None)
    metadata["episodes_requested"] = episodes_target

    summary = summarize(rows.values(), metadata=metadata)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return summary


def build_env(args, univtac_root: Path, spec, action_adapter, task_config: dict):  # noqa: C901
    """Launch Isaac Lab headlessly and construct the wrapped UniVTAC task.

    Mirrors the bootstrap order in ``UniVTAC/scripts/eval_policy.py``:
    ``AppLauncher`` must run before anything imports ``envs.*``, because those
    modules import ``isaaclab``/``omni`` at module scope.
    """

    sys.path.insert(0, str(univtac_root))


    launcher_parser = _argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    app_args = launcher_parser.parse_args([])

    app_args.livestream = 2           # essential, no idea why though.
    app_args.enable_cameras = True    # offscreen RGB rendering
    app_args.num_envs = 1
    if args.device:
        app_args.device = args.device

    global _SIMULATION_APP
    _SIMULATION_APP = AppLauncher(app_args).app


    task_module = importlib.import_module(f"envs.{args.task}")

    env_cfg = task_module.TaskCfg()
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    env_cfg.save_dir = _REPO_ROOT / "eval_result" / "raw" / args.variant / args.task / stamp
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    # Video writing is the main I/O cost per episode; off by default here.
    env_cfg.video_frequency = task_config.get("video_frequency", 0)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    if args.device:
        env_cfg.sim.device = args.device

    task = task_module.Task(env_cfg, mode="eval")


    env = UniVTACGr00tEnv(
        task,
        spec,
        action_adapter=action_adapter,
        instructions=load_instructions(univtac_root, args.task),
        max_steps=args.max_steps,
        gripper_max_qpos=args.gripper_max_qpos,
    )
    return env


_SIMULATION_APP = None


def close_simulation_app() -> None:
    """Close Isaac Sim if this process opened it."""
    global _SIMULATION_APP
    if _SIMULATION_APP is not None:
        try:
            _SIMULATION_APP.close()
        except Exception:
            pass
        _SIMULATION_APP = None


if __name__ == "__main__":
    raise SystemExit(main())
