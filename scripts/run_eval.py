"""Headless evaluation driver: one variant, one UniVTAC task, N episodes.

This is the standalone path — it constructs the Isaac Lab app itself and drives
:class:`~univtac_groot.env_wrapper.UniVTACGr00tEnv` through
:func:`univtac_groot.rollout.evaluate`. Use it when you want this repo's result
files (JSONL + summary JSON) and per-variant SLURM jobs. If you would rather stay
inside UniVTAC's own harness, use ``policy/GR00T`` with ``eval_policy.sh``
instead; both share the same adapters.

Never interactive: no prompts, no GUI, no ``input()``. Rendering is offscreen
(``--headless``) and every parameter comes from flags or the environment, so it
runs unchanged under ``sbatch``.

Must run inside UniVTAC's conda environment, with the GR00T inference server
already listening (``univtac_groot.server.run_server`` in the GR00T environment).

Example::

    python scripts/run_eval.py \
        --task insert_hole --task-config demo --variant baseline \
        --host 127.0.0.1 --port 5555 \
        --episodes 50 --execution-horizon 8 \
        --output eval_result/baseline/insert_hole.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

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
        default="demo",
        help="stem of a file in UniVTAC/task_config (demo, contact, clean)",
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "baseline_finetuned", "tactile"],
        help="ablation variant; 'tactile' concatenates the tactile array onto the state",
    )
    parser.add_argument(
        "--univtac-root",
        default=os.environ.get("UNIVTAC_ROOT", "."),
        help="path to the UniVTAC checkout (must contain envs/ and task_config/)",
    )

    # -- tactile ----------------------------------------------------------
    parser.add_argument(
        "--tactile-mode",
        default="depth_pool",
        choices=["depth_pool", "marker", "video"],
        help="how tactile enters the observation (--variant tactile only)",
    )
    parser.add_argument(
        "--tactile-sensors",
        nargs="+",
        default=["left_tactile", "right_tactile"],
        help="sensor names in observation['tactile']",
    )
    parser.add_argument(
        "--tactile-pool-grid",
        nargs=2,
        type=int,
        default=[8, 6],
        metavar=("ROWS", "COLS"),
        help="pooling grid per sensor; must match the checkpoint's modality config",
    )

    # -- server -----------------------------------------------------------
    parser.add_argument("--host", default=os.environ.get("GROOT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GROOT_PORT", 5555)))
    parser.add_argument("--api-token", default=os.environ.get("GROOT_API_TOKEN"))
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for the server to finish loading the checkpoint",
    )
    parser.add_argument("--request-timeout-ms", type=int, default=120_000)

    # -- control ----------------------------------------------------------
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        help="actions executed per chunk before re-planning; default = full chunk",
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
    parser.add_argument("--episodes", type=int, default=50)
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
    hide them from ``scripts/compare_ablation.py``.
    """
    if args.output:
        return Path(args.output).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return _REPO_ROOT / "eval_result" / args.variant / args.task / f"{stamp}.jsonl"


def build_arm_spec(args: argparse.Namespace):
    """Build the observation spec for the requested variant."""
    from univtac_groot.variants import build_spec

    kwargs = {}
    if args.variant == "tactile":
        grid = (int(args.tactile_pool_grid[0]), int(args.tactile_pool_grid[1]))
        kwargs = {
            "mode": args.tactile_mode,
            "sensor_names": tuple(args.tactile_sensors),
            "pool_grid": grid,
            "marker_pool": grid,
        }
    return build_spec(args.variant, **kwargs)


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
    import yaml

    path = Path(name)
    if path.suffix not in (".yml", ".yaml"):
        path = univtac_root / "task_config" / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(
            f"task config not found: {path}. Pass --task-config with a stem that "
            f"exists in {univtac_root / 'task_config'} (demo, contact, clean)."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_tactile_available(task_config: dict, args: argparse.Namespace) -> None:
    """Fail early if the task config does not expose the tactile data type needed.

    UniVTAC only populates the data types listed under ``observations.tactile``
    (``BaseTask._get_observations`` -> ``TactileManager.get_observations``), so a
    missing entry would surface mid-episode as a ``KeyError`` on seed 1.
    """
    if args.variant != "tactile":
        return
    available = set((task_config.get("observations") or {}).get("tactile") or [])
    needed = {"depth_pool": "depth", "marker": "marker", "video": "rgb"}[args.tactile_mode]
    if needed not in available:
        raise SystemExit(
            f"--tactile-mode {args.tactile_mode} needs observations.tactile to include "
            f"{needed!r}, but {args.task_config} lists {sorted(available) or 'nothing'}. "
            f"Add it to UniVTAC/task_config/{args.task_config}.yml."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    univtac_root = Path(args.univtac_root).resolve()

    from univtac_groot.action_adapter import ActionAdapter, GripperConvention
    from univtac_groot.client import Gr00tClient
    from univtac_groot.metrics import ResultWriter
    from univtac_groot.rollout import (
        BatchingPolicy,
        RolloutConfig,
        evaluate,
        resolve_spec_from_policy,
    )

    task_config = load_task_config(univtac_root, args.task_config)
    check_tactile_available(task_config, args)

    spec = build_arm_spec(args)
    output = resolve_output(args)
    print(f"[eval] variant={args.variant} task={args.task} -> {output}")

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
        "tactile_mode": spec.tactile.mode,
        "state_dim": spec.state_dim,
        "state_keys": list(spec.state_keys),
        "video_keys": sorted(spec.all_video_keys),
        "action_type": args.action_type,
        "gripper_invert": gripper_invert,
        "episodes_requested": args.episodes,
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

    print(json.dumps(summary, indent=2))
    return 0


def build_env(args, univtac_root: Path, spec, action_adapter, task_config: dict):  # noqa: C901
    """Launch Isaac Lab headlessly and construct the wrapped UniVTAC task.

    Mirrors the bootstrap order in ``UniVTAC/scripts/eval_policy.py``:
    ``AppLauncher`` must run before anything imports ``envs.*``, because those
    modules import ``isaaclab``/``omni`` at module scope.
    """
    import argparse as _argparse

    sys.path.insert(0, str(univtac_root))

    from isaaclab.app import AppLauncher

    launcher_parser = _argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    app_args = launcher_parser.parse_args([])
    app_args.headless = True          # no GUI: this must survive `sbatch`
    app_args.enable_cameras = True    # RGB + tactile rendering
    app_args.num_envs = 1
    if args.device:
        app_args.device = args.device

    global _SIMULATION_APP
    _SIMULATION_APP = AppLauncher(app_args).app

    import importlib

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

    from univtac_groot.env_wrapper import UniVTACGr00tEnv

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
