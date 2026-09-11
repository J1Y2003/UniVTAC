"""GR00T N1.7 policy for UniVTAC's native evaluator.

Copy (or symlink) this directory to ``UniVTAC/policy/GR00T/`` and evaluate with
UniVTAC's own harness::

    bash eval_policy.sh insert_hole demo GR00T/deploy_baseline 0

It implements the three-part contract from ``UniVTAC/docs/Deploy.md``
(``__init__(args)`` / ``encode_obs`` / ``eval(task, observation)`` / ``reset``)
and delegates everything substantive to :mod:`univtac_groot`, so the same
adapters and receding-horizon controller are used whether the
run is driven by UniVTAC's ``scripts/eval_policy.py`` or by this repo's
``scripts/run_eval.py``.

The model is **not** loaded in this process. UniVTAC's evaluator runs inside
Isaac Sim's interpreter, which pins its own torch; GR00T is served over ZeroMQ
from its own environment instead — the same split UniVTAC uses for SmolVLA
(``policy/smolvla/smolvla_server.py``). Either point ``groot_host``/``groot_port``
at an already-running server, or set ``groot_python`` and let this class spawn
one as a subprocess.

``eval()`` executes one *decision*: it asks the server for an action chunk and
dispatches the first ``execution_horizon`` actions of it, re-reading the
observation between actions so the caller's ``task.eval_success`` check stays
timely. UniVTAC calls it in a loop until ``step_lim`` or success.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

# UniVTAC puts ``policy/`` on sys.path (scripts/eval_policy.py appends "./policy"),
# and this repo's root must also be importable for ``univtac_groot``.
_THIS_DIR = Path(__file__).resolve().parent
for _candidate in (_THIS_DIR.parents[1], _THIS_DIR.parents[2]):
    if (_candidate / "univtac_groot").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from univtac_groot.action_adapter import ActionAdapter, GripperConvention  # noqa: E402
from univtac_groot.variants import build_spec  # noqa: E402
from univtac_groot.client import Gr00tClient  # noqa: E402
from univtac_groot.history import ObsHistory  # noqa: E402
from univtac_groot.obs_adapter import ObsAdapter  # noqa: E402
from univtac_groot.receding_horizon import RecedingHorizonController  # noqa: E402
from univtac_groot.rollout import batch_observation, resolve_spec_from_policy  # noqa: E402

try:  # UniVTAC's base class; absent when this file is imported for unit tests.
    from .._base_policy import BasePolicy  # type: ignore
except Exception:  # pragma: no cover - standalone import

    class BasePolicy:  # type: ignore[no-redef]
        """Minimal stand-in matching ``UniVTAC/policy/_base_policy.py``."""

        def __init__(self, args: dict) -> None:
            self.model = None

        def encode_obs(self, observation):
            return observation

        def reset(self) -> None:
            pass

        def close(self) -> None:
            pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Policy(BasePolicy):
    """UniVTAC deploy-time wrapper around a remote GR00T N1.7 policy.

    Recognised ``deploy.yml`` fields (all optional unless noted):

    ``policy_name``
        Must be ``GR00T`` -- UniVTAC uses it to import ``policy/GR00T``.
    ``variant``
        ``baseline`` (zero-shot DROID tag) or ``baseline_finetuned``. Selects
        the observation spec from :data:`univtac_groot.variants.VARIANTS`.
    ``execution_horizon``
        Actions executed per chunk before re-planning. Defaults to the full
        chunk the policy declares.
    ``action_type``
        ``qpos`` (default), ``ee`` or ``delta_ee``.
    ``gripper_invert`` / ``gripper_max_qpos``
        Gripper convention; see :class:`~univtac_groot.action_adapter.GripperConvention`.
    ``groot_host`` / ``groot_port`` / ``groot_api_token``
        Where the inference server lives. Port ``0`` picks a free port and
        implies spawning a server locally.
    ``groot_python`` / ``groot_model_path`` / ``groot_embodiment_tag`` /
    ``groot_modality_config``
        Set ``groot_python`` to the interpreter of the GR00T environment to have
        this class launch the server itself.
    ``groot_startup_timeout``
        Seconds to wait for the server (default 900; a 3B checkpoint plus its
        gated backbone is slow to load).
    """

    def __init__(self, args: dict) -> None:
        self.args = dict(args or {})
        self.task_name = str(self.args.get("task_name", "unknown"))

        # -- observation spec ---------------------------------------------
        variant = str(self.args.get("variant", "baseline"))
        spec_kwargs: dict[str, Any] = {}
        if self.args.get("language_key"):
            spec_kwargs["language_key"] = str(self.args["language_key"])
        self.spec = build_spec(variant, **spec_kwargs)
        self.variant = variant

        # -- action handling ----------------------------------------------
        self.action_adapter = ActionAdapter(
            action_type=str(self.args.get("action_type", "qpos")),  # type: ignore[arg-type]
            gripper=GripperConvention(
                max_qpos=float(self.args.get("gripper_max_qpos", 0.039)),
                invert=bool(self.args.get("gripper_invert", variant == "baseline")),
            ),
        )

        # -- transport ----------------------------------------------------
        self._process: subprocess.Popen | None = None
        host = str(self.args.get("groot_host", os.environ.get("GROOT_HOST", "127.0.0.1")))
        port = int(self.args.get("groot_port", os.environ.get("GROOT_PORT", 0)) or 0)
        if port == 0:
            port = _find_free_port()
            self._spawn_server(host="127.0.0.1", port=port)
            host = "127.0.0.1"

        self.client = Gr00tClient(
            host=host,
            port=port,
            timeout_ms=int(self.args.get("groot_request_timeout_ms", 120_000)),
            api_token=self.args.get("groot_api_token"),
        )
        self.client.wait_until_ready(
            timeout_s=float(self.args.get("groot_startup_timeout", 900))
        )
        # ``self.model`` is what BasePolicy.reset()/close() poke at.
        self.model = self.client

        # -- align with the live checkpoint --------------------------------
        self.spec, horizons = resolve_spec_from_policy(
            self.client,
            self.spec,
            execution_horizon=self.args.get("execution_horizon"),
        )
        self.execution_horizon = int(horizons["execution_horizon"])
        self.action_horizon = int(horizons["action_horizon"])

        self.obs_adapter = ObsAdapter(
            self.spec,
            gripper_max_qpos=float(self.args.get("gripper_max_qpos", 0.039)),
        )
        self.history = ObsHistory(
            self.spec.video_delta_indices, self.spec.state_delta_indices
        )
        self.controller = RecedingHorizonController(
            execution_horizon=self.execution_horizon,
            action_horizon=self.action_horizon,
        )
        self._primed = False
        self._instruction = ""

        print(
            f"[GR00T] variant={self.variant} task={self.task_name} "
            f"{self.obs_adapter.describe()} "
            f"action_type={self.action_adapter.action_type} "
            f"exec_horizon={self.execution_horizon}/{self.action_horizon}"
        )

    # -- server lifecycle --------------------------------------------------
    def _spawn_server(self, *, host: str, port: int) -> None:
        """Launch the inference server in the GR00T environment as a subprocess.

        Only used when ``groot_port`` is unset/0. On SLURM it is usually better
        to run the server as its own job step and pass ``groot_port``, so the
        two processes get separate log files.
        """
        python = self.args.get("groot_python")
        if not python:
            raise ValueError(
                "no GR00T server to talk to: set groot_port to an already-running "
                "server, or set groot_python to the interpreter of the GR00T "
                "environment so one can be launched. UniVTAC's Isaac Sim "
                "interpreter cannot load GR00T in-process."
            )
        model_path = self.args.get("groot_model_path")
        embodiment_tag = self.args.get("groot_embodiment_tag")
        if not model_path or not embodiment_tag:
            raise ValueError(
                "spawning a server needs groot_model_path and groot_embodiment_tag "
                "in deploy.yml."
            )

        repo_root = _THIS_DIR.parents[1]
        cmd = [
            str(python),
            "-m",
            "univtac_groot.server.run_server",
            "--model-path",
            str(model_path),
            "--embodiment-tag",
            str(embodiment_tag),
            "--host",
            host,
            "--port",
            str(port),
        ]
        if self.args.get("groot_modality_config"):
            cmd += ["--modality-config-path", str(self.args["groot_modality_config"])]
        if self.args.get("groot_api_token"):
            cmd += ["--api-token", str(self.args["groot_api_token"])]
        if self.args.get("groot_device"):
            cmd += ["--device", str(self.args["groot_device"])]

        log_path = Path(self.args.get("groot_server_log", "groot_server.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[GR00T] launching server: {' '.join(cmd)}  (log: {log_path})")
        self._log_handle = log_path.open("a", encoding="utf-8")

        # Scrub Isaac Sim's CUDA environment: UniVTAC exports CUDA_HOME and
        # LD_LIBRARY_PATH for CUDA 12.4, and leaking them makes the cu128
        # server load a mismatched cuDNN and fail on the first inference.
        server_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX")
        }
        server_env["PYTHONPATH"] = str(repo_root)
        server_env["PYTHONUNBUFFERED"] = "1"

        self._process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=server_env,
        )

    # -- UniVTAC policy contract ------------------------------------------
    def encode_obs(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a UniVTAC observation into a batched GR00T observation.

        Feeds the frame into the history buffer first, so the returned dict
        carries the temporal stack the checkpoint's ``delta_indices`` require.
        """
        frame = self.obs_adapter(observation, self._instruction)
        if not self._primed:
            stacked = self.history.reset(frame)
            self._primed = True
        else:
            self.history.append(frame)
            stacked = self.history.observe()
        return batch_observation(stacked)

    def eval(self, task: Any, observation: Mapping[str, Any]) -> tuple[bool, bool]:
        """Run one decision: predict a chunk if needed, then execute it.

        Dispatches up to ``execution_horizon`` actions, refreshing the
        observation between each so success is detected as soon as UniVTAC sets
        ``eval_success``. Returns ``(exec_success, eval_success)`` like
        ``BaseTask.take_action``.
        """
        if not self._instruction:
            self._instruction = str(
                getattr(task, "instruction", "") or self.task_name.replace("_", " ")
            )

        exec_success, eval_success = True, bool(getattr(task, "eval_success", False))
        current = observation

        for _ in range(max(1, self.execution_horizon)):
            batched = self.encode_obs(current)

            # ``observation`` is bound as a default so the closure cannot
            # capture a later loop iteration's value.
            def infer(observation=batched) -> np.ndarray:
                chunk, _info = self.client.get_action(observation)
                return self.action_adapter.to_univtac(chunk)

            action = self.controller.next_action(infer)
            exec_success, eval_success = task.take_action(
                self._to_tensor(action, task),
                action_type=self.action_adapter.action_type,
            )
            if eval_success:
                break
            if getattr(task, "take_action_cnt", 0) >= getattr(task.cfg, "step_lim", 300):
                break
            if task.check_early_stop():
                break
            current = task._get_observations()

        return exec_success, eval_success

    def reset(self) -> None:
        """Clear per-episode state: history, chunk cache and cached instruction."""
        self._primed = False
        self._instruction = ""
        self.controller.reset()
        try:
            self.client.reset()
        except Exception as exc:  # noqa: BLE001 - a reset failure must not kill the run
            print(f"[GR00T] warning: server reset failed: {exc}")

    def close(self) -> None:
        """Close the socket and, if we spawned it, stop the server."""
        client = getattr(self, "client", None)
        if client is not None:
            if self._process is not None:
                client.kill_server()
            client.close()
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
        handle = getattr(self, "_log_handle", None)
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _to_tensor(action: np.ndarray, task: Any) -> Any:
        """Put the action on the task's device as a float tensor."""
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        try:
            import torch
        except ImportError:  # pragma: no cover
            return arr
        return torch.from_numpy(arr).to(getattr(task, "device", "cpu")).float()
