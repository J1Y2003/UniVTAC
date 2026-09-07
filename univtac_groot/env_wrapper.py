"""Gymnasium-style wrapper standardising UniVTAC's ``BaseTask`` for GR00T.

UniVTAC's ``BaseTask`` is not a Gym env: it is an Isaac Lab ``DirectRLEnvCfg``
scene driven imperatively (``reset(seed=...)``, ``_get_observations()``,
``take_action(action, action_type=...)``, ``check_success()``), and its evaluator
(``scripts/eval_policy.py``) hands the task object to the policy so the policy
can step it. This wrapper puts the standard surface back on top:

* ``reset(seed=...) -> (obs, info)`` and ``step(action) -> (obs, reward, terminated, truncated, info)``;
* a ``gymnasium.spaces.Dict`` observation space over GR00T's flat keys
  (``video.*``, ``state.*``, plus the language key) with the temporal axis
  already folded in;
* one ``use_tactile`` switch — really the :class:`~univtac_groot.spec.TactileSpec`
  carried by the :class:`~univtac_groot.spec.ObsSpec` — that decides whether the
  flattened tactile array is concatenated onto the state vector.

``gymnasium`` is imported lazily so the pure-numpy adapters stay importable on a
login node with no RL stack installed; the wrapper degrades to a duck-typed
object exposing the same methods when gymnasium is absent.

Reward: UniVTAC's tasks expose success, not a shaped reward
(``BaseTask.check_success`` returns a bool and there is no reward function), so
this wrapper reports the sparse success indicator as the reward. That keeps
"logs success rates and rewards" honest: mean reward *is* the success rate here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .action_adapter import ActionAdapter
from .history import ObsHistory
from .obs_adapter import ObsAdapter
from .spec import ObsSpec


def _gym_spaces():
    """Return ``gymnasium.spaces`` or ``None`` when gymnasium is unavailable."""
    try:  # pragma: no cover - environment dependent
        from gymnasium import spaces

        return spaces
    except ImportError:  # pragma: no cover
        return None


def build_observation_space(spec: ObsSpec):
    """Build the ``spaces.Dict`` for ``spec``, or ``None`` without gymnasium.

    Video entries are ``(T, H, W, 3)`` uint8 and state entries ``(T, D)``
    float32, matching what :class:`~univtac_groot.history.ObsHistory` emits and
    what ``Gr00tSimPolicyWrapper.check_observation`` validates once the batch
    axis is added.
    """
    spaces = _gym_spaces()
    if spaces is None:
        return None

    video_t = len(spec.video_delta_indices)
    state_t = len(spec.state_delta_indices)
    h, w = spec.image_size
    th, tw = spec.tactile_image_size

    entries: dict[str, Any] = {}
    for key in spec.video_keys:
        entries[f"video.{key}"] = spaces.Box(0, 255, (video_t, h, w, 3), dtype=np.uint8)
    if spec.tactile.mode == "video":
        for key in spec.tactile_video_keys:
            entries[f"video.{key}"] = spaces.Box(0, 255, (video_t, th, tw, 3), dtype=np.uint8)
    for f in spec.state_fields:
        entries[f"state.{f.key}"] = spaces.Box(
            -np.inf, np.inf, (state_t, f.dim), dtype=np.float32
        )
    entries[spec.language_key] = spaces.Text(max_length=1024)
    return spaces.Dict(entries)


class UniVTACGr00tEnv:
    """Standardised single-environment view of a UniVTAC task.

    Args:
        task: a live UniVTAC ``BaseTask`` (already constructed with
            ``mode='eval'``; the caller owns Isaac Lab's ``AppLauncher``).
        spec: observation contract for this ablation arm.
        action_adapter: converts GR00T action chunks to UniVTAC action vectors.
            Its ``action_type`` also selects the ``take_action`` signature used.
        instructions: candidate language instructions. When given, ``reset``
            forwards them to ``BaseTask.reset(instructions=...)``, which samples
            one with the task's seeded RNG and exposes it as ``task.instruction``.
        max_steps: episode cap. Defaults to the task's own ``cfg.step_lim``.

    Note:
        UniVTAC runs exactly one environment (``scripts/eval_policy.py`` forces
        ``env_cfg.scene.num_envs = 1``), so this wrapper is deliberately
        single-env and adds no batch axis; :class:`univtac_groot.client` does
        that at the wire boundary.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: Any,
        spec: ObsSpec,
        *,
        action_adapter: ActionAdapter | None = None,
        instructions: Sequence[str] | None = None,
        max_steps: int | None = None,
        gripper_max_qpos: float | None = None,
    ) -> None:
        self.task = task
        self.spec = spec
        self.action_adapter = action_adapter or ActionAdapter()
        self.instructions = list(instructions) if instructions else None

        if gripper_max_qpos is None:
            gripper_max_qpos = _read_gripper_max_qpos(task)
        self.obs_adapter = ObsAdapter(spec, gripper_max_qpos=gripper_max_qpos)
        self.history = ObsHistory(spec.video_delta_indices, spec.state_delta_indices)

        self.observation_space = build_observation_space(spec)
        self.action_space = None  # actions arrive as GR00T chunks, not sampled

        self._max_steps = max_steps
        self._instruction = ""
        self._elapsed = 0

    # -- properties --------------------------------------------------------
    @property
    def max_steps(self) -> int:
        """Episode cap: explicit override, else the task's ``cfg.step_lim`` (300)."""
        if self._max_steps is not None:
            return int(self._max_steps)
        return int(getattr(self.task.cfg, "step_lim", 300))

    @property
    def instruction(self) -> str:
        """The instruction sampled for the current episode."""
        return self._instruction

    @property
    def use_tactile(self) -> bool:
        """Whether tactile is contributing to this arm's observations."""
        return self.spec.tactile.enabled

    # -- gym API -----------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the task for ``seed`` and return the first stacked observation.

        ``BaseTask.reset`` takes the seed positionally as ``seed=-1`` meaning
        "keep the current one", so ``None`` is forwarded as ``-1``.
        """
        reset_kwargs: dict[str, Any] = {"seed": -1 if seed is None else int(seed)}
        if self.instructions:
            reset_kwargs["instructions"] = self.instructions
        if options:
            reset_kwargs["options"] = dict(options)
        self.task.reset(**reset_kwargs)

        self._instruction = str(getattr(self.task, "instruction", "") or "")
        self._elapsed = 0
        raw = self.task._get_observations()
        obs = self.history.reset(self.obs_adapter(raw, self._instruction))
        return obs, {"instruction": self._instruction, "seed": seed}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Dispatch one already-converted UniVTAC action vector.

        Args:
            action: the output of :meth:`ActionAdapter.step` — 8-D for ``qpos``
                and ``ee``, 7-D for ``delta_ee``.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` where ``reward`` is
            the sparse success indicator, ``terminated`` marks success or the
            task's own early stop, and ``truncated`` marks the step cap.
        """
        tensor = self._to_task_tensor(action)
        exec_success, eval_success = self.task.take_action(
            tensor, action_type=self.action_adapter.action_type
        )
        self._elapsed += 1

        raw = self.task._get_observations()
        self.history.append(self.obs_adapter(raw, self._instruction))
        obs = self.history.observe()

        success = bool(eval_success)
        early_stop = bool(self.task.check_early_stop())
        # take_action_cnt is the authoritative counter: UniVTAC refuses actions
        # once it reaches cfg.step_lim, which would otherwise spin forever.
        taken = int(getattr(self.task, "take_action_cnt", self._elapsed))
        truncated = taken >= self.max_steps
        terminated = success or early_stop

        info = {
            "success": success,
            "exec_success": bool(exec_success),
            "early_stop": early_stop,
            "take_action_cnt": taken,
            "step_count": int(getattr(self.task, "step_count", taken)),
            "plan_success": bool(getattr(self.task, "plan_success", True)),
        }
        return obs, float(success), terminated, truncated, info

    def step_chunk(
        self, chunk: Mapping[str, np.ndarray], n_steps: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute the first ``n_steps`` actions of a GR00T chunk open-loop.

        Provided for callers that prefer ``MultiStepWrapper``'s macro-step shape
        over :class:`~univtac_groot.receding_horizon.RecedingHorizonController`.
        Rewards are aggregated with ``max`` (i.e. "did any inner step succeed"),
        matching ``MultiStepWrapper``'s default ``reward_agg_method``.
        """
        obs: dict[str, Any] = self.history.observe()
        rewards: list[float] = []
        terminated = truncated = False
        info: dict[str, Any] = {}
        executed = 0
        for i in range(int(n_steps)):
            obs, reward, terminated, truncated, info = self.step(
                self.action_adapter.step(chunk, i)
            )
            rewards.append(reward)
            executed += 1
            if terminated or truncated:
                break
        info = dict(info)
        info["n_env_steps"] = executed
        return obs, (max(rewards) if rewards else 0.0), terminated, truncated, info

    def close(self) -> None:
        """Close the underlying task if it owns resources."""
        close = getattr(self.task, "close", None)
        if callable(close):
            close()

    # -- helpers -----------------------------------------------------------
    def _to_task_tensor(self, action: np.ndarray) -> Any:
        """Move a numpy action onto the task's device as a float tensor.

        ``BaseTask.take_action`` indexes and arithmetics the action as a torch
        tensor on ``task.device``; torch is imported here rather than at module
        scope so the adapters remain importable without it.
        """
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        try:
            import torch
        except ImportError:  # pragma: no cover - only in torch-free unit tests
            return arr
        return torch.from_numpy(arr).to(getattr(self.task, "device", "cpu")).float()


def _read_gripper_max_qpos(task: Any) -> float:
    """Read ``gripper_max_qpos`` off the task's robot manager, with a fallback.

    ``RobotManager`` sets it from ``RobotCfg.gripper_max_qpos``; tasks that
    predate that field fall back to the 0.039 m default in
    :mod:`univtac_groot.spec`.
    """
    from .spec import DEFAULT_GRIPPER_MAX_QPOS

    manager = getattr(task, "_robot_manager", None)
    value = getattr(manager, "gripper_max_qpos", None) if manager is not None else None
    try:
        value = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_GRIPPER_MAX_QPOS
    return value if value > 0 else DEFAULT_GRIPPER_MAX_QPOS
