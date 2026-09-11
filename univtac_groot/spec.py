"""Shared, dependency-light specifications for the UniVTAC x GR00T N1.7 bridge.

Everything in this module is plain numpy/dataclasses so it can be imported both
inside the Isaac Lab process (which must *not* import torch-heavy GR00T code)
and inside the GR00T inference process.

Facts encoded here were read out of the two upstream repos (see docs/UPSTREAM.md):

UniVTAC (``univtac/UniVTAC``)
    * ``envs/_base_task.py::BaseTask._get_observations`` returns::

          {
            'observation': {'head': {'rgb': HWC uint8}, 'wrist': {...}},
            'embodiment':  {'joint': (9,), 'ee': (7,)},
            'actor': {...}, 'step': int, 'atom': {...},
          }

    * ``envs/robot/robot.py`` -> Franka Panda: 7 arm joints
      (``panda_joint1..7``) + 2 finger joints (``panda_finger_joint1/2``), so
      ``embodiment/joint`` is 9-D and ``embodiment/ee`` is a 7-D pose
      (position xyz + quaternion wxyz).
    * ``BaseTask.take_action(action, action_type)`` accepts
      ``'qpos'`` -> 8-D (7 arm + 1 gripper), ``'ee'`` -> 8-D (pos3 + quat4 +
      gripper), ``'delta_ee'`` -> 7-D (dpos3 + drot3 + dgripper).
    * ``RobotManager.gripper_max_qpos`` defaults to ``0.039`` m (per-finger
      opening); larger == more open.

GR00T N1.7 (``NVIDIA/Isaac-GR00T``)
    * ``gr00t/configs/model/gr00t_n1d7.py``: ``max_state_dim = 132``,
      ``max_action_dim = 132``, ``action_horizon = 40``.
    * ``gr00t/policy/gr00t_policy.py::Gr00tSimPolicyWrapper`` consumes the flat
      sim observation convention: ``video.<key>`` (B,T,H,W,C) uint8,
      ``state.<key>`` (B,T,D) float32, language under its modality key as a
      ``tuple[str]`` of length B; it returns ``action.<key>`` (B,T,D) float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# --------------------------------------------------------------------------- #
# Hard limits taken from gr00t/configs/model/gr00t_n1d7.py
# --------------------------------------------------------------------------- #

MAX_STATE_DIM = 132
"""``GR00T_N1d7Config.max_state_dim``. The concatenated state must fit in this."""

MAX_ACTION_DIM = 132
"""``GR00T_N1d7Config.max_action_dim``."""

DEFAULT_ACTION_HORIZON = 40
"""``GR00T_N1d7Config.action_horizon``. N1.7 predicts 40 steps, not the 16 of N1.5/N1.6."""


# --------------------------------------------------------------------------- #
# UniVTAC embodiment constants (envs/robot/robot.py, envs/_base_task.py)
# --------------------------------------------------------------------------- #

N_ARM_JOINTS = 7
"""``panda_joint1..panda_joint7``."""

N_FINGER_JOINTS = 2
"""``panda_finger_joint1``, ``panda_finger_joint2``."""

N_JOINT_STATE = N_ARM_JOINTS + N_FINGER_JOINTS
"""Width of ``observation['embodiment']['joint']`` (``robot.data.joint_pos``)."""

N_EE_STATE = 7
"""Width of ``observation['embodiment']['ee']``: xyz + quaternion (w, x, y, z)."""

DEFAULT_GRIPPER_MAX_QPOS = 0.039
"""``RobotManager.gripper_max_qpos`` default, in metres of per-finger opening."""

UNIVTAC_TASKS = (
    "lift_bottle",
    "lift_can",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "pull_out_key",
    "put_bottle_in_shelf",
    "grasp_classify",
)
"""The eight benchmark tasks under ``UniVTAC/envs/`` (``collect`` is data-gen only)."""


# --------------------------------------------------------------------------- #
# Observation / state layout
# --------------------------------------------------------------------------- #

StateKind = Literal["eef_9d", "joint_position", "gripper_position"]


@dataclass(frozen=True)
class StateField:
    """One named slice of the GR00T ``state`` modality.

    ``key`` is the GR00T modality key (must match the checkpoint's
    ``meta/modality.json`` / registered ``ModalityConfig``), ``dim`` its width.
    """

    key: str
    kind: StateKind
    dim: int


@dataclass
class ObsSpec:
    """Full observation contract for one ablation variant.

    Attributes:
        video_keys: mapping ``gr00t video key -> UniVTAC camera name``, e.g.
            ``{"exterior_image_1_left": "head", "wrist_image_left": "wrist"}``.
        state_fields: ordered state slices.
        language_key: the GR00T language modality key, e.g.
            ``annotation.language.language_instruction``.
        image_size: ``(height, width)`` every RGB stream is resized to.
        video_delta_indices: per-step frame offsets the policy expects
            (e.g. ``(-15, 0)`` for the DROID pretrain tag, ``(0,)`` typically
            for a fresh finetune). Resolved from the server at runtime.
        state_delta_indices: same for the state stream.
    """

    video_keys: dict[str, str]
    state_fields: tuple[StateField, ...]
    language_key: str
    image_size: tuple[int, int] = (256, 256)
    video_delta_indices: tuple[int, ...] = (0,)
    state_delta_indices: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        self.validate()

    # -- derived -----------------------------------------------------------
    @property
    def all_video_keys(self) -> dict[str, str]:
        """The video streams this spec declares."""
        return dict(self.video_keys)

    @property
    def state_dim(self) -> int:
        """Total width of the concatenated state vector."""
        return sum(f.dim for f in self.state_fields)

    @property
    def state_keys(self) -> tuple[str, ...]:
        return tuple(f.key for f in self.state_fields)

    def field_slices(self) -> dict[str, slice]:
        """``{state key: slice}`` into the concatenated state vector."""
        out, offset = {}, 0
        for f in self.state_fields:
            out[f.key] = slice(offset, offset + f.dim)
            offset += f.dim
        return out

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        """Fail fast on layouts GR00T N1.7 cannot represent."""
        if not self.all_video_keys:
            raise ValueError("ObsSpec needs at least one video key.")
        if not self.state_fields:
            raise ValueError("ObsSpec needs at least one state field.")

        dup = [k for k in self.state_keys if self.state_keys.count(k) > 1]
        if dup:
            raise ValueError(f"Duplicate state keys in ObsSpec: {sorted(set(dup))}")

        if self.state_dim > MAX_STATE_DIM:
            raise ValueError(
                f"Concatenated state is {self.state_dim}-D but GR00T N1.7 caps it at "
                f"max_state_dim={MAX_STATE_DIM} (gr00t/configs/model/gr00t_n1d7.py)."
            )

        for name, deltas in (
            ("video_delta_indices", self.video_delta_indices),
            ("state_delta_indices", self.state_delta_indices),
        ):
            assert_delta_indices(np.asarray(deltas), name=name)


def assert_delta_indices(delta_indices: np.ndarray, *, name: str = "delta_indices") -> None:
    """Mirror ``MultiStepWrapper.assert_delta_indices`` from Isaac-GR00T.

    Offsets must be non-positive (the future is not observable), end at 0 (the
    newest frame is always used), and be evenly and positively spaced.
    """
    d = np.asarray(delta_indices)
    if d.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(d <= 0):
        raise ValueError(f"{name}={d.tolist()} must be non-positive.")
    if d[-1] != 0:
        raise ValueError(f"{name}={d.tolist()} must end at 0 (the latest observation).")
    if d.size > 1:
        step = d[1] - d[0]
        if step <= 0:
            raise ValueError(f"{name}={d.tolist()} must be strictly increasing.")
        if not np.all(np.diff(d) == step):
            raise ValueError(f"{name}={d.tolist()} must be evenly spaced; got step {step}.")


def history_length(*delta_index_sets: tuple[int, ...] | None) -> int:
    """Number of past observations that must be retained to serve all offsets.

    Matches ``MultiStepWrapper.get_max_steps_needed``.
    """
    spans = [
        int(np.max(d) - np.min(d) + 1) for d in delta_index_sets if d is not None and len(d) > 0
    ]
    return max(spans) if spans else 1
