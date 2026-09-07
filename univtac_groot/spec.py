"""Shared, dependency-light specifications for the UniVTAC x GR00T N1.7 bridge.

Everything in this module is plain numpy/dataclasses so it can be imported both
inside the Isaac Lab process (which must *not* import torch-heavy GR00T code)
and inside the GR00T inference process.

Facts encoded here were read out of the two upstream repos (see docs/UPSTREAM.md):

UniVTAC (``univtac/UniVTAC``)
    * ``envs/_base_task.py::BaseTask._get_observations`` returns::

          {
            'observation': {'head': {'rgb': HWC uint8}, 'wrist': {...}},
            'tactile':     {'<name>': {'rgb':..., 'rgb_marker':..., 'depth':...,
                                       'marker':..., 'pose':...}},
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

from dataclasses import dataclass, field
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
# Tactile handling
# --------------------------------------------------------------------------- #

TactileMode = Literal["none", "depth_pool", "marker", "video"]
"""How the UniVTAC tactile stream enters the GR00T observation.

``none``
    Baseline. Tactile is dropped entirely; state is proprioception only.
``depth_pool``
    Ablation (default). The per-sensor ``depth`` height map is average-pooled to
    a small grid, flattened, and concatenated onto the 1-D state vector.
``marker``
    Ablation. The per-sensor ``marker`` (marker-motion) field is reduced to 2-D
    displacements, optionally pooled, flattened and concatenated onto the state.
``video``
    Ablation. Tactile RGB images are passed as *additional video streams*
    instead of touching the state vector. This is the architecturally natural
    route for UniVTAC's camera-based sensors (GelSight Mini / GF225 / XenseWS)
    but it changes the video modality keys, so it needs its own finetune.
"""

MarkerLayout = Literal["auto", "dxdy", "xydxdy", "raw"]
"""Interpretation of the trailing axis of the ``marker`` observation.

TacEx marker-motion arrays are commonly ``(N, 4)`` as ``[x, y, dx, dy]`` or
``(N, 2)`` as ``[dx, dy]``; ``auto`` picks ``dxdy`` for width 4 (taking the last
two columns), passes width 2 through, and otherwise falls back to ``raw``.
"""


@dataclass(frozen=True)
class TactileSpec:
    """How to turn UniVTAC tactile observations into state dimensions.

    Attributes:
        mode: see :data:`TactileMode`.
        sensor_names: tactile sensor keys to read, in this exact order, from
            ``observation['tactile']``. UniVTAC names them per task; the
            defaults cover the two-finger GelSight Mini setup and are matched
            with the ``*_gsmini`` aliases seen in the HDF5 dumps.
        pool_grid: ``(rows, cols)`` average-pool target for ``depth_pool`` (and
            for ``marker`` when ``marker_pool`` is set).
        marker_layout: see :data:`MarkerLayout`.
        marker_pool: optional ``(rows, cols)`` pooling of the marker field,
            which is needed because a full GelSight Mini marker grid is 9x7=64
            markers x 2 = 128 dims and would not fit the state budget.
        depth_clip: ``(lo, hi)`` clip applied to the raw height map before
            normalisation, in the sensor's native units.
        normalize: scale pooled tactile values into roughly ``[-1, 1]``.
    """

    mode: TactileMode = "none"
    sensor_names: tuple[str, ...] = ("left_tactile", "right_tactile")
    pool_grid: tuple[int, int] = (8, 6)
    marker_layout: MarkerLayout = "auto"
    marker_pool: tuple[int, int] | None = (8, 6)
    depth_clip: tuple[float, float] = (-1.0, 1.0)
    normalize: bool = True

    @property
    def enabled(self) -> bool:
        """Whether tactile contributes anything at all."""
        return self.mode != "none"

    @property
    def in_state(self) -> bool:
        """Whether tactile is concatenated onto the 1-D state vector."""
        return self.mode in ("depth_pool", "marker")

    def dims_per_sensor(self) -> int:
        """Number of state dimensions contributed by a single sensor."""
        if not self.in_state:
            return 0
        rows, cols = self.pool_grid if self.mode == "depth_pool" else (self.marker_pool or (0, 0))
        if self.mode == "marker":
            if self.marker_pool is None:
                raise ValueError(
                    "TactileSpec(mode='marker') needs marker_pool to be set so the "
                    "flattened marker field has a statically known width; got None."
                )
            return int(rows * cols * 2)  # 2 displacement components per cell
        return int(rows * cols)

    def total_state_dims(self) -> int:
        """Number of state dimensions contributed by all sensors."""
        return self.dims_per_sensor() * len(self.sensor_names)


# --------------------------------------------------------------------------- #
# Observation / state layout
# --------------------------------------------------------------------------- #

StateKind = Literal["eef_9d", "joint_position", "gripper_position", "tactile"]


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
    """Full observation contract for one ablation arm.

    Attributes:
        video_keys: mapping ``gr00t video key -> UniVTAC camera name``, e.g.
            ``{"exterior_image_1_left": "head", "wrist_image_left": "wrist"}``.
        tactile_video_keys: mapping ``gr00t video key -> tactile sensor name``,
            only used when ``tactile.mode == 'video'``.
        state_fields: ordered state slices.
        language_key: the GR00T language modality key, e.g.
            ``annotation.language.language_instruction``.
        tactile: see :class:`TactileSpec`.
        image_size: ``(height, width)`` every RGB stream is resized to.
        tactile_image_size: ``(height, width)`` for tactile RGB streams.
        video_delta_indices: per-step frame offsets the policy expects
            (e.g. ``(-15, 0)`` for the DROID pretrain tag, ``(0,)`` typically
            for a fresh finetune). Resolved from the server at runtime.
        state_delta_indices: same for the state stream.
    """

    video_keys: dict[str, str]
    state_fields: tuple[StateField, ...]
    language_key: str
    tactile: TactileSpec = field(default_factory=TactileSpec)
    tactile_video_keys: dict[str, str] = field(default_factory=dict)
    image_size: tuple[int, int] = (256, 256)
    tactile_image_size: tuple[int, int] = (256, 256)
    video_delta_indices: tuple[int, ...] = (0,)
    state_delta_indices: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        self.validate()

    # -- derived -----------------------------------------------------------
    @property
    def all_video_keys(self) -> dict[str, str]:
        """Visual plus (when enabled) tactile video streams."""
        merged = dict(self.video_keys)
        if self.tactile.mode == "video":
            merged.update(self.tactile_video_keys)
        return merged

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
                f"max_state_dim={MAX_STATE_DIM} "
                f"(gr00t/configs/model/gr00t_n1d7.py). Shrink TactileSpec.pool_grid / "
                f"marker_pool: proprioception here is "
                f"{self.state_dim - self.tactile.total_state_dims()}-D, leaving "
                f"{MAX_STATE_DIM - (self.state_dim - self.tactile.total_state_dims())} "
                f"dims for tactile across {len(self.tactile.sensor_names)} sensor(s)."
            )

        tactile_declared = sum(f.dim for f in self.state_fields if f.kind == "tactile")
        tactile_expected = self.tactile.total_state_dims()
        if tactile_declared != tactile_expected:
            raise ValueError(
                f"ObsSpec declares {tactile_declared} tactile state dims but "
                f"TactileSpec(mode={self.tactile.mode!r}) produces {tactile_expected}."
            )

        if self.tactile.mode == "video" and not self.tactile_video_keys:
            raise ValueError("TactileSpec(mode='video') needs tactile_video_keys.")

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
