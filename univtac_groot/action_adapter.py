"""Turn a GR00T action chunk into the vectors ``BaseTask.take_action`` accepts.

``Gr00tSimPolicyWrapper`` returns ``{"action.<key>": (B, T, D) float32}``. The
keys are whatever the embodiment's ``ModalityConfig`` declares, and the values
are already un-normalised and resolved out of any RELATIVE representation by
``processor.decode_action`` (see ``Gr00tPolicy._get_action`` step 5), so they are
absolute targets in physical units.

UniVTAC's ``BaseTask.take_action(action, action_type=...)`` accepts:

``'qpos'``
    ``Tensor([8])`` = 7 arm joint targets + 1 gripper opening.
    Internally ``set_arm(action[:-1])`` over the 7 ``panda_joint*`` ids and
    ``set_gripper(action[-1])`` over both ``panda_finger_joint*`` ids.
``'ee'``
    ``Tensor([8])`` = position (3) + quaternion wxyz (4) + gripper (1).
``'delta_ee'``
    ``Tensor([7])`` = delta position (3) + delta euler (3) + delta gripper (1).

The gripper needs an explicit convention change: GR00T's ``gripper_position``
for the DROID-style embodiments is a normalised scalar, whereas UniVTAC wants
metres of finger opening bounded by ``RobotManager.gripper_max_qpos`` (0.039 m
by default), with *larger meaning more open*. :class:`GripperConvention` makes
that mapping explicit and invertible rather than hiding it in a magic constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .spec import DEFAULT_GRIPPER_MAX_QPOS

ActionType = Literal["qpos", "ee", "delta_ee"]


@dataclass(frozen=True)
class GripperConvention:
    """Mapping from a GR00T gripper scalar to a UniVTAC finger opening in metres.

    Attributes:
        max_qpos: finger opening at fully open (``RobotManager.gripper_max_qpos``).
        invert: set when the policy's scalar means *closed* at 1.0. DROID-style
            checkpoints are usually trained with 1.0 == closed, so this defaults
            to ``True``; a checkpoint finetuned on UniVTAC data through
            :mod:`scripts.convert_univtac_to_lerobot` emits the UniVTAC
            convention directly and wants ``invert=False``.
        input_range: the range the policy's scalar is expected to occupy. Values
            outside it are clipped, which keeps a mis-scaled checkpoint from
            driving the fingers past their joint limits.
    """

    max_qpos: float = DEFAULT_GRIPPER_MAX_QPOS
    invert: bool = True
    input_range: tuple[float, float] = (0.0, 1.0)

    def to_univtac(self, value: float) -> float:
        """Map a policy gripper scalar to a UniVTAC finger opening in metres."""
        lo, hi = self.input_range
        if hi <= lo:
            raise ValueError(f"invalid input_range {self.input_range}")
        frac = (float(value) - lo) / (hi - lo)
        frac = float(np.clip(frac, 0.0, 1.0))
        if self.invert:
            frac = 1.0 - frac
        return frac * self.max_qpos

    def from_univtac(self, qpos: float) -> float:
        """Inverse of :meth:`to_univtac`; used when building training data."""
        frac = float(np.clip(float(qpos) / self.max_qpos, 0.0, 1.0)) if self.max_qpos else 0.0
        if self.invert:
            frac = 1.0 - frac
        lo, hi = self.input_range
        return lo + frac * (hi - lo)


class ActionAdapter:
    """Slice a GR00T action chunk into per-step UniVTAC action vectors.

    Args:
        action_type: which ``take_action`` signature to target.
        arm_key: action key holding the 7 arm joint targets (``'qpos'`` only).
        gripper_key: action key holding the gripper scalar.
        eef_key: action key holding the end-effector pose (``'ee'``/``'delta_ee'``).
        gripper: see :class:`GripperConvention`.

    The ``'ee'`` path expects ``eef_key`` to be the 9-D ``[xyz, rot6d]`` form
    that ``ActionFormat.XYZ_ROT6D`` produces and converts it back to the
    ``[xyz, quaternion wxyz]`` UniVTAC wants.
    """

    def __init__(
        self,
        *,
        action_type: ActionType = "qpos",
        arm_key: str = "joint_position",
        gripper_key: str = "gripper_position",
        eef_key: str = "eef_9d",
        gripper: GripperConvention | None = None,
    ) -> None:
        if action_type not in ("qpos", "ee", "delta_ee"):
            raise ValueError(f"unsupported action_type {action_type!r}")
        self.action_type = action_type
        self.arm_key = arm_key
        self.gripper_key = gripper_key
        self.eef_key = eef_key
        self.gripper = gripper or GripperConvention()

    # -- chunk handling ----------------------------------------------------
    def required_keys(self) -> tuple[str, ...]:
        """Action keys this adapter reads, without the ``action.`` prefix."""
        if self.action_type == "qpos":
            return (self.arm_key, self.gripper_key)
        return (self.eef_key, self.gripper_key)

    @staticmethod
    def _get(chunk: Mapping[str, np.ndarray], key: str) -> np.ndarray:
        """Fetch ``key`` tolerating the ``action.`` prefix, and drop the batch axis."""
        arr = chunk.get(key)
        if arr is None:
            arr = chunk.get(f"action.{key}")
        if arr is None:
            available = sorted(chunk)
            raise KeyError(
                f"action key {key!r} missing from the policy output (available: "
                f"{available}). The embodiment's ModalityConfig action.modality_keys "
                f"must include it; adjust ActionAdapter's *_key arguments to match."
            )
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:  # (B, T, D) -> (T, D), batch size 1 in UniVTAC
            if arr.shape[0] != 1:
                raise ValueError(
                    f"expected batch size 1 for action {key!r}, got {arr.shape[0]}"
                )
            arr = arr[0]
        if arr.ndim == 1:  # (D,) -> (1, D)
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"action {key!r} must be (T, D); got shape {arr.shape}")
        return arr

    def chunk_length(self, chunk: Mapping[str, np.ndarray]) -> int:
        """Number of predicted steps in ``chunk``, verifying keys agree."""
        lengths = {key: self._get(chunk, key).shape[0] for key in self.required_keys()}
        distinct = set(lengths.values())
        if len(distinct) != 1:
            raise ValueError(f"action keys disagree on horizon: {lengths}")
        return distinct.pop()

    def to_univtac(self, chunk: Mapping[str, np.ndarray]) -> np.ndarray:
        """Convert a whole chunk to ``(T, action_dim)`` UniVTAC action vectors."""
        length = self.chunk_length(chunk)
        return np.stack([self.step(chunk, t) for t in range(length)], axis=0)

    def step(self, chunk: Mapping[str, np.ndarray], index: int) -> np.ndarray:
        """Convert step ``index`` of ``chunk`` to one UniVTAC action vector."""
        gripper_raw = self._get(chunk, self.gripper_key)[index]
        gripper = self.gripper.to_univtac(float(gripper_raw.reshape(-1)[0]))

        if self.action_type == "qpos":
            arm = self._get(chunk, self.arm_key)[index].reshape(-1)
            if arm.shape[0] != 7:
                raise ValueError(
                    f"action {self.arm_key!r} must be 7-D for the Franka Panda arm "
                    f"(panda_joint1..7); got {arm.shape[0]}. UniVTAC's take_action "
                    f"'qpos' path calls set_arm(action[:-1]) over 7 joint ids."
                )
            return np.concatenate([arm, [gripper]]).astype(np.float32)

        pose = self._get(chunk, self.eef_key)[index].reshape(-1)
        if self.action_type == "ee":
            if pose.shape[0] == 9:
                quat = rot6d_to_quat_wxyz(pose[3:9])
                return np.concatenate([pose[:3], quat, [gripper]]).astype(np.float32)
            if pose.shape[0] == 7:  # already xyz + wxyz
                return np.concatenate([pose[:7], [gripper]]).astype(np.float32)
            raise ValueError(
                f"action {self.eef_key!r} must be 9-D ([xyz, rot6d]) or 7-D "
                f"([xyz, quat wxyz]) for action_type='ee'; got {pose.shape[0]}"
            )

        # delta_ee: 3 delta position + 3 delta euler + 1 delta gripper
        if pose.shape[0] < 6:
            raise ValueError(
                f"action {self.eef_key!r} must supply at least 6 dims "
                f"([dxyz, drpy]) for action_type='delta_ee'; got {pose.shape[0]}"
            )
        return np.concatenate([pose[:6], [gripper]]).astype(np.float32)


def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """Invert :func:`univtac_groot.obs_adapter.quat_wxyz_to_rot6d`.

    Gram-Schmidt orthonormalises the two 3-D columns into a rotation matrix
    (the standard 6-D -> SO(3) recovery) and reads off a ``(w, x, y, z)``
    quaternion using the numerically stable branch-on-trace method.
    """
    v = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a, b = v[:3], v[3:]

    n_a = np.linalg.norm(a)
    if n_a < 1e-8:
        raise ValueError("degenerate rot6d: first column has zero norm")
    c0 = a / n_a
    b_perp = b - np.dot(c0, b) * c0
    n_b = np.linalg.norm(b_perp)
    if n_b < 1e-8:
        raise ValueError("degenerate rot6d: columns are parallel")
    c1 = b_perp / n_b
    c2 = np.cross(c0, c1)
    R = np.stack([c0, c1, c2], axis=1)

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:  # canonical hemisphere
        quat = -quat
    return quat
