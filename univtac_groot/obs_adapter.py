"""Convert a UniVTAC observation dict into GR00T's flat sim observation format.

The output convention is the one ``Gr00tSimPolicyWrapper`` validates
(``gr00t/policy/gr00t_policy.py``)::

    {
      "video.<key>":  np.ndarray[np.uint8,   (T, H, W, 3)],
      "state.<key>":  np.ndarray[np.float32, (T, D)],
      "<language_key>": str,
    }

The batch axis is *not* added here; :mod:`univtac_groot.client` adds it right
before the wire call, because UniVTAC evaluates one environment at a time
(``scripts/eval_policy.py`` forces ``num_envs = 1``).

This module is deliberately pure numpy: it must import cleanly inside the Isaac
Lab process, which cannot afford to pull in GR00T's torch/transformers stack.
Torch tensors coming out of Isaac Lab are accepted and converted via
``.detach().cpu().numpy()``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .spec import (
    DEFAULT_GRIPPER_MAX_QPOS,
    N_EE_STATE,
    N_JOINT_STATE,
    ObsSpec,
)


# --------------------------------------------------------------------------- #
# Small array helpers
# --------------------------------------------------------------------------- #


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch tensors / lists / scalars to a numpy array without copying twice."""
    if isinstance(value, np.ndarray):
        return value
    # Duck-type torch.Tensor so this module never imports torch.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and not isinstance(value, (list, tuple)):
        return np.asarray(value.numpy())
    return np.asarray(value)


def as_uint8_hwc(image: Any) -> np.ndarray:
    """Coerce an image to contiguous ``(H, W, 3)`` uint8.

    Handles the shapes Isaac Lab hands back: ``HWC`` RGB, ``HWCA`` RGBA (alpha
    dropped), ``CHW`` (transposed), and float images in ``[0, 1]``.
    """
    arr = to_numpy(image)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D image, got shape {arr.shape}")

    # CHW -> HWC (a 3- or 4-channel leading axis with non-channel trailing axes).
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"expected 3 (RGB) or 4 (RGBA) channels, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        # Isaac Lab RGB output is already 0-255 float; only rescale true [0,1] images.
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(arr)


def resize_nearest(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an ``(H, W, C)`` array to ``(height, width)``.

    Implemented with index arithmetic rather than cv2/PIL so the Isaac Lab side
    needs no extra imaging dependency. GR00T's processor does its own
    (bilinear, aspect-preserving) preprocessing downstream; this resize only
    exists to cap the payload put on the wire.
    """
    target_h, target_w = int(size[0]), int(size[1])
    src_h, src_w = image.shape[:2]
    if (src_h, src_w) == (target_h, target_w):
        return image
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"invalid resize target {size}")
    rows = ((np.arange(target_h) + 0.5) * src_h / target_h).astype(np.intp)
    cols = ((np.arange(target_w) + 0.5) * src_w / target_w).astype(np.intp)
    np.clip(rows, 0, src_h - 1, out=rows)
    np.clip(cols, 0, src_w - 1, out=cols)
    return np.ascontiguousarray(image[rows[:, None], cols[None, :]])


def quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to the 6-D rotation representation.

    The 6-D form is the first two columns of the rotation matrix, which is what
    ``ActionFormat.XYZ_ROT6D`` (used by the ``oxe_droid_*`` embodiment) expects
    alongside a 3-D position to make up ``eef_9d``.

    Isaac Lab / UniVTAC poses are ``wxyz`` (``envs/utils/transforms.py`` builds
    ``Pose`` objects with ``q=[w, x, y, z]`` and ``math_utils.convert_quat(...,
    to="wxyz")``), hence the ordering in the name.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError(f"degenerate quaternion {q.tolist()}")
    w, x, y, z = q / norm
    # Columns 0 and 1 of the rotation matrix.
    col0 = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    col1 = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    return np.concatenate([col0, col1]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Proprioception
# --------------------------------------------------------------------------- #


def split_embodiment(
    embodiment: Mapping[str, Any],
    *,
    gripper_max_qpos: float,
    normalize_gripper: bool = True,
) -> dict[str, np.ndarray]:
    """Decompose UniVTAC proprioception into GR00T state slices.

    Args:
        embodiment: ``observation['embodiment']`` with ``'joint'`` (9,) and
            ``'ee'`` (7,) as produced by ``RobotManager.get_observations``.
        gripper_max_qpos: per-finger opening at full open, for normalising the
            gripper scalar to ``[0, 1]``.
        normalize_gripper: emit the gripper as a ``[0, 1]`` fraction (``1`` ==
            fully open) rather than raw metres.

    Returns:
        ``{"eef_9d": (9,), "joint_position": (7,), "gripper_position": (1,)}``.
        ``eef_9d`` is position (3) + rot6d (6), matching the ``eef_9d`` state key
        of the ``oxe_droid_relative_eef_relative_joint`` embodiment.
    """
    if "joint" not in embodiment or "ee" not in embodiment:
        raise KeyError(
            "observation['embodiment'] needs 'joint' and 'ee'; got "
            f"{sorted(embodiment)}. Set observations.embodiment: ['joint', 'ee'] "
            "in the UniVTAC task config."
        )

    joint = to_numpy(embodiment["joint"]).astype(np.float32).reshape(-1)
    ee = to_numpy(embodiment["ee"]).astype(np.float32).reshape(-1)

    if joint.shape[0] < N_JOINT_STATE:
        raise ValueError(
            f"embodiment['joint'] should be {N_JOINT_STATE}-D for the Franka Panda "
            f"(7 arm + 2 finger joints), got {joint.shape[0]}"
        )
    if ee.shape[0] != N_EE_STATE:
        raise ValueError(
            f"embodiment['ee'] should be {N_EE_STATE}-D (xyz + wxyz quaternion), "
            f"got {ee.shape[0]}"
        )

    variant = joint[:7]
    # Both fingers mirror one commanded opening; take finger 1 as the scalar.
    gripper = float(joint[7])
    if normalize_gripper:
        gripper = gripper / gripper_max_qpos if gripper_max_qpos > 0 else gripper
        gripper = float(np.clip(gripper, 0.0, 1.0))

    eef_9d = np.concatenate([ee[:3], quat_wxyz_to_rot6d(ee[3:7])]).astype(np.float32)

    return {
        "eef_9d": eef_9d,
        "joint_position": variant.astype(np.float32),
        "gripper_position": np.array([gripper], dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
# Full adapter
# --------------------------------------------------------------------------- #


class ObsAdapter:
    """Stateless translator from a UniVTAC observation dict to flat GR00T keys.

    Temporal stacking is handled by :class:`univtac_groot.history.ObsHistory`;
    this class only ever produces the single newest frame per key.
    """

    def __init__(
        self,
        spec: ObsSpec,
        *,
        gripper_max_qpos: float | None = None,
        normalize_gripper: bool = True,
    ) -> None:
        self.spec = spec
        self.gripper_max_qpos = (
            DEFAULT_GRIPPER_MAX_QPOS if gripper_max_qpos is None else float(gripper_max_qpos)
        )
        self.normalize_gripper = normalize_gripper
        self._slices = spec.field_slices()

    # -- pieces ------------------------------------------------------------
    def build_state(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Build ``{state key: (D,) float32}`` for one timestep."""
        parts = split_embodiment(
            observation.get("embodiment", {}),
            gripper_max_qpos=self.gripper_max_qpos,
            normalize_gripper=self.normalize_gripper,
        )
        out: dict[str, np.ndarray] = {}
        for f in self.spec.state_fields:
            value = np.asarray(parts[f.kind], dtype=np.float32).reshape(-1)
            if value.shape[0] != f.dim:
                raise ValueError(
                    f"state field {f.key!r} expects {f.dim} dims, produced {value.shape[0]}"
                )
            out[f.key] = value
        return out

    def build_video(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Build ``{video key: (H, W, 3) uint8}`` for one timestep."""
        cameras = observation.get("observation", {})

        out: dict[str, np.ndarray] = {}
        for gr00t_key, cam_name in self.spec.video_keys.items():
            if cam_name not in cameras:
                raise KeyError(
                    f"camera {cam_name!r} not in observation['observation'] "
                    f"(available: {sorted(cameras)}). Check the task's camera set and "
                    f"policy/task_settings.json camera_type."
                )
            if "rgb" not in cameras[cam_name]:
                raise KeyError(
                    f"camera {cam_name!r} has no 'rgb' entry; add 'rgb' to "
                    f"observations.camera in the UniVTAC task config."
                )
            img = as_uint8_hwc(cameras[cam_name]["rgb"])
            out[gr00t_key] = resize_nearest(img, self.spec.image_size)

        return out

    # -- whole frame -------------------------------------------------------
    def __call__(self, observation: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        """Translate one UniVTAC observation into flat, un-stacked GR00T keys."""
        frame: dict[str, Any] = {}
        for key, value in self.build_video(observation).items():
            frame[f"video.{key}"] = value
        for key, value in self.build_state(observation).items():
            frame[f"state.{key}"] = value
        frame[self.spec.language_key] = str(instruction)
        return frame

    # -- introspection -----------------------------------------------------
    def describe(self) -> str:
        """One-line human summary, used in eval logs."""
        fields = ", ".join(f"{f.key}({f.dim})" for f in self.spec.state_fields)
        return (
            f"video={sorted(self.spec.all_video_keys)} "
            f"state_dim={self.spec.state_dim} [{fields}]"
        )


def flat_keys(spec: ObsSpec) -> Sequence[str]:
    """All flat observation keys an :class:`ObsAdapter` will emit for ``spec``."""
    keys = [f"video.{k}" for k in spec.all_video_keys]
    keys += [f"state.{k}" for k in spec.state_keys]
    keys.append(spec.language_key)
    return tuple(keys)
