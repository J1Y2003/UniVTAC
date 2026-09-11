"""Tests for the observation path: state assembly and stacking.

These run without Isaac Sim, torch or GR00T -- the point of keeping the adapters
pure numpy is that the contract can be checked on a login node before burning a
GPU allocation.
"""

from __future__ import annotations

import numpy as np
import pytest

from univtac_groot.variants import baseline_spec, finetuned_baseline_spec
from univtac_groot.history import ObsHistory
from univtac_groot.obs_adapter import (
    ObsAdapter,
    as_uint8_hwc,
    quat_wxyz_to_rot6d,
    resize_nearest,
    split_embodiment,
)
from univtac_groot.rollout import batch_observation
from univtac_groot.spec import ObsSpec, StateField


# --------------------------------------------------------------------------- #
# Fixtures mimicking UniVTAC's BaseTask._get_observations
# --------------------------------------------------------------------------- #


def make_observation(
    *,
    cameras: tuple[str, ...] = ("head", "wrist"),
    seed: int = 0,
) -> dict:
    """Build a UniVTAC-shaped observation dict."""
    rng = np.random.default_rng(seed)
    obs: dict = {
        "observation": {
            cam: {"rgb": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)}
            for cam in cameras
        },
        "embodiment": {
            # 7 arm joints + 2 finger joints, as robot.data.joint_pos gives.
            "joint": np.array([0.1, -0.2, 0.3, -2.4, 0.5, 2.5, 0.7, 0.02, 0.02], np.float32),
            # xyz + quaternion (w, x, y, z)
            "ee": np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], np.float32),
        },
        "actor": {},
        "step": 3,
        "atom": {"id": 0, "tag": "move"},
    }
    return obs


# --------------------------------------------------------------------------- #
# Array helpers
# --------------------------------------------------------------------------- #


def test_resize_nearest_shape_and_passthrough():
    img = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    assert resize_nearest(img, (24, 32)).shape == (24, 32, 3)
    assert resize_nearest(img, (48, 64)) is img  # no copy when already correct


def test_as_uint8_hwc_variants():
    rng = np.random.default_rng(0)
    # RGBA -> RGB
    assert as_uint8_hwc(rng.integers(0, 256, (8, 8, 4), dtype=np.uint8)).shape == (8, 8, 3)
    # CHW -> HWC
    assert as_uint8_hwc(rng.integers(0, 256, (3, 8, 9), dtype=np.uint8)).shape == (8, 9, 3)
    # float [0, 1] is rescaled, not truncated to zero
    floats = np.full((4, 4, 3), 0.5, np.float32)
    assert as_uint8_hwc(floats).max() > 100
    # float 0-255 is passed through
    assert as_uint8_hwc(np.full((4, 4, 3), 200.0, np.float32)).max() == 200


def test_as_uint8_hwc_rejects_non_image():
    with pytest.raises(ValueError):
        as_uint8_hwc(np.zeros((8, 8)))
    with pytest.raises(ValueError):
        as_uint8_hwc(np.zeros((8, 8, 5)))


def test_quat_to_rot6d_identity_and_roundtrip():
    from univtac_groot.action_adapter import rot6d_to_quat_wxyz

    identity = quat_wxyz_to_rot6d(np.array([1.0, 0, 0, 0]))
    assert np.allclose(identity, [1, 0, 0, 0, 1, 0], atol=1e-6)

    # A 90-degree rotation about z survives the round trip.
    quat = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    recovered = rot6d_to_quat_wxyz(quat_wxyz_to_rot6d(quat))
    assert np.allclose(recovered, quat, atol=1e-5)


def test_quat_to_rot6d_rejects_degenerate():
    with pytest.raises(ValueError):
        quat_wxyz_to_rot6d(np.zeros(4))


# --------------------------------------------------------------------------- #
# Proprioception
# --------------------------------------------------------------------------- #


def test_split_embodiment_shapes_and_gripper_normalisation():
    obs = make_observation()
    parts = split_embodiment(obs["embodiment"], gripper_max_qpos=0.039)
    assert parts["eef_9d"].shape == (9,)
    assert parts["joint_position"].shape == (7,)
    assert parts["gripper_position"].shape == (1,)
    # joint[7] == 0.02 of a 0.039 max -> about half open
    assert 0.4 < float(parts["gripper_position"][0]) < 0.6


def test_split_embodiment_clips_gripper_to_unit_range():
    obs = make_observation()
    obs["embodiment"]["joint"][7] = 0.5  # far past the joint limit
    parts = split_embodiment(obs["embodiment"], gripper_max_qpos=0.039)
    assert float(parts["gripper_position"][0]) == 1.0


def test_split_embodiment_rejects_wrong_widths():
    with pytest.raises(ValueError, match="9-D"):
        split_embodiment(
            {"joint": np.zeros(6, np.float32), "ee": np.zeros(7, np.float32)},
            gripper_max_qpos=0.039,
        )
    with pytest.raises(ValueError, match="7-D"):
        split_embodiment(
            {"joint": np.zeros(9, np.float32), "ee": np.zeros(4, np.float32)},
            gripper_max_qpos=0.039,
        )


def test_split_embodiment_missing_keys_names_the_config_fix():
    with pytest.raises(KeyError, match="observations.embodiment"):
        split_embodiment({"joint": np.zeros(9, np.float32)}, gripper_max_qpos=0.039)


# --------------------------------------------------------------------------- #
# Full adapter
# --------------------------------------------------------------------------- #


def test_baseline_adapter_emits_expected_keys_and_dtypes():
    spec = baseline_spec()
    adapter = ObsAdapter(spec)
    frame = adapter(make_observation(), "insert the tube")

    assert set(frame) == {
        "video.exterior_image_1_left",
        "video.wrist_image_left",
        "state.eef_9d",
        "state.joint_position",
        "state.gripper_position",
        spec.language_key,
    }
    assert frame["video.exterior_image_1_left"].shape == (256, 256, 3)
    assert frame["video.exterior_image_1_left"].dtype == np.uint8
    assert frame["state.eef_9d"].dtype == np.float32
    assert frame[spec.language_key] == "insert the tube"


def test_missing_camera_names_the_available_ones():
    spec = baseline_spec(video_keys={"cam": "nonexistent"})
    with pytest.raises(KeyError, match="head"):
        ObsAdapter(spec)(make_observation(), "x")


def test_adapter_accepts_torch_like_tensors():
    """Isaac Lab hands back tensors; the adapter must not require numpy."""

    class FakeTensor:
        def __init__(self, array):
            self._array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    obs = make_observation()
    obs["embodiment"]["joint"] = FakeTensor(obs["embodiment"]["joint"])
    obs["observation"]["head"]["rgb"] = FakeTensor(obs["observation"]["head"]["rgb"])
    frame = ObsAdapter(baseline_spec())(obs, "x")
    assert frame["state.joint_position"].shape == (7,)
    assert frame["video.exterior_image_1_left"].dtype == np.uint8


# --------------------------------------------------------------------------- #
# Spec validation
# --------------------------------------------------------------------------- #


def test_state_over_budget_is_rejected():
    with pytest.raises(ValueError, match="max_state_dim=132"):
        ObsSpec(
            video_keys={"head": "head"},
            state_fields=(StateField("wide", "eef_9d", 133),),
            language_key="task",
        )


def test_duplicate_state_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        ObsSpec(
            video_keys={"head": "head"},
            state_fields=(
                StateField("q", "joint_position", 7),
                StateField("q", "joint_position", 7),
            ),
            language_key="task",
        )


def test_delta_indices_are_validated():
    with pytest.raises(ValueError, match="must end at 0"):
        baseline_spec().__class__(
            video_keys={"head": "head"},
            state_fields=(StateField("q", "joint_position", 7),),
            language_key="task",
            video_delta_indices=(-2, -1),
        )
    with pytest.raises(ValueError, match="non-positive"):
        ObsSpec(
            video_keys={"head": "head"},
            state_fields=(StateField("q", "joint_position", 7),),
            language_key="task",
            video_delta_indices=(1, 0),
        )
    with pytest.raises(ValueError, match="evenly spaced"):
        ObsSpec(
            video_keys={"head": "head"},
            state_fields=(StateField("q", "joint_position", 7),),
            language_key="task",
            video_delta_indices=(-5, -1, 0),
        )


# --------------------------------------------------------------------------- #
# History stacking
# --------------------------------------------------------------------------- #


def _frame(value: int) -> dict:
    return {
        "video.cam": np.full((2, 2, 3), value, np.uint8),
        "state.q": np.full((3,), float(value), np.float32),
        "task": "go",
    }


def test_history_single_frame_horizon():
    history = ObsHistory((0,), (0,))
    stacked = history.reset(_frame(0))
    assert stacked["video.cam"].shape == (1, 2, 2, 3)
    assert stacked["state.q"].shape == (1, 3)
    assert stacked["task"] == "go"


def test_history_reset_pads_by_repeating_the_first_frame():
    """The DROID tag wants (-15, 0); at t=0 there is no 15-steps-ago frame."""
    history = ObsHistory((-15, 0), (0,))
    stacked = history.reset(_frame(7))
    assert stacked["video.cam"].shape == (2, 2, 2, 3)
    # Both slots are the first frame, not zeros.
    assert np.all(stacked["video.cam"] == 7)


def test_history_offsets_select_the_right_frames():
    history = ObsHistory((-2, 0), (0,))
    history.reset(_frame(0))
    for value in (1, 2, 3, 4):
        history.append(_frame(value))
    stacked = history.observe()
    # Newest is 4; two steps earlier is 2. Order is oldest-first.
    assert stacked["video.cam"][0][0, 0, 0] == 2
    assert stacked["video.cam"][1][0, 0, 0] == 4


def test_history_matches_multistep_wrapper_indexing():
    """``obs[d - 1]`` for offset d: offset 0 must be the last element."""
    history = ObsHistory((-1, 0), (-1, 0))
    history.reset(_frame(10))
    history.append(_frame(11))
    stacked = history.observe()
    assert list(stacked["state.q"][:, 0]) == [10.0, 11.0]


def test_history_drops_state_for_a_vision_only_policy():
    history = ObsHistory((0,), None)
    stacked = history.reset(_frame(0))
    assert "video.cam" in stacked
    assert "state.q" not in stacked


def test_history_requires_reset_first():
    history = ObsHistory((0,), (0,))
    with pytest.raises(RuntimeError, match="reset"):
        history.observe()
    with pytest.raises(RuntimeError, match="reset"):
        history.append(_frame(0))


# --------------------------------------------------------------------------- #
# Wire shape
# --------------------------------------------------------------------------- #


def test_batch_observation_matches_gr00t_validator_shapes():
    """``Gr00tSimPolicyWrapper.check_observation`` wants (B,T,H,W,C)/(B,T,D)."""
    spec = finetuned_baseline_spec()
    adapter = ObsAdapter(spec)
    history = ObsHistory(spec.video_delta_indices, spec.state_delta_indices)
    stacked = history.reset(adapter(make_observation(), "insert"))
    batched = batch_observation(stacked)

    assert batched["video.head"].shape == (1, 1, 256, 256, 3)
    assert batched["video.head"].dtype == np.uint8
    assert batched["state.eef_9d"].shape == (1, 1, 9)
    assert batched["state.eef_9d"].dtype == np.float32
    assert batched[spec.language_key] == ["insert"]
