"""Tests for the observation path: state assembly, tactile encoding, stacking.

These run without Isaac Sim, torch or GR00T -- the point of keeping the adapters
pure numpy is that the contract can be checked on a login node before burning a
GPU allocation.
"""

from __future__ import annotations

import numpy as np
import pytest

from univtac_groot.arms import baseline_spec, tactile_spec
from univtac_groot.history import ObsHistory
from univtac_groot.obs_adapter import (
    ObsAdapter,
    as_uint8_hwc,
    encode_tactile_state,
    pool_2d,
    quat_wxyz_to_rot6d,
    reduce_marker_field,
    resize_nearest,
    split_embodiment,
)
from univtac_groot.rollout import batch_observation
from univtac_groot.spec import MAX_STATE_DIM, ObsSpec, StateField, TactileSpec


# --------------------------------------------------------------------------- #
# Fixtures mimicking UniVTAC's BaseTask._get_observations
# --------------------------------------------------------------------------- #


def make_observation(
    *,
    tactile_types: tuple[str, ...] = ("depth", "marker", "rgb", "rgb_marker"),
    cameras: tuple[str, ...] = ("head", "wrist"),
    sensor_names: tuple[str, ...] = ("left_tactile", "right_tactile"),
    depth_res: tuple[int, int] = (240, 320),
    n_markers: int = 64,
    marker_width: int = 4,
    seed: int = 0,
) -> dict:
    """Build a UniVTAC-shaped observation dict."""
    rng = np.random.default_rng(seed)
    obs: dict = {
        "observation": {
            cam: {"rgb": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)}
            for cam in cameras
        },
        "tactile": {},
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
    for name in sensor_names:
        sensor: dict = {}
        if "depth" in tactile_types:
            sensor["depth"] = rng.normal(0, 0.2, depth_res).astype(np.float32)
        if "marker" in tactile_types:
            sensor["marker"] = rng.normal(0, 3.0, (n_markers, marker_width)).astype(np.float32)
        if "rgb" in tactile_types:
            sensor["rgb"] = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        if "rgb_marker" in tactile_types:
            sensor["rgb_marker"] = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        obs["tactile"][name] = sensor
    return obs


# --------------------------------------------------------------------------- #
# Array helpers
# --------------------------------------------------------------------------- #


def test_pool_2d_averages_and_handles_indivisible_shapes():
    arr = np.arange(240 * 320, dtype=np.float32).reshape(240, 320)
    pooled = pool_2d(arr, (8, 6))
    assert pooled.shape == (8, 6)
    # Pooling is a mean, so it must stay inside the source range.
    assert arr.min() <= pooled.min() and pooled.max() <= arr.max()
    # A constant field pools to that constant.
    assert np.allclose(pool_2d(np.full((13, 7), 2.5, np.float32), (4, 3)), 2.5)


def test_pool_2d_upsamples_when_source_smaller_than_grid():
    pooled = pool_2d(np.array([[1.0, 3.0]], np.float32), (4, 4))
    assert pooled.shape == (4, 4)
    assert np.isfinite(pooled).all()


def test_pool_2d_rejects_bad_input():
    with pytest.raises(ValueError):
        pool_2d(np.zeros((4, 4)), (0, 3))
    with pytest.raises(ValueError):
        pool_2d(np.zeros((4, 4, 4)), (2, 2))


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
# Tactile encoding
# --------------------------------------------------------------------------- #


def test_depth_pool_produces_declared_width_and_is_normalized():
    spec = TactileSpec(mode="depth_pool", pool_grid=(8, 6))
    obs = make_observation()
    vec = encode_tactile_state(obs["tactile"], spec)
    assert vec.shape == (spec.total_state_dims(),) == (96,)
    assert vec.dtype == np.float32
    assert np.all(np.abs(vec) <= 1.0 + 1e-6)  # normalize=True -> [-1, 1]


def test_marker_mode_produces_declared_width():
    spec = TactileSpec(mode="marker", marker_pool=(4, 4))
    obs = make_observation()
    vec = encode_tactile_state(obs["tactile"], spec)
    assert vec.shape == (spec.total_state_dims(),) == (64,)  # 4*4*2 per sensor * 2
    assert np.all(np.abs(vec) <= 1.0)  # tanh-bounded


def test_reduce_marker_field_layouts():
    four = np.arange(8, dtype=np.float32).reshape(2, 4)
    assert np.allclose(reduce_marker_field(four, "auto"), [[2, 3], [6, 7]])
    assert reduce_marker_field(four, "xydxdy").shape == (2, 4)
    two = np.arange(4, dtype=np.float32).reshape(2, 2)
    assert np.allclose(reduce_marker_field(two, "auto"), two)
    with pytest.raises(ValueError):
        reduce_marker_field(np.zeros((2, 3), np.float32), "dxdy")


def test_tactile_modes_that_skip_the_state_return_empty():
    for mode in ("none", "video"):
        spec = TactileSpec(mode=mode)  # type: ignore[arg-type]
        assert encode_tactile_state(make_observation()["tactile"], spec).shape == (0,)


def test_missing_tactile_data_type_names_the_fix():
    spec = TactileSpec(mode="depth_pool")
    obs = make_observation(tactile_types=("rgb",))
    with pytest.raises(KeyError, match="observations.tactile"):
        encode_tactile_state(obs["tactile"], spec)


def test_gsmini_alias_is_accepted():
    """Older UniVTAC dumps name the sensors ``*_gsmini``."""
    spec = TactileSpec(mode="depth_pool", pool_grid=(4, 4))
    obs = make_observation(sensor_names=("left_gsmini", "right_gsmini"))
    assert encode_tactile_state(obs["tactile"], spec).shape == (32,)


def test_unknown_sensor_lists_available_names():
    spec = TactileSpec(mode="depth_pool", sensor_names=("nope",))
    with pytest.raises(KeyError, match="left_tactile"):
        encode_tactile_state(make_observation()["tactile"], spec)


def test_nan_depth_does_not_propagate():
    spec = TactileSpec(mode="depth_pool", pool_grid=(4, 4))
    obs = make_observation()
    obs["tactile"]["left_tactile"]["depth"][0, 0] = np.nan
    obs["tactile"]["left_tactile"]["depth"][1, 1] = np.inf
    assert np.isfinite(encode_tactile_state(obs["tactile"], spec)).all()


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


def test_tactile_adapter_adds_one_state_key_per_sensor():
    spec = tactile_spec(mode="depth_pool", pool_grid=(8, 6))
    adapter = ObsAdapter(spec)
    frame = adapter(make_observation(), "grasp")

    assert frame["state.tactile_left_tactile"].shape == (48,)
    assert frame["state.tactile_right_tactile"].shape == (48,)
    assert spec.state_dim == 17 + 96 == 113
    assert spec.state_dim <= MAX_STATE_DIM


def test_tactile_sensors_are_not_aliased_to_each_other():
    """A copy/paste bug that fed one sensor twice would be invisible in shapes."""
    spec = tactile_spec(mode="depth_pool", pool_grid=(4, 4))
    frame = ObsAdapter(spec)(make_observation(), "x")
    assert not np.allclose(
        frame["state.tactile_left_tactile"], frame["state.tactile_right_tactile"]
    )


def test_tactile_video_mode_adds_streams_not_state_dims():
    spec = tactile_spec(mode="video")
    frame = ObsAdapter(spec)(make_observation(), "x")
    assert "video.tactile_left" in frame and "video.tactile_right" in frame
    assert not any(k.startswith("state.tactile") for k in frame)
    assert spec.state_dim == 17


def test_baseline_and_tactile_share_the_proprioception_slice():
    """The arms must differ *only* by the tactile dims."""
    obs = make_observation()
    base = ObsAdapter(baseline_spec(video_keys={"head": "head", "wrist": "wrist"}))(obs, "x")
    tact = ObsAdapter(tactile_spec())(obs, "x")
    for key in ("state.eef_9d", "state.joint_position", "state.gripper_position"):
        assert np.allclose(base[key], tact[key])


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


def test_state_over_budget_is_rejected_with_the_remaining_budget():
    with pytest.raises(ValueError, match="max_state_dim=132"):
        tactile_spec(mode="depth_pool", pool_grid=(16, 8))  # 128/sensor -> 273-D


def test_declared_tactile_dims_must_match_the_encoder():
    with pytest.raises(ValueError, match="tactile state dims"):
        ObsSpec(
            video_keys={"head": "head"},
            state_fields=(StateField("tactile_x", "tactile", 10),),
            language_key="task",
            tactile=TactileSpec(mode="depth_pool", pool_grid=(8, 6), sensor_names=("a",)),
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
    spec = tactile_spec()
    adapter = ObsAdapter(spec)
    history = ObsHistory(spec.video_delta_indices, spec.state_delta_indices)
    stacked = history.reset(adapter(make_observation(), "insert"))
    batched = batch_observation(stacked)

    assert batched["video.head"].shape == (1, 1, 256, 256, 3)
    assert batched["video.head"].dtype == np.uint8
    assert batched["state.eef_9d"].shape == (1, 1, 9)
    assert batched["state.eef_9d"].dtype == np.float32
    assert batched[spec.language_key] == ["insert"]
