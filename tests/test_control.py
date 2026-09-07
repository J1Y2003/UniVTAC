"""Tests for the action path: chunk slicing, receding horizon, metrics, rollout.

Also covers the rollout loop against a fake environment and a fake policy, so
the seeding, error handling and inference-count accounting are verified without
Isaac Sim or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from univtac_groot.action_adapter import (
    ActionAdapter,
    GripperConvention,
    rot6d_to_quat_wxyz,
)
from univtac_groot.metrics import (
    EpisodeResult,
    ResultWriter,
    compare,
    read_jsonl,
    summarize,
    wilson_interval,
)
from univtac_groot.receding_horizon import (
    RecedingHorizonController,
    resolve_horizons,
)
from univtac_groot.rollout import RolloutConfig, evaluate, run_episode, seed_sequence


# --------------------------------------------------------------------------- #
# Gripper convention
# --------------------------------------------------------------------------- #


def test_gripper_inverted_mapping():
    """DROID-style: 1.0 == closed; UniVTAC: larger metres == more open."""
    conv = GripperConvention(max_qpos=0.039, invert=True)
    assert conv.to_univtac(0.0) == pytest.approx(0.039)  # open
    assert conv.to_univtac(1.0) == pytest.approx(0.0)    # closed
    assert conv.to_univtac(0.5) == pytest.approx(0.0195)


def test_gripper_non_inverted_mapping():
    conv = GripperConvention(max_qpos=0.039, invert=False)
    assert conv.to_univtac(1.0) == pytest.approx(0.039)
    assert conv.to_univtac(0.0) == pytest.approx(0.0)


def test_gripper_clips_out_of_range_predictions():
    """A mis-scaled checkpoint must not drive the fingers past their limits."""
    conv = GripperConvention(max_qpos=0.039, invert=False)
    assert conv.to_univtac(5.0) == pytest.approx(0.039)
    assert conv.to_univtac(-3.0) == pytest.approx(0.0)


def test_gripper_roundtrip():
    for invert in (True, False):
        conv = GripperConvention(max_qpos=0.039, invert=invert)
        for qpos in (0.0, 0.01, 0.02, 0.039):
            assert conv.to_univtac(conv.from_univtac(qpos)) == pytest.approx(qpos, abs=1e-9)


# --------------------------------------------------------------------------- #
# Action adapter
# --------------------------------------------------------------------------- #


def make_chunk(horizon: int = 16, *, prefix: str = "action.") -> dict[str, np.ndarray]:
    """A GR00T-shaped (B, T, D) action chunk for the qpos keys."""
    arm = np.tile(np.arange(7, dtype=np.float32), (horizon, 1))
    arm += np.arange(horizon, dtype=np.float32)[:, None]
    gripper = np.linspace(0, 1, horizon, dtype=np.float32)[:, None]
    return {
        f"{prefix}joint_position": arm[None, ...],
        f"{prefix}gripper_position": gripper[None, ...],
    }


def test_qpos_action_is_eight_dimensional():
    """UniVTAC's take_action('qpos') expects 7 arm joints + 1 gripper."""
    adapter = ActionAdapter(action_type="qpos")
    action = adapter.step(make_chunk(), 0)
    assert action.shape == (8,)
    assert action.dtype == np.float32
    assert np.allclose(action[:7], np.arange(7))


def test_chunk_conversion_preserves_horizon_and_order():
    adapter = ActionAdapter(action_type="qpos")
    actions = adapter.to_univtac(make_chunk(horizon=16))
    assert actions.shape == (16, 8)
    # Step t has arm values offset by t.
    assert np.allclose(actions[3][:7], np.arange(7) + 3)


def test_adapter_accepts_prefixed_and_bare_keys():
    adapter = ActionAdapter(action_type="qpos")
    assert adapter.chunk_length(make_chunk(prefix="action.")) == 16
    assert adapter.chunk_length(make_chunk(prefix="")) == 16


def test_adapter_accepts_unbatched_chunks():
    chunk = {k: v[0] for k, v in make_chunk(horizon=4).items()}
    assert ActionAdapter().to_univtac(chunk).shape == (4, 8)


def test_missing_action_key_lists_what_arrived():
    adapter = ActionAdapter(action_type="qpos", arm_key="nope")
    with pytest.raises(KeyError, match="ModalityConfig"):
        adapter.step(make_chunk(), 0)


def test_disagreeing_horizons_are_rejected():
    chunk = make_chunk(horizon=16)
    chunk["action.gripper_position"] = chunk["action.gripper_position"][:, :8]
    with pytest.raises(ValueError, match="disagree"):
        ActionAdapter().chunk_length(chunk)


def test_wrong_arm_width_names_the_constraint():
    chunk = make_chunk()
    chunk["action.joint_position"] = chunk["action.joint_position"][:, :, :5]
    with pytest.raises(ValueError, match="7-D"):
        ActionAdapter().step(chunk, 0)


def test_batch_size_greater_than_one_is_rejected():
    chunk = make_chunk()
    chunk["action.joint_position"] = np.repeat(chunk["action.joint_position"], 2, axis=0)
    with pytest.raises(ValueError, match="batch size 1"):
        ActionAdapter().step(chunk, 0)


def test_ee_action_converts_rot6d_to_quaternion():
    adapter = ActionAdapter(action_type="ee")
    pose = np.zeros((1, 4, 9), np.float32)
    pose[..., :3] = [0.4, 0.1, 0.3]
    pose[..., 3:9] = [1, 0, 0, 0, 1, 0]  # identity rotation
    chunk = {
        "action.eef_9d": pose,
        "action.gripper_position": np.zeros((1, 4, 1), np.float32),
    }
    action = adapter.step(chunk, 0)
    assert action.shape == (8,)  # pos(3) + quat(4) + gripper(1)
    assert np.allclose(action[3:7], [1, 0, 0, 0], atol=1e-6)


def test_delta_ee_action_is_seven_dimensional():
    adapter = ActionAdapter(action_type="delta_ee")
    chunk = {
        "action.eef_9d": np.zeros((1, 2, 6), np.float32),
        "action.gripper_position": np.zeros((1, 2, 1), np.float32),
    }
    assert adapter.step(chunk, 0).shape == (7,)


def test_rot6d_rejects_degenerate_input():
    with pytest.raises(ValueError, match="zero norm"):
        rot6d_to_quat_wxyz(np.array([0, 0, 0, 1, 0, 0], np.float64))
    with pytest.raises(ValueError, match="parallel"):
        rot6d_to_quat_wxyz(np.array([1, 0, 0, 2, 0, 0], np.float64))


def test_rot6d_roundtrip_over_many_rotations():
    from univtac_groot.obs_adapter import quat_wxyz_to_rot6d

    rng = np.random.default_rng(7)
    for _ in range(64):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        if quat[0] < 0:
            quat = -quat
        assert np.allclose(rot6d_to_quat_wxyz(quat_wxyz_to_rot6d(quat)), quat, atol=1e-5)


# --------------------------------------------------------------------------- #
# Receding horizon
# --------------------------------------------------------------------------- #


def test_controller_reuses_the_chunk_until_the_horizon_is_spent():
    controller = RecedingHorizonController(execution_horizon=4)
    calls = {"n": 0}

    def infer():
        calls["n"] += 1
        return np.arange(16 * 2, dtype=np.float32).reshape(16, 2)

    for _ in range(4):
        controller.next_action(infer)
    assert calls["n"] == 1          # one inference for four actions
    controller.next_action(infer)
    assert calls["n"] == 2          # horizon spent -> re-plan
    assert controller.stats.steps == 5
    assert controller.stats.inferences == 2


def test_controller_full_chunk_when_horizon_is_none():
    controller = RecedingHorizonController(execution_horizon=None)
    chunk = np.zeros((40, 8), np.float32)
    for _ in range(40):
        controller.next_action(lambda: chunk)
    assert controller.stats.inferences == 1
    assert controller.stats.steps == 40


def test_controller_serves_actions_in_order():
    controller = RecedingHorizonController(execution_horizon=3)
    chunk = np.arange(9, dtype=np.float32).reshape(3, 3)
    served = [controller.next_action(lambda: chunk) for _ in range(3)]
    assert np.allclose(np.stack(served), chunk)


def test_controller_rejects_bad_horizons():
    with pytest.raises(ValueError, match=">= 1"):
        RecedingHorizonController(execution_horizon=0)
    with pytest.raises(ValueError, match="exceeds"):
        RecedingHorizonController(execution_horizon=64, action_horizon=40)


def test_controller_rejects_unstable_chunk_length():
    controller = RecedingHorizonController(execution_horizon=1)
    controller.submit(np.zeros((16, 2), np.float32))
    with pytest.raises(ValueError, match="stable"):
        controller.submit(np.zeros((8, 2), np.float32))


def test_controller_rejects_malformed_chunks():
    controller = RecedingHorizonController()
    with pytest.raises(ValueError, match=r"\(T, D\)"):
        controller.submit(np.zeros((4,), np.float32))
    with pytest.raises(ValueError, match="empty"):
        controller.submit(np.zeros((0, 2), np.float32))


def test_controller_reset_clears_cache_and_stats():
    controller = RecedingHorizonController(execution_horizon=2)
    controller.next_action(lambda: np.zeros((4, 2), np.float32))
    controller.reset()
    assert controller.needs_inference
    assert controller.stats.steps == 0


def test_controller_raises_when_inference_returns_nothing():
    controller = RecedingHorizonController()
    with pytest.raises(RuntimeError, match="returned None"):
        controller.next_action(lambda: None)


# --------------------------------------------------------------------------- #
# Horizon resolution from a modality config
# --------------------------------------------------------------------------- #


def modality(action_horizon=40, video_deltas=(-15, 0), state_deltas=(0,)):
    """Dict-shaped modality config, as it arrives over msgpack."""
    return {
        "action": {"delta_indices": list(range(action_horizon))},
        "video": {"delta_indices": list(video_deltas)},
        "state": {"delta_indices": list(state_deltas)},
    }


def test_resolve_horizons_defaults_to_the_full_chunk():
    resolved = resolve_horizons(modality())
    assert resolved["action_horizon"] == 40
    assert resolved["execution_horizon"] == 40
    assert resolved["video_delta_indices"] == (-15, 0)


def test_resolve_horizons_honours_an_explicit_execution_horizon():
    assert resolve_horizons(modality(), execution_horizon=8)["execution_horizon"] == 8


def test_resolve_horizons_rejects_an_oversized_execution_horizon():
    with pytest.raises(ValueError, match="1 <= execution_horizon"):
        resolve_horizons(modality(action_horizon=16), execution_horizon=32)


def test_resolve_horizons_rejects_a_sparse_action_window():
    """A non-contiguous window would silently execute the wrong rows."""
    cfg = modality()
    cfg["action"]["delta_indices"] = [0, 2, 4, 6]
    with pytest.raises(ValueError, match="contiguous"):
        resolve_horizons(cfg)


def test_resolve_horizons_handles_a_vision_only_policy():
    cfg = modality()
    cfg["state"]["delta_indices"] = []
    assert resolve_horizons(cfg)["state_delta_indices"] is None


def test_resolve_horizons_accepts_objects_with_attributes():
    class Cfg:
        def __init__(self, deltas):
            self.delta_indices = deltas

    resolved = resolve_horizons(
        {"action": Cfg(list(range(16))), "video": Cfg([0]), "state": Cfg([0])}
    )
    assert resolved["action_horizon"] == 16


def test_resolve_horizons_requires_an_action_entry():
    with pytest.raises(ValueError, match="no 'action' entry"):
        resolve_horizons({"video": {"delta_indices": [0]}})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def result(seed: int, success: bool, *, steps: int = 20, error: str | None = None):
    return EpisodeResult(
        seed=seed,
        success=success,
        reward=1.0 if success else 0.0,
        steps=steps,
        truncated=not success,
        early_stop=False,
        error=error,
    )


def test_summarize_rate_and_mean_reward_agree():
    """Sparse success: mean reward is the success rate by construction."""
    summary = summarize([result(i, i < 3) for i in range(10)])
    assert summary["episodes_scored"] == 10
    assert summary["successes"] == 3
    assert summary["success_rate"] == pytest.approx(0.3)
    assert summary["mean_reward"] == pytest.approx(0.3)


def test_errored_episodes_are_excluded_from_the_rate():
    """Matches UniVTAC's evaluator, which decrements test_num on an exception."""
    summary = summarize([result(0, True), result(1, False), result(2, False, error="boom")])
    assert summary["episodes_scored"] == 2
    assert summary["episodes_errored"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)


def test_summarize_of_nothing_does_not_divide_by_zero():
    summary = summarize([])
    assert summary["success_rate"] == 0.0
    assert "success_rate_ci95" not in summary


def test_wilson_interval_stays_inside_the_unit_range():
    for successes, total in ((0, 10), (10, 10), (1, 3), (25, 50)):
        lo, hi = wilson_interval(successes, total)
        assert 0.0 <= lo <= hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_result_writer_roundtrips_jsonl_and_writes_a_summary(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultWriter(path, {"arm": "tactile"}) as writer:
        writer.add(result(1, True))
        writer.add(result(2, False))

    # One line per episode, flushed as it completed.
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
    recovered = list(read_jsonl(path))
    assert [r.seed for r in recovered] == [1, 2]
    assert recovered[0].success is True

    summary = (path.with_suffix(".summary.json")).read_text(encoding="utf-8")
    assert '"arm": "tactile"' in summary


def test_summarize_jsonl_recovers_a_partial_run(tmp_path):
    """A job killed at its walltime still leaves usable results."""
    from univtac_groot.metrics import summarize_jsonl

    path = tmp_path / "partial.jsonl"
    with ResultWriter(path) as writer:
        for i in range(4):
            writer.add(result(i, i % 2 == 0))
    assert summarize_jsonl(path)["successes"] == 2


def test_compare_reports_delta_and_interval_overlap():
    low = summarize([result(i, False) for i in range(40)])
    high = summarize([result(i, True) for i in range(40)])
    verdict = compare(low, high)
    assert verdict["delta_pct_points"] == pytest.approx(100.0)
    assert verdict["ci95_overlap"] is False

    same = compare(low, low)
    assert same["delta"] == 0.0
    assert same["ci95_overlap"] is True


# --------------------------------------------------------------------------- #
# Rollout loop against fakes
# --------------------------------------------------------------------------- #


class FakeEnv:
    """Minimal stand-in for UniVTACGr00tEnv."""

    max_steps = 12

    def __init__(self, *, succeed_at: int | None = None, raise_on_seed: set[int] | None = None):
        self.succeed_at = succeed_at
        self.raise_on_seed = raise_on_seed or set()
        self.action_adapter = ActionAdapter()
        self.steps = 0
        self.seeds: list[int] = []

    def reset(self, *, seed=None, options=None):
        if seed in self.raise_on_seed:
            raise RuntimeError(f"bad seed {seed}")
        self.seeds.append(seed)
        self.steps = 0
        return {"state.q": np.zeros((1, 3), np.float32)}, {"instruction": "do it"}

    def step(self, action):
        assert action.shape == (8,), action.shape
        self.steps += 1
        success = self.succeed_at is not None and self.steps >= self.succeed_at
        info = {
            "success": success,
            "take_action_cnt": self.steps,
            "early_stop": False,
        }
        truncated = self.steps >= self.max_steps
        return {"state.q": np.zeros((1, 3), np.float32)}, float(success), success, truncated, info

    def close(self):
        pass


class FakePolicy:
    """Returns a fixed 16-step chunk and counts calls."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def get_action(self, observation):
        self.calls += 1
        return make_chunk(horizon=16), {}

    def reset(self, **options):
        self.resets += 1
        return {}


def test_run_episode_records_success_and_inference_count():
    env, policy = FakeEnv(succeed_at=5), FakePolicy()
    outcome = run_episode(
        env, policy, seed=42, config=RolloutConfig(execution_horizon=4)
    )
    assert outcome.success is True
    assert outcome.steps == 5
    assert outcome.error is None
    # 5 actions at an execution horizon of 4 -> 2 chunk requests.
    assert outcome.inferences == 2 == policy.calls
    assert policy.resets == 1


def test_run_episode_truncates_at_the_step_cap():
    outcome = run_episode(
        FakeEnv(succeed_at=None), FakePolicy(), seed=1,
        config=RolloutConfig(execution_horizon=16),
    )
    assert outcome.success is False
    assert outcome.truncated is True
    assert outcome.steps == FakeEnv.max_steps


def test_run_episode_captures_exceptions_instead_of_propagating():
    outcome = run_episode(
        FakeEnv(raise_on_seed={7}), FakePolicy(), seed=7, config=RolloutConfig()
    )
    assert outcome.error is not None
    assert "bad seed 7" in outcome.error
    assert outcome.success is False


def test_evaluate_scores_the_requested_number_of_episodes(tmp_path):
    env, policy = FakeEnv(succeed_at=3), FakePolicy()
    writer = ResultWriter(tmp_path / "r.jsonl", {"arm": "baseline"})
    summary = evaluate(
        env, policy,
        config=RolloutConfig(num_episodes=5, start_seed=100, execution_horizon=8),
        writer=writer,
        log=lambda _msg: None,
    )
    writer.close()
    assert summary["episodes_scored"] == 5
    assert summary["success_rate"] == 1.0
    assert env.seeds == [100, 101, 102, 103, 104]


def test_evaluate_walks_past_a_failing_seed_without_losing_budget(tmp_path):
    """An errored episode must not silently shrink the episode count."""
    env, policy = FakeEnv(succeed_at=2, raise_on_seed={101}), FakePolicy()
    summary = evaluate(
        env, policy,
        config=RolloutConfig(num_episodes=3, start_seed=100),
        writer=ResultWriter(tmp_path / "r.jsonl"),
        log=lambda _msg: None,
    )
    assert summary["episodes_scored"] == 3
    assert summary["episodes_errored"] == 1
    # Seed 101 raised before reset() recorded it, and the budget advanced to the
    # next seed instead of being consumed.
    assert env.seeds == [100, 102, 103]


def test_evaluate_aborts_after_repeated_failures(tmp_path):
    """A broken checkpoint should fail fast, not burn the whole allocation."""
    env = FakeEnv(raise_on_seed=set(range(100, 200)))
    with pytest.raises(RuntimeError, match="consecutive"):
        evaluate(
            env, FakePolicy(),
            config=RolloutConfig(num_episodes=10, start_seed=100, max_consecutive_errors=3),
            writer=ResultWriter(tmp_path / "r.jsonl"),
            log=lambda _msg: None,
        )


def test_seed_sequence_defaults_match_univtac():
    """UniVTAC uses 1_000_000 * (1 + seed) as the start seed."""
    assert next(iter(seed_sequence(RolloutConfig(seed_offset=0)))) == 1_000_000
    assert next(iter(seed_sequence(RolloutConfig(seed_offset=2)))) == 3_000_000


def test_seed_sequence_respects_max_seed():
    seeds = list(seed_sequence(RolloutConfig(start_seed=5, max_seed=8)))
    assert seeds == [5, 6, 7, 8]


class PlanFailEnv(FakeEnv):
    """Env whose scripted pre-move fails to plan on the given seeds."""

    def __init__(self, *, plan_fail_seeds: set[int], succeed_at: int = 2):
        super().__init__(succeed_at=succeed_at)
        self.plan_fail_seeds = plan_fail_seeds

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["plan_success"] = seed not in self.plan_fail_seeds
        return obs, info


def test_unplannable_seed_is_skipped_not_scored():
    """A cuRobo pre-move failure must not be blamed on the policy."""
    outcome = run_episode(
        PlanFailEnv(plan_fail_seeds={7}), FakePolicy(), seed=7,
        config=RolloutConfig(), log=lambda _m: None,
    )
    assert outcome.skipped == "plan_failure"
    assert outcome.error is None
    assert outcome.steps == 0


def test_skipped_episodes_are_excluded_from_the_success_rate():
    scored = [result(1, True), result(2, False)]
    skipped = EpisodeResult(seed=3, success=False, reward=0.0, steps=0,
                            truncated=False, early_stop=False, skipped="plan_failure")
    summary = summarize([*scored, skipped])
    assert summary["episodes_scored"] == 2
    assert summary["episodes_skipped"] == 1
    assert summary["episodes_errored"] == 0
    assert summary["success_rate"] == pytest.approx(0.5)   # not 1/3
    assert summary["skip_reasons"] == {"plan_failure": 1}


def test_evaluate_walks_past_unplannable_seeds_to_fill_the_budget(tmp_path):
    env = PlanFailEnv(plan_fail_seeds={100, 101, 103}, succeed_at=2)
    summary = evaluate(
        env, FakePolicy(),
        config=RolloutConfig(num_episodes=2, start_seed=100),
        writer=ResultWriter(tmp_path / "r.jsonl"),
        log=lambda _m: None,
    )
    # 100, 101 and 103 unusable -> 102 and 104 are the two scored episodes.
    assert summary["episodes_scored"] == 2
    assert summary["episodes_skipped"] == 3
    assert summary["success_rate"] == 1.0


def test_evaluate_aborts_when_no_seed_can_be_planned(tmp_path):
    """A task whose start poses never plan should fail fast, not spin."""
    env = PlanFailEnv(plan_fail_seeds=set(range(100, 400)))
    with pytest.raises(RuntimeError, match="consecutive seeds were unusable"):
        evaluate(
            env, FakePolicy(),
            config=RolloutConfig(num_episodes=5, start_seed=100, max_consecutive_skips=10),
            writer=ResultWriter(tmp_path / "r.jsonl"),
            log=lambda _m: None,
        )


def test_plan_failures_can_be_scored_when_explicitly_requested():
    outcome = run_episode(
        PlanFailEnv(plan_fail_seeds={7}), FakePolicy(), seed=7,
        config=RolloutConfig(skip_on_plan_failure=False), log=lambda _m: None,
    )
    assert outcome.skipped is None
