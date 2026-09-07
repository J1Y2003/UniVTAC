"""The evaluation loop: episodes, seeding, receding-horizon control, logging.

This is the piece UniVTAC's own evaluator and the standalone driver share. It
deliberately knows nothing about Isaac Lab: it takes an object with the
:class:`~univtac_groot.env_wrapper.UniVTACGr00tEnv` surface and a policy client
with ``get_action``/``reset``, so it is testable against fakes.

Seeding follows UniVTAC's convention in ``scripts/eval_policy.py``: episodes walk
consecutive integer seeds starting at ``1_000_000 * (1 + seed_offset)`` unless a
start seed is given, and a seed is retried-forward (never reused) so that a
failed reset does not silently shrink the episode count.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import traceback
from typing import Any, Callable, Iterator, Mapping, Protocol

import numpy as np

from .action_adapter import ActionAdapter
from .metrics import EpisodeResult, ResultWriter
from .receding_horizon import RecedingHorizonController


class PolicyTransport(Protocol):
    """The slice of :class:`univtac_groot.client.Gr00tClient` a rollout needs."""

    def get_action(
        self, observation: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict]: ...

    def reset(self, **options: Any) -> dict: ...


@dataclass
class RolloutConfig:
    """Knobs for :func:`run_episode` / :func:`evaluate`.

    Attributes:
        num_episodes: episodes to score (errored episodes are retried on the
            next seed and do not consume the budget, as in UniVTAC's evaluator).
        start_seed: first seed. ``None`` derives it from ``seed_offset``.
        seed_offset: used as ``1_000_000 * (1 + seed_offset)`` when
            ``start_seed`` is ``None``, matching UniVTAC's ``deploy_config['seed']``.
        max_seed: stop once seeds pass this, even if short of ``num_episodes``.
            ``None``/``-1`` means unbounded.
        execution_horizon: actions executed per chunk before re-planning.
            ``None`` executes the full chunk.
        max_steps: per-episode action cap. ``None`` uses the env's ``max_steps``.
        max_consecutive_errors: abort the run after this many episodes raise
            back to back, so a broken checkpoint fails fast instead of burning
            the whole SLURM allocation.
        stop_on_success: end the episode as soon as the task reports success.
    """

    num_episodes: int = 50
    start_seed: int | None = None
    seed_offset: int = 0
    max_seed: int | None = None
    execution_horizon: int | None = None
    max_steps: int | None = None
    max_consecutive_errors: int = 5
    stop_on_success: bool = True

    def first_seed(self) -> int:
        """Resolve the starting seed."""
        if self.start_seed is not None and self.start_seed >= 0:
            return int(self.start_seed)
        return 1_000_000 * (1 + int(self.seed_offset))


def _log(message: str, sink: Callable[[str], None] | None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    (sink or print)(line)


def run_episode(
    env: Any,
    policy: PolicyTransport,
    *,
    seed: int,
    action_adapter: ActionAdapter | None = None,
    config: RolloutConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> EpisodeResult:
    """Run one episode and return its result.

    The chunk is requested lazily: :class:`RecedingHorizonController` only calls
    the policy when its cache is exhausted, so a full-chunk execution horizon
    costs one forward pass per ``action_horizon`` environment steps.

    Exceptions inside the episode are captured into
    :attr:`EpisodeResult.error` rather than propagated, so one bad seed cannot
    end a batch job; the caller decides whether to keep going.
    """
    config = config or RolloutConfig()
    adapter = action_adapter or getattr(env, "action_adapter", None) or ActionAdapter()
    controller = RecedingHorizonController(execution_horizon=config.execution_horizon)

    started = time.time()
    reward_total = 0.0
    success = truncated = early_stop = False
    steps = 0
    instruction = ""

    try:
        obs, info = env.reset(seed=seed)
        instruction = str(info.get("instruction", getattr(env, "instruction", "")))
        policy.reset()
        controller.reset()

        max_steps = config.max_steps or getattr(env, "max_steps", 300)
        latest = obs

        def infer(observation) -> np.ndarray:
            """Ask the policy for a chunk and flatten it to per-step vectors."""
            t0 = time.perf_counter()
            chunk, _ = policy.get_action(observation)
            controller.stats.inference_seconds += time.perf_counter() - t0
            return adapter.to_univtac(chunk)

        while steps < max_steps:
            # ``latest`` is bound as a default argument rather than captured, so
            # the callback can only ever see this iteration's observation.
            action = controller.next_action(lambda obs=latest: infer(obs))
            latest, reward, terminated, truncated, step_info = env.step(action)
            steps = int(step_info.get("take_action_cnt", steps + 1))
            reward_total = max(reward_total, float(reward))
            success = bool(step_info.get("success", success))
            early_stop = bool(step_info.get("early_stop", early_stop))

            if success and config.stop_on_success:
                break
            if terminated or truncated:
                break

    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        _log(f"seed {seed} raised: {exc}", log)
        return EpisodeResult(
            seed=seed,
            success=False,
            reward=0.0,
            steps=steps,
            truncated=False,
            early_stop=False,
            instruction=instruction,
            wall_seconds=round(time.time() - started, 2),
            inferences=controller.stats.inferences,
            inference_seconds=round(controller.stats.inference_seconds, 3),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
        )

    return EpisodeResult(
        seed=seed,
        success=success,
        reward=reward_total,
        steps=steps,
        truncated=bool(truncated and not success),
        early_stop=early_stop,
        instruction=instruction,
        wall_seconds=round(time.time() - started, 2),
        inferences=controller.stats.inferences,
        inference_seconds=round(controller.stats.inference_seconds, 3),
        extra=controller.stats.as_dict(),
    )


def seed_sequence(config: RolloutConfig) -> Iterator[int]:
    """Yield consecutive seeds, honouring ``max_seed``."""
    seed = config.first_seed()
    limit = config.max_seed
    while limit is None or limit < 0 or seed <= limit:
        yield seed
        seed += 1


def evaluate(
    env: Any,
    policy: PolicyTransport,
    *,
    config: RolloutConfig | None = None,
    writer: ResultWriter | None = None,
    action_adapter: ActionAdapter | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run ``config.num_episodes`` episodes and return the summary dict.

    Each episode is written to ``writer`` as it finishes, so a job killed at its
    walltime still leaves a re-aggregatable results file.

    Raises:
        RuntimeError: ``config.max_consecutive_errors`` episodes raised in a
            row, which indicates a misconfiguration (wrong embodiment tag,
            missing tactile data type) rather than a hard seed.
    """
    config = config or RolloutConfig()
    scored = 0
    consecutive_errors = 0
    results: list[EpisodeResult] = []

    for seed in seed_sequence(config):
        if scored >= config.num_episodes:
            break

        result = run_episode(
            env,
            policy,
            seed=seed,
            action_adapter=action_adapter,
            config=config,
            log=log,
        )
        results.append(result)
        if writer is not None:
            writer.add(result)

        if result.error is not None:
            consecutive_errors += 1
            if consecutive_errors >= config.max_consecutive_errors:
                raise RuntimeError(
                    f"{consecutive_errors} consecutive episodes failed; aborting. "
                    f"Last error:\n{result.error}"
                )
            continue

        consecutive_errors = 0
        scored += 1
        successes = sum(1 for r in results if r.error is None and r.success)
        _log(
            f"[{scored:>3d}/{config.num_episodes}] seed {seed} "
            f"{'success' if result.success else 'failed'} in {result.steps} steps "
            f"({result.wall_seconds:.1f}s, {result.inferences} inferences) | "
            f"running {successes}/{scored} = {successes / scored * 100:.1f}%",
            log,
        )

    from .metrics import summarize

    summary = summarize(results, metadata=writer.metadata if writer else None)
    _log(
        f"final: {summary['successes']}/{summary['episodes_scored']} "
        f"= {summary['success_rate_pct']:.2f}% success "
        f"({summary['episodes_errored']} errored)",
        log,
    )
    return summary


def resolve_spec_from_policy(
    policy: Any,
    spec: Any,
    *,
    execution_horizon: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Align an :class:`~univtac_groot.spec.ObsSpec` with the live policy.

    Fetches ``get_modality_config`` from the server and rewrites the spec's
    delta indices, so a checkpoint that wants ``video_delta_indices=(-15, 0)``
    (the DROID pretrain tag) is served a two-frame history without the caller
    having to know that.

    Also verifies that the spec's video/state keys are the ones the policy
    declares, which turns an otherwise cryptic ``check_observation`` assertion on
    the server into an actionable message here.

    Returns:
        ``(aligned_spec, horizons)``.
    """
    from .arms import with_horizons
    from .receding_horizon import resolve_horizons

    modality = policy.get_modality_config()
    horizons = resolve_horizons(modality, execution_horizon=execution_horizon)

    def keys_of(entry: object) -> list[str]:
        if entry is None:
            return []
        raw = (
            entry.get("modality_keys")
            if isinstance(entry, Mapping)
            else getattr(entry, "modality_keys", None)
        )
        return [str(k) for k in (raw or [])]

    expected_video = keys_of(modality.get("video"))
    expected_state = keys_of(modality.get("state"))
    expected_language = keys_of(modality.get("language"))

    ours_video = sorted(spec.all_video_keys)
    if expected_video and sorted(expected_video) != ours_video:
        raise ValueError(
            f"video key mismatch: the checkpoint's embodiment declares "
            f"{sorted(expected_video)} but this arm supplies {ours_video}. Fix the "
            f"arm's video_keys (univtac_groot/arms.py) or point --embodiment-tag at "
            f"the matching checkpoint."
        )
    ours_state = sorted(spec.state_keys)
    if expected_state and sorted(expected_state) != ours_state:
        raise ValueError(
            f"state key mismatch: the checkpoint's embodiment declares "
            f"{sorted(expected_state)} but this arm supplies {ours_state}. A tactile "
            f"arm needs a checkpoint finetuned with the matching modality config "
            f"(configs/modality/univtac_tactile_config.py)."
        )
    if expected_language and spec.language_key not in expected_language:
        raise ValueError(
            f"language key mismatch: the checkpoint expects one of "
            f"{expected_language}, this arm sends {spec.language_key!r}."
        )

    aligned = with_horizons(
        spec,
        video_delta_indices=horizons["video_delta_indices"],
        state_delta_indices=horizons["state_delta_indices"],
    )
    _log(
        f"policy horizons: action_horizon={horizons['action_horizon']} "
        f"execution_horizon={horizons['execution_horizon']} "
        f"video_deltas={list(horizons['video_delta_indices'])} "
        f"state_deltas={list(horizons['state_delta_indices'] or [])}",
        log,
    )
    return aligned, horizons


def batch_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Add the batch axis GR00T's validators require.

    ``Gr00tSimPolicyWrapper.check_observation`` wants ``video.*`` as
    ``(B, T, H, W, C)`` uint8, ``state.*`` as ``(B, T, D)`` float32 and the
    language key as a length-``B`` sequence of strings. UniVTAC always runs one
    env, so ``B == 1``.
    """
    out: dict[str, Any] = {}
    for key, value in observation.items():
        if key.startswith("video."):
            arr = np.asarray(value)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            out[key] = arr[None, ...]
        elif key.startswith("state."):
            out[key] = np.asarray(value, dtype=np.float32)[None, ...]
        else:
            out[key] = [str(value)]
    return out


class BatchingPolicy:
    """Adds/removes the batch axis around a :class:`PolicyTransport`.

    Keeps :func:`run_episode` working with un-batched observations while the
    server sees the batched form it validates.
    """

    def __init__(self, transport: PolicyTransport, spec: Any) -> None:
        self.transport = transport
        self.spec = spec

    def get_action(
        self, observation: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict]:
        return self.transport.get_action(batch_observation(observation))

    def reset(self, **options: Any) -> dict:
        return self.transport.reset(**options)

    def get_modality_config(self) -> dict[str, Any]:
        return self.transport.get_modality_config()  # type: ignore[attr-defined]
