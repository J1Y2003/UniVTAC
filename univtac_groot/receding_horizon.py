"""Receding-horizon (open-loop-within-chunk) control over GR00T action chunks.

GR00T N1.7's flow-matching DiT predicts a dense chunk of ``action_horizon``
future steps (40 by default, per ``GR00T_N1d7Config.action_horizon``; the
``libero_sim`` and SO100 configs use 16). Executing the whole chunk before
re-planning is cheap but drifts; re-planning every step is accurate but costs
one full VLA forward pass per environment step. The standard compromise —
what ``MultiStepWrapper`` implements and what ``PolicyHorizonSpec`` calls
``n_action_steps`` — is to execute the first ``execution_horizon`` actions of
each chunk and then re-plan from a fresh observation.

:class:`RecedingHorizonController` owns exactly that bookkeeping, with no
dependency on the environment or the policy transport, so the policy loop reads
as "ask for an action, step the sim".

Contract, matching ``gr00t/eval/_horizon_contract.py``:

* the chunk's ``delta_indices`` must be the contiguous ``range(0, H)`` — the
  chunk is indexed linearly, so a sparse window would silently execute the
  wrong rows;
* ``1 <= execution_horizon <= action_horizon``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np


@dataclass
class HorizonStats:
    """Per-episode accounting of inference cost, reported by the eval driver."""

    steps: int = 0
    """Environment actions dispatched."""
    inferences: int = 0
    """Policy forward passes (chunk requests)."""
    inference_seconds: float = 0.0
    """Wall-clock spent inside the policy call."""

    @property
    def steps_per_inference(self) -> float:
        return self.steps / self.inferences if self.inferences else 0.0

    @property
    def mean_inference_seconds(self) -> float:
        return self.inference_seconds / self.inferences if self.inferences else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "inferences": self.inferences,
            "inference_seconds": round(self.inference_seconds, 4),
            "steps_per_inference": round(self.steps_per_inference, 3),
            "mean_inference_seconds": round(self.mean_inference_seconds, 4),
        }


@dataclass
class RecedingHorizonController:
    """Cache one action chunk and hand out its first ``execution_horizon`` steps.

    Args:
        execution_horizon: how many actions of each chunk to execute before
            re-planning. ``None`` executes the full chunk.
        action_horizon: the chunk length the policy declares, used only to
            validate ``execution_horizon`` up front. ``None`` defers validation
            to the first chunk received.

    Example:
        >>> ctrl = RecedingHorizonController(execution_horizon=2)
        >>> chunks = iter([np.arange(8).reshape(4, 2).astype(np.float32)])
        >>> ctrl.next_action(lambda: next(chunks))
        array([0., 1.], dtype=float32)
        >>> ctrl.next_action(lambda: None)  # served from cache, no new inference
        array([2., 3.], dtype=float32)
    """

    execution_horizon: int | None = None
    action_horizon: int | None = None
    stats: HorizonStats = field(default_factory=HorizonStats)

    _chunk: np.ndarray | None = field(default=None, init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.execution_horizon is not None:
            if self.execution_horizon < 1:
                raise ValueError(
                    f"execution_horizon={self.execution_horizon} must be >= 1; "
                    f"a smaller value would execute nothing."
                )
            if self.action_horizon is not None and self.execution_horizon > self.action_horizon:
                raise ValueError(
                    f"execution_horizon={self.execution_horizon} exceeds the policy's "
                    f"action_horizon={self.action_horizon}; indexing the chunk that far "
                    f"would run off the end."
                )

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Drop the cached chunk. Call at the start of every episode."""
        self._chunk = None
        self._cursor = 0
        self.stats = HorizonStats()

    # -- state -------------------------------------------------------------
    @property
    def needs_inference(self) -> bool:
        """Whether the next action requires a fresh chunk."""
        return self._chunk is None or self._cursor >= self._stride

    @property
    def _stride(self) -> int:
        """How many rows of the current chunk are executed before re-planning."""
        if self._chunk is None:
            raise RuntimeError("no chunk loaded")
        horizon = self._chunk.shape[0]
        if self.execution_horizon is None:
            return horizon
        return min(self.execution_horizon, horizon)

    @property
    def remaining(self) -> int:
        """Actions left in the current chunk before the next re-plan."""
        if self._chunk is None:
            return 0
        return max(0, self._stride - self._cursor)

    # -- driving -----------------------------------------------------------
    def submit(self, chunk: np.ndarray) -> None:
        """Install a freshly predicted ``(T, D)`` chunk and rewind the cursor."""
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"action chunk must be (T, D); got shape {arr.shape}")
        if arr.shape[0] < 1:
            raise ValueError("action chunk is empty")
        if self.action_horizon is None:
            self.action_horizon = int(arr.shape[0])
            if (
                self.execution_horizon is not None
                and self.execution_horizon > self.action_horizon
            ):
                raise ValueError(
                    f"execution_horizon={self.execution_horizon} exceeds the chunk length "
                    f"{self.action_horizon} returned by the policy."
                )
        elif arr.shape[0] != self.action_horizon:
            raise ValueError(
                f"policy returned a chunk of length {arr.shape[0]} but previously "
                f"declared action_horizon={self.action_horizon}; the chunk length must "
                f"be stable for linear indexing to be meaningful."
            )
        self._chunk = arr
        self._cursor = 0

    def next_action(self, infer: Callable[[], np.ndarray | None]) -> np.ndarray:
        """Return the next action, calling ``infer`` only when the cache is dry.

        ``infer`` must return a ``(T, D)`` chunk; it is invoked at most once per
        call and is expected to already have the newest observation in hand.
        """
        if self.needs_inference:
            chunk = infer()
            if chunk is None:
                raise RuntimeError(
                    "a fresh action chunk was required but the inference callback "
                    "returned None"
                )
            self.submit(chunk)
            self.stats.inferences += 1
        assert self._chunk is not None
        action = self._chunk[self._cursor]
        self._cursor += 1
        self.stats.steps += 1
        return action


def resolve_horizons(
    modality_config: Mapping[str, object],
    *,
    execution_horizon: int | None = None,
) -> dict[str, object]:
    """Read the horizon contract out of a policy's modality config.

    Accepts the ``{modality: ModalityConfig}`` mapping returned by
    ``PolicyClient.get_modality_config()`` — over the wire the values arrive as
    ``ModalityConfig`` objects (the server serialises them explicitly) or as
    plain dicts, so both are handled.

    Returns a dict with ``action_horizon``, ``execution_horizon``,
    ``video_delta_indices`` and ``state_delta_indices`` (``None`` when the policy
    declares no state stream), mirroring ``PolicyHorizonSpec.from_modality_config``.
    """

    def deltas(entry: object) -> tuple[int, ...]:
        if entry is None:
            return ()
        raw = entry.get("delta_indices") if isinstance(entry, Mapping) else getattr(
            entry, "delta_indices", None
        )
        if raw is None:
            return ()
        return tuple(int(v) for v in raw)

    if "action" not in modality_config:
        raise ValueError(
            "policy modality config has no 'action' entry; cannot resolve the action "
            f"horizon. Available keys: {sorted(modality_config)}."
        )

    action_deltas = deltas(modality_config["action"])
    action_horizon = len(action_deltas)
    if action_horizon < 1:
        raise ValueError("policy declared an empty action.delta_indices.")
    if list(action_deltas) != list(range(action_horizon)):
        raise ValueError(
            f"action.delta_indices={list(action_deltas)} is not the contiguous "
            f"range(0, {action_horizon}). The chunk is indexed linearly, so a sparse "
            f"or shifted window would silently execute the wrong actions."
        )

    if execution_horizon is None:
        execution_horizon = action_horizon
    execution_horizon = int(execution_horizon)
    if not 1 <= execution_horizon <= action_horizon:
        raise ValueError(
            f"execution_horizon={execution_horizon} must satisfy "
            f"1 <= execution_horizon <= action_horizon={action_horizon}."
        )

    video_deltas = deltas(modality_config.get("video"))
    if not video_deltas:
        raise ValueError("policy modality config declares no video delta_indices.")
    state_deltas = deltas(modality_config.get("state")) or None

    return {
        "action_horizon": action_horizon,
        "execution_horizon": execution_horizon,
        "video_delta_indices": video_deltas,
        "state_delta_indices": state_deltas,
    }
