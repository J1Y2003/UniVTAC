"""Temporal stacking of flat observations for GR00T's ``delta_indices``.

GR00T embodiments declare per-modality observation offsets. The
``oxe_droid_relative_eef_relative_joint`` pretrain tag uses
``video.delta_indices = [-15, 0]``, i.e. the policy wants the current frame
*and* the frame from 15 control steps ago; a fresh finetune usually declares
``[0]``. Feeding a single frame to a two-frame policy fails
``Gr00tSimPolicyWrapper.check_observation`` on the temporal dimension, so the
history has to be maintained on the environment side.

Semantics mirror ``MultiStepWrapper`` in
``gr00t/eval/sim/wrapper/multistep_wrapper.py``, including its indexing
convention: with a ring buffer holding the last ``span + 1`` frames,
offset ``d`` reads ``buffer[d - 1]``, so ``d = 0`` is the newest frame. On reset
the buffer is filled with copies of the first observation, which zero-pads
history by repetition rather than with black frames.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .spec import assert_delta_indices, history_length


class ObsHistory:
    """Ring buffer that turns per-step flat frames into stacked GR00T observations.

    Args:
        video_delta_indices: frame offsets for every ``video.*`` key.
        state_delta_indices: frame offsets for every ``state.*`` key, or ``None``
            for a vision-only policy (no ``state.*`` keys are emitted then).

    Example:
        >>> h = ObsHistory((-1, 0), (0,))
        >>> _ = h.reset({"video.cam": np.zeros((2, 2, 3), np.uint8),
        ...              "state.q": np.zeros((3,), np.float32), "task": "go"})
        >>> obs = h.observe()
        >>> obs["video.cam"].shape, obs["state.q"].shape
        ((2, 2, 2, 3), (1, 3))
    """

    def __init__(
        self,
        video_delta_indices: tuple[int, ...],
        state_delta_indices: tuple[int, ...] | None,
    ) -> None:
        assert_delta_indices(np.asarray(video_delta_indices), name="video_delta_indices")
        if state_delta_indices is not None and len(state_delta_indices) > 0:
            assert_delta_indices(np.asarray(state_delta_indices), name="state_delta_indices")
            self.state_delta_indices: tuple[int, ...] | None = tuple(state_delta_indices)
        else:
            self.state_delta_indices = None

        self.video_delta_indices = tuple(video_delta_indices)
        self.span = history_length(self.video_delta_indices, self.state_delta_indices)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=self.span + 1)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        """Clear the buffer, prime it with ``frame``, and return the stacked view."""
        first = dict(frame)
        self._buffer = deque([first] * (self.span + 1), maxlen=self.span + 1)
        return self.observe()

    def append(self, frame: Mapping[str, Any]) -> None:
        """Push one new frame; the oldest is evicted."""
        if not self._buffer:
            raise RuntimeError("ObsHistory.reset() must be called before append().")
        self._buffer.append(dict(frame))

    # -- read --------------------------------------------------------------
    def observe(self) -> dict[str, Any]:
        """Stack the buffer into ``video.*`` (T,H,W,C), ``state.*`` (T,D) and language.

        Language keys carry only the newest value, matching ``MultiStepWrapper``
        (``spaces.Text`` is never repeated).
        """
        if not self._buffer:
            raise RuntimeError("ObsHistory.reset() must be called before observe().")

        newest = self._buffer[-1]
        out: dict[str, Any] = {}
        for key, value in newest.items():
            if key.startswith("video."):
                out[key] = self._stack(key, self.video_delta_indices)
            elif key.startswith("state."):
                if self.state_delta_indices is None:
                    continue  # vision-only policy: drop the state stream
                out[key] = self._stack(key, self.state_delta_indices)
            else:
                out[key] = value
        return out

    def _stack(self, key: str, delta_indices: tuple[int, ...]) -> np.ndarray:
        # MultiStepWrapper's off-by-one: offset 0 must read the last element.
        frames = [self._buffer[d - 1][key] for d in delta_indices]
        return np.stack(frames, axis=0)

    # -- introspection -----------------------------------------------------
    @property
    def video_horizon(self) -> int:
        return len(self.video_delta_indices)

    @property
    def state_horizon(self) -> int | None:
        return None if self.state_delta_indices is None else len(self.state_delta_indices)

    def __len__(self) -> int:
        return len(self._buffer)
