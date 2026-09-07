"""Benchmark NVIDIA GR00T N1.7 on the UniVTAC visuo-tactile benchmark.

The package is split so that the pieces which must run inside Isaac Sim's
interpreter never import torch or ``gr00t``:

``spec``, ``obs_adapter``, ``history``, ``action_adapter``, ``receding_horizon``,
``metrics``, ``arms``
    Pure numpy. Unit-testable on a login node.
``client``
    ZeroMQ/msgpack client for GR00T's ``PolicyServer`` (no ``gr00t`` import).
``env_wrapper``, ``rollout``
    The Gym-style UniVTAC wrapper and the evaluation loop. ``torch`` and
    ``gymnasium`` are imported lazily.
``server``
    Runs in the *GR00T* environment and loads the model.

See ``README.md`` for the two-process layout and ``docs/ABLATION.md`` for why
the tactile arm needs a finetuned checkpoint.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .action_adapter import ActionAdapter, GripperConvention
from .arms import ARMS, baseline_spec, build_spec, tactile_spec
from .history import ObsHistory
from .metrics import EpisodeResult, ResultWriter, compare, summarize
from .obs_adapter import ObsAdapter
from .receding_horizon import RecedingHorizonController, resolve_horizons
from .spec import (
    DEFAULT_ACTION_HORIZON,
    MAX_STATE_DIM,
    ObsSpec,
    StateField,
    TactileSpec,
)

__all__ = [
    "ARMS",
    "ActionAdapter",
    "DEFAULT_ACTION_HORIZON",
    "EpisodeResult",
    "GripperConvention",
    "MAX_STATE_DIM",
    "ObsAdapter",
    "ObsHistory",
    "ObsSpec",
    "RecedingHorizonController",
    "ResultWriter",
    "StateField",
    "TactileSpec",
    "baseline_spec",
    "build_spec",
    "compare",
    "resolve_horizons",
    "summarize",
    "tactile_spec",
]
