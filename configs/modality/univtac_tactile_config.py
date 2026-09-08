"""GR00T modality config -- UniVTAC Franka Panda **with tactile** (ablation variant B).

Identical to ``univtac_baseline_config`` except that the flattened UniVTAC
tactile array is concatenated onto the 1-D proprioception state as two extra
state keys, one per GelSight Mini finger sensor.

State layout (113-D of the 132-D ``max_state_dim`` budget)::

    eef_9d                  9   end-effector position (3) + rot6d (6)
    joint_position          7   panda_joint1..7
    gripper_position        1   finger opening, normalised to [0, 1]
    tactile_left_tactile   48   8x6 average-pooled gel height map
    tactile_right_tactile  48   8x6 average-pooled gel height map

Why pooled and not raw: UniVTAC's tactile sensors are *camera-based*
(``envs/sensors/tactile.py`` wraps TacEx GelSight Mini / GF225 / XenseWS), so a
raw reading is a 320x240 height map or a 64-marker motion field, not a small
array. Flattening either at full resolution would be 76 800 or 128 dims per
sensor against a 115-dim budget once proprioception is accounted for. An 8x6
average pool keeps the contact-location signal, costs 48 dims per sensor, and
leaves headroom. ``TactileSpec`` in ``univtac_groot/spec.py`` owns the pooling
and its dimensions must agree with the ``tactile_*`` widths declared here --
``ObsSpec.validate`` fails loudly if they drift.

Action layout is byte-identical to the baseline variant, which is what makes the two
runs comparable: only the state input changes.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# --------------------------------------------------------------------------- #
# Cameras are per-task
# --------------------------------------------------------------------------- #
# The UniVTAC paper trains ACT with two views on `insert_tube` and
# `lift_bottle` and the third-person view alone on every other task, so GR00T
# must match per task or the comparison is unfair in GR00T's favour. The table
# lives in univtac_groot.variants (one copy, shared with the dataset
# converter); this module only resolves it.
#
# Resolution order: UNIVTAC_VIDEO_KEYS (explicit, comma-separated), then TASK.
# With neither set this raises rather than falling back to a default, because a
# silently wrong camera set is exactly the class of bug that left UniVTAC's own
# tactile encoder randomly initialised -- see docs/UPSTREAM.md.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if not (_REPO_ROOT / "univtac_groot" / "spec.py").is_file():
    raise ImportError(
        f"cannot locate the univtac-groot checkout from {__file__}: expected it "
        f"two directories up, at {_REPO_ROOT}"
    )
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from univtac_groot.variants import video_keys_for_task as _video_keys_for_task  # noqa: E402

_explicit = _os.environ.get("UNIVTAC_VIDEO_KEYS", "").strip()
if _explicit:
    VIDEO_KEYS = [k.strip() for k in _explicit.split(",") if k.strip()]
else:
    _task = _os.environ.get("UNIVTAC_TASK", _os.environ.get("TASK", "")).strip()
    if not _task:
        raise ImportError(
            "the camera set is per-task, so this modality config needs TASK (or "
            "UNIVTAC_TASK) in the environment -- e.g. TASK=insert_hole. Set "
            "UNIVTAC_VIDEO_KEYS=head,wrist to override it explicitly instead. "
            "Two views are used for insert_tube and lift_bottle, third-person "
            "only for every other task, matching the paper's ACT setup."
        )
    VIDEO_KEYS = list(_video_keys_for_task(_task))
"""Camera set for this run. See univtac_groot.variants.MULTI_VIEW_TASKS."""


ACTION_HORIZON = 40
"""Must match ``univtac_baseline_config.ACTION_HORIZON``."""

TACTILE_POOL_GRID = (8, 6)
"""Average-pool grid per sensor. Mirror any change in
``univtac_groot.variants.tactile_spec(pool_grid=...)`` and in the dataset conversion."""

TACTILE_DIMS_PER_SENSOR = TACTILE_POOL_GRID[0] * TACTILE_POOL_GRID[1]  # 48

TACTILE_STATE_KEYS = ["tactile_left_tactile", "tactile_right_tactile"]
"""One key per UniVTAC tactile sensor, named ``tactile_<sensor name>``."""


univtac_tactile_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=VIDEO_KEYS,
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "eef_9d",
            "joint_position",
            "gripper_position",
            *TACTILE_STATE_KEYS,
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "joint_position",
            "gripper_position",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="joint_position",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="gripper_position",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(univtac_tactile_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
