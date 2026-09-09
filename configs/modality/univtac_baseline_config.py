"""GR00T modality config -- UniVTAC Franka Panda, **no tactile** (ablation variant A).

Registered against ``EmbodimentTag.NEW_EMBODIMENT``, so this is the config for
the *finetuned* baseline: the true control for the tactile variant, identical to it
in every respect except the tactile state dimensions. (The zero-shot baseline
uses the shipped ``oxe_droid_relative_eef_relative_joint`` config instead and
needs no file here.)

Import path convention follows ``examples/SO100/so100_config.py``: importing
this module registers the config as a side effect, which is what
``--modality-config-path`` and ``launch_finetune.py`` rely on.

State layout (17-D, well inside ``max_state_dim=132``)::

    eef_9d            9   end-effector position (3) + rot6d (6)
    joint_position    7   panda_joint1..7
    gripper_position  1   finger opening, normalised to [0, 1] (1 == open)

Action layout (8-D)::

    joint_position    7   RELATIVE -- deltas generalise better across scenes
    gripper_position  1   ABSOLUTE -- a near-binary open/close target

``joint_position`` is relative and ``gripper_position`` absolute, matching the
choice made for both ``oxe_droid_relative_eef_relative_joint`` and the SO100
example. The evaluator does not need to know: ``processor.decode_action``
resolves the relative representation against the observed state and returns
absolute targets.
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


# Cameras are per task: two views on `insert_tube` and `lift_bottle`, the
# third-person view alone elsewhere. The table lives in univtac_groot.variants,
# shared with the dataset converter; this module only resolves it, from
# UNIVTAC_VIDEO_KEYS if set and otherwise TASK. With neither set it raises
# rather than defaulting, since a silently richer observation is invisible in
# the results.
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
"""Predicted chunk length -- GR00T N1.7's own default
(``GR00T_N1d7Config.action_horizon``), and the model's maximum.

This was 16 for a while, chosen to keep an open-loop chunk short relative to
UniVTAC's ``step_lim = 300``. That rationale does not apply: the rollout does
not execute whole chunks, it re-plans every ``EXECUTION_HORIZON`` steps. The
benchmark fixes the data, the observation space and the evaluation protocol --
the chunk length is GR00T's business, so use GR00T's number. (For reference,
ACT's is 50, above what N1.7 can express.)

Keep this identical across both variants -- it changes the action space, so a
mismatch makes the two runs incomparable.
"""

VIDEO_DELTA_INDICES = [0]
"""Single-frame observations. (The DROID pretrain tag instead uses ``[-15, 0]``;
``univtac_groot.rollout.resolve_spec_from_policy`` reads whichever the loaded
checkpoint declares and sizes the history buffer accordingly.)"""


univtac_baseline_config = {
    # Video keys must match the "video" entries of the dataset's meta/modality.json.
    "video": ModalityConfig(
        delta_indices=VIDEO_DELTA_INDICES,
        modality_keys=VIDEO_KEYS,
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "eef_9d",
            "joint_position",
            "gripper_position",
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


register_modality_config(univtac_baseline_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
