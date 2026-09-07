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


ACTION_HORIZON = 16
"""Predicted chunk length.

GR00T N1.7's model default is 40 (``GR00T_N1d7Config.action_horizon``), and the
shipped posttrain configs use 16 (``libero_sim``) or 8 (``simpler_env_*``). 16 is
chosen here to match the UniVTAC control rate: the tasks cap out at
``step_lim = 300`` actions, so a 40-step chunk executed open-loop would commit
over a tenth of an episode per decision. Keep this identical across both variants
-- it changes the action space, so a mismatch makes the two runs incomparable.
"""

VIDEO_DELTA_INDICES = [0]
"""Single-frame observations. (The DROID pretrain tag instead uses ``[-15, 0]``;
``univtac_groot.rollout.resolve_spec_from_policy`` reads whichever the loaded
checkpoint declares and sizes the history buffer accordingly.)"""


univtac_baseline_config = {
    # Video keys must match the "video" entries of the dataset's meta/modality.json.
    "video": ModalityConfig(
        delta_indices=VIDEO_DELTA_INDICES,
        modality_keys=["head", "wrist"],
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
