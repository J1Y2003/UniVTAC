"""The two ablation arms, as observation specs the whole pipeline shares.

Both arms keep vision, language, proprioception and the embodiment id identical;
they differ only in whether the UniVTAC tactile stream is folded into the 1-D
state vector. That is the comparison the study is after, so the specs are built
from one common proprioception layout to make accidental divergence impossible.

Two things constrain what an arm may declare, both verified in the upstream
source rather than assumed:

1.  **State width.** ``GR00T_N1d7Config.max_state_dim`` is 132. Proprioception
    here is 17-D (``eef_9d`` 9 + ``joint_position`` 7 + ``gripper_position`` 1),
    leaving 115 dims for tactile across all sensors. The default 8x6 pooling
    grid spends 48 per sensor, 96 for the two-finger GelSight Mini setup, for a
    113-D state.
2.  **Embodiment tag.** Adding tactile dimensions changes the state layout, and
    GR00T's state projector is embodiment-conditioned. Only ``NEW_EMBODIMENT``
    (and the other ``FINETUNE_ONLY_TAGS``) can carry a custom layout, and those
    tags ship in no released checkpoint — see
    ``gr00t/data/embodiment_tags.py::FINETUNE_ONLY_TAGS``. So:

    * the *baseline* arm runs zero-shot on ``nvidia/GR00T-N1.7-3B`` under
      ``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT``, whose state keys
      (``eef_9d``, ``gripper_position``, ``joint_position``) the UniVTAC Franka
      Panda maps onto directly — DROID is itself a Franka Panda;
    * the *tactile* arm requires a checkpoint finetuned under
      ``NEW_EMBODIMENT`` with a matching modality config
      (``configs/modality/univtac_tactile_config.py``).

    A like-for-like study therefore finetunes both arms with the same recipe and
    compares those two; the zero-shot baseline is a separate, useful reference
    point, not the tactile arm's control. ``docs/ABLATION.md`` spells this out.
"""

from __future__ import annotations

from dataclasses import replace

from .spec import (
    MAX_STATE_DIM,
    ObsSpec,
    StateField,
    TactileSpec,
)

# --------------------------------------------------------------------------- #
# Language / video keys per embodiment tag
# --------------------------------------------------------------------------- #

DROID_LANGUAGE_KEY = "annotation.language.language_instruction"
"""Language key of ``oxe_droid_relative_eef_relative_joint``."""

NEW_EMBODIMENT_LANGUAGE_KEY = "annotation.human.task_description"
"""Language key used by the SO100 example and by our finetune configs."""

DROID_VIDEO_KEYS = {
    "exterior_image_1_left": "head",
    "wrist_image_left": "wrist",
}
"""``oxe_droid_*`` video keys mapped onto UniVTAC's ``head`` / ``wrist`` cameras."""

UNIVTAC_VIDEO_KEYS = {
    "head": "head",
    "wrist": "wrist",
}
"""Video keys for our own finetunes; named after the UniVTAC cameras."""

TACTILE_VIDEO_KEYS = {
    "tactile_left": "left_tactile",
    "tactile_right": "right_tactile",
}
"""Extra video streams for ``TactileSpec(mode='video')``."""


# --------------------------------------------------------------------------- #
# Proprioception layout, shared by both arms
# --------------------------------------------------------------------------- #

PROPRIO_FIELDS: tuple[StateField, ...] = (
    StateField(key="eef_9d", kind="eef_9d", dim=9),
    StateField(key="joint_position", kind="joint_position", dim=7),
    StateField(key="gripper_position", kind="gripper_position", dim=1),
)
"""End-effector pose (xyz + rot6d), the 7 arm joints, and the gripper scalar."""

PROPRIO_DIM = sum(f.dim for f in PROPRIO_FIELDS)  # 17
TACTILE_BUDGET = MAX_STATE_DIM - PROPRIO_DIM  # 115


def tactile_state_fields(tactile: TactileSpec) -> tuple[StateField, ...]:
    """One state field per tactile sensor, named ``tactile_<sensor>``.

    Keeping the sensors as separate modality keys (rather than one wide
    ``tactile`` key) mirrors how the shipped configs split proprioception, and
    lets ``meta/modality.json`` describe each sensor's slice independently.
    """
    if not tactile.in_state:
        return ()
    per_sensor = tactile.dims_per_sensor()
    return tuple(
        StateField(key=f"tactile_{name}", kind="tactile", dim=per_sensor)
        for name in tactile.sensor_names
    )


# --------------------------------------------------------------------------- #
# Arm builders
# --------------------------------------------------------------------------- #


def baseline_spec(
    *,
    language_key: str = DROID_LANGUAGE_KEY,
    video_keys: dict[str, str] | None = None,
    image_size: tuple[int, int] = (256, 256),
) -> ObsSpec:
    """Arm A -- vision + language + proprioception + embodiment id. No tactile.

    Defaults target the zero-shot ``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`` tag.
    ``video_delta_indices`` is left at ``(0,)`` here and overwritten from the
    live policy at runtime (that tag actually asks for ``(-15, 0)``); see
    :func:`univtac_groot.arms.with_horizons`.
    """
    return ObsSpec(
        video_keys=dict(video_keys or DROID_VIDEO_KEYS),
        state_fields=PROPRIO_FIELDS,
        language_key=language_key,
        tactile=TactileSpec(mode="none"),
        image_size=image_size,
    )


def tactile_spec(
    *,
    mode: str = "depth_pool",
    sensor_names: tuple[str, ...] = ("left_tactile", "right_tactile"),
    pool_grid: tuple[int, int] = (8, 6),
    marker_pool: tuple[int, int] | None = (8, 6),
    language_key: str = NEW_EMBODIMENT_LANGUAGE_KEY,
    video_keys: dict[str, str] | None = None,
    image_size: tuple[int, int] = (256, 256),
    tactile_image_size: tuple[int, int] = (256, 256),
) -> ObsSpec:
    """Arm B -- the baseline plus the flattened UniVTAC tactile array in the state.

    Args:
        mode: ``'depth_pool'`` (pooled gel height map, the default),
            ``'marker'`` (pooled marker-motion displacements), or ``'video'``
            (tactile RGB as extra video streams, leaving the state untouched).
        pool_grid: pooling grid for ``depth_pool``.
        marker_pool: pooling grid for ``marker``.

    Raises:
        ValueError: the resulting state would exceed ``max_state_dim``; the
            message reports the remaining budget.
    """
    tactile = TactileSpec(
        mode=mode,  # type: ignore[arg-type]
        sensor_names=sensor_names,
        pool_grid=pool_grid,
        marker_pool=marker_pool,
    )
    return ObsSpec(
        video_keys=dict(video_keys or UNIVTAC_VIDEO_KEYS),
        state_fields=PROPRIO_FIELDS + tactile_state_fields(tactile),
        language_key=language_key,
        tactile=tactile,
        tactile_video_keys=dict(TACTILE_VIDEO_KEYS) if mode == "video" else {},
        image_size=image_size,
        tactile_image_size=tactile_image_size,
    )


def finetuned_baseline_spec(**kwargs) -> ObsSpec:
    """Arm A under ``NEW_EMBODIMENT``, i.e. the tactile arm's true control.

    Same proprioception and video keys as :func:`tactile_spec`, tactile removed,
    so the only difference between the two finetunes is the tactile dimensions.
    """
    kwargs.setdefault("language_key", NEW_EMBODIMENT_LANGUAGE_KEY)
    kwargs.setdefault("video_keys", UNIVTAC_VIDEO_KEYS)
    return baseline_spec(**kwargs)


ARMS = {
    "baseline": baseline_spec,
    "baseline_finetuned": finetuned_baseline_spec,
    "tactile": tactile_spec,
}
"""Registry used by ``scripts/run_eval.py --arm`` and the deploy YAMLs."""


def build_spec(arm: str, **kwargs) -> ObsSpec:
    """Build an :class:`ObsSpec` by arm name.

    Raises:
        KeyError: unknown arm name, listing the valid ones.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    return ARMS[arm](**kwargs)


def with_horizons(
    spec: ObsSpec,
    *,
    video_delta_indices: tuple[int, ...],
    state_delta_indices: tuple[int, ...] | None,
) -> ObsSpec:
    """Return ``spec`` with the delta indices the live policy actually declares.

    The arm builders cannot know these: they are a property of the checkpoint's
    embodiment config, fetched at runtime via ``get_modality_config``. A
    vision-only policy (``state_delta_indices is None``) keeps its state fields
    declared here but :class:`univtac_groot.history.ObsHistory` will drop the
    ``state.*`` keys, matching ``MultiStepWrapper``.
    """
    return replace(
        spec,
        video_delta_indices=tuple(video_delta_indices),
        state_delta_indices=tuple(state_delta_indices) if state_delta_indices else (0,),
    )
