"""Observation specs, as the whole pipeline shares them.

Every spec carries vision, language, proprioception and the embodiment id. The
17-D proprioception layout (``eef_9d`` 9 + ``joint_position`` 7 +
``gripper_position`` 1) is built once, below, so the converter, the trainer and
the evaluator cannot drift apart.

The embodiment tag decides which spec applies, and GR00T's state projector is
embodiment-conditioned:

* ``baseline`` runs zero-shot on ``nvidia/GR00T-N1.7-3B`` under
  ``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT``, whose state keys (``eef_9d``,
  ``gripper_position``, ``joint_position``) the UniVTAC Franka Panda maps onto
  directly -- DROID is itself a Franka Panda. Smoke tests only.
* ``baseline_finetuned`` is the reported model: the same layout finetuned under
  ``NEW_EMBODIMENT`` with ``configs/modality/univtac_baseline_config.py``.
  ``NEW_EMBODIMENT`` and the other ``FINETUNE_ONLY_TAGS`` ship in no released
  checkpoint -- see ``gr00t/data/embodiment_tags.py``.

``docs/BENCHMARK.md`` spells out the protocol.
"""

from __future__ import annotations

from dataclasses import replace

from .spec import ObsSpec, StateField

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

HEAD_ONLY_VIDEO_KEYS = {"head": "head"}
"""Third-person view alone."""

MULTI_VIEW_TASKS = frozenset({"insert_tube", "lift_bottle"})
"""Tasks whose ACT baseline was trained with **two** camera views.

Straight from the UniVTAC paper: *"task-specific configurations vary: both
insert tube and lift bottle utilize multi-view inputs, combining third-person
and wrist-mounted camera views; all other tasks use only the third-person
view."*

This **disagrees with the repo's own `policy/task_settings.json`**, which marks
`lift_can` -- not `lift_bottle` -- as ``camera_type: all``. The paper describes
what was actually trained, so the paper wins. (And per docs/UPSTREAM.md, ACT's
``process_data.py`` reads ``task_settings.json`` through a path that does not
resolve, under an ``if path.exists()`` guard, so it silently defaults every
task to ``head`` anyway -- another reason not to trust that file.)

Getting this wrong is not a small error: giving GR00T a wrist camera on a task
where ACT had only the third-person view is a straightforward unfair advantage,
and it is invisible in the results.
"""


def video_keys_for_task(task: str) -> dict[str, str]:
    """Camera set for ``task``, matching the UniVTAC paper's ACT configuration.

    Two views for :data:`MULTI_VIEW_TASKS`, third-person only for everything
    else. Used by the dataset converter and by both modality configs, so the
    training data and the model's declared inputs cannot drift apart.
    """
    if not task:
        raise ValueError(
            "video_keys_for_task() needs a task name; the camera set is "
            "per-task (see MULTI_VIEW_TASKS)"
        )
    if task in MULTI_VIEW_TASKS:
        return dict(UNIVTAC_VIDEO_KEYS)
    return dict(HEAD_ONLY_VIDEO_KEYS)

# --------------------------------------------------------------------------- #
# Proprioception layout, shared by both variants
# --------------------------------------------------------------------------- #

PROPRIO_FIELDS: tuple[StateField, ...] = (
    StateField(key="eef_9d", kind="eef_9d", dim=9),
    StateField(key="joint_position", kind="joint_position", dim=7),
    StateField(key="gripper_position", kind="gripper_position", dim=1),
)
"""End-effector pose (xyz + rot6d), the 7 arm joints, and the gripper scalar."""

PROPRIO_DIM = sum(f.dim for f in PROPRIO_FIELDS)  # 17


# --------------------------------------------------------------------------- #
# Spec builders
# --------------------------------------------------------------------------- #


def baseline_spec(
    *,
    language_key: str = DROID_LANGUAGE_KEY,
    video_keys: dict[str, str] | None = None,
    image_size: tuple[int, int] = (256, 256),
) -> ObsSpec:
    """Vision + language + proprioception + embodiment id.

    Defaults target the zero-shot ``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`` tag.
    ``video_delta_indices`` is left at ``(0,)`` here and overwritten from the
    live policy at runtime (that tag actually asks for ``(-15, 0)``); see
    :func:`univtac_groot.variants.with_horizons`.
    """
    return ObsSpec(
        video_keys=dict(video_keys or DROID_VIDEO_KEYS),
        state_fields=PROPRIO_FIELDS,
        language_key=language_key,
        image_size=image_size,
    )


def finetuned_baseline_spec(*, task: str | None = None, **kwargs) -> ObsSpec:
    """The reported model: :func:`baseline_spec` under ``NEW_EMBODIMENT``.

    Same proprioception, but the per-task camera set and the language key our
    finetune configs declare.
    """
    kwargs.setdefault("language_key", NEW_EMBODIMENT_LANGUAGE_KEY)
    if "video_keys" not in kwargs:
        kwargs["video_keys"] = video_keys_for_task(task) if task else UNIVTAC_VIDEO_KEYS
    return baseline_spec(**kwargs)


VARIANTS = {
    "baseline": baseline_spec,
    "baseline_finetuned": finetuned_baseline_spec,
}
"""Registry used by ``scripts/run_eval.py --variant`` and the deploy YAMLs."""


def build_spec(variant: str, **kwargs) -> ObsSpec:
    """Build an :class:`ObsSpec` by variant name.

    Raises:
        KeyError: unknown variant name, listing the valid ones.
    """
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[variant](**kwargs)


def with_horizons(
    spec: ObsSpec,
    *,
    video_delta_indices: tuple[int, ...],
    state_delta_indices: tuple[int, ...] | None,
) -> ObsSpec:
    """Return ``spec`` with the delta indices the live policy actually declares.

    The variant builders cannot know these: they are a property of the checkpoint's
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
