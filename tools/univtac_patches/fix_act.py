#!/usr/bin/env python3
"""Repair the path resolution in UniVTAC's ``policy/ACT`` training pipeline.

Why this exists
---------------
Every relative path in the ACT pipeline is resolved against the *process CWD*,
and the intended CWD is ``policy/ACT`` -- ``one.sh`` calls ``bash train.sh``
bare and tests ``./data/sim-$task/...``, and ``eval.sh`` opens with ``cd ../..``.
Most paths agree with that. Three do not, so no single CWD resolves everything:

* ``process_data.py`` reads ``Path(__file__).parent / 'task_settings.json'``,
  but that file lives at ``policy/task_settings.json``, one level up. The read
  is ``.exists()``-guarded, so it fails **open**: ``camera_type`` silently
  falls back to ``head``, quietly dropping the wrist camera for ``lift_can``
  and ``insert_tube``, which are configured ``all``.
* ``constants.py`` reads ``SIM_TASK_CONFIGS.json`` next to itself, but the only
  committed copy is ``policy/SIM_TASK_CONFIGS.json`` -- a leftover from running
  ``process_data.py`` from ``policy/``. (``_base_data_preprocessor.py`` still
  carries a duplicate ``main()`` in which the ``__file__``-relative lookup is
  correct; ``ACT/process_data.py`` is a copy that did not adjust for the moved
  ``__file__``.)
* ``train_config*.yml`` set ``tactile_ckpt: encoder/checkpoints/...``, which is
  repo-root-relative, so from ``policy/ACT`` it points at a directory that does
  not exist. ``backbone.py``'s load is ``.exists()``-guarded too, so this also
  fails open: the tactile encoder is left **randomly initialised** with no
  warning. The released encoder is ``checkpoints/encoder.pth``, fetched by
  ``bash data/download.sh --checkpoint``.

A fourth mismatch blocks the download path specifically: ``data/download.sh``
preserves the *published* dataset layout, so episodes land at
``data/isaac45/<task>/hdf5/*.hdf5``, whereas ``BaseDataPreprocessor`` reads
``data/<task>/<config>/``. There is no ``<config>`` level in the download and
the task sits one directory deeper, so downloaded data is not directly
consumable. This script bridges it with a directory symlink (no copy).

Two further fail-open bugs are fixed while we are here:

* ``camera_names: cam_high`` in ``train_config.yml`` (and ``_all``, ``_vision``,
  ``_freeze``, ``_scrach``) is an ALOHA leftover. The processed HDF5 writes
  ``cam_head``/``cam_wrist``, and ``utils.py`` indexes
  ``/observations/images/{cam_name}`` directly, so this raises
  ``KeyError: cam_high`` only once the first batch is drawn.
* ``backbone.py`` is changed to *raise* on a configured-but-missing
  ``tactile_ckpt`` rather than skip it. A crash at startup is strictly better
  than a silently untrained encoder in an ablation.

Usage
-----
::

    python tools/univtac_patches/fix_act.py --univtac-root "$UNIVTAC_ROOT"
    python tools/univtac_patches/fix_act.py --univtac-root "$UNIVTAC_ROOT" --dry-run
    python tools/univtac_patches/fix_act.py --univtac-root "$UNIVTAC_ROOT" --revert
    python tools/univtac_patches/fix_act.py --univtac-root "$UNIVTAC_ROOT" --check

Idempotent: re-running applies nothing. Every file it edits is copied once to
``<name>.univtac_groot.orig`` beforehand, and ``--revert`` restores from those.
This touches the UniVTAC checkout only -- no dotfiles, no conda environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

BACKUP_SUFFIX = ".univtac_groot.orig"
MARKER = "univtac_groot"

# The rewritten train.sh. Self-locating, so the CWD no longer matters, and it
# refuses to start rather than producing a confusing failure ten minutes in.
TRAIN_SH = """#!/bin/bash
# Patched by univtac_groot (tools/univtac_patches/fix_act.py).
# Original preserved as train.sh""" + BACKUP_SUFFIX + """
#
# Everything below resolves against the process CWD -- imitate_episodes.py,
# ./train_config.yml, ./SIM_TASK_CONFIGS.json, ./act_ckpt, and the dataset_dir
# recorded inside SIM_TASK_CONFIGS.json -- and all of them assume policy/ACT.
# Self-locate so this runs correctly from any directory.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

if [[ $# -lt 5 ]]; then
  echo "usage: bash train.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id> [train_config]" >&2
  echo "example: bash train.sh lift_bottle clean 100 0 0 train_config" >&2
  exit 2
fi

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
train_config=${6:-"train_config"}

export CUDA_VISIBLE_DEVICES=${gpu_id}

task_key="sim-${task_name}-${task_config}-${expert_data_num}"

if [[ ! -f "./${train_config}.yml" ]]; then
  echo "error: ./${train_config}.yml not found in $(pwd)" >&2
  echo "       available: $(ls train_config*.yml 2>/dev/null | tr '\\n' ' ')" >&2
  exit 1
fi

# SIM_TASK_CONFIGS.json is written by process_data.py into whatever directory it
# ran in. A missing file or a missing key means the data step never ran here.
if [[ ! -f "./SIM_TASK_CONFIGS.json" ]]; then
  echo "error: ./SIM_TASK_CONFIGS.json not found in $(pwd)" >&2
  echo "       Run the data step first, from this directory:" >&2
  echo "         bash process_data.sh ${task_name} ${task_config} ${expert_data_num}" >&2
  echo "       (it needs raw episodes at <repo>/data/${task_name}/${task_config}/*.hdf5;" >&2
  echo "        fetch them with: bash data/download.sh --task ${task_name} --version 45)" >&2
  exit 1
fi

python3 - "$task_key" <<'PYCHECK'
import json, sys
from pathlib import Path
key = sys.argv[1]
cfgs = json.loads(Path("SIM_TASK_CONFIGS.json").read_text())
if key not in cfgs:
    print(f"error: '{key}' is not in SIM_TASK_CONFIGS.json", file=sys.stderr)
    print(f"       known keys: {sorted(cfgs) or '(none)'}", file=sys.stderr)
    print("       Re-run process_data.sh with matching arguments.", file=sys.stderr)
    raise SystemExit(1)
entry = cfgs[key]
data_dir = Path(entry["dataset_dir"])
if not data_dir.is_dir():
    print(f"error: dataset_dir '{data_dir}' does not exist relative to {Path.cwd()}", file=sys.stderr)
    print("       dataset_dir is stored CWD-relative, so it only resolves from", file=sys.stderr)
    print("       the directory process_data.py ran in (policy/ACT).", file=sys.stderr)
    raise SystemExit(1)
episodes = sorted(data_dir.glob("episode_*.hdf5"))
if len(episodes) < entry["num_episodes"]:
    print(f"error: {data_dir} holds {len(episodes)} episodes, "
          f"but SIM_TASK_CONFIGS.json claims {entry['num_episodes']}", file=sys.stderr)
    raise SystemExit(1)
print(f"[train.sh] {key}: {len(episodes)} episodes in {data_dir}, "
      f"cameras={entry['camera_names']}")
PYCHECK

ckpt_dir="./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/${train_config}"
mkdir -p "${ckpt_dir}"
echo "[train.sh] cwd=$(pwd)"
echo "[train.sh] ckpt_dir=${ckpt_dir}"

python3 imitate_episodes.py \\
    --task_name "${task_key}" \\
    --ckpt_dir "${ckpt_dir}" \\
    --config_path "./${train_config}.yml" \\
    --seed "${seed}"
"""

PROCESS_DATA_SH = """#!/bin/bash
# Patched by univtac_groot (tools/univtac_patches/fix_act.py).
# Original preserved as process_data.sh""" + BACKUP_SUFFIX + """
#
# process_data.py writes both ./data/sim-<task>/<config>-<n>/ and
# ./SIM_TASK_CONFIGS.json relative to the CWD, and imitate_episodes.py reads
# them back the same way. Pin the CWD to policy/ACT so the two always agree.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

if [[ $# -lt 3 ]]; then
  echo "usage: bash process_data.sh <task_name> <task_config> <expert_data_num>" >&2
  exit 2
fi

task_name=${1}
task_config=${2}
expert_data_num=${3}

repo_root=$(cd ../.. && pwd)
raw_dir="${repo_root}/data/${task_name}/${task_config}"
if [[ ! -d "${raw_dir}" ]]; then
  echo "error: raw episodes not found at ${raw_dir}" >&2
  echo "       (that path is absolute, anchored at the repo root by" >&2
  echo "        policy/_base_data_preprocessor.py -- it is not CWD-relative)" >&2
  echo "       Fetch them with:" >&2
  echo "         cd ${repo_root} && bash data/download.sh --task ${task_name} --version 45" >&2
  exit 1
fi

echo "[process_data.sh] cwd=$(pwd)"
echo "[process_data.sh] raw=${raw_dir} ($(ls "${raw_dir}"/*.hdf5 2>/dev/null | wc -l) hdf5 files)"

python3 process_data.py "$task_name" "$task_config" "$expert_data_num"
"""


class Report:
    """Collects per-fix outcomes so the summary is one block, not a stream."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.rows: list[tuple[str, str, str]] = []
        self.failed = False

    def add(self, status: str, name: str, detail: str = "") -> None:
        if status == "FAIL":
            self.failed = True
        self.rows.append((status, name, detail))

    def render(self) -> str:
        width = max((len(name) for _, name, _ in self.rows), default=0)
        lines = []
        for status, name, detail in self.rows:
            lines.append(f"  [{status:^7}] {name.ljust(width)}  {detail}".rstrip())
        return "\n".join(lines)


def backup(path: Path, report: Report) -> None:
    """Copy ``path`` aside once, so ``--revert`` always has a pristine original."""
    dest = path.with_name(path.name + BACKUP_SUFFIX)
    if dest.exists() or report.dry_run:
        return
    shutil.copy2(path, dest)


# --------------------------------------------------------------------------- #
# Fixes
# --------------------------------------------------------------------------- #
def fix_task_settings(act: Path, policy: Path, report: Report) -> None:
    """Make ``policy/task_settings.json`` visible where ``process_data.py`` looks.

    A copy rather than a symlink: the checkout may sit on a filesystem where
    symlinks are awkward (or be a Windows-side clone), and this file is 839
    bytes of static config that upstream has not changed.
    """
    target = act / "task_settings.json"
    source = policy / "task_settings.json"
    if not source.is_file():
        report.add("FAIL", "task_settings.json", f"source missing: {source}")
        return
    if target.is_file() and target.read_bytes() == source.read_bytes():
        report.add("ok", "task_settings.json", "already present")
        return
    if not report.dry_run:
        shutil.copy2(source, target)
    report.add("FIXED", "task_settings.json", f"copied from {source.name} (was silently defaulting camera_type=head)")


def fix_sim_task_configs(act: Path, policy: Path, report: Report) -> None:
    """Point ``constants.py`` at a ``SIM_TASK_CONFIGS.json`` that exists.

    ``imitate_episodes.py`` reads this CWD-relative and does not import
    ``constants``, so training is unaffected -- but the eval path does import it,
    and a bare ``import constants`` currently raises ``FileNotFoundError``.
    Patched to fall back to the parent directory and to an empty mapping.
    """
    path = act / "constants.py"
    if not path.is_file():
        report.add("FAIL", "constants.py", "not found")
        return
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        report.add("ok", "constants.py", "already patched")
        return

    old = (
        'SIM_TASK_CONFIGS_PATH = os.path.join(current_dir, "./SIM_TASK_CONFIGS.json")\n'
        'with open(SIM_TASK_CONFIGS_PATH, "r") as f:\n'
        "    SIM_TASK_CONFIGS = json.load(f)\n"
    )
    if old not in text:
        report.add("skip", "constants.py", "upstream block changed; inspect manually")
        return

    new = (
        f"# --- {MARKER}: SIM_TASK_CONFIGS.json is written by process_data.py into its\n"
        "# own CWD (policy/ACT), while the only committed copy sits one level up in\n"
        "# policy/. Search both, and tolerate absence -- process_data.py creates it.\n"
        "_SIM_TASK_CONFIG_CANDIDATES = [\n"
        '    os.path.join(current_dir, "SIM_TASK_CONFIGS.json"),\n'
        '    os.path.join(current_dir, os.pardir, "SIM_TASK_CONFIGS.json"),\n'
        "]\n"
        "SIM_TASK_CONFIGS = {}\n"
        "SIM_TASK_CONFIGS_PATH = _SIM_TASK_CONFIG_CANDIDATES[0]\n"
        "for _candidate in _SIM_TASK_CONFIG_CANDIDATES:\n"
        "    if os.path.isfile(_candidate):\n"
        "        SIM_TASK_CONFIGS_PATH = _candidate\n"
        '        with open(_candidate, "r") as f:\n'
        "            SIM_TASK_CONFIGS = json.load(f)\n"
        "        break\n"
    )
    backup(path, report)
    if not report.dry_run:
        path.write_text(text.replace(old, new), encoding="utf-8")
    report.add("FIXED", "constants.py", "SIM_TASK_CONFIGS.json lookup now searches ./ and ../")


def fix_camera_names(act: Path, report: Report) -> None:
    """``cam_high`` -> ``cam_head`` in every train config.

    ``utils.py`` does ``root[f"/observations/images/{cam_name}"]`` with no
    aliasing, and the preprocessor's ``camera_key_map`` saves ``cam_head``.
    """
    changed: list[str] = []
    already: list[str] = []
    for path in sorted(act.glob("train_config*.yml")):
        text = path.read_text(encoding="utf-8")
        if "cam_high" not in text:
            already.append(path.name)
            continue
        backup(path, report)
        if not report.dry_run:
            path.write_text(text.replace("cam_high", "cam_head"), encoding="utf-8")
        changed.append(path.name)
    if changed:
        report.add("FIXED", "camera_names", f"cam_high -> cam_head in {', '.join(changed)}")
    if already and not changed:
        report.add("ok", "camera_names", "no cam_high left")


def find_encoder_checkpoint(root: Path) -> Path | None:
    """Locate the released tactile encoder, wherever download.sh dropped it.

    ``data/download.sh --checkpoint`` preserves the published layout under its
    output directory (``data/`` by default), giving ``data/checkpoints/encoder.pth``.
    """
    candidates = [
        root / "data" / "checkpoints" / "encoder.pth",
        root / "checkpoints" / "encoder.pth",
        root / "encoder" / "encoder.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # The authors' own training output, if the user trained an encoder locally.
    for candidate in sorted((root / "encoder").rglob("best.pth")):
        return candidate
    return None


def fix_tactile_ckpt(act: Path, root: Path, report: Report) -> None:
    """Rewrite ``tactile_ckpt`` to an absolute path, or say why we cannot.

    The shipped value points at the authors' local training output, which was
    never published, and is resolved CWD-relative so it lands in
    ``policy/ACT/encoder/`` regardless.
    """
    resolved = find_encoder_checkpoint(root)
    configs = sorted(act.glob("train_config*.yml"))

    if resolved is None:
        report.add(
            "TODO",
            "tactile_ckpt",
            "encoder weights not on disk -- run: bash data/download.sh --checkpoint",
        )
        return

    changed: list[str] = []
    for path in configs:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        out = []
        touched = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("tactile_ckpt:"):
                value = stripped.split(":", 1)[1].strip()
                # train_config_scrach.yml deliberately trains from scratch.
                if value in ("null", "~", "") or value.strip("\"'") == resolved.as_posix():
                    out.append(line)
                    continue
                # Quoted, and POSIX-separated: an unquoted Windows-style path
                # would make PyYAML choke on the backslashes. Harmless on Linux.
                out.append(f'tactile_ckpt: "{resolved.as_posix()}"\n')
                touched = True
                continue
            out.append(line)
        if touched:
            backup(path, report)
            if not report.dry_run:
                path.write_text("".join(out), encoding="utf-8")
            changed.append(path.name)

    if changed:
        report.add("FIXED", "tactile_ckpt", f"-> {resolved} in {', '.join(changed)}")
    else:
        report.add("ok", "tactile_ckpt", f"already {resolved.as_posix()}")


def fix_backbone_fail_open(act: Path, report: Report) -> None:
    """Make a missing ``tactile_ckpt`` fatal instead of silent.

    ``if ckpt and Path(ckpt).exists(): backbone.load_state_dict(...)`` means a
    typo'd or absent path yields a randomly initialised tactile encoder. In an
    ablation whose whole question is "does tactile help", that failure mode
    produces a plausible-looking null result.
    """
    path = act / "detr" / "models" / "backbone.py"
    if not path.is_file():
        report.add("FAIL", "backbone.py", "not found")
        return
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        report.add("ok", "backbone.py", "already patched")
        return

    old = (
        "        if ckpt and Path(ckpt).exists():\n"
        "            backbone.load_state_dict(torch.load(ckpt, weights_only=True))\n"
    )
    if old not in text:
        report.add("skip", "backbone.py", "upstream block changed; inspect manually")
        return

    new = (
        f"        # --- {MARKER}: fail loudly. The original skipped the load when the\n"
        "        # path did not resolve, leaving a randomly initialised tactile\n"
        "        # encoder and no warning -- indistinguishable from 'tactile does\n"
        "        # not help'. An explicit tactile_ckpt must load or stop the run.\n"
        "        if ckpt:\n"
        "            ckpt_path = Path(ckpt)\n"
        "            if not ckpt_path.is_file():\n"
        "                raise FileNotFoundError(\n"
        '                    f"tactile_ckpt={ckpt!r} does not exist (resolved from "\n'
        '                    f"cwd={Path.cwd()}). The released encoder is "\n'
        '                    "checkpoints/encoder.pth via `bash data/download.sh "\n'
        '                    "--checkpoint`. Set tactile_ckpt: null to train the "\n'
        '                    "tactile encoder from scratch on purpose."\n'
        "                )\n"
        "            backbone.load_state_dict(torch.load(str(ckpt_path), weights_only=True))\n"
    )
    backup(path, report)
    if not report.dry_run:
        path.write_text(text.replace(old, new), encoding="utf-8")
    report.add("FIXED", "backbone.py", "missing tactile_ckpt now raises instead of being skipped")


def fix_shell_scripts(act: Path, report: Report) -> None:
    """Replace ``train.sh`` and ``process_data.sh`` with self-locating versions."""
    for name, body in (("train.sh", TRAIN_SH), ("process_data.sh", PROCESS_DATA_SH)):
        path = act / name
        if not path.is_file():
            report.add("FAIL", name, "not found")
            continue
        current = path.read_text(encoding="utf-8")
        if MARKER in current and current == body:
            report.add("ok", name, "already patched")
            continue
        backup(path, report)
        if not report.dry_run:
            path.write_text(body, encoding="utf-8")
        report.add("FIXED", name, "self-locating (cd to policy/ACT) + preflight checks")


def downloaded_task_dirs(root: Path) -> dict[str, Path]:
    """Map task name -> the ``hdf5/`` directory ``data/download.sh`` created.

    ``download.sh`` sets ``UNIVTAC_RAW_DIR`` to ``data/`` and preserves the
    published layout, so episodes arrive at
    ``data/isaac{45,51}/<task>/hdf5/*.hdf5``. Prefer 45: this ``main`` branch
    installs Isaac Sim 4.5, and the released checkpoints target that data.
    """
    found: dict[str, Path] = {}
    for version in ("isaac45", "isaac51"):
        base = root / "data" / version
        if not base.is_dir():
            continue
        for task_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            hdf5_dir = task_dir / "hdf5"
            if hdf5_dir.is_dir() and any(hdf5_dir.glob("*.hdf5")):
                found.setdefault(task_dir.name, hdf5_dir)
    return found


def link_raw(root: Path, config_name: str, report: Report) -> None:
    """Expose the downloaded episodes where the preprocessor expects them.

    ``BaseDataPreprocessor`` reads ``data/<task>/<config>/`` and picks up
    episodes with ``rglob('*.hdf5')`` sorted by ``int(path.stem)``. The
    downloaded files are already named ``0.hdf5``..``99.hdf5``, so a single
    directory symlink is enough and costs no disk. ``<config>`` has no
    counterpart in the published layout -- it is the collect-time config name,
    and the authors' own ``SIM_TASK_CONFIGS.json`` entry
    (``sim-lift_bottle-clean-2``) says they used ``clean``.
    """
    available = downloaded_task_dirs(root)
    if not available:
        report.add(
            "TODO",
            "raw data link",
            "nothing under data/isaac45/ -- run: bash data/download.sh --task <task> --version 45",
        )
        return

    linked: list[str] = []
    already: list[str] = []
    failed: list[str] = []
    for task, hdf5_dir in available.items():
        target = root / "data" / task / config_name
        if target.exists() or target.is_symlink():
            already.append(task)
            continue
        if report.dry_run:
            linked.append(task)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        relative = Path("..") / hdf5_dir.relative_to(root / "data")
        try:
            target.symlink_to(relative, target_is_directory=True)
            linked.append(task)
        except OSError:
            # Windows without developer mode, or a filesystem with no symlinks.
            # Copying 24 GB per task is not acceptable, so say so rather than
            # silently duplicating.
            failed.append(task)

    if linked:
        report.add("FIXED", "raw data link", f"data/<task>/{config_name} -> ../isaac*/<task>/hdf5 for {', '.join(sorted(linked))}")
    if already:
        report.add("ok", "raw data link", f"already present for {', '.join(sorted(already))}")
    if failed:
        report.add(
            "FAIL",
            "raw data link",
            f"symlink unsupported here; bind-mount or move data/isaac45/<task>/hdf5 "
            f"to data/<task>/{config_name} for {', '.join(sorted(failed))}",
        )


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def describe_state(root: Path, act: Path) -> str:
    """Report what data actually exists, so the next command is unambiguous."""
    lines: list[str] = []

    raw_root = root / "data"
    tasks = []
    if raw_root.is_dir():
        for task_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            if task_dir.name in ("checkpoints", "contact", "isaac45", "isaac51"):
                continue
            for cfg_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                count = len(list(cfg_dir.rglob("*.hdf5")))
                if count:
                    tasks.append(f"{task_dir.name}/{cfg_dir.name}: {count} hdf5")
    lines.append("raw episodes under <root>/data/<task>/<config>/ (what the preprocessor reads):")
    lines.extend(f"    {t}" for t in tasks) if tasks else lines.append("    (none)")

    downloaded = downloaded_task_dirs(root)
    lines.append("downloaded episodes under <root>/data/isaac*/<task>/hdf5/ (what download.sh writes):")
    if downloaded:
        for task, hdf5_dir in sorted(downloaded.items()):
            count = len(list(hdf5_dir.glob("*.hdf5")))
            lines.append(f"    {hdf5_dir.relative_to(root)}: {count} hdf5")
    else:
        lines.append("    (none)")

    processed = act / "data"
    entries = []
    if processed.is_dir():
        for d in sorted(processed.rglob("*")):
            if d.is_dir() and list(d.glob("episode_*.hdf5")):
                entries.append(f"{d.relative_to(processed)}: {len(list(d.glob('episode_*.hdf5')))} episodes")
    lines.append("processed episodes under policy/ACT/data/:")
    lines.extend(f"    {e}" for e in entries) if entries else lines.append("    (none)")

    cfg_path = act / "SIM_TASK_CONFIGS.json"
    if cfg_path.is_file():
        try:
            keys = sorted(json.loads(cfg_path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            keys = [f"<unreadable: {exc}>"]
        lines.append(f"policy/ACT/SIM_TASK_CONFIGS.json keys: {keys or '(empty)'}")
    else:
        lines.append("policy/ACT/SIM_TASK_CONFIGS.json: MISSING (process_data.sh creates it)")

    encoder = find_encoder_checkpoint(root)
    lines.append(f"tactile encoder: {encoder if encoder else 'MISSING (data/download.sh --checkpoint)'}")

    # Confirm the on-disk HDF5 layout matches what utils.py and state_dim expect.
    sample = next((d / "episode_0.hdf5" for d in (processed.rglob("*") if processed.is_dir() else [])
                   if d.is_dir() and (d / "episode_0.hdf5").is_file()), None)
    if sample is not None:
        lines.append(f"sample {sample.name} in {sample.parent}:")
        try:
            import h5py  # noqa: PLC0415 - optional, only for reporting

            with h5py.File(sample, "r") as f:
                images = f["/observations/images"]
                lines.append(f"    image keys: {sorted(images.keys())}")
                lines.append(f"    qpos shape: {f['/observations/qpos'].shape}")
                lines.append(f"    action shape: {f['/action'].shape}")
                lines.append(
                    f"    -> train_config state_dim must equal qpos dim "
                    f"({f['/observations/qpos'].shape[-1]})"
                )
        except ImportError:
            lines.append("    (install h5py in this interpreter to inspect keys/shapes)")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"    (unreadable: {exc})")

    return "\n".join("  " + line for line in lines)


def revert(root: Path, act: Path) -> int:
    """Restore every backed-up file and drop what this patch added.

    Only removes symlinks that resolve into ``data/isaac*/`` -- a real directory
    of episodes at ``data/<task>/<config>`` was put there by data collection,
    not by us, and must not be touched.
    """
    restored = 0
    data = root / "data"
    if data.is_dir():
        for link in sorted(data.glob("*/*")):
            if not link.is_symlink():
                continue
            resolved = str(link.resolve())
            if "isaac45" in resolved or "isaac51" in resolved:
                link.unlink()
                print(f"  unlinked  {link.relative_to(root)}")
                restored += 1
    for backup_path in sorted(act.rglob("*" + BACKUP_SUFFIX)):
        original = backup_path.with_name(backup_path.name[: -len(BACKUP_SUFFIX)])
        shutil.copy2(backup_path, original)
        backup_path.unlink()
        print(f"  restored {original.relative_to(act.parent.parent)}")
        restored += 1
    added = act / "task_settings.json"
    if added.is_file():
        added.unlink()
        print(f"  removed   {added.relative_to(act.parent.parent)}")
        restored += 1
    print(f"reverted {restored} file(s)" if restored else "nothing to revert")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--univtac-root",
        required=True,
        help="path to the UniVTAC checkout (the directory holding envs/ and policy/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument(
        "--raw-config",
        default="clean",
        help=(
            "collect-config name to expose the downloaded episodes under, i.e. "
            "data/<task>/<RAW_CONFIG> -> data/isaac45/<task>/hdf5. Must match the "
            "second argument you pass to process_data.sh / eval_policy.sh"
        ),
    )
    parser.add_argument(
        "--no-link-raw",
        action="store_true",
        help="skip the data/<task>/<config> symlinks (use if you arrange raw data yourself)",
    )
    parser.add_argument("--revert", action="store_true", help="restore the original files")
    parser.add_argument("--check", action="store_true", help="only report the current state")
    args = parser.parse_args(argv)

    root = Path(args.univtac_root).expanduser().resolve()
    act = root / "policy" / "ACT"
    policy = root / "policy"
    if not act.is_dir():
        print(f"error: {act} does not exist -- is --univtac-root correct?", file=sys.stderr)
        return 2

    print(f"UniVTAC root: {root}")

    if args.revert:
        return revert(root, act)

    if args.check:
        print("\nCurrent state:")
        print(describe_state(root, act))
        return 0

    report = Report(dry_run=args.dry_run)
    fix_shell_scripts(act, report)
    fix_task_settings(act, policy, report)
    fix_sim_task_configs(act, policy, report)
    fix_camera_names(act, report)
    fix_tactile_ckpt(act, root, report)
    fix_backbone_fail_open(act, report)
    if not args.no_link_raw:
        link_raw(root, args.raw_config, report)

    print(f"\n{'Would apply' if args.dry_run else 'Applied'}:")
    print(report.render())
    print("\nCurrent state:")
    print(describe_state(root, act))

    if report.failed:
        print("\nSome fixes could not be applied -- see FAIL rows above.", file=sys.stderr)
        return 1

    print(
        "\nNext, from anywhere (the scripts self-locate now):\n"
        f"  bash {act}/process_data.sh <task> {args.raw_config} <n>\n"
        f"  bash {act}/train.sh <task> {args.raw_config} <n> <seed> <gpu_id> [train_config]\n"
        f"Revert with: python {Path(__file__).name} --univtac-root {root} --revert"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
