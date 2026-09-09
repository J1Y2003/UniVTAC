#!/bin/bash
# Download UniVTAC demonstration data by delegating to the simulator's own
# $UNIVTAC_ROOT/data/download.sh, which owns the include patterns, task-name
# validation and the isaac45/isaac51 split.
#
# The full dataset is ~400 GB. Only isaac45 is needed to finetune GR00T, at
# ~24 GB per task; the default here is one task plus the checkpoints.
#
#   env TASKS=lift_bottle UNIVTAC_JOB_CONFIG=1 \
#     bundle-sbatch --job-kind data_process --code-git-root "$REPO_ROOT" -- \
#     --job-name=univtac-groot-download-univtac-demonstration-data-from-modelscope \
#     --wckey=project-short-name:sub_4dpdata --partition=cpu -- \
#     slurm/download_data.sbatch
#
# TASKS="a b" for several, TASKS=all for all eight (~190 GB), TASKS=none
# CHECKPOINT=1 for the encoder alone. modelscope resumes, so resubmitting after
# a walltime kill is safe.
#
# CPU-only, so --partition=cpu is required: the GPU partitions refuse a job that
# requests no GPU. It must be on the command line, since the launcher owns
# --partition and there is no #SBATCH directive to fall back on.
set -uo pipefail





# sbatch copies this script into /var/spool/slurm/d/, so BASH_SOURCE resolves to
# the spool and every derived path lands somewhere unwritable. SLURM_SUBMIT_DIR
# is the right answer under sbatch, BASH_SOURCE under plain `bash`; accept
# whichever actually contains this repo.
if [[ -z "${REPO_ROOT:-}" ]]; then
  for _candidate in \
      "${SLURM_SUBMIT_DIR:-}" \
      "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"; do
    if [[ -n "${_candidate}" && -f "${_candidate}/univtac_groot/spec.py" ]]; then
      REPO_ROOT="${_candidate}"
      break
    fi
  done
fi
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "error: cannot locate the univtac-groot checkout." >&2
  echo "       Tried SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-unset}' and the" >&2
  echo "       script's own directory. Submit from the repo root, or pass" >&2
  echo "       REPO_ROOT=/path/to/UniVTAC explicitly." >&2
  exit 2
fi

# Both runtime guards -- the wckey and the configuration sentinel -- live in one
# place. shellcheck source=slurm/common.sh
. "${REPO_ROOT}/slurm/common.sh"
univtac_job_guard
UNIVTAC_ROOT="${UNIVTAC_ROOT:?set UNIVTAC_ROOT to the UniVTAC checkout}"

TASKS="${TASKS:-lift_bottle}"        # space-separated, or "all", or "none"
VERSION="${VERSION:-45}"             # 45 matches Isaac Sim 4.5 on the main branch
CHECKPOINT="${CHECKPOINT:-1}"        # 1 = also fetch checkpoints/ (includes encoder.pth)
CONTACT="${CONTACT:-0}"              # 1 = also fetch the contact pretraining set
WORKERS="${WORKERS:-8}"
# Default to the checkout's own data/, which is where _base_data_preprocessor.py
# anchors DATA_ROOT_PATH. Overriding this means the ACT pipeline will not see it.
OUTPUT="${OUTPUT:-${UNIVTAC_ROOT}/data}"
# Collect-config name to expose the downloaded episodes under. Must match the
# second argument you later pass to process_data.sh / eval_policy.sh.
RAW_CONFIG="${RAW_CONFIG:-clean}"
PYTHON="${PYTHON:-python3}"

DOWNLOADER="${UNIVTAC_ROOT}/data/download.sh"
if [[ ! -f "${DOWNLOADER}" ]]; then
  echo "error: ${DOWNLOADER} not found." >&2
  echo "       Is UNIVTAC_ROOT correct, and is the checkout recent enough to" >&2
  echo "       ship data/download.sh? Older clones do not have it." >&2
  exit 1
fi

ALL_TASKS="lift_bottle lift_can insert_HDMI insert_hole insert_tube pull_out_key put_bottle_in_shelf grasp_classify"
[[ "${TASKS}" == "all" ]] && TASKS="${ALL_TASKS}"
[[ "${TASKS}" == "none" ]] && TASKS=""

mkdir -p logs "${OUTPUT}"

echo "=========================================================="
echo " univtac_root=${UNIVTAC_ROOT}"
echo " tasks=${TASKS:-<none>}  version=${VERSION}"
echo " checkpoint=${CHECKPOINT}  contact=${CONTACT}"
echo " output=${OUTPUT}  raw_config=${RAW_CONFIG}"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "=========================================================="
df -h "${OUTPUT}" || true

# download.sh does `require_command modelscope`, i.e. it needs the CLI on PATH.
# Downloading needs nothing from Isaac Lab, so rather than installing modelscope
# into the UniVTAC conda env (where it would re-pin requests/urllib3/tqdm under
# a working Isaac Lab + TacEx install) or into ~/.local (which every python3.10
# process on this account would then see), keep it in a throwaway venv. Delete
# DOWNLOAD_VENV any time; it is rebuilt on demand.
if ! command -v modelscope >/dev/null 2>&1; then
  DOWNLOAD_VENV="${DOWNLOAD_VENV:-${REPO_ROOT}/.venv-modelscope}"
  if [[ "${NO_AUTO_VENV:-0}" == "1" ]]; then
    echo "error: 'modelscope' is not on PATH and NO_AUTO_VENV=1." >&2
    echo "       Provide it yourself, or unset NO_AUTO_VENV to let this job" >&2
    echo "       build a private venv at ${DOWNLOAD_VENV}." >&2
    exit 1
  fi
  if [[ ! -x "${DOWNLOAD_VENV}/bin/modelscope" ]]; then
    echo "[job] 'modelscope' not found; building a private venv at ${DOWNLOAD_VENV}"
    echo "[job] (isolated on purpose -- nothing is installed into the UniVTAC"
    echo "[job]  conda env, ~/.local, or the shared base env)"
    "${PYTHON}" -m venv "${DOWNLOAD_VENV}" || {
      echo "error: could not create a venv with ${PYTHON}." >&2
      echo "       Set PYTHON to an interpreter that has the venv module." >&2
      exit 1
    }
    "${DOWNLOAD_VENV}/bin/python" -m pip install --quiet --upgrade pip
    "${DOWNLOAD_VENV}/bin/python" -m pip install --quiet modelscope || {
      echo "error: pip install modelscope failed -- see above." >&2
      exit 1
    }
  fi
  export PATH="${DOWNLOAD_VENV}/bin:${PATH}"
  echo "[job] using modelscope from ${DOWNLOAD_VENV}"
fi
echo "[job] modelscope: $(command -v modelscope)"

# Build the selector list. download.sh accepts repeated selectors, and refuses
# --version unless --task is among them, so only pass it when tasks are wanted.
ARGS=()
for task in ${TASKS}; do
  ARGS+=(--task "${task}")
done
if [[ -n "${TASKS}" ]]; then
  ARGS+=(--version "${VERSION}")
fi
[[ "${CHECKPOINT}" == "1" ]] && ARGS+=(--checkpoint)
[[ "${CONTACT}" == "1" ]] && ARGS+=(--contact)
if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "error: nothing selected. Set TASKS, or CHECKPOINT=1, or CONTACT=1." >&2
  exit 2
fi
ARGS+=(--output "${OUTPUT}" --workers "${WORKERS}")

echo "[job] running: bash ${DOWNLOADER} ${ARGS[*]}"
cd "${UNIVTAC_ROOT}"
bash "${DOWNLOADER}" "${ARGS[@]}"
STATUS=$?
echo "[job] download.sh exited ${STATUS}"

echo "----------------------------------------------------------"
for task in ${TASKS}; do
  count=$(find "${OUTPUT}/isaac${VERSION}/${task}" -name '*.hdf5' 2>/dev/null | wc -l)
  echo "[job] ${task}: ${count} episodes on disk (expect 100)"
done
du -sh "${OUTPUT}" 2>/dev/null || true

# download.sh preserves the published layout (isaac45/<task>/hdf5/{0..99}.hdf5)
# but BaseDataPreprocessor reads data/<task>/<config>/, and no <config> level
# exists in the download. The episodes are already named 0.hdf5..99.hdf5, which
# find_episodes sorts by int(stem), so one directory symlink per task bridges
# the two and costs no disk. isaac45 wins where both exist: `main` installs
# Isaac Sim 4.5 and the released checkpoints target that data.
if [[ ${STATUS} -eq 0 && -n "${TASKS}" ]]; then
  echo "----------------------------------------------------------"
  echo "[job] linking downloaded episodes into data/<task>/${RAW_CONFIG}"
  for hdf5_dir in "${UNIVTAC_ROOT}"/data/isaac45/*/hdf5 \
                  "${UNIVTAC_ROOT}"/data/isaac51/*/hdf5; do
    [[ -d "${hdf5_dir}" ]] || continue
    compgen -G "${hdf5_dir}/*.hdf5" >/dev/null || continue
    task_dir="$(dirname "${hdf5_dir}")"
    task="$(basename "${task_dir}")"
    version="$(basename "$(dirname "${task_dir}")")"
    link="${UNIVTAC_ROOT}/data/${task}/${RAW_CONFIG}"
    if [[ -e "${link}" || -L "${link}" ]]; then
      echo "[job]   ${task}: already linked"
      continue
    fi
    mkdir -p "$(dirname "${link}")"
    # Relative, so the tree survives being moved: from data/<task>/, `..` is
    # data/, hence ../<version>/<task>/hdf5.
    if ln -s "../${version}/${task}/hdf5" "${link}"; then
      echo "[job]   ${task}: data/${task}/${RAW_CONFIG} -> ../${version}/${task}/hdf5"
    else
      # No symlinks on this filesystem. Copying 24 GB per task is not an
      # acceptable fallback, so say so rather than silently duplicating.
      echo "[job]   ${task}: SYMLINK FAILED -- bind-mount or move it by hand" >&2
    fi
  done
fi

echo "----------------------------------------------------------"
echo "[job] next steps:"
for task in ${TASKS}; do
  echo "  env TASK=${task} VARIANT=tactile RAW_DIR=${OUTPUT}/isaac${VERSION}/${task}/hdf5 UNIVTAC_JOB_CONFIG=1 \\"
  echo "    bundle-sbatch --job-kind data_process --code-git-root ${REPO_ROOT} -- \\"
  echo "    --job-name=univtac-groot-convert-univtac-hdf5-demonstrations-to-lerobot-v2 \\"
  echo "    --wckey=project-short-name:sub_4dpdata --partition=cpu -- slurm/convert.sbatch"
done
exit ${STATUS}
