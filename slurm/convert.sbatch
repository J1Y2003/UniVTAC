#!/bin/bash
# Convert UniVTAC HDF5 demonstrations to the GR00T LeRobot v2 format, one job
# per (task, variant).
#
# CPU-only and CPU-heavy: every demonstration frame is decoded and re-encoded as
# one H.264 stream per camera per episode, which takes minutes to hours.
#
#   env TASK=insert_hole VARIANT=baseline_finetuned UNIVTAC_JOB_CONFIG=1 \
#     bundle-sbatch --job-kind data_process --code-git-root "$REPO_ROOT" -- \
#     --job-name=univtac-groot-convert-univtac-hdf5-demonstrations-to-lerobot-v2 \
#     --wckey=project-short-name:sub_4dpdata --partition=cpu -- \
#     slurm/convert.sbatch
#
# For every task at once, pass the array spec to the WRAPPER as separate tokens
# (`#SBATCH --array` is rejected): `bundle-sbatch ... --array 0-7 ...`.
#
# CPU-only, so --partition=cpu is required: the GPU partitions refuse a job that
# requests no GPU. It must be on the command line, since the launcher owns
# --partition and there is no #SBATCH directive to fall back on.
set -euo pipefail





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
# The same default as every other script here: a different value would convert
# to one place and train from another.
DATA_ROOT="${DATA_ROOT:-${HOME}/jaewon/workspace/groot-data}"
case "${DATA_ROOT}" in
  /*) ;;
  *) echo "error: DATA_ROOT must be an absolute path on SHARED storage, got '${DATA_ROOT}'." >&2
     echo "       The convert job and the training job run on different nodes." >&2
     exit 2 ;;
esac

# The converter needs numpy + h5py + pyarrow + imageio[ffmpeg]; it imports
# neither gr00t nor Isaac Sim, so any Python 3.10+ interpreter with
# requirements-convert.txt installed will do. Defaults to the UniVTAC env,
# which already has h5py for its own collection dumps.
CONVERT_PYTHON="${CONVERT_PYTHON:-${UNIVTAC_PYTHON:?set CONVERT_PYTHON or UNIVTAC_PYTHON}}"

# ARM= was the old name for this. Honour it rather than ignoring it: an
# unrecognised environment variable fails *silently*, so a stale
# `ARM=baseline_finetuned` command would quietly convert the tactile variant
# instead -- a wrong dataset that looks like a right one.
if [[ -n "${ARM:-}" && -z "${VARIANT:-}" ]]; then
  echo "[job] note: ARM is deprecated, use VARIANT (taking ARM=${ARM})" >&2
  VARIANT="${ARM}"
fi
VARIANT="${VARIANT:-tactile}"                       # tactile | baseline_finetuned
# Must match download_data.sbatch's RAW_CONFIG, which is what creates the
# data/<task>/<config> symlink the converter reads. Both default to `clean`;
# this used to default to `demo`, so a downloaded task failed to convert with a
# missing-directory error that pointed nowhere useful.
TASK_CONFIG="${TASK_CONFIG:-clean}"
# How many TRAINING episodes to convert; empty means all 100. Prefer
# CONVERT_EPISODES: the eval jobs use EPISODES for evaluation ROLLOUTS and the
# job inherits whatever the submitting shell holds.
CONVERT_EPISODES="${CONVERT_EPISODES:-${EPISODES:-}}"
EPISODES="${CONVERT_EPISODES}"
TACTILE_MODE="${TACTILE_MODE:-depth_pool}"
POOL_ROWS="${POOL_ROWS:-8}"
POOL_COLS="${POOL_COLS:-6}"
FPS="${FPS:-20}"

# The eight benchmark tasks, indexable by SLURM_ARRAY_TASK_ID.
TASKS=(lift_bottle lift_can insert_HDMI insert_hole insert_tube pull_out_key put_bottle_in_shelf grasp_classify)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
else
  TASK="${TASK:?set TASK, or submit with --array=0-7 to sweep all of them}"
fi

RAW_DIR="${RAW_DIR:-${UNIVTAC_ROOT}/data/${TASK}/${TASK_CONFIG}}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/univtac-${TASK}-${VARIANT}}"

mkdir -p logs "$(dirname "${OUT_DIR}")"

# Keep BLAS/ffmpeg thread pools inside the CPU allocation. numpy defaulting to
# every core on the node is the classic way a "CPU-only" job disrupts a shared
# compute node.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-4}}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-4}}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-4}}}"

echo "=========================================================="
echo " task=${TASK}  variant=${VARIANT}  config=${TASK_CONFIG}"
echo " raw=${RAW_DIR}"
echo " out=${OUT_DIR}"
echo " tactile_mode=${TACTILE_MODE} pool=${POOL_ROWS}x${POOL_COLS} fps=${FPS}"
echo " node=$(hostname)  cpus=${SLURM_CPUS_ON_NODE:-?}  job=${SLURM_JOB_ID:-none}"
echo "=========================================================="

if [[ ! -d "${RAW_DIR}" ]]; then
  echo "[job] no raw demonstrations at ${RAW_DIR}" >&2
  echo "[job] fetch them with:" >&2
  echo "[job]   env TASKS=${TASK} UNIVTAC_JOB_CONFIG=1 bundle-sbatch --job-kind data_process \\" >&2
  echo "[job]     --code-git-root ${REPO_ROOT} -- --job-name=univtac-groot-download-univtac-demonstration-data-from-modelscope \\" >&2
  echo "[job]     --wckey=project-short-name:sub_4dpdata --partition=cpu -- slurm/download_data.sbatch" >&2
  echo "[job] or collect them: bash collect_data.sh ${TASK} ${TASK_CONFIG} 0" >&2
  exit 1
fi

# The converter needs numpy + h5py + pyarrow + imageio[ffmpeg] + cv2. Installing
# the missing ones into the UniVTAC env would re-pin packages under a working
# Isaac Lab install, and nothing here imports torch or Isaac Sim, so use a
# private venv.
MISSING=$("${CONVERT_PYTHON}" - <<'PYCHECK' 2>/dev/null || echo "IMPORT_CHECK_FAILED"
mods = {"numpy": "numpy", "h5py": "h5py", "pyarrow": "pyarrow",
        "imageio": "imageio", "cv2": "opencv-python-headless"}
missing = []
for mod, pkg in mods.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)
print(" ".join(missing))
PYCHECK
)
if [[ -n "${MISSING}" && "${MISSING}" != "IMPORT_CHECK_FAILED" ]]; then
  CONVERT_VENV="${CONVERT_VENV:-${REPO_ROOT}/.venv-convert}"
  echo "[job] ${CONVERT_PYTHON} is missing: ${MISSING}"
  if [[ "${NO_AUTO_VENV:-0}" == "1" ]]; then
    echo "error: install them yourself, or unset NO_AUTO_VENV to let this job" >&2
    echo "       build a private venv at ${CONVERT_VENV}." >&2
    echo "       Do NOT install into the shared base conda env." >&2
    exit 1
  fi
  # Every job resolves CONVERT_VENV to the same path, and these run
  # concurrently. `python -m venv` writes bin/python BEFORE pip installs
  # anything, so an existence check is not a valid guard -- the other jobs
  # would skip the build and run against a venv with no pyarrow. Serialise on a
  # lock and gate on the imports working.
  _venv_ready() {
    [[ -x "${CONVERT_VENV}/bin/python" ]] &&
      "${CONVERT_VENV}/bin/python" -c \
        'import numpy, h5py, pyarrow, imageio, cv2' 2>/dev/null
  }

  _build_venv() {
    echo "[job] building a private venv at ${CONVERT_VENV}"
    echo "[job] (isolated on purpose -- the UniVTAC conda env is left untouched)"
    rm -rf "${CONVERT_VENV}"
    "${CONVERT_PYTHON}" -m venv "${CONVERT_VENV}" || {
      echo "error: could not create a venv with ${CONVERT_PYTHON}." >&2
      return 1
    }
    "${CONVERT_VENV}/bin/python" -m pip install --quiet --upgrade pip
    "${CONVERT_VENV}/bin/python" -m pip install --quiet --only-binary=:all: \
      -r "${REPO_ROOT}/requirements-convert.txt" || {
      echo "error: pip install -r requirements-convert.txt failed -- see above." >&2
      return 1
    }
    _venv_ready || {
      echo "error: ${CONVERT_VENV} still cannot import the converter's deps" >&2
      echo "       after a successful pip install. Inspect it by hand." >&2
      return 1
    }
  }

  if ! _venv_ready; then
    # flock holds the lock for as long as fd 9 is open, so a job that waits
    # here resumes only once the builder has finished (or died). Losers then
    # re-test _venv_ready and use what the winner built.
    _LOCK="${CONVERT_VENV}.lock"
    if command -v flock >/dev/null 2>&1; then
      exec 9>"${_LOCK}"
      echo "[job] waiting for the ${CONVERT_VENV} lock"
      flock 9
      _venv_ready || _build_venv || exit 1
      flock -u 9
      exec 9>&-
    else
      # No flock: fall back to a private venv per job. Wastes a few hundred MB
      # per task but cannot race, which matters more.
      CONVERT_VENV="${CONVERT_VENV}-${TASK}-${VARIANT}"
      echo "[job] no flock available; using a per-job venv ${CONVERT_VENV}"
      _venv_ready || _build_venv || exit 1
    fi
  fi
  CONVERT_PYTHON="${CONVERT_VENV}/bin/python"
  echo "[job] using ${CONVERT_PYTHON}"
fi

CMD=(
  "${CONVERT_PYTHON}" "${REPO_ROOT}/scripts/convert_univtac_to_lerobot.py"
  --task "${TASK}"
  --raw-dir "${RAW_DIR}"
  --out "${OUT_DIR}"
  --variant "${VARIANT}"
  --univtac-root "${UNIVTAC_ROOT}"
  --fps "${FPS}"
)
if [[ "${VARIANT}" == "tactile" ]]; then
  CMD+=(--tactile-mode "${TACTILE_MODE}" --tactile-pool-grid "${POOL_ROWS}" "${POOL_COLS}")
fi
if [[ -n "${EPISODES}" ]]; then
  CMD+=(--episodes "${EPISODES}")
fi

echo "[job] ${CMD[*]}"
cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" "${CMD[@]}"

echo "[job] converted dataset: ${OUT_DIR}"
echo "[job] finetune with:"
echo "  TASKS=${TASK} bash ${REPO_ROOT}/slurm/submit_benchmark.sh   # finetune it"
