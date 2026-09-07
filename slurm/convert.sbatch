#!/bin/bash
# Convert UniVTAC HDF5 demonstrations to the GR00T LeRobot v2 format.
#
# CPU-only and CPU-heavy: it decodes every demonstration frame and re-encodes
# one H.264 stream per camera per episode, which takes minutes to hours. That is
# firmly over the "run it on a login node" line, hence this wrapper. No GPU is
# requested -- nothing here touches CUDA.
#
# One job per (task, variant). The state layout differs between variants, so each variant
# needs its own dataset; see docs/ABLATION.md.
#
#   sbatch --export=ALL,TASK=insert_hole,VARIANT=tactile slurm/convert.sbatch
#
# Or sweep every task as an array (one task per array index):
#
#   sbatch --array=0-7 --export=ALL,VARIANT=tactile slurm/convert.sbatch
#
# No --time here on purpose. Requesting more than a partition's limit leaves the
# job pending forever with REASON=PartitionTimeLimit rather than failing fast, so
# the partition's own default is safer. Check the limits with:
#   sinfo -o "%20P %10l %10L %6D %25G"     (%l = TIMELIMIT, %L = DEFAULTTIME)
# and pass --partition explicitly, or export SBATCH_PARTITION. Note `debug` is
# short-lived and is where jobs land by default here -- it is not where long
# training belongs.
#SBATCH --job-name=univtac-groot-convert-univtac-hdf5-demonstrations-to-lerobot-v2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Locate the repo. `dirname "${BASH_SOURCE[0]}"` is NOT reliable here: under
# sbatch, SLURM copies the batch script into a spool directory
# (/var/spool/slurm/d/...), so it resolves to the spool, not the checkout --
# every derived path then points somewhere unwritable. SLURM_SUBMIT_DIR is the
# directory the job was submitted from, which is the right answer under sbatch;
# BASH_SOURCE is the right answer under plain `bash`. Try both and accept only a
# candidate that actually contains this repo.
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
UNIVTAC_ROOT="${UNIVTAC_ROOT:?set UNIVTAC_ROOT to the UniVTAC checkout}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to where converted datasets should land}"

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
TASK_CONFIG="${TASK_CONFIG:-demo}"          # the collection config the demos used
EPISODES="${EPISODES:-}"                    # empty = all available
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

echo "=========================================================="
echo " task=${TASK}  variant=${VARIANT}  config=${TASK_CONFIG}"
echo " raw=${RAW_DIR}"
echo " out=${OUT_DIR}"
echo " tactile_mode=${TACTILE_MODE} pool=${POOL_ROWS}x${POOL_COLS} fps=${FPS}"
echo " node=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  job=${SLURM_JOB_ID:-none}"
echo "=========================================================="

if [[ ! -d "${RAW_DIR}" ]]; then
  echo "[job] no raw demonstrations at ${RAW_DIR}" >&2
  echo "[job] fetch them with: bash slurm/download_data.sbatch  (TASKS=${TASK})" >&2
  echo "[job] or collect them: bash collect_data.sh ${TASK} ${TASK_CONFIG} 0" >&2
  exit 1
fi

# The converter needs numpy + h5py + pyarrow + imageio[ffmpeg] + cv2. The
# UniVTAC env has numpy/h5py/cv2 for its own collection dumps but typically not
# pyarrow or imageio-ffmpeg, and installing them there would re-pin packages
# under a working Isaac Lab + TacEx install. Nothing here imports torch, gr00t
# or Isaac Sim, so a private venv is both safe and sufficient.
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
  if [[ ! -x "${CONVERT_VENV}/bin/python" ]]; then
    echo "[job] building a private venv at ${CONVERT_VENV}"
    echo "[job] (isolated on purpose -- the UniVTAC conda env is left untouched)"
    "${CONVERT_PYTHON}" -m venv "${CONVERT_VENV}" || {
      echo "error: could not create a venv with ${CONVERT_PYTHON}." >&2
      exit 1
    }
    "${CONVERT_VENV}/bin/python" -m pip install --quiet --upgrade pip
    "${CONVERT_VENV}/bin/python" -m pip install --quiet --only-binary=:all: \
      -r "${REPO_ROOT}/requirements-convert.txt" || {
      echo "error: pip install -r requirements-convert.txt failed -- see above." >&2
      exit 1
    }
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
echo "  sbatch --export=ALL,VARIANT=${VARIANT},DATASET=${OUT_DIR} ${REPO_ROOT}/slurm/finetune.sbatch"
