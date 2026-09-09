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
#   env TASK=insert_hole VARIANT=tactile UNIVTAC_JOB_CONFIG=1 bundle-sbatch #     --job-kind data_process --code-git-root "$REPO_ROOT" -- #     --job-name=univtac-groot-convert-univtac-hdf5-demonstrations-to-lerobot-v2 #     --wckey=project-short-name:sub_4dpdata --partition=cpu -- #     slurm/convert.sbatch
#
# Or sweep every task as an array (one task per array index). The SPEC goes to
# the WRAPPER, as separate tokens, and `#SBATCH --array` is forbidden:
#
#   ... bundle-sbatch --job-kind data_process --array 0-7 --code-git-root ...
#
# SITE RULES (see CLAUDE.md, "Cluster rules"). Enforced here; do not "helpfully"
# reintroduce any of them:
#
#   * --time is OPTIONAL and worth passing. Without it a job is assumed to want
#     the partition maximum (2 days), so backfill can only start it in a 2-day
#     gap; a realistic limit makes it eligible for far more gaps and it starts
#     sooner. Two rules: never exceed the partition maximum, or the job sits
#     pending forever with REASON=PartitionTimeLimit, and never set one below
#     what the work needs unless that work is resumable.
#   * NEVER pass --cpus-per-task or --mem. Jobs take the node's per-GPU
#     defaults, and the submit filter rejects the job if either is specified.
#   * ALWAYS pass --wckey=project-short-name:sub_4dpdata.
#   * MODEL_OUTPUT_DIR is injected by bundle-sbatch and equals CODE_OUTPUT_DIR
#     (<bundle>/code-output). Never set it, never pass it with --export, and do
#     not expect it under /rlwrld-unified-checkpoints any more.
#
# Do pass --partition explicitly, or export SBATCH_PARTITION: `debug` is the
# default here and is short-lived, so it is not where long training belongs.
#
# CPU-ONLY JOB: this must go to --partition=cpu. The GPU partitions reject any
# job that does not request a GPU:
#   ❌ ERROR: GPU 파티션에는 GPU를 요청한 잡만 제출할 수 있습니다.
#      CPU 전용 잡은 --partition=cpu 를 사용하세요.
# The header below sets it, but --partition precedence is
# command line > SBATCH_PARTITION > #SBATCH directive, so an exported
# SBATCH_PARTITION pointing at a GPU partition silently overrides it and the
# submission is refused. Pass --partition=cpu on the command line too.

set -euo pipefail

# bundle-sbatch owns --export, so this job's configuration is not handed to it
# as a variable list: it arrives only because the submitter exported it and the
# launcher's environment is inherited. If that ever stops holding, the variables
# below would quietly fall back to their defaults and the job would do the WRONG
# WORK while looking healthy. The submitters set this sentinel; a real job
# without it stops here. A local `bash` run for a dry run is unaffected.
if [[ -n "${SLURM_JOB_ID:-}" && -z "${UNIVTAC_JOB_CONFIG:-}" ]]; then
  echo "error: UNIVTAC_JOB_CONFIG is unset, so this job's environment did not" >&2
  echo "       reach it and its configuration would silently take defaults." >&2
  echo "       Submit through the scripts in slurm/, which export it, or set" >&2
  echo "       UNIVTAC_JOB_CONFIG=1 alongside the rest of the configuration." >&2
  exit 2
fi


# --------------------------------------------------------------------------- #
# The site wckey, checked at runtime
# --------------------------------------------------------------------------- #
# sbatch precedence is: command line > environment variable > #SBATCH directive.
# So a stale SBATCH_WCKEY in the submitting shell (typically a copy of
# env.example.sh from before the value was corrected) SILENTLY OVERRIDES the
# `#SBATCH --wckey=project-short-name:sub_4dpdata` header above. That is exactly how a job once went
# out under the wrong project. The header alone is not enough; check it.
#
# SLURM_WCKEY is only set inside the job when a wckey was applied, so an unset
# value defaults to the right answer rather than failing a valid job.
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_WCKEY:-project-short-name:sub_4dpdata}" != "project-short-name:sub_4dpdata" ]]; then
  echo "error: this job's wckey is '${SLURM_WCKEY}', not 'project-short-name:sub_4dpdata'." >&2
  echo "       Every job here must be accounted to 'project-short-name:sub_4dpdata'." >&2
  echo "       Something in your environment overrode the #SBATCH header --" >&2
  echo "       almost certainly a stale SBATCH_WCKEY in env.sh. Fix it with:" >&2
  echo "         export SBATCH_WCKEY=project-short-name:sub_4dpdata SLURM_WCKEY=project-short-name:sub_4dpdata" >&2
  echo "       then resubmit. 'python scripts/preflight.py' also checks this." >&2
  exit 2
fi


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
# Same default as submit_benchmark.sh, smoke_test.sh and
# benchmark_task.sbatch. This used to be a hard `:?` requirement while
# every other script defaulted it, so forgetting to export it failed the
# convert job after a queue wait -- and, worse, exporting a DIFFERENT value
# here than the trainer uses would convert to one place and train from
# another. One default, everywhere.
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
# How many TRAINING episodes to convert. Empty = all available, which is the
# full released 100 per task -- that is the intended setting.
#
# NOTE the name collision: the eval jobs (eval_ablation.sbatch,
# benchmark_task.sbatch) use EPISODES for the number of EVALUATION
# ROLLOUTS, and the job inherits whatever the submitting shell holds. So
# prefer CONVERT_EPISODES here; EPISODES is still honoured for compatibility.
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
  # CONCURRENCY: every (task, variant) job resolves CONVERT_VENV to the SAME
  # path, and the intended way to run this script is several jobs at once (a
  # per-task loop, or --array). `python -m venv` writes bin/python BEFORE pip
  # installs anything, so an existence check alone is not a valid guard: the
  # first job starts building, the others see bin/python, skip the build, and
  # run the converter against a venv with no pyarrow in it. That is how a
  # three-task loop converted exactly one task. So: serialise on a lock, and
  # gate on the imports actually working rather than on a file existing.
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
