#!/bin/bash
# One-liner submitter for the full ablation. Run from the repo root:
#
#   bash slurm/submit_overnight.sh                 # submit
#   bash slurm/submit_overnight.sh --print         # show the sbatch command only
#   bash slurm/submit_overnight.sh --test-only     # run the submit filter, queue nothing
#   bash slurm/submit_overnight.sh --dry           # DRY_RUN=1 locally, no sbatch at all
#
# Everything below is an overridable default, so a changed path is one variable
# and not a rewritten command line:
#
#   GPUS=4 TASK=insert_hole bash slurm/submit_overnight.sh
#
# Why a wrapper: the `sbatch --export=ALL,...` form needs seven absolute paths
# plus four site-specific flags, which is unreadable to type and easy to get
# subtly wrong -- and a wrong path there fails minutes into a queued job rather
# than here.

set -uo pipefail

# --------------------------------------------------------------------------- #
# Paths -- override any of these in the environment
# --------------------------------------------------------------------------- #
WORKSPACE="${WORKSPACE:-${HOME}/jaewon/workspace}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UNIVTAC_ROOT="${UNIVTAC_ROOT:-${WORKSPACE}/UniVTAC-sim}"
GROOT_ROOT="${GROOT_ROOT:-${WORKSPACE}/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:-${HOME}/miniconda3/envs/UniVTAC/bin/python}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/groot-data}"
HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"
# The submit filter requires this under /rlwrld-unified-checkpoints, and it is
# also where the ~120 GB of checkpoints per variant belongs rather than on the
# shared home mount.
# Shape follows the training-outputs policy: {NFS}/{user}/checkpoints/<job>.
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/${USER}/checkpoints/univtac-groot}"

# --------------------------------------------------------------------------- #
# Job shape
# --------------------------------------------------------------------------- #
# `debug` is the cluster default and caps at 3 hours, which is why training jobs
# sit pending with REASON=PartitionTimeLimit. Name a 2-day partition instead.
PARTITION="${PARTITION:-sjw_alinlab}"
# 2 GPUs schedules far sooner than 4 on partially-allocated nodes, and 4 vs 2
# changes the effective batch size rather than just the speed -- see the recipe
# pinning in overnight_ablation.sbatch. Whatever you pick is locked in for both
# variants once the first one trains.
GPUS="${GPUS:-2}"
# Honest under-request: shorter jobs fit backfill gaps a 2-day job cannot, and a
# walltime kill is safe here because training resumes from its last checkpoint.
TIMELIMIT="${TIMELIMIT:-1-00:00:00}"

TASK="${TASK:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
EPISODES="${EPISODES:-50}"

MODE="${1:-submit}"

# --------------------------------------------------------------------------- #
# Check before submitting -- these are the failures that cost a queued job
# --------------------------------------------------------------------------- #
FAIL=0
check_dir()  { [[ -d "$2" ]] || { echo "  MISSING dir   $1=$2" >&2; FAIL=1; }; }
check_exec() { [[ -x "$2" ]] || { echo "  NOT EXECUTABLE $1=$2" >&2; FAIL=1; }; }

echo "Resolved configuration:"
printf '  %-16s %s\n' \
  REPO_ROOT "${REPO_ROOT}" \
  UNIVTAC_ROOT "${UNIVTAC_ROOT}" \
  GROOT_ROOT "${GROOT_ROOT}" \
  GROOT_PYTHON "${GROOT_PYTHON}" \
  UNIVTAC_PYTHON "${UNIVTAC_PYTHON}" \
  DATA_ROOT "${DATA_ROOT}" \
  HF_HOME "${HF_HOME}" \
  MODEL_OUTPUT_DIR "${MODEL_OUTPUT_DIR}" \
  partition "${PARTITION}" \
  gpus "${GPUS}" \
  time "${TIMELIMIT}" \
  task "${TASK}/${TASK_CONFIG}" \
  episodes "${EPISODES}" \
  wandb "${WANDB_API_KEY:+online}${WANDB_API_KEY:-offline (no WANDB_API_KEY)}"
echo

echo "Checks:"
check_dir  REPO_ROOT      "${REPO_ROOT}"
check_dir  UNIVTAC_ROOT   "${UNIVTAC_ROOT}"
check_dir  GROOT_ROOT     "${GROOT_ROOT}"
check_exec GROOT_PYTHON   "${GROOT_PYTHON}"
check_exec UNIVTAC_PYTHON "${UNIVTAC_PYTHON}"
check_dir  DATA_ROOT      "${DATA_ROOT}"
for variant in tactile baseline_finetuned; do
  dataset="${DATA_ROOT}/univtac-${TASK}-${variant}"
  n=$(find "${dataset}/data" -name '*.parquet' 2>/dev/null | wc -l)
  if [[ "${n}" -gt 0 ]]; then
    echo "  ok  ${variant}: ${n} parquet in ${dataset}"
  else
    echo "  MISSING dataset ${dataset} (convert it first: slurm/convert.sbatch)" >&2
    FAIL=1
  fi
done
# SLURM opens the --output file before the script runs, so a missing logs/ kills
# the job instantly with no log to explain it.
mkdir -p "${REPO_ROOT}/logs" || FAIL=1
[[ -d "${REPO_ROOT}/logs" ]] && echo "  ok  logs/ exists"
if [[ "${FAIL}" -ne 0 ]]; then
  echo >&2
  echo "Fix the above and re-run. Nothing was submitted." >&2
  exit 2
fi
echo

# --------------------------------------------------------------------------- #
# Build and run
# --------------------------------------------------------------------------- #
EXPORTS="ALL"
EXPORTS+=",MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR}"
EXPORTS+=",UNIVTAC_ROOT=${UNIVTAC_ROOT}"
EXPORTS+=",UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"
EXPORTS+=",GROOT_ROOT=${GROOT_ROOT}"
EXPORTS+=",GROOT_PYTHON=${GROOT_PYTHON}"
EXPORTS+=",HF_HOME=${HF_HOME}"
EXPORTS+=",DATA_ROOT=${DATA_ROOT}"
EXPORTS+=",TASK=${TASK}"
EXPORTS+=",TASK_CONFIG=${TASK_CONFIG}"
EXPORTS+=",EPISODES=${EPISODES}"
# DRY_RUN must be pinned off: if it is exported in the calling shell (from
# testing), --export=ALL would carry it in and the job would exit in seconds
# having trained nothing.
EXPORTS+=",DRY_RUN=0"

SBATCH_ARGS=(
  --partition="${PARTITION}"
  --gres="gpu:${GPUS}"
  --time="${TIMELIMIT}"
  --export="${EXPORTS}"
)

case "${MODE}" in
  --dry)
    echo "Running preflight locally (DRY_RUN=1), submitting nothing:"
    env MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" UNIVTAC_ROOT="${UNIVTAC_ROOT}" \
        UNIVTAC_PYTHON="${UNIVTAC_PYTHON}" GROOT_ROOT="${GROOT_ROOT}" \
        GROOT_PYTHON="${GROOT_PYTHON}" HF_HOME="${HF_HOME}" DATA_ROOT="${DATA_ROOT}" \
        TASK="${TASK}" TASK_CONFIG="${TASK_CONFIG}" EPISODES="${EPISODES}" \
        DRY_RUN=1 bash "${REPO_ROOT}/slurm/overnight_ablation.sbatch"
    ;;
  --print)
    echo "sbatch ${SBATCH_ARGS[*]} slurm/overnight_ablation.sbatch"
    ;;
  --test-only)
    sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_ROOT}/slurm/overnight_ablation.sbatch"
    ;;
  submit)
    cd "${REPO_ROOT}"
    sbatch "${SBATCH_ARGS[@]}" slurm/overnight_ablation.sbatch
    status=$?
    if [[ ${status} -eq 0 ]]; then
      echo
      echo "Watch it with:"
      echo "  squeue -u ${USER}                       # ST=R means running"
      echo "  tail -f logs/univtac-groot-full-*.out   # preflight, then stages"
      echo "  tail -f logs/overnight-*/finetune-tactile.log"
    fi
    exit ${status}
    ;;
  *)
    echo "usage: bash slurm/submit_overnight.sh [--print|--test-only|--dry]" >&2
    exit 2
    ;;
esac
