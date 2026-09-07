#!/bin/bash
# One-liner submitter for the full ablation. Run from the repo root:
#
#   bash slurm/submit_overnight.sh                 # submit
#   bash slurm/submit_overnight.sh --print         # show the sbatch command only
#   bash slurm/submit_overnight.sh --test-only     # run the submit filter, queue nothing
#   bash slurm/submit_overnight.sh --dry           # DRY_RUN=1 locally, no sbatch at all
#   bash slurm/submit_overnight.sh --chunks 8      # 8 chained 3h jobs on `debug`
#
# Use --chunks when the long-walltime partitions are contended. The 2-day
# partitions have the walltime but not the availability; eight 3-hour `debug`
# slots that actually start beat one 24-hour slot that pends all night. Chunks
# are chained with --dependency=afterany, so each starts when the previous ends
# and resumes from its last checkpoint.
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

# ${USER} is not guaranteed to be exported; under `set -u` a bare use is fatal.
WHOAMI="${USER:-$(id -un)}"

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
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/${USER:-$(id -un)}/univtac-groot}"

MODE="${1:-submit}"

# --------------------------------------------------------------------------- #
# Job shape
# --------------------------------------------------------------------------- #
# Chunked mode (--chunks N): submit N short jobs chained by SLURM dependencies,
# each resuming where the last was killed. The 2-day partitions have the
# walltime but not the availability -- an 8-hour wait for a 24-hour slot is
# worse than eight 3-hour slots that actually start -- while `debug` is the
# cluster default, so it is the least contended and the quickest to schedule.
#
# This works only because every stage is resumable: finetuning continues via
# --resume-from-checkpoint, and completed stages are skipped by their markers.
if [[ "${MODE}" == "--chunks" ]]; then
  PARTITION="${PARTITION:-debug}"
  # Just inside debug's 3:00:00 cap. Slightly under so the request is never
  # rejected for exceeding it, and so backfill has a little more room.
  TIMELIMIT="${TIMELIMIT:-2:55:00}"
  # The critical parameter for chunking. If a chunk is killed before its first
  # save, it makes ZERO progress and the chain spins forever. We do not yet know
  # the step rate with cuDNN disabled, so save often enough that even a slow
  # chunk banks something; raise it once you can read steps/sec off wandb.
  SAVE_STEPS="${SAVE_STEPS:-250}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
else
  # `debug` is the cluster default and caps at 3 hours, which is why training
  # jobs sit pending with REASON=PartitionTimeLimit. Name a 2-day partition.
  PARTITION="${PARTITION:-sjw_alinlab}"
  SAVE_STEPS="${SAVE_STEPS:-2500}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
fi
# 2 GPUs schedules far sooner than 4 on partially-allocated nodes, and 4 vs 2
# changes the effective batch size rather than just the speed -- see the recipe
# pinning in overnight_ablation.sbatch. Whatever you pick is locked in for both
# variants once the first one trains.
GPUS="${GPUS:-2}"
# Honest under-request: shorter jobs fit backfill gaps a 2-day job cannot, and a
# walltime kill is safe here because training resumes from its last checkpoint.
TIMELIMIT="${TIMELIMIT:-1-00:00:00}"
MAX_STEPS="${MAX_STEPS:-10000}"
CHUNKS="${CHUNKS:-${2:-8}}"

TASK="${TASK:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
EPISODES="${EPISODES:-50}"

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
EXPORTS+=",MAX_STEPS=${MAX_STEPS}"
EXPORTS+=",SAVE_STEPS=${SAVE_STEPS}"
EXPORTS+=",SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
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
      echo "  squeue -u ${WHOAMI}                       # ST=R means running"
      echo "  tail -f logs/univtac-groot-full-*.out   # preflight, then stages"
      echo "  tail -f logs/overnight-*/finetune-tactile.log"
    fi
    exit ${status}
    ;;
  --chunks)
    cd "${REPO_ROOT}"
    echo "Chaining ${CHUNKS} x ${TIMELIMIT} on ${PARTITION} (${GPUS} gpu)."
    echo "Each chunk resumes where the previous one was killed."
    echo
    prev=""
    ids=()
    for ((i = 1; i <= CHUNKS; i++)); do
      args=("${SBATCH_ARGS[@]}")
      # afterany, not afterok: a walltime kill is a *failure* exit, and that is
      # the normal way a chunk ends. afterok would stop the chain on the very
      # event it exists to handle.
      [[ -n "${prev}" ]] && args+=(--dependency="afterany:${prev}")
      out=$(sbatch --parsable "${args[@]}" slurm/overnight_ablation.sbatch) || {
        echo "chunk ${i}: submission failed" >&2
        break
      }
      # --parsable can return "jobid;cluster"; keep the id.
      prev="${out%%;*}"
      ids+=("${prev}")
      printf '  chunk %2d/%d: job %s%s\n' "${i}" "${CHUNKS}" "${prev}" \
        "$([[ ${i} -gt 1 ]] && echo "  (after ${ids[i-2]})")"
    done
    echo
    echo "Submitted ${#ids[@]} chunk(s). Only the first competes for a slot;"
    echo "the rest are held until their predecessor finishes."
    echo
    echo "Watch:      squeue -u ${WHOAMI}"
    echo "Progress:   tail -f logs/univtac-groot-full-*.out"
    echo "Training:   tail -f logs/overnight-*/finetune-tactile.log"
    echo "Stop all:   scancel ${ids[*]}"
    echo
    echo "IMPORTANT: check chunk 1 before trusting the chain. Dependencies are"
    echo "'afterany', so a genuine crash (bad config, missing dataset) would let"
    echo "all ${CHUNKS} chunks run and fail in turn. Preflight exits in seconds"
    echo "in that case, so it is cheap -- but you would wait for nothing."
    echo
    echo "Also verify a checkpoint appears within the first chunk:"
    echo "  ls ${MODEL_OUTPUT_DIR}/${TASK}-tactile/"
    echo "If SAVE_STEPS=${SAVE_STEPS} is still too many steps for ${TIMELIMIT},"
    echo "no checkpoint is written, every chunk restarts from zero, and the chain"
    echo "never progresses. Lower it with SAVE_STEPS=100 and resubmit."
    ;;
  *)
    echo "usage: bash slurm/submit_overnight.sh [--print|--test-only|--dry|--chunks [N]]" >&2
    exit 2
    ;;
esac
