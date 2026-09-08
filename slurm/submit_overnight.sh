#!/bin/bash
# One-liner submitter for the full ablation. Run from the repo root:
#
#   bash slurm/submit_overnight.sh                 # submit
#   bash slurm/submit_overnight.sh --print         # show the sbatch command only
#   bash slurm/submit_overnight.sh --test-only     # run the submit filter, queue nothing
#   bash slurm/submit_overnight.sh --dry           # DRY_RUN=1 locally, no sbatch at all
#   bash slurm/submit_overnight.sh --next          # ONE 3h job on `debug`
#   bash slurm/submit_overnight.sh --status        # progress, submit nothing
#
# Use --next when the long-walltime partitions are contended. The 2-day
# partitions have the walltime but not the availability; a 3-hour `debug` slot
# that actually starts beats a 24-hour slot that pends all night. It submits
# exactly one job and refuses if one of yours is already queued, so it is safe
# to re-run: each job resumes from the last checkpoint. This cluster caps GPUs
# per user across *queued* jobs, not just running ones, so pre-submitting a
# dependency chain blocks itself -- one at a time is the only thing that works.
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
# Must match the CKPT_ROOT default inside overnight_ablation.sbatch, since
# --status reads its stage markers and checkpoints from here.
CKPT_ROOT="${CKPT_ROOT:-${MODEL_OUTPUT_DIR}}"

MODE="${1:-submit}"

# --------------------------------------------------------------------------- #
# Job shape
# --------------------------------------------------------------------------- #
# Short-chunk mode (--next / --status): one short job at a time, re-run by hand,
# each resuming where the last was killed. The 2-day partitions have the
# walltime but not the availability -- an 8-hour wait for a 24-hour slot is
# worse than a 3-hour slot that actually starts -- while `debug` is the cluster
# default, so it is the least contended and the quickest to schedule.
#
# This works only because every stage is resumable: finetuning continues via
# --resume-from-checkpoint, and completed stages are skipped by their markers.
if [[ "${MODE}" == "--next" || "${MODE}" == "--status" ]]; then
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
    echo "--chunks is removed. It submitted all N jobs up front, chained by" >&2
    echo "--dependency, and although only the first could run, the held jobs" >&2
    echo "still sat in the queue with their GPU requests counted against this" >&2
    echo "cluster's per-user GPU cap -- which blocked the whole chain." >&2
    echo >&2
    echo "Use --next instead: it submits exactly ONE job, and refuses if one of" >&2
    echo "yours is already queued or running. Re-run it whenever the previous" >&2
    echo "chunk ends; each one resumes from the last checkpoint." >&2
    exit 2
    ;;
  --status | --next)
    # Progress first, so both modes answer "where are we?" the same way.
    echo "Progress:"
    stage_dir="${CKPT_ROOT}/.stages"
    for variant in tactile baseline_finetuned; do
      out_dir="${CKPT_ROOT}/${TASK}-${variant}"
      if [[ -f "${stage_dir}/finetune-${TASK}-${variant}.done" ]]; then
        state="finetune COMPLETE"
      else
        latest=""
        for d in "${out_dir}"/checkpoint-*; do
          [[ -d "${d}" ]] || continue
          n="${d##*checkpoint-}"
          [[ "${n}" =~ ^[0-9]+$ ]] || continue
          [[ -z "${latest}" || "${n}" -gt "${latest}" ]] && latest="${n}"
        done
        if [[ -n "${latest}" ]]; then
          state="training: step ${latest}/${MAX_STEPS}"
        else
          state="not started (no checkpoint yet)"
        fi
      fi
      summary="${REPO_ROOT}/eval_result/${variant}/${TASK}/seed0-0.summary.json"
      [[ -f "${summary}" ]] && state+="; eval DONE"
      printf '  %-20s %s\n' "${variant}" "${state}"
    done
    echo

    # One job at a time, enforced here rather than relying on the scheduler.
    # squeue's %j truncates at its default width, hence the explicit 200.
    running=$(squeue -h -u "${WHOAMI}" --format="%i %T %200j" 2>/dev/null \
              | grep "univtac-groot" || true)
    if [[ -n "${running}" ]]; then
      echo "You already have a job in the queue:"
      echo "${running}" | sed 's/^/  /'
      echo
      echo "Nothing submitted. Re-run this once that job ends."
      echo "  cancel it with: scancel $(echo "${running}" | awk '{printf "%s ", $1}')"
      exit 0
    fi

    if [[ "${MODE}" == "--status" ]]; then
      echo "No job of yours is queued. Submit the next one with:"
      echo "  bash slurm/submit_overnight.sh --next"
      exit 0
    fi

    cd "${REPO_ROOT}"
    echo "Submitting ONE job: ${TIMELIMIT} on ${PARTITION}, ${GPUS} gpu."
    out=$(sbatch --parsable "${SBATCH_ARGS[@]}" slurm/overnight_ablation.sbatch) || exit $?
    jobid="${out%%;*}"
    echo "  job ${jobid}"
    echo
    echo "Watch:       squeue -u ${WHOAMI}"
    echo "Progress:    tail -f logs/univtac-groot-full-*-${jobid}.out"
    echo "Training:    tail -f logs/overnight-${jobid}/finetune-tactile.log"
    echo "Where am I:  bash slurm/submit_overnight.sh --status"
    echo "Next chunk:  bash slurm/submit_overnight.sh --next   (after this one ends)"
    echo
    echo "If the step number under --status does not advance between chunks,"
    echo "SAVE_STEPS=${SAVE_STEPS} is too many steps to reach in ${TIMELIMIT} and"
    echo "every chunk is restarting from zero. Lower it: SAVE_STEPS=100."
    ;;
  *)
    echo "usage: bash slurm/submit_overnight.sh [--next|--status|--print|--test-only|--dry]" >&2
    exit 2
    ;;
esac
