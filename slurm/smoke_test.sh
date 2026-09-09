#!/bin/bash
# Prove the finetune -> evaluate chain connects, in ~an hour: 20 training steps
# and 1 evaluation episode. The success rate is meaningless at 20 steps; the
# question is whether every interface works.
#
#   bash slurm/smoke_test.sh              # submit
#   bash slurm/smoke_test.sh --print      # show the command
#   bash slurm/smoke_test.sh --check      # report what a previous run produced
#
# Everything it writes goes under SMOKE_ROOT, never the real CKPT_ROOT or
# eval_result/. benchmark_task.sbatch keeps its stage markers and pinned recipe
# in CKPT_ROOT/.stages, so a shared root would let a 20-step run mark the real
# finetune complete and pin the recipe to 20 steps. The dataset is the only
# thing shared, read-only.
set -uo pipefail

WHOAMI="${USER:-$(id -un)}"

WORKSPACE="${WORKSPACE:-${HOME}/jaewon/workspace}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=slurm/common.sh
. "${REPO_ROOT}/slurm/common.sh"
UNIVTAC_ROOT="${UNIVTAC_ROOT:-${WORKSPACE}/UniVTAC-sim}"
GROOT_ROOT="${GROOT_ROOT:-${WORKSPACE}/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:-${HOME}/miniconda3/envs/UniVTAC/bin/python}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/groot-data}"
HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"

# Everything this run writes lives here. Delete it and nothing is lost.
# Ordinary scratch, not the unified folder: bundle-sbatch owns the real output
# location now, and this only holds the smoke run's redirected CKPT_ROOT,
# results and stage markers so they can never collide with the real run's.
SMOKE_ROOT="${SMOKE_ROOT:-${HOME}/jaewon/workspace/groot-smoke}"

TASK="${TASK:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-clean}"

# Deliberately tiny. 20 steps with a save at 10 proves the training loop, the
# optimizer, and checkpoint writing; 1 episode proves the whole eval path.
MAX_STEPS="${MAX_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
EPISODES="${EPISODES:-1}"

# 1 GPU schedules soonest, and the recipe pinning is isolated under SMOKE_ROOT,
# so this does not constrain the real run's GPU count.
#
# `background` rather than `debug`: this job EVALUATES as well as trains -- that
# is the whole point of it -- and evaluation is only allowed on `background`.
# The real run splits the two stages across two partitions; the smoke test
# deliberately does not, because what it proves is that the chain holds inside
# one job. 20 steps plus one episode fits either partition's limit.
PARTITION="${PARTITION:-background}"
# Names must be longer than 50 characters or the submit filter rejects them;
# this one is 57 plus the task. It moved here out of benchmark_task.sbatch's
# #SBATCH header, which the launcher now owns.
JOB_NAME="${JOB_NAME:-univtac-groot-smoke-test-finetune-then-evaluate-one-task-${TASK}}"
GPUS="${GPUS:-1}"
# A short --time so backfill can slot this into a small gap -- the whole point
# of a smoke test is that it schedules and finishes quickly. 20 steps plus one
# episode is ~an hour; 2 h is the headroom. Empty means no --time.
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
# Site rule: every sbatch and srun carries this.
WCKEY="${WCKEY:-project-short-name:sub_4dpdata}"
# Off by default so a throwaway run does not clutter the real project's plots.
USE_WANDB="${USE_WANDB:-0}"

MODE="${1:-submit}"

# --------------------------------------------------------------------------- #
if [[ "${MODE}" == "--check" ]]; then
  echo "Smoke run outputs under ${SMOKE_ROOT}:"
  for variant in tactile baseline_finetuned; do
    out_dir="${SMOKE_ROOT}/${TASK}-${variant}"
    ckpts=$(ls -d "${out_dir}"/checkpoint-* 2>/dev/null | tr '\n' ' ')
    marker="${SMOKE_ROOT}/.stages/finetune-${TASK}-${variant}.done"
    summary="${SMOKE_ROOT}/eval_result/${variant}/${TASK}/seed0-0.summary.json"
    printf '  %-20s finetune=%s  checkpoints=%s\n' "${variant}" \
      "$([[ -f "${marker}" ]] && echo DONE || echo "incomplete")" \
      "${ckpts:-none}"
    if [[ -f "${summary}" ]]; then
      echo "      eval summary:"
      sed 's/^/        /' "${summary}" | head -14
    else
      echo "      eval summary: MISSING"
    fi
  done
  echo
  echo "What to look for:"
  echo "  * a checkpoint-10 or -20 directory     -> trainer accepted the modality"
  echo "                                            config and the 113-D state"
  echo "  * eval summary with episodes_scored=1  -> inference on a finetuned"
  echo "                                            checkpoint works end to end"
  echo "  * episodes_errored=1 instead           -> read the server log:"
  echo "      ${SMOKE_ROOT}/eval_result/<variant>/${TASK}/server-0.log"
  exit 0
fi

# --------------------------------------------------------------------------- #
FAIL=0
for spec in "UNIVTAC_ROOT ${UNIVTAC_ROOT}" "GROOT_ROOT ${GROOT_ROOT}" "DATA_ROOT ${DATA_ROOT}"; do
  set -- ${spec}
  [[ -d "$2" ]] || { echo "MISSING dir $1=$2" >&2; FAIL=1; }
done
for spec in "GROOT_PYTHON ${GROOT_PYTHON}" "UNIVTAC_PYTHON ${UNIVTAC_PYTHON}"; do
  set -- ${spec}
  [[ -x "$2" ]] || { echo "NOT EXECUTABLE $1=$2" >&2; FAIL=1; }
done
for variant in tactile baseline_finetuned; do
  n=$(find "${DATA_ROOT}/univtac-${TASK}-${variant}/data" -name '*.parquet' 2>/dev/null | wc -l)
  [[ "${n}" -gt 0 ]] || { echo "MISSING dataset for ${variant}" >&2; FAIL=1; }
done
# Every GR00T checkpoint loads the gated nvidia/Cosmos-Reason2-2B, so no token
# means a 401 about a minute into training -- after the job already holds a GPU.
# Warn here, where it costs nothing. HF_HOME is redirected, which also moves
# where a stored login is read from ($HF_HOME/token), so HF_TOKEN in the
# environment is the reliable route; the job inherits it from this shell.
if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_HOME}/token" && ! -s "${HOME}/.cache/huggingface/token" ]]; then
  echo "WARNING: no HF_TOKEN and no token file under HF_HOME=${HF_HOME}." >&2
  echo "         The job will fail loading nvidia/Cosmos-Reason2-2B (401)." >&2
  echo "         Fix: export HF_TOKEN=hf_...   then re-run this." >&2
  FAIL=1
fi
mkdir -p "${REPO_ROOT}/logs" 2>/dev/null || FAIL=1
if [[ "${FAIL}" -ne 0 ]]; then
  echo "Fix the above. Nothing submitted." >&2
  exit 2
fi

# The job's configuration travels in the environment: bundle-sbatch owns
# --export. MODEL_OUTPUT_DIR is absent deliberately -- the launcher injects it,
# pointing at this submission's own bundle. CKPT_ROOT still redirects the
# checkpoints into SMOKE_ROOT so the smoke test cannot touch the real run's.
BUNDLE_ENV=(
  "UNIVTAC_JOB_CONFIG=1"
  "CKPT_ROOT=${SMOKE_ROOT}"
  "RESULTS_DIR=${SMOKE_ROOT}/eval_result"
  "STAGE_DIR=${SMOKE_ROOT}/.stages"
  "UNIVTAC_ROOT=${UNIVTAC_ROOT}"
  "UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"
  "GROOT_ROOT=${GROOT_ROOT}"
  "GROOT_PYTHON=${GROOT_PYTHON}"
  "HF_HOME=${HF_HOME}"
  "DATA_ROOT=${DATA_ROOT}"
  "TASK=${TASK}"
  "TASK_CONFIG=${TASK_CONFIG}"
  "EPISODES=${EPISODES}"
  "MAX_STEPS=${MAX_STEPS}"
  "SAVE_STEPS=${SAVE_STEPS}"
  "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
  "USE_WANDB=${USE_WANDB}"
  # DRY_RUN left exported from testing would ride in and the job would exit in
  # seconds having proved nothing.
  "DRY_RUN=0"
)

BUNDLE_SLURM_ARGS=(
  --job-name="${JOB_NAME}"
  --wckey="${WCKEY}"
  --partition="${PARTITION}"
  --gres="gpu:${GPUS}"
)
[[ -n "${TIME_LIMIT}" ]] && BUNDLE_SLURM_ARGS+=(--time="${TIME_LIMIT}")

# --job-kind train with from_scratch: this job trains before it evaluates, and
# it starts from the released weights rather than a local checkpoint. The eval
# it then runs is inside the same job, which is the whole point of a smoke test.
BUNDLE_JOB_KIND=train
BUNDLE_CHECKPOINT=from_scratch
BUNDLE_GIT_ROOT="${REPO_ROOT}"

echo "Smoke test: ${MAX_STEPS} training steps, ${EPISODES} eval episode(s)"
echo "  task       ${TASK}/${TASK_CONFIG}"
echo "  writes to  ${SMOKE_ROOT}          <- isolated, safe to delete"
echo "  reads      ${DATA_ROOT}           <- shared, read-only"
echo "  job        ${GPUS} gpu on ${PARTITION}, wckey=${WCKEY}, time=${TIME_LIMIT:-<partition maximum>}"
echo

if [[ "${MODE}" == "--print" ]]; then
  bundle_print "${REPO_ROOT}/slurm/benchmark_task.sbatch"
  exit $?
fi
bundle_require || exit 2

# A second queued job competes with the real run for this cluster's per-user GPU
# cap. Say so rather than silently delaying the thing that matters.
existing=$(squeue -h -u "${WHOAMI}" --format="%i %T %200j" 2>/dev/null \
           | grep "univtac-groot" || true)
if [[ -n "${existing}" ]]; then
  echo "NOTE: you already have queued/running jobs:"
  echo "${existing}" | sed 's/^/  /'
  echo
  echo "Adding this one competes with them for the per-user GPU cap. If the big"
  echo "run is still PENDING, consider holding it so the smoke test goes first --"
  echo "if an interface is broken, that pending run would fail anyway:"
  echo "  scontrol hold <jobid>     # keeps queue position, stops it starting"
  echo "  scontrol release <jobid>  # after the smoke test passes"
  echo
  read -r -t 15 -p "Submit anyway? [y/N] " reply || reply=""
  echo
  [[ "${reply}" == "y" || "${reply}" == "Y" ]] || { echo "Nothing submitted."; exit 0; }
fi

cd "${REPO_ROOT}"
# Never retried: once the launcher records a submission, a second invocation for
# the same request is forbidden. See slurm/common.sh.
bundle_submit "${REPO_ROOT}/slurm/benchmark_task.sbatch" || exit $?
jobid="${BUNDLE_JOBID:-<see the launcher output above>}"
echo "Submitted smoke job ${jobid}"
echo
echo "Watch:   the bundle's logs/slurm.out (the launcher prints the bundle path)"
echo "Verdict: bash slurm/smoke_test.sh --check"
echo
echo "Expected timeline: a few minutes to load the 3B checkpoint, then 20 steps"
echo "(~1.9 s/it), two checkpoint writes, and a 1-episode eval per variant."
