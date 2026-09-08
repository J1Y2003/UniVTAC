#!/bin/bash
# Submitter for the GR00T N1.7 x UniVTAC benchmark: one finetune-then-evaluate
# job per task. Run from the repo root:
#
#   bash slurm/submit_benchmark.sh                 # submit
#   bash slurm/submit_benchmark.sh --print         # show the sbatch command only
#   bash slurm/submit_benchmark.sh --test-only     # run the submit filter, queue nothing
#   bash slurm/submit_benchmark.sh --dry           # DRY_RUN=1 locally, no sbatch at all
#
# Everything below is an overridable default, so a changed path is one variable
# and not a rewritten command line:
#
#   TASK=insert_hole MAX_STEPS=2000 bash slurm/submit_benchmark.sh
#
# Two tiers of task, submitted in that order:
#
#   TASKS        the three reported tasks -- insert_hole, insert_tube,
#                pull_out_key -- submitted first and unconstrained.
#   EXTRA_TASKS  lift_bottle, submitted last and gated behind all of TASKS
#                with --dependency=afterany, so it cannot take a GPU from a
#                reported task. EXTRA_TASKS="" skips it.
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
# 1, and do not raise it without fixing the cause first: launch_finetune.py
# wraps the model in nn.DataParallel for --num-gpus > 1, which fails with
# "module must have its parameters and buffers ... on device: cuda:0 ... but
# found one on device: cpu". More GPUs would also change the effective batch
# size rather than just the speed, and whatever you pick is locked in for both
# variants by the recipe pinning in benchmark_task.sbatch.
GPUS="${GPUS:-1}"
# NO TIMELIMIT, deliberately. Site rule: never pass --time. The job gets the
# maximum the partition allows, or runs until the script exits, whichever comes
# first -- a limit can only ever cut the run short. A walltime kill is safe
# anyway, since training resumes from its last checkpoint.
# Site rule: every sbatch and srun carries this.
WCKEY="${WCKEY:-project-short-name:sub_4dpdata}"

# One job per task -- GR00T is finetuned PER TASK, matching UniVTAC's ACT,
# which trains one policy per task. Three separate jobs rather than one long
# one: each is independently resumable, each fits a backfill gap, and one
# task failing does not block the others.
#
# Default is the three tasks the project cares about. `TASK=x` still works.
TASKS="${TASKS:-${TASK:-insert_hole insert_tube pull_out_key}}"
# A second, LOWER-PRIORITY tier, submitted only after every job in TASKS has
# been queued and gated behind them with --dependency, so it can never take a
# GPU that a reported task still wants. `lift_bottle` is here because it is
# deliberately NOT one of the three reported tasks: it is the task any
# hyperparameter sweep is allowed to touch without fitting the numbers we
# report (see docs/ABLATION.md#comparability-with-univtacs-act). Having a
# trained lift_bottle model is still useful -- it is the fourth data point and
# the sweep substrate -- it just must not compete for the queue.
#
# Set EXTRA_TASKS="" to submit the three priority tasks alone.
EXTRA_TASKS="${EXTRA_TASKS:-lift_bottle}"
ALL_TASKS="${TASKS} ${EXTRA_TASKS}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
# 100, from the paper: "All policies are trained on 50 automatically collected
# full trajectories per task and evaluated over 100 test rollouts." Comparing a
# 50-episode interval against their 100-episode number would not be comparing
# like with like.
EPISODES="${EPISODES:-100}"
# Vision only. The tactile pipeline stays in the repo and still works, but the
# question right now is how GR00T N1.7 does on UniVTAC WITHOUT touch, so the
# default trains one variant. Set VARIANTS="tactile baseline_finetuned" to run
# the full ablation again.
VARIANTS="${VARIANTS:-baseline_finetuned}"

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
  wckey "${WCKEY}" \
  tasks "${TASKS}" \
  extra_tasks "${EXTRA_TASKS:-<none>}" \
  task_config "${TASK_CONFIG}" \
  variants "${VARIANTS}" \
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
# Only the variants actually being trained -- demanding a tactile dataset we
# deliberately are not training would block the submit for no reason.
# A missing dataset for a PRIORITY task blocks the submit; a missing dataset
# for an EXTRA task only drops that task. Otherwise an unconverted lift_bottle
# would hold up the three tasks we actually report, which is backwards.
EXTRA_TASKS_OK=""
for task in ${ALL_TASKS}; do
  extra=0
  for e in ${EXTRA_TASKS}; do [[ "${task}" == "${e}" ]] && extra=1; done
  task_ok=1
  for variant in ${VARIANTS}; do
    dataset="${DATA_ROOT}/univtac-${task}-${variant}"
    n=$(find "${dataset}/data" -name '*.parquet' 2>/dev/null | wc -l)
    if [[ "${n}" -gt 0 ]]; then
      echo "  ok  ${task}/${variant}: ${n} parquet in ${dataset}"
    else
      task_ok=0
      convert="sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu --export=ALL,TASK=${task},VARIANT=${variant},TASK_CONFIG=${TASK_CONFIG} slurm/convert.sbatch"
      if [[ "${extra}" -eq 1 ]]; then
        echo "  skip ${task}/${variant}: no dataset, and it is an EXTRA task -- not submitting it" >&2
        echo "        convert it later: ${convert}" >&2
      else
        echo "  MISSING dataset ${dataset}" >&2
        echo "        convert it: ${convert}" >&2
        FAIL=1
      fi
    fi
  done
  [[ "${extra}" -eq 1 && "${task_ok}" -eq 1 ]] && EXTRA_TASKS_OK+="${task} "
done
# Only the extra tasks that actually have data.
EXTRA_TASKS="${EXTRA_TASKS_OK% }"
ALL_TASKS="${TASKS} ${EXTRA_TASKS}"
# SLURM opens the --output file before the script runs, so a missing logs/ kills
# the job instantly with no log to explain it.
# Every GR00T checkpoint loads the gated nvidia/Cosmos-Reason2-2B, so no token
# means a 401 about a minute into training -- after the job already holds a GPU.
# Warn here, where it costs nothing. HF_HOME is redirected, which also moves
# where a stored login is read from ($HF_HOME/token), so HF_TOKEN in the
# environment is the reliable route; --export=ALL carries it into the job.
if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_HOME}/token" && ! -s "${HOME}/.cache/huggingface/token" ]]; then
  echo "WARNING: no HF_TOKEN and no token file under HF_HOME=${HF_HOME}." >&2
  echo "         The job will fail loading nvidia/Cosmos-Reason2-2B (401)." >&2
  echo "         Fix: export HF_TOKEN=hf_...   then re-run this." >&2
  FAIL=1
fi
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
# The submit filter rejects job names of 50 characters or fewer, so keep the
# per-task name long. `%x` in the sbatch --output pattern picks this up, which
# is what keeps the three tasks' logs apart.
job_name_for() {
  printf 'univtac-groot-per-task-finetune-then-evaluate-vision-only-%s' "$1"
}

exports_for() {
  local task="$1" e="ALL"
  e+=",MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR}"
  e+=",UNIVTAC_ROOT=${UNIVTAC_ROOT}"
  e+=",UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"
  e+=",GROOT_ROOT=${GROOT_ROOT}"
  e+=",GROOT_PYTHON=${GROOT_PYTHON}"
  e+=",HF_HOME=${HF_HOME}"
  e+=",DATA_ROOT=${DATA_ROOT}"
  e+=",TASK=${task}"
  e+=",TASK_CONFIG=${TASK_CONFIG}"
  e+=",EPISODES=${EPISODES}"
  e+=",VARIANTS=${VARIANTS}"
  # DRY_RUN must be pinned off: if it is exported in the calling shell (from
  # testing), --export=ALL would carry it in and the job would exit in seconds
  # having trained nothing.
  e+=",DRY_RUN=0"
  printf '%s' "${e}"
}

sbatch_args_for() {
  local task="$1" dep="${2:-}"
  # No --time, no --cpus-per-task, no --mem: all three are site rules. See the
  # note at the top of slurm/benchmark_task.sbatch.
  SBATCH_ARGS=(
    --partition="${PARTITION}"
    --gres="gpu:${GPUS}"
    --wckey="${WCKEY}"
    --job-name="$(job_name_for "${task}")"
    --export="$(exports_for "${task}")"
  )
  # afterany, not afterok: an extra task is an independent finetune on its own
  # dataset, so a priority task crashing is no reason to abandon it. The
  # dependency exists purely to order the QUEUE, not to express a data
  # dependency. (afterok would leave lift_bottle stuck in DependencyNeverSatisfied
  # forever if one of the three failed, needing a manual scancel.)
  [[ -n "${dep}" ]] && SBATCH_ARGS+=(--dependency="afterany:${dep}")
}

case "${MODE}" in
  --dry)
    for task in ${ALL_TASKS}; do
      echo "=== preflight ${task} (DRY_RUN=1), submitting nothing ==="
      env MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}" UNIVTAC_ROOT="${UNIVTAC_ROOT}" \
          UNIVTAC_PYTHON="${UNIVTAC_PYTHON}" GROOT_ROOT="${GROOT_ROOT}" \
          GROOT_PYTHON="${GROOT_PYTHON}" HF_HOME="${HF_HOME}" DATA_ROOT="${DATA_ROOT}" \
          TASK="${task}" TASK_CONFIG="${TASK_CONFIG}" EPISODES="${EPISODES}" \
          VARIANTS="${VARIANTS}" \
          DRY_RUN=1 bash "${REPO_ROOT}/slurm/benchmark_task.sbatch"
      echo
    done
    ;;
  --print)
    for task in ${ALL_TASKS}; do
      sbatch_args_for "${task}"
      echo "sbatch ${SBATCH_ARGS[*]} slurm/benchmark_task.sbatch"
    done
    ;;
  --test-only)
    for task in ${ALL_TASKS}; do
      sbatch_args_for "${task}"
      sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_ROOT}/slurm/benchmark_task.sbatch"
    done
    ;;
  submit)
    cd "${REPO_ROOT}"
    submitted=() ; failed=0
    # Phase 1: the reported tasks, unconstrained, so they start as soon as a
    # GPU frees up.
    priority_ids=""
    for task in ${TASKS}; do
      sbatch_args_for "${task}"
      if out=$(sbatch --parsable "${SBATCH_ARGS[@]}" slurm/benchmark_task.sbatch); then
        jobid="${out%%;*}"
        submitted+=("${jobid} ${task}")
        priority_ids+="${jobid}:"
        echo "submitted ${jobid}  ${task}"
      else
        echo "FAILED to submit ${task}" >&2
        failed=1
      fi
    done
    # Phase 2: the extra tasks, held until every phase-1 job has finished. With
    # GPUS=1 and one job per task the three run concurrently if the queue
    # allows; lift_bottle then takes whatever is left.
    dep="${priority_ids%:}"
    for task in ${EXTRA_TASKS}; do
      sbatch_args_for "${task}" "${dep}"
      if out=$(sbatch --parsable "${SBATCH_ARGS[@]}" slurm/benchmark_task.sbatch); then
        jobid="${out%%;*}"
        submitted+=("${jobid} ${task} (after ${dep:-nothing})")
        echo "submitted ${jobid}  ${task}  [held until ${dep:-<no dependency>}]"
      else
        echo "FAILED to submit ${task}" >&2
        failed=1
      fi
    done
    if ((${#submitted[@]})); then
      echo
      echo "Submitted ${#submitted[@]} job(s), one per task (extras held last):"
      printf '  %s\n' "${submitted[@]}"
      echo
      echo "Watch them with:"
      echo "  squeue -u ${USER} -o '%.10i %.60j %.9T %.10M %.20R'"
      echo "  tail -f logs/univtac-groot-per-task-*.out          # preflight, then stages"
      echo "  tail -f logs/benchmark-<task>-<jobid>/finetune-${VARIANTS%% *}.log"
      echo
      echo "Aggregate when they finish:"
      echo "  python scripts/compare_ablation.py --results-dir eval_result"
    fi
    exit ${failed}
    ;;
  *)
    echo "usage: bash slurm/submit_benchmark.sh [--print|--test-only|--dry]" >&2
    echo "  TASKS=\"a b c\"  EXTRA_TASKS=\"d\"  VARIANTS=...  EPISODES=N  GPUS=N" >&2
    echo "  are all overridable. EXTRA_TASKS run last, gated behind TASKS." >&2
    exit 2
    ;;
esac
