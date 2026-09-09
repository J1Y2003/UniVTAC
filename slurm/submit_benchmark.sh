#!/bin/bash
# Submitter for the GR00T N1.7 x UniVTAC benchmark: TWO jobs per task, a
# finetune and then an evaluation held behind it.
#
# They are separate jobs because they belong on different partitions --
# training on sjw_alinlab, evaluation only on `background` -- and one job holds
# one allocation on one partition. Run from the repo root:
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
# Evaluation is NOT allowed on sjw_alinlab -- it belongs on `background`.
# That is why each task is TWO jobs and not one: a job holds one allocation on
# one partition, so a finetune-then-evaluate job would evaluate wherever it
# trained. The eval job is gated behind its own finetune with
# --dependency=afterok, so nothing evaluates a checkpoint that does not exist.
#
# `background` is PriorityTier 1, below sjw_alinlab's 2 -- it wins nothing in a
# contest, it simply has a shorter queue when its nodes are idle.
EVAL_PARTITION="${EVAL_PARTITION:-background}"
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
  partition "${PARTITION} (finetune)" \
  eval_partition "${EVAL_PARTITION} (evaluate)" \
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
# is what keeps the tasks' -- and now the two stages' -- logs apart.
#
# Shortest possible name here is 55 characters ('finetune-only' + 'lift_bottle'),
# comfortably over the floor. Do not trim these.
job_name_for() {
  local task="$1" stage="$2"
  case "${stage}" in
    finetune) printf 'univtac-groot-per-task-finetune-only-vision-only-%s' "${task}" ;;
    eval)     printf 'univtac-groot-per-task-evaluate-only-vision-only-%s' "${task}" ;;
    *) echo "job_name_for: unknown stage '${stage}'" >&2; return 1 ;;
  esac
}

# `stage` is passed through as STAGES=, which is what makes one job train and
# the other evaluate. Everything else is identical between the two.
exports_for() {
  local task="$1" stage="$2" e="ALL"
  # Colon-separated: --export is itself a comma list, so neither a space nor a
  # comma may appear inside a value. benchmark_task.sbatch splits on ':' again.
  case "${stage}" in
    finetune) e+=",STAGES=finetune" ;;
    eval)     e+=",STAGES=eval:compare" ;;
    *) echo "exports_for: unknown stage '${stage}'" >&2; return 1 ;;
  esac
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
  e+=",VARIANTS=${VARIANTS// /:}"   # same reason as STAGES above
  # DRY_RUN must be pinned off: if it is exported in the calling shell (from
  # testing), --export=ALL would carry it in and the job would exit in seconds
  # having trained nothing.
  e+=",DRY_RUN=0"
  printf '%s' "${e}"
}

# sbatch_args_for <task> <stage> [dependency-spec]
#
# The dependency spec is passed WHOLE, e.g. 'afterok:123' or 'afterany:1:2:3',
# because the two callers need different kinds:
#
#   afterok  -- the eval job on its own finetune. It is a real data dependency:
#               there is no checkpoint to evaluate if training did not finish.
#               The cost is that a failed or walltime-killed finetune leaves the
#               eval job in DependencyNeverSatisfied, needing a scancel and a
#               resubmit once the finetune is resumed.
#   afterany -- an EXTRA task's finetune on the reported tasks' finetunes. That
#               one is pure queue ordering, not a data dependency, so a priority
#               task crashing must not abandon it.
#
# --partition is passed on the COMMAND LINE, not left to a #SBATCH header: it
# beats a stale SBATCH_PARTITION in the submitting shell, which would otherwise
# silently put the eval job back on the training partition.
sbatch_args_for() {
  local task="$1" stage="$2" dep="${3:-}" partition="${PARTITION}"
  [[ "${stage}" == "eval" ]] && partition="${EVAL_PARTITION}"
  # No --time, no --cpus-per-task, no --mem: all three are site rules. See the
  # note at the top of slurm/benchmark_task.sbatch.
  SBATCH_ARGS=(
    --partition="${partition}"
    --gres="gpu:${GPUS}"
    --wckey="${WCKEY}"
    --job-name="$(job_name_for "${task}" "${stage}")"
    --export="$(exports_for "${task}" "${stage}")"
  )
  [[ -n "${dep}" ]] && SBATCH_ARGS+=(--dependency="${dep}")
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
      for stage in finetune eval; do
        sbatch_args_for "${task}" "${stage}"
        echo "sbatch ${SBATCH_ARGS[*]} slurm/benchmark_task.sbatch"
      done
    done
    ;;
  --test-only)
    # The dependency is left off here on purpose: --test-only would reject an
    # afterok on a job id that does not exist yet. It tests the submit filter,
    # which is what this mode is for.
    for task in ${ALL_TASKS}; do
      for stage in finetune eval; do
        sbatch_args_for "${task}" "${stage}"
        sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_ROOT}/slurm/benchmark_task.sbatch"
      done
    done
    ;;
  submit)
    cd "${REPO_ROOT}"
    submitted=() ; failed=0

    # Submit one task: its finetune on PARTITION, then its evaluation on
    # EVAL_PARTITION held behind that finetune. Sets TRAIN_JOBID for the caller
    # so the extra tasks can be ordered behind the reported tasks' TRAINING,
    # which is what actually competes for a GPU on the training partition.
    submit_task() {
      local task="$1" train_dep="${2:-}" out jobid
      TRAIN_JOBID=""

      sbatch_args_for "${task}" finetune "${train_dep}"
      if ! out=$(sbatch --parsable "${SBATCH_ARGS[@]}" slurm/benchmark_task.sbatch); then
        echo "FAILED to submit finetune for ${task}" >&2
        failed=1
        return 1
      fi
      jobid="${out%%;*}"
      TRAIN_JOBID="${jobid}"
      submitted+=("${jobid} ${task} finetune on ${PARTITION}${train_dep:+ [held: ${train_dep}]}")
      echo "submitted ${jobid}  ${task}  finetune  (${PARTITION})${train_dep:+  [held: ${train_dep}]}"

      # The eval job is NOT skipped when the finetune fails to submit -- it is
      # never reached, because we returned above. An eval with no finetune to
      # wait on would evaluate a checkpoint that will never exist.
      sbatch_args_for "${task}" eval "afterok:${jobid}"
      if ! out=$(sbatch --parsable "${SBATCH_ARGS[@]}" slurm/benchmark_task.sbatch); then
        echo "FAILED to submit eval for ${task} (its finetune ${jobid} is queued)" >&2
        failed=1
        return 1
      fi
      submitted+=("${out%%;*} ${task} evaluate on ${EVAL_PARTITION} [held: afterok:${jobid}]")
      echo "submitted ${out%%;*}  ${task}  evaluate  (${EVAL_PARTITION})  [held: afterok:${jobid}]"
    }

    # Phase 1: the reported tasks, unconstrained, so they start as soon as a
    # GPU frees up.
    priority_ids=""
    for task in ${TASKS}; do
      # Gate on TRAIN_JOBID, not on submit_task's exit status: the finetune can
      # queue successfully and its eval still fail to submit, and lift_bottle
      # must stay ordered behind that finetune either way.
      submit_task "${task}"
      [[ -n "${TRAIN_JOBID}" ]] && priority_ids+="${TRAIN_JOBID}:"
    done
    # Phase 2: the extra tasks, held until every phase-1 FINETUNE has finished.
    # Ordering behind the finetunes rather than the evals is deliberate: the
    # evals run on a different partition and never compete with training.
    dep="${priority_ids%:}"
    for task in ${EXTRA_TASKS}; do
      submit_task "${task}" "${dep:+afterany:${dep}}"
    done
    if ((${#submitted[@]})); then
      echo
      echo "Submitted ${#submitted[@]} job(s) -- two per task, finetune then evaluate"
      echo "(evaluation on ${EVAL_PARTITION}, extras held last):"
      printf '  %s\n' "${submitted[@]}"
      echo
      echo "Watch them with:"
      echo "  squeue -u ${USER} -o '%.10i %.20P %.70j %.9T %.10M %.20R'"
      echo
      echo "An eval job showing DependencyNeverSatisfied means its finetune failed"
      echo "or was killed. Resume that finetune, then resubmit its eval alone:"
      echo "  TASKS=<task> EXTRA_TASKS= bash slurm/submit_benchmark.sh"
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
    echo "  PARTITION=<train>  EVAL_PARTITION=<eval>  are also overridable." >&2
    echo "  Two jobs per task: finetune on PARTITION, evaluate on EVAL_PARTITION." >&2
    echo "  EXTRA_TASKS run last, gated behind the reported tasks' finetunes." >&2
    exit 2
    ;;
esac
