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
#
# It submits FINETUNES only. Evaluation is one checkpoint at a time, through
# slurm/eval_checkpoint.sh, which files each result in the results library.
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
# Why a wrapper: the bundle-sbatch form needs seven absolute paths
# plus four site-specific flags, which is unreadable to type and easy to get
# subtly wrong -- and a wrong path there fails minutes into a queued job rather
# than here.

set -uo pipefail

# --------------------------------------------------------------------------- #
# Paths -- override any of these in the environment
# --------------------------------------------------------------------------- #
WORKSPACE="${WORKSPACE:-${HOME}/jaewon/workspace}"
# USER is not exported in every shell; squeue and the paths below need a name.
WHOAMI="${USER:-$(id -un)}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Every submission goes through bundle-sbatch; the contract lives in one place.
# shellcheck source=slurm/bundle_submit.sh
. "${REPO_ROOT}/slurm/bundle_submit.sh"
UNIVTAC_ROOT="${UNIVTAC_ROOT:-${WORKSPACE}/UniVTAC-sim}"
GROOT_ROOT="${GROOT_ROOT:-${WORKSPACE}/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:-${HOME}/miniconda3/envs/UniVTAC/bin/python}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/groot-data}"
HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"
# MODEL_OUTPUT_DIR is NOT set here any more. bundle-sbatch creates one output
# bundle per submission and injects
#   MODEL_OUTPUT_DIR = CODE_OUTPUT_DIR = <bundle>/code-output
# and the guide is explicit that we must not supply it. The old
# /rlwrld-unified-checkpoints path went with it.
#
# A --dry run has no bundle and therefore no injected MODEL_OUTPUT_DIR, but
# benchmark_task.sbatch requires one. Hand it a throwaway; nothing is written.
DRY_RUN_OUTPUT_DIR="${DRY_RUN_OUTPUT_DIR:-${TMPDIR:-/tmp}/univtac-dry-run}"

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
# Walltime. A job with no --time is assumed to want the partition maximum
# (2 days), so backfill can only ever start it in a 2-day gap. A realistic limit
# makes it eligible for far more gaps, which on a contended queue is the
# difference between starting tonight and starting tomorrow.
#
# TRAINING defaults to a limit because it is RESUMABLE: 10,000 steps at the
# measured 1.89 s/it is ~5.3 h, so 10 h is roughly 2x headroom, and if it is
# ever wrong the job resumes from its last checkpoint on a resubmit.
TIME_LIMIT="${TIME_LIMIT:-10:00:00}"
# EVALUATION defaults to NO limit, deliberately. scripts/run_eval.py has no
# resume -- a walltime kill restarts it from the first seed, losing the run
# rather than pausing it -- and the cost of 100 rollouts has never been
# measured (docs/STATUS.md, "Open"). Set this once a real eval log exists.
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-}"
# Both are passed straight to sbatch, so any format sbatch accepts works
# ("10:00:00", "8:00", "1-12:00:00"). Empty means no --time at all.
#
# Neither may exceed the partition maximum or the job pends forever with
# REASON=PartitionTimeLimit. Check a partition's cap with:
#   scontrol show partition <name> | grep MaxTime
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
  partition "${PARTITION} (finetune)" \
  time_limit "${TIME_LIMIT:-<none: partition maximum>}" \
  eval_time_limit "${EVAL_TIME_LIMIT:-<none: partition maximum>}" \
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
      convert="env TASK=${task} VARIANT=${variant} TASK_CONFIG=${TASK_CONFIG}"
      convert+=" UNIVTAC_JOB_CONFIG=1 bundle-sbatch --job-kind data_process"
      convert+=" --code-git-root ${REPO_ROOT} --"
      convert+=" --job-name=univtac-groot-convert-univtac-hdf5-demonstrations-to-lerobot-v2"
      convert+=" --wckey=${WCKEY} --partition=cpu -- slurm/convert.sbatch"
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
# The launcher owns the Slurm logs now (they land in the bundle), but
# benchmark_task.sbatch still writes its per-stage logs under REPO_ROOT/logs.
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
# per-task name long. It is also what tells one submission's bundle from
# another's at a glance, now that the launcher owns the log paths.
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

# The job's configuration travels in the ENVIRONMENT, not in an --export list:
# bundle-sbatch owns --export, and sbatch propagates the submitting environment
# by default. env_for prints `NAME=VALUE` pairs for `env`, one per line.
#
# UNIVTAC_JOB_CONFIG is the canary. If this environment ever stops reaching the
# job, benchmark_task.sbatch stops instead of silently training TASK's default.
env_for() {
  local task="$1" stage="$2"
  # Colon-separated: a space inside a value would need quoting through two
  # command-line regions. benchmark_task.sbatch splits ':' back out.
  case "${stage}" in
    finetune) printf '%s
' "STAGES=finetune" ;;
    eval)     printf '%s
' "STAGES=eval:compare" ;;
    *) echo "env_for: unknown stage '${stage}'" >&2; return 1 ;;
  esac
  printf '%s
'     "UNIVTAC_JOB_CONFIG=1"     "UNIVTAC_ROOT=${UNIVTAC_ROOT}"     "UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"     "GROOT_ROOT=${GROOT_ROOT}"     "GROOT_PYTHON=${GROOT_PYTHON}"     "HF_HOME=${HF_HOME}"     "DATA_ROOT=${DATA_ROOT}"     "TASK=${task}"     "TASK_CONFIG=${TASK_CONFIG}"     "EPISODES=${EPISODES}"     "VARIANTS=${VARIANTS// /:}"     "DRY_RUN=0"
  # MODEL_OUTPUT_DIR is deliberately absent: bundle-sbatch injects it, and the
  # guide is explicit that we must not supply it.
}

# slurm_args_for <task> <stage>
#
# Region 2 of the bundle-sbatch command line: Slurm options, attached long form
# only. --output/--error/--export/--parsable/--test-only are the launcher's and
# must not appear; --job-name/--wckey/--partition/--time moved here out of the
# #SBATCH headers.
slurm_args_for() {
  local task="$1" stage="$2" partition="${PARTITION}" time_limit="${TIME_LIMIT}"
  if [[ "${stage}" == "eval" ]]; then
    partition="${EVAL_PARTITION}"
    time_limit="${EVAL_TIME_LIMIT}"
  fi
  # No --cpus-per-task and no --mem: site rules, rejected by the submit filter.
  BUNDLE_SLURM_ARGS=(
    --job-name="$(job_name_for "${task}" "${stage}")"
    --wckey="${WCKEY}"
    --partition="${partition}"
    --gres="gpu:${GPUS}"
  )
  [[ -n "${time_limit}" ]] && BUNDLE_SLURM_ARGS+=(--time="${time_limit}")
  return 0
}

submit_stage() {
  # One submission. BUNDLE_ENV is the job's configuration: --export belongs to
  # the launcher, so the job inherits its environment instead of being handed a
  # variable list.
  local task="$1" stage="$2" checkpoint="$3" label="$4"
  mapfile -t BUNDLE_ENV < <(env_for "${task}" "${stage}") || return 2
  slurm_args_for "${task}" "${stage}" || return 2

  if [[ "${stage}" == "eval" ]]; then
    BUNDLE_JOB_KIND=eval
  else
    BUNDLE_JOB_KIND=train
  fi
  BUNDLE_CHECKPOINT="${checkpoint}"
  BUNDLE_GIT_ROOT="${REPO_ROOT}"

  if [[ "${MODE}" == "--print" ]]; then
    bundle_print "slurm/benchmark_task.sbatch"
    return $?
  fi

  # NOTE: no retry, ever. Once the launcher has recorded submission_started, a
  # second invocation for the same request is forbidden even when the outcome
  # is unclear -- read the diagnostic and the retained bundle instead.
  if bundle_submit "slurm/benchmark_task.sbatch"; then
    submitted+=("${label}${BUNDLE_JOBID:+  (job ${BUNDLE_JOBID})}")
    echo "submitted  ${label}${BUNDLE_JOBID:+  (job ${BUNDLE_JOBID})}"
    return 0
  fi
  echo "FAILED to submit ${label}" >&2
  failed=1
  return 1
}

case "${MODE}" in
  --dry)
    for task in ${ALL_TASKS}; do
      echo "=== preflight ${task} (DRY_RUN=1), submitting nothing ==="
      # MODEL_OUTPUT_DIR is normally the launcher's; a dry run has no bundle,
      # so hand the script a throwaway one purely to get past its check.
      env $(env_for "${task}" finetune) \
          MODEL_OUTPUT_DIR="${DRY_RUN_OUTPUT_DIR}" \
          STAGES="finetune:eval:compare" DRY_RUN=1 \
          bash "${REPO_ROOT}/slurm/benchmark_task.sbatch"
      echo
    done
    ;;
  --print)
    for task in ${TASKS} ${EXTRA_TASKS}; do
      submit_stage "${task}" finetune from_scratch "${task} finetune"
      echo
    done
    echo "# evaluation is submitted separately, once checkpoints exist:"
    echo "#   bash slurm/eval_checkpoint.sh --task <task> --checkpoint <dir>"
    ;;
  submit)
    cd "${REPO_ROOT}"
    bundle_require || exit 2
    submitted=() ; failed=0
    # FINETUNES ONLY. The eval job cannot be queued alongside its finetune any
    # more: bundle-sbatch wants --checkpoint to be an existing physical
    # directory at submit time, and the checkpoint does not exist until the
    # finetune has run. The --dependency=afterok chain went with it, since
    # --parsable is the launcher's and there is no job id to depend on.
    for task in ${TASKS} ${EXTRA_TASKS}; do
      submit_stage "${task}" finetune from_scratch "${task} finetune on ${PARTITION}"
    done
    if ((${#submitted[@]})); then
      echo
      echo "Submitted ${#submitted[@]} finetune(s):"
      printf '  %s\n' "${submitted[@]}"
      echo
      echo "Watch them with:"
      echo "  squeue -u ${WHOAMI} -o '%.10i %.20P %.70j %.9T %.10M %.20R'"
      echo
      echo "THEN, once a finetune has finished, evaluate its checkpoints --"
      echo "one invocation per checkpoint, each filed in the results library:"
      echo "  bash slurm/eval_checkpoint.sh --task <task> --seed-offset 1 \\"
      echo "      --checkpoint <output_dir>/checkpoint-<N>"
    fi
    exit ${failed}
    ;;
  *)
    echo "usage: bash slurm/submit_benchmark.sh [--print|--dry]" >&2
    echo "  (no argument)  submit one FINETUNE per task" >&2
    echo "  --dry          local preflight only, submits nothing" >&2
    echo "  --print        show the bundle-sbatch command lines" >&2
    echo >&2
    echo "  TASKS=\"a b c\"  EXTRA_TASKS=\"d\"  VARIANTS=...  EPISODES=N  GPUS=N" >&2
    echo "  PARTITION=<train>  EVAL_PARTITION=<eval>  are also overridable." >&2
    echo "  TIME_LIMIT=10:00:00 (training) and EVAL_TIME_LIMIT= (unset) set --time;" >&2
    echo "  empty means no --time, i.e. the partition maximum." >&2
    echo "  Evaluation lives in slurm/eval_checkpoint.sh, one checkpoint each." >&2
    echo >&2
    echo "  There is no --test-only: that flag belongs to the launcher." >&2
    exit 2
    ;;
esac
