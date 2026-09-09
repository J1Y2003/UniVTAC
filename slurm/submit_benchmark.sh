#!/bin/bash
# Submitter for the GR00T N1.7 x UniVTAC benchmark: ONE FINETUNE PER TASK.
#
# Evaluation is not submitted here. It belongs on a different partition (lab
# policy: `background`, never sjw_alinlab), one job holds one allocation on one
# partition, and bundle-sbatch wants --checkpoint to be an existing physical
# directory that does not exist until training has finished. So evaluation is
# slurm/eval_checkpoint.sh, one checkpoint per invocation, each result filed in
# the results library. Run from the repo root:
#
#   bash slurm/submit_benchmark.sh                 # submit
#   bash slurm/submit_benchmark.sh --print         # show the command only
#   bash slurm/submit_benchmark.sh --dry           # DRY_RUN=1 locally, no submit
#
# The recipe is 30,000 steps at batch 64, retaining checkpoint-10000,
# checkpoint-20000 and checkpoint-30000 -- the three points of the success-rate
# curve that decides the step count. It lives in benchmark_task.sbatch.
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
#   EXTRA_TASKS  lift_bottle, submitted LAST. There is no --dependency any
#                more (--parsable is the launcher's, so there is no job id to
#                depend on); if it competing for a GPU becomes a problem, hold
#                it with `scontrol hold <jobid>`. EXTRA_TASKS="" skips it.
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
# Evaluation does NOT happen here. EVAL_PARTITION and EVAL_TIME_LIMIT moved to
# slurm/eval_checkpoint.sh, which is where they now take effect; setting them
# for this script does nothing. Lab policy keeps evaluation off sjw_alinlab, a
# job holds one allocation on one partition, and an eval job must declare an
# existing physical --checkpoint that does not exist until training has
# finished. So this script submits finetunes only; eval_checkpoint.sh evaluates
# ONE checkpoint per invocation and owns the `background` choice itself.
#
# GPUS stays 1, and do not raise it without fixing the cause first:
# launch_finetune.py wraps the model in nn.DataParallel for --num-gpus > 1,
# which fails with "module must have its parameters and buffers ... on device:
# cuda:0 ... but found one on device: cpu". More GPUs would also change the
# effective batch size rather than just the speed, and whatever you pick is
# locked in for both variants by the recipe pinning in benchmark_task.sbatch.
GPUS="${GPUS:-1}"
# Walltime. A job with no --time is assumed to want the partition maximum
# (2 days), so backfill can only ever start it in a 2-day gap. A realistic limit
# makes it eligible for far more gaps, which on a contended queue is the
# difference between starting tonight and starting tomorrow.
#
# Training is RESUMABLE, so a limit that turns out short costs a resubmission
# rather than the run. The arithmetic: the first real insert_hole finetune did
# 10,000 steps in 117 minutes, i.e. 0.70 s/it, so benchmark_task.sbatch's
# MAX_STEPS=30000 is ~5.9 h of stepping plus model load, dataset statistics and
# three checkpoint writes -- call it 6.2 h. 9 h is ~1.45x that.
#
# Note the 1.89 s/it in docs/SETUP.md and the cuDNN comments: that was a
# 20-step smoke test, two of whose steps were checkpoint writes. It is not the
# production rate and should not be used to size a walltime.
#
# Raise this if you train more than one variant in a single job -- the finetune
# loop inside benchmark_task.sbatch is sequential, so VARIANTS="tactile
# baseline_finetuned" needs roughly double.
TIME_LIMIT="${TIME_LIMIT:-9:00:00}"
# Passed straight through, so any format sbatch accepts works ("9:00:00",
# "8:00", "1-12:00:00"). Empty means no --time at all.
#
# It may not exceed the partition maximum or the job pends forever with
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
# been queued. Ordering is all we have now -- the --dependency gate went with
# --parsable -- so it can still take a GPU a reported task wants; `scontrol
# hold <jobid>` is the remedy. `lift_bottle` is here because it is
# deliberately NOT one of the three reported tasks: it is the task any
# hyperparameter sweep is allowed to touch without fitting the numbers we
# report (see docs/ABLATION.md#comparability-with-univtacs-act). Having a
# trained lift_bottle model is still useful -- it is the fourth data point and
# the sweep substrate -- it just must not compete for the queue.
#
# Set EXTRA_TASKS="" to submit the three priority tasks alone.
#
# `-` and not `:-`, deliberately: with `:-` an explicitly empty EXTRA_TASKS
# falls back to lift_bottle, so the documented way to skip it silently
# submitted a fourth job instead. `-` honours an empty value and still
# defaults when the variable is unset.
EXTRA_TASKS="${EXTRA_TASKS-lift_bottle}"
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

# NOT `${WANDB_API_KEY:+online}${WANDB_API_KEY:-offline}`. That looks like a
# neat one-liner and it PRINTS THE KEY: when the variable is set, `:+` yields
# "online" and `:-` yields the VALUE, so the two concatenate into
# "online<the-actual-api-key>" in the terminal, in the scrollback, and
# anywhere that output gets pasted. Never render a secret through a
# default-value expansion.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_STATE="online"
else
  WANDB_STATE="offline (no WANDB_API_KEY)"
fi

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
  max_steps "${MAX_STEPS:-<benchmark_task.sbatch default: 30000>}" \
  save_steps "${SAVE_STEPS:-<benchmark_task.sbatch default: 10000>}" \
  save_limit "${SAVE_TOTAL_LIMIT:-<benchmark_task.sbatch default: 4>}" \
  gpus "${GPUS}" \
  wckey "${WCKEY}" \
  tasks "${TASKS}" \
  extra_tasks "${EXTRA_TASKS:-<none>}" \
  task_config "${TASK_CONFIG}" \
  variants "${VARIANTS}" \
  episodes "${EPISODES}" \
  wandb "${WANDB_STATE}"
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
  echo "  FAIL  no HF_TOKEN and no token file under HF_HOME=${HF_HOME}." >&2
  echo "        The job would fail loading nvidia/Cosmos-Reason2-2B (401)." >&2
  echo "        Fix: export HF_TOKEN=hf_...   then re-run this." >&2
  FAIL=1
else
  # PRESENCE IS NOT ACCESS. A token that exists but has not been granted the
  # gated repo looks identical until the request is actually made, so probe it.
  # This runs on a login node, needs no GPU, and takes a second -- the
  # alternative is discovering it a minute into a job that already holds an A100.
  #
  # benchmark_task.sbatch probes the same thing at job start; that stays as the
  # last line of defence, since a token can be revoked between submit and run.
  #
  # A probe that cannot RUN (no huggingface_hub, no network from the login node)
  # is a warning, not a failure -- it says nothing about the token. Only a probe
  # that runs and is refused fails the submission.
  HF_PROBE=$("${GROOT_PYTHON}" - <<'PYHF' 2>&1
import sys
try:
    from huggingface_hub import HfApi
except Exception as exc:
    print("SKIP cannot import huggingface_hub: %s" % exc)
    sys.exit(0)
api = HfApi()
try:
    user = api.whoami().get("name", "?")
except Exception as exc:
    print("BAD token rejected by the hub (%s: %s)"
          % (type(exc).__name__, str(exc)[:160]))
    sys.exit(0)
for repo in ("nvidia/GR00T-N1.7-3B", "nvidia/Cosmos-Reason2-2B"):
    try:
        api.model_info(repo)
    except Exception as exc:
        print("DENIED %s for user %s (%s: %s)"
              % (repo, user, type(exc).__name__, str(exc)[:160]))
        sys.exit(0)
print("OK %s" % user)
PYHF
)
  case "${HF_PROBE}" in
    OK*)
      echo "  ok  HF token valid, gated repos readable (hub user ${HF_PROBE#OK })"
      ;;
    SKIP*)
      echo "  WARN  could not verify the HF token: ${HF_PROBE#SKIP }" >&2
      echo "        Not a failure -- the job re-checks before loading weights." >&2
      ;;
    BAD*|DENIED*)
      echo "  FAIL  ${HF_PROBE}" >&2
      echo "        Create a token with access at" >&2
      echo "          https://huggingface.co/settings/tokens" >&2
      echo "        and request access to the gated repo at its model page." >&2
      echo "        Submitting now would burn a queue slot for a 401." >&2
      FAIL=1
      ;;
    *)
      echo "  WARN  HF token probe returned something unexpected:" >&2
      printf '        %s\n' "${HF_PROBE}" >&2
      ;;
  esac
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
  # `finetune` is the only stage this script submits. Evaluation moved out to
  # slurm/eval_checkpoint.sh, so there is no eval branch here to fall into.
  if [[ "${stage}" != "finetune" ]]; then
    echo "env_for: '${stage}' -- this script submits finetunes only." >&2
    echo "  Evaluation is slurm/eval_checkpoint.sh, one checkpoint each." >&2
    return 1
  fi
  # Colon-separated: a space inside a value would need quoting through two
  # command-line regions. benchmark_task.sbatch splits ':' back out.
  printf '%s
' "STAGES=finetune"
  printf '%s
'     "UNIVTAC_JOB_CONFIG=1"     "UNIVTAC_ROOT=${UNIVTAC_ROOT}"     "UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"     "GROOT_ROOT=${GROOT_ROOT}"     "GROOT_PYTHON=${GROOT_PYTHON}"     "HF_HOME=${HF_HOME}"     "DATA_ROOT=${DATA_ROOT}"     "TASK=${task}"     "TASK_CONFIG=${TASK_CONFIG}"     "EPISODES=${EPISODES}"     "VARIANTS=${VARIANTS// /:}"     "DRY_RUN=0"
  # The recipe. Forwarded ONLY when explicitly set, so benchmark_task.sbatch
  # stays the single source of truth for the defaults (30000 / 10000 / 4) and
  # there is no second copy of those numbers here to drift out of step.
  #
  # It is forwarded at all because inheritance is not a contract: these reach
  # the job today only because `env` keeps the surrounding environment and
  # sbatch propagates it. Naming them puts the recipe in `--print` output and
  # in the bundle's evidence, which is where a reader looks to find out what
  # was actually trained. GPUS is deliberately absent -- the job derives
  # NUM_GPUS from CUDA_VISIBLE_DEVICES, i.e. from the allocation it really got.
  local var
  for var in MAX_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT \
             LEARNING_RATE WEIGHT_DECAY ALLOW_RECIPE_CHANGE; do
    [[ -n "${!var+set}" ]] && printf '%s\n' "${var}=${!var}"
  done
  # MODEL_OUTPUT_DIR is deliberately absent: bundle-sbatch injects it, and the
  # guide is explicit that we must not supply it.
  return 0
}

# slurm_args_for <task> <stage>
#
# Region 2 of the bundle-sbatch command line: Slurm options, attached long form
# only. --output/--error/--export/--parsable/--test-only are the launcher's and
# must not appear; --job-name/--wckey/--partition/--time moved here out of the
# #SBATCH headers.
slurm_args_for() {
  local task="$1" stage="$2"
  # One partition, because this script submits one kind of job. Evaluation's
  # partition belongs to slurm/eval_checkpoint.sh.
  # No --cpus-per-task and no --mem: site rules, rejected by the submit filter.
  BUNDLE_SLURM_ARGS=(
    --job-name="$(job_name_for "${task}" "${stage}")"
    --wckey="${WCKEY}"
    --partition="${PARTITION}"
    --gres="gpu:${GPUS}"
  )
  [[ -n "${TIME_LIMIT}" ]] && BUNDLE_SLURM_ARGS+=(--time="${TIME_LIMIT}")
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
      echo "THEN, once a finetune has finished, evaluate its three retained"
      echo "checkpoints -- 10000, 20000, 30000 -- one invocation each, filed in"
      echo "the results library:"
      echo "  for N in 10000 20000 30000; do"
      echo "    bash slurm/eval_checkpoint.sh --task <task> --seed-offset 1 \\"
      echo "        --checkpoint <output_dir>/checkpoint-\$N"
      echo "  done"
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
    echo "  PARTITION=<train> is overridable. EVAL_PARTITION/EVAL_TIME_LIMIT" >&2
    echo "  belong to slurm/eval_checkpoint.sh and do nothing here." >&2
    echo "  TIME_LIMIT=9:00:00 sets --time; empty means no --time at all," >&2
    echo "  i.e. the partition maximum." >&2
    echo "  MAX_STEPS / SAVE_STEPS / SAVE_TOTAL_LIMIT override the recipe;" >&2
    echo "  unset, benchmark_task.sbatch's 30000 / 10000 / 4 apply, which" >&2
    echo "  retains exactly the 10000, 20000 and 30000 checkpoints." >&2
    echo "  Evaluation lives in slurm/eval_checkpoint.sh, one checkpoint each." >&2
    echo >&2
    echo "  There is no --test-only: that flag belongs to the launcher." >&2
    exit 2
    ;;
esac
