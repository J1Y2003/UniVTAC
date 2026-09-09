#!/bin/bash
# Submit one finetune per task: 30,000 steps at batch 64, retaining checkpoints
# 10k/20k/30k. The recipe itself lives in benchmark_task.sbatch.
#
#   bash slurm/submit_benchmark.sh                 # submit
#   bash slurm/submit_benchmark.sh --print         # show the command only
#   bash slurm/submit_benchmark.sh --dry           # DRY_RUN=1 locally, no submit
#
# Evaluation is not submitted here: it belongs on `background` rather than the
# training partition, one job holds one allocation on one partition, and
# bundle-sbatch needs --checkpoint to exist at submit time. Use
# slurm/eval_checkpoint.sh once a finetune has finished.
#
# Every default below is overridable in the environment, e.g.
# `TASK=insert_hole MAX_STEPS=2000 bash slurm/submit_benchmark.sh`. TASKS holds
# the three reported tasks; EXTRA_TASKS holds lift_bottle and is submitted last.
set -uo pipefail

# --------------------------------------------------------------------------- #
# Paths -- override any of these in the environment
# --------------------------------------------------------------------------- #
WORKSPACE="${WORKSPACE:-${HOME}/jaewon/workspace}"
# USER is not exported in every shell; squeue and the paths below need a name.
WHOAMI="${USER:-$(id -un)}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Every submission goes through bundle-sbatch; the contract lives in one place.
# shellcheck source=slurm/common.sh
. "${REPO_ROOT}/slurm/common.sh"
UNIVTAC_ROOT="${UNIVTAC_ROOT:-${WORKSPACE}/UniVTAC-sim}"
GROOT_ROOT="${GROOT_ROOT:-${WORKSPACE}/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:-${HOME}/miniconda3/envs/UniVTAC/bin/python}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/groot-data}"
HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"
# A --dry run has no bundle, so nothing injects MODEL_OUTPUT_DIR, but
# benchmark_task.sbatch requires one. Hand it a throwaway; nothing is written.
DRY_RUN_OUTPUT_DIR="${DRY_RUN_OUTPUT_DIR:-${TMPDIR:-/tmp}/univtac-dry-run}"

# --------------------------------------------------------------------------- #
# Job shape
# --------------------------------------------------------------------------- #
# `debug` is the cluster default and caps at 3 hours, which is why training jobs
# sit pending with REASON=PartitionTimeLimit. Name a 2-day partition instead.
PARTITION="${PARTITION:-sjw_alinlab}"
# EVAL_PARTITION and EVAL_TIME_LIMIT live in slurm/eval_checkpoint.sh now;
# setting them for this script does nothing.
#
# GPUS stays 1. --num-gpus > 1 makes launch_finetune.py wrap the model in
# nn.DataParallel, which dies with "module must have its parameters and buffers
# ... on device: cuda:0 ... but found one on device: cpu". It would also change
# the effective batch size, and the recipe pin locks in whatever you choose.
GPUS="${GPUS:-1}"
# Walltime. Without one, a job is assumed to want the partition maximum, so
# backfill can only start it in a 2-day gap; a realistic limit makes it eligible
# for many more. 9 h is ~1.45x the measured ~6.2 h: 30,000 steps at 0.70 s/it
# (insert_hole did 10,000 in 117 min) plus model load, dataset statistics and
# three checkpoint writes. Do NOT size this off the 1.89 s/it in docs/SETUP.md
# -- that was a 20-step smoke test, two of whose steps wrote a checkpoint.
#
# Roughly double it if VARIANTS names more than one: the finetune loop inside
# benchmark_task.sbatch is sequential.
TIME_LIMIT="${TIME_LIMIT:-9:00:00}"
# Any format sbatch accepts works. Empty means no --time at all; over the
# partition maximum means pending forever with REASON=PartitionTimeLimit
# (`scontrol show partition <name> | grep MaxTime`).
WCKEY="${WCKEY:-project-short-name:sub_4dpdata}"

# One job per task, matching UniVTAC's ACT, which trains one policy per task.
# Separate jobs also mean each is independently resumable, each fits a backfill
# gap, and one task failing does not block the others. `TASK=x` still works.
TASKS="${TASKS:-${TASK:-insert_hole insert_tube pull_out_key}}"
# Submitted last, and deliberately not one of the three reported tasks:
# lift_bottle is the only task a hyperparameter sweep may touch without fitting
# the reported numbers (docs/BENCHMARK.md#comparability-with-univtacs-act). No
# --dependency holds it back any more, so it can still take a GPU a reported
# task wants -- `scontrol hold <jobid>` is the remedy. EXTRA_TASKS="" skips it.
#
# `-` not `:-`, deliberately: with `:-`, the documented EXTRA_TASKS="" fell back
# to lift_bottle and silently submitted a fourth job.
EXTRA_TASKS="${EXTRA_TASKS-lift_bottle}"
ALL_TASKS="${TASKS} ${EXTRA_TASKS}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
# 100, from the paper's "evaluated over 100 test rollouts". A 50-episode
# interval against their 100-episode number is not like-for-like.
EPISODES="${EPISODES:-100}"
# Vision only for now. The tactile pipeline still works; set
# VARIANTS="tactile baseline_finetuned" to run the full ablation again.
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

# The job's configuration travels in the ENVIRONMENT: bundle-sbatch owns
# --export, and sbatch propagates the submitting environment by default.
# env_for prints `NAME=VALUE` pairs for `env`, one per line.
#
# UNIVTAC_JOB_CONFIG is the canary -- if this environment ever stops reaching
# the job, benchmark_task.sbatch stops instead of training TASK's default.
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
  # The recipe, forwarded only when explicitly set so benchmark_task.sbatch
  # stays the single source of truth for the defaults (30000 / 10000 / 4).
  # Naming them at all puts the recipe in `--print` output and in the bundle's
  # evidence rather than leaving it to environment inheritance. GPUS is absent
  # deliberately: the job derives NUM_GPUS from CUDA_VISIBLE_DEVICES, i.e. from
  # the allocation it actually got.
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
