#!/bin/bash
# One task's stages, unattended: finetune each variant, evaluate each, compare.
# Resumable, so a walltime kill plus a plain resubmit continues where it stopped.
#
# NOT normally submitted as one job any more. Training and evaluation belong on
# DIFFERENT PARTITIONS here -- training on sjw_alinlab, evaluation only on
# `background` -- and one job holds one allocation on one partition. So
# slurm/submit_benchmark.sh submits this script TWICE per task:
#
#   STAGES=finetune        on the training partition, by submit_benchmark.sh
#   STAGES="eval compare"  on `background` -- but nothing submits this any
#                          more: evaluation is slurm/eval_checkpoint.sh, one
#                          checkpoint per job, because bundle-sbatch needs
#                          --checkpoint to exist at submit time and --parsable
#                          is the launcher's, so there is no id to depend on.
#
# STAGES defaults to all three, so a single-partition run still works for a
# smoke test or a local `bash` invocation.
#
# Submitted through bundle-sbatch, never plain sbatch:
#
#   env TASK=lift_bottle EPISODES=50 STAGES=finetune UNIVTAC_JOB_CONFIG=1 #     bundle-sbatch --job-kind train --checkpoint from_scratch #     --code-git-root "$REPO_ROOT" -- #     --job-name=univtac-groot-per-task-finetune-only-vision-only-lift_bottle #     --wckey=project-short-name:sub_4dpdata --partition=sjw_alinlab #     --time=10:00:00 -- slurm/benchmark_task.sbatch
#
# Prefer slurm/submit_benchmark.sh, which resolves every path and both regions
# of that command line for you. Do not raise --gres above gpu:1 -- see the note
# on the #SBATCH --gres line below.
#
# BEFORE submitting, verify it without spending GPU time. DRY_RUN=1 runs the
# full preflight, prints the exact arguments every stage would use, checks the
# aggregator imports, and exits -- cheap enough for a login node:
#
#   bash -n slurm/benchmark_task.sbatch            # syntax only, instant
#   DRY_RUN=1 bash slurm/benchmark_task.sbatch     # paths, datasets, plan
#
# There is no --test-only rehearsal any more: that flag belongs to the launcher
# and cannot be passed through. DRY_RUN and `submit_benchmark.sh --print` are
# what is left.
#
# Stages, in order:
#   1. preflight     -- every path and dataset checked BEFORE any GPU time is spent
#   2. finetune      -- VARIANT=tactile, then VARIANT=baseline_finetuned (same recipe)
#   3. evaluate      -- each finetuned checkpoint on TASK
#   4. compare       -- success rates, Wilson intervals, delta
#
# Two properties that matter for an unattended run:
#
# * **Resumable.** Each stage records a marker and is skipped if its output
#   already exists. A walltime kill, then a plain resubmit, continues where it
#   stopped rather than redoing hours of training.
# * **No single failure ends the night.** A failed stage is recorded and the
#   run continues with everything still possible -- if the tactile finetune
#   dies, the baseline still trains and evaluates, so the morning brings a
#   diagnosis instead of an empty results directory.
#
# This cluster's submit filter requires a job name LONGER THAN 50 characters
# and rejects the submission outright otherwise. Do not shorten these.
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
#   * MODEL_OUTPUT_DIR is INJECTED BY THE LAUNCHER and equals CODE_OUTPUT_DIR,
#     i.e. <bundle>/code-output. Do not set it, do not pass it, and do not
#     expect to choose a path under /rlwrld-unified-checkpoints/jimin/jaewon.
#     Bundle storage is managed storage with its own retention, and nothing
#     here copies checkpoints out of it.
#
# Everything the launcher owns has been removed from the directives below:
# --job-name, --output, --error, --export, --wckey, --partition and --time all
# live on the bundle-sbatch command line now. What remains is the resource
# request, in attached long form.
#
# 1 GPU, deliberately: --num-gpus > 1 makes launch_finetune.py wrap the model
# in nn.DataParallel, which dies with "module must have its parameters ... on
# device: cpu". Overridden by slurm/submit_benchmark.sh's --gres.
#SBATCH --gres=gpu:1

# Deliberately NOT `set -e`: a failing stage must be recorded and stepped over,
# not abort the job. Stage failures are tracked explicitly instead.
set -uo pipefail

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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# bundle-sbatch owns --export, so this job's configuration is NOT passed as an
# explicit variable list any more -- it arrives only because the submitter
# exported it and sbatch propagates the submitting environment by default. If
# that ever stops being true the variables below would quietly fall back to
# their defaults and the job would train the WRONG TASK while looking healthy.
#
# So the submitters set this sentinel, and a real job without it stops here.
# (Plain `bash slurm/benchmark_task.sbatch` for a dry run is unaffected: the
# check only applies inside a Slurm allocation.)
if [[ -n "${SLURM_JOB_ID:-}" && -z "${UNIVTAC_JOB_CONFIG:-}" ]]; then
  echo "error: UNIVTAC_JOB_CONFIG is unset, so this job's environment did not" >&2
  echo "       reach it and TASK/VARIANTS/STAGES would silently take their" >&2
  echo "       defaults. Submit through slurm/submit_benchmark.sh, which" >&2
  echo "       exports the configuration before calling bundle-sbatch." >&2
  echo "       If you are submitting by hand, export UNIVTAC_JOB_CONFIG=1" >&2
  echo "       alongside the rest of the configuration." >&2
  exit 2
fi

TASK="${TASK:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
# 100 test rollouts, matching the paper's evaluation protocol.
EPISODES="${EPISODES:-100}"
# Execute 16 of each 40-action chunk, then re-plan from a fresh observation.
# Valid because the contract is 1 <= execution_horizon <= action_horizon.
# ~19 policy calls per episode at step_lim=300. Must be identical across
# variants: it sets how closed-loop the controller is.
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
# ARM= was the old name for this. Honour it rather than ignoring it: an
# unrecognised environment variable fails *silently*, so a stale
# `ARM=baseline_finetuned` command would quietly run the tactile variant
# instead -- a wrong result that looks like a right one. (A stale --arm flag
# fails loudly on the CLI, so no alias is needed there.)
if [[ -n "${ARMS:-}" && -z "${VARIANTS:-}" ]]; then
  echo "[job] note: ARMS is deprecated, use VARIANTS (taking ARMS=${ARMS})" >&2
  VARIANTS="${ARMS}"
fi
VARIANTS="${VARIANTS:-tactile baseline_finetuned}"
# Accept ':' and ',' as separators as well as spaces. A value crossing the
# submitters' `env NAME=VALUE` list is easiest to keep whole when it contains no
# spaces at all, and a comma still reads as a list separator in too many places
# to be safe. So the submitters send colons and this turns them back into words.
VARIANTS="${VARIANTS//[,:]/ }"

# Which stages this job runs. The default is all three, so a plain
# `bash slurm/benchmark_task.sbatch` still does the whole task end to end.
#
# It is split because THE TWO STAGES BELONG ON DIFFERENT PARTITIONS: training
# runs on sjw_alinlab, evaluation is only allowed on `background`. One job
# holds one allocation on one partition, so finetune-then-evaluate in a single
# job would evaluate wherever it trained. slurm/submit_benchmark.sh therefore
# submits two jobs per task -- STAGES=finetune on the training partition, then
# STAGES="eval compare" on `background`, gated behind it with --dependency.
#
#   STAGES=finetune        train only, leave the checkpoint for a later job
#   STAGES="eval compare"  evaluate an existing checkpoint and aggregate
#
# Unknown names are rejected rather than silently ignored: a typo'd
# STAGES=finetne would otherwise run nothing and look like a clean success.
STAGES="${STAGES:-finetune eval compare}"
STAGES="${STAGES//[,:]/ }"   # same reason as VARIANTS above
for _stage in ${STAGES}; do
  case "${_stage}" in
    finetune|eval|compare) ;;
    *) echo "error: unknown stage '${_stage}' in STAGES='${STAGES}'." >&2
       echo "       Valid stages: finetune eval compare" >&2
       exit 2 ;;
  esac
done
stage_enabled() {
  local want="$1" have
  for have in ${STAGES}; do [[ "${have}" == "${want}" ]] && return 0; done
  return 1
}

# The training recipe: 30,000 steps at batch 64, retaining the 10k, 20k and 30k
# checkpoints and nothing else.
#
# 30,000 is a DELIBERATE DEPARTURE from GR00T's own --max-steps 10000 default
# and has to be disclosed alongside the numbers -- docs/ABLATION.md's recipe
# table marks it as such. The reason: 10,000 steps over the full 100 released
# episodes (~28.9k samples, 452 steps/epoch) is only ~22 epochs, and nothing in
# this setup can tell us whether that is enough. There is no validation split
# and launch_finetune.py exposes no eval metric, so the ONLY instrument is
# success rate as a function of step count. 30,000 (~66 epochs) is far enough
# out that the curve either turns or plainly does not.
#
# SAVE_STEPS=10000 makes the saves land exactly on 10000, 20000 and 30000 --
# the three points of that curve, nothing else. Chosen, not incidental:
#   * at SAVE_STEPS=1000 a limit of 3 deletes every early checkpoint long
#     before training ends, which is what forced the hand-copying of
#     checkpoints on the first insert_hole run;
#   * a 3B checkpoint carries optimizer + scheduler + RNG state and measured
#     ~26 GB (an earlier comment here estimated ~40 GB, which was high), so
#     three of them is ~77 GB per variant and fits a bundle comfortably;
#   * writing one stalls the step it lands on -- 9.82 s against a sub-second
#     steady state -- so three writes rather than thirty is also the cheap
#     setting.
# SAVE_TOTAL_LIMIT is a ceiling, not a target: 4 leaves one slot spare so that
# an extra end-of-training save could never evict checkpoint-10000, while only
# three checkpoints are ever actually written.
#
# The cost of sparse saves is resume granularity. A walltime kill or node
# failure now rolls back to the last 10k boundary -- up to ~1.9 h of redone
# work at the measured 0.70 s/it. That is the trade being made knowingly.
#
# There is still no metric-based "best checkpoint" to select automatically:
# launch_finetune.py exposes no eval metric or load-best-at-end option, so the
# last checkpoint is the final model by definition, and this script points a
# stable `<output_dir>/final` symlink at the highest-numbered one. Preferring
# 10k or 20k over 30k is therefore an explicit act performed at EVALUATION
# time, on a dev seed block -- see docs/ABLATION.md, "Choosing the step count".
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"

# Left unset, so launch_finetune.py's own defaults apply. Set them only for a
# deliberate sweep; the pinned recipe below records whatever was used so a
# resubmission cannot silently train the second variant differently.
LEARNING_RATE="${LEARNING_RATE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
HP_ARGS=""
[[ -n "${LEARNING_RATE}" ]] && HP_ARGS+=" --learning-rate ${LEARNING_RATE}"
[[ -n "${WEIGHT_DECAY}" ]] && HP_ARGS+=" --weight-decay ${WEIGHT_DECAY}"

DATA_ROOT="${DATA_ROOT:-${HOME}/jaewon/workspace/groot-data}"
# Where checkpoints go. Under bundle-sbatch this is not ours to choose: the
# launcher creates one output bundle per submission and injects
#   MODEL_OUTPUT_DIR = CODE_OUTPUT_DIR = <bundle>/code-output
# so every submission writes to a DIFFERENT directory. The old
# /rlwrld-unified-checkpoints layout and its validation are gone with it.
#
# Two consequences worth stating plainly:
#
#  * A resubmission does not see the previous run's checkpoints. Resuming is no
#    longer automatic -- pass the previous physical checkpoint directory as
#    `bundle-sbatch --checkpoint`, which arrives here as CHECKPOINT_DIR.
#  * Bundle storage is managed storage with its own retention, and we do NOT
#    copy checkpoints out of it. If a bundle ages out, that finetune is gone
#    and gets resubmitted -- at ~6.2 h per task that is cheaper than keeping a
#    private duplicate of every checkpoint on NFS. Evaluate while it is live.
: "${MODEL_OUTPUT_DIR:?MODEL_OUTPUT_DIR is injected by bundle-sbatch. Submit through slurm/submit_benchmark.sh, not with plain sbatch}"
CKPT_ROOT="${CKPT_ROOT:-${MODEL_OUTPUT_DIR}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/eval_result}"
# Stage markers must OUTLIVE the bundle: they are what makes a resubmission skip
# work that is already done, and a per-submission bundle would hand every run a
# clean slate. Keep them next to the repo unless told otherwise.
STAGE_DIR="${STAGE_DIR:-${REPO_ROOT}/.stages}"

UNIVTAC_ROOT="${UNIVTAC_ROOT:?set UNIVTAC_ROOT to the UniVTAC checkout}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:?set UNIVTAC_PYTHON to the UniVTAC/Isaac Lab interpreter}"
GROOT_ROOT="${GROOT_ROOT:?set GROOT_ROOT to the Isaac-GR00T checkout}"
GROOT_PYTHON="${GROOT_PYTHON:?set GROOT_PYTHON to the GR00T interpreter}"

# hf_transfer is absent from a uv-created venv unless explicitly added, and
# HF_HUB_ENABLE_HF_TRANSFER=1 without it is a hard error mid-download. Default
# off for an unattended run; the weights are already cached by now anyway.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"
export PYTHONUNBUFFERED=1

# Default OFF. This was 1 for a while on the belief that cuDNN could not
# initialise on this cluster's driver; the real cause was a cuDNN in the GR00T

# SLURM re-indexes allocated GPUs to 0..N-1 inside the job, so counting
# CUDA_VISIBLE_DEVICES gives the usable width for data-parallel training.
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-0}")}"

mkdir -p logs "${STAGE_DIR}" "${CKPT_ROOT}" "${RESULTS_DIR}"

STAGE_LOG_DIR="${REPO_ROOT}/logs/benchmark-${TASK}-${SLURM_JOB_ID:-$$}"
mkdir -p "${STAGE_LOG_DIR}"

declare -a SUMMARY=()
# Both variants are evaluated in the same job, so they would otherwise derive the
# same port from SLURM_JOB_ID. The first server is killed by its own EXIT trap,
# but a socket in TIME_WAIT would still make the second bind fail, so hand each
# variant its own port.
PORT_BASE="${PORT_BASE:-$((20000 + (${SLURM_JOB_ID:-$$} % 15000)))}"
PORT_OFFSET=0

banner() {
  echo
  echo "=========================================================="
  echo " $*"
  echo " $(date '+%Y-%m-%d %H:%M:%S')  node=$(hostname)"
  echo "=========================================================="
}

record() { SUMMARY+=("$1"); echo "[stage] $1"; }

# --------------------------------------------------------------------------- #
# 1. Preflight -- everything checkable, checked before any GPU time is spent
# --------------------------------------------------------------------------- #
banner "preflight"
echo " task=${TASK} config=${TASK_CONFIG} variants='${VARIANTS}' episodes=${EPISODES}"
echo " stages='${STAGES}'"
echo " gpus=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo " data=${DATA_ROOT}"
echo " ckpt=${CKPT_ROOT}"
echo " results=${RESULTS_DIR}"
echo " stage logs=${STAGE_LOG_DIR}"
if [[ "${USE_WANDB:-1}" != "1" ]]; then
  echo " wandb=disabled (USE_WANDB=0) -- stdout logs are the only training record"
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo " wandb=online project=${WANDB_PROJECT:-univtac-groot-ablation}"
else
  echo " wandb=offline (no WANDB_API_KEY) -- runs recorded under each OUTPUT_DIR/wandb,"
  echo "       upload later with: wandb sync <dir>/offline-run-*"
fi

FATAL=0
fatal() { echo "PREFLIGHT FAIL: $*" >&2; FATAL=1; }

for path in "${UNIVTAC_ROOT}" "${GROOT_ROOT}"; do
  [[ -d "${path}" ]] || fatal "not a directory: ${path}"
done
for interp in "${UNIVTAC_PYTHON}" "${GROOT_PYTHON}"; do
  [[ -x "${interp}" ]] || fatal "not executable: ${interp}"
done
[[ -f "${GROOT_ROOT}/gr00t/experiment/launch_finetune.py" ]] \
  || fatal "missing ${GROOT_ROOT}/gr00t/experiment/launch_finetune.py"
"${GROOT_PYTHON}" -c "import gr00t" 2>/dev/null \
  || fatal "${GROOT_PYTHON} cannot import gr00t (try: cd ${GROOT_ROOT} && uv sync)"
# Every GR00T checkpoint loads the gated nvidia/Cosmos-Reason2-2B backbone, so
# without credentials training dies with a 401 about a minute in -- after the
# job has a GPU, which on a contended cluster is an expensive way to learn it.
# Probe actual access, not just the presence of a token: a token that lacks
# access to the gated repo looks identical until the request is made.
#
# Note HF_HOME is redirected above, which also moves where a stored login is
# looked for ($HF_HOME/token, not ~/.cache/huggingface/token). Exporting
# HF_TOKEN avoids that entirely and takes precedence over any stored login.
if ! "${GROOT_PYTHON}" - <<'PYCHECK' 2>/dev/null
import sys
try:
    from huggingface_hub import HfApi
    HfApi().model_info("nvidia/Cosmos-Reason2-2B")
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}"[:300], file=sys.stderr)
    raise SystemExit(1)
PYCHECK
then
  fatal "cannot read the gated nvidia/Cosmos-Reason2-2B (needed by every GR00T
       checkpoint). Export a token with access, in the shell you submit from:
         export HF_TOKEN=hf_...
       The job inherits it from the submitting shell. Request access at
       https://huggingface.co/nvidia/Cosmos-Reason2-2B if you have none.
       HF_TOKEN is currently ${HF_TOKEN:+set}${HF_TOKEN:-UNSET}; HF_HOME=${HF_HOME}."
fi

# Same lesson as hf_transfer: do not pass a flag whose package may be absent.
# --use-wandb with no wandb installed would fail after the weights had loaded.
# Only the finetune stage passes --use-wandb, so an eval-only job must not be
# blocked by a missing wandb.
if stage_enabled finetune && [[ "${USE_WANDB:-1}" == "1" ]]; then
  "${GROOT_PYTHON}" -c "import wandb" 2>/dev/null \
    || fatal "USE_WANDB=1 but ${GROOT_PYTHON} cannot import wandb. Install it in
       that venv, or pass USE_WANDB=0 to rely on the stdout logs alone."
fi

# A missing or truncated dataset is the classic way a long unattended run wastes a
# night, so verify shape now rather than discovering it at step 0 of training.
for variant in ${VARIANTS}; do
  # Evaluation never opens the training dataset -- it talks to the policy server
  # and Isaac Sim -- so an eval-only job must not fail on one that has since been
  # archived off NFS by the 4-day retention.
  stage_enabled finetune || continue
  dataset="${DATA_ROOT}/univtac-${TASK}-${variant}"
  if [[ ! -d "${dataset}" ]]; then
    fatal "no dataset for variant '${variant}' at ${dataset}"
    continue
  fi
  for required in meta/info.json meta/modality.json meta/stats.json; do
    [[ -f "${dataset}/${required}" ]] || fatal "${dataset} is missing ${required}"
  done
  n_parquet=$(find "${dataset}/data" -name '*.parquet' 2>/dev/null | wc -l)
  n_videos=$(find "${dataset}/videos" -name '*.mp4' 2>/dev/null | wc -l)
  echo " ${variant}: ${n_parquet} parquet, ${n_videos} mp4  (${dataset})"
  [[ "${n_parquet}" -gt 0 ]] || fatal "${dataset} has no parquet shards"
done

[[ -f "${REPO_ROOT}/configs/modality/univtac_tactile_config.py" ]] \
  || fatal "missing configs/modality/univtac_tactile_config.py"
[[ -f "${REPO_ROOT}/configs/modality/univtac_baseline_config.py" ]] \
  || fatal "missing configs/modality/univtac_baseline_config.py"

# Pin the training recipe across runs. Both variants must train under identical
# conditions or the ablation measures the recipe, not the tactile modality --
# and the two variants are NOT necessarily trained by the same job invocation: a
# walltime kill mid-way means the second variant trains on a later resubmission,
# which the scheduler may hand a different number of GPUs. That matters because
# with data-parallel training --num-gpus multiplies the effective batch size, so
# the same --max-steps on 4 GPUs sees roughly twice the data as on 2.
#
# So record the recipe when the first variant starts and refuse to continue if a
# later run differs. Set ALLOW_RECIPE_CHANGE=1 to override deliberately (e.g.
# you are restarting the whole experiment from scratch).
RECIPE_FILE="${STAGE_DIR}/recipe-${TASK}.env"
# The pin exists to keep the two variants TRAINED alike; an eval-only job has no
# recipe, and checking one would fail it purely for holding a different GPU count.
if ! stage_enabled finetune; then
  echo " recipe: not checked (this job does not train)"
elif [[ -f "${RECIPE_FILE}" ]]; then
  RECORDED_NUM_GPUS=""; RECORDED_MAX_STEPS=""
  RECORDED_LEARNING_RATE=""; RECORDED_WEIGHT_DECAY=""
  # shellcheck source=/dev/null
  source "${RECIPE_FILE}"
  if [[ "${RECORDED_NUM_GPUS}" != "${NUM_GPUS}" \
     || "${RECORDED_MAX_STEPS}" != "${MAX_STEPS}" \
     || "${RECORDED_LEARNING_RATE}" != "${LEARNING_RATE:-groot-default}" \
     || "${RECORDED_WEIGHT_DECAY}" != "${WEIGHT_DECAY:-groot-default}" ]]; then
    if [[ "${ALLOW_RECIPE_CHANGE:-0}" == "1" ]]; then
      echo " WARNING: recipe changed (was ${RECORDED_NUM_GPUS} gpu / ${RECORDED_MAX_STEPS} steps," >&2
      echo " now ${NUM_GPUS} gpu / ${MAX_STEPS} steps). Proceeding because" >&2
      echo " ALLOW_RECIPE_CHANGE=1 -- the variants are no longer directly comparable." >&2
    else
      fatal "recipe mismatch: this task was started with ${RECORDED_NUM_GPUS} GPU(s),
       --max-steps ${RECORDED_MAX_STEPS}, lr=${RECORDED_LEARNING_RATE},
       wd=${RECORDED_WEIGHT_DECAY}; this run has ${NUM_GPUS} GPU(s),
       --max-steps ${MAX_STEPS}, lr=${LEARNING_RATE:-groot-default},
       wd=${WEIGHT_DECAY:-groot-default}. Training the two variants under different
       effective batch sizes or optimiser settings would confound the ablation.
       Either resubmit with
       --gres=gpu:${RECORDED_NUM_GPUS} and MAX_STEPS=${RECORDED_MAX_STEPS}, or pass
       ALLOW_RECIPE_CHANGE=1 and delete ${STAGE_DIR} to restart the experiment."
    fi
  else
    echo " recipe: ${NUM_GPUS} gpu, ${MAX_STEPS} steps (matches ${RECIPE_FILE})"
  fi
elif [[ "${DRY_RUN:-0}" != "1" ]]; then
  printf 'RECORDED_NUM_GPUS=%s\nRECORDED_MAX_STEPS=%s\nRECORDED_LEARNING_RATE=%s\nRECORDED_WEIGHT_DECAY=%s\n' \
    "${NUM_GPUS}" "${MAX_STEPS}" "${LEARNING_RATE:-groot-default}" "${WEIGHT_DECAY:-groot-default}" \
    >"${RECIPE_FILE}"
  echo " recipe: ${NUM_GPUS} gpu, ${MAX_STEPS} steps, lr=${LEARNING_RATE:-groot-default},"
  echo "         wd=${WEIGHT_DECAY:-groot-default} (recorded in ${RECIPE_FILE})"
fi

df -h "${CKPT_ROOT}" 2>/dev/null || true
# shellcheck disable=SC2086 - word splitting is the point here
N_VARIANTS=$(set -- ${VARIANTS}; echo $#)
# How many checkpoints actually survive: one per SAVE_STEPS, capped by the
# retention ceiling. ~26 GB each, measured on the insert_hole checkpoints
# (25,547,974,206 bytes); the ~40 GB this used to assume was high.
N_CKPT=$(( SAVE_STEPS > 0 ? MAX_STEPS / SAVE_STEPS : 1 ))
(( N_CKPT < 1 )) && N_CKPT=1
(( N_CKPT > SAVE_TOTAL_LIMIT )) && N_CKPT="${SAVE_TOTAL_LIMIT}"
echo " the space above must hold roughly $((N_VARIANTS * N_CKPT * 26)) GB"
echo " (${N_VARIANTS} variants x ${N_CKPT} retained checkpoints x ~26 GB each)"

if [[ "${FATAL}" -ne 0 ]]; then
  echo
  echo "Preflight failed. Nothing was run. Fix the above and resubmit." >&2
  exit 2
fi
record "preflight OK"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# launch_finetune.py's exact output layout is not pinned by this repo, so probe
# rather than assume: prefer the highest-numbered checkpoint-N directory, and
# fall back to the output root if the final model was written there directly.
latest_checkpoint() {
  local root="$1" best="" best_n=-1 dir n
  for dir in "${root}"/checkpoint-*; do
    [[ -d "${dir}" ]] || continue
    n="${dir##*checkpoint-}"
    [[ "${n}" =~ ^[0-9]+$ ]] || continue
    if (( n > best_n )); then best_n="${n}"; best="${dir}"; fi
  done
  if [[ -z "${best}" ]]; then
    if [[ -f "${root}/config.json" ]] \
       || compgen -G "${root}/*.safetensors" >/dev/null 2>&1; then
      best="${root}"
    fi
  fi
  printf '%s' "${best}"
}

# --------------------------------------------------------------------------- #
# 2. Finetune each variant
# --------------------------------------------------------------------------- #
for variant in ${VARIANTS}; do
  if ! stage_enabled finetune; then
    record "finetune ${variant}: SKIPPED (not in STAGES='${STAGES}')"
    continue
  fi
  out_dir="${CKPT_ROOT}/${TASK}-${variant}"
  marker="${STAGE_DIR}/finetune-${TASK}-${variant}.done"
  log="${STAGE_LOG_DIR}/finetune-${variant}.log"

  if [[ -f "${marker}" ]]; then
    record "finetune ${variant}: SKIPPED (already done: ${marker})"
    continue
  fi

  # A checkpoint without the marker means an earlier attempt was killed
  # mid-training. Resume it rather than either restarting from step 0 or
  # treating a partial run as finished -- launch_finetune.py has
  # --resume-from-checkpoint, which picks the latest checkpoint-* in output_dir.
  # It defaults to off precisely so an accidental rerun does not merge two
  # experiments, so it must be requested explicitly.
  resume_args=""
  existing=$(latest_checkpoint "${out_dir}")
  # Under bundle-sbatch, out_dir is a FRESH directory on every submission, so a
  # resumable run has nothing here to find. The previous run's checkpoint comes
  # in as CHECKPOINT_DIR (from `bundle-sbatch --checkpoint <physical path>`).
  # Link it into out_dir under its own name and --resume-from-checkpoint picks
  # it up, since that flag selects the highest-numbered checkpoint-N in
  # output_dir. A link, not a copy: these are ~40 GB each.
  #
  # Only the variant whose name the checkpoint carries may claim it -- with two
  # variants in one job, seeding both from one checkpoint would silently train
  # the baseline from the tactile weights.
  if [[ -z "${existing}" && -n "${CHECKPOINT_DIR:-}" ]]; then
    case "$(basename "${CHECKPOINT_DIR}")" in
      checkpoint-[0-9]*)
        if [[ "${CHECKPOINT_DIR}" == *"${TASK}-${variant}"* ]]; then
          mkdir -p "${out_dir}"
          ln -sfn "${CHECKPOINT_DIR}" "${out_dir}/$(basename "${CHECKPOINT_DIR}")"
          existing=$(latest_checkpoint "${out_dir}")
          echo "[stage] seeded ${out_dir} from CHECKPOINT_DIR=${CHECKPOINT_DIR}"
        else
          echo "[stage] CHECKPOINT_DIR=${CHECKPOINT_DIR} does not name" >&2
          echo "[stage] ${TASK}-${variant}; not seeding this variant with it" >&2
        fi
        ;;
      *)
        echo "[stage] CHECKPOINT_DIR=${CHECKPOINT_DIR} is not a checkpoint-N" >&2
        echo "[stage] directory; ignoring it for resume" >&2
        ;;
    esac
  fi
  if [[ -n "${existing}" ]]; then
    resume_args="--resume-from-checkpoint"
    echo "[stage] resuming ${variant} from ${existing}"
  fi

  banner "finetune ${variant}  (${NUM_GPUS} gpu, max_steps=${MAX_STEPS})"
  echo "[stage] log -> ${log}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would run slurm/finetune.sbatch with:"
    echo "[dry-run]   VARIANT=${variant}"
    echo "[dry-run]   DATASET=${DATA_ROOT}/univtac-${TASK}-${variant}"
    echo "[dry-run]   OUTPUT_DIR=${out_dir}"
    echo "[dry-run]   NUM_GPUS=${NUM_GPUS}"
    echo "[dry-run]   EXTRA_ARGS=--max-steps ${MAX_STEPS} --save-steps ${SAVE_STEPS} --save-total-limit ${SAVE_TOTAL_LIMIT}${HP_ARGS} ${resume_args}"
    record "finetune ${variant}: DRY-RUN (not executed)"
    continue
  fi

  GROOT_ROOT="${GROOT_ROOT}" \
  GROOT_PYTHON="${GROOT_PYTHON}" \
  VARIANT="${variant}" \
  TASK="${TASK}" \
  DATASET="${DATA_ROOT}/univtac-${TASK}-${variant}" \
  OUTPUT_DIR="${out_dir}" \
  NUM_GPUS="${NUM_GPUS}" \
  HF_HOME="${HF_HOME}" \
  USE_WANDB="${USE_WANDB:-1}" \
  WANDB_PROJECT="${WANDB_PROJECT:-univtac-groot-ablation}" \
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-univtac-${TASK}-${SLURM_JOB_ID:-local}}" \
  WANDB_API_KEY="${WANDB_API_KEY:-}" \
  WANDB_MODE="${WANDB_MODE:-}" \
  EXTRA_ARGS="--max-steps ${MAX_STEPS} --save-steps ${SAVE_STEPS} --save-total-limit ${SAVE_TOTAL_LIMIT}${HP_ARGS} ${resume_args} ${EXTRA_ARGS:-}" \
    bash "${REPO_ROOT}/slurm/finetune.sbatch" >"${log}" 2>&1
  status=$?

  ckpt=$(latest_checkpoint "${out_dir}")
  if [[ ${status} -eq 0 && -n "${ckpt}" ]]; then
    # Stable path for evaluation and for anything downstream, so a rolling
    # save-total-limit window does not leave dangling references to a pruned
    # checkpoint-N. Relative target so the tree stays movable.
    if [[ "${ckpt}" != "${out_dir}" ]]; then
      ln -sfn "$(basename "${ckpt}")" "${out_dir}/final"
      echo "[stage] ${out_dir}/final -> $(basename "${ckpt}")"
    fi
    touch "${marker}"
    record "finetune ${variant}: OK -> ${ckpt}"
  else
    record "finetune ${variant}: FAILED (exit ${status}, checkpoint='${ckpt:-none}') see ${log}"
  fi
done

# --------------------------------------------------------------------------- #
# 3. Evaluate each variant that produced a checkpoint
# --------------------------------------------------------------------------- #
for variant in ${VARIANTS}; do
  if ! stage_enabled eval; then
    record "eval ${variant}: SKIPPED (not in STAGES='${STAGES}')"
    continue
  fi
  out_dir="${CKPT_ROOT}/${TASK}-${variant}"
  # Prefer the stable `final` symlink the finetune stage writes; fall back to
  # probing, so an externally trained output_dir still evaluates.
  if [[ -e "${out_dir}/final" ]]; then
    ckpt="${out_dir}/final"
  else
    ckpt=$(latest_checkpoint "${out_dir}")
  fi
  log="${STAGE_LOG_DIR}/eval-${variant}.log"
  summary="${RESULTS_DIR}/${variant}/${TASK}/seed0-0.summary.json"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would run slurm/eval_ablation.sbatch with:"
    echo "[dry-run]   VARIANT=${variant} TASK=${TASK} TASK_CONFIG=${TASK_CONFIG}"
    echo "[dry-run]   GROOT_MODEL=${ckpt:-<from the finetune stage>}"
    echo "[dry-run]   EPISODES=${EPISODES} EXECUTION_HORIZON=${EXECUTION_HORIZON}"
    echo "[dry-run]   PORT=$((PORT_BASE + PORT_OFFSET + 1))"
    record "eval ${variant}: DRY-RUN (not executed)"
    PORT_OFFSET=$((PORT_OFFSET + 1))
    continue
  fi

  if [[ -z "${ckpt}" ]]; then
    record "eval ${variant}: SKIPPED (no checkpoint to evaluate)"
    continue
  fi
  if [[ -f "${summary}" ]]; then
    record "eval ${variant}: SKIPPED (summary exists: ${summary})"
    continue
  fi

  PORT_OFFSET=$((PORT_OFFSET + 1))
  banner "evaluate ${variant}  ckpt=${ckpt}  port=$((PORT_BASE + PORT_OFFSET))"
  echo "[stage] log -> ${log}"
  PORT=$((PORT_BASE + PORT_OFFSET)) \
  UNIVTAC_ROOT="${UNIVTAC_ROOT}" \
  UNIVTAC_PYTHON="${UNIVTAC_PYTHON}" \
  GROOT_PYTHON="${GROOT_PYTHON}" \
  VARIANT="${variant}" \
  TASK="${TASK}" \
  TASK_CONFIG="${TASK_CONFIG}" \
  EPISODES="${EPISODES}" \
  EXECUTION_HORIZON="${EXECUTION_HORIZON}" \
  GROOT_MODEL="${ckpt}" \
  RESULTS_DIR="${RESULTS_DIR}" \
  HF_HOME="${HF_HOME}" \
    bash "${REPO_ROOT}/slurm/eval_ablation.sbatch" >"${log}" 2>&1
  status=$?

  if [[ -f "${summary}" ]]; then
    scored=$("${UNIVTAC_PYTHON}" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(f\"{d.get('episodes_scored',0)} scored, {d.get('successes',0)} success, \"
      f\"SR={d.get('success_rate_pct',0)}%\")
" "${summary}" 2>/dev/null || echo "unparseable")
    record "eval ${variant}: exit ${status} -- ${scored}"
  else
    record "eval ${variant}: FAILED (exit ${status}, no summary) see ${log}"
  fi
done

# --------------------------------------------------------------------------- #
# 4. Compare
# --------------------------------------------------------------------------- #
banner "compare"
# Even in a dry run, check the aggregator parses and its flags exist -- a typo
# here would otherwise only surface after both finetunes and both evals.
if ! stage_enabled compare; then
  record "compare: SKIPPED (not in STAGES='${STAGES}')"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
  if "${UNIVTAC_PYTHON}" "${REPO_ROOT}/scripts/compare_ablation.py" --help >/dev/null 2>&1; then
    record "compare: DRY-RUN (script imports and --help works)"
  else
    record "compare: DRY-RUN FAILED -- compare_ablation.py --help errored"
  fi
else
  "${UNIVTAC_PYTHON}" "${REPO_ROOT}/scripts/compare_ablation.py" \
    --results-dir "${RESULTS_DIR}" \
    --baseline-variant baseline_finetuned \
    --tactile-variant tactile \
    --json "${STAGE_LOG_DIR}/comparison.json" \
    2>&1 | tee "${STAGE_LOG_DIR}/compare.log"
  record "compare: exit ${PIPESTATUS[0]}"
fi

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
banner "summary"
for line in "${SUMMARY[@]}"; do echo "  ${line}"; done
echo
echo "  stage logs:   ${STAGE_LOG_DIR}"
echo "  checkpoints:  ${CKPT_ROOT}"
echo "  results:      ${RESULTS_DIR}"
echo
echo "  Resubmitting this script skips completed stages, so a walltime kill is"
echo "  safe to continue from. Delete a marker in ${STAGE_DIR} to force a redo."

# Exit non-zero if anything failed, so `sacct` reflects reality.
if printf '%s\n' "${SUMMARY[@]}" | grep -q "FAILED"; then
  exit 1
fi
exit 0
