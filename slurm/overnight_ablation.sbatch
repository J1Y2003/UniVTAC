#!/bin/bash
# Full ablation for one task, unattended: finetune both variants, evaluate both,
# compare. Designed to be submitted and left alone overnight.
#
#   sbatch --wckey=sub_4dpdata slurm/overnight_ablation.sbatch
#   sbatch --wckey=sub_4dpdata --export=ALL,TASK=lift_bottle,EPISODES=50 slurm/overnight_ablation.sbatch
#
# Prefer slurm/submit_overnight.sh, which resolves every path for you. Do not
# raise --gres above gpu:1 -- see the note on the #SBATCH --gres line below.
#
# BEFORE submitting, verify it without spending GPU time. DRY_RUN=1 runs the
# full preflight, prints the exact arguments every stage would use, checks the
# aggregator imports, and exits -- cheap enough for a login node:
#
#   bash -n slurm/overnight_ablation.sbatch --wckey=sub_4dpdata          # syntax only, instant
#   DRY_RUN=1 bash slurm/overnight_ablation.sbatch --wckey=sub_4dpdata   # paths, datasets, plan
#   sbatch --wckey=sub_4dpdata --test-only slurm/overnight_ablation.sbatch  # submit filter + resources
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
#   * NEVER pass --time. A job gets the maximum the partition allows, or runs
#     until the script exits, whichever comes first, so a limit can only cut the
#     run short. Over-requesting also leaves the job pending forever with
#     REASON=PartitionTimeLimit rather than failing fast.
#   * NEVER pass --cpus-per-task or --mem. Jobs take the node's per-GPU
#     defaults, and the submit filter rejects the job if either is specified.
#   * ALWAYS pass --wckey=sub_4dpdata.
#
# Do pass --partition explicitly, or export SBATCH_PARTITION: `debug` is the
# default here and is short-lived, so it is not where long training belongs.
#SBATCH --job-name=univtac-groot-full-ablation-finetune-both-variants-then-evaluate-and-compare
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# 1 GPU, deliberately: --num-gpus > 1 makes launch_finetune.py wrap the model
# in nn.DataParallel, which dies with "module must have its parameters ... on
# device: cpu". Overridden by slurm/submit_overnight.sh's --gres.
#SBATCH --gres=gpu:1
#SBATCH --wckey=sub_4dpdata

# Deliberately NOT `set -e`: a failing stage must be recorded and stepped over,
# not abort the job. Stage failures are tracked explicitly instead.
set -uo pipefail

# --------------------------------------------------------------------------- #
# The site wckey, checked at runtime
# --------------------------------------------------------------------------- #
# sbatch precedence is: command line > environment variable > #SBATCH directive.
# So a stale SBATCH_WCKEY in the submitting shell (typically a copy of
# env.example.sh from before the value was corrected) SILENTLY OVERRIDES the
# `#SBATCH --wckey=sub_4dpdata` header above. That is exactly how a job once went
# out under the wrong project. The header alone is not enough; check it.
#
# SLURM_WCKEY is only set inside the job when a wckey was applied, so an unset
# value defaults to the right answer rather than failing a valid job.
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_WCKEY:-sub_4dpdata}" != "sub_4dpdata" ]]; then
  echo "error: this job's wckey is '${SLURM_WCKEY}', not 'sub_4dpdata'." >&2
  echo "       Every job here must be accounted to 'sub_4dpdata'." >&2
  echo "       Something in your environment overrode the #SBATCH header --" >&2
  echo "       almost certainly a stale SBATCH_WCKEY in env.sh. Fix it with:" >&2
  echo "         export SBATCH_WCKEY=sub_4dpdata SLURM_WCKEY=sub_4dpdata" >&2
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

# GR00T's own defaults: --max-steps 10000, --save-steps 1000. Hyperparameters
# are GR00T's business -- the benchmark fixes the data, the observation space
# and the evaluation protocol, not the recipe. See docs/ABLATION.md.
#
# One thing to watch rather than pre-empt: 10000 steps at batch 64 over the
# full 100 released episodes (~28.9k samples) is ~22 epochs. Read the loss
# curve; if it overfits, pick a step count on a task you are NOT reporting
# (lift_bottle) and freeze it before evaluating.
#
# SAVE_TOTAL_LIMIT is 3 rather than GR00T's 5 for disk, not for science: a 3B
# checkpoint carries optimizer + scheduler + RNG state and lands around 40 GB,
# so 5 per variant is ~200 GB on a shared mount already at 91%.
#
# There is no metric-based "best checkpoint" to select: launch_finetune.py
# exposes no eval metric or load-best-at-end option, so the last checkpoint is
# the final model by definition. This script points a stable
# `<output_dir>/final` symlink at the highest-numbered checkpoint after
# training, which is what evaluation uses.
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# Left unset, so launch_finetune.py's own defaults apply. Set them only for a
# deliberate sweep; the pinned recipe below records whatever was used so a
# resubmission cannot silently train the second variant differently.
LEARNING_RATE="${LEARNING_RATE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
HP_ARGS=""
[[ -n "${LEARNING_RATE}" ]] && HP_ARGS+=" --learning-rate ${LEARNING_RATE}"
[[ -n "${WEIGHT_DECAY}" ]] && HP_ARGS+=" --weight-decay ${WEIGHT_DECAY}"

DATA_ROOT="${DATA_ROOT:-${HOME}/jaewon/workspace/groot-data}"
# Training outputs policy ("모델 학습 Outputs 통합 저장 및 Retention 정책"):
# every checkpoint must land under the cluster's unified folder,
# {NFS}/{user}/..., which on Kakao is /rlwrld-unified-checkpoints. Required, not
# defaulted: the policy's own train.sbatch asserts MODEL_OUTPUT_DIR, jobs
# without it are to be rejected outright in future, and a silent fallback to
# home NFS is precisely the sprawl the policy exists to stop.
: "${MODEL_OUTPUT_DIR:?set MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/\$USER/checkpoints/<job> (training-outputs policy)}"
CKPT_ROOT="${CKPT_ROOT:-${MODEL_OUTPUT_DIR}}"
# CKPT_ROOT can still be redirected (the smoke test does), but not off the
# unified folder -- that would put tens of GB back on home NFS.
case "${CKPT_ROOT}" in
  /rlwrld-unified-checkpoints/*) ;;
  *)
    if [[ "${ALLOW_NONSTANDARD_OUTPUT:-0}" == "1" ]]; then
      echo "[job] WARNING: CKPT_ROOT=${CKPT_ROOT} is outside" >&2
      echo "[job] /rlwrld-unified-checkpoints -- against the training-outputs" >&2
      echo "[job] policy. Proceeding because ALLOW_NONSTANDARD_OUTPUT=1." >&2
    else
      echo "error: CKPT_ROOT=${CKPT_ROOT} is outside /rlwrld-unified-checkpoints." >&2
      echo "       The training-outputs policy requires checkpoints under" >&2
      echo "       {NFS}/{user}/... on the unified folder. Set MODEL_OUTPUT_DIR" >&2
      echo "       (and CKPT_ROOT, if you override it) accordingly, or pass" >&2
      echo "       ALLOW_NONSTANDARD_OUTPUT=1 if you really mean it." >&2
      exit 2
    fi
    ;;
esac
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/eval_result}"
STAGE_DIR="${STAGE_DIR:-${CKPT_ROOT}/.stages}"

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
# venv that did not match torch's pin (see docs/SETUP.md, "cuDNN"). Turning
# cuDNN off is NOT cheap: it costs ~86x on the Qwen3-VL vision tower, because
# the patch-embed Conv3d loses the fast kernel and falls back to a per-patch
# ATen loop -- measured at 170 s/step versus a few seconds. Only set this to 1
# as a deliberate last resort, and expect the run to be unusably slow.
# `scripts/preflight.py --deep` now fails if the cuDNN pin has drifted again.
DISABLE_CUDNN="${DISABLE_CUDNN:-0}"

# SLURM re-indexes allocated GPUs to 0..N-1 inside the job, so counting
# CUDA_VISIBLE_DEVICES gives the usable width for data-parallel training.
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-0}")}"

mkdir -p logs "${STAGE_DIR}" "${CKPT_ROOT}" "${RESULTS_DIR}"

STAGE_LOG_DIR="${REPO_ROOT}/logs/overnight-${SLURM_JOB_ID:-$$}"
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
echo " gpus=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo " data=${DATA_ROOT}"
echo " ckpt=${CKPT_ROOT}"
echo " results=${RESULTS_DIR}"
echo " disable_cudnn=${DISABLE_CUDNN}"
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
       --export=ALL then carries it into the job. Request access at
       https://huggingface.co/nvidia/Cosmos-Reason2-2B if you have none.
       HF_TOKEN is currently ${HF_TOKEN:+set}${HF_TOKEN:-UNSET}; HF_HOME=${HF_HOME}."
fi

# Same lesson as hf_transfer: do not pass a flag whose package may be absent.
# --use-wandb with no wandb installed would fail after the weights had loaded.
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  "${GROOT_PYTHON}" -c "import wandb" 2>/dev/null \
    || fatal "USE_WANDB=1 but ${GROOT_PYTHON} cannot import wandb. Install it in
       that venv, or pass USE_WANDB=0 to rely on the stdout logs alone."
fi

# A missing or truncated dataset is the classic way an overnight run wastes a
# night, so verify shape now rather than discovering it at step 0 of training.
for variant in ${VARIANTS}; do
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
if [[ -f "${RECIPE_FILE}" ]]; then
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
echo " the space above must hold roughly $((N_VARIANTS * SAVE_TOTAL_LIMIT * 40)) GB"
echo " (${N_VARIANTS} variants x ${SAVE_TOTAL_LIMIT} retained checkpoints x ~40 GB each)"

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
  if [[ -n "${existing}" ]]; then
    resume_args="--resume-from-checkpoint"
    echo "[stage] resuming ${variant} from ${existing}"
  fi

  banner "finetune ${variant}  (${NUM_GPUS} gpu, max_steps=${MAX_STEPS})"
  echo "[stage] log -> ${log}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would run slurm/finetune.sbatch --wckey=sub_4dpdata with:"
    echo "[dry-run]   VARIANT=${variant}"
    echo "[dry-run]   DATASET=${DATA_ROOT}/univtac-${TASK}-${variant}"
    echo "[dry-run]   OUTPUT_DIR=${out_dir}"
    echo "[dry-run]   NUM_GPUS=${NUM_GPUS}  DISABLE_CUDNN=${DISABLE_CUDNN}"
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
  DISABLE_CUDNN="${DISABLE_CUDNN}" \
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
    echo "[dry-run] would run slurm/eval_ablation.sbatch --wckey=sub_4dpdata with:"
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
  DISABLE_CUDNN="${DISABLE_CUDNN}" \
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
if [[ "${DRY_RUN:-0}" == "1" ]]; then
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
