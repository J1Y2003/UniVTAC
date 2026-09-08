#!/bin/bash
# Finetune GR00T N1.7 on UniVTAC demonstrations for one ablation variant.
#
# Required for the tactile variant (extra state dims => NEW_EMBODIMENT, which ships
# in no released checkpoint) and recommended for the baseline variant too, so the
# comparison is like-for-like. Convert the data first:
#
#   python scripts/convert_univtac_to_lerobot.py --task insert_hole \
#       --raw-dir "${UNIVTAC_ROOT}/data/insert_hole/demo" \
#       --out "${DATA_ROOT}/univtac-insert_hole-tactile" --variant tactile
#
#   sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,TASK=insert_hole,VARIANT=tactile,DATASET=/data/univtac-insert_hole-tactile slurm/finetune.sbatch
#
# SITE RULES (see CLAUDE.md, "Cluster rules"). Enforced here; do not "helpfully"
# reintroduce any of them:
#
#   * NEVER pass --time. A job gets the maximum the partition allows, or runs
#     until the script exits, whichever comes first, so a limit can only cut the
#     run short. Over-requesting also leaves the job pending forever with
#     REASON=PartitionTimeLimit rather than failing fast.
#   * NEVER pass --cpus-per-task or --mem. Jobs take the node's per-GPU
#     defaults, and the submit filter rejects the job if either is specified.
#   * ALWAYS pass --wckey=project-short-name:sub_4dpdata.
#   * MODEL_OUTPUT_DIR must be in the SUBMITTING environment, for every job --
#     including CPU-only ones that write no checkpoints. The filter checks it at
#     submit time, before this script runs, so no in-script default can help:
#       ❌ ERROR: MODEL_OUTPUT_DIR가 없습니다.
#          MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jimin/... 로 지정하세요.
#     `source env.sh` sets it and --export=ALL carries it in.
#
# Do pass --partition explicitly, or export SBATCH_PARTITION: `debug` is the
# default here and is short-lived, so it is not where long training belongs.
#SBATCH --job-name=univtac-groot-finetune-one-ablation-variant-under-new-embodiment-tag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --wckey=project-short-name:sub_4dpdata

set -euo pipefail

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


# ARM= was the old name for this. Honour it rather than ignoring it: an
# unrecognised environment variable fails *silently*, so a stale
# `ARM=baseline_finetuned` command would quietly run the tactile variant
# instead -- a wrong result that looks like a right one. (A stale --arm flag
# fails loudly on the CLI, so no alias is needed there.)
if [[ -n "${ARM:-}" && -z "${VARIANT:-}" ]]; then
  echo "[job] note: ARM is deprecated, use VARIANT (taking ARM=${ARM})" >&2
  VARIANT="${ARM}"
fi
VARIANT="${VARIANT:-tactile}"                 # tactile | baseline_finetuned
DATASET="${DATASET:?set DATASET to the converted LeRobot dataset root}"
# TASK is load-bearing, not just a wandb label: the modality config resolves the
# camera set from it (two views for insert_tube and lift_bottle, third-person
# only otherwise, matching the paper's ACT setup). Without it the config raises
# at import. Set UNIVTAC_VIDEO_KEYS to override explicitly.
TASK="${TASK:?set TASK -- the modality config picks the camera set from it}"
export TASK
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
# Training-outputs policy: checkpoints belong under the cluster's unified
# folder, {NFS}/{user}/... (Kakao: /rlwrld-unified-checkpoints). Default to
# MODEL_OUTPUT_DIR rather than /tmp or scratch, and require one of them --
# see the same check in benchmark_task.sbatch.
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_OUTPUT_DIR:-}}"
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "error: set OUTPUT_DIR, or MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/\$USER/checkpoints/<job>" >&2
  echo "       (training-outputs policy: outputs must live in the unified folder)" >&2
  exit 2
fi
case "${OUTPUT_DIR}" in
  /rlwrld-unified-checkpoints/*) ;;
  *)
    if [[ "${ALLOW_NONSTANDARD_OUTPUT:-0}" == "1" ]]; then
      echo "[job] WARNING: OUTPUT_DIR=${OUTPUT_DIR} is outside the unified folder." >&2
    else
      echo "error: OUTPUT_DIR=${OUTPUT_DIR} is outside /rlwrld-unified-checkpoints." >&2
      echo "       Pass ALLOW_NONSTANDARD_OUTPUT=1 to override deliberately." >&2
      exit 2
    fi
    ;;
esac
NUM_GPUS="${NUM_GPUS:-1}"

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
GROOT_ROOT="${GROOT_ROOT:?set GROOT_ROOT to the Isaac-GR00T checkout}"
GROOT_PYTHON="${GROOT_PYTHON:?set GROOT_PYTHON to the GR00T interpreter, e.g. Isaac-GR00T/.venv/bin/python}"

if [[ "${VARIANT}" == "tactile" ]]; then
  MODALITY_CONFIG="${REPO_ROOT}/configs/modality/univtac_tactile_config.py"
else
  MODALITY_CONFIG="${REPO_ROOT}/configs/modality/univtac_baseline_config.py"
fi

mkdir -p logs "${OUTPUT_DIR}"

export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp}/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# --------------------------------------------------------------------------- #
# cuDNN, checked before anything expensive happens
# --------------------------------------------------------------------------- #
# There is deliberately NO option here to run without cuDNN. A mismatched cuDNN
# reports CUDNN_STATUS_NOT_INITIALIZED, which reads like a too-old driver, and
# the obvious workaround -- stop using cuDNN -- costs ~86x on GR00T's vision
# tower with no error message at all: a step goes from 1.89 s to 170 s and the
# job merely looks slow. So the only supported response is to fix the pin, and
# this guard states the fix.
#
# It fails ONLY on a definite mismatch (wrong version loaded, cannot dlopen, or
# a real convolution fails). Anything it cannot determine is a warning, so it
# never blocks a job it cannot actually indict. SKIP_CUDNN_CHECK=1 skips the
# check itself, for the case where the check is what is broken -- it does not
# change how the job runs.
if [[ "${SKIP_CUDNN_CHECK:-0}" != "1" ]]; then
  if ! "${GROOT_PYTHON}" "${REPO_ROOT}/scripts/check_cudnn.py"; then
    echo "error: aborting before spending GPU time. Fix cuDNN as printed above," >&2
    echo "       or set SKIP_CUDNN_CHECK=1 if this check is itself broken." >&2
    exit 1
  fi
else
  echo "[job] cuDNN check skipped (SKIP_CUDNN_CHECK=1)"
fi

# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #
# Worth enabling: launch_finetune.py exposes no eval metric, so the training
# loss curve is the only evidence for whether --max-steps under- or over-trains,
# and a multi-hour unattended run with no record is hard to reason about after
# the fact.
#
# Credentials, on a shared account: NEVER run `wandb login` -- it writes a token
# into ~/.netrc for the account owner. Export WANDB_API_KEY instead, exactly as
# with HF_TOKEN.
#
# The failure mode to avoid is a hang: with tracking on and no credentials,
# wandb can block waiting on stdin, and in a batch job stdin never answers, so
# the job burns its whole walltime having trained nothing. So resolve a mode
# here rather than leaving it to chance -- offline when there is no key (the run
# is recorded locally and `wandb sync` can upload it later), disabled only when
# explicitly asked.
WANDB_ARGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
  mkdir -p "${WANDB_DIR}"
  # Group both ablation variants under one name so they overlay on the same plots.
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-univtac-${TASK:-unknown}}"
  export WANDB_NAME="${WANDB_NAME:-${TASK:-unknown}-${VARIANT}-${SLURM_JOB_ID:-local}}"
  export WANDB_TAGS="${WANDB_TAGS:-univtac,${VARIANT},gr00t-n1d7}"

  if [[ -z "${WANDB_MODE:-}" ]]; then
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      export WANDB_MODE=online
    else
      export WANDB_MODE=offline
      echo "[job] no WANDB_API_KEY -- logging OFFLINE to ${WANDB_DIR}."
      echo "[job] Upload later with: wandb sync ${WANDB_DIR}/offline-run-*"
      echo "[job] For live logging: export WANDB_API_KEY=... (do NOT run 'wandb login')"
    fi
  fi
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-univtac-groot-ablation}")
  echo "[job] wandb: mode=${WANDB_MODE} project=${WANDB_PROJECT:-univtac-groot-ablation}"
  echo "[job]        group=${WANDB_RUN_GROUP} name=${WANDB_NAME}"
else
  export WANDB_MODE=disabled
  echo "[job] wandb disabled (USE_WANDB=0); the stdout log is the only record"
fi

echo "=========================================================="
echo " variant=${VARIANT}  dataset=${DATASET}"
echo " base=${BASE_MODEL}  modality=${MODALITY_CONFIG}"
echo " out=${OUTPUT_DIR}  gpus=${NUM_GPUS}"
echo "=========================================================="

# launch_finetune.py registers the modality config for NEW_EMBODIMENT by
# importing the --modality-config-path module (examples/SO100 convention).
cd "${GROOT_ROOT}"
"${GROOT_PYTHON}" gr00t/experiment/launch_finetune.py \
  --base-model-path "${BASE_MODEL}" \
  --dataset-path "${DATASET}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${MODALITY_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-gpus "${NUM_GPUS}" \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
  ${EXTRA_ARGS:-}

echo "[job] finetune finished; checkpoints under ${OUTPUT_DIR}"
echo "[job] evaluate with:"
echo "  sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,VARIANT=${VARIANT},TASK=<task>,GROOT_MODEL=${OUTPUT_DIR}/<checkpoint> ${REPO_ROOT}/slurm/eval_ablation.sbatch"
