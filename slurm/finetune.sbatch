#!/bin/bash
# Finetune GR00T N1.7 on a converted UniVTAC dataset. Normally invoked by
# slurm/benchmark_task.sbatch rather than submitted directly.
#
#   env TASK=insert_hole VARIANT=baseline_finetuned \
#     DATASET="$DATA_ROOT/univtac-insert_hole-baseline_finetuned" \
#     UNIVTAC_JOB_CONFIG=1 \
#     bundle-sbatch --job-kind train --checkpoint from_scratch \
#     --code-git-root "$REPO_ROOT" -- \
#     --job-name=univtac-groot-finetune-one-ablation-variant-under-new-embodiment-tag \
#     --wckey=project-short-name:sub_4dpdata --partition=sjw_alinlab -- \
#     slurm/finetune.sbatch
#
# Site rules (job name >50 chars, --wckey, no --cpus-per-task or --mem) and
# the options the launcher owns: CLAUDE.md, "Cluster rules". MODEL_OUTPUT_DIR
# is injected and equals <bundle>/code-output; never set it.
#SBATCH --gres=gpu:1
set -euo pipefail



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
  echo "error: set OUTPUT_DIR, or MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jimin/jaewon/<job>" >&2
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

# sbatch copies this script into /var/spool/slurm/d/, so BASH_SOURCE resolves to
# the spool and every derived path lands somewhere unwritable. SLURM_SUBMIT_DIR
# is the right answer under sbatch, BASH_SOURCE under plain `bash`; accept
# whichever actually contains this repo.
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

# Both runtime guards -- the wckey and the configuration sentinel -- live in one
# place. shellcheck source=slurm/common.sh
. "${REPO_ROOT}/slurm/common.sh"
univtac_job_guard
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

# cuDNN, before anything expensive. A mismatched pin reads like a too-old driver
# (CUDNN_STATUS_NOT_INITIALIZED) and the obvious workaround -- disable cuDNN --
# costs ~86x on GR00T's vision tower with no error at all, so there is
# deliberately no way to run without it. See CLAUDE.md, "cuDNN". The check fails
# only on a definite mismatch and warns otherwise; SKIP_CUDNN_CHECK=1 skips the
# check, not the requirement.
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
echo "  bash ${REPO_ROOT}/slurm/eval_checkpoint.sh --task <task> \\"
echo "      --checkpoint ${OUTPUT_DIR}/checkpoint-<N> --seed-offset 1"
echo "  (it resolves the physical checkpoint path bundle-sbatch --checkpoint needs)"
