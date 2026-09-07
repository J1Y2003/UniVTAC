#!/bin/bash
# Finetune GR00T N1.7 on UniVTAC demonstrations for one ablation arm.
#
# Required for the tactile arm (extra state dims => NEW_EMBODIMENT, which ships
# in no released checkpoint) and recommended for the baseline arm too, so the
# comparison is like-for-like. Convert the data first:
#
#   python scripts/convert_univtac_to_lerobot.py --task insert_hole \
#       --raw-dir "${UNIVTAC_ROOT}/data/insert_hole/demo" \
#       --out "${DATA_ROOT}/univtac-insert_hole-tactile" --arm tactile
#
#   sbatch --export=ALL,ARM=tactile,DATASET=/data/univtac-insert_hole-tactile slurm/finetune.sbatch
#
# No --time here on purpose. Requesting more than a partition's limit leaves the
# job pending forever with REASON=PartitionTimeLimit rather than failing fast, so
# the partition's own default is safer. Check the limits with:
#   sinfo -o "%20P %10l %10L %6D %25G"     (%l = TIMELIMIT, %L = DEFAULTTIME)
# and pass --partition explicitly, or export SBATCH_PARTITION. Note `debug` is
# short-lived and is where jobs land by default here -- it is not where long
# training belongs.
#SBATCH --job-name=univtac-groot-finetune-one-ablation-arm-under-new-embodiment-tag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

ARM="${ARM:-tactile}"                 # tactile | baseline_finetuned
DATASET="${DATASET:?set DATASET to the converted LeRobot dataset root}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/univtac-groot-${ARM}}"
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

if [[ "${ARM}" == "tactile" ]]; then
  MODALITY_CONFIG="${REPO_ROOT}/configs/modality/univtac_tactile_config.py"
else
  MODALITY_CONFIG="${REPO_ROOT}/configs/modality/univtac_baseline_config.py"
fi

mkdir -p logs "${OUTPUT_DIR}"

export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp}/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Training hits the same cuDNN wall as inference on a driver older than GR00T's
# CUDA build. launch_finetune.py spawns its own worker processes, so a flag on
# the parent is not enough -- a sitecustomize hook on PYTHONPATH reaches them
# all. Set DISABLE_CUDNN=1 to enable. See docs/SETUP.md.
if [[ "${DISABLE_CUDNN:-0}" == "1" ]]; then
  export UNIVTAC_DISABLE_CUDNN=1
  export UNIVTAC_DISABLE_CUDNN_VERBOSE=1
  export PYTHONPATH="${REPO_ROOT}/tools/nocudnn${PYTHONPATH:+:${PYTHONPATH}}"
  echo "[job] cuDNN disabled for every child process via sitecustomize"
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
  # Group both ablation arms under one name so they overlay on the same plots.
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-univtac-${TASK:-unknown}}"
  export WANDB_NAME="${WANDB_NAME:-${TASK:-unknown}-${ARM}-${SLURM_JOB_ID:-local}}"
  export WANDB_TAGS="${WANDB_TAGS:-univtac,${ARM},gr00t-n1d7}"

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
echo " arm=${ARM}  dataset=${DATASET}"
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
echo "  sbatch --export=ALL,ARM=${ARM},TASK=<task>,GROOT_MODEL=${OUTPUT_DIR}/<checkpoint> ${REPO_ROOT}/slurm/eval_ablation.sbatch"
