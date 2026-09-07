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
  ${EXTRA_ARGS:-}

echo "[job] finetune finished; checkpoints under ${OUTPUT_DIR}"
echo "[job] evaluate with:"
echo "  sbatch --export=ALL,ARM=${ARM},TASK=<task>,GROOT_MODEL=${OUTPUT_DIR}/<checkpoint> ${REPO_ROOT}/slurm/eval_ablation.sbatch"
