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
#SBATCH --job-name=univtac-groot-finetune
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G

set -euo pipefail

ARM="${ARM:-tactile}"                 # tactile | baseline_finetuned
DATASET="${DATASET:?set DATASET to the converted LeRobot dataset root}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/univtac-groot-${ARM}}"
NUM_GPUS="${NUM_GPUS:-1}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

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
