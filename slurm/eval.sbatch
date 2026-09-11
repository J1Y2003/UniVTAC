#!/bin/bash
# Server (GROOT_MODEL, EMBODIMENT_TAG, PORT) + evaluator ("$@"). See docs/USAGE.md.
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
set -euo pipefail
env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH -u CONDA_PREFIX PYTHONPATH="$REPO_ROOT" "$GROOT_PYTHON" -u -m univtac_groot.server.run_server --model-path "$GROOT_MODEL" --embodiment-tag "$EMBODIMENT_TAG" --host 127.0.0.1 --port "$PORT" --modality-config-path "$REPO_ROOT/configs/modality/univtac_baseline_config.py" &
trap 'kill $! 2>/dev/null' EXIT
PYTHONPATH="$REPO_ROOT:$UNIVTAC_ROOT" "$UNIVTAC_PYTHON" "$REPO_ROOT/scripts/run_eval.py" --host 127.0.0.1 --port "$PORT" "$@"
