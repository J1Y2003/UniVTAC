#!/bin/bash
# Finetune. All flags go to gr00t/experiment/launch_finetune.py -- see docs/USAGE.md.
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
set -euo pipefail
cd "$GROOT_ROOT"
exec "$GROOT_PYTHON" gr00t/experiment/launch_finetune.py "$@"
