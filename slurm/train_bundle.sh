#!/bin/bash
# Same as train.sbatch, submitted through bundle-sbatch.
#   bash slurm/train_bundle.sh <checkpoint|from_scratch> -- <launch_finetune.py args>
set -euo pipefail
ckpt="$1"; shift; [[ "${1:-}" == "--" ]] && shift
exec bundle-sbatch --job-kind train --checkpoint "$ckpt" --code-git-root "$REPO_ROOT" -- \
  --job-name="$JOB_NAME" --wckey=project-short-name:sub_4dpdata \
  --partition="$PARTITION" --gres=gpu:1 -- slurm/train.sbatch "$@"
