#!/bin/bash
# Same as eval.sbatch, submitted through bundle-sbatch.
#   bash slurm/eval_bundle.sh <physical checkpoint dir> -- <run_eval.py args>
set -euo pipefail
ckpt="$1"; shift; [[ "${1:-}" == "--" ]] && shift
exec bundle-sbatch --job-kind eval --checkpoint "$ckpt" --code-git-root "$REPO_ROOT" -- \
  --job-name="$JOB_NAME" --wckey=project-short-name:sub_4dpdata \
  --partition="$PARTITION" --gres=gpu:1 -- slurm/eval.sbatch "$@"
