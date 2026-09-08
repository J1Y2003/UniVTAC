#!/bin/bash
# Submit the full ablation sweep: both variants x all UniVTAC benchmark tasks.
#
# Prints the sbatch commands and submits them. Everything is non-interactive;
# set DRY_RUN=1 to print without submitting.
#
#   UNIVTAC_ROOT=~/UniVTAC \
#   GROOT_PYTHON=~/Isaac-GR00T/.venv/bin/python \
#   UNIVTAC_PYTHON=~/miniconda3/envs/UniVTAC/bin/python \
#   TACTILE_MODEL=/ckpt/univtac-tactile/checkpoint-20000 \
#   BASELINE_MODEL=nvidia/GR00T-N1.7-3B \
#   bash slurm/submit_ablation.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The eight benchmark tasks under UniVTAC/envs/ ('collect' is data-gen only).
TASKS="${TASKS:-lift_bottle lift_can insert_HDMI insert_hole insert_tube pull_out_key put_bottle_in_shelf grasp_classify}"
VARIANTS="${VARIANTS:-baseline tactile}"
EPISODES="${EPISODES:-50}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"

: "${UNIVTAC_ROOT:?set UNIVTAC_ROOT}"
: "${GROOT_PYTHON:?set GROOT_PYTHON}"
: "${UNIVTAC_PYTHON:?set UNIVTAC_PYTHON}"

for variant in ${VARIANTS}; do
  case "${variant}" in
    baseline)           model="${BASELINE_MODEL:-nvidia/GR00T-N1.7-3B}" ;;
    baseline_finetuned) model="${BASELINE_FT_MODEL:?set BASELINE_FT_MODEL}" ;;
    tactile)            model="${TACTILE_MODEL:?set TACTILE_MODEL}" ;;
    *) echo "unknown variant ${variant}" >&2; exit 1 ;;
  esac

  for task in ${TASKS}; do
    exports="ALL,VARIANT=${variant},TASK=${task},TASK_CONFIG=${TASK_CONFIG}"
    exports+=",EPISODES=${EPISODES},EXECUTION_HORIZON=${EXECUTION_HORIZON}"
    exports+=",GROOT_MODEL=${model},UNIVTAC_ROOT=${UNIVTAC_ROOT}"
    exports+=",GROOT_PYTHON=${GROOT_PYTHON},UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"
    exports+=",REPO_ROOT=${REPO_ROOT}"
    [[ -n "${TACTILE_MODE:-}" ]] && exports+=",TACTILE_MODE=${TACTILE_MODE}"

    # The submit filter rejects job names of 50 characters or fewer, so this
    # cannot be the short "uv-<variant>-<task>" it used to be. No --time, no
    # --cpus-per-task, no --mem; --wckey always. See CLAUDE.md, "Cluster rules".
    jobname="univtac-groot-evaluate-one-variant-on-one-task-${variant}-${task}"
    cmd=(sbatch --job-name="${jobname}" --wckey="${WCKEY:-sub_4dpdata}"
         --export="${exports}" "${REPO_ROOT}/slurm/eval_ablation.sbatch")
    echo "${cmd[*]}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      "${cmd[@]}"
    fi
  done
done

echo
echo "When the jobs finish, aggregate with:"
echo "  python ${REPO_ROOT}/scripts/compare_ablation.py --results-dir ${REPO_ROOT}/eval_result --json ablation.json"
