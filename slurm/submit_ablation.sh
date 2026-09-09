#!/bin/bash
# EVALUATION-ONLY sweep: one eval job per (variant, task) against checkpoints
# that ALREADY EXIST. It trains nothing.
#
# This is not the script that runs the benchmark. For that use
# slurm/submit_benchmark.sh, which finetunes and then evaluates, one job per
# task. Come here to re-evaluate a finished checkpoint -- a different
# EXECUTION_HORIZON, more episodes, the zero-shot `baseline` -- without paying
# for training again.
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

# Not `set -e`: one task failing to submit must not abandon the rest, and a
# skipped variant is an ordinary outcome here.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=slurm/bundle_submit.sh
. "${REPO_ROOT}/slurm/bundle_submit.sh"
failed=0

# The eight benchmark tasks under UniVTAC/envs/ ('collect' is data-gen only).
# The three reported tasks, matching submit_benchmark.sh. Pass TASKS="..." for
# others; all eight are lift_bottle lift_can insert_HDMI insert_hole
# insert_tube pull_out_key put_bottle_in_shelf grasp_classify.
TASKS="${TASKS:-insert_hole insert_tube pull_out_key}"
# Vision-only, matching what is actually being trained. `baseline` is the
# ZERO-SHOT model and is not a fair comparator to a finetuned one; `tactile`
# needs a tactile finetune that does not currently exist.
VARIANTS="${VARIANTS:-baseline_finetuned}"
EPISODES="${EPISODES:-100}"   # paper: "evaluated over 100 test rollouts"
# `clean` everywhere: it is what download_data.sbatch creates the
# data/<task>/<config> symlink for, and what convert.sbatch reads.
TASK_CONFIG="${TASK_CONFIG:-clean}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
# Evaluation is not allowed on sjw_alinlab; `background` is where it belongs.
# Passed on the command line, not left to the #SBATCH header in
# eval_ablation.sbatch, because a stale SBATCH_PARTITION in the submitting shell
# outranks the header and would silently put these back on the training queue.
EVAL_PARTITION="${EVAL_PARTITION:-background}"
# No --time by default: run_eval.py has no resume, so a walltime kill restarts
# from the first seed instead of continuing. A limit here would schedule sooner
# via backfill, so set it once you have measured a real eval -- generously.
#   EVAL_TIME_LIMIT=06:00:00 bash slurm/submit_ablation.sh
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-}"

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
    # bundle-sbatch --checkpoint must be an EXISTING PHYSICAL DIRECTORY: the
    # launcher rejects symlinks, so `<out>/final` has to be resolved. The
    # zero-shot `baseline` names a hub model rather than a local checkpoint and
    # therefore cannot be submitted as --job-kind eval at all.
    if [[ "${variant}" == "baseline" ]]; then
      echo "skip ${task}/${variant}: the zero-shot baseline is a hub id" >&2
      echo "     (${model}), and --job-kind eval requires a local checkpoint" >&2
      continue
    fi
    checkpoint=$(readlink -f "${model}" 2>/dev/null || printf '%s' "${model}")
    if [[ ! -d "${checkpoint}" ]]; then
      echo "skip ${task}/${variant}: ${model} is not an existing directory" >&2
      failed=1
      continue
    fi

    BUNDLE_ENV=(
      "UNIVTAC_JOB_CONFIG=1"
      "VARIANT=${variant}"
      "TASK=${task}"
      "TASK_CONFIG=${TASK_CONFIG}"
      "EPISODES=${EPISODES}"
      "EXECUTION_HORIZON=${EXECUTION_HORIZON}"
      "GROOT_MODEL=${checkpoint}"
      "UNIVTAC_ROOT=${UNIVTAC_ROOT}"
      "GROOT_PYTHON=${GROOT_PYTHON}"
      "UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"
      "REPO_ROOT=${REPO_ROOT}"
    )
    [[ -n "${TACTILE_MODE:-}" ]] && BUNDLE_ENV+=("TACTILE_MODE=${TACTILE_MODE}")

    # The submit filter rejects job names of 50 characters or fewer, so this
    # cannot be the short "uv-<variant>-<task>" it used to be. No --cpus-per-task
    # and no --mem; --wckey always. See CLAUDE.md, "Cluster rules".
    BUNDLE_SLURM_ARGS=(
      --job-name="univtac-groot-evaluate-one-variant-on-one-task-${variant}-${task}"
      --wckey="${WCKEY:-project-short-name:sub_4dpdata}"
      --partition="${EVAL_PARTITION}"
      --gres="gpu:1"
    )
    [[ -n "${EVAL_TIME_LIMIT}" ]] && BUNDLE_SLURM_ARGS+=(--time="${EVAL_TIME_LIMIT}")

    BUNDLE_JOB_KIND=eval
    BUNDLE_CHECKPOINT="${checkpoint}"
    BUNDLE_GIT_ROOT="${REPO_ROOT}"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      bundle_print "${REPO_ROOT}/slurm/eval_ablation.sbatch" || failed=1
    else
      # Never retried: see slurm/bundle_submit.sh.
      bundle_submit "${REPO_ROOT}/slurm/eval_ablation.sbatch" || failed=1
    fi
  done
done

echo
echo "When the jobs finish, aggregate with:"
echo "  python ${REPO_ROOT}/scripts/compare_ablation.py --results-dir ${REPO_ROOT}/eval_result --json ablation.json"
exit ${failed}
