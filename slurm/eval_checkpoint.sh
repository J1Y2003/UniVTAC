#!/bin/bash
# Evaluate ONE checkpoint and file the result in the results library.
#
#   bash slurm/eval_checkpoint.sh \
#       --checkpoint /rlwrld-unified-checkpoints/jimin/jaewon/insert_hole-baseline_finetuned/checkpoint-10000 \
#       --task insert_hole --seed-offset 1
#
# One invocation, one checkpoint, one job, one JSON file at
# `<result-dir>/<task>-<variant>/<task>-ckpt<N>-seed<offset>.json`. Each file is
# self-describing: success rate with a Wilson 95% interval, error/skip/
# truncation counts, mean steps and inference timings, plus the checkpoint,
# seed block, git commit and Slurm job id that produced it.
#
# --seed-offset picks WHICH task instances you evaluate on: run_eval.py starts
# at seed `1_000_000 * (1 + offset)`. Offset 0 is the reported block, so use a
# non-zero one for anything you might select a checkpoint on, and keep it the
# same across checkpoints you intend to compare.
#
# Submitted as --job-kind eval on `background`. Site rules: CLAUDE.md,
# "Cluster rules".
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=slurm/common.sh
. "${REPO_ROOT}/slurm/common.sh"

WHOAMI="${USER:-$(id -un)}"

# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
CHECKPOINT=""
TASK=""
VARIANT=""
EPISODES=100
SEED_OFFSET=0
# Beside the checkouts, not inside one: a results library outlives any single
# clone, and bundle storage is retention-managed so it cannot live there.
RESULT_DIR="${RESULT_DIR:-${HOME}/jaewon/workspace/eval_results}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
PARTITION="${EVAL_PARTITION:-background}"
# Unset, i.e. the partition maximum, because the cost of 100 rollouts has never
# been measured. Set it once a finished eval log gives a real figure -- the job
# resumes from max(seed)+1, so a short limit costs a requeue, not the run.
TIME_LIMIT="${EVAL_TIME_LIMIT:-}"
WCKEY="${WCKEY:-project-short-name:sub_4dpdata}"
GPUS="${GPUS:-1}"
MODE="submit"

usage() {
  cat >&2 <<'USAGE'
usage: bash slurm/eval_checkpoint.sh --checkpoint DIR --task TASK [options]

  --checkpoint DIR    the checkpoint to evaluate (required). Must be a physical
                      directory -- bundle-sbatch rejects symlinks, so a `final`
                      link is resolved for you.
  --task TASK         UniVTAC task (required)
  --variant NAME      baseline_finetuned | tactile | baseline
                      (default: inferred from the checkpoint's parent, e.g.
                      .../insert_hole-baseline_finetuned/checkpoint-10000)
  --episodes N        scored episodes (default 100, the reported protocol)
  --seed-offset N     seed block (default 0 = the REPORTED block; use 1+ for
                      anything you might select a checkpoint on)
  --result-dir DIR    results library root (default ~/jaewon/workspace/eval_results)
  --task-config NAME  default clean
  --execution-horizon N   actions per chunk before re-planning (default 16)
  --partition NAME    default background (evaluation is not allowed elsewhere)
  --time HH:MM:SS     optional walltime; unset means the partition maximum
  --print             show the bundle-sbatch command, submit nothing
USAGE
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)        CHECKPOINT="${2:?}"; shift 2 ;;
    --task)              TASK="${2:?}"; shift 2 ;;
    --variant)           VARIANT="${2:?}"; shift 2 ;;
    --episodes)          EPISODES="${2:?}"; shift 2 ;;
    --seed-offset)       SEED_OFFSET="${2:?}"; shift 2 ;;
    --result-dir)        RESULT_DIR="${2:?}"; shift 2 ;;
    --task-config)       TASK_CONFIG="${2:?}"; shift 2 ;;
    --execution-horizon) EXECUTION_HORIZON="${2:?}"; shift 2 ;;
    --partition)         PARTITION="${2:?}"; shift 2 ;;
    --time)              TIME_LIMIT="${2:?}"; shift 2 ;;
    --print)             MODE="--print"; shift ;;
    -h|--help)           usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "${CHECKPOINT}" ]] || { echo "--checkpoint is required" >&2; usage; }
[[ -n "${TASK}" ]] || { echo "--task is required" >&2; usage; }
[[ "${EPISODES}" =~ ^[0-9]+$ ]] || { echo "--episodes must be an integer" >&2; exit 2; }
[[ "${SEED_OFFSET}" =~ ^[0-9]+$ ]] || { echo "--seed-offset must be an integer" >&2; exit 2; }

: "${UNIVTAC_ROOT:?set UNIVTAC_ROOT (source env.sh)}"
: "${UNIVTAC_PYTHON:?set UNIVTAC_PYTHON (source env.sh)}"
: "${GROOT_PYTHON:?set GROOT_PYTHON (source env.sh)}"
HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"

# --------------------------------------------------------------------------- #
# Resolve the checkpoint
# --------------------------------------------------------------------------- #
# bundle-sbatch rejects a symlink outright, and `final` is a symlink, so resolve
# before declaring. Done here rather than left to the caller because the failure
# would otherwise surface as a launcher rejection with no explanation.
RESOLVED="$(readlink -f "${CHECKPOINT}" 2>/dev/null || printf '%s' "${CHECKPOINT}")"
if [[ ! -d "${RESOLVED}" ]]; then
  echo "error: --checkpoint ${CHECKPOINT} is not an existing directory" >&2
  [[ "${RESOLVED}" != "${CHECKPOINT}" ]] && echo "       (resolved to ${RESOLVED})" >&2
  exit 2
fi

# A checkpoint outside the managed user root must carry Hugging Face download
# metadata under .cache/huggingface/download/, which a trainer-written
# checkpoint has none of -- so one copied to home storage cannot be evaluated
# from there. A warning rather than a failure, since the root is inferred and
# the launcher is the authority; override MANAGED_ROOT if the guess is wrong.
MANAGED_ROOT="${MANAGED_ROOT:-/rlwrld-unified-checkpoints/${WHOAMI}/jaewon}"
case "${RESOLVED}" in
  "${MANAGED_ROOT}"/*) ;;
  *)
    if [[ ! -d "${RESOLVED}/.cache/huggingface/download" ]]; then
      echo "WARNING: ${RESOLVED}" >&2
      echo "         is outside the managed user root (${MANAGED_ROOT}) and has" >&2
      echo "         no .cache/huggingface/download/, so the launcher will" >&2
      echo "         refuse it. Copy it under the managed root and evaluate" >&2
      echo "         that copy -- do not fabricate the metadata, it is evidence." >&2
      echo >&2
    fi
    ;;
esac

CKPT_BASE="$(basename "${RESOLVED}")"
if [[ "${CKPT_BASE}" =~ ^checkpoint-([0-9]+)$ ]]; then
  STEP="${BASH_REMATCH[1]}"
else
  # A finetune output root rather than a checkpoint-N, e.g. an externally
  # trained model. Name it after the directory so the entry is still unique.
  STEP="${CKPT_BASE}"
  echo "note: ${CKPT_BASE} is not a checkpoint-N directory; filing it as" >&2
  echo "      ckpt${STEP}. Point --checkpoint at a checkpoint-N for a step number." >&2
fi

# Infer the variant from the finetune output directory the checkpoint sits in --
# `<task>-<variant>` is how every stage in this repo names it.
if [[ -z "${VARIANT}" ]]; then
  PARENT="$(basename "$(dirname "${RESOLVED}")")"
  case "${PARENT}" in
    "${TASK}-"*) VARIANT="${PARENT#"${TASK}"-}" ;;
    *)
      echo "error: cannot infer --variant from ${PARENT}." >&2
      echo "       Expected the checkpoint's parent to be <task>-<variant>." >&2
      echo "       Pass --variant explicitly." >&2
      exit 2
      ;;
  esac
fi

JOB_DIR="${TASK}-${VARIANT}"
EVAL_JSON="${RESULT_DIR}/${JOB_DIR}/${TASK}-ckpt${STEP}-seed${SEED_OFFSET}.json"
# The per-episode JSONL lives beside its summary, so a result can always be
# recomputed and a killed run leaves usable partial data.
RESULTS_DIR="${RESULT_DIR}/${JOB_DIR}/episodes"
RUN_TAG="ckpt${STEP}"

echo "evaluate one checkpoint"
printf '  %-18s %s\n' \
  task "${TASK}/${TASK_CONFIG}" \
  variant "${VARIANT}" \
  checkpoint "${RESOLVED}" \
  step "${STEP}" \
  episodes "${EPISODES}" \
  "seed block" "${SEED_OFFSET} (seeds from $(( 1000000 * (1 + SEED_OFFSET) )))" \
  partition "${PARTITION}" \
  walltime "${TIME_LIMIT:-<partition maximum>}" \
  "result json" "${EVAL_JSON}"
echo

if [[ "${SEED_OFFSET}" == "0" ]]; then
  echo "NOTE: seed block 0 is the one the REPORTED numbers use. Fine for a" >&2
  echo "      headline result; use 1+ if you might pick a checkpoint from it." >&2
  echo >&2
fi

if [[ -f "${EVAL_JSON}" ]]; then
  echo "WARNING: ${EVAL_JSON} already exists and would be overwritten by this" >&2
  echo "         job. Move it aside first if you want to keep both." >&2
  echo >&2
fi

mkdir -p "${RESULT_DIR}/${JOB_DIR}" "${REPO_ROOT}/logs" || exit 2

# --------------------------------------------------------------------------- #
# Submit
# --------------------------------------------------------------------------- #
# The job name must be LONGER than 50 characters or the submit filter rejects
# it, which is why it carries the whole identity of the run.
BUNDLE_SLURM_ARGS=(
  --job-name="univtac-groot-evaluate-one-checkpoint-${TASK}-${VARIANT}-ckpt${STEP}-seed${SEED_OFFSET}"
  --wckey="${WCKEY}"
  --partition="${PARTITION}"
  --gres="gpu:${GPUS}"
)
[[ -n "${TIME_LIMIT}" ]] && BUNDLE_SLURM_ARGS+=(--time="${TIME_LIMIT}")

# MODEL_OUTPUT_DIR is absent deliberately: the launcher injects it.
BUNDLE_ENV=(
  "UNIVTAC_JOB_CONFIG=1"
  "REPO_ROOT=${REPO_ROOT}"
  "TASK=${TASK}"
  "TASK_CONFIG=${TASK_CONFIG}"
  "VARIANT=${VARIANT}"
  "GROOT_MODEL=${RESOLVED}"
  "EPISODES=${EPISODES}"
  "SEED_OFFSET=${SEED_OFFSET}"
  "EXECUTION_HORIZON=${EXECUTION_HORIZON}"
  "RESULTS_DIR=${RESULTS_DIR}"
  "RUN_TAG=${RUN_TAG}"
  "EVAL_JSON=${EVAL_JSON}"
  "UNIVTAC_ROOT=${UNIVTAC_ROOT}"
  "UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"
  "GROOT_PYTHON=${GROOT_PYTHON}"
  "HF_HOME=${HF_HOME}"
)
[[ -n "${TACTILE_MODE:-}" ]] && BUNDLE_ENV+=("TACTILE_MODE=${TACTILE_MODE}")

BUNDLE_JOB_KIND=eval
BUNDLE_CHECKPOINT="${RESOLVED}"
BUNDLE_GIT_ROOT="${REPO_ROOT}"

cd "${REPO_ROOT}"
if [[ "${MODE}" == "--print" ]]; then
  bundle_print "slurm/eval_ablation.sbatch"
  exit $?
fi

bundle_require || exit 2
# Never retried: once the launcher records a submission, a second invocation for
# the same request is forbidden even when the outcome is unclear. Read the
# diagnostic and the retained bundle instead. See slurm/common.sh.
bundle_submit "slurm/eval_ablation.sbatch" || exit $?

echo
echo "submitted${BUNDLE_JOBID:+ job ${BUNDLE_JOBID}}"
echo "  watch:  squeue -u ${WHOAMI} -o '%.10i %.20P %.70j %.9T %.10M %.20R'"
echo "  result: ${EVAL_JSON}"
echo
echo "The library so far:"
echo "  ls ${RESULT_DIR}/*/"
