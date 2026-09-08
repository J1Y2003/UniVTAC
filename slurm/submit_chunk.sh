#!/bin/bash
# Run the ablation as a series of SHORT jobs, one in the queue at a time.
#
#   bash slurm/submit_chunk.sh            # submit one 3h chunk (refuses if one is queued)
#   bash slurm/submit_chunk.sh --status   # progress only, submit nothing
#   bash slurm/submit_chunk.sh --auto 8   # keep one chunk queued until 8 have run
#   bash slurm/submit_chunk.sh --print    # show the sbatch command
#
# Companion to submit_overnight.sh, which asks for one long job on a 2-day
# partition. Use this one instead when those partitions are contended: a 3-hour
# `debug` slot that actually starts beats a 24-hour slot that pends all night.
#
# Why one at a time, rather than pre-submitting a dependency chain: this cluster
# caps GPUs per user across *queued* jobs, not just running ones. Submitting
# eight chained jobs puts eight GPU requests in the queue, trips that cap, and
# blocks the very chain it was meant to create. A held job is not a free job.
#
# This only works because every stage is resumable. Finetuning continues from
# the latest checkpoint via --resume-from-checkpoint, and finished stages are
# skipped by their markers, so N short chunks reach the same place as one long
# job -- minus a few minutes per chunk reloading the model.

set -uo pipefail

# ${USER} is not guaranteed to be exported; under `set -u` a bare use is fatal.
WHOAMI="${USER:-$(id -un)}"

# --------------------------------------------------------------------------- #
# Paths -- same defaults as submit_overnight.sh, same overrides
# --------------------------------------------------------------------------- #
WORKSPACE="${WORKSPACE:-${HOME}/jaewon/workspace}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UNIVTAC_ROOT="${UNIVTAC_ROOT:-${WORKSPACE}/UniVTAC-sim}"
GROOT_ROOT="${GROOT_ROOT:-${WORKSPACE}/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:-${HOME}/miniconda3/envs/UniVTAC/bin/python}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/groot-data}"
HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/${WHOAMI}/univtac-groot}"
# Must match the CKPT_ROOT default inside overnight_ablation.sbatch: --status
# reads its stage markers and checkpoints from here.
CKPT_ROOT="${CKPT_ROOT:-${MODEL_OUTPUT_DIR}}"

# --------------------------------------------------------------------------- #
# Job shape -- short by design
# --------------------------------------------------------------------------- #
# `debug` is the cluster default partition and caps at 3:00:00, which makes it
# the least contended and the quickest to start.
PARTITION="${PARTITION:-debug}"
# Just inside the cap, so the request is never rejected for exceeding it.
TIMELIMIT="${TIMELIMIT:-2:55:00}"
GPUS="${GPUS:-2}"
MAX_STEPS="${MAX_STEPS:-10000}"
# The parameter that decides whether chunking works at all. A chunk killed
# before its first save banks NOTHING, so the next chunk restarts from zero and
# no amount of chunks ever finishes. The step rate with cuDNN disabled is still
# unmeasured, so save often; raise this once --status shows the real rate.
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

TASK="${TASK:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
EPISODES="${EPISODES:-50}"

# How often --auto looks to see whether the previous chunk has ended. Long
# enough not to hammer the scheduler from a login node.
POLL_SECONDS="${POLL_SECONDS:-300}"

MODE="${1:-submit}"
AUTO_CHUNKS="${2:-8}"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Any of our jobs currently queued or running. squeue's %j truncates at its
# default width, hence the explicit 200.
my_jobs() {
  squeue -h -u "${WHOAMI}" --format="%i %T %200j" 2>/dev/null \
    | grep "univtac-groot" || true
}

latest_step() {
  local out_dir="$1" latest="" d n
  for d in "${out_dir}"/checkpoint-*; do
    [[ -d "${d}" ]] || continue
    n="${d##*checkpoint-}"
    [[ "${n}" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${latest}" || "${n}" -gt "${latest}" ]]; then latest="${n}"; fi
  done
  printf '%s' "${latest}"
}

report_progress() {
  echo "Progress (${TASK}):"
  local variant out_dir state step summary
  for variant in tactile baseline_finetuned; do
    out_dir="${CKPT_ROOT}/${TASK}-${variant}"
    if [[ -f "${CKPT_ROOT}/.stages/finetune-${TASK}-${variant}.done" ]]; then
      state="finetune COMPLETE"
    else
      step=$(latest_step "${out_dir}")
      if [[ -n "${step}" ]]; then
        state="training: step ${step}/${MAX_STEPS}"
      else
        state="not started (no checkpoint yet)"
      fi
    fi
    summary="${REPO_ROOT}/eval_result/${variant}/${TASK}/seed0-0.summary.json"
    [[ -f "${summary}" ]] && state+="; eval DONE"
    printf '  %-20s %s\n' "${variant}" "${state}"
  done
}

# --------------------------------------------------------------------------- #
# Preflight -- the failures that would otherwise cost a queued slot
# --------------------------------------------------------------------------- #
FAIL=0
for spec in "UNIVTAC_ROOT ${UNIVTAC_ROOT}" "GROOT_ROOT ${GROOT_ROOT}" "DATA_ROOT ${DATA_ROOT}"; do
  set -- ${spec}
  [[ -d "$2" ]] || { echo "MISSING dir $1=$2" >&2; FAIL=1; }
done
for spec in "GROOT_PYTHON ${GROOT_PYTHON}" "UNIVTAC_PYTHON ${UNIVTAC_PYTHON}"; do
  set -- ${spec}
  [[ -x "$2" ]] || { echo "NOT EXECUTABLE $1=$2" >&2; FAIL=1; }
done
for variant in tactile baseline_finetuned; do
  n=$(find "${DATA_ROOT}/univtac-${TASK}-${variant}/data" -name '*.parquet' 2>/dev/null | wc -l)
  [[ "${n}" -gt 0 ]] || {
    echo "MISSING dataset ${DATA_ROOT}/univtac-${TASK}-${variant}" >&2
    FAIL=1
  }
done
# SLURM opens the --output file before the script runs, so a missing logs/ kills
# the job instantly with no log to explain it.
mkdir -p "${REPO_ROOT}/logs" 2>/dev/null || FAIL=1
if [[ "${FAIL}" -ne 0 ]]; then
  echo "Fix the above and re-run. Nothing was submitted." >&2
  exit 2
fi

EXPORTS="ALL"
EXPORTS+=",MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR}"
EXPORTS+=",UNIVTAC_ROOT=${UNIVTAC_ROOT}"
EXPORTS+=",UNIVTAC_PYTHON=${UNIVTAC_PYTHON}"
EXPORTS+=",GROOT_ROOT=${GROOT_ROOT}"
EXPORTS+=",GROOT_PYTHON=${GROOT_PYTHON}"
EXPORTS+=",HF_HOME=${HF_HOME}"
EXPORTS+=",DATA_ROOT=${DATA_ROOT}"
EXPORTS+=",TASK=${TASK}"
EXPORTS+=",TASK_CONFIG=${TASK_CONFIG}"
EXPORTS+=",EPISODES=${EPISODES}"
EXPORTS+=",MAX_STEPS=${MAX_STEPS}"
EXPORTS+=",SAVE_STEPS=${SAVE_STEPS}"
EXPORTS+=",SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
# Pinned off: DRY_RUN left exported from testing would ride in on --export=ALL
# and the job would exit in seconds having trained nothing.
EXPORTS+=",DRY_RUN=0"

SBATCH_ARGS=(
  --partition="${PARTITION}"
  --gres="gpu:${GPUS}"
  --time="${TIMELIMIT}"
  --export="${EXPORTS}"
)

# Prints human-readable messages on stderr and ONLY the job id on stdout, so a
# caller can do `id=$(submit_one)` without hiding what happened from the user.
submit_one() {
  local existing
  existing=$(my_jobs)
  if [[ -n "${existing}" ]]; then
    echo "Already queued or running -- nothing submitted:" >&2
    echo "${existing}" | sed 's/^/  /' >&2
    return 1
  fi
  local out jobid
  out=$(cd "${REPO_ROOT}" && sbatch --parsable "${SBATCH_ARGS[@]}" \
        slurm/overnight_ablation.sbatch) || return 2
  jobid="${out%%;*}"
  echo "Submitted job ${jobid}  (${TIMELIMIT} on ${PARTITION}, ${GPUS} gpu)" >&2
  printf '%s' "${jobid}"
  return 0
}

# --------------------------------------------------------------------------- #
case "${MODE}" in
  --status)
    report_progress
    echo
    existing=$(my_jobs)
    if [[ -n "${existing}" ]]; then
      echo "In the queue:"
      echo "${existing}" | sed 's/^/  /'
    else
      echo "Nothing queued. Submit the next chunk with:"
      echo "  bash slurm/submit_chunk.sh"
    fi
    ;;

  --print)
    echo "sbatch ${SBATCH_ARGS[*]} slurm/overnight_ablation.sbatch"
    ;;

  --auto)
    # Keeps exactly ONE chunk in the queue, resubmitting as each ends, until
    # AUTO_CHUNKS have been submitted or everything is finished. Run it under
    # tmux: it is a foreground loop and dies with your shell. It only ever calls
    # squeue and sbatch, so it costs the login node nothing between polls.
    echo "Auto mode: up to ${AUTO_CHUNKS} chunks, one in the queue at a time,"
    echo "polling every ${POLL_SECONDS}s. Ctrl-C to stop (the running chunk survives)."
    echo
    done_count=0
    while [[ "${done_count}" -lt "${AUTO_CHUNKS}" ]]; do
      if [[ -z "$(my_jobs)" ]]; then
        # Stop early once both variants are finetuned and evaluated.
        if [[ -f "${CKPT_ROOT}/.stages/finetune-${TASK}-tactile.done" \
           && -f "${CKPT_ROOT}/.stages/finetune-${TASK}-baseline_finetuned.done" \
           && -f "${REPO_ROOT}/eval_result/tactile/${TASK}/seed0-0.summary.json" \
           && -f "${REPO_ROOT}/eval_result/baseline_finetuned/${TASK}/seed0-0.summary.json" ]]; then
          echo "[$(date '+%H:%M:%S')] everything finished; stopping."
          break
        fi
        if jobid=$(submit_one); then
          done_count=$((done_count + 1))
          echo "[$(date '+%H:%M:%S')] chunk ${done_count}/${AUTO_CHUNKS} submitted"
          report_progress | sed 's/^/    /'
          # Confirm the job is actually visible before trusting an empty queue
          # again. Without this, a transient squeue failure reads as "nothing
          # running" and the loop submits repeatedly -- exactly the queue
          # flooding this script exists to avoid.
          confirmed=0
          for _ in 1 2 3; do
            sleep 10
            if my_jobs | grep -q "^${jobid} "; then confirmed=1; break; fi
          done
          if [[ "${confirmed}" -ne 1 ]]; then
            echo "[$(date '+%H:%M:%S')] job ${jobid} never appeared in squeue;" >&2
            echo "stopping rather than risk submitting again. Check: squeue -u ${WHOAMI}" >&2
            break
          fi
        else
          echo "[$(date '+%H:%M:%S')] submission refused or failed; stopping." >&2
          break
        fi
      fi
      sleep "${POLL_SECONDS}"
    done
    echo
    report_progress
    ;;

  submit)
    report_progress
    echo
    jobid=$(submit_one) || exit 0
    echo
    echo "Watch:      squeue -u ${WHOAMI}"
    echo "Log:        tail -f logs/univtac-groot-full-*-${jobid}.out"
    echo "Progress:   bash slurm/submit_chunk.sh --status"
    echo "Next chunk: bash slurm/submit_chunk.sh          (once this one ends)"
    echo
    echo "The step number under --status must ADVANCE between chunks. If it does"
    echo "not, SAVE_STEPS=${SAVE_STEPS} is more steps than fit in ${TIMELIMIT},"
    echo "nothing is ever banked, and every chunk restarts from zero."
    echo "Fix with: SAVE_STEPS=100 bash slurm/submit_chunk.sh"
    ;;

  *)
    echo "usage: bash slurm/submit_chunk.sh [--status|--print|--auto [N]]" >&2
    exit 2
    ;;
esac
