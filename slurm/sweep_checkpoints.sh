#!/bin/bash
# Evaluate one task's checkpoint sweep -- 2k, 4k, 6k, 8k, 10k -- on ONE seed
# block, resumably. Run it once per task:
#
#   bash slurm/sweep_checkpoints.sh --task lift_bottle --seed-offset 1 \
#       --orig   /rlwrld-unified-checkpoints/$USER/checkpoints/univtac-groot/lift_bottle-baseline_finetuned \
#       --rescue $HOME/jaewon/workspace/groot-keep/lift_bottle-baseline_finetuned
#
# WHAT IT IS FOR. Training loss cannot tell you whether a run overfits, and
# there is no validation metric in this pipeline -- the only signal is success
# rate as a function of training step. This walks the retained checkpoints and
# measures exactly that.
#
# WHY ONE SEED BLOCK. --seed-offset picks WHICH 100 task instances you evaluate
# on: run_eval.py starts at seed 1_000_000 * (1 + offset) and counts up, which
# is UniVTAC's own convention. Offset 0 is the block the reported numbers use.
# Use a NON-ZERO offset here (1 gives seeds from 2_000_000) for two reasons:
#
#   * it keeps checkpoint selection off the reported rollouts, which is what
#     docs/ABLATION.md forbids tuning against;
#   * the five points stay comparable to each other. Different offsets mean
#     different object poses and different difficulty, so a curve mixing them
#     would show seed luck rather than training progress.
#
# It is applied uniformly to all five evaluations, by construction.
#
# RESUMING. `background` is preemptible, so this WILL be killed mid-sweep.
# Re-run the same command and it continues:
#
#   * a checkpoint whose eval finished is skipped (its .done marker exists);
#   * a checkpoint killed PART WAY through resumes at the next unseen seed.
#     ResultWriter appends to its JSONL and flushes after every episode, and
#     each line carries its seed, so the partial file is authoritative: the
#     next seed is max(seed)+1 and the remaining budget is the episodes still
#     unscored. Nothing is re-run and nothing is lost.
#
# Errored and skipped episodes are counted the way run_eval.py counts them --
# they consume a seed but not the episode budget -- so the resumed run targets
# the same number of SCORED episodes as a clean run would.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
TASK=""
ORIG=""
RESCUE=""
SEED_OFFSET=""
EPISODES=100
VARIANT="baseline_finetuned"
STEPS="2000 4000 6000 8000 10000"
RESULTS_DIR=""
DRY_RUN=0

usage() {
  cat >&2 <<'USAGE'
usage: bash slurm/sweep_checkpoints.sh --task TASK --seed-offset N \
           --orig DIR [--rescue DIR] [options]

  --task TASK          the UniVTAC task (required)
  --seed-offset N      seed block for EVERY eval here; use non-zero so the
                       reported block 0 stays clean (required)
  --orig DIR           checkpoint directory searched first (required)
  --rescue DIR         fallback when --orig no longer has a checkpoint, i.e.
                       the copies rescued before the rolling window deleted them
  --episodes N         scored episodes per checkpoint (default 100)
  --variant NAME       baseline_finetuned | tactile (default baseline_finetuned)
  --steps "A B C"      checkpoint numbers (default "2000 4000 6000 8000 10000")
  --results-dir DIR    default <repo>/eval_sweep/seed<offset>/<task>
  --dry-run            resolve, report the resume plan, run no evaluation
USAGE
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)         TASK="${2:?}"; shift 2 ;;
    --orig)         ORIG="${2:?}"; shift 2 ;;
    --rescue)       RESCUE="${2:?}"; shift 2 ;;
    --seed-offset)  SEED_OFFSET="${2:?}"; shift 2 ;;
    --episodes)     EPISODES="${2:?}"; shift 2 ;;
    --variant)      VARIANT="${2:?}"; shift 2 ;;
    --steps)        STEPS="${2:?}"; shift 2 ;;
    --results-dir)  RESULTS_DIR="${2:?}"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help)      usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "${TASK}" ]] || { echo "--task is required" >&2; usage; }
[[ -n "${ORIG}" ]] || { echo "--orig is required" >&2; usage; }
[[ -n "${SEED_OFFSET}" ]] || { echo "--seed-offset is required" >&2; usage; }
[[ "${SEED_OFFSET}" =~ ^[0-9]+$ ]] || { echo "--seed-offset must be an integer" >&2; exit 2; }
[[ "${EPISODES}" =~ ^[0-9]+$ ]] || { echo "--episodes must be an integer" >&2; exit 2; }

# Required by eval_ablation.sbatch; fail here rather than five evals deep.
: "${UNIVTAC_ROOT:?set UNIVTAC_ROOT (source env.sh)}"
: "${UNIVTAC_PYTHON:?set UNIVTAC_PYTHON (source env.sh)}"
: "${GROOT_PYTHON:?set GROOT_PYTHON (source env.sh)}"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/eval_sweep/seed${SEED_OFFSET}/${TASK}}"
BASE_SEED=$(( 1000000 * (1 + SEED_OFFSET) ))

if [[ "${SEED_OFFSET}" == "0" ]]; then
  echo "WARNING: --seed-offset 0 is the block the REPORTED numbers use." >&2
  echo "         Selecting a checkpoint on it is the contamination" >&2
  echo "         docs/ABLATION.md rules out. Use 1 unless you mean this." >&2
fi

echo "sweep: task=${TASK} variant=${VARIANT} episodes=${EPISODES}/checkpoint"
echo "  seed block   ${SEED_OFFSET}  (seeds from ${BASE_SEED})"
echo "  orig         ${ORIG}"
echo "  rescue       ${RESCUE:-<none>}"
echo "  results      ${RESULTS_DIR}"
echo "  steps        ${STEPS}"
echo

mkdir -p "${RESULTS_DIR}"

# --------------------------------------------------------------------------- #
# resume_plan <jsonl> -- print "NEXT_SEED SCORED" for a partial result file.
#
# Reads what is already recorded rather than assuming the run got that far:
# ResultWriter flushes every episode, so the file is truthful even after a
# SIGKILL. Errored/skipped episodes consumed a seed but not the budget, which
# is why the two numbers are counted separately.
# --------------------------------------------------------------------------- #
resume_plan() {
  local jsonl="$1" out
  [[ -s "${jsonl}" ]] || { printf '%s %s' "${BASE_SEED}" 0; return 0; }
  out=$("${UNIVTAC_PYTHON}" - "${jsonl}" "${BASE_SEED}" <<'PYRESUME'
import json, sys
path, base = sys.argv[1], int(sys.argv[2])
max_seed, scored = base - 1, 0
with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A kill mid-write can leave one truncated final line. Everything
            # before it still counts; this line does not.
            continue
        seed = row.get("seed")
        if isinstance(seed, int):
            max_seed = max(max_seed, seed)
        if not row.get("error") and not row.get("skipped"):
            scored += 1
print(f"{max_seed + 1} {scored}")
PYRESUME
  ) || { echo "resume_plan: reader failed on ${jsonl}" >&2; return 1; }
  # tr: an interpreter that writes CRLF would otherwise leave a trailing \r on
  # the second field and break the arithmetic that consumes it.
  out=$(printf '%s' "${out}" | tr -d '\r')
  if [[ ! "${out}" =~ ^[0-9]+[[:space:]]+[0-9]+$ ]]; then
    echo "resume_plan: unparseable resume state '${out}' from ${jsonl}" >&2
    return 1
  fi
  printf '%s' "${out}"
}

# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
declare -a SUMMARY=()
STATUS=0

# Each checkpoint's eval starts its own policy server and kills it on the way
# out. eval_ablation.sbatch derives PORT from the job id, which is CONSTANT
# across the five runs here, so they would all bind the same port -- and the
# previous server's socket sitting in TIME_WAIT makes that bind fail. Hand each
# one its own port, the same fix benchmark_task.sbatch applies between variants.
PORT_BASE="${PORT_BASE:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 15000) ))}"
PORT_OFFSET=0

for N in ${STEPS}; do
  ckpt="${ORIG}/checkpoint-${N}"
  where="orig"
  if [[ ! -d "${ckpt}" && -n "${RESCUE}" ]]; then
    ckpt="${RESCUE}/checkpoint-${N}"
    where="rescue"
  fi
  if [[ ! -d "${ckpt}" ]]; then
    SUMMARY+=("checkpoint-${N}: MISSING (not in orig${RESCUE:+ or rescue})")
    continue
  fi

  # Named by us, not by RUN_TAG, so results are self-identifying and two
  # checkpoints cannot overwrite each other's file.
  jsonl="${RESULTS_DIR}/${VARIANT}/${TASK}/seed${SEED_OFFSET}-ckpt${N}.jsonl"
  marker="${RESULTS_DIR}/.done-ckpt${N}"

  if [[ -f "${marker}" ]]; then
    SUMMARY+=("checkpoint-${N}: SKIPPED (done)")
    continue
  fi

  # A failure here must NOT fall through to a fresh run: that would re-evaluate
  # episodes already paid for and double-count them in the JSONL.
  if ! plan=$(resume_plan "${jsonl}"); then
    SUMMARY+=("checkpoint-${N}: SKIPPED (cannot read resume state, see above)")
    STATUS=1
    continue
  fi
  read -r next_seed scored <<<"${plan}"
  remaining=$(( EPISODES - scored ))
  if [[ "${remaining}" -le 0 ]]; then
    touch "${marker}"
    SUMMARY+=("checkpoint-${N}: already had ${scored}/${EPISODES}, marked done")
    continue
  fi

  echo "=========================================================="
  echo " checkpoint-${N}  (${where})"
  echo "   ${ckpt}"
  if [[ "${scored}" -gt 0 ]]; then
    echo "   RESUMING: ${scored}/${EPISODES} scored, ${remaining} to go,"
    echo "             next seed ${next_seed}"
  else
    echo "   fresh run: ${EPISODES} episodes from seed ${next_seed}"
  fi
  echo "   -> ${jsonl}"
  echo "=========================================================="

  if [[ "${DRY_RUN}" == "1" ]]; then
    SUMMARY+=("checkpoint-${N}: DRY-RUN (${remaining} episodes from ${next_seed})")
    continue
  fi

  # One eval per checkpoint, through eval_ablation.sbatch so the server, the
  # two-interpreter split and the cuDNN-safe environment are handled in one
  # place rather than reimplemented here.
  #
  # START_SEED is absolute and overrides the offset-derived default, which is
  # what makes a resume continue instead of re-running from the block start.
  PORT_OFFSET=$(( PORT_OFFSET + 1 ))
  UNIVTAC_JOB_CONFIG=1 \
  PORT=$(( PORT_BASE + PORT_OFFSET )) \
  REPO_ROOT="${REPO_ROOT}" \
  TASK="${TASK}" \
  VARIANT="${VARIANT}" \
  GROOT_MODEL="${ckpt}" \
  EPISODES="${remaining}" \
  SEED_OFFSET="${SEED_OFFSET}" \
  START_SEED="${next_seed}" \
  RUN_TAG="ckpt${N}" \
  RESULTS_DIR="${RESULTS_DIR}" \
    bash "${REPO_ROOT}/slurm/eval_ablation.sbatch"
  rc=$?

  scored_now=0
  if plan=$(resume_plan "${jsonl}"); then
    read -r _ scored_now <<<"${plan}"
  fi
  if [[ "${scored_now}" -ge "${EPISODES}" ]]; then
    touch "${marker}"
    SUMMARY+=("checkpoint-${N}: DONE ${scored_now}/${EPISODES}")
  else
    # Not a failure to shout about: a preemption lands here, and re-running
    # this script picks up from ${scored_now}.
    SUMMARY+=("checkpoint-${N}: PARTIAL ${scored_now}/${EPISODES} (exit ${rc}) -- re-run to continue")
    STATUS=1
  fi
done

echo
echo "=========================================================="
echo " sweep summary -- task=${TASK} seed block ${SEED_OFFSET}"
echo "=========================================================="
printf '  %s\n' "${SUMMARY[@]}"
echo
echo "Success rate per checkpoint:"
echo "  python ${REPO_ROOT}/scripts/compare_ablation.py --results-dir ${RESULTS_DIR}"
echo
echo "Re-run this exact command to continue anything PARTIAL; finished"
echo "checkpoints are skipped via their .done marker."
exit ${STATUS}
