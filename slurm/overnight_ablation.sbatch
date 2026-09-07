#!/bin/bash
# Full ablation for one task, unattended: finetune both arms, evaluate both,
# compare. Designed to be submitted and left alone overnight.
#
#   sbatch slurm/overnight_ablation.sbatch
#   sbatch --gres=gpu:4 --export=ALL,TASK=lift_bottle,EPISODES=50 slurm/overnight_ablation.sbatch
#
# BEFORE submitting, verify it without spending GPU time. DRY_RUN=1 runs the
# full preflight, prints the exact arguments every stage would use, checks the
# aggregator imports, and exits -- cheap enough for a login node:
#
#   bash -n slurm/overnight_ablation.sbatch          # syntax only, instant
#   DRY_RUN=1 bash slurm/overnight_ablation.sbatch   # paths, datasets, plan
#   sbatch --test-only slurm/overnight_ablation.sbatch  # submit filter + resources
#
# Stages, in order:
#   1. preflight     -- every path and dataset checked BEFORE any GPU time is spent
#   2. finetune      -- ARM=tactile, then ARM=baseline_finetuned (same recipe)
#   3. evaluate      -- each finetuned checkpoint on TASK
#   4. compare       -- success rates, Wilson intervals, delta
#
# Two properties that matter for an unattended run:
#
# * **Resumable.** Each stage records a marker and is skipped if its output
#   already exists. A walltime kill, then a plain resubmit, continues where it
#   stopped rather than redoing hours of training.
# * **No single failure ends the night.** A failed stage is recorded and the
#   run continues with everything still possible -- if the tactile finetune
#   dies, the baseline still trains and evaluates, so the morning brings a
#   diagnosis instead of an empty results directory.
#
# This cluster's submit filter requires a job name LONGER THAN 50 characters
# and rejects the submission outright otherwise. Do not shorten these.
# No --time here on purpose. Requesting more than a partition's limit leaves the
# job pending forever with REASON=PartitionTimeLimit rather than failing fast, so
# the partition's own default is safer. Check the limits with:
#   sinfo -o "%20P %10l %10L %6D %25G"     (%l = TIMELIMIT, %L = DEFAULTTIME)
# and pass --partition explicitly, or export SBATCH_PARTITION. Note `debug` is
# short-lived and is where jobs land by default here -- it is not where long
# training belongs.
#SBATCH --job-name=univtac-groot-full-ablation-finetune-both-arms-then-evaluate-and-compare
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:4
#SBATCH --wckey=project-short-name:sub_4dpdata

# Deliberately NOT `set -e`: a failing stage must be recorded and stepped over,
# not abort the job. Stage failures are tracked explicitly instead.
set -uo pipefail

# Locate the repo. `dirname "${BASH_SOURCE[0]}"` is NOT reliable here: under
# sbatch, SLURM copies the batch script into a spool directory
# (/var/spool/slurm/d/...), so it resolves to the spool, not the checkout --
# every derived path then points somewhere unwritable. SLURM_SUBMIT_DIR is the
# directory the job was submitted from, which is the right answer under sbatch;
# BASH_SOURCE is the right answer under plain `bash`. Try both and accept only a
# candidate that actually contains this repo.
if [[ -z "${REPO_ROOT:-}" ]]; then
  for _candidate in \
      "${SLURM_SUBMIT_DIR:-}" \
      "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"; do
    if [[ -n "${_candidate}" && -f "${_candidate}/univtac_groot/spec.py" ]]; then
      REPO_ROOT="${_candidate}"
      break
    fi
  done
fi
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "error: cannot locate the univtac-groot checkout." >&2
  echo "       Tried SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-unset}' and the" >&2
  echo "       script's own directory. Submit from the repo root, or pass" >&2
  echo "       REPO_ROOT=/path/to/UniVTAC explicitly." >&2
  exit 2
fi

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TASK="${TASK:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
EPISODES="${EPISODES:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
ARMS="${ARMS:-tactile baseline_finetuned}"

# launch_finetune.py defaults: --max-steps 10000, --save-steps 1000,
# --save-total-limit 5. That last one is a disk hazard here: a 3B checkpoint
# carries optimizer + scheduler + RNG state (needed for --resume-from-checkpoint),
# which lands around 40 GB each. Five of those per arm is ~200 GB, ~400 GB for
# both arms, on a shared 15T mount that already sits at 91%. Keep fewer and
# save less often -- still enough to resume, far less I/O on a network
# filesystem.
#
# With MAX_STEPS=10000 and SAVE_STEPS=2500 the run saves at 2500/5000/7500/
# 10000 and SAVE_TOTAL_LIMIT=3 keeps a rolling window of the last three, so the
# final checkpoint plus two fallbacks -- roughly 120 GB per arm.
#
# Note there is no metric-based "best checkpoint" to select: launch_finetune.py
# exposes no eval metric or load-best-at-end option, so the last checkpoint is
# the final model by definition. This script therefore points a stable
# `<output_dir>/final` symlink at the highest-numbered checkpoint after
# training, which is what evaluation uses.
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

DATA_ROOT="${DATA_ROOT:-${HOME}/jaewon/workspace/groot-data}"
# This cluster's submit filter requires MODEL_OUTPUT_DIR under
# /rlwrld-unified-checkpoints and rejects the job at `sbatch` time without it.
# Honour it as the checkpoint root by default: it is the cluster's designated
# place for model outputs, and it keeps ~120 GB per arm off the shared home
# mount. CKPT_ROOT still wins if set explicitly.
CKPT_ROOT="${CKPT_ROOT:-${MODEL_OUTPUT_DIR:-${HOME}/jaewon/workspace/groot-ckpt}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/eval_result}"
STAGE_DIR="${STAGE_DIR:-${CKPT_ROOT}/.stages}"

UNIVTAC_ROOT="${UNIVTAC_ROOT:?set UNIVTAC_ROOT to the UniVTAC checkout}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:?set UNIVTAC_PYTHON to the UniVTAC/Isaac Lab interpreter}"
GROOT_ROOT="${GROOT_ROOT:?set GROOT_ROOT to the Isaac-GR00T checkout}"
GROOT_PYTHON="${GROOT_PYTHON:?set GROOT_PYTHON to the GR00T interpreter}"

# hf_transfer is absent from a uv-created venv unless explicitly added, and
# HF_HUB_ENABLE_HF_TRANSFER=1 without it is a hard error mid-download. Default
# off for an unattended run; the weights are already cached by now anyway.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"
export PYTHONUNBUFFERED=1

# cuDNN cannot initialise on a driver older than GR00T's CUDA 12.8 build (see
# docs/SETUP.md). Leave this at 1 until an admin upgrades the driver; it costs
# wall-clock on the conv path but keeps results comparable across arms.
DISABLE_CUDNN="${DISABLE_CUDNN:-1}"

# SLURM re-indexes allocated GPUs to 0..N-1 inside the job, so counting
# CUDA_VISIBLE_DEVICES gives the usable width for data-parallel training.
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-0}")}"

mkdir -p logs "${STAGE_DIR}" "${CKPT_ROOT}" "${RESULTS_DIR}"

STAGE_LOG_DIR="${REPO_ROOT}/logs/overnight-${SLURM_JOB_ID:-$$}"
mkdir -p "${STAGE_LOG_DIR}"

declare -a SUMMARY=()
# Both arms are evaluated in the same job, so they would otherwise derive the
# same port from SLURM_JOB_ID. The first server is killed by its own EXIT trap,
# but a socket in TIME_WAIT would still make the second bind fail, so hand each
# arm its own port.
PORT_BASE="${PORT_BASE:-$((20000 + (${SLURM_JOB_ID:-$$} % 15000)))}"
PORT_OFFSET=0

banner() {
  echo
  echo "=========================================================="
  echo " $*"
  echo " $(date '+%Y-%m-%d %H:%M:%S')  node=$(hostname)"
  echo "=========================================================="
}

record() { SUMMARY+=("$1"); echo "[stage] $1"; }

# --------------------------------------------------------------------------- #
# 1. Preflight -- everything checkable, checked before any GPU time is spent
# --------------------------------------------------------------------------- #
banner "preflight"
echo " task=${TASK} config=${TASK_CONFIG} arms='${ARMS}' episodes=${EPISODES}"
echo " gpus=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo " data=${DATA_ROOT}"
echo " ckpt=${CKPT_ROOT}"
echo " results=${RESULTS_DIR}"
echo " disable_cudnn=${DISABLE_CUDNN}"
echo " stage logs=${STAGE_LOG_DIR}"

FATAL=0
fatal() { echo "PREFLIGHT FAIL: $*" >&2; FATAL=1; }

for path in "${UNIVTAC_ROOT}" "${GROOT_ROOT}"; do
  [[ -d "${path}" ]] || fatal "not a directory: ${path}"
done
for interp in "${UNIVTAC_PYTHON}" "${GROOT_PYTHON}"; do
  [[ -x "${interp}" ]] || fatal "not executable: ${interp}"
done
[[ -f "${GROOT_ROOT}/gr00t/experiment/launch_finetune.py" ]] \
  || fatal "missing ${GROOT_ROOT}/gr00t/experiment/launch_finetune.py"
"${GROOT_PYTHON}" -c "import gr00t" 2>/dev/null \
  || fatal "${GROOT_PYTHON} cannot import gr00t (try: cd ${GROOT_ROOT} && uv sync)"

# A missing or truncated dataset is the classic way an overnight run wastes a
# night, so verify shape now rather than discovering it at step 0 of training.
for arm in ${ARMS}; do
  dataset="${DATA_ROOT}/univtac-${TASK}-${arm}"
  if [[ ! -d "${dataset}" ]]; then
    fatal "no dataset for arm '${arm}' at ${dataset}"
    continue
  fi
  for required in meta/info.json meta/modality.json meta/stats.json; do
    [[ -f "${dataset}/${required}" ]] || fatal "${dataset} is missing ${required}"
  done
  n_parquet=$(find "${dataset}/data" -name '*.parquet' 2>/dev/null | wc -l)
  n_videos=$(find "${dataset}/videos" -name '*.mp4' 2>/dev/null | wc -l)
  echo " ${arm}: ${n_parquet} parquet, ${n_videos} mp4  (${dataset})"
  [[ "${n_parquet}" -gt 0 ]] || fatal "${dataset} has no parquet shards"
done

[[ -f "${REPO_ROOT}/configs/modality/univtac_tactile_config.py" ]] \
  || fatal "missing configs/modality/univtac_tactile_config.py"
[[ -f "${REPO_ROOT}/configs/modality/univtac_baseline_config.py" ]] \
  || fatal "missing configs/modality/univtac_baseline_config.py"

# Pin the training recipe across runs. Both arms must train under identical
# conditions or the ablation measures the recipe, not the tactile modality --
# and the two arms are NOT necessarily trained by the same job invocation: a
# walltime kill mid-way means the second arm trains on a later resubmission,
# which the scheduler may hand a different number of GPUs. That matters because
# with data-parallel training --num-gpus multiplies the effective batch size, so
# the same --max-steps on 4 GPUs sees roughly twice the data as on 2.
#
# So record the recipe when the first arm starts and refuse to continue if a
# later run differs. Set ALLOW_RECIPE_CHANGE=1 to override deliberately (e.g.
# you are restarting the whole experiment from scratch).
RECIPE_FILE="${STAGE_DIR}/recipe-${TASK}.env"
if [[ -f "${RECIPE_FILE}" ]]; then
  RECORDED_NUM_GPUS=""; RECORDED_MAX_STEPS=""
  # shellcheck source=/dev/null
  source "${RECIPE_FILE}"
  if [[ "${RECORDED_NUM_GPUS}" != "${NUM_GPUS}" || "${RECORDED_MAX_STEPS}" != "${MAX_STEPS}" ]]; then
    if [[ "${ALLOW_RECIPE_CHANGE:-0}" == "1" ]]; then
      echo " WARNING: recipe changed (was ${RECORDED_NUM_GPUS} gpu / ${RECORDED_MAX_STEPS} steps," >&2
      echo " now ${NUM_GPUS} gpu / ${MAX_STEPS} steps). Proceeding because" >&2
      echo " ALLOW_RECIPE_CHANGE=1 -- the arms are no longer directly comparable." >&2
    else
      fatal "recipe mismatch: this task was started with ${RECORDED_NUM_GPUS} GPU(s) and
       --max-steps ${RECORDED_MAX_STEPS}, but this run has ${NUM_GPUS} GPU(s) and
       --max-steps ${MAX_STEPS}. Training the two arms under different effective
       batch sizes would confound the ablation. Either resubmit with
       --gres=gpu:${RECORDED_NUM_GPUS} and MAX_STEPS=${RECORDED_MAX_STEPS}, or pass
       ALLOW_RECIPE_CHANGE=1 and delete ${STAGE_DIR} to restart the experiment."
    fi
  else
    echo " recipe: ${NUM_GPUS} gpu, ${MAX_STEPS} steps (matches ${RECIPE_FILE})"
  fi
elif [[ "${DRY_RUN:-0}" != "1" ]]; then
  printf 'RECORDED_NUM_GPUS=%s\nRECORDED_MAX_STEPS=%s\n' "${NUM_GPUS}" "${MAX_STEPS}" \
    >"${RECIPE_FILE}"
  echo " recipe: ${NUM_GPUS} gpu, ${MAX_STEPS} steps (recorded in ${RECIPE_FILE})"
fi

df -h "${CKPT_ROOT}" 2>/dev/null || true
# shellcheck disable=SC2086 - word splitting is the point here
N_ARMS=$(set -- ${ARMS}; echo $#)
echo " the space above must hold roughly $((N_ARMS * SAVE_TOTAL_LIMIT * 40)) GB"
echo " (${N_ARMS} arms x ${SAVE_TOTAL_LIMIT} retained checkpoints x ~40 GB each)"

if [[ "${FATAL}" -ne 0 ]]; then
  echo
  echo "Preflight failed. Nothing was run. Fix the above and resubmit." >&2
  exit 2
fi
record "preflight OK"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# launch_finetune.py's exact output layout is not pinned by this repo, so probe
# rather than assume: prefer the highest-numbered checkpoint-N directory, and
# fall back to the output root if the final model was written there directly.
latest_checkpoint() {
  local root="$1" best="" best_n=-1 dir n
  for dir in "${root}"/checkpoint-*; do
    [[ -d "${dir}" ]] || continue
    n="${dir##*checkpoint-}"
    [[ "${n}" =~ ^[0-9]+$ ]] || continue
    if (( n > best_n )); then best_n="${n}"; best="${dir}"; fi
  done
  if [[ -z "${best}" ]]; then
    if [[ -f "${root}/config.json" ]] \
       || compgen -G "${root}/*.safetensors" >/dev/null 2>&1; then
      best="${root}"
    fi
  fi
  printf '%s' "${best}"
}

# --------------------------------------------------------------------------- #
# 2. Finetune each arm
# --------------------------------------------------------------------------- #
for arm in ${ARMS}; do
  out_dir="${CKPT_ROOT}/${TASK}-${arm}"
  marker="${STAGE_DIR}/finetune-${TASK}-${arm}.done"
  log="${STAGE_LOG_DIR}/finetune-${arm}.log"

  if [[ -f "${marker}" ]]; then
    record "finetune ${arm}: SKIPPED (already done: ${marker})"
    continue
  fi

  # A checkpoint without the marker means an earlier attempt was killed
  # mid-training. Resume it rather than either restarting from step 0 or
  # treating a partial run as finished -- launch_finetune.py has
  # --resume-from-checkpoint, which picks the latest checkpoint-* in output_dir.
  # It defaults to off precisely so an accidental rerun does not merge two
  # experiments, so it must be requested explicitly.
  resume_args=""
  existing=$(latest_checkpoint "${out_dir}")
  if [[ -n "${existing}" ]]; then
    resume_args="--resume-from-checkpoint"
    echo "[stage] resuming ${arm} from ${existing}"
  fi

  banner "finetune ${arm}  (${NUM_GPUS} gpu, max_steps=${MAX_STEPS})"
  echo "[stage] log -> ${log}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would run slurm/finetune.sbatch with:"
    echo "[dry-run]   ARM=${arm}"
    echo "[dry-run]   DATASET=${DATA_ROOT}/univtac-${TASK}-${arm}"
    echo "[dry-run]   OUTPUT_DIR=${out_dir}"
    echo "[dry-run]   NUM_GPUS=${NUM_GPUS}  DISABLE_CUDNN=${DISABLE_CUDNN}"
    echo "[dry-run]   EXTRA_ARGS=--max-steps ${MAX_STEPS} --save-steps ${SAVE_STEPS} --save-total-limit ${SAVE_TOTAL_LIMIT} ${resume_args}"
    record "finetune ${arm}: DRY-RUN (not executed)"
    continue
  fi

  GROOT_ROOT="${GROOT_ROOT}" \
  GROOT_PYTHON="${GROOT_PYTHON}" \
  ARM="${arm}" \
  DATASET="${DATA_ROOT}/univtac-${TASK}-${arm}" \
  OUTPUT_DIR="${out_dir}" \
  NUM_GPUS="${NUM_GPUS}" \
  DISABLE_CUDNN="${DISABLE_CUDNN}" \
  HF_HOME="${HF_HOME}" \
  EXTRA_ARGS="--max-steps ${MAX_STEPS} --save-steps ${SAVE_STEPS} --save-total-limit ${SAVE_TOTAL_LIMIT} ${resume_args} ${EXTRA_ARGS:-}" \
    bash "${REPO_ROOT}/slurm/finetune.sbatch" >"${log}" 2>&1
  status=$?

  ckpt=$(latest_checkpoint "${out_dir}")
  if [[ ${status} -eq 0 && -n "${ckpt}" ]]; then
    # Stable path for evaluation and for anything downstream, so a rolling
    # save-total-limit window does not leave dangling references to a pruned
    # checkpoint-N. Relative target so the tree stays movable.
    if [[ "${ckpt}" != "${out_dir}" ]]; then
      ln -sfn "$(basename "${ckpt}")" "${out_dir}/final"
      echo "[stage] ${out_dir}/final -> $(basename "${ckpt}")"
    fi
    touch "${marker}"
    record "finetune ${arm}: OK -> ${ckpt}"
  else
    record "finetune ${arm}: FAILED (exit ${status}, checkpoint='${ckpt:-none}') see ${log}"
  fi
done

# --------------------------------------------------------------------------- #
# 3. Evaluate each arm that produced a checkpoint
# --------------------------------------------------------------------------- #
for arm in ${ARMS}; do
  out_dir="${CKPT_ROOT}/${TASK}-${arm}"
  # Prefer the stable `final` symlink the finetune stage writes; fall back to
  # probing, so an externally trained output_dir still evaluates.
  if [[ -e "${out_dir}/final" ]]; then
    ckpt="${out_dir}/final"
  else
    ckpt=$(latest_checkpoint "${out_dir}")
  fi
  log="${STAGE_LOG_DIR}/eval-${arm}.log"
  summary="${RESULTS_DIR}/${arm}/${TASK}/seed0-0.summary.json"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would run slurm/eval_ablation.sbatch with:"
    echo "[dry-run]   ARM=${arm} TASK=${TASK} TASK_CONFIG=${TASK_CONFIG}"
    echo "[dry-run]   GROOT_MODEL=${ckpt:-<from the finetune stage>}"
    echo "[dry-run]   EPISODES=${EPISODES} EXECUTION_HORIZON=${EXECUTION_HORIZON}"
    echo "[dry-run]   PORT=$((PORT_BASE + PORT_OFFSET + 1))"
    record "eval ${arm}: DRY-RUN (not executed)"
    PORT_OFFSET=$((PORT_OFFSET + 1))
    continue
  fi

  if [[ -z "${ckpt}" ]]; then
    record "eval ${arm}: SKIPPED (no checkpoint to evaluate)"
    continue
  fi
  if [[ -f "${summary}" ]]; then
    record "eval ${arm}: SKIPPED (summary exists: ${summary})"
    continue
  fi

  PORT_OFFSET=$((PORT_OFFSET + 1))
  banner "evaluate ${arm}  ckpt=${ckpt}  port=$((PORT_BASE + PORT_OFFSET))"
  echo "[stage] log -> ${log}"
  PORT=$((PORT_BASE + PORT_OFFSET)) \
  UNIVTAC_ROOT="${UNIVTAC_ROOT}" \
  UNIVTAC_PYTHON="${UNIVTAC_PYTHON}" \
  GROOT_PYTHON="${GROOT_PYTHON}" \
  ARM="${arm}" \
  TASK="${TASK}" \
  TASK_CONFIG="${TASK_CONFIG}" \
  EPISODES="${EPISODES}" \
  EXECUTION_HORIZON="${EXECUTION_HORIZON}" \
  GROOT_MODEL="${ckpt}" \
  RESULTS_DIR="${RESULTS_DIR}" \
  DISABLE_CUDNN="${DISABLE_CUDNN}" \
  HF_HOME="${HF_HOME}" \
    bash "${REPO_ROOT}/slurm/eval_ablation.sbatch" >"${log}" 2>&1
  status=$?

  if [[ -f "${summary}" ]]; then
    scored=$("${UNIVTAC_PYTHON}" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(f\"{d.get('episodes_scored',0)} scored, {d.get('successes',0)} success, \"
      f\"SR={d.get('success_rate_pct',0)}%\")
" "${summary}" 2>/dev/null || echo "unparseable")
    record "eval ${arm}: exit ${status} -- ${scored}"
  else
    record "eval ${arm}: FAILED (exit ${status}, no summary) see ${log}"
  fi
done

# --------------------------------------------------------------------------- #
# 4. Compare
# --------------------------------------------------------------------------- #
banner "compare"
# Even in a dry run, check the aggregator parses and its flags exist -- a typo
# here would otherwise only surface after both finetunes and both evals.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  if "${UNIVTAC_PYTHON}" "${REPO_ROOT}/scripts/compare_ablation.py" --help >/dev/null 2>&1; then
    record "compare: DRY-RUN (script imports and --help works)"
  else
    record "compare: DRY-RUN FAILED -- compare_ablation.py --help errored"
  fi
else
  "${UNIVTAC_PYTHON}" "${REPO_ROOT}/scripts/compare_ablation.py" \
    --results-dir "${RESULTS_DIR}" \
    --baseline-arm baseline_finetuned \
    --tactile-arm tactile \
    --json "${STAGE_LOG_DIR}/comparison.json" \
    2>&1 | tee "${STAGE_LOG_DIR}/compare.log"
  record "compare: exit ${PIPESTATUS[0]}"
fi

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
banner "summary"
for line in "${SUMMARY[@]}"; do echo "  ${line}"; done
echo
echo "  stage logs:   ${STAGE_LOG_DIR}"
echo "  checkpoints:  ${CKPT_ROOT}"
echo "  results:      ${RESULTS_DIR}"
echo
echo "  Resubmitting this script skips completed stages, so a walltime kill is"
echo "  safe to continue from. Delete a marker in ${STAGE_DIR} to force a redo."

# Exit non-zero if anything failed, so `sacct` reflects reality.
if printf '%s\n' "${SUMMARY[@]}" | grep -q "FAILED"; then
  exit 1
fi
exit 0
