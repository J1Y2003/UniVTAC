#!/bin/bash
# Prove the finetune -> evaluate chain works, in ~an hour, without touching the
# real run's outputs.
#
#   bash slurm/smoke_test.sh              # submit
#   bash slurm/smoke_test.sh --print      # show the sbatch command
#   bash slurm/smoke_test.sh --check      # report what a previous smoke run produced
#
# Three interfaces in this pipeline have never actually executed:
#
#   1. launch_finetune.py accepting our modality config under NEW_EMBODIMENT
#   2. a 113-D state vector surviving the trainer's validation
#   3. Gr00tPolicy loading a *finetuned* checkpoint and answering get_action
#
# Everything else is verified, but those three only appear once training starts,
# and a full run costs a day to find out. So: 20 training steps and 1 evaluation
# episode. The success rate is meaningless at 20 steps -- the only question is
# whether every interface connects.
#
# ISOLATION. This writes checkpoints, stage markers and results under
# SMOKE_ROOT, never the real CKPT_ROOT or eval_result/. That matters for more
# than tidiness: benchmark_task.sbatch keeps its stage markers and its
# pinned training recipe in CKPT_ROOT/.stages, so a shared root would let a
# 20-step smoke run mark the real finetune "complete" and pin the recipe to
# 1 GPU / 20 steps. The dataset is the only thing shared, read-only.

set -uo pipefail

WHOAMI="${USER:-$(id -un)}"

WORKSPACE="${WORKSPACE:-${HOME}/jaewon/workspace}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UNIVTAC_ROOT="${UNIVTAC_ROOT:-${WORKSPACE}/UniVTAC-sim}"
GROOT_ROOT="${GROOT_ROOT:-${WORKSPACE}/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_ROOT}/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:-${HOME}/miniconda3/envs/UniVTAC/bin/python}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/groot-data}"
HF_HOME="${HF_HOME:-${HOME}/jaewon/hf_cache}"

# Everything this run writes lives here. Delete it and nothing is lost.
SMOKE_ROOT="${SMOKE_ROOT:-/rlwrld-unified-checkpoints/${WHOAMI}/checkpoints/univtac-groot-smoke}"
# The submit filter demands MODEL_OUTPUT_DIR under /rlwrld-unified-checkpoints.
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${SMOKE_ROOT}}"

TASK="${TASK:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-clean}"

# Deliberately tiny. 20 steps with a save at 10 proves the training loop, the
# optimizer, and checkpoint writing; 1 episode proves the whole eval path.
MAX_STEPS="${MAX_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
EPISODES="${EPISODES:-1}"

# 1 GPU schedules soonest, and the recipe pinning is isolated under SMOKE_ROOT,
# so this does not constrain the real run's GPU count.
#
# `background` rather than `debug`: this job EVALUATES as well as trains -- that
# is the whole point of it -- and evaluation is only allowed on `background`.
# The real run splits the two stages across two partitions; the smoke test
# deliberately does not, because what it proves is that the chain holds inside
# one job. 20 steps plus one episode fits either partition's limit.
PARTITION="${PARTITION:-background}"
GPUS="${GPUS:-1}"
# No --time: site rule. The job takes the partition maximum automatically.
# Site rule: every sbatch and srun carries this.
WCKEY="${WCKEY:-project-short-name:sub_4dpdata}"
# Off by default so a throwaway run does not clutter the real project's plots.
USE_WANDB="${USE_WANDB:-0}"

MODE="${1:-submit}"

# --------------------------------------------------------------------------- #
if [[ "${MODE}" == "--check" ]]; then
  echo "Smoke run outputs under ${SMOKE_ROOT}:"
  for variant in tactile baseline_finetuned; do
    out_dir="${SMOKE_ROOT}/${TASK}-${variant}"
    ckpts=$(ls -d "${out_dir}"/checkpoint-* 2>/dev/null | tr '\n' ' ')
    marker="${SMOKE_ROOT}/.stages/finetune-${TASK}-${variant}.done"
    summary="${SMOKE_ROOT}/eval_result/${variant}/${TASK}/seed0-0.summary.json"
    printf '  %-20s finetune=%s  checkpoints=%s\n' "${variant}" \
      "$([[ -f "${marker}" ]] && echo DONE || echo "incomplete")" \
      "${ckpts:-none}"
    if [[ -f "${summary}" ]]; then
      echo "      eval summary:"
      sed 's/^/        /' "${summary}" | head -14
    else
      echo "      eval summary: MISSING"
    fi
  done
  echo
  echo "What to look for:"
  echo "  * a checkpoint-10 or -20 directory     -> trainer accepted the modality"
  echo "                                            config and the 113-D state"
  echo "  * eval summary with episodes_scored=1  -> inference on a finetuned"
  echo "                                            checkpoint works end to end"
  echo "  * episodes_errored=1 instead           -> read the server log:"
  echo "      ${SMOKE_ROOT}/eval_result/<variant>/${TASK}/server-0.log"
  exit 0
fi

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
  [[ "${n}" -gt 0 ]] || { echo "MISSING dataset for ${variant}" >&2; FAIL=1; }
done
# Every GR00T checkpoint loads the gated nvidia/Cosmos-Reason2-2B, so no token
# means a 401 about a minute into training -- after the job already holds a GPU.
# Warn here, where it costs nothing. HF_HOME is redirected, which also moves
# where a stored login is read from ($HF_HOME/token), so HF_TOKEN in the
# environment is the reliable route; --export=ALL carries it into the job.
if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_HOME}/token" && ! -s "${HOME}/.cache/huggingface/token" ]]; then
  echo "WARNING: no HF_TOKEN and no token file under HF_HOME=${HF_HOME}." >&2
  echo "         The job will fail loading nvidia/Cosmos-Reason2-2B (401)." >&2
  echo "         Fix: export HF_TOKEN=hf_...   then re-run this." >&2
  FAIL=1
fi
mkdir -p "${REPO_ROOT}/logs" 2>/dev/null || FAIL=1
if [[ "${FAIL}" -ne 0 ]]; then
  echo "Fix the above. Nothing submitted." >&2
  exit 2
fi

EXPORTS="ALL"
EXPORTS+=",MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR}"
EXPORTS+=",CKPT_ROOT=${SMOKE_ROOT}"
EXPORTS+=",RESULTS_DIR=${SMOKE_ROOT}/eval_result"
EXPORTS+=",STAGE_DIR=${SMOKE_ROOT}/.stages"
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
EXPORTS+=",USE_WANDB=${USE_WANDB}"
# DRY_RUN left exported from testing would ride in on --export=ALL and the job
# would exit in seconds having proved nothing.
EXPORTS+=",DRY_RUN=0"

# No --time, no --cpus-per-task, no --mem: all three are site rules.
SBATCH_ARGS=(
  --partition="${PARTITION}"
  --gres="gpu:${GPUS}"
  --wckey="${WCKEY}"
  --export="${EXPORTS}"
)

echo "Smoke test: ${MAX_STEPS} training steps, ${EPISODES} eval episode(s)"
echo "  task       ${TASK}/${TASK_CONFIG}"
echo "  writes to  ${SMOKE_ROOT}          <- isolated, safe to delete"
echo "  reads      ${DATA_ROOT}           <- shared, read-only"
echo "  job        ${GPUS} gpu on ${PARTITION}, wckey=${WCKEY}, no --time"
echo

if [[ "${MODE}" == "--print" ]]; then
  echo "sbatch ${SBATCH_ARGS[*]} slurm/benchmark_task.sbatch"
  exit 0
fi

# A second queued job competes with the real run for this cluster's per-user GPU
# cap. Say so rather than silently delaying the thing that matters.
existing=$(squeue -h -u "${WHOAMI}" --format="%i %T %200j" 2>/dev/null \
           | grep "univtac-groot" || true)
if [[ -n "${existing}" ]]; then
  echo "NOTE: you already have queued/running jobs:"
  echo "${existing}" | sed 's/^/  /'
  echo
  echo "Adding this one competes with them for the per-user GPU cap. If the big"
  echo "run is still PENDING, consider holding it so the smoke test goes first --"
  echo "if an interface is broken, that pending run would fail anyway:"
  echo "  scontrol hold <jobid>     # keeps queue position, stops it starting"
  echo "  scontrol release <jobid>  # after the smoke test passes"
  echo
  read -r -t 15 -p "Submit anyway? [y/N] " reply || reply=""
  echo
  [[ "${reply}" == "y" || "${reply}" == "Y" ]] || { echo "Nothing submitted."; exit 0; }
fi

cd "${REPO_ROOT}"
out=$(sbatch --parsable "${SBATCH_ARGS[@]}" slurm/benchmark_task.sbatch) || exit $?
jobid="${out%%;*}"
echo "Submitted smoke job ${jobid}"
echo
echo "Watch:   tail -f logs/univtac-groot-full-*-${jobid}.out"
echo "Verdict: bash slurm/smoke_test.sh --check"
echo
echo "Expected timeline: a few minutes to load the 3B checkpoint, then 20 steps"
echo "(~1.9 s/it), two checkpoint writes, and a 1-episode eval per variant."
