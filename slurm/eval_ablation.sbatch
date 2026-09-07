#!/bin/bash
# Evaluate one ablation arm on one UniVTAC task, fully headless.
#
# Two processes, one job: the GR00T inference server (Python 3.12, holds the 3B
# checkpoint) and the UniVTAC evaluator (Python 3.10, Isaac Sim). Different minor
# Python versions means they cannot share a virtualenv at all, so they talk over
# a loopback ZeroMQ socket. This mirrors how UniVTAC runs SmolVLA
# (policy/smolvla/smolvla_server.py). See docs/SETUP.md.
#
# Submit one job per (arm, task). To sweep, use the array form below or the
# helper: slurm/submit_ablation.sh
#
#   sbatch --export=ALL,ARM=baseline,TASK=insert_hole slurm/eval_ablation.sbatch
#   sbatch --export=ALL,ARM=tactile,TASK=insert_hole,GROOT_MODEL=/ckpt/tactile slurm/eval_ablation.sbatch
#
#SBATCH --job-name=univtac-groot-eval
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration (override any of these with --export)
# --------------------------------------------------------------------------- #
ARM="${ARM:-baseline}"                       # baseline | baseline_finetuned | tactile
TASK="${TASK:-insert_hole}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
EPISODES="${EPISODES:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
SEED_OFFSET="${SEED_OFFSET:-0}"

# Repo + environment paths.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UNIVTAC_ROOT="${UNIVTAC_ROOT:?set UNIVTAC_ROOT to the UniVTAC checkout}"
GROOT_PYTHON="${GROOT_PYTHON:?set GROOT_PYTHON to the GR00T interpreter, e.g. Isaac-GR00T/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:?set UNIVTAC_PYTHON to the UniVTAC/Isaac Lab interpreter}"

# Checkpoint + embodiment tag. The baseline runs zero-shot on the released
# model; the tactile arm needs a NEW_EMBODIMENT finetune (docs/ABLATION.md).
if [[ "${ARM}" == "baseline" ]]; then
  GROOT_MODEL="${GROOT_MODEL:-nvidia/GR00T-N1.7-3B}"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT}"
  MODALITY_CONFIG="${MODALITY_CONFIG:-}"
elif [[ "${ARM}" == "tactile" ]]; then
  GROOT_MODEL="${GROOT_MODEL:?tactile needs a finetuned checkpoint; set GROOT_MODEL}"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
  MODALITY_CONFIG="${MODALITY_CONFIG:-${REPO_ROOT}/configs/modality/univtac_tactile_config.py}"
else
  GROOT_MODEL="${GROOT_MODEL:?baseline_finetuned needs a checkpoint; set GROOT_MODEL}"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
  MODALITY_CONFIG="${MODALITY_CONFIG:-${REPO_ROOT}/configs/modality/univtac_baseline_config.py}"
fi

# Device placement. Isaac Sim (scene + offscreen rendering) and GR00T N1.7
# (~7 GB in bf16, plus activations) share this job's allocation. On one GPU that
# needs roughly 24 GB free; request --gres=gpu:2 and they are split
# automatically. SLURM re-indexes allocated GPUs to 0..N-1 inside the job.
GPU_COUNT=$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-0}")
if [[ "${GPU_COUNT}" -ge 2 ]]; then
  SERVER_DEVICE="${SERVER_DEVICE:-cuda:1}"
  SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
else
  SERVER_DEVICE="${SERVER_DEVICE:-cuda:0}"
  SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
fi

# A per-job port keeps concurrent array tasks on one node from colliding.
PORT="${PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/eval_result}"
RUN_TAG="${SLURM_ARRAY_TASK_ID:-0}"
OUTPUT="${RESULTS_DIR}/${ARM}/${TASK}/seed${SEED_OFFSET}-${RUN_TAG}.jsonl"
SERVER_LOG="${RESULTS_DIR}/${ARM}/${TASK}/server-${RUN_TAG}.log"

mkdir -p logs "$(dirname "${OUTPUT}")"

echo "=========================================================="
echo " arm=${ARM} task=${TASK} config=${TASK_CONFIG}"
echo " model=${GROOT_MODEL}  tag=${EMBODIMENT_TAG}"
echo " episodes=${EPISODES} execution_horizon=${EXECUTION_HORIZON}"
echo " gpus=${GPU_COUNT}  server=${SERVER_DEVICE}  sim=${SIM_DEVICE}"
echo " port=${PORT}  output=${OUTPUT}"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "=========================================================="

# Keep HF/torch caches on shared scratch: login-node home quotas are tight and
# every rank would otherwise re-download the 3B checkpoint.
export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp}/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Credentials for the gated Cosmos-Reason2-2B backbone. Note the interaction:
# redirecting HF_HOME above also moves where the CLI's stored token is looked
# for, so a token saved under the default ~/.cache/huggingface would NOT be
# found here. Exporting HF_TOKEN avoids that entirely -- the env var takes
# precedence over any stored login, which is also how you override a shared
# machine's ambient `hf auth login` identity.
if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_HOME}/token" && ! -s "${HOME}/.cache/huggingface/token" ]]; then
  echo "[job] WARNING: no HF_TOKEN and no token file under HF_HOME=${HF_HOME}." >&2
  echo "[job] The server will fail to load nvidia/Cosmos-Reason2-2B (GatedRepoError)." >&2
  echo "[job] Fix: export HF_TOKEN=hf_...  then resubmit (see docs/SETUP.md)." >&2
fi
export TOKENIZERS_PARALLELISM=false
# Cap thread pools: 8 CPUs shared between Isaac Sim and the server, and BLAS
# defaulting to all cores causes heavy contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# Unbuffered stdout/stderr so both logs stream live rather than in 8 KB blocks.
export PYTHONUNBUFFERED=1
# Offscreen rendering only. No GUI, no livestream: this must survive sbatch.
export DISPLAY=""

# --------------------------------------------------------------------------- #
# 1. GR00T inference server
# --------------------------------------------------------------------------- #
# -u is not optional here: the server's stdout is redirected to SERVER_LOG, and
# Python block-buffers a non-tty stdout, so without it the log stays empty for
# minutes and a healthy server is indistinguishable from a hung one.
SERVER_CMD=(
  "${GROOT_PYTHON}" -u -m univtac_groot.server.run_server
  --model-path "${GROOT_MODEL}"
  --embodiment-tag "${EMBODIMENT_TAG}"
  --host 127.0.0.1
  --port "${PORT}"
  --device "${SERVER_DEVICE}"
)
if [[ -n "${MODALITY_CONFIG}" ]]; then
  SERVER_CMD+=(--modality-config-path "${MODALITY_CONFIG}")
fi
# Escape hatch for a broken cuDNN in the GR00T env: DISABLE_CUDNN=1
if [[ "${DISABLE_CUDNN:-0}" == "1" ]]; then
  SERVER_CMD+=(--disable-cudnn)
fi

# The server must NOT inherit the UniVTAC environment's CUDA settings. UniVTAC's
# conda env carries CUDA 12.4 (libuipc's env.yaml) and its installer exports
#     CUDA_HOME=$CONDA_PREFIX; LD_LIBRARY_PATH=$CUDA_HOME/lib
# while GR00T's torch is cu128 with its own bundled cuDNN, resolved through its
# venv. A leaked LD_LIBRARY_PATH makes the server load the wrong cuDNN, which
# surfaces as `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` on the first
# get_action -- long after the weights loaded fine, so it looks like a model bug.
# `env -u` scrubs them for the server only; the evaluator keeps them, since
# Isaac Sim genuinely needs them.
SERVER_ENV=(env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH -u CONDA_PREFIX)

echo "[job] starting server: ${SERVER_CMD[*]}"
( cd "${REPO_ROOT}" && PYTHONPATH="${REPO_ROOT}" "${SERVER_ENV[@]}" "${SERVER_CMD[@]}" )   >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "[job] cleaning up server pid ${SERVER_PID}"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# 2. UniVTAC evaluator (waits for the server via Gr00tClient.wait_until_ready)
# --------------------------------------------------------------------------- #
EVAL_CMD=(
  "${UNIVTAC_PYTHON}" "${REPO_ROOT}/scripts/run_eval.py"
  --task "${TASK}"
  --task-config "${TASK_CONFIG}"
  --arm "${ARM}"
  --univtac-root "${UNIVTAC_ROOT}"
  --host 127.0.0.1
  --port "${PORT}"
  --episodes "${EPISODES}"
  --seed-offset "${SEED_OFFSET}"
  --execution-horizon "${EXECUTION_HORIZON}"
  --output "${OUTPUT}"
  --startup-timeout 1800
  --device "${SIM_DEVICE}"
)
if [[ "${ARM}" == "tactile" ]]; then
  EVAL_CMD+=(--tactile-mode "${TACTILE_MODE:-depth_pool}")
fi

echo "[job] starting evaluator: ${EVAL_CMD[*]}"
cd "${UNIVTAC_ROOT}"
# Capture the status rather than letting `set -e` abort here, so the summary
# below (and the server log path) is still printed when a run fails.
set +e
PYTHONPATH="${REPO_ROOT}:${UNIVTAC_ROOT}" "${EVAL_CMD[@]}"
STATUS=$?
set -e

echo "[job] evaluator exited ${STATUS}"
echo "[job] results:    ${OUTPUT}"
echo "[job] server log: ${SERVER_LOG}"
if [[ ${STATUS} -ne 0 ]]; then
  echo "[job] non-zero exit: read the SERVER log first -- a checkpoint that fails"
  echo "[job] to load shows up here only as a wait_until_ready timeout."
fi
exit ${STATUS}
