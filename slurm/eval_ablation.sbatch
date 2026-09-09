#!/bin/bash
# Evaluate one ablation variant on one UniVTAC task, fully headless.
#
# Two processes, one job: the GR00T inference server (Python 3.12, holds the 3B
# checkpoint) and the UniVTAC evaluator (Python 3.10, Isaac Sim). Different minor
# Python versions means they cannot share a virtualenv at all, so they talk over
# a loopback ZeroMQ socket. This mirrors how UniVTAC runs SmolVLA
# (policy/smolvla/smolvla_server.py). See docs/SETUP.md.
#
# Submit one job per (variant, task). To sweep, use the array form below or the
# helper: slurm/submit_ablation.sh
#
#   sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,VARIANT=baseline,TASK=insert_hole slurm/eval_ablation.sbatch
#   sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,VARIANT=tactile,TASK=insert_hole,GROOT_MODEL=/ckpt/tactile slurm/eval_ablation.sbatch
#
# SITE RULES (see CLAUDE.md, "Cluster rules"). Enforced here; do not "helpfully"
# reintroduce any of them:
#
#   * NEVER pass --time. A job gets the maximum the partition allows, or runs
#     until the script exits, whichever comes first, so a limit can only cut the
#     run short. Over-requesting also leaves the job pending forever with
#     REASON=PartitionTimeLimit rather than failing fast.
#   * NEVER pass --cpus-per-task or --mem. Jobs take the node's per-GPU
#     defaults, and the submit filter rejects the job if either is specified.
#   * ALWAYS pass --wckey=project-short-name:sub_4dpdata.
#   * MODEL_OUTPUT_DIR must be in the SUBMITTING environment, for every job --
#     including CPU-only ones that write no checkpoints. The filter checks it at
#     submit time, before this script runs, so no in-script default can help:
#       ❌ ERROR: MODEL_OUTPUT_DIR가 없습니다.
#          MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jimin/... 로 지정하세요.
#     `source env.sh` sets it and --export=ALL carries it in.
#
# Do pass --partition explicitly, or export SBATCH_PARTITION: `debug` is the
# default here and is short-lived, so it is not where long training belongs.
#SBATCH --job-name=univtac-groot-evaluate-one-ablation-variant-on-one-univtac-benchmark-task
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
# Evaluation is not allowed on sjw_alinlab. This header is the floor, not the
# guarantee: --partition on the command line and SBATCH_PARTITION in the shell
# both outrank it, which is why slurm/submit_ablation.sh and
# slurm/submit_benchmark.sh also pass it explicitly.
#SBATCH --partition=background
#SBATCH --wckey=project-short-name:sub_4dpdata

set -euo pipefail

# --------------------------------------------------------------------------- #
# The site wckey, checked at runtime
# --------------------------------------------------------------------------- #
# sbatch precedence is: command line > environment variable > #SBATCH directive.
# So a stale SBATCH_WCKEY in the submitting shell (typically a copy of
# env.example.sh from before the value was corrected) SILENTLY OVERRIDES the
# `#SBATCH --wckey=project-short-name:sub_4dpdata` header above. That is exactly how a job once went
# out under the wrong project. The header alone is not enough; check it.
#
# SLURM_WCKEY is only set inside the job when a wckey was applied, so an unset
# value defaults to the right answer rather than failing a valid job.
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_WCKEY:-project-short-name:sub_4dpdata}" != "project-short-name:sub_4dpdata" ]]; then
  echo "error: this job's wckey is '${SLURM_WCKEY}', not 'project-short-name:sub_4dpdata'." >&2
  echo "       Every job here must be accounted to 'project-short-name:sub_4dpdata'." >&2
  echo "       Something in your environment overrode the #SBATCH header --" >&2
  echo "       almost certainly a stale SBATCH_WCKEY in env.sh. Fix it with:" >&2
  echo "         export SBATCH_WCKEY=project-short-name:sub_4dpdata SLURM_WCKEY=project-short-name:sub_4dpdata" >&2
  echo "       then resubmit. 'python scripts/preflight.py' also checks this." >&2
  exit 2
fi


# --------------------------------------------------------------------------- #
# Configuration (override any of these with --export)
# --------------------------------------------------------------------------- #
# ARM= was the old name for this. Honour it rather than ignoring it: an
# unrecognised environment variable fails *silently*, so a stale
# `ARM=baseline_finetuned` command would quietly run the tactile variant
# instead -- a wrong result that looks like a right one. (A stale --arm flag
# fails loudly on the CLI, so no alias is needed there.)
if [[ -n "${ARM:-}" && -z "${VARIANT:-}" ]]; then
  echo "[job] note: ARM is deprecated, use VARIANT (taking ARM=${ARM})" >&2
  VARIANT="${ARM}"
fi
VARIANT="${VARIANT:-baseline}"                       # baseline | baseline_finetuned | tactile
TASK="${TASK:-insert_hole}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
EPISODES="${EPISODES:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
SEED_OFFSET="${SEED_OFFSET:-0}"

# Repo + environment paths.
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
UNIVTAC_ROOT="${UNIVTAC_ROOT:?set UNIVTAC_ROOT to the UniVTAC checkout}"
GROOT_PYTHON="${GROOT_PYTHON:?set GROOT_PYTHON to the GR00T interpreter, e.g. Isaac-GR00T/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:?set UNIVTAC_PYTHON to the UniVTAC/Isaac Lab interpreter}"

# Checkpoint + embodiment tag. The baseline runs zero-shot on the released
# model; the tactile variant needs a NEW_EMBODIMENT finetune (docs/ABLATION.md).
if [[ "${VARIANT}" == "baseline" ]]; then
  GROOT_MODEL="${GROOT_MODEL:-nvidia/GR00T-N1.7-3B}"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT}"
  MODALITY_CONFIG="${MODALITY_CONFIG:-}"
elif [[ "${VARIANT}" == "tactile" ]]; then
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
OUTPUT="${RESULTS_DIR}/${VARIANT}/${TASK}/seed${SEED_OFFSET}-${RUN_TAG}.jsonl"
SERVER_LOG="${RESULTS_DIR}/${VARIANT}/${TASK}/server-${RUN_TAG}.log"

mkdir -p logs "$(dirname "${OUTPUT}")"

echo "=========================================================="
echo " variant=${VARIANT} task=${TASK} config=${TASK_CONFIG}"
echo " model=${GROOT_MODEL}  tag=${EMBODIMENT_TAG}"
echo " episodes=${EPISODES} execution_horizon=${EXECUTION_HORIZON}"
echo " gpus=${GPU_COUNT}  server=${SERVER_DEVICE}  sim=${SIM_DEVICE}"
echo " port=${PORT}  output=${OUTPUT}"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "=========================================================="

# Keep HF/torch caches on shared scratch: login-node home quotas are tight and
# every rank would otherwise re-download the 3B checkpoint.
export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp}/hf_cache}"

# hf_transfer roughly triples download throughput on the 15 GB of weights, but
# HF_HUB_ENABLE_HF_TRANSFER=1 without the package installed is a hard
# ValueError from huggingface_hub, not a fallback -- and it surfaces mid-download
# inside the server log, looking like a checkpoint problem. So probe for it
# rather than assuming. An explicit setting from the caller always wins.
if [[ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]]; then
  if "${GROOT_PYTHON}" -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
  else
    export HF_HUB_ENABLE_HF_TRANSFER=0
    echo "[job] hf_transfer not installed in ${GROOT_PYTHON}; using the standard"
    echo "[job] downloader. For a faster first run: ${GROOT_PYTHON} -m pip install hf_transfer"
  fi
fi

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
# Cap thread pools. This cluster rejects --cpus-per-task, so the job gets the
# node's full core count, and BLAS defaulting to all of them makes Isaac Sim and
# the inference server fight each other. Two small pools beat one greedy one.
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
# --------------------------------------------------------------------------- #
# cuDNN, checked before anything expensive happens
# --------------------------------------------------------------------------- #
# There is deliberately NO option here to run without cuDNN. A mismatched cuDNN
# reports CUDNN_STATUS_NOT_INITIALIZED, which reads like a too-old driver, and
# the obvious workaround -- stop using cuDNN -- costs ~86x on GR00T's vision
# tower with no error message at all: a step goes from 1.89 s to 170 s and the
# job merely looks slow. So the only supported response is to fix the pin, and
# this guard states the fix.
#
# It fails ONLY on a definite mismatch (wrong version loaded, cannot dlopen, or
# a real convolution fails). Anything it cannot determine is a warning, so it
# never blocks a job it cannot actually indict. SKIP_CUDNN_CHECK=1 skips the
# check itself, for the case where the check is what is broken -- it does not
# change how the job runs.
if [[ "${SKIP_CUDNN_CHECK:-0}" != "1" ]]; then
  if ! "${GROOT_PYTHON}" "${REPO_ROOT}/scripts/check_cudnn.py"; then
    echo "error: aborting before spending GPU time. Fix cuDNN as printed above," >&2
    echo "       or set SKIP_CUDNN_CHECK=1 if this check is itself broken." >&2
    exit 1
  fi
else
  echo "[job] cuDNN check skipped (SKIP_CUDNN_CHECK=1)"
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
  --variant "${VARIANT}"
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
if [[ "${VARIANT}" == "tactile" ]]; then
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
