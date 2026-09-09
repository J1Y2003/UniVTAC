#!/bin/bash
# Evaluate one checkpoint on one UniVTAC task, headless. Prefer
# slurm/eval_checkpoint.sh, which resolves the paths and files the result.
#
# Two processes in one job: the GR00T inference server (Python 3.12, holds the 3B
# checkpoint) and the UniVTAC evaluator (Python 3.10, Isaac Sim), talking over a
# loopback ZeroMQ socket because they cannot share a virtualenv.
#
#   env VARIANT=baseline_finetuned TASK=insert_hole UNIVTAC_JOB_CONFIG=1 \
#     bundle-sbatch --job-kind eval --checkpoint /physical/path/checkpoint-10000 \
#     --code-git-root "$REPO_ROOT" -- \
#     --job-name=univtac-groot-evaluate-one-ablation-variant-on-one-univtac-benchmark-task \
#     --wckey=project-short-name:sub_4dpdata --partition=background -- \
#     slurm/eval_ablation.sbatch
#
# --checkpoint must be a physical directory; the launcher rejects the `final`
# symlink a finetune leaves behind. Evaluation runs on `background`, never
# sjw_alinlab.
#
# Site rules (job name >50 chars, --wckey, no --cpus-per-task or --mem) and
# the options the launcher owns: CLAUDE.md, "Cluster rules". MODEL_OUTPUT_DIR
# is injected and equals <bundle>/code-output; never set it.
#SBATCH --gres=gpu:1
set -euo pipefail





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
# sbatch copies this script into /var/spool/slurm/d/, so BASH_SOURCE resolves to
# the spool and every derived path lands somewhere unwritable. SLURM_SUBMIT_DIR
# is the right answer under sbatch, BASH_SOURCE under plain `bash`; accept
# whichever actually contains this repo.
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

# Both runtime guards -- the wckey and the configuration sentinel -- live in one
# place. shellcheck source=slurm/common.sh
. "${REPO_ROOT}/slurm/common.sh"
univtac_job_guard
UNIVTAC_ROOT="${UNIVTAC_ROOT:?set UNIVTAC_ROOT to the UniVTAC checkout}"
GROOT_PYTHON="${GROOT_PYTHON:?set GROOT_PYTHON to the GR00T interpreter, e.g. Isaac-GR00T/.venv/bin/python}"
UNIVTAC_PYTHON="${UNIVTAC_PYTHON:?set UNIVTAC_PYTHON to the UniVTAC/Isaac Lab interpreter}"

# The baseline runs zero-shot on the released model; a finetune needs
# NEW_EMBODIMENT. GROOT_MODEL defaults to the CHECKPOINT_DIR bundle-sbatch
# injected, i.e. the path the launcher validated, and an explicit value wins.
GROOT_MODEL="${GROOT_MODEL:-${CHECKPOINT_DIR:-}}"
if [[ "${VARIANT}" == "baseline" ]]; then
  GROOT_MODEL="${GROOT_MODEL:-nvidia/GR00T-N1.7-3B}"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT}"
  MODALITY_CONFIG="${MODALITY_CONFIG:-}"
elif [[ "${VARIANT}" == "tactile" ]]; then
  GROOT_MODEL="${GROOT_MODEL:?tactile needs a finetuned checkpoint; set GROOT_MODEL or submit with bundle-sbatch --checkpoint}"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
  MODALITY_CONFIG="${MODALITY_CONFIG:-${REPO_ROOT}/configs/modality/univtac_tactile_config.py}"
else
  GROOT_MODEL="${GROOT_MODEL:?baseline_finetuned needs a checkpoint; set GROOT_MODEL or submit with bundle-sbatch --checkpoint}"
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
# Without a distinct tag every eval writes seed<offset>-0.jsonl and two runs
# collide; it must also be STABLE across a requeue, so the resume below finds
# the work already done. eval_checkpoint.sh sets RUN_TAG=ckpt<N>.
RUN_TAG="${RUN_TAG:-${SLURM_ARRAY_TASK_ID:-0}}"
OUTPUT="${RESULTS_DIR}/${VARIANT}/${TASK}/seed${SEED_OFFSET}-${RUN_TAG}.jsonl"
SERVER_LOG="${RESULTS_DIR}/${VARIANT}/${TASK}/server-${RUN_TAG}.log"

# --------------------------------------------------------------------------- #
# Resume, because `background` preempts with REQUEUE
# --------------------------------------------------------------------------- #
# A requeue re-runs this script from the top with the same OUTPUT path, and
# ResultWriter appends -- so without the arithmetic below a preemption would add
# a second pass from the first seed and inflate the tally. Read what is already
# recorded and continue past it: every episode is flushed as it completes and
# carries its seed, so the file is truthful even after a SIGKILL.
#
# Errored and skipped episodes consumed a seed but not the episode budget, so
# the two are counted separately and the resumed run targets the same number of
# SCORED episodes a clean run would.
EPISODES_TARGET="${EPISODES}"
EVAL_COMPLETE=0
BASE_SEED=$(( 1000000 * (1 + SEED_OFFSET) ))
if [[ -s "${OUTPUT}" ]]; then
  RESUME_RAW=$("${UNIVTAC_PYTHON}" - "${OUTPUT}" "${BASE_SEED}" <<'PYRESUME' 2>/dev/null || true
import json, sys
path, base = sys.argv[1], int(sys.argv[2])
max_seed, scored = base - 1, 0
seen = set()
with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue          # a kill mid-write leaves one truncated line
        seed = row.get("seed")
        if isinstance(seed, int):
            max_seed = max(max_seed, seed)
            if seed in seen:
                continue      # an earlier pass already counted this seed
            seen.add(seed)
        if not row.get("error") and not row.get("skipped"):
            scored += 1
print(f"{max_seed + 1} {scored}")
PYRESUME
)
  RESUME_RAW=$(printf '%s' "${RESUME_RAW}" | tr -d '
')
  if [[ "${RESUME_RAW}" =~ ^([0-9]+)[[:space:]]+([0-9]+)$ ]]; then
    RESUME_SEED="${BASH_REMATCH[1]}"
    RESUME_SCORED="${BASH_REMATCH[2]}"
    if (( RESUME_SCORED >= EPISODES_TARGET )); then
      echo "[job] ${OUTPUT} already holds ${RESUME_SCORED}/${EPISODES_TARGET}"
      echo "[job] scored episodes -- nothing left to run."
      EVAL_COMPLETE=1
    else
      START_SEED="${START_SEED:-${RESUME_SEED}}"
      EPISODES=$(( EPISODES_TARGET - RESUME_SCORED ))
      echo "[job] RESUMING: ${RESUME_SCORED}/${EPISODES_TARGET} already scored;"
      echo "[job] running ${EPISODES} more from seed ${START_SEED}."
    fi
  else
    # Refusing to guess: appending a fresh pass would corrupt the tally.
    echo "error: ${OUTPUT} exists but its resume state is unreadable." >&2
    echo "       Move it aside to start this checkpoint over." >&2
    exit 2
  fi
fi

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

# Roughly triples download throughput, but the flag without the package is a
# hard ValueError rather than a fallback, surfacing mid-download as what looks
# like a checkpoint problem. An explicit setting from the caller wins.
if [[ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]]; then
  if "${GROOT_PYTHON}" -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
  else
    export HF_HUB_ENABLE_HF_TRANSFER=0
    echo "[job] hf_transfer not installed in ${GROOT_PYTHON}; using the standard"
    echo "[job] downloader. For a faster first run: ${GROOT_PYTHON} -m pip install hf_transfer"
  fi
fi

# Credentials for the gated Cosmos-Reason2-2B. Redirecting HF_HOME also moves
# where a stored token is looked for, so exporting HF_TOKEN is the reliable
# route and takes precedence over any stored login.
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
# 3. The results-library entry, when one was asked for
# --------------------------------------------------------------------------- #
# EVAL_JSON is one self-describing file per (task, checkpoint, seed block), so a
# directory of them is a queryable record of every evaluation ever run. It is
# recomputed from the JSONL rather than copied from the run's own summary,
# because after a resume the in-memory summary covers only the final leg while
# the JSONL holds every episode.
emit_library_json() {
  [[ -n "${EVAL_JSON:-}" ]] || return 0
  mkdir -p "$(dirname "${EVAL_JSON}")"
  if "${UNIVTAC_PYTHON}" - <<PYJSON
import json, os, socket, subprocess, sys, time
sys.path.insert(0, "${REPO_ROOT}")
from univtac_groot.metrics import read_jsonl, summarize, wilson_interval

jsonl = "${OUTPUT}"
if not os.path.exists(jsonl):
    print("no JSONL to summarise: " + jsonl, file=sys.stderr)
    raise SystemExit(1)

def git_raw(*args):
    """(succeeded, stdout). Distinguishing the two matters: an empty result from
    `git status --porcelain` means a CLEAN tree, while a failed call means we do
    not know -- and recording "clean" for the second would be a lie."""
    try:
        proc = subprocess.run(("git", "-C", "${REPO_ROOT}") + args,
                              capture_output=True, text=True, timeout=10)
        return proc.returncode == 0, proc.stdout
    except Exception:
        return False, ""

def git(*args):
    ok, out = git_raw(*args)
    return out.strip() if ok and out.strip() else None

_dirty_ok, _dirty_out = git_raw("status", "--porcelain")

ckpt = "${GROOT_MODEL}"
step = os.path.basename(ckpt.rstrip("/"))
step = int(step.split("-")[-1]) if step.startswith("checkpoint-") and step.split("-")[-1].isdigit() else None

# Dedupe by seed, last write wins -- the same rule compare_ablation.py uses.
# A rerun that re-evaluated a seed would otherwise be counted twice.
rows = {r.seed: r for r in read_jsonl(jsonl)}
summary = summarize(rows.values())
lo, hi = wilson_interval(int(summary.get("successes", 0)),
                         int(summary.get("episodes_scored", 0)))
summary["success_rate_ci95"] = [round(lo, 6), round(hi, 6)]
summary["success_rate_ci95_pct"] = [round(lo * 100, 2), round(hi * 100, 2)]

record = {
    "task": "${TASK}",
    "task_config": "${TASK_CONFIG}",
    "variant": "${VARIANT}",
    "checkpoint": ckpt,
    "checkpoint_step": step,
    "seed_offset": int("${SEED_OFFSET}"),
    "first_seed": 1_000_000 * (1 + int("${SEED_OFFSET}")),
    "episodes_requested": int("${EPISODES_TARGET}"),
    "execution_horizon": int("${EXECUTION_HORIZON}"),
    "metrics": summary,
    "provenance": {
        "jsonl": jsonl,
        "server_log": "${SERVER_LOG}",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "bundle": os.environ.get("JOB_OUTPUT_BUNDLE_DIR"),
        "host": socket.gethostname(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(_dirty_out.strip()) if _dirty_ok else None,
        "evaluator_exit": ${STATUS:-0},
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    },
}
with open("${EVAL_JSON}", "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=2, ensure_ascii=False)
    fh.write("\\n")
print("SR={0}% ({1}/{2} scored), CI95 {3}".format(
    summary.get("success_rate_pct"), summary.get("successes"),
    summary.get("episodes_scored"), summary["success_rate_ci95_pct"]))
PYJSON
  then
    echo "[job] library entry: ${EVAL_JSON}"
  else
    echo "[job] WARNING: could not write ${EVAL_JSON}; the JSONL above is intact" >&2
  fi
}

# A requeue after the last episode has nothing left to run, but still owes the
# library its entry: the attempt that finished the episodes was killed before it
# could write one.
if (( EVAL_COMPLETE )); then
  emit_library_json
  echo "[job] nothing to evaluate; the library entry is up to date."
  exit 0
fi

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
# cuDNN, before anything expensive. A mismatched pin reads like a too-old driver
# (CUDNN_STATUS_NOT_INITIALIZED) and the obvious workaround -- disable cuDNN --
# costs ~86x on GR00T's vision tower with no error at all, so there is
# deliberately no way to run without it. See CLAUDE.md, "cuDNN". The check fails
# only on a definite mismatch and warns otherwise; SKIP_CUDNN_CHECK=1 skips the
# check, not the requirement.
if [[ "${SKIP_CUDNN_CHECK:-0}" != "1" ]]; then
  if ! "${GROOT_PYTHON}" "${REPO_ROOT}/scripts/check_cudnn.py"; then
    echo "error: aborting before spending GPU time. Fix cuDNN as printed above," >&2
    echo "       or set SKIP_CUDNN_CHECK=1 if this check is itself broken." >&2
    exit 1
  fi
else
  echo "[job] cuDNN check skipped (SKIP_CUDNN_CHECK=1)"
fi

# The server must not inherit UniVTAC's CUDA settings: its conda env exports
# CUDA_HOME and LD_LIBRARY_PATH for CUDA 12.4, and GR00T's torch is cu128 with
# its own bundled cuDNN. A leaked LD_LIBRARY_PATH surfaces as
# CUDNN_STATUS_NOT_INITIALIZED on the first get_action, long after the weights
# loaded, so it reads as a model bug. The evaluator keeps them; Isaac Sim needs
# them.
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
# START_SEED resumes a preempted run: run_eval.py starts at an absolute seed
# rather than deriving one from SEED_OFFSET, and ResultWriter appends to the
# existing JSONL. The resume block above computes it from what is already
# recorded there. Unset means "start at 1_000_000 * (1 + SEED_OFFSET)".
if [[ -n "${START_SEED:-}" ]]; then
  EVAL_CMD+=(--start-seed "${START_SEED}")
fi
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

emit_library_json

if [[ ${STATUS} -ne 0 ]]; then
  echo "[job] non-zero exit: read the SERVER log first -- a checkpoint that fails"
  echo "[job] to load shows up here only as a wait_until_ready timeout."
fi
exit ${STATUS}
