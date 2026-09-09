#!/bin/bash
# One task's stages: finetune each variant, evaluate each, compare. Resumable --
# each stage records a marker in STAGE_DIR and is skipped if already done, and a
# failed stage is recorded and stepped over rather than ending the run.
#
# In practice only STAGES=finetune is submitted, by slurm/submit_benchmark.sh;
# evaluation is slurm/eval_checkpoint.sh, one checkpoint per job. The eval and
# compare stages still run for a local invocation or a smoke test.
#
# Check it without spending GPU time:
#
#   bash -n slurm/benchmark_task.sbatch            # syntax only
#   DRY_RUN=1 bash slurm/benchmark_task.sbatch     # paths, datasets, plan
#
# Site rules (job name >50 chars, --wckey, no --cpus-per-task or --mem) and the
# options the launcher owns: CLAUDE.md, "Cluster rules". MODEL_OUTPUT_DIR is
# injected and equals <bundle>/code-output; never set it.
#
# 1 GPU, deliberately: --num-gpus > 1 makes launch_finetune.py wrap the model in
# nn.DataParallel, which dies with "module must have its parameters ... on
# device: cpu".
#SBATCH --gres=gpu:1
set -uo pipefail



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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TASK="${TASK:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-clean}"
# 100 test rollouts, matching the paper's evaluation protocol.
EPISODES="${EPISODES:-100}"
# Execute 16 of each 40-action chunk, then re-plan from a fresh observation.
# Valid because the contract is 1 <= execution_horizon <= action_horizon.
# ~19 policy calls per episode at step_lim=300. Must be identical across
# variants: it sets how closed-loop the controller is.
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
# ARM= was the old name for this. Honour it rather than ignoring it: an
# unrecognised environment variable fails *silently*, so a stale
# `ARM=baseline_finetuned` command would quietly run the tactile variant
# instead -- a wrong result that looks like a right one. (A stale --arm flag
# fails loudly on the CLI, so no alias is needed there.)
if [[ -n "${ARMS:-}" && -z "${VARIANTS:-}" ]]; then
  echo "[job] note: ARMS is deprecated, use VARIANTS (taking ARMS=${ARMS})" >&2
  VARIANTS="${ARMS}"
fi
VARIANTS="${VARIANTS:-tactile baseline_finetuned}"
# Accept ':' and ',' as separators as well as spaces. A value crossing the
# submitters' `env NAME=VALUE` list is easiest to keep whole when it contains no
# spaces at all, and a comma still reads as a list separator in too many places
# to be safe. So the submitters send colons and this turns them back into words.
VARIANTS="${VARIANTS//[,:]/ }"

# Which stages to run. submit_benchmark.sh sends STAGES=finetune; the default
# is all three, for a local run. Unknown names are rejected, since a typo'd
# STAGES=finetne would otherwise run nothing and look like success.
STAGES="${STAGES:-finetune eval compare}"
STAGES="${STAGES//[,:]/ }"   # same reason as VARIANTS above
for _stage in ${STAGES}; do
  case "${_stage}" in
    finetune|eval|compare) ;;
    *) echo "error: unknown stage '${_stage}' in STAGES='${STAGES}'." >&2
       echo "       Valid stages: finetune eval compare" >&2
       exit 2 ;;
  esac
done
stage_enabled() {
  local want="$1" have
  for have in ${STAGES}; do [[ "${have}" == "${want}" ]] && return 0; done
  return 1
}

# 30,000 steps at batch 64, retaining exactly checkpoint-10000, -20000 and
# -30000; how one of the three becomes the reported number is in
# docs/BENCHMARK.md, "Choosing the step count".
#
# SAVE_TOTAL_LIMIT is a ceiling, not a target: 4 leaves a spare slot so an extra
# end-of-training save cannot evict checkpoint-10000. Three at ~26 GB each.
# The cost of sparse saves is resume granularity -- a kill rolls back to the
# last 10k boundary, ~1.9 h of redone work.
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"

# Unset, so launch_finetune.py's own defaults apply. The recipe pin below
# records whatever was used.
LEARNING_RATE="${LEARNING_RATE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
HP_ARGS=""
[[ -n "${LEARNING_RATE}" ]] && HP_ARGS+=" --learning-rate ${LEARNING_RATE}"
[[ -n "${WEIGHT_DECAY}" ]] && HP_ARGS+=" --weight-decay ${WEIGHT_DECAY}"

DATA_ROOT="${DATA_ROOT:-${HOME}/jaewon/workspace/groot-data}"
# The launcher injects MODEL_OUTPUT_DIR = <bundle>/code-output, a fresh
# directory per submission. So a resubmission does not see the previous run's
# checkpoints: pass the previous physical checkpoint directory as
# `bundle-sbatch --checkpoint`, which arrives here as CHECKPOINT_DIR. Bundle
# storage is retention-managed and nothing copies checkpoints out, so evaluate
# a checkpoint while its bundle is live.
: "${MODEL_OUTPUT_DIR:?MODEL_OUTPUT_DIR is injected by bundle-sbatch. Submit through slurm/submit_benchmark.sh, not with plain sbatch}"
CKPT_ROOT="${CKPT_ROOT:-${MODEL_OUTPUT_DIR}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/eval_result}"
# Stage markers must OUTLIVE the bundle: they are what makes a resubmission skip
# work that is already done, and a per-submission bundle would hand every run a
# clean slate. Keep them next to the repo unless told otherwise.
STAGE_DIR="${STAGE_DIR:-${REPO_ROOT}/.stages}"

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

# Default OFF. This was 1 for a while on the belief that cuDNN could not
# initialise on this cluster's driver; the real cause was a cuDNN in the GR00T

# SLURM re-indexes allocated GPUs to 0..N-1 inside the job, so counting
# CUDA_VISIBLE_DEVICES gives the usable width for data-parallel training.
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-0}")}"

mkdir -p logs "${STAGE_DIR}" "${CKPT_ROOT}" "${RESULTS_DIR}"

STAGE_LOG_DIR="${REPO_ROOT}/logs/benchmark-${TASK}-${SLURM_JOB_ID:-$$}"
mkdir -p "${STAGE_LOG_DIR}"

declare -a SUMMARY=()
# Both variants are evaluated in the same job, so they would otherwise derive the
# same port from SLURM_JOB_ID. The first server is killed by its own EXIT trap,
# but a socket in TIME_WAIT would still make the second bind fail, so hand each
# variant its own port.
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
echo " task=${TASK} config=${TASK_CONFIG} variants='${VARIANTS}' episodes=${EPISODES}"
echo " stages='${STAGES}'"
echo " gpus=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo " data=${DATA_ROOT}"
echo " ckpt=${CKPT_ROOT}"
echo " results=${RESULTS_DIR}"
echo " stage logs=${STAGE_LOG_DIR}"
if [[ "${USE_WANDB:-1}" != "1" ]]; then
  echo " wandb=disabled (USE_WANDB=0) -- stdout logs are the only training record"
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo " wandb=online project=${WANDB_PROJECT:-univtac-groot-ablation}"
else
  echo " wandb=offline (no WANDB_API_KEY) -- runs recorded under each OUTPUT_DIR/wandb,"
  echo "       upload later with: wandb sync <dir>/offline-run-*"
fi

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
# Probe actual ACCESS to the gated nvidia/Cosmos-Reason2-2B, not just the
# presence of a token: a token without the grant looks identical until the
# request is made, and the 401 lands a minute in, after the job holds a GPU.
#
# Not `${HF_TOKEN:+set}${HF_TOKEN:-UNSET}`: when the variable is set, `:-`
# yields its VALUE, so that one-liner prints the token into slurm.err.
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_STATE="set"
else
  HF_TOKEN_STATE="UNSET"
fi
if ! "${GROOT_PYTHON}" - <<'PYCHECK' 2>/dev/null
import sys
try:
    from huggingface_hub import HfApi
    HfApi().model_info("nvidia/Cosmos-Reason2-2B")
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}"[:300], file=sys.stderr)
    raise SystemExit(1)
PYCHECK
then
  fatal "cannot read the gated nvidia/Cosmos-Reason2-2B (needed by every GR00T
       checkpoint). Export a token with access, in the shell you submit from:
         export HF_TOKEN=hf_...
       The job inherits it from the submitting shell. Request access at
       https://huggingface.co/nvidia/Cosmos-Reason2-2B if you have none.
       HF_TOKEN is currently ${HF_TOKEN_STATE}; HF_HOME=${HF_HOME}."
fi

# Same lesson as hf_transfer: do not pass a flag whose package may be absent.
# --use-wandb with no wandb installed would fail after the weights had loaded.
# Only the finetune stage passes --use-wandb, so an eval-only job must not be
# blocked by a missing wandb.
if stage_enabled finetune && [[ "${USE_WANDB:-1}" == "1" ]]; then
  "${GROOT_PYTHON}" -c "import wandb" 2>/dev/null \
    || fatal "USE_WANDB=1 but ${GROOT_PYTHON} cannot import wandb. Install it in
       that venv, or pass USE_WANDB=0 to rely on the stdout logs alone."
fi

# A missing or truncated dataset is the classic way a long unattended run wastes a
# night, so verify shape now rather than discovering it at step 0 of training.
for variant in ${VARIANTS}; do
  # Evaluation never opens the training dataset -- it talks to the policy server
  # and Isaac Sim -- so an eval-only job must not fail on one that has since been
  # archived off NFS by the 4-day retention.
  stage_enabled finetune || continue
  dataset="${DATA_ROOT}/univtac-${TASK}-${variant}"
  if [[ ! -d "${dataset}" ]]; then
    fatal "no dataset for variant '${variant}' at ${dataset}"
    continue
  fi
  for required in meta/info.json meta/modality.json meta/stats.json; do
    [[ -f "${dataset}/${required}" ]] || fatal "${dataset} is missing ${required}"
  done
  n_parquet=$(find "${dataset}/data" -name '*.parquet' 2>/dev/null | wc -l)
  n_videos=$(find "${dataset}/videos" -name '*.mp4' 2>/dev/null | wc -l)
  echo " ${variant}: ${n_parquet} parquet, ${n_videos} mp4  (${dataset})"
  [[ "${n_parquet}" -gt 0 ]] || fatal "${dataset} has no parquet shards"
done

[[ -f "${REPO_ROOT}/configs/modality/univtac_tactile_config.py" ]] \
  || fatal "missing configs/modality/univtac_tactile_config.py"
[[ -f "${REPO_ROOT}/configs/modality/univtac_baseline_config.py" ]] \
  || fatal "missing configs/modality/univtac_baseline_config.py"

# Record the recipe on the first run for a task and refuse a later run that
# differs, since --num-gpus multiplies the effective batch size and a
# resubmission may be handed a different allocation. ALLOW_RECIPE_CHANGE=1
# overrides.
RECIPE_FILE="${STAGE_DIR}/recipe-${TASK}.env"
# The pin exists to keep the two variants TRAINED alike; an eval-only job has no
# recipe, and checking one would fail it purely for holding a different GPU count.
if ! stage_enabled finetune; then
  echo " recipe: not checked (this job does not train)"
elif [[ -f "${RECIPE_FILE}" ]]; then
  RECORDED_NUM_GPUS=""; RECORDED_MAX_STEPS=""
  RECORDED_LEARNING_RATE=""; RECORDED_WEIGHT_DECAY=""
  # shellcheck source=/dev/null
  source "${RECIPE_FILE}"
  if [[ "${RECORDED_NUM_GPUS}" != "${NUM_GPUS}" \
     || "${RECORDED_MAX_STEPS}" != "${MAX_STEPS}" \
     || "${RECORDED_LEARNING_RATE}" != "${LEARNING_RATE:-groot-default}" \
     || "${RECORDED_WEIGHT_DECAY}" != "${WEIGHT_DECAY:-groot-default}" ]]; then
    if [[ "${ALLOW_RECIPE_CHANGE:-0}" == "1" ]]; then
      echo " WARNING: recipe changed (was ${RECORDED_NUM_GPUS} gpu / ${RECORDED_MAX_STEPS} steps," >&2
      echo " now ${NUM_GPUS} gpu / ${MAX_STEPS} steps). Proceeding because" >&2
      echo " ALLOW_RECIPE_CHANGE=1 -- the variants are no longer directly comparable." >&2
    else
      fatal "recipe mismatch: this task was started with ${RECORDED_NUM_GPUS} GPU(s),
       --max-steps ${RECORDED_MAX_STEPS}, lr=${RECORDED_LEARNING_RATE},
       wd=${RECORDED_WEIGHT_DECAY}; this run has ${NUM_GPUS} GPU(s),
       --max-steps ${MAX_STEPS}, lr=${LEARNING_RATE:-groot-default},
       wd=${WEIGHT_DECAY:-groot-default}. Training the two variants under different
       effective batch sizes or optimiser settings would confound the ablation.
       Either resubmit with
       --gres=gpu:${RECORDED_NUM_GPUS} and MAX_STEPS=${RECORDED_MAX_STEPS}, or pass
       ALLOW_RECIPE_CHANGE=1 and delete ${STAGE_DIR} to restart the experiment."
    fi
  else
    echo " recipe: ${NUM_GPUS} gpu, ${MAX_STEPS} steps (matches ${RECIPE_FILE})"
  fi
elif [[ "${DRY_RUN:-0}" != "1" ]]; then
  printf 'RECORDED_NUM_GPUS=%s\nRECORDED_MAX_STEPS=%s\nRECORDED_LEARNING_RATE=%s\nRECORDED_WEIGHT_DECAY=%s\n' \
    "${NUM_GPUS}" "${MAX_STEPS}" "${LEARNING_RATE:-groot-default}" "${WEIGHT_DECAY:-groot-default}" \
    >"${RECIPE_FILE}"
  echo " recipe: ${NUM_GPUS} gpu, ${MAX_STEPS} steps, lr=${LEARNING_RATE:-groot-default},"
  echo "         wd=${WEIGHT_DECAY:-groot-default} (recorded in ${RECIPE_FILE})"
fi

df -h "${CKPT_ROOT}" 2>/dev/null || true
# shellcheck disable=SC2086 - word splitting is the point here
N_VARIANTS=$(set -- ${VARIANTS}; echo $#)
# How many checkpoints actually survive: one per SAVE_STEPS, capped by the
# retention ceiling. ~26 GB each, measured on the insert_hole checkpoints
# (25,547,974,206 bytes); the ~40 GB this used to assume was high.
N_CKPT=$(( SAVE_STEPS > 0 ? MAX_STEPS / SAVE_STEPS : 1 ))
(( N_CKPT < 1 )) && N_CKPT=1
(( N_CKPT > SAVE_TOTAL_LIMIT )) && N_CKPT="${SAVE_TOTAL_LIMIT}"
echo " the space above must hold roughly $((N_VARIANTS * N_CKPT * 26)) GB"
echo " (${N_VARIANTS} variants x ${N_CKPT} retained checkpoints x ~26 GB each)"

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
# 2. Finetune each variant
# --------------------------------------------------------------------------- #
for variant in ${VARIANTS}; do
  if ! stage_enabled finetune; then
    record "finetune ${variant}: SKIPPED (not in STAGES='${STAGES}')"
    continue
  fi
  out_dir="${CKPT_ROOT}/${TASK}-${variant}"
  marker="${STAGE_DIR}/finetune-${TASK}-${variant}.done"
  log="${STAGE_LOG_DIR}/finetune-${variant}.log"

  if [[ -f "${marker}" ]]; then
    record "finetune ${variant}: SKIPPED (already done: ${marker})"
    continue
  fi

  # An unmarked checkpoint is a killed run: resume it rather than restarting
  # from step 0. --resume-from-checkpoint defaults to off so an accidental
  # rerun cannot merge two experiments, so it has to be asked for.
  resume_args=""
  existing=$(latest_checkpoint "${out_dir}")
  # out_dir is fresh on every submission, so a resume comes from CHECKPOINT_DIR
  # instead. Link it in under its own name and --resume-from-checkpoint finds
  # it, since that flag takes the highest-numbered checkpoint-N in output_dir;
  # a link rather than a copy, at ~26 GB each. Only the variant whose name the
  # checkpoint carries may claim it, or two variants in one job would seed from
  # each other's weights.
  if [[ -z "${existing}" && -n "${CHECKPOINT_DIR:-}" ]]; then
    case "$(basename "${CHECKPOINT_DIR}")" in
      checkpoint-[0-9]*)
        if [[ "${CHECKPOINT_DIR}" == *"${TASK}-${variant}"* ]]; then
          mkdir -p "${out_dir}"
          ln -sfn "${CHECKPOINT_DIR}" "${out_dir}/$(basename "${CHECKPOINT_DIR}")"
          existing=$(latest_checkpoint "${out_dir}")
          echo "[stage] seeded ${out_dir} from CHECKPOINT_DIR=${CHECKPOINT_DIR}"
        else
          echo "[stage] CHECKPOINT_DIR=${CHECKPOINT_DIR} does not name" >&2
          echo "[stage] ${TASK}-${variant}; not seeding this variant with it" >&2
        fi
        ;;
      *)
        echo "[stage] CHECKPOINT_DIR=${CHECKPOINT_DIR} is not a checkpoint-N" >&2
        echo "[stage] directory; ignoring it for resume" >&2
        ;;
    esac
  fi
  if [[ -n "${existing}" ]]; then
    resume_args="--resume-from-checkpoint"
    echo "[stage] resuming ${variant} from ${existing}"
  fi

  banner "finetune ${variant}  (${NUM_GPUS} gpu, max_steps=${MAX_STEPS})"
  echo "[stage] log -> ${log}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would run slurm/finetune.sbatch with:"
    echo "[dry-run]   VARIANT=${variant}"
    echo "[dry-run]   DATASET=${DATA_ROOT}/univtac-${TASK}-${variant}"
    echo "[dry-run]   OUTPUT_DIR=${out_dir}"
    echo "[dry-run]   NUM_GPUS=${NUM_GPUS}"
    echo "[dry-run]   EXTRA_ARGS=--max-steps ${MAX_STEPS} --save-steps ${SAVE_STEPS} --save-total-limit ${SAVE_TOTAL_LIMIT}${HP_ARGS} ${resume_args}"
    record "finetune ${variant}: DRY-RUN (not executed)"
    continue
  fi

  GROOT_ROOT="${GROOT_ROOT}" \
  GROOT_PYTHON="${GROOT_PYTHON}" \
  VARIANT="${variant}" \
  TASK="${TASK}" \
  DATASET="${DATA_ROOT}/univtac-${TASK}-${variant}" \
  OUTPUT_DIR="${out_dir}" \
  NUM_GPUS="${NUM_GPUS}" \
  HF_HOME="${HF_HOME}" \
  USE_WANDB="${USE_WANDB:-1}" \
  WANDB_PROJECT="${WANDB_PROJECT:-univtac-groot-ablation}" \
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-univtac-${TASK}-${SLURM_JOB_ID:-local}}" \
  WANDB_API_KEY="${WANDB_API_KEY:-}" \
  WANDB_MODE="${WANDB_MODE:-}" \
  EXTRA_ARGS="--max-steps ${MAX_STEPS} --save-steps ${SAVE_STEPS} --save-total-limit ${SAVE_TOTAL_LIMIT}${HP_ARGS} ${resume_args} ${EXTRA_ARGS:-}" \
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
    record "finetune ${variant}: OK -> ${ckpt}"
  else
    record "finetune ${variant}: FAILED (exit ${status}, checkpoint='${ckpt:-none}') see ${log}"
  fi
done

# --------------------------------------------------------------------------- #
# 3. Evaluate each variant that produced a checkpoint
# --------------------------------------------------------------------------- #
for variant in ${VARIANTS}; do
  if ! stage_enabled eval; then
    record "eval ${variant}: SKIPPED (not in STAGES='${STAGES}')"
    continue
  fi
  out_dir="${CKPT_ROOT}/${TASK}-${variant}"
  # Prefer the stable `final` symlink the finetune stage writes; fall back to
  # probing, so an externally trained output_dir still evaluates.
  if [[ -e "${out_dir}/final" ]]; then
    ckpt="${out_dir}/final"
  else
    ckpt=$(latest_checkpoint "${out_dir}")
  fi
  log="${STAGE_LOG_DIR}/eval-${variant}.log"
  summary="${RESULTS_DIR}/${variant}/${TASK}/seed0-0.summary.json"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would run slurm/eval_ablation.sbatch with:"
    echo "[dry-run]   VARIANT=${variant} TASK=${TASK} TASK_CONFIG=${TASK_CONFIG}"
    echo "[dry-run]   GROOT_MODEL=${ckpt:-<from the finetune stage>}"
    echo "[dry-run]   EPISODES=${EPISODES} EXECUTION_HORIZON=${EXECUTION_HORIZON}"
    echo "[dry-run]   PORT=$((PORT_BASE + PORT_OFFSET + 1))"
    record "eval ${variant}: DRY-RUN (not executed)"
    PORT_OFFSET=$((PORT_OFFSET + 1))
    continue
  fi

  if [[ -z "${ckpt}" ]]; then
    record "eval ${variant}: SKIPPED (no checkpoint to evaluate)"
    continue
  fi
  if [[ -f "${summary}" ]]; then
    record "eval ${variant}: SKIPPED (summary exists: ${summary})"
    continue
  fi

  PORT_OFFSET=$((PORT_OFFSET + 1))
  banner "evaluate ${variant}  ckpt=${ckpt}  port=$((PORT_BASE + PORT_OFFSET))"
  echo "[stage] log -> ${log}"
  PORT=$((PORT_BASE + PORT_OFFSET)) \
  UNIVTAC_ROOT="${UNIVTAC_ROOT}" \
  UNIVTAC_PYTHON="${UNIVTAC_PYTHON}" \
  GROOT_PYTHON="${GROOT_PYTHON}" \
  VARIANT="${variant}" \
  TASK="${TASK}" \
  TASK_CONFIG="${TASK_CONFIG}" \
  EPISODES="${EPISODES}" \
  EXECUTION_HORIZON="${EXECUTION_HORIZON}" \
  GROOT_MODEL="${ckpt}" \
  RESULTS_DIR="${RESULTS_DIR}" \
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
    record "eval ${variant}: exit ${status} -- ${scored}"
  else
    record "eval ${variant}: FAILED (exit ${status}, no summary) see ${log}"
  fi
done

# --------------------------------------------------------------------------- #
# 4. Compare
# --------------------------------------------------------------------------- #
banner "compare"
# Even in a dry run, check the aggregator parses and its flags exist -- a typo
# here would otherwise only surface after both finetunes and both evals.
if ! stage_enabled compare; then
  record "compare: SKIPPED (not in STAGES='${STAGES}')"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
  if "${UNIVTAC_PYTHON}" "${REPO_ROOT}/scripts/compare_ablation.py" --help >/dev/null 2>&1; then
    record "compare: DRY-RUN (script imports and --help works)"
  else
    record "compare: DRY-RUN FAILED -- compare_ablation.py --help errored"
  fi
else
  "${UNIVTAC_PYTHON}" "${REPO_ROOT}/scripts/compare_ablation.py" \
    --results-dir "${RESULTS_DIR}" \
    --baseline-variant baseline_finetuned \
    --tactile-variant tactile \
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
