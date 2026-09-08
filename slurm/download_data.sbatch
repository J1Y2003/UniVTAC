#!/bin/bash
# Download UniVTAC demonstration data, targeted, via UniVTAC's own downloader.
#
# This delegates to $UNIVTAC_ROOT/data/download.sh rather than calling
# modelscope directly. That script is authoritative about include patterns,
# task-name validation and the isaac45/isaac51 split, and it lands files in the
# published layout that the rest of UniVTAC expects.
#
# The full dataset is ~400 GB. You almost certainly want a subset:
#
#   byml2024/UniVTAC
#     isaac45/<task>/hdf5/{0..99}.hdf5   ~24 GB/task, 8 tasks ~190 GB  <- Isaac Sim 4.5 (what UniVTAC installs)
#     isaac51/<task>/...                 ~190 GB                       <- Isaac Sim 5.1, NOT needed on `main`
#     contact/<shape>/...                                              <- pretraining data for UniVTAC's own encoder
#     checkpoints/<task>/, checkpoints/encoder.pth                     <- their baseline policies + tactile encoder
#
# Only isaac45 is needed to finetune GR00T. The default here is ONE task plus
# the checkpoints, so you can validate convert -> finetune -> eval before
# committing ~190 GB.
#
#   sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu --export=ALL,TASKS=lift_bottle            slurm/download_data.sbatch
#   sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu --export=ALL,TASKS="lift_bottle lift_can" slurm/download_data.sbatch
#   sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu --export=ALL,TASKS=all                    slurm/download_data.sbatch   # all 8, ~190 GB
#   sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu --export=ALL,TASKS=none,CHECKPOINT=1      slurm/download_data.sbatch   # encoder only, ~minutes
#
# No GPU. Network + disk bound. modelscope resumes, so a walltime kill is safe
# to resubmit.
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
#
# CPU-ONLY JOB: this must go to --partition=cpu. The GPU partitions reject any
# job that does not request a GPU:
#   ❌ ERROR: GPU 파티션에는 GPU를 요청한 잡만 제출할 수 있습니다.
#      CPU 전용 잡은 --partition=cpu 를 사용하세요.
# The header below sets it, but --partition precedence is
# command line > SBATCH_PARTITION > #SBATCH directive, so an exported
# SBATCH_PARTITION pointing at a GPU partition silently overrides it and the
# submission is refused. Pass --partition=cpu on the command line too.
#SBATCH --job-name=univtac-groot-download-univtac-demonstration-data-from-modelscope
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --wckey=project-short-name:sub_4dpdata
#SBATCH --partition=cpu

set -uo pipefail

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

TASKS="${TASKS:-lift_bottle}"        # space-separated, or "all", or "none"
VERSION="${VERSION:-45}"             # 45 matches Isaac Sim 4.5 on the main branch
CHECKPOINT="${CHECKPOINT:-1}"        # 1 = also fetch checkpoints/ (includes encoder.pth)
CONTACT="${CONTACT:-0}"              # 1 = also fetch the contact pretraining set
WORKERS="${WORKERS:-8}"
# Default to the checkout's own data/, which is where _base_data_preprocessor.py
# anchors DATA_ROOT_PATH. Overriding this means the ACT pipeline will not see it.
OUTPUT="${OUTPUT:-${UNIVTAC_ROOT}/data}"
# Collect-config name to expose the downloaded episodes under. Must match the
# second argument you later pass to process_data.sh / eval_policy.sh.
RAW_CONFIG="${RAW_CONFIG:-clean}"
PYTHON="${PYTHON:-python3}"

DOWNLOADER="${UNIVTAC_ROOT}/data/download.sh"
if [[ ! -f "${DOWNLOADER}" ]]; then
  echo "error: ${DOWNLOADER} not found." >&2
  echo "       Is UNIVTAC_ROOT correct, and is the checkout recent enough to" >&2
  echo "       ship data/download.sh? Older clones do not have it." >&2
  exit 1
fi

ALL_TASKS="lift_bottle lift_can insert_HDMI insert_hole insert_tube pull_out_key put_bottle_in_shelf grasp_classify"
[[ "${TASKS}" == "all" ]] && TASKS="${ALL_TASKS}"
[[ "${TASKS}" == "none" ]] && TASKS=""

mkdir -p logs "${OUTPUT}"

echo "=========================================================="
echo " univtac_root=${UNIVTAC_ROOT}"
echo " tasks=${TASKS:-<none>}  version=${VERSION}"
echo " checkpoint=${CHECKPOINT}  contact=${CONTACT}"
echo " output=${OUTPUT}  raw_config=${RAW_CONFIG}"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "=========================================================="
df -h "${OUTPUT}" || true

# download.sh does `require_command modelscope`, i.e. it needs the CLI on PATH.
# Downloading needs nothing from Isaac Lab, so rather than installing modelscope
# into the UniVTAC conda env (where it would re-pin requests/urllib3/tqdm under
# a working Isaac Lab + TacEx install) or into ~/.local (which every python3.10
# process on this account would then see), keep it in a throwaway venv. Delete
# DOWNLOAD_VENV any time; it is rebuilt on demand.
if ! command -v modelscope >/dev/null 2>&1; then
  DOWNLOAD_VENV="${DOWNLOAD_VENV:-${REPO_ROOT}/.venv-modelscope}"
  if [[ "${NO_AUTO_VENV:-0}" == "1" ]]; then
    echo "error: 'modelscope' is not on PATH and NO_AUTO_VENV=1." >&2
    echo "       Provide it yourself, or unset NO_AUTO_VENV to let this job" >&2
    echo "       build a private venv at ${DOWNLOAD_VENV}." >&2
    exit 1
  fi
  if [[ ! -x "${DOWNLOAD_VENV}/bin/modelscope" ]]; then
    echo "[job] 'modelscope' not found; building a private venv at ${DOWNLOAD_VENV}"
    echo "[job] (isolated on purpose -- nothing is installed into the UniVTAC"
    echo "[job]  conda env, ~/.local, or the shared base env)"
    "${PYTHON}" -m venv "${DOWNLOAD_VENV}" || {
      echo "error: could not create a venv with ${PYTHON}." >&2
      echo "       Set PYTHON to an interpreter that has the venv module." >&2
      exit 1
    }
    "${DOWNLOAD_VENV}/bin/python" -m pip install --quiet --upgrade pip
    "${DOWNLOAD_VENV}/bin/python" -m pip install --quiet modelscope || {
      echo "error: pip install modelscope failed -- see above." >&2
      exit 1
    }
  fi
  export PATH="${DOWNLOAD_VENV}/bin:${PATH}"
  echo "[job] using modelscope from ${DOWNLOAD_VENV}"
fi
echo "[job] modelscope: $(command -v modelscope)"

# Build the selector list. download.sh accepts repeated selectors, and refuses
# --version unless --task is among them, so only pass it when tasks are wanted.
ARGS=()
for task in ${TASKS}; do
  ARGS+=(--task "${task}")
done
if [[ -n "${TASKS}" ]]; then
  ARGS+=(--version "${VERSION}")
fi
[[ "${CHECKPOINT}" == "1" ]] && ARGS+=(--checkpoint)
[[ "${CONTACT}" == "1" ]] && ARGS+=(--contact)
if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "error: nothing selected. Set TASKS, or CHECKPOINT=1, or CONTACT=1." >&2
  exit 2
fi
ARGS+=(--output "${OUTPUT}" --workers "${WORKERS}")

echo "[job] running: bash ${DOWNLOADER} ${ARGS[*]}"
cd "${UNIVTAC_ROOT}"
bash "${DOWNLOADER}" "${ARGS[@]}"
STATUS=$?
echo "[job] download.sh exited ${STATUS}"

echo "----------------------------------------------------------"
for task in ${TASKS}; do
  count=$(find "${OUTPUT}/isaac${VERSION}/${task}" -name '*.hdf5' 2>/dev/null | wc -l)
  echo "[job] ${task}: ${count} episodes on disk (expect 100)"
done
du -sh "${OUTPUT}" 2>/dev/null || true

# download.sh preserves the published layout (isaac45/<task>/hdf5/), but
# BaseDataPreprocessor reads data/<task>/<config>/. Bridge the two with the
# symlinks fix_act.py manages -- idempotent, so re-running is free.
if [[ ${STATUS} -eq 0 && -n "${TASKS}" ]]; then
  echo "----------------------------------------------------------"
  echo "[job] linking downloaded episodes into data/<task>/${RAW_CONFIG}"
  "${PYTHON}" "${REPO_ROOT}/tools/univtac_patches/fix_act.py" \
    --univtac-root "${UNIVTAC_ROOT}" --raw-config "${RAW_CONFIG}" || true
fi

echo "----------------------------------------------------------"
echo "[job] next steps:"
for task in ${TASKS}; do
  echo "  # UniVTAC's own ACT baseline (its data step, then training):"
  echo "  bash ${UNIVTAC_ROOT}/policy/ACT/process_data.sh ${task} ${RAW_CONFIG} 100"
  echo "  # this repo's GR00T path:"
  echo "  sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu --export=ALL,TASK=${task},VARIANT=tactile,RAW_DIR=${OUTPUT}/isaac${VERSION}/${task}/hdf5 slurm/convert.sbatch"
done
exit ${STATUS}
