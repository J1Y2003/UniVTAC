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
#   sbatch --export=ALL,TASKS=lift_bottle            slurm/download_data.sbatch
#   sbatch --export=ALL,TASKS="lift_bottle lift_can" slurm/download_data.sbatch
#   sbatch --export=ALL,TASKS=all                    slurm/download_data.sbatch   # all 8, ~190 GB
#   sbatch --export=ALL,TASKS=none,CHECKPOINT=1      slurm/download_data.sbatch   # encoder only, ~minutes
#
# No GPU. Network + disk bound. modelscope resumes, so a walltime kill is safe
# to resubmit.
#
#SBATCH --job-name=univtac-download
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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

if ! command -v modelscope >/dev/null 2>&1; then
  echo "error: 'modelscope' is not on PATH in this environment." >&2
  echo "       download.sh requires it. Activate the env that has it, or:" >&2
  echo "         ${PYTHON} -m pip install --user modelscope" >&2
  echo "       Do NOT install into the shared (base) conda env." >&2
  exit 1
fi

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
  echo "  sbatch --export=ALL,TASK=${task},ARM=tactile,RAW_DIR=${OUTPUT}/isaac${VERSION}/${task}/hdf5 slurm/convert.sbatch"
done
exit ${STATUS}
