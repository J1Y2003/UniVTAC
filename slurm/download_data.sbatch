#!/bin/bash
# Download UniVTAC demonstration data from ModelScope, targeted.
#
# The full dataset is ~400 GB. You almost certainly want a subset:
#
#   byml2024/UniVTAC
#     isaac45/<task>/hdf5/{0..99}.hdf5   ~24 GB/task, 8 tasks  ~190 GB  <- Isaac Sim 4.5 (what UniVTAC installs)
#     isaac51/<task>/...                 ~190 GB                        <- Isaac Sim 5.1, NOT needed
#     contact/<shape>/...                                               <- pretraining data for UniVTAC's own encoder
#     checkpoints/<task>/, encoder.pth                                  <- their baseline policies
#
# Only `isaac45` is needed to finetune GR00T. Default here is ONE task, so you
# can validate convert -> finetune -> eval before committing ~190 GB.
#
#   sbatch --export=ALL,TASKS=lift_bottle           slurm/download_data.sbatch
#   sbatch --export=ALL,TASKS="lift_bottle lift_can" slurm/download_data.sbatch
#   sbatch --export=ALL,TASKS=all                   slurm/download_data.sbatch   # all 8, ~190 GB
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

DATASET="${DATASET:-byml2024/UniVTAC}"
ISAAC="${ISAAC:-isaac45}"            # isaac45 matches UniVTAC's Isaac Sim 4.5 install
TASKS="${TASKS:-lift_bottle}"
DEST="${DEST:-${DATA_ROOT:-${SCRATCH:-/tmp}}/univtac-raw}"
PYTHON="${CONVERT_PYTHON:-${UNIVTAC_PYTHON:-python}}"

ALL_TASKS="lift_bottle lift_can insert_HDMI insert_hole insert_tube pull_out_key put_bottle_in_shelf grasp_classify"
[[ "${TASKS}" == "all" ]] && TASKS="${ALL_TASKS}"

mkdir -p logs "${DEST}"

echo "=========================================================="
echo " dataset=${DATASET}  variant=${ISAAC}"
echo " tasks=${TASKS}"
echo " dest=${DEST}"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "=========================================================="
df -h "${DEST}" || true

command -v modelscope >/dev/null 2>&1 || "${PYTHON}" -m pip install --quiet modelscope

# One invocation per task keeps a failure scoped and the log readable. Roughly
# 24 GB each; check `df -h` above before requesting many.
STATUS=0
for task in ${TASKS}; do
  echo "----------------------------------------------------------"
  echo "[job] downloading ${ISAAC}/${task}"
  if ! modelscope download --dataset "${DATASET}" \
        --include "${ISAAC}/${task}/*" \
        --local_dir "${DEST}"; then
    echo "[job] FAILED on ${task}" >&2
    echo "[job] if --include is unsupported by this modelscope version, check:" >&2
    echo "[job]   modelscope download --help" >&2
    STATUS=1
    continue
  fi
  count=$(find "${DEST}/${ISAAC}/${task}" -name '*.hdf5' 2>/dev/null | wc -l)
  echo "[job] ${task}: ${count} episodes on disk"
done

echo "----------------------------------------------------------"
du -sh "${DEST}" 2>/dev/null || true
echo "[job] next: convert each task to the GR00T LeRobot format, e.g."
for task in ${TASKS}; do
  echo "  sbatch --export=ALL,TASK=${task},ARM=tactile,RAW_DIR=${DEST}/${ISAAC}/${task} slurm/convert.sbatch"
done
exit ${STATUS}
