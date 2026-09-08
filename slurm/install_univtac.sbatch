#!/bin/bash
# Install UniVTAC (Isaac Sim + Isaac Lab + cuRobo + TacEx + libuipc) as a batch job.
#
# Needed on clusters where `srun` is restricted to a short debug partition, so a
# multi-hour interactive install is not possible.
#
#   sbatch --export=ALL,UNIVTAC_ROOT=$HOME/jaewon/workspace/UniVTAC-sim \
#          --wckey=project-short-name:sub_4dpdata \
#          slurm/install_univtac.sbatch
#
# UniVTAC's own scripts/install.sh guards every step with
# `if pip show <pkg>; then skip`, so this job is RESUMABLE: if it hits the
# walltime, just submit it again and it continues where it stopped.
#
# Three things in the upstream script do not survive a batch context; this
# wrapper works around all three without editing the repo (see PATCHES below).
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
#SBATCH --job-name=univtac-groot-install-univtac-isaac-lab-tacex-and-curobo-stack
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --wckey=project-short-name:sub_4dpdata

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


UNIVTAC_ROOT="${UNIVTAC_ROOT:?set UNIVTAC_ROOT to the UniVTAC-sim checkout}"
cd "${UNIVTAC_ROOT}"
mkdir -p logs

echo "=========================================================="
echo " UniVTAC install"
echo " root=${UNIVTAC_ROOT}"
echo " node=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-none}"
echo " job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_ON_NODE:-?}"
echo "=========================================================="

# --------------------------------------------------------------------------- #
# Conda must be initialised: install.sh exits if CONDA_PREFIX is unset, and it
# calls `conda activate`, which needs the shell hook. A batch shell is not a
# login shell, so source the hook explicitly.
# --------------------------------------------------------------------------- #
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
if [[ -z "${CONDA_BASE}" ]]; then
  echo "[job] conda not on PATH. Set CONDA_BASE=/path/to/miniconda3 and resubmit." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate base
echo "[job] conda base: ${CONDA_BASE}"

# --------------------------------------------------------------------------- #
# Prerequisite check. install.sh calls `sudo apt install` for these, which
# cannot work in a batch job (no TTY, and no sudo on a shared cluster). They are
# normally already present; if one is missing, load a module or install it via
# conda into base BEFORE resubmitting -- do not try to make sudo work.
# --------------------------------------------------------------------------- #
missing=()
for tool in cmake gcc make git git-lfs; do
  command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
done
if (( ${#missing[@]} )); then
  echo "[job] missing build tools: ${missing[*]}" >&2
  echo "[job] try:  module load cmake gcc   (or)   conda install -n base -c conda-forge ${missing[*]}" >&2
  exit 1
fi
echo "[job] build tools present: cmake gcc make git git-lfs"

# --------------------------------------------------------------------------- #
# vcpkg. Upstream install.sh has an inverted test at step 6:
#
#     if [ -d "Toolchain" ]; then   # relative to the repo, so normally false
#         mkdir ~/Toolchain; git clone .../vcpkg; ./bootstrap-vcpkg.sh
#     fi
#     export CMAKE_TOOLCHAIN_FILE="$HOME/Toolchain/vcpkg/scripts/..."
#
# so vcpkg is never cloned on a fresh machine, yet CMAKE_TOOLCHAIN_FILE points
# into it and the libuipc build fails. Do it here instead.
# --------------------------------------------------------------------------- #
VCPKG_DIR="${VCPKG_DIR:-$HOME/Toolchain/vcpkg}"
if [[ ! -f "${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake" ]]; then
  echo "[job] installing vcpkg into ${VCPKG_DIR}"
  mkdir -p "$(dirname "${VCPKG_DIR}")"
  [[ -d "${VCPKG_DIR}/.git" ]] || git clone https://github.com/microsoft/vcpkg.git "${VCPKG_DIR}"
  ( cd "${VCPKG_DIR}" && ./bootstrap-vcpkg.sh -disableMetrics )
else
  echo "[job] vcpkg already bootstrapped"
fi
export CMAKE_TOOLCHAIN_FILE="${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake"

# --------------------------------------------------------------------------- #
# PATCHES to a scratch copy of install.sh. The repo is left untouched.
#
#   1. `sudo apt install ...`  -> a no-op echo (see the check above).
#   2. `--livestream 2`        -> `--headless`. Livestream waits for a GUI
#                                 client to connect, which never happens in a
#                                 batch job; the TacEx smoke test would hang
#                                 until the walltime.
#   3. `python3 -m pytest .`   -> made non-fatal. install.sh runs `set -e`, so a
#      (cuRobo's test suite)     single environment-specific test failure would
#                                abort the whole install after cuRobo built.
# --------------------------------------------------------------------------- #
PATCHED="${UNIVTAC_ROOT}/.install_batch.sh"
sed -e 's|^\([[:space:]]*\)sudo apt install|\1echo "[patched] skipping sudo apt install"  #|' \
    -e 's|--livestream 2|--headless|g' \
    -e 's|^\([[:space:]]*\)python3 -m pytest \.|\1python3 -m pytest . \|\| echo "[patched] curobo tests failed (non-fatal)"|' \
    scripts/install.sh > "${PATCHED}"
chmod +x "${PATCHED}"

echo "[job] patched install script -> ${PATCHED}"
echo "[job] diff against upstream:"
diff scripts/install.sh "${PATCHED}" || true
echo "----------------------------------------------------------"

# Headless rendering for the smoke tests install.sh runs at the end.
export DISPLAY=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-8}}}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-8}}}"   # limits C++/CUDA build parallelism

bash "${PATCHED}"
STATUS=$?

echo "----------------------------------------------------------"
if [[ ${STATUS} -eq 0 ]]; then
  echo "[job] install finished. Verify, then continue with docs/RUNBOOK.md step 3:"
  echo "  conda activate UniVTAC && python -c 'import isaaclab, tacex; print(\"ok\")'"
else
  echo "[job] install exited ${STATUS}."
  echo "[job] This job is resumable -- every step is guarded by a 'pip show' check,"
  echo "[job] so resubmitting continues from where it stopped:"
  echo "  sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,UNIVTAC_ROOT=${UNIVTAC_ROOT} slurm/install_univtac.sbatch"
fi
exit ${STATUS}
