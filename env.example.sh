# Per-session environment for the GR00T x UniVTAC pipeline.
#
# Copy, edit, and source it at the start of each session:
#
#     cp env.example.sh env.sh
#     $EDITOR env.sh
#     source env.sh
#
# Why a file instead of ~/.bashrc: on a shared or borrowed account, .bashrc
# belongs to whoever owns the account. Sourcing a file you own leaves no
# permanent trace, and `sbatch --export=ALL` propagates whatever is exported in
# the submitting shell -- so nothing here needs to live in a dotfile.
#
# env.sh is gitignored. Keep it out of version control: it holds a token.

# --------------------------------------------------------------------------- #
# Paths -- two SEPARATE repositories
# --------------------------------------------------------------------------- #

# This repo (contains univtac_groot/, scripts/run_eval.py).
export REPO_ROOT="$HOME/jaewon/workspace/UniVTAC"

# The UniVTAC simulator checkout (contains envs/, task_config/, install.sh).
export UNIVTAC_ROOT="$HOME/jaewon/workspace/UniVTAC-sim"

# --------------------------------------------------------------------------- #
# Interpreters
# --------------------------------------------------------------------------- #

# The UniVTAC conda env (Python 3.10, Isaac Sim). Runs the evaluator.
# While that env is active this is just $CONDA_PREFIX/bin/python.
export UNIVTAC_PYTHON="$(conda run -n UniVTAC which python 2>/dev/null | tail -1)"

# GR00T's uv venv (Python 3.12). Runs the model server.
export GROOT_PYTHON="$HOME/Isaac-GR00T/.venv/bin/python"

# Optional: interpreter for dataset conversion (needs h5py/pyarrow/imageio,
# but neither Isaac Sim nor gr00t). Defaults to UNIVTAC_PYTHON.
# export CONVERT_PYTHON="$(conda run -n univtac-groot which python | tail -1)"

# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #

# YOUR token, from https://huggingface.co/settings/tokens (read-only is enough).
#
# Set it here rather than running `hf auth login`: that command OVERWRITES the
# stored token at $HF_HOME/token, which on a shared account is somebody else's
# credential. HF_TOKEN takes precedence over any stored login for your
# processes only, and touches nothing on disk.
#
# Keep this file mode 600 (`chmod 600 env.sh`).
export HF_TOKEN="hf_REPLACE_ME"

# Optional: keep the ~7 GB model cache (and any token file) out of the account
# owner's home quota. Point it at scratch or your own directory.
# export HF_HOME="${SCRATCH:-/tmp}/$USER-hf"

# --------------------------------------------------------------------------- #
# SLURM
# --------------------------------------------------------------------------- #

# Clusters that require a wckey need it on sbatch too, not just srun.
# SLURM reads SBATCH_WCKEY as the default for --wckey.
export SBATCH_WCKEY="project-short-name:sub_4dpdata"

# This cluster's submit filter REJECTS any job without MODEL_OUTPUT_DIR set to a
# path under /rlwrld-unified-checkpoints. The rejection happens at `sbatch` time,
# so the job never starts and no log is written -- the only clue is
# "MODEL_OUTPUT_DIR가 없습니다" on stderr. It is also the right place for
# checkpoints: the shared home mount is far more contended.
export MODEL_OUTPUT_DIR="/rlwrld-unified-checkpoints/$USER/univtac-groot"

# Where finetuned checkpoints go. Defaults to MODEL_OUTPUT_DIR inside the job
# scripts, which keeps tens of GB per arm off the home filesystem.
export CKPT_ROOT="${MODEL_OUTPUT_DIR}"

# Where converted datasets go (only needed for finetuning / the tactile arm).
export DATA_ROOT="${SCRATCH:-/tmp}/$USER-univtac-datasets"

# --------------------------------------------------------------------------- #
echo "environment set:"
echo "  REPO_ROOT       = ${REPO_ROOT}"
echo "  UNIVTAC_ROOT    = ${UNIVTAC_ROOT}"
echo "  UNIVTAC_PYTHON  = ${UNIVTAC_PYTHON:-<unset>}"
echo "  GROOT_PYTHON    = ${GROOT_PYTHON:-<unset>}"
echo "  HF_TOKEN        = ${HF_TOKEN:0:7}... (${#HF_TOKEN} chars)"
echo
echo "verify with:  python \$REPO_ROOT/scripts/preflight.py --deep"
