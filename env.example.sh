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

# This cluster requires a wckey on every sbatch and srun. The job scripts pass
# `--wckey=project-short-name:sub_4dpdata` themselves; this export covers anything you submit by
# hand, since sbatch reads SBATCH_WCKEY.
export SBATCH_WCKEY="project-short-name:sub_4dpdata"

# Do NOT export SLURM_WCKEY. SLURM sets it *inside* a job to report the wckey
# the job actually received, which is what every .sbatch here re-checks at
# runtime. Exporting it from your shell rides in on `--export=ALL` and masks
# that check. For srun, pass --wckey on the command line instead.

# Jobs here land on `debug` by default, whose time limit is short enough that a
# training job sits pending forever with REASON=PartitionTimeLimit. Name a
# long-running partition instead; SLURM reads SBATCH_PARTITION as the default
# for --partition. Confirm the name and its limit first:
#   sinfo -o "%20P %10l %10L %6D %25G"
# export SBATCH_PARTITION="gpu"

# DO NOT export MODEL_OUTPUT_DIR (or OUTPUT_DIR, CODE_OUTPUT_DIR,
# JOB_OUTPUT_BUNDLE_DIR, CHECKPOINT_DIR). bundle-sbatch owns all five: it
# creates one output bundle per submission and injects them, and it REFUSES to
# run if it finds them already set:
#
#   error: inherited bundle path environment is unsupported
#
# An older version of this file exported MODEL_OUTPUT_DIR because the previous
# storage policy demanded it. If your env.sh still does, delete that line --
# env.sh is gitignored, so fixing this template does not fix your copy.
#
# Nothing here needs to point at a checkpoint directory any more -- the
# launcher chooses it. Our area under the managed root is
# /rlwrld-unified-checkpoints/jimin/jaewon; that is what a checkpoint path
# should look like when you hand one to slurm/eval_checkpoint.sh, and it is
# where bundle-sbatch will accept a --checkpoint from without Hugging Face
# download metadata.

# CKPT_ROOT is only for redirecting checkpoints AWAY from the bundle, which the
# smoke test does to keep its output separate. Leave it unset otherwise, so the
# job writes to the bundle the launcher gave it.
# export CKPT_ROOT="$HOME/jaewon/workspace/groot-smoke"

# Where converted datasets go. Must be on SHARED storage: the convert job and
# the training job land on different nodes, and a /tmp default would put the
# dataset somewhere the trainer cannot see. Matches slurm/submit_benchmark.sh's
# own default, so conversion and training agree without extra flags.
export DATA_ROOT="$HOME/jaewon/workspace/groot-data"

# --------------------------------------------------------------------------- #
echo "environment set:"
echo "  REPO_ROOT       = ${REPO_ROOT}"
echo "  UNIVTAC_ROOT    = ${UNIVTAC_ROOT}"
echo "  UNIVTAC_PYTHON  = ${UNIVTAC_PYTHON:-<unset>}"
echo "  GROOT_PYTHON    = ${GROOT_PYTHON:-<unset>}"
echo "  HF_TOKEN        = ${HF_TOKEN:0:7}... (${#HF_TOKEN} chars)"
echo
echo "verify with:  python \$REPO_ROOT/scripts/preflight.py --deep"
