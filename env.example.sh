# Per-session environment. Copy, edit, source:
#
#     cp env.example.sh env.sh && $EDITOR env.sh
#     chmod 600 env.sh
#     source env.sh
#
# env.sh is gitignored and holds two tokens.

# --- Paths ----------------------------------------------------------------- #
export REPO_ROOT="$HOME/jaewon/workspace/UniVTAC"          # this repo
export UNIVTAC_ROOT="$HOME/jaewon/workspace/UniVTAC-sim"   # the simulator
export DATA_ROOT="$HOME/jaewon/workspace/groot-data"       # converted datasets
export HF_HOME="$HOME/jaewon/hf_cache"                     # ~7 GB of weights

# DATA_ROOT must be on shared storage: conversion and training run on different
# nodes.

# --- Interpreters ---------------------------------------------------------- #
# Python 3.10 for Isaac Sim, 3.12 for GR00T. They cannot share a virtualenv.
export UNIVTAC_PYTHON="$(conda run -n UniVTAC which python 2>/dev/null | tail -1)"
export GROOT_PYTHON="$HOME/jaewon/workspace/Isaac-GR00T/.venv/bin/python"

# Optional interpreter for dataset conversion. Defaults to UNIVTAC_PYTHON.
# export CONVERT_PYTHON="$(conda run -n univtac-groot which python | tail -1)"

# --- Tokens ---------------------------------------------------------------- #
# Export them; do not run `hf auth login` or `wandb login`, which write into the
# shared account's credential store. HF_TOKEN needs access to the gated
# nvidia/Cosmos-Reason2-2B.
export HF_TOKEN="hf_REPLACE_ME"
export WANDB_API_KEY="REPLACE_ME"   # unset = offline logs under the bundle

# --- SLURM ----------------------------------------------------------------- #
# Required on every submission, and the value is not arbitrary -- see CLAUDE.md,
# "Cluster rules". The scripts pass --wckey themselves; this covers hand
# submissions, since sbatch reads SBATCH_WCKEY.
export SBATCH_WCKEY="project-short-name:sub_4dpdata"

# Do not export SLURM_WCKEY: SLURM sets it inside a job, and every .sbatch
# re-checks it.

# Do not export MODEL_OUTPUT_DIR, OUTPUT_DIR, CODE_OUTPUT_DIR,
# JOB_OUTPUT_BUNDLE_DIR or CHECKPOINT_DIR. bundle-sbatch owns all five and
# refuses to run if it finds one set (`inherited bundle path environment is
# unsupported`).

# Optional; the submitters name a partition themselves. `debug` is the cluster
# default and caps at 3 h.
# export SBATCH_PARTITION="sjw_alinlab"

# --------------------------------------------------------------------------- #
echo "environment set:"
echo "  REPO_ROOT       = ${REPO_ROOT}"
echo "  UNIVTAC_ROOT    = ${UNIVTAC_ROOT}"
echo "  UNIVTAC_PYTHON  = ${UNIVTAC_PYTHON:-<unset>}"
echo "  GROOT_PYTHON    = ${GROOT_PYTHON:-<unset>}"
echo "  HF_TOKEN        = ${#HF_TOKEN} chars"
echo "  WANDB_API_KEY   = ${#WANDB_API_KEY} chars"
echo
echo "verify with:  python \$REPO_ROOT/scripts/preflight.py --deep"
