# Template for env.sh, which is gitignored and so does not travel with a clone.
#
#   cp env.example.sh env.sh && $EDITOR env.sh && chmod 600 env.sh
#   source env.sh          # per session, never from ~/.bashrc
#
# This belongs on the CLUSTER, where jobs are submitted. A workstation checkout
# needs none of it -- see README.md, "Set up a new machine".
#
# Keep it out of ~/.bashrc: the tokens below end up in every process you start,
# and a stale value here silently outranks a flag (see the three traps at the
# bottom).

# --- The three checkouts -----------------------------------------------------
# Similar names, different repositories. REPO_ROOT is this one.
export REPO_ROOT=$HOME/UniVTAC                 # this repo; submit jobs from here
export UNIVTAC_ROOT=$HOME/UniVTAC-sim          # the simulator (github.com/univtac/UniVTAC)
export GROOT_ROOT=$HOME/Isaac-GR00T            # NVIDIA/Isaac-GR00T

# --- The two interpreters ----------------------------------------------------
# They cannot share a virtualenv: UniVTAC is Python 3.10 (Isaac Sim), GR00T is
# 3.12. That is why the model is served over loopback ZeroMQ.
export UNIVTAC_PYTHON=$(conda run -n UniVTAC which python)
export GROOT_PYTHON=$GROOT_ROOT/.venv/bin/python   # uv's venv, not a system python

# --- Where data and checkpoints live -----------------------------------------
# OUTPUT_ROOT's parent belongs to the shared account owner; keep ours in the
# `jaewon` subtree. ~26 GB per retained checkpoint.
export DATA_ROOT=$HOME/jaewon/data                     # converted LeRobot datasets
export OUTPUT_ROOT=/rlwrld-unified-checkpoints/$USER/jaewon

# --- Submit-filter formalities -----------------------------------------------
# `project-short-name:` is a LITERAL part of the required format, not a
# placeholder to fill in. Strip it and every submission is rejected.
export SBATCH_WCKEY=project-short-name:sub_4dpdata

# --- Tokens ------------------------------------------------------------------
# Export them; never run `hf auth login` or `wandb login`, which write into the
# shared account owner's credential store. `--export=ALL` carries these into the
# job. nvidia/Cosmos-Reason2-2B is gated and every GR00T checkpoint loads it, so
# without HF_TOKEN the server dies at load with a 401.
export HF_TOKEN=hf_...                 # https://huggingface.co/settings/tokens
export WANDB_API_KEY=...               # https://wandb.ai/authorize

# --- Three things to leave unset ---------------------------------------------
# Each of these outranks the flag you would pass on the command line.
#
#   SLURM_WCKEY      SLURM sets it INSIDE a job to report the wckey the job got,
#                    which is what the runtime re-check reads. Exporting it
#                    masks that check. `srun` takes --wckey on the command line.
#   SBATCH_PARTITION Precedence is command line > SBATCH_PARTITION > #SBATCH,
#                    so an exported value silently overrides both job scripts'
#                    headers.
#   DRY_RUN          A stray DRY_RUN=1 left over from testing makes a job exit
#                    in seconds having trained nothing.
unset SLURM_WCKEY SBATCH_PARTITION DRY_RUN

# MODEL_OUTPUT_DIR is required by the submit filter on EVERY submission,
# including evaluation, but its value is per-job -- export it at submission
# time, not here, so it always names the run you are actually submitting.
