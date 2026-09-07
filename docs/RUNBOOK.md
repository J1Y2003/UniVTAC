# Runbook: from nothing to an evaluation number

Ordered. Every step says **where** it runs and **which environment**. Do them in
sequence; a step assumes the previous ones passed.

At any point, run this to see which step you are on:

```bash
python scripts/preflight.py          # add --deep to probe both interpreters
```

It prints a checklist and the exact next command. It needs only numpy, so it is
safe on a login node.

**Notation**

| Tag | Meaning |
| --- | --- |
| `[login]` | fine on the SSH login node: seconds, no GPU, little memory |
| `[compute]` | must be a compute node — `srun --pty` or `sbatch` |
| `(univtac-groot)`, `(UniVTAC)` | which conda env the command runs in |

**Never install into `base`.** On a shared cluster `base` usually belongs to the
system, and `pip install` there affects every user. Step 2 creates a dedicated
env for this repo's tooling; the simulator gets its own (`UniVTAC`, made by its
installer) and GR00T uses a uv venv, not conda.

Two repositories are involved and both are easy to confuse:

* **`$REPO_ROOT`** — *this* repo (contains `univtac_groot/`, `scripts/run_eval.py`)
* **`$UNIVTAC_ROOT`** — the simulator, `github.com/univtac/UniVTAC` (contains `envs/`, `task_config/`)

They must be separate directories. `preflight.py` checks this.

---

## Step 0 — one-time installs `[compute]`

Skip whatever you already have. **Do not run these on the login node**: both
compile CUDA code from source.

```bash
srun --cpus-per-task=16 --mem=64G --time=8:00:00 --pty bash     # allocation to build in

# 0a. The simulator (hours). Creates conda env `UniVTAC`, Python 3.10.
git clone https://github.com/univtac/UniVTAC.git ~/UniVTAC-sim
cd ~/UniVTAC-sim && bash scripts/install.sh

# 0b. Scene assets.
bash data/download.sh

# 0c. Prove the simulator works ON ITS OWN before involving GR00T.
conda activate UniVTAC
bash collect_data.sh grasp_classify demo 0
#    -> should write episodes under data/grasp_classify/demo/

# 0d. GR00T, Python 3.12, separate env.
sudo apt install git-lfs && git lfs install        # BEFORE cloning
git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T
cd ~/Isaac-GR00T && uv sync --python 3.12
uv run python -c "import gr00t; print('ok')"
```

Step 0c is the one people skip and regret. If UniVTAC cannot collect data by
itself, nothing downstream will work, and the failure will look like a GR00T
problem.

## Step 1 — Hugging Face access `[login]`

GR00T's backbone `nvidia/Cosmos-Reason2-2B` is **gated** and every checkpoint
loads it, including the base model.

1. Request access at <https://huggingface.co/nvidia/Cosmos-Reason2-2B> (approval
   is not instant — do this early).
2. Authenticate:

```bash
huggingface-cli login        # or: export HF_TOKEN=<token>
```

Without this the server dies at load with `GatedRepoError` / `401`, which you
will only see as a timeout in the evaluator's log.

## Step 2 — environments and exports `[login]`

Three environments, none of them `base`:

| Env | Owner | Purpose |
| --- | --- | --- |
| `univtac-groot` | **you create it, step 2a** | this repo's tooling: tests, preflight, aggregation, dataset conversion |
| `UniVTAC` | created by UniVTAC's `install.sh` | runs the evaluator (Isaac Sim, Python 3.10) |
| `~/Isaac-GR00T/.venv` | created by `uv sync` | runs the model server (Python 3.12) |

```bash
# 2a. A dedicated env for this repo's tooling. Small and torch-free.
conda create -n univtac-groot python=3.11 -y
conda activate univtac-groot
pip install -r $REPO_ROOT/requirements-dev.txt      # numpy, pytest, pyyaml, zmq, msgpack
pip install -r $REPO_ROOT/requirements-convert.txt  # h5py, pyarrow, imageio -- only if converting

# 2b. The evaluator needs the client packages inside the UniVTAC env, because
#     that is the interpreter that talks to the server. Four packages, no torch.
conda activate UniVTAC
pip install -r $REPO_ROOT/requirements-client.txt
conda deactivate

# 2c. Put these in ~/.bashrc so every shell and every sbatch job agrees.
export REPO_ROOT=~/jaewon/UniVTAC                          # THIS repo
export UNIVTAC_ROOT=~/UniVTAC-sim                          # the simulator
export UNIVTAC_PYTHON=$(conda run -n UniVTAC which python)
export GROOT_PYTHON=~/Isaac-GR00T/.venv/bin/python         # uv's venv
export CONVERT_PYTHON=$(conda run -n univtac-groot which python)
export DATA_ROOT=$SCRATCH/univtac-datasets                 # only for finetuning
```

`CONVERT_PYTHON` points the conversion job at your tooling env, so dataset
conversion never needs h5py/pyarrow installed into the simulator env either.

## Step 3 — cheap verification `[login]`

```bash
conda activate univtac-groot
cd $REPO_ROOT

pytest tests -q                  # expect: 94 passed
python scripts/preflight.py --deep
```

Run every `scripts/preflight.py` and `scripts/compare_ablation.py` command in
this env. They need only numpy, and keeping them out of `base` and out of the
simulator env means neither can be broken by this repo.

Get preflight to all-green before touching a GPU. It catches wrong paths, a
missing HF token, and the wrong interpreter in seconds instead of after a
20-minute queue wait plus a checkpoint load.

## Step 4 — first evaluation, interactively `[compute]`

Do this once by hand before submitting anything. The point is to keep the
server alive across attempts: the checkpoint load is minutes, the evaluator
restarts in seconds.

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 --pty bash
cd $REPO_ROOT

# 4a. Start the server in the background; wait for "listening on".
PYTHONPATH=$REPO_ROOT $GROOT_PYTHON -m univtac_groot.server.run_server \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --port 5555 > /tmp/groot_server.log 2>&1 &
tail -f /tmp/groot_server.log        # Ctrl-C once you see "listening on tcp://..."

# 4b. Check the observation contract. Seconds; no Isaac Sim startup.
$UNIVTAC_PYTHON scripts/run_eval.py \
    --task insert_hole --arm baseline --port 5555 --dry-run

# 4c. One real episode. This starts Isaac Sim, so allow a few minutes.
$UNIVTAC_PYTHON scripts/run_eval.py \
    --task insert_hole --arm baseline --univtac-root $UNIVTAC_ROOT \
    --port 5555 --episodes 1 --execution-horizon 8
```

Use `tmux` so a dropped SSH connection does not kill the allocation.

If 4c produces one line of JSONL under `$REPO_ROOT/eval_result/baseline/insert_hole/`,
the pipeline works end to end and you can submit in bulk.

## Step 5 — the real run `[login]` → submits to `[compute]`

```bash
cd $REPO_ROOT

# One task, 50 episodes.
sbatch --export=ALL,ARM=baseline,TASK=insert_hole slurm/eval_ablation.sbatch

# Or all eight benchmark tasks for the baseline arm.
ARMS=baseline bash slurm/submit_ablation.sh

squeue -u $USER
```

`eval_ablation.sbatch` runs the server *and* the evaluator inside the one job, so
you do not manage the server yourself. Budget ~24 GB VRAM; if your nodes are
smaller, add `--gres=gpu:2` and the script splits the two processes across them.

## Step 6 — read the results `[login]`

```bash
conda activate univtac-groot
python scripts/compare_ablation.py
python scripts/compare_ablation.py --json ablation.json     # machine-readable
```

Results are appended per episode, so this works while jobs are still running and
after a walltime kill.

---

## Steps 7-9 — the tactile arm (only if you need the ablation)

The tactile arm **cannot run zero-shot**: extra state dimensions require the
`NEW_EMBODIMENT` tag, which ships in no released checkpoint. See
[ABLATION.md](ABLATION.md) for why. It needs demonstrations, a conversion, and a
finetune first — days of work, not minutes.

```bash
# 7. Collect demonstrations for each task you will evaluate.   [compute]
cd $UNIVTAC_ROOT && conda activate UniVTAC
bash collect_data.sh insert_hole demo 0

# 8. Convert to GR00T LeRobot v2. CPU-only, minutes-hours.     [login] -> [compute]
#    Runs under $CONVERT_PYTHON (your univtac-groot env), not the simulator's.
cd $REPO_ROOT
sbatch --export=ALL,TASK=insert_hole,ARM=tactile           slurm/convert.sbatch
sbatch --export=ALL,TASK=insert_hole,ARM=baseline_finetuned slurm/convert.sbatch

# 9. Finetune both arms with the SAME recipe.                  [login] -> [compute]
sbatch --export=ALL,ARM=tactile,DATASET=$DATA_ROOT/univtac-insert_hole-tactile \
    slurm/finetune.sbatch
sbatch --export=ALL,ARM=baseline_finetuned,DATASET=$DATA_ROOT/univtac-insert_hole-baseline_finetuned \
    slurm/finetune.sbatch

# 10. Evaluate both, then compare them against each other.
sbatch --export=ALL,ARM=tactile,TASK=insert_hole,GROOT_MODEL=<ckpt> slurm/eval_ablation.sbatch
sbatch --export=ALL,ARM=baseline_finetuned,TASK=insert_hole,GROOT_MODEL=<ckpt> slurm/eval_ablation.sbatch

python scripts/compare_ablation.py --baseline-arm baseline_finetuned --tactile-arm tactile
```

Compare the tactile arm against `baseline_finetuned`, not against the zero-shot
baseline — otherwise the whole finetuning effect gets credited to touch.

## If something fails

1. `python scripts/preflight.py --deep` — names the first broken thing.
2. For a failed eval job, **read the server log first**
   (`eval_result/<arm>/<task>/server-*.log`). A checkpoint that fails to load
   appears in the evaluator's log only as a `wait_until_ready` timeout.
3. [SETUP.md](SETUP.md#common-failures) has a symptom-to-cause table.
