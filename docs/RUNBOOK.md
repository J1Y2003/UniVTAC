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
# 0a. Clone the simulator.                                       [login]
git clone https://github.com/univtac/UniVTAC.git ~/UniVTAC-sim

# 0b. Install it as a BATCH job (hours). Creates conda env `UniVTAC`, Python 3.10.
cd $REPO_ROOT
sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,UNIVTAC_ROOT=$HOME/UniVTAC-sim slurm/install_univtac.sbatch

# 0c. Scene assets.
cd ~/UniVTAC-sim && bash data/download.sh

# 0d. GR00T, Python 3.12, separate env.
git lfs install                                    # no sudo needed if git-lfs exists
git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T
cd ~/Isaac-GR00T && uv sync --python 3.12
uv run python -c "import gr00t; print('ok')"
```

### Why 0b is an sbatch job, not `srun --pty`

Many clusters restrict interactive `srun` to a short `debug` partition, so a
multi-hour install cannot run interactively. `slurm/install_univtac.sbatch`
wraps UniVTAC's `scripts/install.sh` and works around four things that do not
survive a batch context:

| Problem in upstream `scripts/install.sh` | Workaround |
| --- | --- |
| `sudo apt install cmake build-essential` / `git-lfs` — no sudo on a shared cluster, and no TTY to prompt on | checks the tools exist, then patches the `sudo` lines out |
| `--livestream 2` on the TacEx smoke test — waits forever for a GUI client | rewritten to `--headless` |
| `python3 -m pytest .` for cuRobo under `set -e` — one environment-specific test failure aborts the whole install | made non-fatal |
| step 6's `if [ -d "Toolchain" ]` never fires on a fresh machine, so **vcpkg is never cloned** yet `CMAKE_TOOLCHAIN_FILE` points into it → libuipc build fails | clones and bootstraps vcpkg itself |

The repo is not modified; the wrapper writes a patched copy to
`.install_batch.sh` and prints the diff into the job log.

**It is resumable.** Upstream guards every step with
`if pip show <pkg>; then skip`, so if the job hits its walltime just resubmit
and it continues from where it stopped.

### Verify before moving on `[compute, debug partition is fine]`

```bash
srun --partition=debug --gres=gpu:1 --wckey=project-short-name:sub_4dpdata --pty bash -l
conda activate UniVTAC
python -c "import isaaclab, tacex; print('ok')"
cd ~/UniVTAC-sim && bash collect_data.sh grasp_classify demo 0
```

This is the check people skip and regret. If UniVTAC cannot collect data on its
own, nothing downstream works, and the failure will look like a GR00T problem.

## Step 1 — Hugging Face access `[login]`

GR00T's backbone `nvidia/Cosmos-Reason2-2B` is **gated** and every checkpoint
loads it, including the base model.

1. Request access at <https://huggingface.co/nvidia/Cosmos-Reason2-2B> (approval
   is not instant — do this early).
2. Authenticate:

```bash
export HF_TOKEN=hf_...       # add to ~/.bashrc; see SETUP.md for shared servers
hf auth whoami               # must report YOU, not a shared account
```

On a shared machine `hf auth whoami` may already show someone else's identity.
Do not log them out -- `HF_TOKEN` overrides it for your processes only.

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
conda create -n univtac-groot python=3.12 -y
conda activate univtac-groot

# ~19 MB of prebuilt wheels, well under a minute. Safe on a login node.
# --only-binary=:all: makes pip FAIL rather than silently start a source build.
pip install --only-binary=:all: -r $REPO_ROOT/requirements-dev.txt

# Only if you will convert datasets: ~85 MB more, mostly pyarrow (50 MB) and
# the bundled ffmpeg (30 MB). Still wheels-only, but 1-2 minutes -- if your
# cluster is strict about login-node work, run this line inside the step 0
# allocation instead.
pip install --only-binary=:all: -r $REPO_ROOT/requirements-convert.txt

# 2b. The evaluator needs the client packages inside the UniVTAC env, because
#     that is the interpreter that talks to the server. Four packages, no torch.
conda activate UniVTAC
pip install -r $REPO_ROOT/requirements-client.txt
conda deactivate

# 2c. Exports. Use a session file rather than ~/.bashrc -- see the note below.
cd $REPO_ROOT
cp env.example.sh env.sh && chmod 600 env.sh
$EDITOR env.sh                 # fill in paths + your HF_TOKEN
source env.sh
```

**On a shared or borrowed account, do not edit `~/.bashrc`.** It belongs to
whoever owns the account. `env.sh` is gitignored, leaves no permanent trace, and
`sbatch --export=ALL` propagates whatever the submitting shell exported -- so
nothing here needs to be in a dotfile. Source it once per session.

Two things to avoid on someone else's account:

* **`hf auth login`** overwrites the stored token at `$HF_HOME/token`, which is
  their credential. Export `HF_TOKEN` instead; it takes precedence for your
  processes and writes nothing.
* **A default `HF_HOME`** puts the ~7 GB checkpoint into their home quota. Point
  `HF_HOME` at scratch if that matters.

Note that UniVTAC's own `scripts/install.sh` already appended
`export CMAKE_TOOLCHAIN_FILE=...` to `~/.bashrc` (step 6 of that script) and
cloned vcpkg into `~/Toolchain`. Worth mentioning to the account owner.

`CONVERT_PYTHON` points the conversion job at your tooling env, so dataset
conversion never needs h5py/pyarrow installed into the simulator env either.

**Why `--only-binary=:all:`.** Every package here publishes a manylinux wheel
for CPython 3.12, so installing is a download and an unpack -- no compiler runs.
The failure mode worth guarding against is pip falling back to an sdist (wrong
Python version, unusual arch), because building `pyarrow` from source pulls in
CMake and Arrow C++ and takes tens of minutes on many cores -- exactly the kind
of login-node work your cluster forbids. With this flag pip stops with
`No matching distribution` instead, and you can move the install to a compute
node deliberately. Python 3.12 is chosen because numpy >= 2.5 requires it, so
you get current wheels for everything.

## Step 3 — cheap verification `[login]`

```bash
conda activate univtac-groot
cd $REPO_ROOT

pytest tests -q
python scripts/preflight.py --deep
```

Run every `scripts/preflight.py` and `scripts/compare_ablation.py` command in
this env. They need only numpy, and keeping them out of `base` and out of the
simulator env means neither can be broken by this repo.

Get preflight to all-green before touching a GPU. It catches wrong paths, a
missing HF token, and the wrong interpreter in seconds instead of after a
20-minute queue wait plus a checkpoint load.

**Do not skip the cuDNN line in that output.** It is the one check whose
failure is silent: a cuDNN that does not match torch's pin makes training
~86x slower with no error message at all, so the job looks like it is working
and simply never finishes. `--deep` on a login node still verifies the
version (it can only skip the GPU convolution), and every GPU job re-runs the
same check via `scripts/check_cudnn.py` before loading weights, so a mismatch
aborts the job in seconds instead of wasting a night. If it fails, the output
prints the exact reinstall commands. Detail and rationale:
[SETUP.md](SETUP.md#cudnn-check-the-library-not-the-metadata).

## Step 4 — first evaluation, interactively `[compute]`

Do this once by hand before submitting anything. The point is to keep the
server alive across attempts: the checkpoint load is minutes, the evaluator
restarts in seconds.

```bash
srun --gres=gpu:1 --wckey=project-short-name:sub_4dpdata --pty bash
cd $REPO_ROOT

# 4a. Start the server in the background; wait for "listening on".
PYTHONUNBUFFERED=1 PYTHONPATH=$REPO_ROOT $GROOT_PYTHON -u \
    -m univtac_groot.server.run_server \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --port 5555 > /tmp/groot_server.log 2>&1 &
tail -f /tmp/groot_server.log        # Ctrl-C once you see "listening on tcp://..."

# 4b. Check the observation contract. Seconds; no Isaac Sim startup.
$UNIVTAC_PYTHON scripts/run_eval.py \
    --task insert_hole --variant baseline --port 5555 --dry-run

# 4c. One real episode. This starts Isaac Sim, so allow a few minutes.
$UNIVTAC_PYTHON scripts/run_eval.py \
    --task insert_hole --variant baseline --univtac-root $UNIVTAC_ROOT \
    --port 5555 --episodes 1 --execution-horizon 16
```

Use `tmux` so a dropped SSH connection does not kill the allocation.

If 4c produces one line of JSONL under `$REPO_ROOT/eval_result/baseline/insert_hole/`,
the pipeline works end to end and you can submit in bulk.

## Step 5 — the real run `[login]` → submits to `[compute]`

```bash
cd $REPO_ROOT

# The benchmark: finetune then evaluate, one job per task. --dry runs the
# full preflight and submits nothing, which is how to check paths and
# datasets without spending queue time.
bash slurm/submit_benchmark.sh --dry
bash slurm/submit_benchmark.sh

squeue -u $USER -o '%.8i %.12P %.70j %.9T %.10M %.20R'
```

That covers the three reported tasks plus `lift_bottle`, gated behind them.
To evaluate a checkpoint that already exists without retraining, or to add the
zero-shot `baseline` row:

```bash
BASELINE_FT_MODEL=/ckpt/univtac-insert_hole/final bash slurm/submit_ablation.sh
VARIANTS=baseline bash slurm/submit_ablation.sh
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

## Steps 7-9 — the tactile variant (only if you need the ablation)

The tactile variant **cannot run zero-shot**: extra state dimensions require the
`NEW_EMBODIMENT` tag, which ships in no released checkpoint. See
[ABLATION.md](ABLATION.md) for why. It needs demonstrations, a conversion, and a
finetune first — days of work, not minutes.

```bash
# 7. Get demonstrations. The released data (100 episodes per task) is the
#    intended source; collecting your own is only for tasks not in the release.
#    CPU-only, so --partition=cpu. ~24 GB per task.              [login] -> [compute]
cd $REPO_ROOT
sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu \
    --export=ALL,TASKS=insert_hole,VERSION=45,RAW_CONFIG=clean slurm/download_data.sbatch

# 8. Convert to GR00T LeRobot v2, once per variant -- the state layout differs.
#    CPU-only, minutes. Runs under $CONVERT_PYTHON, not the simulator's Python.
for v in tactile baseline_finetuned; do
  sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu \
      --export=ALL,TASK=insert_hole,VARIANT=$v,TASK_CONFIG=clean slurm/convert.sbatch
done

# 9. Finetune and evaluate both, same recipe, in one resumable job.
#    EXTRA_TASKS="" skips the gated lift_bottle run.        [login] -> [compute]
VARIANTS="tactile baseline_finetuned" TASKS=insert_hole EXTRA_TASKS="" \
    bash slurm/submit_benchmark.sh

# 10. Compare them against each other.
python scripts/compare_ablation.py --baseline-variant baseline_finetuned --tactile-variant tactile
```

Every `sbatch` needs `--wckey`, and every CPU-only job needs `--partition=cpu`;
the submit filter rejects them otherwise. `submit_benchmark.sh` adds both for
you, which is the main reason to prefer it over hand-written `sbatch` lines.

Compare the tactile variant against `baseline_finetuned`, not against the zero-shot
baseline — otherwise the whole finetuning effect gets credited to touch.

## If something fails

1. `python scripts/preflight.py --deep` — names the first broken thing.
2. For a failed eval job, **read the server log first**
   (`eval_result/<variant>/<task>/server-*.log`). A checkpoint that fails to load
   appears in the evaluator's log only as a `wait_until_ready` timeout.
3. [SETUP.md](SETUP.md#common-failures) has a symptom-to-cause table.
