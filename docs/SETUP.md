# Setup: what must exist before you run anything

> For the **ordered sequence** of commands, see [RUNBOOK.md](RUNBOOK.md), or run
> `python scripts/preflight.py`. This document explains the *why* behind each
> prerequisite and lists failure modes.

## What "a live GR00T server" means

It is a **separate long-running OS process** that has already loaded the
checkpoint into GPU memory and is listening on a TCP port for inference
requests. Nothing in the UniVTAC-side code loads the model; it only sends
observations over a socket and gets action chunks back.

"Live" specifically means it answers the `ping` endpoint. A server that has been
launched but is still pulling weights from Hugging Face is *not* live yet —
`Gr00tClient.wait_until_ready()` exists precisely to block through that window,
which takes minutes for a 3B checkpoint on a cold cache.

```
$ python -m univtac_groot.server.run_server \
      --model-path nvidia/GR00T-N1.7-3B \
      --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --port 5555

[server] loading nvidia/GR00T-N1.7-3B as OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT on cuda
... minutes of weight loading ...
[server] ready: state_keys=['eef_9d', 'gripper_position', 'joint_position'] ...
Server is ready and listening on tcp://0.0.0.0:5555     <-- now it is "live"
```

From then on it sits idle until the evaluator connects. One server serves a
whole evaluation run; it is not restarted per episode.

### Why it is a separate process, not an import

This is not a stylistic choice — the two stacks **cannot share an interpreter**:

| | UniVTAC | GR00T N1.7 |
| --- | --- | --- |
| Python | **3.10** | **3.12** |
| Torch | 2.5.1 + cu118 | pinned by `uv.lock`, CUDA 12.8 |
| Core deps | Isaac Sim 4.5, Isaac Lab 2.1.1, TacEx, cuRobo | transformers, flash-attn, torchcodec, TensorRT |

Sources: `UniVTAC/docs/Installation.md` ("Python 3.10", `torch==2.5.1`) and
`Isaac-GR00T/README.md` ("dGPU on CUDA 12.8 with Python 3.12", `uv sync --python 3.12`).

Different minor Python versions means no shared virtualenv is even possible, so
the model runs behind a ZeroMQ socket. UniVTAC's own SmolVLA integration does
the same thing for the same reason (`policy/smolvla/smolvla_server.py` runs in
`policy/smolvla/.venv`).

---

## Prerequisites

You need **two separate environments** on a Linux machine with an NVIDIA GPU.

### 1. UniVTAC environment (conda env `UniVTAC`, Python 3.10)

```bash
git clone https://github.com/univtac/UniVTAC.git
cd UniVTAC
bash scripts/install.sh        # builds Isaac Sim, Isaac Lab, TacEx, cuRobo
```

This is the long pole — expect hours, and it is not always first-try clean.
`docs/Installation.md` has the manual path if the all-in-one script fails.
Things worth knowing before you start:

- **TacEx must be built from `third_party/TacEx`**, not the public repo — UniVTAC
  ships project-specific modifications.
- `tacex_uipc` builds libuipc from source and needs **vcpkg**, CMake 3.26,
  GCC 11.4 and CUDA 12.4, plus `CMAKE_TOOLCHAIN_FILE` exported.
- Verify with UniVTAC's own smoke test before involving GR00T at all:
  ```bash
  conda activate UniVTAC
  bash collect_data.sh grasp_classify demo 0
  ```
  If that does not produce data, no amount of GR00T setup will help.

Then add this repo's client dependencies (five small packages, no torch --
about 19 MB of wheels, so this one is fine on a login node):

```bash
conda activate UniVTAC
pip install --only-binary=:all: -r requirements-client.txt
```

### 2. GR00T environment (Python 3.12)

```bash
sudo apt install git-lfs && git lfs install     # BEFORE cloning
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install -y ffmpeg
uv sync --python 3.12
uv run python -c "import gr00t; print('ok')"
```

- **`git-lfs` before cloning**, or `demo_data/`'s parquet files arrive as
  pointer stubs.
- **FFmpeg must be version 4–7.** `torchcodec==0.8.0` is GR00T's only supported
  video backend and cannot load FFmpeg 8, which is what Ubuntu 25.10+/26.04 ship.
  On those, `conda install -c conda-forge 'ffmpeg<8'` and put it on
  `LD_LIBRARY_PATH`.

### 3. Hugging Face access to a gated model (required)

GR00T's VLM backbone is [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B),
which is **gated**, and *every* GR00T checkpoint loads it on first use —
including the base `nvidia/GR00T-N1.7-3B`. Request access on that model page
(approval is not instant), then authenticate:

```bash
export HF_TOKEN=hf_...        # preferred; see the shared-machine note below
# or, to store it:  hf auth login    (older CLIs: huggingface-cli login)
```

Without it, the server dies at load with `GatedRepoError` / `401 Client Error`.
This is the single most common reason `wait_until_ready` times out, which is why
that error message names the model.

### On a shared server

`hf auth whoami` may report **somebody else's** identity: a machine-wide
`HF_TOKEN`, a shared `HF_HOME`, or a token left in a shared cache. Do not run
`hf auth logout` -- that would clobber a credential you do not own.

Export your own token instead. `HF_TOKEN` takes precedence over any stored
login, so it overrides the ambient identity for your processes only:

```bash
export HF_TOKEN=hf_...              # from https://huggingface.co/settings/tokens
hf auth whoami                      # should now report you
```

For full isolation of both token and model cache, redirect `HF_HOME` as well:

```bash
export HF_HOME=$SCRATCH/hf_home     # or $HOME/.cache/hf_mine
hf auth login                       # writes $HF_HOME/token
```

Two cautions:

* **`HF_HOME` moves the token file too.** `slurm/eval_ablation.sbatch` defaults
  `HF_HOME` to scratch, so a token stored under the default
  `~/.cache/huggingface` is not visible inside the job. Exporting `HF_TOKEN`
  sidesteps this, and the job warns when neither is present.
* **Never put the token in a tracked file or in an `#SBATCH` line** -- job
  scripts are often world-readable. Keep it in your environment
  (`chmod 600` any file that holds it) and let `--export=ALL` carry it in.

### 4. UniVTAC assets and (optionally) demonstration data

The task scenes need UniVTAC's assets. Its dataset/assets come from ModelScope:

```bash
cd UniVTAC && bash data/download.sh    # installs modelscope, pulls byml2024/UniVTAC
```

Demonstration data is only needed if you are going to finetune (i.e. for the
tactile arm) — see [ABLATION.md](ABLATION.md).

---

## Wiring the two together

The SLURM script needs to know which interpreter is which:

```bash
export REPO_ROOT=/path/to/this/repo
export UNIVTAC_ROOT=/path/to/UniVTAC
export UNIVTAC_PYTHON=$(conda run -n UniVTAC which python)
export GROOT_PYTHON=/path/to/Isaac-GR00T/.venv/bin/python   # uv's venv
```

`GROOT_PYTHON` is the **uv venv's** interpreter, not a system python — that is
where `uv sync` installed `gr00t`. `slurm/eval_ablation.sbatch` sets
`PYTHONPATH=$REPO_ROOT` when launching the server so `univtac_groot.server` is
importable alongside `gr00t`.

If finetuning fails with `CUDA_HOME is unset`, run
`bash scripts/deployment/dgpu/install_deps.sh` once in the GR00T checkout, or
`export CUDA_HOME=/usr/local/cuda`.

## Preflight, cheapest first

Each step assumes the previous one passed. Step 1 runs anywhere; **steps 2-6
need a compute node** (2 and 3 import CUDA-linked stacks, 4-6 use the GPU) — see
[Working from an SSH login node](#working-from-an-ssh-login-node) for how to get
an interactive allocation to run them in.

```bash
# 1. This repo's logic — login node is fine: no GPU, no Isaac Sim, no gr00t
pip install -r requirements-dev.txt && pytest tests -q          # expect 94 passed

# 2. GR00T imports in its own env
/path/to/Isaac-GR00T/.venv/bin/python -c "import gr00t; print('ok')"

# 3. UniVTAC imports in its own env (must not print an Isaac Sim error)
conda run -n UniVTAC python -c "import isaaclab, tacex; print('ok')"

# 4. Server comes up live and reports the checkpoint's real contract
PYTHONUNBUFFERED=1 PYTHONPATH=$REPO_ROOT $GROOT_PYTHON -u \n    -m univtac_groot.server.run_server \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --port 5555

# 5. The observation contract matches, without paying for Isaac Sim startup
$UNIVTAC_PYTHON scripts/run_eval.py --task insert_hole --arm baseline \
    --port 5555 --dry-run

# 6. A single real episode before committing a sweep
$UNIVTAC_PYTHON scripts/run_eval.py --task insert_hole --arm baseline \
    --univtac-root $UNIVTAC_ROOT --port 5555 --episodes 1
```

Step 5 talks to the live server and prints the resolved horizons, state keys and
video keys, then exits — it is the fastest way to catch an arm/checkpoint
mismatch. Note that it does need the server (it calls `get_modality_config`);
there is no fully offline contract check, because the authoritative answer lives
in the checkpoint. Keep the server from step 4 running while you iterate on
steps 5 and 6 — reloading the checkpoint per attempt is the main time sink.

## Common failures

| Symptom | Cause |
| --- | --- |
| `GatedRepoError` / `401` on server start | No access to `nvidia/Cosmos-Reason2-2B`, or not logged in |
| `wait_until_ready` times out after 900 s | Server died on load — **read the server log**, not the evaluator's |
| `Embodiment tag 'NEW_EMBODIMENT' is not supported by this checkpoint` | Expected: `NEW_EMBODIMENT` needs a finetune, see [ABLATION.md](ABLATION.md) |
| `state key mismatch: the checkpoint's embodiment declares ...` | Arm and checkpoint disagree; wrong `--arm` or wrong `--model-path` |
| `--tactile-mode depth_pool needs observations.tactile to include 'depth'` | Add it to `UniVTAC/task_config/<config>.yml` |
| `Could not load libtorchcodec ... versions 4, 5, 6 and 7` | FFmpeg 8 installed; downgrade to <8 |
| Parquet files in `demo_data/` unreadable | Cloned Isaac-GR00T without `git-lfs` |
| `CUDA_HOME is unset` during finetune | Run GR00T's `scripts/deployment/dgpu/install_deps.sh` |
| `CUDNN_STATUS_NOT_INITIALIZED`, cuDNN debug log says `cudaGetDeviceCount(&count) != cudaSuccess` with `GPU=NULL` and compute capability `0.0` | **Driver older than GR00T's CUDA runtime.** torch is `2.9.0+cu128` (CUDA 12.8); a CUDA 12.4-era driver (e.g. 550.54.14) runs it via minor-version compatibility, so cuBLAS/tensor ops work, but cuDNN 9.13 cannot enumerate the device. Not a misinstall. Workaround: `--disable-cudnn` (or `DISABLE_CUDNN=1`). Real fixes need an admin: driver r570+, or NVIDIA's `cuda-compat-12-8` forward-compatibility package (supported on data-center GPUs like A100). Diagnose with `CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=stdout`. |
| `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` on the first `get_action` | The server inherited `LD_LIBRARY_PATH`/`CUDA_HOME` from the UniVTAC conda env (CUDA 12.4) while its torch is cu128. Launch it with `env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH`; the job scripts do this automatically. Check free VRAM first, since genuine OOM reports the same error. |
| `Arm motion planning failed on action 0` | cuRobo, not GR00T. Verify UniVTAC's own expert works: `bash collect_data.sh grasp_classify demo 0` |
| `ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available` | The flag is a hard error, not a fallback, and it fires mid-download inside the *server* log so it reads like a checkpoint fault. `eval_ablation.sbatch` now probes `GROOT_PYTHON` for the package and only enables the flag when present. Override with `HF_HUB_ENABLE_HF_TRANSFER=0`, or install it: `$GROOT_PYTHON -m pip install hf_transfer` (worth it for the ~15 GB of weights). |
| Port already in use with concurrent jobs | `eval_ablation.sbatch` derives a per-job port from `SLURM_JOB_ID`; pass `PORT=` to override |

---

## Working from an SSH login node

Most clusters forbid intensive CPU work and all GPU work on the login node. The
pipeline is arranged so that nothing you need to run interactively violates
that. What runs where:

| Stage | Where | Cost |
| --- | --- | --- |
| `pytest tests -q` | **login node** | ~2 s, pure numpy, no GPU |
| `scripts/compare_ablation.py` | **login node** | seconds; reads scalar JSONL only |
| `--help`, editing, git | **login node** | free |
| UniVTAC install (`scripts/install.sh`) | **compute node** | builds libuipc/cuRobo from source — hours of `nvcc`/CMake |
| `data/download.sh` (assets) | **compute or transfer node** | large download + unpack |
| Dataset conversion | **`slurm/convert.sbatch`** | CPU-only, minutes–hours (video re-encode) |
| Finetuning | **`slurm/finetune.sbatch`** | GPU, hours |
| Evaluation (server + Isaac Sim) | **`slurm/eval_ablation.sbatch`** | GPU, both processes in one job |
| `run_eval.py --dry-run` | **compute node** | needs the live server, so it is GPU work |

Two entries deserve emphasis:

**The UniVTAC install is itself a heavy CPU job.** `scripts/install.sh` compiles
libuipc and cuRobo from source. Do not run it on the login node — use an
interactive allocation (below) or wrap it in a batch job. This surprises people
because "installing dependencies" sounds like login-node work.

**`--dry-run` is not login-node work.** It queries a live GR00T server for the
checkpoint's modality config, so it needs the GPU job that hosts the server.
Only the test suite is truly offline. Run `--dry-run` inside the interactive
session where you started the server, or as the first step of a batch job.

### Interactive allocation for debugging

Batch turnaround is painful while you are still finding the right flags. Grab an
interactive shell on a compute node and iterate there:

```bash
# Adjust the partition/account names to your cluster.
srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 --pty bash

# Then, inside the allocation, run the two processes in one shell:
export REPO_ROOT=~/UniVTAC-GR00T UNIVTAC_ROOT=~/UniVTAC
PYTHONPATH=$REPO_ROOT ~/Isaac-GR00T/.venv/bin/python     -m univtac_groot.server.run_server     --model-path nvidia/GR00T-N1.7-3B     --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --port 5555 &

conda run -n UniVTAC python $REPO_ROOT/scripts/run_eval.py     --task insert_hole --arm baseline --univtac-root $UNIVTAC_ROOT     --port 5555 --dry-run          # then drop --dry-run for --episodes 1
```

Leaving the server running while you re-run the evaluator is the whole point:
the checkpoint load happens once, and each evaluator iteration starts in
seconds. Use `tmux` or `screen` so a dropped SSH connection does not kill the
allocation.

### GPU memory in the evaluation job

The eval job hosts Isaac Sim (scene plus offscreen rendering) *and* GR00T N1.7
(~7 GB in bf16, plus activations) on the same allocation. Budget roughly 24 GB
of VRAM for a single-GPU run. If your nodes are tighter than that, ask for two:

```bash
sbatch --gres=gpu:2 --export=ALL,ARM=baseline,TASK=insert_hole slurm/eval_ablation.sbatch
```

`eval_ablation.sbatch` counts the GPUs SLURM allocated and puts the model on
`cuda:1` and the simulator on `cuda:0` automatically; override with
`SERVER_DEVICE=` / `SIM_DEVICE=` if you want a different split.

Both processes always live in the **same job on the same node**, talking over
`127.0.0.1`. That is deliberate: no cross-node networking to arrange, no
firewall rules, and when the job ends both processes die together (the script
traps `EXIT` and kills the server). If your cluster instead prefers one GPU per
job step, run the server as its own job and pass its node name as
`--host`; the client takes any reachable host.

### Ordering the whole thing

```bash
# on a compute node / interactive allocation, once
bash scripts/install.sh && bash data/download.sh

# on the login node — free, and catches configuration errors early
pytest tests -q

# batch, in order
sbatch --export=ALL,ARM=baseline,TASK=insert_hole slurm/eval_ablation.sbatch   # zero-shot arm

# only if you need the tactile arm (it requires a finetune):
sbatch --array=0-7 --export=ALL,ARM=tactile slurm/convert.sbatch
sbatch --export=ALL,ARM=tactile,DATASET=$DATA_ROOT/univtac-insert_hole-tactile slurm/finetune.sbatch
sbatch --export=ALL,ARM=tactile,TASK=insert_hole,GROOT_MODEL=<ckpt> slurm/eval_ablation.sbatch

# login node again
python scripts/compare_ablation.py --results-dir eval_result --json ablation.json
```

Results are written to JSONL as each episode completes, so a job killed at its
walltime still leaves a usable partial file — `compare_ablation.py` re-aggregates
whatever is there.

## Note on video codecs

`scripts/convert_univtac_to_lerobot.py` writes **H.264**, which matches GR00T's
constraint: `torchcodec` "supports H.264 on all platforms; AV1 decoding is not
guaranteed" (Isaac-GR00T README). The shipped `demo_data` uses AV1 and ships a
converter (`examples/SimplerEnv/convert_av1_to_h264.py`) for exactly this
reason, so H.264 is the safer output for a dataset you intend to train on.
