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

## cuDNN: check the library, not the metadata

**Requirement: the cuDNN on disk must match torch's pin.** `torch==2.9.0+cu128`
pins `nvidia-cudnn-cu12==9.10.2.21`. Anything else fails with
`CUDNN_STATUS_NOT_INITIALIZED`, which reads like a too-old driver and is not
one.

**Check it by asking the loaded library its own version.** Package metadata is
not evidence: `uv pip list` reports the pinned version while the files on disk
are a different release, and cuBLAS keeps working (CUDA minor-version
compatibility covers the runtime but says nothing about a separate shared
library).

```bash
$GROOT_PYTHON -c "import ctypes, torch; l=ctypes.CDLL('libcudnn.so.9'); \
  l.cudnnGetVersion.restype=ctypes.c_size_t; print(l.cudnnGetVersion())"
```

Want `91002` (encoding is `MAJOR*10000 + MINOR*100 + PATCH`, so `91300` is
9.13.0). `scripts/preflight.py --deep` performs exactly this check and fails on
a mismatch.

**Fix:**

```bash
cd $GROOT_ROOT
env -u CONDA_PREFIX -u VIRTUAL_ENV uv cache clean nvidia-cudnn-cu12
env -u CONDA_PREFIX -u VIRTUAL_ENV uv pip install --python .venv/bin/python \
    --reinstall nvidia-cudnn-cu12==9.10.2.21
```

The `uv cache clean` is not optional. uv hardlinks these `.so` files out of its
content-addressed cache -- `ls -la` shows a link count above 1 -- so a plain
`--reinstall` can re-link the very files you are replacing.

**If the pinned version still will not initialise**, in order of preference: an
older pin-compatible cuDNN (`9.8.0.87`, `9.7.1.26`); the cluster's container
path (`srun --container`, available here -- there is no module system and no
`enroot`/`apptainer` binary); a driver upgrade to r570+ or `cuda-compat-12-8`
from an admin.

## Slow training steps

**There is deliberately no way to run without cuDNN in this repo.** No
`DISABLE_CUDNN` flag, no `--disable-cudnn` on the server, no
`sitecustomize` hook. They existed, and removing them is the point: running
without cuDNN costs ~86x and is the reason a training step once took 170 s
instead of 1.89 s. Because the penalty is silent -- no error, just a job that
never finishes -- an available switch is worse than no switch.

What replaces it: **`scripts/check_cudnn.py` runs automatically in every GPU
job**, from `finetune.sbatch` and `eval_ablation.sbatch`, before any weights
load. It asks the loaded library its own version, compares it against torch's
pin, runs a real GPU convolution, and on a definite mismatch aborts the job
printing the reinstall commands above. It fails only on evidence -- an
unreadable pin or an absent GPU is a warning, never a block --- and
`SKIP_CUDNN_CHECK=1` skips the check itself without changing how the job runs.
`scripts/preflight.py --deep` performs the same check from a login node.

Disabling cuDNN looks like a cheap trade -- only convolutions use cuDNN, and
FlashAttention-2 and the DiT's SDPA path do not. But GR00T has one conv on the
hot path: Qwen3-VL's vision patch embed. Qwen3VL reshapes every visual patch
into its own batch element (`transformers/models/qwen3_vl/modeling_qwen3_vl.py`,
class `Qwen3VLVisionPatchEmbed`), so
`Conv3d(3, 1024, (2,16,16), stride=(2,16,16))` runs over ~32,768 batch elements
per step -- 64 samples x 2 cameras x 256 patches at 256 px. Without cuDNN, ATen
walks that batch with a per-element im2col loop driven from the main Python
thread. GR00T's own `_apply_vision_patch_embed_channels_last` workaround exists
precisely to select the fast *cuDNN* kernel here, so disabling cuDNN disables
what that workaround reaches for.

Measured on an A100 (sm_80), vision-tower forward+backward at 32 images:

| Patch embed | Time | |
| --- | --- | --- |
| shipped `Conv3d`, cuDNN off | 46.16 s | 98.8% of the whole tower |
| algebraically identical matmul | 0.53 s | **86x faster** |

Scaled to the real 128-image batch that is ~185 s against an observed 170 s
step: with cuDNN off this single conv is essentially the entire training step,
and a 10,000-step finetune projects to 470 hours.

**The signature, if you meet it again.** GPU utilisation ~48% but power only
~110 W of a 400 W limit, SM clocks pinned at ~1400 MHz, **~0% of time spent
accessing memory**, and the main Python thread pegged at ~100% of one core with
almost no system time while the dataloader workers sit idle. Kernels resident
half the time while drawing no power and moving no memory is a tiny-kernel
launch flood driven from Python, not data starvation. **Read power draw, not
utilisation.**

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
tactile variant) — see [ABLATION.md](ABLATION.md).

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
$UNIVTAC_PYTHON scripts/run_eval.py --task insert_hole --variant baseline \
    --port 5555 --dry-run

# 6. A single real episode before committing a sweep
$UNIVTAC_PYTHON scripts/run_eval.py --task insert_hole --variant baseline \
    --univtac-root $UNIVTAC_ROOT --port 5555 --episodes 1
```

Step 5 talks to the live server and prints the resolved horizons, state keys and
video keys, then exits — it is the fastest way to catch an variant/checkpoint
mismatch. Note that it does need the server (it calls `get_modality_config`);
there is no fully offline contract check, because the authoritative answer lives
in the checkpoint. Keep the server from step 4 running while you iterate on
steps 5 and 6 — reloading the checkpoint per attempt is the main time sink.

## Training-outputs policy (Kakao cluster)

**Superseded by `bundle-sbatch`.** Checkpoints no longer go to a path we
choose. The launcher creates one output bundle per submission and injects

```
MODEL_OUTPUT_DIR = CODE_OUTPUT_DIR = <bundle>/code-output
```

and the guide is explicit that we must not supply `MODEL_OUTPUT_DIR`
ourselves. The unified-folder roots below are what the previous policy
required, kept only so an old path in someone's `env.sh` is recognisable:

| Cluster | NFS root (retired for our jobs) |
| --- | --- |
| Kakao | `/rlwrld-unified-checkpoints` |
| Naver (MLXP) | `/data/rlwrld-unified-checkpoints` |
| AWS (SKT) | `/fsx/rlwrld-unified-checkpoints` |

**Retention is the part that bites a multi-week study, and bundles do not
change that.** Bundle storage is managed storage, not permanent storage: the
guide says retention may migrate and later delete aged outputs, and tells you
to move long-lived artifacts to designated user storage yourself. The old
unified-folder window was 4 days untouched (§5.1 body; its table says 7 -- the
document contradicts itself) before migration to Object Storage and deletion 90
days later; the bundle window is not documented, so plan for the same. A
checkpoint you want to re-evaluate weeks later will not be there, and
`<output_dir>/final` becomes a dangling symlink.

It is worse than it used to be: a rescued checkpoint is now also the only way
to **resume** that training, since a resubmission gets a fresh bundle and must
be pointed at its predecessor with `--checkpoint`.

§7 gives the remedy: write to the unified folder first, then move anything
needing permanent retention into your own user folder. That is what
`slurm/preserve_outputs.sh` does -- it copies only the *final* checkpoint per
variant and lets intermediates expire, because each is ~40 GB with optimizer
state:

```bash
bash slurm/preserve_outputs.sh --check   # what exists, days until archival
bash slurm/preserve_outputs.sh           # copy finals to KEEP_ROOT
```

Run it as soon as a finetune finishes, not days later.

## Common failures

| Symptom | Cause |
| --- | --- |
| `GatedRepoError` / `401` on server start | No access to `nvidia/Cosmos-Reason2-2B`, or not logged in |
| `wait_until_ready` times out after 900 s | Server died on load — **read the server log**, not the evaluator's |
| `Embodiment tag 'NEW_EMBODIMENT' is not supported by this checkpoint` | Expected: `NEW_EMBODIMENT` needs a finetune, see [ABLATION.md](ABLATION.md) |
| `state key mismatch: the checkpoint's embodiment declares ...` | Variant and checkpoint disagree; wrong `--variant` or wrong `--model-path` |
| `--tactile-mode depth_pool needs observations.tactile to include 'depth'` | Add it to `UniVTAC/task_config/<config>.yml` |
| `Could not load libtorchcodec ... versions 4, 5, 6 and 7` | FFmpeg 8 installed; downgrade to <8 |
| Parquet files in `demo_data/` unreadable | Cloned Isaac-GR00T without `git-lfs` |
| `CUDA_HOME is unset` during finetune | Run GR00T's `scripts/deployment/dgpu/install_deps.sh` |
| `CUDNN_STATUS_NOT_INITIALIZED`, cuDNN debug log says `cudaGetDeviceCount(&count) != cudaSuccess` with `GPU=NULL` and compute capability `0.0` | **The cuDNN in the GR00T venv is not the one torch pins.** Reads like a driver problem and is not. Run `python scripts/preflight.py --deep`, which checks it; fix per [cuDNN](#cudnn-check-the-library-not-the-metadata). |
| `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` on the first `get_action` | The server inherited `LD_LIBRARY_PATH`/`CUDA_HOME` from the UniVTAC conda env (CUDA 12.4) while its torch is cu128. Launch it with `env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH`; the job scripts do this automatically. Check free VRAM first, since genuine OOM reports the same error. |
| `Arm motion planning failed on action 0` | cuRobo, not GR00T. Verify UniVTAC's own expert works: `bash collect_data.sh grasp_classify demo 0` |
| `ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available` | The flag is a hard error, not a fallback, and it fires mid-download inside the *server* log so it reads like a checkpoint fault. `eval_ablation.sbatch` now probes `GROOT_PYTHON` for the package and only enables the flag when present. Override with `HF_HUB_ENABLE_HF_TRANSFER=0`, or install it: `$GROOT_PYTHON -m pip install hf_transfer` (worth it for the ~15 GB of weights). |
| `GPU 파티션에는 GPU를 요청한 잡만 제출할 수 있습니다` | A CPU-only job went to a GPU partition. Add `--partition=cpu`. Applies to `download_data.sbatch` and `convert.sbatch`; both set it in their headers, but an exported `SBATCH_PARTITION` beats a `#SBATCH` directive, so pass it on the command line too. |
| `sbatch: error: ... Batch job submission failed: Unspecified error` | This cluster's submit filter enforces site rules and rejects the job before it queues. Known rules: a job name **longer than 50 characters**, `--wckey=project-short-name:sub_4dpdata` (the `project-short-name:` prefix is **literal**, not a placeholder -- without it the filter answers `WCKey를 project-short-name:<name> 형식으로 지정해야 합니다`), and **no** `--cpus-per-task` or `--mem` (jobs take the node's per-GPU defaults). `--time` is allowed, but must not exceed the partition maximum. `MODEL_OUTPUT_DIR` is still required but bundle-sbatch injects it -- do not set it yourself. There is no `--test-only` rehearsal any more (the launcher owns that flag); `bash slurm/submit_benchmark.sh --print` shows the exact command instead. |
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
srun --gres=gpu:1 --wckey=project-short-name:sub_4dpdata --pty bash

# Then, inside the allocation, run the two processes in one shell:
export REPO_ROOT=~/UniVTAC-GR00T UNIVTAC_ROOT=~/UniVTAC
PYTHONPATH=$REPO_ROOT ~/Isaac-GR00T/.venv/bin/python     -m univtac_groot.server.run_server     --model-path nvidia/GR00T-N1.7-3B     --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --port 5555 &

conda run -n UniVTAC python $REPO_ROOT/scripts/run_eval.py     --task insert_hole --variant baseline --univtac-root $UNIVTAC_ROOT     --port 5555 --dry-run          # then drop --dry-run for --episodes 1
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
sbatch --wckey=project-short-name:sub_4dpdata --gres=gpu:2 --export=ALL,VARIANT=baseline,TASK=insert_hole slurm/eval_ablation.sbatch
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
sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,VARIANT=baseline,TASK=insert_hole slurm/eval_ablation.sbatch   # zero-shot variant

# only if you need the tactile variant (it requires a finetune):
sbatch --wckey=project-short-name:sub_4dpdata --partition=cpu --array=0-7 --export=ALL,VARIANT=tactile slurm/convert.sbatch
sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,VARIANT=tactile,DATASET=$DATA_ROOT/univtac-insert_hole-tactile slurm/finetune.sbatch
sbatch --wckey=project-short-name:sub_4dpdata --export=ALL,VARIANT=tactile,TASK=insert_hole,GROOT_MODEL=<ckpt> slurm/eval_ablation.sbatch

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
