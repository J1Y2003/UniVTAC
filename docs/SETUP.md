# Setup: what must exist before you run anything

> For the **ordered sequence** of commands, see [USAGE.md](USAGE.md), or run
> `python scripts/preflight.py`. This document explains the *why* behind each
> prerequisite and lists failure modes.

## Why it is a separate process, not an import

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

### 3. cuDNN must match torch's pin

`torch==2.9.0+cu128` pins `nvidia-cudnn-cu12==9.10.2.21`. Anything else either
fails with `CUDNN_STATUS_NOT_INITIALIZED` — which reads like a too-old driver
and is not one — or loads and runs **~86x slower** on GR00T's vision tower with
no error at all.

Check it by asking the loaded library its own version. Package metadata is not
evidence: `uv pip list` reports the pin while the files on disk are a different
release. Import `torch` first so the venv's copy is the one found:

```bash
$GROOT_PYTHON -c "import ctypes, torch; l=ctypes.CDLL('libcudnn.so.9'); \
  l.cudnnGetVersion.restype=ctypes.c_size_t; print(l.cudnnGetVersion())"
```

You want `91002` (the encoding is `MAJOR*10000 + MINOR*100 + PATCH`, so `91300`
is 9.13.0). If it is anything else:

```bash
cd $GROOT_ROOT
env -u CONDA_PREFIX -u VIRTUAL_ENV uv cache clean nvidia-cudnn-cu12
env -u CONDA_PREFIX -u VIRTUAL_ENV uv pip install --python .venv/bin/python \
    --reinstall nvidia-cudnn-cu12==9.10.2.21
```

The `uv cache clean` is not optional: uv hardlinks these `.so` files out of its
content-addressed cache, so a plain `--reinstall` can re-link the very files you
are replacing. If `9.10.2.21` still will not initialise, try the pin-compatible
`9.8.0.87` or `9.7.1.26`, then ask an admin about `cuda-compat-12-8` or a driver
upgrade to r570+.

**You should not have to remember any of this.** `scripts/preflight.py --deep`
runs the check from a login node, and `scripts/check_cudnn.py` runs it inside
every GPU job before any weights load, aborting with the commands above on a
definite mismatch and only warning when it cannot tell. There is deliberately
no way to disable cuDNN here: the penalty is silent, so an available switch is
worse than none. (For why it costs 86x, and the wandb signature of the
170 s/step run that found it: `git log --grep=cuDNN`.)

### 4. Hugging Face access to a gated model (required)

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

* **`HF_HOME` moves the token file too.** `slurm/eval_task.sbatch` defaults
  `HF_HOME` to scratch, so a token stored under the default
  `~/.cache/huggingface` is not visible inside the job. Exporting `HF_TOKEN`
  sidesteps this, and the job warns when neither is present.
* **Never put the token in a tracked file or in an `#SBATCH` line** -- job
  scripts are often world-readable. Keep it in your environment
  (`chmod 600` any file that holds it); the job inherits it from the
  submitting shell.

### 5. UniVTAC assets and (optionally) demonstration data

The task scenes need UniVTAC's assets. Its dataset/assets come from ModelScope:

```bash
cd UniVTAC && bash data/download.sh    # installs modelscope, pulls byml2024/UniVTAC
```

Demonstration data is needed for every task you intend to benchmark: the
reported model is a finetune, so there is no data-free path — see
[BENCHMARK.md](BENCHMARK.md#why-there-is-no-zero-shot-number).

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
where `uv sync` installed `gr00t`. `slurm/eval_task.sbatch` sets
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

Checkpoints belong in the cluster's unified folder:

| Cluster | NFS root |
| --- | --- |
| Kakao | `/rlwrld-unified-checkpoints` |
| Naver (MLXP) | `/data/rlwrld-unified-checkpoints` |
| AWS (SKT) | `/fsx/rlwrld-unified-checkpoints` |

Our area under the Kakao root is **`/rlwrld-unified-checkpoints/jimin/jaewon`**
(`jimin` is the shared account, `jaewon` is ours within it). That is what
`univtac_output_root` in `slurm/common.sh` resolves to, overridable with
`OUTPUT_ROOT`, and `finetune.sbatch` refuses an `OUTPUT_DIR` outside
`/rlwrld-unified-checkpoints` unless `ALLOW_NONSTANDARD_OUTPUT=1`.

One directory per `<task>-<variant>`, the same path on every submission, which
is what lets a resubmitted finetune find its own previous checkpoints and
resume. See [USAGE.md](USAGE.md).

**Retention.** Managed storage is not permanent: retention may migrate and
later delete aged outputs (the documented window was 4 days untouched before
migration, 90 days to deletion). Nothing here copies checkpoints out to a
private directory -- duplicating ~26 GB per checkpoint is the sprawl the
storage policy exists to prevent. If a checkpoint ages out, resubmit the
finetune: ~6.2 h per task is cheaper than a shadow copy of everything.

## Common failures

| Symptom | Cause |
| --- | --- |
| `GatedRepoError` / `401` on server start | No access to `nvidia/Cosmos-Reason2-2B`, or not logged in |
| `wait_until_ready` times out after 900 s | Server died on load — **read the server log**, not the evaluator's |
| `Embodiment tag 'NEW_EMBODIMENT' is not supported by this checkpoint` | Expected: `NEW_EMBODIMENT` needs a finetune, see [BENCHMARK.md](BENCHMARK.md) |
| `state key mismatch: the checkpoint's embodiment declares ...` | Variant and checkpoint disagree; wrong `--variant` or wrong `--model-path` |
| `Could not load libtorchcodec ... versions 4, 5, 6 and 7` | FFmpeg 8 installed; downgrade to <8 |
| Parquet files in `demo_data/` unreadable | Cloned Isaac-GR00T without `git-lfs` |
| `CUDA_HOME is unset` during finetune | Run GR00T's `scripts/deployment/dgpu/install_deps.sh` |
| `CUDNN_STATUS_NOT_INITIALIZED`, cuDNN debug log says `cudaGetDeviceCount(&count) != cudaSuccess` with `GPU=NULL` and compute capability `0.0` | **The cuDNN in the GR00T venv is not the one torch pins.** Reads like a driver problem and is not. Run `python scripts/preflight.py --deep`, which checks it; fix per [cuDNN](#3-cudnn-must-match-torchs-pin). |
| `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` on the first `get_action` | The server inherited `LD_LIBRARY_PATH`/`CUDA_HOME` from the UniVTAC conda env (CUDA 12.4) while its torch is cu128. Launch it with `env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH`; the job scripts do this automatically. Check free VRAM first, since genuine OOM reports the same error. |
| `Arm motion planning failed on action 0` | cuRobo, not GR00T. Verify UniVTAC's own expert works: `bash collect_data.sh grasp_classify demo 0` |
| `ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available` | The flag is a hard error, not a fallback, and it fires mid-download inside the *server* log so it reads like a checkpoint fault. `eval_task.sbatch` now probes `GROOT_PYTHON` for the package and only enables the flag when present. Override with `HF_HUB_ENABLE_HF_TRANSFER=0`, or install it: `$GROOT_PYTHON -m pip install hf_transfer` (worth it for the ~15 GB of weights). |
| `GPU 파티션에는 GPU를 요청한 잡만 제출할 수 있습니다` | A CPU-only job went to a GPU partition. Add `--partition=cpu`. Applies to `download_data.sbatch` and `convert.sbatch`; both set it in their headers, but an exported `SBATCH_PARTITION` beats a `#SBATCH` directive, so pass it on the command line too. |
| `sbatch: error: ... Batch job submission failed: Unspecified error` | This cluster's submit filter enforces site rules and rejects the job before it queues. Known rules: a job name **longer than 50 characters**, `--wckey=project-short-name:sub_4dpdata` (the `project-short-name:` prefix is **literal**, not a placeholder -- without it the filter answers `WCKey를 project-short-name:<name> 형식으로 지정해야 합니다`), and **no** `--cpus-per-task` or `--mem` (jobs take the node's per-GPU defaults). `--time` is allowed, but must not exceed the partition maximum. `bash slurm/submit_benchmark.sh --print` shows the exact command without submitting, and `sbatch --test-only` rehearses one. |
| Port already in use with concurrent jobs | `eval_task.sbatch` derives a per-job port from `SLURM_JOB_ID`; pass `PORT=` to override |

---

## Working from an SSH login node

Most clusters forbid intensive CPU work and all GPU work on the login node. The
pipeline is arranged so that nothing you need to run interactively violates
that. What runs where:

| Stage | Where | Cost |
| --- | --- | --- |
| `pytest tests -q` | **login node** | ~2 s, pure numpy, no GPU |
| `scripts/results_table.py` | **login node** | seconds; reads the result JSONs only |
| `--help`, editing, git | **login node** | free |
| UniVTAC install (`scripts/install.sh`) | **compute node** | builds libuipc/cuRobo from source — hours of `nvcc`/CMake |
| `data/download.sh` (assets) | **compute or transfer node** | large download + unpack |
| Dataset conversion | **`slurm/convert.sbatch`** | CPU-only, minutes–hours (video re-encode) |
| Finetuning | **`slurm/finetune.sbatch`** | GPU, hours |
| Evaluation (server + Isaac Sim) | **`slurm/eval_task.sbatch`** | GPU, both processes in one job |
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
of VRAM for a single-GPU run. If your nodes are tighter than that, ask for two
with `GPUS=2 bash slurm/eval_checkpoint.sh ...`.

`eval_task.sbatch` counts the GPUs SLURM allocated and puts the model on
`cuda:1` and the simulator on `cuda:0` automatically; override with
`SERVER_DEVICE=` / `SIM_DEVICE=` if you want a different split.

Both processes always live in the **same job on the same node**, talking over
`127.0.0.1`. That is deliberate: no cross-node networking to arrange, no
firewall rules, and when the job ends both processes die together (the script
traps `EXIT` and kills the server). If your cluster instead prefers one GPU per
job step, run the server as its own job and pass its node name as
`--host`; the client takes any reachable host.

### Ordering the whole thing

[USAGE.md](USAGE.md) is the ordered path. In short:

```bash
python scripts/preflight.py --deep       # login node, free
pytest tests -q                          # login node, free
bash slurm/submit_benchmark.sh --dry     # full preflight, submits nothing
bash slurm/submit_benchmark.sh           # one finetune per task
# then, once a finetune finishes, one eval job per checkpoint:
bash slurm/eval_checkpoint.sh --task insert_hole --seed-offset 1 \
    --checkpoint $OUTPUT_ROOT/insert_hole-baseline_finetuned/checkpoint-30000
```

Jobs are submitted with plain `sbatch`. `slurm/common.sh` encodes the site
rules -- the wckey, the job-name floor, the rejected options -- and validates
before anything is sent. Submitting a job script by hand works too, as long as
you pass `UNIVTAC_JOB_CONFIG=1` and the job's configuration; see the header of
each script. [USAGE.md](USAGE.md) is the reference.

Results are written to JSONL as each episode completes, so a job killed at its
walltime still leaves a usable partial file — `results_table.py` re-aggregates
whatever is there, and `eval_task.sbatch` resumes from `max(seed)+1` rather
than replaying the block.

## Note on video codecs

`scripts/convert_univtac_to_lerobot.py` writes **H.264**, which matches GR00T's
constraint: `torchcodec` "supports H.264 on all platforms; AV1 decoding is not
guaranteed" (Isaac-GR00T README). The shipped `demo_data` uses AV1 and ships a
converter (`examples/SimplerEnv/convert_av1_to_h264.py`) for exactly this
reason, so H.264 is the safer output for a dataset you intend to train on.
