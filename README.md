# GR00T N1.7 on the UniVTAC benchmark

Finetune [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B)
on the [UniVTAC](https://github.com/univtac/UniVTAC) manipulation benchmark, one
policy per task, and measure its success rate over 100 rollouts.

The observation is the task's camera images (third-person, plus a wrist view on
two tasks), the 17-D robot state, and the language instruction.

**Current state, blockers and next actions: [docs/STATUS.md](docs/STATUS.md).**

| | |
| --- | --- |
| Model | `nvidia/GR00T-N1.7-3B`, finetuned per task |
| Observation | images + 17-D state + language instruction |
| Recipe | 30,000 steps at batch 64, retaining checkpoints 10k/20k/30k |
| Protocol | 100 rollouts per task, UniVTAC's `1_000_000 * (1 + seed)` seeds |
| Output | success rate per task, with a Wilson 95 % interval |

Everything runs headlessly as Slurm batch jobs. No script prompts, opens a
GUI, or needs a login-node GPU.

### Start here

On the **cluster**, where jobs are submitted:

```bash
cp env.example.sh env.sh             # env.sh is gitignored; it never travels with a clone
$EDITOR env.sh && chmod 600 env.sh   # paths + two tokens
source env.sh                        # per session, not ~/.bashrc
mkdir -p logs                        # #SBATCH --output does not create it
```

Setting up a machine from nothing — including the git and SSH steps — is
[Set up a new machine](#set-up-a-new-machine) below.

**[docs/USAGE.md](docs/USAGE.md) — a runnable example for each of the seven
scripts**, plus the site rules you have to satisfy yourself.

[docs/BENCHMARK.md](docs/BENCHMARK.md) is the one to read before quoting a
number: the evaluation protocol, the settings that move it, what to state
alongside it, and how the step count is chosen without fitting the reported
rollouts.

---

## Architecture: two processes, one socket

UniVTAC needs **Python 3.10** (Isaac Sim 4.5, Isaac Lab 2.1.1, cuRobo,
torch 2.5.1+cu118); GR00T N1.7 needs **Python 3.12** (CUDA 12.8, transformers,
flash-attn, and the gated `nvidia/Cosmos-Reason2-2B` backbone). Different minor
Python versions cannot share a virtualenv, so the model is served over a
loopback ZeroMQ socket rather than imported in-process — the same split UniVTAC
uses for its own SmolVLA integration (`policy/smolvla/smolvla_server.py`).

```
┌─ GR00T env (3.12) ──────────────────┐        ┌─ UniVTAC / Isaac Lab env (3.10) ───────┐
│ univtac_groot.server.run_server     │        │ scripts/run_eval.py                    │
│   Gr00tPolicy(GR00T-N1.7-3B)        │        │   UniVTACGr00tEnv  (Gym surface)       │
│   Gr00tSimPolicyWrapper             │◄──────►│   ObsAdapter       (images + state)     │
│   PolicyServer  (ZMQ REP)           │  ZMQ   │   ObsHistory       (delta_indices)     │
└─────────────────────────────────────┘ msgpack│   RecedingHorizonController            │
                                               │   Gr00tClient      (ZMQ REQ)           │
                                               └────────────────────────────────────────┘
```

Both processes always live in the **same job on the same node**, talking over
`127.0.0.1`: no cross-node networking, and when the job ends both die together.
The UniVTAC-side client needs only `numpy`, `pyzmq`, `msgpack` and
`msgpack-numpy` — it never imports `gr00t` or torch-heavy code.

## Set up a new machine

Two machines, and it matters which is which. The **workstation** edits code,
runs the tests and reads results; it needs no GPU, no Isaac Sim and no `gr00t`,
and a few minutes of setup. The **cluster** runs everything that costs
anything, and is hours of setup. Nothing below runs on the cluster on your
behalf — you run it there yourself.

### 1. Workstation (any Linux; commands are Ubuntu 24.04)

```bash
sudo apt update && sudo apt install -y git git-lfs python3-venv
git lfs install
```

Ubuntu 24.04 ships Python 3.12 as **`python3`** — there is no `python` on
`PATH` unless you install `python-is-python3`, so read every `python` in these
docs as `python3` or work inside the venv below, where `python` exists. The
same release enforces PEP 668, so a bare `pip install` into the system
interpreter is refused with `externally-managed-environment`. The venv is not
optional.

**SSH.** The git remote here is `git@github.com-kaist:J1Y2003/UniVTAC` — that
host is an *alias*, not a domain, so a clone fails with
`Could not resolve hostname` until `~/.ssh/config` defines it. Copy your
existing `~/.ssh/config` and the keys it names (`id_ed25519_github_kaist` for
GitHub, `id_ed25519` for the cluster) from the old machine, or generate new
ones and register them with GitHub and the cluster. SSH silently ignores a key
that is group- or world-readable:

```bash
chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519* ~/.ssh/config
chmod 644 ~/.ssh/*.pub
ssh -T git@github.com-kaist          # "You've successfully authenticated" = alias + key work
```

**Clone and test.**

```bash
git clone git@github.com-kaist:J1Y2003/UniVTAC.git && cd UniVTAC
python3 -m venv .venv && source .venv/bin/activate   # .venv is gitignored
pip install -r requirements-dev.txt
pytest tests -q                                      # expect 84 passed, ~2 s
```

That is the whole workstation. The numeric core is pure numpy, so a green test
suite here means the contract is right before you spend a GPU allocation.
`scripts/results_table.py` also runs here against a `eval_result/` you have
copied down — it is standard-library-only by design.

Two things deliberately do **not** come down with a clone, because they are
gitignored: `env.sh` (use `env.example.sh`) and `eval_result/`. Copy the latter
from the cluster with `rsync` if you want to aggregate locally.

### 2. Cluster

**[docs/SETUP.md](docs/SETUP.md) is the full prerequisite list.** The short
version — three checkouts with confusingly similar names:

```bash
# 0. This repo. Jobs are submitted from here.
git clone git@github.com-kaist:J1Y2003/UniVTAC.git
cd UniVTAC && cp env.example.sh env.sh && $EDITOR env.sh && chmod 600 env.sh

# 1. The simulator ($UNIVTAC_ROOT), Python 3.10.
#    Builds Isaac Sim, Isaac Lab, TacEx, cuRobo -- hours, and a compute job,
#    not login-node work.
git clone https://github.com/univtac/UniVTAC.git UniVTAC-sim
cd UniVTAC-sim && bash scripts/install.sh
bash data/download.sh                       # scene assets, via modelscope

# 2. GR00T ($GROOT_ROOT), Python 3.12, separate environment
sudo apt install git-lfs && git lfs install # BEFORE cloning
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T && uv sync --python 3.12     # needs ffmpeg 4-7, not 8

# 3. This repo's client dependencies, into the UniVTAC environment
conda activate UniVTAC && pip install -r requirements-client.txt
```

Then `source env.sh && python scripts/preflight.py --deep`, which checks the
above and prints the next command.

Three things bite almost everyone: `nvidia/Cosmos-Reason2-2B` is **gated**, so
the server dies at load with a 401 unless `HF_TOKEN` has access — export it,
never run `hf auth login`, which overwrites the shared account's stored token;
`torchcodec` cannot load **FFmpeg 8**, which recent Ubuntu ships; and a cuDNN
that is not the pinned `9.10.2.21` costs ~86x silently, which is why
`scripts/check_cudnn.py` runs inside every GPU job.

### 3. Day to day

Edit and test on the workstation, push, pull on the cluster, submit there:

```bash
# workstation
git add -p && git commit && git push

# cluster
cd ~/UniVTAC && git pull && source env.sh
```

Keep it one-directional. The cluster checkout is a place to run from, not to
edit in — a local edit there is invisible to the workstation and to git, and
the eval JSON records the commit that produced it, so an uncommitted change on
the cluster makes a result unreproducible.

## Quickstart

```bash
source env.sh

export TASK=insert_hole

# Finetune. Every flag is yours to pass; see docs/USAGE.md for the full line.
sbatch --job-name=univtac-groot-per-task-finetune-vision-only-$TASK \
       --partition=sjw_alinlab --time=9:00:00 \
  slurm/train.sbatch --dataset-path $DATA_ROOT/univtac-$TASK-baseline_finetuned \
    --output-dir $OUTPUT_ROOT/$TASK-baseline_finetuned --num-gpus 1 \
    --max-steps 30000 --save-steps 10000 --save-total-limit 4

# Evaluate one checkpoint, on `background`. Use a non-zero --seed-offset for
# anything you might select a checkpoint on.
export GROOT_MODEL=$OUTPUT_ROOT/$TASK-baseline_finetuned/checkpoint-30000
export EMBODIMENT_TAG=NEW_EMBODIMENT PORT=25555
sbatch --job-name=univtac-groot-evaluate-one-checkpoint-$TASK-ckpt30000 \
       --partition=background \
  slurm/eval.sbatch --task $TASK --variant baseline_finetuned \
    --univtac-root $UNIVTAC_ROOT --seed-offset 1 \
    --output eval_result/$TASK-ckpt30000.json

# Aggregate (safe on a login node: reads scalars only)
python scripts/results_table.py eval_result
```

Each eval writes `<result-dir>/<task>-<variant>/<task>-ckpt<N>-seed<offset>.json`
— success rate with a Wilson 95 % interval, error/skip/truncation counts, mean
steps, inference timings, and the checkpoint, seed block, git commit and job id
that produced it. The per-episode JSONL sits beside it, so anything can be
recomputed. It is requeue-safe, which `background` requires.

Or run the two processes by hand:

```bash
# terminal 1 — GR00T environment
python -m univtac_groot.server.run_server \
    --model-path <checkpoint> --port 5555

# terminal 2 — UniVTAC environment
python scripts/run_eval.py --task insert_hole --variant baseline_finetuned \
    --univtac-root "$UNIVTAC_ROOT" --port 5555 \
    --episodes 100 --execution-horizon 16
```

`--dry-run` queries the running server and prints the resolved observation
contract, then exits without starting Isaac Sim — the cheapest way to check that
a variant and a checkpoint agree. The full preflight ladder is in
[docs/SETUP.md](docs/SETUP.md#preflight-cheapest-first).

### Using UniVTAC's own harness instead

`policy/GR00T/` is a drop-in plug-in implementing UniVTAC's `docs/Deploy.md`
contract, so the same adapters work under `eval_policy.sh`:

```bash
cp -r policy/GR00T "$UNIVTAC_ROOT/policy/"
cd "$UNIVTAC_ROOT"
bash eval_policy.sh insert_hole demo GR00T/deploy_baseline 0
```

## What's here

```
univtac_groot/
  spec.py               Observation contracts + GR00T's hard limits
  obs_adapter.py        UniVTAC obs -> flat video.*/state.* keys
  history.py            delta_indices stacking (mirrors GR00T's MultiStepWrapper)
  action_adapter.py     Action chunk -> take_action vectors; gripper convention
  receding_horizon.py   Chunk cache + execution horizon
  client.py             ZeroMQ client, wire-compatible with GR00T's PolicyServer
  env_wrapper.py        Gymnasium surface over UniVTAC's BaseTask
  rollout.py            Episode loop: seeding, logging, error handling
  metrics.py            JSONL results, success rates, Wilson intervals
  variants.py           The variants and the per-task camera table
  server/run_server.py  GR00T-side inference server
policy/GR00T/           Drop-in plug-in for UniVTAC's own evaluator
configs/modality/       GR00T ModalityConfigs for the finetuned variants
scripts/
  preflight.py                   Setup checklist; prints the next command
  check_cudnn.py                 cuDNN pin guard; runs inside every GPU job
  run_eval.py                    Headless eval driver
  results_table.py               Success rate per task, from the library
  results_plot.py                The same numbers as a graph (needs matplotlib)
  convert_univtac_to_lerobot.py  UniVTAC HDF5 -> GR00T LeRobot v2
  wandb_report.py                Read a run's training + system metrics back
  eval_progress.py               How far along the running evals are
  eval_triage.py                 Why a run's episodes errored or were skipped
  check_eval_logs.py             Cross-check logs/ against eval_result/
  recover_eval.py                Rebuild a summary from a damaged JSONL
slurm/
  train.sbatch            exec launch_finetune.py with your flags
  train_bundle.sh         the same, through bundle-sbatch
  eval.sbatch             server + evaluator in one GPU job
  eval_bundle.sh          the same, through bundle-sbatch
docs/USAGE.md           A runnable example for each script  <- start here
docs/SETUP.md           Prerequisites, environments, common failures
docs/STATUS.md          Current state, decisions in force, failure -> cause
docs/BENCHMARK.md       The protocol and what moves the number
docs/UPSTREAM.md        Every upstream fact this code relies on, with citations
tests/                  No GPU, Isaac Sim, or gr00t needed
```

## Tests

The numeric core is pure numpy, so the contract is checkable before you spend a
GPU allocation:

```bash
pip install -r requirements-dev.txt
pytest tests -q
```
