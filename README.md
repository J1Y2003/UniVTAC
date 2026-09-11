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

```bash
$EDITOR env.sh && chmod 600 env.sh   # paths + two tokens; gitignored
source env.sh                        # per session, not ~/.bashrc
mkdir -p logs                        # #SBATCH --output does not create it
```

**[docs/USAGE.md](docs/USAGE.md) — a runnable example for each of the six
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

## Install

**[docs/SETUP.md](docs/SETUP.md) is the full prerequisite list.** The short
version:

```bash
# 1. UniVTAC, Python 3.10 (builds Isaac Sim, Isaac Lab, TacEx, cuRobo -- hours)
git clone https://github.com/univtac/UniVTAC.git && cd UniVTAC && bash scripts/install.sh
bash data/download.sh                       # scene assets, via modelscope

# 2. Isaac-GR00T, Python 3.12, separate environment
sudo apt install git-lfs && git lfs install # BEFORE cloning
git clone https://github.com/NVIDIA/Isaac-GR00T.git && cd Isaac-GR00T
uv sync --python 3.12                       # needs ffmpeg 4-7, not 8

# 3. This repo's client dependencies, into the UniVTAC environment
conda activate UniVTAC && pip install -r requirements-client.txt
```

Three things bite almost everyone: `nvidia/Cosmos-Reason2-2B` is **gated**, so
the server dies at load with a 401 unless `HF_TOKEN` has access — export it,
never run `hf auth login`, which overwrites the shared account's stored token;
`torchcodec` cannot load **FFmpeg 8**, which recent Ubuntu ships; and a cuDNN
that is not the pinned `9.10.2.21` costs ~86x silently, which is why
`scripts/check_cudnn.py` runs inside every GPU job.

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
    --output eval_result/$TASK-ckpt30000.jsonl

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
  convert_univtac_to_lerobot.py  UniVTAC HDF5 -> GR00T LeRobot v2
  wandb_report.py                Read a run's training + system metrics back
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
