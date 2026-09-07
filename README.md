# GR00T N1.7 on the UniVTAC benchmark

An evaluation pipeline for benchmarking [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B)
on the [UniVTAC](https://github.com/univtac/UniVTAC) visuo-tactile manipulation
benchmark, set up as a two-arm ablation:

| Arm | Observation | Checkpoint |
| --- | --- | --- |
| **Baseline** | vision + language + 1-D proprioception + embodiment id | zero-shot on the released model |
| **Tactile** | the same, with the flattened UniVTAC tactile array concatenated onto the state vector | **requires a finetune** — see [docs/ABLATION.md](docs/ABLATION.md) |

Everything runs headlessly under `sbatch`. No script prompts, opens a GUI, or
needs a login-node GPU.

### Start here

```bash
cp env.example.sh env.sh && chmod 600 env.sh   # paths + HF token; gitignored
source env.sh                                  # per session, not ~/.bashrc
```


**[docs/RUNBOOK.md](docs/RUNBOOK.md) — the ordered sequence** from nothing to an
evaluation number, each step labelled with where it runs and which conda env.
Or just ask the repo where you are:

```bash
python scripts/preflight.py --deep     # checklist + the exact next command
```

> **Read [docs/ABLATION.md](docs/ABLATION.md) before running the tactile arm.**
> Adding tactile dimensions changes the state layout, and GR00T's state
> projector is embodiment-conditioned, so the tactile arm only runs under the
> `NEW_EMBODIMENT` tag — which ships in no released checkpoint. The baseline arm
> runs zero-shot today; the tactile arm needs a finetune first. This is a
> property of the model, not of this code.

---

## Architecture: two processes, one socket

UniVTAC needs **Python 3.10** (Isaac Sim 4.5, Isaac Lab 2.1.1, TacEx, cuRobo,
torch 2.5.1+cu118); GR00T N1.7 needs **Python 3.12** (CUDA 12.8, transformers,
flash-attn, and the gated `nvidia/Cosmos-Reason2-2B` backbone). Different minor
Python versions means they cannot share a virtualenv at all, so the model is
served over a loopback ZeroMQ socket rather than imported in-process — the same
split UniVTAC uses for its own SmolVLA integration
(`policy/smolvla/smolvla_server.py`).

```
┌─ GR00T env ─────────────────────────┐        ┌─ UniVTAC / Isaac Lab env ──────────────┐
│ univtac_groot.server.run_server     │        │ scripts/run_eval.py                    │
│   Gr00tPolicy(GR00T-N1.7-3B)        │        │   UniVTACGr00tEnv  (Gym surface)       │
│   Gr00tSimPolicyWrapper             │◄──────►│   ObsAdapter       (state + tactile)   │
│   PolicyServer  (ZMQ REP)           │  ZMQ   │   ObsHistory       (delta_indices)     │
└─────────────────────────────────────┘ msgpack│   RecedingHorizonController            │
                                               │   Gr00tClient      (ZMQ REQ)           │
                                               └────────────────────────────────────────┘
```

The UniVTAC-side client needs only `numpy`, `pyzmq`, `msgpack` and
`msgpack-numpy` — it never imports `gr00t` or torch-heavy code.

## Install

**[docs/SETUP.md](docs/SETUP.md) is the full prerequisite list**, including what
"a live GR00T server" means, the preflight sequence, and a failure table. The
short version:

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

Two things bite almost everyone: `nvidia/Cosmos-Reason2-2B` is **gated**, so
request access and `huggingface-cli login` or the server dies at load with a
401; and `torchcodec` cannot load **FFmpeg 8**, which recent Ubuntu ships.

## Quickstart

```bash
export REPO_ROOT=$PWD
export UNIVTAC_ROOT=~/UniVTAC
export UNIVTAC_PYTHON=$(conda run -n UniVTAC which python)
export GROOT_PYTHON=~/Isaac-GR00T/.venv/bin/python   # uv's venv, not a conda env

# Baseline, one task, 50 episodes
sbatch --export=ALL,ARM=baseline,TASK=insert_hole slurm/eval_ablation.sbatch

# The full sweep: both arms x eight tasks
TACTILE_MODEL=/ckpt/univtac-tactile/checkpoint-20000 bash slurm/submit_ablation.sh

# Aggregate (safe on a login node: reads scalars only)
python scripts/compare_ablation.py --results-dir eval_result --json ablation.json
```

Or run the two processes by hand:

```bash
# terminal 1 — GR00T environment
python -m univtac_groot.server.run_server \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --port 5555

# terminal 2 — UniVTAC environment
python scripts/run_eval.py --task insert_hole --arm baseline \
    --univtac-root "$UNIVTAC_ROOT" --port 5555 \
    --episodes 50 --execution-horizon 8
```

`--dry-run` queries the running server and prints the resolved observation
contract, then exits without starting Isaac Sim — the cheapest way to check that
an arm and a checkpoint agree. The full preflight ladder is in
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
  spec.py               Observation/tactile contracts + GR00T's hard limits
  obs_adapter.py        UniVTAC obs -> flat video.*/state.* keys; tactile encoders
  history.py            delta_indices stacking (mirrors GR00T's MultiStepWrapper)
  action_adapter.py     Action chunk -> take_action vectors; gripper convention
  receding_horizon.py   Chunk cache + execution horizon
  client.py             ZeroMQ client, wire-compatible with GR00T's PolicyServer
  env_wrapper.py        Gymnasium surface over UniVTAC's BaseTask
  rollout.py            Episode loop: seeding, logging, error handling
  metrics.py            JSONL results, success rates, Wilson intervals
  arms.py               The two ablation arms as one shared spec
  server/run_server.py  GR00T-side inference server
policy/GR00T/           Drop-in plug-in for UniVTAC's own evaluator
configs/modality/       GR00T ModalityConfigs for the finetuned arms
scripts/
  run_eval.py                    Headless eval driver
  compare_ablation.py            Ablation table
  convert_univtac_to_lerobot.py  UniVTAC HDF5 -> GR00T LeRobot v2 (for finetuning)
slurm/
  install_univtac.sbatch  Batch-safe wrapper for UniVTAC's install.sh
  eval_ablation.sbatch    Server + evaluator in one GPU job
  convert.sbatch          Dataset conversion (CPU-only, array-capable)
  finetune.sbatch         GR00T finetune for an arm
  submit_ablation.sh      Sweep submitter
  preflight.py                   Setup checklist; prints the next command
docs/RUNBOOK.md         Ordered: nothing -> an evaluation number  <- start here
docs/SETUP.md           Prerequisites, environments, common failures
docs/ABLATION.md        Experimental design, constraints, calibration checks
docs/UPSTREAM.md        Every upstream fact this code relies on, with citations
tests/                  94 tests; no GPU, Isaac Sim, or gr00t needed
```

## Tests

The numeric core is pure numpy, so the contract is checkable before you spend a
GPU allocation:

```bash
pip install -r requirements-dev.txt
pytest tests -q          # 94 passed
```

`tests/test_client_integration.py` runs the full adapter → history → batching →
socket → chunk → action path against a stand-in server that performs the same
validation as `Gr00tSimPolicyWrapper.check_observation`.

## Design notes

**Tactile is imagery, not a low-dimensional array.** UniVTAC's sensors are
camera-based (`envs/sensors/tactile.py` wraps TacEx GelSight Mini / GF225 /
XenseWS), so a raw reading is a 320×240 gel height map or a 64-marker motion
field — not a small vector. `TactileSpec` average-pools it to a configurable
grid (default 8×6 = 48 dims per sensor) before concatenation, giving a 113-D
state against GR00T's 132-D cap. `--tactile-mode video` instead passes tactile
RGB as extra video streams, which is the architecturally natural route for
camera-based sensors.

**Horizons come from the checkpoint, not from constants.** N1.7 predicts 40-step
chunks by default (`GR00T_N1d7Config.action_horizon`), the shipped posttrain
configs use 16 or 8, and the DROID tag wants a two-frame observation history
(`video_delta_indices = [-15, 0]`). `resolve_spec_from_policy` reads the live
`get_modality_config` and sizes the history buffer and controller from it, and
fails with an actionable message when an arm and a checkpoint disagree.

**Results survive a walltime kill.** Episodes are appended to JSONL as they
finish and never held in memory beyond scalars; `summarize_jsonl` re-aggregates
a partial file.

## Sources

- [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — policy, server, embodiment tags, modality configs
- [univtac/UniVTAC](https://github.com/univtac/UniVTAC) — environments, tactile sensors, deploy contract
- [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) · [GR00T N1.7 announcement](https://huggingface.co/blog/nvidia/gr00t-n1-7)
- [UniVTAC paper](https://arxiv.org/abs/2602.10093) · [project page](https://univtac.github.io/)

Exact file-and-line provenance for every assumption is in
[docs/UPSTREAM.md](docs/UPSTREAM.md).
