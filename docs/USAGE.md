# Usage

Six scripts. Each is a thin wrapper: you pass the flags, it runs the thing.

| script | what it runs |
| --- | --- |
| `slurm/train.sbatch` | `gr00t/experiment/launch_finetune.py` |
| `slurm/train_bundle.sh` | the same, through `bundle-sbatch` |
| `slurm/eval.sbatch` | `univtac_groot.server.run_server` + `scripts/run_eval.py` |
| `slurm/eval_bundle.sh` | the same, through `bundle-sbatch` |
| `scripts/results_table.py` | success rate per run, from `*.summary.json` |
| `scripts/check_cudnn.py` | is cuDNN the version torch pins |

```bash
source env.sh
mkdir -p logs          # #SBATCH --output writes here
```

---

## Site rules you must satisfy yourself

Nothing checks these for you.

- **Job name longer than 50 characters.** A floor, not a ceiling — the opposite
  of every other cluster. A short name is rejected with "Unspecified error".
- **`--wckey=project-short-name:sub_4dpdata`** on every submission.
  `env.sh` exports `SBATCH_WCKEY` so plain `sbatch` picks it up; the bundle wrappers pass it explicitly.
- **`export TASK=<task>`.** The modality config reads it at import to pick the
  camera set (two views for `insert_tube` and `lift_bottle`, third-person only
  otherwise). Both training and evaluation need it.

---

## 1. Train

```bash
export TASK=insert_hole
sbatch --job-name=univtac-groot-per-task-finetune-vision-only-$TASK \
       --partition=sjw_alinlab --time=9:00:00 \
  slurm/train.sbatch \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path   $DATA_ROOT/univtac-$TASK-baseline_finetuned \
    --output-dir     $OUTPUT_ROOT/$TASK-baseline_finetuned \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path $REPO_ROOT/configs/modality/univtac_baseline_config.py \
    --num-gpus 1 --max-steps 30000 --save-steps 10000 --save-total-limit 4 \
    --use-wandb --wandb-project univtac-groot
```

`--save-steps 10000 --save-total-limit 4` retains `checkpoint-10000`, `-20000`
and `-30000`, about 26 GB each. Add `--resume-from-checkpoint` to continue from
the highest `checkpoint-N` already in `--output-dir`.

Keep `--num-gpus 1`: above 1, `launch_finetune.py` wraps the model in
`nn.DataParallel` and dies on a device mismatch.

## 2. Train through bundle-sbatch

```bash
export TASK=insert_hole
export JOB_NAME=univtac-groot-per-task-finetune-vision-only-$TASK
export PARTITION=sjw_alinlab
bash slurm/train_bundle.sh from_scratch -- \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path $DATA_ROOT/univtac-$TASK-baseline_finetuned \
    --output-dir   $OUTPUT_ROOT/$TASK-baseline_finetuned \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path $REPO_ROOT/configs/modality/univtac_baseline_config.py \
    --num-gpus 1 --max-steps 30000 --save-steps 10000 --save-total-limit 4
```

First argument is the `--job-kind train` checkpoint: `from_scratch`, or a
**physical** checkpoint directory to resume from. The launcher rejects symlinks,
so resolve `final` with `readlink -f` first. It also injects `MODEL_OUTPUT_DIR`
and names its own output directory, which is why `--output-dir` above still
decides where checkpoints actually land.

## 3. Evaluate

```bash
export TASK=insert_hole
export GROOT_MODEL=$OUTPUT_ROOT/$TASK-baseline_finetuned/checkpoint-30000
export EMBODIMENT_TAG=NEW_EMBODIMENT
export PORT=25555
sbatch --job-name=univtac-groot-evaluate-one-checkpoint-$TASK-ckpt30000 \
       --partition=background \
  slurm/eval.sbatch \
    --task $TASK --variant baseline_finetuned --univtac-root $UNIVTAC_ROOT \
    --seed-offset 1 --output eval_result/$TASK-ckpt30000-seed1.jsonl
```

Two processes in one job — the GR00T server (Python 3.12) and the UniVTAC
evaluator (Python 3.10) over loopback ZeroMQ, because they cannot share a
virtualenv. The script scrubs `LD_LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH` and
`CONDA_PREFIX` for the server only: UniVTAC's conda env points at CUDA 12.4 and
GR00T's torch is cu128 with its own cuDNN. A leak shows up as
`CUDNN_STATUS_NOT_INITIALIZED` on the first `get_action`, long after the weights
loaded, so it reads as a model bug.

`run_eval.py` writes the JSONL as each episode completes plus
`<output>.summary.json` at the end. Point `--output` at a **distinct** path per
checkpoint, or runs collide in one file.

`--seed-offset N` starts at seed `1_000_000 * (1 + N)`. Offset 0 is the reported
block; use 1+ for anything you might pick a checkpoint on.

**`run_eval.py`'s defaults are the protocol**, so the command above is short:

| flag | default | note |
| --- | --- | --- |
| `--task-config` | `clean` | |
| `--episodes` | `100` | scored episodes, not attempts |
| `--execution-horizon` | `16` | must be identical across runs you compare |
| `--startup-timeout` | `1800` | a 3B checkpoint plus its gated backbone is slow |

Only `--task`, `--variant`, `--univtac-root`, `--seed-offset` and `--output`
have to be passed.

For the zero-shot baseline, set `GROOT_MODEL` to a downloaded snapshot directory,
`EMBODIMENT_TAG=OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, and
`--variant baseline`.

**Resuming is automatic, which `background` requires.** That partition preempts
with `PreemptMode=REQUEUE`: the job re-runs with the same arguments and
`ResultWriter` appends, so without this a preemption would add a fresh pass from
the first seed and inflate the tally.

So `run_eval.py` reads any existing `--output` first and continues from
`max(seed) + 1` for only the episodes still unscored, exiting immediately if the
file is already complete — before Isaac Lab boots. Errored and skipped episodes
consumed a seed but not the budget, so a resumed run targets the same number of
*scored* episodes a clean run would, and duplicate seeds are counted once.

The summary is recomputed from the whole JSONL at the end, deduplicated by seed,
so it reflects every pass rather than the final leg alone.

`--no-resume` evaluates the full block again. It still appends, so the tally
will double-count: move the JSONL aside instead if you want a clean rerun. A
file that exists but cannot be parsed is a hard error rather than a guess.

## 4. Evaluate through bundle-sbatch

```bash
export TASK=insert_hole
export GROOT_MODEL=$OUTPUT_ROOT/$TASK-baseline_finetuned/checkpoint-30000
export EMBODIMENT_TAG=NEW_EMBODIMENT
export PORT=25555
export JOB_NAME=univtac-groot-evaluate-one-checkpoint-$TASK-ckpt30000
export PARTITION=background
bash slurm/eval_bundle.sh "$GROOT_MODEL" -- \
    --task $TASK --variant baseline_finetuned --univtac-root $UNIVTAC_ROOT \
    --seed-offset 1 --output eval_result/$TASK-ckpt30000-seed1.jsonl
```

## 5. Read the results

```bash
python scripts/results_table.py eval_result
python scripts/results_table.py eval_result --json table.json
```

Recursively finds every `*.summary.json` and prints one row each: task, scored
episodes, success rate, Wilson 95 % interval, error and skip counts.

## 6. Check cuDNN

```bash
$GROOT_PYTHON scripts/check_cudnn.py
```

Run it before a long job. `CUDNN_STATUS_NOT_INITIALIZED` means the GR00T venv's
cuDNN is not the `9.10.2.21` that torch 2.9.0+cu128 pins — **not** a driver
problem. `uv pip list` reports the pinned version even when the files on disk are
another release, so only `ctypes.CDLL("libcudnn.so.9").cudnnGetVersion()` catches
it (want `91002`), which is what this script does.

Do not work around it by disabling cuDNN: that costs ~86x on the vision tower's
`Conv3d`, turning a 1.89 s step into 170 s, with no error message.

---

## Other scripts

Not wrapped, so allocate a cpu node before running:

```bash
# Convert UniVTAC HDF5 -> GR00T LeRobot v2 (CPU-only, minutes)
$UNIVTAC_PYTHON scripts/convert_univtac_to_lerobot.py --task $TASK \
    --raw-dir $UNIVTAC_ROOT/data/$TASK/clean \
    --out $DATA_ROOT/univtac-$TASK-baseline_finetuned \
    --variant baseline_finetuned --univtac-root $UNIVTAC_ROOT --fps 20

# Download the released demonstrations (~24 GB per task)
bash $UNIVTAC_ROOT/data/download.sh --task $TASK --version 45 \
    --output $UNIVTAC_ROOT/data --workers 8

# Install the simulator stack (one-off, hours)
bash $UNIVTAC_ROOT/scripts/install.sh
```

Wrap either in `sbatch --partition=cpu` if you want to queue them instead.

## Notes

- Checkpoints go under `$OUTPUT_ROOT` = `/rlwrld-unified-checkpoints/$USER/jaewon`.
  The parent belongs to the account owner; keep ours in the `jaewon` subtree.
- The account is shared. Never `pkill` by pattern, never `wandb login` or
  `hf auth login` (they write into the owner's credential store), never install
  into `(base)` conda or `~/.local`. Filter `squeue` to `univtac-groot`.
- `Isaac-GR00T/.venv` was made by `uv`, so it has no pip. Use
  `uv pip install --python .venv/bin/python`, with `env -u CONDA_PREFIX -u VIRTUAL_ENV`.
- Measured throughput: **0.70 s/it** at 30,000 steps. The 1.89 s/it figure in
  `SETUP.md` was a 20-step smoke test and is 2.7x too slow.
