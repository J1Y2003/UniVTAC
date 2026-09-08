# Working on this repo

Benchmark `nvidia/GR00T-N1.7-3B` on the UniVTAC visuo-tactile benchmark as a
two-condition ablation. Read [docs/RUNBOOK.md](docs/RUNBOOK.md) for the
end-to-end order, [docs/UPSTREAM.md](docs/UPSTREAM.md) before touching anything
that talks to GR00T or UniVTAC, and [docs/STATUS.md](docs/STATUS.md) for where
the work currently stands.

## Hard rule: never touch the cluster directly

**Claude must never access the cluster itself.** No `ssh`, `scp`/`rsync`,
`srun`, `sbatch`, `scancel`, or `squeue`; no installs, no file writes, no
reading logs over the wire. This covers read-only commands too, and it is not
negotiable because a diagnosis would be faster with a live shell.

Applies to every cluster alias in `~/.ssh/config` (`rlwrld_node1`/`2`/`3` and
any other).

Instead: reason and author scripts *locally*, then hand over one
copy-pasteable command -- or one script with sensible defaults -- and let the
operator run it and paste the output back.

The rule exists because it was broken. On 2026-09-08, asked only to help
*isolate* a slow training step, Claude connected unprompted and mutated shared
state: installed `py-spy` into `Isaac-GR00T/.venv`, wrote files under
`~/jaewon/scratch/`, submitted two SLURM jobs on the shared account (one of
which would have started a real finetune), and ran a 37 GB benchmark through
`srun --overlap` inside a live 5-hour training job, taking the A100 to
80,775 MiB of 81,920 and nearly OOM-killing it. This is a **shared, borrowed
account**: a mistake here costs someone else's work, which is why the operator
runs the commands.

## The experiment in one paragraph

Two **variants**, same 100 episodes, same recipe, differing only in whether
pooled tactile depth is appended to the state vector: `baseline_finetuned`
(17-D state) and `tactile` (113-D state = 17 proprioception + 96 pooled
GelSight depth). "Variant" means experimental condition; UniVTAC has exactly one
robot arm (Franka Panda), and the word "arm" in this codebase always means the
manipulator. A third variant `baseline` exists for zero-shot smoke tests only --
it is not a fair comparator to a finetuned model.

## Non-obvious things that will cost you a day

**Two processes, not one.** UniVTAC needs Python 3.10 (Isaac Sim), GR00T needs
3.12. They cannot share a virtualenv, so they talk over loopback ZeroMQ. This is
forced, not a design preference.

**Three checkouts, similar names.** `UniVTAC/` is this repo (submit jobs from
here). `UniVTAC-sim/` is the simulator (`$UNIVTAC_ROOT`). `Isaac-GR00T/` is
GR00T (`$GROOT_ROOT`); its `.venv` was made by `uv` so it has **no pip** --
use `uv pip install --python .venv/bin/python`, with `env -u CONDA_PREFIX -u
VIRTUAL_ENV`, or uv resolves the wrong environment.

**Shared account.** The cluster user is shared with a senior who runs his own
jobs (named `iclr2027*`) and a monitoring dashboard on the same UID. Never
`pkill` by pattern, never `wandb login` or `hf auth login` (they write into his
`~/.netrc` / token store), never install into `(base)` conda or `~/.local`.
Filter `squeue` output to `univtac-groot` to see only our jobs. For tokens,
export `HF_TOKEN` / `WANDB_API_KEY` in the submitting shell -- `--export=ALL`
carries them in.

**`sbatch` copies the script to `/var/spool/slurm/d/`,** so
`dirname "${BASH_SOURCE[0]}"` resolves to the spool, not the repo. All
`slurm/*.sbatch` resolve `REPO_ROOT` from `SLURM_SUBMIT_DIR` first and validate
it contains `univtac_groot/spec.py`. Do not "simplify" that back.

**`--export=ALL` carries your whole shell environment.** A stray `DRY_RUN=1`
left over from testing makes the job exit in seconds having trained nothing;
the submitters pin `DRY_RUN=0` for this reason.

**cuDNN: ask the library, not pip.** `CUDNN_STATUS_NOT_INITIALIZED` means the
GR00T venv's cuDNN is not the `9.10.2.21` that torch 2.9.0+cu128 pins. It is
not the driver. `uv pip list` reports the pinned version even when the files on
disk are another release, so only
`ctypes.CDLL("libcudnn.so.9").cudnnGetVersion()` catches it (want `91002`);
`scripts/preflight.py --deep` does. Never reach for `DISABLE_CUDNN=1`: it costs
~86x on Qwen3-VL's patch-embed `Conv3d` and turns a 1.89 s training step into
170 s.

**Verify upstream, do not assume.** The facts most likely to be guessed wrong:
the action horizon is **40**, not 16; tactile is **320x240 imagery**, not a
small array; in the dataset's HDF5 images are **JPEG byte streams** and
state/action are a **one-step shift of a single `embodiment/joint` array**; and
the marker field's shape is not what its name suggests. `docs/UPSTREAM.md`
records each with its source -- add to it rather than re-deriving.

## Cluster rules (Kakao SLURM)

The submit filter rejects jobs violating any of these, usually with an
unhelpful "Unspecified error":

- job name of **50 characters or fewer** -- the limit is a *floor*, not a
  ceiling, which is the opposite of every other cluster. Verified in both
  directions: `convert.sbatch`'s 63-character name and the benchmark jobs'
  69-70 character names all submit, and a short name is refused. This is why
  the job names here are absurdly descriptive -- do not shorten them
- `MODEL_OUTPUT_DIR` set under `/rlwrld-unified-checkpoints/<user>/checkpoints/<job>`
- `--wckey=project-short-name:sub_4dpdata` on **every** `sbatch` and `srun`.
  The string `project-short-name:` is a **literal part of the required
  format**, not a template to fill in. It looks exactly like a placeholder and
  is not one: strip it and the submit filter answers

  ```
  ❌ ERROR: WCKey를 project-short-name:<name> 형식으로 지정해야 합니다.
     예: #SBATCH --wckey=project-short-name:human
  ```

  Note the filter's own example keeps the prefix verbatim. `sub_4dpdata` is the
  project name and goes after the colon. **Do not "clean this up"** -- it was
  removed once on the assumption it was a placeholder and every submission
  failed until it was put back.

  **A `#SBATCH --wckey` header is not enough.** sbatch precedence is
  **command line > environment variable > `#SBATCH` directive**, so a stale
  `SBATCH_WCKEY` in your shell -- e.g. an `env.sh` copied from the old
  `env.example.sh` -- silently overrides the header and the job goes out under
  the wrong project. `env.sh` is gitignored, so fixing the repo does not fix
  yours: `export SBATCH_WCKEY=project-short-name:sub_4dpdata`.

  **Do not export `SLURM_WCKEY`.** SLURM sets it *inside* a job to report the
  wckey the job actually got, which is what every `.sbatch` re-checks at
  runtime before doing any work. Exporting it from your shell rides in on
  `--export=ALL` and masks that check. `srun` takes `--wckey` on the command
  line instead, and every documented `srun` here does.

  So: `--wckey` on every command line (strongest), a runtime re-check in every
  `.sbatch`, and `scripts/preflight.py` failing on a wrong `SBATCH_WCKEY`.
- **no** `--cpus-per-task` and **no** `--mem` — memory and CPUs are not
  specifiable here at all; a job takes the node's per-GPU defaults
- **no** `--time` — a job gets the partition maximum, or runs until the script
  exits, so a limit can only cut the run short. Over-requesting also leaves it
  pending forever with `REASON=PartitionTimeLimit`
- **`--partition=cpu` for any job that does not request a GPU.** The GPU
  partitions refuse it with
  `GPU 파티션에는 GPU를 요청한 잡만 제출할 수 있습니다` /
  `CPU 전용 잡은 --partition=cpu 를 사용하세요`.
  The two CPU-only jobs here are `download_data.sbatch` and `convert.sbatch`;
  both carry `#SBATCH --partition=cpu`. The other four request `--gres=gpu:N`
  and belong on a GPU partition.

  Same precedence trap as the wckey: `--partition` follows
  **command line > `SBATCH_PARTITION` > `#SBATCH` directive**, and
  `env.example.sh` suggests exporting `SBATCH_PARTITION` for the long-running
  GPU partition — which would silently override the header on exactly these
  two jobs. Pass `--partition=cpu` on the command line as well.

Because `--cpus-per-task` is rejected, `SLURM_CPUS_PER_TASK` is `1` inside a
job even when it really holds 12 CPUs. Size thread pools off
`SLURM_CPUS_ON_NODE` instead — not `nproc`, which returns the node's full 128
since there is no cpuset isolation here.

`sbatch --test-only <script>` runs the filter without queueing -- use it before
blaming the script. `srun` is restricted to the `debug` partition. `debug` is
the *default* partition and caps at 3 hours, which is why untuned jobs sit
pending with `PartitionTimeLimit`; every other partition allows 2 days.
`PriorityTier` is a strict ordering (`background` 1 < `sjw_alinlab` 2 <
`sjw_alinlab_premium` 3), so a single premium job outranks yours permanently.

`logs/` must exist **before** submitting: SLURM opens `--output` before the
script runs, so a missing directory kills the job with no log at all.

Checkpoints under `/rlwrld-unified-checkpoints` are archived off NFS after
**4 days** untouched and deleted 90 days later. Run
`bash slurm/preserve_outputs.sh` as soon as a finetune finishes or you will
retrain it.

## How to run things

```bash
bash slurm/submit_benchmark.sh --dry    # full preflight, no GPU time, no submit
bash slurm/submit_benchmark.sh          # the real run (one job per task)
bash slurm/smoke_test.sh                # 20 steps + 1 eval episode, isolated
bash slurm/preserve_outputs.sh          # rescue checkpoints from retention
```

Everything is an overridable env var (`GPUS`, `MAX_STEPS`, `TASK`, `PARTITION`,
...). Prefer adding a variable over editing a command line.

`submit_benchmark.sh` submits **one job per task**, in two tiers: `TASKS`
(the three reported tasks) go first and unconstrained, then `EXTRA_TASKS`
(`lift_bottle`) is submitted with `--dependency=afterany` on all of them so
it cannot take a GPU a reported task still wants. `lift_bottle` is separate
because it is the only task a hyperparameter sweep may touch without
fitting the reported numbers.

`benchmark_task.sbatch` is resumable: stage markers plus
`--resume-from-checkpoint`. It also **pins the training recipe** on first run
(GPU count and `MAX_STEPS`) and refuses a mismatch, because `--num-gpus`
multiplies the effective batch size -- training the two variants at different
batch sizes would confound the ablation. `ALLOW_RECIPE_CHANGE=1` overrides.

## Working style expected here

Verify before asserting; say plainly when something is unverified or when you
were wrong. Do not claim to have pushed -- the user handles git unless they ask.
Give commands that work against the user's *current* checkout, not against
uncommitted local edits. The user works from an allocated node or a login node
and dislikes long copy-paste blocks: prefer one script with sensible defaults.
