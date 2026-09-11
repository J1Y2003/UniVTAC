# Working on this repo

Finetune `nvidia/GR00T-N1.7-3B` on the UniVTAC benchmark, one policy per task,
and measure its success rate over 100 rollouts. The deliverable is one number
per task. Read [docs/USAGE.md](docs/USAGE.md) for how to run anything,
[docs/UPSTREAM.md](docs/UPSTREAM.md) before touching anything that talks to
GR00T or UniVTAC, and [docs/STATUS.md](docs/STATUS.md) for where the work
currently stands.

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

GR00T N1.7 is finetuned per task on the 100 released episodes. Its observation
is the task's camera images (third-person, plus wrist on `insert_tube` and
`lift_bottle`), the **17-D robot state** (`eef_9d` 9 + `joint_position` 7 +
`gripper_position` 1), and the language instruction. It is evaluated over 100
rollouts per task, and the success rate is the output.

`baseline_finetuned` is the variant that produces that number and the only one
run. `baseline` runs the released weights zero-shot under a pretrained tag, for
smoke tests.

UniVTAC has exactly one robot arm (Franka Panda), so "arm" in this codebase
always means the manipulator.

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
`scripts/preflight.py --deep` does, and `scripts/check_cudnn.py` re-checks it
inside every GPU job before weights load. **There is deliberately no way to run
without cuDNN here** -- no `DISABLE_CUDNN`, no `--disable-cudnn`. It costs ~86x
on Qwen3-VL's patch-embed `Conv3d`, turning a 1.89 s step into 170 s with no
error message, so do not reintroduce a switch for it.

**Verify upstream, do not assume.** The facts most likely to be guessed wrong:
the action horizon is **40**, not 16; in the dataset's HDF5 images are **JPEG
byte streams**; and state/action are a **one-step shift of a single
`embodiment/joint` array**. `docs/UPSTREAM.md` records each with its source --
add to it rather than re-deriving.

## Six scripts, each a thin wrapper

`docs/USAGE.md` is the reference, with a runnable example for each.

| script | runs |
| --- | --- |
| `slurm/train.sbatch` | `launch_finetune.py` |
| `slurm/train_bundle.sh` | the same, through `bundle-sbatch` |
| `slurm/eval.sbatch` | `run_server` + `run_eval.py` |
| `slurm/eval_bundle.sh` | the same, through `bundle-sbatch` |
| `scripts/results_table.py` | success rate per run |
| `scripts/check_cudnn.py` | the cuDNN pin |

Each shell script is under ten lines and does one thing: `exec` the real
program with the flags it was handed. **They validate nothing.** There is no
`common.sh`, no preflight, no stage markers, no recipe pinning, no
configuration sentinel, no submitter that assembles a command for you.

Passing the right flags is the operator's job. Do not add guards, defaults,
fallbacks or `for` loops back into these scripts -- that is the explicit design,
not an oversight. If something needs checking, say so in `docs/USAGE.md`
instead.

`run_eval.py`'s defaults are the protocol -- `--task-config clean`,
`--episodes 100`, `--execution-horizon 16`, `--startup-timeout 1800` -- so an
eval command only has to pass `--task`, `--variant`, `--univtac-root`,
`--seed-offset` and `--output`.

Two things the shell still owns, because they cannot live in a flag:

- `slurm/eval.sbatch` starts **two processes** (the GR00T server and the
  UniVTAC evaluator) and scrubs `LD_LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH` and
  `CONDA_PREFIX` for the server only. A leak surfaces as
  `CUDNN_STATUS_NOT_INITIALIZED` on the first `get_action`, reading as a model
  bug.
- `export TASK=<task>` reaches the modality config, which resolves the camera
  set from it at import. Both training and evaluation need it.


## Cluster rules (Kakao SLURM)

The submit filter rejects jobs violating any of these, usually with an
unhelpful "Unspecified error":

- job name of **50 characters or fewer** -- the limit is a *floor*, not a
  ceiling, which is the opposite of every other cluster. Verified in both
  directions: 63-character and 69-70 character names all submit, and a short
  name is refused. This is why the job names in docs/USAGE.md are absurdly
  descriptive -- do not shorten them
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

  `env.sh` exports `SBATCH_WCKEY`, which plain `sbatch` reads; the
  `*_bundle.sh` wrappers pass `--wckey` explicitly. A stale `SBATCH_WCKEY`
  outranks a `#SBATCH` header, so keep it correct.

  **Do not export `SLURM_WCKEY`.** SLURM sets it *inside* a job to report the
  wckey the job actually got, which is what every `.sbatch` re-checks at
  runtime before doing any work. The job inherits your shell's environment, so
  an exported value masks that check. `srun` takes `--wckey` on the command
  line instead, and every documented `srun` here does.

  So: `--wckey` on every `sbatch` command line (added by `univtac_build` if
  you forget), a runtime re-check in every `.sbatch`, and
  `scripts/preflight.py` failing on a wrong `SBATCH_WCKEY`.
- **no** `--cpus-per-task` and **no** `--mem` — memory and CPUs are not
  specifiable here at all; a job takes the node's per-GPU defaults
- `--time` is **allowed and usually worth setting**. A job without
  one is assumed to want the partition maximum, so backfill can only start it
  in a 2-day gap -- a realistic limit makes it eligible for many more gaps and
  it starts sooner. Two ways to get it wrong: over-requesting past the
  partition maximum leaves it pending forever with `REASON=PartitionTimeLimit`,
  and under-requesting cuts the run short. Both jobs here are resumable, so a
  short limit costs a resubmission rather than a run: finetunes resume from the
  last retained checkpoint (granularity `SAVE_STEPS`, so 10,000 steps -- about
  1.9 h of redone work), and evaluation resumes from `max(seed)+1` for the
  episodes still unscored, which `background`'s `PreemptMode=REQUEUE` forced.
  `9:00:00` is right for training against a measured ~6.2 h. The cost of 100
  eval rollouts has never been timed, so evaluation is usually submitted
  without `--time`
- **`--partition=cpu` for any job that does not request a GPU.** The GPU
  partitions refuse it with
  `GPU 파티션에는 GPU를 요청한 잡만 제출할 수 있습니다` /
  `CPU 전용 잡은 --partition=cpu 를 사용하세요`.
  Conversion and download are run by hand (docs/USAGE.md, "Other things you
  run by hand"); wrap either in `sbatch --partition=cpu` if you would rather
  not hold a login node. Both `slurm/*.sbatch` request a GPU.

  Same precedence trap as the wckey: `--partition` follows
  **command line > `SBATCH_PARTITION` > `#SBATCH` directive**, so an exported
  `SBATCH_PARTITION` would silently override the header on exactly these two
  jobs. `env.example.sh` leaves it commented out for that reason.

Because `--cpus-per-task` is rejected, `SLURM_CPUS_PER_TASK` is `1` inside a
job even when it really holds 12 CPUs. Size thread pools off
`SLURM_CPUS_ON_NODE` instead — not `nproc`, which returns the node's full 128
since there is no cpuset isolation here.

`sbatch --test-only` rehearses a submission without queueing it.
`srun` is restricted to the `debug` partition. `debug` is
the *default* partition and caps at 3 hours, which is why untuned jobs sit
pending with `PartitionTimeLimit`; every other partition allows 2 days.
`PriorityTier` is a strict ordering (`background` 1 < `sjw_alinlab` 2 <
`sjw_alinlab_premium` 3), so a single premium job outranks yours permanently.

**Evaluation may not run on `sjw_alinlab`** -- it goes on `background`, while
training stays on `sjw_alinlab`. This is lab policy, not something the submit
filter enforces, so nothing will stop you getting it wrong. A job holds one
allocation on one partition, which is why finetune and evaluate are separate
jobs -- and with 10k/20k/30k retained, **four jobs per task**: one finetune and
three evaluations. See "How to run things" below. Note `background` is
`PriorityTier` 1, the *lowest* tier: it wins no contest for a busy node, it is
simply a shorter queue when its own nodes are idle. It **does** preempt, with
`PreemptMode=REQUEUE`. A requeued eval job re-runs with the same arguments and
`ResultWriter` appends, so `run_eval.py` reads any existing `--output` first and
continues from `max(seed)+1` for only the episodes still unscored, exiting
before Isaac Lab boots if the file is already complete. Without that, every
preemption would append a fresh pass from the first seed and inflate the tally.
`--no-resume` disables it. The summary is recomputed from the whole JSONL,
deduplicated by seed, so it covers every pass rather than the final leg.

`logs/` has to exist before submitting -- `#SBATCH --output` does not create
it. Logs land in `logs/<job-name>-<jobid>.out`.

## How to run things

See `docs/USAGE.md`. In short:

```bash
source env.sh && mkdir -p logs
sbatch --job-name=<over 50 chars> --partition=sjw_alinlab --time=9:00:00 \
  slurm/train.sbatch <launch_finetune.py flags>
sbatch --job-name=<over 50 chars> --partition=background \
  slurm/eval.sbatch <run_eval.py flags>
python scripts/results_table.py eval_result
```

**The recipe: 30,000 steps, three checkpoints.** `--max-steps 30000
--save-steps 10000 --save-total-limit 4` retains `checkpoint-10000`, `-20000`
and `-30000` and nothing else (the limit is a ceiling with one spare slot, so an
extra end-of-training save cannot evict the first). Measured **0.70 s/it**, so
~6.2 h per task, ~26 GB per checkpoint.

30,000 is the one recipe number **we** chose rather than inherited from GR00T
(its default is 10,000), so it is a deliberate difference that gets disclosed
with the numbers. It exists because there is no validation split and no eval
metric in `launch_finetune.py`: success rate against step count is the only
instrument for whether 22 epochs was enough, and three retained checkpoints are
its three points. Which one becomes the reported model is decided on a
**disjoint seed block** (`--seed-offset 1`), frozen, and applied uniformly to
every task -- never a per-task argmax, which is both an asymmetric advantage
over ACT's uniform 4,000 steps and a winner's curse at +/-10 points of noise.
See docs/BENCHMARK.md, "Choosing the step count".

Do not size a walltime off the **1.89 s/it** in `docs/SETUP.md`. That was a
20-step smoke test, two of whose steps wrote a checkpoint at ~9.82 s; it is the
correct baseline for the cuDNN penalty ratio, but 2.7x the production rate.

Resuming is `--resume-from-checkpoint`, which takes the highest `checkpoint-N`
already in `--output-dir`. Keep `--output-dir` the same across resubmissions and
it just works; `--num-gpus` must not change, because it multiplies the effective
batch size.

Checkpoints go under `$OUTPUT_ROOT` =
`/rlwrld-unified-checkpoints/$USER/jaewon`. The parent belongs to the account
owner; ours stays in the `jaewon` subtree. `bundle-sbatch` is kept as an entry
point (`*_bundle.sh`) because it is the announced path, but it names its own
output directories by ULID under that parent, which is why it is not the
default here.


## Working style expected here

Verify before asserting; say plainly when something is unverified or when you
were wrong. Do not claim to have pushed -- the user handles git unless they ask.
Give commands that work against the user's *current* checkout, not against
uncommitted local edits. The user works from an allocated node or a login node
and dislikes long copy-paste blocks: prefer one script with sensible defaults.
