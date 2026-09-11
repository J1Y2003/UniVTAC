# The benchmark: protocol, settings, and what moves the number

This project produces one number per task: the success rate of a GR00T N1.7
finetune on a UniVTAC task, over 100 rollouts. This file is the reference for
the protocol and for every setting that changes that number.

| | |
| --- | --- |
| Model | `nvidia/GR00T-N1.7-3B`, finetuned per task |
| Observation | camera images + 17-D robot state + language instruction |
| State | `eef_9d` 9 + `joint_position` 7 + `gripper_position` 1 |
| Recipe | 30,000 steps, batch 64, GR00T's own lr and weight decay |
| Retained | `checkpoint-10000`, `-20000`, `-30000` |
| Action | `ACTION_HORIZON=40`, `EXECUTION_HORIZON=16` |
| Protocol | 100 rollouts, seeds from `1_000_000 * (1 + seed_offset)` |

## Why there is no zero-shot number

The finetune is not optional. Three upstream facts force it:

1. GR00T's state projector is *embodiment-conditioned*
   (`gr00t/model/modules/embodiment_conditioned_mlp.py`), and each embodiment
   tag fixes its own state layout in `gr00t/configs/data/embodiment_configs.py`.
2. Only `NEW_EMBODIMENT` (and the other `FINETUNE_ONLY_TAGS`) can carry a custom
   state layout — which UniVTAC's 17-D state is.
3. `FINETUNE_ONLY_TAGS` ships in **no released checkpoint**:
   `Gr00tPolicy.__init__` raises when the tag is absent from
   `processor.get_modality_configs()`, and `gr00t/data/embodiment_tags.py` says
   so explicitly.

So `baseline_finetuned` is the variant that produces the number. `baseline`
runs the released weights zero-shot under a pretrained tag (`OXE_DROID_*`), for
smoke tests.

## Running it end to end

```bash
# Convert (RUNBOOK.md covers the download first). Once per task.
env TASK=insert_hole TASK_CONFIG=clean UNIVTAC_JOB_CONFIG=1 \
  sbatch --job-name=univtac-groot-convert-univtac-hdf5-demonstrations-to-lerobot-v2 \
         --wckey=project-short-name:sub_4dpdata slurm/convert.sbatch

# Finetune: 30,000 steps, retaining checkpoints 10k/20k/30k.
TASKS=insert_hole EXTRA_TASKS="" bash slurm/submit_benchmark.sh

# Evaluate each retained checkpoint, one job each, on a DEV seed block.
for N in 10000 20000 30000; do
  bash slurm/eval_checkpoint.sh --task insert_hole --seed-offset 1 \
      --checkpoint <output_dir>/checkpoint-$N
done

# Table
python scripts/results_table.py
```

`benchmark_task.sbatch` pins the recipe (GPU count, `MAX_STEPS`, lr, weight
decay) on its first run for a task and refuses a mismatch afterwards, so a
resubmission cannot silently train under different conditions.

## Cameras are per task

UniVTAC defines the observation per task: third-person only, except two tasks
that add a wrist view.

| Task | Cameras |
| --- | --- |
| `insert_hole` | `head` only |
| `insert_tube` | `head` + `wrist` |
| `pull_out_key` | `head` only |
| `lift_bottle` | `head` + `wrist` |

The paper states it: *"task-specific configurations vary: both insert tube and
lift bottle utilize multi-view inputs, combining third-person and wrist-mounted
camera views; all other tasks use only the third-person view."*

This **contradicts `policy/task_settings.json`**, which marks `lift_can` — not
`lift_bottle` — as `camera_type: all`. Trust the paper: per
[UPSTREAM.md](UPSTREAM.md), upstream reads `task_settings.json` through a path
that does not resolve, under an `if path.exists()` guard, so it silently
defaults every task to `head` regardless of what the file says.

`univtac_groot.variants.MULTI_VIEW_TASKS` is the single source of truth, used by
both the dataset converter and the modality configs, so the training data and
the model's declared inputs cannot drift apart. The configs resolve it from
`TASK` and **raise** if `TASK` is absent rather than defaulting to two cameras —
a silently richer observation would be invisible in the results.
`UNIVTAC_VIDEO_KEYS=head,wrist` overrides it deliberately.

Datasets converted before this was enforced carry both streams; reconvert the
head-only tasks.

## The recipe

GR00T runs on **its own defaults** for everything except the step count.
`launch_finetune.py`'s learning rate, weight decay, schedule and batch size are
left unset. Confirm what those defaults actually are rather than trusting a
secondhand value:

```bash
"${GROOT_PYTHON}" "${GROOT_ROOT}/gr00t/experiment/launch_finetune.py" --help \
    | grep -A2 -E "learning-rate|weight-decay|warmup|global-batch-size"
```

To sweep them deliberately:

```bash
LEARNING_RATE=1e-5 WEIGHT_DECAY=1e-4 bash slurm/submit_benchmark.sh
```

`benchmark_task.sbatch` records both in its pinned recipe and refuses a
resubmission that changes them, so a walltime kill cannot resume under
different optimiser settings.

**Training data is all 100 released episodes per task** (`0.hdf5`..`99.hdf5`).
At 30,000 steps x batch 64 that is 1,920,000 samples over ~28,932, so ~66 epochs
at 452 steps/epoch. GR00T's own 10,000-step default would be ~22 epochs.

### Choosing the step count

Training runs to 30,000 steps and retains `checkpoint-10000`,
`checkpoint-20000` and `checkpoint-30000`. Picking one of the three IS
selection, so it has to happen somewhere that is not the reported number.

**The mechanism: a disjoint seed block.** `--seed-offset N` starts the episode
seeds at `1_000_000 * (1 + N)`, so offset 0 is the reported protocol (seeds from
1,000,000, matching UniVTAC) and offset 1 is a completely disjoint set (from
2,000,000) drawn from the same distribution. Evaluate all three checkpoints at
**offset 1**, freeze the step count, then run the reported number once at
**offset 0**. Four evaluations per task, only the last of which is reported.

**One uniform step count for every task, not a per-task argmax.** At 100
episodes the Wilson 95 % interval is roughly ±10 points, so the argmax of three
points each carrying that much noise is frequently just noise — the winner's
curse, where the selected value's advantage does not reproduce.

**Read the trend, not the peak.** Three points at ±10 cannot resolve a maximum,
but they can distinguish still-climbing from flat from clearly falling:

* still climbing at 30k — use 30k, and note the budget was the binding
  constraint rather than convergence;
* flat — prefer the smallest, since nothing shows the extra compute bought
  anything;
* clearly falling after some point — that is the overfitting signal, and the
  step before it is the answer.

Then state it in one line beside the number: "step count 30,000, selected from
{10,000, 20,000, 30,000} on a disjoint seed block and applied uniformly."

## Receding-horizon control

N1.7's flow-matching DiT predicts a dense chunk of future actions. Executing all
of it is cheap but drifts; re-planning every step costs a full VLA forward pass
per environment step. `RecedingHorizonController` executes the first
`execution_horizon` actions and then re-plans, which is what GR00T's own
`PolicyHorizonSpec.n_action_steps` and `MultiStepWrapper` do.

Chunk lengths differ by config — 40 for the N1.7 default and the DROID tag, 16
for `libero_sim` and for our finetune configs, 8 for `simpler_env_*` — so
nothing here hard-codes one. `resolve_spec_from_policy` reads
`get_modality_config` from the live server and validates that
`action.delta_indices` is the contiguous `range(0, H)`, because the chunk is
indexed linearly and a sparse window would silently execute the wrong rows.

**Keep `execution_horizon` fixed across every run you intend to compare.** It
sets the closed-loop rate, which strongly affects contact-rich tasks. At
`EXECUTION_HORIZON=16` against UniVTAC's `step_lim = 300` that is roughly 19
decisions per episode; `EXECUTION_HORIZON=1` re-plans every step at 16x the
inference cost.

## Calibration to check before trusting a low number

Each of these can depress a success rate to zero while looking like a genuine
result.

**1. Gripper convention.** UniVTAC wants metres of finger opening bounded by
`RobotManager.gripper_max_qpos` (0.039 m), with larger meaning *more open*.
DROID-style checkpoints usually emit a normalised scalar where 1.0 means
*closed*. `GripperConvention(invert=True)` is the default for the zero-shot
baseline and `invert=False` for anything finetuned through this repo's converter
(which writes the UniVTAC convention). Log a few predicted `gripper_position`
values and check the fingers close when they should.

**2. Action space for the zero-shot variant.** The DROID tag returns `eef_9d`
(relative EEF), `joint_position` (relative joint) and `gripper_position`.
`--action-type qpos` uses `joint_position + gripper_position`, mapping directly
onto UniVTAC's 8-D `qpos` path; `--action-type ee` uses `eef_9d` instead, which
routes through cuRobo planning. Not equivalent in practice: the joint path is
more faithful to what the model predicts, the EEF path more robust to
joint-space offsets between DROID's Panda mounting and UniVTAC's.

**3. Camera correspondence.** The zero-shot baseline maps UniVTAC's `head` onto
DROID's `exterior_image_1_left` and `wrist` onto `wrist_image_left`. The
viewpoints are similar in kind, not calibrated to each other.

**4. Control rate.** UniVTAC steps at `decimation = 1` over a 1/120 s sim step;
DROID data is 15 Hz, so a per-step joint delta learned at 15 Hz means something
different at the sim's rate. This affects the zero-shot variant; a finetune
learns the deployment rate from the data.

## Reading the results

Each eval writes `<result-dir>/<task>-<variant>/<task>-ckpt<N>-seed<offset>.json`,
and `scripts/results_table.py` aggregates a directory of them into per-task
and pooled success rates with Wilson 95 % intervals.

At 100 episodes an interval is roughly ±10 points near 20 % and ±10 near 50 %,
so a single task separates very little. Report the interval, not just the point
estimate, and treat non-overlap as a conservative signal rather than a
hypothesis test.

`episodes_errored` is tracked separately from failures and excluded from the
rate, matching UniVTAC's evaluator, which decrements `test_num` on an exception
rather than scoring it. A non-trivial error count means a broken configuration,
not a hard task — check the JSONL's `error` field.

**Do not select a checkpoint or a hyperparameter on the reported rollouts.**
There is no validation split and `launch_finetune.py` exposes no eval metric, so
choosing the best of three saved checkpoints by success rate on seed offset 0
fits the test set. Use a disjoint seed block, or `lift_bottle`, which is
deliberately not one of the three reported tasks.

### State when reporting the number

Facts about this configuration that a reader needs, none of them problems:

* **State width** 17-D (`eef_9d` 9 + `joint_position` 7 + gripper 1). The
  end-effector pose is a known function of the joints, so this is a
  reparameterisation rather than extra information.
* **Action parameterisation.** `launch_finetune.py` hardcodes
  `use_relative_action = True`; our config marks `joint_position` RELATIVE and
  `gripper_position` ABSOLUTE.
* **Horizons.** `ACTION_HORIZON = 40` (N1.7's default and its ceiling —
  `validate_action_horizons` rejects more) with `EXECUTION_HORIZON = 16`. There
  is no temporal ensembling: `univtac_groot/receding_horizon.py` does
  execute-k-then-replan only.
* **Image preprocessing.** GR00T resizes to 256x256 and expects RGB; the
  converter swaps channels, since UniVTAC stores BGR.
* **Step count** and how it was chosen, per the section above.
* **Episode count** — 100 released episodes per task for training, 100 rollouts
  for evaluation.

## Published numbers on these tasks

From UniVTAC's Table I, for judging whether a result is plausible. Different
policy classes on different observation spaces, trained on 50 episodes with
their own evaluation seeds.

| Task | ACT (vision only) | ACT + UniVTAC Encoder | VITaL |
| --- | --- | --- | --- |
| Lift Bottle | 42.0 | 71.0 | 72.0 |
| Pull-out Key | 28.0 | 46.0 | 47.0 |
| Lift Can | 20.0 | 29.0 | 8.0 |
| Put Bottle in Shelf | 28.0 | 31.0 | 32.0 |
| Insert Hole | 19.0 | 24.0 | 25.0 |
| Insert HDMI | 15.0 | 28.0 | 6.0 |
| Insert Tube | 45.0 | 56.0 | 34.0 |
| Grasp Classify | 50.0 | 99.0 | 100.0 |
| *Average (8 tasks)* | *30.9* | *48.0* | *40.5* |

These are contact-rich tasks where low double digits is the norm: 81 of the 100
`insert_hole` rollouts fail in the best column above. `docs/STATUS.md` has the
failure-to-cause table for telling a weak policy from a broken pipeline.

The eight-task average describes eight tasks; this repo runs three or four.
