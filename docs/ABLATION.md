# The ablation: does tactile in the state vector help GR00T N1.7?

The study compares two variants that differ in exactly one respect:

* **Variant A (baseline).** Vision, language instruction, 1-D proprioception,
  embodiment id.
* **Variant B (tactile).** The same, plus the flattened UniVTAC tactile array
  concatenated onto the proprioception vector.

`univtac_groot/variants.py` builds both from one shared `PROPRIO_FIELDS` tuple, so
the variants cannot drift apart by accident, and `tests/test_obs_adapter.py`
asserts that the proprioception slices are bit-identical between them.

---

## The constraint that shapes the whole experiment

**Variant B cannot run zero-shot on `nvidia/GR00T-N1.7-3B`.** Three facts from the
upstream source combine to force this:

1. GR00T's state projector is *embodiment-conditioned*
   (`gr00t/model/modules/embodiment_conditioned_mlp.py`), and each embodiment tag
   fixes its own state layout in `gr00t/configs/data/embodiment_configs.py`.
2. Only `NEW_EMBODIMENT` (and the other `FINETUNE_ONLY_TAGS`) can carry a custom
   state layout.
3. `FINETUNE_ONLY_TAGS` ships in **no released checkpoint** —
   `Gr00tPolicy.__init__` raises when the tag is absent from
   `processor.get_modality_configs()`, and `gr00t/data/embodiment_tags.py` says
   so explicitly.

So appending tactile dimensions to a pretrained tag's state vector is not a
smaller version of the same experiment — the extra dimensions land in a
projector that never saw them, and the pretrained slices shift meaning. There is
no way to read a useful zero-shot tactile number off the released weights.

### What that means in practice

| Comparison | Runnable today | What it measures |
| --- | --- | --- |
| A zero-shot vs. B zero-shot | ❌ | not possible (see above) |
| **A zero-shot** | ✅ | how far the released model gets on UniVTAC out of the box |
| **A finetuned vs. B finetuned** | after finetuning | **the actual ablation** |

The like-for-like comparison is between two finetunes that share a recipe,
dataset, action space and execution horizon, and differ only in the tactile
state dimensions. That is what `--variant baseline_finetuned` exists for; the
zero-shot baseline is a useful reference point, not Variant B's control.

Reporting the zero-shot baseline against a finetuned tactile variant would
attribute the entire finetuning effect to tactile sensing. `scripts/compare_ablation.py`
takes `--baseline-variant`, so point it at `baseline_finetuned` for the headline
number and report the zero-shot variant separately.

## Running it end to end

The whole tactile ablation for one task, given data already downloaded and
converted (see [RUNBOOK.md](RUNBOOK.md)):

```bash
# One dataset per variant -- the state layout differs, so they cannot share one
for v in baseline_finetuned tactile; do
  env TASK=insert_hole VARIANT=$v TASK_CONFIG=clean UNIVTAC_JOB_CONFIG=1 \
    bundle-sbatch --job-kind data_process --code-git-root $REPO_ROOT -- \
    --job-name=univtac-groot-convert-univtac-hdf5-demonstrations-to-lerobot-v2 \
    --wckey=project-short-name:sub_4dpdata --partition=cpu -- slurm/convert.sbatch
done

# Finetune both variants, same recipe
VARIANTS="tactile baseline_finetuned" TASKS=insert_hole EXTRA_TASKS="" \
  bash slurm/submit_benchmark.sh

# Then, once those have finished, evaluate what they produced
VARIANTS="tactile baseline_finetuned" TASKS=insert_hole EXTRA_TASKS="" \
  bash slurm/submit_benchmark.sh --evals

# Table
python scripts/compare_ablation.py --baseline-variant baseline_finetuned --tactile-variant tactile
```

`benchmark_task.sbatch` pins the recipe (GPU count, `MAX_STEPS`, lr, weight
decay) on its first run for a task and refuses a mismatch afterwards, which is
what keeps the two variants comparable. Add the zero-shot `baseline` row with
`VARIANTS=baseline bash slurm/submit_ablation.sh` -- it needs no finetune, and
it is not a fair comparator to a finetuned model.

## How tactile enters the observation

UniVTAC's tactile sensors are **camera-based**: `envs/sensors/tactile.py` wraps
TacEx GelSight Mini, ViTai GF225 and XenseWS, and a reading is an image, not a
short vector. `TactileManager.get_observations` can emit `rgb`, `rgb_marker`,
`depth` (a gel height map), `marker` (a marker-motion field), `points` and
`pose`. For a GelSight Mini the height map is 320×240. The marker field is
larger than the sensor's `create_*_cfg` grid suggests: the released `isaac45`
dumps store `marker` as `(N, 2, 1200, 2)` — a *pair* of 1200-point rasters per
frame, not a 9×7 = 64 grid. Size nothing off 64.

Proprioception spends 17 of GR00T's 132 state dimensions
(`eef_9d` 9 + `joint_position` 7 + `gripper_position` 1), leaving **115 for
tactile across all sensors**. Raw flattening is therefore impossible: 76 800
dims per sensor for the height map, 4800 for the marker field with two sensors.
`TactileSpec` offers three routes:

| `--tactile-mode` | What is concatenated | Dims/sensor (default) |
| --- | --- | --- |
| `depth_pool` *(default)* | average-pooled gel height map, normalised to [-1, 1] | 8×6 = **48** |
| `marker` | pooled marker-motion displacements, `tanh`-bounded | 8×6×2 = 96 → use a smaller grid |
| `video` | nothing — tactile RGB becomes extra *video* streams | 0 |

`depth_pool` at 8×6 gives a **113-D state**, inside the cap.
`ObsSpec.validate` refuses anything larger and reports the remaining budget, so
a bad grid fails at construction rather than mid-episode.

`video` mode deserves consideration on its merits: GR00T's vision encoder is
built for images, and a marker overlay makes shear visible in a way a pooled
scalar field does not. It is not the ablation as specified (the state vector is
untouched), so it is offered as a third variant rather than the default.

**The pooling grid is part of the model contract.** It must match the
`tactile_*` widths in `configs/modality/univtac_tactile_config.py` and the
dataset the checkpoint was finetuned on. Change it in one place and the other
two must follow; the mismatch surfaces as a `state key mismatch` error from
`resolve_spec_from_policy` at startup, not as a silently worse success rate.

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

**Keep `execution_horizon` identical across variants.** It sets the closed-loop
rate, which strongly affects contact-rich tasks; varying it between variants would
confound the tactile comparison. The default of 8 against UniVTAC's
`step_lim = 300` keeps roughly 37 decisions per episode.

## Calibration you must check

These are honest unknowns that no amount of code can settle, and each can
depress a success rate to zero while looking like a negative tactile result.

**1. Gripper convention.** UniVTAC wants metres of finger opening bounded by
`RobotManager.gripper_max_qpos` (0.039 m), with larger meaning *more open*.
DROID-style checkpoints usually emit a normalised scalar where 1.0 means
*closed*. `GripperConvention(invert=True)` is the default for the zero-shot
baseline and `invert=False` for variants finetuned through this repo's converter
(which writes the UniVTAC convention). Verify against your checkpoint before
trusting a low number: log a few predicted `gripper_position` values and check
whether the fingers close when they should.

**2. Action space for the zero-shot variant.** The DROID tag returns `eef_9d`
(relative EEF), `joint_position` (relative joint) and `gripper_position`.
`--action-type qpos` uses `joint_position + gripper_position`, which maps
directly onto UniVTAC's 8-D `qpos` path; `--action-type ee` uses `eef_9d`
instead, which routes through cuRobo planning. They are not equivalent in
practice — the joint path is more faithful to what the model predicts, the EEF
path is more robust to joint-space offsets between DROID's Panda mounting and
UniVTAC's. Try both on one task before committing the sweep.

**3. Camera correspondence.** The baseline variant maps UniVTAC's `head` camera onto
DROID's `exterior_image_1_left` and `wrist` onto `wrist_image_left`. The
viewpoints are similar in kind, not calibrated to each other. Note also that
`policy/task_settings.json` marks most tasks `camera_type: head` — `lift_can`
and `insert_tube` are the ones with `all` — so on head-only tasks the wrist
stream may be a duplicate or absent, and a zero-shot two-camera tag is being
fed something it was not trained on.

**4. Control rate.** UniVTAC steps at `decimation = 1` over a 1/120 s sim step;
DROID data is 15 Hz. A per-step joint delta learned at 15 Hz means something
different at the sim's rate. This mostly affects the zero-shot variant; a finetune
learns the deployment rate from the data.

## Comparability with UniVTAC's ACT

The paper's ACT recipe, verbatim: *"For each task, we collected 50 episodes of
synthetic data used for training. All ACT models were trained for a total of
4,000 optimization steps using a batch size of 64. The learning rate was set to
1x10^-5 for both the vision and tactile encoders, with a weight decay of
1x10^-4."*

Three of those four knobs we can match exactly, and one we currently do not.

**The stance: this is a benchmark, not a controlled ablation.** A benchmark
fixes the task, the data, the observation space and the evaluation protocol.
Everything downstream -- optimizer, schedule, chunk length, which modules are
trainable -- is the method's business, and GR00T N1.7 runs on **its own
defaults**. Requiring GR00T to use ACT's learning rate would answer "how does
GR00T do under ACT's recipe?", which nobody is asking, and would penalise the
model that is not ACT.

| Knob | Paper (ACT) | This repo | |
| --- | --- | --- | --- |
| training episodes | 50 per task | **100 per task** (the full release) | **deliberate difference -- disclose** |
| cameras | per-task, see below | per-task, matched | **held fixed** |
| evaluation rollouts | **100** | **100**, their seed sequence | **held fixed** |
| optimization steps | 4,000 | `MAX_STEPS=10000` (N1.7 default) | GR00T's own |
| batch size | 64 | 64 (N1.7 default -- coincides) | GR00T's own |
| learning rate | 1e-5 (encoders) | N1.7 default | GR00T's own |
| weight decay | 1e-4 | N1.7 default | GR00T's own |
| action chunk | 50, time aggregation | `ACTION_HORIZON = 40` (N1.7 default and ceiling) | GR00T's own |

**The one trap.** "GR00T at its best" is where test-set fitting enters. There
is no validation split and `launch_finetune.py` exposes no eval metric, so
tuning the step count or the learning rate *against the 100 evaluation
rollouts* — or picking among saved checkpoints by success rate — turns a
benchmark result into an upper bound fitted to the test set. That is the only
thing here that would genuinely invalidate the comparison, and it is far easier
to do by accident than any hyperparameter mismatch.

**Clean way out:** tune on a task you do not report. Sweep on `lift_bottle`
(same pipeline, 50 episodes, not one of the three reported tasks), then apply
the chosen recipe unchanged to `insert_hole`, `insert_tube` and `pull_out_key`.
Genuinely tuned, zero contamination, and it states in one sentence.

**Caveat to report either way:** a tuned GR00T against an untuned published ACT
is partly a tuning-effort comparison. True of nearly every benchmark table; the
accepted mitigation is to say what you spent and not tune on the test rollouts.

The knobs are exposed for a deliberate sweep, left unset so GR00T's defaults
apply:

```bash
LEARNING_RATE=1e-5 WEIGHT_DECAY=1e-4 bash slurm/submit_benchmark.sh
```

`benchmark_task.sbatch` records the learning rate and weight decay in its
pinned recipe alongside the GPU count and step budget, and refuses a
resubmission that changes them — otherwise a walltime kill could train the
second variant at a different learning rate and confound the ablation
invisibly.

Confirm GR00T's actual defaults rather than trusting the extrapolation above:

```bash
"${GROOT_PYTHON}" "${GROOT_ROOT}/gr00t/experiment/launch_finetune.py" --help \
    | grep -A2 -E "learning-rate|weight-decay|warmup|global-batch-size"
```

One further caveat on the comparison, carried over from `docs/STATUS.md`:
before claiming a head-to-head against their *published* numbers, confirm how
many evaluation episodes and which seeds those used. Ours follow their
`1_000_000 * (1 + seed)` convention, but 50 versus 100 evaluation seeds are not
comparable intervals.

## What the paper actually specifies

Read from the paper PDF, Appendix B ("Implementation Details for Simulation
Experiments") and Section V-A / Table I. Quote everything rather than
paraphrasing, because two of these were guessed wrong before the PDF was read.

**Training.** *"For each task, we collected 50 episodes of synthetic data used
for training. All ACT models were trained for a total of 4,000 optimization
steps using a batch size of 64. The learning rate was set to 1e-5 for both the
vision and tactile encoders, with a weight decay of 1e-4."*

Note the learning rate is stated **for the encoders** — ACT's original code
splits `--lr` (transformer) from `--lr_backbone` (vision). The transformer's
learning rate is not stated in the paper; ACT's upstream default is also 1e-5.

**Architecture and action chunking.** *"The model employs a transformer-based
architecture with 4 layers in the encoder and 7 layers in the decoder. For
positional encoding, we adopt fixed sine-cosine embeddings for visual features,
while learnable positional embeddings are used for tactile features. The model
predicts a sequence of **50 future actions** from the current observation, which
are executed using **time aggregation** to produce smoother and more stable
motor outputs."*

**Cameras.** *"Visual observations are primarily captured from a third-person
camera view. However, task-specific configurations vary: both insert tube and
lift bottle utilize multi-view inputs, combining third-person and wrist-mounted
camera views; all other tasks use only the third-person view."*

**Evaluation.** *"All policies are trained on 50 automatically collected full
trajectories per task and evaluated over **100 test rollouts**."*

**Proprioception is not specified.** The paper never states what robot state ACT
receives. `policy/ACT/train_config*.yml` declares `state_dim: 8`, while
`embodiment/joint` on disk is 9-D (see [UPSTREAM.md](UPSTREAM.md)). Treat our
17-D state as a disclosed difference, not a matched setting.

### Table I — ACT without tactile input, per task

This is the row to compare against. **Do not use the 30.9 average** unless you
run all eight tasks.

| Task | ACT (vision only) | ACT + UniVTAC Encoder | VITaL |
| --- | --- | --- | --- |
| Lift Bottle | 42.0 | 71.0 | 72.0 |
| Pull-out Key | **28.0** | 46.0 | 47.0 |
| Lift Can | 20.0 | 29.0 | 8.0 |
| Put Bottle in Shelf | 28.0 | 31.0 | 32.0 |
| Insert Hole | **19.0** | 24.0 | 25.0 |
| Insert HDMI | 15.0 | 28.0 | 6.0 |
| Insert Tube | **45.0** | 56.0 | 34.0 |
| Grasp Classify | 50.0 | 99.0 | 100.0 |
| *Average (8 tasks)* | *30.9* | *48.0* | *40.5* |

Bold are the three tasks currently in flight. At 100 rollouts a Wilson 95 %
interval near 20 % is roughly ±8 points, so `insert_hole` at 19.0 % is
`[12.4, 27.8]` — a beatable bar, but only with 100 rollouts of our own.

### Three things we cannot or do not match

| Item | Paper (ACT) | Us | Status |
| --- | --- | --- | --- |
| action chunk length | **50** | `ACTION_HORIZON = 40` | **cannot match**, and we take N1.7's ceiling. Its `action_horizon` is 40 and `validate_action_horizons` rejects anything above it, so 50 is unreachable |
| chunk execution | **time aggregation** (re-plans and averages every step) | receding horizon, `EXECUTION_HORIZON=16` | **not implemented.** `univtac_groot/receding_horizon.py` does execute-k-then-replan only; there is no temporal ensembling. `EXECUTION_HORIZON=1` re-plans every step, which is the closest behaviour, at 16x the inference cost |
| robot state | unspecified; config says 8-D | 17-D (`eef_9d` + joints + gripper) | disclose |

The execution difference is not neutral. Time aggregation re-plans every
environment step, so ACT is markedly more closed-loop than GR00T at
`EXECUTION_HORIZON=16` — and closed-loop control is exactly what helps on
contact-rich insertion. That difference currently **favours ACT**, so it is a
conservative setting for us rather than a flattering one, which is worth saying
in the writeup either way.

## Fairness checklist against the paper's ACT numbers

Ordered by how badly each one can invalidate the comparison. The first two are
the ones that would make a GR00T win meaningless.

**1. Camera sets — settled by the paper, and now enforced per task.**
The paper states it directly: *"task-specific configurations vary: both insert
tube and lift bottle utilize multi-view inputs, combining third-person and
wrist-mounted camera views; all other tasks use only the third-person view."*

So, for the three tasks in flight:

| Task | Cameras |
| --- | --- |
| `insert_hole` | `head` only |
| `insert_tube` | `head` + `wrist` |
| `pull_out_key` | `head` only |

This **contradicts `policy/task_settings.json`**, which marks `lift_can` — not
`lift_bottle` — as `camera_type: all`. The paper describes what was trained, so
the paper wins. (And per [UPSTREAM.md](UPSTREAM.md), ACT's `process_data.py:12`
reads `task_settings.json` through a path that does not resolve, under an
`if path.exists()` guard, so it silently defaults every task to `head` anyway —
a third reason not to trust that file.)

`univtac_groot.variants.MULTI_VIEW_TASKS` is the single source of truth, used
by both the dataset converter and both modality configs, so the training data
and the model's declared inputs cannot drift apart. The configs resolve it from
`TASK` and **raise** if `TASK` is absent rather than defaulting, because giving
GR00T a wrist camera ACT never had is an unfair advantage that is invisible in
the results. `UNIVTAC_VIDEO_KEYS=head,wrist` overrides it deliberately.

Datasets converted before this was enforced carry both streams; reconvert the
head-only tasks.

**2. Re-evaluate their released ACT checkpoint under our evaluator.**
`data/download.sh --checkpoint` ships it. Running it through *our* harness with
*our* seeds and episode count makes the comparison apples-to-apples by
construction, and doubles as a validation of the harness: if you reproduce
~30.9 % vision-only, the pipeline is trustworthy; if you do not, you have found
a protocol difference *before* publishing a claim built on it. This is the
single highest-value thing on this list. It needs
`tools/univtac_patches/fix_act.py` first — the shipped encoder path
(`encoder/checkpoints/resnet18/.../best.pth`) was never published, and a
missing encoder checkpoint leaves the encoder randomly initialised **with no
warning**.

**3. Per-task numbers, not the benchmark average.** The 30.9 % / 48.0 % figures
are averaged over all eight tasks. Training three tasks and comparing a
three-task mean against their eight-task mean is invalid. Pull ACT's per-task
rates from the paper's table and compare task by task.

**4. Training episodes: 100 — the full release, and a deliberate difference.**
ModelScope ships 100 episodes per task (`0.hdf5`..`99.hdf5`); the paper trained
ACT on 50. We use all 100, so **GR00T trains on twice the demonstrations ACT
had.** That is a real advantage and it must appear next to the number, not in an
appendix. Two things make it defensible: 100 is the released dataset rather
than a private collection, and there is no way to verify *which* 50 the paper
used anyway. But state it plainly — e.g. "GR00T N1.7, 100 released
demonstrations per task; ACT numbers as published, 50 demonstrations."

At 10,000 steps x batch 64 that is 640,000 samples over ~28,932, so ~22
epochs.

**5. Evaluation episodes: 100.** Settled by the paper — *"evaluated over 100
test rollouts."* `EPISODES` now defaults to 100 everywhere; it was 50, which
would have produced intervals roughly 1.4x wider than theirs and invited an
apples-to-oranges comparison. Our seeding already follows their
`1_000_000 * (1 + seed)` convention and errored episodes decrement `test_num`
the same way, so the protocol matches.

**6. Task config.** Keep `TASK_CONFIG` consistent between download, conversion
and evaluation, and confirm which config their ACT models were trained on.

**7. Differences to disclose rather than fix.** These are legitimate
method differences, not unfairness, but they belong next to the number:

* **State width.** Ours is 17-D (`eef_9d` 9 + `joint_position` 7 + gripper 1);
  ACT's `train_config*.yml` declares `state_dim: 8`. The end-effector pose is a
  known function of the joints, so this is a reparameterisation rather than
  extra information — but say so.
* **Action parameterisation.** `launch_finetune.py` hardcodes
  `use_relative_action = True` and our config marks `joint_position` RELATIVE,
  `gripper_position` ABSOLUTE; ACT predicts absolute joint targets.
* **Action / execution horizon.** Ours is `ACTION_HORIZON = 40` with
  `EXECUTION_HORIZON=16`, against GR00T N1.7's default of 40. ACT uses its own
  chunk size and may use temporal ensembling. This changes the effective
  control rate, so check what their evaluator does.
* **Image preprocessing.** GR00T resizes to 256x256 and expects RGB (the
  converter swaps channels, since UniVTAC stores BGR); ACT uses its own
  pipeline at its own resolution.
* **Optimiser.** Each method at its own tuned settings — see the learning-rate
  discussion above. Report both rather than pretending they match.

**8. Do not select checkpoints on evaluation success.** There is no validation
split and `launch_finetune.py` exposes no eval metric, so picking the best of
four saved checkpoints by success rate is test-set fitting. Fix the step count
in advance, or hold out episodes.

## Reading the results

`scripts/compare_ablation.py` reports per-task and pooled success rates with
Wilson 95 % intervals, plus whether the intervals overlap. With 50 episodes per
task an interval is roughly ±14 points near 50 %, so a single task rarely
separates the variants; the pooled and macro rows across the eight tasks are the
numbers to read. Overlap is a conservative signal, not a hypothesis test —
report both intervals rather than claiming significance.

UniVTAC's own reference numbers for context: ACT at 30.9 % vision-only versus
48.0 % with their tactile encoder, averaged over the benchmark. That is a
different policy class and a different tactile pathway, so it sets expectations
about effect size, not a target.

`episodes_errored` is tracked separately from failures and excluded from the
rate, matching UniVTAC's evaluator, which decrements `test_num` on an exception
rather than scoring it. A non-trivial error count means a broken configuration,
not a hard task — check the JSONL's `error` field.
