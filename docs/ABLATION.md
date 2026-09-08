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

```bash
# 0. Collect demonstrations with UniVTAC (its own tooling)
cd "$UNIVTAC_ROOT" && bash collect_data.sh insert_hole demo

# 1. Convert once per variant — the state layout differs
python scripts/convert_univtac_to_lerobot.py --task insert_hole \
    --raw-dir "$UNIVTAC_ROOT/data/insert_hole/demo" \
    --out "$DATA/univtac-insert_hole-baseline" --variant baseline_finetuned
python scripts/convert_univtac_to_lerobot.py --task insert_hole \
    --raw-dir "$UNIVTAC_ROOT/data/insert_hole/demo" \
    --out "$DATA/univtac-insert_hole-tactile" --variant tactile

# 2. Finetune both, same recipe
sbatch --export=ALL,VARIANT=baseline_finetuned,DATASET=$DATA/univtac-insert_hole-baseline slurm/finetune.sbatch
sbatch --export=ALL,VARIANT=tactile,DATASET=$DATA/univtac-insert_hole-tactile           slurm/finetune.sbatch

# 3. Evaluate all three variants
bash slurm/submit_ablation.sh   # VARIANTS="baseline baseline_finetuned tactile"

# 4. Table
python scripts/compare_ablation.py --baseline-variant baseline_finetuned --tactile-variant tactile
```

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

| Knob | Paper (ACT) | This repo | Matched? |
| --- | --- | --- | --- |
| optimization steps | 4,000 | `MAX_STEPS=4000` (default) | yes |
| batch size | 64 | GR00T `--global-batch-size` 64 (default) | yes |
| learning rate | 1e-5 | GR00T default, **~1e-4** peak (see below) | **no, by default** |
| weight decay | 1e-4 | GR00T default | **no, by default** |
| training episodes | **50** per task | **100** per task | **no** |

**Steps and batch size line up for free**, which is worth noting: 4,000 steps
at batch 64 is 256,000 samples seen, and GR00T's default global batch size is
already 64. So GR00T gets the same number of gradient updates over the same
batch size as the published baseline.

**The episode count does not.** Our converted datasets carry 100 episodes
(28,932 samples); the paper used 50. Matching steps and batch while doubling
the data means the same gradient signal drawn from twice the demonstrations:
~8.9 epochs for us versus ~17.7 for them. Neither setting is wrong, but they
are different experiments, and the difference favours GR00T. Two honest
options:

* **Match their data.** Convert and train on 50 episodes, so the only
  difference from ACT is the policy class. This is the cleaner head-to-head.
* **Keep 100 episodes** and state plainly that GR00T saw twice the
  demonstrations at the same step budget. Defensible, but it must appear next
  to the number, not in an appendix.

Whichever you pick, it applies identically to `tactile` and
`baseline_finetuned`, so it never threatens the *internal* ablation — only the
external comparison to the ACT row.

**The optimiser settings are a real decision, not an oversight.** GR00T's
`launch_finetune.py` does not use 1e-5. In a real run the logged learning rate
climbed +2e-7 per step through warmup, which extrapolates to a peak near 1e-4 —
roughly 10x the paper. That is not obviously wrong: ACT is a small policy with
a pretrained ResNet encoder trained largely from scratch, whereas this is a
finetune of 1.62 B trainable parameters, where 1e-4 is on the aggressive side.
Both scripts now expose the knobs so the choice is explicit rather than
inherited:

```bash
LEARNING_RATE=1e-5 WEIGHT_DECAY=1e-4 bash slurm/submit_overnight.sh
```

`overnight_ablation.sbatch` records the learning rate and weight decay in its
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

## Fairness checklist against the paper's ACT numbers

Ordered by how badly each one can invalidate the comparison. The first two are
the ones that would make a GR00T win meaningless.

**1. Camera sets — the biggest live risk.**
`configs/modality/univtac_baseline_config.py` requests `["head", "wrist"]` for
*every* task. But `policy/task_settings.json` marks most tasks
`camera_type: head`, with only `lift_can` and `insert_tube` as `all`. So for
`insert_hole` and `pull_out_key` our pipeline feeds GR00T a **wrist camera ACT
never saw** — a straightforward unfair advantage.

It is worse than that. `docs/UPSTREAM.md` records that ACT's
`process_data.py:12` reads `task_settings.json` through a `__file__`-relative
path that does not resolve, under an `if path.exists()` guard — so the lookup
**silently misses and defaults `camera_type` to `head`**. If the published
numbers came from that code path, ACT was head-only for *all* tasks, including
`insert_tube`. Verify against their released checkpoint rather than the config
file's stated intent, then drop `wrist` from `video_keys` to match. Note also
that the released dumps may not even contain a wrist stream for head-only
tasks.

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

**4. Training episodes: 50, not 100.** Convert with `EPISODES=50`. This also
makes the epoch count match for free — 4,000 steps x batch 64 = 256,000 samples
over 50 x ~310 frames is ~16.5 epochs, essentially their number.

**5. Evaluation episodes and seeds.** Confirm how many evaluation episodes
their published numbers use. Our seeding follows their
`1_000_000 * (1 + seed)` convention and errored episodes decrement `test_num`
the same way, so the protocol matches; only N is in question. Evaluating *more*
episodes than they did is fine and tightens your interval — just report both
Ns, and never compare a 50-episode interval to a 100-episode one as though they
were equivalent.

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
* **Action / execution horizon.** Ours is `ACTION_HORIZON = 16` with
  `EXECUTION_HORIZON=8`, against GR00T N1.7's default of 40. ACT uses its own
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
