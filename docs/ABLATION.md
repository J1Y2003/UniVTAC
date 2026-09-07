# The ablation: does tactile in the state vector help GR00T N1.7?

The study compares two arms that differ in exactly one respect:

* **Arm A (baseline).** Vision, language instruction, 1-D proprioception,
  embodiment id.
* **Arm B (tactile).** The same, plus the flattened UniVTAC tactile array
  concatenated onto the proprioception vector.

`univtac_groot/arms.py` builds both from one shared `PROPRIO_FIELDS` tuple, so
the arms cannot drift apart by accident, and `tests/test_obs_adapter.py`
asserts that the proprioception slices are bit-identical between them.

---

## The constraint that shapes the whole experiment

**Arm B cannot run zero-shot on `nvidia/GR00T-N1.7-3B`.** Three facts from the
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
state dimensions. That is what `--arm baseline_finetuned` exists for; the
zero-shot baseline is a useful reference point, not Arm B's control.

Reporting the zero-shot baseline against a finetuned tactile arm would
attribute the entire finetuning effect to tactile sensing. `scripts/compare_ablation.py`
takes `--baseline-arm`, so point it at `baseline_finetuned` for the headline
number and report the zero-shot arm separately.

## Running it end to end

```bash
# 0. Collect demonstrations with UniVTAC (its own tooling)
cd "$UNIVTAC_ROOT" && bash collect_data.sh insert_hole demo

# 1. Convert once per arm — the state layout differs
python scripts/convert_univtac_to_lerobot.py --task insert_hole \
    --raw-dir "$UNIVTAC_ROOT/data/insert_hole/demo" \
    --out "$DATA/univtac-insert_hole-baseline" --arm baseline_finetuned
python scripts/convert_univtac_to_lerobot.py --task insert_hole \
    --raw-dir "$UNIVTAC_ROOT/data/insert_hole/demo" \
    --out "$DATA/univtac-insert_hole-tactile" --arm tactile

# 2. Finetune both, same recipe
sbatch --export=ALL,ARM=baseline_finetuned,DATASET=$DATA/univtac-insert_hole-baseline slurm/finetune.sbatch
sbatch --export=ALL,ARM=tactile,DATASET=$DATA/univtac-insert_hole-tactile           slurm/finetune.sbatch

# 3. Evaluate all three arms
bash slurm/submit_ablation.sh   # ARMS="baseline baseline_finetuned tactile"

# 4. Table
python scripts/compare_ablation.py --baseline-arm baseline_finetuned --tactile-arm tactile
```

## How tactile enters the observation

UniVTAC's tactile sensors are **camera-based**: `envs/sensors/tactile.py` wraps
TacEx GelSight Mini, ViTai GF225 and XenseWS, and a reading is an image, not a
short vector. `TactileManager.get_observations` can emit `rgb`, `rgb_marker`,
`depth` (a gel height map), `marker` (a marker-motion field), `points` and
`pose`. For a GelSight Mini the height map is 320×240 and the marker grid is
9×7 = 64 markers.

Proprioception spends 17 of GR00T's 132 state dimensions
(`eef_9d` 9 + `joint_position` 7 + `gripper_position` 1), leaving **115 for
tactile across all sensors**. Raw flattening is therefore impossible: 76 800
dims per sensor for the height map, 128 for the marker field with two sensors.
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
untouched), so it is offered as a third arm rather than the default.

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

**Keep `execution_horizon` identical across arms.** It sets the closed-loop
rate, which strongly affects contact-rich tasks; varying it between arms would
confound the tactile comparison. The default of 8 against UniVTAC's
`step_lim = 300` keeps roughly 37 decisions per episode.

## Calibration you must check

These are honest unknowns that no amount of code can settle, and each can
depress a success rate to zero while looking like a negative tactile result.

**1. Gripper convention.** UniVTAC wants metres of finger opening bounded by
`RobotManager.gripper_max_qpos` (0.039 m), with larger meaning *more open*.
DROID-style checkpoints usually emit a normalised scalar where 1.0 means
*closed*. `GripperConvention(invert=True)` is the default for the zero-shot
baseline and `invert=False` for arms finetuned through this repo's converter
(which writes the UniVTAC convention). Verify against your checkpoint before
trusting a low number: log a few predicted `gripper_position` values and check
whether the fingers close when they should.

**2. Action space for the zero-shot arm.** The DROID tag returns `eef_9d`
(relative EEF), `joint_position` (relative joint) and `gripper_position`.
`--action-type qpos` uses `joint_position + gripper_position`, which maps
directly onto UniVTAC's 8-D `qpos` path; `--action-type ee` uses `eef_9d`
instead, which routes through cuRobo planning. They are not equivalent in
practice — the joint path is more faithful to what the model predicts, the EEF
path is more robust to joint-space offsets between DROID's Panda mounting and
UniVTAC's. Try both on one task before committing the sweep.

**3. Camera correspondence.** The baseline arm maps UniVTAC's `head` camera onto
DROID's `exterior_image_1_left` and `wrist` onto `wrist_image_left`. The
viewpoints are similar in kind, not calibrated to each other. Note also that
`policy/task_settings.json` marks most tasks `camera_type: head` — `lift_can`
and `insert_tube` are the ones with `all` — so on head-only tasks the wrist
stream may be a duplicate or absent, and a zero-shot two-camera tag is being
fed something it was not trained on.

**4. Control rate.** UniVTAC steps at `decimation = 1` over a 1/120 s sim step;
DROID data is 15 Hz. A per-step joint delta learned at 15 Hz means something
different at the sim's rate. This mostly affects the zero-shot arm; a finetune
learns the deployment rate from the data.

## Reading the results

`scripts/compare_ablation.py` reports per-task and pooled success rates with
Wilson 95 % intervals, plus whether the intervals overlap. With 50 episodes per
task an interval is roughly ±14 points near 50 %, so a single task rarely
separates the arms; the pooled and macro rows across the eight tasks are the
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
