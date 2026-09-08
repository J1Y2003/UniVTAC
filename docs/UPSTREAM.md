# Upstream facts this code depends on

Every non-obvious assumption in this repo, with its source. Checked against
`NVIDIA/Isaac-GR00T` and `univtac/UniVTAC` at `main` in September 2026. If you
update either dependency, re-check this list first — most of these are the kind
of detail that changes silently between releases.

## GR00T N1.7

| Fact | Value | Source |
| --- | --- | --- |
| Interpreter | **Python 3.12**, CUDA 12.8 (dGPU) | `README.md` "Installation" |
| Dependency manager | `uv sync --python 3.12`; gr00t lands in `.venv` | same |
| Video backend | `torchcodec` only; **FFmpeg 4-7**, H.264 safe, AV1 not guaranteed | same |
| Gated dependency | `nvidia/Cosmos-Reason2-2B`, loaded by *every* checkpoint | same |
| Max state dimension | **132** | `gr00t/configs/model/gr00t_n1d7.py::max_state_dim` |
| Max action dimension | 132 | same, `max_action_dim` |
| Default action horizon | **40** | same, `action_horizon` |
| Max embodiments | 32 | same, `max_num_embodiments` |
| VLM backbone | `nvidia/Cosmos-Reason2-2B` (gated) | model card / README |
| Policy class | `gr00t.policy.gr00t_policy.Gr00tPolicy(embodiment_tag, model_path, *, device, strict)` | `gr00t/policy/gr00t_policy.py` |
| Nested observation format | `{"video": {k: (B,T,H,W,C) uint8}, "state": {k: (B,T,D) float32}, "language": {k: [[str]]}}` | `Gr00tPolicy.check_observation` |
| Flat sim observation format | `video.<k>` (B,T,H,W,C) uint8, `state.<k>` (B,T,D) float32, language as `tuple[str]` (B,) | `Gr00tSimPolicyWrapper.check_observation` |
| Action output | `{"action.<k>": (B,T,D) float32}`, already un-normalised and de-relativised | `Gr00tPolicy._get_action` step 5, `Gr00tSimPolicyWrapper._get_action` |
| Server endpoints | `ping`, `kill`, `reset`, `get_action`, `get_modality_config` | `gr00t/policy/server_client.py::PolicyServer.__init__` |
| Wire format | ZMQ REQ/REP, msgpack + msgpack-numpy, `{"endpoint", "data", "api_token"}`, errors in-band as `{"error": ...}` | `PolicyServer.run` |
| Pickle boundary | object-dtype ndarrays refused on both sides | `MsgSerializer._safe_encode/_safe_decode` |
| Execution horizon contract | `1 <= n_action_steps <= action_horizon`; `action.delta_indices` must be contiguous `range(0, H)` | `gr00t/eval/_horizon_contract.py::PolicyHorizonSpec` |
| Observation history indexing | offset `d` reads `buffer[d - 1]`; reset fills the buffer with copies of the first frame | `gr00t/eval/sim/wrapper/multistep_wrapper.py::MultiStepWrapper._get_obs`, `.reset` |
| Delta-index rules | non-positive, ending at 0, evenly and positively spaced | `MultiStepWrapper.assert_delta_indices` |
| Custom embodiment registration | a `*_config.py` calls `register_modality_config(cfg, embodiment_tag=...)` as an import side effect | `examples/SO100/so100_config.py`, `gr00t/configs/data/embodiment_configs.py` |
| Dataset format | GR00T LeRobot = LeRobot **v2** + `meta/modality.json` | `getting_started/data_preparation.md` |
| `meta/info.json` schema | `codebase_version: "v2.1"`, `features`, `data_path`, `video_path`, `chunks_size: 1000` | `demo_data/cube_to_bowl_5/meta/info.json` |
| Annotation plumbing | `modality.json` key drops the `annotation.` prefix and points at `task_index` | `data_preparation.md`, `demo_data/.../modality.json` |

### Embodiment tags

From `gr00t/data/embodiment_tags.py`:

* **`PRETRAIN_TAGS`** — in the base checkpoint, usable zero-shot:
  `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, `XDOF`, `XDOF_SUBTASK`, `REAL_G1`,
  `REAL_R1_PRO_SHARPA{,_HUMAN,_MAXINSIGHTS,_MECKA}`.
* **`POSTTRAIN_TAGS`** — need a finetuned checkpoint: `UNITREE_G1`,
  `UNITREE_G1_SONIC`, `SIMPLER_ENV_GOOGLE`, `SIMPLER_ENV_WIDOWX`, `LIBERO_PANDA`.
* **`FINETUNE_ONLY_TAGS`** — in **no** shipped checkpoint: `NEW_EMBODIMENT`,
  `ROBOCASA_PANDA_OMRON`, `ROBOCASA_GR1_TABLETOP`.

`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` is the natural zero-shot choice for
UniVTAC because DROID is itself a Franka Panda. Its config:

```python
"video":  delta_indices=[-15, 0], keys=["exterior_image_1_left", "wrist_image_left"]
"state":  delta_indices=[0],      keys=["eef_9d", "gripper_position", "joint_position"]
"action": delta_indices=range(40), keys=["eef_9d", "gripper_position", "joint_position"]
          eef_9d          RELATIVE, EEF,     XYZ_ROT6D
          gripper_position ABSOLUTE, NON_EEF, DEFAULT
          joint_position   RELATIVE, NON_EEF, DEFAULT
"language": ["annotation.language.language_instruction"]
```

The `[-15, 0]` video offsets are the reason `ObsHistory` exists: a single frame
fails `check_observation` on the temporal dimension.

## UniVTAC

| Fact | Value | Source |
| --- | --- | --- |
| Interpreter | **Python 3.10**, torch 2.5.1+cu118 | `docs/Installation.md` "Requirements" |
| Simulator stack | Isaac Sim 4.5 + Isaac Lab 2.1.1 + TacEx (UIPC) + cuRobo | `docs/Installation.md` |
| TacEx source | must be built from `third_party/TacEx`, not upstream | same ("Important" note) |
| Robot | Franka Panda: 7 arm joints + 2 finger joints | `envs/robot/robot.py` (`_arm_joint_names`, `_gripper_joint_names`) |
| Benchmark tasks | `lift_bottle`, `lift_can`, `insert_HDMI`, `insert_hole`, `insert_tube`, `pull_out_key`, `put_bottle_in_shelf`, `grasp_classify` (+ `collect` for data-gen) | `envs/` |
| Observation dict | `{'observation': {cam: {'rgb': ...}}, 'tactile': {sensor: {...}}, 'embodiment': {'joint','ee'}, 'actor', 'step', 'atom'}` | `envs/_base_task.py::BaseTask._get_observations` |
| Proprioception | `joint` = `robot.data.joint_pos` → **9-D**; `ee` = pose → **7-D** (xyz + quat wxyz) | `envs/robot/robot.py::RobotManager.get_observations` |
| Which obs are populated | only the data types listed under `observations.*` in the task config | `_get_observations` guards on `cfg.obs_data_type` |
| Tactile data types | `rgb`, `rgb_marker`, `depth` (height map), `marker` (marker motion), `points`, `pose` | `envs/sensors/tactile.py::VisualTactileSensor.get_observations` |
| Tactile sensors | GelSight Mini (320×240), GF225 (480×480), XenseWS (320×240) | same file's `create_*_cfg` |
| Marker count | the `create_*_cfg` marker grid (9×7 for GelSight Mini) is the *config* default; the released `isaac45` dumps carry **M=1200** per raster, so do not size anything off 64 | config vs. verified on disk |
| Supported for eval | **GelSight Mini only** | `README.md` |
| Action signatures | `qpos` → 8-D (7 arm + 1 gripper); `ee` → 8-D (pos3 + quat4 + gripper); `delta_ee` → 7-D | `BaseTask.take_action` docstring and body |
| Gripper range | `gripper_max_qpos = 0.039` m per finger; larger = more open | `envs/robot/robot.py`, `RobotCfg` |
| Episode cap | `BaseTaskCfg.step_lim = 300` | `envs/_base_task.py` |
| Success signal | `check_success()` → bool, sets `eval_success`; **no shaped reward exists** | `BaseTask.take_action`, `check_success` |
| Eval seeding | `1_000_000 * (1 + seed)`, walking consecutive seeds; errored episodes decrement `test_num` | `scripts/eval_policy.py::eval_policy` |
| Env count | forced to 1 | `scripts/eval_policy.py` (`args_cli.num_envs = 1`) |
| Policy contract | `Policy(args)` / `encode_obs` / `eval(task, obs)` / `reset` under `policy/<Name>/` + `deploy.yml` | `docs/Deploy.md` |
| Bootstrap order | `AppLauncher` **must** run before importing `envs.*` | `scripts/eval_policy.py` (imports after `app_launcher`) |
| HDF5 keys (on disk) | `observation/<cam>/rgb`, `tactile/<sensor>/{rgb,rgb_marker,depth,marker,pose}`, `embodiment/{joint,ee}`, `actor/<name>`, `step`, `atom/{id,tag}` | verified against `isaac45/lift_bottle/0.hdf5` |
| Image storage | **JPEG byte stream**, `(N,)` of `\|S<max>`, one buffer per frame; any key whose last segment contains `rgb` | `envs/utils/data.py::HDF5Handler.{img_to_stream,stream_to_img}` |
| Image colour order | `cv2.imdecode(..., IMREAD_COLOR)` → **BGR**; no `cvtColor` anywhere, so UniVTAC's own ACT trains on BGR | `stream_to_img` |
| State/action derivation | no `joint_state`/`joint_action` on disk — derived as `joint[:-1]` and `joint[1:]`, so an action is the **absolute next joint position** and N frames give N-1 transitions; all other arrays truncated `[:-1]` | `HDF5Handler.batch_gather_hdf5` |
| `embodiment/joint` width | **9** (7 arm + 2 fingers) — note ACT's `train_config*.yml` declares `state_dim: 8` | verified on disk |
| `tactile/*/depth` | `(N, 240, 320)` float32 — present in the released data, so `depth_pool` is viable | verified on disk |
| `tactile/*/marker` | `(N, 2, M, 2)` float32 (M=1200 for GelSight Mini in this release) — a *pair* of marker rasters per frame, not `(M, 2\|4)` | verified on disk |
| Sensor name aliases | `left_tactile` (current) and `left_gsmini` (older dumps) | same file's fallback try/except |
| Prior VLA integration | SmolVLA runs behind a FastAPI service in its own venv | `policy/smolvla/deploy_policy.py` |

## Why two processes

UniVTAC requires Python 3.10 and GR00T requires Python 3.12, so no shared
virtualenv exists — the split is forced, not chosen. See
[SETUP.md](SETUP.md#why-it-is-a-separate-process-not-an-import).

## Easily-mistaken upstream facts

Each of these is load-bearing and each contradicts a plausible first guess.
`guideline.md` is the project's original brief and is superseded wherever it
conflicts with this file.

1. **Custom embodiments register through
   `gr00t/configs/data/embodiment_configs.py`**, which defines
   `MODALITY_CONFIGS` and `register_modality_config`;
   `examples/SO100/so100_config.py` is the worked example. There is no
   `gr00t-leapp-export` package and no `new_embodiment_config_defaults.py`.

2. **N1.7's default action horizon is 40**, not 16 — 16 is N1.5/N1.6, and
   shipped posttrain configs use 16 or 8. Nothing here hard-codes a horizon; it
   is read from the live policy.

3. **UniVTAC's tactile sensors are camera-based.** A reading is a 320x240
   height map or a 64-marker motion field, not a small numeric array.
   Flattening either raw exceeds the 132-D state cap, which is why
   `TactileSpec` pools. UniVTAC is Franka Panda only — no other manipulator
   appears anywhere in it.

4. **`lerobot` is a dataset format here, not a library dependency.** GR00T N1.7
   ships its own `Gr00tPolicy` over `transformers`' `AutoModel`/`AutoProcessor`,
   plus its own server/client, evaluation loop and horizon contract. Only the
   on-disk format matters (v2 + `modality.json`), which
   `scripts/convert_univtac_to_lerobot.py` writes directly. Importing `lerobot`
   would add a torch dependency for no benefit.

5. **The tactile variant cannot be evaluated zero-shot**, because
   `FINETUNE_ONLY_TAGS` ships in no checkpoint. This is the single most
   consequential constraint on the experiment's design — see
   [ABLATION.md](ABLATION.md).

## Known environment constraints

**cuDNN must match torch's pin.** GR00T pins `torch==2.9.0` on cu128 (CUDA
12.8) in `pyproject.toml`, alongside a URL-pinned flash-attn wheel built for
that pair, so the CUDA version is not practically adjustable. torch 2.9.0+cu128
requires **`nvidia-cudnn-cu12==9.10.2.21`** (= `91002`); a venv carrying a
different cuDNN reports `CUDNN_STATUS_NOT_INITIALIZED` while cuBLAS keeps
working, which reads convincingly like a too-old driver and is not.

Two claims to reject on sight, because both are wrong and both are plausible:
that 9.13 fails on a driver older than r570 (it is the pin mismatch, not the
driver), and that disabling cuDNN is a viable workaround (it costs ~86x on
the vision tower). Verify by asking the loaded library, not pip metadata, and
see
[SETUP.md](SETUP.md#cudnn-check-the-library-not-the-metadata).

**Driver versions vary across this fleet.** Observed: `worker-node3` (A100
80GB PCIe) on 550.54.14, `worker-node109` (A100-SXM4-80GB) on 550.163.01. Do
not write "the cluster's driver" as though it were one value.

## Things deliberately left to the operator

* **Gripper sign and scale** for a zero-shot checkpoint — configurable and
  documented, not guessed. See ABLATION.md, "Calibration you must check".
* **`marker` layout** — TacEx marker-motion arrays are commonly `(N, 4)` as
  `[x, y, dx, dy]` or `(N, 2)`; `MarkerLayout='auto'` infers from the trailing
  width, and `reduce_marker_field` can be pinned explicitly. The row/column
  order of the marker raster is not part of UniVTAC's public contract, so
  `_marker_to_grid` pools rather than claiming a spatial layout.
* **Video codec** — the demo dataset uses AV1; the converter writes H.264,
  which is far likelier to exist in a cluster ffmpeg build.
* **Per-task camera sets** — `policy/task_settings.json` marks most tasks
  `camera_type: head`, with `lift_can` and `insert_tube` as `all`. The variants
  request both `head` and `wrist`; drop `wrist` from `video_keys` for
  head-only tasks if the stream turns out to be absent.

## UniVTAC's own ACT baseline: path resolution is inconsistent

Relevant if you run UniVTAC's baselines for comparison numbers. Every relative
path in `policy/ACT` is resolved against the **process CWD**, and the intended
CWD is `policy/ACT` (`one.sh` calls `bash train.sh` bare and tests
`./data/sim-$task/...`; `eval.sh` opens with `cd ../..`). Three paths disagree,
so no single CWD resolves everything.

| Path | Where | Resolved against | Actually at |
| --- | --- | --- | --- |
| `imitate_episodes.py`, `./<train_config>.yml`, `./act_ckpt/…` | `train.sh` | CWD | `policy/ACT` ✅ |
| `./SIM_TASK_CONFIGS.json` (written) | `ACT/process_data.py:36` | CWD | `policy/ACT` ✅ |
| `./SIM_TASK_CONFIGS.json` (read) | `ACT/imitate_episodes.py:39` | CWD | `policy/ACT` ✅ |
| `./data/sim-<task>/…` → `dataset_dir` | `ACT/process_data.py:9` | CWD | `policy/ACT` ✅ |
| raw `data/<task>/<config>/*.hdf5` | `policy/_base_data_preprocessor.py:11` | **absolute, repo root** | ✅ any CWD |
| `task_settings.json` | `ACT/process_data.py:12`, `Path(__file__).parent` | `policy/ACT/` | **`policy/`** ❌ |
| `SIM_TASK_CONFIGS.json` | `ACT/constants.py:6`, `dirname(__file__)` | `policy/ACT/` | **`policy/`** ❌ |
| `encoder/checkpoints/…/best.pth` | all `ACT/train_config*.yml` | CWD → `policy/ACT/encoder/` | repo root ❌ |

The committed `policy/SIM_TASK_CONFIGS.json` (`sim-lift_bottle-clean-2`, 2
episodes) is a leftover from running `process_data.py` from `policy/`.
`_base_data_preprocessor.py` still carries a duplicate `main()` in which the
`__file__`-relative `task_settings.json` lookup *is* correct;
`ACT/process_data.py` is a copy that did not adjust for the moved `__file__`.

Four failures that are **silent** rather than loud:

1. `camera_names: cam_high` in `train_config{,_all,_vision,_freeze,_scrach}.yml`
   is an ALOHA leftover. The preprocessor saves `cam_head`/`cam_wrist` and
   `ACT/utils.py` indexes `/observations/images/{cam_name}` with no aliasing, so
   this raises `KeyError: cam_high` only once the first batch is drawn.
2. `task_settings.json` is read under `if path.exists()`, so the miss above
   silently defaults `camera_type` to `head` — dropping the wrist camera for
   `lift_can` and `insert_tube`, the two tasks configured `all`.
3. `ACT/detr/models/backbone.py:150` loads the tactile encoder under
   `if ckpt and Path(ckpt).exists()`. A path that does not resolve leaves the
   encoder **randomly initialised** with no warning — in an ablation asking
   "does tactile help", that manufactures a plausible null result. The shipped
   `encoder/checkpoints/resnet18/20251128-125750/best.pth` was never published;
   the released encoder is `checkpoints/encoder.pth`.
4. `data/download.sh` preserves the *published* layout, writing
   `data/isaac45/<task>/hdf5/*.hdf5`, while `BaseDataPreprocessor` reads
   `data/<task>/<config>/`. No `<config>` level exists in the download and the
   task sits one directory deeper, so downloaded data is not directly
   consumable. Episodes are named `0.hdf5`…`99.hdf5`, which `find_episodes`
   sorts by `int(path.stem)`, so a directory symlink suffices.

`tools/univtac_patches/fix_act.py` repairs all of the above against a checkout,
idempotently and with `--revert`. It edits the UniVTAC checkout only — no
dotfiles, no conda environments.

### Entry-point signatures

Confirmed against the scripts, since the argument names are easy to mistake:

```
bash eval_policy.sh   <task_name> <task_config> <policy_config> <gpu_id>
bash collect_data.sh  <task_name> <task_config> [gpu_id] [start_seed] [max_seed] [episode_num]
policy/ACT/train.sh   <task_name> <task_config> <expert_data_num> <seed> <gpu_id> [train_config]
policy/ACT/eval.sh    <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id>
```

Note that `collect_data.sh` takes a **gpu id** in position 3, not a seed — the
seed range is positions 4 and 5, defaulting to `-1`. Root `eval_policy.sh`
likewise takes a **gpu id** in position 4, not a seed,
and a **policy config** in position 3 (e.g. `ACT/deploy`, `GR00T/deploy_baseline`)
— not a task or checkpoint name. `policy/ACT/eval.sh` is stale: it invokes
`script/eval_policy.py` (the directory is `scripts/`) and
`policy/ACT/deploy_policy.yml` (the file is `deploy.yml`); `one.sh` bypasses it
in favour of the root entry point.
