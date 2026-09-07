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
| Tactile sensors | GelSight Mini (320×240, 9×7 = 64 markers), GF225 (480×480, 81), XenseWS (320×240, 220) | same file's `create_*_cfg` |
| Supported for eval | **GelSight Mini only** | `README.md` |
| Action signatures | `qpos` → 8-D (7 arm + 1 gripper); `ee` → 8-D (pos3 + quat4 + gripper); `delta_ee` → 7-D | `BaseTask.take_action` docstring and body |
| Gripper range | `gripper_max_qpos = 0.039` m per finger; larger = more open | `envs/robot/robot.py`, `RobotCfg` |
| Episode cap | `BaseTaskCfg.step_lim = 300` | `envs/_base_task.py` |
| Success signal | `check_success()` → bool, sets `eval_success`; **no shaped reward exists** | `BaseTask.take_action`, `check_success` |
| Eval seeding | `1_000_000 * (1 + seed)`, walking consecutive seeds; errored episodes decrement `test_num` | `scripts/eval_policy.py::eval_policy` |
| Env count | forced to 1 | `scripts/eval_policy.py` (`args_cli.num_envs = 1`) |
| Policy contract | `Policy(args)` / `encode_obs` / `eval(task, obs)` / `reset` under `policy/<Name>/` + `deploy.yml` | `docs/Deploy.md` |
| Bootstrap order | `AppLauncher` **must** run before importing `envs.*` | `scripts/eval_policy.py` (imports after `app_launcher`) |
| HDF5 keys | `observation/<cam>/rgb`, `tactile/<sensor>/rgb_marker`, `embodiment/joint_state`, `embodiment/joint_action` | `policy/_base_data_preprocessor.py` |
| Sensor name aliases | `left_tactile` (current) and `left_gsmini` (older dumps) | same file's fallback try/except |
| Prior VLA integration | SmolVLA runs behind a FastAPI service in its own venv | `policy/smolvla/deploy_policy.py` |

## Why two processes

UniVTAC requires Python 3.10 and GR00T requires Python 3.12, so no shared
virtualenv exists — the split is forced, not chosen. See
[SETUP.md](SETUP.md#why-it-is-a-separate-process-not-an-import).

## Corrections to the original guideline

The guideline flagged its own references as preliminary. What research changed:

1. **`gr00t-leapp-export` / `new_embodiment_config_defaults.py` do not exist.**
   The real equivalents are `gr00t/configs/data/embodiment_configs.py` (which
   defines `MODALITY_CONFIGS` and `register_modality_config`) and
   `examples/SO100/so100_config.py` (the worked custom-embodiment example). This
   repo follows those.

2. **The action horizon is 40, not 16.** The guideline's "e.g. 16-step horizons"
   describes N1.5/N1.6. N1.7's default is 40; shipped posttrain configs use 16 or
   8. Nothing here hard-codes a horizon — it is read from the live policy.

3. **UniVTAC's tactile sensors are camera-based.** The guideline's earlier
   premise of a small numeric tactile array does not match the simulator: a
   reading is a 320×240 height map or a 64-marker motion field. Flattening
   either raw exceeds the 132-D state cap, hence `TactileSpec`'s pooling. (The
   guideline's earlier mention of a RobotEra XHAND1 also does not appear
   anywhere in UniVTAC, which is Franka Panda only; the guideline was corrected
   on this point.)

4. **`lerobot` is not used for model instantiation.** GR00T N1.7 ships its own
   `Gr00tPolicy` over `transformers`' `AutoModel`/`AutoProcessor`, and its own
   server/client, evaluation loop and horizon contract. `lerobot` remains
   relevant only as the *dataset format* (v2 + `modality.json`), which
   `scripts/convert_univtac_to_lerobot.py` writes directly. Pulling in
   `lerobot` as a library would add a torch dependency for no benefit.

5. **The tactile arm cannot be evaluated zero-shot.** See
   [ABLATION.md](ABLATION.md). This is the single most consequential finding for
   the experiment's design, and it follows from `FINETUNE_ONLY_TAGS` shipping in
   no checkpoint.

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
  `camera_type: head`, with `lift_can` and `insert_tube` as `all`. The arms
  request both `head` and `wrist`; drop `wrist` from `video_keys` for
  head-only tasks if the stream turns out to be absent.
