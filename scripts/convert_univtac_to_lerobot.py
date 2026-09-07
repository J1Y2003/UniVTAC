"""Convert UniVTAC demonstration HDF5 into the GR00T LeRobot v2 format.

Needed because the tactile ablation variant cannot run zero-shot: extra state
dimensions require the ``NEW_EMBODIMENT`` tag, which ships in no released
checkpoint (``gr00t/data/embodiment_tags.py::FINETUNE_ONLY_TAGS``). Run this on
UniVTAC's collected demonstrations, then finetune both variants with the same recipe
(``slurm/finetune.sbatch``) so the only difference between them is the tactile
state dimensions.

Input — UniVTAC's raw collection dumps, keyed as in
``policy/_base_data_preprocessor.py``::

    data/<task>/<config>/<episode>.hdf5
      observation/head/rgb             (N,)  |S<max>  JPEG-encoded per frame
      observation/wrist/rgb            (N,)  |S<max>  JPEG-encoded per frame
      tactile/<left|right>_gsmini/rgb_marker   (N,) |S<max>  JPEG-encoded
      tactile/<left|right>_gsmini/depth        (N, 240, 320) float32
      tactile/<left|right>_gsmini/marker       (N, 2, M, 2)  float32
      embodiment/joint                 (N, 9)  float32
      embodiment/ee                    (N, 7)  float32

Two conventions here are inherited from ``envs/utils/data.py::HDF5Handler`` and
must be matched exactly, or training and evaluation disagree silently:

* **Images are JPEG byte streams, not arrays.** Any dataset whose final path
  segment contains ``rgb`` holds one encoded buffer per frame
  (``dict_to_hdf5`` writes them with ``cv2.imencode('.jpg', ...)``).
  ``stream_to_img`` decodes with ``cv2.IMREAD_COLOR``, which yields **BGR**;
  UniVTAC's own ACT pipeline then trains on BGR. GR00T's vision backbone
  expects RGB, so this converter swaps the channels -- see
  ``decode_image_stream``.
* **State and action are a one-step shift of a single array.** There is no
  ``joint_state``/``joint_action`` pair on disk;
  ``batch_gather_hdf5`` derives them as ``joint[:-1]`` and ``joint[1:]``, so an
  action is the *absolute next joint position* and an N-frame episode yields
  N-1 transitions. Every other per-frame array is truncated with ``[:-1]`` to
  stay aligned. Dumps that do carry the explicit pair are still accepted.

(Older dumps name the sensors ``*_gsmini``, newer ones ``*_tactile``; both are
accepted. The released ``isaac45`` data uses ``*_gsmini``.)

Output — the layout in ``getting_started/data_preparation.md``::

    <out>/
      meta/{info.json,episodes.jsonl,tasks.jsonl,modality.json}
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.head/episode_000000.mp4

State and action are written as single concatenated float32 arrays whose slices
``meta/modality.json`` names, exactly as the ``cube_to_bowl_5`` demo dataset
does. The state layout matches :mod:`univtac_groot.variants` so that training and
evaluation agree dimension for dimension — the converter imports the same
:class:`~univtac_groot.spec.ObsSpec` rather than restating the layout.

Requires ``pyarrow`` and ``imageio[ffmpeg]`` (or ``opencv-python``) in whichever
environment runs it; this is offline preprocessing, so run it as its own batch
job, not on a login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from univtac_groot.action_adapter import GripperConvention  # noqa: E402
from univtac_groot.variants import build_spec  # noqa: E402
from univtac_groot.obs_adapter import (  # noqa: E402
    as_uint8_hwc,
    encode_tactile_state,
    quat_wxyz_to_rot6d,
    resize_nearest,
)
from univtac_groot.spec import ObsSpec  # noqa: E402

CHUNK_SIZE = 1000
"""``info.json``'s ``chunks_size``; episodes are grouped ``chunk-{index // 1000}``."""


# --------------------------------------------------------------------------- #
# HDF5 reading
# --------------------------------------------------------------------------- #


def _first_present(group: Any, candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in group:
            return key
    return None


def decode_image_stream(raw: Any, *, bgr_to_rgb: bool = True) -> list[np.ndarray]:
    """Decode a UniVTAC image dataset into a list of ``(H, W, 3)`` uint8 frames.

    UniVTAC stores camera and tactile RGB as one JPEG buffer per frame
    (``HDF5Handler.img_to_stream``), so the dataset is ``(N,)`` of ``|S<max>``
    rather than an ``(N, H, W, 3)`` array. Mirrors ``stream_to_img``, with one
    deliberate difference: ``cv2.imdecode`` returns BGR, and GR00T's backbone
    expects RGB, so the channels are swapped by default.

    Accepts an already-decoded array too, so a re-exported dump still works.
    """
    arr = np.asarray(raw)

    # Already decoded: (N, H, W, 3) or (N, H, W, 4).
    if arr.ndim == 4:
        return [as_uint8_hwc(frame) for frame in arr]

    if arr.ndim != 1:
        raise ValueError(
            f"expected either (N,) encoded buffers or (N, H, W, C) frames, "
            f"got shape {arr.shape} dtype {arr.dtype}"
        )

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "decoding UniVTAC image streams needs opencv-python (cv2), which is "
            "how the data was encoded. Install it in the converter environment "
            "(requirements-convert.txt)."
        ) from exc

    frames: list[np.ndarray] = []
    for index, buf in enumerate(arr):
        if isinstance(buf, (bytes, bytearray, np.bytes_)):
            payload = np.frombuffer(bytes(buf), dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            payload = buf
        else:
            raise TypeError(f"frame {index}: unsupported buffer type {type(buf)!r}")
        image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"frame {index}: cv2.imdecode failed on a {payload.size}-byte buffer")
        if bgr_to_rgb:
            image = image[..., ::-1]
        frames.append(np.ascontiguousarray(image))
    return frames


def _split_state_action(emb: Any, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Derive ``(joint_state, joint_action)`` the way UniVTAC's loader does.

    The raw dumps carry a single ``embodiment/joint`` array;
    ``HDF5Handler.batch_gather_hdf5`` splits it into ``joint[:-1]`` (state) and
    ``joint[1:]`` (action), making the action an absolute next-joint target and
    costing one frame per episode. Explicit ``joint_state``/``joint_action``
    datasets, if present, are used as-is.
    """
    state_key = _first_present(emb, ["joint_state"])
    action_key = _first_present(emb, ["joint_action"])
    if state_key is not None and action_key is not None:
        state = np.asarray(emb[state_key], dtype=np.float32)
        action = np.asarray(emb[action_key], dtype=np.float32)
        n = min(len(state), len(action))
        return state[:n], action[:n]

    if "joint" in emb:
        joint = np.asarray(emb["joint"], dtype=np.float32)
        if len(joint) < 2:
            raise ValueError(
                f"{filename}: embodiment/joint has {len(joint)} frame(s); at least 2 "
                f"are needed to form a (state, next-state) transition."
            )
        return joint[:-1], joint[1:]

    raise KeyError(
        f"{filename}: embodiment has none of joint_state/joint_action/joint; found "
        f"{sorted(emb.keys())}"
    )


def read_episode(path: Path, spec: ObsSpec) -> dict[str, np.ndarray]:
    """Load one UniVTAC episode HDF5 into per-frame arrays.

    Returns a dict with ``state`` (N, state_dim), ``action`` (N, action_dim) and
    one ``video.<key>`` entry per configured camera.

    Raises:
        KeyError: a required dataset is missing, naming the HDF5 keys present so
            a collection run with the wrong ``observations`` config is obvious.
    """
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as f:
        # -- proprioception ------------------------------------------------
        emb = f["embodiment"]
        joint_state, joint_action = _split_state_action(emb, path.name)
        # N-1 transitions from N frames. Everything below is truncated to this,
        # which reproduces the `data[:-1]` slicing in batch_gather_hdf5.
        n_frames = len(joint_state)

        ee_key = _first_present(emb, ["ee", "ee_pose"])
        if ee_key is not None:
            ee = np.asarray(emb[ee_key], dtype=np.float32)[:n_frames]
        else:
            raise KeyError(
                f"{path.name}: no end-effector pose ('ee') in embodiment; the eef_9d "
                f"state slice cannot be built. Re-collect with "
                f"observations.embodiment: ['joint', 'ee']."
            )

        joint_state = joint_state[:n_frames]
        joint_action = joint_action[:n_frames]

        # -- cameras -------------------------------------------------------
        for gr00t_key, cam in spec.video_keys.items():
            cam_key = f"observation/{cam}/rgb"
            if cam_key not in f:
                raise KeyError(
                    f"{path.name}: missing {cam_key}. Available observation groups: "
                    f"{sorted(f['observation'].keys()) if 'observation' in f else '[]'}"
                )
            frames = decode_image_stream(f[cam_key][:n_frames])
            out[f"video.{gr00t_key}"] = np.stack(
                [resize_nearest(img, spec.image_size) for img in frames]
            )

        if spec.tactile.mode == "video":
            for gr00t_key, sensor in spec.tactile_video_keys.items():
                group = _tactile_group(f, sensor, path.name)
                source = _first_present(group, ["rgb_marker", "rgb"])
                if source is None:
                    raise KeyError(
                        f"{path.name}: tactile sensor {sensor!r} has no rgb_marker/rgb; "
                        f"found {sorted(group.keys())}"
                    )
                frames = decode_image_stream(group[source][:n_frames])
                out[f"video.{gr00t_key}"] = np.stack(
                    [resize_nearest(img, spec.tactile_image_size) for img in frames]
                )

        # -- tactile state -------------------------------------------------
        tactile_frames: np.ndarray | None = None
        if spec.tactile.in_state:
            needed = "depth" if spec.tactile.mode == "depth_pool" else "marker"
            per_sensor = {}
            for sensor in spec.tactile.sensor_names:
                group = _tactile_group(f, sensor, path.name)
                if needed not in group:
                    raise KeyError(
                        f"{path.name}: tactile sensor {sensor!r} has no {needed!r} "
                        f"(found {sorted(group.keys())}). Re-collect with "
                        f"'{needed}' in observations.tactile."
                    )
                per_sensor[sensor] = np.asarray(group[needed][:n_frames])
            tactile_frames = np.stack(
                [
                    encode_tactile_state(
                        {s: {needed: per_sensor[s][i]} for s in per_sensor},
                        spec.tactile,
                    )
                    for i in range(n_frames)
                ]
            )

    # -- assemble the concatenated state / action --------------------------
    gripper = GripperConvention(invert=False)
    states, actions = [], []
    for i in range(n_frames):
        eef_9d = np.concatenate([ee[i][:3], quat_wxyz_to_rot6d(ee[i][3:7])])
        state = [eef_9d, joint_state[i][:7], [gripper.from_univtac(joint_state[i][7])]]
        if tactile_frames is not None:
            state.append(tactile_frames[i])
        states.append(np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in state]))
        actions.append(
            np.concatenate(
                [
                    joint_action[i][:7],
                    [gripper.from_univtac(joint_action[i][7])],
                ]
            ).astype(np.float32)
        )

    out["state"] = np.stack(states).astype(np.float32)
    out["action"] = np.stack(actions).astype(np.float32)

    if out["state"].shape[1] != spec.state_dim:
        raise ValueError(
            f"{path.name}: built a {out['state'].shape[1]}-D state but the variant's "
            f"ObsSpec declares {spec.state_dim}-D"
        )
    return out


def _tactile_group(f: Any, sensor: str, filename: str) -> Any:
    """Resolve a tactile sensor group, tolerating the ``*_gsmini`` alias."""
    for name in (sensor, sensor.replace("_tactile", "_gsmini"), f"{sensor}_tactile"):
        key = f"tactile/{name}"
        if key in f:
            return f[key]
    available = sorted(f["tactile"].keys()) if "tactile" in f else []
    raise KeyError(
        f"{filename}: tactile sensor {sensor!r} not found (available: {available})"
    )


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    """Write ``(N, H, W, 3)`` uint8 frames as H.264 MP4.

    ``imageio`` is preferred; ``cv2`` is the fallback. The demo dataset uses AV1,
    but libx264 is far more likely to be present in a cluster's ffmpeg build and
    is read fine by LeRobot's decoders.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        return
    except ImportError:
        pass

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "writing videos needs either imageio[ffmpeg] or opencv-python"
        ) from exc

    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR
    finally:
        writer.release()


def write_parquet(
    episode: dict[str, np.ndarray],
    path: Path,
    *,
    episode_index: int,
    global_offset: int,
    task_index: int,
    fps: int,
) -> int:
    """Write one episode's tabular data. Returns the number of frames."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(episode["state"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "observation.state": pa.array(
                [row.tolist() for row in episode["state"]], type=pa.list_(pa.float32())
            ),
            "action": pa.array(
                [row.tolist() for row in episode["action"]], type=pa.list_(pa.float32())
            ),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / float(fps)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(global_offset, global_offset + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
            "next.reward": pa.array(np.zeros(n, dtype=np.float32)),
            "next.done": pa.array(
                np.concatenate([np.zeros(n - 1, dtype=bool), [True]]) if n else np.zeros(0, bool)
            ),
        }
    )
    pq.write_table(table, str(path))
    return n


def build_modality_json(spec: ObsSpec, language_key: str) -> dict[str, Any]:
    """Emit ``meta/modality.json`` describing the concatenated array slices.

    The slice boundaries come straight from the variant's :class:`ObsSpec`, so this
    file and the registered ``ModalityConfig`` cannot disagree.
    """
    state: dict[str, dict[str, int]] = {}
    for key, sl in spec.field_slices().items():
        state[key] = {"start": int(sl.start), "end": int(sl.stop)}

    action = {
        "joint_position": {"start": 0, "end": 7},
        "gripper_position": {"start": 7, "end": 8},
    }
    video = {
        key: {"original_key": f"observation.images.{key}"} for key in spec.all_video_keys
    }
    # The annotation key drops the leading "annotation." (see data_preparation.md),
    # and reads its value out of the parquet's task_index column.
    annotation = {language_key.removeprefix("annotation."): {"original_key": "task_index"}}

    return {"state": state, "action": action, "video": video, "annotation": annotation}


def build_info_json(
    spec: ObsSpec,
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
) -> dict[str, Any]:
    """Emit ``meta/info.json`` in the LeRobot v2.1 schema."""
    features: dict[str, Any] = {
        "action": {"dtype": "float32", "shape": [8], "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": [spec.state_dim],
            "names": None,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in spec.all_video_keys:
        h, w = (
            spec.tactile_image_size if key in spec.tactile_video_keys else spec.image_size
        )
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": "univtac_franka_panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(spec.all_video_keys),
        "total_chunks": max(1, (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def compute_stats(states: list[np.ndarray], actions: list[np.ndarray]) -> dict[str, Any]:
    """Compute ``meta/stats.json`` for the state and action arrays.

    GR00T normalises with these, and ``DataConfig.override_pretraining_statistics``
    defaults to True, so a missing stats file degrades a finetune badly. Only
    the low-dimensional arrays are summarised here; image statistics are not
    used by GR00T's processor.
    """

    def describe(stacked: np.ndarray) -> dict[str, list[float]]:
        return {
            "mean": stacked.mean(axis=0).astype(float).tolist(),
            "std": (stacked.std(axis=0) + 1e-8).astype(float).tolist(),
            "min": stacked.min(axis=0).astype(float).tolist(),
            "max": stacked.max(axis=0).astype(float).tolist(),
            "q01": np.quantile(stacked, 0.01, axis=0).astype(float).tolist(),
            "q99": np.quantile(stacked, 0.99, axis=0).astype(float).tolist(),
        }

    return {
        "observation.state": describe(np.concatenate(states, axis=0)),
        "action": describe(np.concatenate(actions, axis=0)),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def find_episodes(root: Path) -> list[Path]:
    """Collect ``*.hdf5`` under ``root``, ordered numerically like UniVTAC does."""

    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return (int(path.stem), "")
        except ValueError:
            return (1 << 30, path.stem)

    return sorted(root.rglob("*.hdf5"), key=sort_key)


def load_instruction(univtac_root: Path, task: str) -> str:
    """First 'seen' instruction from ``UniVTAC/instructions/<task>.json``."""
    path = univtac_root / "instructions" / f"{task}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        seen = data.get("seen") or []
        if seen:
            return str(seen[0])
    return task.replace("_", " ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert UniVTAC HDF5 demos to the GR00T LeRobot v2 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, help="UniVTAC task name")
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="UniVTAC collection directory, e.g. UniVTAC/data/insert_hole/demo",
    )
    parser.add_argument("--out", required=True, help="output dataset root")
    parser.add_argument(
        "--variant",
        default="tactile",
        choices=["baseline_finetuned", "tactile"],
        help="state layout to emit; must match the variant you will finetune",
    )
    parser.add_argument("--tactile-mode", default="depth_pool",
                        choices=["depth_pool", "marker", "video"])
    parser.add_argument("--tactile-sensors", nargs="+",
                        default=["left_tactile", "right_tactile"])
    parser.add_argument("--tactile-pool-grid", nargs=2, type=int, default=[8, 6],
                        metavar=("ROWS", "COLS"))
    parser.add_argument("--episodes", type=int, default=None,
                        help="cap the number of episodes converted")
    parser.add_argument("--fps", type=int, default=20,
                        help="nominal control rate recorded in info.json")
    parser.add_argument("--image-size", nargs=2, type=int, default=[256, 256],
                        metavar=("H", "W"))
    parser.add_argument("--univtac-root", default=".", help="for instructions/<task>.json")
    parser.add_argument("--no-videos", action="store_true",
                        help="write parquet and metadata only (for a quick schema check)")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise SystemExit(f"no such raw directory: {raw_dir}")
    episodes = find_episodes(raw_dir)
    if not episodes:
        raise SystemExit(f"no .hdf5 episodes under {raw_dir}")
    if args.episodes:
        episodes = episodes[: args.episodes]

    grid = (int(args.tactile_pool_grid[0]), int(args.tactile_pool_grid[1]))
    spec_kwargs: dict[str, Any] = {"image_size": (args.image_size[0], args.image_size[1])}
    if args.variant == "tactile":
        spec_kwargs.update(
            mode=args.tactile_mode,
            sensor_names=tuple(args.tactile_sensors),
            pool_grid=grid,
            marker_pool=grid,
        )
    spec = build_spec(args.variant, **spec_kwargs)

    out_root = Path(args.out)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    instruction = load_instruction(Path(args.univtac_root), args.task)

    print(f"[convert] {len(episodes)} episodes -> {out_root}")
    print(f"[convert] variant={args.variant} state_dim={spec.state_dim} keys={list(spec.state_keys)}")

    episode_lines: list[str] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    total_frames = 0

    for index, path in enumerate(episodes):
        chunk = index // CHUNK_SIZE
        data = read_episode(path, spec)
        n = write_parquet(
            data,
            out_root / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet",
            episode_index=index,
            global_offset=total_frames,
            task_index=0,
            fps=args.fps,
        )
        if not args.no_videos:
            for key in spec.all_video_keys:
                write_video(
                    data[f"video.{key}"],
                    out_root
                    / f"videos/chunk-{chunk:03d}/observation.images.{key}"
                    / f"episode_{index:06d}.mp4",
                    fps=args.fps,
                )
        all_states.append(data["state"])
        all_actions.append(data["action"])
        total_frames += n
        episode_lines.append(
            json.dumps({"episode_index": index, "tasks": [instruction], "length": n})
        )
        print(f"[convert] episode {index:04d} ({path.name}): {n} frames")

    meta = out_root / "meta"
    (meta / "episodes.jsonl").write_text("\n".join(episode_lines) + "\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": instruction}) + "\n", encoding="utf-8"
    )
    (meta / "modality.json").write_text(
        json.dumps(build_modality_json(spec, spec.language_key), indent=4), encoding="utf-8"
    )
    (meta / "info.json").write_text(
        json.dumps(
            build_info_json(
                spec,
                total_episodes=len(episodes),
                total_frames=total_frames,
                total_tasks=1,
                fps=args.fps,
            ),
            indent=4,
        ),
        encoding="utf-8",
    )
    (meta / "stats.json").write_text(
        json.dumps(compute_stats(all_states, all_actions), indent=2), encoding="utf-8"
    )

    print(
        f"[convert] done: {len(episodes)} episodes, {total_frames} frames, "
        f"state_dim={spec.state_dim} -> {out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
