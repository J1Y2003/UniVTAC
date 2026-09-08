#!/usr/bin/env python3
"""Can we train GR00T on someone else's LeRobot conversion?

A GR00T dataset is LeRobot v2 **plus** ``meta/modality.json``, and the state
and action layouts have to agree with the variant's ``ObsSpec`` exactly. A
generic LeRobot export usually has the v2 half and not the rest, and several of
the ways it can be wrong are silent -- channel-swapped video and a mis-sliced
state vector both train happily and just produce a worse model.

So check rather than assume:

    python scripts/check_external_dataset.py /path/to/UniVTAC_lerobot \\
        --tasks insert_hole insert_tube pull_out_key

Read-only. Needs numpy + pyarrow (the `univtac-groot` env); `ffprobe` and
`opencv-python` are used if present and skipped if not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from univtac_groot.variants import build_spec, video_keys_for_task  # noqa: E402

OK, BAD, WARN, INFO = "  ok  ", " FAIL ", " warn ", "  --  "


class Result:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []
        self.fatal = 0

    def add(self, status: str, msg: str) -> None:
        self.rows.append((status, msg))
        if status == BAD:
            self.fatal += 1

    def show(self) -> None:
        for status, msg in self.rows:
            print(f"   [{status}] {msg}")


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return exc


def check_task(root: Path, task: str, variant: str, dump_frame: Path | None) -> Result:
    r = Result()
    spec = build_spec(variant)
    want_cams = sorted(video_keys_for_task(task))

    # ---- meta files -------------------------------------------------- #
    meta = root / "meta"
    for name, required in (("info.json", True), ("modality.json", True),
                           ("stats.json", True), ("episodes.jsonl", False),
                           ("tasks.jsonl", False)):
        p = meta / name
        if p.is_file():
            r.add(OK, f"meta/{name}")
        elif required:
            r.add(BAD, f"meta/{name} MISSING -- GR00T needs it "
                       f"({'the GR00T-specific half of the format' if name == 'modality.json' else 'dataset statistics'})")
        else:
            r.add(WARN, f"meta/{name} missing (LeRobot writes it; may be fine)")

    info = load_json(meta / "info.json")
    if isinstance(info, dict):
        r.add(INFO, f"codebase_version={info.get('codebase_version')!r} "
                    f"fps={info.get('fps')} episodes={info.get('total_episodes')} "
                    f"chunks_size={info.get('chunks_size')}")
        feats = info.get("features", {})
        for key in ("observation.state", "action"):
            f = feats.get(key)
            if f is None:
                r.add(WARN, f"info.json has no feature {key!r} (names may differ)")
                continue
            width = (f.get("shape") or [None])[0]
            expect = spec.state_dim if key.endswith("state") else 8
            status = OK if width == expect else BAD
            r.add(status, f"{key} width {width} (variant {variant!r} wants {expect})")
        vid = sorted(k for k in feats if k.startswith("observation.images"))
        r.add(INFO, f"video features: {[k.split('.')[-1] for k in vid] or 'none'}")
    else:
        r.add(BAD, f"meta/info.json unreadable: {info}")

    # ---- modality.json: the GR00T-specific part ---------------------- #
    mod = load_json(meta / "modality.json")
    if isinstance(mod, dict):
        got_state = sorted(mod.get("state", {}))
        got_action = sorted(mod.get("action", {}))
        got_video = sorted(mod.get("video", {}))
        want_state = sorted(spec.state_keys)
        r.add(OK if got_state == want_state else BAD,
              f"modality.json state keys {got_state}\n"
              f"            variant wants {want_state}")
        r.add(INFO, f"modality.json action keys {got_action}")
        missing = [c for c in want_cams if c not in got_video]
        r.add(OK if not missing else BAD,
              f"modality.json video keys {got_video}; this task needs {want_cams}"
              + (f" -- MISSING {missing}" if missing else ""))
        extra = [c for c in got_video if c not in want_cams]
        if extra:
            r.add(INFO, f"extra cameras present ({extra}); harmless -- the modality "
                        f"config selects only {want_cams}")
    elif (meta / "modality.json").is_file():
        r.add(BAD, f"meta/modality.json unreadable: {mod}")

    # ---- one parquet ------------------------------------------------- #
    shards = sorted((root / "data").rglob("*.parquet"))
    r.add(OK if shards else BAD, f"{len(shards)} parquet shard(s)")
    if shards:
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(shards[0])
            r.add(INFO, f"{shards[0].name}: {t.num_rows} rows, columns {t.column_names}")
            if t.num_rows in (310, 311):
                r.add(INFO if t.num_rows == 310 else WARN,
                      f"{t.num_rows} rows/episode -- ours is 310 (311 frames minus the "
                      f"joint[:-1]/joint[1:] shift). 311 means a different action "
                      f"derivation, so actions may be off by one step")
            for col, expect in (("observation.state", spec.state_dim), ("action", 8)):
                if col in t.column_names:
                    w = len(t.column(col)[0].as_py())
                    r.add(OK if w == expect else BAD, f"{col} is {w}-D (want {expect})")
        except ImportError:
            r.add(WARN, "pyarrow not importable; skipped the parquet check")
        except Exception as exc:
            r.add(BAD, f"could not read {shards[0].name}: {type(exc).__name__}: {exc}")

    # ---- video: codec, and the silent one (channel order) ------------ #
    vids = sorted((root / "videos").rglob("*.mp4"))
    r.add(OK if vids else BAD, f"{len(vids)} mp4 file(s)")
    if vids:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,width,height,nb_frames",
                 "-of", "default=nw=1", str(vids[0])],
                capture_output=True, text=True, timeout=60).stdout.strip()
            codec = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
            name = codec.get("codec_name", "?")
            r.add(OK if name in ("h264", "hevc") else WARN,
                  f"{vids[0].name}: codec={name} {codec.get('width')}x{codec.get('height')} "
                  f"frames={codec.get('nb_frames')}"
                  + ("" if name in ("h264", "hevc")
                     else " -- GR00T's torchcodec guarantees H.264; AV1 is not guaranteed"))
        except FileNotFoundError:
            r.add(WARN, "ffprobe not on PATH; skipped the codec check")
        except Exception as exc:
            r.add(WARN, f"ffprobe failed: {type(exc).__name__}")

        try:
            import cv2
            cap = cv2.VideoCapture(str(vids[0]))
            got, frame = cap.read()
            cap.release()
            if got:
                # cv2 gives BGR. UniVTAC stores BGR JPEGs and GR00T's backbone
                # wants RGB, so a conversion that forgot to swap is a real risk
                # and nothing will ever error. Natural scenes here (wood, tan
                # table, metal) normally have more red than blue.
                b, g, rr = (float(frame[..., i].mean()) for i in range(3))
                r.add(INFO, f"first frame channel means, as stored: "
                            f"ch0={b:.1f} ch1={g:.1f} ch2={rr:.1f}")
                r.add(WARN if b > rr + 8 else INFO,
                      "channel 0 is much brighter than channel 2 -- consistent with "
                      "BGR stored where GR00T expects RGB. CONFIRM BY EYE."
                      if b > rr + 8 else
                      "channel balance looks RGB-ish, but this is a weak signal")
                if dump_frame:
                    cv2.imwrite(str(dump_frame), frame)
                    r.add(INFO, f"wrote {dump_frame} -- open it. Wood/skin/table "
                                f"looking blue means the channels are swapped")
        except ImportError:
            r.add(WARN, "opencv-python not importable; skipped the colour check")
        except Exception as exc:
            r.add(WARN, f"colour check failed: {type(exc).__name__}")

    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory containing one subdirectory per task")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--variant", default="baseline_finetuned")
    ap.add_argument("--dump-frames", type=Path, default=None,
                    help="directory to write one decoded frame per task, for eyeballing")
    args = ap.parse_args(argv)

    if args.dump_frames:
        args.dump_frames.mkdir(parents=True, exist_ok=True)

    print(f"\nChecking {args.root} against variant {args.variant!r}")
    print("=" * 72)
    fatal = 0
    for task in args.tasks:
        root = args.root / task
        print(f"\n{task}  ({root})")
        if not root.is_dir():
            print(f"   [{BAD}] not a directory")
            fatal += 1
            continue
        dump = (args.dump_frames / f"{task}.png") if args.dump_frames else None
        res = check_task(root, task, args.variant, dump)
        res.show()
        fatal += res.fatal

    print("\n" + "=" * 72)
    if fatal:
        print(f"{fatal} blocking problem(s). This conversion is NOT drop-in for GR00T.")
        print("Either fix the gaps or reconvert with scripts/convert_univtac_to_lerobot.py")
        print("(31 MB per task, minutes, and it is already known to work end to end).")
    else:
        print("No blocking problems found. Wire it in WITHOUT copying:")
        print("  for t in <tasks>; do")
        print("    ln -s <root>/$t $DATA_ROOT/univtac-$t-baseline_finetuned")
        print("  done")
        print("Treat the source tree as read-only -- it is not ours.")
        print("Still confirm the colour order by eye; nothing above proves it.")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
