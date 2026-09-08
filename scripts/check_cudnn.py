#!/usr/bin/env python3
"""Fail if the cuDNN GR00T's venv loads is not the one torch pins.

Run under GR00T's interpreter, inside the job, before any weights load:

    $GROOT_PYTHON scripts/check_cudnn.py

Exit 0 if the loaded cuDNN matches torch's pin and a real convolution runs on
the GPU. Exit 1 on a definite mismatch, printing the fix. Exit 0 with a warning
whenever the answer cannot be determined -- an unreadable pin or a missing GPU
is not evidence of a problem, and this guard must never block a job it cannot
actually indict.

WHY THIS RUNS AUTOMATICALLY. A mismatched cuDNN reports
CUDNN_STATUS_NOT_INITIALIZED, which reads exactly like a too-old driver. The
tempting workaround is to stop using cuDNN, and that costs ~86x on GR00T's
vision tower: Qwen3-VL's patch embed reshapes every visual patch into its own
batch element, so its Conv3d runs over ~32,768 batch elements per step and
without cuDNN ATen walks them with a per-element im2col loop from the main
Python thread. A training step goes from 1.89 s to 170 s, and nothing in the
stack complains -- it is merely slow. There is deliberately no flag in this
repo to disable cuDNN, so this check plus the fix below is the whole story.

Nothing here is cheap to check indirectly: `uv pip list` reports the pinned
version even when the files on disk are a different release, and
`torch.backends.cudnn.version()` reports the header it was built against, not
the library that got loaded. Only asking the loaded library its own version
distinguishes them.
"""

from __future__ import annotations

import ctypes
import json
import sys

FIX = """\
FIX -- reinstall the pinned cuDNN into GR00T's venv:

    cd $GROOT_ROOT
    env -u CONDA_PREFIX -u VIRTUAL_ENV uv cache clean nvidia-cudnn-cu12
    env -u CONDA_PREFIX -u VIRTUAL_ENV uv pip install --python .venv/bin/python \\
        --reinstall nvidia-cudnn-cu12=={wheel}

The `uv cache clean` is not optional: uv hardlinks these .so files out of its
content-addressed cache (`ls -la` shows a link count above 1), so a plain
--reinstall can re-link the very files you are trying to replace.

Then re-run this check. Full detail in docs/SETUP.md, section "cuDNN".
If cuDNN still will not initialise, docs/SETUP.md lists the fallbacks in order
of preference. Do not try to run without cuDNN -- see the note in this file.
"""


def encode(wheel: str) -> int | None:
    """``"9.10.2.21"`` -> ``91002``, cuDNN's MAJOR*10000 + MINOR*100 + PATCH."""
    parts = wheel.split(".")
    if len(parts) < 3:
        return None
    try:
        major, minor, patch = (int(p) for p in parts[:3])
    except ValueError:
        return None
    return major * 10000 + minor * 100 + patch


def probe() -> dict:
    out: dict = {}
    try:
        import importlib.metadata as md
        out["wheel"] = md.version("nvidia-cudnn-cu12")
    except Exception:
        out["wheel"] = None
    try:
        # Importing torch is what puts the venv's nvidia/*/lib directories on
        # the loader path, so the CDLL below resolves the same library torch
        # would use rather than a system one.
        import torch
        out["torch"] = torch.__version__
        out["header"] = torch.backends.cudnn.version()
    except Exception as exc:
        out["fatal"] = f"{type(exc).__name__}: {exc}"[:300]
        return out
    try:
        lib = ctypes.CDLL("libcudnn.so.9")
        lib.cudnnGetVersion.restype = ctypes.c_size_t
        out["runtime"] = int(lib.cudnnGetVersion())
    except Exception as exc:
        out["runtime"] = None
        out["dlopen"] = str(exc)[:300]
    # A version match is necessary but not sufficient: run a real convolution
    # through the cuDNN path, which is what actually failed on this cluster.
    if torch.cuda.is_available():
        try:
            import torch.nn as nn
            conv = nn.Conv2d(3, 4, 3).cuda()
            with torch.backends.cudnn.flags(enabled=True):
                conv(torch.randn(2, 3, 16, 16, device="cuda")).sum().item()
            out["conv"] = "ok"
        except Exception as exc:
            out["conv"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    else:
        out["conv"] = "no-gpu"
    return out


def main() -> int:
    as_json = "--json" in sys.argv[1:]
    info = probe()
    if as_json:
        print(json.dumps(info))
        # --json is for scripts/preflight.py, which does its own reporting.
        return 0

    if "fatal" in info:
        print(f"cudnn-check: FAIL -- cannot import torch: {info['fatal']}", file=sys.stderr)
        print("  This interpreter is not a working GR00T environment.", file=sys.stderr)
        return 1

    wheel = info.get("wheel")
    runtime = info.get("runtime")
    expected = encode(wheel) if wheel else None

    if runtime is None:
        print(f"cudnn-check: FAIL -- cannot dlopen libcudnn.so.9: "
              f"{info.get('dlopen', '?')}", file=sys.stderr)
        print(FIX.format(wheel=wheel or "9.10.2.21"), file=sys.stderr)
        return 1

    if expected is None:
        print(f"cudnn-check: WARN -- loaded cuDNN {runtime}, but the pinned "
              f"version is unreadable ({wheel!r}); continuing.")
        return 0

    if runtime != expected:
        print(f"cudnn-check: FAIL -- loaded cuDNN {runtime} but torch "
              f"{info.get('torch')} pins {wheel} (= {expected}).", file=sys.stderr)
        print("  Training would still run, and would be ~86x slower per step "
              "with no error.", file=sys.stderr)
        print(FIX.format(wheel=wheel), file=sys.stderr)
        return 1

    conv = info.get("conv")
    if conv == "no-gpu":
        print(f"cudnn-check: OK on version ({runtime}); no GPU visible, so the "
              f"convolution was not exercised.")
        return 0
    if conv != "ok":
        print(f"cudnn-check: FAIL -- cuDNN {runtime} matches the pin but a real "
              f"convolution failed: {conv}", file=sys.stderr)
        print(FIX.format(wheel=wheel), file=sys.stderr)
        return 1

    print(f"cudnn-check: OK -- cuDNN {runtime} matches torch {info.get('torch')} "
          f"pin {wheel}, and a GPU convolution ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
