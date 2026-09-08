"""Opt-in cuDNN kill switch, injected via ``PYTHONPATH``.

Python imports ``sitecustomize`` automatically at interpreter startup, so
putting this directory on ``PYTHONPATH`` disables cuDNN in *every* child
process -- including the dataloader workers and per-rank processes that
``gr00t/experiment/launch_finetune.py`` spawns, which a command-line flag on the
parent cannot reach.

**This is a last resort, not a default.** It was on by default for a while,
on the theory that this cluster's driver (550.x, CUDA 12.4) was too old for
GR00T's torch build (2.9.0+cu128). That was wrong: the GR00T venv had cuDNN
**9.13** on disk while torch 2.9.0+cu128 pins ``nvidia-cudnn-cu12==9.10.2.21``
(= 91002), and the mismatched library was what reported
``CUDNN_STATUS_NOT_INITIALIZED``. Reinstalling the pinned version fixed it with
no driver change. See docs/SETUP.md, "cuDNN", and
``scripts/preflight.py --deep``, which now fails if the pin drifts again.

Turning cuDNN off is **not** the cheap trade it looks like. It is true that
only convolutions are affected -- FlashAttention-2 and the DiT's SDPA path do
not use cuDNN -- but GR00T has exactly one conv that matters and it is on the
hot path: Qwen3-VL's vision patch embed. Qwen3VL reshapes every visual patch
into its own batch element, so that ``Conv3d(3, 1024, (2,16,16),
stride=(2,16,16))`` runs over ~32,768 batch elements per step (64 samples x 2
cameras x 256 patches at 256px). Without cuDNN, ATen walks that batch with a
per-element im2col loop driven from the main Python thread. Measured on an
A100 (sm_80): **46.2 s vs 0.53 s** for a vision-tower forward+backward at 32
images -- 86x, and 98.8% of the tower -- which is what turned a step into
170 s and a 10,000-step finetune into a 470-hour projection.

Gated behind an environment variable so merely having this on ``PYTHONPATH``
costs nothing and never imports torch unless asked:

    UNIVTAC_DISABLE_CUDNN=1 PYTHONPATH=<repo>/tools/nocudnn:$PYTHONPATH python ...

Remove once the driver is updated (r570+) or ``cuda-compat-12-8`` is installed.
"""

import os

if os.environ.get("UNIVTAC_DISABLE_CUDNN") == "1":
    try:
        import torch

        torch.backends.cudnn.enabled = False
        if os.environ.get("UNIVTAC_DISABLE_CUDNN_VERBOSE") == "1":
            print("[sitecustomize] cuDNN disabled (UNIVTAC_DISABLE_CUDNN=1)", flush=True)
    except Exception:  # noqa: BLE001 - startup hooks must never break the interpreter
        pass
