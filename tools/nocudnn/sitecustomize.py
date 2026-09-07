"""Opt-in cuDNN kill switch, injected via ``PYTHONPATH``.

Python imports ``sitecustomize`` automatically at interpreter startup, so
putting this directory on ``PYTHONPATH`` disables cuDNN in *every* child
process -- including the dataloader workers and per-rank processes that
``gr00t/experiment/launch_finetune.py`` spawns, which a command-line flag on the
parent cannot reach.

Needed because this cluster's driver (550.x, CUDA 12.4) is older than GR00T's
torch build (2.9.0+cu128), so cuDNN 9.13 cannot initialise: its internal
``cudaGetDeviceCount`` fails and it reports ``CUDNN_STATUS_NOT_INITIALIZED``.
Only convolutions are affected -- FlashAttention-2 and the DiT's SDPA path do
not use cuDNN -- so native conv kernels are a workable, if slower, substitute.

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
