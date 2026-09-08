"""Serve a GR00T N1.7 policy over ZeroMQ for the UniVTAC evaluator.

Runs in the **GR00T** environment (the one with ``gr00t``, transformers and
flash-attn installed), never inside Isaac Sim. It is a thin shell over the
upstream pieces:

* ``Gr00tPolicy`` loads the checkpoint and the ``AutoProcessor``;
* ``Gr00tSimPolicyWrapper`` accepts the flat ``video.*``/``state.*`` observation
  convention this repo's adapters emit and returns ``action.*``;
* ``PolicyServer`` binds the socket and dispatches the ``ping``/``reset``/
  ``get_action``/``get_modality_config``/``kill`` endpoints.

Compared with ``python -m gr00t.eval.run_gr00t_server`` this adds one thing:
``--modality-config-path`` is honoured for a *model* policy, not only for the
replay policy. That matters when a finetuned checkpoint was trained before its
modality config was registered, or when the config lives in this repo
(``configs/modality/univtac_tactile_config.py``) rather than in the checkpoint.
If the checkpoint already carries the config, the flag is unnecessary.

Usage (see ``slurm/eval_ablation.sbatch`` for the batch form)::

    python -m univtac_groot.server.run_server \
        --model-path nvidia/GR00T-N1.7-3B \
        --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
        --port 5555

    python -m univtac_groot.server.run_server \
        --model-path /ckpt/univtac-tactile/checkpoint-20000 \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path configs/modality/univtac_tactile_config.py \
        --port 5556
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


DEFAULT_PORT = 5555


def load_modality_config(path: str) -> None:
    """Import a ``*_config.py`` so its ``register_modality_config`` call runs.

    GR00T's convention (``examples/SO100/so100_config.py``) is that the module
    registers itself into ``gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS``
    as an import side effect. Loaded by file path so the config need not sit on
    ``PYTHONPATH``.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"modality config not found: {config_path}")
    if config_path.suffix != ".py":
        raise ValueError(
            f"expected a .py modality config that calls register_modality_config(); "
            f"got {config_path.name}. A dataset's meta/modality.json is a different "
            f"schema and is not accepted here."
        )

    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load modality config {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[config_path.stem] = module
    spec.loader.exec_module(module)
    print(f"[server] registered modality config from {config_path}")


def describe_acceleration(policy: object) -> str:
    """Report dtype and attention kernel actually in use, for the startup log.

    Worth printing because the fallback is silent: ``GR00T_N1d7Config`` defaults
    to ``use_flash_attention=True``, but ``qwen3_backbone.py`` catches the
    ``ImportError`` and downgrades to ``attn_implementation="sdpa"`` with only a
    warning. A run that quietly lost FlashAttention-2 is slower and uses more
    memory, and nothing else in the pipeline would tell you.

    Best-effort and defensive: every attribute is probed, since the layout of
    the config differs between checkpoints.
    """
    bits: list[str] = []

    try:
        import torch

        model = getattr(policy, "model", None)
        dtypes = {str(p.dtype) for p in model.parameters()} if model is not None else set()
        if dtypes:
            bits.append("dtype=" + ",".join(sorted(d.replace("torch.", "") for d in dtypes)))
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            bits.append(f"vram_used={(total - free) / 2**30:.1f}/{total / 2**30:.1f}GiB")
    except Exception:  # noqa: BLE001 - diagnostics must never break startup
        pass

    try:
        import torch

        if not torch.backends.cudnn.enabled:
            bits.append("cudnn=DISABLED")
        else:
            bits.append(f"cudnn={torch.backends.cudnn.version()}")
    except Exception:  # noqa: BLE001
        pass

    try:
        import flash_attn  # noqa: F401

        bits.append("flash_attn=installed")
    except ImportError:
        bits.append("flash_attn=MISSING (backbone falls back to sdpa)")
    except Exception:  # noqa: BLE001
        bits.append("flash_attn=?")

    # Walk a few likely places for the resolved attention implementation.
    try:
        model = getattr(policy, "model", None)
        seen: set[str] = set()
        for obj in (model, getattr(model, "config", None)):
            if obj is None:
                continue
            for attr in ("_attn_implementation", "attn_implementation", "use_flash_attention"):
                value = getattr(obj, attr, None)
                if value is not None:
                    seen.add(f"{attr}={value}")
        for module in (model.modules() if model is not None else []):
            cfg = getattr(module, "config", None)
            impl = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
            if impl:
                seen.add(f"backbone_attn={impl}")
                break
        bits.extend(sorted(seen))
    except Exception:  # noqa: BLE001
        pass

    return "  ".join(bits) if bits else "unavailable"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve GR00T N1.7 over ZeroMQ for UniVTAC evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF repo id or local checkpoint directory, e.g. nvidia/GR00T-N1.7-3B",
    )
    parser.add_argument(
        "--embodiment-tag",
        required=True,
        help=(
            "EmbodimentTag name or value (case-insensitive). Base-checkpoint tags: "
            "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT, XDOF, REAL_G1, ... "
            "NEW_EMBODIMENT requires a finetuned checkpoint."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port")
    parser.add_argument("--device", default="cuda", help="torch device for the model")
    parser.add_argument(
        "--modality-config-path",
        default=None,
        help=(
            "optional .py modality config to register before loading the policy "
            "(needed only when the checkpoint does not already carry it)"
        ),
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="shared secret; the client must pass the same value",
    )
    parser.add_argument(
        "--no-sim-wrapper",
        action="store_true",
        help=(
            "serve the nested Gr00tPolicy observation format instead of the flat "
            "video.*/state.* sim format this repo's adapters emit"
        ),
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="disable GR00T's observation/action validation (not recommended)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Imported here so --help works without the (heavy) GR00T stack loaded.
    from gr00t.data.embodiment_tags import (
        FINETUNE_ONLY_TAGS,
        POSTTRAIN_TAGS,
        EmbodimentTag,
    )
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    if args.modality_config_path:
        load_modality_config(args.modality_config_path)

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    if tag in FINETUNE_ONLY_TAGS or tag in POSTTRAIN_TAGS:
        print(
            f"[server] note: {tag.name} ships in no released checkpoint; "
            f"{args.model_path!r} must be a checkpoint finetuned for it, or "
            f"Gr00tPolicy will refuse the tag."
        )

    print(f"[server] loading {args.model_path} as {tag.name} on {args.device}")
    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=args.model_path,
        device=args.device,
        strict=not args.no_strict,
    )

    modality = policy.get_modality_config()
    action_horizon = len(modality["action"].delta_indices)
    print(
        f"[server] ready: state_keys={modality['state'].modality_keys} "
        f"video_keys={modality['video'].modality_keys} "
        f"action_horizon={action_horizon} "
        f"video_deltas={list(modality['video'].delta_indices)}"
    )
    print(f"[server] acceleration: {describe_acceleration(policy)}")

    if not args.no_sim_wrapper:
        policy = Gr00tSimPolicyWrapper(policy, strict=not args.no_strict)

    with PolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\n[server] shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
