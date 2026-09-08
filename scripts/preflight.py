"""Check the setup and print the next command to run.

Safe on a login node: path and environment checks only, plus optional
subprocess import probes. No GPU, no Isaac Sim, no model loading.

    python scripts/preflight.py           # fast checks
    python scripts/preflight.py --deep    # also probe the two interpreters

Each check prints PASS / FAIL / SKIP and, on the first failure, the exact
command to fix it. Run this whenever you are unsure where you are in the
sequence; the full runbook is docs/RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"

_SYMBOL = {PASS: "[ok]  ", FAIL: "[FAIL]", SKIP: "[--]  ", WARN: "[warn]"}


class Report:
    """Accumulates check results and prints the first actionable failure."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.first_fix: str | None = None

    def add(self, status: str, label: str, detail: str = "", fix: str = "") -> None:
        self.rows.append((status, label, detail))
        if status == FAIL and self.first_fix is None:
            self.first_fix = fix or ""

    def render(self) -> int:
        width = max(len(label) for _s, label, _d in self.rows)
        print()
        for status, label, detail in self.rows:
            line = f"  {_SYMBOL[status]} {label:<{width}}"
            if detail:
                line += f"  {detail}"
            print(line)
        print()

        failures = sum(1 for s, _, _ in self.rows if s == FAIL)
        if failures:
            print(f"{failures} check(s) failed. Next step:\n")
            print(self.first_fix or "  see docs/RUNBOOK.md")
            print()
            return 1
        print("All checks passed. Next step:\n")
        print(self.first_fix or "  see docs/RUNBOOK.md")
        print()
        return 0


def env_path(name: str) -> Path | None:
    """Read an environment variable as an expanded path, if set and non-empty."""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def probe_import(python: Path, module: str, timeout: int = 300) -> tuple[bool, str]:
    """Run ``python -c "import <module>"`` in another interpreter."""
    try:
        result = subprocess.run(
            [str(python), "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, "interpreter not found"
    except subprocess.TimeoutExpired:
        return False, f"import timed out after {timeout}s"
    if result.returncode == 0:
        return True, "ok"
    tail = (result.stderr or "").strip().splitlines()
    return False, tail[-1][:160] if tail else f"exit {result.returncode}"


def check_repo(report: Report) -> None:
    """This repo is intact and importable."""
    if (_REPO_ROOT / "univtac_groot" / "spec.py").is_file():
        report.add(PASS, "this repo", str(_REPO_ROOT))
    else:
        report.add(
            FAIL,
            "this repo",
            f"univtac_groot/ missing under {_REPO_ROOT}",
            "  This script must live in <repo>/scripts/. Re-clone or fix the layout.",
        )
        return

    try:
        import numpy  # noqa: F401

        from univtac_groot.variants import baseline_spec

        spec = baseline_spec()
        report.add(PASS, "package imports", f"baseline state_dim={spec.state_dim}")
    except Exception as exc:  # noqa: BLE001
        report.add(
            FAIL,
            "package imports",
            f"{type(exc).__name__}: {exc}",
            "  conda create -n univtac-groot python=3.12 -y\n"
            "  conda activate univtac-groot\n"
            "  pip install -r requirements-dev.txt   # NOT into (base)",
        )


def check_univtac(report: Report) -> Path | None:
    """UNIVTAC_ROOT points at a real UniVTAC checkout, distinct from this repo."""
    root = env_path("UNIVTAC_ROOT")
    if root is None:
        report.add(
            FAIL,
            "UNIVTAC_ROOT",
            "not set",
            "  export UNIVTAC_ROOT=/path/to/UniVTAC   # the checkout containing envs/",
        )
        return None
    if not root.is_dir():
        report.add(
            FAIL, "UNIVTAC_ROOT", f"no such directory: {root}",
            "  export UNIVTAC_ROOT=/path/to/UniVTAC",
        )
        return None

    # The giveaway that this is the simulator checkout and not this repo.
    markers = ["envs/_base_task.py", "task_config", "instructions"]
    missing = [m for m in markers if not (root / m).exists()]
    if missing:
        hint = (
            "  UNIVTAC_ROOT must be the simulator checkout (github.com/univtac/UniVTAC),\n"
            "  not this repo. Clone it if you have not:\n"
            "    git clone https://github.com/univtac/UniVTAC.git\n"
            "    cd UniVTAC && bash scripts/install.sh   # on a COMPUTE node"
        )
        report.add(FAIL, "UniVTAC checkout", f"missing {missing}", hint)
        return None

    if root.resolve() == _REPO_ROOT.resolve():
        report.add(
            FAIL,
            "UniVTAC checkout",
            "UNIVTAC_ROOT is this repo",
            "  They are two different repositories; point UNIVTAC_ROOT elsewhere.",
        )
        return None

    report.add(PASS, "UniVTAC checkout", str(root))

    tasks = sorted(p.stem for p in (root / "envs").glob("*.py") if not p.stem.startswith("_"))
    report.add(PASS, "UniVTAC tasks", f"{len(tasks)} found")

    if (root / "assets" / "embodiments").is_dir():
        report.add(PASS, "UniVTAC assets", "assets/embodiments present")
    else:
        report.add(
            WARN, "UniVTAC assets", "assets/ looks incomplete",
            "",
        )
    return root


def check_interpreters(report: Report, deep: bool) -> None:
    """UNIVTAC_PYTHON and GROOT_PYTHON exist and can import their stacks."""
    for var, module, hint in (
        (
            "UNIVTAC_PYTHON",
            "isaaclab",
            "  export UNIVTAC_PYTHON=$(conda run -n UniVTAC which python)",
        ),
        (
            "GROOT_PYTHON",
            "gr00t",
            "  export GROOT_PYTHON=/path/to/Isaac-GR00T/.venv/bin/python",
        ),
    ):
        python = env_path(var)
        if python is None:
            report.add(FAIL, var, "not set", hint)
            continue
        if not python.is_file():
            report.add(FAIL, var, f"no such interpreter: {python}", hint)
            continue

        version = subprocess.run(
            [str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True,
        ).stdout.strip()
        report.add(PASS, var, f"{python}  (Python {version or '?'})")

        if not deep:
            report.add(SKIP, f"  import {module}", "use --deep to probe")
            continue
        ok, detail = probe_import(python, module)
        if ok:
            report.add(PASS, f"  import {module}", "ok")
        else:
            report.add(
                FAIL,
                f"  import {module}",
                detail,
                f"  {module} is not installed in {python}. See docs/SETUP.md.",
            )


_CUDNN_PROBE = r"""
import ctypes, json
out = {}
try:
    import importlib.metadata as md
    out["wheel"] = md.version("nvidia-cudnn-cu12")
except Exception:
    out["wheel"] = None
try:
    import torch
    out["torch"] = torch.__version__
    out["header"] = torch.backends.cudnn.version()
except Exception as exc:
    out["fatal"] = f"{type(exc).__name__}: {exc}"[:200]
    print(json.dumps(out)); raise SystemExit(0)
try:
    _lib = ctypes.CDLL("libcudnn.so.9")
    _lib.cudnnGetVersion.restype = ctypes.c_size_t
    out["runtime"] = int(_lib.cudnnGetVersion())
except Exception as exc:
    out["runtime"] = None
    out["dlopen"] = str(exc)[:200]
if torch.cuda.is_available():
    try:
        import torch.nn as nn
        torch.backends.cudnn.enabled = True
        _c = nn.Conv2d(3, 4, 3).cuda()
        with torch.backends.cudnn.flags(enabled=True):
            _c(torch.randn(2, 3, 16, 16, device="cuda")).sum().item()
        out["conv"] = "ok"
    except Exception as exc:
        out["conv"] = f"{type(exc).__name__}: {str(exc)[:160]}"
else:
    out["conv"] = "no-gpu"
print(json.dumps(out))
"""


def _encode_cudnn_version(wheel: str) -> int | None:
    """``"9.10.2.21"`` -> ``91002``, cuDNN's own MAJOR*10000+MINOR*100+PATCH."""
    parts = wheel.split(".")
    if len(parts) < 3:
        return None
    try:
        major, minor, patch = (int(p) for p in parts[:3])
    except ValueError:
        return None
    return major * 10000 + minor * 100 + patch


def check_cudnn(report: Report, deep: bool) -> None:
    """The cuDNN that GR00T's venv actually loads is the one torch pins.

    This exists because getting it wrong cost days. The venv had cuDNN 9.13 on
    disk while ``torch==2.9.0+cu128`` pins ``nvidia-cudnn-cu12==9.10.2.21``, and
    the mismatched library reported ``CUDNN_STATUS_NOT_INITIALIZED``. That was
    misread as "the cluster's driver is too old", cuDNN was disabled as a
    workaround, and disabling it costs ~86x on GR00T's vision tower -- 170 s per
    training step instead of a few. Nothing in the stack complains: pip metadata
    reported the pinned version while the files on disk were a different
    release, so only asking the library its own version catches it.
    """
    python = env_path("GROOT_PYTHON")
    if python is None or not python.is_file():
        report.add(SKIP, "cuDNN", "GROOT_PYTHON not usable")
        return
    if not deep:
        report.add(SKIP, "cuDNN", "use --deep to probe (imports torch)")
        return

    try:
        result = subprocess.run(
            [str(python), "-c", _CUDNN_PROBE],
            capture_output=True, text=True, timeout=300,
        )
        info = json.loads((result.stdout or "").strip().splitlines()[-1])
    except Exception as exc:
        report.add(FAIL, "cuDNN", f"probe failed: {type(exc).__name__}",
                   f"  Run by hand to see why:\n    {python} -c '<see _CUDNN_PROBE>'")
        return

    if "fatal" in info:
        report.add(FAIL, "cuDNN", info["fatal"],
                   f"  {python} cannot import torch. See docs/SETUP.md.")
        return

    wheel = info.get("wheel")
    runtime = info.get("runtime")
    expected = _encode_cudnn_version(wheel) if wheel else None
    fix = (
        "  The loaded cuDNN is not the one torch pins. Reinstall it:\n"
        f"    cd $GROOT_ROOT\n"
        "    env -u CONDA_PREFIX -u VIRTUAL_ENV uv cache clean nvidia-cudnn-cu12\n"
        "    env -u CONDA_PREFIX -u VIRTUAL_ENV uv pip install --python .venv/bin/python \\\n"
        f"        --reinstall nvidia-cudnn-cu12=={wheel}\n"
        "  The cache clean matters: uv hardlinks these .so files out of its\n"
        "  content cache, so a plain --reinstall can re-link the same bad files.\n"
        "  See docs/SETUP.md, \"cuDNN\"."
    )

    if runtime is None:
        report.add(FAIL, "cuDNN", f"cannot dlopen libcudnn.so.9: {info.get('dlopen', '?')}", fix)
        return
    if expected is None:
        report.add(WARN, "cuDNN", f"runtime {runtime}, pinned version unreadable ({wheel})")
    elif runtime != expected:
        report.add(
            FAIL, "cuDNN",
            f"loaded {runtime} but torch pins {wheel} (= {expected}) -- MISMATCH",
            fix,
        )
        return
    else:
        report.add(PASS, "cuDNN", f"{runtime} matches the pin ({wheel})")

    conv = info.get("conv")
    if conv == "ok":
        report.add(PASS, "  cuDNN conv", "runs on this GPU")
    elif conv == "no-gpu":
        report.add(SKIP, "  cuDNN conv", "no GPU here; re-run on a compute node")
    else:
        report.add(
            FAIL, "  cuDNN conv", str(conv),
            "  cuDNN loads but cannot run a convolution. Do NOT paper over this\n"
            "  with DISABLE_CUDNN=1 -- that costs ~86x on the vision tower (170 s\n"
            "  per training step). Try an older pin-compatible cuDNN, or the\n"
            "  cluster's container path (srun --container). See docs/SETUP.md.",
        )


def check_client_deps(report: Report, deep: bool) -> None:
    """The UniVTAC-side interpreter has this repo's four client dependencies."""
    python = env_path("UNIVTAC_PYTHON")
    if python is None or not python.is_file():
        report.add(SKIP, "client deps", "UNIVTAC_PYTHON not usable")
        return
    if not deep:
        report.add(SKIP, "client deps", "use --deep to probe")
        return
    ok, detail = probe_import(python, "zmq, msgpack, msgpack_numpy, yaml", timeout=120)
    if ok:
        report.add(PASS, "client deps", "zmq/msgpack/msgpack_numpy/yaml")
    else:
        report.add(
            FAIL, "client deps", detail,
            f"  {python} -m pip install -r {_REPO_ROOT}/requirements-client.txt",
        )


def check_hf(report: Report) -> None:
    """Hugging Face credentials for the gated Cosmos-Reason2-2B backbone."""
    if os.environ.get("HF_TOKEN"):
        # Per huggingface_hub docs this overrides any stored login, which is how
        # you get past a shared machine's ambient identity.
        report.add(PASS, "HF credentials", "HF_TOKEN set (overrides stored login)")
        return
    token_paths = [
        Path.home() / ".cache/huggingface/token",
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "token",
    ]
    if any(p.is_file() and p.stat().st_size > 0 for p in token_paths):
        report.add(PASS, "HF credentials", "cached token found")
    else:
        report.add(
            FAIL,
            "HF credentials",
            "no HF_TOKEN and no cached token",
            "  Request access to https://huggingface.co/nvidia/Cosmos-Reason2-2B\n"
            "  (every GR00T checkpoint loads it), then:\n"
            "    huggingface-cli login     # or: export HF_TOKEN=<token>",
        )


GATED_BACKBONE = "nvidia/Cosmos-Reason2-2B"
"""GR00T N1.7's VLM backbone. Gated, and every checkpoint loads it."""

_ACCESS_PROBE = """
import os, sys
from huggingface_hub import HfApi
tok = os.environ.get("HF_TOKEN") or None
api = HfApi(token=tok)
try:
    who = api.whoami().get("name", "?")
except Exception as e:
    print("WHOAMI_FAIL", type(e).__name__, str(e)[:120]); sys.exit(2)
try:
    api.model_info("%s")
    print("OK", who)
except Exception as e:
    print("NO_ACCESS", who, type(e).__name__, str(e)[:160]); sys.exit(3)
""" % GATED_BACKBONE


def check_gated_access(report: Report, deep: bool) -> None:
    """Confirm the *effective* token can actually reach the gated backbone.

    A token merely existing proves nothing: on a shared machine the ambient
    login may belong to someone else, and ``hf auth whoami`` reflects whichever
    token the CLI resolved rather than whether the gate is open for you. This
    asks the Hub directly, using GR00T's interpreter because that is the one
    with ``huggingface_hub`` installed -- and the one that will actually load
    the model.
    """
    if not deep:
        report.add(SKIP, "gated access", "use --deep to query the Hub")
        return
    python = env_path("GROOT_PYTHON")
    if python is None or not python.is_file():
        report.add(SKIP, "gated access", "GROOT_PYTHON not usable")
        return

    try:
        result = subprocess.run(
            [str(python), "-c", _ACCESS_PROBE],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        report.add(WARN, "gated access", "Hub query timed out (offline node?)")
        return

    out = (result.stdout or "").strip().splitlines()
    if out:
        line = out[-1]
    else:
        # Last stderr line is the exception; the first is just "Traceback...".
        err = (result.stderr or "").strip().splitlines()
        line = err[-1][:160] if err else ""

    if line.startswith("OK"):
        report.add(PASS, "gated access", f"{GATED_BACKBONE} reachable as {line.split(maxsplit=1)[-1]}")
    elif line.startswith("NO_ACCESS"):
        parts = line.split(maxsplit=2)
        identity = parts[1] if len(parts) > 1 else "?"
        fix = [
            f"  The effective HF identity is {identity!r} and it cannot read",
            f"  {GATED_BACKBONE}. Either that is not your account, or the gate",
            "  is not approved for it yet.",
            f"    1. Request access: https://huggingface.co/{GATED_BACKBONE}",
            "    2. Use YOUR token:  export HF_TOKEN=hf_...",
            "  HF_TOKEN overrides any stored login (huggingface_hub docs: 'If set,",
            "  this value will overwrite the token stored on the machine').",
        ]
        report.add(FAIL, "gated access", f"denied for identity {identity!r}", "\n".join(fix))
    elif line.startswith("WHOAMI_FAIL"):
        report.add(
            FAIL, "gated access", "token rejected by the Hub",
            "  export HF_TOKEN=hf_...   # from https://huggingface.co/settings/tokens",
        )
    else:
        report.add(WARN, "gated access", line[:120] or "probe produced no output")


def check_slurm(report: Report) -> None:
    """Whether we are on a login node, and whether sbatch is available."""
    on_compute = bool(os.environ.get("SLURM_JOB_ID"))
    where = "inside a SLURM job" if on_compute else "login node (no SLURM_JOB_ID)"
    report.add(PASS, "location", where)
    if shutil.which("sbatch"):
        report.add(PASS, "sbatch", shutil.which("sbatch") or "")
    else:
        report.add(WARN, "sbatch", "not on PATH - is this a submit host?")


def check_conda(report: Report) -> None:
    """Report the active conda environment.

    Which env is active decides what ``python`` means. This script and
    ``compare_ablation.py`` only need numpy and belong in a dedicated env
    (``univtac-groot``); the *evaluator* must run under the ``UniVTAC`` env, and
    the server under GR00T's uv venv. ``base`` is typically the cluster's shared
    environment -- installing into it affects other users, and it is the usual
    reason ``import isaaclab`` fails.
    """
    env = os.environ.get("CONDA_DEFAULT_ENV") or ""
    interpreter = f"python {sys.version_info.major}.{sys.version_info.minor}"
    if not env:
        report.add(WARN, "conda env", f"none active ({interpreter})")
    elif env == "base":
        report.add(
            WARN,
            "conda env",
            f"base ({interpreter}) - do not pip install here on a shared cluster; "
            f"see docs/RUNBOOK.md step 2 (conda create -n univtac-groot)",
        )
    else:
        report.add(PASS, "conda env", f"{env} ({interpreter})")


def check_results(report: Report) -> bool:
    """Whether any evaluation results already exist."""
    results = _REPO_ROOT / "eval_result"
    if not results.is_dir():
        report.add(SKIP, "results", "none yet (expected before the first run)")
        return False
    variants = sorted(p.name for p in results.iterdir() if p.is_dir() and p.name != "raw")
    files = list(results.rglob("*.jsonl"))
    if not files:
        report.add(SKIP, "results", f"{results} exists but holds no .jsonl")
        return False
    report.add(PASS, "results", f"{len(files)} file(s), variants={variants}")
    return True


def next_step(*, have_results: bool, deep: bool) -> str:
    """The command to run next, assuming every check passed."""
    if not deep:
        return "  python scripts/preflight.py --deep    # probe both interpreters"
    if not have_results:
        return (
            "  # Interactive first run (COMPUTE node - evaluation is GPU work):\n"
            "  srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 --pty bash\n"
            "  # then follow docs/RUNBOOK.md step 4"
        )
    return (
        "  python scripts/compare_ablation.py     # you have results; aggregate them\n"
        "  # or submit the rest of the sweep: bash slurm/submit_ablation.sh"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the setup and print the next command to run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also probe the two interpreters (imports torch; still no GPU)",
    )
    args = parser.parse_args(argv)

    print("=" * 68)
    print(" GR00T N1.7 x UniVTAC - preflight")
    print("=" * 68)

    report = Report()
    check_slurm(report)
    check_conda(report)
    check_repo(report)
    check_univtac(report)
    check_interpreters(report, args.deep)
    check_cudnn(report, args.deep)
    check_client_deps(report, args.deep)
    check_hf(report)
    check_gated_access(report, args.deep)
    have_results = check_results(report)

    # On success there is no fix to show, so substitute the forward step.
    if report.first_fix is None:
        report.first_fix = next_step(have_results=have_results, deep=args.deep)
    return report.render()


if __name__ == "__main__":
    raise SystemExit(main())
