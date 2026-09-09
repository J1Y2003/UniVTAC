"""Wire-compatible client for GR00T's ``PolicyServer``, with no ``gr00t`` import.

Why a hand-rolled client instead of ``gr00t.policy.server_client.PolicyClient``:
UniVTAC's evaluator runs inside Isaac Sim's interpreter (``AppLauncher`` must be
constructed before anything else imports omni), which is **Python 3.10** with
torch 2.5.1+cu118; GR00T requires **Python 3.12** with CUDA 12.8. Different minor
Python versions means ``gr00t`` cannot be imported into this process at all, so
the model runs behind a socket -- the same arrangement UniVTAC uses for its own
SmolVLA integration (``policy/smolvla/smolvla_server.py``, which lives in its own
venv). This client needs only numpy, pyzmq, msgpack and msgpack-numpy.

The protocol, read off ``gr00t/policy/server_client.py::PolicyServer.run``:

* ZeroMQ ``REQ``/``REP`` over TCP, one msgpack-encoded dict per round trip;
* request ``{"endpoint": <name>, "data": {<kwargs>}, "api_token": <token?>}``;
* endpoints ``ping``, ``kill``, ``reset``, ``get_action``, ``get_modality_config``;
* ``get_action`` takes ``{"observation": obs, "options": None}`` and returns the
  ``(action, info)`` tuple, which msgpack delivers as a 2-element list;
* an error is returned in-band as ``{"error": "..."}``;
* arrays are encoded by ``msgpack_numpy``, and object-dtype arrays are refused
  on both sides (``MsgSerializer`` enforces ``allow_pickle=False``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - exercised only in a fully provisioned env
    import msgpack
    import msgpack_numpy as mnp
    import zmq
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "univtac_groot.client needs pyzmq, msgpack and msgpack-numpy. "
        "Install them into the UniVTAC/Isaac Lab environment with "
        "`pip install -r requirements-client.txt` (they pull no torch)."
    ) from exc


DEFAULT_PORT = 5555
"""``gr00t.eval.run_gr00t_server.DEFAULT_MODEL_SERVER_PORT``."""


class PolicyServerError(RuntimeError):
    """The server answered with an in-band ``{"error": ...}`` payload."""


def _encode(obj: Any) -> Any:
    """msgpack ``default`` hook that refuses pickle-bearing payloads.

    Mirrors ``MsgSerializer._safe_encode``: object-dtype arrays would be
    serialised by ``msgpack_numpy`` via ``pickle``, which the server rejects, so
    fail here with a message that names the offending key instead.
    """
    if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
        raise TypeError(
            f"refusing to encode an object-dtype ndarray (shape={obj.shape}); "
            f"convert it to a concrete numeric dtype first"
        )
    return mnp.encode(obj)


_MODALITY_MARKERS = (
    "__ModalityConfig__",
    b"__ModalityConfig__",
    "__ModalityConfig_class__",
    b"__ModalityConfig_class__",
)


def _decode(obj: Any) -> Any:
    """msgpack ``object_hook`` mirroring ``MsgSerializer._safe_decode``."""
    if isinstance(obj, dict):
        # The ``__ndarray_class__`` envelope carries a raw .npy payload; load it
        # with allow_pickle=False, as the server does.
        if obj.get("__ndarray_class__", obj.get(b"__ndarray_class__")):
            payload = obj.get("as_npy", obj.get(b"as_npy"))
            if payload is None:
                raise ValueError("malformed ndarray payload: marker set but 'as_npy' missing")
            import io

            return np.load(io.BytesIO(payload), allow_pickle=False)

        # ModalityConfig envelope from ``MsgSerializer._encode_custom``. The
        # server rebuilds a real ``ModalityConfig``; importing gr00t is what
        # this client exists to avoid, so return the dataclass dict as-is.
        if any(marker in obj for marker in _MODALITY_MARKERS):
            key = next((k for k in ("as_json", b"as_json") if k in obj), None)
            if key is None:
                raise ValueError(
                    "malformed ModalityConfig payload: marker present but 'as_json' "
                    f"missing. keys={sorted(repr(k) for k in obj)}"
                )
            payload = obj[key]
            if isinstance(payload, bytes):
                payload = payload.decode()
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                return {
                    (k.decode() if isinstance(k, bytes) else k): v for k, v in payload.items()
                }
            return payload

        nd = obj.get(b"nd", obj.get("nd"))
        kind = obj.get(b"kind", obj.get("kind"))
        if nd and kind in (b"O", "O"):
            raise ValueError("refusing to decode an object-dtype (pickle-bearing) payload")
    return mnp.decode(obj)


class Gr00tClient:
    """Blocking client for a remote ``Gr00tPolicy`` wrapped in ``PolicyServer``.

    Args:
        host: server host. Use the SLURM node name when the model runs in a
            separate job step.
        port: server port.
        timeout_ms: per-request send/receive timeout. Loading a 3B checkpoint
            takes minutes, so the eval driver waits with :meth:`wait_until_ready`
            rather than by inflating this.
        api_token: matches ``PolicyServer(api_token=...)`` when the server is set
            up with one.

    Example:
        >>> with Gr00tClient(port=5555) as client:  # doctest: +SKIP
        ...     cfg = client.get_modality_config()
        ...     action, info = client.get_action(observation)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        *,
        timeout_ms: int = 60_000,
        api_token: str | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.api_token = api_token
        self._context = zmq.Context()
        self._socket: Any = None
        self._closed = False
        self._connect()

    # -- transport ---------------------------------------------------------
    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, **data: Any) -> Any:
        """Invoke ``endpoint`` with keyword payload ``data``.

        A timeout leaves a ``REQ`` socket in an unusable state (it may not send
        again until it has received), so the socket is rebuilt before raising.
        """
        request: dict[str, Any] = {"endpoint": endpoint, "data": data}
        if self.api_token is not None:
            request["api_token"] = self.api_token

        payload = msgpack.packb(request, default=_encode)
        try:
            self._socket.send(payload)
            reply = self._socket.recv()
        except zmq.error.Again as exc:
            self._connect()
            raise TimeoutError(
                f"GR00T server at {self.host}:{self.port} did not answer "
                f"{endpoint!r} within {self.timeout_ms} ms"
            ) from exc
        except zmq.error.ZMQError as exc:
            self._connect()
            raise ConnectionError(
                f"transport error talking to {self.host}:{self.port}: {exc}"
            ) from exc

        result = msgpack.unpackb(reply, object_hook=_decode, raw=False)
        if isinstance(result, Mapping) and "error" in result and len(result) == 1:
            raise PolicyServerError(f"{endpoint}: {result['error']}")
        return result

    # -- endpoints ---------------------------------------------------------
    def ping(self) -> bool:
        """Whether the server responds. Never raises on timeout."""
        try:
            reply = self.call("ping")
        except (TimeoutError, ConnectionError, PolicyServerError):
            return False
        return isinstance(reply, Mapping) and reply.get("status") == "ok"

    def wait_until_ready(self, timeout_s: float = 900.0, poll_s: float = 2.0) -> None:
        """Block until :meth:`ping` succeeds, or raise ``TimeoutError``.

        The default is generous because the server has to pull and load
        ``nvidia/GR00T-N1.7-3B`` plus its gated ``nvidia/Cosmos-Reason2-2B``
        backbone before it binds a live policy.
        """
        import time

        deadline = time.monotonic() + timeout_s
        # Ping with a short timeout so a dead server is detected promptly.
        saved, self.timeout_ms = self.timeout_ms, min(self.timeout_ms, 5_000)
        self._connect()
        try:
            while time.monotonic() < deadline:
                if self.ping():
                    return
                time.sleep(poll_s)
        finally:
            self.timeout_ms = saved
            self._connect()
        raise TimeoutError(
            f"GR00T server at {self.host}:{self.port} was not ready within {timeout_s:.0f}s. "
            f"Check the server job's log: a gated-repo 401 on nvidia/Cosmos-Reason2-2B "
            f"is the usual cause."
        )

    def get_modality_config(self) -> dict[str, Any]:
        """Fetch the policy's ``{modality: ModalityConfig}`` mapping.

        Values arrive as dicts (msgpack cannot carry the dataclass), which
        :func:`univtac_groot.receding_horizon.resolve_horizons` accepts directly.
        """
        reply = self.call("get_modality_config")
        if not isinstance(reply, Mapping):
            raise PolicyServerError(f"unexpected get_modality_config reply: {type(reply)}")
        return dict(reply)

    def get_action(self, observation: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict]:
        """Run one chunk inference. Returns ``(action, info)``."""
        reply = self.call("get_action", observation=dict(observation), options=None)
        # BasePolicy.get_action returns a tuple; msgpack delivers it as a list.
        if isinstance(reply, (list, tuple)) and len(reply) == 2:
            action, info = reply
        elif isinstance(reply, Mapping):
            action, info = reply, {}
        else:
            raise PolicyServerError(f"unexpected get_action reply: {type(reply)}")
        if not isinstance(action, Mapping):
            raise PolicyServerError(f"action payload is not a mapping: {type(action)}")
        return {str(k): np.asarray(v) for k, v in action.items()}, dict(info or {})

    def reset(self, **options: Any) -> dict:
        """Reset policy-side episode state."""
        reply = self.call("reset", options=options or None)
        return dict(reply) if isinstance(reply, Mapping) else {}

    def kill_server(self) -> None:
        """Ask the server to stop its run loop."""
        try:
            self.call("kill")
        except (TimeoutError, ConnectionError, PolicyServerError):
            pass  # the server may drop the connection as it shuts down

    # -- cleanup -----------------------------------------------------------
    def close(self) -> None:
        """Release the socket and ZMQ context. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        try:
            self._context.term()
        except Exception:
            pass

    def __enter__(self) -> "Gr00tClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
