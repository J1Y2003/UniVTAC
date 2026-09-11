"""End-to-end test of the ZeroMQ transport against a stand-in policy server.

:mod:`univtac_groot.client` re-implements GR00T's wire format instead of
importing ``gr00t.policy.server_client`` (which would drag torch into Isaac
Sim's interpreter). That makes the encoding a contract worth testing: the
server here is a faithful copy of ``PolicyServer.run``'s dispatch loop
(msgpack over a ``REP`` socket, ``{"endpoint", "data", "api_token"}`` requests,
in-band ``{"error": ...}`` replies), so a drift in the client's framing fails
here rather than after a 3B checkpoint has finished loading on a GPU node.

Skipped when pyzmq / msgpack are unavailable.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
msgpack = pytest.importorskip("msgpack")
mnp = pytest.importorskip("msgpack_numpy")

from univtac_groot.variants import finetuned_baseline_spec  # noqa: E402
from univtac_groot.client import Gr00tClient, PolicyServerError  # noqa: E402
from univtac_groot.metrics import ResultWriter  # noqa: E402
from univtac_groot.obs_adapter import ObsAdapter  # noqa: E402
from univtac_groot.history import ObsHistory  # noqa: E402
from univtac_groot.rollout import (  # noqa: E402
    BatchingPolicy,
    RolloutConfig,
    evaluate,
    resolve_spec_from_policy,
)

from .test_obs_adapter import make_observation  # noqa: E402


ACTION_HORIZON = 16

MODALITY_CONFIGS = {
    "video": {"delta_indices": [0], "modality_keys": ["head", "wrist"]},
    "state": {
        "delta_indices": [0],
        "modality_keys": [
            "eef_9d",
            "joint_position",
            "gripper_position",
        ],
    },
    "action": {
        "delta_indices": list(range(ACTION_HORIZON)),
        "modality_keys": ["joint_position", "gripper_position"],
    },
    "language": {
        "delta_indices": [0],
        "modality_keys": ["annotation.human.task_description"],
    },
}
"""``ModalityConfig`` dataclass fields, before the wire envelope is applied."""


class FakePolicyServer:
    """Mirror of ``gr00t.policy.server_client.PolicyServer``'s dispatch loop."""

    def __init__(self, *, api_token: str | None = None, fail_get_action: bool = False):
        self.api_token = api_token
        self.fail_get_action = fail_get_action
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.running = True
        self.observations: list[dict] = []
        self.resets = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "FakePolicyServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.running = False
        try:
            self.socket.close(linger=0)
            self.context.term()
        except Exception:
            pass

    def __enter__(self) -> "FakePolicyServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- protocol ----------------------------------------------------------
    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            try:
                if not poller.poll(timeout=100):
                    continue
                request = msgpack.unpackb(
                    self.socket.recv(), object_hook=mnp.decode, raw=False
                )
            except Exception:
                return

            try:
                if self.api_token is not None and request.get("api_token") != self.api_token:
                    self._send({"error": "Unauthorized: Invalid API token"})
                    continue
                result = self._dispatch(
                    request.get("endpoint", "get_action"), request.get("data", {})
                )
                self._send(result)
            except Exception as exc:  # noqa: BLE001 - mirrors the real server
                self._send({"error": str(exc)})

    def _send(self, payload: object) -> None:
        self.socket.send(msgpack.packb(payload, default=mnp.encode))

    def _dispatch(self, endpoint: str, data: dict) -> object:
        if endpoint == "ping":
            return {"status": "ok", "message": "Server is running"}
        if endpoint == "kill":
            self.running = False
            return {"status": "ok"}
        if endpoint == "reset":
            self.resets += 1
            return {}
        if endpoint == "get_modality_config":
            # gr00t's ``MsgSerializer._encode_custom`` wraps every ModalityConfig
            # as ``{"__ModalityConfig__": True, "as_json": <dataclass dict>}``.
            # Emitting plain dicts here (as an earlier version of this fake did)
            # concealed a real client bug: the envelope reached the caller
            # un-unwrapped, so ``delta_indices`` came back empty and
            # ``resolve_horizons`` rejected the policy.
            return {
                modality: {"__ModalityConfig__": True, "as_json": cfg}
                for modality, cfg in MODALITY_CONFIGS.items()
            }
        if endpoint == "get_action":
            if self.fail_get_action:
                raise RuntimeError("state dim mismatch")
            observation = data["observation"]
            self.observations.append(observation)
            self._validate(observation)
            return (
                {
                    "joint_position": np.zeros((1, ACTION_HORIZON, 7), np.float32),
                    "gripper_position": np.ones((1, ACTION_HORIZON, 1), np.float32),
                },
                {"latency_ms": 1.0},
            )
        raise ValueError(f"Unknown endpoint: {endpoint}")

    @staticmethod
    def _validate(observation: dict) -> None:
        """The checks ``Gr00tSimPolicyWrapper.check_observation`` performs."""
        for key, value in observation.items():
            if key.startswith("video."):
                assert isinstance(value, np.ndarray), f"{key} is {type(value)}"
                assert value.dtype == np.uint8, f"{key} dtype {value.dtype}"
                assert value.ndim == 5, f"{key} shape {value.shape}"
                assert value.shape[-1] == 3, f"{key} channels {value.shape[-1]}"
            elif key.startswith("state."):
                assert isinstance(value, np.ndarray), f"{key} is {type(value)}"
                assert value.dtype == np.float32, f"{key} dtype {value.dtype}"
                assert value.ndim == 3, f"{key} shape {value.shape}"
            else:
                assert isinstance(value, (list, tuple)), f"{key} is {type(value)}"
                assert isinstance(value[0], str), f"{key}[0] is {type(value[0])}"


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def test_ping_and_modality_config_round_trip():
    with FakePolicyServer() as server:
        with Gr00tClient(port=server.port, timeout_ms=5000) as client:
            assert client.ping() is True
            config = client.get_modality_config()
            assert len(config["action"]["delta_indices"]) == ACTION_HORIZON
            assert "eef_9d" in config["state"]["modality_keys"]


def test_get_action_returns_arrays_with_shapes_preserved():
    spec = finetuned_baseline_spec()
    frame = ObsAdapter(spec)(make_observation(), "insert the tube")
    stacked = ObsHistory(spec.video_delta_indices, spec.state_delta_indices).reset(frame)

    with FakePolicyServer() as server:
        with Gr00tClient(port=server.port, timeout_ms=10_000) as client:
            policy = BatchingPolicy(client, spec)
            action, info = policy.get_action(stacked)

    # msgpack_numpy must preserve dtype and shape across the wire.
    assert action["joint_position"].shape == (1, ACTION_HORIZON, 7)
    assert action["joint_position"].dtype == np.float32
    assert info["latency_ms"] == pytest.approx(1.0)
    # And the server accepted the observation shapes it validates.
    assert server.observations[0]["video.head"].shape == (1, 1, 256, 256, 3)


def test_server_side_error_surfaces_as_an_exception():
    with FakePolicyServer(fail_get_action=True) as server:
        with Gr00tClient(port=server.port, timeout_ms=5000) as client:
            with pytest.raises(PolicyServerError, match="state dim mismatch"):
                client.get_action({"state.q": np.zeros((1, 1, 3), np.float32)})


def test_api_token_is_enforced():
    with FakePolicyServer(api_token="secret") as server:
        with Gr00tClient(port=server.port, timeout_ms=5000, api_token="wrong") as client:
            with pytest.raises(PolicyServerError, match="Unauthorized"):
                client.get_modality_config()
        with Gr00tClient(port=server.port, timeout_ms=5000, api_token="secret") as client:
            assert client.ping() is True


def test_ping_on_a_dead_port_returns_false_without_raising():
    with Gr00tClient(port=1, timeout_ms=300) as client:
        assert client.ping() is False


def test_wait_until_ready_times_out_with_an_actionable_message():
    with Gr00tClient(port=1, timeout_ms=300) as client:
        with pytest.raises(TimeoutError, match="Cosmos-Reason2-2B"):
            client.wait_until_ready(timeout_s=0.5, poll_s=0.1)


def test_client_survives_a_timeout_and_can_be_reused():
    """A timed-out REQ socket is unusable until rebuilt; the client must rebuild it."""
    with FakePolicyServer() as server:
        client = Gr00tClient(port=server.port, timeout_ms=5000)
        try:
            assert client.ping() is True
            # Force a timeout against a dead port, then recover.
            client.host, client.port = "127.0.0.1", 1
            client.timeout_ms = 200
            client._connect()
            assert client.ping() is False
            client.host, client.port = "127.0.0.1", server.port
            client.timeout_ms = 5000
            client._connect()
            assert client.ping() is True
        finally:
            client.close()


def test_object_dtype_arrays_are_refused_before_hitting_the_wire():
    """``MsgSerializer`` bans pickle-bearing payloads; fail on our side first."""
    with FakePolicyServer() as server:
        with Gr00tClient(port=server.port, timeout_ms=5000) as client:
            with pytest.raises(TypeError, match="object-dtype"):
                client.call("get_action", observation={"x": np.array([{"a": 1}], dtype=object)})


# --------------------------------------------------------------------------- #
# Whole pipeline over the socket
# --------------------------------------------------------------------------- #


class ScriptedEnv:
    """UniVTAC-shaped env that emits real adapter output and succeeds at step N."""

    max_steps = 20

    def __init__(self, spec, succeed_at: int = 6):
        self.spec = spec
        self.succeed_at = succeed_at
        self.adapter = ObsAdapter(spec)
        self.history = ObsHistory(spec.video_delta_indices, spec.state_delta_indices)
        from univtac_groot.action_adapter import ActionAdapter

        self.action_adapter = ActionAdapter()
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        frame = self.adapter(make_observation(seed=seed or 0), "insert the tube")
        return self.history.reset(frame), {"instruction": "insert the tube"}

    def step(self, action):
        assert action.shape == (8,)
        self.steps += 1
        frame = self.adapter(make_observation(seed=self.steps), "insert the tube")
        self.history.append(frame)
        success = self.steps >= self.succeed_at
        return (
            self.history.observe(),
            float(success),
            success,
            self.steps >= self.max_steps,
            {"success": success, "take_action_cnt": self.steps, "early_stop": False},
        )

    def close(self):
        pass


def test_full_pipeline_over_the_socket(tmp_path):
    """Adapter -> history -> batching -> socket -> chunk -> action, for real."""
    spec = finetuned_baseline_spec()

    with FakePolicyServer() as server:
        with Gr00tClient(port=server.port, timeout_ms=20_000) as client:
            aligned, horizons = resolve_spec_from_policy(
                client, spec, execution_horizon=4, log=lambda _m: None
            )
            assert horizons["action_horizon"] == ACTION_HORIZON
            assert horizons["execution_horizon"] == 4

            env = ScriptedEnv(aligned, succeed_at=6)
            writer = ResultWriter(tmp_path / "r.jsonl", {"variant": "baseline_finetuned"})
            summary = evaluate(
                env,
                BatchingPolicy(client, aligned),
                config=RolloutConfig(num_episodes=3, start_seed=1, execution_horizon=4),
                writer=writer,
                log=lambda _m: None,
            )
            writer.close()

    assert summary["episodes_scored"] == 3
    assert summary["success_rate"] == 1.0
    # 6 actions at an execution horizon of 4 = 2 chunk requests per episode.
    assert summary["total_inferences"] == 6
    assert server.resets == 3


def test_resolve_spec_rejects_a_state_key_mismatch():
    """A checkpoint whose state layout differs from the spec must not silently run."""
    from univtac_groot.spec import ObsSpec, StateField

    with FakePolicyServer() as server:
        with Gr00tClient(port=server.port, timeout_ms=5000) as client:
            # The fake server advertises all three proprioception keys; this spec
            # supplies only the end-effector pose.
            partial = ObsSpec(
                video_keys={"head": "head", "wrist": "wrist"},
                state_fields=(StateField("eef_9d", "eef_9d", 9),),
                language_key="annotation.human.task_description",
            )
            with pytest.raises(ValueError, match="state key mismatch"):
                resolve_spec_from_policy(client, partial, log=lambda _m: None)


def test_modality_config_envelope_is_unwrapped():
    """Regression: gr00t wraps ModalityConfig in a custom msgpack envelope.

    Leaving ``{"__ModalityConfig__": True, "as_json": {...}}`` un-unwrapped made
    ``resolve_horizons`` see no ``delta_indices`` and reject a perfectly good
    policy with "policy declared an empty action.delta_indices".
    """
    with FakePolicyServer() as server:
        with Gr00tClient(port=server.port, timeout_ms=5000) as client:
            config = client.get_modality_config()

    # The marker keys must be gone, and the dataclass fields present.
    for modality, expected in MODALITY_CONFIGS.items():
        entry = config[modality]
        assert "__ModalityConfig__" not in entry, f"{modality} envelope leaked through"
        assert entry["delta_indices"] == expected["delta_indices"]
        assert entry["modality_keys"] == expected["modality_keys"]


def test_horizons_resolve_through_the_real_envelope():
    """The end-to-end path that failed on the cluster: server -> resolve_horizons."""
    from univtac_groot.receding_horizon import resolve_horizons

    with FakePolicyServer() as server:
        with Gr00tClient(port=server.port, timeout_ms=5000) as client:
            horizons = resolve_horizons(
                client.get_modality_config(), execution_horizon=8
            )

    assert horizons["action_horizon"] == ACTION_HORIZON
    assert horizons["execution_horizon"] == 8
    assert horizons["video_delta_indices"] == (0,)
    assert horizons["state_delta_indices"] == (0,)
