"""Frame protocol: id matching, out-of-order frames, and cloud-init wiring.

``SocketChannel.request`` now matches the response id. Without that, two
concurrent requests could swap their answers, which is a silent correctness bug
rather than a loud failure.
"""

from __future__ import annotations

import json
import threading

import pytest

from mactf.sandbox.channel import Frame, FrameReader, ProtocolError, SocketChannel
from mactf.sandbox.provision import (
    AGENT_INSTALL_PATH,
    agent_cloud_init_lines,
    seed_payload,
    write_cloud_init,
    write_guest_agent,
)


def test_frame_round_trip():
    frame = Frame(kind="exec", payload={"command": ["ls"]}, id=3)
    decoded = Frame.decode(frame.encode().rstrip(b"\n"))
    assert decoded.kind == "exec"
    assert decoded.id == 3
    assert decoded.payload == {"command": ["ls"]}


def test_frame_rejects_malformed_json():
    with pytest.raises(ProtocolError):
        Frame.decode(b"{not json}")


def test_frame_rejects_missing_kind():
    with pytest.raises(ProtocolError):
        Frame.decode(json.dumps({"id": 1}).encode())


class _FakeChannel(SocketChannel):
    """A channel with no socket, whose reader yields scripted frames."""

    def __init__(self, frames):
        super().__init__("unused")
        self.sent = []
        self._scripted = _ScriptedReader(frames)
        self._reader = self._scripted

    def send(self, frame):
        self.sent.append(frame)


class _ScriptedReader:
    def __init__(self, frames):
        self._frames = list(frames)
        self.error = None

    def read(self, timeout=None):
        return self._frames.pop(0) if self._frames else None


def test_request_skips_mismatched_ids():
    channel = _FakeChannel(
        [
            Frame(kind="exec", payload={"stdout": "wrong"}, id=2),
            Frame(kind="exec", payload={"stdout": "right"}, id=1),
        ]
    )
    response = channel.request("exec", {"command": "ls"}, timeout=5.0)
    assert response.payload["stdout"] == "right"


def test_request_uses_a_monotonic_id():
    channel = _FakeChannel([Frame(kind="ping", payload={}, id=1)])
    channel.request("ping", {}, timeout=5.0)
    channel2 = _FakeChannel([Frame(kind="ping", payload={}, id=1)])
    channel2.request("ping", {}, timeout=5.0)
    assert channel.sent[0].id == 1
    assert channel2.sent[0].id == 1


def test_request_raises_on_error_frame():
    from mactf.exceptions import SandboxError

    channel = _FakeChannel([Frame(kind="error", payload={"message": "nope"}, id=1)])
    with pytest.raises(SandboxError):
        channel.request("ping", {}, timeout=5.0)


def test_request_accepts_frames_without_an_id():
    channel = _FakeChannel([Frame(kind="ping", payload={"pong": True}, id=None)])
    assert channel.request("ping", {}, timeout=5.0).payload["pong"] is True


# -- provisioning --------------------------------------------------------


def test_cloud_init_without_agent_is_unchanged(tmp_path):
    seed = write_cloud_init(tmp_path / "seed", peer="p1", with_agent=False)
    text = seed.user_data.read_text(encoding="utf-8")
    assert "guest_agent.py" not in text
    assert "mactf-agent.service" not in text
    assert "echo mactf-ready" in text


def test_cloud_init_with_agent_installs_a_unit(tmp_path):
    seed = write_cloud_init(tmp_path / "seed", peer="p1", with_agent=True)
    text = seed.user_data.read_text(encoding="utf-8")
    assert AGENT_INSTALL_PATH in text
    assert "mactf-agent.service" in text
    assert "write_files:" in text
    assert "runcmd:" in text
    # The unit body must be present, not just its path.
    assert "ExecStart=" in text
    assert "Restart=always" in text


def test_agent_source_is_embedded_verbatim(tmp_path):
    seed = write_cloud_init(tmp_path / "seed", peer="p1", with_agent=True)
    text = seed.user_data.read_text(encoding="utf-8")
    assert "class GuestAgent" in text
    assert "def _confine" in text


def test_agent_cloud_init_lines_mentions_the_port():
    lines = agent_cloud_init_lines("/dev/virtio-ports/mactf.ctl")
    assert any("systemctl" in line for line in lines)
    assert any("mactf.ctl" in line for line in lines)


def test_write_guest_agent_copies_a_runnable_file(tmp_path):
    path = write_guest_agent(tmp_path)
    assert path.exists()
    source = path.read_text(encoding="utf-8")
    assert source.startswith('"""Guest-side control agent')
    compile(source, str(path), "exec")


def test_seed_payload_includes_the_agent(tmp_path):
    write_cloud_init(tmp_path / "seed", peer="p1", with_agent=True)
    payload = seed_payload(tmp_path / "seed")
    assert "mactf-agent.service" in payload["user-data"]
