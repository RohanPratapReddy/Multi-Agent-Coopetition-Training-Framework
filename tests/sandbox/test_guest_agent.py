"""Guest agent: real subprocess execution over real frames.

No test launches a VM. The agent is driven directly with ``Frame`` objects and,
for the streaming path, over a pipe pair, which is the same byte protocol the
virtio-serial transport carries.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time

import pytest

from mactf.sandbox.guest_agent import Frame, GuestAgent


@pytest.fixture()
def workdir(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    return root, scratch


@pytest.fixture()
def agent(workdir):
    root, scratch = workdir
    return GuestAgent(
        workdir=str(root),
        allowed_paths=(str(root), str(scratch)),
        max_output_bytes=4096,
        default_timeout=10.0,
    )


def _exec(agent, command, **payload):
    body = {"command": command}
    body.update(payload)
    return agent.handle(Frame(kind="exec", payload=body, id=7))


# -- ping and health -----------------------------------------------------


def test_ping_shape(agent):
    response = agent.handle(Frame(kind="ping", payload={}, id=1))
    assert response.kind == "ping"
    assert response.payload["pong"] is True
    assert isinstance(response.payload["pid"], int)
    assert response.payload["uptime"] >= 0.0
    assert response.id == 1


def test_health_shape(agent):
    response = agent.handle(Frame(kind="health", payload={}, id=2))
    assert response.kind == "health"
    assert response.payload["ready"] is True
    assert isinstance(response.payload["python"], str)
    assert isinstance(response.payload["free_bytes"], int)
    assert isinstance(response.payload["load"], float)
    assert response.id == 2


# -- exec ----------------------------------------------------------------


def test_exec_list_command(agent):
    response = _exec(agent, [sys.executable, "-c", "print(41+1)"])
    # Windows children emit CRLF; the agent must not rewrite child output.
    assert response.payload["stdout"].replace("\r\n", "\n") == "42\n"
    assert response.payload["exit_code"] == 0
    assert response.payload["timed_out"] is False
    assert response.payload["duration"] >= 0.0


@pytest.mark.skipif(os.name != "posix", reason="shell semantics differ on Windows")
def test_exec_string_uses_a_shell(agent):
    response = _exec(agent, "echo hello && echo world")
    assert response.payload["exit_code"] == 0
    assert "hello" in response.payload["stdout"]
    assert "world" in response.payload["stdout"]


def test_exec_failure_returns_real_code_and_stderr(agent):
    response = _exec(agent, [sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"])
    assert response.payload["exit_code"] == 3
    assert "bad" in response.payload["stderr"]
    assert response.payload["timed_out"] is False


def test_exec_missing_binary(agent):
    response = _exec(agent, ["definitely-not-a-real-binary-xyz"])
    assert response.payload["exit_code"] == 127
    assert "not found" in response.payload["stderr"]


def test_exec_timeout_kills_and_reports(agent):
    started = time.monotonic()
    response = _exec(agent, [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    elapsed = time.monotonic() - started
    assert response.payload["timed_out"] is True
    assert response.payload["exit_code"] == 124
    assert elapsed < 15.0


def test_exec_cwd_outside_roots_is_rejected(agent):
    response = _exec(agent, [sys.executable, "-c", "print(1)"], cwd="/")
    assert response.payload["exit_code"] == 126
    assert "outside allowed roots" in response.payload["stderr"]


def test_exec_env_is_passed(agent):
    response = _exec(
        agent,
        [sys.executable, "-c", "import os; print(os.environ['MACTF_TEST_VALUE'])"],
        env={"MACTF_TEST_VALUE": "abc123"},
    )
    assert response.payload["stdout"].strip() == "abc123"


def test_exec_truncates_oversized_output(agent):
    response = _exec(agent, [sys.executable, "-c", "print('x' * 20000)"])
    assert response.payload["truncated"] is True
    assert response.payload["bytes_total"] > 4096
    assert len(response.payload["stdout"]) <= 4096


# -- put and get ---------------------------------------------------------


def test_put_get_round_trip_bytes(agent, workdir):
    root, _ = workdir
    payload = bytes(range(256)) * 2
    put = agent.handle(
        Frame(
            kind="put",
            payload={
                "path": str(root / "blob.bin"),
                "content_b64": base64.b64encode(payload).decode("ascii"),
                "mode": 0o600,
            },
            id=3,
        )
    )
    assert put.payload["written"] == len(payload)
    get = agent.handle(Frame(kind="get", payload={"path": str(root / "blob.bin")}, id=4))
    assert base64.b64decode(get.payload["content_b64"]) == payload
    assert get.payload["size"] == len(payload)


def test_put_creates_parent_directories(agent, workdir):
    root, _ = workdir
    target = root / "nested" / "deep" / "file.txt"
    response = agent.handle(
        Frame(
            kind="put",
            payload={
                "path": str(target),
                "content_b64": base64.b64encode(b"ok").decode("ascii"),
            },
            id=5,
        )
    )
    assert target.read_bytes() == b"ok"
    assert response.payload["written"] == 2


def test_get_missing_file_is_an_error_frame(agent, workdir):
    root, _ = workdir
    response = agent.handle(
        Frame(kind="get", payload={"path": str(root / "nope.bin")}, id=6)
    )
    assert response.kind == "error"
    assert "not a file" in response.payload["message"]


def test_get_respects_max_bytes(agent, workdir):
    root, _ = workdir
    path = root / "big.bin"
    path.write_bytes(b"a" * 1000)
    response = agent.handle(
        Frame(kind="get", payload={"path": str(path), "max_bytes": 10}, id=8)
    )
    assert response.payload["truncated"] is True
    assert len(base64.b64decode(response.payload["content_b64"])) == 10


def test_path_escape_is_rejected(agent, workdir):
    root, _ = workdir
    outside = root.parent / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    response = agent.handle(
        Frame(kind="get", payload={"path": str(root / ".." / "secret.txt")}, id=9)
    )
    assert response.kind == "error"
    assert "outside allowed roots" in response.payload["message"]


def test_symlink_escape_is_rejected(agent, workdir):
    root, _ = workdir
    outside = root.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    link = root / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    response = agent.handle(Frame(kind="get", payload={"path": str(link)}, id=10))
    assert response.kind == "error"
    assert "outside allowed roots" in response.payload["message"]


# -- robustness ----------------------------------------------------------


def test_unsupported_kind_is_an_error_frame(agent):
    response = agent.handle(Frame(kind="explode", payload={}, id=11))
    assert response.kind == "error"
    assert "unsupported kind" in response.payload["message"]
    assert response.id == 11


def test_handler_exception_becomes_an_error_frame(agent, monkeypatch):
    def broken(frame):
        raise RuntimeError("handler exploded")

    agent._handlers["exec"] = broken
    response = _exec(agent, ["true"])
    assert response.kind == "error"
    assert "handler exploded" in response.payload["message"]
    # The agent still serves the next frame.
    assert agent.handle(Frame(kind="ping", payload={}, id=12)).kind == "ping"


def test_shutdown_stops_the_loop(agent):
    response = agent.handle(Frame(kind="shutdown", payload={"graceful": True}, id=13))
    assert response.payload["stopping"] is True
    assert agent._running is False


# -- stream protocol -----------------------------------------------------


def test_stream_serves_multiple_frames():
    agent = GuestAgent(workdir=os.getcwd(), allowed_paths=(os.getcwd(),))
    incoming = io.BytesIO(
        Frame(kind="ping", payload={}, id=1).encode()
        + Frame(kind="health", payload={}, id=2).encode()
        + Frame(kind="shutdown", payload={}, id=3).encode()
    )

    class _Pipe(io.BytesIO):
        def flush(self):
            pass

    outgoing = _Pipe()
    agent.serve_stream(_Duplex(incoming, outgoing))
    payload = outgoing.getvalue()
    frames = [Frame.decode(line) for line in payload.split(b"\n") if line.strip()]
    assert [frame.kind for frame in frames] == ["ping", "health", "shutdown"]
    assert [frame.id for frame in frames] == [1, 2, 3]


def test_stream_survives_a_malformed_frame():
    agent = GuestAgent(workdir=os.getcwd(), allowed_paths=(os.getcwd(),))
    incoming = io.BytesIO(b"{not json}\n" + Frame(kind="ping", payload={}, id=5).encode())

    class _Pipe(io.BytesIO):
        def flush(self):
            pass

    outgoing = _Pipe()
    agent.serve_stream(_Duplex(incoming, outgoing))
    frames = [
        Frame.decode(line) for line in outgoing.getvalue().split(b"\n") if line.strip()
    ]
    assert frames[0].kind == "error"
    assert frames[1].kind == "ping"


class _Duplex:
    """A minimal read/write duplex over two byte buffers."""

    def __init__(self, incoming, outgoing):
        self._in = incoming
        self._out = outgoing

    def read(self, size=-1):
        return self._in.read(size)

    def write(self, data):
        return self._out.write(data)

    def flush(self):
        pass
