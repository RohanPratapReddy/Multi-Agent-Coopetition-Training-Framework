"""The host/guest protocol for QEMU sandboxes.

The docx describes a host-side orchestrator driving a guest over a serial port.
This module defines that wire format: newline-delimited JSON frames with a
monotonic id, so a dropped or reordered response is detectable rather than
silently misinterpreted.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..exceptions import SandboxError

FRAME_SEPARATOR = b"\n"
MAX_FRAME_BYTES = 1 << 22


class ProtocolError(SandboxError):
    """A frame was malformed or out of sequence."""


@dataclass
class Frame:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    ts: float = 0.0

    def encode(self) -> bytes:
        body = {
            "kind": self.kind,
            "id": self.id,
            "ts": self.ts or time.time(),
            "payload": self.payload,
        }
        return json.dumps(body, separators=(",", ":")).encode("utf-8") + FRAME_SEPARATOR

    @classmethod
    def decode(cls, raw: bytes) -> "Frame":
        if len(raw) > MAX_FRAME_BYTES:
            raise ProtocolError(f"frame exceeds {MAX_FRAME_BYTES} bytes")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"malformed frame: {exc}") from exc
        if not isinstance(body, dict) or "kind" not in body:
            raise ProtocolError("frame must be a JSON object with a 'kind' field")
        payload = body.get("payload") or {}
        if not isinstance(payload, dict):
            raise ProtocolError("frame payload must be an object")
        return cls(
            kind=str(body["kind"]),
            payload=payload,
            id=body.get("id"),
            ts=float(body.get("ts") or 0.0),
        )


class FrameReader(threading.Thread):
    """Background reader that reassembles newline-delimited frames."""

    def __init__(self, stream: Any, *, name: str = "mactf-frame-reader") -> None:
        super().__init__(name=name, daemon=True)
        self._stream = stream
        self._buffer = bytearray()
        self._frames: queue.Queue[Frame] = queue.Queue()
        self._error: Exception | None = None
        self._stop = threading.Event()

    @property
    def error(self) -> Exception | None:
        return self._error

    def run(self) -> None:  # pragma: no cover - exercised via integration tests
        try:
            while not self._stop.is_set():
                chunk = self._stream.read(1)
                if not chunk:
                    if self._stop.is_set():
                        break
                    time.sleep(0.01)
                    continue
                if chunk == FRAME_SEPARATOR:
                    if not self._buffer:
                        continue
                    frame = Frame.decode(bytes(self._buffer))
                    self._buffer.clear()
                    self._frames.put(frame)
                else:
                    self._buffer.extend(chunk)
                    if len(self._buffer) > MAX_FRAME_BYTES:
                        raise ProtocolError("unterminated frame exceeded size limit")
        except Exception as exc:  # noqa: BLE001 - surfaced through ``error``
            self._error = exc
        finally:
            # A partial trailing frame is a protocol violation, not a silent drop.
            if self._buffer and self._error is None:
                self._error = ProtocolError("stream ended mid-frame")

    def read(self, timeout: float | None = None) -> Frame | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[Frame]:
        frames: list[Frame] = []
        while True:
            try:
                frames.append(self._frames.get_nowait())
            except queue.Empty:
                return frames

    def stop(self) -> None:
        self._stop.set()


class SocketChannel:
    """Host side of a virtio-serial socket channel."""

    def __init__(
        self,
        path: str | "os.PathLike[str]",
        *,
        connect_timeout: float = 10.0
    ) -> None:
        import os

        self.path = os.fspath(path)
        self.connect_timeout = connect_timeout
        self._socket: socket.socket | None = None
        self._reader: FrameReader | None = None

    def connect(self, *, deadline: float | None = None) -> "SocketChannel":
        deadline = deadline or time.monotonic() + self.connect_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(1.0)
                client.connect(self.path)
                self._socket = client
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.1)
        if self._socket is None:
            raise SandboxError(
                f"could not connect to sandbox channel {self.path!r}: {last_error}"
            )
        self._reader = FrameReader(self._socket.makefile("rb"), name="mactf-channel")
        self._reader.start()
        return self

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def send(self, frame: Frame) -> None:
        if self._socket is None:
            raise SandboxError("channel is not connected")
        self._socket.sendall(frame.encode())

    def request(self, kind: str, payload: dict[str, Any] | None = None, *, timeout: float = 30.0) -> Frame:
        if self._reader is None:
            raise SandboxError("channel is not connected")
        self.send(Frame(kind=kind, payload=payload or {}))
        response = self._reader.read(timeout=timeout)
        if response is None:
            if self._reader.error is not None:
                raise ProtocolError(str(self._reader.error)) from self._reader.error
            raise SandboxError(f"timed out waiting for a {kind!r} response after {timeout}s")
        if response.kind == "error":
            raise SandboxError(
                f"guest rejected {kind!r}: {response.payload.get('message', 'no message')}"
            )
        return response

    def close(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader.join(timeout=2.0)
            self._reader = None
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def __enter__(self) -> "SocketChannel":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def on_frame(callback: Callable[[Frame], None]) -> Callable[[Frame], None]:
    """Tiny decorator to register a frame handler by annotation-free convention."""

    def handler(frame: Frame) -> None:
        callback(frame)

    return handler
