"""Guest-side control agent for MACTF QEMU sandboxes. Stdlib only.

This file is deliberately standalone: no ``mactf`` import, no third-party
dependency. It is copied into a VM image (or injected by cloud-init) and run as
a service there, where the framework is not installed and must not be.

Wire format
-----------
Identical to :mod:`mactf.sandbox.channel`: newline-delimited JSON objects with
``kind``, ``id``, ``ts``, and ``payload``. The guest is the **server** and the
host's ``SocketChannel`` is the client. In QEMU the transport is a
``virtio-serial`` character device (``/dev/virtio-ports/mactf.ctl``) rather than
a UNIX socket, so this module opens the device for read/write instead of
connecting. Keeping the frame bytes identical is what lets one protocol serve
both transports.

Semantics that a naive implementation gets wrong
------------------------------------------------
1. ``exec`` never uses ``shell=True`` for a list command. A list is argv; a
   string is explicitly shell semantics and goes through ``/bin/sh -c``.
2. Output is truncated at ``max_output_bytes`` per stream and the response says
   so. Silent truncation would let a test pass on a log it never fully saw.
3. A timeout kills the whole process group, not just the child, or a hanging
   grandchild holds the port open forever.
4. Paths are confined to ``allowed_paths`` *after* ``realpath``, so ``..`` and
   symlinks cannot escape.
5. Nothing raises out of the loop. A crashed agent makes the host hang until its
   timeout, which is strictly worse than an error frame.
6. The frame ``id`` is echoed, because the host matches responses by id.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

DEFAULT_PORT = "/dev/virtio-ports/mactf.ctl"
FRAME_SEPARATOR = b"\n"
MAX_FRAME_BYTES = 1 << 22
DEFAULT_ALLOWED_PATHS = ("/app", "/tmp/mactf")


class Frame:
    """The wire frame, duplicated from the host's channel module on purpose."""

    __slots__ = ("kind", "payload", "id", "ts")

    def __init__(self, kind, payload=None, id=None, ts=0.0):
        self.kind = kind
        self.payload = payload or {}
        self.id = id
        self.ts = ts or time.time()

    def encode(self):
        body = {
            "kind": self.kind,
            "id": self.id,
            "ts": self.ts,
            "payload": self.payload,
        }
        return json.dumps(body, separators=(",", ":")).encode("utf-8") + FRAME_SEPARATOR

    @classmethod
    def decode(cls, raw):
        if len(raw) > MAX_FRAME_BYTES:
            raise ValueError("frame exceeds %d bytes" % MAX_FRAME_BYTES)
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict) or "kind" not in body:
            raise ValueError("frame must be a JSON object with a 'kind' field")
        payload = body.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("frame payload must be an object")
        return cls(
            kind=str(body["kind"]),
            payload=payload,
            id=body.get("id"),
            ts=float(body.get("ts") or 0.0),
        )

    def describe(self):
        return "frame[%s id=%s %d payload key(s)]" % (
            self.kind,
            self.id,
            len(self.payload),
        )


class GuestAgent:
    """Serve MACTF control frames from a character device or file object."""

    VERSION = "1.0"

    def __init__(
        self,
        port=DEFAULT_PORT,
        *,
        workdir="/app",
        allowed_paths=DEFAULT_ALLOWED_PATHS,
        max_output_bytes=1 << 20,
        default_timeout=300.0,
    ):
        self.port = port
        self.workdir = workdir
        self.allowed_paths = tuple(
            os.path.realpath(os.path.expanduser(path)) for path in allowed_paths
        )
        self.max_output_bytes = int(max_output_bytes)
        self.default_timeout = float(default_timeout)
        self._started = time.monotonic()
        self._running = True
        self._handlers = {
            "ping": self._handle_ping,
            "health": self._handle_health,
            "exec": self._handle_exec,
            "put": self._handle_put,
            "get": self._handle_get,
            "shutdown": self._handle_shutdown,
        }

    # -- serving ---------------------------------------------------------
    def serve_forever(self):
        """Read frames from ``port`` until shutdown or EOF."""
        fd = os.open(self.port, os.O_RDWR)
        try:
            stream = os.fdopen(fd, "rwb", buffering=0)
            self.serve_stream(stream)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def serve_stream(self, stream):
        """Serve frames from any binary, line-oriented stream.

        Split out from :meth:`serve_forever` so tests and cloud-init wrappers can
        drive the agent over a pipe without a virtio device.
        """
        buffer = bytearray()
        while self._running:
            chunk = stream.read(1)
            if not chunk:
                break
            if chunk == FRAME_SEPARATOR:
                if not buffer:
                    continue
                try:
                    frame = Frame.decode(bytes(buffer))
                except ValueError as exc:
                    response = Frame(
                        kind="error", payload={"message": "bad frame: %s" % exc}
                    )
                else:
                    response = self._safe_handle(frame)
                buffer.clear()
                stream.write(response.encode())
                try:
                    stream.flush()
                except (AttributeError, OSError):
                    pass
                continue
            buffer.extend(chunk)
            if len(buffer) > MAX_FRAME_BYTES:
                buffer.clear()
                stream.write(
                    Frame(
                        kind="error",
                        payload={"message": "unterminated frame exceeded size limit"},
                    ).encode()
                )
                try:
                    stream.flush()
                except (AttributeError, OSError):
                    pass

    def _safe_handle(self, frame):
        """Dispatch one frame, converting every failure into an error frame."""
        if frame.kind == "shutdown":
            response = self._handle_shutdown(frame)
            self._running = False
            return self._with_id(response, frame)
        handler = self._handlers.get(frame.kind)
        if handler is None:
            return self._with_id(
                Frame(
                    kind="error",
                    payload={"message": "unsupported kind %r" % frame.kind},
                ),
                frame,
            )
        try:
            return self._with_id(handler(frame), frame)
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            return self._with_id(
                Frame(
                    kind="error",
                    payload={
                        "message": "%s: %s" % (type(exc).__name__, exc),
                        "traceback": traceback.format_exc()[-2000:],
                    },
                ),
                frame,
            )

    @staticmethod
    def _with_id(response, frame):
        response.id = frame.id
        return response

    def handle(self, frame):
        """Handle one frame and return the response frame (never raises)."""
        return self._safe_handle(frame)

    # -- handlers --------------------------------------------------------
    def _handle_ping(self, frame):
        return Frame(
            kind="ping",
            payload={
                "pong": True,
                "pid": os.getpid(),
                "uptime": time.monotonic() - self._started,
                "version": self.VERSION,
            },
        )

    def _handle_health(self, frame):
        return Frame(
            kind="health",
            payload={
                "ready": True,
                "python": sys.version.split()[0],
                "cwd": os.getcwd(),
                "free_bytes": self._free_bytes(),
                "load": _load_average(),
                "version": self.VERSION,
            },
        )

    def _handle_exec(self, frame):
        payload = frame.payload
        command = payload.get("command")
        if not command:
            return Frame(kind="error", payload={"message": "exec requires 'command'"})

        timeout = float(payload.get("timeout") or self.default_timeout)
        try:
            cwd = self._confine(payload.get("cwd") or self.workdir)
        except ValueError as exc:
            return Frame(
                kind="exec",
                payload={
                    "stdout": "",
                    "stderr": str(exc),
                    "exit_code": 126,
                    "duration": 0.0,
                    "timed_out": False,
                    "truncated": False,
                    "error": "path outside allowed roots",
                },
            )

        environment = dict(os.environ)
        environment.update(
            {str(key): str(value) for key, value in (payload.get("env") or {}).items()}
        )

        if isinstance(command, (list, tuple)):
            argv = [str(part) for part in command]
        else:
            # A string is an explicit request for shell semantics.
            argv = ["/bin/sh", "-c", str(command)]

        started = time.monotonic()
        stdout, stderr, code, timed_out = _run_captured(
            argv,
            cwd=cwd,
            env=environment,
            timeout=timeout,
            max_bytes=self.max_output_bytes,
        )
        duration = time.monotonic() - started

        return Frame(
            kind="exec",
            payload={
                "stdout": stdout["text"],
                "stderr": stderr["text"],
                "exit_code": int(code),
                "duration": duration,
                "timed_out": bool(timed_out),
                "truncated": bool(stdout["truncated"] or stderr["truncated"]),
                "bytes_total": stdout["total"] + stderr["total"],
            },
        )

    def _handle_put(self, frame):
        payload = frame.payload
        path = self._confine(payload.get("path"))
        content = base64.b64decode(payload.get("content_b64") or "")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        mode = payload.get("mode")
        with open(path, "wb") as handle:
            handle.write(content)
        if mode is not None:
            os.chmod(path, int(mode))
        return Frame(kind="put", payload={"written": len(content), "path": path})

    def _handle_get(self, frame):
        payload = frame.payload
        path = self._confine(payload.get("path"))
        if not os.path.isfile(path):
            return Frame(kind="error", payload={"message": "not a file: %s" % path})
        limit = int(payload.get("max_bytes") or self.max_output_bytes)
        with open(path, "rb") as handle:
            raw = handle.read(limit)
        size = os.path.getsize(path)
        return Frame(
            kind="get",
            payload={
                "content_b64": base64.b64encode(raw).decode("ascii"),
                "size": size,
                "truncated": size > len(raw),
            },
        )

    def _handle_shutdown(self, frame):
        self._running = False
        return Frame(kind="shutdown", payload={"stopping": True})

    # -- helpers ---------------------------------------------------------
    def _confine(self, path):
        """Resolve ``path`` and reject anything outside the allowed roots."""
        if not path:
            raise ValueError("path outside allowed roots: empty")
        candidate = os.path.realpath(os.path.expanduser(str(path)))
        for root in self.allowed_paths:
            if candidate == root or candidate.startswith(root + os.sep):
                return candidate
        raise ValueError(
            "path outside allowed roots: %s (allowed: %s)"
            % (candidate, ", ".join(self.allowed_paths))
        )

    def _free_bytes(self):
        for candidate in (self.workdir, "/"):
            try:
                return int(shutil.disk_usage(candidate).free)
            except OSError:
                continue
        return 0


def _load_average():
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return 0.0


class _Stream:
    """A bounded byte accumulator that remembers whether it truncated."""

    __slots__ = ("buffer", "total", "truncated", "limit")

    def __init__(self, limit):
        self.buffer = bytearray()
        self.total = 0
        self.truncated = False
        self.limit = int(limit)

    def append(self, chunk):
        self.total += len(chunk)
        room = self.limit - len(self.buffer)
        if room <= 0:
            self.truncated = True
            return
        if len(chunk) > room:
            self.truncated = True
            chunk = chunk[:room]
        self.buffer.extend(chunk)

    @property
    def text(self):
        return bytes(self.buffer).decode("utf-8", errors="replace")

    def as_dict(self):
        return {"text": self.text, "total": self.total, "truncated": self.truncated}


def _kill_group(process):
    """Kill the process and every descendant, not just the direct child."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _run_captured(argv, *, cwd, env, timeout, max_bytes):
    """Run ``argv`` with a bounded, streamed capture and a group timeout."""
    stdout = _Stream(max_bytes)
    stderr = _Stream(max_bytes)
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            shell=False,
        )
    except FileNotFoundError:
        stderr.append(("command not found: %s" % argv[0]).encode())
        return stdout.as_dict(), stderr.as_dict(), 127, False
    except PermissionError:
        stderr.append(("permission denied: %s" % argv[0]).encode())
        return stdout.as_dict(), stderr.as_dict(), 126, False
    except OSError as exc:
        stderr.append(("could not start %s: %s" % (argv[0], exc)).encode())
        return stdout.as_dict(), stderr.as_dict(), 126, False

    readers = []
    for stream, sink in ((process.stdout, stdout), (process.stderr, stderr)):
        if stream is None:
            continue
        reader = _Drain(stream, sink)
        reader.start()
        readers.append(reader)

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(process)
        try:
            process.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, OSError):
            pass

    for reader in readers:
        reader.join(timeout=5.0)

    code = process.returncode
    if timed_out and (code is None or code >= 0):
        # 124 is the conventional "timed out" status.
        code = 124
    return stdout.as_dict(), stderr.as_dict(), int(code if code is not None else 1), timed_out


class _Drain:
    """Drain one pipe on a daemon thread so neither pipe can block the other."""

    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink
        self._thread = threading.Thread(target=self._pump, daemon=True)

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)

    def _pump(self):
        try:
            while True:
                chunk = self._stream.read(4096)
                if not chunk:
                    break
                self._sink.append(chunk)
        except (OSError, ValueError):
            pass
        finally:
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass


def main(argv=None):
    """Entry point for ``mactf-agent.service``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    port = os.environ.get("MACTF_AGENT_PORT", DEFAULT_PORT)
    workdir = os.environ.get("MACTF_AGENT_WORKDIR", "/app")
    allowed = os.environ.get("MACTF_AGENT_PATHS")
    allowed_paths = tuple(allowed.split(os.pathsep)) if allowed else DEFAULT_ALLOWED_PATHS
    if len(argv) >= 2 and argv[0] == "--port":
        port = argv[1]
    agent = GuestAgent(port, workdir=workdir, allowed_paths=allowed_paths)
    agent.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
