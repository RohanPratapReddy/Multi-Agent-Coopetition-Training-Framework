"""Cross-process run locks: the only liveness signal that survives a crashed process.

Document 16, section 7.3. A run holds ``<PATHS.STATE>/locks/<run_id>.lock`` for as long as a
process drives it. The file is created with ``O_CREAT | O_EXCL`` (atomic everywhere the
framework runs), refreshed by a daemon heartbeat, and judged live by three tests: it parses,
its process exists, and its heartbeat is fresh. This module imports only the standard
library and ``..exceptions``/``.recovery``'s validator, so it is safe on the torch-free CLI
fast path.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..exceptions import StateConflict

__all__ = [
    "HEARTBEAT_SECONDS",
    "STALE_AFTER_SECONDS",
    "UNPARSEABLE_GRACE_SECONDS",
    "LockInfo",
    "RunLock",
    "lock_directory",
    "pid_alive",
]

HEARTBEAT_SECONDS = 10.0
STALE_AFTER_SECONDS = 120.0
UNPARSEABLE_GRACE_SECONDS = 5.0

_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


@dataclass(frozen=True)
class LockInfo:
    """Who holds a run's lock and since when."""

    run_id: str
    pid: int
    host: str
    owner: str            # "cli" or "api"
    started_at: float
    heartbeat_at: float   # the lock file's mtime
    token: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pid": self.pid,
            "host": self.host,
            "owner": self.owner,
            "started_at": self.started_at,
            "heartbeat_at": self.heartbeat_at,
            "token": self.token,
        }


def lock_directory(settings: Any) -> Path:
    """``PATHS.STATE``/locks (created on demand by ``acquire``, never by ``holder``)."""
    return Path(settings.path("STATE")) / "locks"


def pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` exists (even one this user may not signal)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED   # exists, not ours
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class RunLock:
    """The exclusive right to drive one run, as a lock file with a heartbeat."""

    def __init__(
        self,
        directory: Path,
        run_id: str,
        *,
        owner: str = "cli",
        heartbeat: float = HEARTBEAT_SECONDS,
        stale_after: float = STALE_AFTER_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        from .recovery import validate_run_id

        self.run_id = validate_run_id(run_id)
        self.directory = Path(directory)
        self.owner = owner
        self.heartbeat = float(heartbeat)
        self.stale_after = float(stale_after)
        self.clock = clock
        self.token = uuid.uuid4().hex
        self.path = self.directory / f"{self.run_id}.lock"
        self._lost = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._held = False

    # -- inspection ---------------------------------------------------------------------
    @staticmethod
    def holder(
        directory: Path,
        run_id: str,
        *,
        stale_after: float = STALE_AFTER_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> LockInfo | None:
        """The live holder of ``run_id``'s lock, or ``None`` when nobody is home."""
        from .recovery import validate_run_id

        path = Path(directory) / f"{validate_run_id(run_id)}.lock"
        try:
            stat = path.stat()
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("lock content is not an object")
            info = LockInfo(
                run_id=str(data.get("run_id", run_id)),
                pid=int(data["pid"]),
                host=str(data["host"]),
                owner=str(data.get("owner", "cli")),
                started_at=float(data.get("started_at", stat.st_mtime)),
                heartbeat_at=stat.st_mtime,
                token=str(data.get("token", "")),
            )
        except (ValueError, KeyError, TypeError):
            # ``acquire`` creates the file before it writes it: young and unparseable means
            # "being written right now"; older than the grace period means abandoned.
            if clock() - stat.st_mtime <= UNPARSEABLE_GRACE_SECONDS:
                return LockInfo(run_id, 0, socket.gethostname(), "unknown", stat.st_mtime,
                                stat.st_mtime, "")
            return None
        if info.host == socket.gethostname() and not pid_alive(info.pid):
            return None
        if clock() - info.heartbeat_at > stale_after:
            return None
        return info

    @property
    def lost(self) -> bool:
        """True once another process took the lock over (the heartbeat noticed)."""
        return self._lost.is_set()

    # -- acquire / release --------------------------------------------------------------
    def acquire(self) -> "RunLock":
        self.directory.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = self.holder(self.directory, self.run_id, stale_after=self.stale_after,
                                   clock=self.clock)
                if info is not None:
                    raise StateConflict(
                        f"run {self.run_id} is being driven by pid {info.pid} on {info.host} "
                        f"({info.owner}) since {_iso(info.started_at)}",
                        run_id=self.run_id, pid=info.pid, host=info.host, owner=info.owner,
                    ) from None
                graveyard = self.path.with_name(
                    f"{self.path.name}.stale-{os.getpid()}-{uuid.uuid4().hex[:8]}")
                try:
                    os.replace(self.path, graveyard)        # one contender wins this rename
                except FileNotFoundError:
                    continue                                # someone cleared it; retry the create
                except OSError as exc:
                    raise StateConflict(
                        f"cannot take over the stale lock for run {self.run_id}: {exc}",
                        run_id=self.run_id) from None
                with contextlib.suppress(OSError):
                    graveyard.unlink()
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"run_id": self.run_id, "pid": os.getpid(), "host": socket.gethostname(),
                           "owner": self.owner, "started_at": self.clock(), "token": self.token},
                          handle)
                handle.flush()
                os.fsync(handle.fileno())
            self._held = True
            self._lost.clear()
            self._start_heartbeat()
            return self
        raise StateConflict(f"could not acquire the lock for run {self.run_id}; try again",
                            run_id=self.run_id)

    def release(self) -> None:
        """Stop the heartbeat; unlink the file when it still carries our token. Never raises."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if not self._held:
            return
        self._held = False
        with contextlib.suppress(Exception):
            if self._read_token() == self.token:
                self.path.unlink()

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    # -- heartbeat ----------------------------------------------------------------------
    def _read_token(self) -> str | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return str(data.get("token")) if isinstance(data, dict) else None

    def _start_heartbeat(self) -> None:
        self._stop = threading.Event()
        thread = threading.Thread(target=self._beat, args=(self._stop,),
                                  name=f"runlock-{self.run_id}", daemon=True)
        self._thread = thread
        thread.start()

    def _beat(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat):
            if self._read_token() != self.token:
                self._lost.set()
                return
            try:
                os.utime(self.path)
            except OSError:
                self._lost.set()
                return
