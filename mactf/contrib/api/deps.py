"""Application state and the dependency accessors views use (document 02 section 1.3).

Resolution order: ``request.app_state`` (set by the handler), then a process-level
fallback keyed by the current settings object. Everything is built lazily, once, so
importing the app or constructing a ``WSGIHandler`` never opens a database.
"""

from __future__ import annotations

import collections
import hashlib
import json
import threading
import time
from typing import Any

from ...conf import Settings, get_settings
from ...db import Registry, ensure_registry
from ...http.request import Request
from ...orchestration.live import EventBus, RunManager

_LOCK = threading.RLock()
_FALLBACK: dict[int, dict[str, Any]] = {}


class IdempotencyCache:
    """An LRU of ``(principal, key) -> response`` with a TTL (``POST /runs`` only)."""

    def __init__(self, capacity: int = 256, ttl: float = 86400.0) -> None:
        self.capacity = int(capacity)
        self.ttl = float(ttl)
        self._items: collections.OrderedDict[tuple[str, str], dict[str, Any]] = collections.OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def fingerprint(body: Any) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def get(self, principal: str, key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            entry = self._items.get((principal, key))
            if entry is None:
                return None
            if now - entry["stored_at"] > self.ttl:
                del self._items[(principal, key)]
                return None
            self._items.move_to_end((principal, key))
            return entry

    def put(self, principal: str, key: str, *, fingerprint: str, status: int, body: Any, headers: dict[str, str]) -> None:
        with self._lock:
            self._items[(principal, key)] = {
                "stored_at": time.time(),
                "fingerprint": fingerprint,
                "status": status,
                "body": body,
                "headers": dict(headers),
            }
            self._items.move_to_end((principal, key))
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


def app_state(request: Request | None = None) -> dict[str, Any]:
    """The state dict for this request: the handler's, else a per-settings fallback."""
    state = getattr(request, "app_state", None) if request is not None else None
    if isinstance(state, dict):
        state.setdefault("settings", get_settings())
        return state
    settings = get_settings()
    with _LOCK:
        entry = _FALLBACK.get(id(settings))
        if entry is None or entry.get("settings") is not settings:
            entry = _FALLBACK[id(settings)] = {"settings": settings, "started_at": time.time()}
        return entry


def clear_fallback() -> None:
    with _LOCK:
        for entry in _FALLBACK.values():
            _close(entry)
        _FALLBACK.clear()


def _close(state: dict[str, Any]) -> None:
    manager = state.get("run_manager")
    if manager is not None:
        try:
            manager.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass


def _ensure(state: dict[str, Any], key: str, factory) -> Any:
    value = state.get(key)
    if value is None:
        with _LOCK:
            value = state.get(key)
            if value is None:
                value = state[key] = factory()
    return value


def get_settings_dep(request: Request | None = None) -> Settings:
    return app_state(request)["settings"]


def get_registry_dep(request: Request | None = None) -> Registry:
    state = app_state(request)
    return _ensure(state, "registry", lambda: ensure_registry(state["settings"]))


def get_events_dep(request: Request | None = None) -> EventBus:
    state = app_state(request)
    return _ensure(state, "events", lambda: EventBus.from_settings(state["settings"]))


def get_scorer_dep(request: Request | None = None):
    """The app-scoped ``ScoringPipeline``: warning registries persist across requests."""
    from ...scoring.pipeline import build_scorer

    state = app_state(request)
    return _ensure(state, "scorer", lambda: build_scorer(state["settings"]))


def get_run_manager_dep(request: Request | None = None) -> RunManager:
    state = app_state(request)

    def build() -> RunManager:
        manager = RunManager(
            state["settings"],
            registry=get_registry_dep(request),
            bus=get_events_dep(request),
        )
        manager.reap_orphans()  # rows an earlier process left running are "interrupted"
        return manager

    return _ensure(state, "run_manager", build)


def get_idempotency_dep(request: Request | None = None) -> IdempotencyCache:
    state = app_state(request)
    api = state["settings"].get("API") or {}
    return _ensure(
        state,
        "idempotency",
        lambda: IdempotencyCache(256, float(api.get("IDEMPOTENCY_TTL_SECONDS", 86400))),
    )


def sandbox_handles(request: Request | None = None) -> dict[str, tuple[Any, Any]]:
    """``peer -> (provider, handle)`` for sandboxes a run provisioned and registered."""
    return _ensure(app_state(request), "sandbox_handles", dict)


def register_sandbox_handle(state: dict[str, Any], peer: str, provider: Any, handle: Any) -> None:
    """Called by whatever provisions a sandbox so ``POST /sandbox/exec`` can reach it."""
    _ensure(state, "sandbox_handles", dict)[peer] = (provider, handle)


def api_option(request: Request | None, key: str, default: Any = None) -> Any:
    return (get_settings_dep(request).get("API") or {}).get(key, default)
