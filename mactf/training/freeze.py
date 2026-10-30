"""The freeze set: which peers are frozen, and why (document 17 section 11.4).

Phase P2 uses it for the ``optimizer:null`` reason; the strategy and mutation reasons arrive with
their own phases and need no change here.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..exceptions import ImproperlyConfigured

log = logging.getLogger("mactf.freeze")

FREEZE_MODES = ("legacy", "strict")
RETIRED = "mutation:retired"
_KNOWN_PREFIXES = ("strategy:", "optimizer:", "mutation:")


def _check_reason(reason: str) -> str:
    if not isinstance(reason, str) or not reason.startswith(_KNOWN_PREFIXES) or reason.endswith(":"):
        raise ImproperlyConfigured(
            f"E075: freeze reason {reason!r} must look like one of "
            f"{[prefix + '<name>' for prefix in _KNOWN_PREFIXES]}"
        )
    return reason


class FreezeSet:
    """Which peers are frozen, and why. The training loop and ``_write`` read it."""

    def __init__(self, mode: str = "legacy") -> None:
        self._mode = "legacy"
        self.set_mode(mode)
        self._tree: Any = None
        self._by_path: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    # -- configuration ---------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        text = str(mode).lower()
        if text not in FREEZE_MODES:
            raise ImproperlyConfigured(
                f"E070: TRAINER.FREEZE_MODE must be one of {list(FREEZE_MODES)}, got {mode!r}"
            )
        self._mode = text

    def bind(self, tree: Any) -> None:
        """Attach the tree whose node statuses this set mirrors."""
        self._tree = tree

    # -- reading ---------------------------------------------------------
    def reasons(self, path: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._by_path.get(path, ())))

    def paths(self, reason: str | None = None) -> list[str]:
        with self._lock:
            return sorted(
                path for path, held in self._by_path.items() if reason is None or reason in held
            )

    def __contains__(self, path: object) -> bool:
        return path in self._by_path

    def __len__(self) -> int:
        return len(self._by_path)

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths())

    def _counts(self, path: str, *, legacy: tuple[str, ...]) -> bool:
        held = self._by_path.get(path)
        if not held:
            return False
        if self._mode == "strict":
            return True
        return any(reason.startswith(legacy) for reason in held)

    def excludes(self, path: str) -> bool:
        """Weight 0 in the group's report."""
        return self._counts(path, legacy=("strategy:", "mutation:"))

    def skips_training(self, path: str) -> bool:
        return self._counts(path, legacy=(RETIRED,))

    def skips_write(self, path: str) -> bool:
        return self._counts(path, legacy=(RETIRED,))

    def eligible_for(self, members: Sequence[str]) -> dict[str, bool] | None:
        """The eligibility mapping a weight update takes, or ``None`` when nobody is excluded.

        ``None`` (not an all-true mapping) keeps the call identical to the one made when no
        freeze exists, which is what the default-identity invariant needs.
        """
        with self._lock:
            excluded = {path for path in members if self.excludes(path)}
        if not excluded:
            return None
        return {path: path not in excluded for path in members}

    def learners(self, paths: Iterable[str]) -> list[str]:
        with self._lock:
            return [path for path in paths if not self.skips_training(path)]

    def check_learners(self, paths: Iterable[str]) -> None:
        """``E075`` when no node would take a training step."""
        everyone = list(paths)
        if everyone and not self.learners(everyone):
            raise ImproperlyConfigured(
                f"E075: every node ({len(everyone)}) is frozen under TRAINER.FREEZE_MODE={self._mode}, "
                f"so nothing could learn; reasons: "
                f"{ {path: list(self.reasons(path)) for path in sorted(everyone)} }"
            )

    # -- changing --------------------------------------------------------
    def _known(self, path: str) -> bool:
        return self._tree is None or path in self._tree

    def add(self, path: str, reason: str) -> bool:
        """Freeze ``path`` for ``reason``. ``True`` when the peer was not frozen before."""
        _check_reason(reason)
        with self._lock:
            if not self._known(path):
                raise ImproperlyConfigured(f"E075: cannot freeze {path!r} ({reason}): not a node in the tree")
            held = self._by_path.setdefault(path, set())
            fresh = not held
            held.add(reason)
            self._sync(path)
            return fresh

    def discard(self, path: str, reason: str) -> bool:
        """Remove one reason. ``True`` when the peer is no longer frozen by anything."""
        with self._lock:
            held = self._by_path.get(path)
            if not held or reason not in held:
                return False
            held.discard(reason)
            if not held:
                del self._by_path[path]
            self._sync(path)
            return path not in self._by_path

    def clear_reason(self, reason: str) -> list[str]:
        """Remove ``reason`` from every peer; the paths that became thawed."""
        with self._lock:
            return [path for path in self.paths(reason) if self.discard(path, reason)]

    def forget(self, path: str) -> None:
        """The node left the tree: drop every reason without touching a status."""
        with self._lock:
            self._by_path.pop(path, None)

    # -- mutation (document 17 section 12.5.2) ----------------------------
    def is_retired(self, path: str) -> bool:
        with self._lock:
            return RETIRED in self._by_path.get(path, ())

    def retired(self) -> frozenset[str]:
        with self._lock:
            return frozenset(path for path, held in self._by_path.items() if RETIRED in held)

    def snapshot(self) -> dict[str, frozenset[str]]:
        with self._lock:
            return {path: frozenset(held) for path, held in self._by_path.items()}

    def restore(self, snapshot: Mapping[str, frozenset[str]]) -> None:
        """Put the reasons back exactly. Statuses follow the (separately restored) tree."""
        with self._lock:
            self._by_path = {path: set(held) for path, held in snapshot.items()}

    def _sync(self, path: str) -> None:
        node = self._tree.get(path) if self._tree is not None else None
        if node is None:
            return
        held = self._by_path.get(path)
        if held:
            if node.status in ("active", "frozen", "pruned"):
                self._tree.mark(path, "pruned" if RETIRED in held else "frozen")
        elif node.status in ("frozen", "pruned"):
            self._tree.mark(path, "active")

    def sync(self) -> None:
        """Re-mirror every status. ``run_epoch`` resets ``failed`` to ``active`` at its top and
        calls this straight after, so a frozen peer that failed stays frozen."""
        with self._lock:
            for path in list(self._by_path):
                self._sync(path)

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "mode": self._mode,
                "entries": {path: sorted(held) for path, held in sorted(self._by_path.items())},
            }

    def load_state_dict(self, payload: Mapping[str, Any]) -> list[str]:
        """Restore the part of the set that cannot be re-derived: ``mutation:*`` reasons.

        ``strategy:*`` reasons are re-added by the strategy's ``on_epoch_start`` and
        ``optimizer:*`` reasons by the optimizer plan, from the configuration this run was
        built with, so a saved copy of either would only fight a changed configuration.
        Paths that are not in the tree are dropped with ``W036``. The saved *mode* is not
        applied: configuration wins, and a difference is reported (``W034``)."""
        notes: list[str] = []
        entries = dict(payload.get("entries") or {})
        with self._lock:
            for path in list(self._by_path):
                for reason in [item for item in self.reasons(path) if item.startswith("mutation:")]:
                    self.discard(path, reason)
            for path, reasons in sorted(entries.items()):
                kept = [str(item) for item in reasons if str(item).startswith("mutation:")]
                if not kept:
                    continue
                if not self._known(path):
                    notes.append(f"W036: saved freeze entry {path!r} {kept} has no node; dropped")
                    continue
                for reason in kept:
                    self.add(path, reason)
        saved_mode = str(payload.get("mode", self._mode))
        if saved_mode != self._mode:
            notes.append(
                f"W034: the saved run used TRAINER.FREEZE_MODE={saved_mode!r}; this run uses {self._mode!r}"
            )
        for note in notes:
            log.warning(note)
        return notes
