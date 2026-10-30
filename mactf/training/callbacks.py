"""Callbacks: the class-based form of the trainer's hooks (document 08 section 7).

A hook is a function in a mapping; a callback is an object with a stable ``name``
so it can be added, removed, and restored from a checkpoint. Both are supported and
they share the same invocation points, so a user can switch between them without
learning a second lifecycle.

A callback returning ``None`` leaves the value under consideration unchanged, the
same convention :meth:`mactf.signals.Signal.send` uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..exceptions import ValidationError

__all__ = [
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "EpochTimer",
    "GradientNormMonitor",
    "LossHistoryCallback",
    "ProgressCallback",
    "build_callbacks",
    "HOOK_NAMES",
]

#: Every hook name the trainer dispatches. Kept here so the trainer, the strict
#: hook validation, and the API's capability listing cannot drift apart.
HOOK_NAMES: tuple[str, ...] = (
    "on_build",
    "on_epoch_start",
    "on_batch",
    "on_step",
    "on_score",
    "on_aggregate",
    "on_epoch_end",
    "on_resume",
    "on_close",
)


class Callback:
    """Base class. Every method is optional; the default does nothing."""

    name: str = "callback"

    def on_build(self, trainer: Any) -> None:
        return None

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        return None

    def on_batch(self, trainer: Any, node: Any, batch: Any) -> Any:
        """Return a batch to replace the one about to be used, or ``None``."""
        return None

    def on_step(self, trainer: Any, node: Any, result: Any) -> Any:
        """Return a :class:`StepResult` to replace the one produced, or ``None``."""
        return None

    def on_score(self, trainer: Any, scorecard: Any) -> None:
        return None

    def on_aggregate(self, trainer: Any, report: Any) -> None:
        return None

    def on_epoch_end(self, trainer: Any, result: Any) -> None:
        return None

    def on_resume(self, trainer: Any, report: Any) -> None:
        return None

    def on_close(self, trainer: Any) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("class") not in (None, type(self).__name__):
            raise ValidationError(
                f"{type(self).__name__} cannot load state written by "
                f"{payload.get('class')!r}"
            )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class CallbackList:
    """An ordered, name-addressable collection of callbacks.

    Bound to a trainer at construction so ``add``/``remove`` can immediately give
    the new callback the current context instead of waiting an epoch.
    """

    def __init__(
        self,
        callbacks: Sequence[Callback] | None = None,
        *,
        trainer: Any = None,
    ) -> None:
        self.trainer = trainer
        self._callbacks: list[Callback] = []
        for callback in callbacks or ():
            self.add(callback, notify=False)

    # -- membership ------------------------------------------------------
    def add(self, callback: Callback, *, notify: bool = True) -> "CallbackList":
        if not isinstance(callback, Callback):
            raise ValidationError(
                f"a callback must be a Callback subclass, got {type(callback).__name__}"
            )
        if any(existing.name == callback.name for existing in self._callbacks):
            raise ValidationError(
                f"a callback named {callback.name!r} is already registered; remove it first"
            )
        self._callbacks.append(callback)
        if notify and self.trainer is not None and getattr(self.trainer, "_built", False):
            callback.on_build(self.trainer)
        return self

    def remove(self, name: str) -> bool:
        """Remove by name, returning whether anything was removed."""
        for index, callback in enumerate(self._callbacks):
            if callback.name == name:
                del self._callbacks[index]
                return True
        return False

    def get(self, name: str) -> Callback:
        for callback in self._callbacks:
            if callback.name == name:
                return callback
        raise KeyError(f"no callback named {name!r}")

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self):
        return iter(self._callbacks)

    def __contains__(self, name: object) -> bool:
        return any(callback.name == name for callback in self._callbacks)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(callback.name for callback in self._callbacks)

    # -- dispatch --------------------------------------------------------
    def dispatch(self, hook: str, *args: Any) -> list[Any]:
        """Call ``hook`` on every callback, collecting non-``None`` returns.

        A callback that raises propagates: unlike a signal receiver, a callback is
        user code the user attached to *this* run and expects to run. Swallowing it
        would make a broken callback look like a healthy epoch.
        """
        if hook not in HOOK_NAMES:
            raise ValidationError(
                f"unknown callback hook {hook!r}; known hooks: " + ", ".join(HOOK_NAMES)
            )
        results: list[Any] = []
        for callback in list(self._callbacks):
            method = getattr(callback, hook, None)
            if method is None:
                continue
            value = method(*args)
            if value is not None:
                results.append(value)
        return results

    def state_dict(self) -> list[dict[str, Any]]:
        return [callback.state_dict() for callback in self._callbacks]

    def load_state_dict(self, payload: Iterable[Mapping[str, Any]]) -> None:
        by_name = {str(item.get("name")): item for item in payload or ()}
        for callback in self._callbacks:
            state = by_name.get(callback.name)
            if state:
                callback.load_state_dict(state)

    def describe(self) -> list[dict[str, Any]]:
        return [callback.describe() for callback in self._callbacks]


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------
class LossHistoryCallback(Callback):
    """Records each epoch's mean loss and the trailing median."""

    name = "loss_history"

    def __init__(self) -> None:
        self.history: list[float] = []
        self.medians: list[float] = []

    def on_epoch_end(self, trainer: Any, result: Any) -> None:
        value = float(result.mean_loss)
        self.history.append(value)
        window = sorted(self.history[-5:])
        middle = len(window) // 2
        median = (
            window[middle]
            if len(window) % 2
            else (window[middle - 1] + window[middle]) / 2.0
        )
        self.medians.append(median)

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload.update({"history": list(self.history), "medians": list(self.medians)})
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        self.history = [float(value) for value in payload.get("history", [])]
        self.medians = [float(value) for value in payload.get("medians", [])]


class EpochTimer(Callback):
    """Records wall-clock duration per epoch."""

    name = "epoch_timer"

    def __init__(self, *, clock: Any = time.monotonic) -> None:
        self.clock = clock
        self._started = 0.0
        self.durations: list[float] = []

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self._started = float(self.clock())

    def on_epoch_end(self, trainer: Any, result: Any) -> None:
        self.durations.append(float(self.clock()) - self._started)

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["durations"] = list(self.durations)
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        self.durations = [float(value) for value in payload.get("durations", [])]


@dataclass
class GradientNormMonitor(Callback):
    """Keeps the last ``window`` gradient norms and flags an outlier."""

    name: str = "gradient_norm"
    window: int = 32
    warn_above: float | None = None
    norms: list[float] = field(default_factory=list)
    flagged: int = 0

    def on_step(self, trainer: Any, node: Any, result: Any) -> None:
        norm = result.metrics.get("grad_norm") if getattr(result, "metrics", None) else None
        if norm is None:
            return None
        value = float(norm)
        self.norms.append(value)
        if len(self.norms) > self.window:
            self.norms.pop(0)
        if self.warn_above is not None and value > self.warn_above:
            self.flagged += 1
        return None

    def state_dict(self) -> dict[str, Any]:
        payload = {"name": self.name, "class": type(self).__name__}
        payload.update({"norms": list(self.norms), "flagged": self.flagged})
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("class") not in (None, type(self).__name__):
            raise ValidationError(
                f"{type(self).__name__} cannot load state written by "
                f"{payload.get('class')!r}"
            )
        self.norms = [float(value) for value in payload.get("norms", [])]
        self.flagged = int(payload.get("flagged", 0))


class EarlyStopping(Callback):
    """Sets ``trainer.stop_requested`` after ``patience`` non-improving epochs.

    It does not raise and does not patch the loop: the trainer checks the flag at
    the epoch boundary, which is the only place a partial epoch can be avoided
    (document 08 section 5.2).
    """

    name = "early_stopping"

    def __init__(
        self,
        *,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if patience < 1:
            raise ValidationError("EarlyStopping.patience must be >= 1")
        if mode not in ("min", "max"):
            raise ValidationError("EarlyStopping.mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def on_epoch_end(self, trainer: Any, result: Any) -> None:
        value = float(result.mean_loss)
        if self.best is None:
            self.best = value
            return
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            setter = getattr(trainer, "request_stop", None)
            if callable(setter):
                setter(f"early stopping: no improvement for {self.bad_epochs} epochs")
            else:
                setattr(trainer, "stop_requested", True)

    def state_dict(self) -> dict[str, Any]:
        payload = {"name": self.name, "class": type(self).__name__}
        payload.update(
            {
                "best": self.best,
                "bad_epochs": self.bad_epochs,
                "patience": self.patience,
                "mode": self.mode,
            }
        )
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("class") not in (None, type(self).__name__):
            raise ValidationError(
                f"{type(self).__name__} cannot load state written by "
                f"{payload.get('class')!r}"
            )
        self.best = payload.get("best")
        self.bad_epochs = int(payload.get("bad_epochs", 0))


class ProgressCallback(Callback):
    """Calls a user function with a one-line summary per epoch."""

    name = "progress"

    def __init__(self, sink: Any = print) -> None:
        if not callable(sink):
            raise ValidationError("ProgressCallback needs a callable sink")
        self.sink = sink

    def on_epoch_end(self, trainer: Any, result: Any) -> None:
        self.sink(
            f"epoch {result.epoch}: steps={result.steps} "
            f"loss={result.mean_loss:.6f}"
            + ("" if result.completed else f" ({result.reason})")
        )


#: Built-ins addressable by name for a settings- or spec-driven run.
CALLBACK_TYPES: dict[str, type] = {
    "loss_history": LossHistoryCallback,
    "epoch_timer": EpochTimer,
    "gradient_norm": GradientNormMonitor,
    "early_stopping": EarlyStopping,
    "progress": ProgressCallback,
}


def build_callbacks(
    spec: Sequence[Any] | Mapping[str, Any] | Callback | None,
    *,
    trainer: Any = None,
) -> CallbackList:
    """Resolve a callback specification into a :class:`CallbackList`.

    Accepts instances, ``{"type": name, **kwargs}`` mappings, and a bare mapping of
    ``name -> spec``. An unknown type raises with the accepted list.
    """
    if spec is None:
        return CallbackList(trainer=trainer)
    if isinstance(spec, CallbackList):
        spec.trainer = trainer
        return spec
    if isinstance(spec, Callback):
        return CallbackList([spec], trainer=trainer)
    entries: list[Any]
    if isinstance(spec, Mapping):
        if "type" in spec or "name" in spec:
            entries = [spec]
        else:
            entries = [
                {"type": key, **(dict(value) if isinstance(value, Mapping) else {})}
                for key, value in spec.items()
            ]
    else:
        entries = list(spec)

    resolved: list[Callback] = []
    for entry in entries:
        if isinstance(entry, Callback):
            resolved.append(entry)
            continue
        if isinstance(entry, str):
            entry = {"type": entry}
        if not isinstance(entry, Mapping):
            raise ValidationError(
                f"a callback spec must be a Callback, a string, or a mapping; got "
                f"{type(entry).__name__}"
            )
        payload = dict(entry)
        key = str(payload.pop("type", payload.pop("name", ""))).lower()
        target = CALLBACK_TYPES.get(key)
        if target is None:
            raise ValidationError(
                f"unknown callback type {key!r}; accepted: "
                + ", ".join(sorted(CALLBACK_TYPES))
            )
        try:
            resolved.append(target(**payload))
        except TypeError as error:
            raise ValidationError(
                f"callback {key!r} does not accept these arguments: {error}"
            ) from error
    return CallbackList(resolved, trainer=trainer)
