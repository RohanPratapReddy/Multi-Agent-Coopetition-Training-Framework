"""State migration for an optimizer swap (document 17 section 9.6)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

import torch

from ..exceptions import ComponentMigrationError
from . import schedulers

log = logging.getLogger("mactf.training.optimizer_migrations")

Migrate = Callable[[Any, Any], None]


def _native(backend: Any) -> torch.optim.Optimizer | None:
    return getattr(backend, "_instance", None)


def _signature(state: Mapping[str, Any] | None) -> frozenset[str]:
    """The set of per-parameter state keys of a torch ``state_dict()['state']``."""
    keys: set[str] = set()
    for entry in dict((state or {}).get("state") or {}).values():
        if isinstance(entry, Mapping):
            keys.update(str(key) for key in entry)
    return frozenset(keys)


def _backend_holds_state(state: Any) -> bool:
    """Whether a backend ``state_dict`` carries something a run produced.

    A torch state holds something once any parameter has stepped. A router's is a mapping of
    its members' states and holds something when any member does. Any other backend (scipy's
    iterate vector) holds something whenever its dict is non-empty.
    """
    if not isinstance(state, Mapping) or not state:
        return False
    if "state" in state and "param_groups" in state:
        return bool(_signature(state))
    nested = [value for value in state.values() if isinstance(value, Mapping)]
    if nested and len(nested) == len(state):
        return any(_backend_holds_state(value) for value in nested)
    return True


def has_state(optimizer: Any) -> bool:
    """True if discarding ``optimizer`` would lose something a run produced."""
    backend = optimizer.backend
    if getattr(backend, "name", "") == "null":
        return False
    if _backend_holds_state(backend.state_dict()):
        return True
    scheduler = optimizer.scheduler
    position = schedulers.position_of(scheduler) if scheduler is not None else 0
    return bool(position) or bool(getattr(optimizer, "_accumulated", 0))


def hyper_parameters(backend: Any) -> list[dict[str, Any]]:
    """The non-parameter entries of every param group, in order."""
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in backend.param_groups
    ]


def apply_hyper_parameters(backend: Any, hyper: list[dict[str, Any]]) -> None:
    """Put ``hyper`` back over the groups, after a ``load_state_dict`` copied the old ones in."""
    groups = backend.param_groups
    if len(groups) != len(hyper):
        raise ComponentMigrationError(
            f"cannot re-apply hyper-parameters: the optimizer has {len(groups)} groups, "
            f"the new spec {len(hyper)}"
        )
    for group, values in zip(groups, hyper):
        group.update(values)


def compatible_state(saved: Any, backend: Any, saved_class: Any = None) -> str | None:
    """``None`` when ``saved`` can be loaded into ``backend``, else why not.

    torch's ``load_state_dict`` checks only the number and size of parameter groups; it
    does not check which state keys exist, so Adam state loaded into SGD passes and fails at
    the next ``step()``. This is the check it lacks.
    """
    if not isinstance(saved, Mapping) or not saved:
        return None
    native = _native(backend)
    if native is None:
        if getattr(backend, "name", "") == "router":
            return None
        return f"{type(backend).__name__} cannot restore another optimizer's state"
    if "state" not in saved:
        return "the saved state is not a torch optimizer state"
    here = native.state_dict()
    if len(saved.get("param_groups", [])) != len(here["param_groups"]):
        return (
            f"parameter-group count differs ({len(saved.get('param_groups', []))} saved, "
            f"{len(here['param_groups'])} now)"
        )
    saved_groups, here_groups = saved.get("param_groups") or [], here["param_groups"]
    if saved_groups and here_groups:
        # An optimizer that has not stepped has no state keys to compare, but its group keys
        # name its algorithm: Adam and AdamW share them, Adam and SGD or Lion do not.
        # ``initial_lr`` is written into the groups by a torch scheduler, not by the algorithm.
        saved_keys = set(saved_groups[0]) - {"params", "initial_lr"}
        here_keys = set(here_groups[0]) - {"params", "initial_lr"}
        if saved_keys != here_keys:
            return (
                f"hyper-parameter sets differ ({sorted(saved_keys ^ here_keys)} are in only one "
                f"of {saved_class or 'the saved optimizer'} and {type(native).__name__})"
            )
    wanted, found = _signature(saved), _signature(here)
    if wanted and found and wanted != found:
        return f"state keys differ ({sorted(wanted)} saved, {sorted(found)} expected)"
    return None


def reset(old: Any, new: Any) -> None:
    """The new facade starts clean. Partial gradients of the old window are dropped."""
    if has_state(old):
        log.warning(
            "optimizer swap on %r discards the state of %s (reset); use carry or fail to keep or refuse",
            getattr(old, "node_path", ""), old.describe().get("class"),
        )
    new.zero_grad()


def carry(old: Any, new: Any) -> None:
    """Move state from ``old`` to ``new``; raise rather than lose any silently."""
    if not has_state(old):
        return
    reason = compatible_state(old.backend.state_dict(), new.backend, old.describe().get("class"))
    if reason is not None:
        raise ComponentMigrationError(
            f"cannot carry optimizer state from {old.describe().get('class')} to "
            f"{new.describe().get('class')}: {reason}; choose migrate: reset to discard it explicitly"
        )
    hyper = hyper_parameters(new.backend)
    payload = old.backend.state_dict()
    if payload:
        try:
            new.backend.load_state_dict(payload)
        except Exception as error:  # noqa: BLE001 - surfaced as a migration failure
            raise ComponentMigrationError(
                f"{type(new.backend).__name__}.load_state_dict rejected the old state: {error}"
            ) from error
        apply_hyper_parameters(new.backend, hyper)
    position = schedulers.position_of(old.scheduler) if old.scheduler is not None else 0
    if position and new.scheduler is not None:
        try:
            schedulers.fast_forward(new.scheduler, int(position))
        except ValueError as error:
            raise ComponentMigrationError(str(error)) from error
    elif position and new.scheduler is None:
        log.warning(
            "W014 swap on %r carried optimizer state but the new entry has no scheduler; its "
            "position %s is dropped", getattr(old, "node_path", ""), position,
        )
    if old.grad_accumulation == new.grad_accumulation:
        new._accumulated = old._accumulated
    else:
        new.zero_grad()
        log.warning(
            "W014 swap on %r changed GRAD_ACCUMULATION %s -> %s; the open window was dropped",
            getattr(old, "node_path", ""), old.grad_accumulation, new.grad_accumulation,
        )


def fail(old: Any, new: Any) -> None:
    """Refuse when the old facade holds anything worth keeping."""
    if has_state(old):
        raise ComponentMigrationError(
            f"optimizer {old.describe().get('class')} on {getattr(old, 'node_path', '')!r} holds "
            "state and the swap was requested with migrate: fail; use migrate: carry or "
            "migrate: reset to proceed"
        )


_MODES: Mapping[str, Migrate] = {"reset": reset, "carry": carry, "fail": fail}


def migration_for(mode: str | None) -> Migrate:
    chosen = str(mode or "reset").lower()
    if chosen not in _MODES:
        raise ComponentMigrationError(f"unknown migration {chosen!r}; accepted: {list(_MODES)}")
    return _MODES[chosen]
