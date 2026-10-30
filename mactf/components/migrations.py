"""State migrations for a component swap (document 17 section 8.8)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

import torch

from ..exceptions import ComponentMigrationError

log = logging.getLogger("mactf.components")

Migrate = Callable[[Any, Any], None]

MIGRATE_MODES: tuple[str, ...] = ("reset", "carry", "fail")

#: Defaults by family, the last column of the 8.8.1 table. A family not listed here uses "reset".
FAMILY_DEFAULTS: Mapping[str, str] = {
    "loss": "carry",
    "regularization": "carry",
    "gradient": "carry",
    "metric": "carry",
    "value": "fail",
    "synthesis": "fail",
}


def _parameters(instance: Any) -> list[torch.nn.Parameter]:
    getter = getattr(instance, "parameters", None)
    if not callable(getter):
        return []
    return [parameter for parameter in getter() if isinstance(parameter, torch.nn.Parameter)]


def _state(instance: Any) -> Any:
    getter = getattr(instance, "state_dict", None)
    if not callable(getter):
        return None
    return getter()


def is_stateful(instance: Any) -> bool:
    """True if discarding ``instance`` would lose something a run produced.

    An instance is stateful if it owns trainable parameters, or if it defines both
    ``state_dict`` and ``load_state_dict``: a class that can load state has declared that it
    has some. ``AccumulateGradient``, the learned value functions and a learned synthesizer
    qualify; ``ClipGradient`` and the other stateless built-ins do not.
    """
    if _parameters(instance):
        return True
    return callable(getattr(instance, "state_dict", None)) and callable(
        getattr(instance, "load_state_dict", None)
    )


def reset(old: Any, new: Any) -> None:
    """Discard the old state. The new instance starts from its own constructor state.

    Calls ``new.reset()`` when it exists, so an instance that was built eagerly and then
    touched (a probe, a dry run) is clean at the moment it takes over.
    """
    if is_stateful(old):
        log.warning(
            "swap discards the state of %s (reset); use carry or fail to keep or refuse",
            type(old).__name__,
        )
    clear = getattr(new, "reset", None)
    if callable(clear):
        clear()


def carry_parameters(old: Any, new: Any) -> None:
    """Copy trainable parameters from ``old`` to ``new`` by position.

    Both must expose the same number of parameters with equal shapes; anything else is a
    ``ComponentMigrationError``, never a partial copy. Used for value networks and learned
    synthesizers whose replacement has the same architecture and a different rule.
    """
    source, target = _parameters(old), _parameters(new)
    if not source:
        return
    if len(source) != len(target):
        raise ComponentMigrationError(
            f"cannot carry parameters: {type(old).__name__} has {len(source)}, "
            f"{type(new).__name__} has {len(target)}"
        )
    for index, (left, right) in enumerate(zip(source, target)):
        if tuple(left.shape) != tuple(right.shape):
            raise ComponentMigrationError(
                f"cannot carry parameters: parameter {index} is {tuple(left.shape)} in "
                f"{type(old).__name__} and {tuple(right.shape)} in {type(new).__name__}"
            )
    with torch.no_grad():
        for left, right in zip(source, target):
            right.copy_(left.detach().to(right.device, right.dtype))


def carry_state(old: Any, new: Any) -> None:
    """Load ``old.state_dict()`` into ``new``; copy parameters when there is no state to load.

    The old instance being stateless is fine (nothing to carry). The old instance being
    stateful and the new one unable to load it is not: that is exactly the silent loss the
    mode exists to prevent, so it raises.
    """
    if not is_stateful(old):
        return
    payload = _state(old)
    loader = getattr(new, "load_state_dict", None)
    if isinstance(payload, Mapping) and payload and callable(loader):
        try:
            loader(dict(payload))
        except Exception as error:  # noqa: BLE001 - surfaced as a migration failure
            raise ComponentMigrationError(
                f"{type(new).__name__}.load_state_dict rejected the state of "
                f"{type(old).__name__}: {error}"
            ) from error
        return
    if _parameters(old):
        carry_parameters(old, new)
        return
    raise ComponentMigrationError(
        f"{type(old).__name__} carries state but {type(new).__name__} has no load_state_dict; "
        "choose migrate: reset to discard it explicitly"
    )


def fail_if_stateful(old: Any, new: Any) -> None:
    """Refuse the swap when the old instance holds state."""
    if is_stateful(old):
        raise ComponentMigrationError(
            f"{type(old).__name__} holds state and the swap was requested with migrate: fail; "
            "use migrate: carry or migrate: reset to proceed"
        )


_MODES: Mapping[str, Migrate] = {
    "reset": reset,
    "carry": carry_state,
    "fail": fail_if_stateful,
}


def migration_for(mode: str | None, *, family: str = "", override: Mapping[str, str] | None = None) -> Migrate:
    """Pick the migration. Order: the request's ``mode``, ``RL.MIGRATE[family]``, the family default."""
    chosen = mode or (dict(override or {}).get(family)) or FAMILY_DEFAULTS.get(family, "reset")
    chosen = str(chosen).lower()
    if chosen not in _MODES:
        raise ComponentMigrationError(
            f"unknown migration {chosen!r}; accepted: {list(MIGRATE_MODES)}"
        )
    return _MODES[chosen]


def state_digest(instance: Any) -> str:
    """A short stable digest of ``instance``'s state, for swap records and tests."""
    import hashlib
    import json

    state = _state(instance)
    parameters = [parameter.detach().cpu().flatten().tolist() for parameter in _parameters(instance)]
    blob = json.dumps({"state": state, "parameters": parameters}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
