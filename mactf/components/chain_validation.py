"""Validate components and whole trainer specs without starting a run (document 17 section 16).

``probe_instance`` builds a throwaway instance of one spec and runs the author probe of its
kind; ``validate_trainer_spec`` resolves every component a ``TrainerSpec`` names and reports
*all* problems, not the first. Both return findings in the shape ``mactf components validate``
prints and the API's ``422`` body carries::

    {"code": "E004", "severity": "error", "message": "..."}

The module imports torch lazily: the probes need it, a caller that only validates names does not.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..exceptions import ComponentError, ImproperlyConfigured, MACTFError

_CODE = re.compile(r"\b([EWCF]\d{3})\b")


def finding(code: str, message: str, severity: str = "error", **details: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"code": str(code), "severity": severity, "message": str(message)}
    item.update(details)
    return item


def finding_of(error: BaseException, *, where: str = "") -> dict[str, Any]:
    """An exception as a finding. The code is the exception's, else the one in its message."""
    text = str(error)
    code = getattr(error, "code", None)
    if not code:
        found = _CODE.search(text)
        code = found.group(1) if found else ("E008" if not isinstance(error, ImproperlyConfigured) else "E005")
    return finding(code, f"{where}: {text}" if where else text)


def _scratch_aggregator() -> Any:
    from ..coopetition.aggregation import Aggregator
    from ..conf import DEFAULTS

    return Aggregator.from_settings(dict(DEFAULTS["COOPETITION"]), num_peers=3)


def _scratch_view() -> Any:
    from ..coopetition.strategy import tree_view
    from ..coopetition.tree import PeerTree

    return tree_view(PeerTree.flat(["a", "b", "c"]), hierarchy=False)


#: What the framework injects into a kind's factory, so a bare validation does not report it missing.
_OFFERED = {"optimizer": ("params",), "scheduler": ("optimizer",), "weight_update": ("aggregator",)}


def injected_offer(kind_id: str) -> dict[str, None]:
    """The ``inject=`` offer for ``ComponentResolver.validate`` of a kind built by a plan."""
    return {key: None for key in _OFFERED.get(kind_id, ())}


def probe_instance(resolver: Any, spec: Any) -> list[dict[str, Any]]:
    """Build one throwaway instance of ``spec`` and run its kind's probe. Empty means it behaves.

    A build failure is ``E008`` (the factory raised) or the component's own error; a probe
    failure carries the probe's code (``E047`` functions, ``E056`` optimizers, ``E064`` /
    ``E067`` weights, ``E074`` / ``E077`` strategies). Nothing is registered or installed.
    """
    kind_id = spec.kind_id
    found: list[dict[str, Any]] = []
    inject: dict[str, Any] = {}
    if kind_id in ("optimizer", "scheduler"):
        # Their factories need a parameter list / an optimizer that only the plan has, so the
        # spec is validated by name and parameters and then probed on a scratch module.
        try:
            resolver.validate(spec, trusted=True, inject=injected_offer(kind_id))
            found.extend(_probe_optimizer(spec) if kind_id == "optimizer" else _probe_scheduler(spec))
        except (ComponentError, ImproperlyConfigured, MACTFError) as error:
            found.append(finding_of(error))
        except Exception as error:  # noqa: BLE001
            found.append(finding("E008", f"{kind_id}:{spec.type}: probe raised {type(error).__name__}: {error}"))
        return found
    try:
        if kind_id == "weight_update":
            inject = {"aggregator": _scratch_aggregator()}
        resolved = resolver.resolve(spec, trusted=True, inject=inject or None)
    except ComponentError as error:
        return [finding_of(error)]
    except (ImproperlyConfigured, MACTFError, TypeError, ValueError) as error:
        return [finding_of(error)]
    instance = resolved.instance
    try:
        if kind_id.startswith("rl_function."):
            from .function_probes import verify_function

            family = kind_id.split(".", 1)[1]
            items = list(verify_function(family, instance, name=resolved.component.name))
            if items:
                # E047 is the code of the failure; the F-codes are what it lists (17.2.2).
                listed = "; ".join(item.render() if hasattr(item, "render") else str(item) for item in items)
                found.append(finding(
                    "E047", f"E047: {kind_id}:{resolved.component.name} failed its contract probe: {listed}",
                    probe=[item.code for item in items],
                ))
        elif kind_id == "weight_update":
            from ..coopetition.weight_probe import verify_weight_update

            aggregator = inject["aggregator"]
            verify_weight_update(
                lambda: resolver.resolve(spec, trusted=True, inject={"aggregator": _scratch_aggregator()}).instance,
                floor=aggregator.weight_floor, cap=aggregator.weight_cap, tolerance=1e-6,
                slot=f"weight_update:{resolved.component.name}", mode="repair",
            )
        elif kind_id == "peer_strategy":
            from ..coopetition.strategy_probe import verify_peer_strategy

            verify_peer_strategy(
                lambda: resolver.resolve(spec, trusted=True).instance,
                slot=f"peer_strategy:{resolved.component.name}", view=_scratch_view(),
            )
        elif kind_id == "optimizer":
            found.extend(_probe_optimizer(spec))
        elif kind_id == "scheduler":
            found.extend(_probe_scheduler(spec))
        elif kind_id == "trainer_policy":
            if getattr(instance, "unsafe", False):
                found.append(finding(
                    "W004", f"trainer_policy:{resolved.component.name} is declared unsafe; it needs "
                    "TRAINER.ALLOW_UNSAFE_POLICY to run", severity="warning",
                ))
    except ComponentError as error:
        found.append(finding_of(error))
    except (ImproperlyConfigured, MACTFError) as error:
        found.append(finding_of(error))
    except Exception as error:  # noqa: BLE001 - a probe that crashes is a finding about the component
        found.append(finding("E008", f"{kind_id}:{resolved.component.name}: probe raised "
                                     f"{type(error).__name__}: {error}"))
    return found


def _probe_optimizer(spec: Any) -> list[dict[str, Any]]:
    """The optimizer probe over a scratch module, through the ``OPTIMIZER`` block the spec means."""
    from ..training.optimizer_plan import compile_entry, verify_entry

    params = dict(spec.params)
    block: dict[str, Any] = {"TYPE": spec.type or "torch"}
    for key in ("lr", "weight_decay"):
        if key in params:
            block[key.upper()] = params.pop(key)
    if params:
        block["PARAMS"] = params
    entry = compile_entry("OPTIMIZER", block, glob=None, horizon=100, written=None)
    verify_entry(entry)
    return []


def _probe_scheduler(spec: Any) -> list[dict[str, Any]]:
    """Dry-build a scheduler over a scratch SGD optimizer (E053 for bad parameters)."""
    import torch

    from ..training.schedulers import build_scheduler

    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
    try:
        build_scheduler(str(spec.type), optimizer, dict(spec.params))
    except ComponentError:
        raise
    except Exception as error:  # noqa: BLE001
        return [finding("E053", f"scheduler {spec.type!r}: {type(error).__name__}: {error}")]
    return []


# ---------------------------------------------------------------------------
# A whole TrainerSpec
# ---------------------------------------------------------------------------
#: ``TrainerSpec`` field -> component kind id, for the fields that name a registered component.
_FIELD_KINDS = {"peer_strategy": "peer_strategy", "weight_update": "weight_update"}


def validate_trainer_spec(body: Mapping[str, Any], settings: Any, trusted: bool = False) -> list[dict[str, Any]]:
    """Every problem in a trainer spec, in one list.

    A component field is resolved with ``ComponentResolver.validate`` (``trusted=False`` for the
    API, so an ``import:`` spec is ``E006``); the ``optimizer`` block is compiled the way the
    plan would (``E057``, ``E053``); the rest is checked by building the trainer's kwargs and
    the trainer itself is *not* built. A warning is not a failure.
    """
    from .resolver import ComponentResolver, ComponentSpec

    found: list[dict[str, Any]] = []
    resolver = ComponentResolver()
    for field, kind_id in _FIELD_KINDS.items():
        raw = body.get(field)
        if raw is None:
            continue
        try:
            spec = ComponentSpec.parse(kind_id, raw)
            component = resolver.validate(spec, trusted=bool(trusted))
        except ComponentError as error:
            found.append(finding_of(error, where=field))
            continue
        if getattr(component, "unsafe", False):
            found.append(finding("W004", f"{field}: {kind_id}:{component.name} is declared unsafe",
                                 severity="warning"))
    optimizer = body.get("optimizer")
    if isinstance(optimizer, Mapping):
        found.extend(_optimizer_findings(optimizer))
    return found


def _optimizer_findings(optimizer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the facade the spec describes (no parameters bound): ``E057``-style problems surface here."""
    from ..training.optimizer import MACTFOptimizer

    try:
        MACTFOptimizer.from_spec({str(k): v for k, v in optimizer.items() if v is not None})
    except (ComponentError, ImproperlyConfigured, MACTFError, KeyError, TypeError, ValueError) as error:
        return [finding_of(error, where="optimizer")]
    return []
