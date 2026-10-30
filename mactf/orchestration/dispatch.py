"""Engine dispatch: pick ``Orchestrator`` or ``TrainerEngine`` without importing torch.

``mactf.management.driver`` documents that importing it imports no torch
(``driver.py:9-10``), so the choice and the class are separated: this module is
torch-free and imports a class only when a run is actually constructed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from ..exceptions import ImproperlyConfigured

if TYPE_CHECKING:  # pragma: no cover - annotations only, and torch-free by design
    from ..db import Registry
    from .orchestrator import RunConfig
    from .phases import PhaseContext

log = logging.getLogger("mactf.orchestration.dispatch")

#: ``ORCHESTRATOR.ENGINE`` values. The first is the default and the frozen oracle.
ENGINE_ORCHESTRATOR = "orchestrator"
ENGINE_TRAINER = "trainer"
ENGINES: tuple[str, ...] = (ENGINE_ORCHESTRATOR, ENGINE_TRAINER)
DEFAULT_ENGINE = ENGINE_ORCHESTRATOR


@runtime_checkable
class Engine(Protocol):
    """What ``RunDriver``, ``RunManager``, ``mactf redteam`` and the API read.

    Every member below is used by a real call site; the comment names it. The
    protocol is deliberately *structural*: ``Orchestrator`` satisfies it without
    being changed, and ``TrainerEngine`` satisfies it by delegation.
    """

    #: ``driver.py:228`` and ``live.py`` -- the resolved runtime description.
    settings: Any
    #: ``live.py:649-651`` reads and then writes it; ``driver.py:233`` logs it.
    run_id: str
    #: ``driver.py:182``, ``:295``; ``live.py:810``.
    registry: Any
    #: ``_persist_warnings`` and ``RunManager.attach`` consumers.
    scorer: Any
    #: ``driver.py`` does not read it; the engine's ``_resume`` does.
    checkpoints: Any
    #: ``driver.py:229`` (``", ".join(peers)``), ``peers.py:27``.
    peers: dict[str, Any]
    #: ``phases.bootstrap`` and every phase function.
    reference: Any
    #: ``phases.aggregate`` when the engine delegates; ``None`` is allowed.
    aggregator: Any
    #: The engine's own epoch loop uses it; surfaced for parity.
    runner: Any
    #: ``driver.py:228`` (``context.runtime``), ``redteam.py:92-98``.
    context: "PhaseContext | None"
    #: ``driver.py:230``.
    sandbox_choice: Any
    #: ``driver.py:231``, printed as ``note`` lines.
    startup_notes: list[str]
    #: Not read by a driver; recorded so a caller can compute a duration.
    started_at: float

    def prepare(self, config: "RunConfig | None" = None) -> "PhaseContext":
        """Build everything a run needs, raising before any row is written."""
        ...

    def run(
        self,
        config: "RunConfig | None" = None,
        *,
        should_stop: Any = None,
        on_epoch: Any = None,
    ) -> dict[str, Any]:
        """Run the epoch loop; return the summary ``driver.py:293-299`` reads."""
        ...


def engine_name(settings: Any) -> str:
    """``ORCHESTRATOR.ENGINE``, validated. ``E090`` for anything else."""
    block = settings.get("ORCHESTRATOR", {}) if hasattr(settings, "get") else {}
    raw = block.get("ENGINE", DEFAULT_ENGINE) if isinstance(block, Mapping) else DEFAULT_ENGINE
    name = str(DEFAULT_ENGINE if raw is None or raw == "" else raw).strip().lower()
    if name not in ENGINES:
        raise ImproperlyConfigured(
            f"E090: ORCHESTRATOR.ENGINE must be one of {list(ENGINES)}, got {raw!r}"
        )
    return name


def engine_for(settings: Any) -> type:
    """The class, imported on demand so this module stays torch-free."""
    if engine_name(settings) == ENGINE_TRAINER:
        from .engine import TrainerEngine

        return TrainerEngine
    from .orchestrator import Orchestrator

    return Orchestrator


def engine_factory(
    settings: Any,
    registry: "Registry | None" = None,
    *,
    modules: Mapping[str, Any] | None = None,
) -> Engine:
    """Construct the configured engine.

    ``registry=None`` means "resolve one" (``ensure_registry``), which is what
    ``RunDriver`` wants: it has no registry until an engine has one.
    """
    if engine_name(settings) == ENGINE_TRAINER:
        from .engine import TrainerEngine

        return TrainerEngine(settings, modules=modules, registry=registry)
    from .orchestrator import Orchestrator

    return Orchestrator(settings, modules=dict(modules) if modules else None, registry=registry)


def engine_of(engine: Any) -> str:
    """The engine name of a live object, without importing either class.

    Used for the checkpoint's ``identity.engine`` and for ``E091``. A subclass of
    ``TrainerEngine`` is still the trainer engine; anything else is the orchestrator.
    """
    if type(engine).__name__ == "TrainerEngine":
        return ENGINE_TRAINER
    for klass in type(engine).__mro__:
        if klass.__name__ == "TrainerEngine":
            return ENGINE_TRAINER
    return ENGINE_ORCHESTRATOR
