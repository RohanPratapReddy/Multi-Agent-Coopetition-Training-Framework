"""``co_distill``: document 14's step as a built-in ``trainer_policy`` (document 17 section 7.5, 7.9)."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from ..components.registry import HUB
from ..exceptions import ImproperlyConfigured
from ..training.protocol import EpochResult, StepResult


class CoDistillPolicy:
    """Contrastive co-distillation (document 14), selectable by name.

    A thin delegate: ``CoDistillStep`` is built from settings and a tree, not from ``params``,
    so the trainer builds it once (``MACTFTrainer._build_distillation_step``) and this class
    forwards to it through ``host.distill``.
    """

    name: ClassVar[str] = "co_distill"
    unsafe: ClassVar[bool] = False
    self_stepping: ClassVar[bool | None] = True
    requires_terms: ClassVar[frozenset[str]] = frozenset()
    composes_distillation: ClassVar[bool] = True
    consumes_functions: ClassVar[frozenset[str]] = frozenset({"loss"})

    def __init__(self) -> None:
        self._step: Any = None

    def attach(self, host: Any) -> None:
        if getattr(host, "distill", None) is None:
            raise ImproperlyConfigured(
                "TRAINER.POLICY is 'co_distill' but DISTILLATION.ENABLED is false"
            )
        self._step = host.distill

    def setup(self, node: Any) -> None:
        self._step.setup(node)

    def train_step(self, node: Any, batch: Mapping[str, Any]) -> StepResult:
        return self._step.train_step(node, batch)

    def on_epoch_end(self, tree: Any, result: EpochResult) -> None:
        self._step.on_epoch_end(tree, result)

    def card_metadata(self, path: str) -> dict[str, Any]:
        return self._step.card_metadata(path)

    def state_dict(self) -> dict[str, Any]:
        return self._step.state_dict()

    def load_state_dict(self, payload: Mapping[str, Any]) -> Any:
        return self._step.load_state_dict(payload)


def register_builtins() -> None:
    """Register ``co_distill``. Idempotent; the registry's ``builtins_module`` is ``training.policy``."""
    registry = HUB.registry("trainer_policy")
    if registry.find("co_distill") is None:
        registry.register_builtin("co_distill", CoDistillPolicy)


register_builtins()
