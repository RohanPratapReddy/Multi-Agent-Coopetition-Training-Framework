"""Choosing, validating, guarding and attaching the run's training policy (document 17 section 7.5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..exceptions import ImproperlyConfigured, UnsafePolicyDenied
from .guards.step import GuardedPolicy, StepGuard, StepGuardConfig
from .kinds import Origin

log = logging.getLogger("mactf.components.policy")

POLICY_SLOT = "TRAINER.POLICY"
POLICY_KIND = "trainer_policy"


@dataclass(frozen=True)
class PolicyChoice:
    policy: Any
    origin: Origin
    guard: StepGuard | None
    source: str          # "code" | "spec" | "distillation" | "default"
    fingerprint: dict[str, Any]
    #: The policy as configured, before any guard wrapped it (``None`` for the default).
    #: ``MACTFTrainer._trainer_policy`` keeps this meaning; ``policy`` is what runs.
    configured: Any = None


def _trainer_block(settings: Any) -> Mapping[str, Any]:
    block = settings.get("TRAINER", {}) if hasattr(settings, "get") else {}
    return block or {}


def _unsafe_allowed(settings: Any) -> bool:
    return bool(_trainer_block(settings).get("ALLOW_UNSAFE_POLICY", False))


def _guard_config(settings: Any, *, default_mode: str) -> StepGuardConfig:
    block = dict(_trainer_block(settings).get("STEP_GUARD", {}) or {})
    block.setdefault("MODE", default_mode)
    return StepGuardConfig.from_settings(block)


def _wrap(policy: Any, *, origin: Origin, source: str, settings: Any, host: Any) -> tuple[Any, StepGuard | None]:
    """Wrap everything the framework did not write in a ``StepGuard``.

    A built-in policy is returned as is (invariant I1). An unsafe policy is returned as
    is too, *after* the flag check: it has declared that it manages its own safety, and the
    guard's snapshot would not cover a policy that rewrites parameters outside the step.
    """
    if origin is Origin.BUILTIN:
        return policy, None
    unsafe = bool(getattr(policy, "unsafe", False))
    if unsafe:
        if not _unsafe_allowed(settings):
            raise UnsafePolicyDenied(
                getattr(policy, "name", type(policy).__name__),
                "the policy declares unsafe=True; set TRAINER.ALLOW_UNSAFE_POLICY=true to run it",
            )
        log.warning(
            "policy %r runs unsafe: no StepGuard snapshot or quarantine applies",
            getattr(policy, "name", type(policy).__name__),
        )
        return policy, None
    # An in-code ``trainer=`` instance predates this document; it is checked in "warn" mode
    # by default so an existing project is told, not broken. A spec-resolved policy is new
    # code and defaults to "quarantine".
    default_mode = "warn" if source == "code" else "quarantine"
    guard = StepGuard(_guard_config(settings, default_mode=default_mode))
    wrapped = GuardedPolicy(policy, guard, origin=origin)
    return wrapped, guard


def resolve_policy(
    *,
    code_instance: Any,
    settings: Any,
    distillation: Any,
    slots: Any,
    host: Any,
    distill_factory: Callable[[], Any],
) -> PolicyChoice:
    """Pick, validate, guard and attach the run's training policy."""
    # ``default`` and ``co_distill`` are registered by this module's import. It is not the
    # registry's ``builtins_module`` because that import needs torch, and registering a
    # trainer policy must stay possible without it (``test_import_weight``).
    from ..training import policy as _builtin_policies  # noqa: F401

    block = _trainer_block(settings)
    raw_spec = block.get("POLICY")
    distill_on = bool(getattr(distillation, "enabled", False))
    # One distillation step per run, built once and reachable as ``host.distill`` by
    # whichever policy owns or composes it (``co_distill`` and any ``composes_distillation``
    # policy read it in ``attach``). Built first so ``attach`` order cannot matter.
    distill_step = distill_factory() if distill_on else None
    host.distill = distill_step

    if code_instance is not None:
        if not callable(getattr(code_instance, "train_step", None)) and callable(code_instance):
            # The legacy callable form ``(node, batch, trainer=trainer)`` joins the protocol.
            code_instance = CallablePolicy(code_instance)
        policy, source, origin = code_instance, "code", Origin.RUNTIME
        if raw_spec not in (None, "", {}):
            raise ImproperlyConfigured(
                "a trainer= instance and TRAINER.POLICY were both given; the instance would "
                "silently win. Remove one."
            )
        slot = slots.adopt(POLICY_SLOT, POLICY_KIND, code_instance, name=_name_of(code_instance))
    elif raw_spec not in (None, "", {}):
        slot = slots.install(POLICY_SLOT, POLICY_KIND, raw_spec, inject={"host": host})
        policy, source, origin = slot.instance, "spec", slot.component.origin
    elif distill_on:
        policy, source, origin = distill_step, "distillation", Origin.BUILTIN
        slot = slots.adopt_registered(POLICY_SLOT, POLICY_KIND, policy, name="co_distill")
    else:
        # Built with a plain call, not through the registry: a default run makes no
        # registry call at all (section 19.4, condition 4).
        from ..training.policy import DefaultTrainPolicy

        policy, source, origin = DefaultTrainPolicy(), "default", Origin.BUILTIN
        slot = slots.adopt_registered(POLICY_SLOT, POLICY_KIND, policy, name="default")

    if distill_on and source in ("code", "spec"):
        if not bool(getattr(policy, "composes_distillation", False)):
            raise ImproperlyConfigured(
                f"DISTILLATION.ENABLED is true and policy {_name_of(policy)!r} does not declare "
                "composes_distillation=True; two policies cannot both own the step. Disable "
                "distillation, choose TRAINER.POLICY=co_distill, or set composes_distillation "
                "and delegate through host.distill."
            )

    wrapped, guard = _wrap(policy, origin=origin, source=source, settings=settings, host=host)
    attach = getattr(wrapped, "attach", None)
    if callable(attach):
        attach(host)
    return PolicyChoice(
        policy=wrapped,
        origin=origin,
        guard=guard,
        source=source,
        fingerprint=slot.resolved.fingerprint(),
        configured=None if source == "default" else policy,
    )


class CallablePolicy:
    """A legacy ``policy(node, batch, trainer=trainer)`` callable behind the policy protocol.

    ``self_stepping`` is deliberately not declared: the trainer reads it with a ``False``
    default for a callable, which is the behaviour before this document.
    """

    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function
        self.name = getattr(function, "__name__", type(function).__name__)
        self._host: Any = None

    def attach(self, host: Any) -> None:
        self._host = host

    def train_step(self, node: Any, batch: Mapping[str, Any]) -> Any:
        return self.function(node, batch, trainer=self._host)


def _name_of(policy: Any) -> str:
    named = getattr(policy, "name", None)
    return named if isinstance(named, str) and named else type(policy).__name__
