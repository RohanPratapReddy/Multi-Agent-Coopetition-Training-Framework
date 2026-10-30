"""Registration entry points (document 17 section 3.8).

``register`` is the generic form; each kind has a decorator over it. The two legacy
functions, ``register_scorer`` and ``register_function``, keep their old positional
signatures (returning ``None``) and gain the decorator form.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .kinds import ComponentKind, Mutability, Origin, resolve_kind
from .params import ParamSpec
from .registry import HUB, Component


def register(
    kind: ComponentKind | str,
    name: str,
    obj: Any = None,
    *,
    family: str | None = None,
    origin: Origin | None = None,
    mutability: Mutability | None = None,
    version: str = "0",
    description: str = "",
    params_schema: Mapping[str, ParamSpec] | None = None,
    capabilities: Mapping[str, Any] | None = None,
    requires: Any = (),
    replace: bool = False,
    scope: str = "global",
    via: str | None = None,
) -> Any:
    """Register ``obj`` under ``kind``/``name``; or return a decorator when ``obj`` is None.

    Called with an object it returns that object's :class:`Component`. Used as
    ``@register(ComponentKind.SCORER, "x")`` it returns the decorated class unchanged.
    ``kind`` may be a :class:`ComponentKind`, a kind name or a full kind id such as
    ``"rl_function.loss"``.
    """
    registry = HUB.registry(resolve_kind(kind, family))

    def apply(target: Any, resolved_via: str) -> Component:
        return registry.register(
            name,
            target,
            origin=origin,
            mutability=mutability,
            version=version,
            description=description,
            params_schema=params_schema,
            capabilities=capabilities,
            requires=requires,
            replace=replace,
            scope=scope,
            via=resolved_via,
        )

    if obj is not None:
        return apply(obj, via or "api")

    def decorator(target: Any) -> Any:
        apply(target, via or "api")
        return target

    return decorator


def unregister(
    kind: ComponentKind | str,
    name: str,
    *,
    family: str | None = None,
    scope: str = "global",
    via: str = "api",
) -> Component:
    return HUB.registry(resolve_kind(kind, family)).unregister(name, scope=scope, via=via)


def get_component(kind: ComponentKind | str, name: str, *, family: str | None = None) -> Component:
    return HUB.registry(resolve_kind(kind, family)).get(name)


def _kind_decorator(kind: ComponentKind) -> Callable[..., Any]:
    def register_kind(name: str, obj: Any = None, **options: Any) -> Any:
        return register(kind, name, obj, **options)

    register_kind.__name__ = f"register_{kind.value}"
    register_kind.__doc__ = f"Register a {kind.value} component. See ``register``."
    return register_kind


register_reward_term = _kind_decorator(ComponentKind.REWARD_TERM)
register_optimizer = _kind_decorator(ComponentKind.OPTIMIZER)
register_scheduler = _kind_decorator(ComponentKind.SCHEDULER)
register_peer_strategy = _kind_decorator(ComponentKind.PEER_STRATEGY)
register_weight_update = _kind_decorator(ComponentKind.WEIGHT_UPDATE)
register_aggregator = _kind_decorator(ComponentKind.AGGREGATOR)
register_policy = _kind_decorator(ComponentKind.TRAINER_POLICY)
register_peer_factory = _kind_decorator(ComponentKind.PEER_FACTORY)


def register_scorer(
    name: str, scorer: Any = None, *, replace: bool = False, **options: Any
) -> Any:
    """``register_scorer("n", Cls, replace=...)`` (legacy, returns ``None``) or ``@register_scorer("n")``.

    Add a stage type. ``replace=False`` refuses to overwrite an existing name, because a
    silent override would make a stored spec mean something different after a deploy.
    """
    if scorer is not None:
        register(ComponentKind.SCORER, name, scorer, replace=replace, via="legacy", **options)
        return None
    return register(ComponentKind.SCORER, name, None, replace=replace, **options)


def register_function(
    family: str, name: str, cls: Any = None, *, replace: bool = False, **options: Any
) -> Any:
    """``register_function("loss", "n", Cls)`` (legacy, returns ``None``) or ``@register_function("loss", "n")``.

    Add a wrapper type to a family. Applications use this for custom wrappers.
    """
    if cls is not None:
        register(
            ComponentKind.RL_FUNCTION,
            name,
            cls,
            family=family,
            replace=replace,
            via="legacy",
            **options,
        )
        return None
    return register(
        ComponentKind.RL_FUNCTION, name, None, family=family, replace=replace, **options
    )
