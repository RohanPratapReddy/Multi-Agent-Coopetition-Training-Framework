"""Components, registries, the hub and per-run overlays (document 17 sections 3.3-3.5).

The hub is a process singleton holding one :class:`ComponentRegistry` per kind. A run may
stack an :class:`Overlay` over it (a ``contextvars`` stack, so two runs in one process never
see each other's registrations). Nothing here imports torch or any framework module at
import time; the contracts import what they check lazily.
"""

from __future__ import annotations

import contextlib
import contextvars
import fnmatch
import hashlib
import importlib
import itertools
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Mapping, TypeVar

from ..exceptions import (
    ComponentBuildError,
    ComponentCollision,
    ComponentContractError,
    ComponentLocked,
    ComponentMissing,
    ComponentParamError,
    ComponentSpecError,
    ValidationError,
)
from .kinds import KIND_SPECS, KindSpec, Mutability, Origin, all_kind_ids
from .params import ParamSpec
from .reflect import canonical_json, reflect, source_digest

log = logging.getLogger("mactf.components")
T = TypeVar("T")

#: Kinds whose names ``mactf/conf.py`` lower-cases before comparing.
_LOWERCASE_KINDS = frozenset({"peer_strategy", "weight_update"})


@dataclass(frozen=True)
class Component:
    """One registered, named factory. Immutable: replacing means registering a new one."""

    kind: str
    name: str
    factory: Any
    origin: Origin
    mutability: Mutability
    version: str = "0"
    module: str = ""
    qualname: str = ""
    description: str = ""
    params_schema: Mapping[str, ParamSpec] | None = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    #: Names of other components of any kind this one needs, as ``"kind:name"``.
    requires: frozenset[str] = frozenset()
    overrides_builtin: bool = False
    seq: int = 0

    # -- reflection ------------------------------------------------------
    def accepted_params(self) -> frozenset[str]:
        if self.params_schema is not None:
            return frozenset(self.params_schema)
        return reflect(self.factory).keys

    def check_params(
        self, params: Mapping[str, Any], *, ignore: frozenset[str] = frozenset()
    ) -> list[str]:
        """Problems with ``params``, empty when fine. ``ignore`` names keys the
        resolver injects (for example ``config`` for a scorer) and are not user params."""
        problems: list[str] = []
        reflection = reflect(self.factory)
        accepted = self.accepted_params() | ignore
        schema = self.params_schema
        if not (reflection.var_keyword and schema is None):
            for key in sorted(params):
                if key not in accepted:
                    problems.append(
                        f"{self.name} does not accept {key!r}; "
                        f"accepted keys: {sorted(accepted - ignore)}"
                    )
        if schema is not None:
            for key, spec in schema.items():
                if key in params:
                    problems.extend(spec.problems(key, params[key]))
                elif spec.required:
                    problems.append(f"missing required parameter {key!r}")
        else:
            for key in sorted(reflection.required - ignore):
                if key not in params:
                    problems.append(f"missing required parameter {key!r}")
        return problems

    # -- reporting -------------------------------------------------------
    def fingerprint(
        self, *, spec_hash: str | None = None, with_source: bool = True
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "origin": self.origin.value,
            "version": self.version,
        }
        if spec_hash is not None:
            payload["spec_hash"] = spec_hash
        if self.origin is not Origin.BUILTIN and with_source:
            payload["source_digest"] = source_digest(self.factory)
        if self.overrides_builtin:
            payload["overrides_builtin"] = True
        return payload

    def fingerprint_text(self) -> str:
        """Twelve hex digits that change when the component's identity or source does."""
        text = canonical_json({"d": self.describe(), "s": source_digest(self.factory)})
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "origin": self.origin.value,
            "mutability": self.mutability.value,
            "version": self.version,
            "class": self.qualname or getattr(self.factory, "__name__", repr(self.factory)),
            "module": self.module,
            "description": self.description,
            "kwargs": sorted(self.accepted_params()),
            "requires": sorted(self.requires),
            "overrides_builtin": self.overrides_builtin,
        }
        if self.params_schema is not None:
            payload["params"] = {k: v.describe() for k, v in self.params_schema.items()}
        if self.capabilities:
            payload["capabilities"] = dict(self.capabilities)
        return payload


class Overlay:
    """A per-run layer of registrations and hidden names, stacked over the base."""

    __slots__ = ("label", "_entries", "_hidden", "_lock")

    def __init__(self, label: str) -> None:
        self.label = label
        self._entries: dict[str, dict[str, Component]] = {}
        self._hidden: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    def put(self, component: Component) -> None:
        with self._lock:
            self._entries.setdefault(component.kind, {})[component.name] = component
            self._hidden.get(component.kind, set()).discard(component.name)

    def hide(self, kind: str, name: str) -> None:
        with self._lock:
            self._entries.get(kind, {}).pop(name, None)
            self._hidden.setdefault(kind, set()).add(name)

    def entry(self, kind: str, name: str) -> Component | None:
        return self._entries.get(kind, {}).get(name)

    def is_hidden(self, kind: str, name: str) -> bool:
        return name in self._hidden.get(kind, ())

    def entries(self, kind: str) -> Mapping[str, Component]:
        return dict(self._entries.get(kind, {}))

    def hidden(self, kind: str) -> frozenset[str]:
        return frozenset(self._hidden.get(kind, ()))

    def is_empty(self) -> bool:
        return not any(self._entries.values()) and not any(self._hidden.values())


@dataclass(frozen=True)
class HubPolicy:
    """What the hub enforces. Resolved from ``COMPONENTS`` (section 15)."""

    allow_builtin_override: bool = False
    #: ``"kind:name"`` patterns (``fnmatch``); ``"scorer:*"`` locks a whole kind.
    locked: frozenset[str] = frozenset()
    strict_legacy: bool = False
    allow_import: bool = False
    import_allowlist: tuple[str, ...] = ()
    max_components_per_kind: int = 512
    #: ``COMPONENTS.STRICT_RESTORE``: a restore difference is E094, not W052 (section 14).
    strict_restore: bool = False

    def is_locked(self, kind: str, name: str) -> bool:
        key = f"{kind}:{name}"
        return any(fnmatch.fnmatchcase(key, pattern) for pattern in self.locked)


def _called_by_framework() -> bool:
    """Whether the first caller outside ``mactf.components`` is a ``mactf`` module."""
    import sys

    frame = sys._getframe(1)
    while frame is not None:
        module = str(frame.f_globals.get("__name__", ""))
        if not module.startswith("mactf.components"):
            return module == "mactf" or module.startswith("mactf.")
        frame = frame.f_back
    return False


class ComponentRegistry(Generic[T]):
    """Name -> :class:`Component` for one kind. Thread-safe; overlay-aware."""

    def __init__(self, spec: KindSpec, hub: "Hub") -> None:
        self.spec = spec
        self._hub = hub
        self._entries: dict[str, Component] = {}
        self._lock = threading.RLock()
        self._frozen = False
        self._mirror: dict[str, Any] | None = None
        self._seq = itertools.count(1)
        #: ``name -> factory`` the framework shipped; lets a legacy write that puts the
        #: original back (a test restoring what it replaced) restore the built-in origin.
        self._builtin_factories: dict[str, Any] = {}
        self._builtins_ready = spec.builtins_module is None

    @property
    def kind(self) -> str:
        return self.spec.kind_id

    # -- built-ins -------------------------------------------------------
    def _ensure_builtins(self) -> None:
        """Import the module that registers this kind's built-ins, once.

        Without this a plugin loaded before the framework module that owns the legacy table
        could take a built-in's name, and the framework's own import would then collide
        with it. Registration is the only caller that must not race ahead; the import is
        skipped while the module itself is registering (``attach_mirror`` marks it ready).
        """
        if self._builtins_ready:
            return
        self._builtins_ready = True
        module = self.spec.builtins_module
        if module:
            importlib.import_module(module)

    @property
    def builtins_loaded(self) -> bool:
        return self._builtins_ready

    def register_builtin(self, name: str, factory: Any) -> Component:
        """Register (or re-register, on a module reload) a framework-shipped component.

        Built-ins are not subject to locks, freezing or the per-kind cap: they are the
        framework's own import, and a deployment that locks ``scorer:*`` still has to be
        able to import ``mactf.scoring`` after the lock is applied. The kind's contract is
        still enforced.
        """
        name = self._check_name(name)
        problems = self.spec.contract(factory)
        if problems:
            raise ComponentContractError(self.kind, name, problems)
        with self._lock:
            existing = self._entries.get(name)
            if existing is not None and existing.origin is not Origin.BUILTIN:
                raise ComponentCollision(
                    self.kind,
                    name,
                    f"a built-in cannot take a name already registered by origin "
                    f"{existing.origin.value}",
                )
            component = self._make(name, factory, Origin.BUILTIN, self.spec.default_mutability)
            self._entries[name] = component
            self._builtin_factories[name] = factory
            self._mirror_set(name, factory)
            self._hub._bump()
        return component

    def restore_builtin(self, name: str) -> Component:
        """Put the framework's original back under ``name``, whatever replaced it."""
        with self._lock:
            factory = self._builtin_factories.get(name)
            if factory is None:
                raise ComponentMissing(self.kind, name, sorted(self._builtin_factories))
            component = self._make(name, factory, Origin.BUILTIN, self.spec.default_mutability)
            self._entries[name] = component
            self._mirror_set(name, factory)
            self._hub._bump()
        return component

    def is_builtin_factory(self, name: str, factory: Any) -> bool:
        return self._builtin_factories.get(name) is factory

    def is_builtin_class(self, factory: Any) -> bool:
        """Whether ``factory`` is one of the framework's own classes, under any built-in name.

        An in-code *instance* of a built-in class is adopted with the ``runtime`` origin; this
        is how a checkpoint tells it from an instance of a subclass (document 17 section 14.3).
        """
        self._ensure_builtins()
        with self._lock:
            return any(factory is original for original in self._builtin_factories.values())

    # -- mirror (section 3.6) --------------------------------------------
    def attach_mirror(self, table: dict[str, Any]) -> None:
        """Keep ``table`` equal to the base layer's ``name -> factory``."""
        with self._lock:
            self._mirror = table
            self._builtins_ready = True
            dict.clear(table)
            for name, component in self._entries.items():
                dict.__setitem__(table, name, component.factory)

    # -- lookup ----------------------------------------------------------
    def find(self, name: str) -> Component | None:
        self._ensure_builtins()
        name = str(name)
        for overlay in reversed(self._hub.active_overlays()):
            if overlay.is_hidden(self.kind, name):
                return None
            found = overlay.entry(self.kind, name)
            if found is not None:
                return found
        return self._entries.get(name)

    def get(self, name: str) -> Component:
        found = self.find(str(name))
        if found is None:
            raise ComponentMissing(self.kind, str(name), self.names())
        return found

    def has(self, name: str) -> bool:
        return self.find(str(name)) is not None

    def names(self) -> list[str]:
        self._ensure_builtins()
        visible = dict.fromkeys(self._entries)
        for overlay in self._hub.active_overlays():
            for name in overlay.hidden(self.kind):
                visible.pop(name, None)
            for name in overlay.entries(self.kind):
                visible[name] = None
        return sorted(visible)

    def components(self) -> list[Component]:
        return [self.get(name) for name in self.names()]

    def capabilities(self) -> list[dict[str, Any]]:
        return [component.describe() for component in self.components()]

    # -- registration ----------------------------------------------------
    def register(
        self,
        name: str,
        factory: Any,
        *,
        origin: Origin | None = None,
        mutability: Mutability | None = None,
        version: str = "0",
        description: str = "",
        params_schema: Mapping[str, ParamSpec] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        requires: Any = (),
        replace: bool = False,
        scope: str = "global",
        via: str = "api",
    ) -> Component:
        self._ensure_builtins()
        origin = origin or self._hub.default_origin()
        name = self._check_name(name)
        if origin is Origin.BUILTIN and not _called_by_framework():
            # ``builtin`` is what makes a component skip the guards that wrap customisations
            # (a built-in policy is not wrapped, a built-in is never "custom"): a plugin cannot
            # give itself that origin through ``register(..., origin=...)`` or a decorator option.
            raise ComponentContractError(
                self.kind, name,
                ["origin 'builtin' is reserved for the framework's own modules"],
            )
        legacy_check = self.spec.legacy_contract if via == "legacy" else None
        problems = (legacy_check or self.spec.contract)(factory)
        if problems:
            raise ComponentContractError(self.kind, name, problems)
        default = self.spec.default_mutability
        chosen = mutability or default
        if chosen.rank > default.rank:
            raise ComponentContractError(
                self.kind,
                name,
                [f"mutability {chosen.value!r} is looser than the kind default {default.value!r}"],
            )
        policy = self._hub.effective_policy()
        if policy.is_locked(self.kind, name):
            raise ComponentLocked(self.kind, name, "the slot is locked by COMPONENTS.LOCKED")
        with self._lock:
            if scope == "run":
                overlay = self._hub.current_overlay(required=True)
                existing = self.find(name)
            else:
                if self._frozen:
                    raise ComponentLocked(self.kind, name, "the registry is frozen")
                if (
                    len(self._entries) >= policy.max_components_per_kind
                    and name not in self._entries
                ):
                    raise ComponentContractError(
                        self.kind,
                        name,
                        [f"registry is full ({policy.max_components_per_kind} components)"],
                    )
                overlay = None
                existing = self._entries.get(name)
            restoring = (
                existing is not None
                and replace
                and via == "legacy"
                and scope != "run"
                and self.is_builtin_factory(name, factory)
            )
            if restoring:
                # Putting the framework's own class back is undoing a replacement, not a
                # new one: it is allowed whatever replaced it, and it is a built-in again.
                return self.restore_builtin(name)
            overrides = self._check_replace(name, existing, origin, replace, policy, via)
            component = self._make(
                name,
                factory,
                origin,
                chosen,
                version=version,
                description=description,
                params_schema=params_schema,
                capabilities=capabilities,
                requires=requires,
                overrides=overrides,
            )
            if overlay is not None:
                overlay.put(component)
            else:
                self._entries[name] = component
                self._mirror_set(name, factory)
                self._hub._bump()
        if overrides:
            log.warning(
                "W001 %s:%s replaces a built-in component (origin=%s)",
                self.kind,
                name,
                origin.value,
            )
        return component

    def _make(
        self,
        name: str,
        factory: Any,
        origin: Origin,
        mutability: Mutability,
        *,
        version: str = "0",
        description: str = "",
        params_schema: Mapping[str, ParamSpec] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        requires: Any = (),
        overrides: bool = False,
    ) -> Component:
        return Component(
            kind=self.kind,
            name=name,
            factory=factory,
            origin=origin,
            mutability=mutability,
            version=str(version),
            module=getattr(factory, "__module__", "") or "",
            qualname=getattr(factory, "__qualname__", "") or "",
            description=description
            or (getattr(factory, "__doc__", "") or "").strip().split("\n")[0],
            params_schema=dict(params_schema) if params_schema is not None else None,
            capabilities=dict(capabilities or {}),
            requires=frozenset(requires or ()),
            overrides_builtin=overrides,
            seq=next(self._seq),
        )

    def _check_replace(
        self,
        name: str,
        existing: Component | None,
        origin: Origin,
        replace: bool,
        policy: HubPolicy,
        via: str,
    ) -> bool:
        """Decide whether ``name`` may be taken over; ``True`` when it replaces a built-in."""
        if existing is None:
            return False
        if not replace:
            raise ComponentCollision(
                self.kind,
                name,
                f"already registered by origin {existing.origin.value} "
                f"({existing.module}.{existing.qualname}); pass replace=True to override",
            )
        if existing.origin is Origin.BUILTIN:
            if via == "legacy" and policy.strict_legacy and not policy.allow_builtin_override:
                raise ComponentLocked(
                    self.kind,
                    name,
                    "is built in; a legacy registry write may not replace it "
                    "(COMPONENTS.STRICT_LEGACY)",
                )
            legacy_ok = via == "legacy" and not policy.strict_legacy
            if not (policy.allow_builtin_override or legacy_ok):
                raise ComponentCollision(
                    self.kind,
                    name,
                    "is built in; replacing it needs COMPONENTS.ALLOW_BUILTIN_OVERRIDE",
                )
            if via == "legacy":
                log.warning("W005 %s:%s replaced through a legacy registry write", self.kind, name)
            return True
        if origin.rank < existing.origin.rank:
            raise ComponentCollision(
                self.kind,
                name,
                f"origin {origin.value} cannot replace a component from origin "
                f"{existing.origin.value}",
            )
        return False

    def unregister(self, name: str, *, scope: str = "global", via: str = "api") -> Component:
        self._ensure_builtins()
        policy = self._hub.effective_policy()
        if policy.is_locked(self.kind, name):
            raise ComponentLocked(self.kind, name, "the slot is locked by COMPONENTS.LOCKED")
        with self._lock:
            if scope == "run":
                overlay = self._hub.current_overlay(required=True)
                existing = self.find(name)
                if existing is None:
                    raise ComponentMissing(self.kind, name, self.names())
                self._check_unregister(existing, policy, via)
                overlay.hide(self.kind, name)
                return existing
            if self._frozen:
                raise ComponentLocked(self.kind, name, "the registry is frozen")
            existing = self._entries.get(name)
            if existing is None:
                raise ComponentMissing(self.kind, name, self.names())
            self._check_unregister(existing, policy, via)
            del self._entries[name]
            self._mirror_del(name)
            self._hub._bump()
            return existing

    def _check_unregister(self, existing: Component, policy: HubPolicy, via: str) -> None:
        if existing.origin is Origin.BUILTIN:
            legacy_ok = via == "legacy" and not policy.strict_legacy
            if not (policy.allow_builtin_override or legacy_ok):
                raise ComponentLocked(
                    self.kind,
                    existing.name,
                    "is built in; removing it needs COMPONENTS.ALLOW_BUILTIN_OVERRIDE",
                )
            if via == "legacy":
                log.warning(
                    "W005 %s:%s removed through a legacy registry write", self.kind, existing.name
                )

    def freeze(self) -> None:
        with self._lock:
            self._frozen = True

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    # -- construction ----------------------------------------------------
    def build(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        inject: Mapping[str, Any] | None = None,
        check_instance: bool = True,
    ) -> T:
        """Validate ``params``, call the factory, check the instance contract.

        ``check_instance=False`` skips the last step. The legacy ``build_function`` passes it:
        the old function registry never required a wrapper to define its entry method until the
        wrapper was *called*, and a test or a project may register a marker class that way.

        ``inject`` carries keys the framework offers (a scorer's ``config``, a warning
        stage's ``warning_policy``); the factory receives those it declares, and a spec may
        not set them.
        """
        component = self.get(name)
        values = dict(params or {})
        reflection = reflect(component.factory)
        accepted = component.accepted_params()
        takes_any = reflection.var_keyword and component.params_schema is None
        # Injection is an offer: the factory receives only the keys it declares.
        extra = {
            key: value
            for key, value in dict(inject or {}).items()
            if key in accepted or takes_any
        }
        clash = sorted(set(values) & set(extra))
        if clash:
            raise ComponentSpecError(
                self.kind,
                f"E011: {self.kind}:{component.name}: "
                + "; ".join(
                    f"parameter {key!r} is supplied by the framework and cannot be set in a spec"
                    for key in clash
                ),
            )
        problems = component.check_params(values, ignore=frozenset(extra))
        if problems:
            raise ComponentParamError(self.kind, component.name, problems)
        try:
            builder = self.spec.builder
            if builder is not None:
                instance = builder(component, values, extra)
            else:
                instance = component.factory(**values, **extra)
        except (TypeError, ValueError) as error:
            raise ComponentBuildError(self.kind, component.name, error) from error
        bad = self.spec.instance_contract(instance) if check_instance else []
        if bad:
            raise ComponentContractError(self.kind, component.name, bad)
        return instance

    # -- helpers ---------------------------------------------------------
    def _check_name(self, name: str) -> str:
        """The rule every ``register_*`` already applies: ``name.replace("_", "").isalnum()``."""
        text = str(name)
        if not text or not text.replace("_", "").isalnum():
            noun = "function" if self.spec.family else self.kind
            raise ComponentContractError(
                self.kind,
                text,
                [f"invalid {noun} type name {text!r}: letters, digits and underscores only"],
            )
        if self.kind in _LOWERCASE_KINDS and text != text.lower():
            raise ComponentContractError(
                self.kind,
                text,
                ["names of this kind are lower-case: the configuration layer lower-cases them"],
            )
        return text

    def _mirror_set(self, name: str, factory: Any) -> None:
        if self._mirror is not None:
            dict.__setitem__(self._mirror, name, factory)

    def _mirror_del(self, name: str) -> None:
        if self._mirror is not None:
            dict.pop(self._mirror, name, None)


_OVERLAYS: contextvars.ContextVar[tuple[Overlay, ...]] = contextvars.ContextVar(
    "mactf_component_overlays", default=()
)
_POLICY: contextvars.ContextVar[HubPolicy | None] = contextvars.ContextVar(
    "mactf_component_policy", default=None
)
_DEFAULT_ORIGIN: contextvars.ContextVar[Origin] = contextvars.ContextVar(
    "mactf_component_default_origin", default=Origin.PROJECT
)


class Hub:
    """Process singleton: one registry per kind, a policy and an overlay stack."""

    def __init__(self) -> None:
        self._registries: dict[str, ComponentRegistry[Any]] = {}
        self._policy = HubPolicy()
        self._lock = threading.RLock()
        self._generation = 0

    # -- registries ------------------------------------------------------
    def registry(self, kind_id: str) -> ComponentRegistry[Any]:
        with self._lock:
            found = self._registries.get(kind_id)
            if found is None:
                spec = KIND_SPECS.get(kind_id)
                if spec is None:
                    raise ValidationError(
                        f"unknown component kind {kind_id!r}; kinds: "
                        + ", ".join(all_kind_ids())
                    )
                found = ComponentRegistry(spec, self)
                self._registries[kind_id] = found
            return found

    def kinds(self) -> tuple[str, ...]:
        return all_kind_ids()

    # -- policy ----------------------------------------------------------
    @property
    def policy(self) -> HubPolicy:
        return self._policy

    def configure(self, policy: HubPolicy) -> None:
        with self._lock:
            self._policy = policy

    def effective_policy(self) -> HubPolicy:
        return _POLICY.get() or self._policy

    # -- default origin --------------------------------------------------
    def default_origin(self) -> Origin:
        """The origin a ``register_*`` call gets when it names none: ``PROJECT`` unless
        a loader (plugin discovery, a runtime spec import) has scoped another one."""
        return _DEFAULT_ORIGIN.get()

    @contextlib.contextmanager
    def loading(self, origin: Origin) -> Iterator[None]:
        token = _DEFAULT_ORIGIN.set(origin)
        try:
            yield
        finally:
            _DEFAULT_ORIGIN.reset(token)

    # -- overlays --------------------------------------------------------
    def active_overlays(self) -> tuple[Overlay, ...]:
        return _OVERLAYS.get()

    def current_overlay(self, *, required: bool = False) -> Overlay | None:
        stack = _OVERLAYS.get()
        if stack:
            return stack[-1]
        if required:
            raise ComponentLocked("overlay", "-", "no run scope is active; use HUB.scoped()")
        return None

    @contextlib.contextmanager
    def scoped(self, label: str = "run", *, policy: HubPolicy | None = None) -> Iterator[Overlay]:
        """Activate a fresh overlay for the current thread/task; discard it on exit."""
        overlay = Overlay(label)
        stack_token = _OVERLAYS.set(_OVERLAYS.get() + (overlay,))
        policy_token = _POLICY.set(policy) if policy is not None else None
        try:
            yield overlay
        finally:
            if policy_token is not None:
                _POLICY.reset(policy_token)
            _OVERLAYS.reset(stack_token)

    def activate(self, overlay: Overlay) -> contextvars.Token:
        """Re-enter an existing overlay (a worker thread that must see a run's parts)."""
        return _OVERLAYS.set(_OVERLAYS.get() + (overlay,))

    def deactivate(self, token: contextvars.Token) -> None:
        _OVERLAYS.reset(token)

    # -- generation, snapshots -------------------------------------------
    def _bump(self) -> None:
        self._generation += 1

    @property
    def generation(self) -> int:
        return self._generation

    def freeze_all(self) -> None:
        for kind_id in all_kind_ids():
            self.registry(kind_id).freeze()

    def unfreeze_all(self) -> None:
        for kind_id in all_kind_ids():
            self.registry(kind_id).unfreeze()

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Every visible component of every kind: the data behind ``mactf components list``."""
        policy = self.effective_policy()
        return {
            kind_id: {
                c.name: {
                    **c.describe(),
                    "locked": policy.is_locked(kind_id, c.name),
                    "fingerprint": c.fingerprint_text(),
                }
                for c in self.registry(kind_id).components()
            }
            for kind_id in all_kind_ids()
        }


HUB = Hub()
