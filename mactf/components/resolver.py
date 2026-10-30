"""The uniform spec and the resolver (document 17 section 5).

``ComponentSpec`` is the one shape every slot is described by: ``{type | import, params}``.
``ComponentResolver`` turns a spec into an instance through the registries and enforces the
import allow-list. ``merge_layers`` is the resolution order of section 5.3 made executable:
it merges the nine layers lowest to highest and records which layer decided what.
"""

from __future__ import annotations

import fnmatch
import importlib
import re
import types
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..exceptions import (
    ComponentBuildError,
    ComponentCollision,
    ComponentContractError,
    ComponentError,
    ComponentImportDenied,
    ComponentParamError,
    ComponentSpecError,
)
from .kinds import KIND_SPECS, Mutability, Origin
from .reflect import spec_hash
from .registry import HUB, Component

_IMPORT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")

#: The nine layers of section 5.3, lowest precedence first.
LAYERS: tuple[str, ...] = (
    "default",  # 1  built-in defaults (TRAINER_DEFAULTS, ...)
    "settings",  # 2  settings.py
    "conf",  # 3  conf/*.yaml
    "config_files",  # 4  CONFIG_FILES
    "env",  # 5  MACTF_* environment
    "config",  # 6  --config PATH
    "set",  # 7  --set KEY=VALUE
    "api",  # 8  API request spec
    "code",  # 9  in-code instance
)


@dataclass(frozen=True)
class ComponentSpec:
    kind_id: str
    type: str | None = None
    import_path: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    # -- parsing ---------------------------------------------------------
    @classmethod
    def parse(cls, kind_id: str, raw: Any, *, allow_patch: bool = False) -> "ComponentSpec":
        if isinstance(raw, ComponentSpec):
            if raw.kind_id != kind_id:
                raise ComponentSpecError(
                    kind_id, f"a {raw.kind_id} spec was given for a {kind_id} slot"
                )
            return raw
        if isinstance(raw, str):
            if not raw:
                raise ComponentSpecError(kind_id, "an empty component name")
            return cls(kind_id=kind_id, type=raw)
        if not isinstance(raw, Mapping):
            raise ComponentSpecError(
                kind_id, f"a spec is a name or a mapping, got {type(raw).__name__}"
            )
        data = dict(raw)
        type_name = data.pop("type", None)
        import_path = data.pop("import", None)
        params = data.pop("params", None)
        flat = data
        if params is not None and flat:
            raise ComponentSpecError(
                kind_id,
                f"E012: give params under 'params' or flat, not both; flat keys: {sorted(flat)}",
            )
        if params is None:
            params = flat
        if not isinstance(params, Mapping):
            raise ComponentSpecError(kind_id, "'params' must be a mapping")
        loose = [key for key in params if not isinstance(key, str)]
        if loose:
            # YAML reads ``1:`` as an int; a non-string name can neither be a parameter nor be sorted/hashed
            raise ComponentSpecError(kind_id, f"parameter names must be strings, got {loose!r}")
        if type_name is None and import_path is None and not allow_patch:
            raise ComponentSpecError(kind_id, "a spec needs 'type' or 'import'")
        if import_path is not None and not _IMPORT.match(str(import_path)):
            raise ComponentSpecError(
                kind_id, f"'import' must look like 'pkg.mod:Name', got {import_path!r}"
            )
        return cls(
            kind_id=kind_id,
            type=None if type_name is None else str(type_name),
            import_path=None if import_path is None else str(import_path),
            params=dict(params),
        )

    # -- canonical form --------------------------------------------------
    def canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.type is not None:
            payload["type"] = self.type
        if self.import_path is not None:
            payload["import"] = self.import_path
        payload["params"] = dict(self.params)
        return payload

    def hash(self) -> str:
        return spec_hash(self.canonical())

    @property
    def is_patch(self) -> bool:
        """A patch names no component: it only carries params for a lower layer's."""
        return self.type is None and self.import_path is None

    def merged_over(self, base: "ComponentSpec | None") -> "ComponentSpec":
        """Layer ``self`` on ``base``. The same ``type``/``import`` (or a patch) merges
        params key by key; a different one replaces the lower layer wholesale, because
        the lower layer's params belong to another component."""
        if base is None:
            return self
        same = (self.type, self.import_path) == (base.type, base.import_path)
        if not (same or self.is_patch):
            return self
        return ComponentSpec(
            kind_id=self.kind_id,
            type=self.type if self.type is not None else base.type,
            import_path=self.import_path if self.import_path is not None else base.import_path,
            params={**dict(base.params), **dict(self.params)},
        )


@dataclass
class Resolved:
    """A built component: the instance, its registry record and the spec it came from."""

    spec: ComponentSpec
    component: Component
    instance: Any
    spec_hash: str

    def fingerprint(self) -> dict[str, Any]:
        return self.component.fingerprint(spec_hash=self.spec_hash)


@dataclass(frozen=True)
class LayerResult:
    """The outcome of :func:`merge_layers`: the merged spec and who decided what.

    ``decided_by`` is the highest layer that named the component (``type`` or ``import``);
    ``param_layers`` maps each parameter to the highest layer that set it. ``instance`` is
    set when layer 9 (``code``) supplied an object, which wins outright and is adopted
    without a build.
    """

    spec: ComponentSpec | None
    decided_by: str
    param_layers: Mapping[str, str]
    instance: Any = None
    trail: tuple[tuple[str, dict[str, Any]], ...] = ()

    @property
    def adopted(self) -> bool:
        return self.instance is not None

    def describe(self) -> dict[str, Any]:
        return {
            "decided_by": self.decided_by,
            "spec": self.spec.canonical() if self.spec is not None else None,
            "params": dict(self.param_layers),
            "adopted": self.adopted,
            "layers": [{"layer": name, "spec": payload} for name, payload in self.trail],
        }


def merge_layers(kind_id: str, layers: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> LayerResult:
    """Merge a slot's value across the layers of section 5.3, lowest precedence first.

    ``layers`` is a mapping or a sequence of ``(layer_name, raw)`` pairs; a ``None`` raw
    value means the layer says nothing. The order of the *sequence* is the precedence when
    given as a sequence; a mapping is ordered by :data:`LAYERS`, and an unknown layer name
    is an error (a typo must not silently lose precedence). The ``code`` layer may hold an
    instance (anything that is not a name, a mapping or a spec): it wins outright.
    """
    if isinstance(layers, Mapping):
        unknown = sorted(set(layers) - set(LAYERS))
        if unknown:
            raise ComponentSpecError(
                kind_id, f"unknown resolution layer(s) {unknown}; layers: {list(LAYERS)}"
            )
        ordered = [(name, layers[name]) for name in LAYERS if name in layers]
    else:
        ordered = [(str(name), raw) for name, raw in layers]
    merged: ComponentSpec | None = None
    decided = ""
    param_layers: dict[str, str] = {}
    trail: list[tuple[str, dict[str, Any]]] = []
    instance: Any = None
    for name, raw in ordered:
        if raw is None:
            continue
        if not isinstance(raw, (str, Mapping, ComponentSpec)):
            # An object: only the in-code layer may hand one over, and it wins outright.
            if name != "code":
                raise ComponentSpecError(
                    kind_id,
                    f"layer {name!r} gave {type(raw).__name__}; only the 'code' layer may "
                    "supply an instance",
                )
            instance = raw
            decided = name
            trail.append((name, {"instance": type(raw).__qualname__}))
            continue
        spec = ComponentSpec.parse(kind_id, raw, allow_patch=merged is not None)
        trail.append((name, spec.canonical()))
        previous = merged
        merged = spec.merged_over(previous)
        if not spec.is_patch:
            decided = name
        if previous is not None and (merged.type, merged.import_path) != (
            previous.type,
            previous.import_path,
        ):
            # A different component replaced the lower layers wholesale: their params go.
            param_layers = {}
        for key in spec.params:
            param_layers[key] = name
    if instance is None and merged is not None and merged.is_patch:
        raise ComponentSpecError(kind_id, "a spec needs 'type' or 'import'")
    return LayerResult(
        spec=merged,
        decided_by=decided,
        param_layers=param_layers,
        instance=instance,
        trail=tuple(trail),
    )


class ComponentResolver:
    """Turns specs into instances through the registries, honouring the import policy."""

    def validate(
        self,
        spec: ComponentSpec,
        *,
        trusted: bool,
        inject: Mapping[str, Any] | None = None,
    ) -> Component:
        """Everything short of building: the name exists, params are acceptable."""
        component = self._component(spec, trusted=trusted)
        offered = frozenset(
            key
            for key in dict(inject or {})
            if key in component.accepted_params()
        )
        clash = sorted(set(spec.params) & offered)
        if clash:
            raise ComponentSpecError(
                spec.kind_id,
                f"E011: {spec.kind_id}:{component.name}: "
                + "; ".join(
                    f"parameter {key!r} is supplied by the framework and cannot be set in a spec"
                    for key in clash
                ),
            )
        problems = component.check_params(dict(spec.params), ignore=offered)
        if problems:
            raise ComponentParamError(spec.kind_id, component.name, problems)
        return component

    def resolve(
        self,
        spec: ComponentSpec,
        *,
        trusted: bool,
        inject: Mapping[str, Any] | None = None,
        check_instance: bool = True,
    ) -> Resolved:
        component = self._component(spec, trusted=trusted)
        instance = HUB.registry(spec.kind_id).build(
            component.name, dict(spec.params), inject=inject, check_instance=check_instance
        )
        return Resolved(spec=spec, component=component, instance=instance, spec_hash=spec.hash())

    def adopt(self, kind_id: str, instance: Any, *, name: str = "instance") -> Resolved:
        """Wrap an in-code instance (highest precedence). It is not in a registry, so it
        is recorded as a ``runtime`` component whose fingerprint carries a source digest."""
        bad = KIND_SPECS[kind_id].instance_contract(instance)
        if bad:
            raise ComponentContractError(kind_id, name, bad)
        factory = type(instance)
        component = Component(
            kind=kind_id,
            name=str(getattr(instance, "name", None) or name),
            factory=factory,
            origin=Origin.RUNTIME,
            mutability=(
                Mutability.STATIC
                if KIND_SPECS[kind_id].default_mutability is Mutability.STATIC
                else Mutability.EPISODE
            ),
            module=factory.__module__,
            qualname=factory.__qualname__,
        )
        spec = ComponentSpec(kind_id=kind_id, type=component.name, params={})
        return Resolved(
            spec=spec,
            component=component,
            instance=instance,
            spec_hash=spec_hash({"adopted": f"{factory.__module__}.{factory.__qualname__}"}),
        )

    # -- internals -------------------------------------------------------
    def _component(self, spec: ComponentSpec, *, trusted: bool) -> Component:
        registry = HUB.registry(spec.kind_id)
        if spec.import_path is not None:
            return self._import(spec, trusted=trusted)
        if spec.type is None:
            raise ComponentSpecError(spec.kind_id, "a spec needs 'type' or 'import'")
        return registry.get(spec.type)

    def _import(self, spec: ComponentSpec, *, trusted: bool) -> Component:
        assert spec.import_path is not None
        module_name, _, attr = spec.import_path.partition(":")
        policy = HUB.effective_policy()
        if not trusted:
            if not policy.allow_import:
                raise ComponentImportDenied(
                    spec.import_path,
                    "imports are disabled on this surface (COMPONENTS.ALLOW_IMPORT is false)",
                )
            if not any(fnmatch.fnmatchcase(module_name, p) for p in policy.import_allowlist):
                raise ComponentImportDenied(
                    spec.import_path,
                    f"module {module_name!r} is not in COMPONENTS.IMPORT_ALLOWLIST "
                    f"{list(policy.import_allowlist)}",
                )
        registry = HUB.registry(spec.kind_id)
        with HUB.loading(Origin.RUNTIME):
            try:
                module = importlib.import_module(module_name)
            except ComponentError:
                raise
            except Exception as error:  # noqa: BLE001 - chained, never swallowed
                raise ComponentBuildError(
                    spec.kind_id, spec.import_path, error
                ) from error
        target: Any = module
        for part in attr.split("."):
            try:
                target = getattr(target, part)
            except AttributeError:
                raise ComponentSpecError(
                    spec.kind_id, f"{module_name!r} has no attribute {attr!r}"
                ) from None
            # The allow-list names modules. An attribute path that walks *through* a module
            # object (``acme.mod:os.system``, ``acme.mod:torch.optim.SGD``) would reach code
            # of a module the operator never allowed, so each module crossed must be allowed.
            if not trusted and isinstance(target, types.ModuleType) and not any(
                fnmatch.fnmatchcase(target.__name__, p) for p in policy.import_allowlist
            ):
                raise ComponentImportDenied(
                    spec.import_path,
                    f"{attr!r} reaches module {target.__name__!r}, which is not in "
                    f"COMPONENTS.IMPORT_ALLOWLIST {list(policy.import_allowlist)}",
                )
        name = spec.type or _snake(attr.rsplit(".", 1)[-1])
        existing = registry.find(name)
        if existing is not None:
            if existing.factory is target:
                return existing
            raise ComponentCollision(
                spec.kind_id,
                name,
                f"already registered to {existing.module}.{existing.qualname}; "
                f"{spec.import_path} cannot take the same name",
            )
        return registry.register(name, target, origin=Origin.RUNTIME, scope=_scope())


def _scope() -> str:
    """Imports register into the active run overlay when there is one, else globally."""
    return "run" if HUB.current_overlay() is not None else "global"


def _snake(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()
