"""App registry and configuration, modeled on Django's ``AppConfig``."""

from __future__ import annotations

import importlib
from typing import Any

from . import signals
from .conf import Settings, get_settings
from .exceptions import AppRegistryNotReady, ImproperlyConfigured

_apps: dict[str, "AppConfig"] = {}
_populated = False
_ready = False


class AppConfig:
    """Base class for a MACTF app.

    Subclasses set ``name``, optionally ``label``, ``verbose_name``, and
    ``default_auto_field``, and may override :meth:`ready`.
    """

    name: str = ""
    label: str | None = None
    verbose_name: str | None = None
    path: str | None = None

    # Doc 13 section 2: declarative, optional. ``None`` means "try the convention".
    urls_module: str | None = None
    migrations_module: str | None = None
    checks_module: str | None = None
    commands_module: str | None = None

    #: Submodules imported by :meth:`import_models`, in order. ``urls`` is last
    #: because it imports ``views``, which imports ``serializers``; a failure should
    #: carry the name of the module that actually failed.
    SUBMODULES = (
        "models", "peers", "providers", "systems", "receivers",
        "views", "serializers", "urls",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.label = self.label or name.rsplit(".", 1)[-1]
        self.verbose_name = self.verbose_name or self.label.replace("_", " ").title()
        module = importlib.import_module(name)
        self.module = module
        self.path = self.path or str(getattr(module, "__file__", ""))

    def __repr__(self) -> str:
        return f"<AppConfig: {self.label}>"

    # -- lifecycle -------------------------------------------------------
    def import_models(self) -> None:
        """Import the conventional submodules (``SUBMODULES``) if present.

        A ``ModuleNotFoundError`` whose ``name`` is the module being imported means
        "this app has no such module"; anything else is a real failure and propagates.
        """
        names = [f"{self.name}.{submodule}" for submodule in self.SUBMODULES]
        if self.checks_module:
            names.insert(0, self.checks_module)
        if self.urls_module:
            names.append(self.urls_module)
        seen: set[str] = set()
        for module_name in names:
            if module_name in seen:
                continue
            seen.add(module_name)
            module = self._import_optional(module_name)
            if module is not None:
                self._register_declared(module)
        self._register_declared(self.module)

    def _import_optional(self, module_name: str) -> Any | None:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # ``exc.name`` is the first missing module on the dotted path: the target
            # itself or one of its absent parents inside this app.
            missing = exc.name or ""
            absent_here = module_name == missing or (
                module_name.startswith(missing + ".") and missing.startswith(self.name)
            )
            if not absent_here and missing != self.name:
                raise
            return None

    # -- declarative contributions (doc 13 section 2) --------------------------
    def get_urls(self) -> list[Any]:
        """The app's ``urlpatterns``. Pure data; no settings reads, no side effects."""
        module = self._import_optional(self.urls_module or f"{self.name}.urls")
        if module is None:
            return []
        return list(getattr(module, "urlpatterns", []))

    def get_migrations(self) -> dict[int, list[str]]:
        """``{version: [sql, ...]}`` from ``<migrations_module>.operations``, or ``{}``.

        ``operations`` may be a mapping or a list of ``(version, statements)`` pairs.
        A list that repeats a version is an ``ImproperlyConfigured``.
        """
        module = self._import_optional(self.migrations_module or f"{self.name}.migrations")
        if module is None:
            return {}
        operations = getattr(module, "operations", None)
        if operations is None:
            return {}
        pairs = operations.items() if isinstance(operations, dict) else operations
        result: dict[int, list[str]] = {}
        for version, statements in pairs:
            version = int(version)
            if version in result:
                raise ImproperlyConfigured(
                    f"app {self.label!r} declares migration version {version} more than once"
                )
            if isinstance(statements, str):
                statements = [statements]
            result[version] = [str(statement) for statement in statements]
        return result

    def get_commands(self) -> str | None:
        """The dotted package holding this app's commands, if it exists."""
        package = self.commands_module or f"{self.name}.management.commands"
        return package if self._import_optional(package) is not None else None

    def _register_declared(self, module: Any) -> None:
        """Register any ``PEERS`` declared by an app module."""
        from .apps import get_peer_registry

        declared = getattr(module, "PEERS", None)
        if not declared:
            return
        registry = get_peer_registry()
        for candidate in declared:
            if isinstance(candidate, type):
                registry.register(candidate, app=self.label)

    def ready(self) -> None:  # noqa: D401 - Django-compatible name
        """Hook run once the registry is populated."""

    def register_peer(self, peer_class: type) -> type:
        return _peers.register(peer_class, app=self.label)

    def register_provider(self, provider_class: type) -> type:
        return _providers.register(provider_class, app=self.label)

    def register_system_check(self, check) -> Any:
        return checks.register(check, app=self.label)


class AppRegistry:
    """Holds the app configs for the current settings object."""

    def __init__(self) -> None:
        self.app_configs: dict[str, AppConfig] = {}
        self._ready = False

    # -- population ------------------------------------------------------
    def populate(self, installed_apps: list[str] | None = None) -> None:
        global _populated
        if self._ready:
            return
        settings = get_settings()
        entries = installed_apps if installed_apps is not None else settings["INSTALLED_APPS"]
        for entry in entries:
            self._load_app(entry)
        self._ready = True
        _populated = True

    def _load_app(self, entry: str) -> None:
        if isinstance(entry, AppConfig):
            config = entry
        elif isinstance(entry, str):
            label_hint = entry.rsplit(".", 1)[-1]
            try:
                module = importlib.import_module(f"{entry}.apps")
            except ModuleNotFoundError as exc:
                if exc.name not in {f"{entry}.apps", entry}:
                    raise
                config = AppConfig.create(entry)
            else:
                config = self._config_from_module(module, label_hint, entry)
        else:
            raise ImproperlyConfigured(f"invalid INSTALLED_APPS entry: {entry!r}")

        if config.label in self.app_configs:
            raise ImproperlyConfigured(
                f"app label {config.label!r} is already registered; set a unique label"
            )
        self.app_configs[config.label] = config

    @staticmethod
    def _config_from_module(module: Any, label_hint: str, entry: str) -> AppConfig:
        candidates = []
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, AppConfig) and value is not AppConfig:
                if getattr(value, "name", None) == entry or value.label == label_hint:
                    candidates.append(value)
        if not candidates:
            candidates = [
                value
                for value in vars(module).values()
                if isinstance(value, type) and issubclass(value, AppConfig) and value is not AppConfig
            ]
        if len(candidates) == 1:
            return candidates[0](entry)
        if not candidates:
            return AppConfig.create(entry)
        raise ImproperlyConfigured(
            f"multiple AppConfig subclasses found in {entry}.apps; import one explicitly"
        )

    # -- readiness -------------------------------------------------------
    def check_apps_ready(self) -> None:
        if not self._ready:
            raise AppRegistryNotReady(
                "Apps are not loaded yet; call mactf.apps.populate() or use manage.py"
            )

    def check_models_ready(self) -> None:
        self.check_apps_ready()

    def get_app_configs(self) -> list[AppConfig]:
        self.check_apps_ready()
        return list(self.app_configs.values())

    def get_app_config(self, label: str) -> AppConfig:
        self.check_apps_ready()
        try:
            return self.app_configs[label]
        except KeyError as exc:
            raise LookupError(f"no installed app with label {label!r}") from exc

    def get_containing_app_config(self, object_name: str) -> AppConfig | None:
        self.check_apps_ready()
        for config in self.app_configs.values():
            if object_name.startswith(config.name + "."):
                return config
        return None

    def set_ready(self) -> None:
        for config in self.app_configs.values():
            config.import_models()
        for config in self.app_configs.values():
            config.ready()
        self._ready = True
        signals.app_ready.send(sender=self)


class _PeerRegistry:
    """Maps peer names to ``nn.Module`` classes discovered across apps."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[type, str]] = {}

    def register(self, peer_class: type, *, app: str = "unknown", name: str | None = None) -> type:
        key = name or peer_class.__name__
        if key in self._entries and self._entries[key][0] is not peer_class:
            raise ImproperlyConfigured(
                f"a different peer is already registered as {key!r} "
                f"(from app {self._entries[key][1]!r})"
            )
        self._entries[key] = (peer_class, app)
        return peer_class

    def discover(self, module_or_object: Any, *, app: str = "unknown") -> list[type]:
        """Register every ``nn.Module`` declared in ``PEERS`` or defined in the module."""
        import torch.nn as nn

        found: list[type] = []
        declared = getattr(module_or_object, "PEERS", None) or []
        module_name = getattr(module_or_object, "__name__", None)
        candidates = [candidate for candidate in declared if isinstance(candidate, type)]
        for value in vars(module_or_object).values():
            if (
                isinstance(value, type)
                and issubclass(value, nn.Module)
                and value.__module__ == module_name
                and value not in candidates
            ):
                candidates.append(value)
        for candidate in candidates:
            if not issubclass(candidate, nn.Module):
                continue
            self.register(candidate, app=app)
            found.append(candidate)
        return found

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    def get(self, name: str) -> type:
        try:
            return self._entries[name][0]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries)) or "none"
            raise LookupError(f"unknown peer {name!r}; registered peers: {available}") from exc

    def all(self) -> dict[str, type]:
        return {name: entry[0] for name, entry in self._entries.items()}

    def clear(self) -> None:
        self._entries.clear()


class _ProviderRegistry:
    """Maps sandbox provider names to classes."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[type, str]] = {}

    def register(self, provider_class: type, *, app: str = "unknown") -> type:
        key = getattr(provider_class, "name", provider_class.__name__).lower()
        self._entries[key] = (provider_class, app)
        return provider_class

    def get(self, name: str) -> type:
        try:
            return self._entries[name.lower()][0]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries)) or "none"
            raise LookupError(f"unknown sandbox provider {name!r}; available: {available}") from exc

    def all(self) -> dict[str, type]:
        return {name: entry[0] for name, entry in self._entries.items()}

    def clear(self) -> None:
        self._entries.clear()


def _ensure_appconfig_create() -> None:
    if not hasattr(AppConfig, "create"):
        @classmethod  # type: ignore[misc]
        def create(cls, entry: Any) -> AppConfig:
            if isinstance(entry, AppConfig):
                return entry
            return cls(entry)

        AppConfig.create = create  # type: ignore[attr-defined]


_ensure_appconfig_create()

apps = AppRegistry()
_peers = _PeerRegistry()
_providers = _ProviderRegistry()


def get_peer_registry() -> _PeerRegistry:
    return _peers


def get_provider_registry() -> _ProviderRegistry:
    return _providers


def populate(installed_apps: list[str] | None = None) -> None:
    """Populate and ready the app registry. Idempotent."""
    global _ready
    if _ready:
        return
    apps.populate(installed_apps)
    apps.set_ready()
    _ready = True


def reset_registry() -> None:
    """Clear the registry. Used by tests and :func:`mactf.conf.configure`."""
    global _ready, _populated
    apps.app_configs.clear()
    apps._ready = False
    _ready = False
    _populated = False
    _peers.clear()
    _providers.clear()


from . import checks  # noqa: E402  (avoid circular import at module import time)
