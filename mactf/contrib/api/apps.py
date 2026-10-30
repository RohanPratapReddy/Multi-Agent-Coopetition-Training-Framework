"""``ApiConfig``: declares the app; mounts nothing and opens nothing."""

from __future__ import annotations

from ...apps import AppConfig


class ApiConfig(AppConfig):
    name = "mactf.contrib.api"
    label = "mactf_api"
    verbose_name = "MACTF API"
    urls_module = "mactf.contrib.api.urls"
    checks_module = "mactf.contrib.api.systems"

    def ready(self) -> None:
        # Importing registers the checks and the signal receivers. ``ready`` must not
        # mount routes, read tokens or touch the registry: that all happens lazily, on
        # the first request, so importing settings never has side effects.
        from . import receivers, systems  # noqa: F401
