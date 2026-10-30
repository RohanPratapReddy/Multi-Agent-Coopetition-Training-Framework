"""The components contrib app: the checks that report the codes of 17.2 before a run starts."""

from __future__ import annotations

from ...apps import AppConfig


class ComponentsConfig(AppConfig):
    name = "mactf.contrib.components"
    label = "mactf_components"
    verbose_name = "MACTF Components"

    def ready(self) -> None:
        from . import systems  # noqa: F401  (registers checks)
