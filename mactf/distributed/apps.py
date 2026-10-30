"""``DistributedConfig``: declares the app, validates ``DISTRIBUTED`` and mounts nothing."""

from __future__ import annotations

from ..apps import AppConfig


class DistributedConfig(AppConfig):
    name = "mactf.distributed"
    label = "distributed"
    verbose_name = "MACTF distributed runs"
    urls_module = "mactf.distributed.urls"
    checks_module = "mactf.distributed.systems"

    def ready(self) -> None:
        from . import systems  # noqa: F401 - registers the checks
