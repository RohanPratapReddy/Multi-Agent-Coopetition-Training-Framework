"""MACTF sandbox app."""

from __future__ import annotations

from .apps import SandboxConfig

default_app_config = "mactf.contrib.sandbox.apps.SandboxConfig"

__all__ = ["SandboxConfig", "default_app_config"]
