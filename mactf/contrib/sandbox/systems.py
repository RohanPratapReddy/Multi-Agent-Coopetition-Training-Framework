"""System checks registered by the sandbox app."""

from __future__ import annotations

from ... import checks
from .apps import check_isolation

checks.register(check_isolation, app="mactf_sandbox")
