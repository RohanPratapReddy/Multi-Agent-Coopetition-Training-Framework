"""Registers the API checks. Imported by ``ApiConfig.ready``."""

from __future__ import annotations

from ... import checks
from .checks import check_mounted, check_route_names, check_tokens, check_unprotected

for _check in (check_unprotected, check_mounted, check_route_names, check_tokens):
    checks.register(_check, app="mactf_api", requires="mactf.contrib.api")
