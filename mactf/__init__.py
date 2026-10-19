"""MACTF - Multi-Agent Coopetition Training Framework.

A Django-shaped training framework for arbitrary ``torch.nn.Module`` peers.
Numerical scoring follows the formulations in ``docs/Designing_Multi_Agent_
Incentive_Frameworks.docx``; structural integrity constraints follow the
normative rules in ``docs/framework.md``.
"""

from __future__ import annotations

import re

# Single source of truth: pyproject.toml resolves this dynamically and the
# release workflow gates the git tag against it.
__version__ = "0.0.0rc1"

# Numeric release segment only, so pre-release suffixes survive: "1.2.3rc1" -> (1, 2, 3).
_RELEASE = re.match(r"\d+(?:\.\d+)*", __version__)
VERSION = tuple(int(part) for part in _RELEASE.group(0).split(".")) if _RELEASE else ()

__all__ = ["__version__", "VERSION"]
