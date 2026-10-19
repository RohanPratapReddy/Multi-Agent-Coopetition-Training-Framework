"""MACTF - Multi-Agent Coopetition Training Framework.

A Django-shaped training framework for arbitrary ``torch.nn.Module`` peers.
Numerical scoring follows the formulations in ``docs/Designing_Multi_Agent_
Incentive_Frameworks.docx``; structural integrity constraints follow the
normative rules in ``docs/framework.md``.
"""

from __future__ import annotations

__version__ = "0.0.0"
VERSION = tuple(int(part) for part in __version__.split("."))

__all__ = ["__version__", "VERSION"]
