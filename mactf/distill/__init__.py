"""Multi-peer, multi-teacher contrastive distillation (document 14).

Exports are resolved lazily (PEP 562). ``mactf.conf`` imports ``mactf.distill.config``
for the settings defaults, and that import must not pull in ``torch`` or the step
machinery; an eager ``__init__`` would load every submodule on the first
``import mactf.conf``. Attribute access (``from mactf.distill import CoDistillStep``)
imports only the submodule that defines the name.

``mactf.distill`` never imports ``http``, ``contrib.api``, ``client`` or
``orchestration``: it is policy, and the wrappers that reach it are plumbing.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    # config (torch-free)
    "CONFORMANCE_KEYS": "config",
    "CoDistillPolicy": "policy",
    "CONFORMANCE_MODES": "config",
    "DISTILLATION_DEFAULTS": "config",
    "DISTILLATION_KEYS": "config",
    "DistillationConfig": "config",
    "validate_block": "config",
    # batch
    "DistillBatch": "batch",
    "coerce_batch": "batch",
    # label maps
    "LabelMap": "labelmap",
    "build_label_map": "labelmap",
    "shared_axis": "labelmap",
    # views
    "DistributionView": "views",
    "FixedRandomView": "views",
    "HiddenStateView": "views",
    "IdentityView": "views",
    "LinearView": "views",
    "OutputBundle": "views",
    "OutputView": "views",
    "ViewPair": "views",
    "build_view": "views",
    "build_view_pair": "views",
    # teachers
    "Teacher": "teacher",
    "TeacherOutput": "teacher",
    "TeacherSpec": "teacher",
    "build_teacher": "teacher",
    "normalize_output": "teacher",
    # conformance
    "ConformanceVerdict": "conformance",
    "DeclaredAudit": "conformance",
    "GuardAudit": "conformance",
    "TeacherAuditor": "conformance",
    "TrustOnlyAudit": "conformance",
    "TrustProfile": "conformance",
    "TrustState": "conformance",
    "admission": "conformance",
    "build_trust_profile": "conformance",
    # math core
    "ContrastiveReport": "contrastive",
    "contrastive_loss": "contrastive",
    "peer_margin": "contrastive",
    "positive_mask": "contrastive",
    "SigmaState": "sigma",
    "sigma_weights": "sigma",
    "ObjectiveConfig": "objective",
    "PeerLoss": "objective",
    "compound_loss": "objective",
    "cohort_signal": "objective",
    "ramp_factor": "objective",
    # step and state
    "CoDistillStep": "step",
    "PreparedPeer": "step",
    "PreparedStep": "step",
    "DistillationState": "state",
    "DistillResumeReport": "state",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
