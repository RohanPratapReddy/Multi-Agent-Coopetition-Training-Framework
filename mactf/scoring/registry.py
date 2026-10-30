"""The scorer spec registry (document 07 section 10).

Specs are the only way a caller (the API, a settings file, a client) names a
scoring stage. The registry is the closed list of what exists, so
``GET /scorers`` cannot advertise a scorer that is not implemented and a typo
cannot silently train the wrong objective.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..components.decorators import register_scorer
from ..components.legacy import LegacyTable
from .base import Scorer
from .hierarchy import HierarchicalScorer
from .penalty import PenaltyScorer
from .reward_scorer import RewardScorer
from .warning_scorer import WarningScorer

#: ``spec["type"]`` -> the class it constructs.
#:
#: ``hierarchy`` is accepted so ``ScorerSpec(type="hierarchy")`` resolves in the
#: API, but it is only valid as the outermost stage of a hierarchical run: it
#: scores a whole tree, so it cannot sit in a per-node chain (document 07
#: section 10). ``ScoringPipeline._validate_chain`` rejects it anywhere else.
#:
#: A live view of the ``scorer`` registry of ``mactf.components`` (document 17 section 3.6):
#: reads are plain dict reads, writes go through the registry's collision, origin and lock
#: rules.
SCORER_TYPES: dict[str, type[Scorer]] = LegacyTable(
    "scorer",
    {
        "reward": RewardScorer,
        "penalty": PenaltyScorer,
        "warning": WarningScorer,
        "hierarchy": HierarchicalScorer,
    },
)


def build_scorer_from_spec(spec: Mapping[str, Any]) -> Scorer:
    """Construct a single stage from ``{"type": <key>, **kwargs}``."""
    return Scorer.from_spec(spec)


def scorer_capabilities() -> list[dict[str, Any]]:
    """The machine-readable catalog backing ``GET /scorers``."""
    payload: list[dict[str, Any]] = []
    for name in sorted(SCORER_TYPES):
        stage = SCORER_TYPES[name]
        payload.append(
            {
                "type": name,
                "class": stage.__name__,
                "kind": stage.kind.value,
                "order": stage.order,
                "requires": sorted(stage.requires),
                # ``hierarchy`` is only valid outermost; the flag tells the API
                # and a spec validator where it may appear.
                "root_only": stage.kind.value == "hierarchy",
            }
        )
    return payload


# Built-ins registered by their own modules (document 17 sections 6.3 and 6.8):
# ``composed_reward``, ``zero_reward``, ``no_warnings`` and the ``reward_term`` kind.
from . import null_stages as _null_stages  # noqa: E402,F401
from . import reward_terms as _reward_terms  # noqa: E402,F401

__all__ = [
    "SCORER_TYPES",
    "build_scorer_from_spec",
    "register_scorer",
    "scorer_capabilities",
]
