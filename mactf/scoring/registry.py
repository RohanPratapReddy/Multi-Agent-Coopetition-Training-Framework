"""The scorer spec registry (document 07 section 10).

Specs are the only way a caller (the API, a settings file, a client) names a
scoring stage. The registry is the closed list of what exists, so
``GET /scorers`` cannot advertise a scorer that is not implemented and a typo
cannot silently train the wrong objective.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..exceptions import ValidationError
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
SCORER_TYPES: dict[str, type[Scorer]] = {
    "reward": RewardScorer,
    "penalty": PenaltyScorer,
    "warning": WarningScorer,
    "hierarchy": HierarchicalScorer,
}


def register_scorer(name: str, scorer: type[Scorer], *, replace: bool = False) -> None:
    """Add a stage type. Applications use this to expose a custom stage.

    ``replace=False`` refuses to overwrite an existing name, because a silent
    override would make a stored spec mean something different after a deploy.
    """
    if not isinstance(scorer, type) or not issubclass(scorer, Scorer):
        raise ValidationError(f"{scorer!r} is not a Scorer subclass")
    if not name or not name.replace("_", "").isalnum():
        raise ValidationError(f"invalid scorer type name {name!r}")
    if name in SCORER_TYPES and not replace:
        raise ValidationError(
            f"scorer type {name!r} is already registered to "
            f"{SCORER_TYPES[name].__name__}; pass replace=True to override"
        )
    SCORER_TYPES[name] = scorer


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


__all__ = [
    "SCORER_TYPES",
    "build_scorer_from_spec",
    "register_scorer",
    "scorer_capabilities",
]
