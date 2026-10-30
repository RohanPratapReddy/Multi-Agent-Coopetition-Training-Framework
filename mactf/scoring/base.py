"""The scoring stage protocol: :class:`Scorer` and its value objects.

Scoring is split into stages rather than one function so each part of
framework.md has exactly one owner:

===============  =======  ====================================================
Stage            Order    Owns
===============  =======  ====================================================
``WarningScorer``      5    the W1-W9 regime; the only stateful stage
``RewardScorer``      10    ``R = a*r_task + b*r_epi + g*r_collab + d*r_align``
``PenaltyScorer``     20    ``P = sum(lambda_k * p_k) * Omega``
``ScoringPipeline``    0    ordering, precedence, compliance, the scorecard
===============  =======  ====================================================

Two design points matter and are easy to get wrong:

* :class:`ScoreContext` is frozen, and a stage that needs to change it returns a
  new one through :meth:`ScoreContext.evolve`. "Warnings run before penalties" is
  therefore visible in the type instead of being a comment that a future refactor
  can quietly break.
* :meth:`Scorer.prepare` is separate from :meth:`Scorer.score`. Warning detection
  has to hand its result to the penalty stage; merging the two methods is exactly
  what produced the ordering hazard this split removes.

No stage builds a ``PeerScorecard``. That belongs to the pipeline alone, so a
custom stage cannot invent fields or skip compliance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Mapping

from ..exceptions import ValidationError
from ..rules.severity import Severity
from .ledger import Ledger
from .warning import WarningPolicy


class ScoringKind(str, Enum):
    """What a stage computes. Used for chain validation and for the API."""

    REWARD = "reward"
    PENALTY = "penalty"
    WARNING = "warning"
    PIPELINE = "pipeline"
    HIERARCHY = "hierarchy"


@dataclass(frozen=True)
class WarningMutation:
    """One state transition a :class:`~mactf.scoring.warning_scorer.WarningScorer`
    applied to a registry.

    Reporting mutations rather than exposing the registry is what keeps the
    warning stage the only writer: a caller can audit exactly what changed
    without being able to change it.
    """

    peer: str
    rule_id: str
    episode: int
    action: str  # "issue" | "escalate" | "expire" | "clean"
    severity: Severity | None = None
    evidence: str = ""
    penalty: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "peer": self.peer,
            "rule_id": self.rule_id,
            "episode": self.episode,
            "action": self.action,
            "severity": self.severity.label if self.severity is not None else None,
            "evidence": self.evidence,
            "penalty": self.penalty,
        }

    def describe(self) -> str:
        """One-line rendering, matching ``WarningRecord.describe``.

        ``PeerScorecard.warnings_issued`` is typed as ``WarningRecord`` and
        ``PeerScorecard.to_dict`` calls ``describe()`` on every entry; the
        pipeline actually stores mutations, so the two must render alike for
        the scorecard's serialized form to stay stable.
        """
        label = self.severity.label if self.severity is not None else "?"
        text = (
            f"warning[{self.rule_id}] ep={self.episode} sev={label} {self.action}"
        )
        if self.penalty:
            text += f" penalty={self.penalty:.3f}"
        if self.evidence:
            text += f" ({self.evidence})"
        return text


@dataclass(frozen=True)
class ScoreContext:
    """Everything a scorer may read.

    Frozen so a stage cannot mutate its input. Stages return a new context with
    :meth:`evolve`, which is what makes the pipeline's ordering explicit.
    """

    peer: str
    episode: int
    ledger: Ledger
    node_path: str = ""
    depth: int = 0
    config: Mapping[str, Any] = field(default_factory=dict)
    severity_weights: Mapping[str, Any] = field(default_factory=dict)
    warning_policy: WarningPolicy = field(default_factory=WarningPolicy)
    known_rules: frozenset[str] = frozenset()
    warning_total: float = 0.0
    warning_events: tuple[str, ...] = ()
    escalated_rules: frozenset[str] = frozenset()
    children: tuple["ScorerOutput", ...] = ()
    branch_void: frozenset[str] = frozenset()
    extras: Mapping[str, Any] = field(default_factory=dict)

    def evolve(self, **changes: Any) -> "ScoreContext":
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)

    # -- convenience readers --------------------------------------------
    @property
    def warning_only_rules(self) -> frozenset[str]:
        """Rules that produced a warning this episode and did not escalate.

        framework.md W1: these score zero, so both the penalty stage and the
        points projection exclude them. Computed in one place so the two can
        never disagree.
        """
        return frozenset(self.warning_events) - frozenset(self.escalated_rules)

    @property
    def allowed_rules(self) -> frozenset[str] | None:
        """The rules the controlling precedence tier admits, or ``None`` for all."""
        allowed = self.extras.get("allowed_rules")
        if allowed is None:
            return None
        return frozenset(allowed)


@dataclass(frozen=True)
class ScorerOutput:
    """The result of one scoring stage."""

    kind: ScoringKind
    amount: float = 0.0
    breakdown: Mapping[str, Any] = field(default_factory=dict)
    rules: tuple[str, ...] = ()
    mutations: tuple[WarningMutation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    void_rules: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "amount": self.amount,
            "breakdown": dict(self.breakdown),
            "rules": list(self.rules),
            "metadata": dict(self.metadata),
            "void_rules": list(self.void_rules),
        }


@dataclass(frozen=True)
class ScorerDiagnostic:
    """A problem found by :meth:`Scorer.validate`, reportable as a 422."""

    level: str  # "info" | "warning" | "error"
    code: str
    message: str
    hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }


class Scorer(ABC):
    """Base class for every scoring stage.

    Subclasses implement exactly one job. The pipeline owns ordering, the
    registry, compliance, and the scorecard.
    """

    kind: ClassVar[ScoringKind]
    order: ClassVar[int] = 100
    requires: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    # -- lifecycle -------------------------------------------------------
    def prepare(self, ctx: ScoreContext) -> ScoreContext:
        """Return the context the following stages will see.

        Only :class:`~mactf.scoring.warning_scorer.WarningScorer` overrides this,
        because only warning detection can change what later stages must know.
        """
        return ctx

    @abstractmethod
    def score(self, ctx: ScoreContext) -> ScorerOutput:
        """Compute this stage's contribution."""

    def finalize(self, ctx: ScoreContext, output: ScorerOutput) -> ScorerOutput:
        """Post-process an output. The default is a no-op."""
        return output

    # -- introspection ---------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "Scorer":
        """Build a stage from ``{"type": <registry key>, **kwargs}``.

        Resolved through the ``scorer`` component registry, so a stage registered by a
        plugin or a project is addressable here exactly like a built-in one. Unknown keys
        and unknown types raise :class:`~mactf.exceptions.ValidationError` listing what is
        accepted, rather than silently constructing the wrong objective.
        """
        from ..components import HUB
        from ..components.resolver import ComponentResolver, ComponentSpec

        payload = dict(spec or {})
        if payload.get("type") is None and payload.get("import") is None:
            raise ValidationError(
                f"{cls.__name__}.from_spec requires a 'type' key; accepted types: "
                + ", ".join(HUB.registry("scorer").names())
            )
        resolved = ComponentResolver().resolve(
            ComponentSpec.parse("scorer", payload), trusted=True
        )
        stage = resolved.instance
        if not isinstance(stage, cls) and cls is not Scorer:
            raise ValidationError(
                f"spec type {resolved.component.name!r} resolves to {type(stage).__name__}, "
                f"which is not a {cls.__name__}"
            )
        return stage

    @classmethod
    def _build_from_kwargs(cls, kwargs: Mapping[str, Any]) -> "Scorer":
        """Construct ``cls`` with ``kwargs``, rejecting anything ``__init__`` cannot take."""
        from ..components.reflect import reflect

        accepted = reflect(cls).keys
        unknown = sorted(set(kwargs) - accepted)
        if unknown:
            raise ValidationError(
                f"{cls.__name__} does not accept {', '.join(repr(key) for key in unknown)}; "
                f"accepted keys: {', '.join(sorted(accepted)) or '(none)'}"
            )
        return cls(**dict(kwargs))

    def validate(self, settings: Mapping[str, Any]) -> list[ScorerDiagnostic]:
        """Return configuration problems. Empty means healthy."""
        return []

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "class": type(self).__name__,
            "order": self.order,
            "requires": sorted(self.requires),
            "config_keys": sorted(self.config),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, "config": dict(self.config)}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("class") not in (None, type(self).__name__):
            raise ValueError(
                f"{type(self).__name__} cannot load state written by "
                f"{payload.get('class')!r}"
            )
        self.config = dict(payload.get("config", {}))

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} kind={self.kind.value} order={self.order}>"
        )


def severity_fraction(
    severity: Severity, table: Mapping[str, Any], *, default: float = 1.0
) -> float:
    """Look up a severity's multiplier in a table keyed by ``S0``..``S4``.

    A missing label falls back to ``default`` rather than raising: the docx path
    behaves that way today (``continuous.py:130-132``) and a custom table that
    omits a tier should scale by one, not fail a run.
    """
    value = table.get(severity.label)
    return default if value is None else float(value)


def _constructor_keys(cls: type) -> frozenset[str]:
    """The keyword names ``cls.__init__`` accepts. Deprecated alias of ``reflect(cls).keys``.

    ``mactf.components.reflect`` implements the same MRO walk (document 17 section 3.3);
    this stays for callers that imported the private name.
    """
    from ..components.reflect import reflect

    return frozenset(reflect(cls).keys)
