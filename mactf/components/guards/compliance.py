"""ComplianceGuard and ReadOnlyLedger (document 17 section 6.4).

The guard sits around every scoring stage the framework did not write. It never trusts a
stage's number: it re-derives what a breach must cost from the one penalty implementation
(``PenaltyScorer``, hardened) over the same prepared context and compares. The guarantee is
one-sided on purpose (6.4.1): a stage may charge more, never less, and may not hide that a
non-offsettable rule applied.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ...compliance import DEFAULT_MIN_FLOOR_LAMBDA, effective_non_offsettable
from ...exceptions import ComplianceGuardViolation, ImproperlyConfigured, LedgerReadOnlyError
from ...rules.catalog import NEVER_WARNING_FIRST
from ...scoring.base import ScoreContext, Scorer, ScorerOutput, ScoringKind
from ...scoring.ledger import Ledger, RuleEvent
from ...scoring.penalty import PenaltyScorer
from ...scoring.reward_scorer import BUCKETS, RewardScorer
from ...scoring.warning_scorer import WarningScorer

log = logging.getLogger("mactf.components.guards.compliance")

GUARD_MODES: tuple[str, ...] = ("enforce", "fallback")

#: Violations that no mode may repair: there is no reference to substitute for them.
HARD_CODES: frozenset[str] = frozenset({"C006", "C007"})

#: Context fields no stage may change in ``prepare``, except the ones named per kind.
_PROTECTED_FIELDS: tuple[str, ...] = (
    "peer", "episode", "node_path", "depth", "config", "severity_weights",
    "warning_policy", "known_rules", "branch_void",
)
_PROTECTED_EXTRAS: tuple[str, ...] = ("allowed_rules", "precedence")
_WARNING_OWNED_FIELDS: tuple[str, ...] = ("warning_total", "warning_events", "escalated_rules")
_WARNING_OWNED_EXTRAS: tuple[str, ...] = ("warning_mutations",)

_BUILTIN_STAGES: tuple[type, ...] = (PenaltyScorer, RewardScorer, WarningScorer)


def _number(value: Any) -> float | None:
    """``value`` as a finite float, or ``None``. Booleans are not numbers here."""
    if isinstance(value, (bool, str, bytes)):
        return None             # "5.0" converts with float() but breaks the arithmetic downstream
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _non_negative(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0.0 else None


_PRIMITIVES: tuple[type, ...] = (str, int, float, bool, bytes, type(None))


class _Snapshot(tuple):
    """A ledger snapshot: ``(event, field values)`` pairs plus the very list that held them."""

    container: list = []


def _is_primitive(value: Any) -> bool:
    return any(type(value) is kind for kind in _PRIMITIVES)


def _same(left: Any, right: Any, _depth: int = 0) -> bool:
    """Equality that never raises and never asks the *candidate* to vouch for itself.

    A stage can return a ``dict``/``frozenset``/event subclass whose ``__eq__`` always says
    ``True`` (Python tries the right operand's reflected ``__eq__`` first when its type is a
    subclass), so ``==`` alone proves nothing about a value a stage touched. Here the types
    must be identical and containers are compared element by element with the same rule; a
    value that cannot be compared has changed.
    """
    if _depth > 24:
        return False
    try:
        if type(left) is not type(right):
            return False
        if _is_primitive(left):
            return bool(left == right) or (left != left and right != right)  # NaN is NaN
        if isinstance(left, dict):
            if len(left) != len(right):
                return False
            for key, value in dict.items(left):
                if not _is_primitive(key) and type(key) is not tuple:
                    return bool(left == right)
                if key not in right or not _same(value, right[key], _depth + 1):
                    return False
            return True
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(
                _same(a, b, _depth + 1) for a, b in zip(left, right)
            )
        if isinstance(left, (set, frozenset)):
            if not all(_is_primitive(item) for item in left):
                return bool(left == right)
            return len(left) == len(right) and all(item in right for item in left)
        if dataclasses.is_dataclass(left) and not isinstance(left, type):
            return all(
                _same(getattr(left, f.name), getattr(right, f.name), _depth + 1)
                for f in dataclasses.fields(left)
            )
        return bool(left == right)
    except Exception:  # noqa: BLE001 - e.g. an array-valued extra; treat as different
        return False


def _plain(value: Any, _depth: int = 0) -> Any:
    """A one-time snapshot of ``value`` made of exact builtin types.

    The guard validates what it reads, so the pipeline must be handed exactly what was
    validated: an object whose ``__float__`` (or a mapping whose ``__getitem__``) answers
    differently the second time would otherwise pass the check and charge less afterwards.
    Numbers become plain ``float``/``int``; anything else with a ``__float__`` becomes a
    ``float`` once; unknown objects are kept (the inspectors then reject a non-number).
    """
    if _depth > 24:
        raise ValueError("nesting too deep")
    if isinstance(value, bool):
        return bool(int.__int__(value))
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, str):
        return str.__add__(value, "")
    if value is None or isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        return {
            _plain(key, _depth + 1): _plain(item, _depth + 1) for key, item in list(value.items())
        }
    if isinstance(value, (list, tuple)):
        items = [_plain(i, _depth + 1) for i in list(value)]
        return items if isinstance(value, list) else tuple(items)
    if isinstance(value, (set, frozenset)):
        return frozenset(_plain(i, _depth + 1) for i in value)
    try:
        return float(value)
    except Exception:  # noqa: BLE001 - not a number: kept so the inspectors can refuse it
        return value


def _ids(value: Any) -> set[str]:
    """``value`` as a set of rule ids; a non-iterable is the empty set."""
    try:
        return {str(item) for item in value}
    except TypeError:
        return set()


@dataclass(frozen=True)
class GuardPolicy:
    """Tunables of the guard. ``from_settings`` reads the ``COMPLIANCE`` block."""

    mode: str = "enforce"
    min_floor_lambda: float = DEFAULT_MIN_FLOOR_LAMBDA
    reward_ceiling: float = 1.0e4
    tolerance: float = 1.0e-9
    trust_builtin: bool = True

    def __post_init__(self) -> None:
        if self.mode not in GUARD_MODES:
            raise ImproperlyConfigured(
                f"COMPLIANCE.GUARD_MODE must be one of {list(GUARD_MODES)}, got {self.mode!r}"
            )
        if not math.isfinite(self.min_floor_lambda) or self.min_floor_lambda <= 0.0:
            raise ImproperlyConfigured(
                f"COMPLIANCE.MIN_FLOOR_LAMBDA must be finite and > 0, got {self.min_floor_lambda!r}"
            )
        if not math.isfinite(self.reward_ceiling) or self.reward_ceiling <= 0.0:
            raise ImproperlyConfigured(
                f"COMPLIANCE.REWARD_CEILING must be finite and > 0, got {self.reward_ceiling!r}"
            )
        if not math.isfinite(self.tolerance) or not 0.0 <= self.tolerance <= 1.0e-3:
            raise ImproperlyConfigured(
                f"COMPLIANCE.TOLERANCE must be within [0, 1e-3], got {self.tolerance!r}"
            )

    @classmethod
    def from_settings(cls, settings: Any) -> "GuardPolicy":
        block: Mapping[str, Any] = {}
        if hasattr(settings, "get"):
            block = settings.get("COMPLIANCE") or {}
        return cls(
            mode=str(block.get("GUARD_MODE", "enforce")).lower(),
            min_floor_lambda=float(block.get("MIN_FLOOR_LAMBDA", DEFAULT_MIN_FLOOR_LAMBDA)),
            reward_ceiling=float(block.get("REWARD_CEILING", 1.0e4)),
            tolerance=float(block.get("TOLERANCE", 1.0e-9)),
            trust_builtin=bool(block.get("TRUST_BUILTIN", True)),
        )


@dataclass(frozen=True)
class GuardViolation:
    code: str
    stage: str
    kind: str
    message: str
    rule_id: str | None = None
    observed: float | None = None
    required: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "stage": self.stage, "kind": self.kind,
            "rule_id": self.rule_id, "message": self.message,
            "observed": self.observed, "required": self.required,
        }


@dataclass(frozen=True)
class GuardReport:
    mode: str
    violations: tuple[GuardViolation, ...] = ()
    substituted: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "violations": [item.as_dict() for item in self.violations],
            "substituted": list(self.substituted),
        }


@dataclass(frozen=True)
class Pristine:
    """A deep copy of the protected context fields, taken before any stage ran.

    ``ScoreContext`` is frozen, but its mappings are not: a stage that edits
    ``ctx.config["PENALTY_LAMBDAS"]`` in place changes nothing ``prepare``'s return value
    would show. Comparing against this copy catches it, and the reference is computed from
    it, so a stage cannot weaken the very yardstick it is measured by.
    """

    fields: Mapping[str, Any]
    extras: Mapping[str, Any]

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.fields["config"])


class ReadOnlyLedger(Ledger):
    """A live, read-only view of a ledger.

    It *is a* ``Ledger`` (so ``isinstance`` checks and every query method in stages
    keep working) that shares the source's event list and refuses to change it. The
    guard additionally compares the event tuple before and after every stage (C007), so
    a stage reaching for ``_events`` is caught rather than trusted.
    """

    def __init__(self, source: Ledger) -> None:
        super().__init__(peer=source.peer)
        self._events = source._events  # shared on purpose: a view, not a copy
        self._source = source

    def unwrap(self) -> Ledger:
        return self._source

    def record(self, *args: Any, **kwargs: Any) -> RuleEvent:
        raise LedgerReadOnlyError("record")

    def extend(self, events: Iterable[RuleEvent]) -> None:
        raise LedgerReadOnlyError("extend")

    def clear(self) -> None:
        raise LedgerReadOnlyError("clear")


class ComplianceGuard:
    """Checks every stage's output against the reference floor. See 6.4.1."""

    def __init__(self, policy: GuardPolicy | None = None) -> None:
        self.policy = policy or GuardPolicy()

    # -- ledger ----------------------------------------------------------
    @staticmethod
    def view(ledger: Ledger) -> ReadOnlyLedger:
        return ledger if isinstance(ledger, ReadOnlyLedger) else ReadOnlyLedger(ledger)

    @staticmethod
    def _state(event: RuleEvent) -> tuple[Any, ...]:
        return tuple(object.__getattribute__(event, f.name) for f in dataclasses.fields(event))

    @staticmethod
    def snapshot(ledger: Ledger) -> "_Snapshot":
        """Every event with its field values: frozen is a convention, ``object.__setattr__``
        can still rewrite a magnitude in place, which an identity/equality check cannot see."""
        source = ledger.unwrap() if isinstance(ledger, ReadOnlyLedger) else ledger
        container = source._events
        taken = _Snapshot((event, ComplianceGuard._state(event)) for event in list(container))
        taken.container = container
        return taken

    def _trusted(self, stage: Scorer) -> bool:
        # identity, never ``in``: a metaclass may define ``__eq__``/``__hash__`` to look like a built-in
        return self.policy.trust_builtin and any(type(stage) is t for t in _BUILTIN_STAGES)

    # -- the pristine context --------------------------------------------
    @staticmethod
    def pristine(ctx: ScoreContext) -> Pristine:
        """Deep-copy the protected fields of ``ctx`` (call before the first stage)."""
        fields = {name: copy.deepcopy(getattr(ctx, name)) for name in _PROTECTED_FIELDS}
        extras = {key: copy.deepcopy(ctx.extras.get(key)) for key in _PROTECTED_EXTRAS}
        return Pristine(fields, extras)

    # -- structural checks (never substitutable) -------------------------
    def check_prepare(
        self,
        stage: Scorer,
        before: ScoreContext,
        after: ScoreContext,
        source: Ledger,
    ) -> list[GuardViolation]:
        name, kind = type(stage).__name__, stage.kind.value
        found: list[GuardViolation] = []
        # Identity, not isinstance: a stage's object can claim any ``__class__``.
        if (
            after.ledger is not before.ledger
            or type(after.ledger) is not ReadOnlyLedger
            or after.ledger._source is not source
            or after.ledger._events is not source._events
        ):
            found.append(GuardViolation(
                "C007", name, kind, "prepare() replaced the ledger it was given"))
        fields = list(_PROTECTED_FIELDS)
        extras = list(_PROTECTED_EXTRAS)
        if stage.kind is not ScoringKind.WARNING:
            fields += _WARNING_OWNED_FIELDS
            extras += _WARNING_OWNED_EXTRAS
        for field_name in fields:
            if not _same(getattr(before, field_name), getattr(after, field_name)):
                found.append(GuardViolation(
                    "C006", name, kind,
                    f"prepare() changed ScoreContext.{field_name}, which this stage does not own"))
        for key in extras:
            if not _same(before.extras.get(key), after.extras.get(key)):
                found.append(GuardViolation(
                    "C006", name, kind,
                    f"prepare() changed ScoreContext.extras[{key!r}], which this stage does not own"))
        return found

    def check_pristine(
        self, stage: Scorer, ctx: ScoreContext, pristine: Pristine
    ) -> list[GuardViolation]:
        """Protected fields still equal their pre-walk copy (catches in-place edits)."""
        name, kind = type(stage).__name__, stage.kind.value
        found: list[GuardViolation] = []
        for field_name, original in pristine.fields.items():
            if not _same(getattr(ctx, field_name), original):
                found.append(GuardViolation(
                    "C006", name, kind,
                    f"the stage changed ScoreContext.{field_name}, which no stage may change"))
        for key, original in pristine.extras.items():
            if not _same(ctx.extras.get(key), original):
                found.append(GuardViolation(
                    "C006", name, kind,
                    f"the stage changed ScoreContext.extras[{key!r}], which no stage may change"))
        return found

    @staticmethod
    def _same_state(event: RuleEvent, state: tuple[Any, ...]) -> bool:
        return all(
            _same(now, then)
            for now, then in zip(ComplianceGuard._state(event), state)
        )

    def check_after_stage(
        self, stage: Scorer, source: Ledger, snapshot: "_Snapshot"
    ) -> list[GuardViolation]:
        """C007 when the ledger's events changed; the original events are put back."""
        current = list(source._events)
        unchanged = source._events is snapshot.container and len(current) == len(snapshot) and all(
            now is event and self._same_state(now, state)
            for now, (event, state) in zip(current, snapshot)
        )
        if not unchanged:
            changed = len(current)
            # Undo the tampering so the caller's ledger is what it was before the walk.
            for event, state in snapshot:
                for field_def, value in zip(dataclasses.fields(event), state):
                    object.__setattr__(event, field_def.name, value)
            snapshot.container[:] = [event for event, _ in snapshot]
            source._events = snapshot.container
            return [GuardViolation(
                "C007", type(stage).__name__, stage.kind.value,
                f"the ledger changed while {type(stage).__name__} ran "
                f"({len(snapshot)} -> {changed} events)")]
        return []

    # -- the reference ---------------------------------------------------
    def reference_penalty(self, ctx: ScoreContext, config: Mapping[str, Any]) -> ScorerOutput:
        """The built-in penalty stage over ``ctx``, with the floor forced to its minimum."""
        hardened = dict(config)
        hardened["NON_OFFSETTABLE"] = sorted(
            effective_non_offsettable(config.get("NON_OFFSETTABLE"))
        )
        hardened["MIN_FLOOR_LAMBDA"] = max(
            self.policy.min_floor_lambda,
            float(config.get("MIN_FLOOR_LAMBDA", self.policy.min_floor_lambda)),
        )
        return PenaltyScorer(config=hardened).score(ctx)

    # -- per-kind inspection --------------------------------------------
    def _inspect_penalty(
        self, stage: Scorer, output: ScorerOutput, ctx: ScoreContext,
        config: Mapping[str, Any], reference: ScorerOutput,
    ) -> list[GuardViolation]:
        name, kind, tol = type(stage).__name__, stage.kind.value, self.policy.tolerance
        found: list[GuardViolation] = []
        breakdown = output.breakdown
        if not isinstance(breakdown, Mapping):
            return [GuardViolation("C001", name, kind, "breakdown is not a mapping")]
        raw_amounts = breakdown.get("amounts", {})
        amounts: dict[str, float] = {}
        if not isinstance(raw_amounts, Mapping):
            found.append(GuardViolation("C001", name, kind, "breakdown['amounts'] is not a mapping"))
            raw_amounts = {}
        for rule_id, value in raw_amounts.items():
            number = _non_negative(value)
            if number is None:
                found.append(GuardViolation(
                    "C001", name, kind,
                    f"amounts[{rule_id!r}] is {value!r}; a penalty is finite and non-negative",
                    rule_id=str(rule_id)))
            else:
                amounts[str(rule_id)] = number
        if _non_negative(output.amount) is None:
            found.append(GuardViolation(
                "C001", name, kind, f"amount is {output.amount!r}; expected finite and >= 0"))
        if output.mutations:
            found.append(GuardViolation(
                "C006", name, kind, "a PENALTY stage returned warning mutations"))

        ref_amounts: Mapping[str, float] = reference.breakdown["amounts"]
        for rule_id in sorted(set(map(str, raw_amounts)) - set(ref_amounts)):
            found.append(GuardViolation(
                "C009", name, kind,
                f"amount for {rule_id}, which is not an admitted penalty event of this ledger "
                "(absent, outside the controlling tier, or warning-only)", rule_id=rule_id))

        no = effective_non_offsettable(config.get("NON_OFFSETTABLE"))
        for rule_id in sorted(no & set(ref_amounts)):
            required = float(ref_amounts[rule_id])
            observed = amounts.get(rule_id, 0.0)
            if observed + tol < required:
                found.append(GuardViolation(
                    "C002", name, kind,
                    f"{rule_id} is non-offsettable and its floor is {required:g}; the stage "
                    f"charged {observed:g}", rule_id=rule_id, observed=observed, required=required))

        omega = _number(breakdown.get("omega", 1.0))
        omega_ref = float(reference.breakdown["omega"])
        if omega is None or omega < 1.0 - tol:
            found.append(GuardViolation(
                "C003", name, kind,
                f"omega is {breakdown.get('omega')!r}; Omega is an amplifier and is >= 1",
                observed=omega, required=1.0))
        elif omega + tol < omega_ref:
            found.append(GuardViolation(
                "C003", name, kind,
                f"omega {omega:g} is below the reference {omega_ref:g} for the same rules",
                observed=omega, required=omega_ref))

        missing = sorted((no & set(ref_amounts)) - _ids(output.void_rules))
        if missing:
            found.append(GuardViolation(
                "C004", name, kind,
                f"void_rules omits {missing}; a non-offsettable breach must void the episode"))
        return found

    def _inspect_reward(
        self, stage: Scorer, output: ScorerOutput, ctx: ScoreContext
    ) -> list[GuardViolation]:
        name, kind, tol = type(stage).__name__, stage.kind.value, self.policy.tolerance
        found: list[GuardViolation] = []
        breakdown = output.breakdown
        if not isinstance(breakdown, Mapping):
            return [GuardViolation("C001", name, kind, "breakdown is not a mapping")]
        if output.mutations or output.void_rules:
            found.append(GuardViolation(
                "C006", name, kind, "a REWARD stage returned mutations or void_rules"))
        if _non_negative(output.amount) is None:
            found.append(GuardViolation(
                "C001", name, kind, f"amount is {output.amount!r}; expected finite and >= 0"))
        weights = breakdown.get("weights", {})
        if not isinstance(weights, Mapping):
            found.append(GuardViolation("C001", name, kind, "breakdown['weights'] is not a mapping"))
            weights = {}
        values: dict[str, float] = {}
        for bucket in BUCKETS:
            number = _non_negative(breakdown.get(bucket, 0.0))
            if number is None:
                found.append(GuardViolation(
                    "C001", name, kind,
                    f"bucket {bucket} is {breakdown.get(bucket)!r}; expected finite and >= 0"))
                number = 0.0
            values[bucket] = number
        total = 0.0
        for bucket in BUCKETS:
            weight = _non_negative(weights.get(bucket, 1.0))
            if weight is None:
                found.append(GuardViolation(
                    "C001", name, kind,
                    f"weights[{bucket!r}] is {weights.get(bucket)!r}; expected finite and >= 0"))
                weight = 0.0
            total += weight * values[bucket]
        if total > self.policy.reward_ceiling + tol:
            found.append(GuardViolation(
                "C008", name, kind,
                f"reward total {total:g} exceeds COMPLIANCE.REWARD_CEILING "
                f"{self.policy.reward_ceiling:g}", observed=total,
                required=self.policy.reward_ceiling))
        rewarded = {event.rule_id for event in ctx.ledger.rewards()}
        by_rule = breakdown.get("by_rule", {})
        for rule_id, value in (by_rule.items() if isinstance(by_rule, Mapping) else ()):
            if rule_id not in rewarded:
                found.append(GuardViolation(
                    "C009", name, kind,
                    f"by_rule pays {rule_id}, which has no reward event in this ledger",
                    rule_id=str(rule_id)))
            if _non_negative(value) is None:
                found.append(GuardViolation(
                    "C001", name, kind, f"by_rule[{rule_id!r}] is {value!r}",
                    rule_id=str(rule_id)))
        return found

    def _inspect_warning(
        self, stage: Scorer, output: ScorerOutput, ctx: ScoreContext, config: Mapping[str, Any]
    ) -> list[GuardViolation]:
        name, kind = type(stage).__name__, stage.kind.value
        found: list[GuardViolation] = []
        if _non_negative(output.amount) is None:
            found.append(GuardViolation(
                "C001", name, kind, f"amount is {output.amount!r}; expected finite and >= 0"))
        banned = NEVER_WARNING_FIRST | effective_non_offsettable(config.get("NON_OFFSETTABLE"))
        for rule_id in sorted(ctx.warning_only_rules & banned):
            found.append(GuardViolation(
                "C005", name, kind,
                f"{rule_id} was scored as warning-only, but it is never warning-first "
                "(framework.md W9)", rule_id=rule_id))
        return found

    # -- one reading of a stage's output ---------------------------------
    @staticmethod
    def freeze(stage: Scorer, output: ScorerOutput) -> ScorerOutput:
        """``output`` as exact builtin values, read once (see :func:`_plain`).

        The inspectors and the pipeline must see the same numbers: this copy is what both
        get. Raises ``ValueError``/``TypeError``/``RuntimeError`` for an output that cannot
        be read at all (the caller reports ``C001``).
        """
        def names(items: Any) -> tuple[str, ...]:
            try:
                return tuple(str(_plain(item)) for item in list(items))
            except TypeError:
                return ()

        breakdown = output.breakdown
        return ScorerOutput(
            kind=stage.kind,
            amount=_plain(output.amount),
            breakdown=_plain(breakdown) if isinstance(breakdown, Mapping) else breakdown,
            rules=names(output.rules),
            mutations=tuple(output.mutations or ()),
            metadata=dict(output.metadata) if isinstance(output.metadata, Mapping) else {},
            void_rules=names(output.void_rules),
        )

    # -- the entry point -------------------------------------------------
    def enforce(
        self,
        ctx: ScoreContext,
        outputs: Mapping[ScoringKind, ScorerOutput],
        chain: Sequence[Scorer],
        pristine: Pristine | None = None,
    ) -> tuple[dict[ScoringKind, ScorerOutput], GuardReport]:
        """Return the outputs the pipeline may use, and what the guard did.

        ``pristine`` (from :meth:`pristine`) is the yardstick's source: the reference is
        computed from the configuration as it was before any stage ran.
        """
        config = pristine.config if pristine is not None else dict(ctx.config)
        result = dict(outputs)
        cached: dict[str, ScorerOutput] = {}

        def reference() -> ScorerOutput:
            if "penalty" not in cached:
                cached["penalty"] = self.reference_penalty(ctx, config)
            return cached["penalty"]

        found: list[GuardViolation] = []
        by_kind: dict[ScoringKind, list[GuardViolation]] = {}
        for stage in chain:
            output = outputs.get(stage.kind)
            if output is None or self._trusted(stage):
                continue
            try:
                output = self.freeze(stage, output)
            except Exception as error:  # noqa: BLE001 - an unreadable output is refused
                found.append(GuardViolation(
                    "C001", type(stage).__name__, stage.kind.value,
                    f"the output could not be read once and for all: "
                    f"{type(error).__name__}: {error}"))
                by_kind[stage.kind] = [found[-1]]
                continue
            result[stage.kind] = output
            if stage.kind is ScoringKind.PENALTY:
                items = self._inspect_penalty(stage, output, ctx, config, reference())
            elif stage.kind is ScoringKind.REWARD:
                items = self._inspect_reward(stage, output, ctx)
            elif stage.kind is ScoringKind.WARNING:
                items = self._inspect_warning(stage, output, ctx, config)
            else:
                items = []
            if items:
                by_kind[stage.kind] = items
                found.extend(items)

        if not found:
            return result, GuardReport(self.policy.mode)
        hard = ScoringKind.WARNING in by_kind or any(item.code in HARD_CODES for item in found)
        if self.policy.mode == "enforce" or hard:
            raise ComplianceGuardViolation([item.as_dict() for item in found])

        substituted: list[str] = []
        if ScoringKind.PENALTY in by_kind:
            result[ScoringKind.PENALTY] = reference()
            substituted.append("penalty")
        if ScoringKind.REWARD in by_kind:
            result[ScoringKind.REWARD] = RewardScorer(config=config).score(ctx)
            substituted.append("reward")
        log.warning(
            "W003 the compliance guard substituted the reference %s output for a stage that "
            "broke the contract: %s",
            "+".join(substituted),
            "; ".join(f"{item.code} {item.message}" for item in found),
        )
        return result, GuardReport(self.policy.mode, tuple(found), tuple(substituted))
