"""Teacher conformance gating (document 14 section 4.6).

A teacher may or may not have been trained to obey ``docs/framework.md``, so this
module answers one question: on what basis may a peer imitate it?

Contrastive distillation transfers representational geometry. It does not
transfer the compliance objective, which reaches the parameters through ``L_ind``
only. A non-conformant teacher can therefore do exactly one kind of harm: make a
rule-breaking output look like a correct target. The gate removes that harm. It
is an admission filter and not a compliance teacher, and it never feeds a teacher
violation into ``L_ind``.

Two invariants are enforced here and nowhere else:

* I7: no profile, mode, or trust trajectory can admit a sample whose audit found
  a non-offsettable rule. A profile can only narrow trust.
* Trust is never folded into admission. It reaches the objective through
  ``sigma_weights`` alone, so a verdict is reproducible from the audit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import torch

from ..exceptions import ContractError, ImproperlyConfigured, ValidationError
from ..rl.functions._spec import build_from_kwargs
from ..rules.catalog import CATALOG, NON_OFFSETTABLE, PENALTIES, TIERS
from ..rules.severity import Severity
from ..scoring.evidence import ClaimAudit, EpisodeEvidence
from ..scoring.guards import RuleBinding, evaluate_binding, resolve_binding
from ..scoring.ledger import RuleEvent
from .batch import DistillBatch
from .config import CONFORMANCE_KEYS, CONFORMANCE_MODES
from .views import OutputBundle

__all__ = [
    "CONFORMANCE_KEYS",
    "CONFORMANCE_MODES",
    "ConformanceVerdict",
    "DeclaredAudit",
    "FORBIDDEN_TIERS",
    "GuardAudit",
    "SOFTNESS_BY_SEVERITY",
    "TeacherAuditor",
    "TrustOnlyAudit",
    "TrustProfile",
    "TrustState",
    "admission",
    "build_trust_profile",
]

#: Tiers a profile may never declare as trusted. Declaring any of these trusts a
#: teacher to be right about a non-offsettable domain, which I7 forbids.
FORBIDDEN_TIERS: frozenset[int] = frozenset({0, 1, 2})

# CONFORMANCE_MODES and CONFORMANCE_KEYS live in ``config.py`` (no torch import) so
# ``conf.py`` can validate the settings block without loading this module.



@dataclass(frozen=True)
class TrustProfile:
    """What a teacher is admitted as a target for, and by whose evidence.

    ``tiers``
        Precedence tiers whose violations this profile tolerates at a graduated
        admission. A violation in a tier outside this set is refused outright,
        because the profile never vouched for that tier. With the default
        ``{3, 4, 5, 6}`` an epistemic, collaboration, task, or efficiency
        violation costs a teacher a fraction of its admission for that sample,
        while a safety, honesty, or privacy violation is refused.
    ``deny``
        Rule ids refused regardless of tier or audit. Must contain every member
        of ``NON_OFFSETTABLE``; may contain more.
    ``trust_init`` / ``trust_floor``
        The starting trust in ``[trust_floor, 1]`` and the level below which trust
        never decays.
    ``trust_min``
        Below this value the teacher is excluded from ``sigma`` entirely for the
        peers it has fallen below, rather than continuing at a small weight.
    ``require_audit``
        When true an auditor must be configured; a declared-only profile is a
        configuration error.
    ``source``
        Free-form provenance string recorded in the checkpoint.
    ``declared``
        False when this profile was produced by the fail-closed default rather
        than written down. ``STRICT_UNDECLARED`` and the checkpoint's resume note
        both read it.
    """

    tiers: frozenset[int] = frozenset({3, 4, 5, 6})
    deny: frozenset[str] = NON_OFFSETTABLE
    trust_init: float = 1.0
    trust_floor: float = 0.0
    trust_min: float = 0.05
    require_audit: bool = False
    source: str | None = None
    declared: bool = False

    def __post_init__(self) -> None:
        try:
            tiers = frozenset(int(tier) for tier in self.tiers)
            deny = frozenset(str(rule_id) for rule_id in self.deny)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"TrustProfile tiers/deny are malformed: {exc}") from exc
        object.__setattr__(self, "tiers", tiers)
        object.__setattr__(self, "deny", deny)

        forbidden = tiers & FORBIDDEN_TIERS
        if forbidden:
            raise ValidationError(
                "a TrustProfile may not tolerate tier(s) "
                + ", ".join(TIERS[tier] for tier in sorted(forbidden))
                + "; tiers 0-2 are non-offsettable (docs/framework.md section 6.1) and "
                "no declared profile can waive them (invariant I7)"
            )
        unknown = tiers - set(TIERS)
        if unknown:
            raise ValidationError(
                f"TrustProfile.tiers contains unknown tier(s) {sorted(unknown)}; "
                f"accepted: {sorted(TIERS)}"
            )

        waives = NON_OFFSETTABLE - deny
        if waives:
            raise ValidationError(
                "TrustProfile.deny must contain every non-offsettable rule; it is "
                "missing " + ", ".join(sorted(waives)) + ". A profile can only narrow "
                "trust, never widen it (invariant I7)."
            )
        unknown_rules = deny - set(CATALOG)
        if unknown_rules:
            raise ValidationError(
                f"TrustProfile.deny names unknown rule(s) {sorted(unknown_rules)}; "
                "see mactf.rules.catalog.CATALOG"
            )

        for label, value in (("trust_init", self.trust_init), ("trust_floor", self.trust_floor)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"TrustProfile.{label} must be a number, got {value!r}")
        if not (0.0 <= self.trust_floor <= self.trust_init <= 1.0):
            raise ValidationError(
                f"TrustProfile requires 0 <= trust_floor <= trust_init <= 1, got "
                f"floor={self.trust_floor!r} init={self.trust_init!r}"
            )
        if not 0.0 <= self.trust_min <= 1.0:
            raise ValidationError(f"TrustProfile.trust_min must be in [0, 1], got {self.trust_min!r}")

    def permits(self, rule_ids: Sequence[str]) -> bool:
        """``True`` when the profile admits *any* of a sample whose audit fired
        ``rule_ids``.

        Empty input is admitted. Otherwise every fired rule must be outside
        ``deny`` and sit in a tolerated tier. It is deliberately "all rules must
        be tolerated", so a sample firing one T3 rule and one T0 rule is refused.
        """
        if not rule_ids:
            return True
        for rule_id in rule_ids:
            if rule_id in self.deny:
                return False
            rule = CATALOG.get(rule_id)
            if rule is None:
                # An unknown rule id cannot be tiered, so it cannot be admitted.
                return False
            if rule.tier not in self.tiers:
                return False
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "tiers": sorted(self.tiers),
            "deny": sorted(self.deny),
            "trust_init": self.trust_init,
            "trust_floor": self.trust_floor,
            "trust_min": self.trust_min,
            "require_audit": self.require_audit,
            "source": self.source,
            "declared": self.declared,
        }


def build_trust_profile(
    spec: Mapping[str, Any] | TrustProfile | None,
    *,
    declared: bool | None = None,
) -> TrustProfile:
    """Resolve a spec into a profile, recording whether it was *declared*.

    ``declared`` is not cosmetic. ``STRICT_UNDECLARED`` has to distinguish "the
    operator wrote a profile" from "nobody said anything and the fail-closed
    default filled the gap". The flag is stamped at the point where the absence is
    still observable and travels on the profile into the checkpoint.

    ``declared=None`` leaves the flag alone for a ``TrustProfile`` instance, and
    means "declared" for a mapping. For ``None`` it means undeclared.
    """
    if isinstance(spec, TrustProfile):
        return replace(spec, declared=spec.declared if declared is None else bool(declared))
    if spec is None:
        return TrustProfile(declared=False if declared is None else bool(declared))
    if not isinstance(spec, Mapping):
        raise ValidationError(
            f"a trust profile must be a mapping of TrustProfile fields, got "
            f"{type(spec).__name__}; write trust: {{trust_init: 0.7}} rather than a bare number"
        )
    payload = dict(spec)
    if "tiers" in payload:
        payload["tiers"] = frozenset(int(tier) for tier in payload["tiers"])
    if "deny" in payload:
        payload["deny"] = frozenset(str(rule) for rule in payload["deny"])
    payload["declared"] = True if declared is None else bool(declared)
    return build_from_kwargs(TrustProfile, payload)


@dataclass
class ConformanceVerdict:
    """One teacher's admission for one step.

    ``admission`` is per-sample and in ``[0, 1]``. ``1`` is full admission, ``0``
    is refusal, and a value in between is a graduated admission for a tolerated
    violation. It is a tensor, not a bool, because severity is graded.

    ``trust`` is informational only: it is **not** folded into ``admission``.
    Trust enters the objective through ``sigma_weights`` and nowhere else. An
    auditor constructs a verdict with the default ``1.0`` and the gate replaces it
    with the live value, so exactly one place stamps it.

    ``hard_denied`` records *why* the zeros are absolute. It does not mean "all
    rows are zero": a teacher that fabricated on sample 3 of 8 keeps its other
    seven rows and loses sample 3.
    """

    teacher: str
    admission: torch.Tensor  # (B,) in [0, 1], from the audit only
    rules: tuple[str, ...] = ()
    severity: Severity = Severity.NONE
    hard_denied: bool = False
    trust: float = 1.0
    #: The audit's own per-row refusals. Kept separate because ``masked_fraction``
    #: drives trust decay: a teacher at trust 0 whose audit found nothing must not
    #: be read as "fully masked", or trust would decay on its own penalty.
    audit_mask: torch.Tensor | None = None  # (B,) bool, True where refused
    reasons: tuple[str, ...] = ()
    source: str = "audit"

    def __post_init__(self) -> None:
        if self.admission.ndim != 1:
            raise ContractError(
                f"ConformanceVerdict.admission must be 1-D, got {tuple(self.admission.shape)}"
            )
        if not bool(torch.isfinite(self.admission).all()):
            raise ContractError(
                f"teacher {self.teacher!r} has a non-finite admission entry; a "
                "refusal is 0, never nan (a nan propagates into every peer's loss)"
            )
        if bool((self.admission < 0).any()) or bool((self.admission > 1).any()):
            raise ContractError(f"teacher {self.teacher!r} has an admission entry outside [0, 1]")
        if self.audit_mask is not None:
            if self.audit_mask.shape != self.admission.shape or self.audit_mask.dtype != torch.bool:
                raise ContractError(
                    f"teacher {self.teacher!r} audit_mask must be a bool tensor shaped "
                    f"like admission {tuple(self.admission.shape)}"
                )
        # The hard-denial check is about consistency, not about being all-zero:
        # a hard-denied verdict with no zero row would claim I7 applied and then
        # admit everything.
        if self.hard_denied and self.masked == 0:
            raise ContractError(
                f"teacher {self.teacher!r} is hard-denied for rules "
                + (", ".join(self.rules) or "-")
                + " but no admission entry is zero; a non-offsettable violation is "
                "absolute (invariant I7)"
            )

    @property
    def admitted(self) -> int:
        return int((self.admission > 0).sum().item())

    @property
    def refused(self) -> torch.Tensor:
        """Rows the **audit** refused.

        This is what ``TrustState.observe`` reads. It is deliberately not
        ``admission <= 0``: a hand-built verdict may leave ``audit_mask`` unset,
        and a decay rule keyed on admission would be a ratchet with no way back up
        the moment anything scaled admission downstream.
        """
        if self.audit_mask is not None:
            return self.audit_mask
        return self.admission <= 0

    @property
    def masked(self) -> int:
        return int(self.refused.sum().item())

    @property
    def masked_fraction(self) -> float:
        total = int(self.refused.numel())
        return float(self.masked) / total if total else 0.0

    @property
    def all_denied(self) -> bool:
        return self.admitted == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "teacher": self.teacher,
            "admitted": self.admitted,
            "masked": self.masked,
            "masked_fraction": round(self.masked_fraction, 6),
            "rules": list(self.rules),
            "severity": self.severity.label,
            "hard_denied": self.hard_denied,
            "trust": round(self.trust, 6),
            "reasons": list(self.reasons),
            "source": self.source,
        }

    def gather(self, batch: int) -> torch.Tensor:
        """Return ``(B,)`` admission, refusing a verdict built for a different batch."""
        if int(self.admission.numel()) != int(batch):
            raise ContractError(
                f"verdict for teacher {self.teacher!r} covers "
                f"{int(self.admission.numel())} samples, but the batch has {batch}"
            )
        return self.admission


@runtime_checkable
class TeacherAuditor(Protocol):
    """Decides what a teacher's output for this batch is admissible for.

    Deliberately narrow. An auditor sees the teacher's bundle and the batch it was
    computed from, and nothing else: not the peer set, not the loss, not the trust
    history. An auditor that could see the loss could tune admission to make
    training easier, which is the failure mode this section exists to stop.
    """

    name: str

    def audit(
        self,
        bundle: OutputBundle,
        batch: DistillBatch,
        *,
        profile: TrustProfile,
    ) -> ConformanceVerdict: ...


@dataclass(frozen=True)
class DeclaredAudit:
    """A pre-computed, digest-verified conformance record.

    The artifact maps sample index to fired rule ids, plus the digest of the
    teacher it was computed for. A digest mismatch is a hard error, not a warning:
    an audit of a different teacher is not an audit of this one.
    """

    name: str
    rules_by_sample: Mapping[int, tuple[str, ...]]
    teacher_digest: str
    source: str = "declared"
    batch_fingerprint: str | None = None

    @classmethod
    def from_file(cls, path: Path | str) -> "DeclaredAudit":
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ImproperlyConfigured(f"conformance artifact does not exist: {target}") from exc
        except json.JSONDecodeError as exc:
            raise ImproperlyConfigured(f"conformance artifact {target} is not valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ImproperlyConfigured(f"conformance artifact {target} must be a JSON object")
        if "teacher_digest" not in payload:
            raise ImproperlyConfigured(
                f"conformance artifact {target} has no 'teacher_digest'; an audit that "
                "does not say which teacher it audited cannot be used"
            )
        by_sample = {
            int(index): tuple(str(rule) for rule in rules)
            for index, rules in dict(payload.get("rules_by_sample") or {}).items()
        }
        for index, rules in sorted(by_sample.items()):
            unknown = sorted(set(rules) - set(CATALOG))
            if unknown:
                raise ImproperlyConfigured(
                    f"conformance artifact {target} names unknown rule(s) {unknown} at "
                    f"sample {index}"
                )
        return cls(
            name=str(payload.get("name", target.stem)),
            rules_by_sample=by_sample,
            teacher_digest=str(payload["teacher_digest"]),
            source=str(payload.get("source", "declared")),
            batch_fingerprint=payload.get("batch_fingerprint"),
        )

    @property
    def digest(self) -> str:
        """Digest of the artifact contents, for the checkpoint's ``audit_digests``.

        Independent of how the file was serialised, and independent of the teacher
        being audited: a retrained teacher audited under the same record is a
        teacher-digest mismatch with a matching audit digest, and those are
        different events.
        """
        payload = json.dumps(
            {
                "teacher_digest": self.teacher_digest,
                "rules_by_sample": {
                    str(index): list(rules) for index, rules in sorted(self.rules_by_sample.items())
                },
                "batch_fingerprint": self.batch_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def verify_teacher(self, teacher_digest: str) -> None:
        """Raise unless this record was computed for the teacher with this digest."""
        if teacher_digest != self.teacher_digest:
            raise ContractError(
                f"conformance artifact {self.name!r} was computed for teacher digest "
                f"{self.teacher_digest[:12]}... but the loaded teacher is "
                f"{teacher_digest[:12]}...; an audit of a different teacher is not an "
                "audit of this one"
            )

    def audit(
        self,
        bundle: OutputBundle,
        batch: DistillBatch,
        *,
        profile: TrustProfile,
    ) -> ConformanceVerdict:
        if self.batch_fingerprint is not None:
            fingerprint = batch.fingerprint
            if fingerprint != self.batch_fingerprint:
                raise ContractError(
                    f"conformance artifact {self.name!r} was computed for batch "
                    f"{self.batch_fingerprint[:12]}... but this batch is "
                    f"{fingerprint[:12]}...; an audit is only valid for the samples it audited"
                )
        n = batch.batch
        if any(index >= n or index < 0 for index in self.rules_by_sample):
            raise ContractError(
                f"conformance artifact {self.name!r} covers sample indices outside the "
                f"batch size {n}; the artifact and the batch disagree"
            )
        return admission(
            teacher=self.name,
            profile=profile,
            fired=dict(self.rules_by_sample),
            batch=n,
            source=self.source,
        )


@dataclass(frozen=True)
class GuardAudit:
    """Run the framework's own guards over a teacher output.

    ``evidence_factory`` turns one teacher bundle into ``EpisodeEvidence``; the
    guards then run unmodified through ``evaluate_binding``, so a conformance
    verdict and a scorecard cannot disagree about what P1 means.

    Two sources of candidate violations are supported:

    * ``detector`` (recommended): a callable returning the ``RuleEvent`` records
      (or bare penalty ids) that an upstream judge observed in the teacher's
      output. Each is passed through its binding's guard, which decides whether
      the rule really applies. This is the faithful use of a guard: guards decide
      eligibility, they do not detect.
    * no ``detector``: one event is synthesized per rule in ``rule_ids`` (default:
      every penalty) and the guard's verdict stands. This is deliberately
      fail-closed: several guards allow by default, so the all-penalty default
      refuses the batch. Pass an explicit ``rule_ids`` to audit only the rules the
      evidence can actually discriminate.

    **Scope is the whole batch, not a sample.** A guard answers a question about
    an episode, and ``RuleEvent`` has no sample index. So a guard that denies
    makes the verdict batch-level: every row is refused, or none is. This is
    deliberately coarser than ``DeclaredAudit``, and it is the honest mapping:
    pretending a guard finding applied to row 3 would fabricate precision.
    """

    evidence_factory: Callable[[OutputBundle, DistillBatch], EpisodeEvidence]
    name: str = "guard"
    rule_ids: tuple[str, ...] | None = None
    bindings: Mapping[str, RuleBinding] | None = None
    detector: Callable[[OutputBundle, DistillBatch], Iterable[RuleEvent | str]] | None = None

    def _rules(self) -> tuple[str, ...]:
        if self.rule_ids is not None:
            unknown = sorted(set(self.rule_ids) - set(CATALOG))
            if unknown:
                raise ContractError(f"GuardAudit {self.name!r} names unknown rule(s) {unknown}")
            penalties = {rule.id for rule in PENALTIES}
            rewards = sorted(set(self.rule_ids) - penalties)
            if rewards:
                raise ContractError(
                    f"GuardAudit {self.name!r} names non-penalty rule(s) {rewards}; "
                    "an audit reports penalties"
                )
            return tuple(sorted(self.rule_ids))
        return tuple(sorted(rule.id for rule in PENALTIES))

    def _candidates(self, bundle: OutputBundle, batch: DistillBatch) -> list[RuleEvent]:
        if self.detector is None:
            return [
                RuleEvent(
                    rule_id=rule_id,
                    evidence=f"teacher {self.name} batch audit",
                    source="distill.conformance",
                )
                for rule_id in self._rules()
            ]
        allowed = None if self.rule_ids is None else set(self._rules())
        events: list[RuleEvent] = []
        for item in self.detector(bundle, batch) or ():
            event = (
                item
                if isinstance(item, RuleEvent)
                else RuleEvent(
                    rule_id=str(item),
                    evidence=f"teacher {self.name} detector",
                    source="distill.conformance",
                )
            )
            if event.rule_id not in CATALOG:
                raise ContractError(
                    f"GuardAudit {self.name!r} detector reported unknown rule {event.rule_id!r}"
                )
            if CATALOG[event.rule_id].kind == "reward":
                raise ContractError(
                    f"GuardAudit {self.name!r} detector reported reward rule "
                    f"{event.rule_id!r}; an audit reports penalties"
                )
            if allowed is not None and event.rule_id not in allowed:
                continue
            events.append(event)
        return events

    def audit(
        self,
        bundle: OutputBundle,
        batch: DistillBatch,
        *,
        profile: TrustProfile,
    ) -> ConformanceVerdict:
        evidence = self.evidence_factory(bundle, batch)
        if not isinstance(evidence, EpisodeEvidence):
            raise ContractError(
                f"GuardAudit {self.name!r} evidence_factory returned "
                f"{type(evidence).__name__}, expected EpisodeEvidence"
            )
        events = self._candidates(bundle, batch)
        audit_row = ClaimAudit.from_evidence(
            self.name,
            0,
            evidence,
            penalties_fired=sorted({event.rule_id for event in events}),
        )
        fired: set[str] = set()
        reasons: list[str] = []
        for event in events:
            binding = resolve_binding(event.rule_id, self.bindings)
            verdict = evaluate_binding(binding, event, audit_row)
            if verdict.earned:
                fired.add(event.rule_id)
            if verdict.missing:
                # "Could not check" and "checked and clean" are different, and a
                # profile with require_audit=True must not be satisfiable by silence.
                reasons.append(
                    f"{event.rule_id}: required evidence absent ({', '.join(verdict.missing)})"
                )
        ordered = tuple(sorted(fired))
        rows = {index: ordered for index in range(batch.batch)} if ordered else {}
        return admission(
            teacher=self.name,
            profile=profile,
            fired=rows,
            batch=batch.batch,
            reasons=tuple(reasons),
            source="guard",
        )


@dataclass(frozen=True)
class TrustOnlyAudit:
    """Admit on declaration alone. Only valid when ``require_audit`` is false.

    Named so that its use is visible in a config file and in a log line. A silent
    "no auditor" would be indistinguishable from a passing audit.

    It reports admission ``1.0`` and holds **no** trust value. Trust is the gate's
    input, not the auditor's, and reaches the objective through ``sigma_weights``
    alone.
    """

    name: str = "trust-only"

    def audit(
        self,
        bundle: OutputBundle,
        batch: DistillBatch,
        *,
        profile: TrustProfile,
    ) -> ConformanceVerdict:
        if profile.require_audit:
            raise ImproperlyConfigured(
                "TrustProfile.require_audit is set but the only auditor configured is "
                "TrustOnlyAudit, which audits nothing; configure a DeclaredAudit or a "
                "GuardAudit, or clear require_audit"
            )
        return ConformanceVerdict(
            teacher=self.name,
            admission=torch.ones((batch.batch,), dtype=torch.float32),
            source="declared",
        )


def admission(
    *,
    teacher: str,
    profile: TrustProfile,
    fired: Mapping[int, Sequence[str]],
    batch: int,
    reasons: Sequence[str] = (),
    source: str = "audit",
) -> ConformanceVerdict:
    """Per-sample admission from fired rule ids.

    Deliberately **not** parameterised by trust. Trust is the gate's state and
    changes every step; an auditor that received it could raise admission by
    claiming a high trust. This function answers only what the audit can answer:
    is this row admissible at all, at ``1.0``, ``softness``, or ``0.0``.

    The rules, in order:

    1. An empty fired set for a sample is admitted at ``1.0``.
    2. A sample firing anything in ``profile.deny`` is admitted at exactly 0 and
       marks the verdict ``hard_denied``. That is I7; no branch skips it.
    3. A sample firing a rule from a tier the profile does not tolerate is
       admitted at exactly 0. This is deliberately narrower than ``deny`` so a
       future non-offsettable rule is refused even before it reaches the set.
    4. A sample firing only rules from tolerated tiers is admitted at
       ``softness``, which falls with severity.
    """
    if batch < 1:
        raise ValidationError(f"admission needs a positive batch size, got {batch!r}")

    values = torch.ones((batch,), dtype=torch.float32)
    fired_rules: set[str] = set()
    hard = False
    severity = Severity.NONE
    notes: list[str] = list(reasons)

    for index in sorted(fired):
        if not 0 <= int(index) < batch:
            raise ContractError(
                f"admission for teacher {teacher!r} names sample {index}, outside a "
                f"batch of {batch}"
            )
        rules = tuple(str(rule) for rule in fired[index] if rule)
        if not rules:
            continue
        unknown = sorted(set(rules) - set(CATALOG))
        if unknown:
            raise ContractError(
                f"admission for teacher {teacher!r} names unknown rule(s) {unknown}; "
                "an untierable rule cannot be admitted"
            )
        # Only penalties can be violations. A reward id reaching `admission` means
        # the auditor reported achievement in a channel that only understands
        # infringement; treating it as a violation would refuse a sample for being
        # good.
        rewarded = sorted(rule for rule in rules if CATALOG[rule].kind == "reward")
        if rewarded:
            raise ContractError(
                f"admission for teacher {teacher!r} names reward rule(s) {rewarded}; "
                "admission reports penalties, because a reward is not a reason to "
                "refuse a teacher's target"
            )
        fired_rules.update(rules)
        denied = [rule for rule in rules if rule in profile.deny]
        if denied:
            hard = True
            values[int(index)] = 0.0
            notes.append(f"sample {index}: non-offsettable {', '.join(sorted(denied))} (I7)")
            severity = _worse(severity, _severity_of(denied))
            continue
        untolerated = [rule for rule in rules if CATALOG[rule].tier not in profile.tiers]
        if untolerated:
            values[int(index)] = 0.0
            notes.append(
                f"sample {index}: {', '.join(sorted(untolerated))} fired in a tier this "
                "profile does not tolerate; a violation there is treated as "
                "unrecoverable rather than graduated"
            )
            severity = _worse(severity, _severity_of(untolerated))
            continue
        softness = _softness(_severity_of(rules))
        values[int(index)] = softness
        notes.append(f"sample {index}: {', '.join(sorted(rules))} at softness {softness:g}")
        severity = _worse(severity, _severity_of(rules))

    values = values.clamp(min=0.0, max=1.0)
    # Restated, not redundant. The per-sample branch above zeros a denied sample,
    # and this re-zeros it from the raw `fired` mapping. If a future edit softens
    # the branch, this still forces a hard zero, so the failure mode is "the
    # redundancy looks silly" rather than "I7 was quietly violated".
    # `ConformanceVerdict.__post_init__` is the third check.
    denied_rows = torch.tensor(
        [
            any(rule in profile.deny for rule in fired.get(index, ()))
            for index in range(batch)
        ],
        dtype=torch.bool,
    )
    if hard and not bool((values[denied_rows] == 0).all()):
        raise ContractError(
            f"admission for teacher {teacher!r} softens a non-offsettable denial; "
            "this cannot happen and indicates a bug in the severity table (I7)"
        )
    values = torch.where(denied_rows, torch.zeros_like(values), values)
    return ConformanceVerdict(
        teacher=teacher,
        admission=values,
        rules=tuple(sorted(fired_rules)),
        severity=severity,
        hard_denied=hard,
        audit_mask=values <= 0,
        reasons=tuple(notes),
        source=source,
    )


#: Admission retained at each severity for an offsettable violation. A minor
#: violation costs a tenth of the sample. S4 is inadmissible even here, which
#: today coincides with the catalog (every S4 rule is non-offsettable) and keeps
#: the guarantee if that ever changes.
SOFTNESS_BY_SEVERITY: dict[Severity, float] = {
    Severity.NONE: 1.0,
    Severity.MINOR: 0.9,
    Severity.MODERATE: 0.6,
    Severity.MAJOR: 0.25,
    Severity.SEVERE: 0.0,
}


def _softness(severity: Severity) -> float:
    return float(SOFTNESS_BY_SEVERITY[severity])


def _severity_of(rule_ids: Sequence[str]) -> Severity:
    """The worst default severity among the fired rules."""
    worst = Severity.NONE
    for rule_id in rule_ids:
        rule = CATALOG.get(rule_id)
        if rule is not None:
            worst = _worse(worst, rule.default_severity)
    return worst


def _worse(left: Severity, right: Severity) -> Severity:
    return left if int(left) >= int(right) else right


class TrustState:
    """Per-teacher trust in ``[trust_floor, 1]``.

    ``SigmaState`` answers "can this peer match this teacher", an EMA of
    ``-L_ij``. It is a difficulty measure and a hard teacher is not a bad teacher.
    ``TrustState`` answers "is this teacher's target legitimate", which is a
    provenance measure and has nothing to do with difficulty. Multiplying them
    (section 6.5) is the composition; merging them would make a hard-but-honest
    teacher indistinguishable from an easy-but-dishonest one.
    """

    def __init__(
        self,
        *,
        init: float = 1.0,
        floor: float = 0.0,
        decay: float = 0.5,
        restore: float = 0.05,
    ) -> None:
        if not 0.0 <= floor <= init <= 1.0:
            raise ValidationError(
                f"TrustState requires 0 <= floor <= init <= 1, got floor={floor!r} init={init!r}"
            )
        if not 0.0 < decay <= 1.0:
            raise ValidationError(f"trust decay must be in (0, 1], got {decay!r}")
        if not 0.0 <= restore <= 1.0:
            raise ValidationError(f"trust restore must be in [0, 1], got {restore!r}")
        self.init = float(init)
        self.floor = float(floor)
        self.decay = float(decay)
        self.restore = float(restore)
        self._values: dict[str, float] = {}

    def get(self, teacher: str) -> float:
        return self._values.get(teacher, self.init)

    def set(self, teacher: str, value: float) -> float:
        """Force a trust value, clamped to ``[floor, 1]``.

        Used only by the resume path. Not used by ``observe``, which computes the
        decay itself, so the two paths cannot drift.
        """
        if not math.isfinite(float(value)):
            raise ContractError(f"trust for teacher {teacher!r} must be finite, got {value!r}")
        clamped = min(1.0, max(self.floor, float(value)))
        self._values[str(teacher)] = clamped
        return clamped

    def observe(self, teacher: str, verdict: ConformanceVerdict) -> float:
        """Update one teacher's trust from one step's verdict.

        A step with no violations restores trust toward ``init`` at ``restore``;
        a step with any violation multiplies trust by ``(1 - decay)`` scaled by the
        audit's masked fraction. Decay is multiplicative because trust lost to a
        pattern of violations should compound, while a single bad batch should
        barely move a teacher that has been clean for a long time.

        ``hard_denied`` pins trust to ``floor`` immediately. A teacher that
        fabricated a citation does not keep 90% of its trust because the
        fabrication was confined to one row.

        The masked fraction is derived from the **audit's** refusals
        (``audit_mask``), never from a trust-scaled tensor, so decay is never
        driven by its own penalty. ``observe`` never reads ``admission``.
        """
        current = self.get(teacher)
        if verdict.hard_denied:
            value = self.floor
        elif verdict.masked_fraction > 0.0:
            value = current * (1.0 - self.decay * verdict.masked_fraction)
        else:
            value = current + (self.init - current) * self.restore
        value = min(1.0, max(self.floor, value))
        self._values[teacher] = value
        return value

    def weights(self, teachers: Sequence[str]) -> dict[str, float]:
        return {name: self.get(name) for name in sorted(teachers)}

    def state_dict(self) -> dict[str, Any]:
        return {
            "init": self.init,
            "floor": self.floor,
            "decay": self.decay,
            "restore": self.restore,
            "values": {name: value for name, value in sorted(self._values.items())},
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.init = float(payload.get("init", self.init))
        self.floor = float(payload.get("floor", self.floor))
        self.decay = float(payload.get("decay", self.decay))
        self.restore = float(payload.get("restore", self.restore))
        self._values = {
            str(name): min(1.0, max(self.floor, float(value)))
            for name, value in dict(payload.get("values") or {}).items()
        }
