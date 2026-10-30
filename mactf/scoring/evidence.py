"""Episode evidence and the read-only audit guards consult (document 11 section 4.2-4.3).

A guard without evidence is an assertion, so eligibility is decided from
*evidence records* rather than from the ledger alone. Two rules shape this module:

* Missing sections are **absent**, never defaulted to a permissive value. A guard
  that needs ``verifications`` and finds none denies with ``missing=(...)``; it
  does not assume the peer verified anything.
* Guards never receive the whole evidence object. They receive
  :class:`ClaimAudit`, computed once per ``(peer, episode)``, so 40 guards do not
  each re-scan 200 claims.

Imports are the standard library and :mod:`mactf.rules` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..rules.catalog import CATALOG

__all__ = [
    "ClaimAudit",
    "ClaimRecord",
    "CreditRecord",
    "CritiqueRecord",
    "EpisodeEvidence",
    "EvidenceLink",
    "FlagRecord",
    "HelpEvent",
    "RevisionRecord",
    "SuggestionRecord",
    "SynthesisRecord",
    "UnknownStatement",
    "VerificationTrace",
    "CorrectionRecord",
]


@dataclass(frozen=True)
class ClaimRecord:
    """One material claim from the artifact."""

    claim_id: str
    text: str = ""
    material: bool = True
    supported: bool = False
    labelled_uncertain: bool = False
    evidence_ids: tuple[str, ...] = ()
    contradicted: bool = False
    asserted_as_fact: bool = True
    #: P10: a claim taken from a peer rather than produced, and whether it could be checked.
    accepted_from_peer: bool = False
    verification_available: bool = True
    #: P8: confidence language beyond the evidence, and whether it was withdrawn.
    confidence_exceeds_evidence: bool = False
    retracted: bool = False


@dataclass(frozen=True)
class EvidenceLink:
    """A claim-to-source mapping the evaluator asserts."""

    claim_id: str
    source: str = ""
    resolves: bool = True
    fabricated: bool = False


@dataclass(frozen=True)
class SuggestionRecord:
    """One peer suggestion and what this peer did with it."""

    author: str
    suggestion_id: str
    addressed: bool = False
    decision: str = "ignored"  # "accepted" | "rejected" | "ignored"
    reason_references_substance: bool = False
    incorporated: bool = False
    credited: bool = False
    #: An immaterial suggestion may be ignored without P6 (framework.md 6-P6).
    relevant: bool = True


@dataclass(frozen=True)
class CreditRecord:
    """A credit the peer gave, independent of whether a suggestion was incorporated."""

    suggestion_id: str
    author: str = ""


@dataclass(frozen=True)
class FlagRecord:
    """One misconduct flag this peer raised."""

    target_peer: str
    rule_id: str
    accepted: bool = False
    evidence_cited: bool = False
    adjudicated: bool = True
    #: ``deliberate`` selects P3; ``negligent``/``accidental`` select P4.
    intent: str = "negligent"


@dataclass(frozen=True)
class VerificationTrace:
    """Evidence that a claim was actually checked."""

    claim_id: str
    method: str = "test"  # "test" | "computation" | "citation" | "observation"
    trivial: bool = False
    reproduced: bool = False


@dataclass(frozen=True)
class HelpEvent:
    """A help action this peer gave or received."""

    direction: str  # "given" | "received"
    other_peer: str = ""
    effective: bool = False
    documented_blocker: bool = False
    subtask: str = ""


@dataclass(frozen=True)
class UnknownStatement:
    """R6: an explicit statement of what is not known."""

    claim_id: str
    resolution_path: bool = False


@dataclass(frozen=True)
class RevisionRecord:
    """R7: a prior claim that changed, and whether the correcting evidence was valid."""

    claim_id: str
    changed: bool = True
    evidence_valid: bool = False
    #: Changing under social pressure rather than evidence (P10 boundary).
    capitulation: bool = False


@dataclass(frozen=True)
class CritiqueRecord:
    """R11: a critique of a peer's artifact."""

    target: str
    names_flaw: bool = True
    remedy: bool = False
    remedy_testable: bool = False
    flaw_correct: bool = True


@dataclass(frozen=True)
class SynthesisRecord:
    """R10: a combination of contributions."""

    sources: tuple[str, ...] = ()
    concatenation: bool = False


@dataclass(frozen=True)
class CorrectionRecord:
    """A correction issued against a claim (P2, P9)."""

    claim_id: str
    valid: bool = True
    revised: bool = False
    counter_evidence: bool = False
    disclosed: bool = False


@dataclass(frozen=True)
class EpisodeEvidence:
    """Everything a guard may consult, assembled before scoring.

    Absent scalar fields are ``None`` and absent collections are empty; both are
    reported by :meth:`present_sections` so a guard can name what it lacked.
    """

    claims: tuple[ClaimRecord, ...] = ()
    evidence_links: tuple[EvidenceLink, ...] = ()
    suggestions: tuple[SuggestionRecord, ...] = ()
    credits: tuple[CreditRecord, ...] = ()
    flags: tuple[FlagRecord, ...] = ()
    verifications: tuple[VerificationTrace, ...] = ()
    help_events: tuple[HelpEvent, ...] = ()
    unknowns: tuple[UnknownStatement, ...] = ()
    revisions: tuple[RevisionRecord, ...] = ()
    critiques: tuple[CritiqueRecord, ...] = ()
    syntheses: tuple[SynthesisRecord, ...] = ()
    corrections: tuple[CorrectionRecord, ...] = ()
    task_checks_passed: bool | None = None
    refusal: str | None = None  # boundary named by the refusal, if any
    partial: bool = False
    partial_limits_stated: bool = False
    intent_aligned: bool | None = None
    assumptions_documented: bool = False
    disclosed_ambiguity: tuple[str, ...] = ()
    correctness_verified: bool | None = None
    step_count: int | None = None
    task_class_baseline: int | None = None
    all_refusals: bool = False  # the P18 pattern from 5-R2
    #: Deliverable produced (P23, R17); ``None`` means unknown.
    deliverable: bool | None = None
    #: P11 progress, P12 noise, P13 independence, P14 honest failure, P18 hazard.
    progress_made: bool | None = None
    substantive_ratio: float | None = None
    independent_support: bool | None = None
    honest_failure: bool | None = None
    input_hazardous: bool | None = None
    helper_artifact_intact: bool | None = None
    boundary_attempts: int | None = None
    safety_critical: bool = False

    # -- derived ---------------------------------------------------------
    @property
    def supports_claim_audit(self) -> bool:
        return bool(self.claims)

    @property
    def has_fabrication(self) -> bool:
        """True when any evidence link is fabricated. Drives P1 detection."""
        return any(link.fabricated for link in self.evidence_links)

    def present_sections(self) -> frozenset[str]:
        """The names of every section that carries data."""
        present: set[str] = set()
        for name in (
            "claims", "evidence_links", "suggestions", "credits", "flags",
            "verifications", "help_events", "unknowns", "revisions", "critiques",
            "syntheses", "corrections",
        ):
            if getattr(self, name):
                present.add(name)
        for name in (
            "task_checks_passed", "intent_aligned", "correctness_verified", "step_count",
            "task_class_baseline", "deliverable", "progress_made", "substantive_ratio",
            "independent_support", "honest_failure", "input_hazardous",
            "helper_artifact_intact", "boundary_attempts", "refusal",
        ):
            if getattr(self, name) is not None:
                present.add(name)
        if self.disclosed_ambiguity:
            present.add("disclosed_ambiguity")
        return frozenset(present)

    def digest(self) -> dict[str, Any]:
        """A compact summary for ``scorecard.metadata["evidence"]``."""
        return {
            "sections": sorted(self.present_sections()),
            "claims": len(self.claims),
            "evidence_links": len(self.evidence_links),
            "suggestions": len(self.suggestions),
            "flags": len(self.flags),
            "verifications": len(self.verifications),
            "help_events": len(self.help_events),
            "task_checks_passed": self.task_checks_passed,
            "correctness_verified": self.correctness_verified,
            "has_fabrication": self.has_fabrication,
        }

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeEvidence":
        """Build from a plain mapping, rejecting unknown keys with the allowed list."""
        import dataclasses

        record_types: dict[str, type] = {
            "claims": ClaimRecord,
            "evidence_links": EvidenceLink,
            "suggestions": SuggestionRecord,
            "credits": CreditRecord,
            "flags": FlagRecord,
            "verifications": VerificationTrace,
            "help_events": HelpEvent,
            "unknowns": UnknownStatement,
            "revisions": RevisionRecord,
            "critiques": CritiqueRecord,
            "syntheses": SynthesisRecord,
            "corrections": CorrectionRecord,
        }
        allowed = {item.name for item in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                f"EpisodeEvidence got unknown field(s) {unknown}; allowed: {sorted(allowed)}"
            )
        values: dict[str, Any] = {}
        for key, value in payload.items():
            record_type = record_types.get(key)
            if record_type is not None:
                names = {item.name for item in dataclasses.fields(record_type)}
                rows = []
                for row in value or ():
                    if isinstance(row, record_type):
                        rows.append(row)
                        continue
                    extra = sorted(set(row) - names)
                    if extra:
                        raise ValueError(
                            f"{record_type.__name__} got unknown field(s) {extra}; "
                            f"allowed: {sorted(names)}"
                        )
                    row = dict(row)
                    for tuple_field in ("evidence_ids", "sources"):
                        if tuple_field in row:
                            row[tuple_field] = tuple(row[tuple_field])
                    rows.append(record_type(**row))
                values[key] = tuple(rows)
            elif key == "disclosed_ambiguity":
                values[key] = tuple(str(item) for item in (value or ()))
            else:
                values[key] = value
        return cls(**values)

    @classmethod
    def from_subject(cls, subject: Any) -> "EpisodeEvidence":
        """Adapt an :class:`~mactf.scoring.evaluator.EvaluationSubject`.

        Duck-typed so this module need not import the evaluator. Only what the
        subject actually recorded is carried over; anything it cannot know stays
        absent.
        """
        claims: list[ClaimRecord] = []
        links: list[EvidenceLink] = []
        verifications: list[VerificationTrace] = []
        unknowns: list[UnknownStatement] = []
        for index, claim in enumerate(getattr(subject, "claims", ()) or ()):
            claim_id = str(getattr(claim, "metadata", {}).get("id", f"c{index}"))
            kind = getattr(claim, "support_kind", "none")
            label = getattr(claim, "uncertainty_label", None)
            claims.append(
                ClaimRecord(
                    claim_id=claim_id,
                    text=str(getattr(claim, "text", "")),
                    material=bool(getattr(claim, "material", True)),
                    supported=kind == "evidence",
                    labelled_uncertain=bool(label),
                    asserted_as_fact=bool(getattr(claim, "asserted_as_fact", True)),
                    contradicted=bool(getattr(claim, "metadata", {}).get("contradicted", False)),
                )
            )
            if kind == "evidence":
                links.append(
                    EvidenceLink(claim_id=claim_id, source=str(getattr(claim, "support", "")))
                )
            elif kind == "assertion" and getattr(claim, "asserted_as_fact", True):
                # An assertion offered as support is not a source; it is not a
                # fabricated one either, so it stays unlinked and unsupported.
                pass
            if label:
                unknowns.append(UnknownStatement(claim_id=claim_id, resolution_path=bool(label)))
        suggestions: list[SuggestionRecord] = []
        for item in getattr(subject, "suggestions", ()) or ():
            if getattr(item, "critique_of", None) is not None:
                continue
            suggestions.append(
                SuggestionRecord(
                    author=str(getattr(item, "author", "")),
                    suggestion_id=str(getattr(item, "id", "")),
                    addressed=bool(getattr(item, "considered", False)),
                    decision="accepted" if getattr(item, "incorporated", False) else (
                        "rejected" if getattr(item, "considered", False) else "ignored"
                    ),
                    reason_references_substance=bool(getattr(item, "consideration_reason", None)),
                    incorporated=bool(getattr(item, "incorporated", False)),
                    credited=bool(getattr(item, "credited", False)),
                    relevant=bool(getattr(item, "material", True)),
                )
            )
        critiques = [
            CritiqueRecord(
                target=str(item.critique_of),
                remedy=bool(getattr(item, "critique_fix", None)),
                remedy_testable=bool(getattr(item, "critique_fix", None)),
            )
            for item in getattr(subject, "suggestions", ()) or ()
            if getattr(item, "critique_of", None) is not None
        ]
        flags = [
            FlagRecord(
                target_peer=str(item.target_peer),
                rule_id=str(item.rule_id),
                accepted=bool(item.accepted),
                evidence_cited=bool(str(getattr(item, "evidence", "")).strip()),
                adjudicated=item.accepted is not None,
                intent=str(getattr(item, "intent", "negligent")),
            )
            for item in getattr(subject, "flags", ()) or ()
        ]
        corrections = [
            CorrectionRecord(
                claim_id=str(item.claim),
                valid=bool(item.valid),
                revised=bool(item.revised),
                counter_evidence=bool(item.counter_evidence),
            )
            for item in getattr(subject, "corrections", ()) or ()
        ]
        revisions = [
            RevisionRecord(
                claim_id=str(item.claim),
                changed=bool(item.revised),
                evidence_valid=bool(item.valid),
            )
            for item in getattr(subject, "corrections", ()) or ()
            if item.revised
        ]
        help_events = [
            HelpEvent(
                direction="given",
                other_peer=str(step.helped_peer),
                effective=bool(step.help_effective),
            )
            for step in getattr(subject, "trace", ()) or ()
            if getattr(step, "helped_peer", None)
        ]
        trace = tuple(getattr(subject, "trace", ()) or ())
        substantive = None
        if trace:
            tokens = sum(max(0, int(step.tokens)) for step in trace)
            useful = sum(max(0, int(step.substantive_tokens)) for step in trace)
            substantive = (useful / tokens) if tokens else 1.0
        actions = tuple(getattr(subject, "actions", ()) or ())
        boundary_attempts = (
            sum(1 for action in actions if action.targets_boundary and not action.allowed)
            if actions
            else None
        )
        return cls(
            claims=tuple(claims),
            evidence_links=tuple(links),
            suggestions=tuple(suggestions),
            flags=tuple(flags),
            verifications=tuple(verifications),
            help_events=tuple(help_events),
            unknowns=tuple(unknowns),
            revisions=tuple(revisions),
            critiques=tuple(critiques),
            corrections=tuple(corrections),
            substantive_ratio=substantive,
            progress_made=(
                any(bool(step.new_information) for step in trace) if trace else None
            ),
            boundary_attempts=boundary_attempts,
        )


def _ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class ClaimAudit:
    """Pre-computed answers a guard may consult in O(1)."""

    peer: str
    episode: int
    has_claims: bool = False
    material_claims: int = 0
    unsupported_material: tuple[str, ...] = ()
    fabricated_links: tuple[str, ...] = ()
    labelled_uncertain: int = 0
    contradicted: tuple[str, ...] = ()
    task_checks_passed: bool | None = None
    correctness_verified: bool | None = None
    all_refusals: bool = False
    refusal_boundary: str | None = None
    partial: bool = False
    partial_limits_stated: bool = False
    intent_aligned: bool | None = None
    assumptions_documented: bool = False
    disclosures: frozenset[str] = frozenset()
    suggestions_addressed: int = 0
    suggestions_ignored_relevant: tuple[str, ...] = ()
    suggestions_non_substantive: tuple[str, ...] = ()
    suggestions_incorporated: int = 0
    distinct_contributors: int = 0
    credits_present: int = 0
    credits_missing: tuple[str, ...] = ()
    credits_unearned: tuple[str, ...] = ()
    accepted_flags: tuple[str, ...] = ()
    rejected_flags: tuple[str, ...] = ()
    evidence_free_flags: tuple[str, ...] = ()
    verification_traces: int = 0
    trivial_verifications: int = 0
    reproduced_verifications: int = 0
    effective_help_given: int = 0
    help_received: int = 0
    documented_blocker: bool = False
    # -- fields beyond the document's list, each answering one guard's question
    sections: frozenset[str] = frozenset()
    penalties_fired: frozenset[str] = frozenset()
    void: bool = False
    deliverable: bool | None = None
    unknown_statements: int = 0
    unknowns_with_path: int = 0
    valid_revisions: int = 0
    capitulations: int = 0
    critiques_with_remedy: int = 0
    critiques_wrong: int = 0
    critiques: int = 0
    syntheses: int = 0
    concatenations: int = 0
    undisclosed_contradictions: tuple[str, ...] = ()
    contradictions_disclosed: int = 0
    unrevised_valid_corrections: int = 0
    corrections: int = 0
    deliberate_rejected_flags: int = 0
    good_faith_rejected_flags: int = 0
    flag_count: int = 0
    flag_precision: float | None = None
    unverified_accepted: tuple[str, ...] = ()
    accepted_claims: int = 0
    overclaims: int = 0
    unretracted_overclaims: int = 0
    asserted_as_fact: int = 0
    substantive_ratio: float | None = None
    progress_made: bool | None = None
    independent_support: bool | None = None
    honest_failure: bool | None = None
    input_hazardous: bool | None = None
    helper_artifact_intact: bool | None = None
    boundary_attempts: int | None = None
    safety_critical: bool = False
    step_count: int | None = None
    task_class_baseline: int | None = None
    # -- thresholds, so a guard never reads settings
    flag_cap: int = 3
    p_min: float = 0.5
    p12_noise_ratio: float = 0.5
    r9_step_factor: float = 1.25

    @property
    def accepted_evidenced_flags(self) -> int:
        return len(self.accepted_flags)

    @property
    def tier_breach(self) -> tuple[str, ...]:
        """Penalties from tiers T0-T2 that fired this episode."""
        return tuple(
            sorted(
                rule_id
                for rule_id in self.penalties_fired
                if rule_id in CATALOG and CATALOG[rule_id].tier <= 2
            )
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, name)
            if isinstance(value, frozenset):
                value = sorted(value)
            elif isinstance(value, tuple):
                value = list(value)
            payload[name] = value
        return payload

    @classmethod
    def from_evidence(
        cls,
        peer: str,
        episode: int,
        evidence: EpisodeEvidence,
        *,
        penalties_fired: Iterable[str] = (),
        void: bool = False,
        flag_cap: int = 3,
        p_min: float = 0.5,
        p12_noise_ratio: float = 0.5,
        r9_step_factor: float = 1.25,
    ) -> "ClaimAudit":
        """One pass over the evidence."""
        material = [claim for claim in evidence.claims if claim.material]
        unsupported = _ids(
            claim.claim_id
            for claim in material
            if not claim.supported and not claim.labelled_uncertain
        )
        fabricated = _ids(link.claim_id for link in evidence.evidence_links if link.fabricated)

        relevant = [item for item in evidence.suggestions if item.relevant]
        ignored = _ids(item.suggestion_id for item in relevant if not item.addressed)
        non_substantive = _ids(
            item.suggestion_id
            for item in relevant
            if item.addressed and not item.reason_references_substance
        )
        incorporated = [item for item in evidence.suggestions if item.incorporated]
        credited_ids = {item.suggestion_id for item in evidence.credits}
        credits_missing = _ids(
            item.suggestion_id
            for item in incorporated
            if not (item.credited or item.suggestion_id in credited_ids)
        )
        credits_present = sum(
            1 for item in incorporated if item.credited or item.suggestion_id in credited_ids
        )
        incorporated_ids = {item.suggestion_id for item in incorporated}
        credited_all = credited_ids | {
            item.suggestion_id for item in evidence.suggestions if item.credited
        }
        credits_unearned = tuple(sorted(credited_all - incorporated_ids))

        accepted_flags = _ids(
            f"{flag.target_peer}:{flag.rule_id}"
            for flag in evidence.flags
            if flag.accepted and flag.evidence_cited
        )
        rejected_flags = _ids(
            f"{flag.target_peer}:{flag.rule_id}"
            for flag in evidence.flags
            if flag.adjudicated and not flag.accepted
        )
        evidence_free = _ids(
            f"{flag.target_peer}:{flag.rule_id}" for flag in evidence.flags if not flag.evidence_cited
        )
        adjudicated = [flag for flag in evidence.flags if flag.adjudicated]
        precision = (
            sum(1 for flag in adjudicated if flag.accepted) / len(adjudicated)
            if adjudicated
            else None
        )
        deliberate = sum(
            1 for flag in evidence.flags
            if flag.adjudicated and not flag.accepted and flag.intent == "deliberate"
        )
        good_faith = sum(
            1 for flag in evidence.flags
            if flag.adjudicated and not flag.accepted and flag.intent != "deliberate"
        )

        given = [event for event in evidence.help_events if event.direction == "given"]
        received = [event for event in evidence.help_events if event.direction == "received"]

        trivial = sum(1 for item in evidence.verifications if item.trivial)
        reproduced = sum(
            1 for item in evidence.verifications if item.reproduced and not item.trivial
        )
        verified_claims = {item.claim_id for item in evidence.verifications}

        contributors = 0
        for synthesis in evidence.syntheses:
            contributors = max(contributors, len(set(synthesis.sources)))
        if not evidence.syntheses:
            contributors = len({item.author for item in incorporated if item.author})

        contradicted = _ids(claim.claim_id for claim in evidence.claims if claim.contradicted)
        disclosed_corrections = {
            item.claim_id for item in evidence.corrections if item.disclosed or item.revised
        }
        undisclosed = tuple(item for item in contradicted if item not in disclosed_corrections)
        unverified = _ids(
            claim.claim_id
            for claim in evidence.claims
            if claim.accepted_from_peer
            and claim.verification_available
            and claim.claim_id not in verified_claims
        )
        overclaims = [claim for claim in evidence.claims if claim.confidence_exceeds_evidence]

        return cls(
            peer=peer,
            episode=episode,
            has_claims=bool(evidence.claims),
            material_claims=len(material),
            unsupported_material=unsupported,
            fabricated_links=fabricated,
            labelled_uncertain=sum(1 for claim in evidence.claims if claim.labelled_uncertain),
            contradicted=contradicted,
            task_checks_passed=evidence.task_checks_passed,
            correctness_verified=evidence.correctness_verified,
            all_refusals=bool(evidence.all_refusals),
            refusal_boundary=(evidence.refusal or None),
            partial=bool(evidence.partial),
            partial_limits_stated=bool(evidence.partial_limits_stated),
            intent_aligned=evidence.intent_aligned,
            assumptions_documented=bool(evidence.assumptions_documented),
            disclosures=frozenset(str(item) for item in evidence.disclosed_ambiguity),
            suggestions_addressed=sum(1 for item in evidence.suggestions if item.addressed),
            suggestions_ignored_relevant=ignored,
            suggestions_non_substantive=non_substantive,
            suggestions_incorporated=len(incorporated),
            distinct_contributors=contributors,
            credits_present=credits_present,
            credits_missing=credits_missing,
            credits_unearned=credits_unearned,
            accepted_flags=accepted_flags,
            rejected_flags=rejected_flags,
            evidence_free_flags=evidence_free,
            verification_traces=len(evidence.verifications),
            trivial_verifications=trivial,
            reproduced_verifications=reproduced,
            effective_help_given=sum(1 for event in given if event.effective),
            help_received=len(received),
            documented_blocker=any(event.documented_blocker for event in evidence.help_events),
            sections=evidence.present_sections(),
            penalties_fired=frozenset(str(item) for item in penalties_fired),
            void=bool(void),
            deliverable=evidence.deliverable,
            unknown_statements=len(evidence.unknowns),
            unknowns_with_path=sum(1 for item in evidence.unknowns if item.resolution_path),
            valid_revisions=sum(
                1 for item in evidence.revisions if item.changed and item.evidence_valid
            ),
            capitulations=sum(1 for item in evidence.revisions if item.capitulation),
            critiques=len(evidence.critiques),
            critiques_with_remedy=sum(
                1
                for item in evidence.critiques
                if item.names_flaw and item.remedy and item.remedy_testable and item.flaw_correct
            ),
            critiques_wrong=sum(1 for item in evidence.critiques if not item.flaw_correct),
            syntheses=len(evidence.syntheses),
            concatenations=sum(1 for item in evidence.syntheses if item.concatenation),
            undisclosed_contradictions=undisclosed,
            contradictions_disclosed=len(contradicted) - len(undisclosed),
            unrevised_valid_corrections=sum(
                1
                for item in evidence.corrections
                if item.valid and not item.revised and not item.counter_evidence
            ),
            corrections=len(evidence.corrections),
            deliberate_rejected_flags=deliberate,
            good_faith_rejected_flags=good_faith,
            flag_count=len(evidence.flags),
            flag_precision=precision,
            unverified_accepted=unverified,
            accepted_claims=sum(1 for claim in evidence.claims if claim.accepted_from_peer),
            overclaims=len(overclaims),
            unretracted_overclaims=sum(1 for claim in overclaims if not claim.retracted),
            asserted_as_fact=sum(1 for claim in evidence.claims if claim.asserted_as_fact),
            substantive_ratio=evidence.substantive_ratio,
            progress_made=evidence.progress_made,
            independent_support=evidence.independent_support,
            honest_failure=evidence.honest_failure,
            input_hazardous=evidence.input_hazardous,
            helper_artifact_intact=evidence.helper_artifact_intact,
            boundary_attempts=evidence.boundary_attempts,
            safety_critical=bool(evidence.safety_critical),
            step_count=evidence.step_count,
            task_class_baseline=evidence.task_class_baseline,
            flag_cap=int(flag_cap),
            p_min=float(p_min),
            p12_noise_ratio=float(p12_noise_ratio),
            r9_step_factor=float(r9_step_factor),
        )

    @classmethod
    def empty(cls, peer: str = "", episode: int = 0, **kwargs: Any) -> "ClaimAudit":
        """An audit over no evidence at all: every reward guard will deny."""
        return cls.from_evidence(peer, episode, EpisodeEvidence(), **kwargs)
