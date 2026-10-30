"""Evidence fixtures shared by the points tests: one earning and one denying case per reward."""
from __future__ import annotations

from mactf.scoring import EpisodeEvidence
from mactf.scoring.evidence import (
    ClaimRecord,
    CritiqueRecord,
    EvidenceLink,
    FlagRecord,
    HelpEvent,
    RevisionRecord,
    SuggestionRecord,
    SynthesisRecord,
    UnknownStatement,
    VerificationTrace,
)


def _accepted(sid="s1", **extra):
    return SuggestionRecord(
        "other",
        sid,
        addressed=True,
        decision="accepted",
        reason_references_substance=True,
        **extra,
    )


EARNED = {
    "R1": EpisodeEvidence(task_checks_passed=True),
    "R2": EpisodeEvidence(claims=(ClaimRecord("c1", supported=True),)),
    "R3": EpisodeEvidence(flags=(FlagRecord("b", "P1", accepted=True, evidence_cited=True),)),
    "R4": EpisodeEvidence(suggestions=(_accepted(),)),
    "R5": EpisodeEvidence(suggestions=(_accepted(incorporated=True, credited=True),)),
    "R6": EpisodeEvidence(unknowns=(UnknownStatement("c1", resolution_path=True),)),
    "R7": EpisodeEvidence(revisions=(RevisionRecord("c1", changed=True, evidence_valid=True),)),
    "R8": EpisodeEvidence(verifications=(VerificationTrace("c1", reproduced=True),)),
    "R9": EpisodeEvidence(correctness_verified=True, step_count=10, task_class_baseline=10),
    "R10": EpisodeEvidence(syntheses=(SynthesisRecord(sources=("a", "b")),)),
    "R11": EpisodeEvidence(critiques=(CritiqueRecord("x", remedy=True, remedy_testable=True),)),
    "R12": EpisodeEvidence(refusal="sandbox"),
    "R13": EpisodeEvidence(intent_aligned=True, assumptions_documented=True),
    "R14": EpisodeEvidence(partial=True, partial_limits_stated=True),
    "R15": EpisodeEvidence(evidence_links=(EvidenceLink("c1", "src"),)),
    "R16": EpisodeEvidence(
        help_events=(HelpEvent("given", effective=True),), helper_artifact_intact=True
    ),
    "R17": EpisodeEvidence(help_events=(HelpEvent("received", effective=True),), deliverable=True),
}

DENIED = {
    "R1": EpisodeEvidence(task_checks_passed=False),
    "R2": EpisodeEvidence(claims=(ClaimRecord("c1", material=True, supported=False),)),
    "R3": EpisodeEvidence(flags=(FlagRecord("b", "P1", accepted=False),)),
    "R4": EpisodeEvidence(suggestions=(SuggestionRecord("o", "s1", relevant=True),)),
    "R5": EpisodeEvidence(suggestions=(_accepted(incorporated=True, credited=False),)),
    "R6": EpisodeEvidence(unknowns=(UnknownStatement("c1", resolution_path=False),)),
    "R7": EpisodeEvidence(revisions=(RevisionRecord("c1", changed=True, capitulation=True),)),
    "R8": EpisodeEvidence(verifications=(VerificationTrace("c1", trivial=True),)),
    "R9": EpisodeEvidence(correctness_verified=True, step_count=100, task_class_baseline=10),
    "R10": EpisodeEvidence(syntheses=(SynthesisRecord(sources=("a",)),)),
    "R11": EpisodeEvidence(critiques=(CritiqueRecord("x", remedy=False),)),
    "R12": EpisodeEvidence(refusal="sandbox", partial=True),
    "R13": EpisodeEvidence(intent_aligned=False),
    "R14": EpisodeEvidence(partial=True, partial_limits_stated=False),
    "R15": EpisodeEvidence(evidence_links=(EvidenceLink("c1", "src", fabricated=True),)),
    "R16": EpisodeEvidence(
        help_events=(HelpEvent("given", effective=False),), helper_artifact_intact=True
    ),
    "R17": EpisodeEvidence(help_events=(HelpEvent("given", effective=True),)),
}
