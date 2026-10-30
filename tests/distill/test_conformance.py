"""Conformance gate: I7, graduated admission, auditors, trust (NB1-NB6)."""

from __future__ import annotations

import json

import pytest
import torch

from mactf.distill.batch import DistillBatch
from mactf.distill.conformance import (
    ConformanceVerdict,
    DeclaredAudit,
    GuardAudit,
    SOFTNESS_BY_SEVERITY,
    TrustOnlyAudit,
    TrustProfile,
    TrustState,
    admission,
    build_trust_profile,
)
from mactf.distill.views import OutputBundle
from mactf.exceptions import ContractError, ImproperlyConfigured, ValidationError
from mactf.rules.catalog import CATALOG, NON_OFFSETTABLE
from mactf.rules.severity import Severity
from mactf.scoring.evidence import EpisodeEvidence

BUNDLE = OutputBundle(logits=torch.zeros(4, 3), source="teacher", side="t")


def _batch(n: int = 4) -> DistillBatch:
    return DistillBatch(states=torch.arange(n * 2, dtype=torch.float32).reshape(n, 2))


# -- TrustProfile (I7) ------------------------------------------------------
@pytest.mark.parametrize("tier", [0, 1, 2])
def test_profile_cannot_tolerate_a_non_offsettable_tier(tier):
    with pytest.raises(ValidationError, match="invariant I7"):
        TrustProfile(tiers=frozenset({3, tier}))


def test_profile_cannot_waive_a_non_offsettable_rule():
    for rule in sorted(NON_OFFSETTABLE):
        with pytest.raises(ValidationError, match="missing"):
            TrustProfile(deny=NON_OFFSETTABLE - {rule})


def test_profile_may_deny_more_but_not_unknown_rules():
    wider = TrustProfile(deny=NON_OFFSETTABLE | {"P8"})
    assert "P8" in wider.deny
    with pytest.raises(ValidationError, match="unknown rule"):
        TrustProfile(deny=NON_OFFSETTABLE | {"P999"})
    with pytest.raises(ValidationError, match="unknown tier"):
        TrustProfile(tiers=frozenset({3, 42}))


def test_profile_numeric_validation():
    with pytest.raises(ValidationError):
        TrustProfile(trust_init=0.2, trust_floor=0.5)
    with pytest.raises(ValidationError):
        TrustProfile(trust_init=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TrustProfile(trust_min=2.0)


def test_permits_requires_every_fired_rule_to_be_tolerated():
    profile = TrustProfile()
    assert profile.permits([])
    assert profile.permits(["P8"])
    assert not profile.permits(["P8", "P1"])
    assert not profile.permits(["P17"])
    assert not profile.permits(["NOPE"])


def test_build_trust_profile_records_declaration():
    assert build_trust_profile(None).declared is False
    assert build_trust_profile({"trust_init": 0.5}).declared is True
    base = TrustProfile()
    assert build_trust_profile(base, declared=True).declared is True
    assert build_trust_profile(base).declared is False
    with pytest.raises(ValidationError, match="bare number"):
        build_trust_profile(0.7)  # type: ignore[arg-type]
    coerced = build_trust_profile({"tiers": [3, 4], "deny": sorted(NON_OFFSETTABLE)})
    assert coerced.tiers == frozenset({3, 4})


# -- admission --------------------------------------------------------------
def test_clean_samples_are_fully_admitted():
    verdict = admission(teacher="t", profile=TrustProfile(), fired={}, batch=3)
    assert verdict.admission.tolist() == [1.0, 1.0, 1.0]
    assert not verdict.hard_denied
    assert verdict.masked == 0


@pytest.mark.parametrize("rule", sorted(NON_OFFSETTABLE))
def test_non_offsettable_rules_always_zero_the_row(rule):
    """NB1: no profile, tier set, or audit shape can admit a denied sample."""
    verdict = admission(teacher="t", profile=TrustProfile(), fired={1: (rule,)}, batch=3)
    assert verdict.admission.tolist() == [1.0, 0.0, 1.0]
    assert verdict.hard_denied
    assert verdict.rules == (rule,)
    assert verdict.audit_mask.tolist() == [False, True, False]


def test_a_tolerated_violation_is_graduated_by_severity():
    # P8 is a tier-3 MINOR rule, P9 a tier-3 MODERATE rule.
    verdict = admission(teacher="t", profile=TrustProfile(), fired={0: ("P8",), 1: ("P9",)}, batch=2)
    assert verdict.admission[0].item() == pytest.approx(SOFTNESS_BY_SEVERITY[Severity.MINOR])
    assert verdict.admission[1].item() == pytest.approx(SOFTNESS_BY_SEVERITY[Severity.MODERATE])
    assert not verdict.hard_denied


def test_a_violation_in_an_untolerated_tier_is_refused():
    narrow = TrustProfile(tiers=frozenset({3}))
    verdict = admission(teacher="t", profile=narrow, fired={0: ("P11",)}, batch=2)  # tier 6
    assert verdict.admission.tolist() == [0.0, 1.0]
    assert not verdict.hard_denied


def test_mixed_rules_take_the_worst_case():
    verdict = admission(teacher="t", profile=TrustProfile(), fired={0: ("P8", "P1")}, batch=1)
    assert verdict.admission.tolist() == [0.0]
    assert verdict.hard_denied


def test_admission_input_validation():
    with pytest.raises(ValidationError):
        admission(teacher="t", profile=TrustProfile(), fired={}, batch=0)
    with pytest.raises(ContractError, match="outside a batch"):
        admission(teacher="t", profile=TrustProfile(), fired={5: ("P8",)}, batch=2)
    with pytest.raises(ContractError, match="unknown rule"):
        admission(teacher="t", profile=TrustProfile(), fired={0: ("ZZ",)}, batch=2)
    reward = next(rule.id for rule in CATALOG.values() if rule.kind == "reward")
    with pytest.raises(ContractError, match="reward rule"):
        admission(teacher="t", profile=TrustProfile(), fired={0: (reward,)}, batch=2)


def test_severe_is_inadmissible_even_if_it_were_offsettable():
    assert SOFTNESS_BY_SEVERITY[Severity.SEVERE] == 0.0
    assert SOFTNESS_BY_SEVERITY[Severity.NONE] == 1.0


# -- ConformanceVerdict ----------------------------------------------------
def test_verdict_rejects_nan_range_and_shape_errors():
    with pytest.raises(ContractError, match="non-finite"):
        ConformanceVerdict(teacher="t", admission=torch.tensor([float("nan")]))
    with pytest.raises(ContractError, match="outside"):
        ConformanceVerdict(teacher="t", admission=torch.tensor([1.5]))
    with pytest.raises(ContractError, match="1-D"):
        ConformanceVerdict(teacher="t", admission=torch.ones(2, 2))
    with pytest.raises(ContractError, match="audit_mask"):
        ConformanceVerdict(teacher="t", admission=torch.ones(2), audit_mask=torch.zeros(3, dtype=torch.bool))


def test_hard_denied_verdict_must_actually_zero_something():
    with pytest.raises(ContractError, match="absolute"):
        ConformanceVerdict(teacher="t", admission=torch.ones(3), hard_denied=True)


def test_verdict_gather_refuses_a_different_batch():
    verdict = ConformanceVerdict(teacher="t", admission=torch.ones(3))
    assert verdict.gather(3).shape == (3,)
    with pytest.raises(ContractError, match="covers 3 samples"):
        verdict.gather(4)
    assert verdict.as_dict()["admitted"] == 3


# -- auditors --------------------------------------------------------------
def test_trust_only_audit_admits_everything_unless_an_audit_is_required():
    verdict = TrustOnlyAudit().audit(BUNDLE, _batch(), profile=TrustProfile())
    assert verdict.admission.tolist() == [1.0] * 4
    assert verdict.source == "declared"
    with pytest.raises(ImproperlyConfigured, match="require_audit"):
        TrustOnlyAudit().audit(BUNDLE, _batch(), profile=TrustProfile(require_audit=True))


def test_declared_audit_maps_samples_to_admission():
    audit = DeclaredAudit(name="a", rules_by_sample={2: ("P17",), 0: ("P8",)}, teacher_digest="d" * 64)
    verdict = audit.audit(BUNDLE, _batch(), profile=TrustProfile())
    assert verdict.admission[2].item() == 0.0
    assert 0.0 < verdict.admission[0].item() < 1.0
    assert verdict.hard_denied


def test_declared_audit_refuses_a_foreign_teacher_batch_or_range():
    audit = DeclaredAudit(name="a", rules_by_sample={}, teacher_digest="a" * 64)
    audit.verify_teacher("a" * 64)
    with pytest.raises(ContractError, match="different teacher"):
        audit.verify_teacher("b" * 64)

    bound = DeclaredAudit(
        name="a", rules_by_sample={}, teacher_digest="a" * 64, batch_fingerprint=_batch().fingerprint
    )
    bound.audit(BUNDLE, _batch(), profile=TrustProfile())
    with pytest.raises(ContractError, match="only valid for the samples"):
        bound.audit(BUNDLE, _batch(5), profile=TrustProfile())

    outside = DeclaredAudit(name="a", rules_by_sample={9: ("P8",)}, teacher_digest="a" * 64)
    with pytest.raises(ContractError, match="outside the batch"):
        outside.audit(BUNDLE, _batch(), profile=TrustProfile())


def test_declared_audit_file_round_trip_and_errors(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps({"name": "x", "teacher_digest": "abc", "rules_by_sample": {"1": ["P8"]}}),
        encoding="utf-8",
    )
    audit = DeclaredAudit.from_file(path)
    assert audit.rules_by_sample == {1: ("P8",)}
    assert audit.digest == DeclaredAudit.from_file(path).digest

    with pytest.raises(ImproperlyConfigured, match="does not exist"):
        DeclaredAudit.from_file(tmp_path / "missing.json")
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ImproperlyConfigured, match="not valid JSON"):
        DeclaredAudit.from_file(path)
    path.write_text(json.dumps({"rules_by_sample": {}}), encoding="utf-8")
    with pytest.raises(ImproperlyConfigured, match="teacher_digest"):
        DeclaredAudit.from_file(path)
    path.write_text(
        json.dumps({"teacher_digest": "a", "rules_by_sample": {"0": ["ZZ"]}}), encoding="utf-8"
    )
    with pytest.raises(ImproperlyConfigured, match="unknown rule"):
        DeclaredAudit.from_file(path)
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(ImproperlyConfigured, match="JSON object"):
        DeclaredAudit.from_file(path)


def test_audit_digest_is_independent_of_key_order():
    one = DeclaredAudit(name="a", rules_by_sample={1: ("P8",), 0: ("P9",)}, teacher_digest="d")
    two = DeclaredAudit(name="a", rules_by_sample={0: ("P9",), 1: ("P8",)}, teacher_digest="d")
    assert one.digest == two.digest


def test_guard_audit_runs_the_framework_guards():
    # Whatever the guards decide, the verdict is batch-level and I7 holds.
    audit = GuardAudit(
        evidence_factory=lambda bundle, batch: EpisodeEvidence(),
        name="g",
        detector=lambda bundle, batch: ["P17"],
    )
    verdict = audit.audit(BUNDLE, _batch(), profile=TrustProfile())
    rows = verdict.admission.tolist()
    assert len(set(rows)) == 1
    if verdict.rules:
        assert verdict.hard_denied and rows == [0.0] * 4


def test_guard_audit_without_a_detector_is_fail_closed():
    audit = GuardAudit(evidence_factory=lambda bundle, batch: EpisodeEvidence(), name="g")
    verdict = audit.audit(BUNDLE, _batch(), profile=TrustProfile())
    assert len(set(verdict.admission.tolist())) == 1


def test_guard_audit_validation():
    factory = lambda bundle, batch: EpisodeEvidence()  # noqa: E731
    with pytest.raises(ContractError, match="unknown rule"):
        GuardAudit(evidence_factory=factory, rule_ids=("ZZ",)).audit(BUNDLE, _batch(), profile=TrustProfile())
    reward = next(rule.id for rule in CATALOG.values() if rule.kind == "reward")
    with pytest.raises(ContractError, match="non-penalty"):
        GuardAudit(evidence_factory=factory, rule_ids=(reward,)).audit(BUNDLE, _batch(), profile=TrustProfile())
    with pytest.raises(ContractError, match="expected EpisodeEvidence"):
        GuardAudit(evidence_factory=lambda b, c: None).audit(BUNDLE, _batch(), profile=TrustProfile())
    with pytest.raises(ContractError, match="unknown rule"):
        GuardAudit(evidence_factory=factory, detector=lambda b, c: ["ZZ"]).audit(
            BUNDLE, _batch(), profile=TrustProfile()
        )
    with pytest.raises(ContractError, match="reward rule"):
        GuardAudit(evidence_factory=factory, detector=lambda b, c: [reward]).audit(
            BUNDLE, _batch(), profile=TrustProfile()
        )


# -- TrustState ------------------------------------------------------------
def _verdict(masked: int, total: int = 4, *, hard: bool = False) -> ConformanceVerdict:
    mask = torch.zeros(total, dtype=torch.bool)
    mask[:masked] = True
    return ConformanceVerdict(
        teacher="t",
        admission=(~mask).float(),
        audit_mask=mask,
        hard_denied=hard and masked > 0,
    )


def test_trust_decays_with_the_masked_fraction_and_recovers():
    state = TrustState(init=1.0, floor=0.0, decay=0.5, restore=0.1)
    assert state.get("t") == 1.0
    after = state.observe("t", _verdict(2))
    assert after == pytest.approx(1.0 * (1 - 0.5 * 0.5))
    recovered = state.observe("t", _verdict(0))
    assert recovered == pytest.approx(after + (1.0 - after) * 0.1)


def test_hard_denial_pins_trust_to_the_floor():
    state = TrustState(init=1.0, floor=0.2)
    assert state.observe("t", _verdict(1, hard=True)) == 0.2
    assert state.observe("t", _verdict(0)) > 0.2  # restoration is possible


def test_trust_decay_is_not_driven_by_its_own_penalty():
    """A trust-scaled admission must not look like an audit refusal."""
    state = TrustState(init=1.0, decay=0.9, restore=0.5)
    state.set("t", 0.1)
    verdict = ConformanceVerdict(
        teacher="t",
        admission=torch.full((4,), 0.0),  # zero because trust was multiplied in downstream
        audit_mask=torch.zeros(4, dtype=torch.bool),
    )
    assert state.observe("t", verdict) > 0.1


def test_trust_state_validation_and_round_trip():
    with pytest.raises(ValidationError):
        TrustState(init=0.5, floor=0.9)
    with pytest.raises(ValidationError):
        TrustState(decay=0.0)
    with pytest.raises(ValidationError):
        TrustState(restore=2.0)
    state = TrustState(init=0.9, floor=0.1, decay=0.4, restore=0.2)
    state.observe("a", _verdict(2))
    state.set("b", 5.0)  # clamped
    assert state.get("b") == 1.0
    with pytest.raises(ContractError):
        state.set("c", float("nan"))
    clone = TrustState()
    clone.load_state_dict(state.state_dict())
    assert clone.state_dict() == state.state_dict()
    assert clone.weights(["b", "a"]) == state.weights(["a", "b"])
