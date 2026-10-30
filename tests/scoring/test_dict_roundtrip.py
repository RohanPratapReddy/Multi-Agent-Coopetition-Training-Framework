"""Document 13 section 5 / criterion 52: ``from_dict`` is the inverse of ``to_dict``."""

from __future__ import annotations

import dataclasses

import pytest

from mactf.compliance import ComplianceResult, constrain
from mactf.coopetition.aggregation import AggregationReport
from mactf.rules.precedence import PrecedenceDecision
from mactf.sandbox.base import Capability, CapabilityReport, ExecResult, IsolationLevel
from mactf.scoring import EpisodeEvidence, Ledger, PointsEngine, ScoringPipeline
from mactf.scoring.continuous import ContinuousScore
from mactf.scoring.normalize import CohortNormalizer, NormalizationReport
from mactf.scoring.points import PointsBreakdown
from mactf.scoring.scorecard import EpisodeScorecard, PeerScorecard


def _ledger(*rules, peer="p"):
    ledger = Ledger(peer=peer)
    for rule_id in rules:
        ledger.record(rule_id, evidence="e" if rule_id.startswith("P") else "")
    return ledger


def _card(*rules, evidence=None):
    return ScoringPipeline().score_peer(_ledger(*rules), peer="p", episode=2, evidence=evidence)


def _dump(obj):
    return obj.to_dict() if hasattr(obj, "to_dict") else obj.as_dict()


def _samples():
    clean = _card("R1", "R2")
    suppressed = _card("R1", "P6")
    void = _card("R1", "P1")
    warned = _card("P11", "P11", "P11", "P11")
    pointed = _card("R1", "P12", evidence=EpisodeEvidence())
    normalized = CohortNormalizer().normalize({"a": 80.0, "b": 55.0, "c": 20.0}, episode=3, void={"c": True})
    breakdown = PointsEngine().score(_ledger("R1", "P12"), evidence=EpisodeEvidence())
    aggregation = AggregationReport(
        weights={"a": 0.6, "b": 0.4},
        raw_weights={"a": 0.7, "b": 0.3},
        kappa=1.5,
        entropy=0.67,
        adjustments=["floor a"],
        gated_peers=("c",),
        trimmed_peers=("a",),
        order=("a", "b"),
        signal_source="points",
        normalization=normalized,
    )
    return {
        "PeerScorecard/clean": clean,
        "PeerScorecard/suppressed": suppressed,
        "PeerScorecard/void": void,
        "PeerScorecard/warned": warned,
        "PeerScorecard/points": pointed,
        "EpisodeScorecard": EpisodeScorecard(episode=2, peers={"p": clean, "q": void}, metadata={"k": 1}),
        "ContinuousScore": void.continuous,
        "AggregationReport": aggregation,
        "AggregationReport/plain": AggregationReport(weights={"a": 1.0}, raw_weights={"a": 1.0}, kappa=1.0, entropy=0.0),
        "CapabilityReport": CapabilityReport(
            provider="local",
            isolation=IsolationLevel.PROCESS,
            usable=True,
            capabilities=[Capability("fs", True), Capability("net", False)],  # detail is not on the wire
            reasons=["no kvm"],
            metadata={"host": "x"},
        ),
        "ExecResult": ExecResult(command=["echo", "hi"], stdout="hi\n", exit_code=0, duration=0.25),
        "ExecResult/str": ExecResult(command="echo hi", stderr="boom", exit_code=2, timed_out=True),
        "ComplianceResult": constrain(void.continuous),
        "ComplianceResult/clean": constrain(clean.continuous),
        "PointsBreakdown": breakdown,
        "NormalizationReport": normalized,
    }


SAMPLES = _samples()


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_serialized_form_round_trips(name):
    original = SAMPLES[name]
    payload = _dump(original)
    rebuilt = type(original).from_dict(payload)
    assert isinstance(rebuilt, type(original))
    assert _dump(rebuilt) == payload


EXACT = [
    "ExecResult", "ExecResult/str", "AggregationReport/plain", "PointsBreakdown", "NormalizationReport",
    "ComplianceResult", "ComplianceResult/clean", "PeerScorecard/points",
]


#: Warnings travel as ``describe()`` text and a precedence decision as its description
#: (document 13 section 5), so these compare through the serialized form only.
LOSSY = {
    "PeerScorecard/suppressed", "PeerScorecard/warned", "PeerScorecard/void", "PeerScorecard/points",
    "ComplianceResult", "ComplianceResult/clean", "EpisodeScorecard",
}


@pytest.mark.parametrize("name", [n for n in sorted(SAMPLES) if n.split("/")[0] != "AggregationReport" or n in EXACT])
def test_dataclass_fields_equal_where_the_form_is_lossless(name):
    if name in LOSSY:
        pytest.skip("holds a warning description or a rule decision, compared through the serialized form")
    original = SAMPLES[name]
    if not dataclasses.is_dataclass(original):
        pytest.skip("not a dataclass")
    rebuilt = type(original).from_dict(_dump(original))
    assert dataclasses.asdict(rebuilt) == dataclasses.asdict(original), name


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_unknown_keys_are_ignored(name):
    original = SAMPLES[name]
    payload = dict(_dump(original), future_field={"nested": 1})
    assert _dump(type(original).from_dict(payload)) == _dump(original)


REQUIRED = {
    "PeerScorecard/clean": "final_score",
    "EpisodeScorecard": "peers",
    "ContinuousScore": "warning",
    "AggregationReport/plain": "kappa",
    "CapabilityReport": "provider",
    "ExecResult": "exit_code",
    "ComplianceResult": "value",
    "PointsBreakdown": "final_points",
    "NormalizationReport": "stats",
}


@pytest.mark.parametrize("name,key", sorted(REQUIRED.items()))
def test_missing_required_key_raises_key_error_naming_it(name, key):
    original = SAMPLES[name]
    payload = _dump(original)
    del payload[key]
    with pytest.raises(KeyError) as caught:
        type(original).from_dict(payload)
    assert key in str(caught.value)


def test_none_and_empty_stay_distinguishable():
    card = _card("R1")
    payload = card.to_dict()
    payload["floor_reason"] = ""
    assert PeerScorecard.from_dict(payload).floor_reason == ""
    payload["floor_reason"] = None
    assert PeerScorecard.from_dict(payload).floor_reason is None
    assert PeerScorecard.from_dict(dict(payload, points=None)).points is None
    assert PeerScorecard.from_dict(dict(payload, competitive=None)).competitive is None


def test_scorecard_keeps_properties_after_rebuild():
    void = PeerScorecard.from_dict(_card("R1", "P1").to_dict())
    assert void.is_void and void.violated_non_offsettable == ("P1",)
    assert void.final_score < 0
    assert void.severity == _card("R1", "P1").severity
    assert void.continuous.decision is not None


def test_precedence_description_round_trip():
    decision = _card("R1", "P6").continuous.decision
    again = PrecedenceDecision.from_description(decision.describe())
    assert again == decision
    with pytest.raises(ValueError):
        PrecedenceDecision.from_description("garbage")


def test_tuples_are_rebuilt_not_left_as_lists():
    rebuilt = AggregationReport.from_dict(SAMPLES["AggregationReport"].as_dict())
    assert isinstance(rebuilt.gated_peers, tuple) and isinstance(rebuilt.order, tuple)
    assert isinstance(rebuilt.normalization, NormalizationReport)
    frozen = NormalizationReport.from_dict(SAMPLES["NormalizationReport"].as_dict())
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.episode = 9  # type: ignore[misc]
