"""CoDistillStep end to end: invariants I1-I4, I7 through the step, rendezvous, signals."""

from __future__ import annotations

import pytest
import torch

from distill_support import (
    build_step,
    parameters_of,
    shared_batch,
    signals as peer_signals,
    teacher_spec,
)
from mactf import signals
from mactf.distill import CoDistillStep
from mactf.distill.conformance import TrustProfile, admission
from mactf.exceptions import ContractError, ImproperlyConfigured, ValidationError


def exploding(states):
    raise RuntimeError("teacher exploded")


class FixedAudit:
    """A user-supplied auditor that reports a fixed fired-rule map."""

    name = "fixed"

    def __init__(self, fired):
        self.fired = fired

    def audit(self, bundle, batch, *, profile: TrustProfile):
        return admission(teacher="t1", profile=profile, fired=self.fired, batch=batch.batch)


class Recorder:
    def __init__(self, signal):
        self.events = []
        self.signal = signal
        signal.connect(self, weak=False)

    def __call__(self, sender=None, **payload):
        self.events.append(payload)

    def close(self):
        self.signal.disconnect(self)


@pytest.fixture
def cohort(settings, tmp_path):
    return build_step(settings, tmp_path)


# -- construction -----------------------------------------------------------
def test_requires_a_teacher(settings, tmp_path):
    with pytest.raises(ImproperlyConfigured, match="at least one teacher"):
        build_step(settings, tmp_path, teachers=[])


def test_duplicate_teacher_names_are_refused(settings, tmp_path):
    with pytest.raises(ValidationError, match="duplicate teacher"):
        build_step(settings, tmp_path, teachers=[teacher_spec("a"), teacher_spec("a")])


def test_unknown_sigma_mode_and_conformance_key_are_refused(settings, tmp_path):
    with pytest.raises(ValidationError, match="sigma_mode"):
        build_step(settings, tmp_path, sigma_mode="bogus")
    with pytest.raises(ValidationError, match="does not accept"):
        build_step(settings, tmp_path, conformance={"TRUST_DECAYY": 0.1})


def test_an_undeclared_teacher_is_refused_under_strict_undeclared(settings, tmp_path):
    with pytest.raises(ImproperlyConfigured, match="STRICT_UNDECLARED"):
        build_step(settings, tmp_path, teachers=[teacher_spec(trust=None)])
    step, _ = build_step(
        settings,
        tmp_path,
        teachers=[teacher_spec(trust=None)],
        conformance={"STRICT_UNDECLARED": False},
    )
    assert step.teachers["t1"].spec.trust.declared is False


def test_audit_mode_without_an_auditor_is_an_error(settings, tmp_path):
    with pytest.raises(ImproperlyConfigured, match="MODE='audit'"):
        build_step(settings, tmp_path, conformance={"MODE": "audit"})


def test_from_settings_refuses_an_off_config_and_a_non_config(settings):
    from mactf.distill.config import DistillationConfig

    with pytest.raises(ValidationError, match="DistillationConfig"):
        CoDistillStep.from_settings({"ENABLED": True})  # type: ignore[arg-type]
    off = DistillationConfig.from_spec(False, settings=settings)
    with pytest.raises(ImproperlyConfigured, match="disabled"):
        CoDistillStep.from_settings(off)


def test_a_peer_path_that_collides_with_a_teacher_is_refused(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, teachers=[teacher_spec("peer-1")], register=False)
    with pytest.raises(ValidationError, match="collides"):
        step.setup(trainer.peers["peer-1"])


# -- one cohort step -----------------------------------------------------------
def test_a_step_trains_every_peer_and_leaves_the_teacher_inert(cohort):
    """I2: the teacher never accumulates a gradient or changes."""
    step, trainer = cohort
    teacher = step.teachers["t1"]
    digest = teacher.digest
    before = {path: parameters_of(peer.module) for path, peer in trainer.peers.items()}
    result = step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))

    assert result.stepped
    assert torch.isfinite(result.loss)
    for path, peer in trainer.peers.items():
        assert not torch.equal(parameters_of(peer.module), before[path])
    assert all(p.grad is None for p in teacher.module.parameters())
    teacher._digest = None
    assert teacher.digest == digest
    assert result.metrics["teachers_active"] == 1.0
    assert step.last["peer-1"].sigma == {"t1": pytest.approx(1.0)}


def test_peer_gradients_do_not_cross_peers_or_reach_the_teacher(cohort):
    """I3: backward of one peer's loss touches that peer only."""
    step, trainer = cohort
    for peer in trainer.peers.values():
        step.setup(peer)
    batch = shared_batch()
    prepared = step.prepare(trainer.peers, batch)
    for peer in trainer.peers.values():
        for p in peer.module.parameters():
            p.grad = None
    loss = step._peer_loss(
        prepared=prepared,
        entry=prepared.peers["peer-1"],
        signal={"peer-1": 0.0, "peer-2": 0.0},
        centring=0.0,
        scorecards=None,
    )
    loss.total.backward()
    assert any(p.grad is not None for p in trainer.peers["peer-1"].module.parameters())
    assert all(p.grad is None for p in trainer.peers["peer-2"].module.parameters())
    assert all(p.grad is None for p in step.teachers["t1"].module.parameters())


def test_one_shared_batch_object_reaches_every_participant(cohort):
    """I1: peers and the teacher read the very same states tensor."""
    step, trainer = cohort
    batch = shared_batch()
    step.step(trainer.peers, batch, signal=peer_signals(trainer))
    assert step.last_prepared.batch.states is batch


def test_the_same_seed_reproduces_the_same_run(settings, tmp_path):
    """I4: determinism."""
    runs = []
    for index in range(2):
        step, trainer = build_step(settings, tmp_path / f"run{index}", seed=4)
        for _ in range(2):
            step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
        runs.append({p: parameters_of(peer.module) for p, peer in trainer.peers.items()})
    for path in runs[0]:
        assert torch.equal(runs[0][path], runs[1][path])


def test_warmup_turns_the_distillation_term_off_exactly(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, config={"warmup_steps": 2, "peer_margin": 0.0})
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    for loss in step.last.values():
        assert loss.lambda_cd_effective == 0.0
        assert float(loss.total) == pytest.approx(float(loss.ind), rel=1e-6)
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    assert all(loss.lambda_cd_effective > 0.0 for loss in step.last.values())


def test_step_requires_a_signal_for_every_peer_on_the_first_step(cohort):
    step, trainer = cohort
    with pytest.raises(ContractError, match="no signal"):
        step.step(trainer.peers, shared_batch())
    with pytest.raises(ValidationError, match="at least 2"):
        step.step(trainer.peers, shared_batch(1), signal=peer_signals(trainer))


def test_a_second_step_may_default_the_signal_to_the_previous_loss(cohort):
    step, trainer = cohort
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    step.step(trainer.peers, shared_batch())  # falls back to -loss
    assert step.state.step == 2


def test_cooperative_term_is_zero_sum_across_peers(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, config={"lambda_coop": 1.0})
    step.step(trainer.peers, shared_batch(), signal={"peer-1": 1.0, "peer-2": 3.0})
    centred = [float(loss.coop) for loss in step.last.values()]
    assert all(c != 0.0 for c in centred)
    assert step.last["peer-1"].breakdown.extras["coop"] == pytest.approx(float(step.last["peer-1"].coop))


def test_void_peer_is_excluded_from_the_cohort_centring(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, config={"lambda_coop": 1.0})
    step.step(
        trainer.peers,
        shared_batch(),
        signal={"peer-1": 1.0, "peer-2": 3.0},
        eligible={"peer-1": True, "peer-2": False},
    )
    assert float(step.last["peer-2"].coop) == 0.0


# -- multiple teachers ----------------------------------------------------------
def test_two_teachers_share_a_normalised_sigma_and_see_hard_negatives(settings, tmp_path):
    step, trainer = build_step(
        settings, tmp_path, teachers=[teacher_spec("a"), teacher_spec("b")]
    )
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    for loss in step.last.values():
        assert sorted(loss.sigma) == ["a", "b"]
        assert sum(loss.sigma.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(step.last_reports) == {
        ("peer-1", "a"),
        ("peer-1", "b"),
        ("peer-2", "a"),
        ("peer-2", "b"),
    }


def test_a_failing_teacher_is_excluded_for_the_step_not_the_run(settings, tmp_path):
    bad = teacher_spec("bad", kind="callable", factory="test_step.exploding", kwargs={})
    bad.pop("view")
    recorder = Recorder(signals.distillation_teacher_error)
    try:
        step, trainer = build_step(settings, tmp_path, teachers=[teacher_spec("good"), bad])
        step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    finally:
        recorder.close()
    assert step.last_prepared.failed.keys() == {"bad"}
    assert step.last_prepared.active == ("good",)
    assert [event["teacher"] for event in recorder.events] == ["bad"]
    assert step.last["peer-1"].sigma == {"good": pytest.approx(1.0)}


def test_every_teacher_failing_trains_on_the_individual_loss_alone(settings, tmp_path):
    bad = teacher_spec("bad", kind="callable", factory="test_step.exploding", kwargs={})
    bad.pop("view")
    step, trainer = build_step(settings, tmp_path, teachers=[bad], config={"peer_margin": 0.0})
    result = step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    assert result.metrics["individual_only"] == 1.0
    for loss in step.last.values():
        assert loss.sigma == {}
        assert float(loss.total) == pytest.approx(float(loss.ind), rel=1e-6)


# -- conformance through the step (NB1-NB6) ----------------------------------
def test_a_non_offsettable_teacher_contributes_nothing_and_is_pinned(settings, tmp_path):
    """NB1: P17 on every row zeroes the term, removes the teacher from sigma, pins trust."""
    rows = {index: ("P17",) for index in range(6)}
    recorder = Recorder(signals.distillation_conformance_violation)
    try:
        step, trainer = build_step(
            settings,
            tmp_path,
            teachers=[teacher_spec("t1"), teacher_spec("t2")],
            auditors={"t1": FixedAudit(rows)},
            config={"peer_margin": 0.0},
        )
        step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    finally:
        recorder.close()
    for loss in step.last.values():
        assert "t1" not in loss.sigma
        assert loss.sigma == {"t2": pytest.approx(1.0)}
        assert loss.cd_reports["t1"].admitted_rows == 0
    assert step._trust.get("t1") == 0.0
    assert step._trust.get("t2") == 1.0
    assert any(event["hard_denied"] and event["teacher"] == "t1" for event in recorder.events)


def test_a_fully_denied_only_teacher_leaves_the_individual_objective(settings, tmp_path):
    rows = {index: ("P1",) for index in range(6)}
    step, trainer = build_step(
        settings, tmp_path, auditors={"t1": FixedAudit(rows)}, config={"peer_margin": 0.0}
    )
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    for loss in step.last.values():
        assert loss.sigma == {}
        assert float(loss.total) == pytest.approx(float(loss.ind), rel=1e-6)
        assert float(loss.cd) == 0.0


def test_a_partially_refused_teacher_keeps_its_clean_rows(settings, tmp_path):
    """NB2: a refused row leaves the loss; the other rows and the teacher stay."""
    narrow = teacher_spec(trust={"tiers": [3]})
    step, trainer = build_step(
        settings, tmp_path, teachers=[narrow], auditors={"t1": FixedAudit({2: ("P11",)})}
    )
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    report = step.last_reports[("peer-1", "t1")]
    assert report.admitted_rows == 5 and report.masked_rows == 1
    assert float(step.last["peer-1"].cd) > 0.0
    assert step.last["peer-1"].sigma == {"t1": pytest.approx(1.0)}
    assert 0.0 < step._trust.get("t1") < 1.0  # decayed by the masked fraction, not pinned


def test_one_hard_denied_row_pins_trust_and_drops_the_teacher_from_sigma(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, auditors={"t1": FixedAudit({2: ("P17",)})})
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    report = step.last_reports[("peer-1", "t1")]
    assert report.admitted_rows == 5 and report.masked_rows == 1
    assert step._trust.get("t1") == 0.0
    assert step.last["peer-1"].sigma == {}
    assert float(step.last["peer-1"].cd) == 0.0


def test_a_tolerated_violation_softens_but_does_not_remove_the_teacher(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, auditors={"t1": FixedAudit({0: ("P8",)})})
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    report = step.last_reports[("peer-1", "t1")]
    assert report.admitted_rows == 6
    assert report.soft_admission is not None and report.soft_admission < 1.0
    assert step.last["peer-1"].sigma == {"t1": pytest.approx(1.0)}


def test_an_auditor_that_raises_is_treated_as_a_denial_not_an_admission(settings, tmp_path):
    class Broken:
        name = "broken"

        def audit(self, bundle, batch, *, profile):
            raise RuntimeError("audit crashed")

    step, trainer = build_step(settings, tmp_path, auditors={"t1": Broken()})
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    verdict = step.last_prepared.verdicts["t1"]
    assert verdict.source == "audit-error"
    assert verdict.admitted == 0
    assert "audit crashed" in verdict.reasons[0]
    assert step.last["peer-1"].sigma == {}


def test_conformance_off_installs_no_fallback_auditor(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, conformance={"ENABLED": False})
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    assert step.last_prepared.verdicts["t1"].source == "disabled"
    assert step.last_prepared.verdicts["t1"].admitted == 6


def test_an_explicit_auditor_still_runs_when_the_gate_is_off(settings, tmp_path):
    rows = {index: ("P17",) for index in range(6)}
    step, trainer = build_step(
        settings, tmp_path, conformance={"ENABLED": False}, auditors={"t1": FixedAudit(rows)}
    )
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    assert step.last_prepared.verdicts["t1"].admitted == 0


def test_trust_collapse_signal_fires_on_the_crossing(settings, tmp_path):
    recorder = Recorder(signals.distillation_trust_collapsed)
    rows = {index: ("P17",) for index in range(6)}
    try:
        step, trainer = build_step(settings, tmp_path, auditors={"t1": FixedAudit(rows)})
        step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
        step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    finally:
        recorder.close()
    assert len(recorder.events) == 1
    assert recorder.events[0]["teacher"] == "t1"


def test_step_signal_carries_losses_and_sigma(cohort):
    step, trainer = cohort
    recorder = Recorder(signals.distillation_step)
    try:
        step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    finally:
        recorder.close()
    assert len(recorder.events) == 1
    assert set(recorder.events[0]["losses"]) == {"peer-1", "peer-2"}


def test_sigma_collapse_signal_fires_once_for_a_dominant_teacher(settings, tmp_path):
    recorder = Recorder(signals.distillation_sigma_collapsed)
    try:
        step, trainer = build_step(
            settings,
            tmp_path,
            teachers=[teacher_spec("a"), teacher_spec("b", weight=1000.0)],
            sigma_mode="uniform",
        )
        step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
        step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    finally:
        recorder.close()
    assert len(recorder.events) == 2  # one crossing per peer, not one per step
    assert {event["peer"] for event in recorder.events} == {"peer-1", "peer-2"}


# -- rendezvous ------------------------------------------------------------------
def test_rendezvous_buffers_until_every_peer_arrives(cohort):
    step, trainer = cohort
    batch = {"states": shared_batch(), "signal": 0.0}
    first = step.train_step(trainer.peers["peer-1"], batch)
    assert not first.stepped
    assert first.metrics["pending"] == 1.0
    assert float(first.loss) == 0.0
    second = step.train_step(trainer.peers["peer-2"], batch)
    assert second.stepped
    assert "pending" not in second.metrics
    assert step.state.step == 1


def test_a_peer_arriving_twice_is_an_error(cohort):
    step, trainer = cohort
    batch = {"states": shared_batch(), "signal": 0.0}
    step.train_step(trainer.peers["peer-1"], batch)
    with pytest.raises(ContractError, match="arrived twice"):
        step.train_step(trainer.peers["peer-1"], batch)


def test_different_batches_in_one_cohort_are_refused(cohort):
    """I1: peers must submit the identical states tensor."""
    step, trainer = cohort
    step.train_step(trainer.peers["peer-1"], {"states": shared_batch(seed=1), "signal": 0.0})
    with pytest.raises(ContractError, match="different batch"):
        step.train_step(trainer.peers["peer-2"], {"states": shared_batch(seed=2), "signal": 0.0})
    # The failed cohort is discarded, so the next one starts clean.
    first = step.train_step(trainer.peers["peer-1"], {"states": shared_batch(), "signal": 0.0})
    assert not first.stepped


def test_void_and_floor_metadata_reach_the_compound_loss(cohort):
    step, trainer = cohort
    batch = {"states": shared_batch(), "signal": 0.0, "void": True, "floor": 0.5}
    step.train_step(trainer.peers["peer-1"], batch)
    step.train_step(trainer.peers["peer-2"], batch)
    assert step._eligible == {"peer-1": False, "peer-2": False}


def test_incomplete_cohort_is_discarded_at_epoch_end(cohort):
    import logging

    class Capture(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    step, trainer = cohort
    step.train_step(trainer.peers["peer-1"], {"states": shared_batch(), "signal": 0.0})
    # Attached to the module logger itself: other tests configure the ``mactf`` logger
    # with propagation off, which would hide the record from ``caplog``.
    logger = logging.getLogger("mactf.distill.step")
    handler = Capture()
    logger.addHandler(handler)
    try:
        step.on_epoch_end(None, type("R", (), {"completed": True})())
    finally:
        logger.removeHandler(handler)
    assert any("incomplete cohort" in message for message in handler.messages)
    assert not step._buffered


# -- weights, blending, metadata ---------------------------------------------------
def test_update_weights_covers_exactly_the_peers(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, managed=False)
    step.step(trainer.peers, shared_batch(), signal={"peer-1": 1.0, "peer-2": 2.0})
    report = step.update_weights()
    assert set(report.weights) == {"peer-1", "peer-2"}
    assert not set(report.weights) & set(step.teachers)
    assert sum(report.weights.values()) == pytest.approx(1.0)
    with pytest.raises(ContractError, match="no score"):
        step.update_weights(scores={"peer-1": 1.0})


def test_assert_peer_only_refuses_a_teacher_in_the_weights(settings, tmp_path):
    from mactf.exceptions import ComplianceError

    step, trainer = build_step(settings, tmp_path, managed=False)
    step.step(trainer.peers, shared_batch(), signal={"peer-1": 1.0, "peer-2": 2.0})
    report = step.update_weights()
    report.weights["t1"] = 0.0
    with pytest.raises(ComplianceError, match="output-only"):
        step._assert_peer_only(report)


def test_an_unmanaged_step_without_an_aggregator_reports_instead_of_blending(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path, managed=False)
    step.step(trainer.peers, shared_batch(), signal={"peer-1": 1.0, "peer-2": 2.0})
    before = {p: parameters_of(peer.module) for p, peer in trainer.peers.items()}
    report = step.blend()
    assert any("not blended" in note for note in report.adjustments)
    for path, peer in trainer.peers.items():
        assert torch.equal(parameters_of(peer.module), before[path])


def test_a_managed_step_never_blends_on_epoch_end(cohort):
    step, trainer = cohort
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    step.on_epoch_end(None, type("R", (), {"completed": True})())
    assert step.state.blends == 0 and step.state.aggregations == 0
    assert step.managed


def test_card_metadata_and_describe_are_json_safe(cohort):
    import json

    step, trainer = cohort
    assert step.card_metadata("peer-1") == {"enabled": True, "step": 0, "teachers": ["t1"]}
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    meta = step.card_metadata("peer-1")
    assert meta["sigma"] == {"t1": pytest.approx(1.0)}
    assert {"per_teacher", "cd_weighted", "peer_margin", "step"} <= set(meta)
    json.dumps(meta)
    description = step.describe()
    assert description["type"] == "co_distill"
    assert description["managed"] is True
    json.dumps(description, default=str)
