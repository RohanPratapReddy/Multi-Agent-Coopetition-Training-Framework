"""Checkpoint section and resume behaviour (document 14 section 11)."""

from __future__ import annotations

import json

import pytest
import torch

from distill_support import (
    Tiny,
    build_step,
    distillation_block,
    shared_batch,
    signals as peer_signals,
    teacher_spec,
    tune,
)
from mactf.distill.conformance import TrustProfile, admission
from mactf.distill.state import (
    STATE_FORMAT_VERSION,
    STATE_HISTORY,
    DistillationState,
    DistillResumeReport,
)
from mactf.training import MACTFTrainer


class HardDeny:
    name = "hard"

    def audit(self, bundle, batch, *, profile: TrustProfile):
        return admission(teacher="t1", profile=profile, fired={0: ("P17",)}, batch=batch.batch)


def run_steps(step, trainer, count: int = 2) -> None:
    for index in range(count):
        step.step(trainer.peers, shared_batch(seed=index), signal=peer_signals(trainer))


def test_state_record_keeps_a_bounded_numeric_history():
    state = DistillationState()
    for index in range(STATE_HISTORY + 20):
        state.record({"cd": index, "total": index * 2})
    assert len(state.loss_history) == STATE_HISTORY
    assert state.loss_history[0]["cd"] == 20.0
    assert state.loss_history[-1]["cd"] == float(STATE_HISTORY + 19)
    assert all(isinstance(value, float) for row in state.loss_history for value in row.values())


def test_state_dict_is_json_serializable_and_complete(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path)
    run_steps(step, trainer, 2)
    payload = step.state_dict()
    json.dumps(payload)
    assert payload["format_version"] == STATE_FORMAT_VERSION
    assert payload["enabled"] is True
    assert payload["teacher_names"] == ["t1"]
    assert set(payload["teacher_digests"]) == {"t1"}
    assert payload["step"] == 2
    assert payload["trust_state"]["values"]["t1"] == 1.0
    assert payload["trust_profiles"]["t1"]
    assert payload["sigma_state"]
    assert payload["config"]["tau"] == pytest.approx(0.5)


def test_teacher_weights_are_never_stored(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path)
    run_steps(step, trainer, 1)
    text = json.dumps(step.state_dict())
    # A 3x4 Linear teacher would put twelve floats in a nested list if weights leaked.
    weights = step.teachers["t1"].module.weight.detach().flatten().tolist()
    assert str(weights[0]) not in text


def test_round_trip_restores_step_sigma_and_digest(settings, tmp_path):
    first, trainer = build_step(settings, tmp_path)
    run_steps(first, trainer, 3)
    payload = first.state_dict()

    second, _ = build_step(settings, tmp_path / "again")
    report = second.load_state_dict(payload)
    assert isinstance(report, DistillResumeReport)
    assert second._step == 3
    assert second.sigma.state_dict() == first.sigma.state_dict()
    assert "teacher_digest:t1" in report.restored
    assert "sigma_state" in report.restored
    assert "trust_state" in report.restored
    assert report.notes == ()
    assert second.state_dict()["step"] == 3


def test_a_retrained_teacher_is_a_note_not_an_error(settings, tmp_path):
    first, trainer = build_step(settings, tmp_path, seed=0)
    run_steps(first, trainer, 1)
    payload = first.state_dict()

    other, _ = build_step(settings, tmp_path / "other", seed=7)
    assert other.teachers["t1"].digest != first.teachers["t1"].digest
    report = other.load_state_dict(payload)
    assert any("digest changed" in note for note in report.notes)
    assert "teacher_digest:t1" not in report.restored


def test_a_checkpointed_teacher_that_is_no_longer_configured_is_reported(settings, tmp_path):
    first, trainer = build_step(
        settings, tmp_path, teachers=[teacher_spec("t1"), teacher_spec("t2")]
    )
    run_steps(first, trainer, 1)
    payload = first.state_dict()

    second, _ = build_step(settings, tmp_path / "second")
    report = second.load_state_dict(payload)
    assert any("'t2'" in note and "does not configure" in note for note in report.notes)


def test_a_new_teacher_has_no_recorded_digest(settings, tmp_path):
    first, trainer = build_step(settings, tmp_path)
    run_steps(first, trainer, 1)
    payload = first.state_dict()

    second, _ = build_step(
        settings, tmp_path / "second", teachers=[teacher_spec("t1"), teacher_spec("t2")]
    )
    report = second.load_state_dict(payload)
    assert any("'t2'" in note and "new since" in note for note in report.notes)
    assert "teacher_digest:t2" in " ".join(report.skipped)


def test_resume_never_raises_trust(settings, tmp_path):
    """I7: a resumed run must not re-earn trust it lost."""
    spoiled, trainer = build_step(
        settings, tmp_path, auditors={"t1": HardDeny()}, conformance={"SIGMA_TRUST_MIN": False}
    )
    run_steps(spoiled, trainer, 1)
    lost = spoiled._trust.get("t1")
    assert lost < 1.0
    payload = spoiled.state_dict()

    fresh, _ = build_step(settings, tmp_path / "fresh")
    assert fresh._trust.get("t1") == 1.0
    report = fresh.load_state_dict(payload)
    assert fresh._trust.get("t1") == pytest.approx(lost)
    assert any("cannot re-earn" in note for note in report.notes)


def test_resume_keeps_the_lower_of_stored_and_current_trust(settings, tmp_path):
    clean, trainer = build_step(settings, tmp_path)
    run_steps(clean, trainer, 1)
    payload = clean.state_dict()  # stored trust is 1.0

    spoiled, trainer2 = build_step(
        settings,
        tmp_path / "spoiled",
        auditors={"t1": HardDeny()},
        conformance={"SIGMA_TRUST_MIN": False},
    )
    run_steps(spoiled, trainer2, 1)
    current = spoiled._trust.get("t1")
    spoiled.load_state_dict(payload)
    assert spoiled._trust.get("t1") == pytest.approx(current)


def test_a_version_one_checkpoint_resumes_with_a_note(settings, tmp_path):
    first, trainer = build_step(settings, tmp_path)
    run_steps(first, trainer, 1)
    payload = first.state_dict()
    for key in ("trust_state", "trust_profiles", "audit_digests", "format_version"):
        payload.pop(key, None)

    second, _ = build_step(settings, tmp_path / "second")
    report = second.load_state_dict(payload)
    assert "trust_state" in report.skipped
    assert "trust_profiles" in report.skipped
    assert any("no trust history" in note for note in report.notes)
    assert second._trust.get("t1") == second._trust.init


def test_a_changed_trust_profile_is_reported(settings, tmp_path):
    first, trainer = build_step(settings, tmp_path)
    run_steps(first, trainer, 1)
    payload = first.state_dict()

    second, _ = build_step(
        settings, tmp_path / "second", teachers=[teacher_spec(trust={"tiers": [3]})]
    )
    report = second.load_state_dict(payload)
    assert any("trust profile changed" in note for note in report.notes)


def test_a_payload_without_sigma_history_restarts_from_the_prior(settings, tmp_path):
    first, trainer = build_step(settings, tmp_path)
    run_steps(first, trainer, 1)
    payload = first.state_dict()
    payload.pop("sigma_state")
    second, _ = build_step(settings, tmp_path / "second")
    report = second.load_state_dict(payload)
    assert "sigma_state" in report.skipped
    assert any("restart from the prior" in note for note in report.notes)


def test_resume_report_is_json_serializable(settings, tmp_path):
    first, trainer = build_step(settings, tmp_path)
    run_steps(first, trainer, 1)
    second, _ = build_step(settings, tmp_path / "second")
    report = second.load_state_dict(first.state_dict())
    json.dumps(report.as_dict())
    assert report.as_dict()["section"] == "distillation"


def test_the_ramp_resumes_instead_of_restarting(settings, tmp_path):
    """A resumed run at step 0 with warmup would silently turn distillation off again."""
    first, trainer = build_step(settings, tmp_path, config={"warmup_steps": 2})
    run_steps(first, trainer, 3)
    payload = first.state_dict()
    second, _ = build_step(settings, tmp_path / "second", config={"warmup_steps": 2})
    second.load_state_dict(payload)
    assert second._step == 3
    second.step(second._peers, shared_batch(), signal=peer_signals(trainer))
    assert second.last["peer-1"].lambda_cd_effective > 0.0


def test_trainer_checkpoint_carries_the_distillation_section(settings, tmp_path):
    tune(settings)
    torch.manual_seed(0)
    trainer = MACTFTrainer(
        model=Tiny(),
        num_peers=2,
        settings=settings,
        checkpoint_dir=tmp_path / "ckpt",
        distillation=distillation_block(),
    )
    trainer.fit(epochs=2)
    payload = trainer.state_dict()
    section = payload["identity"]["trainer"]["distillation"]
    assert section["enabled"] is True
    assert section["step"] == 2
    assert section["teacher_names"] == ["t1"]

    torch.manual_seed(0)
    resumed = MACTFTrainer(
        model=Tiny(),
        num_peers=2,
        settings=settings,
        checkpoint_dir=tmp_path / "ckpt2",
        distillation=distillation_block(),
    ).build()
    resumed.checkpoint.run_id = trainer.checkpoint.run_id
    resumed.load_state_dict(payload)
    assert resumed.distillation_resume is not None
    assert resumed.distillation_resume.section == "distillation"
    assert resumed._trainer_policy._step == 2
