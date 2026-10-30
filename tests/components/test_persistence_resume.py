"""Save with custom components, restore in a fresh registry state, behave identically (section 14.5, 14.7).

"Fresh process" is a fresh overlay: the extensions are registered again from nothing, exactly
what a new Python process does when it imports the project's modules.
"""

from __future__ import annotations

import copy
import logging

import pytest
import torch

from persist_support import (
    CUSTOM_OPTIMIZER, CUSTOM_TRAINER, EVERY_KIND_BLOCKS, custom_trainer, default_trainer, params,
    register_every_kind, register_extensions,
)

from p2_support import configure, make_trainer, tune
from p3_examples import EmaSoftmax

from mactf.components import overlay, register_weight_update
from mactf.db import Registry
from mactf.exceptions import ImproperlyConfigured
from mactf.orchestration.checkpoint_v2 import CheckpointPayload

W053 = "W053: this checkpoint records no components; nothing to verify"


def _equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def _fresh(settings):
    """A settings object a second 'process' would load: loaded again from the same module."""
    from mactf import conf

    return conf.load_settings("mactf_test_settings")


# -- the round trip -----------------------------------------------------------------------------------------
def test_a_customised_run_resumes_bit_identically_in_a_fresh_registry_state(settings, tmp_path):
    with overlay():
        register_extensions()
        reference = custom_trainer(_fresh(settings), tmp_path / "ref")
        reference.fit(epochs=4)
        expected = params(reference)

        first = custom_trainer(_fresh(settings), tmp_path / "run")
        first.fit(epochs=2)
        manifest = first._component_manifest()
    assert {"optimizer", "scheduler", "weight_update"} <= set(manifest)

    with overlay():                                     # a new process: nothing registered yet
        register_extensions()
        second = custom_trainer(_fresh(settings), tmp_path / "run", seed=999)   # other init proves the restore
        result = second.fit(epochs=4, resume=True)
        assert result.epochs == 2
        assert result.component_notes == () and second.component_notes == []
        _equal(expected, params(second))
        assert second._component_manifest() == manifest
        # the stateful strategy and the scheduler continued where they were
        assert second.weight_plan.strategy_for("root", second.aggregator)._ema == \
            reference.weight_plan.strategy_for("root", reference.aggregator)._ema
        assert second.optimizers["peer-1"].backend.param_groups[0]["lr"] == \
            reference.optimizers["peer-1"].backend.param_groups[0]["lr"]


def test_every_kind_round_trips(settings, tmp_path):
    def build(directory, seed=3, epochs=4):
        tune(settings, EPOCHS=epochs, CHECKPOINT_EVERY=1, DETERMINISTIC=True)
        configure(settings, **copy.deepcopy(EVERY_KIND_BLOCKS))
        trainer = make_trainer(settings, directory, seed=seed, peers=2)
        trainer.run_id = "kinds-run"
        return trainer

    with overlay():
        register_every_kind()
        reference = build(tmp_path / "ref")
        reference.fit(epochs=4)
        first = build(tmp_path / "run")
        first.fit(epochs=2)
        manifest = first._component_manifest()
    assert len(manifest) == 8
    with overlay():
        register_every_kind()
        second = build(tmp_path / "run", seed=999)
        result = second.fit(epochs=4, resume=True)
        assert result.component_notes == ()
        _equal(params(reference), params(second))


def test_the_plain_state_dict_path_applies_the_same_checks(settings, tmp_path):
    with overlay():
        register_extensions()
        first = custom_trainer(_fresh(settings), tmp_path / "a").build()
        first.run_epoch(0)
        payload = first.state_dict()
    with overlay():
        register_extensions()
        second = custom_trainer(_fresh(settings), tmp_path / "b", seed=7).build()
        report = second.load_state_dict(payload, strict=False)
        assert report.component_notes == () and second.component_notes == []


# -- P3: nothing moves after a fatal finding --------------------------------------------------------------------
def _snapshot(trainer):
    return (
        params(trainer),
        {path: copy.deepcopy(o.backend.state_dict()) for path, o in trainer.optimizers.items()},
        torch.get_rng_state().clone(),
        trainer.global_step, trainer.start_epoch, trainer._restored,
        {label: copy.deepcopy(c.state_dict()) for label, c in trainer.controllers.items()},
    )


def _same(left, right):
    _equal(left[0], right[0])
    for path in left[1]:
        assert repr(left[1][path]) == repr(right[1][path])
    assert torch.equal(left[2], right[2]) and left[3:6] == right[3:6] and left[6] == right[6]


def test_nothing_is_restored_after_a_fatal_finding(settings, tmp_path):
    with overlay():
        register_extensions()
        first = custom_trainer(_fresh(settings), tmp_path / "run")
        first.fit(epochs=2)
    registry = Registry(tmp_path / "reg.sqlite3")
    registry.migrate()
    with overlay():
        # the module providing ema_softmax and Lion is not installed in this process, and the
        # settings no longer name them: the saved run needed them
        second = custom_trainer(_fresh(settings), tmp_path / "run", seed=999, trainer_block={"WEIGHT_UPDATE": "uniform"},
                                optimizer={"CLASS": "AdamW", "LR": 1e-3})
        second.build()
        second.checkpoint.registry = registry
        before = _snapshot(second)
        rows_before = registry.checkpoints(second.run_id)
        with pytest.raises(ImproperlyConfigured) as caught:
            second.fit(epochs=4, resume=True)
        message = str(caught.value)
        assert "E093" in message and "ema_softmax" in message and "Lion" in message
        assert "TRAINER.WEIGHT_UPDATE" in message and "OPTIMIZER" in message          # every fatal slot, one error
        _same(before, _snapshot(second))
        assert registry.checkpoints(second.run_id) == rows_before
        assert second.component_notes == [] or all(not n.startswith("W05") for n in second.component_notes)


def test_a_poisoned_payload_leaves_the_trainer_untouched(settings, tmp_path):
    with overlay():
        register_extensions()
        first = custom_trainer(_fresh(settings), tmp_path / "a").build()
        first.run_epoch(0)
        payload = first.state_dict()
        second = custom_trainer(_fresh(settings), tmp_path / "b", seed=5).build()
        before = _snapshot(second)
        for mutate, code in (
            (lambda e: e.update(schema=2), "E096"),
            (lambda e: e.pop("name"), "E095"),
            (lambda e: e.update(name="not_registered_anywhere"), "E093"),
        ):
            poisoned = copy.deepcopy(payload)
            mutate(poisoned["identity"]["components"]["weight_update"]["TRAINER.WEIGHT_UPDATE"])
            with pytest.raises(ImproperlyConfigured, match=code):
                second.load_state_dict(poisoned, strict=False)
            _same(before, _snapshot(second))


# -- P4: no silent substitution -----------------------------------------------------------------------------------
def _resume_notes(settings, tmp_path, register, *, strict=False, **overrides):
    """Save under the original extensions, resume under ``register`` and return (trainer, result)."""
    with overlay():
        register_extensions()
        custom_trainer(_fresh(settings), tmp_path / "run").fit(epochs=2)
    with overlay():
        register()
        resumed_settings = _fresh(settings)
        if strict:
            configure(resumed_settings, COMPONENTS={"STRICT_RESTORE": True})
        trainer = custom_trainer(resumed_settings, tmp_path / "run", seed=999, **overrides)
        return trainer, trainer.fit(epochs=3, resume=True)


def test_a_changed_spec_resumes_with_w052_naming_the_field(settings, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        trainer, result = _resume_notes(
            settings, tmp_path, register_extensions,
            trainer_block={"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alpha": 0.9}}},
        )
    (note,) = [n for n in result.component_notes if n.startswith("W052")]
    assert "TRAINER.WEIGHT_UPDATE" in note and "params" in note and "0.5" in note and "0.9" in note
    assert any(r.getMessage().startswith("W052") for r in caplog.records)
    assert result.epochs == 1                                       # the run continued


def test_a_changed_spec_under_strict_restore_is_e094_before_anything_moves(settings, tmp_path):
    with overlay():
        register_extensions()
        custom_trainer(_fresh(settings), tmp_path / "run").fit(epochs=2)
    with overlay():
        register_extensions()
        strict = _fresh(settings)
        configure(strict, COMPONENTS={"STRICT_RESTORE": True})
        trainer = custom_trainer(
            strict, tmp_path / "run", seed=999,
            trainer_block={"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alpha": 0.9}}},
        ).build()
        before = _snapshot(trainer)
        with pytest.raises(ImproperlyConfigured, match="E094") as caught:
            trainer.fit(epochs=3, resume=True)
        assert "params" in str(caught.value) and "spec_hash" in str(caught.value)
        _same(before, _snapshot(trainer))


class EmaSoftmaxEdited(EmaSoftmax):
    """The same component after somebody edited its code."""

    def apply(self, vectors, scores, **kwargs):
        return super().apply(vectors, scores, **kwargs)


def _register_edited():
    register_weight_update("ema_softmax", EmaSoftmaxEdited, version="1", scope="run")
    from persist_support import register_optimizer, register_scheduler
    from p2_support import Lion, step_then_flat

    register_optimizer("Lion", Lion, version="1.0", scope="run")
    register_scheduler("step_then_flat", step_then_flat, version="1", scope="run")


def test_source_digest_drift_is_w052_and_under_strict_e094(settings, tmp_path):
    _, result = _resume_notes(settings, tmp_path, _register_edited)
    (note,) = [n for n in result.component_notes if n.startswith("W052")]
    assert "source_digest" in note and "params" not in note
    strict_root = tmp_path / "strict"
    with pytest.raises(ImproperlyConfigured, match="E094.*source_digest|E094(.|\n)*source_digest"):
        _resume_notes(settings, strict_root, _register_edited, strict=True)


def test_version_skew_is_w052(settings, tmp_path):
    def newer():
        register_extensions(version="2")

    _, result = _resume_notes(settings, tmp_path, newer)
    (note,) = [n for n in result.component_notes if n.startswith("W052")]
    assert "version" in note and "'1' -> '2'" in note


def test_an_orphan_entry_is_w052_and_the_resume_goes_on(settings, tmp_path):
    with overlay():
        register_extensions()
        first = custom_trainer(_fresh(settings), tmp_path / "a").build()
        first.run_epoch(0)
        payload = first.state_dict()
        payload["identity"]["components"].setdefault("rl_function.loss", {})["RL.loss.peer-9"] = {
            "schema": 1, "kind": "rl_function.loss", "name": "retired_loss", "origin": "project",
            "version": "0", "spec_hash": "x", "params": {}, "slot": "RL.loss.peer-9",
        }
        second = custom_trainer(_fresh(settings), tmp_path / "b", seed=5).build()
        report = second.load_state_dict(payload, strict=False)
        assert [n for n in report.component_notes if "RL.loss.peer-9" in n and n.startswith("W052")]
        assert len(report.component_notes) == 1


# -- P5: legacy tolerance ---------------------------------------------------------------------------------------------------
def test_a_legacy_file_is_one_w053_on_resume(settings, tmp_path, caplog):
    """A checkpoint written by the previous release (a default run writes exactly its bytes)."""
    first = default_trainer(_fresh(settings), tmp_path / "run")
    first.fit(epochs=2)
    stored = first.checkpoint.load(first.checkpoint.latest().path)
    assert "components" not in stored.identity                      # what every pre-doc-17 file looks like
    second = default_trainer(_fresh(settings), tmp_path / "run", seed=999, epochs=3)
    with caplog.at_level(logging.INFO, logger="mactf.components"):
        result = second.fit(epochs=3, resume=True)
    assert result.component_notes == (W053,) and second.component_notes == [W053]
    assert len([r for r in caplog.records if "W053" in r.getMessage()]) == 1
    assert result.epochs == 1
    assert result.as_dict()["component_notes"] == [W053]


def test_a_legacy_file_restored_into_a_customised_run_is_one_w053(settings, tmp_path):
    first = default_trainer(_fresh(settings), tmp_path / "run")
    first.fit(epochs=2)
    with overlay():
        register_extensions()
        second = custom_trainer(_fresh(settings), tmp_path / "run", seed=999)
        result = second.fit(epochs=3, resume=True)
    assert result.component_notes == (W053,)


def test_a_legacy_payload_dict_is_one_w053(settings, tmp_path):
    first = default_trainer(_fresh(settings), tmp_path / "a").build()
    first.run_epoch(0)
    payload = first.state_dict()
    payload["identity"].pop("components", None)                      # a v2 file from before this document
    second = default_trainer(_fresh(settings), tmp_path / "b", seed=5).build()
    report = second.load_state_dict(payload, strict=False)
    assert report.component_notes == (W053,)
    assert report.as_dict()["component_notes"] == [W053]


def test_orchestrator_files_are_untouched(settings, tmp_path):
    """T3: a v1 file has no ``identity`` at all, so it is a legacy file: one W053."""
    from mactf.orchestration.checkpoint import CheckpointManager
    from mactf.orchestration.checkpoint_v2 import MACTFCheckpoint
    from mactf.runtime import load_checkpoint

    class Peer:
        def state_dict(self):
            return {"w": torch.zeros(2)}

    manager = CheckpointManager(tmp_path / "v1", run_id="orch")
    record = manager.save(
        episode=1, peers={"peer-1": Peer()}, reference=None, controller=None, warnings=None,
        runtime={}, kappa=1.0, entropy=1.0, score=0.5,
    )
    raw = load_checkpoint(record.path)
    assert "identity" not in raw and "components" not in raw
    migrated = CheckpointPayload.from_dict(MACTFCheckpoint._migrate_v1(raw))
    assert "components" not in migrated.identity
    trainer = default_trainer(_fresh(settings), tmp_path / "b").build()
    assert trainer._component_plan(migrated).notes() == [W053]


def test_a_default_resume_reports_nothing_new_beyond_the_note(settings, tmp_path):
    first = default_trainer(_fresh(settings), tmp_path / "run")
    first.fit(epochs=2)
    second = default_trainer(_fresh(settings), tmp_path / "run", seed=999, epochs=3)
    second.fit(epochs=3, resume=True)
    assert second.component_notes == [W053]
    assert not second.weight_warnings and not second.strategy_warnings


# -- tampered files -------------------------------------------------------------------------------------------------------------
def test_a_tampered_checkpoint_file_fails_verification(settings, tmp_path):
    from mactf.runtime import atomic_torch_save

    with overlay():
        register_extensions()
        first = custom_trainer(_fresh(settings), tmp_path / "run")
        first.fit(epochs=2)
        path = first.checkpoint.latest().path
        raw = torch.load(path, weights_only=False)
        raw["identity"]["components"]["weight_update"]["TRAINER.WEIGHT_UPDATE"]["name"] = "forged"
        atomic_torch_save(raw, path)
        verification = first.checkpoint.verify(path)
        assert not verification.ok and any("identity" in p for p in verification.problems)
