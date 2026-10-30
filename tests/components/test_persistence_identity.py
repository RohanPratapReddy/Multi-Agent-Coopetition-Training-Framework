"""What a checkpoint's identity does and does not gain (document 17 section 14.2, 14.4, 14.8).

P1 is the invariant: a run that customises nothing writes the bytes it always wrote.
"""

from __future__ import annotations

import torch
import pytest

import default_identity as recorder
from persist_support import (
    FRAMEWORK_IDENTITY_KEYS, custom_trainer, default_trainer, register_extensions,
)

from mactf.components.kinds import Boundary
from mactf.exceptions import ImproperlyConfigured
from mactf.orchestration.checkpoint_v2 import CheckpointContext, _HashWriter


def _payload(trainer, epoch=0):
    context = CheckpointContext(trainer=trainer, episode=epoch).resolve()
    return trainer.checkpoint._collect(context, episode=epoch, step=0)


# -- P1 ---------------------------------------------------------------------------------------------
def test_a_default_run_writes_the_same_identity_hash(settings, tmp_path):
    """P1: no components and no trainer extras; since P5 the one new key is ``engine`` (14.4, 14.8)."""
    trainer = default_trainer(settings, tmp_path).build()
    assert trainer._identity_extra() == {"engine": "trainer"} and trainer._component_manifest() == {}
    trainer.run_epoch(0)
    assert trainer._identity_extra() == {"engine": "trainer"}
    payload = _payload(trainer, 1)
    # the 15 framework keys plus ``engine``, and not one more
    assert sorted(payload.identity) == sorted([*FRAMEWORK_IDENTITY_KEYS, "engine"])
    assert "components" not in payload.identity and "trainer" not in payload.identity
    original = {key: payload.identity[key] for key in FRAMEWORK_IDENTITY_KEYS}
    assert payload.section_hashes()["identity"] != _HashWriter.digest(original)      # 14.8: this section changes
    assert payload.section_hashes()["identity"] == _HashWriter.digest({**original, "engine": "trainer"})


def test_a_default_run_with_in_code_builtin_instances_writes_no_components(settings, tmp_path):
    from mactf.coopetition.strategy import KappaSoftmaxStrategy
    from mactf.coopetition.weights import UniformAverage

    trainer = default_trainer(
        settings, tmp_path, weight_update=UniformAverage(), peer_strategy=KappaSoftmaxStrategy()
    ).build()
    assert {path for path in trainer.slots.paths()} >= {"TRAINER.WEIGHT_UPDATE", "TRAINER.PEER_STRATEGY"}
    assert trainer._component_manifest() == {}
    # (the plan's own in-code record is P3's ``identity["trainer"]["weight_plan"]``, not a component)
    assert "components" not in trainer._identity_extra()


def test_checkpoint_bytes_unchanged_when_all_builtin(tmp_path):
    """The recorded fixture's identity key set and content hash are reproduced exactly."""
    fixtures = recorder.load_fixtures()
    recorded = fixtures["run"]["torch"]
    current = torch.__version__.split("+")[0].rsplit(".", 1)[0]
    if recorded != current:
        pytest.skip(f"fixtures were recorded with torch {recorded}.x, this is {current}.x")
    run = recorder.run_trainer(tmp_path)
    want = fixtures["scalars"]["trainer"]
    assert run["checkpoint_identity_keys"] == want["checkpoint_identity_keys"]
    # P5 added the one key (``engine``); the other fifteen sections and the hash over them are the
    # pre-change ones, which is what the next two lines compare against the fixture.
    assert sorted(run["checkpoint_identity_keys"]) == sorted([*FRAMEWORK_IDENTITY_KEYS, "engine"])
    assert run["checkpoint_sections"] == want["checkpoint_sections"]
    assert run["checkpoint_content_hash"] == want["checkpoint_content_hash"] == fixtures["checkpoint"]


# -- P-G1, P-G2, 14.4.3 ----------------------------------------------------------------------------------------
def test_periodic_checkpoint_carries_the_manifest(settings, tmp_path, component_overlay):
    register_extensions()
    trainer = custom_trainer(settings, tmp_path, epochs=2)
    trainer.fit(epochs=2)
    stored = trainer.checkpoint.load(trainer.checkpoint.latest().path)
    manifest = stored.identity["components"]
    assert set(manifest) == {"optimizer", "scheduler", "weight_update"}
    assert manifest["weight_update"]["TRAINER.WEIGHT_UPDATE"]["name"] == "ema_softmax"
    assert manifest["optimizer"]["OPTIMIZER"]["name"] == "Lion"
    assert manifest["scheduler"]["OPTIMIZER:scheduler"]["params"] == {"drop_at": 3}
    # the trainer-owned state a plain resume needs travels with it, and nothing else of the trainer's
    assert set(stored.identity["trainer"]) == {"weight_plan"}
    assert stored.identity["engine"] == "trainer"
    # one function: the file and the live trainer agree
    assert manifest == trainer._component_manifest()


def test_state_dict_carries_the_manifest(settings, tmp_path, component_overlay):
    register_extensions()
    trainer = custom_trainer(settings, tmp_path).build()
    trainer.run_epoch(0)
    data = trainer.state_dict()
    assert data["identity"]["components"] == trainer._component_manifest() != {}
    assert {"spec", "history", "callbacks", "weight_plan"} <= set(data["identity"]["trainer"])


def test_watchdog_snapshot_carries_the_manifest(settings, tmp_path, component_overlay):
    from types import SimpleNamespace

    from mactf.training.trainer import _CheckpointBridge

    register_extensions()
    trainer = custom_trainer(settings, tmp_path).build()
    trainer.run_epoch(0)
    bridge = _CheckpointBridge(trainer.checkpoint, trainer)
    health = SimpleNamespace(epoch=0, step=1, scores={"peer-1": 1.0}, kappa=1.0, entropy=1.0)
    record = bridge.save(health, reason="watchdog")
    stored = trainer.checkpoint.load(record.path)
    assert stored.identity["components"] == trainer._component_manifest()
    assert "weight_plan" in stored.identity["trainer"]


def test_a_hand_built_context_gets_the_manifest_too(settings, tmp_path, component_overlay):
    register_extensions()
    trainer = custom_trainer(settings, tmp_path).build()
    record = trainer.checkpoint.save(CheckpointContext(trainer=trainer, episode=0), reason="manual")
    assert "components" in trainer.checkpoint.load(record.path).identity


def test_engine_key_is_written_once(settings, tmp_path, component_overlay):
    """14.4: ``identity["engine"]`` is what ``E091`` compares; written for a default run and a custom one."""
    assert default_trainer(settings, tmp_path / "plain").build()._identity_extra() == {"engine": "trainer"}
    # an unbuilt trainer has nothing to say
    assert default_trainer(settings, tmp_path / "unbuilt")._identity_extra() == {}
    register_extensions()
    trainer = custom_trainer(settings, tmp_path).build()
    extra = trainer._identity_extra()
    assert extra["engine"] == "trainer" and list(extra).count("engine") == 1


def test_identity_extra_cannot_replace_a_framework_key(settings, tmp_path):
    trainer = default_trainer(settings, tmp_path).build()
    for key in FRAMEWORK_IDENTITY_KEYS:
        context = CheckpointContext(trainer=trainer, identity_extra={key: "x"}).resolve()
        with pytest.raises(ImproperlyConfigured, match=key):
            trainer.checkpoint._collect(context, episode=0, step=0)


# -- P6: the registry row ----------------------------------------------------------------------------------------
class _Rows:
    def __init__(self):
        self.calls = []
        self.checkpoints = 0

    def record_checkpoint(self, *args, **kwargs):
        self.checkpoints += 1

    def record_components(self, run_id, components, *, source="engine"):
        self.calls.append((run_id, components, source))


def test_a_default_run_makes_no_registry_component_write(settings, tmp_path):
    trainer = default_trainer(settings, tmp_path)
    trainer.checkpoint = trainer._build_checkpoint()
    rows = trainer.checkpoint.registry = _Rows()
    trainer.fit(epochs=2)
    assert rows.checkpoints == 2 and rows.calls == []


def test_every_save_rewrites_the_row_and_an_emptied_manifest_is_null(settings, tmp_path, component_overlay):
    register_extensions()
    trainer = custom_trainer(settings, tmp_path, epochs=2)
    trainer.checkpoint = trainer._build_checkpoint()
    rows = trainer.checkpoint.registry = _Rows()
    trainer.fit(epochs=2)
    assert [call[2] for call in rows.calls] == ["checkpoint", "checkpoint"]
    assert rows.calls[-1][1] == trainer._component_manifest()
    # the components go away (a swap back to built-ins): the row is cleared once, then left alone
    writer = trainer.checkpoint
    trainer.slots.swap("TRAINER.WEIGHT_UPDATE", "uniform", boundary=Boundary.RUN_START,
                       inject={"aggregator": trainer.aggregator})
    trainer.optimizer_plan._entries.clear()
    context = CheckpointContext(trainer=trainer, episode=9)
    writer.save(context, reason="manual")
    assert rows.calls[-1][1] is None
    count = len(rows.calls)
    writer.save(CheckpointContext(trainer=trainer, episode=10), reason="manual")
    assert len(rows.calls) == count


def test_a_registry_without_the_method_is_not_an_error(settings, tmp_path, component_overlay):
    register_extensions()
    trainer = custom_trainer(settings, tmp_path, epochs=1)
    trainer.checkpoint = trainer._build_checkpoint()

    class Old:
        def record_checkpoint(self, *args, **kwargs):
            pass

    trainer.checkpoint.registry = Old()
    trainer.fit(epochs=1)


def test_no_run_row_is_not_an_error(settings, tmp_path, component_overlay):
    from mactf.db import Registry

    register_extensions()
    trainer = custom_trainer(settings, tmp_path, epochs=1)
    trainer.checkpoint = trainer._build_checkpoint()
    registry = Registry(tmp_path / "r.sqlite3")
    registry.migrate()
    trainer.checkpoint.registry = registry
    trainer.fit(epochs=1)               # no start_run: the file is written, the row does not exist
    assert registry.components(trainer.run_id) is None


def test_the_run_row_equals_the_checkpoint_key_and_follows_a_swap(settings, tmp_path, component_overlay):
    """P6: one function writes both. A mid-run swap reaches the row at the next save."""
    import json

    from mactf.db import Registry
    from mactf.components.kinds import Boundary

    register_extensions()
    trainer = custom_trainer(settings, tmp_path, epochs=2)
    trainer.checkpoint = trainer._build_checkpoint()
    registry = Registry(tmp_path / "reg.sqlite3")
    registry.migrate()
    registry.start_run(trainer.run_id, settings={}, runtime={})
    trainer.checkpoint.registry = registry
    trainer.fit(epochs=2)
    stored = trainer.checkpoint.load(trainer.checkpoint.latest().path)
    row = registry.components(trainer.run_id)
    assert row == json.loads(json.dumps(stored.identity["components"])) != {}
    assert registry.get_run(trainer.run_id)["components"] == row

    trainer.slots.swap("TRAINER.WEIGHT_UPDATE", "uniform", boundary=Boundary.RUN_START,
                       inject={"aggregator": trainer.aggregator})
    assert registry.components(trainer.run_id) == row            # durable only in the file until the next save
    trainer.checkpoint.save(CheckpointContext(trainer=trainer, episode=7), reason="manual")
    after = registry.components(trainer.run_id)
    assert "weight_update" not in after and set(after) == {"optimizer", "scheduler"}
