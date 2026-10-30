"""The component manifest and its diff (document 17 section 14.3; invariants P1, P2, P6)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from component_helpers import OtherPolicy, Policy, Term

import mactf.training.policy  # noqa: F401  (registers the built-in policies)
from mactf.components import register_policy, register_reward_term
from mactf.components.fingerprint import (
    ENTRY_SCHEMA,
    IDENTITY_KEY,
    RECORD_FLAGS,
    Change,
    SlotView,
    component_manifest,
    diff,
    entry_for,
    recorded_slots,
    records_in_checkpoint,
    render,
)
from mactf.components.resolver import ComponentResolver
from mactf.components.slots import SlotTable
from mactf.components.registry import HUB


def _table() -> SlotTable:
    return SlotTable(ComponentResolver())


class Unsafe(Policy):
    unsafe = True


class Asks(Policy):
    record_in_checkpoint = True


# -- P1, P2 -------------------------------------------------------------------------------
def test_a_builtin_with_default_params_is_not_recorded():
    slots = _table()
    slots.install("TRAINER.POLICY", "trainer_policy", "default")
    assert component_manifest(slots) == {}
    # a built-in with parameters set in configuration is still not recorded
    slots.install("T.REWARD", "reward_term", {"type": "severity_scaled"})
    assert component_manifest(slots) == {}


def test_a_project_component_is_recorded(component_overlay):
    register_policy("mine", Policy, version="2.1", scope="run")
    slots = _table()
    slots.install("TRAINER.POLICY", "trainer_policy", {"type": "mine", "params": {"rate": 0.5}})
    manifest = component_manifest(slots)
    entry = manifest["trainer_policy"]["TRAINER.POLICY"]
    assert set(entry) >= {
        "schema", "kind", "name", "origin", "version", "spec_hash", "params", "slot",
        "source_digest",
    }
    assert entry["schema"] == ENTRY_SCHEMA == 1
    assert (entry["kind"], entry["name"], entry["version"]) == ("trainer_policy", "mine", "2.1")
    assert entry["origin"] == "project" and entry["slot"] == "TRAINER.POLICY"
    assert entry["params"] == {"rate": 0.5}
    assert entry["spec_hash"] and entry["source_digest"]


def test_an_unsafe_policy_is_recorded_with_its_flag(component_overlay):
    assert "unsafe" in RECORD_FLAGS and "record_in_checkpoint" in RECORD_FLAGS
    register_policy("risky", Unsafe, scope="run")
    slots = _table()
    slots.install("TRAINER.POLICY", "trainer_policy", "risky")
    entry = component_manifest(slots)["trainer_policy"]["TRAINER.POLICY"]
    assert entry["unsafe"] is True


def test_the_record_flag_forces_a_builtin_shaped_slot_into_the_manifest():
    component = HUB.registry("trainer_policy").get("default")
    assert records_in_checkpoint(component) is False
    assert records_in_checkpoint(component, Asks()) is True
    assert records_in_checkpoint(component, Unsafe()) is True
    # a mock that answers every attribute is not "asking": only a real True counts
    class Everything:
        def __getattr__(self, name):
            return object()

    assert records_in_checkpoint(component, Everything()) is False


def test_an_in_code_instance_of_a_builtin_class_is_not_an_extension():
    from mactf.coopetition.weights import UniformAverage

    slots = _table()
    slots.adopt("TRAINER.WEIGHT_UPDATE", "weight_update", UniformAverage(), name="uniform")
    assert slots.get("TRAINER.WEIGHT_UPDATE").component.origin.value == "runtime"
    assert component_manifest(slots) == {}


def test_an_in_code_instance_of_a_subclass_is_recorded():
    from mactf.coopetition.weights import UniformAverage

    class Mine(UniformAverage):
        pass

    slots = _table()
    slots.adopt("TRAINER.WEIGHT_UPDATE", "weight_update", Mine(), name="uniform")
    entry = component_manifest(slots)["weight_update"]["TRAINER.WEIGHT_UPDATE"]
    assert entry["origin"] == "runtime"


def test_a_builtin_replacement_is_recorded():
    from mactf.components import overlay
    from mactf.components.registry import HubPolicy

    with overlay(policy=HubPolicy(allow_builtin_override=True)):
        register_policy("default", OtherPolicy, replace=True, scope="run")
        slots = _table()
        slots.install("TRAINER.POLICY", "trainer_policy", "default")
        entry = component_manifest(slots)["trainer_policy"]["TRAINER.POLICY"]
        assert entry["overrides_builtin"] is True
        assert records_in_checkpoint(slots.get("TRAINER.POLICY").component)


# -- determinism --------------------------------------------------------------------------
def test_manifest_is_sorted_and_grouped_by_kind(component_overlay):
    register_policy("zeta", Policy, scope="run")
    register_policy("alpha", OtherPolicy, scope="run")
    register_reward_term("t", Term, scope="run")
    slots = _table()
    slots.install("B.POLICY", "trainer_policy", "zeta")
    slots.install("A.POLICY", "trainer_policy", "alpha")
    slots.install("Z.TERM", "reward_term", "t")
    manifest = component_manifest(slots)
    assert list(manifest) == ["reward_term", "trainer_policy"]
    assert list(manifest["trainer_policy"]) == ["A.POLICY", "B.POLICY"]
    assert recorded_slots(manifest) == [
        ("reward_term", "Z.TERM"), ("trainer_policy", "A.POLICY"), ("trainer_policy", "B.POLICY"),
    ]
    again = component_manifest(slots)
    assert json.dumps(manifest, sort_keys=False, default=str) == json.dumps(again, default=str)


def test_manifest_is_stable_across_processes(tmp_path):
    """The hash of the manifest does not depend on dict insertion order or the process."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys, json\n"
        f"sys.path.insert(0, {str(__import__('pathlib').Path(__file__).parent)!r})\n"
        "from component_helpers import Policy\n"
        "from mactf.components import register_policy\n"
        "from mactf.components.fingerprint import component_manifest\n"
        "from mactf.components.resolver import ComponentResolver\n"
        "from mactf.components.slots import SlotTable\n"
        "register_policy('mine', Policy, version='3')\n"
        "s = SlotTable(ComponentResolver())\n"
        "s.install('TRAINER.POLICY', 'trainer_policy', {'type': 'mine', 'params': {'rate': 2.0}})\n"
        "print(json.dumps(component_manifest(s), sort_keys=True))\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-I", str(script)], capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(outputs) == 1


# -- the view --------------------------------------------------------------------------------
def test_a_slot_view_serves_extras_beside_the_table(component_overlay):
    register_policy("mine", Policy, scope="run")
    slots = _table()
    slots.install("TRAINER.POLICY", "trainer_policy", "mine")
    other = _table()
    other.install("EXTRA.POLICY", "trainer_policy", "mine")
    view = SlotView(slots, [other.get("EXTRA.POLICY")])
    assert view.paths() == ["EXTRA.POLICY", "TRAINER.POLICY"]
    assert set(component_manifest(view)["trainer_policy"]) == {"EXTRA.POLICY", "TRAINER.POLICY"}
    with pytest.raises(Exception, match="NOPE"):
        view.get("NOPE")


# -- diff ---------------------------------------------------------------------------------
def _manifest(**fields):
    base = {
        "schema": 1, "kind": "weight_update", "name": "trimmed_mean", "origin": "project",
        "version": "0", "spec_hash": "sha256:a", "source_digest": "sha256:s",
        "params": {"trim": 0.2}, "slot": "TRAINER.WEIGHT_UPDATE",
    }
    base.update(fields)
    return {"weight_update": {"TRAINER.WEIGHT_UPDATE": base}}


def test_diff_reports_one_present_change_for_a_missing_slot():
    saved = {**_manifest(), "optimizer": {"OPTIMIZER": {"name": "Lion", "kind": "optimizer"}}}
    live = _manifest()
    changes = diff(saved, live)
    assert [(c.kind, c.slot, c.field) for c in changes] == [("optimizer", "OPTIMIZER", "present")]
    assert changes[0].describe() == "OPTIMIZER: absent now"
    assert diff(live, saved)[0].describe() == "OPTIMIZER: absent when saved"
    assert changes[0].structural


def test_diff_reports_params_and_spec_hash_together():
    changes = diff(_manifest(), _manifest(params={"trim": 0.1}, spec_hash="sha256:b"))
    fields = [change.field for change in changes]
    assert fields == ["spec_hash", "params"]
    assert changes[1].saved == {"trim": 0.2} and changes[1].live == {"trim": 0.1}


def test_diff_of_equal_manifests_is_empty_and_none_is_empty():
    assert diff(_manifest(), _manifest()) == []
    assert diff(None, None) == []
    assert diff(None, _manifest())[0].field == "present"


def test_a_version_bump_alone_is_reported_but_not_structural():
    (change,) = diff(_manifest(), _manifest(version="1"))
    assert change.field == "version" and not change.structural


def test_render_matches_the_documented_shape():
    text = render(diff(_manifest(), _manifest(name="ema_softmax")))
    assert text == "TRAINER.WEIGHT_UPDATE: name 'trimmed_mean' -> 'ema_softmax'"


def test_entry_for_matches_the_slot_fingerprint(component_overlay):
    register_policy("mine", Policy, version="9", scope="run")
    slots = _table()
    slot = slots.install("TRAINER.POLICY", "trainer_policy", "mine")
    entry = entry_for(slot)
    fingerprint = slots.fingerprints()["TRAINER.POLICY"]
    for key in ("kind", "name", "origin", "version", "spec_hash", "source_digest"):
        assert entry[key] == fingerprint[key]
    assert IDENTITY_KEY == "components"
    assert Change("k", "s", "name", 1, 2).structural
