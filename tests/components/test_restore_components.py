"""``ComponentRestorer``: what a checkpoint's manifest means for this run (document 17 section 14.5).

Invariants P3 (fail before writing), P4 (no silent substitution), P5 (legacy tolerance).
"""

from __future__ import annotations

import copy
import logging

import pytest

from component_helpers import OtherPolicy, Policy

from mactf.components import register_policy
from mactf.components.fingerprint import component_manifest
from mactf.components.registry import HubPolicy
from mactf.components.resolver import ComponentResolver
from mactf.components.restore import (
    ABSENT_NOTE,
    ComponentRestorer,
    Finding,
    RestorePlan,
    extract_manifest,
)
from mactf.components.slots import SlotTable
from mactf.exceptions import ImproperlyConfigured

SLOT = "TRAINER.POLICY"
KIND = "trainer_policy"


def _slots(spec) -> SlotTable:
    table = SlotTable(ComponentResolver())
    table.install(SLOT, KIND, spec)
    return table


@pytest.fixture
def mine(component_overlay):
    register_policy("mine", Policy, version="1", scope="run")
    return component_overlay


def _saved(spec="mine"):
    return component_manifest(_slots(spec))


def _plan(saved, live_spec="mine", *, strict=False):
    restorer = ComponentRestorer(_slots(live_spec), policy=HubPolicy(strict_restore=strict))
    return restorer.plan({"identity": {"components": saved}})


def _entry(manifest):
    return manifest[KIND][SLOT]


# -- 12 -----------------------------------------------------------------------------------
def test_a_matching_manifest_is_all_ok(mine):
    plan = _plan(_saved())
    assert plan.checked == 1 and not plan.absent and plan.ok
    assert [f.status for f in plan.findings] == ["ok"]
    assert plan.notes() == [] and plan.warnings == [] and plan.fatal == []
    plan.raise_for_errors()                         # nothing to raise
    assert plan.status_of(SLOT) == "ok" and plan.status_of("nope") is None
    assert plan.describe()["statuses"] == {"ok": 1}


# -- 13, 21 -------------------------------------------------------------------------------
@pytest.mark.parametrize("strict", [False, True])
def test_a_missing_component_is_e093(mine, strict):
    saved = _saved()
    _entry(saved)["name"] = "gone_from_this_build"
    plan = _plan(saved, strict=strict)
    (finding,) = plan.findings
    assert (finding.status, finding.code) == ("missing", "E093") and finding.fatal
    assert not plan.ok
    with pytest.raises(ImproperlyConfigured) as caught:
        plan.raise_for_errors()
    text = str(caught.value)
    assert "E093" in text and "gone_from_this_build" in text and KIND in text and SLOT in text
    assert "COMPONENTS.MODULES" in text


def test_every_fatal_slot_is_named(mine):
    saved = _saved()
    saved[KIND]["TRAINER.OTHER"] = {**_entry(saved), "name": "also_gone", "slot": "TRAINER.OTHER"}
    saved["weight_update"] = {"TRAINER.WEIGHT_UPDATE": {"schema": 1, "kind": "weight_update",
                                                        "name": "ghost", "origin": "project"}}
    table = _slots("mine")
    table.install("TRAINER.OTHER", KIND, "mine")
    table.install("TRAINER.WEIGHT_UPDATE", "weight_update", "uniform")
    _entry(saved)["name"] = "gone_one"
    plan = ComponentRestorer(table, policy=HubPolicy()).plan({"identity": {"components": saved}})
    assert len(plan.fatal) == 3
    with pytest.raises(ImproperlyConfigured) as caught:
        plan.raise_for_errors()
    message = str(caught.value)
    for needle in ("gone_one", "also_gone", "ghost"):
        assert needle in message, message


def test_a_runtime_component_that_is_not_supplied_again_is_a_difference_not_a_missing_plugin(mine):
    """An in-code instance cannot be installed from a module; it is reported, never E093."""
    saved = _saved()
    _entry(saved).update(name="CodePolicy", origin="runtime")
    permissive = _plan(saved)
    assert [(f.status, f.code) for f in permissive.findings] == [("mismatch", "W052")]
    strict = _plan(saved, strict=True)
    assert [(f.status, f.code) for f in strict.findings] == [("mismatch", "E094")]


# -- 14 -----------------------------------------------------------------------------------
def test_a_changed_spec_is_w052_or_e094(mine, caplog):
    saved = _saved({"type": "mine", "params": {"rate": 0.5}})
    live = {"type": "mine", "params": {"rate": 0.9}}
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        plan = _plan(saved, live)
    (finding,) = plan.findings
    assert (finding.status, finding.code) == ("mismatch", "W052") and not finding.fatal
    assert plan.ok and plan.warnings == [finding]
    assert "params" in finding.message and "spec_hash" in finding.message
    plan.raise_for_errors()                              # a warning never blocks a resume
    assert any(r.getMessage().startswith("W052") for r in caplog.records)

    strict = _plan(saved, live, strict=True)
    (hard,) = strict.findings
    assert (hard.status, hard.code) == ("mismatch", "E094") and hard.fatal
    with pytest.raises(ImproperlyConfigured, match="E094.*spec_hash.*params|E094(.|\n)*params"):
        strict.raise_for_errors()


def test_a_changed_name_is_structural(mine, component_overlay):
    register_policy("other", OtherPolicy, scope="run")
    plan = _plan(_saved("mine"), "other")
    (finding,) = plan.findings
    assert finding.status == "mismatch" and "name 'mine' -> 'other'" in finding.message


# -- 15 -----------------------------------------------------------------------------------
def test_a_changed_source_digest_is_e094_under_strict(mine):
    saved = _saved()
    _entry(saved)["source_digest"] = "0123456789abcdef"      # the code was edited, params the same
    assert [(f.status, f.code) for f in _plan(saved).findings] == [("mismatch", "W052")]
    strict = _plan(saved, strict=True)
    (finding,) = strict.findings
    assert finding.code == "E094" and "source_digest" in finding.message
    assert "spec_hash" not in finding.message               # params are untouched


def test_a_version_bump_alone_is_w052_permissive_and_e094_strict(mine):
    saved = _saved()
    _entry(saved)["version"] = "0.9"
    (soft,) = _plan(saved).findings
    assert soft.code == "W052" and "version" in soft.message
    (hard,) = _plan(saved, strict=True).findings
    assert hard.code == "E094"


# -- 16 -----------------------------------------------------------------------------------
@pytest.mark.parametrize("strict", [False, True])
def test_an_orphan_slot_is_w052(mine, strict):
    saved = _saved()
    saved["rl_function.loss"] = {"RL.loss.peer-9": {**_entry(saved), "kind": "rl_function.loss",
                                                    "slot": "RL.loss.peer-9"}}
    plan = _plan(saved, strict=strict)
    orphan = [f for f in plan.findings if f.status == "orphan"]
    assert len(orphan) == 1 and orphan[0].code == "W052" and not orphan[0].fatal
    assert plan.ok                                            # a retired peer's slot never blocks


# -- 17, 18 -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda e: e.pop("schema"), "schema"),
        (lambda e: e.pop("name"), "name"),
        (lambda e: e.pop("kind"), "kind"),
        (lambda e: e.update(schema="1"), "non-integer"),
        (lambda e: e.update(schema=True), "non-integer"),
        (lambda e: e.update(schema=1.0), "non-integer"),
        (lambda e: e.update(name=7), "name"),
        (lambda e: e.update(name=""), "name"),
        (lambda e: e.update(kind="weight_update"), "weight_update"),
        (lambda e: e.update(params=[1, 2]), "params"),
    ],
)
def test_a_malformed_entry_is_e095(mine, mutate, needle):
    saved = _saved()
    mutate(_entry(saved))
    plan = _plan(saved)
    (finding,) = plan.findings
    assert (finding.status, finding.code) == ("malformed", "E095") and finding.fatal, finding
    assert needle in finding.message
    with pytest.raises(ImproperlyConfigured, match="E095"):
        plan.raise_for_errors()


def test_an_entry_that_is_not_a_mapping_is_e095(mine):
    saved = _saved()
    saved[KIND][SLOT] = "just a string"
    (finding,) = _plan(saved).findings
    assert (finding.status, finding.code) == ("malformed", "E095")
    assert "not a mapping" in finding.message


@pytest.mark.parametrize("raw", ["a string", 7, ["x"], {"trainer_policy": 5}, {"trainer_policy": ["x"]}])
def test_a_manifest_that_is_not_nested_mappings_is_e095_never_ignored(mine, raw):
    plan = _plan(raw)
    assert plan.fatal and all(f.code == "E095" for f in plan.fatal)
    with pytest.raises(ImproperlyConfigured, match="E095"):
        plan.raise_for_errors()


@pytest.mark.parametrize("schema", [2, 3, 99, 0, -1])
def test_a_newer_schema_is_e096(mine, schema):
    saved = _saved()
    _entry(saved)["schema"] = schema
    plan = _plan(saved)
    (finding,) = plan.findings
    assert (finding.status, finding.code) == ("unsupported", "E096") and finding.fatal
    assert "Upgrade mactf" in finding.message


def test_a_tampered_entry_cannot_smuggle_a_nested_failure(mine):
    """Hostile values in optional fields are compared, never executed or trusted."""
    saved = _saved()
    _entry(saved).update(params={"x": object()}, spec_hash=object(), version=None, slot=3)
    plan = _plan(saved)                                      # no raw exception
    assert plan.findings[0].status == "mismatch"


# -- 19 -----------------------------------------------------------------------------------
def test_a_legacy_file_is_one_w053(mine, caplog):
    restorer = ComponentRestorer(_slots("mine"), policy=HubPolicy())
    for legacy in (
        {"identity": {"run_id": "r", "episode": 0}},                  # a torch.load mapping
        {"identity": {"run_id": "r", "trainer": {"epoch": 1}}},       # trainer state, no components
        type("Payload", (), {"identity": {"run_id": "r"}})(),         # a CheckpointPayload shape
    ):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="mactf.components"):
            plan = restorer.plan(legacy)
        assert plan.absent and plan.findings == [] and plan.ok and plan.checked == 0
        assert plan.notes() == [ABSENT_NOTE] and ABSENT_NOTE.startswith("W053")
        assert restorer.report(plan) == ABSENT_NOTE
        plan.raise_for_errors()
        assert [r for r in caplog.records if "W053" in r.getMessage()].__len__() == 1


def test_an_empty_manifest_is_not_absent_and_checks_nothing(mine):
    plan = ComponentRestorer(_slots("mine"), policy=HubPolicy()).plan({"identity": {"components": {}}})
    assert not plan.absent and plan.findings == [] and plan.checked == 0
    assert ComponentRestorer(_slots("mine"), policy=HubPolicy()).report(plan) == "components: nothing recorded"


# -- the report and the shapes -------------------------------------------------------------------------
def test_the_report_counts(mine):
    saved = _saved({"type": "mine", "params": {"rate": 0.5}})
    restorer = ComponentRestorer(_slots({"type": "mine", "params": {"rate": 0.9}}), policy=HubPolicy())
    plan = restorer.plan({"identity": {"components": saved}})
    assert restorer.report(plan) == "components: 1 checked, 1 differ, 0 fatal"


def test_extract_manifest_accepts_the_four_shapes_and_rejects_nothing_silently(mine):
    manifest = _saved()
    identity = {"run_id": "r", "components": manifest}
    assert extract_manifest({"identity": identity}) == manifest
    assert extract_manifest(type("P", (), {"identity": identity})()) == manifest
    assert extract_manifest(identity) == manifest
    assert extract_manifest(manifest) == manifest
    assert extract_manifest({"identity": {"run_id": "r"}}) is None
    assert extract_manifest({"identity": "garbage"}) is None
    assert extract_manifest(object()) is None
    assert extract_manifest({"identity": {"components": 5}}) == {"components": {"?": 5}}


def test_the_restorer_never_edits_a_slot(mine):
    table = _slots({"type": "mine", "params": {"rate": 0.9}})
    before = copy.deepcopy(table.fingerprints())
    saved = _saved({"type": "mine", "params": {"rate": 0.5}})
    ComponentRestorer(table, policy=HubPolicy(strict_restore=True)).plan({"identity": {"components": saved}})
    assert table.fingerprints() == before


def test_strict_is_read_from_the_hub_policy_when_none_is_given(mine):
    from mactf.components import overlay

    saved = _saved({"type": "mine", "params": {"rate": 0.5}})
    restorer = ComponentRestorer(_slots({"type": "mine", "params": {"rate": 0.9}}))
    assert restorer.plan({"identity": {"components": saved}}).findings[0].code == "W052"
    with overlay(policy=HubPolicy(strict_restore=True)):
        assert restorer.plan({"identity": {"components": saved}}).findings[0].code == "E094"


def test_finding_and_plan_basics():
    finding = Finding("k", "S", "ok")
    assert finding.describe() == "S [ok]" and not finding.fatal
    assert RestorePlan().ok and RestorePlan(absent=True).notes() == [ABSENT_NOTE]
