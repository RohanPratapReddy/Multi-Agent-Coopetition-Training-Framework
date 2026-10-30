"""Adversarial verification of D2 on the untrusted surface: a request to change a live run's
components (the API's ``PATCH runs/{id}/components``) cannot import outside the allow-list, cannot
set what the framework injects, and cannot touch a locked slot (doc 17 sections 5.4, 15, 16)."""

from __future__ import annotations

import sys
import types

import pytest

from mactf.components import HUB, HubPolicy
from mactf.exceptions import (
    ComponentError,
    ComponentImportDenied,
    ComponentLocked,
    ComponentMissing,
    ComponentSpecError,
)
from mactf.orchestration.live import _request_components

from persist_support import default_trainer


def _ask(trainer, slot, raw, **kw):
    return _request_components("r1", trainer, slot, raw, trusted=False, boundary="episode",
                               immediate=False, **kw)


@pytest.fixture
def trainer(settings, tmp_path):
    return default_trainer(settings, tmp_path, epochs=2).build()


@pytest.mark.parametrize("slot", ["TRAINER.WEIGHT_UPDATE", "TRAINER.PEER_STRATEGY"])
@pytest.mark.parametrize("target", ["os:system", "subprocess:Popen", "builtins:eval", "pickle:loads"])
def test_an_import_is_refused_by_default_and_nothing_is_queued(trainer, slot, target):
    before = dict(trainer.slots.pending())
    with pytest.raises(ComponentImportDenied) as caught:
        _ask(trainer, slot, {"import": target})
    assert caught.value.code == "E006"
    assert dict(trainer.slots.pending()) == before


def test_an_import_outside_the_allow_list_is_refused_even_when_imports_are_on(trainer):
    with HUB.scoped(policy=HubPolicy(allow_import=True, import_allowlist=("adv_ok.*",))):
        with pytest.raises(ComponentImportDenied, match="IMPORT_ALLOWLIST"):
            _ask(trainer, "TRAINER.WEIGHT_UPDATE", {"import": "os:system"})


def test_a_walk_through_an_allowed_module_is_refused(trainer, monkeypatch):
    import os

    module = types.ModuleType("adv_ok.mod")
    module.os = os
    monkeypatch.setitem(sys.modules, "adv_ok", types.ModuleType("adv_ok"))
    monkeypatch.setitem(sys.modules, "adv_ok.mod", module)
    with HUB.scoped(policy=HubPolicy(allow_import=True, import_allowlist=("adv_ok.*",))):
        with pytest.raises(ComponentImportDenied, match="reaches module 'os'"):
            _ask(trainer, "TRAINER.WEIGHT_UPDATE", {"import": "adv_ok.mod:os.system"})


@pytest.mark.parametrize("raw", [
    {"type": "no_such_strategy"}, {"type": "uniform", "params": {"nonexistent": 1}},
    {"params": {"x": 1}}, 7, ["uniform"], {"type": "uniform", "import": "no way"},
])
def test_a_bad_spec_is_refused_with_a_code_and_queues_nothing(trainer, raw):
    before = dict(trainer.slots.pending())
    with pytest.raises(ComponentError) as caught:
        _ask(trainer, "TRAINER.WEIGHT_UPDATE", raw)
    assert caught.value.code in {"E001", "E003", "E004", "E005", "E006", "E007"}, caught.value.code
    assert dict(trainer.slots.pending()) == before


def test_a_locked_slot_is_not_changed_through_the_api(trainer):
    """Refused at the request, or - if queued - refused at the boundary with the old strategy running."""
    before = type(trainer.weight_plan.strategy_for("root", trainer.aggregator)).__name__
    with HUB.scoped(policy=HubPolicy(locked=("slot:TRAINER.WEIGHT_UPDATE*", "slot:TRAINER.WEIGHT_UPDATE.*"))):
        try:
            _ask(trainer, "TRAINER.WEIGHT_UPDATE", "uniform")
        except ComponentLocked:
            return
        trainer.run_epoch(0)
    assert type(trainer.weight_plan.strategy_for("root", trainer.aggregator)).__name__ == before
