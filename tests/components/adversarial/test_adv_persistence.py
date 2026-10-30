"""Adversarial verification of D2 across checkpoint and resume (doc 17 section 14).

The restorer verifies; it never installs. So no manifest - missing, doctored or claiming the
default components - can switch a guard off: the live components are the ones in settings, and
the guards wrap what is live.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from mactf.components import register_scorer
from mactf.exceptions import ComplianceGuardViolation

from adv_support import inject_breach, run_epochs
from p2_support import configure, make_trainer, tune
from persist_support import default_trainer
from scoring_helpers import penalty_variant


def _zero(out, ctx):
    return replace(out, amount=0.0, void_rules=(), breakdown={**out.breakdown, "amounts": {}})


def _evil_trainer(settings, tmp_path):
    register_scorer("adv_zero_pen", penalty_variant(_zero), scope="run")
    configure(settings, SCORING={"STAGES": {"penalty": "adv_zero_pen"}})
    tune(settings, EPOCHS=2)
    trainer = make_trainer(settings, tmp_path, peers=2).build()
    inject_breach(trainer, ["peer-1"])
    return trainer


def _honest_payload(settings, tmp_path):
    trainer = default_trainer(settings, tmp_path / "honest", epochs=1).build()
    trainer.run_epoch(0)
    return trainer.state_dict()


@pytest.mark.parametrize("doctor", ["keep", "drop_components", "claim_default", "empty_components"])
def test_no_manifest_turns_the_guard_off(doctor, settings, tmp_path, component_overlay):
    payload = copy.deepcopy(_honest_payload(settings, tmp_path))
    identity = payload["identity"]
    if doctor == "drop_components":
        identity.pop("components", None)
    elif doctor == "empty_components":
        identity["components"] = {}
    elif doctor == "claim_default":
        identity["components"] = {"scorer": {"scoring.penalty": {
            "schema": 1, "kind": "scorer", "name": "penalty", "origin": "builtin"}}}
    trainer = _evil_trainer(settings, tmp_path / "evil")
    trainer.load_state_dict(payload, strict=False)         # a verification, never an installation
    with pytest.raises(ComplianceGuardViolation):
        run_epochs(trainer, 1)


def test_resuming_with_a_cheaper_stage_than_the_run_started_with_is_still_refused(settings, tmp_path, component_overlay):
    first = default_trainer(settings, tmp_path / "run", epochs=2).build()
    first.fit(epochs=1)
    register_scorer("adv_zero_pen", penalty_variant(_zero), scope="run")
    configure(settings, SCORING={"STAGES": {"penalty": "adv_zero_pen"}})
    second = default_trainer(settings, tmp_path / "run", epochs=2, seed=9).build()
    inject_breach(second, ["peer-1"])
    with pytest.raises(ComplianceGuardViolation):
        second.fit(epochs=2, resume=True)
