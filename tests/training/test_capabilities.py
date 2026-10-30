"""``TrainerCapabilities`` distillation flags (doc 14 section 19.4)."""

from __future__ import annotations

import json

import pytest

from mactf.training import TrainerCapabilities
from train_support import make_trainer, tuned  # noqa: F401
from test_trainer_distillation_arg import block


def test_defaults_are_false():
    capabilities = TrainerCapabilities()
    assert capabilities.supports_distillation is False
    assert capabilities.distillation_conformance is False
    assert capabilities.as_dict()["supports_distillation"] is False
    assert capabilities.as_dict()["distillation_conformance"] is False


def test_an_off_trainer_reports_neither_flag(tuned, tmp_path):
    capabilities = make_trainer(tuned, tmp_path).build().capabilities()
    assert capabilities.supports_distillation is False
    assert capabilities.distillation_conformance is False


def test_an_on_trainer_reports_both_flags(tuned, tmp_path):
    capabilities = make_trainer(tuned, tmp_path, distillation=block()).build().capabilities()
    assert capabilities.supports_distillation is True
    assert capabilities.distillation_conformance is True


def test_conformance_flag_follows_conformance_enabled(tuned, tmp_path):
    spec = block(CONFORMANCE={"ENABLED": False})
    capabilities = make_trainer(tuned, tmp_path, distillation=spec).build().capabilities()
    assert capabilities.supports_distillation is True
    assert capabilities.distillation_conformance is False


def test_false_over_an_enabled_block_reports_off(tuned, tmp_path):
    tuned["DISTILLATION"].update({"ENABLED": True, **block()})
    capabilities = make_trainer(tuned, tmp_path, distillation=False).build().capabilities()
    assert capabilities.supports_distillation is False


def test_capabilities_are_json_safe_and_keep_the_existing_fields(tuned, tmp_path):
    payload = make_trainer(tuned, tmp_path, distillation=block()).build().capabilities().as_dict()
    json.dumps(payload)
    for key in (
        "optimizer_granularity",
        "supports_amp",
        "supports_hierarchy",
        "supports_peft",
        "supports_aggregation",
        "deterministic",
        "reason",
    ):
        assert key in payload
    assert payload["supports_aggregation"] is True


def test_the_probe_does_not_depend_on_an_importable_step_module(tuned, tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "mactf.distill" else real(name, *a, **k)
    )
    capabilities = make_trainer(tuned, tmp_path, distillation=block()).build().capabilities()
    assert capabilities.supports_distillation is False
    assert capabilities.distillation_conformance is True
