"""``--set`` and component specs (document 17 section 15.7, 15.10 rows 29-34)."""

from __future__ import annotations

import copy

import pytest

from mactf import conf
from mactf.conf_overrides import apply_overrides, is_open_container, parse_override
from mactf.exceptions import CommandError


def setting(*texts, trainer=None):
    values = copy.deepcopy(conf.DEFAULTS)
    if trainer is not None:
        values["TRAINER"].update(trainer)
    apply_overrides(values, [parse_override(text) for text in texts], defaults=conf.DEFAULTS)
    return values


def test_a_name_still_replaces_the_spec_wholesale():
    assert setting("TRAINER.WEIGHT_UPDATE=uniform")["TRAINER"]["WEIGHT_UPDATE"] == "uniform"


def test_a_spec_leaf_promotes_a_scalar():
    assert setting("TRAINER.WEIGHT_UPDATE.type=uniform")["TRAINER"]["WEIGHT_UPDATE"] == {"type": "uniform"}


def test_a_params_leaf_promotes_two_levels():
    got = setting("TRAINER.WEIGHT_UPDATE.params.trim=0.3")["TRAINER"]["WEIGHT_UPDATE"]
    # Document 17 15.7.1 writes ``{"params": {...}}``; the name of the scalar it replaces is kept
    # as ``type``, because a spec with no name is refused at load and the lower layer is gone.
    assert got == {"type": "kappa_softmax", "params": {"trim": 0.3}}


def test_params_merge_into_an_existing_mapping_spec():
    got = setting(
        "TRAINER.WEIGHT_UPDATE.params.trim=0.3",
        trainer={"WEIGHT_UPDATE": {"type": "trimmed_mean", "params": {"trim": 0.1, "other": 1}}},
    )["TRAINER"]["WEIGHT_UPDATE"]
    assert got == {"type": "trimmed_mean", "params": {"trim": 0.3, "other": 1}}


def test_an_import_leaf_promotes():
    got = setting("TRAINER.WEIGHT_UPDATE.import=pkg.mod:Cls")["TRAINER"]["WEIGHT_UPDATE"]
    assert got == {"import": "pkg.mod:Cls"}


def test_a_null_default_promotes_too():
    assert setting("TRAINER.AGGREGATOR.type=default")["TRAINER"]["AGGREGATOR"] == {"type": "default"}


def test_a_whole_mapping_is_parsed_as_yaml():
    got = setting("TRAINER.WEIGHT_UPDATE={type: trimmed_mean, params: {trim: 0.2}}")
    assert got["TRAINER"]["WEIGHT_UPDATE"] == {"type": "trimmed_mean", "params": {"trim": 0.2}}


def test_a_non_spec_leaf_is_refused():
    with pytest.raises(CommandError, match="TRAINER.EPOCHS is a int, not a mapping"):
        setting("TRAINER.EPOCHS.value=3")


def test_a_deep_non_spec_path_under_a_scalar_is_refused():
    with pytest.raises(CommandError, match="is a str, not a mapping"):
        setting("TRAINER.WEIGHT_UPDATE.options.x=1")


def test_promotion_does_not_accept_a_float_container():
    with pytest.raises(CommandError, match="float, not a mapping"):
        setting("OPTIMIZER.LR.type=x")


def test_a_bool_container_is_refused():
    with pytest.raises(CommandError, match="bool, not a mapping"):
        setting("COMPONENTS.STRICT_LEGACY.params.x=1")


def test_a_list_still_refuses_a_walk():
    with pytest.raises(CommandError, match="is a list; set the whole list"):
        setting("COMPONENTS.MODULES.type=x")


def test_a_params_write_of_a_mapping_over_the_scalar_is_still_refused():
    with pytest.raises(CommandError):
        setting("TRAINER.WEIGHT_UPDATE.params={}", trainer={"WEIGHT_UPDATE": "uniform"})
    # (the leaf is created inside a promoted mapping, which is fine: it is the name that survives
    # only in the whole-value form)


def test_is_open_container_is_public_and_reads_defaults():
    assert is_open_container({"A": {}}, ["A"])
    assert not is_open_container({"A": {"x": 1}}, ["A"])
    assert is_open_container({}, ["Z"])
