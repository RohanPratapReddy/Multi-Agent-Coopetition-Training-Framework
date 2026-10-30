"""``mactf.conf_files``: the layout table, the directory resolver and the loader."""

from __future__ import annotations

import os
import time

import pytest
import yaml

from mactf.conf_files import (
    CONF_LAYOUT,
    OWNED_KEYS,
    PYTHON_ONLY_KEYS,
    RESERVED_KEYS,
    layout_by_filename,
    layout_by_template,
    load_conf_tree,
    owner_of,
    resolve_conf_dir,
)
from mactf.exceptions import ImproperlyConfigured


def _write(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(yaml.safe_dump(payload), encoding="utf-8")


# -- CONF_LAYOUT ------------------------------------------------------------------
def test_layout_has_nineteen_distinct_files():
    names = [entry.filename for entry in CONF_LAYOUT]
    assert len(names) == 19
    assert len(set(names)) == 19
    assert all(name.endswith(".yaml") for name in names)


def test_every_owned_key_has_exactly_one_owner():
    seen: dict[str, str] = {}
    for entry in CONF_LAYOUT:
        for key in entry.owns:
            assert key not in seen, f"{key} owned by both {seen[key]} and {entry.filename}"
            seen[key] = entry.filename


def test_owned_keys_is_the_union_of_every_entry():
    assert OWNED_KEYS == frozenset().union(*[set(entry.owns) for entry in CONF_LAYOUT])


def test_layout_indexes_round_trip():
    by_name, by_template = layout_by_filename(), layout_by_template()
    assert set(by_name) == {entry.filename for entry in CONF_LAYOUT}
    for entry in CONF_LAYOUT:
        assert by_name[entry.filename] is entry
        assert entry.template_name == entry.filename + ".tpl"
        assert by_template[entry.template_name] is entry


def test_owner_of_names_the_file_for_every_owned_key_and_none_otherwise():
    for entry in CONF_LAYOUT:
        for key in entry.owns:
            assert owner_of(key) == entry.filename
    assert owner_of("SECRET_KEY") is None
    assert owner_of("NOT_A_BLOCK") is None


def test_the_new_blocks_are_owned_by_the_documented_files():
    assert owner_of("TRAINER") == owner_of("TRAINING") == "trainer.yaml"
    assert owner_of("ORCHESTRATOR") == owner_of("COOPETITION") == "orchestrator.yaml"
    assert owner_of("OPTIMIZER") == "optimizer.yaml"
    assert owner_of("LOSS") == "loss.yaml"
    assert owner_of("LOGGER") == "logger.yaml"
    assert owner_of("TELEMETRY") == "telemetry.yaml"
    assert owner_of("WATCHDOG") == "watchdog.yaml"
    assert owner_of("CHECKPOINT") == "checkpoint.yaml"
    assert owner_of("CORS") == owner_of("API") == "api.yaml"
    assert owner_of("DISTILLATION") == "distillation.yaml"


def test_reserved_keys_are_never_owned_by_a_file():
    assert not RESERVED_KEYS & OWNED_KEYS
    assert "SECRET_KEY" in RESERVED_KEYS and "CONF_DIR" in RESERVED_KEYS


# -- resolve_conf_dir -------------------------------------------------------------
def test_explicit_absolute_conf_dir_wins_over_the_implicit_one(tmp_path):
    (tmp_path / "conf").mkdir()
    _write(tmp_path / "conf", "conf.yaml", {"SEED": 1})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert resolve_conf_dir({"CONF_DIR": elsewhere}, tmp_path) == elsewhere.resolve()


def test_relative_conf_dir_resolves_against_root(tmp_path):
    (tmp_path / "settings_tree").mkdir()
    assert resolve_conf_dir({"CONF_DIR": "settings_tree"}, tmp_path) == (
        tmp_path / "settings_tree"
    ).resolve()


def test_missing_explicit_conf_dir_raises(tmp_path):
    with pytest.raises(ImproperlyConfigured, match="not a directory"):
        resolve_conf_dir({"CONF_DIR": tmp_path / "nope"}, tmp_path)


def test_implicit_conf_directory_with_one_yaml_is_found(tmp_path):
    _write(tmp_path / "conf", "conf.yaml", {"SEED": 1})
    assert resolve_conf_dir({}, tmp_path) == (tmp_path / "conf").resolve()
    assert resolve_conf_dir({"CONF_DIR": None}, tmp_path) == (tmp_path / "conf").resolve()


def test_implicit_conf_directory_without_yaml_is_none(tmp_path):
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "notes.txt").write_text("x", encoding="utf-8")
    assert resolve_conf_dir({}, tmp_path) is None


def test_no_directory_is_none(tmp_path):
    assert resolve_conf_dir({}, tmp_path) is None


# -- load_conf_tree ---------------------------------------------------------------
def test_no_directory_loads_nothing_and_says_so(tmp_path):
    values, notes = load_conf_tree({}, tmp_path)
    assert values == {}
    assert notes == ["no conf/ directory; settings.py is the configuration"]


def test_merge_order_is_layout_order_not_filesystem_order(tmp_path):
    conf_dir = tmp_path / "conf"
    first_key = {
        entry.filename: next(key for key in entry.owns if key not in PYTHON_ONLY_KEYS)
        for entry in CONF_LAYOUT
    }
    # created in reverse layout order, with mtimes running backwards too, so neither
    # directory order nor modification time can accidentally produce the layout order
    for index, entry in enumerate(reversed(CONF_LAYOUT)):
        _write(conf_dir, entry.filename, {first_key[entry.filename]: {"FROM": entry.filename}})
        stamp = time.time() - index
        os.utime(conf_dir / entry.filename, (stamp, stamp))
    values, notes = load_conf_tree({}, tmp_path)
    assert list(values) == [first_key[entry.filename] for entry in CONF_LAYOUT]
    assert not [line for line in notes if line.endswith("absent, defaults apply")]


def test_an_absent_file_contributes_nothing_and_produces_a_note(tmp_path):
    _write(tmp_path / "conf", "conf.yaml", {"SEED": 7})
    values, notes = load_conf_tree({}, tmp_path)
    assert values == {"SEED": 7}
    assert "trainer.yaml: absent, defaults apply" in notes
    assert "conf.yaml: absent, defaults apply" not in notes


def test_conf_notes_record_the_resolved_directory(tmp_path):
    _write(tmp_path / "conf", "conf.yaml", {"SEED": 7})
    _, notes = load_conf_tree({}, tmp_path)
    assert notes[0] == f"conf/ resolved to {(tmp_path / 'conf').resolve()}"


def test_unknown_file_name_is_refused_by_name(tmp_path):
    _write(tmp_path / "conf", "conf.yaml", {"SEED": 7})
    _write(tmp_path / "conf", "extra.yaml", {"SEED": 8})
    with pytest.raises(ImproperlyConfigured, match=r"no loader reads: extra\.yaml"):
        load_conf_tree({}, tmp_path)


def test_a_foreign_block_is_refused_naming_both_files(tmp_path):
    _write(tmp_path / "conf", "logger.yaml", {"TRAINING": {"EPOCHS": 2}})
    with pytest.raises(ImproperlyConfigured) as caught:
        load_conf_tree({}, tmp_path)
    message = str(caught.value)
    assert "logger.yaml" in message and "trainer.yaml" in message and "TRAINING" in message


def test_an_unknown_block_is_refused(tmp_path):
    _write(tmp_path / "conf", "conf.yaml", {"NOT_A_BLOCK": 1})
    with pytest.raises(ImproperlyConfigured, match="unknown block NOT_A_BLOCK"):
        load_conf_tree({}, tmp_path)


@pytest.mark.parametrize("key", ["SECRET_KEY", "CONF_DIR", "BASE_DIR", "SETTINGS_MODULE"])
def test_reserved_keys_are_refused(tmp_path, key):
    _write(tmp_path / "conf", "conf.yaml", {key: "x"})
    with pytest.raises(ImproperlyConfigured, match=f"conf.yaml sets {key}"):
        load_conf_tree({}, tmp_path)


@pytest.mark.parametrize("key", sorted(PYTHON_ONLY_KEYS))
def test_python_only_keys_are_refused_in_yaml(tmp_path, key):
    _write(tmp_path / "conf", "mactf.yaml", {key: []})
    with pytest.raises(ImproperlyConfigured, match=f"mactf.yaml sets {key}"):
        load_conf_tree({}, tmp_path)


def test_non_strict_notes_a_foreign_block_instead_of_raising(tmp_path):
    _write(tmp_path / "conf", "logger.yaml", {"TRAINING": {"EPOCHS": 2}})
    values, notes = load_conf_tree({}, tmp_path, strict=False)
    assert values == {"TRAINING": {"EPOCHS": 2}}
    assert any("accepted: strict=False" in line and "TRAINING" in line for line in notes)


def test_a_none_mapping_block_is_refused(tmp_path):
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "trainer.yaml").write_text("TRAINING:\n", encoding="utf-8")
    with pytest.raises(ImproperlyConfigured, match="TRAINING must be a mapping"):
        load_conf_tree({}, tmp_path)


def test_a_nested_mapping_deep_merges_across_non_strict_files(tmp_path):
    _write(tmp_path / "conf", "trainer.yaml", {"TRAINING": {"EPOCHS": 2, "BATCH_SIZE": 8}})
    _write(tmp_path / "conf", "logger.yaml", {"TRAINING": {"EPOCHS": 5}})
    values, _ = load_conf_tree({}, tmp_path, strict=False)
    # logger.yaml sorts after trainer.yaml in CONF_LAYOUT or before it; either way
    # the sibling key must survive rather than the mapping being replaced.
    assert values["TRAINING"]["BATCH_SIZE"] == 8
    assert set(values["TRAINING"]) == {"EPOCHS", "BATCH_SIZE"}


def test_relative_paths_root_resolves_against_the_project_directory(tmp_path):
    _write(tmp_path / "conf", "conf.yaml", {"PATHS": {"ROOT": ".", "VAR": "var"}})
    values, _ = load_conf_tree({}, tmp_path)
    assert values["PATHS"]["ROOT"] == str(tmp_path.resolve())
    assert values["PATHS"]["VAR"] == "var"


def test_the_two_new_blocks_have_owners():
    from mactf.conf_files import owner_of

    assert owner_of("COMPONENTS") == "components.yaml"
    assert owner_of("COMPLIANCE") == "components.yaml"
    assert owner_of("RL") == "rl.yaml"


def test_load_order_places_components_second_and_rl_after_loss():
    names = [entry.filename for entry in CONF_LAYOUT]
    assert names[:3] == ["conf.yaml", "mactf.yaml", "components.yaml"]
    assert names.index("rl.yaml") == names.index("loss.yaml") + 1


def test_a_key_written_by_the_wrong_file_names_both(tmp_path):
    from mactf.conf_files import load_conf_tree
    from mactf.exceptions import ImproperlyConfigured

    conf = tmp_path / "conf"
    _write(conf, "scorer.yaml", {"COMPLIANCE": {"GUARD_MODE": "fallback"}})
    with pytest.raises(ImproperlyConfigured) as caught:
        load_conf_tree({"CONF_DIR": str(conf)}, tmp_path)
    assert "scorer.yaml" in str(caught.value) and "components.yaml" in str(caught.value)
