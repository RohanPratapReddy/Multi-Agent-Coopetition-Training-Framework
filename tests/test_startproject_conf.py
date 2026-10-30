"""``startproject`` writes the ``conf/`` tree, and the project it writes loads."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conf_project import load, scaffold
from mactf.conf_files import CONF_LAYOUT
from mactf.exceptions import CommandError
from mactf.management.commands import startproject


def _written(tmp_path):
    return {path.relative_to(tmp_path).as_posix() for path in startproject.write_project("demo_swarm", target_root=tmp_path)}


def test_layout_writes_twenty_one_files_plus_gitignore(tmp_path):
    names = _written(tmp_path)
    assert len(startproject.LAYOUT) == 21
    assert len(names) == 22
    assert all(name.startswith("demo_swarm/") for name in names)
    assert "demo_swarm/.gitignore" in names


def test_conf_holds_exactly_the_seventeen_yaml_files(tmp_path):
    names = _written(tmp_path)
    in_conf = {name for name in names if name.startswith("demo_swarm/conf/")}
    assert in_conf == {f"demo_swarm/conf/{entry.filename}" for entry in CONF_LAYOUT}
    assert len(in_conf) == 17
    assert all(name.endswith(".yaml") for name in in_conf)


def test_no_configs_directory_is_created(tmp_path):
    _written(tmp_path)
    assert not (tmp_path / "demo_swarm" / "configs").exists()


def test_the_project_name_appears_only_where_it_should(tmp_path):
    _written(tmp_path)
    mentions = set()
    for path in (tmp_path / "demo_swarm").rglob("*"):
        if path.is_file() and "demo_swarm" in path.read_text(encoding="utf-8"):
            mentions.add(path.relative_to(tmp_path / "demo_swarm").as_posix())
    # settings.py names the peers app; peers.yaml's header names the module that defines
    # MLPPeer; manage.py points MACTF_SETTINGS_MODULE at the project
    assert {"settings.py", "conf/peers.yaml", "manage.py"} <= mentions
    assert not any(name.startswith("configs/") for name in mentions)
    for path in (tmp_path / "demo_swarm").rglob("*"):
        if path.is_file():
            assert "{{ project_name }}" not in path.read_text(encoding="utf-8"), path


def test_every_conf_file_parses_to_a_mapping(tmp_path):
    _written(tmp_path)
    for entry in CONF_LAYOUT:
        payload = yaml.safe_load((tmp_path / "demo_swarm" / "conf" / entry.filename).read_text(encoding="utf-8"))
        assert isinstance(payload, dict) and payload, entry.filename


def test_every_conf_file_writes_only_keys_it_owns(tmp_path):
    _written(tmp_path)
    for entry in CONF_LAYOUT:
        payload = yaml.safe_load((tmp_path / "demo_swarm" / "conf" / entry.filename).read_text(encoding="utf-8"))
        assert set(payload) <= set(entry.owns), (entry.filename, set(payload) - set(entry.owns))


def test_generated_project_loads(tmp_path, monkeypatch):
    scaffold(tmp_path)
    loaded = load(tmp_path, monkeypatch)
    conf_dir = (tmp_path / "demo_swarm" / "conf").resolve()
    assert Path(str(loaded["CONF_DIR"])).resolve() == conf_dir
    assert any(str(conf_dir) in note for note in loaded["_CONF_NOTES"])
    assert [peer["name"] for peer in loaded["PEERS"]] == ["peer_alpha", "peer_beta", "peer_gamma"]
    assert all(peer["class"] == "MLPPeer" for peer in loaded["PEERS"])
    assert loaded["TRAINING"]["EPOCHS"] == 3
    assert loaded["PATHS"]["ROOT"] == str((tmp_path / "demo_swarm").resolve())


def test_settings_py_has_no_peers_so_they_come_from_peers_yaml(tmp_path):
    scaffold(tmp_path)
    text = (tmp_path / "demo_swarm" / "settings.py").read_text(encoding="utf-8")
    assert "PEERS" not in text
    assert "TRAINING" not in text
    assert "CONFIG_FILES: list[str] = []" in text


def test_a_trainer_yaml_edit_arrives_through_the_fold(tmp_path, monkeypatch):
    package = scaffold(tmp_path)
    path = package / "conf" / "trainer.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("EPOCHS: 3", "EPOCHS: 9", 1), encoding="utf-8")
    loaded = load(tmp_path, monkeypatch)
    assert loaded["TRAINER"]["EPOCHS"] == 9
    assert loaded["TRAINING"]["EPOCHS"] == 9


def test_layout_and_conf_layout_agree(tmp_path):
    written = {name.removeprefix("conf/") for _, name in startproject.LAYOUT if name.startswith("conf/")}
    assert written == {entry.filename for entry in CONF_LAYOUT}
    for entry in CONF_LAYOUT:
        source, package = startproject.template_source(f"conf/{entry.template_name}")
        assert source.is_file(), entry.template_name
        assert package == startproject.CONF_TEMPLATE_PACKAGE


def test_every_packaged_template_is_in_the_layout():
    shipped = {path.name for path in startproject.conf_template_root().glob("*.tpl")}
    assert shipped == {entry.template_name for entry in CONF_LAYOUT}


def test_a_template_missing_from_the_package_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        startproject,
        "LAYOUT",
        startproject.LAYOUT + (("conf/foo.yaml.tpl", "conf/foo.yaml"),),
    )
    with pytest.raises(CommandError, match=r"template 'conf/foo\.yaml\.tpl' is missing from mactf\.templates\.conf"):
        startproject.write_project("demo_swarm", target_root=tmp_path)


def test_gitignore_names_the_runtime_directories(tmp_path):
    package = scaffold(tmp_path)
    lines = (package / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert {"var/", "var/runs/", "var/teachers/", "var/audits/", "var/cache/", "var/state/"} <= set(lines)


def test_next_steps_point_at_the_configuration(tmp_path, capsys):
    from mactf.management import execute_from_command_line

    execute_from_command_line(["mactf", "startproject", "demo_swarm", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "edit conf/trainer.yaml to set epochs, batch size, and the task" in out
    assert str(Path("demo_swarm") / "conf" / "conf.yaml") in out
