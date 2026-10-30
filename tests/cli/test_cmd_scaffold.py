"""``startproject`` and ``startpeer`` (document 16, section 8.1 and 8.11)."""

from __future__ import annotations

import ast
import sys

import pytest

from mactf.testing.cli import invoke


def tree(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


# -- startproject ---------------------------------------------------------------------
def test_startproject_writes_a_project_that_loads(tmp_path):
    out = invoke(["startproject", "alpha_swarm", "--dir", str(tmp_path), "--format", "json"])
    assert out.code == 0, out.stderr
    payload = out.json()
    assert payload["project"] == "alpha_swarm" and payload["dry_run"] is False
    for expected in ("alpha_swarm/settings.py", "alpha_swarm/manage.py", "alpha_swarm/conf/trainer.yaml"):
        assert expected in payload["created"]
        assert (tmp_path / expected).is_file()
    ast.parse((tmp_path / "alpha_swarm" / "settings.py").read_text(encoding="utf-8"))
    assert payload["next_steps"][0] == "cd alpha_swarm"


def test_startproject_dry_run_touches_nothing(tmp_path):
    out = invoke(["startproject", "alpha_swarm", "--dir", str(tmp_path), "--dry-run", "--format", "json"])
    assert out.code == 0 and out.json()["dry_run"] is True and out.json()["created"]
    assert tree(tmp_path) == []
    text = invoke(["startproject", "alpha_swarm", "--dir", str(tmp_path), "--dry-run"])
    assert "would create" in text.stdout and "next steps" not in text.stdout


def test_startproject_refuses_an_existing_project_unless_forced(tmp_path):
    assert invoke(["startproject", "alpha_swarm", "--dir", str(tmp_path)]).code == 0
    settings = tmp_path / "alpha_swarm" / "settings.py"
    settings.write_text("# edited\n", encoding="utf-8")
    again = invoke(["startproject", "alpha_swarm", "--dir", str(tmp_path)])
    assert again.code == 4 and settings.read_text(encoding="utf-8") == "# edited\n"
    assert invoke(["startproject", "alpha_swarm", "--dir", str(tmp_path), "--force"]).code == 0
    assert settings.read_text(encoding="utf-8") != "# edited\n"


@pytest.mark.parametrize("name", ["2fast", "has-dash", "class", "with space", ""])
def test_startproject_rejects_names_that_cannot_be_imported(tmp_path, name):
    out = invoke(["startproject", name, "--dir", str(tmp_path)])
    assert out.code == 2 and tree(tmp_path) == []


def test_startproject_needs_no_existing_project(tmp_path):
    (tmp_path / "empty").mkdir()
    out = invoke(["startproject", "fresh"], cwd=tmp_path / "empty")
    assert out.code == 0 and (tmp_path / "empty" / "fresh" / "settings.py").exists()


def test_the_gitignore_keeps_runtime_state_out(tmp_path):
    invoke(["startproject", "alpha_swarm", "--dir", str(tmp_path)])
    lines = (tmp_path / "alpha_swarm" / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "var/" in lines


# -- startpeer ------------------------------------------------------------------------
def test_startpeer_writes_a_valid_module(tmp_path):
    out = invoke(["startpeer", "critic_peer", "--dir", str(tmp_path), "--state-dim", "32",
                  "--action-dim", "5", "--format", "json"])
    assert out.code == 0, out.stderr
    assert out.json()["class"] == "CriticPeer" and out.json()["overwrote"] is False
    source = (tmp_path / "critic_peer.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    assert any(isinstance(node, ast.ClassDef) and node.name == "CriticPeer" for node in module.body)
    assert "state_dim: int = 32" in source and "action_dim: int = 5" in source
    assert "PEERS = [CriticPeer]" in source


def test_startpeer_module_builds_and_runs_a_forward_pass(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    invoke(["startpeer", "tiny_peer", "--dir", str(tmp_path), "--state-dim", "6", "--action-dim", "3"])
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("tiny_peer", None)
    try:
        import tiny_peer

        logits, value = tiny_peer.PEERS[0]()(torch.zeros(2, 6))
        assert logits.shape == (2, 3) and value.shape == (2, 1)
    finally:
        sys.modules.pop("tiny_peer", None)


def test_startpeer_nested_names_create_directories(tmp_path):
    out = invoke(["startpeer", "peers/critic_peer", "--dir", str(tmp_path)])
    assert out.code == 0 and (tmp_path / "peers" / "critic_peer.py").is_file()


def test_startpeer_dry_run_writes_nothing(tmp_path):
    out = invoke(["startpeer", "critic_peer", "--dir", str(tmp_path), "--dry-run", "--format", "json"])
    assert out.code == 0 and out.json()["dry_run"] is True and tree(tmp_path) == []
    assert "would create" in invoke(["startpeer", "critic_peer", "--dir", str(tmp_path), "--dry-run"]).stdout


def test_startpeer_will_not_overwrite_without_force(tmp_path):
    assert invoke(["startpeer", "critic_peer", "--dir", str(tmp_path)]).code == 0
    target = tmp_path / "critic_peer.py"
    target.write_text("# mine\n", encoding="utf-8")
    assert invoke(["startpeer", "critic_peer", "--dir", str(tmp_path)]).code == 4
    assert target.read_text(encoding="utf-8") == "# mine\n"
    forced = invoke(["startpeer", "critic_peer", "--dir", str(tmp_path), "--force", "--format", "json"])
    assert forced.code == 0 and forced.json()["overwrote"] is True
    assert target.read_text(encoding="utf-8") != "# mine\n"


@pytest.mark.parametrize("name", ["2peer", "peer-one", "class", "../escape", "a/../../escape"])
def test_startpeer_rejects_bad_or_escaping_names(tmp_path, name):
    base = tmp_path / "base"
    base.mkdir()
    out = invoke(["startpeer", name, "--dir", str(base)])
    assert out.code == 2
    assert tree(base) == [] and not (tmp_path / "escape.py").exists()


def test_startpeer_rejects_non_positive_dimensions(tmp_path):
    assert invoke(["startpeer", "p", "--dir", str(tmp_path), "--state-dim", "0"]).code == 2
    assert invoke(["startpeer", "p", "--dir", str(tmp_path), "--action-dim", "-3"]).code == 2


def test_startpeer_in_a_project_defaults_to_the_project_root(project):
    out = project.invoke("startpeer", "scout_peer", "--format", "json")
    assert out.code == 0, out.stderr
    assert (project.package / "scout_peer.py").exists() or (project.root / "scout_peer.py").exists()
