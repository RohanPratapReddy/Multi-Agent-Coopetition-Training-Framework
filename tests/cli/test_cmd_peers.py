import json

import pytest

pytest.importorskip("torch")

from mactf.management.commands.peers import confidence_pair, eligible_pair, score_pair  # noqa: E402

pytestmark = pytest.mark.needs_torch


@pytest.fixture()
def swarm(project):
    """The generated project's configured peers."""
    out = project.invoke("peers", "list", "--format", "json")
    assert out.code == 0, out.stderr
    return project, [item["name"] for item in out.json()["items"]]


def test_list_reports_the_source_and_every_peer(swarm):
    project, names = swarm
    out = project.invoke("peers", "list", "--format", "json")
    assert out.json()["source"] == "settings" and out.json()["count"] == len(names) >= 1
    for item in out.json()["items"]:
        assert set(item) == {"name", "class", "parameter_count", "layout_hash", "source"}
        assert item["parameter_count"] > 0 and len(item["layout_hash"]) == 16


def test_show_and_contract_for_each_peer(swarm):
    project, names = swarm
    for name in names:
        shown = project.invoke("peers", "show", name, "--format", "json")
        contract = project.invoke("peers", "contract", name, "--format", "json")
        assert shown.code == contract.code == 0
        assert shown.json()["name"] == name
        assert {"contract", "input_names", "outputs_value", "output_kind", "notes"} <= set(contract.json())


def test_unknown_peer_is_not_found_with_a_hint(swarm):
    project, names = swarm
    out = project.invoke("peers", "show", names[0][:-1])
    assert out.code == 3 and names[0] in out.stderr


def test_weights_match_the_core_function(swarm):
    from mactf import conf
    from mactf.coopetition.introspect import preview_weights

    project, _ = swarm
    out = project.invoke("peers", "weights", "--score", "a=0.8", "--score", "b=0.6", "--score", "c=0.2",
                         "--eligible", "c=false", "--confidence", "b=0.5", "--format", "json")
    assert out.code == 0, out.stderr
    expected = preview_weights(conf.load_settings(project.module), {"a": 0.8, "b": 0.6, "c": 0.2},
                               eligible={"a": True, "b": True, "c": False},
                               confidence={"a": 1.0, "b": 0.5, "c": 1.0})
    assert out.json()["report"]["weights"] == expected["weights"]
    assert out.json()["input"]["eligible"]["c"] is False


def test_weights_need_at_least_one_score(swarm):
    project, _ = swarm
    assert project.invoke("peers", "weights").code == 2


def test_weights_reject_assignments_without_a_score_and_malformed_pairs(swarm):
    project, _ = swarm
    assert project.invoke("peers", "weights", "--score", "a=1", "--eligible", "b=true").code == 2
    assert project.invoke("peers", "weights", "--score", "a").code == 2
    assert project.invoke("peers", "weights", "--score", "a=nan").code == 2
    assert project.invoke("peers", "weights", "--score", "a=1", "--confidence", "a=2").code == 2


def test_scores_file_and_flag_precedence(swarm, tmp_path):
    project, _ = swarm
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"scores": {"a": 0.5, "b": 0.4}, "confidence": {"a": 0.9}}), encoding="utf-8")
    out = project.invoke("peers", "weights", "--scores-file", str(path), "--score", "b=0.9", "--format", "json")
    assert out.code == 0, out.stderr
    assert out.json()["input"]["scores"] == {"a": 0.5, "b": 0.9}           # a flag beats the file
    assert out.json()["input"]["confidence"] == {"a": 0.9, "b": 1.0}


def test_scores_file_from_stdin_and_yaml(swarm, tmp_path):
    project, _ = swarm
    out = project.invoke("peers", "weights", "--scores-file", "-", "--format", "json",
                         input='{"scores": {"a": 0.5, "b": 0.4}, "advance": true}')
    assert out.code == 0 and out.json()["input"]["advance"] is True
    yaml_file = tmp_path / "scores.yaml"
    yaml_file.write_text("scores:\n  a: 0.5\n  b: 0.4\n", encoding="utf-8")
    assert project.invoke("peers", "weights", "--scores-file", str(yaml_file)).code == 0


def test_scores_file_errors(swarm, tmp_path):
    project, _ = swarm
    assert project.invoke("peers", "weights", "--scores-file", str(tmp_path / "missing.json")).code == 3
    assert project.invoke("peers", "weights", "--scores-file", "-", input="{not json").code == 2
    assert project.invoke("peers", "weights", "--scores-file", "-", input='{"scores": {"a": 1}, "bogus": 1}').code == 2
    assert project.invoke("peers", "weights", "--scores-file", "-", input='{"scores": {"a": "x"}}').code == 2


def test_infeasible_floor_is_a_usage_error(swarm):
    project, _ = swarm
    out = project.invoke("peers", "weights", "--score", "a=1", "--score", "b=1", "--score", "c=1",
                         "--score", "d=1", "--score", "e=1", "--set", "COOPETITION.WEIGHT_FLOOR=0.5",
                         "--set", "COOPETITION.WEIGHT_CAP=0.9")
    assert out.code == 2


def test_argument_types():
    assert score_pair("a=0.5") == ("a", 0.5)
    assert confidence_pair("a.b-c=1") == ("a.b-c", 1.0)
    assert eligible_pair("a=Yes") == ("a", True) and eligible_pair("a=off") == ("a", False)
    import argparse

    for bad in ("=1", "a", "a=inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            score_pair(bad)
