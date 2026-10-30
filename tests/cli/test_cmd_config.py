import json

import pytest
import yaml

from mactf.management.commands.config import MASK, lookup, origin_for, redact, secret_fragments
from mactf.exceptions import ResourceNotFound
from mactf.testing.cli import invoke

SECRET = "hunter2-token-value"


def _with_token(project):
    return ("--set", f"API.TOKENS.alice={SECRET}")


def test_show_masks_secrets_in_every_format(project):
    for fmt in ("text", "json", "yaml"):
        out = project.invoke("config", "show", "API", *_with_token(project), "--format", fmt)
        assert out.code == 0, out.stderr
        assert SECRET not in out.stdout, fmt
        assert MASK in out.stdout, fmt


def test_reveal_unmasks_and_warns_on_stderr(project):
    out = project.invoke("config", "show", "API.TOKENS", *_with_token(project), "--reveal", "--format", "json")
    assert out.code == 0
    assert out.json()["value"] == {"alice": SECRET}
    assert "clear text" in out.stderr


def test_show_secrets_is_not_an_option(project):
    assert project.invoke("config", "show", "--show-secrets").code == 2


def test_get_prints_one_bare_value(project):
    out = project.invoke("config", "get", "TRAINING.EPOCHS")
    assert out.code == 0
    assert out.stdout == "3\n"


def test_get_is_case_insensitive_when_unambiguous(project):
    assert project.invoke("config", "get", "training.epochs").stdout == "3\n"


def test_get_unknown_key_is_not_found_with_a_hint(project):
    out = project.invoke("config", "get", "TRAINING.EPOCH")
    assert out.code == 3
    assert "TRAINING.EPOCHS" in out.stderr


def test_get_private_or_malformed_path_is_usage(project):
    assert project.invoke("config", "get", "_ORIGINS").code == 2
    assert project.invoke("config", "get", "TRAINING..EPOCHS").code == 2


def test_get_origin_on_a_table_is_usage(project):
    assert project.invoke("config", "get", "TRAINING", "--origin").code == 2


def test_origin_names_the_layer_of_a_set_override(project):
    out = project.invoke("config", "get", "TRAINING.EPOCHS", "--origin", "--set", "TRAINING.EPOCHS=2")
    assert out.code == 0
    assert out.stdout.strip().split("\t") == ["2", "cli:--set"]


def test_origin_names_a_config_file_layer(project, tmp_path):
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("TRAINING:\n  EPOCHS: 7\n", encoding="utf-8")
    out = project.invoke("config", "show", "TRAINING.EPOCHS", "--origin", "--config", str(overlay),
                         "--format", "json")
    assert out.code == 0, out.stderr
    assert out.json()["value"] == 7
    assert out.json()["origins"]["TRAINING.EPOCHS"].startswith("cli:")


def test_changed_lists_only_non_default_values(project):
    out = project.invoke("config", "show", "--changed", "--origin", "--format", "json")
    assert out.code == 0, out.stderr
    assert out.json()["origins"]
    assert "default" not in set(out.json()["origins"].values())


def test_files_lists_layers_lowest_first(project):
    out = project.invoke("config", "files", "--format", "json")
    assert out.code == 0, out.stderr
    labels = [layer["label"] for layer in out.json()["layers"]]
    assert labels[0] == "default"
    assert labels[1] == "settings:demo_swarm.settings"


def test_validate_ok_on_a_generated_project(project):
    out = project.invoke("config", "validate", "--format", "json")
    assert out.code == 0, out.stderr
    assert out.json()["ok"] is True and out.json()["errors"] == []


def test_validate_reports_a_module_that_does_not_import(tmp_path):
    out = invoke(["config", "validate", "--settings", "no_such_settings_module", "--format", "json"],
                 env={"MACTF_NO_DISCOVER": "1"})
    assert out.code == 2
    assert out.json()["ok"] is False
    assert "no_such_settings_module" in out.json()["errors"][0]


def test_set_type_rules_are_enforced(project):
    assert project.invoke("config", "get", "TRAINING.EPOCHS", "--set", "TRAINING.EPOCHS=bad").code == 2


def test_machine_formats_keep_stdout_clean(project):
    out = project.invoke("config", "show", "TRAINING", "--reveal", "--format", "yaml")
    assert yaml.safe_load(out.stdout)["key"] == "TRAINING"
    assert out.stderr.startswith("warning:")


# -- pure functions ------------------------------------------------------------------------

def test_redact_masks_by_name_fragment_and_hides_keys_under_a_secret_table():
    tree = {"API": {"TOKENS": {"alice": "a", "bob": "b"}, "PASSWORD": "p", "OTHER": "x"},
            "LIST": [{"api_key": "k", "n": 1}]}
    out = redact(tree, ("token", "password", "api_key"))
    assert out["API"]["TOKENS"] == {"<redacted 1>": MASK, "<redacted 2>": MASK}
    assert out["API"]["PASSWORD"] == MASK and out["API"]["OTHER"] == "x"
    assert out["LIST"] == [{"api_key": MASK, "n": 1}]
    assert tree["API"]["PASSWORD"] == "p"                          # the input is untouched


def test_redact_keeps_empty_and_non_string_secrets_readable():
    assert redact({"PASSWORD": ""}, ("password",)) == {"PASSWORD": ""}
    assert redact({"TOKEN_COUNT": 3}, ("token",)) == {"TOKEN_COUNT": 3}


def test_lookup_walks_case_insensitively_and_names_close_keys():
    path, value = lookup({"TRAINING": {"EPOCHS": 3}}, "training.epochs")
    assert path == ("TRAINING", "EPOCHS") and value == 3
    with pytest.raises(ResourceNotFound, match="TRAINING.EPOCHS"):
        lookup({"TRAINING": {"EPOCHS": 3}}, "TRAINING.EPOCH")
    with pytest.raises(ResourceNotFound, match="not a table"):
        lookup({"TRAINING": {"EPOCHS": 3}}, "TRAINING.EPOCHS.X")


def test_origin_for_falls_back_to_the_nearest_ancestor():
    origins = {"API.TOKENS.alice": "cli:--set", "A.B": "default"}
    assert origin_for(origins, ("API", "TOKENS")) == "cli:--set"
    assert origin_for(origins, ("A", "B", "C")) == "default"
    assert origin_for(origins, ("Z",)) == "unknown"


def test_secret_fragments_union_the_default_and_configured_lists():
    fragments = secret_fragments({"LOGGER": {"REDACT": ["Custom_Field"]}})
    assert "custom_field" in fragments and "token" in fragments
    assert len(fragments) == len(set(fragments))
