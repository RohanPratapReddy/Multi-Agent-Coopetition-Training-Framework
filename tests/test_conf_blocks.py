"""The blocks document 15 adds to ``DEFAULTS``, their validators and the template drift test."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conf_project import edit_conf, leaves, load, scaffold
from mactf import conf
from mactf.conf import (
    CHECKPOINT_DEFAULTS,
    CORS_DEFAULTS,
    DEFAULTS,
    LOGGER_DEFAULTS,
    LOSS_DEFAULTS,
    OPTIMIZER_DEFAULTS,
    ORCHESTRATOR_DEFAULTS,
    TRAINER_DEFAULTS,
    WATCHDOG_DEFAULTS,
)
from mactf.conf_files import CONF_LAYOUT, PYTHON_ONLY_KEYS
from mactf.exceptions import ImproperlyConfigured
from mactf.training.logging import DEFAULT_REDACT

TEMPLATES = Path(conf.__file__).resolve().parent / "templates" / "conf"

NEW_BLOCKS = {
    "TRAINER": TRAINER_DEFAULTS,
    "OPTIMIZER": OPTIMIZER_DEFAULTS,
    "LOSS": LOSS_DEFAULTS,
    "LOGGER": LOGGER_DEFAULTS,
    "WATCHDOG": WATCHDOG_DEFAULTS,
    "CHECKPOINT": CHECKPOINT_DEFAULTS,
    "ORCHESTRATOR": ORCHESTRATOR_DEFAULTS,
    "CORS": CORS_DEFAULTS,
}


# -- the new blocks exist, with the documented defaults ---------------------------
@pytest.mark.parametrize("name", sorted(NEW_BLOCKS))
def test_each_new_block_is_in_defaults_with_its_documented_values(name):
    assert DEFAULTS[name] == NEW_BLOCKS[name]
    # a copy, so mutating DEFAULTS never edits the module-level source of truth
    assert DEFAULTS[name] is not NEW_BLOCKS[name]


def test_paths_gain_the_five_new_directories():
    paths = DEFAULTS["PATHS"]
    assert {"RUNS", "TEACHERS", "AUDITS", "CACHE", "STATE"} <= set(paths)
    assert paths["RUNS"] == "var/runs" and paths["STATE"] == "var/state"


def test_documented_default_values():
    assert TRAINER_DEFAULTS["SHARING"] == "independent"
    assert OPTIMIZER_DEFAULTS["CLASS"] == "AdamW" and OPTIMIZER_DEFAULTS["BETAS"] == [0.9, 0.999]
    assert LOSS_DEFAULTS["TYPE"] == "composite"
    assert LOGGER_DEFAULTS["SINKS"] == ["console", "file"]
    assert WATCHDOG_DEFAULTS["ON_CRITICAL"] == "rollback"
    assert CHECKPOINT_DEFAULTS["DIR"] is None and CHECKPOINT_DEFAULTS["CHECKSUM"] == "sha256"
    assert ORCHESTRATOR_DEFAULTS["MAX_EPOCHS"] is None
    assert CORS_DEFAULTS["ORIGINS"] == []


# -- drift: DEFAULTS and the shipped YAML are two copies of the same numbers ------
#: Template values that are a project decision rather than a framework default.
#: Each is a deliberate, argued difference; anything else that differs is drift.
PROJECT_CHOICES = {
    # the generated project switches mixed precision on, as its old settings.py did
    "AMP",
    # conf/ is the directory this layout creates; the default key keeps its old value
    "PATHS.CONFIGS",
    # a generated project ships a three-peer swarm; the framework default is none
    "PEERS",
    # the default is [] meaning "DEFAULT_REDACT"; the file spells the list out, and
    # test_the_shipped_redact_list_is_the_logging_default pins the two together
    "LOGGER.REDACT",
}

#: ``trainer.yaml`` lists the keys that stayed in TRAINING; the rest moved to
#: TRAINER and LOSS, so TRAINING is compared one way (template leaves only).
ONE_WAY_BLOCKS = {"TRAINING"}


def _template(entry):
    text = (TEMPLATES / entry.template_name).read_text(encoding="utf-8")
    return yaml.safe_load(text.replace("{{ project_name }}", "demo_swarm")) or {}


@pytest.mark.parametrize("entry", CONF_LAYOUT, ids=lambda entry: entry.filename)
def test_every_template_leaf_equals_the_defaults_leaf(entry):
    differing: list[str] = []
    for key, value in _template(entry).items():
        template_leaves = leaves({key: value})
        default_leaves = leaves({key: DEFAULTS[key]})
        paths = set(template_leaves) if key in ONE_WAY_BLOCKS else set(template_leaves) | set(default_leaves)
        for path in sorted(paths):
            if path in PROJECT_CHOICES or key in PROJECT_CHOICES:
                continue
            ours, theirs = template_leaves.get(path, "<absent>"), default_leaves.get(path, "<absent>")
            if ours != theirs or type(ours) is not type(theirs):
                differing.append(f"{entry.filename}: {path}: template={ours!r} DEFAULTS={theirs!r}")
    assert differing == []


def test_every_owned_block_that_has_a_default_is_written_by_its_file():
    """A default with no template line would be invisible to the person editing conf/."""
    missing = []
    for entry in CONF_LAYOUT:
        written = set(_template(entry))
        for key in entry.owns:
            if key in PYTHON_ONLY_KEYS or key in written or key not in DEFAULTS:
                continue
            missing.append(f"{entry.filename} does not mention {key}")
    assert missing == []


def test_templates_that_differ_from_defaults_are_exactly_the_documented_choices():
    amp = {entry.filename: _template(entry).get("AMP") for entry in CONF_LAYOUT}
    assert amp["conf.yaml"] is True and DEFAULTS["AMP"] is False
    assert _template(CONF_LAYOUT[0])["PATHS"]["CONFIGS"] == "conf"


def test_the_shipped_redact_list_is_the_logging_default():
    logger = next(entry for entry in CONF_LAYOUT if entry.filename == "logger.yaml")
    assert tuple(_template(logger)["LOGGER"]["REDACT"]) == DEFAULT_REDACT


def test_no_template_scalar_is_an_unquoted_yaml_one_float():
    """PyYAML reads ``1.0e3`` (no exponent sign) as a string, which a float() accepts
    but a type check does not. The drift test above compares types for that reason."""
    for path in sorted(TEMPLATES.glob("*.tpl")):
        for value in leaves(yaml.safe_load(path.read_text(encoding="utf-8").replace("{{ project_name }}", "x")) or {}).values():
            if isinstance(value, str):
                try:
                    float(value)
                except ValueError:
                    continue
                pytest.fail(f"{path.name} has a numeric-looking string {value!r}; quote it or fix the exponent")


# -- _fold_trainer ----------------------------------------------------------------
@pytest.fixture()
def project(tmp_path):
    return scaffold(tmp_path)


def test_trainer_epochs_fold_into_training(tmp_path, monkeypatch, project):
    edit_conf(project, "trainer.yaml", lambda p: p["TRAINER"].update(EPOCHS=8))
    edit_conf(project, "trainer.yaml", lambda p: p["TRAINING"].pop("EPOCHS", None))
    loaded = load(tmp_path, monkeypatch)
    assert loaded["TRAINING"]["EPOCHS"] == 8
    assert loaded["TRAINER"]["EPOCHS"] == 8


def test_a_trainer_key_with_no_training_counterpart_stays_in_trainer_only(tmp_path, monkeypatch, project):
    edit_conf(project, "trainer.yaml", lambda p: p["TRAINER"].update(HIERARCHY=True))
    loaded = load(tmp_path, monkeypatch)
    assert loaded["TRAINER"]["HIERARCHY"] is True
    assert "HIERARCHY" not in loaded["TRAINING"]


def test_without_trainer_the_training_block_is_untouched(tmp_path, monkeypatch):
    module = tmp_path / "legacy_proj"
    module.mkdir()
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "settings.py").write_text(
        "from pathlib import Path\nBASE_DIR = Path(__file__).resolve().parent\n"
        "CONFIG_FILES = []\nDEVICE = 'cpu'\nTRAINING = {'EPOCHS': 6, 'LEARNING_RATE': 0.01}\n",
        encoding="utf-8",
    )
    loaded = load(tmp_path, monkeypatch, name="legacy_proj")
    assert loaded["TRAINING"]["EPOCHS"] == 6
    assert loaded["TRAINING"]["LEARNING_RATE"] == 0.01
    assert loaded["TRAINING"]["BATCH_SIZE"] == DEFAULTS["TRAINING"]["BATCH_SIZE"]


# -- validators: one test per refusal ---------------------------------------------
def _set(block, **values):
    def mutate(payload):
        payload.setdefault(block, {}).update(values)

    return mutate


REFUSALS = [
    ("trainer.yaml", _set("TRAINER", EPOCHS=0), "TRAINER.EPOCHS must be >= 1"),
    ("trainer.yaml", _set("TRAINER", BATCH_SIZE=0), "TRAINER.BATCH_SIZE must be >= 1"),
    ("trainer.yaml", _set("TRAINER", STEPS_PER_EPOCH=0), "TRAINER.STEPS_PER_EPOCH must be >= 1"),
    ("trainer.yaml", _set("TRAINER", GRAD_ACCUMULATION=0), "TRAINER.GRAD_ACCUMULATION must be >= 1"),
    ("trainer.yaml", _set("TRAINER", PEER_STRATEGY="nonsense"), "TRAINER.PEER_STRATEGY must be one of"),
    ("trainer.yaml", _set("TRAINER", WEIGHT_UPDATE="nonsense"), "TRAINER.WEIGHT_UPDATE must be one of"),
    ("trainer.yaml", _set("TRAINER", SHARING="nonsense"), "TRAINER.SHARING must be one of"),
    ("optimizer.yaml", _set("OPTIMIZER", TYPE="nonsense"), "OPTIMIZER.TYPE must be"),
    ("optimizer.yaml", _set("OPTIMIZER", LR=-1.0), "OPTIMIZER.LR must be >= 0"),
    ("optimizer.yaml", _set("OPTIMIZER", WEIGHT_DECAY=-1.0), "OPTIMIZER.WEIGHT_DECAY must be >= 0"),
    ("optimizer.yaml", _set("OPTIMIZER", GRAD_ACCUMULATION=0), "OPTIMIZER.GRAD_ACCUMULATION must be >= 1"),
    ("loss.yaml", _set("LOSS", KL_COEF=-0.1), "LOSS.KL_COEF must be >= 0"),
    ("loss.yaml", _set("LOSS", ENTROPY_COEF=-0.1), "LOSS.ENTROPY_COEF must be >= 0"),
    ("loss.yaml", _set("LOSS", CLIP_EPSILON=0.0), "LOSS.CLIP_EPSILON must be in (0, 1]"),
    ("logger.yaml", _set("LOGGER", LEVEL="LOUD"), "LOGGER.LEVEL must be a standard level"),
    ("logger.yaml", _set("LOGGER", SINKS=[]), "LOGGER.SINKS must be a non-empty list"),
    ("logger.yaml", _set("LOGGER", SINKS=["smoke"]), "LOGGER.SINKS names unknown sink"),
    ("logger.yaml", _set("LOGGER", SINKS=["file"], LOG_FILE=None), "LOGGER.SINKS includes 'file' but LOG_FILE is unset"),
    ("logger.yaml", _set("LOGGER", SINKS=["jsonl"], JSONL_FILE=None), "LOGGER.SINKS includes 'jsonl' but JSONL_FILE is unset"),
    ("logger.yaml", _set("LOGGER", SINKS=["tensorboard"]), "LOGGER.SINKS includes 'tensorboard' but TENSORBOARD_DIR is unset"),
    ("watchdog.yaml", _set("WATCHDOG", MAX_GRAD_NORMN=100.0), "WATCHDOG names unknown key"),
    ("watchdog.yaml", _set("WATCHDOG", ON_CRITICAL="explode"), "WATCHDOG.ON_CRITICAL must be one of"),
    ("watchdog.yaml", _set("WATCHDOG", ON_DEGRADATION="explode"), "WATCHDOG.ON_DEGRADATION must be one of"),
    ("checkpoint.yaml", _set("CHECKPOINT", KEEP=-1), "CHECKPOINT.KEEP and KEEP_BEST must be >= 0"),
    ("checkpoint.yaml", _set("CHECKPOINT", CHECKSUM="md5"), "CHECKPOINT.CHECKSUM must be 'sha256' or 'blake2b'"),
    ("orchestrator.yaml", _set("ORCHESTRATOR", MAX_EPOCHS=0), "ORCHESTRATOR.MAX_EPOCHS must be None or >= 1"),
]


@pytest.mark.parametrize(
    ("filename", "mutate", "message"),
    REFUSALS,
    ids=[f"{message[:60]}" for _, _, message in REFUSALS],
)
def test_validator_refuses_and_names_the_block_and_key(tmp_path, monkeypatch, project, filename, mutate, message):
    edit_conf(project, filename, mutate)
    with pytest.raises(ImproperlyConfigured, match=message.replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
        load(tmp_path, monkeypatch)


def test_a_block_that_is_not_a_mapping_is_refused(tmp_path, monkeypatch, project):
    (project / "conf" / "loss.yaml").write_text("LOSS: 3\n", encoding="utf-8")
    with pytest.raises(ImproperlyConfigured, match="LOSS must be a mapping"):
        load(tmp_path, monkeypatch)


def test_api_cors_origins_and_cors_origins_together_are_refused(tmp_path, monkeypatch, project):
    def both(payload):
        payload["API"]["CORS_ORIGINS"] = ["https://a.example"]
        payload["CORS"]["ORIGINS"] = ["https://b.example"]

    edit_conf(project, "api.yaml", both)
    with pytest.raises(ImproperlyConfigured, match=r"API\.CORS_ORIGINS and CORS\.ORIGINS both set"):
        load(tmp_path, monkeypatch)


def test_disagreeing_learning_rates_are_refused(tmp_path, monkeypatch, project):
    edit_conf(project, "optimizer.yaml", _set("OPTIMIZER", LR=0.5))
    with pytest.raises(ImproperlyConfigured, match=r"TRAINING\.LEARNING_RATE=0\.001 and OPTIMIZER\.LR=0\.5 disagree"):
        load(tmp_path, monkeypatch)


def test_agreeing_learning_rates_are_accepted(tmp_path, monkeypatch, project):
    edit_conf(project, "optimizer.yaml", _set("OPTIMIZER", LR=0.002))
    edit_conf(project, "trainer.yaml", _set("TRAINING", LEARNING_RATE=0.002))
    loaded = load(tmp_path, monkeypatch)
    assert loaded["OPTIMIZER"]["LR"] == loaded["TRAINING"]["LEARNING_RATE"] == 0.002


def test_a_freshly_generated_conf_tree_passes_validation(tmp_path, monkeypatch, project):
    loaded = load(tmp_path, monkeypatch)
    assert loaded["TRAINING"]["EPOCHS"] == 3
    assert len(loaded["PEERS"]) == 3
    # the loader strips nothing the validators needed, and records authorship
    assert set(loaded["_AUTHORED"]) >= {"TRAINER", "OPTIMIZER", "LOSS", "LOGGER", "WATCHDOG", "CHECKPOINT"}


def test_a_default_only_project_is_not_refused_by_the_new_validators():
    """DEFAULTS alone (a legacy project) must satisfy every new validator."""
    from mactf.conf import Settings, _validate_blocks

    import copy

    _validate_blocks(Settings(copy.deepcopy(DEFAULTS)))
