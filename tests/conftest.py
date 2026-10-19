"""Shared pytest fixtures."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    """A settings object rooted in a temp directory, with YAML overlays disabled."""
    from mactf import conf

    conf.reset()

    module = type(sys)("mactf_test_settings")
    module.BASE_DIR = str(tmp_path)
    module.CONFIG_FILES = []
    module.DEVICE = "cpu"
    module.PEERS = []
    sys.modules["mactf_test_settings"] = module
    monkeypatch.setenv("MACTF_SETTINGS_MODULE", "mactf_test_settings")
    loaded = conf.load_settings("mactf_test_settings")
    yield loaded
    conf.reset()


@pytest.fixture()
def project_settings():
    """The real training project settings, used for integration tests."""
    from mactf import conf

    conf.reset()
    loaded = conf.load_settings("training.settings")
    yield loaded
    conf.reset()


@pytest.fixture()
def score_config():
    """The docx scoring configuration used across scoring tests."""
    return {
        "REWARD_WEIGHTS": {"r_task": 1.0, "r_epi": 1.5, "r_collab": 1.2, "r_align": 1.0},
        "PENALTY_LAMBDAS": {"P1": 5.0, "P6": 1.0, "P11": 0.5, "P20": 1000.0},
        "DEFAULT_LAMBDA": 1.0,
        "SANDBOX_MULTIPLIER": 1000.0,
        "OMEGA_RULES": ["P20"],
        "PRECEDENCE_MODE": "strict",
        "NON_OFFSETTABLE": ["P1", "P2", "P17", "P20", "P21", "P22"],
        "ALLOW_NEGATIVE_REWARDS": False,
    }


@pytest.fixture()
def severity_weights():
    return {"S0": 0.0, "S1": 0.25, "S2": 1.0, "S3": 4.0, "S4": 25.0}


@pytest.fixture()
def warning_policy():
    from mactf.scoring.warning import WarningPolicy

    return WarningPolicy(k_esc=3, n_win=5, n_decay=3, w_cap=2, p_warn=5.0)


@pytest.fixture()
def scorer(score_config, severity_weights, warning_policy):
    from mactf.scoring.pipeline import Scorer

    return Scorer(
        score_config=copy.deepcopy(score_config),
        severity_weights=copy.deepcopy(severity_weights),
        warning_policy=warning_policy,
    )
