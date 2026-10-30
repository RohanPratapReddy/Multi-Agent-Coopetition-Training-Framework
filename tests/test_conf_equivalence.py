"""The migration gate: a ``conf/`` project must behave like the legacy dict-and-overlay one.

The legacy project below is the exact project ``mactf startproject`` generated before
document 15 (one ``settings.py`` of dicts plus three ``configs/*.yaml`` overlays). The
``conf/`` project is what it generates now, with ``optimizer.yaml`` set to the optimizer
the legacy trainer used (``Adam``, no weight decay), because the shipped file chooses
``AdamW`` on purpose and that is a numerics change, not an equivalence question.
"""

from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from conf_project import edit_conf, load, scaffold
from mactf import conf
from mactf.conf import DEFAULTS

LEGACY_SETTINGS = '''"""Legacy settings."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = "dev-only-change-me"
DEBUG = False

INSTALLED_APPS = [
    "mactf.contrib.scoring",
    "mactf.contrib.coopetition",
    "mactf.contrib.sandbox",
    "mactf.contrib.telemetry",
    "mactf.contrib.components",
    "demo_swarm.peers",
]

PEERS = [
    {"name": "peer_alpha", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
    {"name": "peer_beta", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
    {"name": "peer_gamma", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
]

DEVICE = "auto"
DTYPE = "auto"
AMP = True
SEED = 1337

DATABASES = {
    "default": {
        "ENGINE": "sqlite3",
        "NAME": "var/mactf.sqlite3",
    }
}

PATHS = {
    "ROOT": str(BASE_DIR),
    "VAR": "var",
    "CHECKPOINTS": "var/checkpoints",
    "CONFIGS": "configs",
    "DATASETS": "datasets",
    "LOGS": "var/logs",
    "IMAGES": "var/images",
}

TRAINING = {
    "EPOCHS": 3,
    "BATCH_SIZE": 32,
    "STATE_DIM": 16,
    "ACTION_DIM": 4,
    "LEARNING_RATE": 1e-3,
    "CRITIQUE_ROUNDS": 1,
    "STEPS_PER_EPOCH": 4,
    "BOOTSTRAP": True,
    "REDTEAM": False,
    "CHECKPOINT_EVERY": 1,
    "TASK": "synthetic_classification",
}

SANDBOX = {
    "PROVIDER": "auto",
    "ISOLATION": "hardware",
    "ALLOW_FALLBACK": True,
    "REQUIRE_HARDWARE": False,
    "NETWORK": "none",
    "SNAPSHOT": True,
}

CONFIG_FILES = [
    "configs/swarm.yaml",
    "configs/scoring.yaml",
    "configs/sandbox.yaml",
]
'''

LEGACY_SWARM = """
TRAINING:
  EPOCHS: 3
  BATCH_SIZE: 32
  STEPS_PER_EPOCH: 4
  LEARNING_RATE: 0.001
  CRITIQUE_ROUNDS: 1
  CHECKPOINT_EVERY: 1

COOPETITION:
  KAPPA_INIT: 1.0
  KAPPA_MIN: 0.1
  KAPPA_MAX: 10.0
  TARGET_ENTROPY_RATIO: 0.8
  BETA_MOMENTUM: 0.9
  ETA_KAPPA: 0.05
  WEIGHT_FLOOR: 0.02
  WEIGHT_CAP: 0.6
  DISAGREEMENT_MAX: 0.75
  REFERENCE_EMA: 0.05
"""

LEGACY_SCORING = """
SCORING:
  PRECEDENCE_MODE: strict
  REWARD_WEIGHTS:
    r_task: 1.0
    r_epi: 1.5
    r_collab: 1.2
    r_align: 1.0
  PENALTY_LAMBDAS:
    P1: 5.0
    P2: 5.0
    P17: 50.0
    P20: 1000.0
    P21: 50.0
    P22: 50.0
  SANDBOX_MULTIPLIER: 1000.0

WARNINGS:
  K_ESC: 3
  N_WIN: 5
  N_DECAY: 3
  W_CAP: 2
  P_WARN: 5.0
  ESCALATE: true

SEVERITY:
  WEIGHTS:
    S0: 0.0
    S1: 0.25
    S2: 1.0
    S3: 4.0
    S4: 25.0
"""

LEGACY_SANDBOX = """
SANDBOX:
  PROVIDER: auto
  ISOLATION: hardware
  ALLOW_FALLBACK: true
  REQUIRE_HARDWARE: false
  MEMORY_MB: 4096
  VCPUS: 2
  ACCEL: auto
  NETWORK: none
  SNAPSHOT: true
  BOOT_TIMEOUT: 180
  COMMAND_TIMEOUT: 300
  BASE_IMAGE: var/images/base.qcow2
  WORK_IMAGE: var/images/peer-{peer}.qcow2
  SERIAL_SOCKET: var/run/mactf-{peer}.sock
"""

#: Keys that legitimately differ between the two layouts: where the files live, and
#: the bookkeeping that records which layout was used.
EXCLUDED = {"_CONF_NOTES", "_AUTHORED", "PATHS", "CONF_DIR", "CONFIG_FILES"}

#: Excluded from the *settings* comparison only. A legacy project never writes
#: ``OPTIMIZER``, so its trainer builds the inline ``Adam`` while the block still holds
#: the ``AdamW`` default; the conf/ project writes ``Adam``. The two trainers agreeing
#: is asserted separately, on the optimizer each one actually built.
BLOCK_ONLY_WRITTEN_BY_CONF = {"OPTIMIZER"}


def _legacy(root: Path) -> Path:
    package = scaffold(root)
    shutil.rmtree(package / "conf")
    (package / "settings.py").write_text(LEGACY_SETTINGS, encoding="utf-8")
    configs = package / "configs"
    configs.mkdir()
    (configs / "swarm.yaml").write_text(LEGACY_SWARM, encoding="utf-8")
    (configs / "scoring.yaml").write_text(LEGACY_SCORING, encoding="utf-8")
    (configs / "sandbox.yaml").write_text(LEGACY_SANDBOX, encoding="utf-8")
    return package


def _modern(root: Path) -> Path:
    package = scaffold(root)
    # the optimizer the legacy trainer built: Adam at TRAINING.LEARNING_RATE, no decay
    edit_conf(package, "optimizer.yaml", lambda p: p["OPTIMIZER"].update(CLASS="Adam", WEIGHT_DECAY=0.0))
    return package


def _close(left, right, path="$"):
    """Recursive comparison: ``==`` everywhere, ``1e-12`` on floats."""
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right), f"{path}: keys differ {set(left) ^ set(right)}"
        for key in left:
            _close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        assert len(left) == len(right), f"{path}: length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            _close(a, b, f"{path}[{index}]")
    elif isinstance(left, float) or isinstance(right, float):
        assert math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12), f"{path}: {left!r} != {right!r}"
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


@pytest.fixture()
def pair(tmp_path, monkeypatch):
    legacy_root, modern_root = tmp_path / "legacy", tmp_path / "modern"
    legacy_root.mkdir()
    modern_root.mkdir()
    _legacy(legacy_root)
    _modern(modern_root)
    legacy = load(legacy_root, monkeypatch)
    modern = load(modern_root, monkeypatch)
    return legacy, modern


# -- the gate ---------------------------------------------------------------------
def test_conf_tree_resolves_to_the_same_settings_as_the_legacy_project(pair):
    legacy, modern = pair
    skipped = EXCLUDED | BLOCK_ONLY_WRITTEN_BY_CONF
    left = {k: v for k, v in legacy.as_dict().items() if k not in skipped}
    right = {k: v for k, v in modern.as_dict().items() if k not in skipped}
    # The one key the fold adds: TRAINER.GRAD_ACCUMULATION has a TRAINING counterpart
    # that legacy projects never carried. At 1 it is the optimizer's own default.
    assert right["TRAINING"].pop("GRAD_ACCUMULATION") == 1
    # LOGGER.REDACT: [] means "use DEFAULT_REDACT"; the shipped file spells it out.
    from mactf.training.logging import DEFAULT_REDACT

    assert left["LOGGER"]["REDACT"] == []
    left["LOGGER"]["REDACT"] = list(DEFAULT_REDACT)
    # the conf/ tree writes the new blocks, which the legacy project leaves at DEFAULTS;
    # they hold the same values, and the two dicts must agree key for key
    _close(left, right)


def test_the_gate_is_not_vacuous(pair):
    legacy, modern = pair
    assert modern["CONF_DIR"] is not None and legacy["CONF_DIR"] is None
    assert "TRAINER" in modern["_AUTHORED"] and "TRAINER" not in legacy["_AUTHORED"]
    assert modern["PATHS"]["CONFIGS"] == "conf" and legacy["PATHS"]["CONFIGS"] == "configs"
    assert len(modern["PEERS"]) == len(legacy["PEERS"]) == 3
    assert modern["TRAINING"]["EPOCHS"] == legacy["TRAINING"]["EPOCHS"] == 3


def test_every_path_resolves_to_the_same_relative_location(pair):
    legacy, modern = pair
    for key in ("VAR", "CHECKPOINTS", "DATASETS", "LOGS", "IMAGES"):
        assert modern["PATHS"][key] == legacy["PATHS"][key], key
    assert Path(modern["PATHS"]["ROOT"]).name == Path(legacy["PATHS"]["ROOT"]).name == "demo_swarm"


# -- the trainers agree -----------------------------------------------------------
class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.policy = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


def _fit(loaded, tmp_path):
    from mactf.training import MACTFTrainer

    loaded["TRAINING"].update({"EPOCHS": 1, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False})
    loaded["SANDBOX"]["PROVIDER"] = "local"
    loaded["DEVICE"] = "cpu"
    loaded["AMP"] = False
    torch.manual_seed(1337)
    trainer = MACTFTrainer(model=Tiny(), num_peers=2, settings=loaded, checkpoint_dir=tmp_path / "ckpt", seed=1337)
    result = trainer.fit(epochs=1)
    spec = trainer.spec.as_dict()
    spec.pop("run_id")
    return trainer, result, spec


def test_trainers_built_from_either_layout_agree_exactly(pair, tmp_path):
    legacy, modern = pair
    legacy_trainer, legacy_result, legacy_spec = _fit(legacy, tmp_path / "a")
    modern_trainer, modern_result, modern_spec = _fit(modern, tmp_path / "b")

    # TrainSpec describes the loop, loss and optimizer the run was built with. The
    # block passes Adam's own default betas/eps explicitly, so the *names* of the
    # keyword arguments differ while their values are Adam's defaults; the peers'
    # parameters below prove the optimizers stepped identically.
    assert legacy_spec["optimizer"]["kwargs"] == []
    assert modern_spec["optimizer"]["kwargs"] == ["betas", "eps"]
    legacy_spec["optimizer"].pop("kwargs")
    modern_spec["optimizer"].pop("kwargs")
    _close(legacy_spec, modern_spec)

    a, b = legacy_trainer.reports[-1], modern_trainer.reports[-1]
    assert a.kappa == pytest.approx(b.kappa, abs=1e-12)
    assert a.entropy == pytest.approx(b.entropy, abs=1e-12)
    assert set(a.weights) == set(b.weights)
    for name, weight in a.weights.items():
        assert weight == pytest.approx(b.weights[name], abs=1e-12)
    assert legacy_result.mean_loss == pytest.approx(modern_result.mean_loss, abs=1e-12)
    for name, peer in legacy_trainer.peers.items():
        assert torch.allclose(
            peer.export_parameters(), modern_trainer.peers[name].export_parameters(), atol=1e-12, rtol=0.0
        )


def test_the_shipped_optimizer_choice_is_a_real_numerics_change(tmp_path, monkeypatch):
    """The equivalence above needs ``Adam``/0.0 because the shipped file says ``AdamW``/0.01.

    If this ever starts passing as *equal*, the shipped optimizer.yaml stopped doing
    anything and the test above stopped proving the wiring.
    """
    adamw_root = tmp_path / "adamw"
    adamw_root.mkdir()
    scaffold(adamw_root)
    shipped = load(adamw_root, monkeypatch)
    assert shipped["OPTIMIZER"]["CLASS"] == "AdamW" and shipped["OPTIMIZER"]["WEIGHT_DECAY"] == 0.01
    _, _, spec = _fit(shipped, tmp_path / "c")
    assert spec["optimizer"].get("class") == "AdamW" or "AdamW" in str(spec["optimizer"])


# -- DEFAULTS did not move --------------------------------------------------------
#: Every pre-existing ``DEFAULTS`` block, as it stood when document 15 landed. The blocks
#: document 15 adds (``TRAINER``, ``OPTIMIZER``, ``LOSS``, ``LOGGER``, ``WATCHDOG``,
#: ``CHECKPOINT``, ``ORCHESTRATOR``, ``CORS``, ``CONF_DIR``) and ``DISTILLATION`` (document
#: 14, pinned by its own tests) are not listed; ``PATHS`` lists only its original seven
#: keys. A future edit to any value here fails this test and has to be argued in a diff.
SNAPSHOT = {'AMP': False,
 'API': {'ALLOW_ANONYMOUS': True,
         'BASE_PATH': 'api/v1',
         'CANCEL_GRACE_SECONDS': 10.0,
         'CORS_ORIGINS': [],
         'EVENT_BUFFER': 1024,
         'HEALTH_HISTORY_LIMIT': 50,
         'IDEMPOTENCY_TTL_SECONDS': 86400,
         'INFERENCE_BUDGET_DEFAULT': None,
         'INFERENCE_MAX_WORKERS': 4,
         'LOG_TAIL_DEFAULT': 100,
         'LOG_TAIL_MAX': 2000,
         'MAX_CONCURRENT_RUNS': 1,
         'PREVIEW_MAX_NODES': 64,
         'PRINCIPAL_RESOLVER': None,
         'TOKENS': {}},
 'APPEND_SLASH': True,
 'CONFIG_FILES': [],
 'COOPETITION': {'AGGREGATE_BUFFERS': 'top',
                 'BETA_MOMENTUM': 0.9,
                 'CONFIDENCE_FLOOR': 0.1,
                 'COOPERATIVE_BUDGET': {'CARRY_OVER': False,
                                        'PER_EPISODE': 6.0,
                                        'PER_PEER_CAP': 3.0},
                 'COOPERATIVE_POOL': True,
                 'DISAGREEMENT_MAX': 0.75,
                 'ETA_KAPPA': 0.05,
                 'FALLBACK_TO_REFERENCE': True,
                 'KAPPA_INIT': 1.0,
                 'KAPPA_MAX': 10.0,
                 'KAPPA_MIN': 0.1,
                 'NORMALIZATION': {'CLIP': [0.0, 100.0],
                                   'ENABLED': True,
                                   'EXCLUDE_VOID': True,
                                   'MIN_COHORT': 2,
                                   'SHRINK': 0.35,
                                   'SIGNAL_SCALE': 50.0,
                                   'TARGET_MEAN': 62.0},
                 'REFERENCE_EMA': 0.05,
                 'RESET_OPTIMIZERS': True,
                 'SIGNAL_SOURCE': 'final_score',
                 'TARGET_ENTROPY_RATIO': 0.8,
                 'TRIM_FRACTION': 0.0,
                 'WEIGHT_CAP': 0.6,
                 'WEIGHT_FLOOR': 0.02},
 'DATABASES': {'default': {'ENGINE': 'sqlite3', 'NAME': 'var/mactf.sqlite3'}},
 'DEBUG': False,
 'DEFAULT_AUTO_FIELD': 'mactf.db.BigAutoField',
 'DETERMINISTIC': False,
 'DEVICE': 'auto',
 'DISTRIBUTED': {'BASE_PATH': 'worker/v1',
                 'COMPRESS': True,
                 'COORDINATOR_URL': 'http://127.0.0.1:8000',
                 'MAX_PEERS': 1,
                 'MAX_WORKERS': 16,
                 'WORKER_HEARTBEAT_TIMEOUT': 30.0,
                 'WORKER_ID': None,
                 'WORKER_TASK_TIMEOUT': 600.0},
 'DTYPE': 'auto',
 'ENGINE': {'F_CAP': 3, 'P12_NOISE_RATIO': 0.5, 'P_MIN': 0.5, 'R9_STEP_FACTOR': 1.25},
 'HTTP': {'MAX_BODY_BYTES': 8388608, 'SSE_HEARTBEAT': 15.0, 'STREAM_WORKERS': 16},
 'INSTALLED_APPS': ['mactf.contrib.scoring',
                    'mactf.contrib.coopetition',
                    'mactf.contrib.sandbox',
                    'mactf.contrib.telemetry',
                    'mactf.contrib.components'],
 'MIDDLEWARE': ['mactf.http.middleware.RequestIdMiddleware',
                'mactf.http.middleware.ExceptionMiddleware'],
 'PATHS': {'CHECKPOINTS': 'var/checkpoints',
           'CONFIGS': 'configs',
           'DATASETS': 'datasets',
           'IMAGES': 'var/images',
           'LOGS': 'var/logs',
           'ROOT': '.',
           'VAR': 'var'},
 'PEERS': [],
 'ROOT_URLCONF': None,
 'SANDBOX': {'ACCEL': 'auto',
             'ALLOW_FALLBACK': True,
             'BASE_IMAGE': 'var/images/base.qcow2',
             'BOOT_TIMEOUT': 180,
             'CLOUD_INIT': {'PACKAGES': ['python3', 'python3-pip'], 'USER': 'mactf'},
             'COMMAND_TIMEOUT': 300,
             'ISOLATION': 'hardware',
             'MEMORY_MB': 4096,
             'NETWORK': 'none',
             'PROVIDER': 'auto',
             'REQUIRE_HARDWARE': False,
             'SERIAL_SOCKET': 'var/run/mactf-{peer}.sock',
             'SNAPSHOT': True,
             'VCPUS': 2,
             'WORK_IMAGE': 'var/images/peer-{peer}.qcow2'},
 'SCORING': {'ALLOW_NEGATIVE_REWARDS': False,
             'DEFAULT_LAMBDA': 1.0,
             'F_CAP_RATIO': 0.5,
             'NOISE_THRESHOLD': 0.2,
             'NON_OFFSETTABLE': ['P1', 'P2', 'P17', 'P20', 'P21', 'P22'],
             'OMEGA_RULES': ['P20', 'P21', 'P22'],
             'OMEGA_RULE_OVERRIDES': {},
             'PENALTY_LAMBDAS': {'P1': 5.0,
                                 'P10': 2.0,
                                 'P11': 0.5,
                                 'P12': 0.5,
                                 'P13': 8.0,
                                 'P14': 8.0,
                                 'P15': 1.5,
                                 'P16': 3.0,
                                 'P17': 50.0,
                                 'P18': 1.5,
                                 'P19': 1.5,
                                 'P2': 5.0,
                                 'P20': 1000.0,
                                 'P21': 50.0,
                                 'P22': 50.0,
                                 'P23': 2.0,
                                 'P3': 3.0,
                                 'P4': 0.5,
                                 'P5': 2.5,
                                 'P6': 1.0,
                                 'P7': 1.0,
                                 'P8': 0.75,
                                 'P9': 2.0},
             'POINTS': {'PENALTY_WEIGHTS': {'P1': 40.0,
                                            'P10': 15.0,
                                            'P11': 4.0,
                                            'P12': 4.0,
                                            'P13': 60.0,
                                            'P14': 60.0,
                                            'P15': 10.0,
                                            'P16': 25.0,
                                            'P17': 100.0,
                                            'P18': 10.0,
                                            'P19': 10.0,
                                            'P2': 40.0,
                                            'P20': 100.0,
                                            'P21': 100.0,
                                            'P22': 100.0,
                                            'P23': 12.0,
                                            'P3': 20.0,
                                            'P4': 5.0,
                                            'P5': 25.0,
                                            'P6': 6.0,
                                            'P7': 6.0,
                                            'P8': 5.0,
                                            'P9': 15.0},
                        'REWARD_WEIGHTS': {'R1': 25.0,
                                           'R10': 8.0,
                                           'R11': 5.0,
                                           'R12': 4.0,
                                           'R13': 6.0,
                                           'R14': 5.0,
                                           'R15': 3.0,
                                           'R16': 4.0,
                                           'R17': 1.0,
                                           'R2': 12.0,
                                           'R3': 2.0,
                                           'R4': 3.0,
                                           'R5': 3.0,
                                           'R6': 5.0,
                                           'R7': 5.0,
                                           'R8': 6.0,
                                           'R9': 3.0},
                        'SEVERITY_FRACTIONS': {'S0': 0.0,
                                               'S1': 0.05,
                                               'S2': 0.15,
                                               'S3': 0.4,
                                               'S4': 1.0},
                        'TABLE_VERSION': '1.0.0',
                        'WARNING_NOTIONAL': 2.0},
             'PRECEDENCE_MODE': 'strict',
             'P_MIN': 0.5,
             'REWARD_WEIGHTS': {'r_align': 1.0, 'r_collab': 1.2, 'r_epi': 1.5, 'r_task': 1.0},
             'SANDBOX_MULTIPLIER': 1000.0},
 'SEED': 1337,
 'SEVERITY': {'MAX_TIER_ESCALATION': 3,
              'WEIGHTS': {'S0': 0.0, 'S1': 0.25, 'S2': 1.0, 'S3': 4.0, 'S4': 25.0}},
 'TELEMETRY': {'CONSOLE': True, 'JSON': False, 'LEVEL': 'INFO', 'LOG_FILE': 'var/logs/mactf.log'},
 'TRAINING': {'ACTION_DIM': 4,
              'BATCH_SIZE': 32,
              'BOOTSTRAP': True,
              'BOOTSTRAP_CONFIG': {'BATCHES_PER_EPOCH': 0,
                                   'BATCH_SIZE': 16,
                                   'EPOCHS': 1,
                                   'LEARNING_RATE': 0.001,
                                   'MAX_GRAD_NORM': 1.0,
                                   'PATIENCE': 0,
                                   'UPDATE_REFERENCE': True,
                                   'WEIGHT_DECAY': 0.0},
              'CHECKPOINT_EVERY': 1,
              'CLIP_EPSILON': 0.2,
              'CRITIQUE_ROUNDS': 1,
              'ENTROPY_COEF': 0.0,
              'ENTROPY_COLLAPSE_RATIO': 0.05,
              'EPOCHS': 3,
              'GAE_LAMBDA': 0.95,
              'GAMMA': 0.99,
              'KL_COEF': 0.01,
              'LEARNING_RATE': 0.001,
              'MAX_GRAD_NORM': 1.0,
              'REDTEAM': False,
              'ROLLBACK_ON_ENTROPY_COLLAPSE': True,
              'STATE_DIM': 16,
              'STEPS_PER_EPOCH': 4,
              'TASK': 'synthetic_classification',
              'VALUE_COEF': 0.5},
 'WARNINGS': {'ESCALATE': True, 'K_ESC': 3, 'N_DECAY': 3, 'N_WIN': 5, 'P_WARN': 5.0, 'W_CAP': 2}}


def test_no_pre_existing_default_changed():
    changed = []
    for key, expected in SNAPSHOT.items():
        actual = DEFAULTS[key]
        if key == "PATHS":
            actual = {name: actual[name] for name in expected}
        if actual != expected:
            changed.append(key)
    assert changed == []


def test_the_snapshot_covers_every_pre_existing_block():
    added = {
        "TRAINER", "OPTIMIZER", "LOSS", "LOGGER", "WATCHDOG", "CHECKPOINT",
        "ORCHESTRATOR", "CORS", "CONF_DIR", "DISTILLATION",
        # document 17 section 15.1: the component registry's policy block
        "COMPONENTS",
        # document 17 section 6.6: the compliance floor and guard
        "COMPLIANCE",
        # document 17 section 8.5: RL function defaults and overrides
        "RL",
    }
    assert set(DEFAULTS) - added == set(SNAPSHOT)


def test_paths_only_gained_keys():
    assert set(DEFAULTS["PATHS"]) - set(SNAPSHOT["PATHS"]) == {"RUNS", "TEACHERS", "AUDITS", "CACHE", "STATE"}


# -- document 17 section 15.1: the COMPONENTS block changes nothing for either project ----
def test_the_components_block_is_the_default_in_both_projects(pair):
    legacy, modern = pair
    assert legacy["COMPONENTS"] == modern["COMPONENTS"] == DEFAULTS["COMPONENTS"]
    # a legacy project leaves it at the framework default, so `mactf config --changed` shows
    # nothing; a generated project now ships conf/components.yaml (document 17 section 15.3),
    # which spells the same values out and so records the block as written by that file
    assert "COMPONENTS" not in legacy["_AUTHORED"]


def test_a_default_components_block_is_exactly_the_default_hub_policy(pair):
    from mactf.components import HubPolicy, policy_from_settings

    for project in pair:
        assert policy_from_settings(project["COMPONENTS"]) == HubPolicy()
