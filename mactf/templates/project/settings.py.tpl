"""Settings for the {{ project_name }} MACTF project.

Structure mirrors a Django project's ``settings.py``: apps, peers, and one dict
per subsystem. Values not set here come from ``mactf.conf.DEFAULTS``, so this
file only records what the swarm actually decides. YAML overlays listed in
``CONFIG_FILES`` are merged on top at load time.
"""

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
    "{{ project_name }}.peers",
]

# Reward-modulated aggregation averages parameter vectors, so every peer in one
# swarm must share a layout. Distinct initialisations give diversity instead.
PEERS = [
    {"name": "peer_alpha", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
    {"name": "peer_beta", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
    {"name": "peer_gamma", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
]

DEVICE = "auto"          # auto | cpu | cuda | cuda:0 | mps
DTYPE = "auto"           # auto | float32 | float16 | bfloat16 | float64
AMP = True               # automatic mixed precision, CUDA only
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

# Optional YAML overlays, merged in order after this module.
CONFIG_FILES = [
    "configs/swarm.yaml",
    "configs/scoring.yaml",
    "configs/sandbox.yaml",
]
