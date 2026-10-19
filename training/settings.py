"""Settings for the MACTF training project.

Structure mirrors a Django project's ``settings.py``: apps, peers, and one dict
per subsystem. YAML overlays listed in ``CONFIG_FILES`` are merged on top at load
time, so deployment tweaks never require editing Python.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "dev-only-not-a-real-secret"
DEBUG = False

# apps
INSTALLED_APPS = [
    "mactf.contrib.scoring",
    "mactf.contrib.coopetition",
    "mactf.contrib.sandbox",
    "mactf.contrib.telemetry",
    "training.peers",
]

# peers: instantiated once per swarm member.
# Reward-modulated aggregation averages parameter vectors, so every peer must
# share one layout. Distinct initialisations give diversity without that problem.
PEERS = [
    {"name": "peer_alpha", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
    {"name": "peer_beta", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
    {"name": "peer_gamma", "class": "MLPPeer", "kwargs": {"state_dim": 16, "action_dim": 4, "hidden": 64}},
]


# execution
DEVICE = "auto"          # auto | cpu | cuda | cuda:0 | mps
DTYPE = "auto"           # auto | float32 | float16 | bfloat16 | float64
AMP = True               # automatic mixed precision, CUDA only
SEED = 1337
DETERMINISTIC = False    # True is slower but reproducible

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
    "CONFIGS": "training/configs",
    "DATASETS": "training/datasets",
    "LOGS": "var/logs",
    "IMAGES": "var/images",
}

SEVERITY = {
    "WEIGHTS": {"S0": 0.0, "S1": 0.25, "S2": 1.0, "S3": 4.0, "S4": 25.0},
    "MAX_TIER_ESCALATION": 3,
}

SCORING = {
    # docx Turn 3: R = a*r_task + b*r_epi + g*r_collab + d*r_align
    "REWARD_WEIGHTS": {"r_task": 1.0, "r_epi": 1.5, "r_collab": 1.2, "r_align": 1.0},
    # docx Turn 3: P = sum(lambda_k * p_k) * Omega
    "PENALTY_LAMBDAS": {
        "P1": 5.0, "P2": 5.0, "P3": 3.0, "P4": 0.5, "P5": 2.5,
        "P6": 1.0, "P7": 1.0, "P8": 0.75, "P9": 2.0, "P10": 2.0,
        "P11": 0.5, "P12": 0.5, "P13": 8.0, "P14": 8.0, "P15": 1.5,
        "P16": 3.0, "P17": 50.0, "P18": 1.5, "P19": 1.5, "P20": 1000.0,
        "P21": 50.0, "P22": 50.0, "P23": 2.0,
    },
    "DEFAULT_LAMBDA": 1.0,
    "SANDBOX_MULTIPLIER": 1000.0,
    "OMEGA_RULES": ["P20", "P21", "P22"],
    # strict = framework.md 3.2 (controlling tier only); layered = score all tiers
    "PRECEDENCE_MODE": "strict",
    "NON_OFFSETTABLE": ["P1", "P2", "P17", "P20", "P21", "P22"],
    "ALLOW_NEGATIVE_REWARDS": False,
}

WARNINGS = {
    "K_ESC": 3,       # active warnings of one rule that trigger escalation
    "N_WIN": 5,       # rolling episode window
    "N_DECAY": 3,     # consecutive clean episodes to expire one warning
    "W_CAP": 2,       # simultaneous active warnings per rule
    "P_WARN": 5.0,    # penalty applied on escalation
    "ESCALATE": True,
}

COOPETITION = {
    # docx Turn 5 adaptive temperature
    "KAPPA_INIT": 1.0,
    "KAPPA_MIN": 0.1,
    "KAPPA_MAX": 10.0,
    "TARGET_ENTROPY_RATIO": 0.8,
    "BETA_MOMENTUM": 0.9,
    "ETA_KAPPA": 0.05,
    # framework.md section 10 constraints
    "WEIGHT_FLOOR": 0.02,
    "WEIGHT_CAP": 0.60,
    "CONFIDENCE_FLOOR": 0.10,
    "TRIM_FRACTION": 0.0,
    "DISAGREEMENT_MAX": 0.75,
    "FALLBACK_TO_REFERENCE": True,
    "REFERENCE_EMA": 0.05,
    "RESET_OPTIMIZERS": True,
    "AGGREGATE_BUFFERS": "top",
    "COOPERATIVE_POOL": True,
}

SANDBOX = {
    "PROVIDER": "auto",
    "ISOLATION": "hardware",
    "BASE_IMAGE": "var/images/base.qcow2",
    "WORK_IMAGE": "var/images/peer-{peer}.qcow2",
    "MEMORY_MB": 4096,
    "VCPUS": 2,
    "ACCEL": "auto",
    "NETWORK": "none",
    "SNAPSHOT": True,
    "BOOT_TIMEOUT": 180,
    "COMMAND_TIMEOUT": 300,
    "REQUIRE_HARDWARE": False,
    "ALLOW_FALLBACK": True,
    "SERIAL_SOCKET": "var/run/mactf-{peer}.sock",
}

TRAINING = {
    "EPOCHS": 3,
    "BATCH_SIZE": 32,
    "STATE_DIM": 16,
    "ACTION_DIM": 4,
    "LEARNING_RATE": 1e-3,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPSILON": 0.2,
    "VALUE_COEF": 0.5,
    "KL_COEF": 0.01,
    "ENTROPY_COEF": 0.0,
    "MAX_GRAD_NORM": 1.0,
    "CRITIQUE_ROUNDS": 1,
    "STEPS_PER_EPOCH": 4,
    "BOOTSTRAP": True,
    "REDTEAM": False,
    "CHECKPOINT_EVERY": 1,
    "ROLLBACK_ON_ENTROPY_COLLAPSE": True,
    "ENTROPY_COLLAPSE_RATIO": 0.05,
    "TASK": "synthetic_classification",
}

TELEMETRY = {
    "LEVEL": "INFO",
    "JSON": False,
    "CONSOLE": True,
    "LOG_FILE": "var/logs/mactf.log",
}

# Optional YAML overlays, merged in order after this module.
CONFIG_FILES = [
    "training/configs/swarm.yaml",
    "training/configs/scoring.yaml",
    "training/configs/sandbox.yaml",
]
