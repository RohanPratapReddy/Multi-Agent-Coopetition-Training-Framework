"""Settings loading, merging, and typed access.

Resolution order, later winning:

1. Framework defaults (:data:`DEFAULTS`).
2. A settings module named by ``MACTF_SETTINGS_MODULE``.
2.5. The project's ``conf/`` tree (document 15), one file per subsystem, merged in
   :data:`mactf.conf_files.CONF_LAYOUT` order.
3. ``settings.CONFIG_FILES`` YAML overlays, in order.
4. ``MACTF_<SETTING>`` environment variables for scalar overrides.
"""

from __future__ import annotations

import copy
import difflib
import importlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .components.known import (
    BUILTIN_NAMES,
    LOCKABLE_KINDS,
    component_ref_problem,
    is_dotted_name,
    lock_pattern_problem,
)
from .distill.config import DISTILLATION_DEFAULTS, validate_block
from .exceptions import ImproperlyConfigured

log = logging.getLogger("mactf.conf")

ENV_SETTINGS_MODULE = "MACTF_SETTINGS_MODULE"
#: ``MACTF_*`` variables that configure the CLI or the loader, never a settings key.
_ENV_NON_SETTINGS = frozenset({ENV_SETTINGS_MODULE, "MACTF_NO_DISCOVER", "MACTF_DISCOVERED_ROOT"})
# The distribution cannot assume a project exists, so the default is the
# framework's own minimal settings module. Projects set MACTF_SETTINGS_MODULE.
DEFAULT_SETTINGS_MODULE = "mactf.settings_default"

# Document 15 section 5: the blocks the ``conf/`` tree adds. Each one is a default
# that reproduces the behaviour of a project that never writes it; whether a block
# was *written* is recorded separately under ``_AUTHORED`` so consumers can tell
# "the framework default" from "the project's choice".
TRAINER_DEFAULTS: dict[str, Any] = {
    # The loop's shape.
    "EPOCHS": 3,
    "BATCH_SIZE": 32,
    "STEPS_PER_EPOCH": 4,
    "GRAD_ACCUMULATION": 1,
    "MAX_GRAD_NORM": 1.0,
    # The tree's shape.
    "HIERARCHY": False,
    "PEER_STRATEGY": "kappa_softmax",
    "WEIGHT_UPDATE": "kappa_softmax",
    # Document 17 sections 10 and 11: per-group strategies, their guards, and the freeze mode.
    "WEIGHT_UPDATE_OVERRIDES": {},
    "WEIGHT_GUARD": {"MODE": "repair", "SCOPE": "custom", "TOLERANCE": 1e-6, "PROBE": True},
    "WEIGHT_OPTIONS": {"STRICT_GLOBS": True, "MIGRATE": "reset"},
    "AGGREGATOR": None,
    "PEER_STRATEGY_OVERRIDES": {},
    "PEER_STRATEGY_OPTIONS": {"STRICT_GLOBS": True, "MIGRATE": "reset"},
    "STRATEGY_GUARD": {"MODE": "skip", "SCOPE": "custom", "PROBE": True, "CHECK_FROZEN": True},
    "FREEZE_MODE": "legacy",
    # Document 17 section 12: tree mutation is off until ``ENABLED`` is true. Every key is listed so
    # that ``--set TRAINER.MUTATION.<KEY>=...`` can address it (a key outside the defaults is a typo).
    "MUTATION": {
        "ENABLED": False,
        "ALLOW_PURGE": True,
        "TOMBSTONE": "run",
        "WATCHDOG_PRUNE": "purge",
        "STRICT_REPLAY": True,
        "JOURNAL_WARN": 1000,
        "MAX_OPS_PER_PLAN": 64,
        "MAX_PLANS_PER_EPOCH": 4,
        "MIN_ACTIVE_ROOTS": None,       # None: min(2, roots at build)
        "PROTECTED": [],
        "SCHEDULE": [],
    },
    "SHARING": "independent",
    # The task.
    "TASK": "synthetic_classification",
    # Checkpoint cadence, which is a trainer decision, not a writer decision.
    "CHECKPOINT_EVERY": 1,
    # Observability.
    "LOG_EVERY_STEPS": 1,
    "WATCHDOG_EVERY_EPOCHS": 1,
}

OPTIMIZER_DEFAULTS: dict[str, Any] = {
    # "torch", "scipy", or "callable". Matches MACTFOptimizer.from_spec's dispatch.
    "TYPE": "torch",
    # The torch.optim class name, resolved by getattr(torch.optim, name).
    "CLASS": "AdamW",
    "LR": 1e-3,
    "WEIGHT_DECAY": 0.01,
    "BETAS": [0.9, 0.999],
    "EPS": 1e-8,
    "GRAD_ACCUMULATION": 1,
    "MAX_GRAD_NORM": 1.0,
    # A scheduler is named, not constructed from YAML: the parameters differ per
    # class and a mapping cannot be validated without importing torch.
    "SCHEDULER": None,
    "SCHEDULER_KWARGS": {},
    # Document 17 section 9.4: per-node blocks and the plan's own switches. ``TYPE`` also
    # accepts any registered optimizer name; the plan checks the name, not this module.
    "OVERRIDES": {},
    "MIGRATE": "reset",
    "VERIFY": "custom",
    "STRICT_GLOBS": True,
}

LOSS_DEFAULTS: dict[str, Any] = {
    # "composite" uses CompositeLoss over these coefficients. Any other value is
    # a mactf.rl.functions registry name.
    "TYPE": "composite",
    "VALUE_COEF": 0.5,
    "KL_COEF": 0.01,
    "ENTROPY_COEF": 0.0,
    "CLIP_EPSILON": 0.2,
    "COMPLIANCE_COEF": 1.0,
    "ADVANTAGE_NORMALIZATION": True,
    # registry-name extras, passed through to build_function
    "ARGS": {},
}

LOGGER_DEFAULTS: dict[str, Any] = {
    "LEVEL": "INFO",
    "RUNS_DIR": "var/runs",
    "SINKS": ["console", "file"],
    "JSON": False,
    "LOG_FILE": "var/logs/mactf.log",
    "JSONL_FILE": None,
    "ROTATE_BYTES": 10485760,
    "ROTATE_BACKUPS": 5,
    "BUFFER_SIZE": 1000,
    "ASYNC_WRITE": True,
    "QUEUE_SIZE": 8192,
    # Empty means DEFAULT_REDACT (mactf/training/logging.py). Naming fragments
    # replaces it, which is why the shipped file lists the defaults explicitly.
    "REDACT": [],
    "SCHEMA_VERSION": 1,
    "TENSORBOARD_DIR": None,
    "WANDB_PROJECT": None,
}

WATCHDOG_DEFAULTS: dict[str, Any] = {
    # WatchdogPolicy, in declaration order.
    "NAN_POLICY": "abort",
    "INF_POLICY": "abort",
    "LOSS_INCREASE_RATIO": 3.0,
    "LOSS_PATIENCE": 2,
    "MAX_GRAD_NORM": 100.0,
    "GRAD_NORM_PATIENCE": 1,
    "SCORE_REGRESSION_TOLERANCE": 0.10,
    "SCORE_PATIENCE": 3,
    "ENTROPY_COLLAPSE_RATIO": 0.05,
    "ENTROPY_PATIENCE": 2,
    "KAPPA_MIN": 1e-3,
    "KAPPA_MAX": 1e3,
    "STALE_EPOCH_SECONDS": 1800.0,
    "MIN_THROUGHPUT_RATIO": 0.25,
    "MAX_STEP_TIME_DRIFT": 5.0,
    "MAX_RSS_GROWTH_RATIO": 2.0,
    "MAX_GPU_MEMORY_RATIO": 0.95,
    "MAX_CHECKPOINT_AGE_SECONDS": 3600.0,
    "WARNING_SPIKE_THRESHOLD": 10,
    "MAX_TREE_GROWTH_RATIO": 2.0,
    "DEAD_BRANCH_PATIENCE": 2,
    "CHILD_STARVATION_PATIENCE": 3,
    "DEPTH_COST_RATIO": 4.0,
    # MACTFWatchdog.__init__, beyond the policy.
    "SNAPSHOT_EVERY_EPOCHS": 1,
    "SNAPSHOT_EVERY_SECONDS": None,
    "SNAPSHOT_ON_DEGRADATION": True,
    "ON_DEGRADATION": "snapshot",
    "ON_CRITICAL": "rollback",
    "KEEP_HISTORY": 512,
}

CHECKPOINT_DEFAULTS: dict[str, Any] = {
    "DIR": None,  # None resolves to PATHS["CHECKPOINTS"]
    "KEEP": 5,
    "KEEP_BEST": 1,
    "COMPRESS": True,
    # "sha256" or "blake2b"; MACTFCheckpoint accepts exactly these two.
    "CHECKSUM": "sha256",
    "ASYNC_WRITE": False,
    "INCLUDE_OPTIMIZER": True,
    "INCLUDE_RNG": True,
    "INCLUDE_DATA_POSITION": True,
    "INCLUDE_BUFFERS": True,
    "INCLUDE_ADAPTERS": True,
}

ORCHESTRATOR_DEFAULTS: dict[str, Any] = {
    # Which task source the run uses. None defers to TRAINER.TASK.
    "TASK_SOURCE": None,
    # Stop after this many epochs. None defers to TRAINER.EPOCHS.
    "MAX_EPOCHS": None,
    # Where a resumed run looks when --resume names no file.
    "RESUME_FROM": None,
    # Print the startup notes (path resolution, conf/ notes).
    "ANNOUNCE": True,
    # Validated but not read: see document 15 section 7.3.
    "PAUSE_IS_FATAL": False,
    # Document 17 section 13: which class runs the epochs ("orchestrator" or "trainer").
    "ENGINE": "orchestrator",
    # Document 17 section 13.8: reproduce the orchestrator's cumulative signal window.
    "ENGINE_COMPAT": True,
    # Allow ``--resume`` to continue a checkpoint written by the other engine (E091).
    "ALLOW_ENGINE_SWITCH": False,
}

CORS_DEFAULTS: dict[str, Any] = {
    "ORIGINS": [],
    "ALLOW_CREDENTIALS": False,
    "ALLOW_HEADERS": ["authorization", "content-type", "x-request-id"],
    "MAX_AGE": 600,
}

# Document 17 section 15.1: the component registry's own policy. Every value is the one
# that makes a project that writes nothing behave as it always has: no override, no lock,
# legacy writes tolerated (with a warning), no import from an untrusted surface, ``auto``
# discovery (which finds the framework's own entry-point group, empty), nothing imported,
# registries left open, and a restore difference reported as a warning.
COMPONENTS_DEFAULTS: dict[str, Any] = {
    # -- who may replace a built-in ------------------------------------------------
    "ALLOW_BUILTIN_OVERRIDE": False,
    # -- what may not change at all (``kind:name`` or ``slot:<path>``, fnmatch) -----
    "LOCKED": [],
    # -- whether an old-style direct dict write is an error or just noise ----------
    "STRICT_LEGACY": False,
    # -- importing code named by a spec -------------------------------------------
    "ALLOW_IMPORT": False,
    "IMPORT_ALLOWLIST": [],
    # -- how many components of one kind may exist --------------------------------
    "MAX_PER_KIND": 512,
    # -- discovery ----------------------------------------------------------------
    "PLUGINS": "auto",  # "auto" | "off" | [entry-point names]
    "MODULES": [],  # dotted modules imported once, in order
    "FREEZE_AFTER_LOAD": False,
    # -- restore ---------------------------------------------------------------
    "STRICT_RESTORE": False,
}

# Document 17 section 6.6: the floor every customization is held to, and what happens when a
# scoring stage breaks it. The defaults are inert: 1.0 equals ``SCORING.DEFAULT_LAMBDA`` and
# every built-in lambda of a non-offsettable rule is >= 5.
COMPLIANCE_DEFAULTS: dict[str, Any] = {
    "MIN_FLOOR_LAMBDA": 1.0,  # lambda floor for every non-offsettable rule; > 0
    "GUARD_MODE": "enforce",  # "enforce" | "fallback"  (6.4.2)
    "REWARD_CEILING": 10000.0,  # matches compliance.constrain's value_cap
    "TOLERANCE": 1e-9,  # floating-point slack in guard comparisons; [0, 1e-3]
    "TRUST_BUILTIN": True,  # skip inspection of the three built-in stage classes
}

# Document 17 section 8.5.2: the three switches of the ``RL`` block. A family name
# (``ADVANTAGE``, ``LOSS``...) or ``OVERRIDES`` is written only when a project selects one.
RL_DEFAULTS: dict[str, Any] = {
    "UNCONSUMED": "error",  # "error" | "warn": a configured family nobody reads
    "STRICT_GLOBS": True,  # an override glob that matches no node at build is an error
    "VERIFY": "custom",  # "custom" | "all" | "off": the contract probe
}

#: Spellings kept for compatibility / readability. The value is the canonical key.
_COMPONENTS_ALIASES: dict[str, str] = {
    "API_ALLOW_IMPORT": "ALLOW_IMPORT",
}

#: ``TRAINER`` keys that have a ``TRAINING`` counterpart. Folded at load.
_TRAINER_TO_TRAINING: dict[str, str] = {
    "EPOCHS": "EPOCHS",
    "BATCH_SIZE": "BATCH_SIZE",
    "STEPS_PER_EPOCH": "STEPS_PER_EPOCH",
    "TASK": "TASK",
    "CHECKPOINT_EVERY": "CHECKPOINT_EVERY",
    "MAX_GRAD_NORM": "MAX_GRAD_NORM",
    "GRAD_ACCUMULATION": "GRAD_ACCUMULATION",
}

DEFAULTS: dict[str, Any] = {
    "DEBUG": False,
    # Document 15 section 4.2: where the ``conf/`` tree lives. Only settings.py may
    # set it; ``None`` falls back to ``<root>/conf`` when that holds a ``*.yaml``.
    "CONF_DIR": None,
    "INSTALLED_APPS": [
        "mactf.contrib.scoring",
        "mactf.contrib.coopetition",
        "mactf.contrib.sandbox",
        "mactf.contrib.telemetry",
        "mactf.contrib.components",
    ],
    "PEERS": [],
    # Document 12 section 14.1: the HTTP transport layer.
    "ROOT_URLCONF": None,
    "MIDDLEWARE": [
        "mactf.http.middleware.RequestIdMiddleware",
        "mactf.http.middleware.ExceptionMiddleware",
    ],
    "APPEND_SLASH": True,
    "HTTP": {
        "MAX_BODY_BYTES": 8388608,
        "STREAM_WORKERS": 16,
        "SSE_HEARTBEAT": 15.0,
    },
    "API": {
        "ALLOW_ANONYMOUS": True,
        "TOKENS": {},
        "BASE_PATH": "api/v1",
        "EVENT_BUFFER": 1024,
        "MAX_CONCURRENT_RUNS": 1,
        "CANCEL_GRACE_SECONDS": 10.0,
        "CORS_ORIGINS": [],
        "PRINCIPAL_RESOLVER": None,
        "IDEMPOTENCY_TTL_SECONDS": 86400,
        "LOG_TAIL_DEFAULT": 100,
        "LOG_TAIL_MAX": 2000,
        "HEALTH_HISTORY_LIMIT": 50,
        "PREVIEW_MAX_NODES": 64,
        "INFERENCE_MAX_WORKERS": 4,
        "INFERENCE_BUDGET_DEFAULT": None,
    },
    # Document 15 section 5.10: the richer CORS form; ``API.CORS_ORIGINS`` stays.
    "CORS": copy.deepcopy(CORS_DEFAULTS),
    # Document 17 section 15.1: the component registry's own policy.
    "COMPONENTS": copy.deepcopy(COMPONENTS_DEFAULTS),
    # Document 17 section 6.6: the compliance floor every custom scorer is held to.
    "COMPLIANCE": copy.deepcopy(COMPLIANCE_DEFAULTS),
    # Document 17 section 8.5: RL function defaults and per-node overrides. Empty changes nothing.
    "RL": copy.deepcopy(RL_DEFAULTS),
    # Document 05 section 8: read only when ``mactf.distributed`` is installed.
    "DISTRIBUTED": {
        "COORDINATOR_URL": "http://127.0.0.1:8000",
        "BASE_PATH": "worker/v1",
        "WORKER_ID": None,
        "MAX_PEERS": 1,
        "COMPRESS": True,
        "WORKER_HEARTBEAT_TIMEOUT": 30.0,
        "WORKER_TASK_TIMEOUT": 600.0,
        "MAX_WORKERS": 16,
    },
    "DEFAULT_AUTO_FIELD": "mactf.db.BigAutoField",
    "DEVICE": "auto",
    "DTYPE": "auto",
    "AMP": False,
    "SEED": 1337,
    "DETERMINISTIC": False,
    "DATABASES": {
        "default": {
            "ENGINE": "sqlite3",
            "NAME": "var/mactf.sqlite3",
        }
    },
    "PATHS": {
        "ROOT": ".",
        "VAR": "var",
        "CHECKPOINTS": "var/checkpoints",
        "CONFIGS": "configs",
        "DATASETS": "datasets",
        "LOGS": "var/logs",
        "IMAGES": "var/images",
        # Document 15 section 5.1: every directory a run writes to.
        "RUNS": "var/runs",
        "TEACHERS": "var/teachers",
        "AUDITS": "var/audits",
        "CACHE": "var/cache",
        "STATE": "var/state",
    },
    "SEVERITY": {
        # Magnitude per severity tier. Ordering and caps are fixed by
        # framework.md section 4; the numbers themselves are configuration.
        "WEIGHTS": {"S0": 0.0, "S1": 0.25, "S2": 1.0, "S3": 4.0, "S4": 25.0},
        "MAX_TIER_ESCALATION": 3,
    },
    "SCORING": {
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
        "OMEGA_RULE_OVERRIDES": {},
        # "strict" applies framework.md 3.2 (only the controlling tier scores).
        # "layered" scores every tier but still refuses to net across tiers.
        "PRECEDENCE_MODE": "strict",
        "NON_OFFSETTABLE": ["P1", "P2", "P17", "P20", "P21", "P22"],
        "ALLOW_NEGATIVE_REWARDS": False,
        # Evaluator thresholds (doc 04.3).
        "F_CAP_RATIO": 0.5,
        "P_MIN": 0.5,
        "NOISE_THRESHOLD": 0.2,
        # Doc 11: the 0-100 points economy. Reward weights sum to 100.
        "POINTS": {
            "REWARD_WEIGHTS": {
                "R1": 25.0, "R2": 12.0, "R3": 2.0, "R4": 3.0, "R5": 3.0, "R6": 5.0,
                "R7": 5.0, "R8": 6.0, "R9": 3.0, "R10": 8.0, "R11": 5.0, "R12": 4.0,
                "R13": 6.0, "R14": 5.0, "R15": 3.0, "R16": 4.0, "R17": 1.0,
            },
            "PENALTY_WEIGHTS": {
                "P1": 40.0, "P2": 40.0, "P3": 20.0, "P4": 5.0, "P5": 25.0, "P6": 6.0,
                "P7": 6.0, "P8": 5.0, "P9": 15.0, "P10": 15.0, "P11": 4.0, "P12": 4.0,
                "P13": 60.0, "P14": 60.0, "P15": 10.0, "P16": 25.0, "P17": 100.0,
                "P18": 10.0, "P19": 10.0, "P20": 100.0, "P21": 100.0, "P22": 100.0,
                "P23": 12.0,
            },
            "SEVERITY_FRACTIONS": {"S0": 0.0, "S1": 0.05, "S2": 0.15, "S3": 0.4, "S4": 1.0},
            "WARNING_NOTIONAL": 2.0,
            "TABLE_VERSION": "1.0.0",
        },
    },
    "ENGINE": {
        "F_CAP": 3,
        "P_MIN": 0.5,
        "P12_NOISE_RATIO": 0.5,
        "R9_STEP_FACTOR": 1.25,
    },
    "WARNINGS": {
        "K_ESC": 3,
        "N_WIN": 5,
        "N_DECAY": 3,
        "W_CAP": 2,
        "P_WARN": 5.0,
        "ESCALATE": True,
    },
    "COOPETITION": {
        "KAPPA_INIT": 1.0,
        "KAPPA_MIN": 0.1,
        "KAPPA_MAX": 10.0,
        "TARGET_ENTROPY_RATIO": 0.8,
        "BETA_MOMENTUM": 0.9,
        "ETA_KAPPA": 0.05,
        "WEIGHT_FLOOR": 0.02,
        "WEIGHT_CAP": 0.60,
        "CONFIDENCE_FLOOR": 0.10,
        "TRIM_FRACTION": 0.0,
        "DISAGREEMENT_MAX": 0.75,
        "FALLBACK_TO_REFERENCE": True,
        "REFERENCE_EMA": 0.05,
        "RESET_OPTIMIZERS": True,
        # Doc 11: which scorecard field drives aggregation.
        "SIGNAL_SOURCE": "final_score",
        "NORMALIZATION": {
            "ENABLED": True,
            "TARGET_MEAN": 62.0,
            "SHRINK": 0.35,
            "EXCLUDE_VOID": True,
            "MIN_COHORT": 2,
            "SIGNAL_SCALE": 50.0,
            "CLIP": [0.0, 100.0],
        },
        "AGGREGATE_BUFFERS": "top",
        # framework.md 9.2: report the cooperative component separately, from a
        # bounded per-episode pool so R10/R11/R16 cannot be farmed.
        "COOPERATIVE_POOL": True,
        "COOPERATIVE_BUDGET": {
            "PER_EPISODE": 6.0,
            "PER_PEER_CAP": 3.0,
            "CARRY_OVER": False,
        },
    },
    # Document 15 section 5.10: the run-level decisions that belong to neither the
    # trainer nor the aggregator.
    "ORCHESTRATOR": copy.deepcopy(ORCHESTRATOR_DEFAULTS),
    "SANDBOX": {
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
        "CLOUD_INIT": {"USER": "mactf", "PACKAGES": ["python3", "python3-pip"]},
    },
    # Document 15 sections 5.8 and 5.9.
    "WATCHDOG": copy.deepcopy(WATCHDOG_DEFAULTS),
    "CHECKPOINT": copy.deepcopy(CHECKPOINT_DEFAULTS),
    "TRAINING": {
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
        "BOOTSTRAP": True,
        "REDTEAM": False,
        "CHECKPOINT_EVERY": 1,
        "ROLLBACK_ON_ENTROPY_COLLAPSE": True,
        "ENTROPY_COLLAPSE_RATIO": 0.05,
        "TASK": "synthetic_classification",
        "STEPS_PER_EPOCH": 4,
        "BOOTSTRAP_CONFIG": {
            "EPOCHS": 1,
            "BATCH_SIZE": 16,
            "LEARNING_RATE": 1e-3,
            "WEIGHT_DECAY": 0.0,
            "MAX_GRAD_NORM": 1.0,
            "UPDATE_REFERENCE": True,
            "PATIENCE": 0,
            "BATCHES_PER_EPOCH": 0,
        },
    },
    # Document 15 sections 5.2, 5.4, 5.5. ``TRAINING`` stays the resolved runtime
    # block; ``TRAINER`` is the authored one, folded into it at load.
    "TRAINER": copy.deepcopy(TRAINER_DEFAULTS),
    "OPTIMIZER": copy.deepcopy(OPTIMIZER_DEFAULTS),
    "LOSS": copy.deepcopy(LOSS_DEFAULTS),
    # Document 14. Installed from the one source of truth so the settings path and the
    # ``distillation=`` argument path can never disagree about keys or defaults.
    "DISTILLATION": copy.deepcopy(DISTILLATION_DEFAULTS),
    "TELEMETRY": {
        "LEVEL": "INFO",
        "JSON": False,
        "CONSOLE": True,
        "LOG_FILE": "var/logs/mactf.log",
    },
    # Document 15 section 5.6: wins over TELEMETRY for the logger when written.
    "LOGGER": copy.deepcopy(LOGGER_DEFAULTS),
    "CONFIG_FILES": [],
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(value: str) -> Any:
    """Guess a type for an environment value whose key has no typed default.

    ``"1"`` and ``"0"`` are numbers here, not booleans (document 16, P11); a bool-typed
    key is handled by :func:`_coerce_env`, which accepts them as booleans.
    """
    lowered = value.strip().lower()
    if lowered in _TRUE - {"1"}:
        return True
    if lowered in _FALSE - {"0"}:
        return False
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    if value[:1] in "[{(":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # ``{"team-*": independent}`` is YAML flow style, not JSON (document 17 15.9.2)
            try:
                import yaml

                loaded = yaml.safe_load(value)
            except Exception:  # noqa: BLE001 - not YAML either: keep the text
                return value
            return loaded if isinstance(loaded, (dict, list)) else value
    return value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImproperlyConfigured(
            "PyYAML is required to load CONFIG_FILES; install pyyaml or drop the entry"
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ImproperlyConfigured(f"{path} must contain a YAML mapping at the top level")
    return data


class Settings:
    """A resolved settings object with attribute and mapping access."""

    def __init__(self, values: dict[str, Any], *, module: str | None = None) -> None:
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(f"settings have no attribute {name!r}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self._values[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._values[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def as_dict(self, *, include_trace: bool = False) -> dict[str, Any]:
        """A deep copy of the values; provenance (``_ORIGINS``, ``_LAYERS``) only on request."""
        copied = copy.deepcopy(self._values)
        if not include_trace:
            copied.pop("_ORIGINS", None)
            copied.pop("_LAYERS", None)
        return copied

    @property
    def module(self) -> str | None:
        return self._module

    def path(self, key: str) -> Path:
        """Resolve a ``PATHS`` entry against ``PATHS['ROOT']``."""
        paths = self._values.get("PATHS", {})
        root = Path(paths.get("ROOT", ".")).expanduser().resolve()
        if key == "ROOT":
            return root
        if key not in paths:
            raise ImproperlyConfigured(f"unknown PATHS entry {key!r}")
        candidate = Path(paths[key]).expanduser()
        if candidate.is_absolute():
            return candidate
        return (root / candidate).resolve()

    def ensure_path(self, key: str) -> Path:
        directory = self.path(key)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def database_path(self) -> Path:
        database = self._values["DATABASES"]["default"]
        engine = database.get("ENGINE", "sqlite3")
        if engine != "sqlite3":
            raise ImproperlyConfigured(f"unsupported database engine {engine!r}")
        name = Path(database["NAME"]).expanduser()
        if not name.is_absolute():
            name = self.path("ROOT") / name
        return name.resolve()

    def describe(self) -> str:
        return f"<Settings module={self._module!r} keys={len(self._values)}>"


def _coerce_env(name: str, raw: str, current: Any) -> Any:
    """Coerce by the type the key already has, so ``MACTF_SEED=1`` is ``1``, not ``True``."""
    text = raw.strip()
    try:
        if isinstance(current, bool):                # before int: bool is an int subclass
            lowered = text.lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            raise ValueError("expected true/false, 1/0, yes/no, or on/off")
        if isinstance(current, int):
            return int(text)
        if isinstance(current, float):
            return float(text)
        if isinstance(current, str):
            return raw
    except ValueError as exc:
        raise ImproperlyConfigured(f"MACTF_{name}={raw!r}: {exc}") from exc
    return _coerce(raw)                              # current is None or a list: guess


#: ``MACTF_A__B__C__D`` is the deepest form the documented blocks need
#: (``TRAINER.WEIGHT_UPDATE.params.alpha`` is four). A fifth segment is a typo.
_MAX_ENV_SEGMENTS = 4


def _close(word: str, candidates: Sequence[str]) -> str:
    """``"; did you mean X?"`` for a typo in a variable name, or the empty string."""
    close = difflib.get_close_matches(word, list(candidates), n=3, cutoff=0.6)
    return f"; did you mean {', '.join(close)}?" if close else ""


def _env_segments(key: str) -> tuple[str, ...]:
    """``MACTF_TRAINER__MUTATION__ENABLED`` -> ``("TRAINER", "MUTATION", "ENABLED")``."""
    return tuple(part.strip() for part in key[len("MACTF_") :].split("__"))


def _apply_env(values: dict[str, Any]) -> None:
    """Apply ``MACTF_*`` variables: flat scalars and ``__``-separated block leaves.

    Document 15 section 14 non-goal 2 said this walks scalars only. Document 17 section 15.6
    amends it, because a component spec is a mapping and a mapping is what every block a
    project tunes has become. Nothing changes for a variable without ``__``.
    """
    from .conf_overrides import _promotion, is_open_container, is_spec_slot

    for key in sorted(os.environ):                  # deterministic, so a note is stable
        if not key.startswith("MACTF_") or key in _ENV_NON_SETTINGS:
            continue
        segments = _env_segments(key)
        raw = os.environ[key]
        if not segments[0]:
            raise ImproperlyConfigured(f"E018: {key} names no setting")

        first = segments[0]
        if len(segments) == 1 and (first not in values or first.startswith("_")):
            # A flat variable that names no setting has always been ignored (MACTF_HOME,
            # MACTF_CONFIG_DIR, ...): only the ``__`` form promises a settings key.
            continue
        if first not in values or first.startswith("_"):
            raise ImproperlyConfigured(
                f"E018: {key}: unknown setting {first!r}"
                + _close(first, [name for name in values if not name.startswith("_")])
            )
        if len(segments) == 1:
            if isinstance(values[first], dict):
                raise ImproperlyConfigured(
                    f"E018: {key} sets {first}, which is a mapping; override one key "
                    f"instead: {key}__<KEY>"
                )
            values[first] = _coerce_env(first, raw, values[first])
            continue

        if len(segments) > _MAX_ENV_SEGMENTS:
            raise ImproperlyConfigured(
                f"E018: {key} has {len(segments)} segments; at most {_MAX_ENV_SEGMENTS} "
                "are supported (BLOCK__KEY__SUB__LEAF)"
            )
        if any(not segment for segment in segments):
            raise ImproperlyConfigured(f"E018: {key} has an empty segment")

        container: Any = values
        walked: list[str] = []
        in_spec = False                 # below a component spec: keys are the spec's own (lower case)
        for index, segment in enumerate(segments[:-1]):
            if not isinstance(container, dict):
                raise ImproperlyConfigured(
                    f"E018: {key}: {'.'.join(walked)} is not a mapping"
                )
            if in_spec and segment not in container:
                segment = segment.lower()
            if segment not in container:
                if not (in_spec or is_open_container(DEFAULTS, walked)):
                    raise ImproperlyConfigured(
                        f"E018: {key}: {'.'.join(walked) or 'settings'} has no key "
                        f"{segment!r}{_close(segment, list(container))}"
                    )
                container[segment] = {}
                if not in_spec:
                    # Created rather than declared: worth one line, because the value is now
                    # in the settings and not in any file the reader can grep.
                    log.warning(
                        "W007 %s creates %s.%s, which no default declares",
                        key, ".".join(walked), segment,
                    )
            child = container[segment]
            walked.append(segment)
            if not isinstance(child, dict) and is_spec_slot(walked) and isinstance(child, (str, type(None))):
                promoted = _promotion([part.lower() for part in segments[index + 1 :]])
                if promoted is not None:
                    container[segment] = child = promoted
            if not isinstance(child, dict):
                raise ImproperlyConfigured(
                    f"E018: {key}: {'.'.join(walked)} is a {type(child).__name__}, not a "
                    "mapping; set it whole or use a key of the mapping"
                )
            in_spec = in_spec or is_spec_slot(walked)
            container = child

        last = segments[-1]
        if in_spec and last not in container:
            last = last.lower()
        if last not in container:
            if not (in_spec or is_open_container(DEFAULTS, walked)):
                raise ImproperlyConfigured(
                    f"E018: {key}: {'.'.join(walked)} has no key {last!r}"
                    f"{_close(last, list(container))}"
                )
            if not in_spec:
                log.warning(
                    "W007 %s creates %s.%s, which no default declares",
                    key, ".".join(walked), last,
                )
            container[last] = _coerce(raw)
            continue
        container[last] = _coerce_env("__".join((*walked, last)), raw, container[last])


def validate_settings(settings: Settings) -> None:
    """Fail fast on settings that cannot produce a coherent run."""
    for key in (
        "SCORING", "WARNINGS", "COOPETITION", "TRAINING", "SANDBOX", "SEVERITY", "DISTILLATION",
    ):
        if not isinstance(settings[key], dict):
            raise ImproperlyConfigured(f"{key} must be a mapping")

    weights = settings["SEVERITY"]["WEIGHTS"]
    tiers = ["S0", "S1", "S2", "S3", "S4"]
    for tier in tiers:
        if tier not in weights:
            raise ImproperlyConfigured(f"SEVERITY.WEIGHTS is missing {tier!r}")
    sequence = [float(weights[tier]) for tier in tiers]
    if sequence != sorted(sequence) or len(set(sequence)) != len(sequence):
        raise ImproperlyConfigured("SEVERITY.WEIGHTS must be strictly increasing S0..S4")

    coop = settings["COOPETITION"]
    floor, cap = float(coop["WEIGHT_FLOOR"]), float(coop["WEIGHT_CAP"])
    if not 0.0 <= floor <= cap:
        raise ImproperlyConfigured("COOPETITION.WEIGHT_FLOOR must be within [0, WEIGHT_CAP]")
    if float(coop["KAPPA_MIN"]) <= 0.0:
        raise ImproperlyConfigured("COOPETITION.KAPPA_MIN must be strictly positive")
    if float(coop["KAPPA_MAX"]) < float(coop["KAPPA_MIN"]):
        raise ImproperlyConfigured("COOPETITION.KAPPA_MAX must be >= KAPPA_MIN")
    if not 0.0 <= float(coop["BETA_MOMENTUM"]) < 1.0:
        raise ImproperlyConfigured("COOPETITION.BETA_MOMENTUM must be in [0, 1)")
    if not 0.0 <= float(coop["TRIM_FRACTION"]) < 0.5:
        raise ImproperlyConfigured("COOPETITION.TRIM_FRACTION must be in [0, 0.5)")
    if not 0.0 < float(coop["CONFIDENCE_FLOOR"]) <= 1.0:
        raise ImproperlyConfigured("COOPETITION.CONFIDENCE_FLOOR must be in (0, 1]")
    if not 0.0 <= float(coop["REFERENCE_EMA"]) < 1.0:
        raise ImproperlyConfigured("COOPETITION.REFERENCE_EMA must be in [0, 1)")

    budget = coop.get("COOPERATIVE_BUDGET", {})
    if not isinstance(budget, dict):
        raise ImproperlyConfigured("COOPETITION.COOPERATIVE_BUDGET must be a mapping")
    for key in ("PER_EPISODE", "PER_PEER_CAP"):
        value = budget.get(key)
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ImproperlyConfigured(
                f"COOPETITION.COOPERATIVE_BUDGET.{key} must be a finite "
                f"non-negative number, got {value!r}"
            )
    per_episode = float(budget.get("PER_EPISODE", 6.0))
    per_peer_cap = float(budget.get("PER_PEER_CAP", 3.0))
    if per_peer_cap > per_episode:
        # A cap above the pool would let one peer claim the whole thing while
        # pretending a per-peer bound still applied.
        raise ImproperlyConfigured(
            "COOPETITION.COOPERATIVE_BUDGET.PER_PEER_CAP must be <= PER_EPISODE"
        )

    warnings = settings["WARNINGS"]
    if int(warnings["K_ESC"]) < 2:
        raise ImproperlyConfigured("WARNINGS.K_ESC must be at least 2")
    if int(warnings["N_WIN"]) < 1 or int(warnings["N_DECAY"]) < 1:
        raise ImproperlyConfigured("WARNINGS.N_WIN and N_DECAY must be >= 1")
    if int(warnings["W_CAP"]) > int(warnings["K_ESC"]):
        raise ImproperlyConfigured("WARNINGS.W_CAP must be <= K_ESC")

    scoring = settings["SCORING"]
    if str(scoring["PRECEDENCE_MODE"]).lower() not in {"strict", "layered"}:
        raise ImproperlyConfigured("SCORING.PRECEDENCE_MODE must be 'strict' or 'layered'")
    if not 0.0 < float(scoring.get("F_CAP_RATIO", 0.5)) <= 1.0:
        raise ImproperlyConfigured("SCORING.F_CAP_RATIO must be in (0, 1]")
    if not 0.0 < float(scoring.get("P_MIN", 0.5)) <= 1.0:
        raise ImproperlyConfigured("SCORING.P_MIN must be in (0, 1]")
    if not 0.0 < float(scoring.get("NOISE_THRESHOLD", 0.2)) < 1.0:
        raise ImproperlyConfigured("SCORING.NOISE_THRESHOLD must be in (0, 1)")
    reserved = {"P1", "P2", "P17", "P20", "P21", "P22"}
    missing = reserved - set(scoring["NON_OFFSETTABLE"])
    if missing:
        # Dropping one of these would silently permit netting across tiers.
        raise ImproperlyConfigured(
            f"SCORING.NON_OFFSETTABLE must include {sorted(missing)}"
        )

    # Document 17 section 6.6: the floor every customization is held to. Before
    # _validate_points, because PENALTY_LAMBDAS is what it reads.
    _validate_compliance(settings)
    _validate_points(settings)
    _validate_http(settings)
    # Document 14 section 10.4: the same validator the ``distillation=`` argument runs.
    validate_block(settings["DISTILLATION"])

    if str(settings["DEVICE"]).lower() not in {"auto", "cpu", "cuda", "mps"}:
        raise ImproperlyConfigured(f"unsupported DEVICE {settings['DEVICE']!r}")

    # Before the TRAINING checks: ``TRAINER`` folds into ``TRAINING`` at load, so a bad
    # ``TRAINER.EPOCHS`` would otherwise be reported under the block the author never wrote.
    _validate_blocks(settings)

    training = settings["TRAINING"]
    if int(training["EPOCHS"]) < 1:
        raise ImproperlyConfigured("TRAINING.EPOCHS must be >= 1")
    if int(training["STEPS_PER_EPOCH"]) < 1:
        raise ImproperlyConfigured("TRAINING.STEPS_PER_EPOCH must be >= 1")
    if not 0.0 <= float(training["GAMMA"]) <= 1.0:
        raise ImproperlyConfigured("TRAINING.GAMMA must be in [0, 1]")


#: Mirrors ``mactf.training.trainer.SHARING_MODES``; duplicated so validating
#: settings does not import the trainer (and torch) at load.
_SHARING_MODES = ("independent", "shared_trunk", "tied_weights")

#: Mirror ``coopetition.strategy.PEER_STRATEGIES`` and ``coopetition.weights.WEIGHT_STRATEGIES``
#: for the same reason: those modules import torch, and every command loads settings
#: (document 16, section 7.9). ``tests/cli/test_conf_mirrors.py`` keeps them equal.
#: They are views of ``components.known.BUILTIN_NAMES`` (document 17 section 15.4), not a
#: second copy of the names.
_PEER_STRATEGY_NAMES = BUILTIN_NAMES["peer_strategy"]
_WEIGHT_STRATEGY_NAMES = BUILTIN_NAMES["weight_update"]

_NEW_BLOCKS = (
    "TRAINER", "OPTIMIZER", "LOSS", "LOGGER", "WATCHDOG", "CHECKPOINT",
    "ORCHESTRATOR", "CORS",
    # Document 17: the blocks its sections own.
    "COMPONENTS", "COMPLIANCE", "RL",
)


def _canonical_components(block: Mapping[str, Any]) -> dict[str, Any]:
    """Upper-case the keys and fold an alias onto its canonical key.

    Two spellings that resolve to different values is a configuration the reader cannot
    predict, so it is refused rather than resolved by order; the same value twice is
    accepted with ``W006`` so a project can migrate one line at a time.
    """
    folded: dict[str, Any] = {}
    seen: dict[str, str] = {}
    for raw_key, value in dict(block or {}).items():
        key = str(raw_key).upper()
        canonical = _COMPONENTS_ALIASES.get(key, key)
        if canonical in folded:
            if folded[canonical] != value:
                raise ImproperlyConfigured(
                    f"E013: COMPONENTS.{key} and COMPONENTS.{seen[canonical]} both set "
                    f"{canonical} to different values ({value!r} vs {folded[canonical]!r})"
                )
            log.warning(
                "W006 COMPONENTS.%s is an alias of COMPONENTS.%s; keep one", key, canonical
            )
            continue
        folded[canonical] = value
        seen[canonical] = key
    return folded


_COMPONENTS_FLAGS = (
    "ALLOW_BUILTIN_OVERRIDE", "STRICT_LEGACY", "ALLOW_IMPORT", "FREEZE_AFTER_LOAD",
    "STRICT_RESTORE",
)


def _validate_components(settings: Any) -> None:
    """Shape and enumerations of ``COMPONENTS`` (document 17 section 15.1). No registry, no torch.

    Names that a *component reference* uses are checked by ``component_ref_problem`` at the
    place that names them (the ``TRAINER`` branch below); this function checks the block
    itself.
    """
    raw = settings.get("COMPONENTS")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ImproperlyConfigured("E013: COMPONENTS must be a mapping")
    block = _canonical_components(raw)
    unknown = sorted(set(block) - set(COMPONENTS_DEFAULTS))
    if unknown:
        raise ImproperlyConfigured(
            f"E013: COMPONENTS has unknown key(s) {unknown}; accepted: "
            + ", ".join(sorted(COMPONENTS_DEFAULTS))
        )
    for key in _COMPONENTS_FLAGS:
        value = block.get(key, COMPONENTS_DEFAULTS[key])
        if not isinstance(value, bool):
            raise ImproperlyConfigured(
                f"E013: COMPONENTS.{key} must be true or false, got {value!r}"
            )

    maximum = block.get("MAX_PER_KIND", COMPONENTS_DEFAULTS["MAX_PER_KIND"])
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ImproperlyConfigured(
            f"E013: COMPONENTS.MAX_PER_KIND must be an integer >= 1, got {maximum!r}"
        )

    modules = block.get("MODULES", [])
    if not isinstance(modules, (list, tuple)):
        raise ImproperlyConfigured(
            f"E013: COMPONENTS.MODULES must be a list of dotted module names, got "
            f"{type(modules).__name__}"
        )
    for entry in modules:
        # A path would be imported from the working directory, which is how a typo
        # becomes "it works on my machine". The message names the two usual mistakes.
        if not is_dotted_name(entry):
            raise ImproperlyConfigured(
                f"E014: COMPONENTS.MODULES entry {entry!r} is not a dotted module name; "
                "write 'myproject.components' and put its parent on --pythonpath"
            )

    plugins = block.get("PLUGINS", COMPONENTS_DEFAULTS["PLUGINS"])
    if isinstance(plugins, (list, tuple)):
        bad = [name for name in plugins if not isinstance(name, str) or not name.strip()]
        if bad:
            raise ImproperlyConfigured(
                f"E015: COMPONENTS.PLUGINS names must be non-empty strings, got {bad!r}"
            )
    elif str(plugins).lower() not in {"auto", "off"}:
        raise ImproperlyConfigured(
            "E015: COMPONENTS.PLUGINS must be 'auto', 'off' or a list of entry-point names, "
            f"got {plugins!r}"
        )

    locked = block.get("LOCKED", [])
    if not isinstance(locked, (list, tuple)):
        raise ImproperlyConfigured(
            f"E013: COMPONENTS.LOCKED must be a list of patterns, got {type(locked).__name__}"
        )
    for pattern in locked:
        problem = lock_pattern_problem(pattern, kinds=LOCKABLE_KINDS)
        if problem is not None:
            raise ImproperlyConfigured(f"E016: {problem}")

    allowlist = block.get("IMPORT_ALLOWLIST", [])
    if not isinstance(allowlist, (list, tuple)):
        raise ImproperlyConfigured(
            f"E013: COMPONENTS.IMPORT_ALLOWLIST must be a list of module patterns, got "
            f"{type(allowlist).__name__}"
        )
    for pattern in allowlist:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ImproperlyConfigured(
                f"E017: COMPONENTS.IMPORT_ALLOWLIST entry {pattern!r} is not a pattern"
            )
        if pattern in {"*", "**"}:
            # A blanket allow-list would make ALLOW_IMPORT meaningless on the one
            # surface it protects. Refused rather than documented as a footgun.
            raise ImproperlyConfigured(
                f"E017: COMPONENTS.IMPORT_ALLOWLIST cannot be {pattern!r}: it would admit "
                "every module an HTTP request names; list the packages instead"
            )


def _validate_compliance(settings: Any) -> None:
    """Document 17 section 6.6 (H2): the ``COMPLIANCE`` block, and the floor it implies."""
    raw = settings.get("COMPLIANCE") or {}
    if not isinstance(raw, Mapping):
        raise ImproperlyConfigured("COMPLIANCE must be a mapping")
    block = dict(raw)
    unknown = sorted(set(block) - set(COMPLIANCE_DEFAULTS))
    if unknown:
        raise ImproperlyConfigured(
            f"COMPLIANCE has unknown key(s) {unknown}; accepted: "
            + ", ".join(sorted(COMPLIANCE_DEFAULTS))
        )
    mode = str(block.get("GUARD_MODE", "enforce")).lower()
    if mode not in {"enforce", "fallback"}:
        raise ImproperlyConfigured("COMPLIANCE.GUARD_MODE must be 'enforce' or 'fallback'")
    # (key, default, lowest allowed, highest allowed, lowest is inclusive)
    for key, default_value, low, high, inclusive in (
        ("MIN_FLOOR_LAMBDA", 1.0, 0.0, math.inf, False),  # a zero floor would erase a penalty
        ("REWARD_CEILING", 1e4, 0.0, math.inf, False),
        ("TOLERANCE", 1e-9, 0.0, 1e-3, True),
    ):
        try:
            value = float(block.get(key, default_value))
        except (TypeError, ValueError):
            raise ImproperlyConfigured(f"COMPLIANCE.{key} must be a number") from None
        too_low = value < low if inclusive else value <= low
        if not math.isfinite(value) or too_low or value > high:
            raise ImproperlyConfigured(f"COMPLIANCE.{key} is out of range: {value!r}")
    if not isinstance(block.get("TRUST_BUILTIN", True), bool):
        raise ImproperlyConfigured("COMPLIANCE.TRUST_BUILTIN must be true or false")
    scoring = settings["SCORING"]
    floor = float(block.get("MIN_FLOOR_LAMBDA", 1.0))
    lambdas = dict(scoring.get("PENALTY_LAMBDAS") or {})
    default = float(scoring.get("DEFAULT_LAMBDA", 1.0))
    reserved = {"P1", "P2", "P17", "P20", "P21", "P22"}
    for rule_id in sorted(set(scoring["NON_OFFSETTABLE"]) | reserved):
        try:
            effective = float(lambdas.get(rule_id, default))
        except (TypeError, ValueError):
            raise ImproperlyConfigured(
                f"SCORING.PENALTY_LAMBDAS.{rule_id} is not a number: {lambdas.get(rule_id)!r}"
            ) from None
        if effective < floor:
            raise ImproperlyConfigured(
                f"SCORING.PENALTY_LAMBDAS.{rule_id} is {effective:g}, below "
                f"COMPLIANCE.MIN_FLOOR_LAMBDA {floor:g}; a non-offsettable floor cannot be weakened"
            )


_RL_RESERVED = frozenset({"OVERRIDES", "UNCONSUMED", "STRICT_GLOBS", "VERIFY", "MIGRATE"})


def _spec_shape_problem(raw: Any) -> str | None:
    """``None`` if ``raw`` can be a component spec, else why not. Names are not checked here."""
    if raw is None or isinstance(raw, str):
        return None if raw is None or raw.strip() else "an empty string is not a type name"
    if not isinstance(raw, dict):
        return f"a spec is a string or a mapping, not {type(raw).__name__}"
    has_type, has_import = "type" in raw, "import" in raw
    if has_type == has_import:
        return "a spec needs exactly one of 'type' and 'import'"
    params = raw.get("params")
    if params is not None and not isinstance(params, dict):
        return "'params' must be a mapping"
    return None


_SCORING_STAGE_KINDS = ("warning", "reward", "penalty")


def _validate_scoring_stages(settings: Any) -> None:
    """``SCORING.STAGES`` (document 17 section 6.2): a mapping of stage kind to a spec.

    Shape only: whether a name is registered is decided at build, when a plugin that
    registers it may have loaded.
    """
    scoring = settings.get("SCORING")
    stages = scoring.get("STAGES") if isinstance(scoring, dict) else None
    if stages is None:
        return
    if not isinstance(stages, dict):
        raise ImproperlyConfigured("SCORING.STAGES must map a stage kind to a component spec")
    unknown = sorted(str(kind) for kind in stages if str(kind) not in _SCORING_STAGE_KINDS)
    if unknown:
        raise ImproperlyConfigured(
            f"SCORING.STAGES has unknown kind(s) {unknown}; per-node stages are "
            f"{list(_SCORING_STAGE_KINDS)}"
        )
    for kind, raw in stages.items():
        problem = _spec_shape_problem(raw)
        if problem is not None:
            raise ImproperlyConfigured(f"SCORING.STAGES.{kind}: {problem}")


def _validate_rl(settings: Any) -> None:
    block = settings.get("RL")
    if block is None:
        return
    if not isinstance(block, dict):
        raise ImproperlyConfigured("RL must be a mapping")
    from .components.kinds import RL_FAMILIES

    families = {name.upper(): name for name in RL_FAMILIES}
    unknown = sorted(
        str(key) for key in block
        if str(key).upper() not in _RL_RESERVED and str(key).upper() not in families
    )
    if unknown:
        raise ImproperlyConfigured(
            f"RL has unknown key(s) {unknown}; accepted: {sorted(_RL_RESERVED)} and {sorted(families)}"
        )
    if str(block.get("UNCONSUMED", "error")).lower() not in {"error", "warn"}:
        raise ImproperlyConfigured("RL.UNCONSUMED must be 'error' or 'warn'")
    if str(block.get("VERIFY", "custom")).lower() not in {"custom", "all", "off"}:
        raise ImproperlyConfigured("RL.VERIFY must be 'custom', 'all' or 'off'")
    migrate = block.get("MIGRATE")
    if migrate is not None:
        if not isinstance(migrate, dict):
            raise ImproperlyConfigured("RL.MIGRATE must map FAMILY -> reset|carry|fail")
        for family_key, mode in migrate.items():
            if str(family_key).upper() not in families:
                raise ImproperlyConfigured(
                    f"RL.MIGRATE.{family_key}: unknown family; accepted: {sorted(families)}"
                )
            if str(mode).lower() not in {"reset", "carry", "fail"}:
                raise ImproperlyConfigured(
                    f"RL.MIGRATE.{family_key} must be 'reset', 'carry' or 'fail', got {mode!r}"
                )
    if not isinstance(block.get("STRICT_GLOBS", True), bool):
        raise ImproperlyConfigured("RL.STRICT_GLOBS must be a boolean")
    for key, raw in block.items():
        if str(key).upper() in families:
            problem = _spec_shape_problem(raw)
            if problem is not None:
                raise ImproperlyConfigured(f"RL.{key}: {problem}")
    overrides = block.get("OVERRIDES")
    if overrides is None:
        return
    if not isinstance(overrides, dict):
        raise ImproperlyConfigured("RL.OVERRIDES must map FAMILY -> {glob: spec}")
    for family_key, table in overrides.items():
        family = str(family_key).upper()
        if family not in families:
            raise ImproperlyConfigured(
                f"RL.OVERRIDES.{family_key}: unknown family; accepted: {sorted(families)}"
            )
        if family == "ROLLUP":
            raise ImproperlyConfigured(
                "E042: 'rollup' is served by one global object; RL.OVERRIDES.ROLLUP cannot target "
                "nodes. Use RL.ROLLUP."
            )
        if not isinstance(table, dict):
            raise ImproperlyConfigured(f"RL.OVERRIDES.{family_key} must map glob -> spec")
        for glob, raw in table.items():
            if not str(glob).strip():
                raise ImproperlyConfigured(f"RL.OVERRIDES.{family_key}: a glob cannot be empty")
            problem = _spec_shape_problem(raw)
            if problem is not None:
                raise ImproperlyConfigured(f"RL.OVERRIDES.{family_key}.{glob}: {problem}")


def _validate_optimizer(settings: Mapping[str, Any]) -> None:
    """Shape checks only. Whether a ``TYPE`` is registered is the plan's check (9.5): the
    registry is not populated until components load, which is after this runs on some paths."""
    optimizer = dict(settings.get("OPTIMIZER") or {})
    if not optimizer:
        return
    _validate_optimizer_block(optimizer, "OPTIMIZER")
    mode = str(optimizer.get("MIGRATE", "reset")).lower()
    if mode not in {"reset", "carry", "fail"}:
        raise ImproperlyConfigured(f"OPTIMIZER.MIGRATE must be 'reset', 'carry' or 'fail', got {mode!r}")
    verify = str(optimizer.get("VERIFY", "custom")).lower()
    if verify not in {"custom", "all", "off"}:
        raise ImproperlyConfigured(f"OPTIMIZER.VERIFY must be 'custom', 'all' or 'off', got {verify!r}")
    if not isinstance(optimizer.get("STRICT_GLOBS", True), bool):
        raise ImproperlyConfigured("OPTIMIZER.STRICT_GLOBS must be a boolean")
    overrides = optimizer.get("OVERRIDES") or {}
    if not isinstance(overrides, dict):
        raise ImproperlyConfigured("OPTIMIZER.OVERRIDES must map a node-path glob to an OPTIMIZER block")
    for glob, block in overrides.items():
        if not isinstance(glob, str) or not glob:
            raise ImproperlyConfigured(f"OPTIMIZER.OVERRIDES keys must be non-empty strings, got {glob!r}")
        if not isinstance(block, dict):
            raise ImproperlyConfigured(f"OPTIMIZER.OVERRIDES[{glob!r}] must be a mapping")
        if "OVERRIDES" in {str(k).upper() for k in block}:
            raise ImproperlyConfigured(f"OPTIMIZER.OVERRIDES[{glob!r}] cannot contain OVERRIDES")
        _validate_optimizer_block(block, f"OPTIMIZER.OVERRIDES[{glob!r}]")


def _validate_optimizer_block(block: Mapping[str, Any], where: str) -> None:
    upper = {str(k).upper(): v for k, v in block.items()}
    kind = upper.get("TYPE")
    if kind is not None and (not isinstance(kind, str) or not kind.strip()):
        raise ImproperlyConfigured(f"{where}.TYPE must be a non-empty string, got {kind!r}")
    lr = upper.get("LR")
    if lr is not None and float(lr) < 0.0:
        raise ImproperlyConfigured(f"{where}.LR must be >= 0")
    if float(upper.get("WEIGHT_DECAY", 0.0) or 0.0) < 0.0:
        raise ImproperlyConfigured(f"{where}.WEIGHT_DECAY must be >= 0")
    accumulation = upper.get("GRAD_ACCUMULATION")
    if accumulation is not None and int(accumulation) < 1:
        raise ImproperlyConfigured(f"{where}.GRAD_ACCUMULATION must be >= 1")
    norm = upper.get("MAX_GRAD_NORM")
    if norm is not None and float(norm) < 0.0:
        raise ImproperlyConfigured(f"{where}.MAX_GRAD_NORM must be >= 0")
    params = upper.get("PARAMS")
    if params is not None and not isinstance(params, dict):
        raise ImproperlyConfigured(f"{where}.PARAMS must be a mapping")
    kwargs = upper.get("SCHEDULER_KWARGS")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise ImproperlyConfigured(f"{where}.SCHEDULER_KWARGS must be a mapping")


_LOSS_COMPLIANCE_MODES = ("off", "require", "enforce")
_STEP_GUARD_KEYS = frozenset({
    "MODE", "SNAPSHOT", "MAX_CONSECUTIVE", "GRAD_NORM_CEILING", "MAX_GRAD_NORM",
    "CHECK_PARAMS", "VOID_CLAMP", "REQUIRE_VOID_EVIDENCE",
})
_STEP_GUARD_MODES = ("off", "warn", "enforce", "quarantine")
_STEP_GUARD_SNAPSHOTS = ("auto", "always", "never")


#: Torch-free mirrors of ``weight_guard.GUARD_MODES`` / ``GUARD_SCOPES``,
#: ``strategy`` guard modes / scopes, ``strategy_plan.MIGRATE_MODES`` and
#: ``freeze.FREEZE_MODES`` (kept equal by ``tests/cli/test_conf_mirrors.py``).
_WEIGHT_GUARD_MODES = ("repair", "fallback", "enforce")
_WEIGHT_GUARD_SCOPES = ("custom", "all", "off")
_WEIGHT_GUARD_KEYS = frozenset({"MODE", "SCOPE", "TOLERANCE", "PROBE"})
_STRATEGY_GUARD_MODES = ("skip", "fallback", "enforce")
_STRATEGY_GUARD_SCOPES = ("custom", "all", "off")
_STRATEGY_GUARD_KEYS = frozenset({"MODE", "SCOPE", "PROBE", "CHECK_FROZEN"})
_MIGRATE_MODES = ("reset", "carry", "fail")
_FREEZE_MODES = ("legacy", "strict")


def _upper_keys(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).upper(): item for key, item in dict(value).items()}


def _validate_override_map(trainer: Mapping[str, Any], key: str, kind_id: str, settings: Any) -> None:
    overrides = trainer.get(key) or {}
    if not isinstance(overrides, Mapping):
        raise ImproperlyConfigured(f"TRAINER.{key} must be a mapping of glob to spec")
    for glob, value in overrides.items():
        if not isinstance(glob, str) or not glob or glob != glob.strip():
            raise ImproperlyConfigured(f"TRAINER.{key}: {glob!r} is not a usable glob")
        problem = component_ref_problem(
            f"TRAINER.{key}[{glob!r}]", value, kind_id=kind_id, settings=settings
        )
        if problem is not None:
            raise ImproperlyConfigured(problem)


def _validate_weight_and_strategy_blocks(settings: Any) -> None:
    """Shape of the ``TRAINER`` keys document 17 sections 10 and 11 add.

    Only shapes and enumerations: accepted parameters, ``validate``, the probe, ambiguity and
    placement are the plans', at build, because they need the tree.
    """
    trainer = dict(settings.get("TRAINER") or {})
    if not trainer:
        return
    _validate_override_map(trainer, "WEIGHT_UPDATE_OVERRIDES", "weight_update", settings)
    _validate_override_map(trainer, "PEER_STRATEGY_OVERRIDES", "peer_strategy", settings)
    for key in ("WEIGHT_GUARD", "WEIGHT_OPTIONS", "STRATEGY_GUARD", "PEER_STRATEGY_OPTIONS"):
        if not isinstance(trainer.get(key) or {}, Mapping):
            raise ImproperlyConfigured(f"TRAINER.{key} must be a mapping")
    aggregator = trainer.get("AGGREGATOR")
    if aggregator is not None:
        problem = component_ref_problem(
            "TRAINER.AGGREGATOR", aggregator, kind_id="aggregator", settings=settings
        )
        if problem is not None:
            raise ImproperlyConfigured(problem)
    guard = _upper_keys(trainer.get("WEIGHT_GUARD") or {})
    unknown = sorted(set(guard) - _WEIGHT_GUARD_KEYS)
    if unknown:
        raise ImproperlyConfigured(
            f"E060: TRAINER.WEIGHT_GUARD has unknown keys {unknown}; accepted: {sorted(_WEIGHT_GUARD_KEYS)}"
        )
    if str(guard.get("MODE", "repair")).lower() not in _WEIGHT_GUARD_MODES:
        raise ImproperlyConfigured(
            f"E060: TRAINER.WEIGHT_GUARD.MODE must be one of {list(_WEIGHT_GUARD_MODES)}, got {guard.get('MODE')!r}"
        )
    if str(guard.get("SCOPE", "custom")).lower() not in _WEIGHT_GUARD_SCOPES:
        raise ImproperlyConfigured(
            f"E060: TRAINER.WEIGHT_GUARD.SCOPE must be one of {list(_WEIGHT_GUARD_SCOPES)}, got {guard.get('SCOPE')!r}"
        )
    tolerance = guard.get("TOLERANCE", 1e-6)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0.0 < tolerance < 0.1:
        raise ImproperlyConfigured(f"E060: TRAINER.WEIGHT_GUARD.TOLERANCE must be in (0, 0.1), got {tolerance!r}")
    options = _upper_keys(trainer.get("WEIGHT_OPTIONS") or {})
    unknown = sorted(set(options) - {"STRICT_GLOBS", "MIGRATE"})
    if unknown:
        raise ImproperlyConfigured(
            f"E060: TRAINER.WEIGHT_OPTIONS has unknown keys {unknown}; accepted: MIGRATE, STRICT_GLOBS"
        )
    if str(options.get("MIGRATE", "reset")).lower() not in _MIGRATE_MODES:
        raise ImproperlyConfigured(
            f"E060: TRAINER.WEIGHT_OPTIONS.MIGRATE must be one of {list(_MIGRATE_MODES)}, got {options.get('MIGRATE')!r}"
        )
    sguard = _upper_keys(trainer.get("STRATEGY_GUARD") or {})
    unknown = sorted(set(sguard) - _STRATEGY_GUARD_KEYS)
    if unknown:
        raise ImproperlyConfigured(
            f"E070: TRAINER.STRATEGY_GUARD has unknown keys {unknown}; accepted: {sorted(_STRATEGY_GUARD_KEYS)}"
        )
    if str(sguard.get("MODE", "skip")).lower() not in _STRATEGY_GUARD_MODES:
        raise ImproperlyConfigured(
            f"E070: TRAINER.STRATEGY_GUARD.MODE must be one of {list(_STRATEGY_GUARD_MODES)}, got {sguard.get('MODE')!r}"
        )
    if str(sguard.get("SCOPE", "custom")).lower() not in _STRATEGY_GUARD_SCOPES:
        raise ImproperlyConfigured(
            f"E070: TRAINER.STRATEGY_GUARD.SCOPE must be one of {list(_STRATEGY_GUARD_SCOPES)}, got {sguard.get('SCOPE')!r}"
        )
    soptions = _upper_keys(trainer.get("PEER_STRATEGY_OPTIONS") or {})
    unknown = sorted(set(soptions) - {"STRICT_GLOBS", "MIGRATE"})
    if unknown:
        raise ImproperlyConfigured(
            f"E070: TRAINER.PEER_STRATEGY_OPTIONS has unknown keys {unknown}; accepted: MIGRATE, STRICT_GLOBS"
        )
    if str(soptions.get("MIGRATE", "reset")).lower() not in _MIGRATE_MODES:
        raise ImproperlyConfigured(
            f"E070: TRAINER.PEER_STRATEGY_OPTIONS.MIGRATE must be one of {list(_MIGRATE_MODES)}, "
            f"got {soptions.get('MIGRATE')!r}"
        )
    if str(trainer.get("FREEZE_MODE", "legacy")).lower() not in _FREEZE_MODES:
        raise ImproperlyConfigured(
            f"E070: TRAINER.FREEZE_MODE must be one of {list(_FREEZE_MODES)}, got {trainer.get('FREEZE_MODE')!r}"
        )
    kappa = dict(settings.get("COOPETITION") or {}).get("KAPPA_OVERRIDES")
    if kappa is not None and not isinstance(kappa, Mapping):
        raise ImproperlyConfigured("E062: COOPETITION.KAPPA_OVERRIDES must be a mapping of glob to block")


def _validate_trainer_policy(settings: Any) -> None:
    """Shape of the ``TRAINER`` keys document 17 section 7 adds. Names are checked at build."""
    trainer = dict(settings.get("TRAINER") or {})
    if not trainer:
        return
    policy = trainer.get("POLICY")
    if policy not in (None, "", {}):
        problem = component_ref_problem(
            "TRAINER.POLICY", policy, kind_id="trainer_policy", settings=settings
        )
        if problem is not None:
            raise ImproperlyConfigured(problem)
    if "ALLOW_UNSAFE_POLICY" in trainer and not isinstance(trainer["ALLOW_UNSAFE_POLICY"], bool):
        raise ImproperlyConfigured("TRAINER.ALLOW_UNSAFE_POLICY must be a boolean")
    mode = str(trainer.get("LOSS_COMPLIANCE", "require")).lower()
    if mode not in _LOSS_COMPLIANCE_MODES:
        raise ImproperlyConfigured(
            f"TRAINER.LOSS_COMPLIANCE must be one of {list(_LOSS_COMPLIANCE_MODES)}, got {mode!r}"
        )
    coefficient = trainer.get("COMPLIANCE_COEFFICIENT", 1.0)
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ImproperlyConfigured("TRAINER.COMPLIANCE_COEFFICIENT must be a number >= 0")
    guard = trainer.get("STEP_GUARD")
    if guard is None:
        return
    if not isinstance(guard, dict):
        raise ImproperlyConfigured("TRAINER.STEP_GUARD must be a mapping")
    data = {str(key).upper(): value for key, value in guard.items()}
    unknown = sorted(set(data) - _STEP_GUARD_KEYS)
    if unknown:
        raise ImproperlyConfigured(
            "unknown TRAINER.STEP_GUARD key(s): " + ", ".join(unknown)
            + "; accepted: " + ", ".join(sorted(_STEP_GUARD_KEYS))
        )
    if "MODE" in data and str(data["MODE"]).lower() not in _STEP_GUARD_MODES:
        raise ImproperlyConfigured(
            f"TRAINER.STEP_GUARD.MODE must be one of {list(_STEP_GUARD_MODES)}, got {data['MODE']!r}"
        )
    if "SNAPSHOT" in data and str(data["SNAPSHOT"]).lower() not in _STEP_GUARD_SNAPSHOTS:
        raise ImproperlyConfigured(
            f"TRAINER.STEP_GUARD.SNAPSHOT must be one of {list(_STEP_GUARD_SNAPSHOTS)}, "
            f"got {data['SNAPSHOT']!r}"
        )
    if "MAX_CONSECUTIVE" in data and int(data["MAX_CONSECUTIVE"]) < 1:
        raise ImproperlyConfigured("TRAINER.STEP_GUARD.MAX_CONSECUTIVE must be >= 1")
    if "GRAD_NORM_CEILING" in data and not float(data["GRAD_NORM_CEILING"]) > 0.0:
        raise ImproperlyConfigured("TRAINER.STEP_GUARD.GRAD_NORM_CEILING must be > 0")
    if data.get("MAX_GRAD_NORM") is not None and not float(data["MAX_GRAD_NORM"]) > 0.0:
        raise ImproperlyConfigured("TRAINER.STEP_GUARD.MAX_GRAD_NORM must be > 0 when set")


def _validate_mutation_block(settings: Any) -> None:
    """Shape and enumerations of ``TRAINER.MUTATION`` (document 17 section 12.10).

    ``MutationLimits.validate_block`` is the one validator, so the configuration and the trainer
    cannot drift. ``mactf.training.mutation`` imports ``mactf.exceptions`` and
    ``mactf.coopetition.tree`` only, both torch-free, so this check is safe in the load path
    every command pays for.
    """
    trainer = dict(settings.get("TRAINER") or {})
    block = trainer.get("MUTATION")
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise ImproperlyConfigured("TRAINER.MUTATION must be a mapping")
    from .training.mutation import MutationLimits

    clean = MutationLimits.validate_block(block)
    if clean.get("SCHEDULE") and not clean.get("ENABLED", False):
        # A schedule is meaningless without the engine that reads it, and silently ignoring it
        # would be the kind of dead setting section 2.2 is about.
        raise ImproperlyConfigured(
            "TRAINER.MUTATION.SCHEDULE is set but TRAINER.MUTATION.ENABLED is false; "
            "the schedule would never run"
        )


def _validate_blocks(settings: Any) -> None:
    """Document 15 section 5.11: the blocks the ``conf/`` tree adds."""
    for name in _NEW_BLOCKS:
        if name in settings and not isinstance(settings[name], dict):
            prefix = "E013: " if name == "COMPONENTS" else ""
            raise ImproperlyConfigured(f"{prefix}{name} must be a mapping")
    _validate_components(settings)

    trainer = dict(settings.get("TRAINER") or {})
    if trainer:
        if int(trainer.get("EPOCHS", 3)) < 1:
            raise ImproperlyConfigured("TRAINER.EPOCHS must be >= 1")
        if int(trainer.get("BATCH_SIZE", 32)) < 1:
            raise ImproperlyConfigured("TRAINER.BATCH_SIZE must be >= 1")
        if int(trainer.get("STEPS_PER_EPOCH", 4)) < 1:
            raise ImproperlyConfigured("TRAINER.STEPS_PER_EPOCH must be >= 1")
        if int(trainer.get("GRAD_ACCUMULATION", 1)) < 1:
            raise ImproperlyConfigured("TRAINER.GRAD_ACCUMULATION must be >= 1")
        for where, key, kind_id in (
            ("TRAINER.PEER_STRATEGY", "PEER_STRATEGY", "peer_strategy"),
            ("TRAINER.WEIGHT_UPDATE", "WEIGHT_UPDATE", "weight_update"),
        ):
            problem = component_ref_problem(
                where, trainer.get(key, "kappa_softmax"), kind_id=kind_id, settings=settings
            )
            if problem is not None:
                raise ImproperlyConfigured(problem)
        if str(trainer.get("SHARING", "independent")).lower() not in _SHARING_MODES:
            raise ImproperlyConfigured(
                f"TRAINER.SHARING must be one of {list(_SHARING_MODES)}, "
                f"got {trainer.get('SHARING')!r}"
            )

    _validate_optimizer(settings)

    _validate_learning_rate(settings)

    _validate_rl(settings)
    _validate_scoring_stages(settings)
    _validate_trainer_policy(settings)
    _validate_weight_and_strategy_blocks(settings)
    _validate_mutation_block(settings)

    loss = dict(settings.get("LOSS") or {})
    if loss:
        if not 0.0 <= float(loss.get("KL_COEF", 0.01)):
            raise ImproperlyConfigured("LOSS.KL_COEF must be >= 0")
        if not 0.0 <= float(loss.get("ENTROPY_COEF", 0.0)):
            raise ImproperlyConfigured("LOSS.ENTROPY_COEF must be >= 0")
        if not 0.0 < float(loss.get("CLIP_EPSILON", 0.2)) <= 1.0:
            raise ImproperlyConfigured("LOSS.CLIP_EPSILON must be in (0, 1]")

    logger = dict(settings.get("LOGGER") or {})
    if logger:
        level = str(logger.get("LEVEL", "INFO")).upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ImproperlyConfigured(
                f"LOGGER.LEVEL must be a standard level, got {level!r}"
            )
        sinks = logger.get("SINKS", ["console", "file"])
        known_sinks = {"console", "file", "jsonl", "tensorboard", "wandb"}
        if not isinstance(sinks, (list, tuple)) or not sinks:
            raise ImproperlyConfigured("LOGGER.SINKS must be a non-empty list")
        unknown_sinks = sorted(set(map(str, sinks)) - known_sinks)
        if unknown_sinks:
            raise ImproperlyConfigured(
                f"LOGGER.SINKS names unknown sink(s) {unknown_sinks}; use {sorted(known_sinks)}"
            )
        if "file" in sinks and not logger.get("LOG_FILE"):
            raise ImproperlyConfigured("LOGGER.SINKS includes 'file' but LOG_FILE is unset")
        if "jsonl" in sinks and not logger.get("JSONL_FILE"):
            raise ImproperlyConfigured("LOGGER.SINKS includes 'jsonl' but JSONL_FILE is unset")
        if "tensorboard" in sinks and not logger.get("TENSORBOARD_DIR"):
            raise ImproperlyConfigured(
                "LOGGER.SINKS includes 'tensorboard' but TENSORBOARD_DIR is unset"
            )

    watchdog = dict(settings.get("WATCHDOG") or {})
    if watchdog:
        from .training.watchdog import ACTION_NAMES, WatchdogPolicy

        for key in ("ON_DEGRADATION", "ON_CRITICAL"):
            action = str(watchdog.get(key, "snapshot")).lower()
            if action not in ACTION_NAMES:
                raise ImproperlyConfigured(
                    f"WATCHDOG.{key} must be one of {list(ACTION_NAMES)}, got {action!r}"
                )
        known = {field.lower() for field in WatchdogPolicy.__dataclass_fields__}
        known |= {
            "snapshot_every_epochs", "snapshot_every_seconds", "snapshot_on_degradation",
            "on_degradation", "on_critical", "keep_history",
        }
        unknown = sorted({str(key).lower() for key in watchdog} - known)
        if unknown:
            raise ImproperlyConfigured(
                f"WATCHDOG names unknown key(s) {unknown}; a typo in a threshold is a "
                "watchdog that never fires"
            )

    checkpoint = dict(settings.get("CHECKPOINT") or {})
    if checkpoint:
        if int(checkpoint.get("KEEP", 5)) < 0 or int(checkpoint.get("KEEP_BEST", 1)) < 0:
            raise ImproperlyConfigured("CHECKPOINT.KEEP and KEEP_BEST must be >= 0")
        checksum = str(checkpoint.get("CHECKSUM", "sha256")).lower()
        if checksum not in {"sha256", "blake2b"}:
            raise ImproperlyConfigured(
                f"CHECKPOINT.CHECKSUM must be 'sha256' or 'blake2b', got {checksum!r}"
            )

    orchestrator = dict(settings.get("ORCHESTRATOR") or {})
    if orchestrator:
        from .orchestration.dispatch import ENGINES  # torch-free, so settings stay torch-free

        raw_engine = orchestrator.get("ENGINE")
        engine = "orchestrator" if raw_engine is None or raw_engine == "" else str(raw_engine).strip().lower()
        if engine not in ENGINES:
            raise ImproperlyConfigured(
                f"E090: ORCHESTRATOR.ENGINE must be one of {list(ENGINES)}, got "
                f"{orchestrator.get('ENGINE')!r}"
            )
        for key in ("ENGINE_COMPAT", "ALLOW_ENGINE_SWITCH"):
            if key in orchestrator and not isinstance(orchestrator[key], bool):
                raise ImproperlyConfigured(
                    f"ORCHESTRATOR.{key} must be true or false, got {orchestrator[key]!r}"
                )
        epochs = orchestrator.get("MAX_EPOCHS")
        if epochs is not None and (isinstance(epochs, bool) or int(epochs) < 1):
            raise ImproperlyConfigured("ORCHESTRATOR.MAX_EPOCHS must be None or >= 1")

    api, cors = settings.get("API"), settings.get("CORS")
    authored_cors = (cors or {}).get("ORIGINS") if isinstance(cors, dict) else None
    if isinstance(api, dict) and authored_cors and api.get("CORS_ORIGINS"):
        raise ImproperlyConfigured(
            "API.CORS_ORIGINS and CORS.ORIGINS both set; use CORS for the rich form "
            "and leave API.CORS_ORIGINS empty"
        )


def _validate_learning_rate(settings: Any) -> None:
    """``TRAINING.LEARNING_RATE`` and ``OPTIMIZER.LR`` must not disagree.

    Checked only when the project wrote both: a framework default is not a
    conflict, and a plain mapping with no authorship record is not checked.
    """
    from .conf_files import authored_keys

    optimizer_keys = authored_keys(settings, "OPTIMIZER")
    training_keys = authored_keys(settings, "TRAINING")
    if not optimizer_keys or not training_keys:
        return
    if "LR" not in optimizer_keys or "LEARNING_RATE" not in training_keys:
        return
    lr = settings["OPTIMIZER"].get("LR")
    rate = settings["TRAINING"].get("LEARNING_RATE")
    if lr is None or rate is None:
        return
    if abs(float(lr) - float(rate)) > 1e-12:
        raise ImproperlyConfigured(
            f"TRAINING.LEARNING_RATE={rate!r} and OPTIMIZER.LR={lr!r} disagree; "
            "set one, or set both to the same value"
        )


def _validate_http(settings: Any) -> None:
    """Doc 13 section 3: URL conf, middleware and the HTTP/API blocks."""
    urlconf = settings["ROOT_URLCONF"]
    if urlconf is not None and (not isinstance(urlconf, str) or not urlconf.strip()):
        raise ImproperlyConfigured("ROOT_URLCONF must be None or a non-empty dotted string")
    middleware = settings["MIDDLEWARE"]
    if not isinstance(middleware, (list, tuple)) or not all(
        isinstance(entry, str) and entry for entry in middleware
    ):
        raise ImproperlyConfigured("MIDDLEWARE must be a list of dotted-path strings")
    if not isinstance(settings["APPEND_SLASH"], bool):
        raise ImproperlyConfigured("APPEND_SLASH must be a bool")
    api, http = settings["API"], settings["HTTP"]
    if not isinstance(api, dict) or not isinstance(http, dict):
        raise ImproperlyConfigured("API and HTTP must be mappings")
    base = api.get("BASE_PATH", "api/v1")
    if not isinstance(base, str) or not base or base.startswith("/") or base.endswith("/"):
        raise ImproperlyConfigured("API.BASE_PATH must be non-empty with no leading or trailing slash")
    body = http.get("MAX_BODY_BYTES")
    if isinstance(body, bool) or not isinstance(body, int) or body <= 0:
        raise ImproperlyConfigured("HTTP.MAX_BODY_BYTES must be a positive int")
    workers = http.get("STREAM_WORKERS")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 0 < workers <= 256:
        raise ImproperlyConfigured("HTTP.STREAM_WORKERS must be an int in [1, 256]")
    heartbeat = http.get("SSE_HEARTBEAT")
    if isinstance(heartbeat, bool) or not isinstance(heartbeat, (int, float)) or not (
        math.isfinite(heartbeat) and heartbeat > 0
    ):
        raise ImproperlyConfigured("HTTP.SSE_HEARTBEAT must be a positive number")
    if not isinstance(api.get("TOKENS", {}), dict):
        raise ImproperlyConfigured("API.TOKENS must be a mapping")
    for key in ("EVENT_BUFFER", "MAX_CONCURRENT_RUNS", "LOG_TAIL_DEFAULT", "LOG_TAIL_MAX"):
        value = api.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ImproperlyConfigured(f"API.{key} must be a positive int")


def _validate_points(settings: Any) -> None:
    """Doc 11 section 9.2: the points table, engine bounds and signal source."""
    from .scoring.guards import validate_bindings
    from .scoring.normalize import NormalizationPolicy
    from .scoring.points import PointsTable

    scoring = settings["SCORING"]
    table = PointsTable.from_settings(scoring, validate=False)
    problems = table.validate()
    if problems:
        raise ImproperlyConfigured("SCORING.POINTS invalid: " + "; ".join(problems))
    problems = validate_bindings()
    if problems:
        raise ImproperlyConfigured("; ".join(problems))

    engine = settings["ENGINE"] if "ENGINE" in settings else {}
    if int(engine.get("F_CAP", 3)) < 1:
        raise ImproperlyConfigured("ENGINE.F_CAP must be >= 1")
    if not 0.0 < float(engine.get("P_MIN", 0.5)) <= 1.0:
        raise ImproperlyConfigured("ENGINE.P_MIN must be in (0, 1]")
    if not 0.0 <= float(engine.get("P12_NOISE_RATIO", 0.5)) <= 1.0:
        raise ImproperlyConfigured("ENGINE.P12_NOISE_RATIO must be in [0, 1]")
    if float(engine.get("R9_STEP_FACTOR", 1.25)) < 1.0:
        raise ImproperlyConfigured("ENGINE.R9_STEP_FACTOR must be >= 1")

    coop = settings["COOPETITION"]
    source = str(coop.get("SIGNAL_SOURCE", "final_score")).lower()
    if source not in {"final_score", "points"}:
        raise ImproperlyConfigured(
            f"COOPETITION.SIGNAL_SOURCE must be 'final_score' or 'points', got {source!r}"
        )
    NormalizationPolicy.from_settings(coop.get("NORMALIZATION"))  # raises on bad values


_settings: Settings | None = None


def iter_leaves(tree: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(path, value)`` for every leaf of a settings tree.

    Mappings recurse; lists, scalars, and *empty* mappings are leaves (an empty
    ``API.TOKENS`` is a setting, not an absence); keys that start with ``_`` are skipped.
    The tuple form exists because a key may itself contain a dot.
    """
    if isinstance(tree, Mapping) and tree:
        for key, item in tree.items():
            if not str(key).startswith("_"):
                yield from iter_leaves(item, (*prefix, str(key)))
    elif prefix or not isinstance(tree, Mapping):
        yield prefix, tree


def flatten_leaves(values: Mapping[str, Any]) -> dict[str, Any]:
    return {".".join(path): value for path, value in iter_leaves(values)}


class _Trace:
    """Attributes every leaf to the layer that last changed it (``trace=True``)."""

    def __init__(self) -> None:
        self.origins: dict[str, str] = {}
        self.paths: dict[str, str | None] = {}
        self._order: list[str] = []
        self._previous: dict[str, Any] = {}

    def _note(self, label: str, path: str | None) -> None:
        if label not in self.paths:
            self.paths[label] = path
            self._order.append(label)

    def layer(self, values: Mapping[str, Any], label: Any, path: str | None = None, *,
              always: bool = False) -> None:
        """Snapshot ``values``; each changed leaf is attributed to ``label``.

        ``label`` may be a callable taking the leaf's top-level key, for layers (the
        ``conf/`` tree) whose origin differs per key.
        """
        current = flatten_leaves(values)
        if always and isinstance(label, str):
            self._note(label, path)
        for key, value in current.items():
            if key in self._previous and self._previous[key] == value:
                continue
            name = label(key.split(".", 1)[0]) if callable(label) else label
            self.origins[key] = name
            self._note(name, path)
        for key in set(self._previous) - set(current):
            self.origins.pop(key, None)
        self._previous = copy.deepcopy(current)

    def resync(self, values: Mapping[str, Any]) -> None:
        """Absorb ``_sync_trainer`` (TRAINER mirrors TRAINING) without attributing it."""
        current = flatten_leaves(values)
        for key, target in _TRAINER_TO_TRAINING.items():
            origin = self.origins.get(f"TRAINING.{target}")
            if origin and f"TRAINER.{key}" in current:
                self.origins[f"TRAINER.{key}"] = origin
        self._previous = copy.deepcopy(current)

    def layers(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for label in self.origins.values():
            counts[label] = counts.get(label, 0) + 1
        return [{"label": label, "path": self.paths[label], "leaves": counts.get(label, 0)}
                for label in self._order]


def load_settings(
    module_path: str | None = None,
    *,
    overrides: Sequence[str] = (),
    config_files: Sequence[str | os.PathLike[str]] = (),
    trace: bool = False,
) -> Settings:
    """Build a :class:`Settings` instance from a module, YAML overlays, and the command line.

    ``config_files`` and ``overrides`` are the two command-line layers (``--config`` and
    ``--set``); both sit above the environment. ``trace`` records per-leaf provenance in
    ``_ORIGINS`` and ``_LAYERS`` (document 16, section 5.3).
    """
    module_path = module_path or os.environ.get(ENV_SETTINGS_MODULE) or DEFAULT_SETTINGS_MODULE
    values = copy.deepcopy(DEFAULTS)
    config_files_cli = tuple(config_files)
    tracer = _Trace() if trace else None
    if tracer:
        tracer.layer(values, "default")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImproperlyConfigured(
            f"could not import settings module {module_path!r}: {exc}"
        ) from exc

    root = Path(getattr(module, "BASE_DIR", os.getcwd())).expanduser().resolve()
    values["PATHS"]["ROOT"] = str(root)

    # What the project wrote, as opposed to what DEFAULTS supplied: block name to
    # the top-level keys it set. Consumers read it through conf_files.is_authored.
    authored: dict[str, set[str]] = {}

    for key, value in vars(module).items():
        if key.isupper() and key in DEFAULTS:
            if isinstance(value, dict) and isinstance(values[key], dict):
                _deep_merge(values[key], value)
            else:
                values[key] = copy.deepcopy(value)
            _mark_authored(authored, key, value)

    if values.get("CONF_DIR") is not None:
        # A Path in settings.py becomes a string so as_dict() stays JSON-friendly.
        values["CONF_DIR"] = str(values["CONF_DIR"])
    if tracer:
        tracer.layer(values, f"settings:{module_path}", getattr(module, "__file__", None))

    # Step 2.5: the project's conf/ tree. It sits below CONFIG_FILES because an
    # overlay is an exception to the project's configuration, not a peer of it:
    # a CI run that needs six epochs passes an overlay, and the committed tree
    # must not silently win over it.
    from .conf_files import load_conf_tree

    conf_values, conf_notes = load_conf_tree(values, root)
    for key, value in conf_values.items():
        if isinstance(value, dict) and isinstance(values.get(key), dict):
            _deep_merge(values[key], value)
        else:
            values[key] = copy.deepcopy(value)
        _mark_authored(authored, key, value)
    if tracer:
        from .conf_files import owner_of, resolve_conf_dir

        conf_dir = resolve_conf_dir(values, root)
        tracer.layer(
            values,
            lambda key: f"conf:{owner_of(key) or 'conf'}",
            str(conf_dir) if conf_dir is not None else None,
        )

    # TRAINER is folded before the overlays so an overlay's TRAINING still wins.
    fold_notes = _fold_trainer(values, authored.get("TRAINER", set()))
    conf_notes = [*conf_notes, *fold_notes]
    if tracer:
        tracer.layer(values, "fold:TRAINER")

    config_files = []
    for entry in values.get("CONFIG_FILES") or []:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise ImproperlyConfigured(f"CONFIG_FILES entry does not exist: {candidate}")
        config_files.append(candidate)

    for path in config_files:
        overlay = _load_yaml(path)
        _deep_merge(values, overlay)
        for key, value in overlay.items():
            _mark_authored(authored, str(key), value)
        trainer_overlay = overlay.get("TRAINER")
        if isinstance(trainer_overlay, dict):
            conf_notes.extend(_fold_trainer(values, set(map(str, trainer_overlay))))
        if tracer:
            tracer.layer(values, f"overlay:{path}", str(path), always=True)

    _sync_trainer(values)
    if tracer:
        tracer.resync(values)
    _apply_env(values)
    if tracer:
        tracer.layer(values, lambda key: f"env:MACTF_{key}")

    for entry in config_files_cli:                   # layer 6: --config
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise ImproperlyConfigured(f"--config file does not exist: {path}")
        overlay = _load_yaml(path)
        _deep_merge(values, overlay)
        for key, value in overlay.items():
            _mark_authored(authored, str(key), value)
        trainer_overlay = overlay.get("TRAINER")
        if isinstance(trainer_overlay, dict):
            conf_notes.extend(_fold_trainer(values, set(map(str, trainer_overlay))))
        if tracer:
            tracer.layer(values, f"cli:--config {path}", str(path), always=True)

    if overrides:                                    # layer 7: --set
        from .conf_overrides import apply_overrides, parse_override

        parsed = [parse_override(text) for text in overrides]
        written = apply_overrides(values, parsed, defaults=DEFAULTS)
        for override in parsed:
            if len(override.path) > 1:
                _mark_authored(authored, override.path[0], {override.path[1]: None})
            else:
                _mark_authored(authored, override.path[0], override.value)
        conf_notes.extend(_fold_trainer(values, written.trainer_keys))
        conf_notes.extend(f"--set {o.text}" for o in parsed)
        if tracer:
            tracer.layer(values, "cli:--set", always=True)

    if config_files_cli or overrides:
        _sync_trainer(values)                        # layers 6 and 7 may have written TRAINING
        if tracer:
            tracer.resync(values)
    values["_AUTHORED"] = {name: sorted(keys) for name, keys in sorted(authored.items())}
    notes = conf_notes
    settings = Settings(values, module=module_path)
    validate_settings(settings)
    # Set after validation so a validator never sees it; as_dict() carries it so a
    # support bundle can say which file a value came from.
    settings["_CONF_NOTES"] = notes
    if tracer:
        settings["_ORIGINS"] = dict(tracer.origins)
        settings["_LAYERS"] = tracer.layers()
    return settings


def _mark_authored(authored: dict[str, set[str]], key: str, value: Any) -> None:
    """Record that the project wrote ``key`` (and, for a mapping, which sub-keys)."""
    keys = authored.setdefault(key, set())
    if isinstance(value, dict):
        keys.update(str(item) for item in value)


def _fold_trainer(values: dict[str, Any], written: set[str]) -> list[str]:
    """``TRAINER`` is the new spelling of ``TRAINING``; keep one live copy.

    Folding at load means a consumer reads ``TRAINING`` and gets the project's
    ``TRAINER`` values, without every consumer growing a fallback branch. Only the
    keys the project *wrote* are folded: ``TRAINER``'s framework defaults must not
    clobber a legacy project's ``TRAINING.EPOCHS``. ``TRAINER`` stays in
    ``as_dict()`` so a support bundle shows what the project wrote.
    """
    trainer = values.get("TRAINER")
    training = values.get("TRAINING")
    notes: list[str] = []
    if not isinstance(trainer, dict) or not isinstance(training, dict):
        return notes
    for key in sorted(written):
        target = _TRAINER_TO_TRAINING.get(key)
        if target is None or key not in trainer:
            continue
        if target in training and training[target] != trainer[key]:
            notes.append(
                f"TRAINER.{key}={trainer[key]!r} replaces TRAINING.{target}={training[target]!r}"
            )
        training[target] = copy.deepcopy(trainer[key])
    return notes


def _sync_trainer(values: dict[str, Any]) -> None:
    """Make ``TRAINER`` report what ``TRAINING`` resolved to, for the shared keys."""
    trainer = values.get("TRAINER")
    training = values.get("TRAINING")
    if not isinstance(trainer, dict) or not isinstance(training, dict):
        return
    for key, target in _TRAINER_TO_TRAINING.items():
        if target in training:
            trainer[key] = copy.deepcopy(training[target])


def get_settings() -> Settings:
    """Return the process-wide settings object, loading it on first use."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def configure(
    module_path: str | None = None,
    *,
    force: bool = False,
    overrides: Sequence[str] = (),
    config_files: Sequence[str | os.PathLike[str]] = (),
) -> Settings:
    """(Re)load settings and reset the app registry.

    ``overrides`` and ``config_files`` are the command-line layers (``--set``,
    ``--config``); a caller that passes either should also pass ``force=True``, because
    a cached ``Settings`` would otherwise silently ignore them.
    """
    global _settings
    if _settings is not None and not force:
        return _settings
    from . import apps as apps_module

    _settings = load_settings(module_path, overrides=overrides, config_files=config_files)
    apps_module.reset_registry()
    _clear_http_caches()
    # Document 17 section 15.5: the registry is configured with the settings that describe
    # it, before any consumer resolves a spec. Idempotent and torch-free unless a MODULE
    # imports torch, which is the project's own decision.
    from .components.plugins import ensure_components_loaded

    ensure_components_loaded(_settings)
    return _settings


def reset() -> None:
    """Discard cached settings. Intended for tests."""
    global _settings
    _settings = None
    _clear_http_caches()


def _clear_http_caches() -> None:
    """Drop resolver/application caches, but only if the http package is loaded."""
    import sys

    http = sys.modules.get("mactf.http")
    if http is not None and hasattr(http, "clear_caches"):
        http.clear_caches()
