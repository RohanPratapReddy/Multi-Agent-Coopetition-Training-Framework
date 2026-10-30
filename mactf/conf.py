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
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .distill.config import DISTILLATION_DEFAULTS, validate_block
from .exceptions import ImproperlyConfigured

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
}

CORS_DEFAULTS: dict[str, Any] = {
    "ORIGINS": [],
    "ALLOW_CREDENTIALS": False,
    "ALLOW_HEADERS": ["authorization", "content-type", "x-request-id"],
    "MAX_AGE": 600,
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
            return value
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


def _apply_env(values: dict[str, Any]) -> None:
    for key, raw in os.environ.items():
        if not key.startswith("MACTF_") or key in _ENV_NON_SETTINGS:
            continue
        name = key[len("MACTF_") :]
        if name not in values or isinstance(values[name], dict):
            continue
        values[name] = _coerce_env(name, raw, values[name])


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
_PEER_STRATEGY_NAMES = frozenset({
    "independent", "cooperative", "kappa_softmax", "frozen_peer", "hierarchical",
})
_WEIGHT_STRATEGY_NAMES = frozenset({
    "kappa_softmax", "uniform", "score_weighted", "reference_only", "none", "explicit",
})

_NEW_BLOCKS = (
    "TRAINER", "OPTIMIZER", "LOSS", "LOGGER", "WATCHDOG", "CHECKPOINT",
    "ORCHESTRATOR", "CORS",
)


def _validate_blocks(settings: Any) -> None:
    """Document 15 section 5.11: the blocks the ``conf/`` tree adds."""
    PEER_STRATEGIES = _PEER_STRATEGY_NAMES
    WEIGHT_STRATEGIES = _WEIGHT_STRATEGY_NAMES

    for name in _NEW_BLOCKS:
        if name in settings and not isinstance(settings[name], dict):
            raise ImproperlyConfigured(f"{name} must be a mapping")

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
        if str(trainer.get("PEER_STRATEGY", "kappa_softmax")).lower() not in PEER_STRATEGIES:
            raise ImproperlyConfigured(
                "TRAINER.PEER_STRATEGY must be one of " + ", ".join(sorted(PEER_STRATEGIES))
            )
        if str(trainer.get("WEIGHT_UPDATE", "kappa_softmax")).lower() not in WEIGHT_STRATEGIES:
            raise ImproperlyConfigured(
                "TRAINER.WEIGHT_UPDATE must be one of " + ", ".join(sorted(WEIGHT_STRATEGIES))
            )
        if str(trainer.get("SHARING", "independent")).lower() not in _SHARING_MODES:
            raise ImproperlyConfigured(
                f"TRAINER.SHARING must be one of {list(_SHARING_MODES)}, "
                f"got {trainer.get('SHARING')!r}"
            )

    optimizer = dict(settings.get("OPTIMIZER") or {})
    if optimizer:
        kind = str(optimizer.get("TYPE", "torch")).lower()
        if kind not in {"torch", "scipy", "callable"}:
            raise ImproperlyConfigured(
                f"OPTIMIZER.TYPE must be 'torch', 'scipy', or 'callable', got {kind!r}"
            )
        lr = optimizer.get("LR")
        if lr is not None and float(lr) < 0.0:
            raise ImproperlyConfigured("OPTIMIZER.LR must be >= 0")
        if float(optimizer.get("WEIGHT_DECAY", 0.0)) < 0.0:
            raise ImproperlyConfigured("OPTIMIZER.WEIGHT_DECAY must be >= 0")
        if int(optimizer.get("GRAD_ACCUMULATION", 1)) < 1:
            raise ImproperlyConfigured("OPTIMIZER.GRAD_ACCUMULATION must be >= 1")

    _validate_learning_rate(settings)

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
