"""Settings loading, merging, and typed access.

Resolution order, later winning:

1. Framework defaults (:data:`DEFAULTS`).
2. A settings module named by ``MACTF_SETTINGS_MODULE``.
3. ``settings.CONFIG_FILES`` YAML overlays, in order.
4. ``MACTF_<SETTING>`` environment variables for scalar overrides.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
from typing import Any

from .exceptions import ImproperlyConfigured

ENV_SETTINGS_MODULE = "MACTF_SETTINGS_MODULE"
# The distribution cannot assume a project exists, so the default is the
# framework's own minimal settings module. Projects set MACTF_SETTINGS_MODULE.
DEFAULT_SETTINGS_MODULE = "mactf.settings_default"

DEFAULTS: dict[str, Any] = {
    "DEBUG": False,
    "INSTALLED_APPS": [
        "mactf.contrib.scoring",
        "mactf.contrib.coopetition",
        "mactf.contrib.sandbox",
        "mactf.contrib.telemetry",
    ],
    "PEERS": [],
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
        "AGGREGATE_BUFFERS": "top",
        "COOPERATIVE_POOL": True,
    },
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
    },
    "TELEMETRY": {
        "LEVEL": "INFO",
        "JSON": False,
        "CONSOLE": True,
        "LOG_FILE": "var/logs/mactf.log",
    },
    "CONFIG_FILES": [],
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
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

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._values)

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


def _apply_env(values: dict[str, Any]) -> None:
    for key, raw in os.environ.items():
        if not key.startswith("MACTF_") or key == ENV_SETTINGS_MODULE:
            continue
        name = key[len("MACTF_") :]
        if name not in values or isinstance(values[name], dict):
            continue
        values[name] = _coerce(raw)


def validate_settings(settings: Settings) -> None:
    """Fail fast on settings that cannot produce a coherent run."""
    for key in ("SCORING", "WARNINGS", "COOPETITION", "TRAINING", "SANDBOX", "SEVERITY"):
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
    reserved = {"P1", "P2", "P17", "P20", "P21", "P22"}
    missing = reserved - set(scoring["NON_OFFSETTABLE"])
    if missing:
        # Dropping one of these would silently permit netting across tiers.
        raise ImproperlyConfigured(
            f"SCORING.NON_OFFSETTABLE must include {sorted(missing)}"
        )

    if str(settings["DEVICE"]).lower() not in {"auto", "cpu", "cuda", "mps"}:
        raise ImproperlyConfigured(f"unsupported DEVICE {settings['DEVICE']!r}")

    training = settings["TRAINING"]
    if int(training["EPOCHS"]) < 1:
        raise ImproperlyConfigured("TRAINING.EPOCHS must be >= 1")
    if int(training["STEPS_PER_EPOCH"]) < 1:
        raise ImproperlyConfigured("TRAINING.STEPS_PER_EPOCH must be >= 1")
    if not 0.0 <= float(training["GAMMA"]) <= 1.0:
        raise ImproperlyConfigured("TRAINING.GAMMA must be in [0, 1]")


_settings: Settings | None = None


def load_settings(module_path: str | None = None) -> Settings:
    """Build a :class:`Settings` instance from a module and YAML overlays."""
    module_path = module_path or os.environ.get(ENV_SETTINGS_MODULE) or DEFAULT_SETTINGS_MODULE
    values = copy.deepcopy(DEFAULTS)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImproperlyConfigured(
            f"could not import settings module {module_path!r}: {exc}"
        ) from exc

    root = Path(getattr(module, "BASE_DIR", os.getcwd())).expanduser().resolve()
    values["PATHS"]["ROOT"] = str(root)

    for key, value in vars(module).items():
        if key.isupper() and key in DEFAULTS:
            if isinstance(value, dict) and isinstance(values[key], dict):
                _deep_merge(values[key], value)
            else:
                values[key] = copy.deepcopy(value)

    config_files = []
    for entry in values.get("CONFIG_FILES") or []:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise ImproperlyConfigured(f"CONFIG_FILES entry does not exist: {candidate}")
        config_files.append(candidate)

    for path in config_files:
        _deep_merge(values, _load_yaml(path))

    _apply_env(values)
    settings = Settings(values, module=module_path)
    validate_settings(settings)
    return settings


def get_settings() -> Settings:
    """Return the process-wide settings object, loading it on first use."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def configure(module_path: str | None = None, *, force: bool = False) -> Settings:
    """(Re)load settings and reset the app registry."""
    global _settings
    if _settings is not None and not force:
        return _settings
    from . import apps as apps_module

    _settings = load_settings(module_path)
    apps_module.reset_registry()
    return _settings


def reset() -> None:
    """Discard cached settings. Intended for tests."""
    global _settings
    _settings = None
