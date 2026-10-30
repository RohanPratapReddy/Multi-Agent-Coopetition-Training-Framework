"""Named task-source registry.

A source is selected by name, which is what makes ``TRAINING["TASK"]`` a real
setting and what lets ``RunCreateRequest.task_source`` choose a dataset per run
without the caller importing a class. Resolution goes through the settings
object so ``PATHS["DATASETS"]`` is honoured relative to the project root.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..exceptions import ImproperlyConfigured
from .sources import (
    JsonlTaskSource,
    MixedTaskSource,
    SyntheticTaskSource,
    TaskSourceProtocol,
)

__all__ = [
    "register_source",
    "unregister_source",
    "available_sources",
    "build_source",
    "source_names",
]

_SOURCES: dict[str, type] = {}

#: Dataset file per named source (``None`` means "every ``*.jsonl`` in DATASETS").
_DATASET_FILES: dict[str, str | None] = {
    "jsonl": None,
    "truth": "truth.jsonl",
    "ambiguity": "ambiguity.jsonl",
    "interaction": "interaction.jsonl",
}

#: Names accepted for the synthetic source, including the historical setting.
_SYNTHETIC_ALIASES = frozenset({"synthetic", "synthetic_classification"})


def register_source(name: str) -> Callable[[type], type]:
    """Register a source class under ``name``."""

    def decorator(source: type) -> type:
        _SOURCES[name] = source
        return source

    return decorator


def unregister_source(name: str) -> None:
    _SOURCES.pop(name, None)


def available_sources() -> dict[str, type]:
    """A copy of the registry, so callers cannot mutate it."""
    return dict(_SOURCES)


def source_names() -> list[str]:
    return sorted(_SOURCES)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _training(settings: Any) -> dict[str, Any]:
    return dict(settings["TRAINING"])


def _dataset_paths(settings: Any, name: str) -> list[Path]:
    """Resolve the dataset files a named source reads."""
    directory = settings.path("DATASETS")
    filename = _DATASET_FILES.get(name)
    if filename is not None:
        target = directory / filename
        if not target.exists():
            raise ImproperlyConfigured(
                f"task source {name!r} needs {target}, which does not exist; set "
                f"PATHS['DATASETS'] or choose another TRAINING['TASK']"
            )
        return [target]
    if not directory.exists():
        raise ImproperlyConfigured(
            f"task source {name!r} needs a dataset directory at {directory}; "
            f"set PATHS['DATASETS']"
        )
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise ImproperlyConfigured(
            f"task source {name!r} found no *.jsonl files under {directory}"
        )
    return paths


def _jsonl_source(
    settings: Any,
    *,
    name: str,
    paths: list[Path] | None = None,
    overrides: dict[str, Any] | None = None,
) -> JsonlTaskSource:
    training = _training(settings)
    options: dict[str, Any] = {
        "state_dim": int(training.get("STATE_DIM", 16)),
        "action_dim": int(training.get("ACTION_DIM", 4)),
        "batch_size": int(training.get("BATCH_SIZE", 32)),
        "seed": int(settings["SEED"]),
        "shuffle": True,
        "label_strategy": "hash",
    }
    options.update(overrides or {})
    resolved = paths if paths is not None else _dataset_paths(settings, name)
    return JsonlTaskSource(resolved, **options)


def _synthetic(settings: Any, overrides: dict[str, Any]) -> SyntheticTaskSource:
    training = _training(settings)
    options: dict[str, Any] = {
        "state_dim": int(training.get("STATE_DIM", 16)),
        "action_dim": int(training.get("ACTION_DIM", 4)),
        "batch_size": int(training.get("BATCH_SIZE", 32)),
        "seed": int(settings["SEED"]),
    }
    options.update(overrides)
    return SyntheticTaskSource(**options)


def _mixed(settings: Any, overrides: dict[str, Any]) -> MixedTaskSource:
    """Weighted interleave of every shipped dataset plus the synthetic source.

    Weights favour the real corpora but keep the synthetic generator large
    enough that a default run always has a learnable signal even when the
    shipped seed files hold only a couple of records each.
    """
    weights: dict[str, float] = dict(
        overrides.pop("weights", None)
        or {
            "synthetic": 0.25,
            "truth": 0.25,
            "ambiguity": 0.25,
            "interaction": 0.25,
        }
    )
    sources: list[tuple[TaskSourceProtocol, float]] = []
    directory = settings.path("DATASETS")
    for name, weight in weights.items():
        if weight <= 0.0:
            continue
        if name in _SYNTHETIC_ALIASES:
            sources.append((_synthetic(settings, dict(overrides)), float(weight)))
            continue
        filename = _DATASET_FILES.get(name)
        candidate = directory / filename if filename else None
        if candidate is None or not candidate.exists():
            # A missing optional corpus must not break a mixed run.
            continue
        sources.append(
            (
                _jsonl_source(
                    settings, name=name, paths=[candidate], overrides=dict(overrides)
                ),
                float(weight),
            )
        )
    if not sources:
        sources.append((_synthetic(settings, dict(overrides)), 1.0))
    return MixedTaskSource(sources)


_BUILDERS: dict[str, Callable[[Any, dict[str, Any]], TaskSourceProtocol]] = {
    "synthetic": _synthetic,
    "mixed": _mixed,
}
for _name in _DATASET_FILES:
    _BUILDERS.setdefault(
        _name,
        lambda settings, overrides, _n=_name: _jsonl_source(
            settings, name=_n, overrides=overrides
        ),
    )


def build_source(
    name: str, *, settings: Any = None, **overrides: Any
) -> TaskSourceProtocol:
    """Build the named source from settings, with caller overrides.

    ``settings`` defaults to the process-wide settings object. ``overrides`` are
    forwarded to the source constructor, so ``batch_size=8`` works without a
    second settings object.
    """
    if settings is None:
        from ..conf import get_settings

        settings = get_settings()

    key = str(name or "").strip()
    if not key:
        raise ImproperlyConfigured("task source name must be non-empty")
    normalized = "synthetic" if key in _SYNTHETIC_ALIASES else key

    builder = _BUILDERS.get(normalized)
    if builder is None:
        registered = _SOURCES.get(normalized)
        if registered is None:
            raise ImproperlyConfigured(
                f"unknown task source {name!r}; available: {', '.join(sorted(_BUILDERS))}"
            )
        return registered(**overrides)  # type: ignore[call-arg]
    return builder(settings, dict(overrides))


# Register the classes so ``available_sources()`` is a complete inventory.
_SOURCES.setdefault("synthetic", SyntheticTaskSource)
_SOURCES.setdefault("jsonl", JsonlTaskSource)
_SOURCES.setdefault("mixed", MixedTaskSource)
for _alias in _DATASET_FILES:
    _SOURCES.setdefault(_alias, JsonlTaskSource)
_SOURCES.setdefault("synthetic_classification", SyntheticTaskSource)
