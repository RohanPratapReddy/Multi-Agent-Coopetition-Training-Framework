"""The shipped ``conf/`` tree: file names, owners, and load order.

Every entry names the top-level settings blocks the file owns. A file that writes
a block it does not own is refused at load, so "where do I change the batch size"
has one answer and a diff has one meaning.

The order below is the order the loader merges in. It is also the order
``startproject`` writes, so a person reading a fresh project sees the same order
this module declares.

This module imports nothing from :mod:`mactf.conf` at import time. ``conf`` imports
it inside ``load_settings``, and the two helpers it needs back (``_load_yaml`` and
``DEFAULTS``) are imported lazily inside :func:`load_conf_tree`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ImproperlyConfigured

#: Suffix appended to a destination name to get the packaged template's name, so
#: ``trainer.yaml`` is shipped as ``trainer.yaml.tpl``.
TEMPLATE_SUFFIX = ".tpl"

#: Directory under ``mactf/templates`` that holds the master files.
TEMPLATE_SUBDIR = "conf"

#: Directory ``startproject`` creates in a project and the loader reads.
CONF_SUBDIR = "conf"


@dataclass(frozen=True)
class ConfFile:
    """One file in the tree."""

    #: Destination name inside a project's ``conf/``, for example ``trainer.yaml``.
    filename: str
    #: Top-level settings keys this file may write.
    owns: tuple[str, ...]
    #: One sentence for the ``check`` command and the generated header comment.
    purpose: str

    @property
    def template_name(self) -> str:
        """The packaged template's file name."""
        return self.filename + TEMPLATE_SUFFIX


#: The tree, in load order. Keep this list sorted by subsystem, not by size.
CONF_LAYOUT: tuple[ConfFile, ...] = (
    ConfFile(
        "conf.yaml",
        ("PATHS", "DEBUG", "DEVICE", "DTYPE", "AMP", "SEED", "DETERMINISTIC",
         "DEFAULT_AUTO_FIELD"),
        "Where every directory lives, and the process-wide execution knobs.",
    ),
    ConfFile(
        "mactf.yaml",
        ("INSTALLED_APPS", "MIDDLEWARE", "ROOT_URLCONF", "APPEND_SLASH", "HTTP",
         "CONFIG_FILES"),
        "Framework wiring: apps, middleware, and the HTTP transport limits.",
    ),
    ConfFile(
        "components.yaml",
        ("COMPONENTS", "COMPLIANCE"),
        "Which components may be registered, replaced or imported, and the floor they "
        "are held to.",
    ),
    ConfFile(
        "peers.yaml",
        ("PEERS",),
        "The swarm: one entry per peer, resolved from a class name.",
    ),
    ConfFile(
        "trainer.yaml",
        ("TRAINING", "TRAINER"),
        "The training loop: epochs, batch, dimensions, task, bootstrap.",
    ),
    ConfFile(
        "optimizer.yaml",
        ("OPTIMIZER",),
        "Optimizer backend, learning rate, clipping, accumulation, scheduler.",
    ),
    ConfFile(
        "loss.yaml",
        ("LOSS",),
        "The objective's coefficients and its backend.",
    ),
    ConfFile(
        "rl.yaml",
        ("RL",),
        "Per-family RL function selection: advantage, loss, rollup, synthesis and the rest.",
    ),
    ConfFile(
        "scorer.yaml",
        ("SCORING", "SEVERITY", "WARNINGS", "ENGINE"),
        "Scoring: reward and penalty magnitudes, severity tiers, warning policy.",
    ),
    ConfFile(
        "orchestrator.yaml",
        ("COOPETITION", "ORCHESTRATOR"),
        "Aggregation: kappa, weight bounds, signal source, cooperative pool.",
    ),
    ConfFile(
        "sandbox.yaml",
        ("SANDBOX",),
        "Isolation: provider, images, limits, and the fallback policy.",
    ),
    ConfFile(
        "distillation.yaml",
        ("DISTILLATION",),
        "Contrastive distillation toward a teacher set.",
    ),
    ConfFile(
        "logger.yaml",
        ("LOGGER",),
        "Log sinks, levels, rotation, redaction, and the async writer.",
    ),
    ConfFile(
        "telemetry.yaml",
        ("TELEMETRY",),
        "The legacy telemetry block, kept for projects that never adopted LOGGER.",
    ),
    ConfFile(
        "watchdog.yaml",
        ("WATCHDOG",),
        "Health thresholds and the action taken on degradation.",
    ),
    ConfFile(
        "checkpoint.yaml",
        ("CHECKPOINT",),
        "Checkpoint retention, compression, checksum, and included sections.",
    ),
    ConfFile(
        "api.yaml",
        ("API", "CORS"),
        "The service layer: tokens, base path, limits, CORS origins.",
    ),
    ConfFile(
        "db.yaml",
        ("DATABASES",),
        "The registry database. SQLite only.",
    ),
    ConfFile(
        "distributed.yaml",
        ("DISTRIBUTED",),
        "Worker and coordinator endpoints for the distributed transport.",
    ),
)


def layout_by_filename() -> dict[str, ConfFile]:
    """Index the layout by destination name."""
    return {entry.filename: entry for entry in CONF_LAYOUT}


def layout_by_template() -> dict[str, ConfFile]:
    """Index the layout by packaged template name."""
    return {entry.template_name: entry for entry in CONF_LAYOUT}


#: Every key any file may write. Built once so the loader's check is a set lookup.
OWNED_KEYS: frozenset[str] = frozenset(
    key for entry in CONF_LAYOUT for key in entry.owns
)


def owner_of(key: str) -> str | None:
    """The file that owns ``key``, or ``None`` when no file may set it."""
    for entry in CONF_LAYOUT:
        if key in entry.owns:
            return entry.filename
    return None


#: Keys a ``conf/`` file may never write. See document 15 section 3.3.
RESERVED_KEYS: frozenset[str] = frozenset(
    {"CONF_DIR", "SETTINGS_MODULE", "SECRET_KEY", "BASE_DIR"}
)

#: Keys ``mactf.yaml`` owns for documentation only. They are read before ``conf/``
#: is (apps are import-time, the others are wired before a settings load finishes),
#: so a live value in YAML would be a value nothing honours. Refused like a
#: reserved key rather than silently ignored.
PYTHON_ONLY_KEYS: frozenset[str] = frozenset(
    {"INSTALLED_APPS", "MIDDLEWARE", "ROOT_URLCONF", "APPEND_SLASH"}
)


def _get(settings: Any, name: str, default: Any = None) -> Any:
    """Read one key from a settings object, a module, or a plain mapping."""
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    getter = getattr(settings, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(settings, name, default)


def authored_keys(settings: Any, name: str) -> frozenset[str] | None:
    """The keys of block ``name`` that the project *wrote*, as opposed to defaults.

    ``load_settings`` records, under ``_AUTHORED``, every block that the settings
    module, the ``conf/`` tree or a ``CONFIG_FILES`` overlay set, with the keys it
    set. A framework default is not authored. That distinction is what lets a
    legacy project (``TRAINING.LEARNING_RATE`` only) keep its behaviour even though
    ``DEFAULTS`` now carries an ``OPTIMIZER`` block.

    ``None`` means the settings object carries no authorship record (a plain
    mapping built by a caller or a test): the block's own contents are then all
    the evidence there is, and :func:`is_authored` falls back to truthiness.
    """
    record = _get(settings, "_AUTHORED")
    if not isinstance(record, Mapping):
        return None
    keys = record.get(name)
    if keys is None:
        return frozenset()
    return frozenset(str(key) for key in keys)


def is_authored(settings: Any, name: str) -> bool:
    """Whether the project wrote block ``name`` (see :func:`authored_keys`)."""
    record = _get(settings, "_AUTHORED")
    if isinstance(record, Mapping):
        return name in record
    return bool(_get(settings, name))


def resolve_conf_dir(settings: Any, root: Path) -> Path | None:
    """The project's ``conf/`` directory, or ``None`` when the project has none.

    Precedence, first hit wins:

    1. ``settings.CONF_DIR``, resolved against ``root`` when relative.
    2. ``<root>/conf``, when it exists and holds at least one ``*.yaml``.

    A project without the directory is not an error: it is a project using
    ``settings.py`` the way it always did, and every consumer falls back to the
    legacy block. An explicit ``CONF_DIR`` that does not exist is a broken
    deployment, not a project that opted out, and raises.
    """
    declared = _get(settings, "CONF_DIR")
    if declared is not None:
        candidate = Path(str(declared)).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise ImproperlyConfigured(
                f"CONF_DIR points at {candidate}, which is not a directory"
            )
        return candidate

    implicit = (root / CONF_SUBDIR).resolve()
    if implicit.is_dir() and any(implicit.glob("*.yaml")):
        return implicit
    return None


def load_conf_tree(
    settings: Any,
    root: Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Read every file in ``CONF_LAYOUT`` that exists, merging in order.

    Returns ``(values, notes)``. ``values`` is a flat mapping keyed by top-level
    settings name; ``notes`` is one line per decision a reader would want in a
    startup log: the directory used, a file skipped, a key overridden.

    ``strict`` refuses a file that writes a key it does not own. It defaults to
    ``True`` because the refusal is the point of the ownership table; a project
    migrating by hand passes ``strict=False`` and gets a note instead.

    The merge order is ``CONF_LAYOUT``, never the filesystem's. The glob is used
    only to answer "is there a file the layout does not know about".
    """
    from .conf import DEFAULTS, _deep_merge, _load_yaml

    directory = resolve_conf_dir(settings, root)
    if directory is None:
        return {}, ["no conf/ directory; settings.py is the configuration"]

    by_filename = layout_by_filename()
    present = {path.name for path in directory.glob("*.yaml")}
    notes: list[str] = [f"conf/ resolved to {directory}"]

    unknown = sorted(present - set(by_filename))
    if unknown:
        # A file the loader will never read is a trap: the author believes it is
        # live. Refused rather than warned, because a warning scrolls away.
        raise ImproperlyConfigured(
            f"conf/ contains file(s) no loader reads: {', '.join(unknown)}; "
            "remove them or add them to mactf.conf_files.CONF_LAYOUT"
        )

    merged: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for entry in CONF_LAYOUT:
        path = directory / entry.filename
        if not path.is_file():
            notes.append(f"{entry.filename}: absent, defaults apply")
            continue

        payload = _load_yaml(path)
        for key in payload:
            if key in RESERVED_KEYS or key in PYTHON_ONLY_KEYS:
                raise ImproperlyConfigured(
                    f"{entry.filename} sets {key}, which only settings.py may set"
                )
        problems = _ownership_problems(payload, entry)
        if problems:
            if strict:
                raise ImproperlyConfigured("; ".join(problems))
            notes.extend(f"{problem} (accepted: strict=False)" for problem in problems)

        for key, value in payload.items():
            if value is None and isinstance(DEFAULTS.get(key), dict):
                raise ImproperlyConfigured(f"{entry.filename}: {key} must be a mapping")
            if key in owners and owners[key] != entry.filename:
                notes.append(
                    f"{entry.filename} overrides {key} already set by {owners[key]}"
                )
            owners[key] = entry.filename
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                _deep_merge(merged[key], value)
            else:
                merged[key] = _copy(value)

    _resolve_relative_root(merged, root)
    return merged, notes


def _copy(value: Any) -> Any:
    import copy

    return copy.deepcopy(value)


def _resolve_relative_root(merged: dict[str, Any], root: Path) -> None:
    """Make a relative ``PATHS.ROOT`` mean "relative to the project directory".

    ``conf.yaml`` ships ``ROOT: "."``. Merged verbatim it would overwrite the real
    project root that ``load_settings`` computed from ``BASE_DIR``, and every
    ``Settings.path`` call would then resolve against the process's cwd.
    """
    paths = merged.get("PATHS")
    if not isinstance(paths, dict) or "ROOT" not in paths:
        return
    declared = paths["ROOT"]
    if declared is None:
        paths["ROOT"] = str(root)
        return
    candidate = Path(str(declared)).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    paths["ROOT"] = str(candidate.resolve())


def _ownership_problems(payload: Mapping[str, Any], entry: ConfFile) -> list[str]:
    """Keys ``entry`` wrote that it does not own."""
    problems: list[str] = []
    for key in payload:
        if key in RESERVED_KEYS or key in PYTHON_ONLY_KEYS or key in entry.owns:
            continue
        owner = owner_of(str(key))
        if owner:
            problems.append(
                f"{entry.filename} sets {key}, which is owned by {owner}; "
                "move the block or rename the file"
            )
        else:
            problems.append(
                f"{entry.filename} sets unknown block {key}; known blocks are "
                + ", ".join(sorted(OWNED_KEYS))
            )
    return problems
