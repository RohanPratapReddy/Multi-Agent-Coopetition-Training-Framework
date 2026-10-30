---
title: Project `conf/` Layout and `startproject` Copy
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0
owner: mactf-core
last_reviewed: 2026-10-21
---

# 15. Project `conf/` Layout and `startproject` Copy

Today a generated project keeps its configuration in `settings.py` as Python
dicts, plus three YAML overlays under `configs/`. That means a deployment change
is a code change: an operator who wants eight epochs edits a `.py` file, and a
reviewer who wants to see what differs between staging and production diffs
Python.

This document moves every subsystem argument into a `conf/` directory of YAML
files that `mactf startproject` writes, and specifies the loader, the validating
blocks, and the consumer bindings that make those files real. `settings.py` keeps
identity only: `BASE_DIR`, `SECRET_KEY`, `CONF_DIR`, and `CONFIG_FILES`.

It is additive. A project with no `conf/` directory behaves exactly as it does
today — the three `configs/*.yaml` overlays keep working, `settings.py` keeps its
dicts, and nothing in the load path changes shape. The equivalence gate in
section 14 proves it.

Three decisions were made before writing, and they constrain everything below:

| Decision | Choice | Consequence |
|---|---|---|
| Authority | **Hybrid** | `settings.py` owns identity, apps, and peers. All subsystem arguments live in YAML. `conf/` wins over `settings.py` for any block it declares. |
| Destination | **`<project>/conf/`** | The framework's package name never appears in a project's import path. `settings.CONF_DIR = BASE_DIR / "conf"`. |
| Master files | **`mactf/templates/conf/`** | `mactf/conf.py` stays a module; the shipped YAMLs are package data copied verbatim by `startproject`. |
| Wiring depth | **Full** | New `DEFAULTS` blocks, `from_settings` factories, real consumers in the orchestrator and trainer, validation, and tests. Nothing is documented-but-unused. |

## 1. Scope, non-goals, and what this replaces

### 1.1 What exists today

| Thing | Where | Behaviour |
|---|---|---|
| Framework defaults | `mactf/conf.py:28` (`DEFAULTS`) | One dict, 30+ top-level keys |
| Project settings | `<project>/settings.py` | Upper-case module attributes deep-merged over `DEFAULTS` |
| Overlays | `settings.CONFIG_FILES` | YAML files, deep-merged in order |
| Env overrides | `mactf/conf.py:368` (`_apply_env`) | `MACTF_<KEY>` for scalar top-level keys only |
| Validation | `mactf/conf.py:378` (`validate_settings`) | Structural checks, raises `ImproperlyConfigured` |
| Scaffolder | `mactf/management/commands/startproject.py` | 7 templates plus 3 YAML overlays |

The resolution order in `mactf/conf.py:1-8` is the contract this document
extends, not replaces:

1. `DEFAULTS`
2. the settings module named by `MACTF_SETTINGS_MODULE`
3. `CONFIG_FILES` overlays, in order
4. `MACTF_<SETTING>` environment variables

### 1.2 What is wrong with it

| # | Problem | Consequence |
|---|---|---|
| P1 | A deployment change is a Python edit | Operators edit `.py`; no schema; no diffable artifact |
| P2 | One `settings.py` holds every subsystem | No file ownership; two people editing scoring and logging conflict |
| P3 | The three overlays have no owner table | `configs/scoring.yaml` can set `TRAINING` and nothing notices |
| P4 | `DEFAULTS` has no `TRAINER`/`OPTIMIZER`/`LOSS`/`LOGGER`/`WATCHDOG`/`CHECKPOINT` block | Those are constructor arguments only; a YAML file cannot express them |
| P5 | `MACTFTrainer` reads `TRAINING` for loss coefficients | `LOSS` and `TRAINING` both own `KL_COEF`; no stated winner |
| P6 | `MACTFLogger` reads `TELEMETRY` | A project cannot name multiple sinks, rotation, or redaction from YAML |
| P7 | `MACTFCheckpoint` takes 11 constructor arguments and reads no settings | `CHECKPOINT.KEEP` cannot be expressed |
| P8 | `WatchdogPolicy` has 20 fields and reads no settings | `WATCHDOG.MAX_GRAD_NORM` cannot be expressed |
| P9 | `startproject` writes `configs/` | The directory name says "overlays", not "configuration" |

### 1.3 What this document adds

1. A `conf/` directory of 17 YAML files with a **disjoint ownership table**
   (section 3): each top-level settings key has exactly one owning file.
2. A loader, `mactf/conf_files.py` (section 4), that discovers `CONF_DIR`, merges
   the files in a deterministic order, and refuses a file that writes a key it
   does not own.
3. Eight new `DEFAULTS` blocks — `TRAINER`, `OPTIMIZER`, `LOSS`, `LOGGER`,
   `ORCHESTRATOR`, `WATCHDOG`, `CHECKPOINT`, `PATHS` additions — with validators
   (section 5).
4. `from_settings` factories on four classes that currently require constructor
   arguments (section 6), and the consumer call sites that use them.
5. `startproject` changes: write `conf/`, stop writing `configs/`, shrink
   `settings.py` (section 7).
6. Tests and packaging changes (sections 8 and 9).

### 1.4 Non-goals, stated as refusals

1. **No change to the resolution order.** `conf/` is loaded as step 2.5, between
   the settings module and `CONFIG_FILES`, because it *is* the project's
   configuration and an overlay is an exception to it. Section 4.2 states why.
2. **No nested env overrides.** `MACTF_DISTILLATION_TAU` still does nothing.
   `_apply_env` (`conf.py:368`) walks scalars only, and this document does not
   change it. A per-run override passes the constructor argument.
3. **No YAML for `PEERS`, `INSTALLED_APPS`, or `DATABASES`.** Peers reference
   Python classes by dotted path and apps are import-time; they stay in
   `settings.py`. Section 3.4 explains.
4. **No schema language.** No JSON Schema, no Pydantic model for the files.
   Validation is Python in `mactf/conf.py`, where the numeric bounds already live.
5. **No deletion of `CONFIG_FILES`.** An overlay outside the shipped tree (a CI
   override, a one-off experiment) is a real need, and it merges after `conf/`.
6. **No migration of existing projects.** A project with `settings.py` dicts and
   `configs/*.yaml` keeps working, unchanged. Section 14.1 is the gate.
7. **No change to `DEFAULTS` values.** The blocks move; the numbers do not.
   Section 14.2 asserts every leaf value.

## 2. Module layout

```
mactf/
  conf.py                       # unchanged module; gains DEFAULTS blocks + validators
  conf_files.py                 # NEW: CONF_LAYOUT, resolve_conf_dir, load_conf_tree
  templates/
    project/
      manage.py.tpl
      package_init.py.tpl
      settings.py.tpl           # shrinks to identity (section 7.3)
      peers/__init__.py.tpl
    conf/                       # NEW: the 17 master files
      conf.yaml.tpl
      mactf.yaml.tpl
      trainer.yaml.tpl
      optimizer.yaml.tpl
      loss.yaml.tpl
      scorer.yaml.tpl
      orchestrator.yaml.tpl
      logger.yaml.tpl
      telemetry.yaml.tpl
      watchdog.yaml.tpl
      checkpoint.yaml.tpl
      sandbox.yaml.tpl
      distillation.yaml.tpl
      api.yaml.tpl
      db.yaml.tpl
      peers.yaml.tpl
      distributed.yaml.tpl
```

`mactf/templates/project/configs/` is **deleted**. Its three files are superseded
by `conf/sandbox.yaml` and `conf/scorer.yaml` (which absorbs
`configs/scoring.yaml`); `configs/swarm.yaml`'s contents are split between
`conf/trainer.yaml` and `conf/orchestrator.yaml` per the ownership table.

Why `mactf/templates/conf/` and not `mactf/conf/`: `mactf/conf.py` exists, so a
`mactf/conf/` package cannot coexist with it. Converting `conf.py` into a package
would move the runtime settings module and make the template directory a Python
package at once — two changes for one problem. The template package is already
the established place for shipped text (`mactf/templates/project`), so the master
files go beside it.

Why `conf_files.py` and not code inside `conf.py`: `conf.py` is imported by every
module in the framework, including `mactf.settings_default`, which must work from
a bare wheel with no project. Keeping YAML discovery in a separate module means
`conf.py` gains no `importlib.resources` call and no filesystem walk at import
time, and the loader can be tested against a directory without touching global
state.

## 3. The file ownership table

### 3.1 The rule

Every top-level key in `DEFAULTS` that is a mapping and is not in the "reserved"
set below has **exactly one** owning file in `conf/`. A file that writes a key it
does not own is refused at load with both names in the message:

```
ImproperlyConfigured: conf/logger.yaml sets TRAINING, which is owned by
conf/trainer.yaml; move the block or rename the file
```

The rule exists because the alternative — "any file may set anything" — makes
`conf/` a directory of suggestions. With ownership, the answer to "where do I
change the batch size" is a file name, and a reviewer reading a diff knows which
subsystem moved.

### 3.2 The table

| File | Owns | Never touches |
|---|---|---|
| `conf.yaml` | `PATHS`, `DEBUG`, `DEVICE`, `DTYPE`, `AMP`, `SEED`, `DETERMINISTIC`, `DEFAULT_AUTO_FIELD` | any subsystem block |
| `mactf.yaml` | `INSTALLED_APPS`, `MIDDLEWARE`, `ROOT_URLCONF`, `APPEND_SLASH`, `HTTP`, `CONFIG_FILES` | subsystem arguments |
| `trainer.yaml` | `TRAINING`, `TRAINER` | `OPTIMIZER`, `LOSS`, `SCORER` |
| `optimizer.yaml` | `OPTIMIZER` | `TRAINING` |
| `loss.yaml` | `LOSS` | `TRAINING` |
| `scorer.yaml` | `SCORING`, `SEVERITY`, `WARNINGS`, `ENGINE` | `COOPETITION` |
| `orchestrator.yaml` | `COOPETITION`, `ORCHESTRATOR` | `SCORING` |
| `logger.yaml` | `LOGGER` | `TELEMETRY` |
| `telemetry.yaml` | `TELEMETRY` | `LOGGER` |
| `watchdog.yaml` | `WATCHDOG` | `CHECKPOINT` |
| `checkpoint.yaml` | `CHECKPOINT` | `WATCHDOG` |
| `sandbox.yaml` | `SANDBOX` | — |
| `distillation.yaml` | `DISTILLATION` | `LOSS` |
| `api.yaml` | `API`, `CORS` | `HTTP` |
| `db.yaml` | `DATABASES` | `PATHS` |
| `peers.yaml` | `PEERS` | `TRAINING` |
| `distributed.yaml` | `DISTRIBUTED` | `API` |

### 3.3 The reserved keys

These are loadable from `settings.py` only, and a `conf/` file that sets one is
refused. The reason is per key:

| Key | Why not YAML |
|---|---|
| `CONF_DIR` | It names the directory being read; a file inside it cannot define it. Refused explicitly rather than ignored. |
| `SETTINGS_MODULE` | Same: identity of the loader, not configuration. |
| `SECRET_KEY` | A secret in a committed file is a leak waiting for a `git add`. It stays in `settings.py` where environment interpolation is already the norm. |
| `BASE_DIR` | Computed from `__file__`. |
| `INSTALLED_APPS` (in `mactf.yaml`) | Apps are import-time. Moving imports behind a YAML read would make `populate()` order-dependent on file I/O. Named in the table as owned by `mactf.yaml` for documentation, but `mactf.yaml` is copied as a *reference* file and `settings.py` remains the source. Section 3.4. |

`INSTALLED_APPS` is the one entry in the table that is documentation: the file
records the apps for a reader, and `settings.py` still wins because apps are
merged before `conf/` and `conf/` does not declare the key. A `conf/` file that
tries to set it gets the reserved-key refusal.

### 3.4 Why `PEERS` is in YAML but `INSTALLED_APPS` is not

`PEERS` is a list of dicts whose `class` field is a dotted path resolved lazily by
`mactf/apps.py` at `populate()`, not at import. A YAML list of the same dicts is
therefore expressible, and `peers.yaml` owns it.

`INSTALLED_APPS` is read by `populate()` to *perform imports*. A YAML file that
changed it would mean the framework reads configuration to decide which
configuration-reading code exists. That circularity is why it stays in Python,
and why `mactf.yaml` carries it as a comment block rather than a live key.

## 4. The loader: `mactf/conf_files.py`

### 4.1 `CONF_LAYOUT`

One tuple of `ConfFile` records, in load order. Load order is the ownership table
read top to bottom, and it matters for exactly one case: a file may `$ref` a
value another file declares, and forward references are not resolved.

```python
"""The shipped ``conf/`` tree: file names, owners, and load order.

Every entry names the top-level settings blocks the file owns. A file that writes
a block it does not own is refused at load, so "where do I change the batch size"
has one answer and a diff has one meaning.

The order below is the order the loader merges in. It is also the order
``startproject`` writes, so a person reading a fresh project sees the same order
this module declares.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Suffix the packaged templates carry; stripped for the destination name.
TEMPLATE_SUFFIX = ".yaml.tpl"

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
        "peers.yaml",
        ("PEERS",),
        "The swarm: one entry per peer, resolved from a dotted class path.",
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
```

`OWNED_KEYS` and `owner_of` exist so the loader reports *which* file owns a
stray key rather than only that the key is stray. The message in section 3.1 is
built from `owner_of`.

### 4.2 `resolve_conf_dir`

```python
#: Keys a ``conf/`` file may never write. See section 3.3.
RESERVED_KEYS: frozenset[str] = frozenset(
    {"CONF_DIR", "SETTINGS_MODULE", "SECRET_KEY", "BASE_DIR"}
)


def resolve_conf_dir(settings: Any, root: Path) -> Path | None:
    """The project's ``conf/`` directory, or ``None`` when the project has none.

    Precedence, first hit wins:

    1. ``settings.CONF_DIR``, resolved against ``root`` when relative.
    2. ``<root>/conf``, when it exists and holds at least one ``*.yaml``.

    A project without the directory is not an error: it is a project using
    ``settings.py`` the way it always did, and every consumer in section 6 falls
    back to the legacy block.
    """
    declared = _get(settings, "CONF_DIR")
    if declared is not None:
        candidate = Path(str(declared)).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            # An explicit CONF_DIR that does not exist is a broken deployment,
            # not a project that opted out.
            raise ImproperlyConfigured(
                f"CONF_DIR points at {candidate}, which is not a directory"
            )
        return candidate

    implicit = (root / CONF_SUBDIR).resolve()
    if implicit.is_dir() and any(implicit.glob("*.yaml")):
        return implicit
    return None
```

The asymmetry is deliberate: an **absent** `CONF_DIR` falls back to
`<root>/conf`, and an **explicit but broken** `CONF_DIR` raises. A project that
typed the path has said what it means; a project that typed nothing gets the
convention.

`_get(settings, name)` reads a settings object or a mapping, matching the helper
already used in `mactf/training/logging.py` and `checkpoint_v2.py`:

```python
def _get(settings: Any, name: str, default: Any = None) -> Any:
    """Read one key from a settings object or a plain mapping."""
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    getter = getattr(settings, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(settings, name, default)
```

### 4.3 `load_conf_tree`

```python
def load_conf_tree(
    settings: Any,
    root: Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Read every file in ``CONF_LAYOUT`` that exists, merging in order.

    Returns ``(values, notes)``. ``values`` is a flat mapping keyed by top-level
    settings name; ``notes`` is one line per decision a reader would want in a
    startup log — the directory used, a file skipped, a key overridden.

    ``strict`` refuses a file that writes a key it does not own. It defaults to
    ``True`` because the refusal is the point of the ownership table; a project
    migrating by hand passes ``strict=False`` and gets a note instead.
    """
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
    for entry in CONF_LAYOUT:
        path = directory / entry.filename
        if not path.is_file():
            notes.append(f"{entry.filename}: absent, defaults apply")
            continue

        payload = _load_yaml(path)
        problems = _ownership_problems(payload, entry, strict=strict)
        if problems:
            raise ImproperlyConfigured("; ".join(problems))

        for key, value in payload.items():
            if key in RESERVED_KEYS:
                raise ImproperlyConfigured(
                    f"{entry.filename} sets {key}, which only settings.py may set"
                )
            if key in merged and merged[key] != value:
                notes.append(
                    f"{entry.filename} overrides {key} already set by "
                    f"{owner_of(key) or 'an earlier file'}"
                )
            merged[key] = value

    return merged, notes


def _ownership_problems(
    payload: Mapping[str, Any], entry: ConfFile, *, strict: bool
) -> list[str]:
    """Keys ``entry`` wrote that it does not own."""
    problems: list[str] = []
    for key in payload:
        if key in RESERVED_KEYS or key in entry.owns:
            continue
        owner = owner_of(key)
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
```

`_load_yaml` is reused from `mactf/conf.py:285` rather than reimplemented: it
already raises `ImproperlyConfigured` for a missing PyYAML, a non-mapping
document, and a malformed file.

### 4.4 The insertion point in `load_settings`

`mactf/conf.py:544` today:

```python
    root = Path(getattr(module, "BASE_DIR", os.getcwd())).expanduser().resolve()
    values["PATHS"]["ROOT"] = str(root)

    for key, value in vars(module).items():
        if key.isupper() and key in DEFAULTS:
            ...
```

It gains one step, between the module merge and the `CONFIG_FILES` merge:

```python
    root = Path(getattr(module, "BASE_DIR", os.getcwd())).expanduser().resolve()
    values["PATHS"]["ROOT"] = str(root)

    for key, value in vars(module).items():
        if key.isupper() and key in DEFAULTS:
            if isinstance(value, dict) and isinstance(values[key], dict):
                _deep_merge(values[key], value)
            else:
                values[key] = copy.deepcopy(value)

    # Step 2.5: the project's conf/ tree. It sits below CONFIG_FILES because an
    # overlay is an exception to the project's configuration, not a peer of it:
    # a CI run that needs six epochs passes an overlay, and the committed tree
    # must not silently win over it.
    from .conf_files import load_conf_tree

    conf_values, conf_notes = load_conf_tree(module, root)
    for key, value in conf_values.items():
        if isinstance(value, dict) and isinstance(values.get(key), dict):
            _deep_merge(values[key], value)
        else:
            values[key] = copy.deepcopy(value)
    values["_CONF_NOTES"] = conf_notes

    config_files = []
    ...
```

Two details matter:

- **`_CONF_NOTES`** is an underscore key so it cannot collide with a `DEFAULTS`
  entry, and `Settings.as_dict()` carries it. `check` prints it; a support bundle
  includes it. It is the answer to "which file did this value come from".
- **The merge is `_deep_merge`, not assignment.** `conf/trainer.yaml` that sets
  only `EPOCHS: 8` keeps `DEFAULTS["TRAINING"]`'s other keys. A file is a patch,
  which is what makes a 17-file tree readable instead of 17 copies of `DEFAULTS`.

`load_settings` keeps its signature and its validation call. `_CONF_NOTES` is
stripped before `validate_settings` runs, so a validator never sees it.

### 4.5 Determinism

`glob("*.yaml")` returns filesystem order, which differs between machines. The
loader never iterates it for merging: it iterates `CONF_LAYOUT`, which is a tuple,
and uses the glob only to answer "is there a file the layout does not know about".
The merge order is therefore a property of the module, not of the disk.

## 5. The new `DEFAULTS` blocks

Eight blocks are added to `DEFAULTS` (`mactf/conf.py:28`). Each is placed beside
the block it is most similar to, not appended at the end, so a reader scanning
`conf.py` sees `TRAINER` next to `TRAINING` and `CHECKPOINT` next to `SANDBOX`.

Every default below is a value that reproduces today's behaviour exactly when the
key is absent. That is the equivalence gate's precondition.

### 5.1 `PATHS` additions

`PATHS` (`conf.py:89`) gains five entries. The existing seven are unchanged; the
order in the block is the order `ensure_path` is called in a typical run.

```python
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
```

| Key | Default | Written by | Why a separate directory |
|---|---|---|---|
| `RUNS` | `var/runs` | `MACTFLogger`'s run directory | A run's logs are per-run; `LOGS` holds the process log |
| `TEACHERS` | `var/teachers` | `trainer.yaml`'s `TEACHERS[].path` | A teacher weights file is an input, not a checkpoint |
| `AUDITS` | `var/audits` | `distillation.yaml`'s `AUDIT_ARTIFACTS` | Audits are produced by an eval suite, not by training |
| `CACHE` | `var/cache` | `mactf.distill`'s view cache | Derived data; safe to delete |
| `STATE` | `var/state` | The watchdog's history, the registry's WAL | Survives a checkpoint prune |

`PATHS` stays a flat mapping of strings. A nested mapping would break
`Settings.path` (`conf.py:336`), which resolves one key against `ROOT` and must
keep doing so.

### 5.2 `TRAINER`

The block that `TRAINING` should have been. It is **additive**: `TRAINING` keeps
every key it has today, for the reason in section 5.3.

```python
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
```

`TRAINER` is bound by `MACTFTrainer.build` (`trainer.py:527`) through
`_trainer_settings`, which reads `TRAINER` when present and falls back to
`TRAINING` otherwise. Section 6.2 gives the method.

**The two blocks are reconciled once, at load, not per consumer.** After
`validate_settings` passes, `load_settings` folds `TRAINER` into `TRAINING` so the
20 call sites that read `settings["TRAINING"]` keep working without a fallback
branch:

```python
def _fold_trainer(values: dict[str, Any]) -> None:
    """``TRAINER`` is the new spelling of ``TRAINING``; keep one live copy.

    Folding at load means a consumer reads ``TRAINING`` and gets the project's
    ``TRAINER`` values, without every consumer growing a fallback branch. The
    ``TRAINER`` block stays in ``as_dict()`` so a support bundle shows what the
    project actually wrote.
    """
    trainer = values.get("TRAINER") or {}
    if not trainer:
        return
    mapping = _TRAINER_TO_TRAINING.get
    for key, value in trainer.items():
        target = mapping(key)
        if target is None:
            continue
        values["TRAINING"][target] = copy.deepcopy(value)
```

`_TRAINER_TO_TRAINING` is the identity for the seven shared keys (`EPOCHS`,
`BATCH_SIZE`, `STEPS_PER_EPOCH`, `TASK`, `CHECKPOINT_EVERY`, `MAX_GRAD_NORM`,
`GRAD_ACCUMULATION`) and `None` for the six that have no `TRAINING` counterpart
(`HIERARCHY`, `PEER_STRATEGY`, `WEIGHT_UPDATE`, `SHARING`, `LOG_EVERY_STEPS`,
`WATCHDOG_EVERY_EPOCHS`). Those are read from `TRAINER` directly, by
`MACTFTrainer.build`, which is where the tree is built.

So the rule is: **`TRAINING` is the resolved runtime block; `TRAINER` is the
authored one.** A consumer never reads both.

### 5.3 `TRAINING` is not removed

`TRAINING` (`conf.py:214`) keeps all 30 keys. Three reasons, in order of weight:

1. `MACTFTrainer._training_settings` (`trainer.py:505`) reads it in 14 places, and
   `Orchestrator.prepare` (`orchestrator.py:103`) reads it in 6. Renaming a block
   consumed by 20 call sites is a separate change with its own gate.
2. `LOSS` needs `KL_COEF`/`VALUE_COEF`/`ENTROPY_COEF`/`CLIP_EPSILON` to move out
   of `TRAINING`, and that move is *not* behaviour-preserving unless both are read
   with a stated precedence. `LOSS` wins, per section 5.5.
3. A project's existing `configs/swarm.yaml` sets `TRAINING`. Deleting the block
   breaks every such project on upgrade, which is exactly the migration non-goal.

So `TRAINING` becomes the legacy spelling, `TRAINER` the new one, and the
reconciliation is the fold in 5.2. `MACTFTrainer._training_settings`
(`trainer.py:505`) is unchanged: it reads `TRAINING`, which the fold has already
made authoritative.

### 5.4 `OPTIMIZER`

```python
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
```

`from_spec` (`optimizer.py:1101`) already accepts a mapping whose `type` selects
the backend and whose remaining keys go to the backend constructor. `OPTIMIZER`
is that mapping with uppercase keys, so the adapter is a key-lowering shim and
nothing else:

```python
def optimizer_spec(block: Mapping[str, Any]) -> dict[str, Any]:
    """``OPTIMIZER`` (upper case, YAML) to ``from_spec``'s lower-case payload."""
    payload = {key.lower(): value for key, value in dict(block).items()}
    payload.setdefault("type", payload.pop("backend", "torch"))
    payload["class"] = payload.pop("class", "Adam")
    # BETA0/BETA1 are how a flat YAML expresses torch's tuple; Adam wants one key.
    beta0, beta1 = payload.pop("beta0", None), payload.pop("beta1", None)
    betas = payload.pop("betas", None)
    if betas is None and beta0 is not None and beta1 is not None:
        betas = [beta0, beta1]
    if betas is not None and payload["type"] == "torch":
        payload["betas"] = tuple(float(item) for item in betas)
    return payload
```

`scheduler_kwargs` and `scheduler` are popped before the payload reaches
`from_spec`, because a torch optimizer rejects an unknown keyword; the trainer
binds the scheduler separately at `optimizer.py:894`.

### 5.5 `LOSS`

```python
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
```

These are `LossConfig`'s six fields (`mactf/rl/loss.py:53`) plus a backend
selector and a pass-through for a registry function.

**Precedence.** `MACTFTrainer._loss_config` (`trainer.py:992`) reads `TRAINING`
today. It becomes:

```python
    def _loss_config(self) -> LossConfig:
        loss = dict(self.settings.get("LOSS", {}) or {})
        training = self._trainer_settings()
        source = loss or training
        return LossConfig(
            value_coef=float(source.get("VALUE_COEF", 0.5)),
            kl_coef=float(source.get("KL_COEF", 0.01)),
            entropy_coef=float(source.get("ENTROPY_COEF", 0.0)),
            clip_epsilon=float(source.get("CLIP_EPSILON", 0.2)),
            compliance_coef=float(source.get("COMPLIANCE_COEF", 1.0)),
            advantage_normalization=bool(source.get("ADVANTAGE_NORMALIZATION", True)),
        )
```

`LOSS` wins **as a block**, not per key. A project that sets `LOSS.KL_COEF` and
leaves `LOSS.VALUE_COEF` at its default gets the default, not
`TRAINING.VALUE_COEF`. Per-key merging of two blocks that both claim the same
name is how a configuration becomes unexplainable; block-level precedence is one
rule a reader can hold.

### 5.6 `LOGGER`

```python
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
    # Empty means DEFAULT_REDACT (logging.py:53). Naming fragments replaces it,
    # which is why the shipped file lists the seven defaults explicitly.
    "REDACT": [],
    "SCHEMA_VERSION": 1,
    "TENSORBOARD_DIR": None,
    "WANDB_PROJECT": None,
}
```

`SINKS` is the one ordering that matters: `["console", "file"]` writes to both,
in that order, on every record. `MACTFLogger._default_sinks` (`logging.py:545`)
already builds console, file, and jsonl from `TELEMETRY`; the `LOGGER` path
builds the same three plus the two optional sinks the constructor accepts.

| Key | Maps to | Consumer |
|---|---|---|
| `LEVEL` | `MACTFLogger(level=...)` | `logging.py:487` |
| `SINKS` | `MACTFLogger(sinks=...)` | Dispatch in section 6.3 |
| `JSON` | `ConsoleSink(json_format=...)`, `FileSink(json_format=...)` | `logging.py:255`, `:276` |
| `LOG_FILE` | `FileSink(path)` | `logging.py:276` |
| `JSONL_FILE` | `JsonlSink(path)` | `logging.py:346` |
| `ROTATE_BYTES` / `ROTATE_BACKUPS` | `FileSink(max_bytes=, backup_count=)` | `logging.py:281` |
| `BUFFER_SIZE` | `MACTFLogger(buffer_size=)` | `logging.py:493` |
| `ASYNC_WRITE` / `QUEUE_SIZE` | `MACTFLogger(async_write=, queue_size=)` | `logging.py:495` |
| `REDACT` | `MACTFLogger(redact=)` | `logging.py:494` |
| `TENSORBOARD_DIR` | `TensorBoardSink(log_dir=)` | `logging.py:414` |
| `WANDB_PROJECT` | `WandBSink(project=)` | `logging.py:450` |

### 5.7 `TELEMETRY` is not removed

`TELEMETRY` (`conf.py:246`) stays with its four keys. `MACTFLogger._default_sinks`
and `_telemetry_settings` read it, and so does the console formatter's level
lookup. `telemetry.yaml` ships so a project can set it, and `LOGGER` wins when both
are declared:

```python
def _logger_config(settings: Any) -> dict[str, Any]:
    """``LOGGER`` when declared, else ``TELEMETRY`` widened to LOGGER's shape."""
    block = dict(_get(settings, "LOGGER", {}) or {})
    if block:
        return block
    telemetry = dict(_get(settings, "TELEMETRY", {}) or {})
    return {
        "LEVEL": telemetry.get("LEVEL", "INFO"),
        "JSON": telemetry.get("JSON", False),
        "SINKS": ["console", "file"] if telemetry.get("CONSOLE", True) else ["file"],
        "LOG_FILE": telemetry.get("LOG_FILE"),
        "ASYNC_WRITE": True,
        "BUFFER_SIZE": 1000,
    }
```

### 5.8 `WATCHDOG`

`WatchdogPolicy` (`watchdog.py:92`) has 20 fields; `MACTFWatchdog.__init__`
(`watchdog.py:826`) adds four more. The block is the union, in the class's field
order, then the constructor's:

```python
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
```

`WatchdogPolicy` is a frozen dataclass, so the binding is a field-subset
constructor:
`WatchdogPolicy(**{k.lower(): v for k, v in block.items() if k.lower() in field_names})`.
Keys the policy does not declare go to `MACTFWatchdog`'s own arguments. A key in
neither list is a `ValidationError` from the validator in section 5.11, not a
silent drop.

### 5.9 `CHECKPOINT`

`MACTFCheckpoint.__init__` (`checkpoint_v2.py:395`) takes 11 arguments after
`directory` and `run_id`:

```python
CHECKPOINT_DEFAULTS: dict[str, Any] = {
    "DIR": None,               # None resolves to PATHS["CHECKPOINTS"]
    "KEEP": 5,
    "KEEP_BEST": 1,
    "COMPRESS": True,
    # "sha256" or "blake2b"; checkpoint_v2.py:417 accepts exactly these two.
    "CHECKSUM": "sha256",
    "ASYNC_WRITE": False,
    "INCLUDE_OPTIMIZER": True,
    "INCLUDE_RNG": True,
    "INCLUDE_DATA_POSITION": True,
    "INCLUDE_BUFFERS": True,
    "INCLUDE_ADAPTERS": True,
}
```

`DIR` defaults to `None` rather than `"var/checkpoints"` because the string would
duplicate `PATHS["CHECKPOINTS"]` and the two could drift. `None` means "ask
`PATHS`", which is one owner for one path.

### 5.10 `ORCHESTRATOR` and `CORS`

`ORCHESTRATOR` is the run-level block — the decisions that belong to neither the
trainer nor the aggregator:

```python
ORCHESTRATOR_DEFAULTS: dict[str, Any] = {
    # Which task source the run uses. None defers to TRAINER.TASK.
    "TASK_SOURCE": None,
    # Stop after this many epochs. None defers to TRAINER.EPOCHS.
    "MAX_EPOCHS": None,
    # Where a resumed run looks when --resume names no file.
    "RESUME_FROM": None,
    # Print the startup notes (sandbox choice, path resolution, conf/ notes).
    "ANNOUNCE": True,
    # Fail the run when the watchdog asks for a pause, rather than honouring it.
    "PAUSE_IS_FATAL": False,
}

CORS_DEFAULTS: dict[str, Any] = {
    "ORIGINS": [],
    "ALLOW_CREDENTIALS": False,
    "ALLOW_HEADERS": ["authorization", "content-type", "x-request-id"],
    "MAX_AGE": 600,
}
```

`API.CORS_ORIGINS` (`conf.py:56`) already exists and stays. `CORS` is the richer
block; `api.yaml` sets one or the other, and the validator refuses both in the
same file (section 5.11).

### 5.11 Validators

Appended to `validate_settings` (`conf.py:378`), after the `TRAINING` checks at
`conf.py:463-469`. Each block is checked with the same shape: a mapping, then the
bounds the consumer enforces. The bounds are duplicated from the consumers on
purpose — the constructor's guard protects a caller, and the validator's job is to
fail before a peer is built, at `check` time, with a message that names the file.

```python
def _validate_blocks(settings: Any) -> None:
    """Document 15 section 5.11: the blocks this document adds."""
    for name in ("TRAINER", "OPTIMIZER", "LOSS", "LOGGER", "WATCHDOG",
                 "CHECKPOINT", "ORCHESTRATOR", "CORS"):
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
                "TRAINER.PEER_STRATEGY must be one of "
                + ", ".join(sorted(PEER_STRATEGIES))
            )
        if str(trainer.get("WEIGHT_UPDATE", "kappa_softmax")).lower() not in WEIGHT_STRATEGIES:
            raise ImproperlyConfigured(
                "TRAINER.WEIGHT_UPDATE must be one of "
                + ", ".join(sorted(WEIGHT_STRATEGIES))
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
        known = {"console", "file", "jsonl", "tensorboard", "wandb"}
        if not isinstance(sinks, (list, tuple)) or not sinks:
            raise ImproperlyConfigured("LOGGER.SINKS must be a non-empty list")
        unknown = sorted(set(map(str, sinks)) - known)
        if unknown:
            raise ImproperlyConfigured(
                f"LOGGER.SINKS names unknown sink(s) {unknown}; use {sorted(known)}"
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
        known |= {"SNAPSHOT_EVERY_EPOCHS", "SNAPSHOT_EVERY_SECONDS", "SNAPSHOT_ON_DEGRADATION",
                  "ON_DEGRADATION", "ON_CRITICAL", "KEEP_HISTORY"}
        unknown = sorted({key.lower() for key in watchdog} - known)
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

    api, cors = settings.get("API"), settings.get("CORS")
    if api and cors and api.get("CORS_ORIGINS"):
        raise ImproperlyConfigured(
            "API.CORS_ORIGINS and CORS.ORIGINS both set; use CORS for the rich form "
            "and leave API.CORS_ORIGINS empty"
        )
```

Three of these are worth a sentence each.

**`WATCHDOG` rejects unknown keys.** Every other block tolerates one, because a
new key is forward-compatible. `WATCHDOG` does not, because its failure mode is
silence: `MAX_GRAD_NORMN: 100` would leave `max_grad_norm` at its default and the
probe would never fire, and the run would look healthy while it diverged. The cost
of a typo is asymmetric, so the check is.

**`LOGGER.SINKS` validates against what the sink needs.** `"file"` without
`LOG_FILE` is a `FileSink(None)` that raises at construction with a `TypeError`
three frames deep. Naming the pair in the validator is the difference between a
config error and a stack trace.

**`CORS` versus `API.CORS_ORIGINS`.** Both exist because `API` predates this
document and `api.yaml` owns both. The refusal keeps one authority: a reader of
`api.yaml` does not have to know which of two lists is live.

`PEER_STRATEGIES` and `WEIGHT_STRATEGIES` are imported at the top of the function
from `mactf.coopetition.strategy` (`strategy.py:376`) and
`mactf.coopetition.weights` — both are already imported by
`mactf/orchestration/orchestrator.py:132`, so the dependency is not new.

## 6. The 17 files

Every file below is a complete, literal template. `{{ project_name }}` is the only
token `startproject` substitutes, per `startproject.render`. Nothing is elided,
and every value is one the code in section 7 will accept.

Each file opens with a three-line header that says what it owns, because the file
is the documentation a deployment actually reads.

### 6.1 `conf.yaml` — paths and process-wide execution

```yaml
# conf/conf.yaml — owns: PATHS, DEBUG, DEVICE, DTYPE, AMP, SEED,
# DETERMINISTIC, DEFAULT_AUTO_FIELD. Loaded first; every other file's
# relative paths resolve against PATHS.ROOT.

DEBUG: false

# auto | cpu | cuda | cuda:1 | mps
DEVICE: auto
# auto | float32 | float16 | bfloat16 | float64
DTYPE: auto
# CUDA only; ignored elsewhere by mactf.runtime.
AMP: true

SEED: 1337
# true makes torch use deterministic kernels. Slower, and the only way to make a
# bitwise comparison across two runs meaningful.
DETERMINISTIC: false

DEFAULT_AUTO_FIELD: mactf.db.BigAutoField

PATHS:
  # "." means the project package directory, which is where settings.BASE_DIR
  # points. Leave it alone unless the project runs from outside its package.
  ROOT: "."
  VAR: var
  CHECKPOINTS: var/checkpoints
  CONFIGS: conf
  DATASETS: datasets
  LOGS: var/logs
  IMAGES: var/images
  RUNS: var/runs
  TEACHERS: var/teachers
  AUDITS: var/audits
  CACHE: var/cache
  STATE: var/state
```

`CONFIGS` changes from `configs` to `conf` because the directory it names is the
one this document creates. `PATHS["CONFIGS"]` is read by the `check` command's
summary and by nothing at runtime, so the change is cosmetic — but a stale value
would print a directory the project does not have.

### 6.2 `mactf.yaml` — framework wiring

```yaml
# conf/mactf.yaml — owns: HTTP, and records INSTALLED_APPS, MIDDLEWARE,
# ROOT_URLCONF, APPEND_SLASH for a reader. The apps list below is documentation:
# apps are imported before configuration is read, so settings.py is the source of
# truth for INSTALLED_APPS. A conf/ file that sets it is refused (reserved key).

# --- documented, not authoritative -----------------------------------------
# INSTALLED_APPS:
#   - mactf.contrib.scoring
#   - mactf.contrib.coopetition
#   - mactf.contrib.sandbox
#   - mactf.contrib.telemetry
#   - "{{ project_name }}.peers"
#
# MIDDLEWARE:
#   - mactf.http.middleware.RequestIdMiddleware
#   - mactf.http.middleware.ExceptionMiddleware
#
# ROOT_URLCONF: null
# APPEND_SLASH: true

HTTP:
  # 8 MiB. The largest accepted request body; a score submission with 64 peer
  # records is under 200 KiB, so this is headroom, not a target.
  MAX_BODY_BYTES: 8388608
  # Worker threads for streaming responses.
  STREAM_WORKERS: 16
  # Seconds between SSE keepalives. Must be below the shortest proxy timeout in
  # front of the service.
  SSE_HEARTBEAT: 15.0

# Extra overlays, merged after this directory. A path outside the project is
# allowed; a path that does not exist is refused at load.
CONFIG_FILES: []
```

### 6.3 `peers.yaml` — the swarm

```yaml
# conf/peers.yaml — owns: PEERS. Each entry is a class with kwargs; the class is a
# dotted path resolved when the app registry populates, not when this file is read.

PEERS:
  - name: peer_alpha
    class: "{{ project_name }}.peers.MLPPeer"
    kwargs:
      state_dim: 16
      action_dim: 4
      hidden: 64
  - name: peer_beta
    class: "{{ project_name }}.peers.MLPPeer"
    kwargs:
      state_dim: 16
      action_dim: 4
      hidden: 64
  - name: peer_gamma
    class: "{{ project_name }}.peers.MLPPeer"
    kwargs:
      state_dim: 16
      action_dim: 4
      hidden: 64
```

Three peers with identical kwargs is not redundancy: `Orchestrator.prepare`
(`orchestrator.py:114`) deep-copies the module per peer, and the different
initialisations are what give the swarm diversity. A project that wants one peer
deletes two entries.

`PEERS` cannot live in `settings.py` *and* here. `peers.yaml` wins when it exists,
because `conf/` merges after the module (section 4.4), and `PEERS` is a list, so
the merge replaces rather than appends — which is the behaviour a reader expects
from "this file is the swarm".

### 6.4 `trainer.yaml` — the loop

```yaml
# conf/trainer.yaml — owns: TRAINING and TRAINER. TRAINER is folded into TRAINING
# at load (section 5.2), so a consumer reads one block.

TRAINER:
  EPOCHS: 3
  BATCH_SIZE: 32
  STEPS_PER_EPOCH: 4
  # Accumulate gradients over N steps before one optimizer step.
  GRAD_ACCUMULATION: 1
  MAX_GRAD_NORM: 1.0

  # A flat run: every peer trains independently and kappa-softmax blends.
  HIERARCHY: false
  PEER_STRATEGY: kappa_softmax
  WEIGHT_UPDATE: kappa_softmax
  # independent | shared_trunk | tied_weights. Siblings are averaged, so they must
  # not share parameters; shared_trunk shares a frozen trunk only.
  SHARING: independent

  TASK: synthetic_classification
  CHECKPOINT_EVERY: 1
  LOG_EVERY_STEPS: 1
  WATCHDOG_EVERY_EPOCHS: 1

# The model's input and output widths. They must match every peer's kwargs in
# peers.yaml; the trainer refuses a mismatch when it binds optimizers.
TRAINING:
  STATE_DIM: 16
  ACTION_DIM: 4
  LEARNING_RATE: 0.001

  GAMMA: 0.99
  GAE_LAMBDA: 0.95
  CRITIQUE_ROUNDS: 1

  # One bootstrap pass before the real loop: the reference policy is trained on a
  # small slice so the first contrast is not against random weights.
  BOOTSTRAP: true
  BOOTSTRAP_CONFIG:
    EPOCHS: 1
    BATCH_SIZE: 16
    LEARNING_RATE: 0.001
    WEIGHT_DECAY: 0.0
    MAX_GRAD_NORM: 1.0
    UPDATE_REFERENCE: true
    PATIENCE: 0
    BATCHES_PER_EPOCH: 0

  REDTEAM: false

  # Stop the run when the entropy ratio falls below this, after a snapshot.
  ROLLBACK_ON_ENTROPY_COLLAPSE: true
  ENTROPY_COLLAPSE_RATIO: 0.05
```

`LEARNING_RATE` appears here **and** in `optimizer.yaml`. They are not duplicates:
`TRAINING.LEARNING_RATE` is what `build_peer` (`orchestrator.py:120`) hands a peer,
and `OPTIMIZER.LR` is what the optimizer is constructed with. The validator
refuses a project where both are set to different values, because a peer whose
optimizer has one rate and whose construction was told another is a bug that
presents as "training is slow".

### 6.5 `optimizer.yaml` — the backend

```yaml
# conf/optimizer.yaml — owns: OPTIMIZER. Adapter is mactf.training.optimizer:
# from_spec accepts torch, scipy, and callable backends.

OPTIMIZER:
  # torch | scipy | callable
  TYPE: torch
  # A class on torch.optim. Unknown names are refused at construction with the
  # accepted set, never at epoch 4.
  CLASS: AdamW
  LR: 0.001
  WEIGHT_DECAY: 0.01
  # Adam/AdamW only. Set via BETAS so the mapping stays a plain YAML list.
  BETAS: [0.9, 0.999]
  EPS: 1.0e-8

  GRAD_ACCUMULATION: 1
  MAX_GRAD_NORM: 1.0

  # A scheduler is named, not constructed from YAML. Supported values:
  # null | step | cosine | exponential | plateau
  SCHEDULER: null
  SCHEDULER_KWARGS: {}
```

For a scipy backend the same file becomes:

```yaml
OPTIMIZER:
  TYPE: scipy
  METHOD: L-BFGS-B
  MAX_ITER: 100
  TOL: 1.0e-6
  LR: null
  GRAD_ACCUMULATION: 1
  MAX_GRAD_NORM: null
```

`LR: null` is meaningful: `MACTFOptimizer.from_spec` strips `None` values
(`optimizer.py:1110`), so the backend's own default applies rather than a literal
`None` reaching `scipy.optimize.minimize`.

### 6.6 `loss.yaml` — the objective

```yaml
# conf/loss.yaml — owns: LOSS. These six coefficients are LossConfig's fields
# (mactf/rl/loss.py:53). LOSS wins over TRAINING as a block, not per key.

LOSS:
  # composite uses CompositeLoss over the coefficients below. Any other value is
  # a mactf.rl.functions registry name, and ARGS is passed to it.
  TYPE: composite

  VALUE_COEF: 0.5
  KL_COEF: 0.01
  ENTROPY_COEF: 0.0
  CLIP_EPSILON: 0.2
  COMPLIANCE_COEF: 1.0
  ADVANTAGE_NORMALIZATION: true

  ARGS: {}
```

`CLIP_EPSILON` moves here from `TRAINING` (`conf.py:222`). It is a PPO surrogate
parameter, which makes it a loss coefficient; leaving it in `TRAINING` is how a
project ends up setting it twice.

### 6.7 `scorer.yaml` — scoring, severity, warnings, engine

The legal magnitudes. Every number traces to `docs/framework.md`; this file is a
transcription of `DEFAULTS` (`conf.py:105-158`), not a re-tuning.

```yaml
# conf/scorer.yaml — owns: SCORING, SEVERITY, WARNINGS, ENGINE.

SCORING:
  # framework.md 3.2: only the controlling tier scores. "layered" scores every
  # tier but still refuses to net across tiers.
  PRECEDENCE_MODE: strict

  # R = a*r_task + b*r_epi + g*r_collab + d*r_align
  REWARD_WEIGHTS:
    r_task: 1.0
    r_epi: 1.5
    r_collab: 1.2
    r_align: 1.0

  # P = sum(lambda_k * p_k) * Omega
  PENALTY_LAMBDAS:
    P1: 5.0
    P2: 5.0
    P3: 3.0
    P4: 0.5
    P5: 2.5
    P6: 1.0
    P7: 1.0
    P8: 0.75
    P9: 2.0
    P10: 2.0
    P11: 0.5
    P12: 0.5
    P13: 8.0
    P14: 8.0
    P15: 1.5
    P16: 3.0
    P17: 50.0
    P18: 1.5
    P19: 1.5
    P20: 1000.0
    P21: 50.0
    P22: 50.0
    P23: 2.0
  DEFAULT_LAMBDA: 1.0

  SANDBOX_MULTIPLIER: 1000.0
  OMEGA_RULES: [P20, P21, P22]
  OMEGA_RULE_OVERRIDES: {}

  # Removing any of these would silently permit netting across tiers. The
  # validator refuses a list that drops one.
  NON_OFFSETTABLE: [P1, P2, P17, P20, P21, P22]
  ALLOW_NEGATIVE_REWARDS: false

  F_CAP_RATIO: 0.5
  P_MIN: 0.5
  NOISE_THRESHOLD: 0.2

  # Doc 11: the 0-100 points economy. Reward weights sum to 100.
  POINTS:
    TABLE_VERSION: "1.0.0"
    WARNING_NOTIONAL: 2.0
    REWARD_WEIGHTS:
      R1: 25.0
      R2: 12.0
      R3: 2.0
      R4: 3.0
      R5: 3.0
      R6: 5.0
      R7: 5.0
      R8: 6.0
      R9: 3.0
      R10: 8.0
      R11: 5.0
      R12: 4.0
      R13: 6.0
      R14: 5.0
      R15: 3.0
      R16: 4.0
      R17: 1.0
    PENALTY_WEIGHTS:
      P1: 40.0
      P2: 40.0
      P3: 20.0
      P4: 5.0
      P5: 25.0
      P6: 6.0
      P7: 6.0
      P8: 5.0
      P9: 15.0
      P10: 15.0
      P11: 4.0
      P12: 4.0
      P13: 60.0
      P14: 60.0
      P15: 10.0
      P16: 25.0
      P17: 100.0
      P18: 10.0
      P19: 10.0
      P20: 100.0
      P21: 100.0
      P22: 100.0
      P23: 12.0
    SEVERITY_FRACTIONS:
      S0: 0.0
      S1: 0.05
      S2: 0.15
      S3: 0.4
      S4: 1.0

SEVERITY:
  # Must be strictly increasing S0..S4; the validator checks it.
  WEIGHTS:
    S0: 0.0
    S1: 0.25
    S2: 1.0
    S3: 4.0
    S4: 25.0
  MAX_TIER_ESCALATION: 3

WARNINGS:
  # K_ESC consecutive episodes escalate a warning.
  K_ESC: 3
  # N_WIN episodes form a window; N_DECAY episodes of clean behaviour decay it.
  N_WIN: 5
  N_DECAY: 3
  # W_CAP warnings in the window; must be <= K_ESC.
  W_CAP: 2
  P_WARN: 5.0
  ESCALATE: true

ENGINE:
  F_CAP: 3
  P_MIN: 0.5
  P12_NOISE_RATIO: 0.5
  R9_STEP_FACTOR: 1.25
```

The penalty lambdas are a mapping with 23 entries and the points table has 17
reward and 23 penalty weights. Writing them out is deliberate: a project that
raises `P17` to 75 needs to see `P17` on the adjacent line to know it is raising
a non-offsettable fail, and a `<<: *defaults` anchor would hide that.

### 6.8 `orchestrator.yaml` — aggregation

```yaml
# conf/orchestrator.yaml — owns: COOPETITION and ORCHESTRATOR.

COOPETITION:
  # Adaptive temperature. Kappa softens or sharpens the weight distribution.
  KAPPA_INIT: 1.0
  KAPPA_MIN: 0.1
  KAPPA_MAX: 10.0
  TARGET_ENTROPY_RATIO: 0.8
  BETA_MOMENTUM: 0.9
  ETA_KAPPA: 0.05

  # framework.md 10: no peer may take the whole blend or vanish from it.
  WEIGHT_FLOOR: 0.02
  WEIGHT_CAP: 0.6
  # Above this disagreement the blend falls back to the reference.
  DISAGREEMENT_MAX: 0.75
  FALLBACK_TO_REFERENCE: true
  TRIM_FRACTION: 0.0
  CONFIDENCE_FLOOR: 0.1
  REFERENCE_EMA: 0.05
  RESET_OPTIMIZERS: true
  AGGREGATE_BUFFERS: top

  # final_score | points. Which scorecard field drives aggregation.
  SIGNAL_SOURCE: final_score

  NORMALIZATION:
    ENABLED: true
    TARGET_MEAN: 62.0
    SHRINK: 0.35
    EXCLUDE_VOID: true
    MIN_COHORT: 2
    SIGNAL_SCALE: 50.0
    CLIP: [0.0, 100.0]

  # framework.md 9.2: report the cooperative component from a bounded
  # per-episode pool so R10/R11/R16 cannot be farmed.
  COOPERATIVE_POOL: true
  COOPERATIVE_BUDGET:
    PER_EPISODE: 6.0
    PER_PEER_CAP: 3.0
    CARRY_OVER: false

ORCHESTRATOR:
  TASK_SOURCE: null
  MAX_EPOCHS: null
  RESUME_FROM: null
  ANNOUNCE: true
  PAUSE_IS_FATAL: false
```

`WEIGHT_FLOOR: 0.02` and `WEIGHT_CAP: 0.6` are the values `DEFAULTS` already has,
and `feasible_bounds` (`aggregation.py:347`) refuses a swarm where
`floor * peers > 1` or `cap * peers < 1`. With three peers both hold easily; a
30-peer swarm needs a lower floor, and the error message names the arithmetic.

### 6.9 `sandbox.yaml` — isolation

```yaml
# conf/sandbox.yaml — owns: SANDBOX. QEMU is preferred; the local provider is the
# fallback. Set ALLOW_FALLBACK: false to make hardware isolation mandatory.

SANDBOX:
  # auto | qemu | local. "auto" probes for QEMU and degrades per ALLOW_FALLBACK.
  PROVIDER: auto
  # hardware | process. "hardware" means a VM or container boundary.
  ISOLATION: hardware
  ALLOW_FALLBACK: true
  REQUIRE_HARDWARE: false

  BASE_IMAGE: var/images/base.qcow2
  # {peer} is replaced with the peer name, so each peer gets its own overlay.
  WORK_IMAGE: var/images/peer-{peer}.qcow2
  SERIAL_SOCKET: var/run/mactf-{peer}.sock

  MEMORY_MB: 4096
  VCPUS: 2
  # auto | kvm | hvf | tcg
  ACCEL: auto
  # none | host. A sandbox that needs the network says so here, not in code.
  NETWORK: none
  SNAPSHOT: true

  BOOT_TIMEOUT: 180
  COMMAND_TIMEOUT: 300

  CLOUD_INIT:
    USER: mactf
    PACKAGES:
      - python3
      - python3-pip
```

### 6.10 `distillation.yaml` — teachers

```yaml
# conf/distillation.yaml — owns: DISTILLATION. Every key here is optional; with
# ENABLED: false the trainer produces the numbers it produced before this block
# existed (document 14 section 14.1).

DISTILLATION:
  ENABLED: false

  # A teacher list is a project decision. ENABLED: true with an empty list is
  # refused at load, because a distilled run with no teacher is a typo.
  TEACHERS: []

  OUT_DIM: 64
  TAU: 0.07
  TAU_MIN: 0.01
  TAU_MAX: 1.0
  SYMMETRIC: true
  BETA: 1.0

  SIGMA_MODE: quality_softmax
  SIGMA_FLOOR: 0.0
  SIGMA_CAP: 1.0
  SIGMA_DECAY: 0.9

  LAMBDA_COOP: 0.0
  LAMBDA_IND: 1.0
  LAMBDA_REG: 0.0
  LAMBDA_CD: 1.0
  PEER_MARGIN: 0.25

  WARMUP_STEPS: 0
  RAMP_STEPS: 0

  REGULARIZERS: []
  VIEWS: {}
  STRICT: true

  CONFORMANCE:
    ENABLED: true
    MODE: hybrid
    TRUST_INIT: 1.0
    TRUST_FLOOR: 0.0
    TRUST_DECAY: 0.5
    TRUST_RESTORE: 0.05
    TRUST_MIN: 0.05
    STRICT_UNDECLARED: true
    AUDIT_ARTIFACTS: {}
    SIGMA_TRUST_MIN: true
```

Adopting distillation means editing two lines and adding a teacher:

```yaml
DISTILLATION:
  ENABLED: true
  TEACHERS:
    - name: wide-teacher
      kind: module
      factory: "{{ project_name }}.peers.WidePeer"
      kwargs: {state_dim: 16, action_dim: 4}
      view: {kind: fixed_random, out_dim: 128}
      weight: 1.0
      trust: {tiers: [3, 4, 5], trust_init: 0.9, source: "operator review"}
```

That block is document 14's shape verbatim; this document only decides the file
name and the header.

### 6.11 `logger.yaml` — sinks

```yaml
# conf/logger.yaml — owns: LOGGER. When this block is present it replaces
# TELEMETRY for the logger; TELEMETRY stays live for projects that never adopt it.

LOGGER:
  LEVEL: INFO
  RUNS_DIR: var/runs

  # Order matters: every record goes to each sink, in this order.
  # console | file | jsonl | tensorboard | wandb
  SINKS: [console, file]

  JSON: false
  LOG_FILE: var/logs/mactf.log
  JSONL_FILE: null

  ROTATE_BYTES: 10485760
  ROTATE_BACKUPS: 5

  BUFFER_SIZE: 1000
  ASYNC_WRITE: true
  QUEUE_SIZE: 8192

  # Empty means the seven defaults in mactf/training/logging.py:53. Naming
  # fragments REPLACES the list, so the shipped file spells them out: a project
  # that adds "session_cookie" should not also lose "authorization".
  REDACT:
    - token
    - secret
    - password
    - api_key
    - authorization
    - hf_token
    - wandb_key

  SCHEMA_VERSION: 1

  # Required when "tensorboard" is in SINKS.
  TENSORBOARD_DIR: null
  # Required when "wandb" is in SINKS.
  WANDB_PROJECT: null
```

The `REDACT` list is the reason the default in code is `()` and the shipped file
is explicit: `MACTFLogger.__init__` (`logging.py:511`) treats an empty `redact` as
"use `DEFAULT_REDACT`", so a project that adds one fragment would otherwise
*replace* the seven. Spelling them out makes the file the complete list and
removes the asymmetry.

### 6.12 `telemetry.yaml` — the legacy block

```yaml
# conf/telemetry.yaml — owns: TELEMETRY. Read by MACTFLogger._default_sinks when
# LOGGER is absent. Keep this file if the project predates conf/logger.yaml.

TELEMETRY:
  LEVEL: INFO
  JSON: false
  CONSOLE: true
  LOG_FILE: var/logs/mactf.log
```

### 6.13 `watchdog.yaml` — health

```yaml
# conf/watchdog.yaml — owns: WATCHDOG. Unknown keys are REFUSED: a typo in a
# threshold is a watchdog that never fires, and silence is the one failure this
# component cannot report about itself.

WATCHDOG:
  # abort | rollback | ignore. A NaN loss that is ignored is a run that wastes
  # the remaining epochs.
  NAN_POLICY: abort
  INF_POLICY: abort

  LOSS_INCREASE_RATIO: 3.0
  LOSS_PATIENCE: 2

  MAX_GRAD_NORM: 100.0
  GRAD_NORM_PATIENCE: 1

  SCORE_REGRESSION_TOLERANCE: 0.10
  SCORE_PATIENCE: 3

  ENTROPY_COLLAPSE_RATIO: 0.05
  ENTROPY_PATIENCE: 2

  KAPPA_MIN: 1.0e-3
  KAPPA_MAX: 1.0e3

  STALE_EPOCH_SECONDS: 1800.0
  MIN_THROUGHPUT_RATIO: 0.25
  MAX_STEP_TIME_DRIFT: 5.0

  MAX_RSS_GROWTH_RATIO: 2.0
  MAX_GPU_MEMORY_RATIO: 0.95

  MAX_CHECKPOINT_AGE_SECONDS: 3600.0
  WARNING_SPIKE_THRESHOLD: 10

  MAX_TREE_GROWTH_RATIO: 2.0
  DEAD_BRANCH_PATIENCE: 2
  CHILD_STARVATION_PATIENCE: 3
  DEPTH_COST_RATIO: 4.0

  # The watchdog's own cadence and response.
  SNAPSHOT_EVERY_EPOCHS: 1
  SNAPSHOT_EVERY_SECONDS: null
  SNAPSHOT_ON_DEGRADATION: true
  # none | log | snapshot | rollback | prune_branch | pause | abort
  ON_DEGRADATION: snapshot
  ON_CRITICAL: rollback
  KEEP_HISTORY: 512
```

### 6.14 `checkpoint.yaml` — retention

```yaml
# conf/checkpoint.yaml — owns: CHECKPOINT. Mirrors MACTFCheckpoint.__init__
# (mactf/orchestration/checkpoint_v2.py:395).

CHECKPOINT:
  # null resolves to PATHS["CHECKPOINTS"]. Set an absolute path to put a run's
  # checkpoints on a different volume, which is what a large swarm does.
  DIR: null

  KEEP: 5
  KEEP_BEST: 1

  COMPRESS: true
  # sha256 | blake2b
  CHECKSUM: sha256
  ASYNC_WRITE: false

  # These five decide what a checkpoint can restore. Turning one off makes a
  # resume lossy in a way that is not detectable afterwards, so each is explicit.
  INCLUDE_OPTIMIZER: true
  INCLUDE_RNG: true
  INCLUDE_DATA_POSITION: true
  INCLUDE_BUFFERS: true
  INCLUDE_ADAPTERS: true
```

### 6.15 `api.yaml` — the service layer

```yaml
# conf/api.yaml — owns: API and CORS. Read by mactf.orchestration.live and the
# mactf.contrib.api views.

API:
  ALLOW_ANONYMOUS: true

  # {"<token>": ["read", "run:write"]} or
  # {"<token>": {scopes: ["read"], name: "ci"}}. Scopes: read, run:write,
  # score:write, sandbox:exec, admin, worker.
  TOKENS: {}

  BASE_PATH: api/v1
  EVENT_BUFFER: 1024
  MAX_CONCURRENT_RUNS: 1
  CANCEL_GRACE_SECONDS: 10.0

  # Leave empty and use CORS.ORIGINS below; the validator refuses both.
  CORS_ORIGINS: []
  PRINCIPAL_RESOLVER: null
  IDEMPOTENCY_TTL_SECONDS: 86400

  LOG_TAIL_DEFAULT: 100
  LOG_TAIL_MAX: 2000
  HEALTH_HISTORY_LIMIT: 50
  PREVIEW_MAX_NODES: 64

  INFERENCE_MAX_WORKERS: 4
  INFERENCE_BUDGET_DEFAULT: null

CORS:
  ORIGINS: []
  ALLOW_CREDENTIALS: false
  ALLOW_HEADERS: [authorization, content-type, x-request-id]
  MAX_AGE: 600
```

The token map is a **secret store** in a committed file, which is why the shipped
value is `{}` and the header does not encourage an inline token. A deployment sets
`MACTF_API`-scoped values through its secret manager into a
`settings.py` override or an untracked overlay listed in `CONFIG_FILES` — the
overlay mechanism this document keeps.

### 6.16 `db.yaml` — the registry database

```yaml
# conf/db.yaml — owns: DATABASES. SQLite is the only supported engine; a
# non-sqlite ENGINE is refused by Settings.database_path.

DATABASES:
  default:
    ENGINE: sqlite3
    NAME: var/mactf.sqlite3
```

### 6.17 `distributed.yaml` — workers

```yaml
# conf/distributed.yaml — owns: DISTRIBUTED. Read only when mactf.distributed is
# installed; harmless otherwise.

DISTRIBUTED:
  COORDINATOR_URL: http://127.0.0.1:8000
  BASE_PATH: worker/v1

  # null derives an id from the hostname. Set it when several workers share a
  # host, which is the only case where the derivation collides.
  WORKER_ID: null
  MAX_PEERS: 1

  COMPRESS: true
  WORKER_HEARTBEAT_TIMEOUT: 30.0
  WORKER_TASK_TIMEOUT: 600.0
  MAX_WORKERS: 16
```

### 6.18 What the set covers

Every top-level `DEFAULTS` key and the file that owns it, checked once so a reader
does not have to:

| Key | File | Key | File |
|---|---|---|---|
| `DEBUG` | `conf.yaml` | `LOGGER` | `logger.yaml` |
| `INSTALLED_APPS` | `settings.py` | `TELEMETRY` | `telemetry.yaml` |
| `PEERS` | `peers.yaml` | `WATCHDOG` | `watchdog.yaml` |
| `ROOT_URLCONF` | `settings.py` | `CHECKPOINT` | `checkpoint.yaml` |
| `MIDDLEWARE` | `settings.py` | `DISTILLATION` | `distillation.yaml` |
| `APPEND_SLASH` | `settings.py` | `ORCHESTRATOR` | `orchestrator.yaml` |
| `HTTP` | `mactf.yaml` | `API` | `api.yaml` |
| `API` | `api.yaml` | `CORS` | `api.yaml` |
| `DISTRIBUTED` | `distributed.yaml` | `DATABASES` | `db.yaml` |
| `DEFAULT_AUTO_FIELD` | `conf.yaml` | `PATHS` | `conf.yaml` |
| `DEVICE` | `conf.yaml` | `TRAINING` | `trainer.yaml` |
| `DTYPE` | `conf.yaml` | `TRAINER` | `trainer.yaml` |
| `AMP` | `conf.yaml` | `OPTIMIZER` | `optimizer.yaml` |
| `SEED` | `conf.yaml` | `LOSS` | `loss.yaml` |
| `DETERMINISTIC` | `conf.yaml` | `SCORING` | `scorer.yaml` |
| `ENGINE` | `scorer.yaml` | `SEVERITY` | `scorer.yaml` |
| `WARNINGS` | `scorer.yaml` | `COOPETITION` | `orchestrator.yaml` |
| `SANDBOX` | `sandbox.yaml` | `CONFIG_FILES` | `mactf.yaml` |

## 7. Consumer bindings

A block that nothing reads is a comment with a schema. This section is the table
of readers, with the exact call site each one changes.

### 7.1 The `from_settings` factories

Four classes take constructor arguments today and read no settings. Each gains one
classmethod. The pattern is `Aggregator.from_settings`
(`aggregation.py:333`), which already exists and is the model:

```python
@classmethod
def from_settings(cls, settings: dict[str, Any], *, num_peers: int) -> "Aggregator":
    controller = AdaptiveKappa.from_settings(settings, num_peers=num_peers)
    return cls(
        controller=controller,
        weight_floor=float(settings.get("WEIGHT_FLOOR", 0.02)),
        ...
    )
```

`from_settings` reads a **block**, not the whole settings object. That is why the
signature is `settings: dict[str, Any]` and the caller passes
`settings["LOGGER"]`: a factory that can reach every block is a factory that can
read a block it does not own.

#### `MACTFLogger.from_settings`

```python
@classmethod
def from_settings(cls, settings: Any, *, run_id: str | None = None) -> "MACTFLogger":
    """Build a logger from ``LOGGER``, falling back to ``TELEMETRY``.

    A sink named in ``SINKS`` that needs a path and has none raises here, with
    the key name, rather than at the first write.
    """
    config = _logger_config(settings)
    sinks = cls._build_sinks(config, run_id=run_id)
    return cls(
        settings=settings,
        run_id=run_id,
        level=str(config.get("LEVEL", "INFO")),
        sinks=sinks,
        buffer_size=int(config.get("BUFFER_SIZE", 1000)),
        redact=tuple(config.get("REDACT") or ()),
        async_write=bool(config.get("ASYNC_WRITE", True)),
        queue_size=int(config.get("QUEUE_SIZE", 8192)),
        schema_version=int(config.get("SCHEMA_VERSION", 1)),
    )

@staticmethod
def _build_sinks(config: Mapping[str, Any], *, run_id: str | None) -> list[LogSink]:
    """The sinks ``SINKS`` names, in order. ``memory`` is added by ``__init__``."""
    json_format = bool(config.get("JSON", False))
    built: list[LogSink] = []
    for name in config.get("SINKS", ["console", "file"]):
        key = str(name).lower()
        if key == "console":
            built.append(ConsoleSink(json_format=json_format))
        elif key == "file":
            built.append(
                FileSink(
                    config["LOG_FILE"],
                    json_format=json_format,
                    max_bytes=int(config.get("ROTATE_BYTES", 10_485_760)),
                    backup_count=int(config.get("ROTATE_BACKUPS", 5)),
                )
            )
        elif key == "jsonl":
            built.append(JsonlSink(config["JSONL_FILE"]))
        elif key == "tensorboard":
            built.append(TensorBoardSink(config["TENSORBOARD_DIR"]))
        elif key == "wandb":
            built.append(WandBSink(project=config.get("WANDB_PROJECT"), config=config))
        else:  # pragma: no cover - the validator refuses this first
            raise ImproperlyConfigured(f"unknown LOGGER sink {name!r}")
    return built
```

`MemorySink` is not in `SINKS` and cannot be: `MACTFLogger.__init__`
(`logging.py:526`) always constructs one and appends it if absent, because
`tail()` must work regardless of configuration. Listing it would be a no-op that
implies otherwise.

#### `MACTFCheckpoint.from_settings`

```python
@classmethod
def from_settings(
    cls,
    settings: Any,
    *,
    run_id: str,
    directory: str | Path | None = None,
) -> "MACTFCheckpoint":
    """Build a writer from ``CHECKPOINT``, resolving ``DIR`` through ``PATHS``."""
    block = dict(_get(settings, "CHECKPOINT", {}) or {})
    target = directory or block.get("DIR")
    if target is None:
        ensure = getattr(settings, "ensure_path", None)
        if not callable(ensure):
            raise ImproperlyConfigured(
                "CHECKPOINT.DIR is unset and settings cannot resolve PATHS; "
                "pass an explicit directory"
            )
        target = ensure("CHECKPOINTS")
    return cls(
        target,
        run_id=run_id,
        keep=int(block.get("KEEP", 5)),
        keep_best=int(block.get("KEEP_BEST", 1)),
        compress=bool(block.get("COMPRESS", True)),
        checksum=str(block.get("CHECKSUM", "sha256")),
        async_write=bool(block.get("ASYNC_WRITE", False)),
        include_optimizer=bool(block.get("INCLUDE_OPTIMIZER", True)),
        include_rng=bool(block.get("INCLUDE_RNG", True)),
        include_data_position=bool(block.get("INCLUDE_DATA_POSITION", True)),
        include_buffers=bool(block.get("INCLUDE_BUFFERS", True)),
        include_adapters=bool(block.get("INCLUDE_ADAPTERS", True)),
    )
```

#### `MACTFWatchdog.from_settings`

```python
@classmethod
def from_settings(
    cls,
    settings: Any,
    *,
    checkpoint: Any,
    logger: Any,
) -> "MACTFWatchdog":
    """Build a watchdog from ``WATCHDOG``.

    ``checkpoint`` and ``logger`` stay required: a watchdog without a writer
    cannot snapshot, and ``__init__`` already refuses one.
    """
    block = {str(key).lower(): value for key, value in
             dict(_get(settings, "WATCHDOG", {}) or {}).items()}
    fields = set(WatchdogPolicy.__dataclass_fields__)
    policy = WatchdogPolicy(**{key: value for key, value in block.items() if key in fields})
    own = {"snapshot_every_epochs", "snapshot_every_seconds", "snapshot_on_degradation",
           "on_degradation", "on_critical", "keep_history"}
    return cls(
        checkpoint=checkpoint,
        logger=logger,
        policy=policy,
        **{key: block[key] for key in own if key in block},
    )
```

#### `MACTFOptimizer.from_block`

`MACTFOptimizer.from_spec` (`optimizer.py:1101`) already handles a mapping. It does
not need a settings reader; it needs the key-lowering shim from section 5.4:

```python
@classmethod
def from_block(cls, block: Mapping[str, Any], *, node_path: str = "") -> "MACTFOptimizer":
    """``OPTIMIZER`` (upper-case YAML) to a bound optimizer facade."""
    payload = optimizer_spec(block)
    payload.setdefault("node_path", node_path)
    return cls.from_spec(payload)
```

### 7.2 The reader table

Every consumer, the block it reads, and the change. "Fallback" means the code path
when the block is absent, which is what the equivalence gate exercises.

| # | Consumer | Site | Reads | Change | Fallback |
|---|---|---|---|---|---|
| C1 | `Orchestrator.__init__` | `orchestrator.py:74` | — | unchanged | — |
| C2 | `Orchestrator.prepare` | `orchestrator.py:103` | `TRAINING` | unchanged (the fold made it live) | `DEFAULTS` |
| C3 | `Orchestrator.prepare` | `orchestrator.py:144` | `CHECKPOINT` | `CheckpointManager(...)` → adds the block's retention kwargs | today's `CheckpointManager` |
| C4 | `Orchestrator.prepare` | `orchestrator.py:200` | `SANDBOX` | unchanged | — |
| C5 | `Orchestrator._checkpoint` | `orchestrator.py:489` | `TRAINING.CHECKPOINT_EVERY` | unchanged | — |
| C6 | `build_scorer` | `pipeline.py:700` | `SCORING`, `SEVERITY`, `ENGINE` | unchanged | — |
| C7 | `Aggregator.from_settings` | `aggregation.py:333` | `COOPETITION` | unchanged | — |
| C8 | `MACTFTrainer.build` | `trainer.py:527` | `TRAINING`, `COOPETITION` | reads `ORCHESTRATOR.MAX_EPOCHS` when set | `TRAINING.EPOCHS` |
| C9 | `MACTFTrainer._bind_optimizers` | `trainer.py:845` | `OPTIMIZER` | `from_spec({...Adam...})` → `MACTFOptimizer.from_block(settings["OPTIMIZER"])` | the inline Adam dict |
| C10 | `MACTFTrainer._loss_config` | `trainer.py:992` | `LOSS` | reads `LOSS` as a block, else `TRAINING` | `TRAINING` |
| C11 | `MACTFTrainer._build_checkpoint` | `trainer.py:955` | `CHECKPOINT` | `MACTFCheckpoint(directory, run_id=...)` → `from_settings` | last two args only |
| C12 | `MACTFTrainer._build_runtime` | `trainer.py:664` | `DEVICE`, `DTYPE`, `AMP` | unchanged | — |
| C13 | `MACTFLogger.__init__` | `logging.py:487` | `TELEMETRY` | gains `from_settings`; the constructor itself unchanged | `TELEMETRY` |
| C14 | `MACTFWatchdog.__init__` | `watchdog.py:826` | — | gains `from_settings`; the constructor unchanged | policy defaults |

Two of these deserve the code, because they are the only ones where "unchanged" is
not the whole story.

#### C9 — optimizer binding

`trainer.py:845-859` today:

```python
        for path, peer in self.peers.items():
            module = peer.adapter
            spec = self._optimizer_spec
            if spec is None:
                optimizer = MACTFOptimizer.from_spec(
                    {
                        "type": "torch",
                        "class": "Adam",
                        "lr": float(self._training_settings().get("LEARNING_RATE", 1e-3)),
                    }
                )
            else:
                optimizer = MACTFOptimizer.from_spec(spec)
```

becomes:

```python
        for path, peer in self.peers.items():
            module = peer.adapter
            spec = self._optimizer_spec
            if spec is not None:
                optimizer = MACTFOptimizer.from_spec(spec)
            elif "OPTIMIZER" in self.settings and self.settings["OPTIMIZER"]:
                optimizer = MACTFOptimizer.from_block(self.settings["OPTIMIZER"], node_path=path)
            else:
                optimizer = MACTFOptimizer.from_spec(
                    {
                        "type": "torch",
                        "class": "Adam",
                        "lr": float(self._training_settings().get("LEARNING_RATE", 1e-3)),
                    }
                )
```

The explicit `spec=` argument still wins over the block, which is the precedence
everywhere else in this document: an argument is an override, a block is
configuration, a default is a default. `node_path` is passed here rather than in
the block because it is per-peer, and a per-peer key in a shared file is a bug
waiting to be copy-pasted.

`MACTFOptimizer.from_block` must not set `node_path` when the caller passes `""`
and the block already carries one; `from_spec` does `node_path = str(payload.pop(...))`
so the explicit value is the only one. Section 7.1's `setdefault` handles it.

#### C13 — logger construction

`MACTFLogger.__init__` is unchanged; only the *default sinks* path changes, and it
goes through `from_settings`:

```python
        if sinks is None:
            sinks = self._default_sinks(telemetry)
```

stays, and `telemetry` becomes `_logger_config(settings)` (section 5.7) so a
project with `LOGGER` gets its sinks from there. `from_settings` exists for callers
who want the block resolved without constructing a settings object — the `run`
command is the main one:

```python
logger = MACTFLogger.from_settings(settings, run_id=orchestrator.run_id)
```

### 7.3 What is deliberately *not* wired

| Thing | Why not |
|---|---|
| `ORCHESTRATOR.PAUSE_IS_FATAL` | `Orchestrator.run`'s stop path is a per-epoch callback, not a policy read. A block that changed it would need a `RunConfig` field, and `RunConfig` is frozen for the run. The key ships for the *next* document that touches the stop path; until then `from_settings` does not read it and the validator accepts it. |
| `TRAINER.SHARING` | `MACTFTrainer(sharing=...)` is a constructor argument, and the shared-trunk path requires a `HierarchySpec` to place the trunk. Reading it from YAML without that structure would silently mean `independent`. It is validated for its enum and otherwise left to the caller. |
| `DISTILLATION.*` | Document 14 owns that block's consumers in full. This document ships the file and the validator hook; it does not duplicate 14's wiring. |
| `LOGGER.WANDB_PROJECT` | `WandBSink` is only constructed when `"wandb"` is in `SINKS`, and that is a project's explicit choice. There is no silent enable. |

Shipping a key the loader validates but no consumer reads is not a lie as long as
the table says so. The alternative — omitting the key — hides the decision from
the project's configuration file, which is worse.

## 8. `startproject`

### 8.1 The new layout

`startproject.LAYOUT` (`startproject.py:24`) becomes:

```python
LAYOUT: tuple[tuple[str, str], ...] = (
    ("manage.py.tpl", "manage.py"),
    ("package_init.py.tpl", "__init__.py"),
    ("settings.py.tpl", "settings.py"),
    ("peers/__init__.py.tpl", "peers/__init__.py"),
    ("conf/conf.yaml.tpl", "conf/conf.yaml"),
    ("conf/mactf.yaml.tpl", "conf/mactf.yaml"),
    ("conf/peers.yaml.tpl", "conf/peers.yaml"),
    ("conf/trainer.yaml.tpl", "conf/trainer.yaml"),
    ("conf/optimizer.yaml.tpl", "conf/optimizer.yaml"),
    ("conf/loss.yaml.tpl", "conf/loss.yaml"),
    ("conf/scorer.yaml.tpl", "conf/scorer.yaml"),
    ("conf/orchestrator.yaml.tpl", "conf/orchestrator.yaml"),
    ("conf/sandbox.yaml.tpl", "conf/sandbox.yaml"),
    ("conf/distillation.yaml.tpl", "conf/distillation.yaml"),
    ("conf/logger.yaml.tpl", "conf/logger.yaml"),
    ("conf/telemetry.yaml.tpl", "conf/telemetry.yaml"),
    ("conf/watchdog.yaml.tpl", "conf/watchdog.yaml"),
    ("conf/checkpoint.yaml.tpl", "conf/checkpoint.yaml"),
    ("conf/api.yaml.tpl", "conf/api.yaml"),
    ("conf/db.yaml.tpl", "conf/db.yaml"),
    ("conf/distributed.yaml.tpl", "conf/distributed.yaml"),
)
```

Twenty-one entries, and the `conf/` block is generated from `CONF_LAYOUT` rather
than typed, so the two cannot drift:

```python
from ...conf_files import CONF_LAYOUT

_LAYOUT: list[tuple[str, str]] = [
    ("manage.py.tpl", "manage.py"),
    ("package_init.py.tpl", "__init__.py"),
    ("settings.py.tpl", "settings.py"),
    ("peers/__init__.py.tpl", "peers/__init__.py"),
]
_LAYOUT.extend(
    (f"conf/{entry.template_name}", f"conf/{entry.filename}") for entry in CONF_LAYOUT
)
LAYOUT: tuple[tuple[str, str], ...] = tuple(_LAYOUT)
```

This is the one import `startproject` gains, and it is what makes "the files the
loader reads" and "the files the scaffolder writes" one list. A file added to
`CONF_LAYOUT` without a template is caught by the existing missing-template guard
(`startproject.py:77`), which already raises with the template name.

### 8.2 `write_project` changes

Three changes, all small:

```python
def write_project(project_name: str, *, target_root: Path, force: bool = False) -> list[Path]:
    validate_project_name(project_name)
    package_dir = target_root / project_name
    settings_path = package_dir / "settings.py"
    if settings_path.exists() and not force:
        raise CommandError(
            f"{package_dir} already exists; pass --force to overwrite the generated files"
        )

    root = template_root()
    written: list[Path] = []
    for template_name, relative in LAYOUT:
        source = root / template_name
        if not source.is_file():  # pragma: no cover - packaging regression guard
            raise CommandError(
                f"template {template_name!r} is missing from {TEMPLATE_PACKAGE}; "
                "the installation is incomplete"
            )
        destination = package_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # conf/*.yaml is copied verbatim: YAML has no Python braces to preserve,
        # and only the peers and settings files carry the project name.
        text = source.read_text(encoding="utf-8")
        if relative.endswith(".py"):
            text = render(text, project_name)
        else:
            text = text.replace(TOKEN, project_name)
        destination.write_text(text, encoding="utf-8")
        written.append(destination)

    ignore = package_dir / ".gitignore"
    if not ignore.exists() or force:
        ignore.write_text(
            "var/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n*.egg-info/\ndist/\nbuild/\n",
            encoding="utf-8",
        )
        written.append(ignore)
    return written
```

The `relative.endswith(".py")` branch is not a new mechanism: `render` is
`str.replace(TOKEN, name)`, which is already brace-safe for Python. The branch
exists to document that non-Python files are also substitution-safe, and to make
the line greppable when someone adds a template format that is not.

### 8.3 `settings.py.tpl`

The template shrinks from 78 lines to this, because every subsystem block moved:

```python
"""Settings for the {{ project_name }} MACTF project.

Identity only. Every subsystem argument lives in conf/, which is read after this
module and wins for any block it declares. See conf/ for the values; this file
answers "what is this project", not "how does it train".
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

# Where the configuration tree lives. The loader reads <CONF_DIR>/*.yaml in the
# order declared by mactf.conf_files.CONF_LAYOUT.
CONF_DIR = BASE_DIR / "conf"

# Overlays merged AFTER conf/, for a CI run or a one-off experiment. An empty
# list is the normal case.
CONFIG_FILES: list[str] = []
```

`PEERS` is gone. `peers.yaml` owns it, and two authorities for one list is the
conflict this document exists to remove. A project that prefers Python can still
set `PEERS` in `settings.py`; `peers.yaml` merges after and replaces it, so the
rule is "the file wins", stated once.

`DEBUG` stays though `conf.yaml` also declares it. `DEBUG` in `settings.py` is
what `manage.py` reads before the settings load completes; `conf.yaml`'s value
wins afterwards. A project that sets it in both and gets different answers during
startup is a genuine hazard, so the template sets `DEBUG = False` in both and the
header says which one wins.

### 8.4 `manage.py.tpl` and `peers/__init__.py.tpl`

`manage.py.tpl` is unchanged: it already does
`os.environ.setdefault("MACTF_SETTINGS_MODULE", "{{ project_name }}.settings")`.

`peers/__init__.py.tpl` is unchanged: it defines `MLPPeer`, which `peers.yaml`
names as `{{ project_name }}.peers.MLPPeer`.

### 8.5 `.gitignore`

The generated `.gitignore` gains two lines, because the run directory and the
teacher directory are new:

```
var/runs/
var/teachers/
var/audits/
var/cache/
var/state/
```

`var/` already covers all five. The explicit lines exist so a project that
un-ignores `var/logs` (to keep logs in the repo) does not accidentally commit
checkpoints or teacher weights.

### 8.6 The `next steps` output

`startproject.Command.handle` (`startproject.py:120`) prints the tree. It gains
one line pointing at the configuration:

```
created project 'demo_swarm' in /path/to/parent
  demo_swarm/manage.py
  ...
  demo_swarm/conf/conf.yaml
  demo_swarm/conf/trainer.yaml
  ...

next steps:
  cd demo_swarm
  edit conf/trainer.yaml to set epochs, batch size, and the task
  python manage.py check
  python manage.py migrate
  python manage.py run --epochs 3
```

## 9. Packaging

Three files change, and one test asserts each. The `.gitignore` is written inline
by `write_project` (`startproject.py:91`), not from a template, so adding lines to
it is a one-line edit there (section 8.5) rather than a new file.

### 9.1 `MANIFEST.in`

`recursive-include mactf *.tpl` (`MANIFEST.in:10`) already covers the new files.
No change.

### 9.2 `pyproject.toml`

The `[tool.setuptools.package-data]` table already declares `mactf = ["py.typed", "templates/**/*.tpl"]`.
No change. The new templates are `.tpl` under `mactf/templates`, which the glob
catches.

If the table lists patterns individually instead of the glob, it gains:

```toml
"templates/conf/*.tpl",
```

`tests/test_packaging.py:296` (`test_wheel_ships_py_typed_and_templates_and_no_tests`)
asserts `any(name.endswith(".tpl"))`, which passes either way; section 10 adds a
sharper assertion.

### 9.3 `mactf/settings_default.py`

The fallback module gains `CONF_DIR = None` and nothing else:

```python
# No project, so no conf/ directory. The implicit <root>/conf fallback in
# mactf.conf_files.resolve_conf_dir covers the case where a user runs from a
# directory that happens to have one.
CONF_DIR = None

# No overlays: a wheel install ships no YAML.
CONFIG_FILES: list[str] = []
```

`None` rather than absence, because `resolve_conf_dir` distinguishes "not
declared" (fall back to `<root>/conf`) from "declared as null" (fall back too).
Both are `None` here, and the distinction only matters for a project, so the
default module needs no special case.

## 10. Tests

Ten files, one per concern. Every test names the failure it catches.

| File | Covers | Oracle |
|---|---|---|
| `tests/test_conf_files.py` | `CONF_LAYOUT` is exhaustive and disjoint: every `ConfFile.owns` key appears in exactly one entry; `OWNED_KEYS == frozenset().union(*[e.owns for e in CONF_LAYOUT])`; `layout_by_filename` and `layout_by_template` round-trip; `owner_of` returns the file for every owned key and `None` for `SECRET_KEY` | exact set equality |
| `tests/test_conf_files.py` | `resolve_conf_dir`: explicit `CONF_DIR` wins; relative `CONF_DIR` resolves against root; a missing explicit `CONF_DIR` raises `ImproperlyConfigured`; an implicit `<root>/conf` with one YAML is found; a `<root>/conf` with no YAML returns `None`; no directory returns `None` | exact |
| `tests/test_conf_files.py` | `load_conf_tree`: merges in `CONF_LAYOUT` order, not filesystem order (create the files in reverse); an absent file contributes nothing and produces a note; an unknown file name raises with the name; a file writing a foreign key raises naming **both** files; a reserved key raises; `strict=False` notes instead of raising; a nested mapping deep-merges rather than replacing; `_CONF_NOTES` records the resolved directory | exact messages, `==` on merged dicts |
| `tests/test_conf_blocks.py` | every block in section 5 exists in `DEFAULTS` with the documented default; every default value equals the value in the corresponding template when parsed (a drift test: parse `mactf/templates/conf/*.tpl` with `yaml.safe_load` and compare leaf-for-leaf to the block) | `==` on every leaf |
| `tests/test_conf_blocks.py` | the `_fold_trainer` reconciliation: `TRAINER.EPOCHS: 8` makes `TRAINING.EPOCHS == 8`; a `TRAINER` key with no `TRAINING` counterpart stays in `TRAINER` only; no `TRAINER` leaves `TRAINING` untouched | `==` |
| `tests/test_conf_blocks.py` | validators: one test per `raise` in section 5.11, asserting the message includes the block and key; a valid project's `conf/` passes; `WATCHDOG.MAX_GRAD_NORMN` is refused (the typo case); `LOGGER.SINKS: [file]` without `LOG_FILE` is refused; `API.CORS_ORIGINS` and `CORS.ORIGINS` together are refused | exact messages |
| `tests/test_conf_equivalence.py` | **the gate**: a project with `settings.py` dicts and `configs/*.yaml` produces the same `settings.as_dict()` as one with the equivalent `conf/` tree, excluding `_CONF_NOTES`, `PATHS`, and `CONFIGS`. This is the migration gate. | `1e-12` on floats, `==` on everything else |
| `tests/test_conf_equivalence.py` | the trainers agree: a `MACTFTrainer` built from `conf/` and one built from `settings.py` produce identical `TrainSpec.as_dict()`, and one epoch at `SEED=1337` produces identical `final_score`, `kappa`, `entropy` | `1e-12` |
| `tests/test_conf_equivalence.py` | every leaf value in `DEFAULTS` is unchanged by this document: a snapshot dict in the test names each pre-existing key and its value, so a future edit to `DEFAULTS` fails here and must be argued in a diff | `==` |
| `tests/test_startproject_conf.py` | `LAYOUT` writes exactly 21 files plus `.gitignore`; every path is under `demo_swarm/`; `conf/` has 17 `*.yaml`; **no `configs/` directory exists**; `{{ project_name }}` appears in `settings.py`, `peers.yaml`, and `peers/__init__.py`, and nowhere else; each `conf/*.yaml` parses with `yaml.safe_load` and is a mapping | exact sets |
| `tests/test_startproject_conf.py` | the generated project loads: `syspath_prepend(tmp_path)`, `conf.load_settings("demo_swarm.settings")` succeeds, `loaded["CONF_DIR"]` is the `conf` dir, `_CONF_NOTES` mentions it, `PEERS` has 3 entries from `peers.yaml` (not from `settings.py`, which has none), and `TRAINING.EPOCHS == 3` arrives through the fold | exact |
| `tests/test_startproject_conf.py` | `LAYOUT` and `CONF_LAYOUT` agree: `{name.removeprefix("conf/") for _, name in LAYOUT if name.startswith("conf/")} == {entry.filename for entry in CONF_LAYOUT}`, and every template source exists (`template_root() / entry.template_name`) | exact set equality |
| `tests/test_conf_consumers.py` | `MACTFLogger.from_settings`: each sink in `SINKS` is constructed; order preserved; `memory` is present exactly once; `REDACT` replaces `DEFAULT_REDACT`; a `LOGGER`-less project falls back to `TELEMETRY` with `SINKS == ["console", "file"]` | exact sink `name` sequences |
| `tests/test_conf_consumers.py` | `MACTFCheckpoint.from_settings`: `DIR: null` resolves through `PATHS["CHECKPOINTS"]`; an explicit `DIR` wins; every retention kwarg lands on the attribute; `CHECKSUM: blake2b` is accepted and `md5` is not | exact |
| `tests/test_conf_consumers.py` | `MACTFWatchdog.from_settings`: policy fields land; a key the policy does not declare goes to the constructor; `on_critical: abort` produces `stops_training`; the unknown-key refusal fires | exact |
| `tests/test_conf_consumers.py` | `MACTFOptimizer.from_block`: `BETAS` becomes a tuple; `LR: null` is stripped so the backend default applies; `TYPE: scipy` with `METHOD` works; an unknown `CLASS` raises naming the class | exact |
| `tests/test_packaging.py` (extended) | the wheel contains `mactf/templates/conf/conf.yaml.tpl` and every other template in `CONF_LAYOUT`; the wheel contains no `mactf/templates/project/configs/` | exact |

The drift test in `test_conf_blocks.py` is the one that keeps this document
honest after it is implemented. The YAML in section 6 and the `DEFAULTS` blocks in
section 5 are two copies of the same numbers; parsing one and comparing to the
other means a change to either without the other fails, which is the only way a
17-file tree stays consistent with 8 dicts.

### 10.1 The CI gate

`.github/workflows/deploy.yml` gains one job, `conf-layout`, and one entry in
`build.needs`:

```yaml
  conf-layout:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - name: conf/ tree round-trips through startproject
        run: pytest tests/test_conf_files.py tests/test_conf_blocks.py tests/test_conf_equivalence.py tests/test_startproject_conf.py tests/test_conf_consumers.py -q
```

`tests/test_packaging.py:336` asserts every gate names a test file that exists, so
the job is checked by the suite it runs.

## 11. Failure modes this design refuses

| Failure | Detected by | Message |
|---|---|---|
| A file sets a block it does not own | `_ownership_problems` | `conf/logger.yaml sets TRAINING, which is owned by conf/trainer.yaml` |
| A file the loader never reads sits in `conf/` | `load_conf_tree` unknown-file check | `conf/ contains file(s) no loader reads: extra.yaml` |
| `SECRET_KEY` in a committed file | reserved-key check | `conf/conf.yaml sets SECRET_KEY, which only settings.py may set` |
| A typo in a watchdog threshold | `WATCHDOG` unknown-key check | `WATCHDOG names unknown key(s) {'max_grad_normn'}` |
| A sink named without its path | `LOGGER.SINKS` check | `LOGGER.SINKS includes 'file' but LOG_FILE is unset` |
| Two CORS authorities | `API`/`CORS` check | `API.CORS_ORIGINS and CORS.ORIGINS both set` |
| A template added to the tree but not packaged | `startproject`'s missing-template guard | `template 'conf/foo.yaml.tpl' is missing from mactf.templates.project` |
| A template packaged but not in `CONF_LAYOUT` | `test_startproject_conf.py` set equality | test failure naming the file |
| A `DEFAULTS` value drifting from the template | drift test | test failure naming the key path |
| An existing project breaking on upgrade | equivalence test | `1e-12` diff |

The shape is the same in every row: the failure is a load-time error naming a file
and a key, not a run-time surprise at epoch 9. That is the whole argument for
loading YAML through a validating loader instead of `yaml.safe_load` in each
consumer.

## 12. Sequencing

```
conf_files.py                (CONF_LAYOUT, resolve_conf_dir, load_conf_tree)
  -> conf.py: DEFAULTS blocks, _fold_trainer, _validate_blocks, load_settings hook
  -> tests/test_conf_files.py, tests/test_conf_blocks.py     [green before any template]
  -> mactf/templates/conf/*.yaml.tpl (17 files)
  -> tests/test_conf_blocks.py drift test                    [now meaningful]
  -> from_settings on Logger, Checkpoint, Watchdog, from_block on Optimizer
  -> consumer edits: C3, C8, C9, C10, C11 in section 7.2
  -> tests/test_conf_consumers.py
  -> startproject: LAYOUT from CONF_LAYOUT, settings.py.tpl shrink, configs/ delete
  -> tests/test_startproject_conf.py, extended test_packaging.py
  -> equivalence gate: tests/test_conf_equivalence.py         [the migration gate]
  -> CI job
```

`conf_files.py` first, because both the loader and the scaffolder read
`CONF_LAYOUT`, and writing it once means the template names and destinations are
one list from the start.

The drift test goes green *before* the templates exist is the wrong way to read
that order: the test is written with the blocks, and it fails until the templates
land. It is listed at the template step because that is when it stops failing.

The equivalence gate is last because it compares two whole configurations, and it
is only meaningful once both paths exist. It is the gate that matters: everything
before it adds a capability, and this one proves the capability did not change an
answer.

## 13. Cross-references

| Topic | Document |
|---|---|
| the settings contract this extends | `mactf/conf.py:1-8` |
| overlays and `CONFIG_FILES` | `docs/plans/13-app-integration.md` section on settings |
| logger, watchdog, checkpoint sections | `docs/plans/09-logging-watchdog-checkpoint.md` |
| trainer constructor arguments | `docs/plans/08-trainer-optimizer-strategies.md` section 5.1 |
| weight strategies and the fallback | `docs/plans/08-trainer-optimizer-strategies.md` section 6 |
| `SIGNAL_SOURCE` and normalization | `docs/plans/11-points-and-normalization.md` section 8 |
| the `DISTILLATION` block | `docs/plans.new/14-contrastive-distillation.md` section 10 |
| precedence tiers, non-offsettable rules | `docs/framework.md` sections 3 and 6.1 |
| `startproject` as shipped today | `mactf/management/commands/startproject.py` |
| packaging guarantees | `tests/test_packaging.py` |

## 14. Non-goals, restated

1. **No change to the resolution order.** `conf/` is step 2.5, between the settings
   module and `CONFIG_FILES`, so an overlay still wins. Section 4.4.
2. **No nested environment overrides.** `_apply_env` (`conf.py:368`) walks scalars
   only, unchanged. `MACTF_TRAINER` does nothing.
3. **No YAML for `INSTALLED_APPS`, `MIDDLEWARE`, `ROOT_URLCONF`, or
   `APPEND_SLASH`.** Apps are import-time; the other three are wired before a
   settings load finishes. `mactf.yaml` records them as comments.
4. **No schema library.** Validation is Python, beside the numeric bounds that
   already exist.
5. **No removal of `TRAINING`, `TELEMETRY`, or `CONFIG_FILES`.** Each is a live
   consumer's input, and removing it is a migration this document refuses.
6. **No migration tool.** An existing project keeps working; the equivalence gate
   is the proof. Converting is an edit, not a command.
7. **No change to any default value.** Section 10's third test asserts it, leaf by
   leaf.
8. **No secrets in `conf/`.** `SECRET_KEY` is reserved and `API.TOKENS` ships empty.
   A secret manager or an untracked overlay was already the mechanism.
9. **No second authority for `PEERS`.** `peers.yaml` wins when present;
   `settings.py` is the fallback for a project without the tree.
10. **No consumer for four shipped keys.** `ORCHESTRATOR.PAUSE_IS_FATAL`,
    `TRAINER.SHARING`, and the `DISTILLATION` wiring are validated and documented
    as unread, per section 7.3. A key with no reader is named in a table; a key
    omitted from the file is invisible.
