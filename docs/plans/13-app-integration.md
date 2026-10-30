---
title: App Integration
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0
owner: TBD
last_reviewed: 2026-10-21
---

# 13. App Integration

Document 12 defines the transport. This document defines how the service layer
becomes part of the framework: the app control the framework already has, the
`mactf.contrib.api` app that uses it, the `mactf/client` package that speaks to
it, and the `mactf.testing.Client` that tests it without a socket.

The claim this document has to make good on is narrow and testable: **removing
`mactf.contrib.api` from `INSTALLED_APPS` removes the HTTP API and breaks
nothing else.** That is what `django.contrib.admin` does and what the previous
`mactf/api/` package could not do, because it was imported by the entry point
rather than selected by settings.

## 1. What already exists

These are the controls this document extends. They are not new.

| Control | File | What it does today |
|---|---|---|
| `AppConfig` | [mactf/apps.py:17](mactf/apps.py#L17) | `name`, `label`, `verbose_name`, `path`, `ready()`, `import_models()`, `register_peer`, `register_provider`, `register_system_check` |
| `AppRegistry` | [mactf/apps.py:79](mactf/apps.py#L79) | `populate()`, `set_ready()`, `get_app_configs()`, `get_containing_app_config()` |
| `import_models` submodules | [mactf/apps.py:44](mactf/apps.py#L44) | imports `models`, `peers`, `providers`, `systems`, `receivers` |
| Checks | [mactf/checks.py:67](mactf/checks.py#L67) | `register(check, app=...)`, `run_checks()` |
| Signals | [mactf/signals.py](mactf/signals.py) | 14 signals including `app_ready` |
| Commands | [mactf/management/base.py:17](mactf/management/base.py#L17) | `COMMAND_PACKAGES` hardcoded to `mactf.management.commands`; `load_command`, `get_commands` |
| Migrations | [mactf/db.py:20](mactf/db.py#L20) | `SCHEMA_VERSION`, `MIGRATIONS: dict[int, list[str]]`, `Registry.migrate()` |
| Settings | [mactf/conf.py:27](mactf/conf.py#L27) | `DEFAULTS`; `load_settings` drops any project key absent from it |

The last row is the sharpest constraint in this document.
`load_settings` copies `DEFAULTS`, then merges only uppercase module attributes
whose key is already in `DEFAULTS` ([mactf/conf.py:346](mactf/conf.py#L346)). A
setting that is not declared there is silently ignored — not an error, not a
warning, just absent. So every key section 3 introduces **must** be added to
`DEFAULTS` in the same change that documents it, or a project's
`ROOT_URLCONF = "project.urls"` does nothing and the app serves the wrong conf.

## 2. `AppConfig` extensions

Four declarative attributes and three methods. All are optional; an app that
declares none behaves exactly as it does today.

```python
class AppConfig:
    name: str = ""
    label: str | None = None
    verbose_name: str | None = None
    path: str | None = None

    # -- new, all optional ----------------------------------------------
    urls_module: str | None = None
    migrations_module: str | None = None
    checks_module: str | None = None
    commands_module: str | None = None

    def get_urls(self) -> list[URLPattern | URLResolver]: ...
    def get_migrations(self) -> dict[int, list[str]]: ...
    def get_commands(self) -> str | None: ...
```

| Attribute | Default | Meaning |
|---|---|---|
| `urls_module` | `None` | submodule holding `urlpatterns`; `None` means `f"{name}.urls"` is tried and absence is fine |
| `migrations_module` | `None` | submodule holding `operations`; `None` means `f"{name}.migrations"` is tried |
| `checks_module` | `None` | submodule whose import registers checks; `None` means `f"{name}.systems"`, which is what `mactf.contrib.scoring` already uses |
| `commands_module` | `None` | package of command modules; `None` means `f"{name}.management.commands"` |

`import_models()` gains three submodule names, in this order, after the existing
five:

```python
for submodule in (
    "models", "peers", "providers", "systems", "receivers",   # existing
    "views", "serializers", "urls",                           # new
):
```

The order matters: `urls` is imported last because it imports `views`, and a
`views` module that imports `serializers` should fail with its own `ImportError`
name rather than through the URL conf. Import failures are still filtered by the
existing rule — a `ModuleNotFoundError` whose `exc.name` equals the module being
imported means "absent, skip"; anything else propagates. That rule is what lets
`mactf/http` stay importable with no app installed.

`get_urls()` returns the app's patterns:

```python
def get_urls(self):
    if self.urls_module is None:
        return []
    module = importlib.import_module(self.urls_module)
    return list(getattr(module, "urlpatterns", []))
```

The returned patterns carry the module's `app_name` in the `URLResolver` that
`include()` builds, so `reverse("api:rule-list")` works with no explicit
`namespace=` at the mount site. An app that wants a second, independent namespace
under the same prefix declares `namespace=` on its own `include()`; the api app
does not, because one namespace per app is the point.

`get_migrations()` returns `{version: [sql, ...]}` from
`migrations_module.operations`, or `{}`. `get_commands()` returns the command
package name.

Three properties these methods must keep:

- **No side effects at import.** `get_urls()` imports a module that declares
  data. A module that opens a database connection at import time is a bug this
  contract forbids.
- **Idempotent.** Calling `get_urls()` twice returns patterns from the same cached
  module object. It does not construct new pattern objects, so `reverse()` and
  `resolve()` see stable identities.
- **No settings reads.** The app describes what it has. Whether the URL conf
  mounts it is `ROOT_URLCONF`'s decision, and whether migrations run is the
  behind-`migrate` decision. An app that reads `DEBUG` to decide its own URL
  shape makes the conf untestable.

### 2.1 Command discovery from apps

`COMMAND_PACKAGES` is a tuple constant, and `load_command`/`get_commands` iterate
it ([mactf/management/base.py:93](mactf/management/base.py#L93) and
[:114](mactf/management/base.py#L114)). It becomes a function with the same
observable behaviour for a bare install:

```python
CORE_COMMAND_PACKAGES = ("mactf.management.commands",)

def command_packages() -> tuple[str, ...]:
    packages = list(CORE_COMMAND_PACKAGES)
    from .. import apps
    for config in apps.apps.get_app_configs():
        package = config.get_commands()
        if package and package not in packages:
            packages.append(package)
    return tuple(packages)
```

- `COMMAND_PACKAGES` stays as a module-level alias for
  `CORE_COMMAND_PACKAGES`, because existing tests and third-party code read it.
  It is documented as the core set only.
- `command_packages()` calls `get_app_configs()`, which raises
  `AppRegistryNotReady` before `populate()`. `load_command` therefore populates on
  demand when the registry is not ready, which is exactly what
  `execute_from_command_line` already does before dispatch
  ([mactf/management/base.py:140](mactf/management/base.py#L140)).
- Command name collisions across apps are an `ImproperlyConfigured` raised by
  `get_commands()`, listing both apps. `load_command` keeps first-match-wins
  order for a single-name lookup, because a user typing `mactf serve` should get
  the app's `serve` even if a future core command shares the name — but
  `get_commands()` must not silently hide one.
- `mactf.contrib.api` ships `serve` this way. There is no `serve` in core, and a
  bare install therefore has no `mactf serve`, which is correct: without the app
  there is nothing to serve.

### 2.2 App-scoped migrations

`MIGRATIONS` is a single `dict[int, list[str]]`, and `validate_settings` /
`migrate` assume one global sequence. Two apps adding to it would collide on
version numbers.

The fix keeps the existing table and adds an app table:

```python
# mactf/db.py
SCHEMA_VERSION = 1                    # core, unchanged
MIGRATIONS: dict[int, list[str]] = {...}   # core, unchanged, label "core"

CORE_LABEL = "core"

def migration_plan() -> list[tuple[str, int, list[str]]]:
    """(app_label, version, statements) in apply order."""
```

Schema addition (migration 1 is untouched; a later migration adds it):

```sql
CREATE TABLE IF NOT EXISTS mactf_migrations (
    app        TEXT NOT NULL,
    version    INTEGER NOT NULL,
    applied_at REAL NOT NULL,
    PRIMARY KEY (app, version)
)
```

Rules:

- **Core is an app named `core`.** Its existing rows are recorded in
  `mactf_schema` and stay there; `current_version` keeps reading `mactf_schema`
  so an existing database is not re-migrated. `mactf_migrations` is additional
  bookkeeping and is seeded for `core` at the current `mactf_schema` version on
  first use.
- **Order is `core` first, then apps by label.** Deterministic, and it means a
  core schema change can never land after an app that depends on it.
- **Within an app, versions are integers and applied in ascending order.**
  Starting at `1`. No timestamps in names; a label plus a small integer is
  readable in a `SELECT` and stable in a diff.
- **Duplicate `(app, version)` across two apps is legal**, because the table's
  key includes the app. Duplicate versions *within* an app raise
  `ImproperlyConfigured` when `migration_plan()` builds the list.
- **`mactf migrate` prints the plan** as `app/version` lines before applying, and
  applies each app's statements in one transaction. A failure rolls back that app
  only and reports which app failed.
- **A migration statement list is SQL text.** No Python callbacks, no data
  migrations beyond SQL. The framework's own migrations already work this way,
  and a callback would make the plan untestable without executing it.

`mactf.contrib.api` needs **no** migration. Nothing in document 02 persists
through the app; all persistence is core's (`runs`, `episodes`,
`rule_events`, and the tables document 04.4 adds). The app mechanism exists so a
third-party app can persist, and so the framework is not forced to keep every
table in `mactf/db.py` forever.

### 2.3 Checks from apps

`mactf.contrib.scoring` declares `Checks` through a `systems` module imported in
`ready()` ([mactf/contrib/scoring/apps.py](mactf/contrib/scoring/apps.py),
[mactf/contrib/scoring/systems.py](mactf/contrib/scoring/systems.py)). That is
already the right pattern and this document does not change it.

Two additions:

- `checks_module` lets an app name its check registrar explicitly instead of
  relying on the `systems` convention. `None` keeps the convention, so no existing
  app changes.
- `run_checks()` callers gain nothing new. The `check` command iterates registered
  checks and prints `CheckMessage.id`; an app that registers `mactf.api.E001`
  shows up there with no change to the command.

`mactf.contrib.api` registers four checks in
`mactf/contrib/api/checks.py`:

| ID | Level | Condition |
|---|---|---|
| `mactf.api.E001` | ERROR | `API.ALLOW_ANONYMOUS` is false and `API.TOKENS` is empty — every request would be unauthenticated and the API is unprotected |
| `mactf.api.W002` | WARNING | `ROOT_URLCONF` is `None`, or it is set to a module whose `urlpatterns` does not `include()` `mactf.contrib.api.urls` — the app is loaded but no route is mounted, which is almost always a project mistake |
| `mactf.api.E003` | ERROR | a route `name` in `mactf.contrib.api.urls` is duplicated — `reverse()` would be ambiguous and the OpenAPI `operationId` would collide |
| `mactf.api.E004` | ERROR | an `API.TOKENS` entry grants a scope that is not in the `SCOPES` vocabulary in `mactf/contrib/api/auth.py`, or its value is not a `sha256:`/`plain:` prefixed digest — a typo in a scope is a startup error, not a silent `403` at request time |

`W002` checks the composed conf, not just the settings key, because the common
mistake is a correct `ROOT_URLCONF` that forgot one `include()`. It resolves the
URL conf and tests membership with `resolve(reverse("api:rule-list"))`, which is
cheap and exercises the same machinery a request would.

`E001` mirrors `check_settings_consistency` in
[mactf/contrib/scoring/apps.py:32](mactf/contrib/scoring/apps.py#L32): a settings
combination that produces a silently insecure deployment is an error, not a
warning. `mactf serve` refuses to start when `run_checks()` reports an error and
`--skip-checks` was not passed, matching `requires_system_checks` on the existing
`run` command ([mactf/management/commands/run.py:10](mactf/management/commands/run.py#L10)).

## 3. New settings

Added to `DEFAULTS` in [mactf/conf.py:27](mactf/conf.py#L27). Every one is
required for the reason stated in section 1.

```python
"DEFAULT_AUTO_FIELD": "mactf.db.BigAutoField",
"APPEND_SLASH": True,
"ROOT_URLCONF": None,
"MIDDLEWARE": [
    "mactf.http.middleware.RequestIdMiddleware",
    "mactf.http.middleware.ExceptionMiddleware",
],
"HTTP": {
    "MAX_BODY_BYTES": 8 * 1024 * 1024,
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
```

| Key | Why it is in `DEFAULTS` |
|---|---|
| `DEFAULT_AUTO_FIELD` | `startproject` writes it; without a default, a project that omits it and an app that reads it disagree |
| `APPEND_SLASH` | read by the resolver on every request; absent means the resolver invents a policy |
| `ROOT_URLCONF` | `None` is meaningful (framework default conf) and must be settable |
| `MIDDLEWARE` | the handler builds the chain from it; the default is the minimum that produces problem+json |
| `HTTP.MAX_BODY_BYTES` | a project must be able to raise it for a large evaluation payload |
| `HTTP.STREAM_WORKERS`, `HTTP.SSE_HEARTBEAT` | `mactf serve` reads them at startup; without defaults a host serving many streams could not raise the pool |
| `API.*` | read by `mactf.contrib.api`; documented in document 02 section 4.4 |

`API.TOKENS` is `{}` and `API.ALLOW_ANONYMOUS` is `True` by default, which is the
development posture and is exactly what `mactf.api.E001` guards against being
carried into production unchanged. That pairing is deliberate: the default is
convenient, the check makes shipping the default loud.

`validate_settings` gains five checks:

- `ROOT_URLCONF` is `None` or a non-empty dotted string.
- `MIDDLEWARE` is a list of dotted strings; an entry is not resolved at settings
  time (that would import the world during `mactf check`), but a non-list is an
  `ImproperlyConfigured`.
- `API.BASE_PATH` has no leading or trailing slash, so route mounting is
  unambiguous.
- `HTTP.MAX_BODY_BYTES` is a positive int, and `HTTP.STREAM_WORKERS` is a positive
  int no greater than 256.
- `HTTP.SSE_HEARTBEAT` is a positive float; `0` would defeat the purpose, so it is
  rejected rather than treated as "no heartbeat".

`CONFIG_FILES` YAML overlays apply to these keys like any other, because
`_deep_merge` operates over `DEFAULTS` after module values. A deployment that
sets tokens in YAML rather than in a settings module therefore works, with the
usual caveat that the YAML file must be readable by the process only.

## 4. `mactf/contrib/api/`

The app. Its shape is what `mactf.contrib.scoring` already establishes: an
`apps.py` with a config class whose `ready()` imports registration modules.

```
mactf/contrib/api/
  __init__.py
  apps.py                 ApiConfig(AppConfig)
  urls.py                 urlpatterns with a name for every route
  views/
    __init__.py
    meta.py               health, meta, rules, validation, severities, tiers,
                          score scale, system checks, openapi.json
    runs.py               create, list, get, cancel, summary
    episodes.py           episodes, peers, rule-events, rule-counts,
                          aggregations, checkpoints, rollback, warnings,
                          points, normalization
    streams.py            SSE
    evaluations.py        evaluations, ledger, flags, adjudications
    peers.py              peers, contract, weight preview
    sandbox.py            providers, status, probe, exec
    scorers.py            scorers, trainers, capabilities, preview
    hierarchy.py          tree, node, infer
    observability.py      logs, watchdog, snapshot, verify, resume
    admin.py              config reload, capabilities
  serializers.py          every Serializer subclass
  auth.py                 Principal, bearer parsing, scope table
  checks.py               mactf.api.E001, W002, E003, E004
  systems.py              imports checks.py so ready() registers them
  management/
    __init__.py
    commands/
      __init__.py
      serve.py            wsgiref server over get_wsgi_application()
```

`apps.py`:

```python
from ...apps import AppConfig
from ...http import register_handler, HttpError

class ApiConfig(AppConfig):
    name = "mactf.contrib.api"
    label = "mactf_api"
    verbose_name = "MACTF HTTP API"
    urls_module = "mactf.contrib.api.urls"
    checks_module = "mactf.contrib.api.systems"

    def ready(self) -> None:
        from . import systems          # noqa: F401  registers checks
        from . import receivers        # noqa: F401  app_ready -> bind RunManager
```

- `label` is `mactf_api`, not `api`, following the existing
  `mactf_scoring`/`mactf_coopetition` convention, so a user app called `api` does
  not collide.
- `ready()` does not build the application. It registers checks and connects
  signal receivers. `get_wsgi_application()` builds the handler lazily on first
  call, because an app that constructs a `RunManager` at `ready()` would start
  reaping orphan runs during `mactf check`.
- Removing the app is removing two lines: the `INSTALLED_APPS` entry and the
  `include()` in a project's `urls.py`. `mactf run`, `mactf check`, `mactf
  migrate`, scoring, training, and the client all still work.

`urls.py` names every route. The names are the contract, not the paths, and
document 02 section 3 carries them in a `name` column:

```python
from ...http import path

urlpatterns = [
    path("health", views.meta.health, name="health"),
    path("meta", views.meta.meta, name="meta"),
    path("rules", views.meta.rules, name="rule-list"),
    path("rules/{rule_id}", views.meta.rule_detail, name="rule-detail"),
    path("scores/scale", views.meta.score_scale, name="score-scale"),
    path("runs", views.runs.collection, name="run-list"),
    path("runs/{run_id}", views.runs.detail, name="run-detail"),
    path("runs/{run_id}/cancel", views.runs.cancel, name="run-cancel"),
    path("runs/{run_id}/events", views.streams.events, name="run-events"),
    path("runs/{run_id}/episodes/{episode:int}",
         views.episodes.detail, name="run-episode-detail"),
    # ... one entry per row of document 02 section 3
]
```

`name` values are kebab-case and stable. `tests/contrib_api/test_route_names.py`
asserts that every name resolves through `reverse()` and that the set equals the
names in document 02's table, which is the same completeness idea as document 02
section 5 applied to names instead of paths.

### 4.1 What `ready()` must not do

- It must not mount routes. Mounting is `ROOT_URLCONF`'s job, and an app that
  appended to a global list would be indistinguishable from a package import.
- It must not read `API.TOKENS` and cache it. A token rotated in settings requires
  a restart, and a cached table would make it require a restart *and* a cache
  clear; `auth.py` reads settings per request.
- It must not create the database. `mactf migrate` does that. An app whose
  `ready()` writes to sqlite makes `mactf check` mutate state.

## 5. `from_dict` on framework dataclasses

The client returns framework objects (document 03). For that to be possible, the
objects must be constructible from the JSON the API emits. `to_dict()` already
exists; `from_dict()` is the missing half.

Classes gaining a `from_dict` classmethod:

| Class | File | Notes |
|---|---|---|
| `PeerScorecard` | [mactf/scoring/scorecard.py:22](mactf/scoring/scorecard.py#L22) | reconstructs `ContinuousScore` through its own `from_dict` |
| `EpisodeScorecard` | [mactf/scoring/scorecard.py:98](mactf/scoring/scorecard.py#L98) | peers keyed by name |
| `AggregationReport` | [mactf/coopetition/aggregation.py:32](mactf/coopetition/aggregation.py#L32) | tuples rebuilt from lists |
| `CapabilityReport` | [mactf/sandbox/base.py:61](mactf/sandbox/base.py#L61) | capability flags preserved |
| `ExecResult` | [mactf/sandbox/base.py:138](mactf/sandbox/base.py#L138) | |
| `ComplianceResult` | [mactf/compliance.py:31](mactf/compliance.py#L31) | |
| `PointsBreakdown` | `mactf/scoring/points.py` (document 11) | nested `RulePoints` and `DenialRecord` |
| `NormalizationReport` | `mactf/scoring/normalize.py` (document 11) | nested `CohortStats`, `NormalizedScore` |
| `ContinuousScore` | [mactf/scoring/continuous.py:91](mactf/scoring/continuous.py#L91) | required by `PeerScorecard.from_dict` |

Rules for every `from_dict`:

1. **Exact inverse of `to_dict`.** `cls.from_dict(obj.to_dict()) == obj` for every
   field, with `dataclasses.asdict` equality. This is a test, not an aspiration:
   `tests/scoring/test_dict_roundtrip.py` parametrizes over the classes above and
   fails on the first differing field, printed as `Class.field: sent != received`.
2. **Tolerates unknown keys.** A response from a newer server may carry a field
   this version does not know. `from_dict` ignores it. It must not raise, because
   a client that cannot read a newer response is a client that cannot be upgraded
   independently of the server.
3. **Is strict about missing required keys.** A missing non-defaulted field raises
   `KeyError` naming the class and field. Silently defaulting a missing
   `final_score` to `0.0` would turn a truncated response into a zero score.
4. **Rebuilds containers.** A JSON list becomes whatever the dataclass declares:
   `list`, `tuple`, or `dict` keyed by peer. A `frozen=True` dataclass keeps its
   immutability.
5. **No defaults invented for `None`-meaningful fields.** `floor_reason=None` and
   `floor_reason=""` stay distinguishable; `points=None` and
   `PointsBreakdown(0.0)` stay distinguishable (document 03 section 4 states the
   same rule from the client side).

`to_dict` gains nothing. It already emits what `from_dict` reads. If a field
appears in one and not the other, rule 1 fails; that is the whole enforcement.

### 5.1 Why not a generic `dataclass_from_dict`

A generic helper that inspects `__dataclass_fields__` and recurses works for
flat dataclasses and fails for every interesting case here: `PeerScorecard`'s
`continuous` is a class with its own dict protocol, `EpisodeScorecard.peers` is a
dict of objects, `severity` is an `IntEnum` that serializes as a label,
`warnings_issued` is a tuple of `WarningRecord` that serializes through
`describe()`. A generic helper would need a registry of converters, at which
point it is a hand-written `from_dict` with extra indirection and worse error
messages.

Also, the reverse direction is not symmetric: `warnings_issued` serializes to
`list[str]` through `describe()`, but `PeerScorecard.warnings_issued` is a tuple
of `WarningRecord`. `from_dict` therefore cannot reconstruct a `WarningRecord` from
a string, and does not try: it stores the strings, because the JSON is the
authority on what the server observed. That asymmetry is documented in the
class docstring where it applies, and the round-trip test for `PeerScorecard`
compares the *serialized* form for that field rather than the object identity.

## 6. `mactf/client/`

The client lives in core, not in a separate distribution, because it dispatches
through `mactf.http` and returns `mactf` dataclasses. Splitting it out would
force it to re-declare both.

```
mactf/client/
  __init__.py        MACTFClient
  client.py          MACTFClient and its namespaces
  base.py            Transport protocol, and the transport envelopes
                     RunHandle, Event, Page, Run, Episode, RuleEvent
  transport.py       InProcessTransport, HTTPTransport
  errors.py          MACTFClientError hierarchy
  pagination.py      Page iterator
  specs.py           dict builders for TrainerSpec and friends
```

`Run`, `Episode`, and `RuleEvent` are defined in `base.py` because they are what
the routes return, not what the engine computes; document 03 section 6.1 states
why each is not the same-named engine class.

### 6.1 `Transport`

```python
class Transport(Protocol):
    def request(self, method: str, path: str, *,
                json: Any | None = None,
                params: Mapping[str, Any] | None = None,
                headers: Mapping[str, str] | None = None) -> Response: ...
    def stream(self, method: str, path: str, *,
               headers: Mapping[str, str] | None = None) -> Iterator[Event]: ...
```

Two implementations, one request object, one response object. `stream()` yields
parsed `Event` objects rather than raw bytes: SSE framing is parsed once, in the
transport, because the in-process path produces the same frames a socket would and
document 03's parity suite compares them.

The `Transport` protocol is stated normatively in
[03-python-sdk.md](03-python-sdk.md) section 4; this is the same declaration,
repeated because this is where the two implementations live.

### 6.2 `InProcessTransport`

```python
class InProcessTransport:
    def __init__(self, application=None) -> None:
        self._application = application    # resolved lazily

    def request(self, method, path, **kwargs) -> Response:
        environ = self._environ(method, path, **kwargs)
        return dispatch_wsgi(self._app(), environ)
```

- It **builds a real WSGI `environ`** and calls `get_wsgi_application()` directly.
  Everything that a socket request goes through happens: `resolve()`, the
  middleware chain, auth, the view, serialization. There is no shortcut that
  calls a view function with a dict.
- `wsgi.input` is a `io.BytesIO` over the JSON body, `CONTENT_LENGTH` is set from
  it, and `wsgi.errors` is a `io.StringIO` so a handler logging an error does not
  print into a notebook.
- `SCRIPT_NAME` is `""`, `SERVER_NAME`/`SERVER_PORT` are `localhost`/`80`,
  `SERVER_PROTOCOL` is `HTTP/1.1`, and `wsgi.url_scheme` is `http`. This is the
  minimum a WSGI app may rely on; a view that reads another key gets `None` and
  is a bug in that view.
- No socket is opened, so it works in a test, in a notebook, and in a process
  where binding a port is forbidden.
- `mactf.testing.Client` uses the same two helpers: `dispatch_wsgi` and
  `build_environ` from `mactf/http/testing.py`. That module is the one place a
  WSGI `environ` is built and the one place response bytes are assembled, so the
  transport and the test client cannot disagree about how a request is shaped or
  a response is read. Re-implementing either is the failure mode section 8's last
  row guards against.

The consequence worth stating: the in-process path serializes and then parses
JSON, exactly like the socket path. It would be faster to hand the dataclass
straight back. It would also mean the in-process client never exercised a
serializer, and the equivalence obligation in section 8 would be a promise
instead of a test.

### 6.3 `HTTPTransport`

```python
class HTTPTransport:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", *,
                 token: str | None = None,
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 backoff: float = 0.5) -> None: ...
```

- **`urllib.request` only.** `urllib.request.urlopen` with a `Request` built from
  the method, URL, headers, and body. No `httpx`, no `requests`, no `aiohttp`.
- Base URL normalization: strip a trailing `/`. The prefix is resolved once by
  probing `GET {base}/api/v1/health`, falling back to `GET {base}/health`. A
  failure at construction is a `TransportError` with the attempted URLs, because
  a client that retries forever against a wrong prefix is worse than one that
  reports the prefix it tried.
- `Authorization: Bearer <token>` when a token is given.
- Retries on `429` and `502`/`503`/`504`, with `backoff * 2**attempt` and full
  jitter, honouring `Retry-After` when present. Never on a `POST` without an
  `Idempotency-Key`; the transport generates a `uuid4` key itself for `POST
  /runs`, so a retry is safe by default.
- `stream()` uses `urlopen`'s file object and an SSE frame parser over
  `readline()`. Reconnect with `Last-Event-ID` up to `max_retries`, then raise
  `TransportError`. A `stream_lost` event raises `TransportError` carrying the
  oldest available id rather than skipping silently.
- Problem responses are converted by status first, `type` second, into the same
  exception classes `InProcessTransport` raises (section 6.5).

`AsyncHTTPTransport` is **not** provided. There is no async client because there
is no async server; a caller in an async context runs the client in a
`ThreadPoolExecutor`, which is what `mactf.http/asgi.py` does in the other
direction. Document 03 section 6 states this as a deliberate omission.

### 6.4 `MACTFClient`

```python
class MACTFClient:
    def __init__(self, *, transport=None, settings=None,
                 base_url: str = "http://127.0.0.1:8000",
                 token: str | None = None) -> None: ...

    runs: RunsAPI
    scoring: ScoringAPI
    rules: RulesAPI
    peers: PeersAPI
    sandbox: SandboxAPI
    trainers: TrainersAPI
    scores: ScoresAPI
    observability: ObservabilityAPI
```

Construction rules:

- `transport=None` and no `base_url` means `InProcessTransport()`: the default is
  local, because the common case is a script in the same process as the engine.
- `base_url` given means `HTTPTransport(base_url, token=token)`.
- An explicit transport wins over both, which is how a test injects a recording
  transport.
- `settings` is passed to `InProcessTransport` only, through
  `mactf.conf.configure` semantics: the client does **not** mutate global
  settings. A test constructs a `Settings` and a client together; the client uses
  it to build the application.

Namespaces are thin facades that call the transport and deserialize:

```python
class RunsAPI:
    def create(self, request: Mapping[str, Any]) -> RunHandle: ...
    def list(self, *, limit: int = 50, offset: int = 0) -> Page[Run]: ...
    def get(self, run_id: str) -> Run: ...
    def cancel(self, run_id: str) -> RunHandle: ...
    def stream(self, run_id: str, *, last_event_id: int | None = None) -> Iterator[Event]: ...
```

`RunHandle` and `Event` are the only two client-side dataclasses that have no
framework counterpart. `RunHandle` is `{"run_id", "status", "links"}` — a
transport envelope, not a domain object. `Event` is one SSE frame. Everything
else is a framework type: `Page` wraps them, and the `Page` itself is the only
generic container.

### 6.5 Errors

```python
class MACTFClientError(Exception):
    def __init__(self, message, *, status=None, type_uri=None,
                 trace_id=None, extensions=None): ...

class ValidationError(MACTFClientError): ...      # 422
class NotFound(MACTFClientError): ...             # 404
class Conflict(MACTFClientError): ...             # 409
class Unauthorized(MACTFClientError): ...         # 401
class Forbidden(MACTFClientError): ...            # 403
class RateLimited(MACTFClientError): ...          # 429, exposes retry_after
class ServerError(MACTFClientError): ...          # 5xx
class CapabilityUnavailable(MACTFClientError): ...# provider cannot do it
class TransportError(MACTFClientError): ...       # no connection, timeout, bad frame
```

- `HTTPTransport` maps problem+json by `status` then `type`.
- `InProcessTransport` maps identically, because the response it receives is the
  same problem+json the handler produced. There is no separate exception
  translation for the local path — that is the point of routing it through WSGI.
- `CapabilityUnavailable` is raised by the client only when the server says so
  (a `502` with `urn:mactf:error:sandbox`), or by the local path when the view
  raised `SandboxUnavailable`. It is never inferred client-side from a missing
  extra, because the client cannot know what is installed on the host.

### 6.6 Deliberate omissions

| Omitted | Reason |
|---|---|
| A parallel model family | The framework classes are the models. `from_dict` closes the round trip; a second family would drift (document 00 P2) |
| `httpx` | `urllib.request` is stdlib and is enough for a synchronous client |
| An async client | No async server to talk to. A caller in an async context uses a thread |
| Auto-generated stubs from OpenAPI | The protocol is hand-written so `InProcessTransport` can satisfy it. A generated client can only speak HTTP |
| A CLI | `mactf` already is the CLI and stays canonical |
| Client-side scoring | A second rubric is the failure mode document 00 P10 forbids |

## 7. `mactf.testing.Client`

The Django analogue, and the replacement for `fastapi.testclient.TestClient` in
the fixtures.

```python
class Client:
    def __init__(self, *, urlconf: str | None = None,
                 settings: Settings | None = None,
                 raise_request_exception: bool = False) -> None: ...

    def get(self, path, data=None, **extra) -> TestResponse: ...
    def post(self, path, data=None, content_type="application/json", **extra) -> TestResponse: ...
    def put(self, path, data=None, **extra) -> TestResponse: ...
    def patch(self, path, data=None, **extra) -> TestResponse: ...
    def delete(self, path, data=None, **extra) -> TestResponse: ...

class TestResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes
    def json(self) -> Any: ...
    def __getitem__(self, header: str) -> str: ...
```

Rules:

- It builds a WSGI `environ` and calls `get_wsgi_application()`. No server, no
  port, no thread. A test suite of 400 cases runs in the time the old plan spent
  starting uvicorn once.
- `urlconf` lets a test mount a tiny conf with one route, which is how
  `tests/http/` tests a view without the whole API.
- `raise_request_exception=False` is the default: a `500` is returned as a
  response, because most tests assert on problem bodies. Setting it to `True`
  re-raises the original exception, which is what a test debugging a handler
  wants.
- `data=` is serialized to JSON when it is not `bytes` or `str`, and
  `content_type` defaults to `application/json` for that reason. Form-encoded
  bodies are sent by passing `data=` with
  `content_type="application/x-www-form-urlencoded"`.
- It **does not** share state between requests. No cookies are persisted; a test
  that needs a token passes it per request. Cookie support would be a small
  addition that makes a test depend on ordering, which is the thing a test
  client should not encourage.
- `mactf/testing/__init__.py` exports it, so `from mactf.testing import Client` is
  the import in every test. That module currently exports the harness
  ([mactf/testing/__init__.py](mactf/testing/__init__.py)) and gains one name.

`tests/testing/test_client.py` asserts that a `TestResponse` from
`mactf.testing.Client` and a `Response` from `InProcessTransport` for the same
request have identical status, headers, and body bytes. They share
`dispatch_wsgi`, so this is a regression test rather than a second
implementation.

## 8. The equivalence obligation

Stated once, testable, and the reason the two transports exist:

> For any request the API accepts, `InProcessTransport` and `HTTPTransport`
> produce the same framework object, and neither path can produce an object the
> other cannot.

What that means concretely:

| Property | How it is tested |
|---|---|
| Same status and body bytes for the same request | the fixtures run both transports against one parametrized request list |
| Same framework object after `from_dict` | each response is deserialized and compared field by field, not by dict equality |
| Neither transport can reach a route the other cannot | the route list is derived from `ROOT_URLCONF`, which both use |
| Errors map to the same exception classes | a `422`, `404`, `409`, and `502` case per transport |
| The local path exercises serializers | a test replaces a serializer's `to_representation` with a sentinel and asserts it is called on the in-process path |

The last row is the one that keeps the obligation from decaying. The tempting
optimization is to have `InProcessTransport` return the view's object directly;
that would pass every equality test and silently stop testing every serializer.
A sentinel call test fails the moment someone adds the shortcut.

Two asymmetries are allowed and must be documented where they occur:

1. **A `callable` spec** that names a module path is accepted by the in-process
   transport, because the module is already in the caller's process, and rejected
   by `HTTPTransport`'s server side as remote code execution (document 03 section
   10). The client raises `ValidationError` with a message naming the reason.
2. **Latency and concurrency limits.** `InProcessTransport` has no
   `MAX_CONCURRENT_RUNS` semaphore, because the caller can see the thread; the
   HTTP path enforces it. Behaviour after the limit is reached is the one
   observable difference, and the server returns `429` where the local path
   simply starts the run.

Every other difference is a bug.

## 9. Failure modes this document prevents

| Failure | Prevented by |
|---|---|
| A project sets `ROOT_URLCONF` and it is silently ignored | section 1: the key is in `DEFAULTS`, and section 3 lists why |
| Removing `mactf.contrib.api` breaks `mactf run` | the app registers nothing on the run path; criterion 47 tests it |
| Two apps both add migration `2` and one silently does not run | `(app, version)` key, section 2.2 |
| An app's `serve` command is invisible to `mactf help` | `command_packages()`, section 2.1 |
| The client returns its own `Scorecard` and drifts from the server's | `from_dict` on framework classes, section 5 |
| The in-process path is faster but tests nothing | it goes through WSGI and serializers, section 6.2 |
| A test needs a port and a server thread | `mactf.testing.Client`, section 7 |
| `POST /runs` is retried into two runs | `Idempotency-Key` generated by the transport, section 6.3 |
| A 500 returns an HTML traceback to a client | `ExceptionMiddleware` plus the handler's last-resort catch, document 12 section 8 |

## 10. Acceptance criteria for this document

| # | Criterion |
|---|---|
| 47 | `INSTALLED_APPS` without `mactf.contrib.api` leaves `mactf run`, `mactf check`, `mactf migrate`, and `MACTFClient()` working, and `mactf serve` absent with a helpful error |
| 48 | `mactf serve` is discovered through `command_packages()` from the app, not from a core module |
| 49 | A project's `ROOT_URLCONF`, `MIDDLEWARE`, `APPEND_SLASH`, `HTTP.*`, and `API.*` overrides take effect, proving they are in `DEFAULTS` |
| 50 | `mactf.contrib.api` registers `mactf.api.E001`, `W002`, `E003`, and `E004`, and `mactf check` reports them |
| 51 | An app declaring `migrations_module` has its statements applied after core's, recorded in `mactf_migrations` |
| 52 | `from_dict(to_dict(x)) == x` for every class in section 5, and `from_dict` ignores an unknown key |
| 53 | `InProcessTransport` and `HTTPTransport` return identical framework objects for the same request, including identical exception classes for a `422`, `404`, `409`, and `502` |
| 54 | `mactf.testing.Client` serves a request with no socket, and its response bytes equal those of `InProcessTransport` |

Criteria 47-54 plus 39-46 from document 12 are the release gate in document 06.
Criterion 47 is the one that decides whether the layer is integrated or merely
co-located, which is the whole purpose of this pass.
