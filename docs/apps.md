# Apps

Capabilities beyond the core are installed apps, selected the way `django.contrib.admin` is: list
them in `INSTALLED_APPS`. The core runs without any of them.

| App | Adds |
|---|---|
| `mactf.contrib.scoring` | the scoring pipeline and its checks |
| `mactf.contrib.coopetition` | the aggregator and pool |
| `mactf.contrib.sandbox` | sandbox providers and the `sandbox` command |
| `mactf.contrib.telemetry` | logging and watchdog settings |
| `mactf.contrib.api` | the HTTP API, its checks, and the `serve` command |
| `mactf.distributed` | the coordinator and worker protocol (`DISTRIBUTED` settings) |

## Install and remove the API

```python
INSTALLED_APPS = [
    "mactf.contrib.scoring", "mactf.contrib.coopetition",
    "mactf.contrib.api",
]
```

Remove the line and the routes, `serve`, and the `mactf.api.*` checks disappear. Nothing else
changes, which `tests/contrib_api` and `tests/http/test_layering.py` enforce: the core never
imports `mactf.http`, `mactf.contrib.api` or `mactf.client`.

## What an app declares

An `AppConfig` subclass in `apps.py` (named in `default_app_config` or the module's only
config) may declare `name`, `label`, `verbose_name`, `ready()`, and system checks. The registry
populates configs in order, calls `ready()` once, and rejects duplicate labels with a clear error.
Apps may also ship `management/commands/*.py`; a command is discovered only from installed apps.

## How `urls.py` mounts it

`ROOT_URLCONF = None` (the default) means "mount each installed app's `urls.py`". The API app
contributes its routes under `API.BASE_PATH` (default `api/v1`); `mactf.distributed` contributes
`DISTRIBUTED.BASE_PATH` (default `worker/v1`) only when the API is also installed. To control
mounting yourself, set a root URLconf and use `include`:

```python
# project/urls.py
from mactf.http.urls import include, path

urlpatterns = [path("api/v1/", include("mactf.contrib.api.urls"))]
```

Route names are unique per app namespace (`api:run-detail`); `reverse()` and the OpenAPI
`operationId` use the same string. `run-list` is one name for both `GET` and `POST` on the same
path, so `operationId` is unique per path rather than globally.
