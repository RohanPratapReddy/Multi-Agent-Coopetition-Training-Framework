"""Built-in default settings so a bare ``pip install mactf`` works.

Django requires a project before anything runs; MACTF does not. The management
commands that only inspect the installation (``check``, ``doctor``, ``version``,
``startproject``) must work against the distribution alone, so this module is the
fallback settings module when ``MACTF_SETTINGS_MODULE`` names nothing.

It is deliberately minimal: every value not set here comes from
:data:`mactf.conf.DEFAULTS`, which is validated by
:func:`mactf.conf.validate_settings`. Real training needs a project with peers,
which ``mactf startproject`` writes.
"""

from __future__ import annotations

from pathlib import Path

# A default module has no project directory; resolve relative paths from the
# process working directory, which is what ``manage.py`` would do anyway.
BASE_DIR = Path.cwd()

DEBUG = False

# The framework's own apps provide scoring, aggregation, sandbox, and telemetry
# system checks. Peer apps are added by a generated project.
INSTALLED_APPS = [
    "mactf.contrib.scoring",
    "mactf.contrib.coopetition",
    "mactf.contrib.sandbox",
    "mactf.contrib.telemetry",
    "mactf.contrib.components",
]

# No peers until a project declares them. ``check`` reports this as a warning
# rather than an error, because the framework itself is peer-agnostic.
PEERS: list[dict] = []

DEVICE = "auto"
DTYPE = "auto"
AMP = True

# No project, so no conf/ directory. The implicit <root>/conf fallback in
# mactf.conf_files.resolve_conf_dir covers the case where a user runs from a
# directory that happens to have one.
CONF_DIR = None

# No overlays: a wheel install ships no YAML.
CONFIG_FILES: list[str] = []
