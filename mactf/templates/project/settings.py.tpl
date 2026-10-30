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
