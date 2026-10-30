"""Fixtures for the transport tests that need the API app installed."""

from __future__ import annotations

import pytest

from mactf import apps as apps_module
from mactf import conf
from mactf.http import urls as http_urls
from mactf.http.handler import clear_application_cache


@pytest.fixture()
def api_settings_for_urls(settings):
    settings["INSTALLED_APPS"] = [*settings.get("INSTALLED_APPS", []), "mactf.contrib.api"]
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    http_urls.clear_url_caches()
    clear_application_cache()
    yield settings
    clear_application_cache()
    apps_module.reset_registry()
    http_urls.clear_url_caches()
