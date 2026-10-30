"""Criterion 50: E001, W002, E003 and E004 are registered, reported and fixable."""

from __future__ import annotations

import sys
import types

import pytest

from mactf.checks import run_checks
from mactf.http import path
from mactf.http import urls as http_urls


def _ids(result):
    return [message.id for message in result.messages]


def _find(result, check_id):
    return [message for message in result.messages if message.id == check_id]


def test_a_healthy_installation_reports_none_of_them(api_settings):
    result = run_checks()
    assert not [i for i in _ids(result) if i.startswith("mactf.api.")], result.messages


def test_e001_when_nothing_can_authenticate(api_settings):
    api_settings["API"] = {**api_settings["API"], "ALLOW_ANONYMOUS": False, "TOKENS": {}}
    result = run_checks()
    found = _find(result, "mactf.api.E001")
    assert len(found) == 1
    assert found[0].level == "ERROR"
    assert "API.TOKENS" in found[0].hint
    assert not result.ok


def test_e001_is_cleared_by_a_token_or_a_resolver(api_settings):
    api_settings["API"] = {**api_settings["API"], "ALLOW_ANONYMOUS": False, "TOKENS": {"t": ["read"]}}
    assert not _find(run_checks(), "mactf.api.E001")
    api_settings["API"] = {
        **api_settings["API"], "TOKENS": {}, "PRINCIPAL_RESOLVER": "pkg.mod:resolver",
    }
    assert not _find(run_checks(), "mactf.api.E001")


def test_w002_when_the_routes_are_not_mounted(api_settings):
    module = types.ModuleType("check_empty_urlconf")
    module.urlpatterns = []
    sys.modules["check_empty_urlconf"] = module
    try:
        api_settings["ROOT_URLCONF"] = "check_empty_urlconf"
        http_urls.clear_url_caches()
        result = run_checks()
        found = _find(result, "mactf.api.W002")
        assert len(found) == 1 and found[0].level == "WARNING"
        assert "ROOT_URLCONF" in found[0].hint
        assert result.ok, "a missing mount is a warning, not an error"
    finally:
        sys.modules.pop("check_empty_urlconf", None)
        api_settings["ROOT_URLCONF"] = None
        http_urls.clear_url_caches()


def test_e003_on_a_duplicate_route_name(api_settings, monkeypatch):
    from mactf.contrib.api import urls

    duplicate = path("again", lambda request: None, name=urls.urlpatterns[0].name)
    monkeypatch.setattr(urls, "urlpatterns", [*urls.urlpatterns, duplicate])
    found = _find(run_checks(), "mactf.api.E003")
    assert [message.message for message in found] == [
        f"duplicate route name {duplicate.name!r} in mactf.contrib.api.urls"
    ]
    assert found[0].level == "ERROR"


@pytest.mark.parametrize(
    "tokens, fragment",
    [
        ({"t": ["read", "teleport"]}, "unknown scope"),
        ({"sha256:abc": ["read"]}, "64 hex"),
        ({"plain:": ["read"]}, "empty"),
    ],
)
def test_e004_on_a_bad_token_table(api_settings, tokens, fragment):
    api_settings["API"] = {**api_settings["API"], "TOKENS": tokens}
    found = _find(run_checks(), "mactf.api.E004")
    assert found and fragment in found[0].message
    assert "worker" in found[0].hint


def test_the_worker_scope_is_a_valid_scope(api_settings):
    api_settings["API"] = {**api_settings["API"], "TOKENS": {"w": ["worker"]}}
    assert not _find(run_checks(), "mactf.api.E004")


def test_the_checks_do_not_run_when_the_app_is_not_installed(settings):
    from mactf import apps as apps_module
    from mactf import conf

    settings["INSTALLED_APPS"] = []
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    assert not [i for i in _ids(run_checks()) if i.startswith("mactf.api.")]
