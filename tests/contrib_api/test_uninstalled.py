"""../contrib api/test uninstalled (split from test_app_extensions.py)."""

from __future__ import annotations

import pytest


from mactf import apps as apps_module


from mactf import conf


from mactf.exceptions import CommandError, ImproperlyConfigured


from mactf.management import base


def test_app_without_contributions_returns_empty(settings):
    settings["INSTALLED_APPS"] = ["mactf.contrib.telemetry"]
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    config = apps_module.apps.get_app_config("mactf_telemetry")
    assert config.get_urls() == []
    assert config.get_migrations() == {}
    assert config.get_commands() is None


def test_unknown_serve_has_a_helpful_error(settings):
    settings["INSTALLED_APPS"] = []
    conf._settings = settings
    apps_module.reset_registry()
    with pytest.raises(CommandError, match="INSTALLED_APPS"):
        base.load_command("serve")


@pytest.fixture()
def bare_settings(settings):
    settings["INSTALLED_APPS"] = []
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    yield settings
    apps_module.reset_registry()


def test_core_commands_exist_and_serve_does_not(bare_settings):
    commands = set(base.get_commands())
    assert {"run", "check", "migrate", "doctor", "inspect"} <= commands
    assert "serve" not in commands


def test_check_and_migrate_run_without_the_app(bare_settings, capsys):
    assert base.execute_from_command_line(["manage.py", "check"]) == 0
    assert base.execute_from_command_line(["manage.py", "migrate"]) == 0
    assert "applied migrations" in capsys.readouterr().out


def test_run_works_without_the_app(bare_settings, capsys):
    from api_peers import ApiTinyPeer

    bare_settings["PEERS"] = [
        {"name": "p0", "class": "ApiTinyPeer", "kwargs": {}},
        {"name": "p1", "class": "ApiTinyPeer", "kwargs": {}},
    ]
    bare_settings["TRAINING"].update({"STATE_DIM": 4, "ACTION_DIM": 3})
    apps_module.get_peer_registry().register(ApiTinyPeer)
    code = base.execute_from_command_line(
        ["manage.py", "run", "--dry-run", "--epochs", "1", "--json"]
    )
    assert code == 0
    assert "dry_run" in capsys.readouterr().out


def test_the_serve_error_names_the_app_to_install(bare_settings):
    with pytest.raises(CommandError) as raised:
        base.load_command("serve")
    assert "mactf.contrib.api" in str(raised.value)


def test_the_client_constructs_and_reports_an_unmounted_api_honestly(bare_settings):
    from mactf.client import HTTPTransport, MACTFClient, NotFound, TransportError

    client = MACTFClient()
    with pytest.raises(NotFound):
        client.health()
    with pytest.raises(TransportError, match="no MACTF API answered"):
        MACTFClient(base_url="http://127.0.0.1:9", token="x")
    assert HTTPTransport is not None


def test_the_http_package_imports_without_the_app(bare_settings):
    import mactf.http  # noqa: F401
    from mactf.http import urls

    assert urls.default_urlpatterns() == []
