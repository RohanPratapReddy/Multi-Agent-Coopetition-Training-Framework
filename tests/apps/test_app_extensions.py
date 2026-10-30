"""Doc 13 section 2: AppConfig extensions, command discovery, app migrations."""

from __future__ import annotations

import sys


import textwrap


import pytest


from mactf import apps as apps_module


from mactf import conf


from mactf.exceptions import CommandError, ImproperlyConfigured


from mactf.management import base


def _write(root, relative, text=""):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(text), encoding="utf-8")


@pytest.fixture()
def demo_app(tmp_path, monkeypatch, settings):
    pkg = tmp_path / "pkgs"
    _write(pkg, "demoapp/__init__.py")
    _write(pkg, "demoapp/urls.py", "from mactf.http import path\napp_name='demo'\nurlpatterns=[path('x', lambda r: None, name='x')]\n")
    _write(
        pkg,
        "demoapp/migrations.py",
        "operations = {1: ['CREATE TABLE demo_a (id INTEGER)'], 2: ['CREATE TABLE demo_b (id INTEGER)']}\n",
    )
    _write(pkg, "demoapp/management/__init__.py")
    _write(pkg, "demoapp/management/commands/__init__.py")
    _write(
        pkg,
        "demoapp/management/commands/hello.py",
        "from mactf.management.base import BaseCommand\nclass Command(BaseCommand):\n    help='hi'\n    def handle(self, **o):\n        return 0\n",
    )
    monkeypatch.syspath_prepend(str(pkg))
    settings["INSTALLED_APPS"] = ["demoapp"]
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    yield "demoapp"
    apps_module.reset_registry()
    for name in [m for m in sys.modules if m == "demoapp" or m.startswith("demoapp.")]:
        del sys.modules[name]


def test_declared_attributes_have_conventional_defaults(demo_app):
    config = apps_module.apps.get_app_config("demoapp")
    assert [p.name for p in config.get_urls()] == ["x"]
    assert config.get_urls()[0] is config.get_urls()[0]  # same cached objects
    assert config.get_migrations() == {
        1: ["CREATE TABLE demo_a (id INTEGER)"],
        2: ["CREATE TABLE demo_b (id INTEGER)"],
    }
    assert config.get_commands() == "demoapp.management.commands"


def test_command_packages_include_apps_and_core_alias(demo_app):
    packages = base.command_packages()
    assert packages[0] == "mactf.management.commands"
    assert "demoapp.management.commands" in packages
    assert base.COMMAND_PACKAGES == base.CORE_COMMAND_PACKAGES
    assert base.load_command("hello").help == "hi"
    assert "hello" in base.get_commands()


def test_command_name_collision_is_reported(demo_app, tmp_path, monkeypatch):
    pkg = tmp_path / "pkgs"
    _write(pkg, "otherapp/__init__.py")
    _write(pkg, "otherapp/management/__init__.py")
    _write(pkg, "otherapp/management/commands/__init__.py")
    _write(
        pkg,
        "otherapp/management/commands/hello.py",
        "from mactf.management.base import BaseCommand\nclass Command(BaseCommand):\n    def handle(self, **o):\n        return 0\n",
    )
    cfg = conf.get_settings()
    cfg["INSTALLED_APPS"] = ["demoapp", "otherapp"]
    apps_module.reset_registry()
    apps_module.populate()
    with pytest.raises(ImproperlyConfigured, match="hello"):
        base.get_commands()
    for name in [m for m in sys.modules if m.startswith("otherapp")]:
        del sys.modules[name]
