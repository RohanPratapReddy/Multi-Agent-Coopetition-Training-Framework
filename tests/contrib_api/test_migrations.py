"""../contrib api/test migrations (split from test_app_extensions.py)."""

from __future__ import annotations

import sys


import textwrap


import pytest


from mactf import apps as apps_module


from mactf import conf


from mactf.db import CORE_LABEL, MigrationFailed, Registry, migration_plan


from mactf.exceptions import CommandError, ImproperlyConfigured


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


def test_migration_plan_orders_core_first_then_apps_by_label():
    plan = migration_plan({"zeta": {1: ["SELECT 1"]}, "alpha": {2: ["SELECT 1"], 1: ["SELECT 1"]}})
    labels = [entry[0] for entry in plan]
    assert labels[0] == CORE_LABEL
    assert [(a, v) for a, v, _ in plan if a != CORE_LABEL] == [("alpha", 1), ("alpha", 2), ("zeta", 1)]
    # the same version in two apps is legal
    migration_plan({"a": {1: []}, "b": {1: []}})
    with pytest.raises(ImproperlyConfigured):
        migration_plan({"core": {9: []}})


def test_app_migrations_apply_after_core_and_are_recorded(demo_app, tmp_path):
    registry = Registry(tmp_path / "x.sqlite3")
    registry.migrate()
    done = registry.migrate_apps()
    assert done == [("demoapp", 1), ("demoapp", 2)]
    applied = registry.applied_migrations()
    assert (CORE_LABEL, 1) in applied and ("demoapp", 2) in applied
    assert registry.migrate_apps() == []  # idempotent
    with registry.connect() as connection:
        tables = {r["name"] for r in connection.execute("SELECT name FROM sqlite_master")}
    assert {"demo_a", "demo_b", "mactf_migrations"} <= tables


def test_failed_app_migration_rolls_back_that_app_only(tmp_path):
    registry = Registry(tmp_path / "y.sqlite3")
    registry.migrate()
    plan = migration_plan(
        {
            "good": {1: ["CREATE TABLE good_t (id INTEGER)"]},
            "zbad": {1: ["CREATE TABLE bad_t (id INTEGER)"], 2: ["THIS IS NOT SQL"]},
        }
    )
    with pytest.raises(MigrationFailed) as info:
        registry.migrate_apps(plan)
    assert info.value.app == "zbad"
    applied = registry.applied_migrations()
    assert ("good", 1) in applied
    assert not any(app == "zbad" for app, _ in applied)
    with registry.connect() as connection:
        tables = {r["name"] for r in connection.execute("SELECT name FROM sqlite_master")}
    assert "good_t" in tables and "bad_t" not in tables


def test_existing_database_is_not_remigrated(tmp_path):
    registry = Registry(tmp_path / "z.sqlite3")
    registry.migrate()
    assert registry.migrate() == []
    assert registry.applied_migrations() >= {(CORE_LABEL, 1), (CORE_LABEL, 3)}


def test_duplicate_version_in_app_list_is_rejected(settings):
    from mactf.apps import AppConfig

    class Dup(AppConfig):
        name = "mactf.contrib.telemetry"
        migrations_module = "dup_migrations"

    module = type(sys)("dup_migrations")
    module.operations = [(1, ["SELECT 1"]), (1, ["SELECT 2"])]
    sys.modules["dup_migrations"] = module
    try:
        with pytest.raises(ImproperlyConfigured):
            Dup("mactf.contrib.telemetry").get_migrations()
    finally:
        del sys.modules["dup_migrations"]


def test_the_api_app_declares_no_migration_and_adds_no_table(settings, tmp_path):
    settings["INSTALLED_APPS"] = ["mactf.contrib.api"]
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    try:
        config = apps_module.apps.get_app_config("mactf_api")
        assert config.get_migrations() == {}
        plan = apps_module.migration_plan() if hasattr(apps_module, "migration_plan") else None
        if plan is not None:
            assert {label for label, _, _ in plan} <= {"core"}
    finally:
        apps_module.reset_registry()
