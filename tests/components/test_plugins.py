"""Plugin loading: entry points and ``COMPONENTS.MODULES`` (document 17 sections 4.2, 15.5)."""

from __future__ import annotations

import sys
import types

import pytest

from component_helpers import OtherPolicy, Policy
from mactf.components import HUB, HubPolicy, Origin, load_components, plugins, policy_from_settings
from mactf.exceptions import (
    ComponentBuildError,
    ComponentCollision,
    ComponentLocked,
    ImproperlyConfigured,
)

POLICIES = "trainer_policy"


@pytest.fixture(autouse=True)
def _clean(restore_hub):
    plugins.reset_for_tests()
    yield
    plugins.reset_for_tests()


class FakeEntry:
    """An ``importlib.metadata.EntryPoint`` stand-in."""

    def __init__(self, name, target, value=None):
        self.name = name
        self.value = value or f"fake.{name}:install"
        self._target = target
        self.loaded = 0

    def load(self):
        self.loaded += 1
        if isinstance(self._target, BaseException):
            raise self._target
        return self._target


@pytest.fixture
def entry_points(monkeypatch):
    registered: list[FakeEntry] = []

    def fake(group=None, **kwargs):
        assert group == plugins.ENTRY_POINT_GROUP == "mactf.components"
        return list(registered)

    monkeypatch.setattr(plugins.metadata, "entry_points", fake)
    return registered


@pytest.fixture
def modules(monkeypatch):
    """Importable fake modules; each import is counted, and may register on import."""
    imports: list[str] = []
    made: dict[str, types.ModuleType] = {}

    def add(name, register=None, raises=None):
        module = types.ModuleType(name)
        made[name] = module
        real = plugins.importlib.import_module

        def fake_import(requested, *a, **k):
            if requested == name:
                imports.append(name)
                if raises is not None:
                    raise raises
                if register is not None:
                    register()
                return module
            return real(requested, *a, **k)

        monkeypatch.setattr(plugins.importlib, "import_module", fake_import)
        return module

    add.imports = imports
    return add


# -- policy_from_settings -----------------------------------------------------------
def test_defaults_build_the_default_policy():
    from mactf.conf import COMPONENTS_DEFAULTS

    assert policy_from_settings(COMPONENTS_DEFAULTS) == HubPolicy()
    assert policy_from_settings({}) == HubPolicy()


def test_every_key_reaches_the_policy():
    policy = policy_from_settings({
        "ALLOW_BUILTIN_OVERRIDE": True, "LOCKED": ["scorer:penalty", "slot:X"], "STRICT_LEGACY": True,
        "ALLOW_IMPORT": True, "IMPORT_ALLOWLIST": ["acme.*"], "MAX_PER_KIND": 7, "STRICT_RESTORE": True,
    })
    assert policy == HubPolicy(
        allow_builtin_override=True, locked=frozenset({"scorer:penalty", "slot:X"}),
        strict_legacy=True, allow_import=True, import_allowlist=("acme.*",),
        max_components_per_kind=7, strict_restore=True,
    )


def test_strict_restore_reaches_the_hub_policy(modules):
    load_components({"STRICT_RESTORE": True, "PLUGINS": "off"})
    assert HUB.policy.strict_restore is True
    assert getattr(HUB.effective_policy(), "strict_restore", False) is True
    load_components({"PLUGINS": "off"})
    assert HUB.policy.strict_restore is False


def test_the_alias_sets_allow_import_and_the_canonical_key_wins():
    assert policy_from_settings({"API_ALLOW_IMPORT": True}).allow_import is True
    assert policy_from_settings({"API_ALLOW_IMPORT": True, "ALLOW_IMPORT": False}).allow_import is False


def test_locks_are_left_out_of_the_phase_one_policy():
    block = {"LOCKED": ["scorer:*"]}
    assert policy_from_settings(block, locked=False).locked == frozenset()
    assert policy_from_settings(block, locked=True).locked == frozenset({"scorer:*"})


# -- COMPONENTS.MODULES ---------------------------------------------------------------------
def test_modules_are_imported_once_in_order_with_project_origin(modules, entry_points):
    order = []
    modules("zz_first", register=lambda: (order.append(1), HUB.registry(POLICIES).register("zz_first", Policy)))
    # a second module through a second patch of import_module
    modules("zz_second", register=lambda: (order.append(2), HUB.registry(POLICIES).register("zz_second", OtherPolicy)))
    report = load_components({"MODULES": ["zz_first", "zz_second"], "PLUGINS": "off"})
    assert report.modules == ["zz_first", "zz_second"] and report.plugins == []
    assert HUB.registry(POLICIES).get("zz_first").origin is Origin.PROJECT
    assert HUB.registry(POLICIES).get("zz_second").origin is Origin.PROJECT
    assert order == [1, 2]


def test_loading_twice_imports_nothing_twice(modules, entry_points):
    modules("zz_once", register=lambda: HUB.registry(POLICIES).register("zz_once", Policy))
    block = {"MODULES": ["zz_once"], "PLUGINS": "off"}
    load_components(block)
    report = load_components(block)  # a second call with the same settings
    assert modules.imports == ["zz_once"] and report.modules == ["zz_once"]


def test_a_missing_module_is_improperly_configured(entry_points):
    with pytest.raises(ImproperlyConfigured, match="could not import module 'zz_no_such_module_xx'"):
        load_components({"MODULES": ["zz_no_such_module_xx"], "PLUGINS": "off"})


def test_a_module_that_raises_is_e008_with_the_cause(modules, entry_points):
    modules("zz_boom", raises=RuntimeError("module exploded"))
    with pytest.raises(ComponentBuildError) as caught:
        load_components({"MODULES": ["zz_boom"], "PLUGINS": "off"})
    assert caught.value.code == "E008" and isinstance(caught.value.cause, RuntimeError)
    assert "module exploded" in str(caught.value)


def test_a_component_error_inside_a_module_propagates_unwrapped(modules, entry_points):
    def collide():
        HUB.registry(POLICIES).register("zz_dup", Policy)
        HUB.registry(POLICIES).register("zz_dup", Policy)

    modules("zz_dupmod", register=collide)
    with pytest.raises(ComponentCollision):
        load_components({"MODULES": ["zz_dupmod"], "PLUGINS": "off"})


def test_a_failed_load_puts_the_previous_policy_back(modules, entry_points):
    HUB.configure(HubPolicy(max_components_per_kind=33))
    modules("zz_boom2", raises=RuntimeError("x"))
    with pytest.raises(ComponentBuildError):
        load_components({"MODULES": ["zz_boom2"], "ALLOW_BUILTIN_OVERRIDE": True, "PLUGINS": "off"})
    assert HUB.policy == HubPolicy(max_components_per_kind=33)


# -- entry points ----------------------------------------------------------------------------
def test_an_entry_point_install_function_registers_with_plugin_origin(entry_points):
    def install(hub):
        assert hub is HUB
        hub.registry(POLICIES).register("zz_plug", Policy)

    entry = FakeEntry("mactf-extras", install, "mactf_extras.components:install")
    entry_points.append(entry)
    report = load_components({"PLUGINS": "auto"})
    assert report.plugins == ["mactf-extras"] and report.skipped == []
    component = HUB.registry(POLICIES).get("zz_plug")
    assert component.origin is Origin.PLUGIN
    assert entry.loaded == 1


def test_an_entry_point_that_is_a_module_registers_on_import_only(entry_points):
    entry = FakeEntry("modulestyle", types.ModuleType("m"))  # not callable: nothing to call
    entry_points.append(entry)
    assert load_components({"PLUGINS": "auto"}).plugins == ["modulestyle"]


def test_entry_points_load_once_per_process(entry_points):
    calls = []
    entry_points.append(FakeEntry("e", lambda hub: calls.append(1)))
    load_components({"PLUGINS": "auto"})
    load_components({"PLUGINS": "auto"})
    assert calls == [1]


@pytest.mark.parametrize(
    "setting, expected_loaded, expected_skipped",
    [
        ("auto", ["a", "b"], []),
        ("AUTO", ["a", "b"], []),
        ("off", [], ["a", "b"]),
        (["b"], ["b"], ["a"]),
        ([], [], ["a", "b"]),
        (("a", "b"), ["a", "b"], []),
        (["nonexistent"], [], ["a", "b"]),
    ],
)
def test_plugins_forms(entry_points, setting, expected_loaded, expected_skipped):
    entry_points.extend([FakeEntry("a", lambda hub: None), FakeEntry("b", lambda hub: None)])
    report = load_components({"PLUGINS": setting})
    assert report.plugins == expected_loaded and report.skipped == expected_skipped
    assert [e.loaded for e in entry_points] == [1 if e.name in expected_loaded else 0 for e in entry_points]


def test_a_bad_plugins_setting_is_e015_and_changes_nothing(entry_points):
    before = HUB.policy
    with pytest.raises(ImproperlyConfigured, match="E015"):
        load_components({"PLUGINS": "sometimes"})
    assert HUB.policy == before


def test_a_failing_entry_point_is_e008_naming_the_plugin(entry_points):
    entry_points.append(FakeEntry("broken", ImportError("missing dependency")))
    with pytest.raises(ComponentBuildError, match="could not build plugin 'broken'"):
        load_components({"PLUGINS": "auto"})
    entry_points.clear()
    entry_points.append(FakeEntry("raises", lambda hub: 1 / 0))
    with pytest.raises(ComponentBuildError) as caught:
        load_components({"PLUGINS": "auto"})
    assert isinstance(caught.value.cause, ZeroDivisionError)


def test_a_plugin_cannot_shadow_a_project_component_but_a_project_can_replace_a_plugin(entry_points, modules):
    def install(hub):
        hub.registry(POLICIES).register("zz_shared", Policy)

    entry_points.append(FakeEntry("p", install))
    load_components({"PLUGINS": "auto"})
    assert HUB.registry(POLICIES).get("zz_shared").origin is Origin.PLUGIN
    # project over plugin: allowed (rank 2 >= 1)
    HUB.registry(POLICIES).register("zz_shared", OtherPolicy, replace=True)
    assert HUB.registry(POLICIES).get("zz_shared").origin is Origin.PROJECT
    # plugin over project: refused (rank 1 < 2)
    with pytest.raises(ComponentCollision, match="cannot replace a component from origin project"):
        HUB.registry(POLICIES).register("zz_shared", Policy, replace=True, origin=Origin.PLUGIN)


# -- phases and freezing -------------------------------------------------------------------------
def test_locks_apply_after_registration_and_the_freeze_after_that(modules, entry_points):
    modules("zz_lockmod", register=lambda: HUB.registry(POLICIES).register("zz_pinned", Policy))
    report = load_components({
        "MODULES": ["zz_lockmod"], "PLUGINS": "off", "LOCKED": ["trainer_policy:zz_*"],
        "FREEZE_AFTER_LOAD": True,
    })
    assert report.policy.locked == frozenset({"trainer_policy:zz_*"})
    assert HUB.registry(POLICIES).frozen and HUB.registry("scorer").frozen
    with pytest.raises(ComponentLocked):
        HUB.registry(POLICIES).register("zz_other", Policy)
    # a run still brings its own into an overlay, except where locked
    with HUB.scoped():
        HUB.registry(POLICIES).register("free_name", Policy, scope="run")
        with pytest.raises(ComponentLocked):
            HUB.registry(POLICIES).register("zz_other", Policy, scope="run")


def test_reloading_a_changed_block_starts_from_open_registries(modules, entry_points):
    load_components({"FREEZE_AFTER_LOAD": True, "PLUGINS": "off"})
    assert HUB.registry(POLICIES).frozen
    modules("zz_after", register=lambda: HUB.registry(POLICIES).register("zz_after", Policy))
    load_components({"MODULES": ["zz_after"], "PLUGINS": "off"})
    assert not HUB.registry(POLICIES).frozen and HUB.registry(POLICIES).has("zz_after")


# -- ensure_components_loaded ----------------------------------------------------------------------
class Settings(dict):
    pass


def test_ensure_components_loaded_is_idempotent(modules, entry_points):
    modules("zz_ens", register=lambda: HUB.registry(POLICIES).register("zz_ens", Policy))
    settings = Settings(COMPONENTS={"MODULES": ["zz_ens"], "PLUGINS": "off"})
    first = plugins.ensure_components_loaded(settings)
    second = plugins.ensure_components_loaded(settings)
    assert first is second and modules.imports == ["zz_ens"]


def test_a_changed_block_reloads(modules, entry_points):
    modules("zz_one", register=lambda: HUB.registry(POLICIES).register("zz_one", Policy))
    plugins.ensure_components_loaded(Settings(COMPONENTS={"MODULES": ["zz_one"], "PLUGINS": "off"}))
    report = plugins.ensure_components_loaded(
        Settings(COMPONENTS={"MODULES": ["zz_one"], "PLUGINS": "off", "MAX_PER_KIND": 9})
    )
    assert HUB.policy.max_components_per_kind == 9 and report.modules == ["zz_one"]
    assert modules.imports == ["zz_one"]  # the module itself is still imported once


def test_a_block_written_in_another_key_order_is_the_same_block(entry_points):
    a = Settings(COMPONENTS={"PLUGINS": "off", "MAX_PER_KIND": 5})
    b = Settings(COMPONENTS={"MAX_PER_KIND": 5, "PLUGINS": "off"})
    assert plugins.ensure_components_loaded(a) is plugins.ensure_components_loaded(b)


def test_ensure_components_loaded_without_a_block_loads_the_defaults(entry_points):
    report = plugins.ensure_components_loaded(Settings())
    assert report.policy == HubPolicy() and report.plugins == []


def test_ensure_components_loaded_reads_the_configured_settings(settings, entry_points):
    report = plugins.ensure_components_loaded()
    assert report.policy == HubPolicy()


def test_reset_for_tests_forgets_what_was_loaded(modules, entry_points):
    modules("zz_reset", register=lambda: None)
    load_components({"MODULES": ["zz_reset"], "PLUGINS": "off", "MAX_PER_KIND": 5})
    plugins.reset_for_tests()
    assert HUB.policy == HubPolicy() and plugins._LOADED == {} and plugins._LAST_BLOCK is None
    load_components({"MODULES": ["zz_reset"], "PLUGINS": "off"})
    assert modules.imports == ["zz_reset", "zz_reset"]


# -- the block is read the way conf validates it ----------------------------------------------
def test_a_block_with_lower_case_keys_and_the_alias_reaches_the_policy():
    policy = policy_from_settings({"max_per_kind": 7, "api_allow_import": True, "locked": ["scorer:x"]})
    assert policy == HubPolicy(max_components_per_kind=7, allow_import=True,
                               locked=frozenset({"scorer:x"}))
    assert plugins.fold_block({"Modules": ["a"], "API_ALLOW_IMPORT": 1}) == {"MODULES": ["a"], "ALLOW_IMPORT": 1}
    assert plugins.fold_block({"ALLOW_IMPORT": 2, "API_ALLOW_IMPORT": 1}) == {"ALLOW_IMPORT": 2}
    assert plugins.fold_block({"API_ALLOW_IMPORT": 1, "ALLOW_IMPORT": 2}) == {"ALLOW_IMPORT": 2}
    assert plugins.fold_block(None) == {}


def test_load_components_reads_lower_case_keys_too(modules, entry_points):
    modules("zz_lc", register=lambda: HUB.registry(POLICIES).register("zz_lc", Policy))
    report = load_components({"modules": ["zz_lc"], "plugins": "off", "freeze_after_load": True})
    assert report.modules == ["zz_lc"] and HUB.registry(POLICIES).frozen
