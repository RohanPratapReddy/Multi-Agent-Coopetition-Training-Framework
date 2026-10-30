"""Fixtures for the component-model tests (document 17 section 19.2)."""

from __future__ import annotations

import pytest

from mactf.components import HUB, Hub, overlay
from mactf.components.registry import HubPolicy


@pytest.fixture
def component_overlay():
    """Register freely; the process registry is checked identical afterwards."""
    before = HUB.snapshot()
    with overlay() as scope:
        yield scope
    assert HUB.snapshot() == before, "a test leaked a registration into the process registry"


@pytest.fixture
def hub() -> Hub:
    """A private hub: a test may register, replace, lock and freeze without touching ``HUB``."""
    return Hub()


@pytest.fixture
def policy_for():
    """Build a :class:`HubPolicy` from keyword flags, for ``hub.configure``."""

    def make(**flags):
        return HubPolicy(**flags)

    return make


@pytest.fixture
def policy_classes():
    from component_helpers import OtherPolicy, Policy

    return Policy, OtherPolicy


@pytest.fixture
def restore_hub():
    """Snapshot and restore the process hub exactly: registries, mirrors, freeze, policy.

    For the tests that exercise global behaviour (plugin loading, freezing, the legacy
    tables). It loads every kind's built-ins first so they are part of the snapshot.
    """
    from mactf.components import plugins

    HUB.snapshot()  # force-load every kind's built-ins
    saved = {
        kind: (
            dict(registry._entries),
            registry._frozen,
            dict(registry._builtin_factories),
            dict(registry._mirror) if registry._mirror is not None else None,
        )
        for kind, registry in HUB._registries.items()
    }
    policy = HUB.policy
    loaded = dict(plugins._LOADED)
    last = (plugins._LAST_BLOCK, plugins._LAST_REPORT)
    yield HUB
    for kind, (entries, frozen, builtins, mirror) in saved.items():
        registry = HUB._registries[kind]
        registry._entries = entries
        registry._frozen = frozen
        registry._builtin_factories = builtins
        if registry._mirror is not None and mirror is not None:
            dict.clear(registry._mirror)
            dict.update(registry._mirror, mirror)
    HUB.configure(policy)
    plugins._LOADED.clear()
    plugins._LOADED.update(loaded)
    plugins._LAST_BLOCK, plugins._LAST_REPORT = last
