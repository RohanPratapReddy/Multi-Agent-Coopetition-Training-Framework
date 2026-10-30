"""Fixtures for the training tests that register components (document 17 section 19.2)."""

from __future__ import annotations

import pytest

from mactf.components import HUB, overlay


@pytest.fixture
def component_overlay():
    """Register freely; the process registry is checked identical afterwards."""
    import mactf.training.peer_factories  # noqa: F401  (the built-ins must be in the snapshot)

    before = HUB.snapshot()
    with overlay() as scope:
        yield scope
    assert HUB.snapshot() == before, "a test leaked a registration into the process registry"
