"""Loading the shipped worked examples (document 17 section 18) into a test's overlay."""

from __future__ import annotations

import copy
import importlib
import sys
from typing import Any

from mactf.contrib.components.examples import EXAMPLES, load_conf


def load_example(number: str):
    """Import the example's module afresh, so its decorators register into the active overlay."""
    name = EXAMPLES[number]["module"]
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def fragment(number: str) -> dict[str, Any]:
    """The example's settings fragment without ``COMPONENTS.MODULES`` (the test loads the module itself)."""
    blocks = copy.deepcopy(load_conf(number))
    components = blocks.get("COMPONENTS")
    if components is not None:
        components.pop("MODULES", None)
        if not components:
            blocks.pop("COMPONENTS")
    return blocks


def settings_with(number: str, settings, **extra: Any):
    """``settings`` with the fragment merged in the way ``conf/`` files merge (blocks deep-merged)."""
    from p2_support import configure

    return configure(settings, **{**fragment(number), **extra})
