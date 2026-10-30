"""A model-based fuzz of the registry, overlays and hub, and a thread-safety check of reads.

The model is the documented rule: the innermost overlay that registers or hides a name decides, the
base layer decides otherwise. The registry (a private ``Hub``) must agree after every operation.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack

import pytest
from hypothesis import settings as hyp_settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from component_helpers import Policy
from mactf.components import Hub
from mactf.exceptions import ComponentCollision, ComponentLocked, ComponentMissing

NAMES = st.sampled_from(["a", "b", "c", "d", "e"])
KIND = "trainer_policy"


def _fresh_class(tag: str):
    return type(f"Policy_{tag}", (Policy,), {})


class RegistryMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.hub = Hub()
        self.registry = self.hub.registry(KIND)
        self.base: dict[str, type] = {}
        self.stack: list[dict] = []          # per overlay: {"entries": {}, "hidden": set()}
        self.exits = ExitStack()
        self.counter = 0

    def teardown(self) -> None:
        self.exits.close()

    # -- model ---------------------------------------------------------
    def visible(self, name):
        for layer in reversed(self.stack):
            if name in layer["entries"]:
                return layer["entries"][name]
            if name in layer["hidden"]:
                return None
        return self.base.get(name)

    def _new(self) -> type:
        self.counter += 1
        return _fresh_class(str(self.counter))

    # -- rules ---------------------------------------------------------
    @rule(name=NAMES, replace=st.booleans())
    def register_global(self, name, replace):
        factory = self._new()
        if name in self.base and not replace:
            with pytest.raises(ComponentCollision):
                self.registry.register(name, factory)
            return
        self.registry.register(name, factory, replace=replace)
        self.base[name] = factory

    @rule(name=NAMES, replace=st.booleans())
    @precondition(lambda self: self.stack)
    def register_run(self, name, replace):
        factory = self._new()
        if self.visible(name) is not None and not replace:
            with pytest.raises(ComponentCollision):
                self.registry.register(name, factory, scope="run")
            return
        self.registry.register(name, factory, scope="run", replace=replace)
        layer = self.stack[-1]
        layer["entries"][name] = factory
        layer["hidden"].discard(name)

    @rule(name=NAMES)
    def unregister_global(self, name):
        if name not in self.base:
            with pytest.raises(ComponentMissing):
                self.registry.unregister(name)
            return
        self.registry.unregister(name)
        del self.base[name]

    @rule(name=NAMES)
    @precondition(lambda self: self.stack)
    def unregister_run(self, name):
        if self.visible(name) is None:
            with pytest.raises(ComponentMissing):
                self.registry.unregister(name, scope="run")
            return
        self.registry.unregister(name, scope="run")
        layer = self.stack[-1]
        layer["entries"].pop(name, None)
        layer["hidden"].add(name)

    @rule()
    @precondition(lambda self: len(self.stack) < 3)
    def push(self):
        self.exits.enter_context(self.hub.scoped(f"o{len(self.stack)}"))
        self.stack.append({"entries": {}, "hidden": set()})

    @rule()
    def freeze_blocks_global_writes(self):
        self.registry.freeze()
        try:
            with pytest.raises(ComponentLocked):
                self.registry.register("zz", self._new())
        finally:
            self.registry.unfreeze()

    # -- invariants ----------------------------------------------------
    @invariant()
    def the_registry_matches_the_model(self):
        # overlays are context managers entered on one ExitStack, so only the top can be popped; the model
        # therefore never pops (a stack of at most three covers the nesting rules)
        universe = {"a", "b", "c", "d", "e"}
        expected = sorted(n for n in universe if self.visible(n) is not None)
        assert [n for n in self.registry.names() if n in universe] == expected
        for name in universe:
            found = self.registry.find(name)
            want = self.visible(name)
            assert (found.factory if found else None) is want


RegistryMachine.TestCase.settings = hyp_settings(max_examples=60, stateful_step_count=25, deadline=None, database=None)
TestRegistryModel = RegistryMachine.TestCase


def test_reads_from_threads_never_see_a_torn_registry():
    """Readers iterate names()/find()/snapshot while a writer registers, replaces and unregisters."""
    from mactf.components.known import import_builtins

    import_builtins()          # first use from many threads is the serve command's job (see the next test)
    hub = Hub()
    registry = hub.registry(KIND)
    stop = threading.Event()
    failures: list[BaseException] = []

    def reader():
        try:
            while not stop.is_set():
                names = registry.names()
                assert names == sorted(names)
                for name in names:
                    component = registry.find(name)        # may have been unregistered meanwhile: None is fine
                    assert component is None or component.name == name
                hub.snapshot()
        except BaseException as error:                      # noqa: BLE001
            failures.append(error)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    try:
        for index in range(400):
            name = f"n{index % 17}"
            registry.register(name, _fresh_class(str(index)), replace=True)
            if index % 3 == 0:
                registry.unregister(name)
            with hub.scoped("w"):
                registry.register("scoped", _fresh_class("s"), scope="run", replace=True)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
    assert not failures, failures[0]
    assert not any(thread.is_alive() for thread in threads)


def test_two_overlays_in_two_threads_do_not_leak():
    hub = Hub()
    registry = hub.registry(KIND)
    seen: dict[str, list[str]] = {}
    barrier = threading.Barrier(2)

    def run(label: str):
        with hub.scoped(label):
            registry.register(label, _fresh_class(label), scope="run")
            barrier.wait(timeout=30)                  # both overlays hold their component at the same time
            seen[label] = registry.names()
            barrier.wait(timeout=30)

    threads = [threading.Thread(target=run, args=(label,)) for label in ("left", "right")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert seen == {"left": ["left"], "right": ["right"]}
    assert registry.names() == []


RACE = """
import threading
from mactf.contrib.api.management.commands.serve import build_server
from mactf.components import HUB
server = build_server("127.0.0.1", 0)
errors, counts = [], set()
gate = threading.Barrier(8)
def go():
    try:
        gate.wait(); counts.add(sum(len(v) for v in HUB.snapshot().values()))
    except BaseException as error:
        errors.append(repr(error))
threads = [threading.Thread(target=go) for _ in range(8)]
[t.start() for t in threads]; [t.join() for t in threads]
server.server_close()
print("RESULT", errors, sorted(counts))
"""


@pytest.mark.subprocess
def test_a_server_reads_the_registry_from_many_threads_on_its_first_requests():
    """Regression: before ``build_server`` imported the built-ins, 8 first readers raised an import
    ``_DeadlockError`` or saw 28..111 components depending on the interleaving."""
    import subprocess
    import sys

    for _ in range(3):
        done = subprocess.run([sys.executable, "-c", RACE], capture_output=True, text=True, timeout=120)
        line = [row for row in done.stdout.splitlines() if row.startswith("RESULT")]
        assert line, done.stderr[-800:]
        assert line[0].startswith("RESULT [] ["), line[0]
        assert line[0].count(",") == 0 or "," not in line[0].split("] [")[1], f"readers disagreed: {line[0]}"
