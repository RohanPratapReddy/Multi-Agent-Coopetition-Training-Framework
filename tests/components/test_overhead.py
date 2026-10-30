"""Doc 17 section 20.4: the default path costs nothing it did not cost before.

The gate REPORTS the ratio and does not fail on it: a wall-clock threshold on a shared CI
runner is noise (a ratio above 1.02 on the reference machine is a defect to explain in
review). What is asserted is the structural half of the claim, which is not noise: a
default epoch builds no component, registers none, and the package pulls in no torch.
"""

from __future__ import annotations

import subprocess
import sys
import time

import default_identity as recorder


def _time_run(tmp_path, label):
    start = time.perf_counter()
    recorder.run_trainer(tmp_path / label)
    return time.perf_counter() - start


def test_default_epoch_overhead_is_reported_and_builds_no_component(tmp_path, capsys):
    from mactf.components.registry import ComponentRegistry
    from mactf.components.registry import HUB

    calls = {"build": 0, "register": 0}
    build, register = ComponentRegistry.build, ComponentRegistry.register

    def counted(name, original):
        def wrapper(self, *a, **k):
            calls[name] += 1
            return original(self, *a, **k)

        return wrapper

    # "without" = the package imported but its slots never touched; "with" = the hub's
    # per-run activation entered explicitly around the same scenario.
    _time_run(tmp_path, "warmup")  # imports and first-call costs belong to neither side
    without = _time_run(tmp_path, "without")
    ComponentRegistry.build = counted("build", build)
    ComponentRegistry.register = counted("register", register)
    try:
        start = time.perf_counter()
        with HUB.scoped("overhead"):
            recorder.run_trainer(tmp_path / "with")
        with_slots = time.perf_counter() - start
    finally:
        ComponentRegistry.build, ComponentRegistry.register = build, register
    ratio = with_slots / without if without > 0 else float("nan")
    with capsys.disabled():
        print(f"\n[doc17 20.4] epoch time with/without component slots: {ratio:.3f}")
    assert calls == {"build": 0, "register": 0}
    assert without > 0 and with_slots > 0


def test_importing_the_component_package_pulls_in_no_torch():
    code = "import sys, mactf.components; raise SystemExit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
