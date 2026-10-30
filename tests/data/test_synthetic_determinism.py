"""The historical synthetic source must remain bit-identical.

This is a regression test rather than a fix: the original implementation derives
its seed from ``hash((seed, epoch, counter))`` over integers only, which Python
does not randomize. The test exists so a future change cannot silently make
reruns irreproducible.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_compatibility_shim_matches_the_new_source():
    import torch

    from mactf.data.sources import SyntheticTaskSource
    from mactf.orchestration.episode import TaskSource

    legacy = TaskSource(state_dim=5, action_dim=3, batch_size=4, seed=3)
    modern = SyntheticTaskSource(state_dim=5, action_dim=3, batch_size=4, seed=3)
    for epoch in range(3):
        left = legacy.next(epoch)
        right = modern.next(epoch)
        assert torch.equal(left.states, right.states)
        assert torch.equal(left.targets, right.targets)


def test_generator_override_still_works():
    from mactf.orchestration.episode import Task, TaskSource

    calls = []

    def generator(epoch, counter, state_dim, action_dim):
        calls.append((epoch, counter))
        import torch

        return Task(
            id=f"custom-{counter}",
            states=torch.zeros(2, state_dim),
            targets=torch.zeros(2, dtype=torch.long),
        )

    source = TaskSource(
        state_dim=4, action_dim=2, batch_size=2, seed=0, generator=generator
    )
    task = source.next(5)
    assert task.id == "custom-1"
    assert calls == [(5, 1)]


def test_cross_process_stability():
    script = (
        "from mactf.orchestration.episode import TaskSource;"
        "src = TaskSource(state_dim=6, action_dim=3, batch_size=4, seed=42);"
        "t = src.next(1);"
        "print(t.targets.tolist(), [round(float(x), 9) for x in t.states.flatten()[:5]])"
    )
    outputs = []
    for seed in ("0", "12345", "random"):
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT))
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
            cwd=str(ROOT),
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.strip())
    assert len(set(outputs)) == 1, outputs
