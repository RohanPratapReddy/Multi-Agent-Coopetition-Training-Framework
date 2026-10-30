"""Inference determinism (document 10 sections 7.4 and 9.1).

The same tree, the same seed, and two different ``PYTHONHASHSEED`` values must
produce identical outputs. This is a subprocess test because ``PYTHONHASHSEED``
is read once, at interpreter start.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import torch

from mactf.coopetition.delegation import SelectKDelegation
from mactf.coopetition.peer import build_peer
from mactf.coopetition.tree import HierarchySpec, LevelSpec, node_seed
from mactf.runtime import Runtime
from mactf.scoring.hierarchy import run_inference


class Constant(torch.nn.Module):
    def __init__(self, value: float, *, width: int = 3) -> None:
        super().__init__()
        self.width = int(width)
        self.bias = torch.nn.Parameter(torch.full((self.width,), float(value)))

    def forward(self, state):
        return self.bias.expand(state.shape[0], self.width).clone()


def _runtime() -> Runtime:
    return Runtime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        amp=False,
        seed=0,
        deterministic=True,
    )


_PROBE = textwrap.dedent(
    """
    import json, sys, torch
    from mactf.coopetition.peer import build_peer
    from mactf.coopetition.delegation import SelectKDelegation
    from mactf.coopetition.tree import HierarchySpec, LevelSpec, node_seed
    from mactf.runtime import Runtime
    from mactf.scoring.hierarchy import run_inference

    class Constant(torch.nn.Module):
        def __init__(self, value, width=3):
            super().__init__()
            self.width = width
            self.bias = torch.nn.Parameter(torch.full((width,), float(value)))

        def forward(self, state):
            return self.bias.expand(state.shape[0], self.width).clone()

    runtime = Runtime(device=torch.device("cpu"), dtype=torch.float32, amp=False,
                      seed=0, deterministic=True)
    counter = {"n": 0}

    def factory(path, depth, level):
        counter["n"] += 1
        # The value depends on the blake2b seed, so a `hash()`-derived seed would
        # show up as a different number under a different PYTHONHASHSEED.
        value = node_seed(5, path, 0) % 1000 / 1000.0
        module = Constant(value)
        with torch.no_grad():
            module.bias.copy_(torch.full((3,), value))
        return build_peer(module, name=path, runtime=runtime, action_dim=3, state_dim=3)

    spec = HierarchySpec(levels=(LevelSpec("sub", 2),), max_depth=4)
    tree = spec.build(peer_factory=factory)
    inputs = {"state": torch.ones(1, 3)}
    result = run_inference(tree, inputs, seed=5)
    sys.stdout.write(json.dumps({
        "outputs": {p: v.detach().cpu().tolist() for p, v in result.outputs.items()},
        "selected": {p: v["selected"] for p, v in result.delegation.items()},
        "seeds": {p: node_seed(5, p, 0) for p in result.outputs},
    }, sort_keys=True))
    """
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_probe(hash_seed: str) -> dict:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    environment["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_two_hash_seeds_produce_identical_outputs():
    first = _run_probe("0")
    second = _run_probe("99991")
    assert first == second
    assert first["outputs"], "the probe produced no outputs"


def test_the_derived_seed_is_stable_within_a_process():
    assert node_seed(5, "peer-1/sub-1", 0) == node_seed(5, "peer-1/sub-1", 0)
    assert node_seed(5, "peer-1/sub-1", 0) != node_seed(5, "peer-1/sub-2", 0)


def test_the_seed_changes_with_the_run_seed_and_the_counter():
    path = "peer-1/sub-1"
    assert node_seed(1, path, 0) != node_seed(2, path, 0)
    assert node_seed(1, path, 0) != node_seed(1, path, 1)


def test_the_seed_is_in_range():
    for index in range(200):
        seed = node_seed(0, f"peer-{index}", index)
        assert 0 <= seed < 2**31 - 1


def _tree():
    counter = {"n": 0}

    def factory(path, depth, level):
        counter["n"] += 1
        return build_peer(
            Constant(counter["n"] * 0.25),
            name=path,
            runtime=_runtime(),
            action_dim=3,
            state_dim=3,
        )

    return HierarchySpec(levels=(LevelSpec("sub", 2),), max_depth=4).build(peer_factory=factory)


def test_repeated_serial_calls_agree():
    tree = _tree()
    inputs = {"state": torch.randn(2, 3)}
    first = run_inference(tree, inputs, seed=3)
    second = run_inference(tree, inputs, seed=3)
    for path, value in first.outputs.items():
        assert torch.equal(value, second.outputs[path]), path


def test_repeated_parallel_calls_agree():
    tree = _tree()
    inputs = {"state": torch.randn(2, 3)}
    first = run_inference(tree, inputs, parallel=True, seed=3)
    second = run_inference(tree, inputs, parallel=True, seed=3)
    for path, value in first.outputs.items():
        assert torch.equal(value, second.outputs[path]), path


def test_delegation_selection_is_a_pure_function_of_the_tree():
    tree = _tree()
    inputs = {"state": torch.randn(1, 3)}
    first = run_inference(tree, inputs, delegation=SelectKDelegation(k=1), seed=1)
    second = run_inference(tree, inputs, delegation=SelectKDelegation(k=1), seed=1)
    assert first.delegation == second.delegation
    # SelectKDelegation ranks deterministically, so the first child always wins
    # a tie rather than whichever happened to be visited first.
    assert first.delegation["peer-1"]["selected"] == ["peer-1/sub-1"]


def test_the_inference_seed_is_reported():
    tree = _tree()
    result = run_inference(tree, {"state": torch.randn(1, 3)}, seed=42)
    assert result.seed == 42


def test_no_seed_still_produces_a_stable_result():
    tree = _tree()
    inputs = {"state": torch.randn(1, 3)}
    first = run_inference(tree, inputs)
    second = run_inference(tree, inputs)
    assert first.seed is None
    for path, value in first.outputs.items():
        assert torch.equal(value, second.outputs[path]), path
