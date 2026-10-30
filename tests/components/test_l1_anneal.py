"""The worked regularizer of section 8.9.2: a schedule that is real state."""

from __future__ import annotations

import pytest
import torch

from mactf.components import register_function
from mactf.components.migrations import carry_state, fail_if_stateful, is_stateful

from p2_support import L1AnnealRegularization, configure, make_trainer, tune


def test_schedule_decays_and_state_carries() -> None:
    module = torch.nn.Linear(2, 2)
    old = L1AnnealRegularization(coefficient=1.0, steps=4)
    first = float(old(module))
    old(module)
    assert is_stateful(old)
    new = L1AnnealRegularization(coefficient=1.0, steps=4)
    carry_state(old, new)
    assert float(new(module)) < first                      # picks up at step 2, not step 0
    try:
        fail_if_stateful(old, new)
    except Exception as error:
        assert "migrate: fail" in str(error)
    else:  # pragma: no cover
        raise AssertionError("fail_if_stateful must refuse a stateful instance")


def test_l1_anneal(settings, tmp_path, component_overlay):
    """Through a real trainer: the penalty is added inside the step and anneals to zero."""
    register_function("regularization", "l1_anneal", L1AnnealRegularization, scope="run")
    tune(settings)
    configure(settings, RL={"REGULARIZATION": {"type": "l1_anneal", "params": {"coefficient": 0.5, "steps": 3}}})
    trainer = make_trainer(settings, tmp_path).build()
    node = trainer.tree.get("peer-1")
    states = torch.randn(8, 4, generator=torch.Generator().manual_seed(5))
    batch = {"states": states, "targets": torch.zeros(8, dtype=torch.long), "signal": 0.0, "void": False, "floor": 0.0}
    penalties = []
    for _ in range(5):
        result = trainer.train_step(node, dict(batch))
        penalties.append(result.breakdown.extras["regularization"])
    assert penalties[0] > penalties[1] > penalties[2] > 0.0     # decaying
    assert penalties[3] == 0.0 and penalties[4] == 0.0          # annealed away after `steps` calls
    # the counter lives on the instance of this node, and only this node
    assert trainer.functions.get("regularization", "peer-1")._calls == 5
    assert trainer.functions.get("regularization", "peer-2")._calls == 0


def test_the_regularizer_changes_the_objective_a_closure_optimizer_sees(settings, tmp_path, component_overlay):
    """It is added inside ``evaluate()``, so a full-batch backend optimises the regularised loss."""
    pytest.importorskip("scipy")
    register_function("regularization", "l1_anneal", L1AnnealRegularization, scope="run")
    tune(settings)
    configure(settings, RL={"REGULARIZATION": {"type": "l1_anneal", "params": {"coefficient": 0.1, "steps": 50}}},
              OPTIMIZER={"TYPE": "scipy", "METHOD": "L-BFGS-B"})
    trainer = make_trainer(settings, tmp_path).build()
    batch = {"states": torch.randn(8, 4), "targets": torch.zeros(8, dtype=torch.long),
             "signal": 0.0, "void": False, "floor": 0.0}
    result = trainer.train_step(trainer.tree.get("peer-1"), batch)
    assert "regularization" in result.breakdown.extras


def test_checkpointed_run_continues_the_schedule(settings, tmp_path, component_overlay):
    register_function("regularization", "l1_anneal", L1AnnealRegularization, scope="run")
    tune(settings)
    rl = {"REGULARIZATION": {"type": "l1_anneal", "params": {"coefficient": 0.5, "steps": 6}}}
    configure(settings, RL=rl)
    first = make_trainer(settings, tmp_path / "a").build()
    first.run_epoch(0), first.run_epoch(1)
    payload = first.state_dict()
    second = make_trainer(settings, tmp_path / "b").build()
    second.load_state_dict(payload, strict=False)
    assert second.functions.get("regularization", "peer-1")._calls == 2
