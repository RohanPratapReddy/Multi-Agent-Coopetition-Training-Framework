"""The worked advantage of section 8.9.1, and the rules that keep an advantage hook from being a bypass."""

from __future__ import annotations

import pytest
import torch

from mactf.components import register_function
from mactf.exceptions import ContractError, ValidationError
from mactf.training.policy import DefaultTrainPolicy

from p2_support import ClippedAdvantage, configure, make_trainer, tune


def test_clip_bounds_and_identity() -> None:
    function = ClippedAdvantage(clip=0.5)
    # ``GAEAdvantage`` takes one bootstrap value (``next_values`` has a single element), so the
    # document's ``zeros(1, 3)`` is spelled the way the real estimator accepts it.
    rewards = torch.tensor([10.0, -10.0, 0.1])
    values = torch.zeros(3)
    batch = function(rewards, values, next_values=torch.zeros(1))
    assert float(batch.advantages.abs().max()) <= 0.5 + 1e-6
    assert torch.allclose(batch.returns - values, batch.advantages)


def test_clipped_advantage_rejects_a_non_positive_clip() -> None:
    with pytest.raises(ValidationError):
        ClippedAdvantage(clip=0.0)


@pytest.fixture
def clipped(component_overlay):
    register_function("advantage", "clipped", ClippedAdvantage, scope="run")


def _batch(*, void: bool, signal: float, floor: float = 0.0):
    generator = torch.Generator().manual_seed(11)
    return {"states": torch.randn(8, 4, generator=generator), "targets": torch.zeros(8, dtype=torch.long),
            "signal": signal, "void": void, "floor": floor}


def test_clipped_advantage(settings, tmp_path, clipped):
    """The advantage hook is read from ``RL.ADVANTAGE`` and changes what the step learns."""
    tune(settings)
    configure(settings, RL={"ADVANTAGE": {"type": "clipped", "params": {"clip": 0.05}}})
    hooked = make_trainer(settings, tmp_path / "a").build()
    assert isinstance(hooked._policy, DefaultTrainPolicy)
    assert hooked._policy._hook("advantage", "peer-1").clip == 0.05
    plain = make_trainer(settings, tmp_path / "b")                # same weights, no hook
    settings["RL"].pop("ADVANTAGE")
    plain = make_trainer(settings, tmp_path / "b").build()
    batch = _batch(void=False, signal=3.0)
    clipped_result = hooked.train_step(hooked.tree.get("peer-1"), dict(batch))
    plain_result = plain.train_step(plain.tree.get("peer-1"), dict(batch))
    assert clipped_result.breakdown.extras["advantage_mean"] == pytest.approx(0.05, abs=1e-6)
    assert plain_result.breakdown.extras["advantage_mean"] == pytest.approx(3.0, abs=1e-6)
    assert clipped_result.loss_value != plain_result.loss_value


def test_a_void_node_stays_at_or_below_the_floor_whatever_the_function_returns(settings, tmp_path, component_overlay):
    """7.3 rule 1: the void floor is re-applied *after* a custom advantage."""

    class Generous:
        """Gives every row +5, void or not."""

        def __call__(self, rewards, values, **kwargs):
            from mactf.rl.advantage import AdvantageBatch

            return AdvantageBatch(advantages=torch.full_like(rewards, 5.0), returns=rewards)

    register_function("advantage", "generous", Generous, scope="run")
    tune(settings)
    configure(settings, RL={"ADVANTAGE": {"type": "generous"}})
    trainer = make_trainer(settings, tmp_path).build()
    void = trainer.train_step(trainer.tree.get("peer-1"), _batch(void=True, signal=5.0))
    healthy = trainer.train_step(trainer.tree.get("peer-1"), _batch(void=False, signal=5.0))
    assert void.breakdown.extras["advantage_mean"] <= -1.0 + 1e-6      # held at the floor
    assert healthy.breakdown.extras["advantage_mean"] == pytest.approx(5.0)


def test_a_non_finite_advantage_is_refused_before_it_reaches_the_loss(settings, tmp_path, component_overlay):
    class Broken:
        def __call__(self, rewards, values, **kwargs):
            from mactf.rl.advantage import AdvantageBatch

            return AdvantageBatch(advantages=rewards * float("nan"), returns=rewards)

    register_function("advantage", "broken_adv", Broken, scope="run")
    tune(settings)
    configure(settings, RL={"VERIFY": "off", "ADVANTAGE": {"type": "broken_adv"}})
    trainer = make_trainer(settings, tmp_path).build()
    with pytest.raises(ContractError, match="non-finite"):
        trainer.train_step(trainer.tree.get("peer-1"), _batch(void=False, signal=1.0))


@pytest.mark.parametrize("masks,expected", [("none", None), ("void", True), ("valid", False)])
def test_advantage_masks_declare_what_they_pass(masks, expected, component_overlay):
    """The `masks` argument has one declared meaning per policy setting (7.3 rule 2)."""
    seen = {}

    class Spy:
        def __call__(self, rewards, values, *, masks=None, **kwargs):
            from mactf.rl.advantage import AdvantageBatch

            seen["masks"] = masks
            return AdvantageBatch(advantages=rewards, returns=rewards)

    policy = DefaultTrainPolicy(advantage=Spy(), advantage_masks=masks)
    signal, void = torch.full((4,), 0.5), torch.full((4,), True)
    policy._custom_advantage(policy._spec_hooks.advantage, signal, torch.zeros(4, 1), void, 4)
    if expected is None:
        assert seen["masks"] is None
    else:
        assert bool(seen["masks"].all()) is expected


def test_default_policy_argument_validation():
    with pytest.raises(ValidationError, match="advantage_masks"):
        DefaultTrainPolicy(advantage_masks="both")
    with pytest.raises(ValidationError, match="gamma and lam"):
        DefaultTrainPolicy(gamma=1.5)
