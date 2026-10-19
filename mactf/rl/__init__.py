"""RL primitives: policy, value, advantage, loss, and rollout."""

from __future__ import annotations

from .advantage import (
    AdvantageBatch,
    generalized_advantage,
    penalise_void_advantage,
    relative_to_reference,
    temporal_difference,
)
from .loss import LossBreakdown, LossConfig, clip_gradients, compute_loss, policy_loss
from .policy import (
    PolicyOutput,
    behaviour_probabilities,
    clip_ratio,
    greedy_action,
    importance_ratio,
    make_policy,
    reverse_kl,
    sample_action,
)
from .rollout import Rollout, Transition, attach_signals, collect
from .value import ValueEstimate, ValueHead, normalise_advantages, value_loss

__all__ = [
    "AdvantageBatch",
    "LossBreakdown",
    "LossConfig",
    "PolicyOutput",
    "Rollout",
    "Transition",
    "ValueEstimate",
    "ValueHead",
    "attach_signals",
    "behaviour_probabilities",
    "clip_gradients",
    "clip_ratio",
    "collect",
    "compute_loss",
    "generalized_advantage",
    "greedy_action",
    "importance_ratio",
    "make_policy",
    "normalise_advantages",
    "penalise_void_advantage",
    "policy_loss",
    "relative_to_reference",
    "reverse_kl",
    "sample_action",
    "temporal_difference",
    "value_loss",
]
