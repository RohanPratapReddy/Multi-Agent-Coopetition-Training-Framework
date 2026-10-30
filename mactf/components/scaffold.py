"""Source text for ``mactf components scaffold`` (document 17 section 16.1.3).

One template per kind, written against the real contracts of ``components/contracts.py``: the
module imports, registers, and resolves with ``ComponentResolver.validate`` as it is written; the
only thing missing is the body of the method that does the work, which raises
``NotImplementedError`` so a half-finished component fails loudly in the method, never at import.
This module imports nothing heavy: the templates are text.
"""

from __future__ import annotations

import re

#: The kinds a person writes code for (peers have ``mactf startpeer``).
SCAFFOLD_KINDS: tuple[str, ...] = (
    "scorer", "reward_term", "weight_update", "peer_strategy", "rl_function",
    "optimizer", "scheduler", "trainer_policy",
)

#: The contract function that judges each kind (``mactf.components.contracts``).
CONTRACTS: dict[str, str] = {
    "scorer": "scorer_contract",
    "reward_term": "reward_term_contract",
    "weight_update": "weight_update_contract",
    "peer_strategy": "peer_strategy_contract",
    "rl_function": "rl_function_contract",
    "optimizer": "optimizer_contract",
    "scheduler": "scheduler_contract",
    "trainer_policy": "trainer_policy_contract",
}

#: The kind id a scaffold registers (``rl_function`` is the ``loss`` family).
KIND_IDS: dict[str, str] = {kind: kind for kind in SCAFFOLD_KINDS} | {"rl_function": "rl_function.loss"}

_HEADER = '''"""{kind} component: {name}.

Load it by adding this module to COMPONENTS.MODULES (conf/components.yaml). The contract
for a {kind} is mactf.components.contracts.{contract}.
"""

from __future__ import annotations
'''

_BODIES: dict[str, str] = {
    "weight_update": '''
from mactf.components import register_weight_update
from mactf.coopetition.weights import WeightStrategy


@register_weight_update("{name}", version="1", description="TODO: one line for `components list`")
class {cls}(WeightStrategy):
    """Turn per-peer scores into simplex weights. WeightGuard projects whatever you return."""

    name = "{name}"

    def __init__(self, *, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        raise NotImplementedError("return an AggregationReport (mactf.coopetition.weights.make_report)")
''',
    "scorer": '''
from mactf.components import register_scorer
from mactf.scoring.base import ScoreContext, ScorerOutput
from mactf.scoring.reward_scorer import RewardScorer


@register_scorer("{name}", version="1", description="TODO: one line for `components list`")
class {cls}(RewardScorer):
    """A reward stage. ComplianceGuard checks its output against the reference stage."""

    def score(self, ctx: ScoreContext) -> ScorerOutput:
        raise NotImplementedError("return a ScorerOutput; the guard refuses an inflated reward (C008)")
''',
    "peer_strategy": '''
from typing import ClassVar

from mactf.components import register_peer_strategy
from mactf.coopetition.strategy import PeerStrategy, StrategyTraits


@register_peer_strategy("{name}", version="1", description="TODO: one line for `components list`")
class {cls}(PeerStrategy):
    """Decide, per group, what mixes at an epoch boundary. StrategyGuard checks the choice."""

    name: ClassVar[str] = "{name}"
    traits: ClassVar[StrategyTraits] = StrategyTraits(aggregates=True, walk="tree")

    def coordinate(self, ctx) -> None:
        raise NotImplementedError("call ctx.aggregate_group(group) or ctx.skip(group, reason) for each group")
''',
    "reward_term": '''
from typing import ClassVar

from mactf.components import register_reward_term
from mactf.scoring.reward_terms import RewardTerm


@register_reward_term("{name}", version="1", description="TODO: one line for `components list`")
class {cls}(RewardTerm):
    """One factor of a composed reward: a finite, non-negative number per event."""

    name: ClassVar[str] = "{name}"

    def __init__(self, *, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def contribution(self, event, ctx) -> float:
        raise NotImplementedError("return a finite, non-negative float; the stage and the guard check it")
''',
    "rl_function": '''
from mactf.components import register_function


@register_function("loss", "{name}", version="1", description="TODO: one line for `components list`")
class {cls}:
    """A loss. The framework adds the compliance term; the probe checks finiteness and gradients."""

    required_fields = frozenset({{"values", "value_targets"}})

    def __init__(self, *, delta: float = 1.0) -> None:
        self.delta = float(delta)

    def compute(self, batch):
        raise NotImplementedError("return (loss tensor, LossBreakdown); the probe reports F001-F012")
''',
    "optimizer": '''
import torch

from mactf.components import register_optimizer


@register_optimizer("{name}", version="1", description="TODO: one line for `components list`")
class {cls}(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("update every p in each group['params'] from p.grad; return the closure loss")
''',
    "scheduler": '''
from mactf.components import register_scheduler


@register_scheduler("{name}", version="1", description="TODO: one line for `components list`")
def {name}(optimizer, *, step_size: int = 100, factor: float = 0.1):
    """Return an object with step(), bound to the optimizer given."""
    raise NotImplementedError("return a torch.optim.lr_scheduler object built over `optimizer`")
''',
    "trainer_policy": '''
from mactf.components import register_policy


@register_policy("{name}", version="1", description="TODO: one line for `components list`")
class {cls}:
    """Own the step. StepGuard wraps it unless ``unsafe = True`` and the operator allows that."""

    unsafe = False

    def train_step(self, node, batch):
        raise NotImplementedError("return a mactf.training.protocol.StepResult")
''',
}

_TEST = '''

# tests/components/test_{name}.py
#   from mactf.components import ComponentResolver, ComponentSpec
#   def test_{name}_meets_its_contract():
#       spec = ComponentSpec.parse("{kind_id}", {{"type": "{name}"}})
#       ComponentResolver().validate(spec, trusted=True)
'''

_PLACEHOLDER = re.compile(r"\{(name|cls|kind|kind_id|contract)\}")


def class_name(name: str) -> str:
    """``my_trimmed_mean`` -> ``MyTrimmedMean``."""
    return "".join(part.capitalize() for part in name.split("_") if part)


def render_scaffold(kind: str, name: str) -> str:
    """The text of a ready-to-edit module for ``kind`` named ``name``."""
    body = _BODIES.get(kind)
    if body is None:
        raise ValueError(f"no scaffold for kind {kind!r}; kinds: {list(SCAFFOLD_KINDS)}")
    substitutions = {"name": name, "cls": class_name(name), "kind": kind,
                     "kind_id": KIND_IDS[kind], "contract": CONTRACTS[kind]}
    head = _HEADER.replace("{kind}", kind).replace("{name}", name).replace("{contract}", CONTRACTS[kind])
    text = head + body.replace("{{", "\0(").replace("}}", ")\0")
    text = _PLACEHOLDER.sub(lambda match: substitutions[match.group(1)], text)
    text = text.replace("\0(", "{").replace(")\0", "}")
    tail = _TEST.replace("{{", "\0(").replace("}}", ")\0")
    tail = _PLACEHOLDER.sub(lambda match: substitutions[match.group(1)], tail)
    return text + tail.replace("\0(", "{").replace(")\0", "}")
