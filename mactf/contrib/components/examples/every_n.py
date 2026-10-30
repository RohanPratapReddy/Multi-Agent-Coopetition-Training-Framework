"""18.4: a peer strategy that mixes only every N epochs.

Peers train independently and mix their parameters every ``period`` epochs. The strategy is
context-driven (``coordinate``), so it can be applied to one group only.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from mactf.components import register_peer_strategy
from mactf.contrib.components.examples import registration_scope
from mactf.coopetition.strategy import PeerStrategy, StrategyTraits


@register_peer_strategy(
    "every_n", version="1", description="Mix parameters every N epochs, train independently otherwise",
    scope=registration_scope(),
)
class EveryN(PeerStrategy):
    name: ClassVar[str] = "every_n"
    traits: ClassVar[StrategyTraits] = StrategyTraits(aggregates=True, walk="tree")

    def __init__(self, *, period: int = 3) -> None:
        if isinstance(period, bool) or int(period) < 1:
            raise ValueError("period must be >= 1")
        self.period = int(period)
        self._epoch = 0

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self._epoch = int(epoch)

    def coordinate(self, ctx) -> None:
        self._epoch = int(ctx.epoch)
        for group in ctx.groups:
            if self._epoch % self.period != 0:
                ctx.skip(group, f"mixing only every {self.period} epochs")
                continue
            eligible = ctx.eligible_for(group.members)
            if eligible is not None and not any(eligible.values()):
                ctx.skip(group, "every member is frozen")
            elif group.is_root:
                ctx.aggregate_root(ctx.update_root(eligible=eligible))
            elif ctx.peers_void(group.members):
                ctx.skip(group, "every member was voided this epoch")
            else:
                ctx.aggregate_group(group, eligible=eligible)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "period": self.period}

    def state_dict(self) -> Mapping[str, Any]:
        return {"epoch": self._epoch}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self._epoch = int(payload.get("epoch", 0))


class LegacyEveryN(PeerStrategy):
    """Only the legacy ``on_epoch_end`` entry point: it cannot serve one group (E072)."""

    name: ClassVar[str] = "legacy_every_n"
    traits: ClassVar[StrategyTraits] = StrategyTraits(aggregates=False)

    def on_epoch_end(self, trainer, epoch, report):
        pass
