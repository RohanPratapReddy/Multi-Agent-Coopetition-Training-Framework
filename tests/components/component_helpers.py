"""Small component classes shared by the component tests (no torch is needed to register them)."""

from __future__ import annotations


class Policy:
    """A trainer policy: the lightest registrable kind."""

    def __init__(self, *, rate: float = 1.0) -> None:
        self.rate = rate

    def train_step(self, node, batch):  # pragma: no cover - never run
        return None


class OtherPolicy(Policy):
    pass


class Term:
    rule_ids = frozenset({"R1"})

    def __init__(self, *, scale: float = 1.0) -> None:
        self.scale = scale

    def contribution(self, *args, **kwargs):  # pragma: no cover - never run
        return 0.0


def make_scheduler(optimizer, *, gamma: float = 0.5):  # pragma: no cover - never run
    class _S:
        def step(self):
            return None

    return _S()
