"""Helpers shared by the phase-3 tests: trainers, call traces and parameter snapshots."""

from __future__ import annotations

from typing import Any

import torch

from mactf.coopetition.strategy_probe import _Synthetic
from p2_support import configure, hierarchy_trainer, make_trainer, tune

__all__ = ["flat", "hier", "trace_calls", "flat_parameters", "run_trace", "Recording"]


def flat(settings, tmp_path, trainer_block=None, *, peers=3, epochs=2, coopetition=None, **kwargs):
    tune(settings, EPOCHS=epochs)
    if trainer_block is not None:
        configure(settings, TRAINER=trainer_block)
    if coopetition is not None:
        configure(settings, COOPETITION=coopetition)
    return make_trainer(settings, tmp_path, peers=peers, **kwargs)


def hier(settings, tmp_path, trainer_block=None, *, epochs=2, coopetition=None, **kwargs):
    tune(settings, EPOCHS=epochs)
    if trainer_block is not None:
        configure(settings, TRAINER=trainer_block)
    if coopetition is not None:
        configure(settings, COOPETITION=coopetition)
    return hierarchy_trainer(settings, tmp_path, **kwargs)


def trace_calls(trainer) -> list[tuple]:
    """Record every coordination call the trainer receives, in order, as comparable tuples."""
    calls: list[tuple] = []

    def watch(name, summary):
        original = getattr(trainer, name)

        def wrapper(*args, **kwargs):
            calls.append((name, *summary(args, kwargs)))
            return original(*args, **kwargs)

        setattr(trainer, name, wrapper)

    watch("update_weights", lambda a, k: (None if (k.get("eligible") if k else None) is None
                                          else tuple(sorted(k["eligible"].items())),))
    watch("aggregate", lambda a, k: (k.get("update_reference", True),))
    watch("aggregate_group", lambda a, k: (a[0], tuple(a[1]), None if k.get("eligible") is None
                                           else tuple(sorted(k["eligible"].items()))))
    watch("record_signal_sharing", lambda a, k: (tuple(sorted(a[0])), k.get("epoch")))
    return calls


def flat_parameters(trainer) -> dict[str, torch.Tensor]:
    out = {path: peer.adapter.export_parameters().detach().clone() for path, peer in trainer.peers.items()}
    out["__reference__"] = trainer.reference.adapter.export_parameters().detach().clone()
    return out


def run_trace(trainer, epochs: int = 2) -> dict[str, Any]:
    """Run ``epochs`` epochs; return the call trace, the reports and the final parameters."""
    trainer.build()
    calls = trace_calls(trainer)
    for epoch in range(epochs):
        trainer.run_epoch(epoch)
    return {
        "calls": calls,
        "reports": [(tuple(r.order), tuple(r.weights.items()), r.kappa, r.entropy, r.fallback)
                    for r in trainer.reports],
        "parameters": flat_parameters(trainer),
        "history": [item.as_dict() for item in trainer.history],
    }


class Recording(_Synthetic):
    """The synthetic backend, plus a log of what was written and a way to alter a peer."""

    def __init__(self, view, void=frozenset()):
        super().__init__(view, frozenset(void))
        self.writes: list[tuple] = []
        self.digests: dict[str, str] = {}

    def aggregate(self, report=None, *, update_reference=True):
        self.writes.append(("root", update_reference))
        return super().aggregate(report, update_reference=update_reference)

    def aggregate_group(self, parent, members, *, eligible=None):
        self.writes.append((parent, tuple(members)))
        return super().aggregate_group(parent, members, eligible=eligible)

    def peer_digest(self, path):
        return self.digests.get(path, "synthetic")
