"""Shared pieces of the section 12 tests: a small trainer that can mutate, and a state snapshot."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

import mactf.training.policy  # noqa: F401  (registers the built-in policies)
from mactf.training import MACTFTrainer


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.policy = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


class Wide(nn.Module):
    """A different parameter layout from ``Tiny`` with the same input and output sizes."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, 3)
        self.value = nn.Linear(hidden, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


def mutation_settings(settings, **mutation: Any):
    """Training sized for a test, with ``TRAINER.MUTATION`` enabled and ``mutation`` merged in."""
    settings["TRAINING"].update(
        {"EPOCHS": 3, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False,
         "CHECKPOINT_EVERY": 1}
    )
    settings["DEVICE"] = "cpu"
    settings["SANDBOX"]["PROVIDER"] = "local"
    settings["TRAINER"]["MUTATION"] = {**dict(settings["TRAINER"].get("MUTATION") or {}), "ENABLED": True, **mutation}
    settings["WATCHDOG"]["MAX_TREE_GROWTH_RATIO"] = 8.0
    return settings


def mutable_trainer(settings, tmp_path, *, peers: int = 3, seed: int = 0, build: bool = True, **kwargs):
    """``peers`` roots built from one ``Tiny``; ``add`` works for any path (the single-model source)."""
    torch.manual_seed(seed)
    trainer = MACTFTrainer(
        model=Tiny(), num_peers=peers, settings=settings, checkpoint_dir=Path(tmp_path) / "ckpt", **kwargs
    )
    return trainer.build() if build else trainer


def parameters(trainer) -> dict[str, torch.Tensor]:
    return {path: peer.export_parameters().detach().clone() for path, peer in trainer.peers.items()}


def snapshot(trainer) -> dict[str, Any]:
    """Everything a mutation may touch, in a form two snapshots can be compared with ``==``."""
    tree = trainer.tree
    return {
        "nodes": [(n.path, n.role, n.status, n.children, dict(n.metadata)) for n in tree.walk("root-first")],
        "node_peers": {n.path: id(n.peer) for n in tree.walk("root-first")},
        "peers": {path: id(peer) for path, peer in trainer.peers.items()},
        "weights": {path: peer.export_parameters().detach().clone().tolist() for path, peer in trainer.peers.items()},
        "optimizers": {path: id(opt) for path, opt in trainer.optimizers.items()},
        "peer_optimizers": {path: id(peer.optimizer) for path, peer in trainer.peers.items()},
        "optimizer_entries": {slot: entry.spec_hash for slot, entry in sorted(trainer.optimizer_plan.entries.items())},
        "optimizer_history": len(trainer.optimizer_plan.history),
        "optimizer_swapped": (sorted(trainer.optimizer_plan._swapped), sorted(trainer.optimizer_plan._removed)),
        "controllers": {key: (id(c), c.snapshot()) for key, c in trainer.controllers.items()},
        "level_controllers": {str(key): id(c) for key, c in trainer._level_controllers.items()},
        "group_aggregators": sorted(trainer._group_aggregators),
        "freeze": {path: sorted(trainer.freeze.reasons(path)) for path in trainer.freeze.paths()},
        "journal": copy.deepcopy(trainer.mutation_journal),
        "topology_epoch": trainer.topology_epoch,
        "tombstones": sorted(trainer._tombstones),
        "node_specs": copy.deepcopy(trainer._node_specs),
        "node_meta": copy.deepcopy(trainer._node_meta),
        "budget": trainer._mutations_in_cycle,
        "queue": trainer.engine.pending(),
        "poisoned": trainer.engine.poisoned is None,
        "active_view": trainer._active_view is None,
        "per_path": {
            name: sorted(getattr(trainer, name))
            for name in ("signal_by_peer", "_last_weights", "_previous_scores", "_dead_counts", "_epoch_node_metrics")
        },
        "function_cache": sorted(map(str, trainer.functions._cache)),
        "weight_cache": sorted(trainer.weight_plan._cache),
        "strategy_cache": sorted(map(str, trainer.strategy_plan._group_cache)),
        "rng": torch.get_rng_state().clone().tolist(),
    }
