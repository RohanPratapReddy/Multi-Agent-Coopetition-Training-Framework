"""Test helpers for operators writing their own MACTF tests.

Provides an in-memory-ish harness that builds a swarm, a scorer, and an
orchestrator without touching the project's real database or checkpoints.
"""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .. import conf
from ..conf import Settings
from ..orchestration.orchestrator import Orchestrator
from ..runtime import Runtime, build_runtime
from ..scoring.pipeline import ScoringPipeline as Scorer
from ..scoring.pipeline import build_scorer


class ToyPeer(nn.Module):
    """A minimal peer used by harness-driven tests."""

    def __init__(self, state_dim: int = 8, action_dim: int = 3, hidden: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, action_dim)
        self.value = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


@dataclass
class Harness:
    """A ready-to-use training environment rooted in a temp directory."""

    settings: Settings
    runtime: Runtime
    scorer: Scorer
    directory: Path
    peers: dict[str, ToyPeer] = field(default_factory=dict)

    def orchestrator(self, count: int = 2) -> Orchestrator:
        modules = {
            f"peer{index}": ToyPeer(
                state_dim=int(self.settings["TRAINING"]["STATE_DIM"]),
                action_dim=int(self.settings["TRAINING"]["ACTION_DIM"]),
            )
            for index in range(count)
        }
        self.peers = modules
        return Orchestrator(self.settings, modules=modules)

    def engine(self, count: int = 2, engine: str = "trainer", **blocks: Any) -> Any:
        """The configured engine over a *copy* of the settings (document 17 sections 13.10, 19.2).

        ``engine`` is ``"orchestrator"`` or ``"trainer"``; it is written to ``ORCHESTRATOR.ENGINE``
        on the copy, so the shared ``settings`` is never edited and one scenario can be run under
        both engines. ``blocks`` are settings blocks merged into the copy
        (``TRAINER={"MUTATION": {...}}``); nested mappings merge key by key.
        """
        from ..orchestration.dispatch import engine_factory

        copied = Settings(self.settings.as_dict(), module=self.settings.module)
        copied["ORCHESTRATOR"] = {**dict(copied.get("ORCHESTRATOR") or {}), "ENGINE": engine}
        for name, value in blocks.items():
            _merge_block(copied, str(name).upper(), value)
        conf.validate_settings(copied)
        modules = {
            f"peer{index}": ToyPeer(
                state_dim=int(copied["TRAINING"]["STATE_DIM"]),
                action_dim=int(copied["TRAINING"]["ACTION_DIM"]),
            )
            for index in range(count)
        }
        self.peers = modules
        return engine_factory(copied, modules=modules)


def _merge_block(settings: Settings, name: str, value: Any) -> None:
    """Merge ``value`` into ``settings[name]`` key by key (a non-mapping replaces)."""
    current = settings.get(name)
    if isinstance(current, dict) and isinstance(value, dict):
        merged = dict(current)
        for key, item in value.items():
            if isinstance(merged.get(key), dict) and isinstance(item, dict):
                merged[key] = {**merged[key], **item}
            else:
                merged[key] = item
        settings[name] = merged
    else:
        settings[name] = value


def build_harness(
    *,
    state_dim: int = 8,
    action_dim: int = 3,
    peers: int = 2,
    directory: Path | None = None,
) -> Harness:
    """Create a self-contained harness in a temporary directory."""
    root = Path(directory or tempfile.mkdtemp(prefix="mactf-harness-"))
    values = copy.deepcopy(conf.DEFAULTS)
    values["PATHS"]["ROOT"] = str(root)
    values["TRAINING"].update(
        {
            "STATE_DIM": state_dim,
            "ACTION_DIM": action_dim,
            "BATCH_SIZE": 8,
            "EPOCHS": 1,
            "STEPS_PER_EPOCH": 1,
        }
    )
    values["DEVICE"] = "cpu"
    values["AMP"] = False
    values["SANDBOX"]["PROVIDER"] = "local"
    values["COOPETITION"]["DISAGREEMENT_MAX"] = 1.0
    settings = Settings(values, module="mactf.testing")
    conf.validate_settings(settings)
    runtime = build_runtime(settings)
    return Harness(
        settings=settings,
        runtime=runtime,
        scorer=build_scorer(settings),
        directory=root,
    )


def make_ledger(peer: str = "peer", **events: float):
    """Build a ledger from ``rule_id=magnitude`` keyword arguments."""
    from ..scoring.ledger import Ledger

    ledger = Ledger(peer=peer)
    for rule_id, magnitude in events.items():
        ledger.record(rule_id.upper(), magnitude=magnitude)
    return ledger
