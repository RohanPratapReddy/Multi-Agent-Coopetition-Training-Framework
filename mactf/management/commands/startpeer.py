"""``startpeer`` - scaffold a peer module.

Mirrors Django's ``startapp``: it writes a ready-to-use ``nn.Module`` with the
value head and contract documented, so operators only fill in their architecture.
"""

from __future__ import annotations

from pathlib import Path

from ..base import BaseCommand
from ...conf import get_settings
from ...exceptions import CommandError

TEMPLATE = '''"""Peer module: {name}.

MACTF wraps this automatically with :class:`mactf.coopetition.adapter.PeerAdapter`,
so the only contract to honour is the return shape. Return either:

* ``(logits, value)`` - preferred, and lets the framework train your value head, or
* ``logits`` only - the adapter attaches a value head for you.

``logits`` must have shape ``(batch, action_dim)`` and ``value`` ``(batch, 1)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class {class_name}(nn.Module):
    """A peer in the swarm."""

    def __init__(self, state_dim: int = {state_dim}, action_dim: int = {action_dim}, hidden: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return logits, value


# Declared so MACTF's app discovery registers this peer automatically.
PEERS = [{class_name}]
'''


class Command(BaseCommand):
    help = "Scaffold a new peer nn.Module inside an app directory."
    arguments = [
        (("name",), {"help": "Peer name, for example `critic_peer` or `peers/critic_peer`."}),
        (("--dir",), {"default": None, "help": "Target directory; defaults to the project root."}),
        (("--state-dim",), {"type": int, "default": 16}),
        (("--action-dim",), {"type": int, "default": 4}),
        (("--force",), {"action": "store_true", "default": False}),
    ]

    def handle(self, **options):
        settings = get_settings()
        raw = options["name"].replace("\\", "/").strip("/")
        if not raw:
            raise CommandError("a peer name is required")

        base = Path(options.get("dir") or settings.path("ROOT"))
        target = base / (raw if raw.endswith(".py") else f"{raw}.py")
        if target.exists() and not options.get("force"):
            raise CommandError(f"{target} already exists; pass --force to overwrite")

        class_name = "".join(part.capitalize() for part in Path(raw).stem.split("_")) or "Peer"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            TEMPLATE.format(
                name=Path(raw).stem,
                class_name=class_name,
                state_dim=options["state_dim"],
                action_dim=options["action_dim"],
            ),
            encoding="utf-8",
        )
        self.success(f"created {target}")
        self.success(
            "register it by adding the module to INSTALLED_APPS, or by listing "
            f"{class_name!r} in PEERS"
        )
        return 0
