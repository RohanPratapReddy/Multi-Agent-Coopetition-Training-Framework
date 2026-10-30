"""``startpeer`` - scaffold a peer module.

Mirrors Django's ``startapp``: it writes a ready-to-use ``nn.Module`` with the
value head and contract documented, so operators only fill in their architecture.
"""

from __future__ import annotations

import keyword
from pathlib import Path

from ..base import BaseCommand
from ..bootstrap import discover_project
from ..runsupport import positive_int
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


def peer_target(name: str, base: Path) -> tuple[Path, str]:
    """``(file, class_name)`` for a peer name, or a usage error."""
    raw = name.replace("\\", "/").strip("/")
    if not raw:
        raise CommandError.usage("a peer name is required")
    target = base / (raw if raw.endswith(".py") else f"{raw}.py")
    stem = Path(raw).stem
    if not stem.isidentifier() or keyword.iskeyword(stem):
        raise CommandError.usage(
            f"{stem!r} cannot be a module name; use letters, digits, and underscores, "
            "and do not start with a digit")
    root = base.resolve()
    resolved = target.resolve()
    if root != resolved and root not in resolved.parents:
        raise CommandError.usage(f"{name!r} would write outside {root}")
    class_name = "".join(part.capitalize() for part in stem.split("_")) or "Peer"
    return target, class_name


class Command(BaseCommand):
    help = "Scaffold a new peer nn.Module inside an app directory."
    category = "Scaffold"
    requires_settings = False
    output_formats = ("text", "json")
    examples = ("mactf startpeer critic_peer", "mactf startpeer peers/critic_peer --state-dim 32 --dry-run")
    arguments = [
        (("name",), {"help": "Peer name, for example `critic_peer` or `peers/critic_peer`."}),
        (("--dir",), {"default": None, "metavar": "PATH", "completer": "directory",
                      "help": "Base directory; defaults to the project root."}),
        (("--state-dim",), {"type": positive_int, "default": 16, "metavar": "N",
                            "help": "Input width of the generated module (default 16)."}),
        (("--action-dim",), {"type": positive_int, "default": 4, "metavar": "N",
                             "help": "Output width of the policy head (default 4)."}),
        (("--force",), {"action": "store_true", "default": False,
                        "help": "Overwrite an existing file."}),
        (("--dry-run",), {"action": "store_true", "default": False,
                          "help": "Print the path and class that would be written."}),
    ]

    def handle(self, **options):
        target, class_name = peer_target(options["name"], self._base(options["dir"]))
        existed = target.exists()
        if existed and not options["force"]:
            raise CommandError.conflict(f"{target} already exists; pass --force to overwrite")
        payload = {"peer": target.stem, "class": class_name, "path": str(target),
                   "overwrote": existed and not options["dry_run"], "dry_run": options["dry_run"]}
        if not options["dry_run"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                TEMPLATE.format(name=target.stem, class_name=class_name,
                                state_dim=options["state_dim"], action_dim=options["action_dim"]),
                encoding="utf-8", newline="\n")

        def text() -> None:
            self.out.result(f"{'would create' if options['dry_run'] else 'created'} {target}")
            self.out.result("register it by adding the module to INSTALLED_APPS, or by listing "
                            f"{class_name!r} in PEERS")
        self.out.emit(payload, text=text)
        return 0

    def _base(self, option: str | None) -> Path:
        if option:
            return Path(option).expanduser()
        if self.settings_error is None:
            return self.settings.path("ROOT")
        found = discover_project()
        return found.root if found is not None else Path.cwd()
