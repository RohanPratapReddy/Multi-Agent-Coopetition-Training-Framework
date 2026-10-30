"""Management utilities: the commands' framework and its stable surface (document 16, 8.21).

``__all__`` is the stable surface; every other name in this package may change in any
release. The classes, errors, and the two functions import eagerly; the argparse types
resolve lazily from ``runsupport`` so a caller that only wants ``BaseCommand`` does not
import the registry layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import lazy_exports
from ..exceptions import CapabilityUnavailable, CommandError, ResourceNotFound, StateConflict
from .base import BaseCommand, CommandParser, execute_from_command_line, get_commands
from .exitcodes import ExitCode
from .group import GroupCommand, Sub
from .output import Column

_EXPORTS: dict[str, str] = {
    "positive_int": "runsupport", "non_negative_int": "runsupport",
    "duration_arg": "runsupport", "when_arg": "runsupport", "device_arg": "runsupport",
}

if TYPE_CHECKING:  # pragma: no cover
    from .runsupport import device_arg, duration_arg, non_negative_int, positive_int, when_arg

__all__ = [
    "BaseCommand", "GroupCommand", "Sub",              # the classes to subclass
    "Column",                                          # table columns for self.out.table
    "ExitCode",                                        # the contract of section 3
    "CommandError", "ResourceNotFound", "StateConflict", "CapabilityUnavailable",   # raise these
    "positive_int", "non_negative_int", "duration_arg", "when_arg", "device_arg",   # argparse types
    "execute_from_command_line", "get_commands",
]
# CommandParser stays importable from here for existing callers but is not part of the surface.
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
