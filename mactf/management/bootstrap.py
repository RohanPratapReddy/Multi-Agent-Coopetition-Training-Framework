"""Options that change *which settings are loaded* (document 16, section 4).

They are read by a pre-parser that knows nothing about any command, because they must be
understood before settings exist.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..exceptions import CommandError


class _PreParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise CommandError.usage(message)


@dataclass(frozen=True)
class Bootstrap:
    settings: str | None          # --settings MODULE
    pythonpath: tuple[str, ...]   # --pythonpath PATH, repeatable, os.pathsep-separable
    config: tuple[str, ...]       # --config FILE, repeatable, in order
    overrides: tuple[str, ...]    # --set KEY=VALUE, repeatable, in order
    no_discover: bool             # --no-discover
    traceback: bool               # --traceback
    rest: tuple[str, ...]         # the arguments with the options above removed


BOOTSTRAP_FLAGS = ("--settings", "--pythonpath", "--config", "--set", "--no-discover", "--traceback")


def parse_bootstrap(args: Sequence[str]) -> Bootstrap:
    """Split the bootstrap options from the arguments a command parser will see.

    Everything after a literal ``--`` is returned untouched in ``rest``; no mactf
    command takes a ``--`` remainder today, but the rule means none will have its
    arguments eaten if one does.
    """
    args = list(args)
    tail: list[str] = []
    if "--" in args:
        cut = args.index("--")
        args, tail = args[:cut], args[cut:]
    parser = _PreParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--no-discover", action="store_true", default=False)
    parser.add_argument("--traceback", action="store_true", default=False)
    namespace, rest = parser.parse_known_args(args)
    paths: list[str] = []
    for entry in namespace.pythonpath:
        paths.extend(part for part in entry.split(os.pathsep) if part)
    return Bootstrap(
        settings=namespace.settings,
        pythonpath=tuple(paths),
        config=tuple(namespace.config),
        overrides=tuple(namespace.overrides),
        no_discover=namespace.no_discover or os.environ.get("MACTF_NO_DISCOVER") == "1",
        traceback=namespace.traceback,
        rest=tuple(rest) + tuple(tail),
    )


_MODULE_NAME = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


def apply_bootstrap(boot: Bootstrap):
    """Make the bootstrap options true: ``sys.path``, then the settings module, then configure.

    Returns the resolved ``Settings``. ``ImproperlyConfigured`` propagates to the caller.
    """
    from .. import conf

    resolved: list[str] = []
    for entry in boot.pythonpath:
        directory = Path(entry).expanduser().resolve()
        if not directory.is_dir():
            raise CommandError.usage(f"--pythonpath entry {entry!r} is not a directory")
        resolved.append(str(directory))
    # Inserting each at the front in reverse leaves the first entry first.
    for text in reversed(resolved):
        if text not in sys.path:
            sys.path.insert(0, text)

    module = boot.settings
    if module is not None and not _MODULE_NAME.match(module):
        hint = ""
        if "/" in module or "\\" in module or module.endswith(".py"):
            hint = " (pass the dotted module name and use --pythonpath for its parent)"
        raise CommandError.usage(f"--settings {module!r} is not a dotted module name{hint}")
    explicit = bool(boot.settings or boot.config or boot.overrides)
    return conf.configure(
        module, force=explicit, overrides=boot.overrides, config_files=boot.config
    )


@dataclass(frozen=True)
class Discovery:
    module: str          # "myswarm.settings"
    root: Path           # the directory holding manage.py
    pythonpath: Path     # root.parent, exactly what the generated manage.py inserts
    manage_py: Path


_MANAGE_RE = re.compile(
    r"""os\.environ\.setdefault\(\s*["']MACTF_SETTINGS_MODULE["']\s*,\s*["']([A-Za-z_][\w.]*)["']\s*\)"""
)


def discover_project(start: Path | None = None, *, max_levels: int = 6) -> Discovery | None:
    """Find the generated project at or above ``start``; never executes anything."""
    directory = (start or Path.cwd()).resolve()
    for level, candidate in enumerate((directory, *directory.parents)):
        if level > max_levels:
            break
        manage = candidate / "manage.py"
        if not manage.is_file():
            continue
        try:
            text = manage.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _MANAGE_RE.search(text)
        if match:
            return Discovery(match.group(1), candidate, candidate.parent, manage)
    return None


def settings_chosen(args: Sequence[str]) -> bool:
    """True when ``--settings`` is in ``args`` or ``MACTF_SETTINGS_MODULE`` is non-empty."""
    for arg in args:
        if arg == "--":
            break
        if arg == "--settings" or arg.startswith("--settings="):
            return True
    return bool(os.environ.get("MACTF_SETTINGS_MODULE"))


def discovery_disabled(args: Sequence[str]) -> bool:
    return "--no-discover" in args or os.environ.get("MACTF_NO_DISCOVER") == "1"
