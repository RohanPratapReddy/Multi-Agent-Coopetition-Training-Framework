"""``GroupCommand``: subcommands without a second dispatcher (document 16, section 6.2)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from ..exceptions import CommandError
from .base import BaseCommand, CommandParser, _declare, _epilog, add_global_options


@dataclass(frozen=True)
class Sub:
    name: str
    help: str
    arguments: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = ()
    #: None inherits the group's output_formats.
    output_formats: tuple[str, ...] | None = None
    destructive: bool = False
    aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    exclusive: tuple[tuple[str, ...], ...] = ()
    #: Method on the Command; defaults to "sub_" + name with "-" as "_".
    handler: str = ""

    @property
    def method(self) -> str:
        return self.handler or "sub_" + self.name.replace("-", "_")


class GroupCommand(BaseCommand):
    """A command whose first positional is a verb: ``mactf runs list``."""

    subcommands: tuple[Sub, ...] = ()
    #: When set the verb may be omitted. That verb's own arguments are then absent from
    #: ``options``, so its handler must give each of them a default.
    default_subcommand: str | None = None

    def _formats(self) -> tuple[str, ...]:
        """The union of every verb's formats, so ``--format`` is accepted before the verb."""
        formats = list(self.output_formats)
        for sub in self.subcommands:
            for fmt in sub.output_formats or ():
                if fmt not in formats:
                    formats.append(fmt)
        return tuple(formats)

    def _destructive(self) -> bool:
        return self.destructive or any(sub.destructive for sub in self.subcommands)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        action = parser.add_subparsers(
            dest="subcommand", metavar="<subcommand>", parser_class=CommandParser,
            required=self.default_subcommand is None,
        )
        for sub in self.subcommands:
            child = action.add_parser(
                sub.name, aliases=list(sub.aliases), help=sub.help, description=sub.help,
                epilog=_epilog(self.prog, sub.examples),
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            _declare(child, sub.arguments, sub.exclusive)
            add_global_options(
                child, formats=sub.output_formats or self.output_formats,
                destructive=sub.destructive, skip_checks=False, suppress=True,
            )

    def lookup(self, name: str | None) -> Sub:
        for sub in self.subcommands:
            if name == sub.name or name in sub.aliases:
                return sub
        raise CommandError.usage(
            f"{self.name} needs a subcommand: {', '.join(s.name for s in self.subcommands)}"
        )

    def handle(self, **options: Any) -> Any:
        sub = self.lookup(options.pop("subcommand", None) or self.default_subcommand)
        allowed = sub.output_formats or self.output_formats
        if options["format"] not in allowed:
            raise CommandError.usage(
                f"{self.name} {sub.name} does not support --format {options['format']} "
                f"(supported: {', '.join(allowed)})"
            )
        self.destructive_sub = sub.destructive
        return getattr(self, sub.method)(**options)
