"""mactf help: the commands, or one command's options."""

from __future__ import annotations

import difflib

from ...exceptions import CommandError
from ..base import BaseCommand, get_commands, unknown_command
from ..introspect import (
    GLOBAL_OPTIONS_HELP, chooser_of, describe_command, ordered, overview_text,
)


class Command(BaseCommand):
    help = "Show the commands, or one command's options."
    category = "Meta"
    requires_settings = False
    output_formats = ("text", "json")
    arguments = [
        (("topic",), {"nargs": "*", "metavar": "COMMAND [SUBCOMMAND]", "completer": "command",
                      "help": "A command, optionally followed by one of its subcommands."}),
        (("--all",), {"action": "store_true", "default": False,
                      "help": "List every subcommand under its command."}),
    ]
    examples = ("mactf help", "mactf help runs list", "mactf help --all",
                "mactf help --format json | jq '.commands[] | select(.destructive) | .name'")

    def _commands(self) -> dict[str, type]:
        """Without importing any app: ``help`` must answer in milliseconds (P18)."""
        if self.settings_error is not None:
            self.out.warn(f"app commands are not listed: {self.settings_error}")
            return get_commands(populate=False)
        return get_commands(populate=False, static=True)

    def handle(self, **options):
        import mactf

        topic = options["topic"]
        if len(topic) > 2:
            raise CommandError.usage("help takes a command and at most one subcommand")
        commands = self._commands()
        if not topic:
            infos = [describe_command(cls, name, prog=self.prog) for name, cls in commands.items()]
            payload = {
                "schema": 1, "program": self.prog, "version": mactf.__version__,
                "global_options": [{"usage": label, "help": text}
                                   for label, text in GLOBAL_OPTIONS_HELP],
                "commands": ordered(infos),
            }
            self.out.emit(payload, text=lambda: self.out.result(
                overview_text(infos, prog=self.prog, verbs=options["all"])))
            return 0
        name = topic[0]
        if name not in commands:
            raise unknown_command(name, commands)
        instance = commands[name]()
        instance.prog, instance.name = self.prog, name
        info = describe_command(commands[name], name, prog=self.prog)
        parser = instance.create_parser(self.prog, name)
        if len(topic) == 1:
            self.out.emit(info, text=lambda: self.out.result(parser.format_help().rstrip()))
            return 0
        if not info.subcommands:
            raise CommandError.usage(f"{name} has no subcommands")
        sub = next((s for s in info.subcommands if topic[1] == s.name or topic[1] in s.aliases), None)
        if sub is None:
            names = [s.name for s in info.subcommands]
            close = difflib.get_close_matches(topic[1], names, n=3, cutoff=0.5)
            raise CommandError.usage(
                f"{name} has no subcommand {topic[1]!r}; it has {', '.join(names)}"
                + (f". Did you mean {', '.join(close)}?" if close else ""))
        chooser = chooser_of(parser)
        self.out.emit(sub, text=lambda: self.out.result(chooser.choices[sub.name].format_help().rstrip()))
        return 0
