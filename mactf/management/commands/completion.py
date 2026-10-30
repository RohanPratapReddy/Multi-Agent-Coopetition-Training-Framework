"""mactf completion: print a tab-completion script for a shell."""

from __future__ import annotations

import re

from ...exceptions import CommandError
from ..base import BaseCommand

SHELLS = ("bash", "zsh", "fish", "powershell")
_PROG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")      # starts with a letter, digit, or '_'


class Command(BaseCommand):
    help = "Print a tab-completion script for bash, zsh, fish, or PowerShell."
    category = "Meta"
    requires_settings = False
    output_formats = ("text",)
    arguments = [
        (("shell",), {"choices": list(SHELLS), "completer": "shell"}),
        (("--prog",), {"default": "mactf", "metavar": "NAME",
                       "help": "The program name to complete (default: mactf)."}),
    ]
    examples = ("mactf completion bash >> ~/.bashrc",
                "mactf completion powershell | Out-String | Invoke-Expression",
                "mactf completion zsh --prog manage.py")

    def handle(self, **options):
        if not _PROG.fullmatch(options["prog"]):
            raise CommandError.usage(
                f"--prog {options['prog']!r}: start with a letter, digit, or '_'; "
                "then use letters, digits, '.', '_' or '-'")
        from ..completion import write_raw
        from ..completion.scripts import render_script

        write_raw(self.stdout, render_script(options["shell"], options["prog"]))
        return 0
