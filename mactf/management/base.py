"""Management command framework, modeled on Django's ``manage.py``.

Commands declare ``help``, ``arguments``, and implement ``handle``. Arguments are
added with a tiny subset of ``argparse`` (positional and ``--flag``) so command
modules stay declarative and testable.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Sequence

from ..exceptions import CommandError, ImproperlyConfigured

COMMAND_PACKAGES = (
    "mactf.management.commands",
)


class CommandParser(argparse.ArgumentParser):
    """Parser that raises ``CommandError`` instead of exiting."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise CommandError(message)


class BaseCommand:
    """Base class for every management command."""

    help: str = ""
    requires_system_checks: bool = False
    arguments: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def __init__(self, *, stdout=None, stderr=None, settings: Any | None = None) -> None:
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.settings = settings

    # -- argparse wiring -------------------------------------------------
    def create_parser(self, prog_name: str, subcommand: str) -> CommandParser:
        parser = CommandParser(
            prog=f"{prog_name} {subcommand}",
            description=self.help or None,
            add_help=True,
        )
        self.add_arguments(parser)
        return parser

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        for flags, kwargs in self.arguments:
            parser.add_argument(*flags, **kwargs)

    # -- execution -------------------------------------------------------
    def run_from_argv(self, argv: Sequence[str]) -> int:
        parser = self.create_parser(argv[0], argv[1])
        options = parser.parse_args(list(argv[2:]))
        try:
            self.execute(options)
        except CommandError as exc:
            self.stderr.write(f"CommandError: {exc}\n")
            return 1
        except ImproperlyConfigured as exc:
            self.stderr.write(f"ImproperlyConfigured: {exc}\n")
            return 2
        return 0

    def execute(self, options: argparse.Namespace) -> Any:
        if self.requires_system_checks:
            from ..checks import run_checks

            outcome = run_checks()
            if not outcome.ok:
                for message in outcome.messages:
                    self.stderr.write(f"{message}\n")
                raise CommandError(outcome.summary())
        return self.handle(**vars(options))

    def handle(self, **options: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__} must implement handle(**options)"
        )

    # -- output helpers --------------------------------------------------
    def success(self, message: str) -> None:
        self.stdout.write(f"{message}\n")

    def style(self, message: str) -> str:
        return message


def load_command(name: str) -> type[BaseCommand]:
    """Import a command class by name, searching the command packages."""
    errors: list[str] = []
    for package in COMMAND_PACKAGES:
        module_path = f"{package}.{name}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            errors.append(str(exc))
            continue
        command_class = getattr(module, "Command", None)
        if command_class is None or not issubclass(command_class, BaseCommand):
            raise CommandError(
                f"{module_path} must define a Command class deriving from BaseCommand"
            )
        return command_class
    raise CommandError(
        f"unknown command {name!r}; no module found in {', '.join(COMMAND_PACKAGES)}"
    )


def get_commands() -> dict[str, type[BaseCommand]]:
    """Discover every available command."""
    import pkgutil

    commands: dict[str, type[BaseCommand]] = {}
    for package_name in COMMAND_PACKAGES:
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            continue
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name.startswith("_"):
                continue
            try:
                commands[module_info.name] = load_command(module_info.name)
            except CommandError:
                continue
    return commands


def execute_from_command_line(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``manage.py``."""
    from .. import apps, conf

    argv = list(argv if argv is not None else sys.argv)
    prog_name = argv[0] if argv else "manage.py"
    conf.configure()

    if len(argv) < 2:
        _print_help(prog_name)
        return 1

    subcommand = argv[1]
    if subcommand in {"help", "--help", "-h"}:
        _print_help(prog_name)
        return 0

    if subcommand in {"version", "--version"}:
        from .. import __version__

        print(__version__)
        return 0

    try:
        command_class = load_command(subcommand)
    except CommandError as exc:
        print(f"Unknown command: {subcommand}\n{exc}", file=sys.stderr)
        return 1

    apps.populate()
    command = command_class()
    return command.run_from_argv(argv)


def _print_help(prog_name: str) -> None:
    print(f"Usage: {prog_name} <command> [options]\n")
    print("Available commands:")
    for name, command_class in sorted(get_commands().items()):
        summary = (command_class.help or "").strip().split("\n")[0]
        print(f"  {name:<14} {summary}")
