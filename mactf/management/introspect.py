"""One introspection layer for ``help``, ``__complete``, and the generated reference.

Document 16, section 8.19.1. They all read argparse's own parser, so they cannot disagree
about an option. ``argparse._HelpAction``, ``_SubParsersAction``, ``_CountAction``,
``_AppendAction`` and the parser's ``_actions`` / ``_mutually_exclusive_groups`` are private
names, used because argparse offers no public way to enumerate a parser; a test builds a
parser with every action type handled here so a rename is a failing test rather than a
silent hole in completion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable

from .output import clean

GLOBAL_FLAGS = frozenset({
    "-v", "--verbose", "-q", "--quiet", "--format", "--json", "--no-color", "--force-color",
    "--traceback", "-y", "--yes", "--skip-checks", "--settings", "--pythonpath", "--config",
    "--set", "--no-discover"})

CATEGORY_ORDER = ("Training", "Runs", "Swarm", "Project", "Server", "Scaffold", "Meta")

GLOBAL_OPTIONS_HELP = (
    ("--settings MODULE", "Settings module to load (default: the project found from the working directory)."),
    ("--pythonpath PATH", "Add a directory to sys.path; repeatable."),
    ("--config FILE", "Overlay a YAML settings file above the environment; repeatable."),
    ("--set KEY=VALUE", "Override one setting, for example TRAINING.EPOCHS=2; repeatable."),
    ("--no-discover", "Do not look for a project; use the framework defaults."),
    ("--format FORMAT", "text, json, jsonl, or yaml, where the command supports it."),
    ("--json", "Same as --format json."),
    ("-v / -q", "More / less output; -v may be repeated."),
    ("--no-color / --force-color", "Colour is automatic; NO_COLOR and FORCE_COLOR are honoured."),
    ("-y, --yes", "Answer yes to the confirmation of a destructive command."),
    ("--traceback", "Show the Python stack instead of one error line."),
)


@dataclass(frozen=True)
class ArgInfo:
    dest: str
    flags: tuple[str, ...]            # () for a positional
    kind: str                         # "flag", "count", "option", "positional"
    help: str
    metavar: str | None
    choices: tuple[str, ...] | None
    default: Any
    required: bool
    repeatable: bool
    nargs: str | None                 # "?", "*", "+", an integer as text, or None
    completer: str | None
    is_global: bool


@dataclass(frozen=True)
class CommandInfo:
    name: str
    summary: str
    description: str
    category: str
    aliases: tuple[str, ...]
    requires_settings: bool
    destructive: bool
    formats: tuple[str, ...]
    examples: tuple[str, ...]
    exclusive: tuple[tuple[str, ...], ...]
    arguments: tuple[ArgInfo, ...]
    subcommands: tuple["CommandInfo", ...]


def _arg_info(action: argparse.Action) -> ArgInfo | None:
    if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)) \
            or action.help == argparse.SUPPRESS:
        return None                                   # help itself, the verb chooser, hidden aliases
    flags = tuple(action.option_strings)
    if not flags:
        kind = "positional"
    elif isinstance(action, argparse._CountAction):
        kind = "count"
    elif action.nargs == 0:
        kind = "flag"
    else:
        kind = "option"
    metavar = action.metavar[0] if isinstance(action.metavar, tuple) else action.metavar
    completer = getattr(action, "completer", None)
    return ArgInfo(
        dest=action.dest, flags=flags, kind=kind, help=" ".join((action.help or "").split()),
        metavar=metavar,
        choices=None if action.choices is None else tuple(str(c) for c in action.choices),
        default=None if action.default is argparse.SUPPRESS else clean(action.default),
        required=bool(action.required),
        repeatable=isinstance(action, (argparse._AppendAction, argparse._CountAction)),
        nargs=None if action.nargs is None else str(action.nargs),
        completer=completer if isinstance(completer, str) else None,
        is_global=bool(flags) and flags[0] in GLOBAL_FLAGS)


def _arguments(parser: argparse.ArgumentParser) -> tuple[ArgInfo, ...]:
    return tuple(info for info in map(_arg_info, parser._actions) if info is not None)


def _exclusive(parser: argparse.ArgumentParser) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(a.option_strings[0] for a in group._group_actions if a.option_strings)
        for group in parser._mutually_exclusive_groups
    )


def chooser_of(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    return next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)


def describe_command(command_class: type, name: str, *, prog: str = "mactf") -> CommandInfo:
    command = command_class()
    command.prog = prog
    command.name = name
    parser = command.create_parser(prog, name)
    chooser = chooser_of(parser)
    subs: tuple[CommandInfo, ...] = ()
    if chooser is not None:
        subs = tuple(CommandInfo(
            name=sub.name, summary=sub.help, description=sub.help, category="",
            aliases=tuple(sub.aliases), requires_settings=command.requires_settings,
            destructive=sub.destructive,
            formats=tuple(sub.output_formats or command.output_formats),
            examples=tuple(sub.examples),
            exclusive=_exclusive(chooser.choices[sub.name]),
            arguments=_arguments(chooser.choices[sub.name]),
            subcommands=()) for sub in command.subcommands)
    text = (command.help or "").strip()
    return CommandInfo(
        name=name, summary=text.splitlines()[0] if text else "",
        description=(command.description or command.help or "").strip(),
        category=command.category or "General", aliases=(),
        requires_settings=command.requires_settings,
        destructive=bool(command.destructive or any(sub.destructive for sub in subs)),
        formats=tuple(command.output_formats), examples=tuple(command.examples),
        exclusive=_exclusive(parser), arguments=_arguments(parser), subcommands=subs)


def ordered(infos: Iterable[CommandInfo]) -> list[CommandInfo]:
    rank = {name: position for position, name in enumerate(CATEGORY_ORDER)}

    def key(info: CommandInfo) -> tuple:
        if info.category in rank:
            return (0, rank[info.category], info.name)
        if info.category == "General":
            return (2, 0, info.name)
        return (1, info.category.lower(), info.name)

    return sorted(infos, key=key)


def overview_text(infos: Iterable[CommandInfo], *, prog: str, verbs: bool = False) -> str:
    items = ordered(infos)
    width = max((len(info.name) for info in items), default=0)
    if verbs:
        width = max([width] + [len(f"{info.name} {sub.name}") - 2
                               for info in items for sub in info.subcommands])
    lines = [f"usage: {prog} [global options] <command> [options]", ""]
    current = None
    for info in items:
        if info.category != current:
            lines += [""] if current is not None else []
            lines.append(info.category)
            current = info.category
        lines.append(f"  {info.name:<{width}}  {info.summary}".rstrip())
        if verbs:
            for sub in info.subcommands:
                lines.append(f"    {info.name + ' ' + sub.name:<{width - 2}}  {sub.summary}".rstrip())
    lines += ["", "global options (valid before or after the command name):"]
    option_width = max(len(label) for label, _ in GLOBAL_OPTIONS_HELP)
    lines += [f"  {label:<{option_width}}  {text}" for label, text in GLOBAL_OPTIONS_HELP]
    lines += ["", f"Run '{prog} help <command>' for a command's options, '{prog} help --all' to list every",
              f"subcommand, and '{prog} completion <shell>' to enable tab completion."]
    return "\n".join(lines)
