"""The engine behind ``mactf __complete``: what could the user type here?

Document 16, section 9. It reads the same ``ArgInfo`` records ``mactf help`` reads, so
completion cannot offer an option ``--help`` does not list, nor miss one it does.
"""

from __future__ import annotations

import contextlib
import functools
import io
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO

from ..introspect import ArgInfo, CommandInfo, describe_command

FILES = 1
NOSPACE = 2
DIRS = 4
SHELLS = ("bash", "zsh", "fish", "powershell")
MAX_CANDIDATES = 200
_FOREVER = 1 << 30
_NO_SPACE_AFTER = ("=", "/", ".")
BOOTSTRAP_OPTIONS = frozenset({"--settings", "--pythonpath", "--config", "--set", "--no-discover"})

#: Every command declares the bootstrap options identically (4.1), so their completers
#: are named here once instead of on each declaration.
GLOBAL_COMPLETERS = {"--pythonpath": "directory", "--config": "file", "--set": "setting_assignment"}


@dataclass(frozen=True)
class Candidate:
    value: str
    description: str = ""


@dataclass
class Completion:
    candidates: list[Candidate]
    directive: int = 0


@functools.lru_cache(maxsize=1)
def _settings_for(words: tuple[str, ...]) -> Any | None:
    """The settings these words select, or ``None`` when they cannot be loaded. Never raises."""
    from ...exceptions import CommandError
    from ..bootstrap import apply_bootstrap, parse_bootstrap

    try:
        try:
            boot = parse_bootstrap(words)
        except CommandError:
            boot = parse_bootstrap(words[:-1])          # a dangling "--set" that has no value yet
        apply_bootstrap(boot)
        from ... import conf

        return conf.get_settings()
    except Exception:  # noqa: BLE001 - a broken project completes statically
        return None


@dataclass
class CompletionContext:
    """What a completer is told. ``prefix`` is the text typed for *this* value (after the
    ``=`` of ``--opt=value``), and the engine filters whatever a completer returns by it,
    so a completer may return everything cheap and let the engine narrow it, or use
    ``prefix`` to avoid an expensive full listing."""

    prefix: str
    command: str | None
    sub: str | None
    dest: str                                  # the argument being completed
    given: Mapping[str, Sequence[str]]         # dest -> raw values already on the line
    slot: Sequence[str]                        # values already given to this very argument
    words: Sequence[str]                       # the completed words after the program name
    commands: Mapping[str, type]

    @property
    def settings(self) -> Any | None:
        """The loaded settings, or ``None``. Loaded once per process, on first use."""
        return _settings_for(tuple(self.words))

    def populate(self) -> bool:
        """Populate the app registry (needed to see apps' checks); False when it cannot."""
        if self.settings is None:
            return False
        try:
            from ...apps import apps

            apps.populate()
            return True
        except Exception:  # noqa: BLE001
            return False


Completer = Callable[[CompletionContext], Iterable["str | Candidate"]]


# -- the walk: which argument is the cursor in -----------------------------------------
def _need(arg: ArgInfo) -> tuple[int, int]:
    """``(minimum, maximum)`` values an argument takes."""
    if arg.kind in ("flag", "count"):
        return (0, 0)
    nargs = arg.nargs
    if nargs is None:
        return (1, 1)
    if nargs == "?":
        return (0, 1)
    if nargs == "*":
        return (0, _FOREVER)
    if nargs == "+":
        return (1, _FOREVER)
    if nargs.isdigit():
        return (int(nargs), int(nargs))
    return (1, 1)


@dataclass
class _State:
    args: list[ArgInfo]
    exclusive: tuple[tuple[str, ...], ...]
    sub: CommandInfo | None = None
    pending: ArgInfo | None = None            # an option still expecting values
    pending_count: int = 0
    positional_only: bool = False             # after a literal "--"
    used: dict[str, int] = field(default_factory=dict)
    given: dict[str, list[str]] = field(default_factory=dict)
    filled: dict[str, int] = field(default_factory=dict)     # positional dest -> values taken

    def index(self) -> dict[str, ArgInfo]:
        return {flag: arg for arg in self.args for flag in arg.flags}

    def positionals(self) -> list[ArgInfo]:
        return [arg for arg in self.args if arg.kind == "positional"]


def _mark(state: _State, arg: ArgInfo) -> None:
    state.used[arg.dest] = state.used.get(arg.dest, 0) + 1
    state.given.setdefault(arg.dest, [])


def _positional(state: _State, word: str) -> None:
    for arg in state.positionals():
        if state.filled.get(arg.dest, 0) < _need(arg)[1]:
            state.filled[arg.dest] = state.filled.get(arg.dest, 0) + 1
            state.given.setdefault(arg.dest, []).append(word)
            return


def _step(state: _State, root: CommandInfo, word: str) -> None:
    index = state.index()
    if state.positional_only:
        _positional(state, word)
        return
    if word == "--":
        state.positional_only, state.pending = True, None
        return
    if state.pending is not None and (not word.startswith("-") or word == "-"):
        state.given[state.pending.dest].append(word)
        state.pending_count += 1
        if state.pending_count >= _need(state.pending)[1]:
            state.pending = None
        return
    if word.startswith("--"):
        flag, equals, value = word.partition("=")
        arg = index.get(flag)
        state.pending = None
        if arg is None:
            return
        _mark(state, arg)
        if equals:
            state.given[arg.dest].append(value)
        elif arg.kind == "option" and _need(arg)[1] > 0:
            state.pending, state.pending_count = arg, 0
        return
    if word.startswith("-") and len(word) > 1:               # -v, -vv, -n20, -n
        state.pending = None
        body = word[1:]
        for position, char in enumerate(body):
            arg = index.get("-" + char)
            if arg is None:
                return
            _mark(state, arg)
            if arg.kind in ("flag", "count"):
                continue
            attached = body[position + 1:]
            if attached:
                state.given[arg.dest].append(attached)
            else:
                state.pending, state.pending_count = arg, 0
            return
        return
    state.pending = None                                     # a plain word
    if root.subcommands and state.sub is None:
        chosen = next((sub for sub in root.subcommands
                       if word == sub.name or word in sub.aliases), None)
        if chosen is not None:
            # A verb's parser does not repeat the settings-selection options; they stay
            # valid because the group's parser declares them.
            taken = {flag for arg in chosen.arguments for flag in arg.flags}
            inherited = [arg for arg in root.arguments
                         if arg.flags and arg.flags[0] in BOOTSTRAP_OPTIONS
                         and not taken & set(arg.flags)]
            state.sub = chosen
            state.args = [*chosen.arguments, *inherited]
            state.exclusive = chosen.exclusive
            state.filled = {}
        return
    _positional(state, word)


def _walk(root: CommandInfo, words: Sequence[str]) -> _State:
    state = _State(args=list(root.arguments), exclusive=root.exclusive)
    # The group's own arguments stay valid before the verb; the verb's are added to them.
    for word in words:
        _step(state, root, word)
    return state


# -- candidate sources -------------------------------------------------------------------
def _brief(text: str, limit: int = 72) -> str:
    """The first sentence of ``text``, on one line, at most ``limit`` characters."""
    text = " ".join((text or "").split())
    cut = text.find(". ")
    if cut != -1:
        text = text[: cut + 1]
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _options(state: _State, current: str) -> list[Candidate]:
    """Flags that start with ``current`` and may still be given."""
    index = state.index()
    blocked: set[str] = set()
    for group in state.exclusive:                            # one of a mutually exclusive set is used
        dests = [index[flag].dest for flag in group if flag in index]
        if any(dest in state.used for dest in dests):
            blocked.update(dest for dest in dests if dest not in state.used)
    if "format" in state.used:
        blocked.add("json")
    if "json" in state.used:
        blocked.add("format")
    found: list[Candidate] = []
    for arg in state.args:
        if not arg.flags or arg.dest in blocked:
            continue
        if arg.dest in state.used and not arg.repeatable:
            continue
        found.extend(Candidate(flag, _brief(arg.help)) for flag in arg.flags if flag.startswith(current))
    return found


def _values(arg: ArgInfo, partial: str, state: _State, command_class: type,
            commands: Mapping[str, type], done: Sequence[str]) -> tuple[list[Candidate], int]:
    """Candidates for one argument's value, and the fallback directive when there are none."""
    if arg.choices is not None:
        return [Candidate(choice) for choice in arg.choices if choice.startswith(partial)], 0
    key = arg.completer or GLOBAL_COMPLETERS.get(arg.flags[0] if arg.flags else "")
    if key == "file":
        return [], FILES
    if key == "directory":
        return [], DIRS
    if key is None:
        return [], 0
    from .completers import COMPLETERS

    function = (getattr(command_class, "completers", None) or {}).get(key) or COMPLETERS.get(key)
    if function is None:
        return [], 0
    context = CompletionContext(
        prefix=partial, command=done[0], sub=state.sub.name if state.sub else None, dest=arg.dest,
        given=state.given, slot=list(state.given.get(arg.dest, ())), words=list(done),
        commands=commands)
    found: list[Candidate] = []
    try:
        for item in function(context):
            candidate = item if isinstance(item, Candidate) else Candidate(str(item))
            if candidate.value.startswith(partial):
                found.append(candidate)
    except Exception:  # noqa: BLE001 - a completer that raises costs the user only its suggestions
        return [], 0
    return found, 0


def _finish(found: Iterable[Candidate], directive: int = 0) -> Completion:
    unique: dict[str, Candidate] = {}
    for candidate in found:
        unique.setdefault(candidate.value, candidate)
    ordered = list(unique.values())[:MAX_CANDIDATES]
    if any(candidate.value.endswith(_NO_SPACE_AFTER) for candidate in ordered):
        directive |= NOSPACE
    if ordered:
        directive &= ~(FILES | DIRS)                         # file names are a fallback only
    elif directive & DIRS:
        directive &= ~FILES
    return Completion(ordered, directive)


def _commands(done: Sequence[str]) -> dict[str, type]:
    from ..base import get_commands

    core = get_commands(populate=False)
    if done and done[0] in core:
        return core
    if _settings_for(tuple(done)) is None:                   # cannot see app commands without settings
        return core
    try:
        return get_commands()
    except Exception:  # noqa: BLE001
        return core


def _first_line(text: str) -> str:
    return _brief((text or "").strip().splitlines()[0] if (text or "").strip() else "")


def _command_names(current: str, commands: Mapping[str, type]) -> list[Candidate]:
    if current.startswith("-"):
        meta = (("--help", "Show the commands."), ("-h", "Show the commands."),
                ("--version", "Print the version."))
        return [Candidate(flag, text) for flag, text in meta if flag.startswith(current)]
    return [Candidate(name, _first_line(cls.help)) for name, cls in sorted(commands.items())
            if name.startswith(current)]


# -- the decision --------------------------------------------------------------------------
def complete(words: Sequence[str]) -> Completion:
    """``words`` are the arguments after the program name; the last is the word being completed."""
    done, current = list(words[:-1]), words[-1]
    if not done:
        return _finish(_command_names(current, _commands(done)))
    if done[0].startswith("-"):
        return Completion([], 0)
    commands = _commands(done)
    command_class = commands.get(done[0])
    if command_class is None:
        return Completion([], 0)
    info = describe_command(command_class, done[0])
    state = _walk(info, done[1:])
    index = state.index()

    def values(arg: ArgInfo, partial: str) -> tuple[list[Candidate], int]:
        return _values(arg, partial, state, command_class, commands, done)

    # 1. --option=partial
    if current.startswith("--") and "=" in current and not state.positional_only:
        flag, _, partial = current.partition("=")
        arg = index.get(flag)
        if arg is not None and arg.kind == "option":
            found, _ = values(arg, partial)
            return _finish(Candidate(f"{flag}={item.value}", item.description) for item in found)
        return Completion([], 0)
    # 2. the value of an option that is still waiting for one
    pending = state.pending
    if pending is not None and not state.positional_only:
        if not (current.startswith("-") and state.pending_count >= _need(pending)[0]):
            found, directive = values(pending, current)
            return _finish(found, directive)
    # 3. an option name
    if current.startswith("-") and not state.positional_only:
        return _finish(_options(state, current))
    # 4. a subcommand
    if info.subcommands and state.sub is None:
        return _finish(Candidate(sub.name, sub.summary) for sub in info.subcommands
                       if sub.name.startswith(current))
    # 5. the next positional
    for arg in state.positionals():
        if state.filled.get(arg.dest, 0) < _need(arg)[1]:
            found, directive = values(arg, current)
            return _finish(found, directive)
    return Completion([], 0)


# -- output and the entry point -------------------------------------------------------------
def _flat(text: str) -> str:
    return " ".join(str(text).split())


def render(completion: Completion, shell: str) -> str:
    lines = [f":{completion.directive}"]
    for candidate in completion.candidates:
        value = _flat(candidate.value)
        description = _flat(candidate.description)
        if shell == "bash":
            description = ""                                  # bash cannot show one
        elif shell == "powershell":
            description = description.encode("ascii", "replace").decode("ascii")
        lines.append(f"{value}\t{description}" if description else value)
    return "\n".join(lines) + "\n"


def write_raw(stream: TextIO, text: str) -> None:
    """Write ``text`` as UTF-8 with ``\\n`` line endings, bypassing text-mode translation."""
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write(text.encode("utf-8"))
        buffer.flush()
    else:                                                     # an in-memory stream (tests)
        stream.write(text)
        stream.flush()


def complete_main(args: Sequence[str], *, stdout: TextIO | None = None, prog: str = "mactf") -> int:
    """``__complete <shell> [--new-word] -- <words...>``. Always returns 0; stderr stays empty."""
    stream = stdout if stdout is not None else sys.stdout
    shell = args[0] if args else ""
    rest = list(args[1:])
    new_word = rest[:1] == ["--new-word"]
    if new_word:
        rest = rest[1:]
    if shell not in SHELLS or rest[:1] != ["--"]:
        return 0
    words = rest[1:]
    if new_word:
        words.append("")
    if not words:
        words = [prog, ""]
    elif len(words) == 1:
        words.append("")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            completion = complete(words[1:])
    except (Exception, SystemExit):  # noqa: BLE001 - never let a completion bug reach the prompt
        completion = Completion([], 0)
    write_raw(stream, render(completion, shell))
    return 0
