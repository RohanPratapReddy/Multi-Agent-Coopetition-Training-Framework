"""Management command framework, modeled on Django's ``manage.py``.

Commands declare ``help``, ``arguments``, and implement ``handle``. Arguments are
added with ``argparse`` keywords (plus one extension key, ``completer``) so command
modules stay declarative and testable. Document 16 sections 3, 4, and 6.

The pipeline guarantees: a command's return value is its exit code; a parse error is a
usage message, never a traceback; every exception maps to one documented exit code.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import logging
import sys
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..exceptions import CommandError, ImproperlyConfigured, StateConflict
from .exitcodes import ExitCode, classify
from .output import Printer, resolve_color, silence_stdout

if TYPE_CHECKING:  # pragma: no cover
    from ..db import Registry
    from .bootstrap import Bootstrap

CORE_COMMAND_PACKAGES = (
    "mactf.management.commands",
)
#: The core set only; kept as an alias because existing code and tests read it.
COMMAND_PACKAGES = CORE_COMMAND_PACKAGES

#: Output formats a command may list in ``output_formats``.
ALL_FORMATS = ("text", "json", "jsonl", "yaml")

#: Help-only declarations of the bootstrap options (section 4.1): the pre-parser removes
#: their values before the command parser runs, so these exist for ``--help`` and completion.
_BOOTSTRAP_DESTS = ("bootstrap_settings", "bootstrap_pythonpath", "bootstrap_config",
                    "bootstrap_set", "bootstrap_no_discover")


def command_packages(*, populate: bool = True) -> tuple[str, ...]:
    """Core packages followed by every installed app's command package.

    ``populate`` makes an unready registry populate on demand, which is what
    ``execute_from_command_line`` does before dispatch.
    """
    from .. import apps as apps_module

    if populate and not apps_module.apps._ready:
        apps_module.populate()
    packages = list(CORE_COMMAND_PACKAGES)
    if apps_module.apps._ready:
        for config in apps_module.apps.get_app_configs():
            package = config.get_commands()
            if package and package not in packages:
                packages.append(package)
    return tuple(packages)


def static_app_command_packages() -> tuple[str, ...]:
    """Command packages of ``INSTALLED_APPS`` found by path alone, importing no app.

    Importing an app's package can import ``torch``; ``help`` and ``__complete`` must stay
    cheap, so they look for ``<app>/management/commands`` on disk instead. An app that
    relocates its commands through ``AppConfig.commands_module`` is found once the
    registry is populated, not here.
    """
    import importlib.util
    import os

    from .. import conf

    try:
        entries = conf.get_settings().get("INSTALLED_APPS") or []
    except Exception:  # noqa: BLE001 - broken settings: core commands only
        return ()
    found: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        head, *rest = entry.split(".")
        try:
            spec = importlib.util.find_spec(head)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            directory = os.path.join(location, *rest, "management", "commands")
            if os.path.isdir(directory):
                found.append(f"{entry}.management.commands")
                break
    return tuple(found)


class CommandParser(argparse.ArgumentParser):
    """Parser that raises ``CommandError`` instead of exiting.

    Two constructor defaults pin the parser's behaviour for every command, verb, and
    interpreter (``add_subparsers(parser_class=CommandParser)`` forwards the keywords).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # No unambiguous-prefix matching. The bootstrap pre-parser cannot see a prefix:
        # without this, ``--sett x`` passes it untouched, is accepted here as an
        # abbreviation of the help-only ``--settings``, and the command silently runs
        # against the wrong project.
        kwargs.setdefault("allow_abbrev", False)
        # Python 3.14 colours argparse's own help; ``--no-color`` and the Printer's single
        # colour decision could not reach it. Help is plain; ``Printer`` owns colour.
        if sys.version_info >= (3, 14):
            kwargs.setdefault("color", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:  # type: ignore[override]
        raise CommandError.usage(message)


# -- declarations ---------------------------------------------------------------------
def _epilog(prog_name: str, examples: Sequence[str]) -> str | None:
    """Render ``examples`` under an ``examples:`` heading, ``mactf`` replaced by ``prog_name``."""
    if not examples:
        return None
    lines = ["examples:"]
    for example in examples:
        text = example
        if text.startswith("mactf"):
            text = prog_name + text[len("mactf"):]
        lines.append(f"  {text}")
    return "\n".join(lines)


def _declare(parser: argparse.ArgumentParser, arguments: Sequence[Any],
             exclusive: Sequence[Sequence[str]] = ()) -> None:
    """Add declared arguments, wiring ``exclusive`` groups and the ``completer`` extension."""
    grouped = {flag for names in exclusive for flag in names}
    groups = {tuple(names): parser.add_mutually_exclusive_group() for names in exclusive}
    for names, kwargs in arguments:
        kwargs = dict(kwargs)                                  # never mutate a shared declaration
        completer = kwargs.pop("completer", None)
        target = next((groups[tuple(group)] for group in exclusive if names[0] in group), parser)
        action = target.add_argument(*names, **kwargs)
        action.completer = completer
    unknown = grouped - {names[0] for names, _ in arguments}
    if unknown:                                                # a typo here would silently drop a rule
        raise ValueError(f"exclusive names no declared argument: {sorted(unknown)}")


def _taken(parser: argparse.ArgumentParser, *flags: str) -> bool:
    return any(flag in parser._option_string_actions for flag in flags)


def add_global_options(
    parser: argparse.ArgumentParser, *, formats: Sequence[str], destructive: bool,
    skip_checks: bool, suppress: bool,
) -> None:
    """Declare the options every command accepts (section 6.1).

    ``suppress`` is True for subcommand parsers: their absent flags must not overwrite
    the value the top-level parser already parsed (``mactf runs --format json list`` and
    ``mactf runs list --format json`` mean the same thing). A flag the command declared
    itself wins: the global declaration is skipped when the flag is already taken.
    """
    hidden: Any = argparse.SUPPRESS
    group = parser.add_argument_group("global options")

    def default(value: Any) -> Any:
        return hidden if suppress else value

    if not _taken(parser, "-v", "--verbose"):
        group.add_argument("-v", "--verbose", action="count", default=default(0),
                           help="More output; repeat for more (-vv shows decisions, -vvv debug logging).")
    if not _taken(parser, "-q", "--quiet"):
        group.add_argument("-q", "--quiet", action="count", default=default(0),
                           help="Less output; only errors and the result with -q.")
    ordered = [fmt for fmt in ALL_FORMATS if fmt in formats]
    if len(ordered) > 1 and not _taken(parser, "--format"):
        group.add_argument("--format", choices=ordered, default=hidden if suppress else None,
                           help=f"Output format ({', '.join(ordered)}); text is for humans, "
                                "the others are a stable contract.")
    if "json" in formats and not _taken(parser, "--json"):
        group.add_argument("--json", action="store_true", default=default(False),
                           help="Same as --format json.")
    if not _taken(parser, "--no-color"):
        group.add_argument("--no-color", action="store_true", default=default(False),
                           help="Never emit ANSI colour.")
    if not _taken(parser, "--force-color"):
        group.add_argument("--force-color", action="store_true", default=default(False),
                           help="Always emit ANSI colour.")
    if not _taken(parser, "--traceback"):
        group.add_argument("--traceback", action="store_true", default=default(False),
                           help="Show the Python stack instead of one error line.")
    if destructive and not _taken(parser, "-y", "--yes"):
        group.add_argument("-y", "--yes", action="store_true", default=default(False),
                           help="Answer yes to the confirmation prompt.")
    if skip_checks and not _taken(parser, "--skip-checks"):
        group.add_argument("--skip-checks", action="store_true", default=default(False),
                           help="Skip the system checks.")
    if suppress:
        return                                                 # the group parser carries these
    boot = parser.add_argument_group("settings selection (read before the command runs)")
    boot_specs = (
        ("--settings", {"metavar": "MODULE", "help": "Settings module to load."}),
        ("--pythonpath", {"metavar": "PATH", "action": "append",
                          "help": "Add a directory to sys.path; repeatable."}),
        ("--config", {"metavar": "FILE", "action": "append",
                      "help": "Overlay a YAML settings file above the environment; repeatable."}),
        ("--set", {"metavar": "KEY=VALUE", "action": "append",
                   "help": "Override one setting, e.g. TRAINING.EPOCHS=2; repeatable."}),
        ("--no-discover", {"action": "store_true",
                           "help": "Do not look for a project; use the framework defaults."}),
    )
    for (flag, kwargs), dest in zip(boot_specs, _BOOTSTRAP_DESTS):
        if not _taken(parser, flag):
            boot.add_argument(flag, dest=dest, default=hidden, **kwargs)


class BaseCommand:
    """Base class for every management command."""

    #: One line, shown in `mactf help`.
    help: str = ""
    #: Long description for ``--help``; falls back to ``help``.
    description: str = ""
    #: Shell lines shown under "examples:" in ``--help``.
    examples: tuple[str, ...] = ()
    #: Heading in `mactf help`.
    category: str = "General"
    requires_system_checks: bool = False
    #: False for commands that must work with broken or absent settings.
    requires_settings: bool = True
    #: Formats the command can produce. "text" must be present.
    output_formats: tuple[str, ...] = ("text",)
    #: Adds -y/--yes and enables confirm().
    destructive: bool = False
    arguments: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    #: Sets of flags that may not be combined. The first flag of each argument names it.
    exclusive: tuple[tuple[str, ...], ...] = ()
    #: Command-local completers by name (section 9.3); they shadow the built-ins.
    completers: dict[str, Any] = {}

    def __init__(
        self, *, stdout=None, stderr=None, settings: Any | None = None, stdin=None,
        bootstrap: "Bootstrap | None" = None, settings_error: Exception | None = None,
    ) -> None:
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.stdin = stdin or sys.stdin
        self._settings = settings
        self.bootstrap = bootstrap
        self.settings_error = settings_error
        self.prog = "mactf"
        self.name = ""
        self.assume_yes = False
        self.destructive_sub = False
        self.traceback_wanted = False
        self.out = Printer(self.stdout, self.stderr, color=False)

    @property
    def settings(self) -> Any:
        """The constructor's ``settings`` argument, else ``conf.get_settings()``."""
        if self._settings is not None:
            return self._settings
        from .. import conf

        return conf.get_settings()

    @settings.setter
    def settings(self, value: Any) -> None:
        self._settings = value

    # -- argparse wiring -------------------------------------------------
    def _formats(self) -> tuple[str, ...]:
        return tuple(self.output_formats)

    def _destructive(self) -> bool:
        return self.destructive

    def create_parser(self, prog_name: str, subcommand: str) -> CommandParser:
        parser = CommandParser(
            prog=f"{prog_name} {subcommand}",
            description=(self.description or self.help or None),
            epilog=_epilog(prog_name, self.examples),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=True,
        )
        self.add_arguments(parser)
        add_global_options(
            parser, formats=self._formats(), destructive=self._destructive(),
            skip_checks=self.requires_system_checks, suppress=False,
        )
        return parser

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _declare(parser, self.arguments, self.exclusive)

    # -- execution -------------------------------------------------------
    def run_from_argv(self, argv: Sequence[str]) -> int:
        self.prog = argv[0]
        self.name = argv[1]
        parser = self.create_parser(argv[0], argv[1])
        try:
            options = parser.parse_args(list(argv[2:]))
        except CommandError as exc:                       # CommandParser.error
            self.stderr.write(f"{parser.prog}: error: {exc}\n")
            self.stderr.write(f"Try '{parser.prog} --help' for usage.\n")
            return int(ExitCode.USAGE)
        except SystemExit as exc:                         # argparse's own --help path
            return _exit_status(exc.code)
        return self._run(options)

    def _run(self, options: argparse.Namespace) -> int:
        traceback_wanted = bool(
            getattr(options, "traceback", False) or (self.bootstrap and self.bootstrap.traceback)
        )
        self.traceback_wanted = traceback_wanted
        handler = None
        try:
            handler = self._install_debug_logging(options)
            result = self.execute(options)
            return self._exit_code(result)
        except (KeyboardInterrupt, BrokenPipeError):
            raise                                          # execute_from_command_line owns these
        except Exception as exc:                           # noqa: BLE001 - classify() decides
            if traceback_wanted:
                raise
            code, line = classify(exc)
            self.stderr.write(line + "\n")
            if code == ExitCode.SOFTWARE:
                self.stderr.write("re-run with --traceback for the stack\n")
            return code
        finally:
            if handler is not None:
                logging.getLogger("mactf").removeHandler(handler)

    def _install_debug_logging(self, options: argparse.Namespace) -> logging.Handler | None:
        level = _verbosity(vars(options))
        if level < 3:
            return None
        handler = logging.StreamHandler(self.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger = logging.getLogger("mactf")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return handler

    @staticmethod
    def _exit_code(result: Any) -> int:
        """``handle()`` returns ``None`` (exit 0) or an ``int``; anything else is a bug."""
        if result is None:
            return 0
        if isinstance(result, bool) or not isinstance(result, int):
            raise TypeError(
                f"handle() must return an int or None, got {type(result).__name__}"
            )
        return int(result)

    def _resolve_globals(self, values: dict[str, Any]) -> None:
        """Derive ``format``, ``json``, ``verbosity``, ``color``, and ``yes`` from the flags."""
        for dest in _BOOTSTRAP_DESTS:                      # help-only declarations
            values.pop(dest, None)
        fmt = values.pop("format", None)
        as_json = bool(values.get("json"))
        if fmt is not None and as_json and fmt != "json":
            raise CommandError.usage(f"--json conflicts with --format {fmt}")
        fmt = fmt or ("json" if as_json else "text")
        values["format"] = fmt
        values["json"] = fmt == "json"                     # handlers that read options["json"] keep working
        values["verbosity"] = _verbosity(values)
        no_color = bool(values.get("no_color"))
        force_color = bool(values.get("force_color"))
        values["color"] = False if no_color else True if force_color else None
        values["yes"] = bool(values.get("yes"))

    def execute(self, options: argparse.Namespace) -> Any:
        values = vars(options)
        self._resolve_globals(values)                      # format, verbosity, color, yes
        self.out = Printer(self.stdout, self.stderr, fmt=values["format"],
                           verbosity=values["verbosity"], color=values["color"])
        self.assume_yes = values["yes"]
        if self.requires_settings and self.settings_error is not None:
            raise self.settings_error
        if self.requires_system_checks and not values.get("skip_checks"):
            from ..checks import run_checks

            outcome = run_checks()
            if not outcome.ok:
                for message in outcome.messages:
                    if getattr(message, "level", "ERROR") == "ERROR":
                        self.out.error(str(message))
                raise CommandError(outcome.summary())
        return self.handle(**values)

    def handle(self, **options: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__} must implement handle(**options)"
        )

    # -- settings and registry ---------------------------------------------
    def need_settings(self) -> Any:
        """The resolved settings, or the error that stopped them loading."""
        if self._settings is not None:
            return self._settings
        if self.settings_error is not None:
            raise self.settings_error
        return self.settings

    def registry_for_write(self) -> "Registry":
        """The registry with the schema guaranteed: ``ensure_registry(settings)``, which migrates."""
        from ..db import ensure_registry

        return ensure_registry(self.need_settings())

    def registry_for_read(self) -> "Registry | None":
        """``None`` when the database file does not exist; never creates it.

        Raises ``StateConflict`` when the schema is behind: a reader must not alter the
        file it is reading.
        """
        from ..db import SCHEMA_VERSION, Registry

        path = self.need_settings().database_path()
        if not path.exists():
            return None
        registry = Registry(path, read_only=True)
        version = registry.schema_version()
        if version < SCHEMA_VERSION:
            raise StateConflict(
                f"database schema {version} is behind {SCHEMA_VERSION}; run `{self.prog} migrate`",
                schema=version, expected=SCHEMA_VERSION,
            )
        return registry

    # -- output and prompts --------------------------------------------------
    def success(self, message: str) -> None:
        """The command's narrative line; kept for compatibility, equals ``out.result``."""
        self.out.result(message)

    def style(self, message: str) -> str:
        return message

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask on stderr (stdout stays clean for data); read from ``self.stdin``."""
        if self.assume_yes:
            return True
        isatty = getattr(self.stdin, "isatty", None)
        if not (isatty and isatty()):
            raise CommandError.usage(
                f"{message}: refusing to prompt without a terminal; pass --yes to proceed"
            )
        self.stderr.write(f"{message} [{'Y/n' if default else 'y/N'}] ")
        self.stderr.flush()
        try:
            answer = self.stdin.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.stderr.write("\n")
            return False
        return default if answer == "" else answer in ("y", "yes")

    def aborted(self) -> int:
        """What a destructive command returns when the operator declines."""
        self.out.result("aborted")
        return int(ExitCode.FAILURE)


def _exit_status(code: Any) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    sys.stderr.write(f"{code}\n")
    return int(ExitCode.FAILURE)


def _verbosity(values: Mapping[str, Any]) -> int:
    """``clamp(1 + verbose - quiet, 0, 3)``; ``quiet`` may be a count or a bool."""
    verbose = int(values.get("verbose") or 0)
    quiet = int(values.get("quiet") or 0)
    return max(0, min(3, 1 + verbose - quiet))


# -- command lookup -------------------------------------------------------------------
def unknown_command(name: str, commands: Mapping[str, Any]) -> CommandError:
    close = difflib.get_close_matches(name, sorted(commands), n=3, cutoff=0.6)
    hint = f". Did you mean {', '.join(close)}?" if close else ""
    if name == "serve" and "serve" not in commands:
        hint += " (add 'mactf.contrib.api' to INSTALLED_APPS to get serve)"
    return CommandError.usage(f"unknown command {name!r}{hint}")


def load_command(name: str, *, populate: bool | None = None) -> type[BaseCommand]:
    """Import a command class by name.

    ``populate=None`` is the classic behaviour (core first, then apps, populating on
    demand); ``False`` searches core only; ``True`` searches the populated app packages.
    """
    searched: list[str] = []
    if not name.isidentifier() or name.startswith("_"):
        # ``foo.bar`` would be imported as a submodule path; a leading underscore is hidden.
        raise unknown_command(name, _safe_commands())
    # Core first and without touching the app registry: a bare ``mactf run`` must
    # not import every installed app just to find its own command.
    modes = (False, True) if populate is None else (populate,)
    for mode in modes:
        for package in command_packages(populate=mode):
            if package in searched:
                continue
            searched.append(package)
            module_path = f"{package}.{name}"
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as exc:
                if exc.name not in {module_path, package}:
                    raise
                continue
            command_class = getattr(module, "Command", None)
            if command_class is None or not issubclass(command_class, BaseCommand):
                raise CommandError(
                    f"{module_path} must define a Command class deriving from BaseCommand"
                )
            return command_class
    raise unknown_command(name, _safe_commands())


def _safe_commands() -> dict[str, type[BaseCommand]]:
    try:
        return get_commands(populate=False, static=True)
    except Exception:  # noqa: BLE001 - suggestions are a courtesy
        return {}


def get_commands(*, populate: bool = True, static: bool = False) -> dict[str, type[BaseCommand]]:
    """Discover every available command.

    Two packages providing the same command name is an ``ImproperlyConfigured`` listing
    both: silently hiding one would make ``mactf help`` lie. ``populate=False`` lists core
    commands only (plus, with ``static=True``, app commands found by path without
    importing the apps; what ``help`` and completion do to stay cheap).
    """
    import pkgutil

    from .. import apps as apps_module

    if populate:
        packages = list(command_packages())
    else:
        packages = list(CORE_COMMAND_PACKAGES)
        if apps_module.apps._ready:
            packages = list(command_packages(populate=False))
        elif static:
            packages += [p for p in static_app_command_packages() if p not in packages]
    commands: dict[str, type[BaseCommand]] = {}
    origin: dict[str, str] = {}
    for package_name in packages:
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            continue
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name.startswith("_"):
                continue
            module_path = f"{package_name}.{module_info.name}"
            module = importlib.import_module(module_path)
            command_class = getattr(module, "Command", None)
            if command_class is None or not issubclass(command_class, BaseCommand):
                continue
            if module_info.name in origin:
                raise ImproperlyConfigured(
                    f"command {module_info.name!r} is provided by both "
                    f"{origin[module_info.name]!r} and {package_name!r}"
                )
            origin[module_info.name] = package_name
            commands[module_info.name] = command_class
    return commands


# -- the pipeline (section 4.4) ---------------------------------------------------------
def execute_from_command_line(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``manage.py`` and ``mactf``. Returns the process exit code."""
    argv = list(argv if argv is not None else sys.argv)
    prog = argv[0] if argv else "mactf"
    traceback_wanted = "--traceback" in argv
    try:
        return _dispatch(prog, argv[1:])
    except BrokenPipeError:
        silence_stdout()
        return int(ExitCode.PIPE)
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return int(ExitCode.INTERRUPTED)
    except Exception as exc:  # noqa: BLE001 - classify() decides what the user sees
        if traceback_wanted:
            raise
        code, line = classify(exc)
        sys.stderr.write(line + "\n")
        if code == ExitCode.SOFTWARE:
            sys.stderr.write("re-run with --traceback for the stack\n")
        return code


def _dispatch(prog: str, args: list[str]) -> int:
    from . import bootstrap as bootstrap_module

    # 1. No arguments: grouped help to stderr, usage error.
    if not args:
        _print_overview(prog, sys.stderr)
        return int(ExitCode.USAGE)

    name = args[0]
    # 2. Meta forms. ``__complete`` is dispatched before anything else loads.
    if name == "__complete":
        from .completion import complete_main

        return complete_main(args[1:], prog=prog)
    if name in ("-h", "--help"):
        name, args = "help", ["help", *args[1:]]
    elif name in ("--version", "-V"):
        name, args = "version", ["version", *args[1:]]
    elif name.startswith("-"):
        raise CommandError.usage(
            f"expected a command, got {name!r}; global options go after the command name"
        )

    # 3. Pre-parse the options that decide which settings load.
    boot = bootstrap_module.parse_bootstrap(args[1:])

    # 4. Resolve the command class from core packages only.
    command_class: type[BaseCommand] | None = None
    try:
        command_class = load_command(name, populate=False)
    except CommandError:
        command_class = None

    # 5. Configure.
    settings_error: Exception | None = None
    if command_class is not None and not command_class.requires_settings:
        try:
            bootstrap_module.apply_bootstrap(boot)
        except Exception as exc:  # noqa: BLE001 - a settings module may fail with anything
            # help, version, completion, doctor, scaffolding still work (P3); the command
            # reports the failure itself. A non-ImproperlyConfigured error is wrapped so it
            # prints as a configuration problem with its cause, not as an internal error.
            if isinstance(exc, ImproperlyConfigured):
                settings_error = exc
            else:
                settings_error = ImproperlyConfigured(f"settings failed to load: {type(exc).__name__}: {exc}")
                settings_error.__cause__ = exc
    else:
        bootstrap_module.apply_bootstrap(boot)

    # 6. App commands.
    if command_class is None:
        from .. import apps

        apps.populate()
        try:
            command_class = load_command(name, populate=True)
        except CommandError:
            raise unknown_command(name, get_commands()) from None

    # 7. Populate, when the command needs the installed apps.
    if command_class.requires_settings and settings_error is None:
        from .. import apps

        apps.populate()

    # 8. Instantiate and run.
    command = command_class(bootstrap=boot, settings_error=settings_error)
    return command.run_from_argv([prog, name, *boot.rest])


def _print_overview(prog: str, stream: Any) -> None:
    from .introspect import describe_command, overview_text

    infos = [describe_command(cls, name, prog=prog)
             for name, cls in get_commands(populate=False, static=True).items()]
    stream.write(overview_text(infos, prog=prog) + "\n")
