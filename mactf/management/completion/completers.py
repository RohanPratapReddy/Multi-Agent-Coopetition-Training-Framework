"""The named completers argument declarations refer to (document 16, section 9.3)."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Iterable, Iterator, Mapping

from . import Candidate, CompletionContext

LOG_KINDS = ("run", "epoch", "step", "loss", "score", "aggregation", "warning",
             "checkpoint", "health", "resume", "mutation", "log")


def _first_line(text: str) -> str:
    stripped = (text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _registry(ctx: CompletionContext):
    """A read-only run registry, or ``None``. Opens an existing file only; never creates, migrates, or locks."""
    settings = ctx.settings
    if settings is None:
        return None
    path = settings.database_path()
    if not path.is_file():
        return None
    from ...db import Registry

    return Registry(path, read_only=True, busy_timeout=0.25)      # cannot write, never waits long


def command(ctx: CompletionContext) -> Iterator[Candidate]:
    if not ctx.slot:
        for name, cls in sorted(ctx.commands.items()):
            yield Candidate(name, _first_line(cls.help))
    elif len(ctx.slot) == 1 and ctx.slot[0] in ctx.commands:
        from ..introspect import describe_command

        for sub in describe_command(ctx.commands[ctx.slot[0]], ctx.slot[0]).subcommands:
            yield Candidate(sub.name, sub.summary)


def shell(ctx: CompletionContext) -> Iterator[str]:
    from . import SHELLS

    yield from SHELLS


def _runs(ctx: CompletionContext, statuses: Iterable[str] | None) -> Iterator[Candidate]:
    registry = _registry(ctx)
    if registry is None:
        return
    from ..durations import format_timestamp

    for row in registry.list_runs(limit=50, id_prefix=ctx.prefix or None,
                                  status=None if statuses is None else sorted(statuses)):
        yield Candidate(row["id"], f"{row['status']} {format_timestamp(row.get('started_at'))}")


def run_id(ctx: CompletionContext) -> Iterator[Candidate]:
    yield Candidate("latest", "the newest run")
    yield from _runs(ctx, None)


def resumable_run_id(ctx: CompletionContext) -> Iterator[Candidate]:
    from ...orchestration.recovery import RESUMABLE_STATUSES

    yield from _runs(ctx, RESUMABLE_STATUSES)


def run_status(ctx: CompletionContext) -> Iterator[Candidate]:
    from ..runsupport import ALL_STATUSES, STATUS_GROUPS

    for name in sorted(ALL_STATUSES):
        yield Candidate(name)
    for name in sorted(STATUS_GROUPS):
        yield Candidate(name, "group: " + ", ".join(sorted(STATUS_GROUPS[name])))


def injector(ctx: CompletionContext) -> list[str]:
    return sorted(import_module("mactf.management.commands.redteam").INJECTORS)


def check_app(ctx: CompletionContext) -> list[str]:
    ctx.populate()
    from ...checks import list_checks

    return sorted({info.app for info in list_checks()})


def check_tag(ctx: CompletionContext) -> list[str]:
    ctx.populate()
    from ...checks import list_checks

    return sorted({tag for info in list_checks() for tag in info.tags})


def log_kind(ctx: CompletionContext) -> tuple[str, ...]:
    return LOG_KINDS


def _paths(ctx: CompletionContext) -> list[tuple[str, ...]]:
    settings = ctx.settings
    if settings is None:
        return []
    from ...conf import iter_leaves

    return [path for path, _ in iter_leaves(settings.as_dict())]


def _children(paths: list[tuple[str, ...]], partial: str) -> Iterator[tuple[str, bool]]:
    """The next path segment under the table ``partial`` names: ``(text, is_leaf)``."""
    head, dot, _ = partial.rpartition(".")
    base = head + dot
    depth = base.count(".")
    for path in paths:
        if len(path) <= depth or not ".".join(path).startswith(base):
            continue
        step = base + path[depth]
        leaf = len(path) == depth + 1
        yield (step if leaf else step + "."), leaf


def setting_key(ctx: CompletionContext) -> Iterator[str]:
    for text, _ in _children(_paths(ctx), ctx.prefix):
        yield text


def _value_at(settings: Any, dotted: str) -> Any:
    node: Any = settings.as_dict() if settings is not None else None
    for segment in dotted.split("."):
        if not isinstance(node, Mapping) or segment not in node:
            return None
        node = node[segment]
    return node


def setting_assignment(ctx: CompletionContext) -> Iterator[str]:
    key, equals, _ = ctx.prefix.partition("=")
    if not equals:
        for text, leaf in _children(_paths(ctx), key):
            yield text + "=" if leaf else text
        return
    if isinstance(_value_at(ctx.settings, key), bool):        # the one value a user cannot guess wrongly
        yield f"{key}=true"
        yield f"{key}=false"


def rule_id(ctx: CompletionContext) -> Iterator[Candidate]:
    from ...rules import ALL_RULES

    for rule in ALL_RULES:
        yield Candidate(rule.id, rule.name)


def rule_tag(ctx: CompletionContext) -> list[str]:
    from ...rules import ALL_RULES

    return sorted({tag for rule in ALL_RULES for tag in rule.tags})


def _peer_names(ctx: CompletionContext) -> list[str]:
    settings = ctx.settings
    declared = (settings.get("PEERS") if settings is not None else None) or []
    return [str(entry["name"]) for entry in declared if isinstance(entry, Mapping) and entry.get("name")]


def peer_name(ctx: CompletionContext) -> list[str]:
    return _peer_names(ctx)


def peer_assignment(ctx: CompletionContext) -> Iterator[str]:
    taken = {value.partition("=")[0] for value in ctx.given.get(ctx.dest, ())}
    for name in _peer_names(ctx):
        if name not in taken:
            yield name + "="


def component_kind(ctx: CompletionContext) -> list[str]:
    """Every kind id, including ``rl_function.<family>`` (document 17, 15.4)."""
    from ...components.kinds import all_kind_ids

    return sorted(all_kind_ids())


def component_name(ctx: CompletionContext) -> list[str]:
    """Names for the kind already typed on the line (``--kind K`` or the first positional)."""
    from ...components.known import BUILTIN_NAMES, known_names

    given = ctx.given.get("kind") or ()
    kind = given[0] if given else None
    if kind is None or kind not in BUILTIN_NAMES:
        return sorted({name for names in BUILTIN_NAMES.values() for name in names})
    return sorted(known_names(kind) or ())


def _slot_paths(ctx: CompletionContext) -> set[str]:
    """The slots a project has: the ones settings name, plus the ones every run has."""
    from ...components.kinds import RL_FAMILIES
    from ...components.known import configured_specs

    paths = {"TRAINER.WEIGHT_UPDATE", "TRAINER.PEER_STRATEGY", "TRAINER.POLICY", "TRAINER.AGGREGATOR",
             "OPTIMIZER", "SCORING.STAGES.reward", "SCORING.STAGES.penalty", "SCORING.STAGES.warning"}
    paths |= {f"RL.{family.upper()}" for family in RL_FAMILIES}
    if ctx.settings is not None:
        paths |= {slot for slot, _kind, _raw in configured_specs(ctx.settings)}
    return paths


def slot_path(ctx: CompletionContext) -> list[str]:
    """Slot paths of the configured project: ``TRAINER.WEIGHT_UPDATE``, ``RL.LOSS`` and so on."""
    return sorted(_slot_paths(ctx))


COMPLETERS: dict[str, Callable[[CompletionContext], Iterable[Any]]] = {
    "command": command, "shell": shell, "run_id": run_id, "resumable_run_id": resumable_run_id,
    "run_status": run_status, "injector": injector, "check_app": check_app, "check_tag": check_tag,
    "log_kind": log_kind, "setting_key": setting_key, "setting_assignment": setting_assignment,
    "rule_id": rule_id, "rule_tag": rule_tag, "peer_name": peer_name,
    "peer_assignment": peer_assignment, "component_kind": component_kind,
    "component_name": component_name, "slot_path": slot_path,
}
