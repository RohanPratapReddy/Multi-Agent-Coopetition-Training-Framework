"""test_every_command_meets_contract (document 16, sections 8.21 and 10.2)."""

import ast
import inspect

import pytest

from mactf.management import get_commands
from mactf.management.group import GroupCommand
from mactf.management.introspect import describe_command
from mactf.testing.cli import invoke

COMMANDS = sorted(get_commands().items())
IDS = [name for name, _ in COMMANDS]
# Argument names whose values come from a known set; add a name here when a new one appears.
KNOWN_SETS = {"run", "status", "injector", "app", "tag", "peer", "rule", "shell", "kind", "key", "topic",
              "slot"}


def _verbs(cls):
    return list(getattr(cls, "subcommands", ()) or ())


def _every_arg(info):
    yield from info.arguments
    for sub in info.subcommands:
        yield from _every_arg(sub)


def test_the_core_commands_are_all_present():
    assert {"run", "redteam", "check", "migrate", "inspect", "doctor", "shell", "sandbox", "startproject",
            "startpeer", "runs", "checkpoints", "logs", "config", "rules", "peers", "version", "help",
            "completion"} <= set(IDS)


@pytest.mark.parametrize("name,cls", COMMANDS, ids=IDS)
def test_every_command_meets_contract(name, cls):
    info = describe_command(cls, name)
    # 1. help, examples, category
    assert cls.help and cls.help.strip() == cls.help and "\n" not in cls.help
    assert cls.help.rstrip().endswith("."), f"{name}: help is one sentence ending in a full stop"
    assert any(example.startswith("mactf ") for example in cls.examples) or _verbs(cls), name
    assert cls.category and cls.category != "General", f"{name}: no category"
    # 2. text is always available, and no format is promised that the command lacks
    assert "text" in cls.output_formats
    assert set(cls.output_formats) <= {"text", "json", "jsonl", "yaml"}
    # 3. destructive verbs confirm and preview: --yes and --dry-run exist on them
    verbs = {sub.name: sub for sub in info.subcommands}
    targets = [verbs[sub.name] for sub in _verbs(cls) if sub.destructive]
    if cls.destructive and not _verbs(cls):
        targets.append(info)
    for target in targets:
        flags = {flag for arg in target.arguments for flag in arg.flags}
        assert "--dry-run" in flags, f"{name} {target.name}: a destructive verb needs --dry-run"
    # 5. the module's source: no input(), print(), sys.exit(), bare except, or unexplained
    #    `except Exception` (a handler must raise the exceptions of section 3)
    source = inspect.getsource(inspect.getmodule(cls))
    tree = ast.parse(source)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"input", "print"}, f"{name}: use self.out and self.confirm"
    assert not any(isinstance(n, ast.Attribute) and n.attr == "exit"
                   and isinstance(n.value, ast.Name) and n.value.id == "sys"
                   for n in ast.walk(tree)), f"{name}: raise a CommandError instead of sys.exit"
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert node.type is not None, f"{name}: bare except"
            if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                assert "noqa: BLE001" in lines[node.lineno - 1], \
                    f"{name} line {node.lineno}: `except Exception` needs a stated reason"
    # 7. every argument whose values are a known set declares them
    for arg in _every_arg(info):
        if arg.dest in KNOWN_SETS and arg.choices is None:
            assert arg.completer, f"{name} {arg.dest}: declare `choices` or a `completer`"


@pytest.mark.parametrize("name,cls", COMMANDS, ids=IDS)
def test_every_listed_format_is_accepted_by_argparse(name, cls):
    assert invoke([name, "--help"]).code == 0
    if len(cls.output_formats) < 2:                      # a text-only command declares no --format
        return
    for fmt in cls.output_formats:
        out = invoke([name, "--format", fmt, "--help"])
        assert out.code == 0, (name, fmt, out.stderr)
        assert out.stderr == "", (name, fmt, out.stderr)


@pytest.mark.parametrize("name,cls", COMMANDS, ids=IDS)
def test_an_unlisted_format_is_a_usage_error(name, cls):
    missing = sorted({"text", "json", "jsonl", "yaml"} - set(cls.output_formats))
    if missing and not _verbs(cls):
        assert invoke([name, "--format", missing[0]]).code == 2


@pytest.mark.parametrize("name,cls", COMMANDS, ids=IDS)
def test_group_commands_declare_their_verbs(name, cls):
    if issubclass(cls, GroupCommand):
        assert _verbs(cls), name
        assert len({sub.name for sub in _verbs(cls)}) == len(_verbs(cls))
        taken = [alias for sub in _verbs(cls) for alias in (sub.name, *sub.aliases)]
        assert len(taken) == len(set(taken)), f"{name}: a verb alias collides"
        for sub in _verbs(cls):
            assert hasattr(cls, f"sub_{sub.name}"), f"{name}: no sub_{sub.name} handler"


@pytest.mark.parametrize("name,cls", COMMANDS, ids=IDS)
def test_every_verb_has_an_example(name, cls):
    for sub in _verbs(cls):
        assert sub.examples, f"{name} {sub.name}: give each verb an example"
