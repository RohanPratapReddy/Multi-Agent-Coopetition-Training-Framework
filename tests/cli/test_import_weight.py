import subprocess
import sys
from pathlib import Path

import pytest

import mactf.management.commands as _commands

HEAVY = ("torch", "transformers", "peft", "scipy")
# Commands whose handlers build a swarm still import nothing heavy at module level.
MODULES = sorted(
    f"mactf.management.commands.{path.stem}"
    for path in Path(_commands.__file__).parent.glob("*.py")
    if not path.name.startswith("_")
) + ["mactf.contrib.api.management.commands.serve"]

pytestmark = pytest.mark.subprocess


def _probe(statements: str) -> subprocess.CompletedProcess:
    code = "import sys\n" + statements + (
        f"\nbad = sorted(m for m in {HEAVY!r} if m in sys.modules)\nassert not bad, bad\n")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("module", MODULES)
def test_command_module_import_is_torch_free(module):
    run = _probe(f"import importlib; importlib.import_module({module!r})")
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("statements", [
    "from mactf.management import execute_from_command_line, get_commands, BaseCommand",
    "from mactf.testing.cli import invoke",
    "from mactf.management.introspect import describe_command\n"
    "from mactf.management import get_commands\n"
    "[describe_command(c, n) for n, c in get_commands(populate=False).items()]",
    "import mactf.management.completion as c, io\n"
    "c.complete_main(['bash', '--', 'mactf', 'runs', 'l'], stdout=io.StringIO())",
    "from mactf.management.cli import main; main(['version'])",
    "from mactf.management.cli import main; main(['completion', 'bash'])",
    "from mactf.management.cli import main; main(['help'])",
    "from mactf.management.cli import main; main(['rules', 'list'])",
    "from mactf.management.cli import main; main(['config', 'validate'])",
])
def test_cheap_paths_do_not_import_torch(statements):
    run = _probe(statements)
    assert run.returncode == 0, run.stderr
