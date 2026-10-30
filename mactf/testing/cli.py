"""``invoke``: the supported way to test a management command (document 16, section 8.21).

It drives the real entry point, so a test sees what a user sees: bootstrap, exit-code
classification, and stream separation included. Importing this module imports no torch.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence


@dataclass
class CliResult:
    code: int
    stdout: str
    stderr: str

    def json(self) -> Any:
        return json.loads(self.stdout)

    def jsonl(self) -> list[Any]:
        return [json.loads(line) for line in self.stdout.splitlines() if line.strip()]


class _Stdin(io.StringIO):
    """A non-interactive stdin; ``isatty`` is False so a confirmation refuses to prompt."""

    def isatty(self) -> bool:
        return False


def _reset_state() -> None:
    from .. import apps, conf

    conf.reset()
    apps.reset_registry()


@contextlib.contextmanager
def _environment(env: Mapping[str, str] | None, cwd: str | os.PathLike[str] | None) -> Iterator[None]:
    saved_env = dict(os.environ)
    saved_cwd = os.getcwd()
    saved_path = list(sys.path)
    try:
        if env:
            for key, value in env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = str(value)
        if cwd is not None:
            os.chdir(os.fspath(cwd))
        yield
    finally:
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
        sys.path[:] = saved_path


def invoke(argv: Sequence[str], *, input: str | None = None, env: Mapping[str, str] | None = None,
           cwd: str | os.PathLike[str] | None = None, prog: str = "mactf") -> CliResult:
    """Run ``execute_from_command_line([prog, *argv])`` in-process with captured streams.

    ``env`` is applied for the call and removed afterwards (a ``None`` value unsets a
    variable for the call), ``cwd`` likewise, and the settings cache is reset on entry and
    exit, so one call cannot leak a configuration into the next. ``input`` becomes
    ``stdin``. Returns the exit code instead of raising ``SystemExit``.
    """
    from ..management import execute_from_command_line

    out, err = io.StringIO(), io.StringIO()
    stdin = _Stdin(input or "")
    saved = sys.stdin, sys.stdout, sys.stderr
    with _environment(env, cwd):
        _reset_state()
        sys.stdin, sys.stdout, sys.stderr = stdin, out, err
        try:
            try:
                code = execute_from_command_line([prog, *argv])
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        finally:
            sys.stdin, sys.stdout, sys.stderr = saved
            _reset_state()
    return CliResult(int(code), out.getvalue(), err.getvalue())
