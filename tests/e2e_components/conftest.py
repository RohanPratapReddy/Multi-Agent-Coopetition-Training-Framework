"""Fixtures of the independent end-to-end tests of document 17.

``project`` is a real ``startproject`` tree (made once per session, copied per test) driven through
real subprocesses, so a kill, a resume and a second process are the thing they are named after.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from e2e_support import PROJECT_NAME


@pytest.fixture(scope="session")
def _template(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("e2e-template")
    done = subprocess.run(
        [sys.executable, "-m", "mactf", "startproject", PROJECT_NAME],
        cwd=base, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return base / PROJECT_NAME


@pytest.fixture
def project(_template, tmp_path) -> Path:
    target = tmp_path / PROJECT_NAME
    shutil.copytree(_template, target)
    return target


# The component tests keep their helpers beside them; the end-to-end tests reuse two of those modules.
import sys as _sys
for _name in ("training", "components"):
    _path = str(Path(__file__).resolve().parent.parent / _name)
    if _path not in _sys.path:
        _sys.path.append(_path)
