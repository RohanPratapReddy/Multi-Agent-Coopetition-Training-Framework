"""test client layering (split from test_structure.py)."""

from __future__ import annotations

import ast


from pathlib import Path


import mactf.client


PACKAGE = Path(mactf.client.__file__).parent


def test_module_scope_imports_only_http_exceptions_and_wire():
    allowed = {"http", "exceptions", "wire"}
    for source in PACKAGE.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level >= 2:
                head = (node.module or "").split(".")[0]
                assert head in allowed, f"{source.name} imports mactf.{node.module} at module scope"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("torch", "mactf.")), f"{source.name}: {alias.name}"
