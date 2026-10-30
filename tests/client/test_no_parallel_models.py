"""test no parallel models (split from test_structure.py)."""

from __future__ import annotations

import ast


from pathlib import Path


import mactf.client


PACKAGE = Path(mactf.client.__file__).parent


def test_no_dataclass_duplicates_an_engine_model():
    """Only transport shapes are dataclasses here; every domain object is the framework's own."""
    names = set()
    for source in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef) and any(
                getattr(d, "id", getattr(getattr(d, "func", None), "id", "")) == "dataclass" for d in node.decorator_list
            ):
                names.add(node.name)
    assert names <= {"RunHandle", "Event", "Page", "Run", "Episode", "RuleEvent", "Checkpoint", "PointsEntry"}
