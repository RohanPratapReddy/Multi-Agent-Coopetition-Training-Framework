"""Helpers shared by the ``conf/`` tests: build and load throwaway projects."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


def purge_project(name: str) -> None:
    """Forget a generated project's modules so a rewrite is re-imported."""
    for module in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
        del sys.modules[module]


def scaffold(tmp_path: Path, name: str = "demo_swarm") -> Path:
    """``startproject`` into ``tmp_path``; return the package directory."""
    from mactf.management.commands import startproject

    startproject.write_project(name, target_root=tmp_path)
    return tmp_path / name


def load(tmp_path: Path, monkeypatch, name: str = "demo_swarm"):
    """Load ``<name>.settings`` from ``tmp_path`` with a clean module state."""
    from mactf import conf

    monkeypatch.syspath_prepend(str(tmp_path))
    purge_project(name)
    conf.reset()
    try:
        return conf.load_settings(f"{name}.settings")
    finally:
        conf.reset()


def edit_conf(package: Path, filename: str, mutate) -> None:
    """Parse ``conf/<filename>``, apply ``mutate(payload)``, write it back."""
    path = package / "conf" / filename
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_conf(package: Path, filename: str, payload: dict[str, Any]) -> None:
    """Write ``conf/<filename>`` from a mapping."""
    directory = package / "conf"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings to ``{"A.B.C": leaf}``; lists and scalars are leaves."""
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(leaves(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}
