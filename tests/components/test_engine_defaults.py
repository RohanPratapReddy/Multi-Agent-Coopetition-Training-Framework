"""T3, the Orchestrator is untouched, and a project that writes no ``ENGINE`` cannot tell (document 17 section 13.2)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mactf
from mactf.orchestration.dispatch import engine_for
from mactf.orchestration.orchestrator import Orchestrator

FIXTURE = Path(__file__).parent / "fixtures" / "orchestrator.sha256"


def test_no_engine_key_changes_nothing(settings):
    assert engine_for(settings) is Orchestrator
    settings["ORCHESTRATOR"].pop("ENGINE", None)
    assert engine_for(settings) is Orchestrator
    settings["ORCHESTRATOR"] = {}
    assert engine_for(settings) is Orchestrator
    settings["ORCHESTRATOR"] = None
    assert engine_for(settings) is Orchestrator


def test_orchestrator_files_are_untouched():
    """The bytes of ``Orchestrator`` and of the two modules it runs on equal the digests recorded before P5 (T3)."""
    root = Path(mactf.__file__).parent.parent
    recorded = dict(line.split("  ", 1)[::-1] for line in FIXTURE.read_text().splitlines() if line.strip())
    assert set(recorded) == {
        "mactf/orchestration/orchestrator.py", "mactf/orchestration/phases.py", "mactf/orchestration/checkpoint.py",
    }
    for name, digest in recorded.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, f"{name} was edited"
