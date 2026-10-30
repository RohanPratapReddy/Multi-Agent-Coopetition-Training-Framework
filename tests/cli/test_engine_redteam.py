"""``mactf redteam`` under the trainer engine (row 29)."""

from __future__ import annotations

import pytest

from mactf.management.commands.redteam import INJECTORS

pytest.importorskip("torch")

CPU = ("--device", "cpu")


def test_injectors_run_under_the_trainer(project):
    out = project.invoke(
        "redteam", "--epochs", "1", "--run-id", "rt-trainer", "--format", "json",
        "--set", "ORCHESTRATOR.ENGINE=trainer", *CPU,
    )
    assert out.code == 0, out.stdout + out.stderr
    payload = out.json()
    assert payload["injectors"] == len(INJECTORS) and payload["failed"] == 0
    assert {item["injector"] for item in payload["findings"]} == set(INJECTORS)
    card = payload["scores"]["jailbreak_attempt"] if "jailbreak_attempt" in payload.get("scores", {}) else None
    assert card is None or card["void"] is True
