"""``components scaffold`` -> edit -> configure -> train, and the same without the edit.

An unedited scaffold raises ``NotImplementedError`` on purpose; the run must refuse it up front with a coded,
one-line error (the probes), never start training and never print a raw traceback.
"""

from __future__ import annotations

import re

import torch

import pytest

from e2e_support import BASE_FLAGS, checkpoint_file, manage

pytestmark = pytest.mark.subprocess

WIRING = {
    "weight_update": ["--set", "TRAINER.WEIGHT_UPDATE=my_weight_update"],
    "peer_strategy": ["--set", "TRAINER.PEER_STRATEGY=my_peer_strategy"],
    "optimizer": ["--set", "OPTIMIZER.TYPE=my_optimizer"],
}
ENGINE = ["--set", "ORCHESTRATOR.ENGINE=trainer"]
EDITS = {
    "weight_update": ("raise NotImplementedError(\"return an AggregationReport (mactf.coopetition.weights.make_report)\")",
                      "from mactf.coopetition.weights import make_report\n        names = sorted(vectors)\n        flags = {n: bool(eligible.get(n, True)) if eligible else True for n in names}\n        live = [n for n in names if flags[n]]\n        return make_report({n: 1.0 / len(live) for n in live}, names, kappa=kappa, eligible=flags)"),
    "optimizer": ("raise NotImplementedError(\"update every p in each group['params'] from p.grad; return the closure loss\")",
                  "loss = closure() if closure is not None else None\n        for group in self.param_groups:\n            for p in group['params']:\n                if p.grad is not None:\n                    p.add_(p.grad, alpha=-group['lr'])\n        return loss"),
}


def _scaffold(project, kind):
    done = manage(project, "components", "scaffold", kind, f"my_{kind}")
    assert done.returncode == 0, done.stderr
    return project / "components" / f"my_{kind}.py"


def _run(project, kind, run_id="scaf"):
    return manage(project, "run", "--epochs", "2", "--run-id", run_id, *BASE_FLAGS, *ENGINE, *WIRING[kind],
                  "--set", f'COMPONENTS.MODULES=["components.my_{kind}"]')


@pytest.mark.parametrize("kind", sorted(WIRING))
def test_an_unedited_scaffold_is_refused_before_training(project, kind):
    _scaffold(project, kind)
    done = _run(project, kind)
    assert done.returncode != 0
    assert "Traceback" not in done.stderr + done.stdout
    assert re.search(r"\bE\d{3}\b", done.stderr) and ("NotImplementedError" in done.stderr or f"my_{kind}" in done.stderr)
    assert len(done.stderr.strip().splitlines()) <= 3
    assert not checkpoint_file(project, "scaf", 0).exists()


@pytest.mark.parametrize("kind", sorted(EDITS))
def test_an_edited_scaffold_trains(project, kind):
    path = _scaffold(project, kind)
    old, new = EDITS[kind]
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    done = _run(project, kind)
    assert done.returncode == 0, done.stderr + done.stdout[-600:]
    payload = torch.load(checkpoint_file(project, "scaf", 1), map_location="cpu", weights_only=False)
    recorded = payload["identity"]["components"]
    assert any(f"my_{kind}" in str(entry) for entry in recorded.values()), recorded
