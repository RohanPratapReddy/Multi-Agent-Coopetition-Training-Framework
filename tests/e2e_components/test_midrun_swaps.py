"""Swap the optimizer, the weight update and the tree while a run trains; stop; resume in a real new process.

The resumed run must finish exactly where an uninterrupted run with the same swaps finishes: the swaps
are part of the checkpoint, not of the configuration the new process starts from.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from swap_support import build, final_state, register_extensions, settings_for

from mactf.components import overlay

HERE = Path(__file__).parent
EPOCHS, CUT = 7, 4


def test_a_run_with_mid_run_swaps_resumes_in_a_new_process(settings, tmp_path):
    settings_for(settings)
    with overlay():
        register_extensions()
        reference = build(settings, tmp_path / "ref")
        reference.fit(epochs=EPOCHS)
        expected = final_state(reference)
        reference.close()

        first = build(settings, tmp_path / "split")
        first.fit(epochs=CUT)                                   # swaps at epochs 1, 2 are in the file; the plan of 3 is pending
        assert sorted(first.peers) == ["peer-1", "peer-2", "peer-3"] and first.engine.pending()
        first.close()

    job = {"settings": settings.as_dict(), "seed": 7, "epochs": EPOCHS, "checkpoints": str(tmp_path / "split"),
           "out": str(tmp_path / "child.json")}
    (tmp_path / "job.json").write_text(json.dumps(job, default=str))
    run = subprocess.run([sys.executable, str(HERE / "swap_resume_child.py"), str(tmp_path / "job.json")],
                         capture_output=True, text=True, timeout=240)
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads((tmp_path / "child.json").read_text())

    assert got["epochs_run"] == EPOCHS - CUT
    assert sorted(got["params"]) == ["peer-1", "peer-2", "peer-3", "peer-4"]     # the pending plan ran in the new process
    assert got["optimizer"] == "Lion" and got["topology"] == expected["topology"]
    assert got["params"] == expected["params"]                  # bit-for-bit: float.tolist round-trips exactly
    assert got["ema"] == expected["ema"]
    assert got["lr"] == expected["lr"]
