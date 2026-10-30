"""Helpers of the independent end-to-end tests (subprocess driver, checkpoint reader)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch

PROJECT_NAME = "e2eproj"
BASE_FLAGS = (
    "--skip-checks", "-q",
    "--set", "TRAINING.BOOTSTRAP=false",
    "--set", "DETERMINISTIC=true",
    "--set", "TRAINER.BATCH_SIZE=8",
    "--set", "TRAINER.STEPS_PER_EPOCH=2",
)


def manage(project: Path, *args: str, timeout: float = 240) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(project), os.environ.get("PYTHONPATH", "")]))}
    return subprocess.run(
        [sys.executable, "manage.py", *args], cwd=project, capture_output=True, text=True, timeout=timeout, env=env
    )


def spawn(project: Path, *args: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(project), os.environ.get("PYTHONPATH", "")]))}
    return subprocess.Popen(
        [sys.executable, "manage.py", *args], cwd=project, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )


def checkpoint_file(project: Path, run_id: str, epoch: int) -> Path:
    return project / "var" / "checkpoints" / f"{run_id}-ep{epoch:04d}.pt"


def wait_for(path: Path, proc: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            time.sleep(0.05)
            return
        if proc.poll() is not None:
            raise AssertionError(f"process ended ({proc.returncode}) before {path.name} appeared")
        time.sleep(0.02)
    raise AssertionError(f"{path.name} never appeared")


def deep_equal(a, b, path: str = "") -> str | None:
    """The first path at which two checkpoint sections differ (tensors compared with ``torch.equal``)."""
    import numpy as np

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        ok = isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
        return None if ok else path
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return None if np.array_equal(a, b) else path
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return path + "{keys}"
        for key in a:
            found = deep_equal(a[key], b[key], f"{path}/{key}")
            if found:
                return found
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return path + "[len]"
        for index, (x, y) in enumerate(zip(a, b)):
            found = deep_equal(x, y, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if a == b else path


def state_of(project: Path, run_id: str, epoch: int, sections: Iterable[str]):
    payload = torch.load(checkpoint_file(project, run_id, epoch), map_location="cpu", weights_only=False)
    return {name: payload.get(name) for name in sections}


def assert_same_state(a, b) -> None:
    for section in a:
        assert a[section] or section == "schedulers", f"{section}: nothing to compare"
        found = deep_equal(a[section], b[section])
        assert found is None, f"{section}{found} differs after resume"
