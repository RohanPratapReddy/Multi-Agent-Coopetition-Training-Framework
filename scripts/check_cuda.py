"""Verify the CUDA training path explicitly (device, AMP, dtype, aggregation)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MACTF_SETTINGS_MODULE"] = "training.settings"

from mactf import apps  # noqa: E402
from mactf.conf import configure, get_settings  # noqa: E402
from mactf.orchestration.orchestrator import Orchestrator, RunConfig  # noqa: E402


def main() -> int:
    import torch

    configure()
    apps.populate()
    if not torch.cuda.is_available():
        print("CUDA not available; skipping")
        return 0

    orchestrator = Orchestrator(get_settings())
    summary = orchestrator.run(RunConfig(epochs=2, device="cuda"))
    runtime = summary["runtime"]
    print("device  :", runtime["device"])
    print("dtype   :", runtime["dtype"])
    print("amp     :", runtime["amp"])
    print("gpu     :", runtime.get("gpu_name"))
    print("status  :", summary["status"])
    print("epochs  :", len(summary["history"]))

    assert runtime["device"].startswith("cuda"), runtime["device"]
    assert summary["status"] == "completed"
    assert len(summary["history"]) == 2
    for entry in summary["history"]:
        assert abs(sum(entry["weights"].values()) - 1.0) < 1e-5
    print("CUDA path OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
