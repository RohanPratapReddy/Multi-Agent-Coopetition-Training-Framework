"""The second process of ``test_a_run_with_mid_run_swaps_resumes_in_a_new_process``: ``python swap_resume_child.py job.json``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from swap_support import build, final_state, register_extensions  # noqa: E402

from mactf.components import overlay  # noqa: E402
from mactf.conf import Settings  # noqa: E402


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    settings = Settings(job["settings"], module="mactf.testing")
    with overlay():
        register_extensions()                     # what importing the project's modules does in a fresh interpreter
        trainer = build(settings, Path(job["checkpoints"]), seed=job["seed"] + 1000, script=False)
        result = trainer.fit(epochs=int(job["epochs"]), resume=True)
        payload = final_state(trainer)
        payload.update(epochs_run=result.epochs, notes=list(trainer.component_notes))
    Path(job["out"]).write_text(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
