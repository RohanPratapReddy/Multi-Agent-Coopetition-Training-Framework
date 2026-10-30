"""The second process of ``test_a_mutated_run_resumes_in_a_new_process``.

Not a test: it is run with ``python mutation_resume_child.py <job.json>``. It builds the trainer
the way a fresh process would (the configuration, nothing else), resumes the checkpoint directory
and writes the final parameters, the topology and the journal to the job's ``out`` file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from mutation_support import Tiny  # noqa: E402

from mactf.conf import Settings  # noqa: E402
from mactf.training import MACTFTrainer  # noqa: E402


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    settings = Settings(job["settings"], module="mactf.testing")
    torch.manual_seed(int(job["seed"]))
    trainer = MACTFTrainer(
        model=Tiny(), num_peers=int(job["peers"]), settings=settings, checkpoint_dir=Path(job["checkpoints"])
    )
    trainer.run_id = job["run_id"]
    trainer.build()
    started = {
        "peers_after_build": sorted(trainer.peers),
        "journal_after_build": len(trainer.mutation_journal),
    }
    result = trainer.fit(epochs=int(job["epochs"]), resume=True)
    payload = {
        **started,
        "start_epoch": trainer.start_epoch,
        "epochs_run": result.epochs,
        "peers": sorted(trainer.peers),
        "retired": sorted(trainer.freeze.retired()),
        "topology": trainer.topology(),
        "journal": trainer.mutation_journal,
        "component_notes": list(trainer.component_notes),
        "params": {path: peer.export_parameters().detach().tolist() for path, peer in trainer.peers.items()},
        "audit": trainer.path_state.audit(),
        "pending": trainer.engine.pending(),
    }
    Path(job["out"]).write_text(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
