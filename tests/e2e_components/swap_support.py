"""A run that swaps its optimizer, weight update and tree while it trains, shared by the test and its child process."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
for _name in ("training", "components"):
    _path = str(HERE.parent / _name)
    if _path not in sys.path:
        sys.path.append(_path)

from mutation_support import Tiny  # noqa: E402
from persist_support import register_extensions  # noqa: E402

from mactf.training import MACTFTrainer  # noqa: E402
from mactf.training.callbacks import Callback  # noqa: E402
from mactf.training.mutation import MutationPlan, add_peer  # noqa: E402

LION = {"TYPE": "Lion", "LR": 1e-3, "WEIGHT_DECAY": 0.0, "SCHEDULER": "step_then_flat", "SCHEDULER_KWARGS": {"drop_at": 2}}
EMA = {"type": "ema_softmax", "params": {"alpha": 0.5}}
RUN_ID = "swap-run"


class Script(Callback):
    """Epoch 1: optimizer; epoch 2: weight update; epoch 3: queue a plan that grows the tree (applied at the top of epoch 4)."""

    name = "e2e_swap_script"

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        if epoch == 1:
            trainer.swap_optimizer("OPTIMIZER", LION)
        elif epoch == 2:
            trainer.swap_weight_update(None, EMA)
        elif epoch == 3:
            trainer.request_mutation(MutationPlan.of(add_peer("peer-4", like="peer-1")))   # queued: it is pending in the checkpoint


def build(settings, checkpoints: Path, *, seed: int = 7, script: bool = True) -> MACTFTrainer:
    torch.manual_seed(seed)
    trainer = MACTFTrainer(model=Tiny(), num_peers=3, settings=settings, checkpoint_dir=Path(checkpoints))
    trainer.run_id = RUN_ID
    if script:
        trainer.add_callback(Script())
    return trainer.build()


def final_state(trainer: MACTFTrainer) -> dict[str, Any]:
    return {
        "params": {path: peer.export_parameters().detach().tolist() for path, peer in trainer.peers.items()},
        "topology": trainer.topology()["digest"],
        "ema": dict(trainer.weight_plan.strategy_for("root", trainer.aggregator)._ema),
        "lr": {path: opt.backend.param_groups[0]["lr"] for path, opt in trainer.optimizers.items()},
        "optimizer": next(iter(trainer.optimizers.values())).backend._class_name,
    }


def settings_for(settings) -> Any:
    from mutation_support import mutation_settings
    from p2_support import configure

    mutation_settings(settings)
    settings["DETERMINISTIC"] = True
    settings["TRAINING"]["EPOCHS"] = 7
    configure(settings, COMPONENTS={"MODULES": []})
    return settings


__all__ = ["register_extensions", "build", "final_state", "settings_for", "RUN_ID"]
