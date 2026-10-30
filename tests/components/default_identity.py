"""Recorder for the default-identity fixtures (doc 17, section 19.4).

Run against the UNMODIFIED framework to (re)record the fixtures, and imported by
``test_default_identity`` to replay the same scenario against the current code::

    python -m tests.components.default_identity --record

The scenario is a two-peer, three-epoch run with no ``COMPONENTS`` configuration, driven
two ways: the ``Orchestrator`` (the frozen oracle) and ``MACTFTrainer``. Everything that
could differ between two processes (run ids, timestamps, durations) is excluded; what is
kept is numeric state compared bitwise.

A fixture changes only in a commit that names the behaviour change justifying it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "default_identity"
RUN = {
    "seed": 1337,
    "peers": 2,
    "epochs": 3,
    "state_dim": 4,
    "action_dim": 3,
    "batch_size": 8,
}


def _imports():
    import numpy as np
    import torch

    return np, torch


def canonical(value: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, floats by ``repr``."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)


def _default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_dict"):
        return value.as_dict()
    raise TypeError(f"not serialisable: {type(value).__name__}")


def _seed_everything(seed: int) -> None:
    np, torch = _imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _make_settings(directory: Path):
    from mactf import conf

    values = copy.deepcopy(conf.DEFAULTS)
    values["PATHS"]["ROOT"] = str(directory)
    values["TRAINING"].update(
        {
            "STATE_DIM": RUN["state_dim"],
            "ACTION_DIM": RUN["action_dim"],
            "BATCH_SIZE": RUN["batch_size"],
            "EPOCHS": RUN["epochs"],
            "STEPS_PER_EPOCH": 1,
            "BOOTSTRAP": False,
            "CHECKPOINT_EVERY": 1,
        }
    )
    values["SEED"] = RUN["seed"]
    values["DEVICE"] = "cpu"
    values["AMP"] = False
    values["SANDBOX"]["PROVIDER"] = "local"
    values["SANDBOX"]["ALLOW_FALLBACK"] = True
    values["COOPETITION"]["DISAGREEMENT_MAX"] = 1.0
    settings = conf.Settings(values, module="mactf.testing")
    conf.validate_settings(settings)
    return settings


def _models():
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(nn.Linear(RUN["state_dim"], 16), nn.Tanh())
            self.policy = nn.Linear(16, RUN["action_dim"])
            self.value = nn.Linear(16, 1)

        def forward(self, state):
            hidden = self.body(state)
            return self.policy(hidden), self.value(hidden)

    _seed_everything(RUN["seed"])
    return {f"peer-{index + 1}": Tiny() for index in range(RUN["peers"])}


def _flat(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().double().flatten().tolist()]


def _cards(cards: Any) -> Any:
    return json.loads(canonical(cards))


def run_orchestrator(directory: Path) -> dict[str, Any]:
    from mactf.orchestration.orchestrator import Orchestrator, RunConfig

    settings = _make_settings(directory)
    modules = _models()
    _seed_everything(RUN["seed"])
    orchestrator = Orchestrator(settings, modules=copy.deepcopy(modules))
    epochs: list[dict[str, Any]] = []

    def on_epoch(entry: dict[str, Any]) -> None:
        epochs.append(
            {
                "params": {
                    name: _flat(peer.export_parameters())
                    for name, peer in orchestrator.peers.items()
                },
                "kappa": entry["kappa"],
                "entropy": entry["entropy"],
                "weights": entry["weights"],
                "score": entry["score"],
                "signal": entry["signal_by_peer"],
                "scorecards": _cards(entry["scorecards"]),
                "adjustments": _cards(entry["adjustments"]),
            }
        )

    orchestrator.run(RunConfig(epochs=RUN["epochs"]), on_epoch=on_epoch)
    return {"epochs": epochs, "rng_after": _rng_digest()}


def run_trainer(directory: Path) -> dict[str, Any]:
    from mactf.training import MACTFTrainer
    from mactf.training.callbacks import Callback

    settings = _make_settings(directory)
    modules = _models()
    _seed_everything(RUN["seed"])
    snapshots: list[dict[str, Any]] = []

    class Spy(Callback):
        name = "default-identity-spy"

        def on_epoch_end(self, trainer, result):
            report = trainer.reports[-1]
            snapshots.append(
                {
                    "params": {
                        name: _flat(peer.export_parameters())
                        for name, peer in trainer.peers.items()
                    },
                    "loss": result.mean_loss,
                    "steps": result.steps,
                    "kappa": report.kappa,
                    "entropy": report.entropy,
                    "weights": dict(report.weights),
                    "signal": dict(trainer.signal_by_peer),
                }
            )

    trainer = MACTFTrainer(
        models=modules,
        settings=settings,
        checkpoint_dir=directory / "trainer-ckpt",
        callbacks=[Spy()],
    )
    result = trainer.fit(epochs=RUN["epochs"])
    for entry, card in zip(snapshots, _per_epoch_cards(result.episodes)):
        entry["scorecards"] = card
    latest = trainer.checkpoint.latest()
    out: dict[str, Any] = {"epochs": snapshots, "rng_after": _rng_digest()}
    if latest is not None:
        out.update(checkpoint_facts(latest.path, latest.episode))
    return out


#: Identity fields that are stable across processes. The rest (run id, timestamps,
#: host, command line, versions) vary by construction and are excluded from the hash.
STABLE_IDENTITY = (
    "episode",
    "global_step",
    "schema_version",
    "device",
    "dtype",
    "amp",
    "mode",
    "score",
)


def checkpoint_facts(path: Path, episode: int) -> dict[str, Any]:
    """The process-independent facts of a v2 checkpoint (14.3: its content hash).

    ``identity`` carries a run id, timestamps and host details, so the recorded hash is
    taken over the other fifteen section digests plus the stable identity fields, and the
    identity key set is recorded so a new ``components`` key is noticed.
    """
    import torch

    payload = torch.load(path, weights_only=False)
    sections = dict(payload["manifest"]["sections"])
    identity = payload["identity"]
    stable = {key: identity.get(key) for key in STABLE_IDENTITY}
    sections["identity"] = hashlib.sha256(canonical(stable).encode()).hexdigest()
    return {
        "checkpoint_episode": episode,
        "checkpoint_sections": {k: v for k, v in sections.items()},
        "checkpoint_identity_keys": sorted(identity),
        "checkpoint_content_hash": "sha256:"
        + hashlib.sha256(canonical(sections).encode()).hexdigest(),
    }


def _per_epoch_cards(episodes: Any) -> list[Any]:
    out = []
    for card in episodes:
        data = card.to_dict() if hasattr(card, "to_dict") else card
        out.append(_strip_volatile(json.loads(canonical(data))))
    return out


CHECKPOINT_KEYS = (
    "checkpoint_episode",
    "checkpoint_sections",
    "checkpoint_identity_keys",
    "checkpoint_content_hash",
)
_VOLATILE = {"created_at", "timestamp", "duration", "elapsed", "run_id", "time"}


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def _rng_digest() -> str:
    _, torch = _imports()
    state = torch.get_rng_state().numpy().tobytes()
    return hashlib.sha256(state).hexdigest()


def record_all() -> dict[str, Any]:
    """Run both legs from a fresh temp directory each; return the comparable payload."""
    import torch

    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(prefix="mactf-identity-a-") as a:
        orchestrator = run_orchestrator(Path(a))
    with tempfile.TemporaryDirectory(prefix="mactf-identity-b-") as b:
        trainer = run_trainer(Path(b))
    for leg in (orchestrator, trainer):
        for epoch in leg["epochs"]:
            if "scorecards" in epoch:
                epoch["scorecards"] = _strip_volatile(epoch["scorecards"])
    return {
        "torch": torch.__version__.split("+")[0].rsplit(".", 1)[0],
        "orchestrator": orchestrator,
        "trainer": trainer,
    }


def write_fixtures(payload: dict[str, Any], directory: Path = FIXTURE_DIR) -> None:
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(
        json.dumps({"run": RUN, "torch": payload["torch"], "legs": ["orchestrator", "trainer"]}, indent=2, sort_keys=True)
        + "\n"
    )
    arrays: dict[str, Any] = {}
    cards: dict[str, Any] = {}
    scalars: dict[str, Any] = {}
    for leg in ("orchestrator", "trainer"):
        data = payload[leg]
        scalars[leg] = {"rng_after": data["rng_after"]}
        for key in CHECKPOINT_KEYS:
            if key in data:
                scalars[leg][key] = data[key]
        cards[leg] = []
        scalars[leg]["epochs"] = []
        for index, epoch in enumerate(data["epochs"]):
            for name, flat in epoch["params"].items():
                arrays[f"{leg}/{index}/params/{name}"] = np.asarray(flat, dtype=np.float64)
            cards[leg].append(epoch.get("scorecards"))
            scalars[leg]["epochs"].append(
                {k: v for k, v in epoch.items() if k not in ("params", "scorecards")}
            )
    np.savez(directory / "epochs.npz", **arrays)
    (directory / "scorecards.json").write_text(canonical(cards) + "\n")
    (directory / "scalars.json").write_text(json.dumps(scalars, indent=2, sort_keys=True) + "\n")
    content = scalars["trainer"].get("checkpoint_content_hash") or ""
    (directory / "checkpoint.sha256").write_text(content + "\n")


def load_fixtures(directory: Path = FIXTURE_DIR) -> dict[str, Any]:
    import numpy as np

    run = json.loads((directory / "run.json").read_text())
    with np.load(directory / "epochs.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    return {
        "run": run,
        "arrays": arrays,
        "scorecards": json.loads((directory / "scorecards.json").read_text()),
        "scalars": json.loads((directory / "scalars.json").read_text()),
        "checkpoint": (directory / "checkpoint.sha256").read_text().strip(),
    }


def diff(payload: dict[str, Any], fixtures: dict[str, Any]) -> list[str]:
    """Every difference between a fresh run and the fixtures, bitwise; empty means identical."""
    import numpy as np

    problems: list[str] = []
    for leg in ("orchestrator", "trainer"):
        data = payload[leg]
        want = fixtures["scalars"][leg]
        if data["rng_after"] != want["rng_after"]:
            problems.append(f"{leg}: torch RNG state after the run differs")
        for key in CHECKPOINT_KEYS:
            if key in want and data.get(key) != want[key]:
                problems.append(f"{leg}: {key} {data.get(key)!r} != {want[key]!r}")
        if len(data["epochs"]) != len(want["epochs"]):
            problems.append(f"{leg}: epoch count differs")
            continue
        for index, epoch in enumerate(data["epochs"]):
            for name, flat in epoch["params"].items():
                recorded = fixtures["arrays"][f"{leg}/{index}/params/{name}"]
                if not np.array_equal(np.asarray(flat, dtype=np.float64), recorded):
                    problems.append(f"{leg}: epoch {index} parameters of {name} differ")
            for key, value in want["epochs"][index].items():
                if epoch.get(key) != value:
                    problems.append(f"{leg}: epoch {index} {key}: {epoch.get(key)!r} != {value!r}")
            if canonical(epoch.get("scorecards")) != canonical(fixtures["scorecards"][leg][index]):
                problems.append(f"{leg}: epoch {index} scorecards differ")
    return problems


def main(argv: list[str]) -> int:
    payload = record_all()
    if "--record" in argv:
        write_fixtures(payload)
        print(f"recorded fixtures in {FIXTURE_DIR}")
        return 0
    problems = diff(payload, load_fixtures())
    for line in problems:
        print(line)
    print("default identity: OK" if not problems else f"default identity: {len(problems)} difference(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
