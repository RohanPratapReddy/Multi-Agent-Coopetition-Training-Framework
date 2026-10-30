# Training

`mactf.training.MACTFTrainer` is the wrapper layer for training your own model as a swarm of
peers. It is in-process Python; the HTTP API's orchestrator runs do not execute a `trainer`
spec (a `trainer` block on `POST /runs` is validated and stored only).

## The shape of a run

```python
from mactf.coopetition.tree import HierarchySpec
from mactf.orchestration.checkpoint_v2 import MACTFCheckpoint
from mactf.training import MACTFTrainer
from mactf.training.logging import MACTFLogger
from mactf.training.watchdog import MACTFWatchdog

checkpoints = MACTFCheckpoint("runs/ckpt", run_id="demo")
watchdog = MACTFWatchdog(checkpoint=checkpoints, logger=MACTFLogger(run_id="demo", sinks=[]))

trainer = MACTFTrainer(
    model_factory=build_model,                  # called per node: factory(path) or factory()
    hierarchy=HierarchySpec.from_spec({"roots": 2, "levels": [{"label": "sub", "count": 2}]}),
    peer_strategy="hierarchical",
    weight_update="kappa_softmax",
    watchdog=watchdog,
)
trainer.build()
result = trainer.fit(epochs=2)
```

Pass exactly one way of describing peers: `model`, `models`, `model_factory` (with `num_peers` or
`peer_names`), or `hierarchy` (a factory may accompany a hierarchy). `result` is a `TrainResult`
with `history`, final weights, and `episodes` (the scorecards). `trainer.infer({...}, node_path=...)`
runs one node or the whole tree; `trainer.context` is the live health context.

Each epoch the trainer collects an episode per peer, scores it, derives points, aggregates with
the chosen `weight_update` and applies the optimizer step. Points are always derived and attached,
but only `COOPETITION.SIGNAL_SOURCE` decides whether they feed the signal; see
[calibration.md](calibration.md).

## `MACTFOptimizer`

Wraps a backend and adds gradient clipping, schedulers and per-group routing
(`ParamGroupRouter`). Backends: `TorchOptimizer` (any `torch.optim` class), `ScipyOptimizer`
(needs `scipy`; a missing install raises at construction, not at the first step), and
`CallableOptimizer` (your function). Pass `optimizer="adamw"`, an instance, or a
`MACTFOptimizer`.

## `FinetuneConfig`

A frozen, validated description of parameter-efficient fine-tuning: `peft_type` (`lora`, `adalora`,
`prefix`, `qlora` or `None`), `r`, `alpha`, `dropout`, `target_modules` (`"auto"` resolves by layer
type), `quantization` (`4bit`/`8bit`), `compute_dtype`, `freeze_base`. It needs the optional `peft`
(and `bitsandbytes` for quantisation) packages; `pip install mactf[peft]`. Importing them is
checked when you apply it, with an error naming the missing package.

## Recipes

**Custom reward scorer.** Subclass `mactf.scoring.Scorer`, return a `ScorerOutput`, and pass `scorers=[...]` to the trainer
(mutually exclusive with `hierarchical_scorer`).

**Custom loss.** Pass `loss=` a `LossFunction` from `mactf.rl.functions.loss`, a mapping spec for one, or a plain
callable (wrapped in `CallableLoss`; see its docstring for the call signature).

**LoRA.** `MACTFTrainer(model=base, finetuning=FinetuneConfig(peft_type="lora", r=8, alpha=16))`.
The adapters are saved in checkpoints (`adapter_state`) and `merge_adapters(model)` folds them back.

**SciPy.** `optimizer=ScipyOptimizer(method="L-BFGS-B")` for small, deterministic problems.

Checkpointing, resume and the watchdog are in [recovery.md](recovery.md); logging sinks in
[observability.md](observability.md).
