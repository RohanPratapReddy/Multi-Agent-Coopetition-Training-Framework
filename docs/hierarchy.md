# Hierarchy

A swarm can be a tree of peers: roots delegate to sub-agents, which delegate to mini-agents. A flat
run is a depth-1 tree, so one code path serves both.

## Describing a tree

```python
from mactf.coopetition.tree import HierarchySpec

spec = HierarchySpec.from_spec({
    "roots": 2,
    "levels": [{"label": "subagent", "count": 2}, {"label": "miniagent", "count": 2}],
})
```

`PeerTree` holds `PeerNode`s addressed by path. Pass the spec to `MACTFTrainer(hierarchy=spec)`
(see [training.md](training.md)); the trainer builds a module per node with your `model_factory`.
`DelegationPolicy` (`mactf.coopetition.delegation`) decides which child handles an input and
`SynthesisFunction` (`mactf.coopetition.synthesis`) combines children's outputs for the parent.

## Scoring: non-offsettable propagation

`HierarchicalScorer` scores the tree leaves-first, each node through its own `ScoringPipeline`.
The rule it exists to guarantee:

> If any node in a branch violated a non-offsettable rule, every ancestor of that node is void for
> the episode and takes the floor.

Averaging a voided child with healthy siblings would produce a plausible number and launder a
T0/T1/T2 breach, so `branch_void` unions each node's void set into every ancestor.

Parents roll children up with a `RollupStrategy`, named in `ROLLUP_STRATEGIES`: `kappa_weighted`
(alias `kappa`), `mean`, `sum`, `minimum`, `worst_case`, `explicit`. `build_rollup(spec)` accepts a
name or a mapping.

## Inference

`run_inference(...)` (also `trainer.infer(inputs, node_path=None)` for the whole tree, or a path for
one node) returns an `InferenceResult` with per-node results, cost per level, and the batching
decision it made. A node `budget` caps total calls; exceeding it is reported, not silently
truncated. Seeds are derived per node path (`node_seed`), so inference is reproducible.

Inputs are keyed by the module's forward argument names (for example `{"state": tensor}`).
