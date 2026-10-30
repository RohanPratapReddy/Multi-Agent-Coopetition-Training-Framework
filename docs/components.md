# Customising components

Every pluggable part of training is a *component* of a *kind*, registered under a name and
chosen by a spec. This page is the operator's map: what to type, where each piece lives, and
which document explains it. The design is `docs/plans.new/17-component-customization.md`; the
deviations the implementation made from it are in `docs/plans.new/17-implementation-notes.md`.

## The kinds

`scorer` (a scoring stage), `reward_term`, `rl_function.<family>` (`advantage`, `loss`, `rollup`,
`synthesis`, and the rest), `optimizer`, `scheduler`, `weight_update`, `aggregator`,
`peer_strategy`, `trainer_policy` and `peer_factory`. `mactf components list` prints every
visible component with its origin, mutability and fingerprint.

## Turning a component on

1. Write the module. `mactf components scaffold <kind> <name>` writes a stub that already imports,
   registers and resolves; only the method that does the work raises `NotImplementedError`.
   Eight finished, runnable examples ship in `mactf/contrib/components/examples/`.
2. List the module in `conf/components.yaml` under `COMPONENTS.MODULES` (registration is never a
   side effect of being importable). For one run, `mactf run --component-module acme.components`.
3. Select it with a spec in the file that owns the block: `TRAINER.WEIGHT_UPDATE`, `RL.LOSS`,
   `OPTIMIZER`, `SCORING.STAGES.reward`. A spec is a name, `{type, params}`, or `{import, params}`.
   The same spec can come from the environment (`MACTF_TRAINER__WEIGHT_UPDATE__TYPE=uniform`) or
   from `--set TRAINER.WEIGHT_UPDATE.params.trim=0.3`.
4. Check it before a run: `mactf components validate --kind weight_update --name ema_softmax
   --param alpha=0.3 --probe`, and `mactf check` (the `mactf_components` app).

## Changing a live run

Component changes and tree mutations are queued and applied at the next boundary, never
mid-episode: `mactf runs components show <run>`, `mactf runs components set <run> <slot> --spec ...`,
`mactf peers mutate plan.yaml [--run <run> --apply]`, and over HTTP `GET/PATCH
/api/v1/runs/{run_id}/components` and `POST /api/v1/runs/{run_id}/mutations`
(see [api.md](api.md)). Only a process that drives the run can change it; from another process
these verbs answer exit code 4.

## Where to look

| Question | Section of the design |
|---|---|
| What is a component, an origin, a mutability? | 3, 4 |
| Spec shape and resolution order | 5 |
| Scoring stages and the compliance guard | 6 |
| Policies, RL functions, optimizers | 7, 8, 9 |
| Weight updates and peer strategies | 10, 11 |
| Adding, retiring and purging peers | 12 |
| The two engines | 13 |
| Checkpoints and resume | 14 |
| Configuration keys | 15 (`mactf/components/known.py`, `mactf/conf.py`) |
| Commands and routes | 16 ([cli.md](cli.md), [api.md](api.md)) |
| Error and warning codes | 17 (`mactf/components/codes.py`) |
| Worked examples | 18 (`mactf/contrib/components/examples/`) |
