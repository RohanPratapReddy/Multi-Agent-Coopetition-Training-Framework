# Document 17: implementation notes

Deviations from `17-component-customization.md`, recorded per phase: what differs from the
document, and why. The real tree wins for existing behaviour; where the document's code had a
defect it is fixed here and listed.

## P0 - Foundation (sections 3, 4, 5 and the `COMPONENTS` block of 15)

### Fixtures first (rule 2 of 20.1)

* `tests/components/default_identity.py` is the recorder *and* the replay. `python -m
  tests.components.default_identity --record` writes the fixtures; without `--record` it replays
  and diffs. The fixtures were recorded on the unmodified tree before any production file changed
  (verified by re-running the replay after the change: bitwise identical).
* The scenario is run through **both** engines that exist today (`Orchestrator` and
  `MACTFTrainer`); `TrainerEngine` does not exist yet, so section 19.4's "both engines" is
  satisfied with the two engines that do. Layout is `run.json`, `epochs.npz`, `scorecards.json`,
  `checkpoint.sha256` as in 19.4, plus `scalars.json` (the per-epoch scalars and the RNG digest).
* **The checkpoint hash is not the file's `content_hash`.** A v2 checkpoint's `identity` section
  carries the run id, timestamps, host name, command line and library versions, so its
  `content_hash` differs between two runs of the same scenario. The recorded hash is taken over the
  fifteen other section digests plus the stable identity fields (`episode`, `global_step`,
  `schema_version`, `device`, `dtype`, `amp`, `mode`, `score`), and the identity key set is recorded
  so P4's "no `components` key" assertion has something to compare. The recorder is the single
  definition (`checkpoint_facts`).
* `test_default_identity` skips (with a message) when the installed torch major.minor differs from
  `run.json`, as 19.4 condition 2 says. Condition 4 ("zero guard calls") is checked with what
  exists: a default run makes **zero** calls to `ComponentRegistry.build` and `.register`. P1 and
  later add the guards to the spy list.

### Model

* `KindSpec` gains two fields: `legacy_contract` and `builtins_module`.
  * `legacy_contract`: the RL families validate a *legacy* write (`register_function(family, name,
    Cls)`, a direct write to a family table) with the check the old `register_function` applied (the
    value is a class). The document's strict contract (the entry method exists) would make
    `tests/rl/test_function_registry.py::test_register_function_can_replace_explicitly` fail, which
    registers a class with no `rollup()`. The strict contract still applies to the decorator form,
    to `register()` and to a built instance (`rl_function_instance_contract`).
  * `builtins_module`: the module that defines a kind's legacy table. `ComponentRegistry` imports it
    before the first write, so a plugin loaded early cannot take a built-in's name and make the
    framework's own import collide with it. `attach_mirror` marks the registry ready, which stops the
    recursion while that module is registering. Reads that must stay torch-free
    (`known.registry_names`) never trigger the import.
* `ComponentRegistry.register_builtin` / `restore_builtin` / `is_builtin_factory` are new. Built-ins
  are registered by `LegacyTable` through `register_builtin`, which bypasses locks, freezing and the
  per-kind cap (a deployment that locks `scorer:*` must still be able to import `mactf.scoring`
  afterwards) but still enforces the kind's contract. A legacy write that puts the framework's
  *original* class back over a replacement restores the `builtin` origin and clears
  `overrides_builtin`; every existing test restores what it replaces that way
  (`FUNCTION_TYPES["rollup"]["mean"] = original`), and without this the registry would keep a
  `runtime` record for a built-in forever.
* `STRICT_LEGACY` refuses a legacy write over a built-in with `ComponentLocked` (15.8.1 says
  `ComponentLocked`; the document's code raised `ComponentCollision`). `ALLOW_BUILTIN_OVERRIDE`
  still lets it through. A legacy `del`/`pop` of a built-in logs `W005` as a legacy replacement does.
* `E011` (a spec sets a key the framework injects) is raised as `ComponentSpecError("E011: ...")`,
  as the 17.2 table says, not as `ComponentParamError` as the code in 3.4 did. `ComponentResolver.
  validate` gained an `inject=` argument for the same check and so that an injected required key is
  not reported as missing.
* `ParamSpec`: an `int` is accepted where `float` is declared (YAML writes `lr: 1`).
* `reflect`: a C-level `__init__` with no signature is treated as opaque (accepts anything) rather
  than raising.
* `LegacyTable` takes an optional `hub=` (default `HUB`) so tests can build a view over a private
  hub. `Hub.unfreeze_all()` is new (the loader needs it).
* `ComponentMissing` appends `; did you mean 'x'?` (difflib) when a registered name is close. 19.3.3
  row 20 asks for "the nearest registered name"; the 17.1 listing only had the accepted set.

### Contracts

* `optimizer_contract` and `optimizer_instance_contract` accept `BackendBase` (section 9.1, phase P2)
  and fall back to `mactf.training.optimizer._Backend`, the class that has that role today, so a
  plugin written against the final name registers now. **P2 must export `BackendBase`** (an alias is
  enough); nothing else in `contracts.py` changes.
* `reward_term`, `trainer_policy` and `aggregator` contracts check structure only (the classes they
  will plug into - `RewardTerm`, `TrainerPolicy` - belong to P1/P2). `aggregator` imports
  `mactf.coopetition.aggregation.Aggregator`, which exists.
* `optimizer_builder` copies `values` before popping `lr` / `weight_decay` (the document's code
  mutated its argument).

### Resolver, slots, plugins

* `merge_layers(kind_id, layers) -> LayerResult` and `LAYERS` (the nine layers of 5.3) are new.
  Section 5.3 is prose, but `test_resolution_order` needs an executable form that "names which
  layer" decided. `LayerResult` records the deciding layer, the layer of each parameter, and the
  adopted instance of layer 9.
* `ComponentResolver._import` wraps any exception raised while importing the target module in
  `ComponentBuildError` (E008, cause chained); the document let it escape raw.
* `SlotTable.swap` wraps a non-component exception from `migrate` in `ComponentMigrationError`
  (E048) so the old component demonstrably keeps running.
* `load_components`: an invalid `PLUGINS` value is refused before the policy is touched; on any
  failure the previous policy is restored; registries are unfrozen at the start of a (re)load, since
  a freeze belongs to the configuration that asked for it; a module that raises is E008, an
  `ImportError` stays `ImproperlyConfigured`, and a `ComponentError` from a plugin propagates
  unwrapped. `policy_from_settings` folds the `API_ALLOW_IMPORT` alias so the hub never ignores a key
  a validator accepted. The lock is `RLock`, not `Lock`, because `ensure_components_loaded` calls
  `load_components` while holding it.

### Configuration (`mactf/conf.py`)

* Added: `COMPONENTS_DEFAULTS`, `DEFAULTS["COMPONENTS"]`, `_COMPONENTS_ALIASES`,
  `_canonical_components`, `_validate_components` (E013-E017), `"COMPONENTS"` in `_NEW_BLOCKS`,
  `ensure_components_loaded` in `configure`, and the two `TRAINER` name checks rewritten over
  `components.known.component_ref_problem`; `_PEER_STRATEGY_NAMES` / `_WEIGHT_STRATEGY_NAMES` are
  views of `known.BUILTIN_NAMES`.
* **Defect in 15.4.2:** it builds the lock-pattern kinds from `set(BUILTIN_NAMES) | {"slot"}`, which
  would reject `optimizer:AdamW`, `rl_function.loss:ppo` and every other kind `known.py` does not
  list names for. `known.LOCKABLE_KINDS` (every kind id) is used instead. A bare `rl_function:x` is
  refused (it matches no registry key, so it would lock nothing).
* `known.BUILTIN_NAMES` lists only the three kinds whose built-ins exist today (`scorer`,
  `peer_strategy`, `weight_update`). The document also lists `reward_term`, `aggregator` and
  `trainer_policy` names; each later phase adds its row when it adds the built-ins.
  `LOWERCASE_KINDS` is `{peer_strategy, weight_update}`, matching what the registry enforces (the
  document's wider set would accept `Penalty` at load and fail at build).
* **Not done in P0, because 20.2 gives them to P6** ("the remainder of section 15"): the `conf/`
  layout entries and templates (`components.yaml`, `rl.yaml`) and the tests that count them, the
  nested `MACTF_A__B__C` environment rule (E018, W007), the `--set` spec promotion, and the
  `COMPLIANCE` / `RL` blocks (P1 / P2). Until P6, `COMPONENTS` is set in `settings.py`, which
  `tests/test_conf_blocks.py` covers. `mactf/settings_default.py` needs no change: its only P0 row in
  20.3 is the `INSTALLED_APPS` anchor for `mactf.contrib.components`, the system-check app, which is
  P6; the block's values come from `conf.DEFAULTS`, which that module defers to.

### Legacy tables

* `register_scorer` and `register_function` are re-exports of `components.decorators` (the old
  definitions were deleted); `mactf.rl.functions.registry` and `mactf.scoring.registry` keep every
  name in `__all__`. `registry.py` of the RL package was *not* rewritten around the resolver (that is
  8.2/P2); only the dict literal and `register_function` changed. The `kappa_weighted` rollup alias
  of 8.6 is therefore not added yet.
* The behaviours the document lists as changed: a legacy write over a built-in now logs `W001` and
  `W005`; an invalid scorer now fails at write time with `ComponentContractError` (a
  `ValidationError`; the existing messages `already registered`, `not a Scorer subclass`,
  `invalid ... type name`, `is not a class` are preserved verbatim).

### Existing tests touched

* `tests/test_conf_equivalence.py::test_the_snapshot_covers_every_pre_existing_block`: `COMPONENTS`
  joins the set of blocks added after the snapshot (the document says `DEFAULTS` gains the block).
* `tests/test_conf_blocks.py`: `COMPONENTS` joins `NEW_BLOCKS`; a few tests added.
* `tests/rl/test_function_registry.py`, `tests/cli/test_import_weight.py`: extended, nothing
  removed or weakened.

## P1 - Scoring and guards (section 6)

The 20.3 anchors for P1 were re-checked against the tree before starting: all fourteen rows
matched (`compliance.py` 35/108/129/198, `base.py` 190, `guards.py` 474/700, `hierarchy.py` 97/145,
`penalty.py` 52/70, `pipeline.py` 73/329/499). `guards.py` (the points engine) needed no edit: it
reads the ledger, not stage outputs (6.1 fact 3).

### Files

New: `mactf/components/guards/compliance.py`, `mactf/scoring/{reward_terms,null_stages,
chain_validation}.py`. Changed: `mactf/compliance.py` (`DEFAULT_MIN_FLOOR_LAMBDA`,
`effective_non_offsettable`), `scoring/{base,penalty,hierarchy,pipeline,registry}.py`,
`components/{contracts,known}.py`, `conf.py` (`COMPLIANCE_DEFAULTS`, `DEFAULTS["COMPLIANCE"]`,
`_validate_compliance`, `_NEW_BLOCKS`). `mactf/settings_default.py` needs no change (the block comes
from `conf.DEFAULTS`).

### Deviations from the document's code

* **Built-ins are registered with `register_builtin`, not `@register_scorer`.** The decorator gives a
  new class the default origin (`project`), so `composed_reward`, `zero_reward`, `no_warnings`,
  `severity_scaled` and `capped` would not be built-ins (no `W001` on replacement, not in
  `known.BUILTIN_NAMES`). `scoring/registry.py` imports `null_stages` and `reward_terms` after
  `SCORER_TYPES`; `_BUILTINS_MODULE["reward_term"] = "mactf.scoring.reward_terms"` makes the
  `reward_term` registry load them. `known.BUILTIN_NAMES` gains the three scorer names and a
  `reward_term` row.
* **Guard: in-place edits and a private context.** `ScoreContext` is frozen but its mappings are
  not, and `_base_context` hands every stage a *shallow* copy of `score_config`, so a stage editing
  `ctx.config["PENALTY_LAMBDAS"]` in place changed the pipeline's own configuration and left no
  trace in `prepare`'s return value. `_walk` now gives a chain with any non-built-in stage a deep
  copy of `config`/`severity_weights`, and the guard keeps a `Pristine` deep copy of the protected
  fields (`ComplianceGuard.pristine`, `check_pristine`); every custom stage is compared against it
  after `prepare` and `score`, and the reference penalty is computed from it. `enforce` takes an
  optional `pristine=`. A default chain does none of this (no copies, no extra comparisons).
* **C006 and C007 are never substituted.** The document's `_inspect_*` put a REWARD/PENALTY C006
  (mutations / `void_rules` returned) into the per-kind lists that `fallback` substitutes, while
  6.4.2 says C006 and C007 always raise. `enforce` raises when any violation is in `HARD_CODES`.
* **A tampered ledger is put back.** `check_after_stage` restores the original events when it
  reports C007, and `_walk` runs the same check when a stage raises, so a stage that edits the
  ledger and then fails cannot leave the edit behind (the original error is chained).
* **Hardened against malformed outputs.** A non-mapping `breakdown`, a `void_rules` that is not
  iterable, numeric strings and booleans are reported as `C001`/`C004`, never as a raw
  `AttributeError`/`TypeError` from inside the guard.
* **`_walk` resets `_last_guard_report`** at the start, so a card never carries the previous
  episode's report.
* **`ScoringPipeline._validate_chain` rejects a nested `ScoringPipeline`** (kind `PIPELINE`).
  `validate_stage_chain` reports `chain.pipeline_nested` (doc 6.7) but `_validate_chain` accepted
  such a chain, so `test_chain_validation_agrees` could not hold. The message is new; nothing
  nests pipelines.
* **`validate_stage_chain` order.** It reports in the order `_validate_chain` meets problems
  (non-scorers first, then per stage in chain order, collisions, missing kinds, warning order) and
  only then the stage-level diagnostics, unknown `requires` and the lambda floor. The document's
  order (all stage-level diagnostics first, duplicates after every stage) made "the first error
  corresponds to the exception" false for chains with several problems.
* **`sync_from_slots` checks the slot's kind.** A slot `scoring.reward` that came to hold a
  PENALTY stage (a swap without `migrate_stage_state`) would otherwise have replaced the *penalty*
  stage; it now raises `ImproperlyConfigured` and the chain is unchanged.
* **A `ValidationError` raised by a reward term's constructor is not wrapped in `E008`.** 18.2
  shows `E008: could not build reward_term ...: ValidationError: ...`, but P0's
  `ComponentRegistry.build` (and its test `test_a_validation_error_from_the_component_itself_
  propagates_unchanged`) deliberately lets a `ValidationError` from the component reach the caller
  unchanged; only `TypeError`/`ValueError` become `E008`. P1 keeps P0's rule, so a bad term
  parameter surfaces as the term's own `ValidationError`.
* **`PenaltyScorer.validate` does not add `penalties.floor_lambda_below_minimum` for a negative
  lambda**: `penalties.negative_lambda` already names it, and
  `test_validate_rejects_a_negative_lambda` (unchanged) expects exactly one diagnostic. A
  non-numeric `MIN_FLOOR_LAMBDA`/`DEFAULT_LAMBDA` is `penalties.non_numeric_floor`.
* **`_validate_compliance`** also rejects unknown `COMPLIANCE` keys, a non-mapping block, a
  non-boolean `TRUST_BUILTIN`, and a non-numeric `PENALTY_LAMBDAS` value for a non-offsettable rule
  (an `ImproperlyConfigured` instead of a raw `ValueError`). It is called from `validate_settings`
  right after the existing `NON_OFFSETTABLE` check, before `_validate_points`, as 15.2 says.
* `build_scorer` logs `W003` when `GUARD_MODE` is `fallback`; the guard logs it again each time it
  substitutes. (`check_guard_modes`, the system check, is P6.)

### Existing tests changed (the document changes the behaviour)

* `tests/scoring/test_pipeline_equivalence.py::test_no_bypass_gate`: its `LyingReward` returns
  `1e6` of reward, which is `C008` now. The original assertions (void, score <= 0) are kept under
  `GUARD_MODE: fallback`; the test also asserts the `C008` record and that `enforce` raises.
* `tests/scoring/test_pipeline_validation.py::test_registry_lists_the_three_per_node_stages_plus_hierarchy`
  and `tests/components/test_legacy.py::test_the_documented_builtin_names_are_unchanged`: the
  scorer table gains the three names 6.2/6.8 add.
* `tests/components/test_known.py`: `reward_term` is now a name-checked kind.
* `tests/hierarchy/test_hierarchical_scoring.py::test_describe_reports_the_configuration`:
  `HierarchicalScorer(non_offsettable=("P1",)).describe()` now reports the effective superset
  (H1); it used to echo the narrowed set.
* `tests/test_conf_blocks.py`, `tests/test_conf_equivalence.py`: `COMPLIANCE` joins the new blocks.
* `tests/cli/test_import_weight.py`: one more statement (the guard and the new scoring modules stay
  torch-free).

### Not done in P1 (belongs to a later phase by 20.2)

`conf/components.yaml` and its template for `COMPLIANCE` (P6), the `check_guard_modes` system check
and `W003` as a check (P6), the engine calls to `slots.apply_pending` + `pipeline.sync_from_slots`
at the episode boundary and the API/CLI `PATCH` (P5/P6). `build_scorer(settings, slots=..., trusted=...)`
and `ScoringPipeline.stage_inject/sync_from_slots` are the hooks they use. `SCORING.STAGES` has no
load-time shape check yet (15.4's `_spec_shape_problem`, P6); a bad spec fails at `build_scorer`.

### What the guard is, and is not

It catches a stage that is wrong or careless (every shape in 6.4.1 plus the in-place and
ledger-reach variants above). It is not a sandbox: arbitrary Python in a stage can still reach
other objects (for example the nested dicts a stage shares with its own `self.config`, or a
monkey-patched built-in). `TRUST_BUILTIN` trusts the three built-in *classes* exactly; a subclass
is always inspected.

### Tests

`tests/components/`: `test_scoring_identity.py` (`test_default_pipeline_is_byte_identical`, 200
random episodes against the pre-split oracle with `torch.equal`, plus "no guard call on a default
run"), `test_compliance_guard.py` (about 130 adversarial cases, one per C-code and per misbehaviour,
in both modes), `test_guard_properties.py` (Hypothesis: `test_guard_property_compliance_floor`,
`test_guard_property_void_stays_void`, no false positives, floor lambda), `test_chain_validation.py`
(`test_chain_validation_agrees` over every 1-3 stage chain from a 15-stage catalogue, and Hypothesis
over longer ones), `test_scoring_stages.py`, `test_reward_terms.py` (`test_composed_matches_builtin`),
`test_scoring_hardening.py`; helpers in `scoring_helpers.py`.

## P2 - RL functions, policies, optimizers and schedulers (sections 7, 8, 9)

### Files

New: `mactf/training/{policy,freeze,schedulers,optimizer_builtins,optimizer_plan,optimizer_migrations}.py`,
`mactf/components/{policy_resolver,loss_compliance,function_probes,function_book,migrations}.py`,
`mactf/components/guards/step.py` (and `guards/__init__.py` if P1 had not made it),
`mactf/distill/policy.py`. Changed: `training/{trainer,optimizer}.py`, `rl/functions/{registry,loss}.py`,
`coopetition/synthesis.py`, `components/{contracts,known,slots,registry,resolver}.py`, `conf.py`,
`exceptions.py`, `templates/conf/optimizer.yaml.tpl`, `distill/__init__.py`.

### What the next phases must know

* **`MACTFTrainer._trainer_policy` keeps its pre-17 meaning** (what the caller configured: the
  `trainer=` object, the distillation step, or `None` for the default step). The document gave the
  name to "the object `train_step` calls"; four existing tests pin the old meaning
  (`trainer._trainer_policy is None` for a default run, `isinstance(..., Policy)` for a wrapped
  in-code policy). The object that runs is `trainer._policy` (the default policy, or a
  `GuardedPolicy`); `_policy_origin` / `_policy_guard` as in the document. Everything that read
  `_trainer_policy` to *call* it (`train_step`, `setup`, `on_epoch_end`, metadata, state) reads
  `_policy`. `PolicyChoice` gained `configured` for this.
* The default policy and the distillation-only policy are put in the `TRAINER.POLICY` slot with the
  new `SlotTable.adopt_registered` (a registered component, built outside the registry), not with
  `slots.install(..., "default")`: the latter calls `ComponentRegistry.build`, and the P0 gate
  `test_a_default_run_uses_no_component_machinery` (19.4 condition 4) counts those calls.
* `trainer_policy` has **no** `builtins_module`: importing `mactf.training.policy` needs torch, and
  `test_import_weight` registers a trainer policy without it. `default` and `co_distill` are
  registered when that module is imported; `resolve_policy` imports it. Anything that lists
  built-in policies without a trainer (P6's `components list`) must import it first. The P0
  `component_overlay` fixture compares registry snapshots, so the P2 tests import it at module
  level (`tests/components/p2_support.py`).
* `optimizer` and `scheduler` do have a `builtins_module` (`mactf.training.optimizer`,
  `.schedulers`): they register through `register_builtin`, so locks and freezing do not apply to
  the framework's own import.
* **`mactf/training/freeze.py` (`FreezeSet`, section 11.4) is created here** because
  `optimizer:null` uses it (section 9.7). Only the set exists and is bound to the tree
  (`trainer.freeze`, `TRAINER.FREEZE_MODE` is read). The two loop hunks of 11.4 (`_train_epoch`
  skipping `skips_training`, `_write` skipping `skips_write`) are P3's and were not applied: under
  the `legacy` mode they only matter for `mutation:retired`, and a `null` node's step is a no-op.
* **Section 8.6 is not done** (the rollup table unification in `scoring/hierarchy.py`, the
  `RollupGuard`, `mactf/components/guards/rollup.py`): `hierarchy.py` is P1's file and the guard is
  P3's. What exists: the `kappa_weighted` alias in the rollup table (8.2), `FunctionBook` serving
  `RL.ROLLUP` through `guards.rollup.guard_rollup` (an `ImproperlyConfigured` naming the missing
  module if it is absent), `MACTFTrainer._rollup_choice` and `_rollup_arguments`: the trainer passes
  `rollup_source=` to `HierarchicalScorer` when its constructor accepts one and **refuses** a
  configured `RL.ROLLUP` when it does not, so the setting can never be a silent no-op. P3 adds the
  `rollup_source` argument and property (8.6.1) and nothing in the trainer changes.
* `FunctionBook.validate` is called with `in_code` also naming `"loss"` when `loss=` was given
  (E040 for `loss=` plus `RL.LOSS`); the document listed only synthesis and rollup.
* Exported names: `mactf.training.policy.{DefaultTrainPolicy, PolicyHost, TrainerPolicy, VOID_ADVANTAGE_FLOOR}`;
  `mactf.components.guards.step.{StepGuard, StepGuardConfig, GuardedPolicy, VoidProbe, register_term_check, TERM_CHECKS, CODES}`;
  `mactf.components.policy_resolver.{resolve_policy, PolicyChoice, CallablePolicy}`;
  `mactf.components.function_book.{FunctionBook, BookSettings, specificity}`;
  `mactf.components.function_probes.{verify_function, check_function, ProbeContext, Finding}`;
  `mactf.components.migrations.*`; `mactf.training.optimizer.{BackendBase, NullOptimizer, RouterOptimizer}`
  (`_Backend` stays an alias); `mactf.training.optimizer_plan.{OptimizerPlan, OptimizerEntry, compile_entry, verify_entry}`;
  trainer methods `resolve_node`, `backward`, `loss_for`, `synthesis_for`, `swap_optimizer`,
  `_ensure_optimizer`, attributes `functions`, `slots`, `freeze`, `optimizer_plan`, `distill`.

### Deviations, and defects of the document fixed

Policies and the `StepGuard`

* `StepGuard._invoke` passes `trainer=` to a plain callable policy (the document dropped it, which
  would break the legacy `(node, batch, trainer=...)` form once it is guarded); `resolve_policy`
  wraps such a callable in `CallablePolicy` so it satisfies the `trainer_policy` instance contract.
* For a live-loss policy the guard now holds an `after_step` entry in **every** judging mode, not
  only `enforce`/`quarantine` and only when it took no snapshot before: a `warn`-mode guard still
  reports `S002`/`S003` for the step the trainer takes, and a policy of unknown `self_stepping`
  (snapshotted up front) is judged after the trainer's step too.
* `S003` for a self-stepping policy is checked whether or not a snapshot exists
  (`SNAPSHOT=never` gives up the undo, not the detection, as 7.6.1 describes).
* `GuardedPolicy.__getattr__` refuses dunder names and its own attributes (infinite recursion on
  copy/pickle otherwise).
* Existing behaviour kept: `trainer=` together with `DISTILLATION.ENABLED` is still refused unless
  the policy declares `composes_distillation` (the new rule of 7.5, applied in the constructor
  check); the message names the new way out.
* Card metadata: `metadata["policy"]` is written for every policy with `card_metadata` and for an
  unsafe one; a guarded policy therefore adds `metadata["policy"]["step_guard"]`. A default run adds
  nothing.
* The checkpoint gains `policy`, `functions` and `optimizer_plan` keys under `identity["trainer"]`
  **only when they have content** (a guarded policy whose own state is empty writes no `policy`).

Functions

* `ComponentRegistry.build` / `ComponentResolver.resolve` take `check_instance=` (default `True`);
  the legacy `build_function` passes `False`. `tests/rl/test_function_registry.py` registers and
  builds a marker class with no entry method, which the old registry allowed; the strict instance
  contract still applies to the function book, to `register()` and to anything else built through
  the registry.
* `ComponentContractError` takes `code=` so a probe failure carries `E047` (the document says so but
  its class had no way to).
* `FunctionBook._adopted` / `_build_raw` used `component.kind_id`; the field is `kind`.
* `rl/functions/registry.py` is the document's file with those changes; `kappa_weighted` joins the
  rollup table, so the capability list has seven rollup entries (`tests/components/test_legacy.py`
  states them).
* `WeightedSumLoss` also mirrors a term's `advantage_mean` under the unprefixed key, because that is
  the one number the step guard's void probe reads and `ensure_compliance` wraps every custom loss
  in a `WeightedSumLoss`.
* `_resolve_loss` extends every non-`CompositeLoss` loss, including an in-code `loss=` (so a custom
  `loss=` callable now carries the compliance term; `TRAINER.LOSS_COMPLIANCE: off` restores the old
  behaviour). The coefficient is `TRAINER.COMPLIANCE_COEFFICIENT` when written, else
  `LossConfig.compliance_coef` (the document names both).
* The worked advantage test of 8.9.1 passes 2-D rewards and a 3-element `next_values` to
  `GAEAdvantage`, which the real estimator refuses (`next_values` is one bootstrap value); the test
  in `tests/components/test_clipped_advantage.py` uses the real contract.

Optimizers and schedulers

* The `OPTIMIZER` block is validated by `_validate_optimizer`; the `TYPE` row of
  `tests/test_conf_blocks.py::REFUSALS` changed (any registered name is a valid `TYPE` at load, as 9.4
  says), and `GRAD_ACCUMULATION` is checked without the document's `or 1` (it hid `0`).
* Unknown backends keep the old messages: `unknown optimizer backend 'x'; accepted: torch, scipy,
  callable, <registered>` and `unknown torch optimizer 'x'; torch.optim has no such class and ...`.
* `compile_entry`: a shipped `BETAS`/`EPS` default is dropped when the same key is under `PARAMS`
  (the document's `E057` would fire for `TYPE: Lion` + `PARAMS: {betas: ...}`, the 18.5 example), and
  is never passed to a registered non-torch backend; a scheduler is dry-built over a scratch
  optimizer, so `warmup_steps >= total_steps`, an unknown keyword or `drop_at: 0` is `E053` at build;
  an entry that resolves to `null` ignores an inherited `SCHEDULER` instead of failing with `E052`
  (otherwise the 9.9 example, `TYPE: "null"` under a scheduled default, would not load).
* `merge_blocks`: an override's `PARAMS` were overwritten by the generic loop before the merge line
  could use the default's; fixed.
* `MACTFOptimizer._from_registry` forwards `weight_decay` only to a class that accepts it.
* `OptimizerPlan.state_dict` stores only the entries a live swap changed (plus removals and the
  history), not every `block` entry; with the document's version an edited `conf/optimizer.yaml`
  would always lose to the checkpoint, contradicting the restore table of 9.8 ("OPTIMIZER block
  edited: W015 per node, carry or drop"). `load_state_dict` also rebuilds the swap history, and
  `identity["trainer"]["optimizer_plan"]` is written only once something was swapped. Facades record
  `entry_hash` (never checkpointed) so a restored plan that selects another *legacy-looking* entry
  (`AdamW` -> `SGD`, neither "customized") still replaces the facade.
* `compatible_state` ignores `initial_lr` in the group keys (a torch scheduler writes it, so an
  `Adam` with a scheduler could never be carried into an optimizer without one).
* `swap_optimizer` assigned `self.optimizer_warnings`; it is `_optimizer_warnings`.
* The scratch-optimizer probe (`verify_entry`) and the scheduler dry-build use momentum SGD so a
  `CyclicLR`/`OneCycleLR` is not refused for a missing momentum.
* `optimizer.yaml.tpl` gains the four plan keys (the template/DEFAULTS equality test needs them).

### Existing tests touched

`tests/components/test_known.py` (the `trainer_policy` row), `tests/components/test_legacy.py`
(`kappa_weighted`), `tests/test_conf_blocks.py` (the `OPTIMIZER.TYPE` row, see above),
`tests/test_conf_equivalence.py` (`RL` joins the blocks added after the snapshot). Nothing removed
or weakened.

### Tests added (all in `tests/components/`)

`test_step_guard.py` (every `S` code, the adversarial policies, a Hypothesis property),
`test_policies.py` (resolver, precedence, distillation composition, unsafe, the 7.10 policy, the
compliance extension of an in-code loss), `test_function_book.py` (R1-R3, R7-R12, E040-E045),
`test_function_probes.py` (F001-F012), `test_clipped_advantage.py`, `test_l1_anneal.py`,
`test_median_synthesis.py`, `test_rl_registry_and_conf.py`, `test_optimizer_plan.py` (O1-O10, E050-E057,
the 18.5 Lion example), `test_optimizer_migrations.py` (`test_optimizer_migration_*`, the restore
check, schedulers, resume). `p2_support.py` holds the worked examples as plain classes.

## P3 - Weight update, aggregation and peer strategies (sections 10, 11 and 8.6)

### Files

New: `mactf/coopetition/{weight_guard,weight_probe,strategy_probe}.py`,
`mactf/components/{weight_plan,kappa_plan,strategy_plan}.py`,
`mactf/components/guards/{rollup,strategy}.py`, `mactf/training/coordination.py`,
`mactf/testing/strategy.py`. Changed: `coopetition/{weights,aggregation,temperature,strategy}.py`,
`scoring/hierarchy.py`, `components/{function_book,contracts,known}.py`, `training/trainer.py`,
`orchestration/checkpoint_v2.py` (two small hunks, see below), `conf.py`,
`templates/conf/trainer.yaml.tpl`, `testing/__init__.py`. `training/freeze.py` and the two
`exceptions.py` classes already existed (P2 and P0) and needed no change. Tests: eleven files under
`tests/components/` (`test_rollup_guard`, `test_weight_guard`, `test_weight_plan`, `test_strategy`,
`test_coordination`, `test_strategy_plan`, `test_staggered_mix`, `test_conf_weights_strategies`, and the
helpers `p3_examples.py` and `p3_support.py`).

### What the next phases must know

* **Public names.** `mactf.coopetition.weights`: `WeightTraits`, `traits_of`, `GroupView`,
  `WeightStrategy`, `make_report`, `BUILTIN_WEIGHT_TYPES`; `mactf.coopetition.weight_guard`:
  `WeightGuard` (`covers`, `inspect`, `repair`, `check`, `run`), `Violation`, `GuardOutcome`;
  `mactf.coopetition.aggregation`: `Aggregator.options/clone_with/pristine`, `build_aggregator`,
  `verify_aggregator` (the `aggregator` kind has its first built-in, `default`, and a `known.py` row);
  `mactf.coopetition.temperature`: `AdaptiveKappa.snapshot/restore/resize`;
  `mactf.components.weight_plan`: `WeightPlan`, `pick_glob`, `group_label`, `ROOT_LABEL`;
  `mactf.components.kappa_plan.KappaPlan`; `mactf.coopetition.strategy`: `StrategyTraits`, `GroupRef`,
  `TreeView`, `tree_view`, `traits_of`, `coordinates_via_context`, `defines_nothing`, `legacy_dispatch`,
  `BUILTIN_STRATEGY_TYPES`; `mactf.components.strategy_plan.StrategyPlan` (`run`, `epoch_start`, `set`,
  `remove`, `request*`, `take_pending`, `drain_errors`, `adopt_framework_default`);
  `mactf.components.guards.strategy.StrategyGuard`; `mactf.components.guards.rollup.{RollupGuard,
  guard_rollup}`; `mactf.training.coordination.{CoordinationContext, CoordinationState,
  CoordinationError, AuditEvent}`; `mactf.testing.{synthetic_backend, sweep, written}`.
  Trainer: `weight_plan`, `weight_guard`, `kappa_plan`, `strategy_plan`, `write_count`,
  `swap_weight_update`, `remove_weight_override`, `request_weight_update`, `request_remove_weight_override`,
  `swap_peer_strategy`, `remove_peer_override`, `request_peer_strategy`, `request_remove_peer_override`,
  `legal_report`, `peer_digest`, `_group_views`, `_tree_view`.
* **P5 (tree mutation) hooks.** `AdaptiveKappa.resize` exists for a group that changes size;
  `_validate_weight_plan(initial=False)` / `_validate_strategy_plan(initial=False)` are the calls to make
  after a mutation (the `W022` / `W033` path); the `mutation:retired` reason and `FreezeSet.sync()` are in
  place. `trainer.controllers` is built per group by `KappaPlan`, so a new group needs
  `kappa_plan.build(label, ...)` (what `_group_aggregator` already does lazily).
* **P4 (persistence) hooks.** `identity["trainer"]` gains `weight_plan`, `strategy_plan` and `freeze`
  only when there is something to write; the manifest entries named in 10.9 / 11.10
  (`identity["components"]["weight_update" | "aggregator" | "peer_strategy"]`) are P4's.

### The default run makes no registry call (19.4 condition 4)

`WeightPlan.install_default` (a plain `kappa_softmax`) and `StrategyPlan.adopt_framework_default` (the
unwritten default) use `SlotTable.adopt_registered` with an instance built by a plain call, as P2 did for
the policy. `WeightPlan._build_group` builds `KappaSoftmaxAverage(aggregator)` directly for that slot, and
`_check_group` does not build a scratch instance to look for a `validate` the class does not define. The
document's `install(...)` would have called `ComponentRegistry.build` several times per default run, which
`test_a_default_run_uses_no_component_machinery` forbids. Both plans lower-case a name (`_prepared`,
`_lowered`) because `SlotTable.install` does not, and `build_*` always did.

### Deviations from the document's code, and defects fixed

Weights

* **The probe tolerates bounds under `MODE: repair`.** The document's `ema_softmax` (10.11) returns a
  plain softmax and fails the document's own probe (`two_void`: 0.646 against a cap of 0.6). Invariant A3
  says the framework owns `WEIGHT_FLOOR`/`WEIGHT_CAP`, and `repair` exists to project exactly that, so
  `verify_weight_update(..., mode=)` skips a `U005` that names a peer (a participant outside the bounds)
  when the guard repairs. A vector that does not sum to 1, a weight on a void peer and every hard
  violation remain findings in every mode, so "weights summing to 3" still fails at build; under
  `fallback` and `enforce` a bound is a finding too.
* **`KappaSoftmaxAverage.traits = WeightTraits(may_fallback=True)`.** `Aggregator.compute_weights` flags
  `fallback` by design (no eligible peer, or the disagreement check), and the trainer has always accepted
  that from the default strategy. A subclassed aggregator makes the guard cover the default strategy, and
  without the trait every probe of it failed with `U006`.
* **`Aggregator.pristine()` passes only the base class's options.** The document passed
  `self.options()`, which a subclass extends, so `pristine()` raised `TypeError` for exactly the subclasses
  it exists for.
* **Reference-only now runs (12.12 change 1).** `tests/training/test_weight_update.py::
  test_a_strategy_that_reports_fallback_is_replaced_and_recorded` pinned the old masking with
  `ReferenceOnlyAverage`; it now reaches the same "it reported fallback" branch with `uniform` and an
  all-void cohort, and a new test pins the reset to the reference.
* **`weight_warnings` after a restore.** The document extended the list and then called
  `_validate_*_plan`, which assigns it, so the `W021`/`W031` notes were lost. The notes are added after
  the validation (weights and strategies).
* `WeightPlan.from_settings` / `install_*` strip a legacy `aggregator` key from a mapping (the plan
  injects the group's own aggregator); `state_dict` writes the document's shape unchanged.

Strategies and freezing

* **`legacy_dispatch` lives in `coopetition/strategy.py`.** The document defines it in `strategy_plan.py`
  and imports it from `strategy` in `strategy_probe.py`; one definition, exported from `strategy`, imported
  by the plan.
* **The freeze set is merged into the eligibility `_weights_for` computes** (`freeze.eligible_for`), not
  only into what a strategy asks for. Otherwise a `hierarchical` default sweeping a group that holds a
  `strict`-frozen peer would give it weight (T3/T7), and the guard's own `legal_report` (which already
  assumes the merge) would call a built-in report illegal. `eligible_for` is `None` when nobody is frozen,
  so a default run's call is what it was.
* `StrategyPlan.epoch_start(trainer, epoch, view=None)` builds the per-group instances first, so a
  `per_group` strategy receives `on_epoch_start` before its first call. `StrategyPlan.errors` /
  `drain_errors()`: an `attach` that raises after the slot was written is reported to
  `trainer.failures` instead of escaping (the document says `attach` "cannot fail the swap" but gives it
  no handler).
* `MACTFTrainer.peer_digest` accepts the flat tensor `export_parameters()` returns (the document iterated
  a mapping). `record_signal_sharing` keeps the last 64 entries (11.9).
* **`E075` is also checked at the end of `build()`**, as 11.2 T9 says ("at build and at every change").
* The `peer_strategy` contract still rejects a class with its own abstract methods (the document drops
  `inspect.isabstract`); only `on_epoch_end` stopped being abstract. `tests/components/test_contracts.py::
  AbstractStrategy` now declares an abstract method of its own to keep its assertion.

Rollup (8.6)

* `build_rollup(spec, *, guard_mode="enforce")`, `ROLLUP_STRATEGIES` is the `LegacyTable` itself, and
  `HierarchicalScorer.rollup` is a property over `rollup_source`, all as 8.6.1. The P2 stopgaps are gone:
  `MACTFTrainer._rollup_arguments` (and its "build with no `rollup_source`" refusal) and the `ImportError`
  branch of `FunctionBook._guard`. `_build_scorer` passes `rollup_source=self._rollup_choice` directly.
* `RollupGuard` imports torch (through `rl.functions.rollup`), like the module that uses it; it is not in
  the torch-free import list.

Checkpoint (two hunks outside P3's file list, both specified in 10.9 / 11.10)

* `orchestration/checkpoint_v2.py`: restore step 8 drops a stored temperature that has no live group
  (`W021`, recorded as `kappa:<label>:dropped`) instead of loading it into the root controller (W-G10),
  and `_topology` compares `frozen` and `failed` as `active` (P-G7). The module had no logger; `log` was
  added. P4 should keep both hunks.

Configuration

* Eight `TRAINER_DEFAULTS` keys (`WEIGHT_UPDATE_OVERRIDES`, `WEIGHT_GUARD`, `WEIGHT_OPTIONS`, `AGGREGATOR`,
  `PEER_STRATEGY_OVERRIDES`, `PEER_STRATEGY_OPTIONS`, `STRATEGY_GUARD`, `FREEZE_MODE`) and the same keys in
  `trainer.yaml.tpl`. `_validate_weight_and_strategy_blocks` replaces the document's two functions
  (`_validate_weight_blocks`, `_validate_strategy_blocks`) and reuses `components.known.component_ref_problem`
  from P0 instead of the document's `_validate_component_ref`/`_names_may_be_extended` pair, which P0 had
  already made.
* **`COOPETITION.KAPPA_OVERRIDES` and `KAPPA_STRICT_GLOBS` are not in `COOPETITION`'s defaults**:
  `tests/test_conf_equivalence.py` pins that block key for key, and `KappaPlan` reads both with a default,
  so nothing is lost. The shape of `KAPPA_OVERRIDES` is checked at load (`E062`).
* `tests/cli/test_conf_mirrors.py` does not exist in the tree; the four mirrors are pinned by
  `tests/components/test_conf_weights_strategies.py` instead.

### Existing tests touched

`tests/training/test_weight_update.py` (above), `tests/components/test_contracts.py` (the abstract
strategy), `tests/components/test_known.py` (the `aggregator` row). Nothing removed or weakened.

### Tests

`test_weights_default_identity`, `test_strategy_default_identity` (ten built-in cases, the new
`coordinate` path against the legacy dispatch through a wrapper subclass, comparing the call trace, every
report and every parameter bit for bit), `test_strategy_swap_atomic` (nine fault steps, plus through the
trainer), `test_guard_property_weights_are_a_simplex`, `test_guard_property_strategy_cannot_write`,
`test_guard_property_rollup_union_survives_trimming`, `test_trimmed_mean_rollup`, `test_staggered_mix`; the
four swap rules each have a `unit` and an `integration` test in `test_strategy_plan.py`
(`test_rule_1_*` .. `test_rule_3_*`; rule 4 is `test_strategy_swap_atomic` and
`test_a_request_fails_at_request_time_for_what_needs_no_tree`). Every `U` and `Y` code, both guards in all
modes, the probes (`E064`, `E067`, `E074`, `E077`), `E060`-`E062`, `E065`, `E066`, `E070`-`E072`, `E075`,
`E076`, `W021`, `W022`, `W030`-`W036` have a test that triggers them.

## P4 - Persistence (section 14)

### Files

New: `mactf/components/{fingerprint,restore}.py`; tests `tests/components/{test_fingerprint,
test_restore_components,test_persistence_identity,test_persistence_kinds,test_persistence_resume}.py`
with the helper `persist_support.py`, `tests/db/test_db_migration.py`, `tests/cli/test_cmd_migrate_v4.py`.
Changed: `mactf/db.py` (migration 4, `record_components`, `components`, `_decode_run`),
`mactf/orchestration/checkpoint_v2.py`, `mactf/training/trainer.py`, `mactf/training/protocol.py`
(`TrainResult.component_notes`), `mactf/components/registry.py` (`is_builtin_class`),
`mactf/coopetition/aggregation.py` (`build_aggregator(record=)`), `mactf/scoring/reward_terms.py`
(`term_records`), `mactf/components/known.py` (`PERSISTENCE_CODES`). The two P3 hunks in
`checkpoint_v2.py` (W021 drop of an orphan temperature, `frozen`/`failed` read as `active`) are kept.

### What the next phases must know

* `MACTFTrainer._identity_extra()` exists now (section 12.7's method did not, P5 had not run). It
  returns `{"trainer": <extension state>, "components": <manifest>}`, each key only when it has
  content; **`identity["engine"]` is not written** (see deviations). P5 adds `mutation` to the
  `trainer` mapping and `engine` to the result; `CheckpointContext.identity_extra` and the merge in
  `_collect` (a framework key may not be replaced) are in place. `CheckpointContext.resolve()` fills
  `identity_extra` from `type(trainer)._identity_extra` when empty, so every writer (periodic,
  watchdog bridge, `state_dict`, a hand-built context) carries the manifest; `_observe`, `state_dict`
  and `_CheckpointBridge` also pass it explicitly as 14.4 shows.
* `engine.py` does not exist, so the 14.6.4 `record_components(source="engine")` call at `start_run`
  is **P5's**: add `trainer._component_manifest()` after `start_run`. The checkpoint writer already
  re-records on every save (`_record_components`; no registry call at all for a run that never had a
  manifest).
* `MACTFTrainer._extension_state()` is the single definition of the doc-17 keys of `identity["trainer"]`
  (`policy`, `functions`, `optimizer_plan`, `weight_plan`, `strategy_plan`, `freeze`); `_trainer_state()`
  adds the pre-17 keys to it. A periodic checkpoint carries only the extension state (so a default run
  writes nothing new), and `state_dict()` carries everything.
* **Restore is one path.** `MACTFCheckpoint.restore` calls `type(trainer)._prepare_restore(stored,
  strict=)` before the topology check (component plan, `E093`-`E096`, signal-source check, optimizer
  and weight plans) and `_finish_restore(stored)` after the logger/watchdog step (freeze set, strategy
  plan, callbacks, distillation, policy, functions). `MACTFTrainer.load_state_dict` is now only
  `restore` + `on_resume`, so `fit(resume=True)` (which never applied the plans before) and
  `load_state_dict` restore the same things. The hooks are looked up on the *class*, so a mock trainer
  is unaffected. `ResumeReport.component_notes` and `TrainResult.component_notes` carry the notes.
* `mactf.components.fingerprint.SlotView(slots, extras)` puts the components the plans hold outside the
  slot table behind `paths()`/`get()`: `optimizer_slots(plan)` (`OPTIMIZER`, `OPTIMIZER.<glob>`, and
  `<slot>:scheduler`), `aggregator_slots(resolved)` (`TRAINER.AGGREGATOR`), `reward_term_slots(stages)`
  (`scoring.reward.terms.<label>`). The trainer builds it in `_slot_view()`; the CLI (P6) should use
  `trainer._slot_view()` / `component_manifest(view)` rather than the bare `SlotTable`. In-code
  `scorers=[...]` stages that are not the framework's classes appear as `scoring.<kind>`.
* Manifest slot paths by kind: `scorer` `scoring.<warning|reward|penalty>`; `reward_term`
  `scoring.reward.terms.<label>`; `rl_function.<family>` `RL.<family>` / override paths of the
  `FunctionBook`; `trainer_policy` `TRAINER.POLICY`; `weight_update` `TRAINER.WEIGHT_UPDATE[.<glob>]`;
  `peer_strategy` `TRAINER.PEER_STRATEGY[.<glob>]` (the plans' paths); `aggregator` `TRAINER.AGGREGATOR`;
  `optimizer` / `scheduler` as above.
* `SCORING.STAGES` stage slots now live in the **trainer's** `SlotTable` (`_build_scorer` passes
  `slots=self.slots, trusted=`); P5's `apply_pending` + `sync_from_slots` can use them directly.

### Deviations from the document's code, and defects fixed

* **`identity["engine"]` is not written.** The document writes it "unconditionally" to feed `E091`,
  which belongs to `TrainerEngine` (P5, not built). The same document requires a default run to write
  the pre-change bytes (14.8.1 asserts "no engine key" while 14.8 says every trainer hash changes) and
  the recorded fixture pins the 15-key identity; writing it now fails `test_default_identity`. P5 adds
  `ENGINE_NAME` and the key together with `E091`, and re-records the fixture in that commit.
* **Strict means any difference.** The document's `_check` raised `E094` only for a *structural* change,
  but its own table (14.7.1) lists "`version` bumped only" as `E094` under strict. Strict is now any
  difference; `Finding.fatal` is `status in FATAL or code == "E094"` (the document's property made a
  strict mismatch non-fatal, so `raise_for_errors` would never have raised it). `Change.structural`
  also counts `params`.
* **In-code instances of the framework's own classes are not extensions.** `SlotTable.adopt` gives them
  the `runtime` origin, so `weight_update=UniformAverage()` would have been recorded and changed the
  bytes of a run that is default in effect. `records_in_checkpoint` treats a `runtime` component whose
  class is a registered built-in factory (`ComponentRegistry.is_builtin_class`, new) as a built-in; a
  subclass is recorded.
* **A `runtime` entry that is not registered is not `E093`.** No module can provide an in-code
  instance, so it is a difference (`W052` / `E094`), not "install the module".
* **`extract_manifest` is stricter.** A non-mapping manifest or kind value is returned as a `"?"` entry
  so it is reported `E095`; the document's version dropped such kinds silently, so a hand-edited file
  resumed as if it recorded nothing. An entry whose own `kind` differs from its outer key, a non-string
  or empty `name`, and a non-mapping `params` are `E095` too; `schema` below 1 is `E096`.
* **`W053` is a note in `RestorePlan.notes()`** (the document's `notes()` returned findings only, and
  `report()` carried the text), so `trainer.component_notes` / `TrainResult.component_notes` /
  `ResumeReport.component_notes` hold exactly one line for a legacy file. It is also logged once at INFO.
  Neither new field is in `as_dict()` when empty, so existing payload shapes are unchanged.
* **Optimizer, scheduler, aggregator and reward-term entries.** The document names slots only; these
  four kinds are held by `OptimizerPlan`, `build_aggregator` and `ComposedRewardScorer`, not by the
  table. `OptimizerPlan` entries become slots at manifest time (`params` are the compiled payload, the
  hash is the entry's own `spec_hash`); `build_aggregator(record=)` and `ComposedRewardScorer.
  term_records` expose the `Resolved` they built. A reward term that fails the instance contract is
  simply not recorded (it ran before this phase, so adopting it must not newly refuse it).
* **`get_run` decodes only `components`.** The document also decodes `settings`/`runtime`/`sandbox`
  there; `orchestration/live.py` calls `json.loads(row["settings"])` on the row and
  `runview.decode_run` decodes it again, so decoding in the registry would break the first. Section 16
  can do it together with those readers. `components_json` is renamed `components` as documented;
  `list_runs` returns raw rows.
* `record_components` raises `ImproperlyConfigured` (not a raw `sqlite3.OperationalError`) on a schema-3
  file, so the checkpoint writer's `except ImproperlyConfigured` keeps a save from failing on a
  database that was not migrated; `components()` returns `None` there. The JSON is written with sorted
  keys.
* `MACTFCheckpoint._record_components` remembers whether it (or a resume that found a manifest) ever put
  one in the row, and writes `NULL` once when the manifest becomes empty; a run that never had one does
  no registry call per save. The document called the recorder on every save.
* The scheduler slot is `<optimizer slot>:scheduler`, not `<slot>.SCHEDULER`, which would collide with
  an `OPTIMIZER.OVERRIDES` glob named `SCHEDULER`.

### Existing tests touched

`tests/db/test_migrations_v2.py` and `tests/test_e2e.py`: the schema version is 4 and a fresh
`migrate()` applies `[1, 2, 3, 4]` (the document changes `SCHEMA_VERSION`). Nothing removed or weakened.

### Tests

`test_a_default_run_writes_the_same_identity_hash`, `test_checkpoint_bytes_unchanged_when_all_builtin`
(against the recorded fixture), `test_a_legacy_file_is_one_w053`, `test_a_missing_component_is_e093`,
`test_a_changed_spec_is_w052_or_e094`, `test_migration_4_is_idempotent`, `test_db_migration`, the 14.10
rows 1-28, and: a bit-identical resume of a run with a custom weight update, optimizer and scheduler in
a fresh registry state, the same for one extension of every kind, `E093` listing every missing
component with the trainer, optimizer, RNG and registry rows unchanged, poisoned payloads (`E093`,
`E095`, `E096`) leaving the trainer untouched, `W052`/`E094` for params, source digest and version
skew, orphan entries, a tampered file failing verification, a v1 (Orchestrator) file and a payload
without the key each producing exactly one `W053`.

## P5 - Engine integration and tree mutation (sections 12 and 13)

### Files

New: `mactf/training/{mutation,mutation_engine,mutation_reconcile,path_state,peer_factories}.py`,
`mactf/orchestration/{dispatch,engine}.py`, `mactf/testing/tree.py`; tests `tests/training/test_mutation_*.py`
(plan, limits, simulator, engine, peers, retire, groups, watchdog, checkpoint, equivalence, atomicity,
defaults, properties) with `mutation_support.py`, `mutation_resume_child.py`, `conftest.py`;
`tests/cli/test_conf_mutation.py`, `tests/cli/test_engine_{driver,redteam}.py`;
`tests/components/test_engine_{surface,dispatch,parity,equivalence,signals,db,resume,rollback,feature_gate,defaults}.py`
with `engine_support.py` and `fixtures/orchestrator.sha256`; `tests/orchestration/test_engine_live.py`;
`tests/contrib_api/test_engine_api.py`.
Changed: `trainer.py` (12.7 sites, `ENGINE_NAME`), `watchdog.py`, `logging.py`, `callbacks.py`, `signals.py`,
`coordination.py` (`request_mutation`, `retired`), `tree.py`, `freeze.py`, `optimizer_plan.py`,
`weight_plan.py`, `strategy_plan.py`, `function_book.py`, `contracts.py` (`peer_factory` builtins module),
`checkpoint_v2.py`, `conf.py`, `exceptions.py` (`MutationError`), `driver.py`, `live.py`,
`orchestration/__init__.py`, `testing/{harness,__init__}.py`, `scoring/hierarchy.py` (one line).
`mactf/orchestration/orchestrator.py` (and `phases.py`, `checkpoint.py`) are byte-identical to HEAD; the
digests are pinned in `tests/components/fixtures/orchestrator.sha256`.

### The identity re-record (14.8)

`identity["engine"] = "trainer"` is written by `MACTFTrainer._identity_extra()` on every built trainer.
The only fixture fact that changed is `scalars.json -> trainer -> checkpoint_identity_keys`, which gains
`"engine"`; it was edited by hand (not by `--record`, which would regenerate every number from the new
tree). The fifteen other section digests, the stable-identity hash and `checkpoint.sha256` are the values
recorded on the unmodified tree and still match, which `test_default_identity` and the new
`test_the_engine_key_is_the_only_identity_change` assert. Three P4 tests that asserted "no engine key" were
changed to assert the documented behaviour (`test_persistence_identity.py`).

### Deviations and defects fixed

* **Doc anchors renamed to the real tree**: `_trainer_policy`/`_policy`, `ComponentSpec` lives in
  `components/resolver.py` (not `catalog`), `GroupView`/`tree_view` in `coopetition/{weights,strategy}.py`,
  `adapter.parameter_shapes()` for layouts, `CoordinationContext` is in `training/coordination.py`,
  `_run_epoch_body` split out of `run_epoch` so the `try/finally` phase reset wraps it.
* **Simulator**: an unknown `@token` for `like`/`init_from`/`factory` is `E080` (the doc test expects it, the
  doc code accepted it); `retire` then `revive` (or the reverse) of one node in one plan cancels in
  `retired_now`/`revived_now`.
* **`StrategyPlan.forget_groups`** keys on `key[1]` (the doc used `key[0]`, the slot path). The
  `tree_growth` advice and the `dead_branch` advice share one `if` chain (the doc's second `dead_branch`
  branch was unreachable). `commit_topology` also records per-node blocks in `_swapped`/`_removed` so a
  checkpoint carries `modify ... optimizer` through the optimizer plan, as 12.9.6 assumes.
* **A failed rollback ends the run (R9)**: the engine keeps `poisoned`; every later plan and the next
  `run_epoch` re-raise the `E089`; a queued plan, a schedule entry and the watchdog prune re-raise it
  instead of recording it in `failures`.
* **Layout check** also covers the group of a *rebuilt* peer (membership unchanged); `W042`/`W043` are
  emitted by `_stage`, so a dry run reports them. The schedule key is `(epoch, plan digest)`, so one plan
  scheduled at two epochs runs at both. `MutationLimits.from_settings` emits `W044` when given
  `initial_nodes` and the growth ratio.
* **conf**: every `TRAINER.MUTATION` key (including `MIN_ACTIVE_ROOTS`, `PROTECTED`, `SCHEDULE`) is in
  `TRAINER_DEFAULTS`, because `--set` rejects keys that are not. The env form
  `MACTF_TRAINER__MUTATION__...` of 12.10 does not exist in this tree (`_apply_env` reads top-level scalars);
  row 60 is tested through `--set`. `E090` treats `""` as unset in both `conf` and `engine_name`.
* **Engine**: `load` of a checkpoint takes a path (`record.path`), not an episode; a v1 file is read through
  `_migrate_v1` (which now strips the `module.` key prefix and builds a real flat tree, so a cross-engine
  resume under `ALLOW_ENGINE_SWITCH` works); a run resumed from an explicit file of another run id is a
  *branch* (the file is adopted under the new id, with a note); an explicit checkpoint *directory* is
  restored too (the orchestrator started at `episode + 1` without restoring); `resume_from=False` is "no
  resume"; the `E092` peer-count check runs after `build()`; a second `prepare` keeps the sandbox note.
  `EnginePhaseBridge` uses the doc's mapping. `record_components(source="engine")` follows `start_run`.
* **`HierarchicalScorer.registry_for`** called `registry_for(peer, node_path=)` on a `WarningScorer` that
  has no such argument (always a `TypeError`); it now calls `registry_for_node(node_path, peer)`.
* `RunManager` gains the `tree` event (`tree_mutated` signal), `EVENT_NAMES` gains `"tree"`.
* Not done: `"mutation"` is not yet a filterable record kind in the API serializer/completer lists (P6).

### Findings about the two engines (not P5 defects)

* **D-13.8 - `Orchestrator` never clips gradients.** `rl.loss.clip_gradients` builds `grads` from the
  `parameters` generator and then calls `clip_grad_norm_(list(parameters))` on the exhausted generator, so
  the clip is a no-op; the trainer's facade clips. Hence the documented bit-equality holds only with a
  `TRAINING.MAX_GRAD_NORM` that never engages (the equivalence tests set `1e12`); at the default `1.0` the
  engines agree to `atol=1e-5` and no better (pinned by `test_the_default_gradient_clip_is_the_only_divergence`).
* **D-13.9** - the trainer derives points evidence beside `final_score`; the scorecards differ only in the
  `points`, `evidence_digest` and `points_table_version` keys.
* **D-13.10** - the trainer rolls nodes out one by one, so `episode_started` fires once per node and epoch
  (13.7.1 says once). The run manager does not subscribe to it.
* The `hierarchy` scenario of 19.5 cannot be configured from settings (nothing reads `TRAINER.HIERARCHY`);
  it is compared between `TrainerEngine(trainer=...)` and `MACTFTrainer.fit` on a `HierarchySpec`.
* A retired peer is absent from an `AggregationReport` (weight 0 by omission), not listed with `0.0`: a
  listed zero would violate `WEIGHT_FLOOR` in `legal_report`.

### Tests

Exit criterion: the 33 rows of 13.9 (the doc says 32), `test_orchestrator_trainer_equivalence[plain|hierarchy|
mutating|custom]` (parameters by `torch.equal`), `test_mutation_atomicity[grow|shrink]` (a fault at every
call of 24 steps, then the same plan commits cleanly), `test_a_failed_rollback_is_e089_not_a_half_run` (every
inverse in turn), `test_a_mutated_run_resumes_in_a_new_process` (real subprocess, bit-identical to an
uninterrupted run), `test_orchestrator_files_are_untouched`, `test_guard_property_mutation_plans`, and the
adversarial plan tests (root removal, duplicate names, limits, tied parameters, phase boundary, protected
and infeasible plans, tombstones, tampered journals).

## P6 - CLI, API, system checks, examples and documentation (sections 15 rest, 16, 17, 18)

### Files

New: `mactf/management/commands/components.py`, `mactf/management/structured.py`,
`mactf/components/{scaffold,codes,chain_validation}.py`, `mactf/contrib/components/{__init__,apps,systems,checks}.py`,
`mactf/contrib/components/examples/` (eight modules, `conf/18_N.yaml`, `plans/*.yaml`, an `EXAMPLES` index),
`mactf/contrib/api/views/components.py`, `mactf/templates/conf/{components,rl}.yaml.tpl`,
`scripts/gen_api_docs.py`, `docs/components.md`. Changed: `mactf/conf.py` (`RL_DEFAULTS`, `_apply_env`,
`_close`, SCORING.STAGES shape check, default `INSTALLED_APPS`), `conf_overrides.py`, `conf_files.py`,
`components/{known,registry}.py`, `management/{exitcodes,runsupport}.py`, `commands/{runs,peers,run}.py`,
`completion/completers.py`, `orchestration/live.py`, `contrib/api/{urls,serializers,views/base}.py`,
`contrib/api/management/commands/serve.py`, `http/serializers.py`, `client/{routes,specs,client}.py`,
`exceptions.py`, `training/optimizer_plan.py`, `settings_default.py`, `pyproject.toml` (package data of the
examples), `docs/{cli,api}.md` and `tests/cli/golden/*` (regenerated by their generators),
`docs/plans/02-api-contract.md` (section 3.13, because `test_contract_completeness` pins mounted routes to it).

### What exists for later work

* `RunManager.components / request_components / simulate_mutation / request_mutation`, module-level
  `set_cli_manager` / `manager_for_cli`; `mactf.components.chain_validation.{probe_instance, validate_trainer_spec,
  finding, finding_of, injected_offer}`; `mactf.components.known.{configured_specs, import_builtins}`;
  `Hub.snapshot()` entries now carry `locked` and `fingerprint` (`Component.fingerprint_text()`);
  `mactf.components.codes.{CODE_TABLE, FAMILIES, RESERVED}`; `mactf.components.scaffold.{render_scaffold,
  SCAFFOLD_KINDS, KIND_IDS}`.
* A CLI process can only change a run it drives: `runs components set` and `peers mutate --apply` answer
  exit 4 for any other run (a `RunStateError`), exactly as 16.1.4 says. `set_cli_manager(manager)` lets a
  process that does drive a run (a test, an embedding application) expose it to the same verbs.

### Deviations from the document, and defects fixed

CLI

* `ComponentResolver()` takes no constructor arguments (the document's `ComponentResolver(HUB, trusted=...)`
  does not exist); `validate(spec, trusted=..., inject=injected_offer(kind))`. The manifest helpers are in
  `components/fingerprint.py` (`diff`, `render`), not `components/manifest.py`.
* `components validate` is not given `requires_system_checks`: the attribute is a bool in this tree, and
  `validate` must work on a project whose checks fail. `mactf run` runs the checks as before.
* `components list` imports every built-in module first (`known.import_builtins()`; `mactf.training.policy` is
  in it) so `trainer_policy` shows its two built-ins. `HUB.snapshot()` gained `locked` and `fingerprint`.
* `components inspect` adds the contract verdict, the `locked` flag and the fingerprint; `--probe` of an RL
  function reports one `E047` finding that lists the `F` codes (it carries `probe: [codes]`) instead of one
  finding per `F` code, matching the code table.
* `runs components` and `peers mutate` take `--dry-run` (the contract requires it for `set`/`mutate`); the verb
  `components` is not marked destructive, so `show` never asks for confirmation.
* `peers mutate` without `--run` simulates against a flat tree of the configured peers (nothing is built); with
  `--run` it needs a live run. Exit 1 on a hard finding, as 16.1.4.
* `--component-module` is handled in `add_component_modules` (`runsupport.py`) and shared by `run` and `serve`;
  it is applied after settings load, and the modules are loaded like any in `COMPONENTS.MODULES`. Provenance
  (`mactf config --origin`) does not list it.
* `KNOWN_SETS` in `test_command_contract.py` gained `slot` only. `name` cannot join it: `startproject` and
  `startpeer` have a `name` argument that names something that does not exist yet and has no candidates.
* The scaffold templates are written against the real contracts (`register_scorer`, `RewardScorer`,
  `RewardTerm`, `WeightStrategy`, `PeerStrategy`, `register_function("loss", ...)`, `torch.optim.Optimizer`,
  `register_policy`); the document's versions (`@register("weight_update", ..., params=...)`, `role=`,
  `plan(ctx)`, `step(trainer, batch)`) do not satisfy the contracts. Each template resolves under
  `ComponentResolver.validate` as written (`test_scaffold_renders_and_imports`).
* `classify()`: `ComponentBuildError` is exit 1 (the 16.2.1 table says so; it derives from `ImproperlyConfigured`,
  so it needs its own branch next to the three of 16.2.2). `RunStateError` was already a conflict by class name.
* The document's example `rl_function.loss ppo` names no registered loss; the help example uses
  `policy_gradient`.

Configuration (worker notes, section 15)

* A flat `MACTF_FOO` that names no setting is still ignored (the document's `E018` would break
  `MACTF_UPDATE_GOLDEN`, `MACTF_AGENT_PORT` and the other non-setting variables); only a `__` form raises.
  The env override matches keys case-insensitively and promotes a string spec; values in YAML flow style parse
  as mappings.
* `--set 'TRAINER.WEIGHT_UPDATE={...}'` over a string default failed in the tree; it works for spec slots now.
  `--set X.params.k=v` over a bare name keeps the name: `{type: name, params: {k: v}}`. The document writes
  `{params: {k: v}}`, which is a spec with no name, refused at load, and the lower layer is gone by then.
  `...params=` as the last segment is refused, not promoted.
* `INSTALLED_APPS` defaults gain `mactf.contrib.components`; `test_conf_files` (17 -> 19 files),
  `test_startproject_conf` (counts), `test_conf_equivalence` (the snapshot's apps; "COMPONENTS not authored"
  relaxed because `components.yaml` ships) and two `test_conf_blocks` tests changed (the conf tree now owns
  `COMPONENTS`, so they edit `conf/components.yaml`). The document says these change.
* `check_configured_specs` resolves with `trusted=True` (settings are a trusted surface) and does not check
  `OPTIMIZER` (its `TYPE` is a backend word as often as a name; the plan validates it at build).
* `W016` was in the code table and raised nowhere; `OptimizerPlan.validate` raises it now.

API and client

* `ComponentChoiceField.refresh` is called from `Serializer.__init__` (fields resolve once per class here, not
  per instance). `TrainerSpecSerializer.peer_strategy/weight_update` use a new `ComponentSpecField` (name or
  mapping; keeps the legacy `frozen` and `weights` keys). `GET`/`PATCH runs/{id}/components` share one
  `operationId` (`run-components`) because existing tests pin `operationId == route name`.
* The client has a `client.components` namespace (`ComponentsAPI`) rather than four flat `Client` methods.
* `trainer_policy` probe checks the contract and the unsafe flag (`W004`); there is no synthetic `StepGuard` step.
  No route returns `201`, so `test_status_table` covers 200, 202, 400, 404, 409 and 422.
* `docs/api.md` has no generator in the tree; `scripts/gen_api_docs.py` renders the route table between markers
  from `mactf.http.openapi.schema()`, and `tests/contrib_api/test_openapi_in_sync.py` checks it.
* `"mutation"` is a log-record kind in the API serializer and in the completer list; `EVENT_NAMES` already had `"tree"`.

Examples (section 18)

* The document's code is corrected to the real tree (`dataclasses.replace` for `ScorerOutput`, no-arg resolver,
  `modify` uses `set:`). Paths and globs use `peer-N` rather than `root/worker-N`. `STRATEGY_GUARD.MODE: repair`
  is invalid (`E070`), so 18.4 uses `skip`. 18.2: a negative term raises `ComplianceError`, a bad `scale` raises the
  term's own `ValidationError` (P1's rule). 18.7: the static `TRAINER.POLICY` is `E003` from `SlotTable.swap`;
  `SlotTable.request` queues it and the swap is refused at the boundary. 18.8's purge plan has no `force`, so
  the refusal is real.

Other

* `exceptions.py` defined `MutationError` twice (P5 left both); the first copy was removed.

## Conformance audit additions (sections 12-13)

* Names that differ from the document, behaviour identical: `mutation_reconcile.TopologyReconciler` (doc
  `MutationReconciler`) and `ReconcilePlan` (doc `Diff`); the doc's `StagedTopology` is the `Staged.optimizer`
  field; `structure_digest` is a function in `training/mutation.py` (doc also lists a `PeerTree` method);
  `CoordinationContext` lives in `training/coordination.py`. The `E080`/`E088`/`E089`/`W045` rows in
  `components/codes.py` keep the doc's `MutationReconciler` wording.
* Matrix 12.11.5 row 40 asserts omission from the `AggregationReport` (weight 0 by omission), see P5 notes.
* Matrix test files live in `tests/training/` (section 12) and `tests/components/`, `tests/orchestration/`,
  `tests/cli/`, `tests/contrib_api/` (section 13) rather than the doc's directories; every one of the 93 named
  rows resolves to a test, row 60's env half is `tests/test_conf_env.py`.

## Independent verification - compliance

An adversarial pass on claim D2 ("customisation never weakens compliance; enforced by guards"). Tests:
`tests/components/adversarial/` (`adv_support.py` is the harness: a real `MACTFTrainer` whose scorer is made
to record a non-offsettable `P1` breach on `peer-1`, so every attack is judged on three facts - the node is
still void, it is scored no better than the built-in cost of -5, and it carries zero weight - or the attack
is refused with a documented code). About 330 cases over scorer stages (output, `prepare`, ledger, config,
number-subclass and metaclass attacks, both `GUARD_MODE`s), reward terms, RL functions, weight strategies,
aggregators, peer strategies, trainer policies, peer factories and mutation plans, registries and the legacy
dict views, the import allow-list, the API request surface, slot swaps in a running engine, and resume with a
doctored manifest. Ten real defects were found and fixed; each has a regression test.

### Defects found and fixed

1. **A metaclass impersonated a built-in stage (guard bypass).** Trust was decided with
   `type(stage) in (PenaltyScorer, ...)` (and the same shape for strategies, weight strategies and rollups).
   `in` calls `==` on the class, so a stage whose metaclass defines `__eq__` returning `True` (and a `__hash__`
   for the `frozenset` tables) was "built-in" and skipped the guard: a zero-cost, no-void penalty scored
   `0.0` instead of `-5.0`. All six sites (`guards/compliance.py` `_trusted`, `scoring/pipeline.py`,
   `guards/strategy.py` `covers`, `components/strategy_plan.py` `customized`, `guards/rollup.py`,
   `coopetition/weight_guard.py`) now compare by identity (`type(x) is built`).
2. **Ledger tampering the guard could not see (C007 bypass).** (a) A `prepare` could return a context whose
   ledger is any object claiming `__class__ == ReadOnlyLedger` with an `unwrap()` that returns the source;
   `isinstance` accepted it and every later stage - and the reference penalty - scored an empty ledger.
   (b) `object.__setattr__(event, "magnitude", 0.0)` rewrote a frozen event in place; the snapshot compared
   the same objects with `==`. (c) A lookalike event with `__eq__ -> True` replaced a real one. Now:
   `check_prepare` requires the *same* ledger object (identity, exact `ReadOnlyLedger`, same `_events` list);
   `snapshot` records every event with its field values and the list that held them; `check_after_stage`
   compares identity and values, and restores the objects, their fields and the list itself.
3. **`==` is not a proof of "unchanged" (C006 bypass).** `_same` trusted `before == after`; a `dict`/`frozenset`
   subclass whose `__eq__` always says `True` replaced `ctx.config`, `ctx.branch_void`, `known_rules`. `_same`
   now requires identical types and compares containers and dataclasses element by element.
4. **A stage's numbers were read twice (guard/pipeline TOCTOU).** The guard validated `float(x)`; the pipeline
   later used the raw object, so a `float` subclass with `__float__ -> 5.0` and value `0.0` passed C002 and
   charged nothing. `ComplianceGuard.freeze` snapshots each custom stage's output into exact builtin types
   once; the inspectors and the pipeline both use that copy (an unreadable output is `C001`). The weight guard
   (`weight_guard.read_once`: a `float` subclass overriding its comparisons slipped under `> tol`, so a void
   peer kept weight) and the rollup guard (`_read_once`) got the same treatment.
5. **Numeric strings passed as numbers.** `_number("5.0")` is `5.0`, so `amounts = {"P1": "5.0"}` passed every
   check and crashed `constrain` with a raw `TypeError` (`C001` was documented). `str`/`bytes` are now
   rejected in the compliance and rollup guards.
6. **A weight strategy could un-void a peer by editing its input.** The trainer passed its own `eligible`
   dict to `apply()` and then judged the report against the same dict: `eligible[void] = True` gave the void
   peer a third of the blend with the guard on. A custom strategy now gets copies of `eligible`, `confidence`,
   `scores` and the vectors (the default `KappaSoftmaxAverage` is passed the originals: identity unchanged).
7. **Void peer in a blend through the coordination context and through a custom aggregator.**
   `ctx.unchecked()` and `ctx.checks = False` (both public) switched off the Y001/Y002 checks, so a
   strategy could `aggregate_root(<report that weights the void peer>)`; `Aggregator.aggregate` of a subclass
   was shown every adapter. Fixes: `CoordinationContext.checks` is read-only, `unchecked()` keeps Y001/Y002
   (only scope/once/order bookkeeping is dropped), `MACTFTrainer._write` - the one place parameters are
   written - refuses a report that weights a void or frozen peer (`WeightGuardViolation`, `U004`), a custom
   `aggregate` is handed only the peers with positive weight, and a custom blend that is not finite writes
   nothing (`ComplianceError`).
8. **A loss could claim to carry the compliance charge.** `carries_compliance` read each term's
   `spec_type` and walked any `.terms` attribute, so `spec_type = "compliance_floor"` (or a decoy `terms`)
   made `TRAINER.LOSS_COMPLIANCE: require` add nothing. It now trusts the class (exact
   `ComplianceFloorLoss`/`CompositeLoss`) and walks only `ChainedLoss`/`WeightedSumLoss`. `CARRIES_COMPLIANCE`
   stays, as the names of the two carriers.
9. **The allow-list could be walked around, and `builtin` could be claimed.** `acme.mod:os.system` /
   `acme.mod:torch.optim.SGD` resolved code of modules the allow-list never named; an attribute path may no
   longer cross a module that is not allow-listed (`E006`). `register(..., origin=Origin.BUILTIN)` (also a
   decorator option) gave a plugin the origin that skips `StepGuard`, the `unsafe` gate and the "custom"
   scope of every guard; the origin is now refused unless the first caller outside `mactf.components` is a
   `mactf` module (the framework's own `peer_factories` registration). `register_builtin` is unchanged (see
   residual risks).
10. **The weight-strategy probe handed its module-level cases to user code.** `weight_probe._CASES`,
    `_SPREAD` and the probe vectors went to `apply()` as they were, so a strategy that set
    `eligible[void] = True` while being probed rewrote the cases for every strategy probed after it in
    the process, and a void-weight bug then passed `E064`. `apply()` now gets copies.

Also fixed (not compliance-weakening, found on the way): the engine applied a queued `scoring.*` slot swap
to the slot only, with no `inject` (the new stage got an empty config) and no `sync_from_slots`, so the slot -
and the checkpoint manifest - named a stage the run was not using (`orchestration/engine.py::_apply_boundary`,
not `orchestrator.py`); a peer factory could return non-finite parameters/buffers and the new peer joined the blend
(`E085` now, `training/mutation_engine.py`; a module shared with another node was already refused by the tie check).

### Attacks that were already refused or harmless (kept as tests)

Penalty output (zero, scaled, omega < 1/NaN, NaN/inf/negative/bool/None amounts, phantom rules, mutations,
non-mapping breakdown, lazy iterators, mapping that answers differently on the second read, random cost);
`prepare` attacks (in-place and nested `config` edits, `severity_weights.clear()`, `allowed_rules`,
`precedence`, peer rename, warning-only hiding of `P1` -> `C005`, ledger `record`/`clear`/`_events` ->
`C007` with the ledger restored, tamper-then-raise); reward stages and terms (NaN/inf/negative/huge, ceiling
`C008`, edits to the ledger/config from inside `contribution`); weight strategies (17 shapes, three guard
modes, build-time probe `E064`); aggregator NaN/inf/wrong-size blends; peer strategies (fabricated reports,
private backend, write twice, raise); policies (void batch is clamped and read-only, NaN loss, raise after
writing, unsafe without consent); RL functions (NaN, input mutation, non-determinism refused at build,
rollup dropping a child's void set -> `C010`/`F010`, non-finite rollup amount `C011`); legacy dict views
(`STRICT_LEGACY` refuses every write/delete/pop/clear/update/`|=`; a default-mode replacement logs `W001` and
the new class is still inspected; `dict.__setitem__` does not touch the frozen trust sets); a slot swap to a
cheap stage is refused from its first episode; resume with a missing, empty or doctored manifest never turns a
guard off (the restorer verifies, it does not install); the API surface refuses imports and bad specs and
queues nothing.

### Residual risks (not fixed; by design or out of reach of an in-process guard)

* **The guards are not a sandbox.** A component is Python in the same process: it can monkey-patch
  `PenaltyScorer.score`, a guard, `HUB`, `torch`, or call `HUB.registry(k).register_builtin(...)` (the
  framework's own entry point, left callable because the existing tests use it). Plugins and `COMPONENTS.MODULES`
  are operator-trusted code; the allow-list protects the *untrusted* surface (API) only.
* **`requires_terms` is opt-in (7.6.4).** A `trainer_policy` that declares no `compliance_floor` is not checked
  for the charge; the doc says so and points at `unsafe`. `LOSS_COMPLIANCE: require` covers the `loss` family;
  a whole policy is only as honest as its declaration.
* **A custom optimizer under the built-in default policy** is probed for two steps at build (`O003`) and is not
  wrapped by `StepGuard` (a built-in policy never is): a time-bomb `step()` that writes NaN later poisons the
  peers silently. Suggested: a finite-parameter check at the epoch boundary for non-built-in backends.
* **A thread started by a stage** that edits the ledger after the walk is outside the snapshot window.
* `GUARD_MODE: fallback` hides a buggy stage behind the reference output (by design, `W003`); C006/C007 still raise.

## Independent verification - e2e

Tests: `tests/e2e_components/` (real `startproject` tree copied per test; `manage.py` subprocesses with SIGTERM/SIGKILL;
Hypothesis stateful and garbage-input fuzz; Python client against a real `serve` subprocess). Helpers `e2e_support.py`,
`swap_support.py`, `swap_resume_child.py`.

### Scenarios that pass

* CLI `run` interrupted by SIGTERM and SIGKILL (+ `runs reap`) and resumed with `--resume`: **bit-identical** to an
  uninterrupted run (models, reference, optimizers, schedulers) for the default configuration and for a customised one
  (project `COMPONENTS.MODULES` `ema_softmax` weight update + `Lion` + `step_then_flat`), TrainerEngine,
  `ORCHESTRATOR.ENGINE_COMPAT=false`.
* `manage.py run` and `serve` + `MACTFClient` (HTTP) produce identical final tensors; a live run is steered through the
  client (component PATCH, unknown name -> 404, mutation dry run) and cancelled.
* Swap the optimizer (epoch 1), the weight update (epoch 2) and queue a tree-growing plan (epoch 3), stop after epoch 3,
  resume in a real new interpreter that starts from the *unswapped* configuration: the swaps and the pending plan come
  from the checkpoint and the run ends bit-identical to the uninterrupted one.
* `components scaffold` -> edit -> `run` trains (weight update, optimizer); the unedited scaffolds are refused up front
  with one coded line (`E056`/`E064`/`E074`), no traceback; a real entry-point plugin (dist-info on `PYTHONPATH`) is listed
  with origin `plugin` and used; `COMPONENTS.PLUGINS=off` refuses it cleanly.
* Fuzz: 60-example (250 ad hoc) stateful sequences of plans and epochs on a live trainer (simulator and engine agree on verdict,
  error code and `after_digest`; refused plans leave nothing; the last checkpoint restores into a fresh trainer with the
  same topology and tensors); model-based registry/overlay/hub state machine; `merge_layers` against a reference model;
  `canonical_json` stability; 400 random `--set` lists and 150 `MACTF_*` variables through `apply_overrides` +
  `validate_settings` (only `MACTFError`); thread-safe reads; two overlays in two threads do not leak.
* Performance (HEAD = pre-doc-17 tree exported with `git archive`, same bench, 12 epochs x 3 peers, median of 7):
  `MACTFTrainer.fit` 0.546 s (HEAD) vs 0.541 s and 0.535 vs 0.597 s (working tree) - no measurable change; `import
  mactf.conf` 0.12 s vs 0.11 s; `mactf --help` 0.35 s vs 0.31 s. `TrainerEngine` vs `Orchestrator` (6 epochs): ratio 1.6
  (different engines, not a regression). `registry.find` 0.3 us (0.6 us inside an overlay). 60-epoch run: 1.0 MiB
  retained between epoch 10 and 60, file descriptors 7 -> 7.

### Defects found and fixed

1. **`TrainerEngine.prepare` built the peers' modules before seeding** (`orchestration/engine.py`), so one `SEED` gave
   different initial weights in every process (the Orchestrator seeds first through `build_runtime`). Two identical
   `manage.py run --seed 7` commands with `ORCHESTRATOR.ENGINE=trainer` diverged from epoch 0, and an interrupted run could
   never equal an uninterrupted one. Fix: `seed_everything(SEED, deterministic=DETERMINISTIC)` before `_default_modules`
   (only when the engine builds the modules itself). Test: `test_the_same_seed_gives_the_same_run_in_two_fresh_processes`.
2. **First concurrent use of the registries from several threads** (`mactf serve` is threaded): the lazy built-in import
   raised `_DeadlockError` (in 6 of 6 trials with 8 threads) or returned a half-filled registry (28 to 111 components
   depending on the interleaving). Fix: `build_server` calls `known.import_builtins()` once, on its own thread, before
   serving. Test: `test_a_server_reads_the_registry_from_many_threads_on_its_first_requests` (subprocess, 3 trials).
   Residual: user code that reads a fresh `HUB` from several threads before anything imported the built-ins can still
   hit this; `ComponentRegistry._ensure_builtins` sets its ready flag before the import finishes and takes no lock. A
   lock would trade the loud `_DeadlockError` for a possible hang against a plain `import`, so it was left alone.
3. **Non-string parameter names in a spec** (`{type: x, params: {1: 0.5}}`, which YAML produces for `1:`) reached
   `Component.check_params` and `canonical_json` as a raw `TypeError` (`sorted` / `sort_keys`). Fix:
   `ComponentSpec.parse` raises `ComponentSpecError` ("parameter names must be strings"). Test:
   `test_non_string_parameter_names_are_a_spec_error_not_a_typeerror`.

### Findings that are not defects of the document (residual risks)

* **A resumed run is not bit-identical under `ORCHESTRATOR.ENGINE_COMPAT=true` (the default) or under the Orchestrator.**
  `CumulativeSignal.cards` (and the Orchestrator's `PhaseContext.scorecards`) is in memory only, so after a resume the
  cumulative signal window restarts: epoch 4 of a resumed run aggregated uniform weights where the uninterrupted run
  used a non-uniform blend. The run continues validly but differs from the uninterrupted one. With
  `ENGINE_COMPAT=false` the resume is exact (pinned by `test_an_interrupted_trainer_run_resumes_exactly`); the compat
  and Orchestrator cases are pinned as "resumes and completes" by `test_the_legacy_signal_window_resumes_without_error`.
  A fix needs the window persisted in `identity["trainer"]` (final_score: per-peer sum and count; points: the cards'
  `final_points`); not done because it changes the checkpoint of a default run (invariant I1, the pinned identity hash).
* A SIGKILLed run stays `running` until `runs reap` (documented); `--resume` answers a conflict naming that state.
* `--set SCORING.STAGES...` / `--set TRAINER.POLICY=...` are refused until the key exists (documented in 15.7.3); a conf
  file or `--config` is the way to introduce them. The `COMPONENTS.PLUGINS=off` refusal does not echo the offending
  name ("TRAINER.WEIGHT_UPDATE must be one of ...").
* Hot reload of a plugin module is `E002` by design (Q4); pinned.
