# 17 conformance audit

Independent audit of `17-component-customization.md` against the implemented tree (P0-P6). Status: OK, GAP-fixed (gap found and closed in this audit), DEVIATION-justified (see `17-implementation-notes.md`). Rows group related items; every row was verified by reading code and running or locating a test.

## Sections 3-5 Component model, origins, resolver

| item (doc section) | code location | test | status |
|---|---|---|---|
| 3.1 package layout (all listed modules incl. guards/{compliance,step,weights}) | `mactf/components/*.py`; guards dir has compliance/rollup/step/strategy (weights guard is `weight_plan`/WeightGuard elsewhere, 10.4) | `test_import_weight::test_components_import_pulls_in_no_framework_package` | OK (all 153 `class`/`def` names in 207-2500 exist in `mactf/`) |
| 3.1 imports only exceptions+stdlib at import; no torch | `components/__init__.py` | probe: `torch` not in sys.modules after `import mactf.components`; test above | OK |
| 3.2 Origin/Mutability/Boundary ranks, ComponentKind, RL_FAMILIES, kind_id, parse_kind_id, KindSpec, KIND_SPECS, all_kind_ids | `kinds.py` | `test_kinds.py` (12) | OK; extras `resolve_kind`, `legacy_contract`, `builtins_module` recorded in notes P0 |
| 3.2 default-mutability table (scorer/reward_term episode; loss/regularization/gradient/metric step; other rl episode; optimizer/scheduler/weights/peer_strategy/peer_factory episode; aggregator/trainer_policy static) | `contracts.py` KIND_DEFAULTS | probe of `KIND_SPECS` matches the table row by row; `test_kinds`, `test_contracts` | OK |
| 3.2 looser mutability than kind default refused | `registry.py register` | `test_registry::test_a_looser_mutability_than_the_kind_default_is_refused` | OK |
| 3.3 ParamSpec (type/default/required/choices/min/max/doc, problems, describe; bool is not a number) | `params.py` | `test_reflect::test_param_spec_problems`, `..._describe_...` | OK (int accepted for float: DEVIATION-justified, notes P0) |
| 3.3 reflect: constructor_keys/Reflection, source_digest, canonical_json, spec_hash (16 hex, pinned), float repr | `reflect.py` | `test_reflect.py` (16 incl. pinned hash, key-order property) | OK; C-level opaque init: DEVIATION-justified |
| 3.3/3.4 Component record fields, describe JSON-ready, description default = docstring line, builtin has no source digest | `registry.py` | `test_registry::test_the_component_record_is_complete`, `..._describe_is_json_ready`, `..._builtin_has_no_source_digest...` | OK |
| 3.4 name rules (invalid names contract error, lower-case for strategy/weight, historical family message) | `registry._check_name` | `test_registry` name tests (3) | OK |
| 3.4 register atomic (contract failure/collision leave registry+mirror untouched), replace, per-kind cap, frozen, unregister, generation counter | `registry.py` | `test_registry` (34), `test_properties::test_the_registry_agrees_with_the_reference_model` (hypothesis) | OK |
| 3.4 build: E001 unknown (+did-you-mean), E004 unknown param listing accepted, missing required named, var-kw accepts unless schema, schema validated before ctor, injection offered not demanded, E011, E008 wrap only TypeError/ValueError, ValidationError propagates (I8), instance contract after build, custom builder | `registry.build` | `test_registry` tests lines 40-401 | OK (E011 as ComponentSpecError, E001 suffix: DEVIATION-justified, notes P0) |
| 3.4 W001 on builtin replace; ALLOW_BUILTIN_OVERRIDE; overrides_builtin | `registry._check_replace` | `test_registry::test_w001_...`, `test_isolation::test_builtin_replacement_*` | OK |
| 3.5 Hub: configure/policy/effective_policy, scoped overlay stack via contextvars, innermost wins, hide in overlay, frozen still takes overlay writes, loading(origin), default_origin, snapshot, freeze_all | `registry.py Hub/Overlay` | `test_overlays.py` (20) | OK |
| 3.5 threads: plain Thread loses context; RunManager uses copy_context().run | `orchestration/live.py:803` | `test_overlays::test_a_plain_thread_starts_with_an_empty_context...`, `test_activate_re_enters...`, asyncio + two-thread isolation | OK |
| I9 per-run isolation; overlay never reaches legacy table | `Hub`, `LegacyTable` | `test_isolation.py` (5) | OK |
| 3.6 LegacyTable: dict subclass, reads plain, writes via registry, `_mirror_set/_mirror_del` bypass | `legacy.py` | `test_legacy.py` (20); probe `isinstance(SCORER_TYPES, dict)` and mirror == `{name: factory}` | OK |
| 3.6 installation sites: SCORER_TYPES, PEER_STRATEGIES, WEIGHT_STRATEGIES, 12 RL families as LegacyTable in `FUNCTION_TYPES` (outer plain dict), aliases same object | scoring/registry, coopetition/strategy+weights, rl/functions | probe: types verified; `test_legacy`, `tests/rl/test_function_registry.py` | OK |
| 3.6 behaviour changes: legacy builtin replace logs W001+W005; invalid scorer fails at write time | `registry._check_replace`, `scorer_contract` | `test_registry`, `test_legacy`, `test_decorators::test_register_scorer_legacy_form_keeps_the_old_refusals` | OK |
| 3.7 contracts per kind (+instance contracts, hierarchy kind accepted, reward_term) | `contracts.py` | `test_contracts.py` (24) | OK (strict vs legacy_contract for RL: DEVIATION-justified, notes P0) |
| 3.8 `register`, `unregister`, `get_component`, 8 kind decorators, legacy `register_scorer/function` (None return) and decorator forms, re-exports at old import paths, all 9 options via `**options`/explicit, scope="run" without overlay -> ComponentLocked | `decorators.py` | `test_decorators.py` (14); probe: `rs1 is d.register_scorer` | OK (resolve_kind accepts full kind ids: superset) |
| 3.9 invariants 1-5 (sorted names, mirror equality, atomic register, frozen rule, no I/O) | `registry.py` | `test_registry` sorted/mirror/frozen; `test_properties` | OK |
| 4.1 origin ranks and who-may-replace rule | `registry._check_replace` | `test_replacement_table.py` (9 tests, cells written from the doc table, with/without flag) | OK |
| 4.2 PLUGINS forms (auto/list/off), entry-point group `mactf.components`, install(hub) gets plugin origin, MODULES loaded once in order with project origin, missing module ImproperlyConfigured, E008 on raise, load once per process, phased: plugins, modules, locks, freeze | `plugins.py` (ENTRY_POINT_GROUP verified) | `test_plugins.py` (28) | OK; RLock, restore-on-failure: DEVIATION-justified (notes P0) |
| 4.3 table rows incl. unregister, locked, frozen (global vs run scope), legacy over builtin allowed/STRICT_LEGACY | `registry.py` | `test_replacement_table`, `test_isolation::test_builtin_removal_needs_the_flag_too` | OK; STRICT_LEGACY -> ComponentLocked per 15.8.1 (notes) |
| 4.3 unregistering a registered component does not affect a live slot (slot holds instance+Component) | `slots.py Slot` | `test_slots::test_a_failing_factory_leaves...`, `test_adopt_records...` (slot keeps record); no direct unregister-while-running test | OK (behaviour structural) |
| 4.4 slot paths installed by the engine | pipeline `scoring.<kind>`, `scoring.reward.terms.<label>`, `OPTIMIZER[.glob]`, `TRAINER.POLICY`, `TRAINER.WEIGHT_UPDATE*`, `TRAINER.PEER_STRATEGY*`, `RL.<family>` | `test_scoring_stages`, `test_weight_plan`, `test_strategy_plan`, `test_function_book`, `test_optimizer_plan` | DEVIATION-justified: doc 4.4 table spellings `SCORING.CHAIN.x`, `TRAINER.OPTIMIZER.x` differ from the doc's own 6.2/9.x code (`scoring.<kind>`, `OPTIMIZER`); the real code follows 6.2/9.x |
| 4.4 boundary rule `rank(b) <= min(old,new)` ; run_start/episode/step; static only at run start; episode never between steps | `slots.SlotTable.swap` | `test_slots::test_a_slot_changes_only_at_its_boundary` (param matrix), `..._message_names_the_boundary...` | OK |
| 4.4 swap accepts raw or parsed spec; failed swap/factory/migrate leaves old running (E048 wrap); migrate before commit sees both instances; SwapRecord; history | `slots.py` | `test_slots.py` (18) | OK |
| 4.4 request/apply_pending queued to next boundary, eager validation, slot order, failing build stays queued; E010 missing slot lists paths | `slots.py` | `test_slots` lines 121, 247-296 | OK |
| 4.4 caller table (engine install run_start, apply_pending(EPISODE), request) | `orchestration/engine.py`, `live.py` | `test_engine_*`, `test_components_api` | OK |
| 4.5 lock grammar `<kind>:<name>` / `slot:<path>` (fnmatch), bad pattern explained, kind lock != slot lock, slot lock blocks swap+request but not install, locked component not shadowed by overlay, locks applied in phase 2 | `known.py`, `registry.py`, `slots._slot_locked`, `plugins.py` | `test_locking.py` (10) | OK |
| 4.5 recommended pin `slot:SCORING.CHAIN.penalty` / `.warning` actually locks the scoring slots | engine slot is `scoring.penalty`, so the documented pin matched nothing | none (the 2 existing tests locked an unrelated slot) | GAP-fixed: `slots.slot_lock_names/slot_locked_by` map `scoring.<k>`->`SCORING.CHAIN.<k>`, `scoring.reward.terms.<l>`->`SCORING.REWARD_TERMS.<l>`, `OPTIMIZER[.g]`->`TRAINER.OPTIMIZER.<default|g>`; `optimizer_plan._boundary_allowed` uses it; new test `test_a_lock_in_the_settings_spelling_locks_the_engine_slot` (8 cases) |
| 5.1 spec shape: bare string = `{type}`; flat RL form valid; flat+params = E012 | `resolver.ComponentSpec.parse` | `test_resolver::test_parse_accepts/rejects`; probe shows `E012` and flat parse | OK |
| 5.2 ComponentSpec/Resolved/ComponentResolver: parse (allow_patch), merged_over (same type merge, different replaces, patch onto lower, import kept), canonical form/hash, resolve/validate/adopt, associativity | `resolver.py` | `test_resolver.py` (30 incl. hypothesis associativity) | OK |
| 5.3 nine layers in order, layers 8 over 7 via merged_over, layer 9 wins/adopted without build | `resolver.LAYERS/merge_layers/LayerResult` | `test_resolution.py` (14): order, worked examples, "which layer decided", does not mutate | OK (`merge_layers` is an addition, notes P0) |
| 5.4 allow-list: trusted surfaces any path; untrusted refused unless ALLOW_IMPORT and fnmatch IMPORT_ALLOWLIST (default empty); registered names need no permission; denied import never imports | `resolver._import`, `COMPONENTS_DEFAULTS` (verified `ALLOW_IMPORT False`, `IMPORT_ALLOWLIST []`) | `test_resolver` lines 251-366; `test_slots::test_an_untrusted_table_refuses_an_import_spec` | OK (module raising -> E008: DEVIATION-justified) |
| 5.5 worked resolution (trim 0.1 -> 0.2, spec_hash, manifest entry shape) | resolver + fingerprint | `test_resolution::test_the_worked_example_of_section_5_5`, `test_fingerprint` | OK |
| 1.4 invariants I5, I7, I8, I9, I10 (in this range) | see above | `test_plugins` (I5), `test_slots` (I7), `test_registry` (I8), `test_isolation` (I9/I10) | OK |
| Doc-named tests in 207-2500 (every `test_*` token) | n/a | all exist (checked by script) | OK |

## Sections 6-7 Scoring and training policies

| item (doc §) | code location | test | status |
|---|---|---|---|
| 6.1 stage kinds/orders, pipeline-owned precedence | scoring/{warning_scorer,reward_scorer,penalty,pipeline}.py | tests/scoring/*, test_scoring_identity.py | OK |
| 6.2 SCORING.STAGES, STAGE_KINDS, build_stages (unknown kind, kind mismatch, E011 injected key, untrusted import) | scoring/pipeline.py build_stages | test_scoring_stages.py (unknown_stage_kinds, wrong_kind, may_not_set_a_key_the_framework_injects, import_spec_refused) | OK |
| 6.2 WARNING contract line (subclass WarningScorer) | components/contracts.py:73 | test_a_warning_stage_must_subclass_warning_scorer | OK |
| 6.2 Scorer.from_spec / _build_from_kwargs via registry; _constructor_keys alias | scoring/base.py | test_scorer_from_spec_goes_through_the_registry; tests/scoring/test_pipeline_validation.py | OK |
| 6.2 verbs table (no zero_penalty) | null_stages.py | test_a_zero_penalty_stage_cannot_be_written | OK |
| 6.3 reward terms, composed_reward, admitted_only=False, >=1 term, finite non-neg, rules subset | scoring/reward_terms.py | test_reward_terms.py (test_composed_matches_builtin + 25 more) | OK |
| 6.4 C001-C009 | components/guards/compliance.py | test_compliance_guard.py, test_guard_properties.py | OK |
| 6.4.2 enforce/fallback; C005-C007 always raise; metadata["compliance_guard"] | same | test_fallback_* , test_a_warning_stage_violation_is_not_repaired_in_fallback_mode | OK |
| 6.4.2 W003 on fallback start | pipeline.py build_scorer | none before | GAP-fixed: test_w003_is_logged_when_a_run_starts_in_fallback_and_not_otherwise |
| 6.5 pipeline edits, _BUILTIN_STAGE_TYPES, default byte identity | pipeline.py | test_default_pipeline_is_byte_identical | OK |
| 6.6 H1 effective_non_offsettable, penalty/hierarchy | compliance.py, penalty.py, hierarchy.py | test_scoring_hardening.py | OK |
| 6.6 H2 COMPLIANCE defaults/ranges/validator | conf.py | test_a_bad_compliance_block_is_refused etc. | OK |
| 6.7 validate_stage_chain | scoring/chain_validation.py | test_chain_validation_agrees (+random chains) | OK (documented ordering/pipeline_nested deviations sound) |
| 6.8 replace/sync_from_slots/stage_inject, null stages, I6 | pipeline.py, null_stages.py | test_scoring_stages.py (swap atomic, warning state kept) | OK |
| 6.9 verbs table | - | covered by above | OK |
| 7.2 TrainerPolicy/PolicyHost/aliases | training/policy.py, trainer.py | test_the_trainer_implements_the_host_surface | OK |
| 7.3 DefaultTrainPolicy hooks, advantage_masks, void floor after custom advantage | training/policy.py | test_the_default_policy_is_the_old_step, test_clipped_advantage.py | OK |
| 7.4 trainer changes (loss_for, metadata, state) | trainer.py | test_policies.py | OK |
| 7.5 precedence/co_distill/static slot | policy_resolver.py | test_policies.py (conflict, composes_distillation, slot static) | OK |
| 7.6 S001-S009, modes, config keys/defaults, snapshot, after_step | guards/step.py, conf.py | test_step_guard.py | OK |
| 7.6.2 CHECK_PARAMS=false | step.py | none before | GAP-fixed: test_check_params_false_switches_the_s003_check_off |
| 7.6.5 wrapping table | policy_resolver.py | test_the_default_policy_is_never_wrapped, test_a_trainer_instance_in_code_warns_first | OK |
| 7.7 unsafe: flag, error at build, W004, identity, card metadata | policy_resolver.py, check | test_an_unsafe_policy_*, test_system_checks | OK |
| 7.8 weighted_sum, ensure_compliance, LOSS_COMPLIANCE modes | rl/functions/loss.py, loss_compliance.py | test_rl_registry_and_conf.py, test_policies.py | OK |
| 7.9 co_distill | distill/policy.py | test_co_distill_by_name_* | OK |
| 7.10 worked example | - | test_the_worked_example_policy_runs_under_the_guard | OK |

## Sections 8-9 RL functions, optimizers

| item (doc section) | code location | test | status |
|---|---|---|---|
| 8.1-8.2 twelve RL families, registry integration, `register_function` | mactf/components/kinds.py RL_FAMILIES, mactf/rl/functions/registry.py | tests/rl/test_function_registry.py, tests/components/test_rl_registry_and_conf.py | OK |
| 8.3 FunctionContract probes F001-F012 | mactf/components/function_probes.py | test_function_probes.py (one test per code, F001..F012) | OK |
| 8.3 probe triggers / RL.VERIFY custom,all,off | function_book.py `_verify` | test_function_book.py (test_probe_runs_before_commit etc.) | OK |
| 8.10.2 W013 "at build" when RL.VERIFY off | function_book.py `_verify` (was system check only) | test_function_book.py::test_w013_is_logged_once_when_a_custom_function_runs_unprobed | GAP-fixed |
| 8.4 FunctionBook select/specificity/instances (R7, R8) | function_book.py `_select`, `_instance` | test_override_specificity_*, test_instances_are_per_node | OK |
| 8.5.1 RL block defaults (UNCONSUMED=error, STRICT_GLOBS=true, VERIFY=custom) | conf.py RL_DEFAULTS | test_conf_known.py, test_rl_registry_and_conf.py | OK |
| 8.5.2-8.5.4 validation, consumers, E040/E041/E042/W010 | conf._validate_rl, FunctionBook.validate | test_function_book.py, test_rl_registry_and_conf.py | OK |
| E043/E044/E045/W011 | FunctionBook._select/validate | test_function_book.py | OK |
| 8.6 rollup unification, RollupGuard C010/C011/C012 (R4) | components/guards/rollup.py | test_rollup_guard.py (c010/c011/c012 + hypothesis property tests) | OK |
| R5 branch_void beats custom rollup | coopetition/tree.py branch_void, scoring/hierarchy.py | test_rollup_guard.py::test_branch_void_beats_custom_rollup (named test was absent) | GAP-fixed |
| 8.7 synthesis adapter, build_synthesis, E046, R6 | rl synthesis adapter | test_median_synthesis.py (incl. test_synthesis_failure_is_node_local) | OK |
| 8.8.1 migration defaults per family | components/migrations.py FAMILY_DEFAULTS (matches table; synthesis "fail" is a no-op for stateless) | test_function_book.py test_migration_* | OK |
| 8.8.2-8.8.4 migrate/swap atomic/delete (R9, R10, E048) | migrations.py, function_book.py `_swap` | test_swap_is_atomic, test_migration_*  | OK |
| 8.8.5 state in checkpoint, E049/W012 (R11) | FunctionBook.state_dict/load_state_dict | test_function_state_roundtrip + E049/W012 tests | OK |
| R1-R3, R12 | FunctionBook | test_default_run_unchanged, test_unconsumed_family_is_error, test_override_loss_gets_compliance, test_probe_runs_before_commit | OK |
| 8.9.1 ClippedAdvantage | tests/components/p3_examples/example module | test_clipped_advantage.py (test_clip_bounds_and_identity...) | OK |
| 8.9.2 L1AnnealRegularization | example | test_l1_anneal.py | OK |
| 8.9.3 TrimmedMeanRollup + guard test body | tests/components/p3_examples.py | doc test bodies literally in test_rollup_guard.py (test_union_survives_trimming, test_laundering_...); separate file name differs | DEVIATION-justified (consolidated in test_rollup_guard.py) |
| 8.9.4 MedianSynthesis | example | test_median_synthesis.py | OK |
| 9.1-9.2 BackendBase, NullOptimizer, RouterOptimizer, registrations | mactf/training/optimizer.py | test_optimizer_migrations.py (test_backend_base_keeps_its_old_name, router tests) | OK |
| 9.2.2 MACTFOptimizer spec identity, `_reconcile`, E055/W015 | optimizer.py | test_resume_with_changed_optimizer_* | OK |
| 9.3 schedulers warmup_cosine/warmup_linear/constant_with_warmup/inverse_sqrt + torch schedulers, E052/E053 | training/schedulers | test_optimizer_migrations.py (scheduler tests) | OK |
| 9.4 block/overrides/entries, merge_blocks, E057, O3 | optimizer_plan.py compile_entry/merge_blocks | test_optimizer_plan.py (inherits_facade_keys, backend_replaces_backend_keys) | OK |
| 9.4.1 OPTIMIZER defaults (MIGRATE reset, VERIFY custom, STRICT_GLOBS true) | conf.py | test_optimizer_plan.py / test_conf_* | OK |
| 9.5 OptimizerPlan select/validate/stage/commit, E050/E051/E054 (O2, O6, O8, O9) | optimizer_plan.py | test_optimizer_plan.py (specificity, swap_is_atomic, all_null_is_refused, null thaw) | OK |
| 9.5 probe verify_entry E056 O002/O003 | optimizer_plan.py verify_entry | test_optimizer_plan.py | OK |
| 9.5 probe findings O001, O004, O005, O006 | verify_entry | tests/components/test_optimizer_probe_findings.py (new, parametrised + pass case) | GAP-fixed (untested before) |
| 9.6 migration carry/reset/fail, W014 (O7) | optimizer_migrations | test_optimizer_migration_* | OK |
| 9.7 trainer integration, O10 reset_state/rebind | trainer.py, BackendBase | test_custom_backend_is_reset, test_router_survives_device_move*, test_the_trainer_clears... | OK |
| 9.8 checkpoint/resume, swapped plan wins, E055 | OptimizerPlan.load_state_dict | test_a_swapped_plan_is_checkpointed..., test_a_checkpoint_round_trip_with_overrides... | OK |
| 9.9 config keys, optimizer_spec none->null | optimizer.py optimizer_spec | test_optimizer_plan.py | OK |
| 9.10 codes E048-E057, W011, W014, W015 in code table | components/codes.py | test_code_table.py; per-code tests all present | OK |

## Sections 10-11 Weight update, peer strategies

| item (doc section) | code location | test | status |
|---|---|---|---|
| 10.2 A1 default identity (weights) | `coopetition/weights.py` `build_weight_update`, `BUILTIN_WEIGHT_TYPES`; guard skips built-ins under SCOPE custom (`weight_guard.py:126`) | `test_weight_plan.py::test_weights_default_identity`, `..._in_a_hierarchy` | OK |
| A2 legal weights (finite, simplex, floor/cap, 0 on ineligible/trimmed) | `WeightGuard.inspect/check` U001-U007 | `test_weight_guard.py` U001..U007, repair/fallback/enforce, property test | OK |
| A3 spec cannot set framework bounds | `ComponentRegistry.build` injected-key rule | `test_weight_plan.py::test_spec_cannot_set_floor` | OK |
| A4 one instance per (slot, group) | `WeightPlan.strategy_for` | `test_group_state_isolated` | OK |
| A5 deterministic selection, E061 tie | `weight_plan.pick_glob`, `WeightPlan.validate` | `test_override_specificity_*` (weight_plan, function_book, optimizer_plan) | OK |
| A6 atomic swap | `WeightPlan.set` | `test_swap_atomic` (parametrised per step), `..._boundary_and_adding_an_override` | OK |
| A7 rejected proposal costs nothing | `AdaptiveKappa.snapshot/restore` in `_weights_for` | `test_fallback_does_not_double_step`, `test_snapshot_restore_undoes_a_step_exactly` | OK |
| A8 restore exact or loud (W021/E066) | `WeightPlan.load_state_dict`, `checkpoint_v2` label fix | `test_resume_*` (weight_plan 416-461, 792), `test_a_stored_temperature_with_no_live_group_is_dropped...` | OK |
| A9 fail at build | `WeightPlan.validate`, `verify_weight_update` | `test_bad_spec_fails_at_build_*` | OK |
| 10.4 WeightGuard: modes repair/fallback/enforce, scopes, TOLERANCE (0,0.1), `CODES`, `Violation`, `GuardOutcome` | `coopetition/weight_guard.py` | `test_weight_guard.py` (incl. `CODES == U001..U007`) | OK |
| W017 logged on repair/replace/raise | `weight_guard.py:261-294` | none asserted the log | GAP-fixed: `test_w017_is_logged_when_an_output_is_repaired` |
| 10.6 probe `verify_weight_update` (E064/E067) | `coopetition/weight_probe.py` | `test_weight_probe.py` | OK |
| 10.3 `WeightTraits`, `GroupView`, `WeightStrategy`, `make_report`, `traits_of`, `WEIGHT_STRATEGIES` LegacyTable | `weights.py:69-604` | `test_weight_plan.py`, `test_legacy.py` | OK |
| 10.7 Aggregator `options/clone_with/pristine`, `build_aggregator`, `verify_aggregator` (E065) | `aggregation.py:77-493` | `tests/test_aggregation.py`, `test_weight_plan.py` | OK |
| 10.7 temperature `snapshot/restore/resize`, `load_state_dict` restores num_peers | `temperature.py:185,200` | `test_snapshot_restore_undoes_a_step_exactly`, `test_resize_retargets_the_entropy_and_load_restores_the_arity` | OK |
| 10.5 `WeightPlan`, `KappaPlan` (E062, W022) | `components/weight_plan.py`, `kappa_plan.py` | `test_weight_plan.py` | OK |
| 10.8 trainer sites: `swap_weight_update`, `remove_weight_override`, `_apply_pending_weights`, `_group_views`, `_validate_weight_plan`, state entry | `training/trainer.py:2423-2514` | `test_weight_plan.py` trainer tests | OK |
| 10.9 checkpoint (plan entry, W021) | `trainer._trainer_state`, `checkpoint_v2` | `test_resume_a_weight_plan_through_the_trainer` | OK |
| 10.10 TRAINER keys + defaults (WEIGHT_UPDATE_OVERRIDES, WEIGHT_GUARD, WEIGHT_OPTIONS) and E060 validation | `conf.py:60-62`, `_validate_weight_and_strategy_blocks` (named `_validate_weight_blocks` in doc; merged) | `test_conf_weights_strategies.py` | DEVIATION-justified (one validator for 10.10+11.11, recorded P3) |
| 10.10 torch-free mirrors equal live enumerations | `conf._WEIGHT_GUARD_MODES` etc. | `test_conf_weights_strategies.py::test_the_torch_free_mirrors_equal_the_live_enumerations`; doc names `tests/cli/test_conf_mirrors.py` | DEVIATION-justified (test lives in components dir) |
| 10.12 codes E060-E067, W021, W022, U001-U007 present in code table | `components/codes.py`, raising sites | `test_code_table.py`, per-code tests | OK |
| 10.12 behaviour changes 1-5 (reference_only reset, unknown key error, trainer honours settings, stale temperature dropped, raising strategy falls back W017) | weights.py, trainer, checkpoint_v2, weight_guard | `test_reference_only_is_no_longer_masked`, `test_reference_only_resets_to_the_reference...`, `test_u007_apply_raising_becomes_a_fallback`, stored-temperature test | OK |
| 10.11 worked example (stateful per-group) | `tests/components/p3_examples.py` | `test_weight_plan.py` / examples | OK |
| 11.2 T1 default identity of strategies | `strategy.py` `legacy_dispatch`, `StrategyPlan.run` | `test_strategy_default_identity*` | OK |
| T2 strategy cannot write (Y005, Y006) | `coordination.py` surface, `guards/strategy.py` | `test_strategy_cannot_write`, `test_guard_property_strategy_cannot_write` | OK |
| T3 eligibility only shrinks, excluded peer = 0 (U004) | `CoordinationContext.eligible_for`, `legal_report` | `test_the_ineligible_gets_zero_in_the_context_report`; named `test_ineligible_gets_zero` was MISSING | GAP-fixed: added `test_coordination.py::test_ineligible_gets_zero` (exact weight 0 + Y001/U004 on adding back) |
| T4 parents never after children (Y004) | `coordination._precheck_write` | `test_parent_after_child`, `test_y004_a_deep_ancestor_counts_too` | OK |
| T5 one write per group (Y003) | `_precheck_write` | `test_group_written_once` | OK |
| T6 void group never blended (Y002) | `CoordinationContext.aggregate_group` | `test_y002_a_void_group_is_never_blended`, `test_test_void_group_not_mixed_nothing_is_written` (odd name; exact `test_void_group_not_mixed` absent but behaviour covered) | OK |
| T7 frozen means frozen (strict/legacy) | `freeze.py`, `trainer._train_epoch/_write` | `test_strict_freeze_*`, `test_legacy_freeze_keeps_the_old_behaviour_and_says_so` | OK |
| T8 reasons tracked | `FreezeSet.discard/clear_reason` | `test_freeze_reasons_are_independent`, coordination freeze test | OK |
| T9 someone can learn (E075) | `FreezeSet.check_learners` | `test_all_frozen_is_refused` | OK |
| T10 atomic strategy swap | `StrategyPlan.set` | `test_strategy_swap_atomic*` | OK |
| T11 restore exact or loud (W031, W036, E076) | `StrategyPlan/FreezeSet.load_state_dict`, `checkpoint_v2._topology` | `test_resume_*` (strategy_plan 619-688), `test_freeze_state_keeps_only_what_cannot_be_rederived`, `test_resume_with_frozen_status` | OK |
| T12 no dead settings | `StrategyPlan.from_settings`, trainer build | `test_no_dead_strategy_settings` | OK |
| 11.3 `StrategyTraits`, `GroupRef`, `TreeView`, `tree_view`, `coordinates_via_context`, `PeerStrategy.coordinate/attach/detach/validate`, `on_epoch_end` non-abstract | `coopetition/strategy.py` | `test_strategy.py` (traits, entry point, on_epoch_end) | OK |
| 11.4 `FreezeSet` (strict/legacy, bind/add/discard/reasons/skips_write) | `training/freeze.py` | `test_strategy.py`, `test_coordination.py` | OK |
| 11.5 `StrategyPlan` (select E071, E072, E074, MIGRATE reset/carry/fail, W033, STRICT_GLOBS) | `components/strategy_plan.py` | `test_strategy_plan.py` | OK |
| 11.6 `CoordinationContext` verbs, `CoordinationState`, `AuditEvent`, `CoordinationError`, `CoordinationBackend` | `training/coordination.py` | `test_coordination.py::test_the_context_reads`, verbs tests | OK |
| 11.7 `StrategyGuard` Y001-Y008, modes skip/fallback/enforce, scopes, `topology_digest`, W030 | `components/guards/strategy.py` | `test_coordination.py` (one test per Y code, `test_w030_...`) | OK |
| 11.8 `verify_peer_strategy` (E074, E077) | `coopetition/strategy_probe.py` | `test_the_probe_*` | OK |
| 11.9 trainer integration, W032 skip in `_write` | `trainer.py:2335-2355` | none asserted W032 (built-ins skip first) | GAP-fixed: `test_strategy.py::test_w032_a_group_with_every_member_frozen_is_skipped_and_logged` |
| 11.9 W034 legacy frozen_peer, W035 hierarchy groups unserved | trainer | `test_legacy_freeze_keeps_the_old_behaviour_and_says_so`, strategy tests | OK |
| 11.10 checkpoint `_topology` ignores transient statuses | `checkpoint_v2.py` | `test_resume_with_frozen_status`, `..._failed_status_and_a_real_topology_change` | OK |
| 11.11 TRAINER keys, defaults (STRATEGY_GUARD MODE skip, FREEZE_MODE legacy), enum validation | `conf.py:64-67, 1291-1295, 1364+` | `test_conf_weights_strategies.py`, `test_bad_strategy_settings_are_refused_at_load` | OK |
| 11.11 `_validate_strategy_blocks` name | merged into `_validate_weight_and_strategy_blocks` | as above | DEVIATION-justified |
| 11.12 behaviour changes 1-4 | strategy.py, trainer, checkpoint_v2 | `test_strategy_unknown_key_rejected`, `test_frozen_child_gets_zero_weight`, `test_hierarchical_update_reference_false`, `test_resume_with_frozen_status` | OK |
| 11.12 codes E070-E077, W030-W036, Y001-Y008 in code table | `codes.py` | `test_code_table.py` | OK |
| 11.12 files list (all new modules, `WeightGuardViolation`/`StrategyGuardViolation`) | verified present | `test_exceptions.py` | OK |
| 11.12 `tests/cli/test_conf_mirrors.py` | absent; mirror test in `tests/components/test_conf_weights_strategies.py` | see above | DEVIATION-justified |
| 11 worked example staggered_mix | `tests/components/p3_examples.py`, `contrib/components/examples/every_n.py` | `test_staggered_mix.py` | OK |

## Sections 12-13 Mutation and engine

| item (doc §) | code location | test | status |
|---|---|---|---|
| 12.3 PeerSpec/Op/OpKind/MutationPlan, canonical/digest/parse, 10 E080 shapes | mactf/training/mutation.py | tests/training/test_mutation_plan.py (rows 1-3) | OK |
| 12.4 MutationLimits (keys, defaults, from_settings, validate_block, W044, MIN_ACTIVE_ROOTS vs tree) | mutation.py; conf.TRAINER_DEFAULTS["MUTATION"], _validate_mutation_block | test_mutation_limits.py (rows 4-6), tests/cli/test_conf_mutation.py (row 59) | OK |
| 12.5.1 TopologySimulator, Simulation, GroupChange, E081-E084, E086, W040/W041, stop at first hard finding | mutation.py | test_mutation_simulator.py (rows 7-20) | OK (unknown @token = E080, documented P5) |
| 12.5.2 PeerTree adopt/restore/without; FreezeSet; structure_digest | coopetition/tree.py, freeze.py, mutation.structure_digest | test_mutation_retire/engine | DEVIATION-justified (digest is a module function, noted) |
| 12.5.3 optimizer staging (Staged.optimizer, plans E087) | training/mutation_engine.py | test_mutation_engine, atomicity | DEVIATION-justified (StagedTopology folded into Staged field) |
| 12.5.4 MutationEngine: gate E088, stage, 8-step commit, undo stack, rollback, post-commit audit E089, dry-run | mutation_engine.py | test_mutation_engine.py rows 21-30; test_mutation_atomicity.py (row 57, fault at every step) | OK |
| R9 failed rollback poisons the run | mutation_engine.py | test_a_failed_rollback_is_e089_not_a_half_run | OK |
| 12.3 factories, seeding (M4), init_from strict, shared trunk, tied params, rebuild W043, missing factory E085 | peer_factories.py, mutation_engine._build_peer/_check_layouts | test_mutation_peers.py rows 31-37 | OK |
| 12.6 retire/revive/purge, PathState registry + audit, tombstone | path_state.py, freeze.py | test_mutation_retire.py rows 38-43 | OK |
| M10 retired peer weight 0 (row 40) | freeze.reasons, trainer epoch report | test_a_retired_peer_has_weight_zero | DEVIATION-justified (omission from report instead of listed 0.0; legal_report floor) |
| 12.6 groups: controller/aggregator rebuild, W042 | mutation_engine._rebind_groups | test_mutation_groups.py rows 44-45 | OK |
| 12.7 trainer rebinding sites, _assemble_peer, mutate/request/apply_pending, _after_mutation, active view, journal epoch | training/trainer.py | test_mutation_engine rows 26-30 | OK |
| 12.8 HealthContext fields, LossTrendProbe slice, TreeGrowthProbe, prune_branch via engine, E082 recorded | watchdog.py | test_mutation_watchdog.py rows 46-50 | OK |
| 12.8.6 MutationRecord, MACTFLogger.mutation, on_tree_mutated, EarlyStopping reset, tree_mutated signal | logging.py, callbacks.py, signals.py, live.py | test_mutation_watchdog (early stopping, logged), test_engine_live | OK |
| 12.9 MutationState, journal chain/verify, plan_reconciliation, reconciler, E089/W045, resume in new process, rollback past purge | training/mutation_reconcile.py, checkpoint_v2.py | test_mutation_checkpoint.py rows 51-54 (real subprocess) | DEVIATION-justified (classes TopologyReconciler/ReconcilePlan vs doc MutationReconciler/Diff; noted) |
| 12.10 TRAINER.MUTATION config, --set and env forms | conf.py | tests/cli/test_conf_mutation.py, tests/test_conf_env.py (row 60) | OK (stale docstring fixed) |
| 12.11.5 equivalence (rows 55-56), defaults/I1 (row 58), properties | tests | test_mutation_equivalence/defaults/properties | OK |
| 12.11.6 codes E080-E089, W040-W045 present in code table and raised | components/codes.py + raise sites | grep: every code has src and test hits | OK |
| 12.11.7 file table (new modules, testing/tree.py, MutationError) | all present, import check | n/a | OK |
| 12.11.8 behaviour changes (WATCHDOG_PRUNE, default_probes +1, failures line) | watchdog.py, trainer.py | test_mutation_defaults, test_mutation_watchdog | OK |
| 13.3-13.4 TrainerEngine surface (T1), prepare, feature gate E092 (T9), parity helpers, PhaseContext, run loop, rollback, resume | orchestration/engine.py | test_engine_surface/parity/feature_gate/rollback/resume (rows 1-3,6-8,19-22,31-32) | OK |
| 13.5 dispatch: Engine, ENGINES, engine_name/for/factory/of, E090, ORCHESTRATOR keys, exports | orchestration/dispatch.py, conf.py, __init__.py | test_engine_dispatch/defaults (rows 4-5, 33) | OK |
| 13.6 driver factory, RunManager contextvars, tree(), handle.engine, Harness.engine | management/driver.py, orchestration/live.py, testing/harness.py | tests/cli/test_engine_driver.py, tests/orchestration/test_engine_live.py (rows 23-28) | OK |
| 13.6 redteam and API peers under trainer | contrib/api, redteam | test_engine_redteam.py, tests/contrib_api/test_engine_api.py (rows 29-30) | OK |
| 13.7 CumulativeSignal (T6), EnginePhaseBridge, T7 phases closed | engine.py | test_engine_signals.py, test_per_epoch_window_when_compat_is_off (rows 12-15) | OK (episode_started per node: D-13.10 noted) |
| 13.8 equivalence with Orchestrator, DB rows (T4), E091/W050/W051 | engine.py | test_engine_equivalence.py, test_engine_db.py, test_engine_resume.py (rows 9-11,16-18) | OK (grad-clip finding D-13.8 noted) |
| 13.9 matrix: 33 rows | n/a | all 33 names resolve to tests; 269 tests across both sections pass | OK |
| 13.10 codes E090-E092, W050-W051, files table | codes.py, engine.py | grep: all raised and tested | OK |
| T3 orchestrator.py untouched | sha pinned | tests/components/fixtures/orchestrator.sha256, test_orchestrator_files_are_untouched | OK |

## Sections 14-15 Persistence and configuration

| item (doc §) | code location | test | status |
|---|---|---|---|
| 14.3.1 entry fields (schema,kind,name,origin,version,spec_hash,source_digest,params,slot), ENTRY_SCHEMA=1 | components/fingerprint.py `entry_for`, `ENTRY_SCHEMA` | test_fingerprint.py::test_a_project_component_is_recorded | OK |
| 14.3.2 builtin-with-defaults not recorded; RECORD_FLAGS (record_in_checkpoint, unsafe) | fingerprint.py `records_in_checkpoint`, `RECORD_FLAGS` | test_a_builtin_with_default_params_is_not_recorded, test_an_unsafe_policy_is_recorded_with_its_flag | OK |
| 14.3.2 manifest sorted/grouped, deterministic | `component_manifest` | test_manifest_is_sorted_and_grouped_by_kind | OK |
| 14.3.3 diff (present change for missing slot; params+spec_hash together) | `diff`, `render` | test_diff_reports_one_present_change_for_a_missing_slot, test_diff_reports_params_and_spec_hash_together | OK |
| 14.4 three write sites (periodic _observe, state_dict, watchdog snapshot), engine key written once, identity_extra can't replace framework key | trainer.py / checkpoint identity_extra | test_persistence_identity.py rows 8-11 + test_identity_extra_cannot_replace_a_framework_key | OK |
| 14.4.1 `HubPolicy.strict_restore` + `policy_from_settings` (15.1.3) | components/registry.py, plugins.py:65-76 | NEW test_conf_components.py::test_strict_restore_reaches_the_hub_policy (row 12 had no direct test) | GAP-fixed |
| 14.5.1 status table ok/builtin/mismatch/orphan/missing/malformed/unsupported/absent -> E093/E094/E095/E096/W052/W053, strict vs permissive | components/restore.py `ComponentRestorer`, `_check_entry`, `extract_manifest` | test_restore_components.py rows 12-21 (all names present) | OK |
| 14.5.3/4 verification before any state touched; poisoned payload leaves trainer untouched | trainer.py `_restore` (plan before write), restore.py `RestorePlan` | test_persistence_resume.py::test_nothing_is_restored_after_a_fatal_finding, test_a_poisoned_payload_leaves_the_trainer_untouched | OK |
| 14.5 resume reports notes (ResumeReport / TrainResult.component_notes) | training/protocol.py:129, trainer.py:3179 | test_persistence_resume.py (component_notes asserts); matrix name test_persistence_engine::test_resume_reports_component_notes absent, covered under other name | OK (renamed) |
| 14.6 migration 4 (nullable components_json), record_components/components(), get_run decode, empty->NULL, idempotent, missing run refused | db registry migration 4 | tests/db/test_db_migration.py (rows 22-27 names present) | OK |
| 14.6/14.7.2 v3 DB refuses a new run until migrate | management/base guard | tests/cli/test_cmd_migrate_v4.py::test_a_version_3_database_refuses_a_new_run_until_migrated (row 28, longer name) | OK (renamed) |
| 14.6.4 engine records manifest at start; swap updates row at next save (rows 29,30) | trainer record_components call | test_persistence_identity.py::test_the_run_row_equals_the_checkpoint_key_and_follows_a_swap; test_engine_db.py::test_a_default_run_records_no_components_and_an_extension_does | OK (no test_persistence_engine.py file; merged) |
| 14.7.3 / row 32 Orchestrator files untouched (T3, v1 has no identity) | orchestrator not edited | test_persistence_resume.py::test_orchestrator_files_are_untouched | OK |
| 14.7.1 file/compat table (v1, v2 pre-doc -> W053, retired peer orphans W052) | restore.py | test_a_legacy_file_is_one_w053*, test_an_orphan_entry_is_w052_and_the_resume_goes_on | OK |
| 14.8 identity bytes/hash unchanged for default run; checkpoint bytes | identity | test_a_default_run_writes_the_same_identity_hash, test_checkpoint_bytes_unchanged_when_all_builtin, test_default_identity | OK |
| 14.9 every kind round-trips | persistence kinds | test_persistence_kinds.py, test_every_kind_round_trips | OK |
| 14.11 codes E093-E096, W052, W053 in code table | exceptions/code table | test_code_table.py::test_every_persistence_code_the_restorer_emits_is_in_the_table (test_known.py:189) | OK |
| 15.1.1 COMPONENTS_DEFAULTS (10 keys, values identical to doc) | conf.py:230 | test_conf_components::test_defaults_block_has_every_documented_key, test_defaults_are_inert, test_the_block_is_a_copy_of_the_constant | OK |
| 15.1.1 COMPLIANCE_DEFAULTS (1.0/enforce/10000.0/1e-9/True), RL_DEFAULTS (error/True/custom), DEFAULTS copies | conf.py | test_conf_blocks.py | OK |
| 15.1.2 validators E013 (non-bool, unknown key, non-mapping), E014 module path, E015 PLUGINS, E016 lock pattern, E017 allowlist | conf.py `_validate_components` | test_conf_components rows 4-9 (unknown-key test is `..._e013_and_lists_the_accepted_set`) | OK |
| 15.1.2 API_ALLOW_IMPORT alias W006 / conflict E013 | conf.py `_canonical_components`, plugins.fold_block | test_the_alias_is_w006, test_the_alias_conflict_is_e013 | OK |
| 15.2 COMPLIANCE wired (validated, unknown keys refused) | conf.py `_validate_compliance` | tests/components/test_scoring_hardening.py, test_conf_blocks | OK |
| 15.3.1 CONF_LAYOUT: components.yaml second owning COMPONENTS+COMPLIANCE; rl.yaml directly after loss.yaml; 19 files | conf_files.py (verified order: conf, mactf, components, peers, trainer, optimizer, loss, rl, scorer, ...) | tests/test_conf_files.py rows 35-37 (count test is `..._nineteen_...`: doc 17->19 change) | OK |
| 15.3.2 components.yaml.tpl byte-equal to doc | templates/conf/components.yaml.tpl | diff vs doc = identical; test_conf_blocks template drift | OK |
| 15.3.3 rl.yaml.tpl | templates/conf/rl.yaml.tpl | OVERRIDES block commented instead of `OVERRIDES: {}`; RL_DEFAULTS has no OVERRIDES key (8.5.2 three keys), so a live line would fail the leaf==default drift test | DEVIATION-justified |
| 15.3.4 startproject count / shipped templates == layout / wheel ships .tpl | test_startproject_conf.py (nineteen, every_packaged_template_is_in_the_layout), test_packaging.py | OK |
| 15.4 known.py: torch-free, BUILTIN_NAMES per kind, LOCKABLE_KINDS, may_be_extended, component_ref_problem | components/known.py | tests/components/test_known.py (table==live registries, every kind covered or None, plugin name accepted, MODULES defer, entry point defer, mapping spec shape) ; matrix names live in components/test_known.py not test_conf_known.py | OK (relocated; no test_conf_mirrors needed) |
| 15.4.2 lock-pattern kinds (defect in doc: BUILTIN_NAMES|slot) | known.LOCKABLE_KINDS | documented in notes P0; test_every_documented_lock_pattern_is_accepted_for_every_kind | DEVIATION-justified |
| 15.4.3 TRAINER name checks keep message and case folding | conf.py | test_the_trainer_names_keep_their_message_and_case_folding | OK |
| 15.5 ensure_components_loaded idempotent, reload on changed block, called from configure; reset_for_tests clears globals | plugins.py, conf.configure | test_ensure_components_loaded_is_idempotent_across_configure, test_a_changed_block_reloads, test_a_bad_block_never_reaches_the_hub | OK |
| 15.6 MACTF_A__B__C: 2-4 segments, coercion, E018 (mapping flat var, unknown block w/ suggestion, scalar container, fifth segment, empty segment), W007 on creation, flat scalar unchanged | conf.py `_apply_env` | tests/test_conf_env.py rows 20-28 (all present) | OK |
| 15.7 --set promotion (_promotion/_SPEC_LEAVES; type/import/params; refuse int/bool/float/list) | conf_overrides.py | tests/test_conf_overrides.py rows 29-34 | OK |
| 15.8.1 key reference (10 keys + alias) | conf.py | test_conf_components | OK |
| 15.8.2 COMPLIANCE key ranges (MIN_FLOOR_LAMBDA>0, TOLERANCE in [0,1e-3], GUARD_MODE) | conf.py | test_scoring_hardening / test_conf_blocks | OK |
| 15.8.3 extended TRAINER/OPTIMIZER/SCORING/COOPETITION/ORCHESTRATOR keys | conf.py | notes P3 (KAPPA_OVERRIDES not in defaults, E062 at load) | DEVIATION-justified |
| 15.10 row 44 settings load imports no torch | tests/cli/test_import_weight.py (probe `conf.configure()`; `import mactf.components, known`) | named test absent; probes cover it | OK (renamed) |
| 15.10 row 45 help snapshot unchanged | tests/cli/test_help_snapshot.py | test_help_json_is_stable etc. | OK (renamed) |
| 15.10 rows 13-14,19 mirrors (tests/test_conf_mirrors.py) | known.py vs registries | tests/components/test_known.py::test_the_table_matches_the_live_registries | OK |

## Sections 16-18 CLI, API, checks, examples

| item (doc section) | code location | test | status |
|---|---|---|---|
| 16.1.1 surfaces (components group, runs components, peers mutate, run/serve --component-module, completers) | commands/components.py, runs.py, peers.py, run.py, contrib/api/.../serve.py, completion/completers.py | golden help.json/overview.txt; tests/cli/test_help_snapshot.py | OK |
| 16.1.2 `components list` (--kind/--origin/--locked, alias ls, columns, unknown kind exit 3) | commands/components.py sub_list | test_list_inspect_and_validate_end_to_end, test_component_exit_codes | OK |
| 16.1.2 `inspect` (describe, yaml, unknown name -> 3 with accepted set) | sub_inspect | same | OK |
| 16.1.2 `validate` (--name/--import/--param JSON/--spec-file '-'/--probe, E004 finding exit 1, W004 advisory, usage exit 2) | sub_validate, _raw_spec, _advisories | test_list_inspect..., test_validate_an_unsafe_policy_warns_and_probe_finds_a_broken_function | OK |
| 16.1.2 `scaffold` (kinds, identifier check, path-escape check, --dry-run, conflict exit 4, --force) | sub_scaffold | test_component_exit_codes, test_every_verb_dry_runs, test_scaffold_through_the_cli_then_load_it | OK |
| 16.1.2 `diff` (A/B, live, --structural, exit 0/1) | sub_diff | test_runs_components_and_diff_read_back_a_real_run | OK |
| 16.1.2 contract: `name` added to KNOWN_SETS | tests/cli/test_command_contract.py has `slot` only | contract test | DEVIATION-justified: scaffold's `name` is a new module name, not a known set; the doc gives it no completer, so adding `name` would fail the contract on the doc's own scaffold |
| 16.1.2 requires_system_checks per verb | GroupCommand has one bool per command | n/a | DEVIATION-justified: `validate` validates the spec directly; list/inspect never run checks |
| 16.1.3 scaffolds (render_scaffold, 8 kinds, no placeholder survives, imports) | components/scaffold.py | test_scaffold_renders_and_imports (all kinds) | OK |
| 16.1.4 `runs components show/set` (--spec, --now, not-live 4, unknown run 3, E010 3) | runs.py sub_components, orchestration/live.py | test_runs_components_set_changes_a_live_run, test_component_exit_codes | OK |
| 16.1.4 `peers mutate` (simulate, exit 1 on hard finding, --apply needs --run -> 2, not live -> 4) | peers.py | test_peers_mutate_*, test_component_exit_codes | OK |
| 16.1.5 `run --component-module` / `serve --component-module` (this run only, no settings write, allow-list) | run.py, serve.py, management/runsupport.py | test_runs_components_and_diff_read_back_a_real_run (cm1, bad module refused) | OK (serve flag present in golden; exercised via shared helper) |
| 16.1.6 completers component_kind / component_name / slot_path (static, no torch, degrade) | completion/completers.py | NEW test_the_component_completers_answer_without_importing_torch | GAP-fixed (no test existed) |
| 16.2.1 exit-code table, every row | management/exitcodes.py classify | test_component_exit_codes_by_exception (10 rows), test_component_exit_codes | OK |
| 16.2.2 classify() branches placed before ImproperlyConfigured | exitcodes.py | same | OK |
| 16.2.3 goldens / docs/cli.md / docs/api.md regenerated | tests/cli/golden, docs/cli.md, docs/api.md | test_help_snapshot, test_docs_generated, test_openapi_in_sync | OK |
| 16.3.1 six routes, scopes (read / run:write), literal-before-parameterised | contrib/api/urls.py, views/components.py | test_literal_routes_precede_parameterised | OK |
| 16.3.1 operation_ids {get: run-components-get, patch: run-components-patch} | views/components.py uses route name for both | test_openapi_in_sync | DEVIATION-justified: in this tree `operation_ids` maps a method to the route *name* it serves (runs.py does the same); doc's ids are not route names |
| 16.3.2 ComponentChoiceField + Serializer refresh, ComponentKindField, Spec/Validate/Patch/Mutation serializers, TrainerSpec dynamic choices | contrib/api/serializers.py, http/serializers.py:656 | test_choice_field_refreshes, test_trainer_spec_names_come_from_the_live_registry, test_old_trainer_bodies_still_validate | OK |
| 16.3.3 views: list/filters/paginate, detail 404 with known, validate untrusted + probe, trainer-validate all problems, run components GET/PATCH 202 idempotent, mutations dry_run/queue | views/components.py | tests/contrib_api/test_components_api.py (20 tests) | OK |
| 16.3.4 _translate + status table (200/404/422/409/202/403) | views/base.py _translate | test_status_table, test_patch_refusals_carry_their_codes | OK |
| 16.3.4/17.1.3 HTTP status per exception class: UnsafePolicyDenied 422, ComponentMigrationError 409, ComponentBuildError 500 with code | views/base.py _translate (these three branches were missing: E009 fell to 409, E048/E008 untranslated) | NEW test_every_exception_has_a_code_and_a_status | GAP-fixed |
| 16.3.5 client ROUTES (6 entries), SAMPLE gains kind | client/routes.py, tests/client/test_structure.py | test_route_table_matches_reverse, test_routes_cover_every_component_view | OK |
| 16.4.1 specs.component/components/mutation, trainer accepts spec dict | client/specs.py | test_specs_are_plain_dicts_the_server_accepts | OK |
| 16.4.1 Client methods (doc: components(), validate_component(), patch_components(), mutate()) | client/client.py ComponentsAPI: client.components.{list,get,validate,validate_trainer,show,patch,mutate} | NEW test_the_components_namespace_reaches_every_route | DEVIATION-justified (namespaced like every other client area) + GAP-fixed (methods were untested) |
| 16.4.3 additive compatibility (old bodies, ExitCode unchanged, GET runs/{id} unchanged) | various | test_old_trainer_bodies_still_validate, test_help_json_schema_is_additive_only | OK |
| 17.1.2 exception classes + codes (ComponentError family, Compliance/Step/Ledger, Weight/Strategy guard, MutationError) | mactf/exceptions.py | test_exceptions.py (code attr + message prefix, prefix-carrying sibling codes) | OK |
| 17.1.3 class -> base -> code -> exit -> HTTP table | exceptions.py, exitcodes.py, views/base.py | test_component_exit_codes_by_exception + NEW HTTP test | OK (HTTP column GAP-fixed above) |
| 17.2.2 code table: 127 rows, severity/section identical to the doc rows | components/codes.py CODE_TABLE | test_code_table_is_complete, test_the_table_is_well_formed, test_the_table_matches_the_live_registries | OK (compared programmatically: 0 missing, 0 extra, 0 severity/section diffs; every code is also named in at least one test) |
| 17.2.3 families S/O/U/Y | codes.py FAMILIES | test_the_table_is_well_formed | OK |
| 17.2.4 rule for adding a code | test_code_table_is_complete both directions | same | OK |
| 17.3.1 app (apps.py, systems.py, requires, tags), default INSTALLED_APPS, startproject | contrib/components/*, conf.py:295, settings_default.py:31 | test_the_checks_are_registered_under_the_app_and_tagged, test_a_clean_project_has_no_messages; `mactf check --tag components` on a fresh startproject: 0 errors/0 warnings | OK |
| 17.3.2 seven checks, ids `mactf.components.<code>`, codes E007/E002, E001..E012, W001, W002, W003/W013, W004/E009, E090/E092 | contrib/components/checks.py | test_system_checks.py (16 tests incl. test_each_check_has_its_code_and_hint) | OK (check_configured_specs also reports E008 for an unimportable module: superset of the doc, sound) |
| 17.3.3 settings validation vs checks | conf.validate_settings vs checks | test_conf_components | OK |
| 17.4 files and behaviour changes | listed files exist | n/a | OK |
| 18.1-18.8 modules, YAML fragments, plan files, run + misbehaviour per example | mactf/contrib/components/examples/* | tests/components/test_examples.py (fragments validate, module registers only on import, run, misbehaviours, test_every_worked_example_runs) | OK |
| 18.9 codes per example provoked | n/a | misbehaviour tests plus the per-phase tests; several 18.9 codes (e.g. E051-E057, E084-E089, S001-S009) are asserted in the P2/P5 files, not test_examples.py | OK (covered elsewhere) |
| test hygiene: tests/cli/test_components_cli.py autouse fixture snapshotted the hub before trainer policies were imported, so after the first test the trainer_policy kind was emptied and later tests failed when the file ran alone (order-dependent) | tests/cli/test_components_cli.py restore_process_registry | whole file now passes alone (30 tests) | GAP-fixed (now calls known.import_builtins() first) |

## Cross-cutting checks (section 19, 20, 21 and environment)

| item | check | status |
|---|---|---|
| 19.3 / 20.2 named tests | all 351 `test_*` names in the doc resolved: exact, as a file name, or mapped in the section tables above; exceptions: `test_trimmed_mean_rollup.py` (bodies live in `test_rollup_guard.py`), `test_persistence_engine.py` rows 29-31 (in `test_persistence_identity.py`/`_resume.py`), `test_conf_mirrors` (in `test_conf_weights_strategies.py`), nineteen vs seventeen layout names (doc says re-read the count) | OK / DEVIATION-justified |
| 20.4 `tests/components/test_overhead.py` | missing; added (reports the with/without ratio, asserts zero builds/registrations, torch-free import) | GAP-fixed |
| import weight | `import mactf.components` leaves `torch` out of `sys.modules` (subprocess probe + `test_import_weight`) | OK |
| Python 3.10 | no 3.11+ stdlib/syntax: grep for tomllib, Self, StrEnum, ExceptionGroup, `except*`, datetime.UTC, TaskGroup, NotRequired, PEP 695/701 syntax (tokenizer scan for nested-quote f-strings) clean; `ast.parse(feature_version=(3,10))` clean over mactf/ and tests/ (11 pre-existing BOM files only); no python3.10/3.11 interpreter installed, tests ran under 3.14 | OK (not executed under 3.10) |
| packaging | `pyproject.toml` package-data covers `mactf.templates.conf *.tpl` (components/rl templates) and `mactf.contrib.components.examples` conf/plans yaml; `MANIFEST.in` has `recursive-include mactf *.tpl` | OK |
| CHANGELOG / README | neither mentioned the feature; added a CHANGELOG entry, a README section and a `components` row in the command table | GAP-fixed |
| 21.2 open questions Q1-Q8 | assumed answers hold: ENGINE default stays `orchestrator`, `E092` refuses distributed, `ALLOW_UNSAFE_POLICY` is not settable over the API (no reference in `mactf/contrib/api`) | OK |
