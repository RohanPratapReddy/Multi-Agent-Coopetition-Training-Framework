"""The consolidated code table of document 17 section 17.2 as data.

Every ``E``, ``W``, ``C`` and ``F`` code the framework raises or logs is a row here, and
``tests/components/test_code_table.py`` scans the source tree in both directions: a code
used in ``mactf/`` but missing from :data:`CODE_TABLE` fails, and so does a row nothing
raises (unless it is listed in :data:`RESERVED` with the reason). The rule for adding a
code (17.2.4): add the row, add a test that provokes it, say whether ``mactf check``
reports it.

The ``S``, ``O``, ``U`` and ``Y`` families label items *inside* a finding and are not
rows (:data:`FAMILIES`).
"""

from __future__ import annotations

from typing import NamedTuple


class CodeRow(NamedTuple):
    code: str
    severity: str     # error | warning | violation | finding
    section: str      # where the behaviour is specified
    raised_by: str    # the one place that raises it
    meaning: str


#: ``code -> row``, in the order of the table.
CODE_TABLE: dict[str, CodeRow] = {row.code: row for row in (
    CodeRow('E001', 'error', '3.4', 'ComponentRegistry.get, ComponentResolver, build_*', 'A component of this kind and name is not registered (ComponentMissing); the message lists the accepted names'),
    CodeRow('E002', 'error', '3.4', 'ComponentRegistry.register, ComponentResolver._import', 'The name is taken by another factory and replace was not given (ComponentCollision)'),
    CodeRow('E003', 'error', '4.3', 'ComponentRegistry, SlotTable.swap/request, OptimizerPlan', 'The component, slot or boundary is locked: COMPONENTS.LOCKED, an origin of lower rank, a built-in without ALLOW_BUILTIN_OVERRIDE, a slot whose mutability forbids this boundary (ComponentLocked)'),
    CodeRow('E004', 'error', '5.2', 'Component.check_params, ComponentResolver.validate', 'A spec key the component does not accept, or a value outside its declared range or type (ComponentParamError); the message lists the accepted keys'),
    CodeRow('E005', 'error', '5.1', 'ComponentSpec.parse', 'A spec that is not a name or a mapping, has neither type nor import, or has a malformed import (ComponentSpecError)'),
    CodeRow('E006', 'error', '5.4', 'ComponentResolver._import', 'An import spec on a surface that is not trusted, or a module outside IMPORT_ALLOWLIST (ComponentImportDenied)'),
    CodeRow('E007', 'error', '3.7', 'ComponentRegistry.register, ComponentResolver.adopt', "A factory or an instance does not meet its kind's contract, or asks for a looser mutability than the kind allows (ComponentContractError)"),
    CodeRow('E008', 'error', '3.4', 'ComponentRegistry.build, build_scheduler, load_components', 'A factory raised while building, or a plugin module raised at import (ComponentBuildError, with the cause chained)'),
    CodeRow('E009', 'error', '7.7', 'MACTFTrainer._attach_policy', 'A policy declared unsafe = True and TRAINER.ALLOW_UNSAFE_POLICY is not set (UnsafePolicyDenied); always raised at build, never mid-run'),
    CodeRow('E010', 'error', '4.4', 'SlotTable.get, SlotTable.request', 'A slot path that does not exist (ComponentMissing with kind slot)'),
    CodeRow('E011', 'error', '6.2', 'ComponentSpec.parse, ComponentRegistry.build', 'A spec names a parameter the framework injects (policy, registries, ...); it would be silently overridden, so it is refused'),
    CodeRow('E012', 'error', '5.1', 'ComponentSpec.parse', 'A spec gives parameters both flat and under params'),
    CodeRow('E013', 'error', '15.11', '_validate_components, _canonical_components', 'The COMPONENTS block is not a mapping, has an unknown key, gives a flag a non-bool, or sets an alias and its canonical key to different values'),
    CodeRow('E014', 'error', '15.11', '_validate_components', 'A COMPONENTS.MODULES entry is not a dotted module name (a path, an empty segment, a trailing dot)'),
    CodeRow('E015', 'error', '15.11', '_validate_components', 'COMPONENTS.PLUGINS is neither auto, off, nor a list of non-empty names'),
    CodeRow('E016', 'error', '15.11', '_validate_components via lock_pattern_problem', 'A COMPONENTS.LOCKED entry is not kind:name or slot:<path>, or names an unknown kind'),
    CodeRow('E017', 'error', '15.11', '_validate_components', 'A COMPONENTS.IMPORT_ALLOWLIST entry is not a pattern, or is a blanket */**'),
    CodeRow('E018', 'error', '15.11', '_apply_env', 'An environment override that cannot be applied: an empty segment, an unknown block or key, a mapping-valued flat variable, a scalar treated as a mapping, or more than four segments'),
    CodeRow('E040', 'error', '8.10', 'FunctionBook.validate, conf._validate_rl', 'Two sources name the same default (LOSS and RL.LOSS; RL.SYNTHESIS and synthesis=; RL.ROLLUP and hierarchical_scorer=)'),
    CodeRow('E041', 'error', '8.10', 'FunctionBook.validate', 'A family is configured that no active consumer reads (W010 when RL.UNCONSUMED: warn)'),
    CodeRow('E042', 'error', '8.10', 'conf._validate_rl, FunctionBook.install_override, swap_override', 'A per-node override for a global-only family (rollup)'),
    CodeRow('E043', 'error', '8.10', 'FunctionBook._select', 'Two overrides of equal specificity match the same node'),
    CodeRow('E044', 'error', '8.10', 'FunctionBook.validate', 'An override glob matches no node at build (W011 after a tree mutation or when STRICT_GLOBS: false)'),
    CodeRow('E045', 'error', '8.10', 'FunctionBook.validate', 'RL.ROLLUP or RL.SYNTHESIS is set on a run with no hierarchy'),
    CodeRow('E046', 'error', '8.10', 'build_synthesis', 'A custom rl_function.synthesis name equals a node-level built-in name'),
    CodeRow('E047', 'error', '8.10', 'verify_function', 'A function failed its contract probe (F001..F012 in the message)'),
    CodeRow('E048', 'error', '8.10', 'migrations.py', 'A migration refused or failed (ComponentMigrationError)'),
    CodeRow('E049', 'error', '8.10', 'FunctionBook.load_state_dict', 'Strict restore: saved function state does not match the configured spec'),
    CodeRow('E050', 'error', '9.10', 'OptimizerPlan.validate', 'An OPTIMIZER.OVERRIDES glob matches no node at build (W011 after a mutation or with STRICT_GLOBS: false)'),
    CodeRow('E051', 'error', '9.10', 'OptimizerPlan.select', 'Two overrides of equal specificity match the same node'),
    CodeRow('E052', 'error', '9.10', 'compile_entry, attach_scheduler, build_scheduler', 'A scheduler cannot be used: the backend has no torch optimizer, or the scheduler needs a metric (ReduceLROnPlateau)'),
    CodeRow('E053', 'error', '9.10', '_compile_scheduler', "A scheduler's parameters are invalid (unknown key, warmup_steps >= total_steps, total_steps < 1)"),
    CodeRow('E054', 'error', '9.10', 'OptimizerPlan.validate, stage, request', 'Every node would be frozen, or the default entry would be removed'),
    CodeRow('E055', 'error', '9.10', 'MACTFOptimizer._reconcile, OptimizerPlan.load_state_dict', 'Strict restore: saved optimizer state or plan entry does not match the configured spec'),
    CodeRow('E056', 'error', '9.10', 'verify_entry', 'An optimizer failed its probe (O001..O006 in the message)'),
    CodeRow('E057', 'error', '9.10', 'compile_entry', 'An entry is inconsistent: an unaccepted key, a key given flat and under PARAMS, scipy with accumulation, a bad MIGRATE'),
    CodeRow('E060', 'error', '10.5', 'WeightGuardConfig, WeightPlan.validate, conf.py', 'A weight-update configuration problem: an unknown or malformed WEIGHT_GUARD or WEIGHT_OPTIONS key, an override block that is not a mapping, a bad glob, an override that matches no group, or two aggregation groups with one label'),
    CodeRow('E061', 'error', '10.5', 'pick_glob via WeightPlan.select', 'Two WEIGHT_UPDATE_OVERRIDES globs of equal specificity match one group'),
    CodeRow('E062', 'error', '10.5', 'pick_glob via KappaPlan', 'Two temperature-override globs of equal specificity match one group'),
    CodeRow('E063', 'error', '10.4', 'WeightGuard (WeightGuardViolation)', "A weight strategy's output broke the weight contract (U001-U007) under MODE=enforce"),
    CodeRow('E064', 'error', '10.6', 'verify_aggregator, WeightPlan.validate', "A strategy's validate rejected the tree, or its output probe failed; at build and after a tree mutation"),
    CodeRow('E065', 'error', '10.7', 'WeightStrategy.clone_with check', 'clone_with returned a different class or different options, so a per-group copy would not be the strategy that was configured'),
    CodeRow('E066', 'error', '10.9', 'WeightPlan.load_state_dict', 'The saved plan differs from the configured one under COMPONENTS.STRICT_RESTORE'),
    CodeRow('E067', 'error', '10.6', 'verify_aggregator', "A strategy's state_dict is not JSON, has no load_state_dict, or does not round-trip"),
    CodeRow('E070', 'error', '11.12', 'StrategyPlan, FreezeSet, StrategyGuard, conf.py', 'A strategy-configuration problem: unknown PEER_STRATEGY_OPTIONS or STRATEGY_GUARD key, bad MODE / SCOPE / MIGRATE / FREEZE_MODE, an override that matches no group (under STRICT_GLOBS), an invalid glob, colliding group labels'),
    CodeRow('E071', 'error', '11.12', 'StrategyPlan.select', 'Two override globs match one group with equal specificity'),
    CodeRow('E072', 'error', '11.12', 'StrategyPlan', 'A legacy strategy (no coordinate) used where only a context-driven one can work: as an override, or beside overrides'),
    CodeRow('E073', 'error', '11.12', 'StrategyGuard', 'StrategyGuardViolation: a hard rule (Y005, Y006), or any violation under MODE=enforce'),
    CodeRow('E074', 'error', '11.12', 'StrategyPlan, probe', "A strategy's validate rejected the tree; it defines no entry point; the probe found a violation; MIGRATE=fail refused a swap that would drop state; bad mode argument"),
    CodeRow('E075', 'error', '11.12', 'FreezeSet', 'A malformed freeze reason; a freeze of a path that is not a node; no node would train'),
    CodeRow('E076', 'error', '11.12', 'StrategyPlan.load_state_dict', 'The saved plan differs from the configured one under COMPONENTS.STRICT_RESTORE'),
    CodeRow('E077', 'error', '11.12', 'probe', 'state_dict is not JSON, has no load_state_dict, or does not round-trip'),
    CodeRow('E080', 'error', '12.11', 'PeerSpec.parse, Op.parse, MutationPlan.parse, MutationLimits.validate_block, MutationReconciler.reconcile', 'A plan or a limit block is not valid data: unknown key, bad verb, bad path, bad role, non-JSON value, a plan that sets both init_from and rebuild, a schedule entry without AT_EPOCH, or a checkpoint whose journal holds user mutations while ENABLED is false'),
    CodeRow('E081', 'error', '12.11', 'TopologySimulator._add, _modify, _retire, _revive, _purge', 'A structural rule: no such node, the node exists, a tombstoned path, a missing or retired parent, the node is retired (revive first), a revive whose parent is still retired'),
    CodeRow('E082', 'error', '12.11', 'TopologySimulator._retire, _purge, _finalize', 'A protection or a size rule: PROTECTED glob, ALLOW_PURGE false, active children without recursive, active nodes without force, fewer than MIN_ACTIVE_ROOTS roots, every child of a branch retired'),
    CodeRow('E083', 'error', '12.11', 'TopologySimulator._check_reference, MutationEngine._build_peer', 'A reference (like, init_from) names an unknown node, the node itself, a purged node, or a node with no peer to read from'),
    CodeRow('E084', 'error', '12.11', 'TopologySimulator.run, _add', 'A size limit: more operations than MAX_OPS_PER_PLAN, or a node that would exceed max_nodes (from TreeTooLargeError)'),
    CodeRow('E085', 'error', '12.11', 'MutationEngine._build_peer, _copy_weights, _check_layouts, _check_plans', 'The new topology cannot exist: the factory returned something that is not an nn.Module, @like would unshare a trunk, a parameter count or layout differs from init_from, siblings in one group disagree on layout (M6), tied parameters would appear (M7), or the factory is no longer registered (ComponentMissing)'),
    CodeRow('E086', 'error', '12.11', 'TopologySimulator._finalize', "The group's new active count admits no legal weight vector under WEIGHT_FLOOR/WEIGHT_CAP (feasible_bounds) (M8)"),
    CodeRow('E087', 'error', '12.11', 'MutationEngine._check_plans', 'A plan refused the topology the mutation would create: the optimizer plan, the strategy plan, the weight plan or the kappa plan raised ImproperlyConfigured for it'),
    CodeRow('E088', 'error', '12.11', 'MutationEngine._gate, MutationReconciler.reconcile', "The plan cannot run now: the trainer is not built, mutation is off, the phase is epoch, a blocker is present (DISTILLATION with a policy that cannot follow a tree), the epoch's budget is spent, or a reconciliation is blocked by the same list"),
    CodeRow('E089', 'error', '12.11', 'MutationEngine._commit, apply, PathState.audit, MutationState.parse, MutationReconciler.reconcile', "Atomicity or integrity: the post-commit audit found stale state, the rollback did not complete (rolled_back=False: discard the trainer), a checkpoint's mutation state or tree section does not parse, the journal does not reproduce its tree under STRICT_REPLAY, or a reconciliation did not reach the stored digest"),
    CodeRow('E090', 'error', '13.10', 'dispatch.engine_name, conf._validate_blocks', 'ORCHESTRATOR.ENGINE is not one of ENGINES (including a non-string value that stringifies to something else)'),
    CodeRow('E091', 'error', '13.10', 'TrainerEngine._check_identity', 'The checkpoint was written by the other engine and ORCHESTRATOR.ALLOW_ENGINE_SWITCH is false'),
    CodeRow('E092', 'error', '13.10', 'TrainerEngine._refuse_unsupported', 'The configuration cannot be served by the trainer engine: mactf.distributed installed, or RunConfig.peers disagrees with a pre-built trainer'),
    CodeRow('E093', 'error', '14.11', 'ComponentRestorer._check', 'A checkpoint needs a component (name and kind) that this build has not registered: the providing module is not loaded'),
    CodeRow('E094', 'error', '14.11', 'ComponentRestorer._check', 'A live slot differs from the recorded entry and COMPONENTS.STRICT_RESTORE is on; the message lists every differing field'),
    CodeRow('E095', 'error', '14.11', 'ComponentRestorer._check via _check_entry', 'The components section, or one entry, is not interpretable: not a mapping, no schema/kind/name, or a non-integer schema'),
    CodeRow('E096', 'error', '14.11', '_check_entry', "An entry's schema is newer than ENTRY_SCHEMA"),
    CodeRow('W001', 'warning', '4.3', 'ComponentRegistry._check_replace', 'A built-in component was replaced under ALLOW_BUILTIN_OVERRIDE'),
    CodeRow('W002', 'warning', '2.2', 'check mactf.components.W002', 'TRAINER.PEER_STRATEGY or TRAINER.WEIGHT_UPDATE is set to a non-default value but the active engine never reads it (the Orchestrator ignores both)'),
    CodeRow('W003', 'warning', '6.4', 'ComplianceGuard, check mactf.components.W003', 'A run started with COMPLIANCE.GUARD_MODE: fallback; a plugin bug costs a substituted score instead of the run'),
    CodeRow('W004', 'warning', '7.7', 'mactf components validate, check mactf.components.W004', 'A policy with unsafe = True is configured; StepGuard does not wrap it'),
    CodeRow('W005', 'warning', '3.6', 'LegacyTable.__setitem__', 'A component was replaced through a write to a legacy dict (SCORER_TYPES, PEER_STRATEGIES, ...)'),
    CodeRow('W006', 'warning', '15.11', '_canonical_components', 'COMPONENTS.API_ALLOW_IMPORT and COMPONENTS.ALLOW_IMPORT are both set to the same value: one line can be deleted'),
    CodeRow('W007', 'warning', '15.11', '_apply_env', 'An environment override created a key no DEFAULTS block declares'),
    CodeRow('W010', 'warning', '8.10', 'FunctionBook.validate', 'An unconsumed family under RL.UNCONSUMED: warn'),
    CodeRow('W011', 'warning', '8.10', 'FunctionBook.validate', 'A glob that matches no node, after mutation or with STRICT_GLOBS: false'),
    CodeRow('W012', 'warning', '8.10', 'FunctionBook.load_state_dict', 'Saved function state was dropped because the slot or its spec changed'),
    CodeRow('W013', 'warning', '8.10', 'FunctionBook at build', 'RL.VERIFY: off: custom functions run without a contract probe'),
    CodeRow('W014', 'warning', '9.10', 'optimizer_migrations.carry', 'A carry dropped something it could not move (the scheduler position onto a scheduler-less entry; an open accumulation window when GRAD_ACCUMULATION changed)'),
    CodeRow('W015', 'warning', '9.10', 'MACTFOptimizer._reconcile, OptimizerPlan.load_state_dict', 'Optimizer state or a swapped entry in the checkpoint differs from the configured spec; says whether it was carried or dropped'),
    CodeRow('W016', 'warning', '9.5', 'OptimizerPlan.validate', 'A null optimizer entry selects the only node of a group, so it would never be updated'),
    CodeRow('W017', 'warning', '10.4', 'WeightGuard, build_weight_update', 'A weight strategy was rejected, repaired by projection onto the bounded simplex, or replaced by kappa_softmax (MODE repair or fallback)'),
    CodeRow('W021', 'warning', '10.9', 'WeightPlan.load_state_dict', 'State saved for an aggregation group does not fit the live strategy at that label; dropped'),
    CodeRow('W022', 'warning', '10.5', 'WeightPlan.validate', 'An override matches no aggregation group after a tree mutation, or with STRICT_GLOBS: false'),
    CodeRow('W030', 'warning', '11.12', 'StrategyGuard', 'A soft violation was handled (skip or fallback)'),
    CodeRow('W031', 'warning', '11.12', 'StrategyPlan', 'Saved state or plan did not match the live one; the saved plan won or the state was dropped'),
    CodeRow('W032', 'warning', '11.12', 'MACTFTrainer._write', 'A group was skipped because every member is frozen'),
    CodeRow('W033', 'warning', '11.12', 'StrategyPlan.validate', 'An override matches no group (when STRICT_GLOBS is off, or after a mutation)'),
    CodeRow('W034', 'warning', '11.12', 'trainer, FreezeSet', 'frozen_peer under legacy (frozen peers still train and receive write-back), or the saved FREEZE_MODE differs'),
    CodeRow('W035', 'warning', '11.12', 'trainer', 'In a hierarchy, groups that no strategy serves (the default walks the root only)'),
    CodeRow('W036', 'warning', '11.12', 'FreezeSet.load_state_dict', 'A saved freeze entry has no node; dropped'),
    CodeRow('W040', 'warning', '12.11', 'TopologySimulator._retire', 'The node was already retired; the plan is a no-op for it'),
    CodeRow('W041', 'warning', '12.11', 'TopologySimulator._revive', 'The node was not retired; the plan is a no-op for it'),
    CodeRow('W042', 'warning', '12.11', 'MutationEngine._rebind_groups', "A group's membership changed, so its aggregator, its controller state and the per-group strategy state were rebuilt"),
    CodeRow('W043', 'warning', '12.11', 'MutationEngine._commit (a modify ... rebuild)', "The node's architecture was replaced, so its optimizer was rebuilt and its moments are gone"),
    CodeRow('W044', 'warning', '12.11', 'MutationLimits.from_settings', 'A SCHEDULE that can grow the tree beyond WATCHDOG.MAX_TREE_GROWTH_RATIO will be reported by TreeGrowthProbe'),
    CodeRow('W045', 'warning', '12.11', 'MutationEngine._finish, MutationReconciler.reconcile', 'The journal reached JOURNAL_WARN, or a journal that does not reproduce its tree was tolerated because STRICT_REPLAY is false'),
    CodeRow('W050', 'warning', '13.10', 'TrainerEngine._check_identity', 'A cross-engine resume was explicitly allowed; parameters are restored, epoch bookkeeping is not resumed'),
    CodeRow('W051', 'warning', '13.10', 'TrainerEngine._resume', '--resume found no valid checkpoint in the configured directory; the run starts at epoch 0'),
    CodeRow('W052', 'warning', '14.11', 'ComponentRestorer._check', 'A slot differs (permissive mode), or an entry names a slot the run no longer has'),
    CodeRow('W053', 'warning', '14.11', 'ComponentRestorer.plan', 'The checkpoint records no components at all: a legacy file, or an all-built-in run'),
    CodeRow('C001', 'violation', '6.4', 'ComplianceGuard, stage: every stage', 'A non-finite or negative number: amount, every bucket, every weight, every amounts[k], omega must be finite and >= 0 (omega >= 1)'),
    CodeRow('C002', 'violation', '6.4', 'ComplianceGuard, stage: PENALTY', 'Floor below reference: For every k in NO (the effective non-offsettable set) with a reference amount: amounts[k] >= m_k * max(lambda_k, MIN_FLOOR_LAMBDA) within tolerance, where m_k is the sum of the magnitudes of the admitted events of rule k'),
    CodeRow('C003', 'violation', '6.4', 'ComplianceGuard, stage: PENALTY', "Omega below reference: omega >= 1.0 and omega >= omega_ref, where omega_ref is the reference stage's bounded product"),
    CodeRow('C004', 'violation', '6.4', 'ComplianceGuard, stage: PENALTY', 'void_rules too small: void_rules is a superset of {k in NO : reference amount exists}'),
    CodeRow('C005', 'violation', '6.4', 'ComplianceGuard, stage: WARNING', 'Warning-first violation: ctx.warning_only_rules contains no rule in NEVER_WARNING_FIRST or NO (framework.md W9)'),
    CodeRow('C006', 'violation', '6.4', 'ComplianceGuard, stage: every stage', 'Out-of-role output: A REWARD or PENALTY stage returned mutations, void_rules (REWARD) or changed a protected context field in prepare'),
    CodeRow('C007', 'violation', '6.4', 'ComplianceGuard, stage: every stage', "Ledger tampering: The ledger's event tuple changed during the walk, or prepare replaced the ledger"),
    CodeRow('C008', 'violation', '6.4', 'ComplianceGuard, stage: REWARD', 'Reward above ceiling: sum(weights[b] * bucket[b]) <= COMPLIANCE.REWARD_CEILING'),
    CodeRow('C009', 'violation', '6.4', 'ComplianceGuard, stage: REWARD, PENALTY', "Amount for an event not admitted: by_rule / amounts keys are within the ledger's rules for that side; a penalty key must also be tier-admitted and not warning-only"),
    CodeRow('C010', 'violation', '8.10', 'RollupGuard', 'A rollup dropped a void a child reported'),
    CodeRow('C011', 'violation', '8.10', 'RollupGuard', 'A rollup raised, returned a non-ScorerOutput, or returned a non-finite amount'),
    CodeRow('C012', 'violation', '8.10', 'RollupGuard', 'A rollup emitted warning mutations'),
    CodeRow('F001', 'finding', '8.3', 'function_probes.py (all)', 'The entry method raised on a well-formed input'),
    CodeRow('F002', 'finding', '8.3', 'function_probes.py (all)', 'Wrong return type (not a tensor, not a mapping, not a ScorerOutput, ...)'),
    CodeRow('F003', 'finding', '8.3', 'function_probes.py (all)', 'Wrong shape or size'),
    CodeRow('F004', 'finding', '8.3', 'function_probes.py (all)', 'A non-finite value in the output, or in a loss breakdown'),
    CodeRow('F005', 'finding', '8.3', 'function_probes.py (value, loss, regularization, objective)', 'Not differentiable: the output does not depend on the parameters or inputs it claims; skippable with probe["skip"]'),
    CodeRow('F006', 'finding', '8.3', 'function_probes.py (all except sampling)', 'Not deterministic: the same input gave a different output; skippable with probe["skip"]'),
    CodeRow('F007', 'finding', '8.3', 'function_probes.py (all)', 'The function mutated an input it was handed; skippable with probe["skip"]'),
    CodeRow('F008', 'finding', '8.3', 'function_probes.py (loss)', 'A loss breakdown.total disagrees with the returned tensor'),
    CodeRow('F009', 'finding', '8.3', 'function_probes.py (loss)', 'A loss requires a batch key the probe cannot supply and the loss provides no probe_batch'),
    CodeRow('F010', 'finding', '8.3', 'function_probes.py (rollup)', "A rollup dropped a child's void_rules"),
    CodeRow('F011', 'finding', '8.3', 'function_probes.py (all)', 'probe["skip"] names a code that cannot be skipped'),
    CodeRow('F012', 'finding', '8.3', 'function_probes.py (synthesis)', 'A synthesis output has the wrong shape for its declared shape rule'),
)}

#: Prefixes whose codes label the items inside a finding (17.2.3); none is raised alone.
FAMILIES: dict[str, str] = {
    "S": "step guard (7.6): inside StepGuardViolation and the card's metadata",
    "O": "optimizer probe (9.6): inside E056",
    "U": "weight guard (10.4): inside E063, E064, W017",
    "Y": "strategy audit (11.7): inside E073, E074, W030",
}

#: Codes in the table that are deliberately raised nowhere in the tree, with the reason.
RESERVED: dict[str, str] = {}
