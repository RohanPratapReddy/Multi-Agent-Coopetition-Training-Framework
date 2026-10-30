"""The trainer facade (document 08 section 5, document 10 section 6).

``MACTFTrainer`` ties together pieces that already exist: the peer tree, the
optimizers, the loss functions, the scoring pipeline, the peer strategies, the
weight updates, the logger, the watchdog, and the checkpoint format. It does not
re-implement any of them.

Two rules here are load-bearing:

* **Node parameters are never copied beyond the initial deep copy.** Every node's
  module comes from ``copy.deepcopy`` of the user's module, and the
  ``PeerAdapter`` holds that same object, so an optimizer built on the adapter
  trains the original.
* **Epoch order is fixed.** Delegate, rollout, synthesize, score, local update,
  aggregate, reference EMA, log/watchdog/checkpoint. Scoring happens *before*
  training, because a gradient produced before scoring would change every number
  the equivalence gate compares.

``normalize_episode`` lives here rather than only in ``scoring/normalize.py``
because the trainer owns the cohort boundary: the cohort is whatever ``score()``
returned for that epoch.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from .. import signals
from ..components.function_book import FunctionBook
from ..components.kappa_plan import KappaPlan
from ..components.kinds import Boundary
from ..components.loss_compliance import ensure_compliance
from ..components.resolver import ComponentResolver
from ..components.slots import SlotTable, SwapRecord
from ..components.strategy_plan import DEFAULT_PATH, StrategyPlan
from ..components.weight_plan import ROOT_LABEL, WeightPlan, group_label
from ..conf import LOSS_DEFAULTS, Settings, get_settings
from ..conf_files import authored_keys, is_authored
from ..coopetition.aggregation import (
    AggregateResult,
    AggregationReport,
    Aggregator,
    build_aggregator,
    feasible_bounds,
)
from ..coopetition.delegation import build_delegation
from ..coopetition.peer import (
    Peer,
    build_peer,
    build_reference,
    require_same_layout,
)
from ..coopetition.strategy import (
    FrozenPeerStrategy,
    PeerStrategy,
    TreeView,
    deduplicate_tied,
    group_layouts,
    tree_view,
)
from ..coopetition.synthesis import build_synthesis
from ..coopetition.temperature import AdaptiveKappa
from ..coopetition.tree import (
    HierarchySpec,
    PeerNode,
    PeerTree,
    aggregation_groups,
    branch_void,
)
from ..distill.config import DistillationConfig
from ..coopetition.weight_guard import Violation as WeightViolation
from ..coopetition.weight_guard import WeightGuard
from ..coopetition.weights import (
    GroupView,
    KappaSoftmaxAverage,
    WeightTraits,
    WeightUpdateStrategy,
    traits_of,
)
from ..exceptions import (
    ComplianceError,
    ContractError,
    ImproperlyConfigured,
    MutationError,
    ValidationError,
    WeightGuardViolation,
)
from ..orchestration.checkpoint_v2 import (
    CheckpointContext,
    MACTFCheckpoint,
    ResumeReport,
)
from ..orchestration.episode import Episode, EpisodeRunner, Task
from ..rl.functions.loss import CompositeLoss, LossFunction
from ..rl.loss import LossConfig
from ..rl.policy import make_policy
from ..runtime import Runtime, build_runtime, seed_everything
from ..scoring.hierarchy import HierarchicalScorer, run_inference
from ..scoring.normalize import (
    CohortNormalizer,
    NormalizationPolicy,
    NormalizationReport,
)
from ..scoring.judge import TaskJudge
from ..scoring.pipeline import ScoringPipeline, build_scorer
from ..scoring.scorecard import EpisodeScorecard
from ..training_utils import state_targets_for
from .callbacks import Callback, CallbackList, HOOK_NAMES, build_callbacks
from .finetune import FinetuneConfig, apply_finetuning
from .freeze import RETIRED, FreezeSet
from .mutation import MutationLimits, MutationPlan, MutationReport, structure_digest
from .mutation import add_peer as _op_add
from .mutation import modify_peer as _op_modify
from .mutation import purge_peer as _op_purge
from .mutation import retire_peer as _op_retire
from .mutation import revive_peer as _op_revive
from .mutation_engine import MutationEngine
from .mutation_reconcile import MutationState
from .optimizer import MACTFOptimizer, StepOutcome
from .optimizer_plan import PLAN_KEYS, OptimizerEntry, OptimizerPlan, verify_entry
from .path_state import PathState
from .protocol import (
    EpochResult,
    StepResult,
    TrainResult,
    TrainerCapabilities,
    TrainerProtocol,
    coerce_step_result,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .logging import MACTFLogger
    from .watchdog import HealthReport, MACTFWatchdog

log = logging.getLogger("mactf.training.trainer")

__all__ = ["MACTFTrainer", "TrainSpec"]

#: The engine that writes a schema-2 checkpoint. ``Orchestrator`` writes schema 1 and no
#: ``identity`` section, so the key's absence is how a resume reads "orchestrator" (section 13.4.8,
#: ``E091``).
ENGINE_NAME = "trainer"

#: How a node's parameters relate to another node's (document 08 section 5.3).
SHARING_MODES = ("independent", "shared_trunk", "tied_weights")

#: Accepted ``COOPETITION.SIGNAL_SOURCE`` values (document 11 section 8.2).
SIGNAL_SOURCES = ("points", "final_score")

#: Epoch orders a tree can be walked in.
WALK_ORDERS = ("leaves-first", "root-first", "post-order")


@dataclass(frozen=True)
class TrainSpec:
    """The resolved, immutable description of one training run.

    Produced by :meth:`MACTFTrainer.build` from the constructor arguments and the
    settings. Having it as a value object is what lets ``state_dict`` record the
    run's shape without reaching into mutable trainer fields.
    """

    run_id: str
    epochs: int
    steps_per_epoch: int
    batch_size: int
    learning_rate: float
    seed: int
    signal_source: str
    weight_update: str
    peer_strategy: str
    sharing: str
    hierarchy: bool
    num_nodes: int
    node_paths: tuple[str, ...]
    optimizer: Mapping[str, Any] = field(default_factory=dict)
    loss: Mapping[str, Any] = field(default_factory=dict)
    finetuning: Mapping[str, Any] = field(default_factory=dict)
    normalization: Mapping[str, Any] = field(default_factory=dict)
    amp: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    #: The resolved distillation config, or an empty mapping when off. Recorded so
    #: a checkpoint's shape can be compared without importing ``mactf.distill``.
    distillation: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "signal_source": self.signal_source,
            "weight_update": self.weight_update,
            "peer_strategy": self.peer_strategy,
            "sharing": self.sharing,
            "hierarchy": self.hierarchy,
            "num_nodes": self.num_nodes,
            "node_paths": list(self.node_paths),
            "optimizer": dict(self.optimizer),
            "loss": dict(self.loss),
            "finetuning": dict(self.finetuning),
            "normalization": dict(self.normalization),
            "amp": self.amp,
            "device": self.device,
            "dtype": self.dtype,
            "distillation": dict(self.distillation),
        }


class MACTFTrainer:
    """The composable trainer.

    Every constructor argument has a defined effect; none is decorative. See
    document 08 section 5.1's table for the meaning of each one.
    """

    def __init__(
        self,
        *,
        # -- what is being trained ------------------------------------
        model: nn.Module | None = None,
        model_factory: Callable[[str], nn.Module] | None = None,
        models: Mapping[str, nn.Module] | None = None,
        num_peers: int | None = None,
        peer_names: Sequence[str] | None = None,
        hierarchy: HierarchySpec | None = None,
        # -- how it is trained ----------------------------------------
        optimizer: Any = None,
        loss: Any = None,
        trainer: TrainerProtocol | Callable[..., Any] | None = None,
        finetuning: FinetuneConfig | Mapping[str, Any] | None = None,
        distillation: bool | Mapping[str, Any] | DistillationConfig | None = None,
        # -- how it is scored -----------------------------------------
        scorers: Any = None,
        hierarchical_scorer: HierarchicalScorer | None = None,
        scorer_settings: Mapping[str, Any] | None = None,
        # -- how peers interact ---------------------------------------
        peer_strategy: PeerStrategy | str | None = None,
        weight_update: WeightUpdateStrategy | str | None = None,
        aggregator: Aggregator | None = None,
        kappa: AdaptiveKappa | None = None,
        delegation: Any = None,
        synthesis: Any = None,
        # -- execution -------------------------------------------------
        runtime: Runtime | None = None,
        settings: Any | None = None,
        seed: int | None = None,
        device: str | None = None,
        dtype: str | None = None,
        amp: bool | None = None,
        grad_accumulation: int = 1,
        max_grad_norm: float | None = None,
        # -- data ------------------------------------------------------
        data: Any = None,
        task_source: Any = None,
        tokenizer: Any = None,
        data_collator: Callable[..., Any] | None = None,
        batch_size: int | None = None,
        steps_per_epoch: int | None = None,
        # -- observability and recovery --------------------------------
        logger: "MACTFLogger | None" = None,
        watchdog: "MACTFWatchdog | None" = None,
        checkpoint: MACTFCheckpoint | None = None,
        callbacks: Sequence[Callback] | None = None,
        checkpoint_dir: str | Path | None = None,
        # -- escape hatches --------------------------------------------
        hooks: Mapping[str, Callable[..., Any]] | None = None,
        strict: bool = True,
    ) -> None:
        self.settings = self._resolve_settings(settings)
        self.strict = bool(strict)

        # -- validation 1: exactly one model source -------------------
        given = [
            name
            for name, value in (
                ("model", model),
                ("model_factory", model_factory),
                ("models", models),
                ("hierarchy", hierarchy),
            )
            if value is not None
        ]
        # A hierarchy may be paired with a ``model_factory``: the factory then builds
        # every node whose spec carries no model of its own. Any other pairing is
        # ambiguous about which source wins, so it stays an error.
        paired = sorted(given) == ["hierarchy", "model_factory"]
        if len(given) != 1 and not paired:
            raise ImproperlyConfigured(
                "exactly one of model, model_factory, models, hierarchy must be given "
                "(a hierarchy may also carry a model_factory for nodes without a model); "
                f"got {given or 'none'}"
            )
        # -- validation 2: peer counts and names ----------------------
        if hierarchy is not None and num_peers is not None:
            raise ImproperlyConfigured(
                "hierarchy overrides num_peers; give one or the other, not both"
            )
        if num_peers is not None and int(num_peers) < 1:
            raise ImproperlyConfigured(
                f"num_peers must be >= 1, got {num_peers}"
            )
        if models is not None and num_peers is not None:
            raise ImproperlyConfigured(
                "num_peers is derived from models; they are mutually exclusive"
            )
        resolved_names = self._validate_names(peer_names)
        if resolved_names and models is not None:
            if set(resolved_names) != set(models):
                raise ImproperlyConfigured(
                    "peer_names must name exactly the modules in models; got "
                    f"{sorted(resolved_names)} versus {sorted(models)}"
                )
        if hierarchy is not None:
            self._validate_hierarchy_names(hierarchy, resolved_names)

        # -- validation 3: scorer exclusivity -------------------------
        if scorers is not None and hierarchical_scorer is not None:
            raise ImproperlyConfigured(
                "scorers and hierarchical_scorer are mutually exclusive; a hierarchical "
                "scorer owns its own pipeline per node"
            )
        # -- validation 4: delegation/synthesis require a hierarchy ---
        if hierarchy is None and (delegation is not None or synthesis is not None):
            raise ImproperlyConfigured(
                "delegation and synthesis are only meaningful with a hierarchy; a flat "
                "run has no parent to delegate from or synthesize at"
            )
        # -- validation 5: accumulator and clipping -------------------
        if int(grad_accumulation) < 1:
            raise ImproperlyConfigured(
                f"grad_accumulation must be >= 1, got {grad_accumulation}"
            )
        if max_grad_norm is not None and float(max_grad_norm) <= 0:
            raise ImproperlyConfigured(
                f"max_grad_norm must be > 0 when set, got {max_grad_norm}"
            )
        # -- validation 6: finetuning resolvable ----------------------
        self.finetuning = self._resolve_finetuning(finetuning)
        # -- validation 6b: distillation resolvable (document 14 section 19.1) ---
        self._distillation = self._resolve_distillation(distillation, self.settings)
        if self._distillation.enabled:
            if trainer is not None and not getattr(trainer, "composes_distillation", False):
                raise ImproperlyConfigured(
                    "trainer=... and distillation=... are mutually exclusive; both "
                    "install the per-step policy. Pass the teachers to distillation= and "
                    "drop trainer=, construct CoDistillStep yourself and pass it as "
                    "trainer=, declare composes_distillation=True on the policy, or pass "
                    "distillation=False to turn a project-level DISTILLATION.ENABLED off "
                    "for this run."
                )
            if steps_per_epoch is not None and int(steps_per_epoch) != 1:
                raise ImproperlyConfigured(
                    "distillation runs one cohort step per epoch, so steps_per_epoch must "
                    f"be 1 (got {steps_per_epoch}); every peer takes exactly one "
                    "train_step per cohort step"
                )
        # -- validation 7: chain validity -----------------------------
        self.scoring_settings = dict(scorer_settings or {})

        # -- storage --------------------------------------------------
        self._model = model
        self._model_factory = model_factory
        self._models = dict(models) if models is not None else None
        self._num_peers = int(num_peers) if num_peers is not None else None
        self._peer_names = resolved_names
        self._hierarchy = hierarchy
        self._tree_in_build: PeerTree | None = None
        self._trainer_policy = trainer
        self._loss_spec = loss
        self._scorers_spec = scorers
        self._hierarchical_scorer = hierarchical_scorer
        self._peer_strategy_spec = peer_strategy
        self._weight_update_spec = weight_update
        self._aggregator = aggregator
        self._kappa = kappa
        self._delegation_spec = delegation
        self._synthesis_spec = synthesis
        self._runtime = runtime
        self._seed_override = seed
        self._device_override = device
        self._dtype_override = dtype
        self._amp_override = amp
        self.grad_accumulation = int(grad_accumulation)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm else None
        self._optimizer_spec = optimizer
        self._data = data
        self._tokenizer = tokenizer
        self._data_collator = data_collator
        self._batch_size_override = int(batch_size) if batch_size else None
        self._steps_override = int(steps_per_epoch) if steps_per_epoch else None
        self.logger = logger
        self.watchdog = watchdog
        self.checkpoint = checkpoint
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.callbacks = build_callbacks(callbacks, trainer=self)
        self.hooks: dict[str, Callable[..., Any]] = self._validate_hooks(hooks)

        # -- mutable run state ----------------------------------------
        self.built = False
        self.tree: PeerTree | None = None
        self.peers: dict[str, Peer] = {}
        self.reference: Peer | None = None
        self.optimizers: dict[str, MACTFOptimizer] = {}
        self.controllers: dict[str, AdaptiveKappa] = {}
        self.controller: AdaptiveKappa | None = None
        self.aggregator: Aggregator | None = None
        self.scorer: Any = None
        self.strategy: PeerStrategy | None = None
        self.weight_strategy: WeightUpdateStrategy | None = None
        self.delegation_policy: Any = None
        self.synthesis_function: Any = None
        self.normalizer = CohortNormalizer(
            NormalizationPolicy.from_settings(self._coopetition_settings())
        )
        self.task_source = task_source
        self.runner: EpisodeRunner | None = None
        self.spec: TrainSpec | None = None
        self.start_epoch = 0
        self.global_step = 0
        self.stop_requested = False
        self.stop_reason = ""
        self.history: list[EpochResult] = []
        self.episodes: list[Episode] = []
        self.scorecards: list[EpisodeScorecard] = []
        self.reports: list[AggregationReport] = []
        self.signal_by_peer: dict[str, float] = {}
        #: ``(episode, node_path) -> evidence`` hook for the points layer (doc 11).
        self.evidence_provider: Callable[[Episode, str], Any] | None = None
        self.signal_sharing: list[dict[str, Any]] = []
        self.losses: list[float] = []
        self.grad_norms: list[float] = []
        self.failures: list[str] = []
        self._rng: torch.Generator | None = None
        self._restored = False
        #: The ``DistillResumeReport`` from the last ``load_state_dict``, if any.
        self.distillation_resume: Any = None
        self._closed = False
        self._last_health: Any = None
        self._started_at = 0.0
        self._epoch_seconds: list[float] = []
        self._level_durations: dict[int, float] = {}
        self._initial_rss: int | None = self._rss()

        #: Exposed so checkpoint restoration can find controller state.
        self.warnings = None

        # -- per-epoch scratch ----------------------------------------
        self._epoch = 0
        self._pending: Episode | None = None
        self._epoch_cards: list[EpisodeScorecard] = []
        self._epoch_step_losses: list[float] = []
        self._epoch_node_metrics: dict[str, dict[str, list[float]]] = {}
        self._normalization: NormalizationReport | None = None
        self._epoch_report: AggregationReport | None = None
        self._last_weights: dict[str, float] = {}
        self._previous_scores: dict[str, float] = {}
        self._dead_counts: dict[str, int] = {}
        self._initial_nodes: int | None = None
        self._group_aggregators: dict[str, Aggregator] = {}
        self._level_controllers: dict[str | None, AdaptiveKappa] = {}
        self._loss_function: LossFunction | None = None

        # -- tree mutation (document 17 section 12.7) -----------------------------------
        self.mutation_limits: MutationLimits = MutationLimits()
        self.mutation_journal: list[dict[str, Any]] = []
        self.topology_epoch: int = 0
        self._tombstones: set[str] = set()
        self._phase: str = "idle"                  # idle | epoch | observe
        self._mutations_in_cycle: int = 0
        self._active_view: PeerTree | None = None
        self._shared_trunk: nn.Module | None = None
        self._loss_baseline: int = 0               # first ``history`` index judged by the loss probes
        self._rate_baseline: int = 0               # first ``_epoch_seconds`` index judged by the rate probe
        self._node_specs: dict[str, dict[str, Any]] = {}   # path -> effective spec of a created or rebuilt node
        self._node_meta: dict[str, dict[str, Any]] = {}    # path -> final metadata of a created or touched node
        self._base_digest: str = ""                        # structure digest of the tree as built
        self._build_meta: dict[str, dict[str, Any]] = {}   # path -> metadata the tree was built with
        self.engine: MutationEngine = MutationEngine(self)
        self.path_state: PathState = PathState(self)

        # -- document 17: policies, functions, optimizer plan, freeze set ----------------
        # ``_trainer_policy`` keeps the meaning it always had: the policy the caller supplied
        # (``trainer=``), the distillation step, or ``None`` for the default step. The object
        # ``train_step`` calls (the default policy, or a guard around a foreign one) is
        # ``_policy``; both are set by ``_attach_policy`` in ``build``.
        self._policy: Any = None
        self._policy_origin: Any = None
        self._policy_guard: Any = None
        self.distill: Any = None
        #: ``False`` for a run created through the API without ``COMPONENTS.API_ALLOW_IMPORT``.
        self._components_trusted = True
        self._strict_restore = bool(
            dict(self.settings.get("COMPONENTS") or {}).get("STRICT_RESTORE", False)
        )
        # A trainer built in code with its own ``settings=`` may be the first thing to look at
        # ``COMPONENTS``: load the named modules before any name below is resolved. A no-op when
        # ``conf.configure`` already loaded the same block.
        from ..components.plugins import ensure_components_loaded

        ensure_components_loaded(self.settings)
        self.resolver = ComponentResolver()
        self.slots = SlotTable(self.resolver, trusted=self._components_trusted)
        trainer_block = dict(self.settings.get("TRAINER") or {})
        self.functions = FunctionBook.from_settings(
            self.slots,
            self.settings.get("RL"),
            loss_compliance=str(trainer_block.get("LOSS_COMPLIANCE", "require")).lower(),
            compliance_coefficient=self._compliance_coefficient(),
            guard_mode=str(
                dict(self.settings.get("COMPLIANCE") or {}).get("GUARD_MODE", "enforce")
            ).lower(),
            trusted=self._components_trusted,
        )
        self.functions.configure(self.settings.get("RL"))
        self.freeze = FreezeSet(mode=str(trainer_block.get("FREEZE_MODE", "legacy")))
        # -- document 17 sections 10 and 11: the weight plan and the strategy plan ------
        self.weight_plan: WeightPlan | None = None
        self.weight_guard: WeightGuard | None = None
        self.kappa_plan: KappaPlan | None = None
        self.weight_warnings: list[str] = []
        self.strategy_plan: StrategyPlan | None = None
        self.strategy_warnings: list[str] = []
        #: Notes from the document 17 section 14 component verification of a resume
        #: (``W052``, ``W053``). ``mactf runs show`` prints them.
        self.component_notes: list[str] = []
        #: The aggregator's resolved component when ``TRAINER.AGGREGATOR`` named one.
        self._aggregator_resolved: Any = None
        #: Parameter writes performed so far (``_write`` increments it); the strategy guard
        #: compares its delta over one call with the writes the context audited (``Y003``).
        self.write_count: int = 0
        self.optimizer_plan: OptimizerPlan | None = None
        self._verified_optimizers: set[str] = set()
        self._optimizer_warnings: list[str] = []
        self._function_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Construction helpers and validation
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_settings(settings: Any) -> Any:
        if settings is None:
            return get_settings()
        if isinstance(settings, Settings):
            return settings
        if isinstance(settings, Mapping):
            merged = copy.deepcopy(get_settings().as_dict())
            for key, value in settings.items():
                if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
            return Settings(merged)
        raise ImproperlyConfigured(
            f"settings must be a Settings, a mapping, or None; got {type(settings).__name__}"
        )

    @staticmethod
    def _validate_names(names: Sequence[str] | None) -> tuple[str, ...]:
        if names is None:
            return ()
        from ..coopetition.tree import validate_path

        resolved: list[str] = []
        for name in names:
            path = validate_path(str(name))
            if "/" in path:
                raise ImproperlyConfigured(
                    f"peer_names entries must be root names without '/'; got {path!r}"
                )
            if path in resolved:
                raise ImproperlyConfigured(f"peer_names has duplicate entry {path!r}")
            resolved.append(path)
        return tuple(resolved)

    @staticmethod
    def _validate_hierarchy_names(
        hierarchy: HierarchySpec, names: Sequence[str]
    ) -> None:
        if not names:
            return
        declared = tuple(hierarchy.root_names)
        if declared and tuple(names) != declared:
            raise ImproperlyConfigured(
                "peer_names and hierarchy.root_names disagree: "
                f"{list(names)} versus {list(declared)}"
            )

    @staticmethod
    def _resolve_finetuning(spec: Any) -> FinetuneConfig | None:
        if spec is None:
            return None
        try:
            resolved = FinetuneConfig.from_spec(spec)
        except Exception as error:  # noqa: BLE001 - re-raised as config error
            raise ImproperlyConfigured(f"finetuning is not resolvable: {error}") from error
        return resolved  # FinetuneConfig.__post_init__ has already validated every field

    @staticmethod
    def _resolve_distillation(spec: Any, settings: Any) -> DistillationConfig:
        """Resolve the polymorphic ``distillation=`` argument once, at construction.

        Validation errors (unknown key, enabled without teachers, bad value) surface
        here, not after a run has started. Teacher *construction* stays in ``build()``.
        """
        return DistillationConfig.from_spec(spec, settings=settings)

    @staticmethod
    def _validate_hooks(
        hooks: Mapping[str, Callable[..., Any]] | None,
    ) -> dict[str, Callable[..., Any]]:
        if not hooks:
            return {}
        unknown = sorted(set(hooks) - set(HOOK_NAMES))
        if unknown:
            raise ImproperlyConfigured(
                "unknown hook name(s): "
                + ", ".join(unknown)
                + "; known hooks: "
                + ", ".join(HOOK_NAMES)
            )
        validated: dict[str, Callable[..., Any]] = {}
        for name, hook in hooks.items():
            if not callable(hook):
                raise ImproperlyConfigured(f"hook {name!r} is not callable")
            validated[name] = hook
        return validated

    # -- settings views ------------------------------------------------
    def _coopetition_settings(self) -> dict[str, Any]:
        return dict(self.settings.get("COOPETITION", {}) or {})

    def _training_settings(self) -> dict[str, Any]:
        return dict(self.settings.get("TRAINING", {}) or {})

    @property
    def signal_source(self) -> str:
        source = str(
            self._coopetition_settings().get("SIGNAL_SOURCE", "final_score")
        ).strip().lower()
        return source if source in SIGNAL_SOURCES else "final_score"

    @property
    def node_paths(self) -> tuple[str, ...]:
        if self.tree is not None:
            return tuple(node.path for node in self.tree.walk("root-first"))
        return tuple(self.peers)

    @property
    def stages(self) -> tuple[str, ...]:
        return self.node_paths

    # ------------------------------------------------------------------
    # build()
    # ------------------------------------------------------------------
    def build(self) -> "MACTFTrainer":
        """Resolve runtime, build the tree and peers, bind optimizers."""
        if self.built:
            return self
        training = self._training_settings()
        coopetition = self._coopetition_settings()

        runtime = self._runtime or self._build_runtime(training)
        self._runtime = runtime
        self._seed_rng(runtime.seed)

        tree = self._build_tree()
        self.tree = tree
        trainer_block = dict(self.settings.get("TRAINER") or {})
        self.mutation_limits = MutationLimits.from_settings(
            trainer_block.get("MUTATION"),
            initial_roots=len(tree.root_paths),
            initial_nodes=len(tree),
            growth_ratio=float(
                dict(self.settings.get("WATCHDOG") or {}).get("MAX_TREE_GROWTH_RATIO", 0) or 0
            ) or None,
        )
        self.freeze.bind(tree)
        self.freeze.set_mode(str(trainer_block.get("FREEZE_MODE", "legacy")).lower())
        self.peers = self._build_peers(tree, runtime)

        # Reference policy: one per run, frozen, from the first root.
        first = next(iter(self.peers.values()))
        self.reference = build_reference(first, runtime=runtime)

        # Sibling layouts must match; different branches may not.
        self._validate_group_layouts(tree)

        # Optimizers bind *after* finetuning and freezing.
        self._bind_optimizers(runtime)

        # Temperature, aggregator, strategies.
        roots = len(tree.root_paths)
        self.kappa_plan = KappaPlan.from_settings(coopetition)
        root_settings = self.kappa_plan.settings_for(ROOT_LABEL, coopetition)
        # One owner of the temperature: the aggregator's controller *is* the root
        # controller, so the checkpointed kappa is the kappa that was used.
        aggregator_record: dict[str, Any] = {}
        self.aggregator = self._aggregator or build_aggregator(
            trainer_block.get("AGGREGATOR"),
            root_settings,
            num_peers=max(1, roots),
            resolver=self.resolver,
            trusted=self._components_trusted,
            record=aggregator_record,
        )
        self._aggregator_resolved = aggregator_record.get("resolved")
        if self._kappa is not None:
            self.aggregator.controller = self._kappa
        self.controller = self.aggregator.controller
        levels = self._build_controllers(coopetition)
        levels[None] = self.controller
        self._level_controllers = levels
        self.controllers = {
            (ROOT_LABEL if parent is None else parent): controller
            for parent, controller in levels.items()
        }
        for _parent, members in aggregation_groups(tree):
            feasible_bounds(
                max(1, len(members)),
                floor=float(coopetition.get("WEIGHT_FLOOR", 0.02)),
                cap=float(coopetition.get("WEIGHT_CAP", 0.6)),
            )
        self.weight_plan = WeightPlan.from_settings(
            self.settings, self.slots,
            aggregator=self.aggregator,
            code_spec=self._weight_update_spec,
            resolver=self.resolver,
            trusted=self._components_trusted,
        )
        self.weight_guard = self.weight_plan.guard
        self.weight_strategy = self.weight_plan.strategy_for(ROOT_LABEL, self.aggregator)
        self._validate_weight_plan(initial=True)
        # The strategy plan: ``peer_strategy=`` beats an authored ``TRAINER.PEER_STRATEGY``,
        # which beats the default for the tree's shape (``hierarchical`` on a hierarchy, else
        # ``kappa_softmax``, the mixing strategy a default flat run needs to match ``Orchestrator``).
        self.strategy_plan = StrategyPlan.from_settings(
            self.settings, self.slots,
            code_spec=self._peer_strategy_spec,
            hierarchy=self._hierarchy is not None,
            resolver=self.resolver,
            trusted=self._components_trusted,
        )
        self.strategy = self.strategy_plan.default_instance()
        self._validate_strategy_plan(initial=True)
        for instance in self.strategy_plan.live_instances():
            instance.attach(self)
        self.freeze.check_learners(node.path for node in tree.walk("root-first") if node.peer is not None)

        # Delegation and synthesis only exist in a hierarchy.
        if self._hierarchy is not None:
            self.delegation_policy = build_delegation(self._delegation_spec)
            self.synthesis_function = build_synthesis(self._synthesis_spec)

        # Scoring.
        self.scorer = self._build_scorer()
        self.warnings = getattr(self.scorer, "warning_scorer", None)

        # Data, runner, checkpoint manager.
        self.task_source = self.task_source or self._build_task_source(training)
        self.runner = EpisodeRunner(runtime=runtime)
        if self.checkpoint is None and self.watchdog is not None:
            # The watchdog already owns a writer; reuse it so the trainer, the watchdog and
            # ``resume_from_last_checkpoint`` all read and write one directory.
            self.checkpoint = self.watchdog.checkpoint
        else:
            self.checkpoint = self._build_checkpoint()
        if self.watchdog is not None:
            if self.checkpoint is None:
                self.checkpoint = self.watchdog.checkpoint
            if not isinstance(self.watchdog.checkpoint, _CheckpointBridge):
                # The watchdog hands its writer a ``HealthContext``; the writer
                # wants a ``CheckpointContext`` built from this trainer.
                self.watchdog.checkpoint = _CheckpointBridge(
                    self.watchdog.checkpoint, self
                )
        self._attach_policy()
        self._validate_functions()
        self._setup_custom_trainer()

        batch_size = self._batch_size_override or int(
            training.get("BATCH_SIZE", 32)
        )
        # One update per node per epoch reproduces ``Orchestrator``; a caller
        # who wants more passes ``steps_per_epoch`` explicitly.
        steps = self._steps_override or 1
        self.spec = TrainSpec(
            run_id=self.run_id,
            epochs=int(training.get("EPOCHS", 1)),
            steps_per_epoch=steps,
            batch_size=batch_size,
            learning_rate=float(training.get("LEARNING_RATE", 1e-3)),
            seed=int(runtime.seed),
            signal_source=self.signal_source,
            weight_update=getattr(self.weight_strategy, "name", "kappa_softmax"),
            peer_strategy=getattr(self.strategy, "name", "independent"),
            sharing=str(getattr(self._hierarchy, "sharing", "independent")),
            hierarchy=self._hierarchy is not None,
            num_nodes=len(self.peers),
            node_paths=self.node_paths,
            optimizer=self.optimizers[next(iter(self.optimizers))].describe(),
            loss=self._describe_loss(),
            finetuning=self.finetuning.to_dict() if self.finetuning else {},
            normalization=self.normalizer.policy.as_dict(),
            amp=bool(runtime.amp),
            device=str(runtime.device),
            dtype=str(runtime.dtype).replace("torch.", ""),
            distillation=self._distillation_record(),
        )
        self._initial_nodes = len(self.active_tree)
        self._base_digest = structure_digest(self.tree, ())
        self._build_meta = {
            node.path: copy.deepcopy(node.metadata) for node in self.tree.walk("root-first") if node.metadata
        }
        if self.mutation_limits.enabled:
            problems = self.path_state.audit()
            if problems:
                raise MutationError(
                    "E089",
                    "the trainer is inconsistent before any mutation ran, so one would only make it worse: "
                    + "; ".join(problems),
                )
        self.built = True
        self._dispatch("on_build", self)
        self.callbacks.dispatch("on_build", self)
        return self

    @property
    def run_id(self) -> str:
        if getattr(self, "_run_id", None) is None:
            self._run_id = f"train-{int(time.time()) % 100000}-{id(self) % 10000}"
        return self._run_id

    @run_id.setter
    def run_id(self, value: str) -> None:
        if self.built:
            raise ImproperlyConfigured("run_id cannot change after build()")
        if not value:
            raise ValidationError("run_id must be non-empty")
        self._run_id = str(value)

    def _build_runtime(self, training: Mapping[str, Any]) -> Runtime:
        if any(
            value is not None
            for value in (
                self._device_override,
                self._dtype_override,
                self._amp_override,
                self._seed_override,
            )
        ):
            merged = copy.deepcopy(self.settings.as_dict())
            if self._device_override is not None:
                merged["DEVICE"] = self._device_override
            if self._dtype_override is not None:
                merged["DTYPE"] = self._dtype_override
            if self._amp_override is not None:
                merged["AMP"] = bool(self._amp_override)
            if self._seed_override is not None:
                merged["SEED"] = int(self._seed_override)
            return build_runtime(Settings(merged))
        del training
        return build_runtime(self.settings)

    def _seed_rng(self, seed: int) -> None:
        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))
        seed_everything(int(seed))

    # -- tree ----------------------------------------------------------
    def _build_tree(self) -> PeerTree:
        if self._hierarchy is not None:
            return self._hierarchy.build(runtime=self._runtime, settings=self.settings)
        names = self._peer_names
        if not names:
            if self._models is not None:
                names = tuple(self._models)
            else:
                count = self._num_peers or len(self.settings.get("PEERS", []) or []) or 1
                names = tuple(f"peer-{index + 1}" for index in range(count))
        if self._num_peers is not None and len(names) != self._num_peers:
            raise ImproperlyConfigured(
                f"num_peers={self._num_peers} but {len(names)} names were resolved"
            )
        return PeerTree.flat(names)

    def _module_for(self, path: str) -> nn.Module:
        """A fresh module for one node, deep-copied from the user's source."""
        if self._models is not None:
            source = self._models.get(path)
            if source is None:
                raise ImproperlyConfigured(
                    f"models has no entry for node {path!r}; known: {sorted(self._models)}"
                )
            return copy.deepcopy(source)
        if self._hierarchy is not None:
            return self._hierarchy_module(path)
        if self._model_factory is not None:
            return self._from_factory(path)
        if self._model is None:
            raise ImproperlyConfigured("no model source available for build")
        return copy.deepcopy(self._model)

    def _from_factory(self, path: str) -> nn.Module:
        """Call ``model_factory`` with the node path, or with nothing if it takes no argument."""
        import inspect

        factory = self._model_factory
        try:
            takes_path = any(
                parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
                for parameter in inspect.signature(factory).parameters.values()
            )
        except (TypeError, ValueError):  # a builtin or C callable: assume the documented form
            takes_path = True
        produced = factory(path) if takes_path else factory()
        if not isinstance(produced, nn.Module):
            raise ContractError(
                f"model_factory({path!r}) returned {type(produced).__name__}, not nn.Module"
            )
        return produced

    def _hierarchy_module(self, path: str) -> nn.Module:
        """A node's module from the spec: ``root_models`` for roots, ``LevelSpec.model`` below.

        A source is an ``nn.Module`` (deep-copied per node) or a callable taking the
        node path. A node with no source raises and names where one is expected.
        """
        spec = self._hierarchy
        tree = self._tree_in_build
        node = tree[path] if tree is not None else None
        depth = int(node.depth) if node is not None else 1
        if depth <= 1:
            sources = dict(spec.root_models or {})
            source = sources.get(path, sources.get(""))
            where = f"root_models[{path!r}]"
        else:
            level = spec.levels[depth - 2] if depth - 2 < len(spec.levels) else None
            source = getattr(level, "model", None)
            where = f"levels[{depth - 2}].model ({getattr(level, 'label', '?')!r})"
        if source is None and self._model_factory is not None:
            return self._from_factory(path)
        if source is None:
            raise ImproperlyConfigured(
                f"no model source for hierarchy node {path!r}; set {where} on the HierarchySpec "
                "or give the trainer a model_factory"
            )
        if isinstance(source, nn.Module):
            return copy.deepcopy(source)
        if callable(source):
            produced = source(path)
            if not isinstance(produced, nn.Module):
                raise ContractError(
                    f"{where} returned {type(produced).__name__} for {path!r}, not nn.Module"
                )
            return produced
        raise ImproperlyConfigured(
            f"{where} must be an nn.Module or a callable, got {type(source).__name__}"
        )

    def _build_peers(self, tree: PeerTree, runtime: Runtime) -> dict[str, Peer]:
        self._tree_in_build = tree
        sharing = str(getattr(self._hierarchy, "sharing", "independent"))
        freeze_base = bool(
            self.finetuning.freeze_base if self.finetuning is not None else True
        )

        peers: dict[str, Peer] = {}
        trunk: nn.Module | None = None
        for node in tree.walk("root-first"):
            module = self._module_for(node.path)

            if sharing == "shared_trunk" and node.depth > 1:
                # Children deep-copy the trunk *once* and own their heads. A
                # gradient from one child then accumulates into the shared trunk,
                # which is why the trunk is frozen by default.
                if trunk is None:
                    trunk = copy.deepcopy(module)
                    if freeze_base:
                        for parameter in trunk.parameters():
                            parameter.requires_grad_(False)
                    self._shared_trunk = trunk
                module = _with_shared_trunk(trunk, module)

            peer = self._assemble_peer(node.path, node.depth, module, runtime)
            node.peer = peer
            peers[node.path] = peer
        if not peers:  # pragma: no cover - the tree always has one node
            raise ImproperlyConfigured("the tree produced no nodes")
        return peers

    def _assemble_peer(
        self,
        path: str,
        depth: int,
        module: nn.Module,
        runtime: Runtime,
        *,
        finetuned: bool = False,
    ) -> Peer:
        """Module -> peer: finetuning, the optimizer-less ``build_peer``, and the metadata.

        ``finetuned`` says the module is a copy of a peer that already went through
        ``apply_finetuning``; the call is then a no-op (``_already_applied``) and the flag is only
        recorded. The peer is *not* placed in the tree or in ``self.peers``: the caller does that,
        which is what lets the engine stage a peer without touching the live trainer.
        """
        training = self._training_settings()
        sharing = str(getattr(self._hierarchy, "sharing", "independent"))
        module, report = apply_finetuning(module, self.finetuning, device=runtime.device)
        peer = build_peer(
            module,
            name=path,
            runtime=runtime,
            learning_rate=float(training.get("LEARNING_RATE", 1e-3)),
            action_dim=int(training.get("ACTION_DIM", 4)),
            state_dim=int(training.get("STATE_DIM", 16)),
            weight_decay=float(training.get("WEIGHT_DECAY", 0.0)),
        )
        peer.metadata["depth"] = depth
        peer.metadata["sharing"] = sharing
        peer.metadata["finetuning"] = report
        if finetuned:
            peer.metadata["finetuning_inherited"] = True
        if depth == 1:
            peer.metadata["role"] = "root"
        return peer

    def _validate_group_layouts(self, tree: PeerTree) -> None:
        for _parent, members in aggregation_groups(tree):
            if len(members) < 2:
                continue
            group_layouts(self.peers, members)

    def _default_block_authored(self) -> frozenset[str] | None:
        """Keys the project wrote in ``OPTIMIZER``, *excluding* the plan keys.

        Authoring only ``OVERRIDES`` (or ``MIGRATE``, ``VERIFY``, ``STRICT_GLOBS``) does not
        author the default optimizer: the nodes the overrides do not select keep the legacy
        Adam at ``TRAINING.LEARNING_RATE`` (invariant O1).
        """
        written = authored_keys(self.settings, "OPTIMIZER")
        if written is None:
            return None
        return frozenset(written) - PLAN_KEYS

    def _optimizer_block(self) -> dict[str, Any]:
        """The ``OPTIMIZER`` block when the project wrote a default optimizer, else ``{}``.

        Same as ``trainer.py:883-905`` except that writing only plan keys (``OVERRIDES`` and
        friends) does not count as writing the default optimizer.
        """
        written = self._default_block_authored()
        if not is_authored(self.settings, "OPTIMIZER") or written == frozenset():
            return {}
        block = dict(self.settings.get("OPTIMIZER", {}) or {})
        if not block:
            return {}
        training = self._training_settings()
        if written is not None:
            if "LR" not in written and "LEARNING_RATE" in training:
                block["LR"] = training["LEARNING_RATE"]
            if "MAX_GRAD_NORM" not in written and "MAX_GRAD_NORM" in training:
                block["MAX_GRAD_NORM"] = training["MAX_GRAD_NORM"]
        return block

    def _build_optimizer_plan(self) -> OptimizerPlan:
        from .optimizer_plan import OptimizerPlan
        from .schedulers import horizon

        training = self._training_settings()
        steps = self._steps_override or 1
        accumulation = max(
            int(self.grad_accumulation or 1),
            int(training.get("GRAD_ACCUMULATION", 1) or 1),
        )
        plan = OptimizerPlan.from_settings(
            self.settings,
            block=self._optimizer_block(),
            written=self._default_block_authored(),
            horizon=horizon(int(training.get("EPOCHS", 1)), steps, accumulation),
            code_spec=self._optimizer_spec,
            inline_lr=float(training.get("LEARNING_RATE", 1e-3)),
            trusted=self._components_trusted,
        )
        self.optimizer_plan = plan
        return plan

    def _make_optimizer(
        self, path: str, entry: OptimizerEntry, runtime: Runtime, peer: Peer | None = None
    ) -> MACTFOptimizer:
        """Build, finish and bind the facade of one node for one entry.

        ``peer`` binds the facade to a peer the mutation engine has built but not installed.
        """
        peer = peer if peer is not None else self.peers[path]
        module = peer.adapter
        optimizer = self.optimizer_plan.instantiate(path, entry)
        training = self._training_settings()
        explicit = entry.explicit
        # Precedence, highest first: a per-node override's own key, the constructor argument,
        # the block, ``TRAINING``. The first two lines are the original code (trainer.py:929-945).
        if "GRAD_ACCUMULATION" not in explicit:
            if self.grad_accumulation != 1:
                optimizer.grad_accumulation = self.grad_accumulation
            elif entry.kind != "code" and optimizer.grad_accumulation == 1:
                configured = int(training.get("GRAD_ACCUMULATION", 1) or 1)
                if configured > 1:
                    optimizer.grad_accumulation = configured
        if "MAX_GRAD_NORM" not in explicit:
            if self.max_grad_norm is not None:
                optimizer.max_grad_norm = self.max_grad_norm
            elif optimizer.max_grad_norm is None:
                configured_norm = float(training.get("MAX_GRAD_NORM", 0.0) or 0.0)
                optimizer.max_grad_norm = configured_norm if configured_norm > 0 else None
        optimizer.scaler = runtime.grad_scaler
        optimizer.node_path = path
        optimizer.entry_hash = entry.spec_hash
        optimizer.bind(list(module.parameters()), named=list(module.named_parameters()))
        return optimizer

    def _install_optimizer(self, path: str, optimizer: MACTFOptimizer) -> None:
        self.optimizers[path] = optimizer
        guard = self._policy_guard
        if guard is not None:
            guard.rebind(path, optimizer)      # TRAINER.STEP_GUARD.MAX_GRAD_NORM caps a new optimizer too
        # The facade replaces the legacy Adam on the peer so ``Peer.state_dict`` carries it.
        self.peers[path].optimizer = optimizer  # type: ignore[assignment]

    def _bind_optimizers(self, runtime: Runtime) -> None:
        plan = self._build_optimizer_plan()
        self._optimizer_warnings = plan.validate(self.node_paths, initial=True)
        for path in self.peers:
            entry = plan.select(path)
            self._verify_optimizer(entry)
            self._install_optimizer(path, self._make_optimizer(path, entry, runtime))
            if entry.frozen:
                self.freeze.add(path, "optimizer:null")
        self._validate_tied_parameters()


    def _verify_optimizer(self, entry: OptimizerEntry) -> None:
        from .optimizer_plan import verify_entry

        mode = self.optimizer_plan.verify
        if mode == "off" or entry.spec_hash in self._verified_optimizers:
            return
        if mode == "custom" and not entry.customized:
            return
        verify_entry(entry)
        self._verified_optimizers.add(entry.spec_hash)


    def swap_optimizer(
        self, slot: str, block: Mapping[str, Any] | None, *,
        boundary: Boundary = Boundary.EPISODE, mode: str | None = None,
    ) -> list[SwapRecord]:
        """Replace (``block``) or remove (``None``) an optimizer entry, now.

        ``slot`` is ``"OPTIMIZER"`` or ``"OPTIMIZER.<glob>"``. Atomic: on any failure the old
        optimizers, the plan and the freeze set are exactly as they were.
        """
        self._require_built()
        plan = self.optimizer_plan
        runtime = self._runtime
        staged = plan.stage(
            slot, block, boundary=boundary, node_paths=list(self.peers),
            live=self.optimizers,
            make=lambda path, entry: self._make_optimizer(path, entry, runtime),
            mode=mode,
        )
        if staged.entry is not None:           # a replacement entry is probed before it is committed
            self._verify_optimizer(staged.entry)
        plan.commit(staged)
        for path, optimizer in staged.optimizers.items():
            self._install_optimizer(path, optimizer)
        for path in staged.frozen:
            self.freeze.add(path, "optimizer:null")
        for path in staged.thawed:
            self.freeze.discard(path, "optimizer:null")
        self._optimizer_warnings = plan.validate(list(self.peers), initial=False)
        return list(staged.records)

    def _apply_pending_optimizers(self) -> list[SwapRecord]:
        applied: list[SwapRecord] = []
        for slot in self.optimizer_plan.pending_slots():
            block, mode = self.optimizer_plan.take_pending(slot)
            try:
                applied.extend(self.swap_optimizer(slot, block, mode=mode))
            except Exception as error:  # noqa: BLE001 - a failed queued swap must not end the run
                self.failures.append(f"queued optimizer swap {slot!r} failed: {error}")
                log.error("queued optimizer swap %s failed: %s", slot, error)
        return applied

    def _ensure_optimizer(self, path: str) -> MACTFOptimizer | None:
        """What ``checkpoint_v2.restore`` calls (``checkpoint_v2.py:1041``) for a node with
        stored optimizer state and no live optimizer. Fixes O-G8."""
        peer = self.peers.get(path)
        if peer is None or self.optimizer_plan is None:
            return None
        optimizer = self._make_optimizer(path, self.optimizer_plan.select(path), self._runtime)
        self._install_optimizer(path, optimizer)
        return optimizer

    def _validate_tied_parameters(
        self,
        *,
        tree: PeerTree | None = None,
        peers: Mapping[str, Peer] | None = None,
        skip: Iterable[str] | None = None,
    ) -> None:
        """A tied parameter in two groups cannot receive two blended values.

        The keyword arguments let the mutation engine ask the same question about a tree and a
        set of peers that are not installed yet, over active members only (M7). With none given
        it reads the live tree and peers and skips the retired ones, which at build is nobody.
        """
        tree = tree if tree is not None else self.tree
        peers = peers if peers is not None else self.peers
        skipped = set(skip) if skip is not None else set(self.freeze.retired())
        owners: dict[int, str] = {}
        for parent, members in aggregation_groups(tree):
            group = "root" if parent is None else parent
            seen_in_group: dict[int, str] = {}
            for member in members:
                if member in skipped:
                    continue
                peer = peers.get(member)
                if peer is None:
                    continue
                for name, parameter in peer.module.named_parameters():
                    if not parameter.requires_grad:
                        # A frozen shared trunk holds identical values everywhere,
                        # so averaging it is harmless and it is never double counted.
                        continue
                    key = id(parameter)
                    if key in seen_in_group:
                        raise ContractError(
                            f"parameter {name!r} is tied between {seen_in_group[key]!r} "
                            f"and {member!r} in group {group!r}; siblings must not share "
                            "parameters because they are averaged"
                        )
                    seen_in_group[key] = member
                    previous = owners.get(key)
                    if previous is not None and previous != group:
                        raise ContractError(
                            f"parameter {name!r} is tied across groups {previous!r} and "
                            f"{group!r}; a tied parameter cannot receive two different "
                            "blended values"
                        )
                    owners[key] = group

    def _build_controllers(
        self, coopetition: Mapping[str, Any]
    ) -> dict[str | None, AdaptiveKappa]:
        if self._kappa is not None:
            return {None: self._kappa}
        controllers: dict[str | None, AdaptiveKappa] = {}
        for parent, members in aggregation_groups(self.tree):
            controllers[parent] = self.kappa_plan.build(
                group_label(parent), coopetition, num_peers=max(1, len(members))
            )
        return controllers

    def _build_scorer(self) -> Any:
        if self._hierarchical_scorer is not None:
            return self._hierarchical_scorer
        if self._scorers_spec is not None:
            pipeline = build_scorer(self.settings, scorers=_as_scorers(self._scorers_spec))
        else:
            # The stage slots live in the trainer's table so a checkpoint can record them.
            pipeline = build_scorer(
                self.settings, slots=self.slots, trusted=self._components_trusted
            )
        if self._hierarchy is not None:
            return HierarchicalScorer(pipeline, rollup_source=self._rollup_choice)
        return pipeline

    def _rollup_choice(self) -> Any:
        """The book's guarded rollup, or ``None`` so the scorer keeps its default."""
        return self.functions.get("rollup", "")

    def _build_task_source(self, training: Mapping[str, Any]) -> Any:
        from ..data.registry import build_source

        if self._data is not None:
            factory = getattr(self._data, "next", None)
            if not callable(factory):
                raise ImproperlyConfigured(
                    "data must be an object with a next(epoch) method, or use task_source"
                )
            return self._data
        name = str(training.get("TASK") or "synthetic")
        try:
            return build_source(name, settings=self.settings)
        except Exception as error:  # noqa: BLE001 - degraded, recorded, not silent
            message = f"task source {name!r} unavailable ({error}); using synthetic"
            self.failures.append(message)
            return build_source("synthetic", settings=self.settings)

    def _build_checkpoint(self) -> MACTFCheckpoint | None:
        if self.checkpoint is not None:
            return self.checkpoint
        try:
            # ``checkpoint_dir=`` wins over CHECKPOINT.DIR, which wins over PATHS.
            return MACTFCheckpoint.from_settings(
                self.settings, run_id=self.run_id, directory=self._checkpoint_dir
            )
        except ImproperlyConfigured:
            if self._checkpoint_dir is not None or _checkpoint_block_written(self.settings):
                raise  # the project asked for something specific; do not hide a bad value
            return None
        except Exception:  # noqa: BLE001 - checkpointing is optional
            return None

    def _describe_loss(self) -> dict[str, Any]:
        resolved = self._resolve_loss()
        describe = getattr(resolved, "describe", None)
        return describe() if callable(describe) else {"class": type(resolved).__name__}

    def _loss_block(self) -> dict[str, Any]:
        """The ``LOSS`` block when the project wrote it, else ``{}``."""
        if not is_authored(self.settings, "LOSS"):
            return {}
        return dict(self.settings.get("LOSS", {}) or {})

    def _resolve_loss(self) -> LossFunction:
        """The loss ``LOSS``/``loss=`` names, extended with the compliance term (7.8).

        A ``CompositeLoss`` already carries ``compliance_floor``, so the default objective is
        returned as it is (invariant I1); any other loss goes through ``ensure_compliance``
        under ``TRAINER.LOSS_COMPLIANCE``.
        """
        resolved = self._resolve_loss_unchecked()
        if isinstance(resolved, CompositeLoss):
            return resolved
        mode = str(dict(self.settings.get("TRAINER") or {}).get("LOSS_COMPLIANCE", "require"))
        return ensure_compliance(
            resolved, mode=mode.lower(), coefficient=self._compliance_coefficient()
        )

    def _resolve_loss_unchecked(self) -> LossFunction:
        if self._loss_spec is None:
            block = self._loss_block()
            kind = str(block.get("TYPE", "composite")).lower()
            if block and kind != "composite":
                # Any other TYPE names a mactf.rl.functions registry entry.
                from ..rl.functions.registry import build_function

                payload = {"type": kind, **dict(block.get("ARGS") or {})}
                return build_function(payload, family="loss")
            return CompositeLoss(config=self._loss_config())
        from ..rl.functions._spec import SpecMixin

        if isinstance(self._loss_spec, SpecMixin) or hasattr(self._loss_spec, "compute"):
            return self._loss_spec
        if isinstance(self._loss_spec, Mapping):
            from ..rl.functions.registry import build_function

            payload = dict(self._loss_spec)
            return build_function(payload, family="loss")
        if callable(self._loss_spec):
            from ..rl.functions.loss import CallableLoss

            return CallableLoss(self._loss_spec)
        raise ValidationError(
            f"loss must be a LossFunction, a mapping, or a callable; got "
            f"{type(self._loss_spec).__name__}"
        )

    def _loss_config(self) -> LossConfig:
        """``LOSS`` wins as a whole block when written; otherwise ``TRAINING``.

        Block-level, not per key: a project that sets ``LOSS.KL_COEF`` and leaves
        ``LOSS.VALUE_COEF`` at its default gets the default, not ``TRAINING``'s.
        """
        loss = self._loss_block()
        source = loss or self._training_settings()
        return LossConfig(
            value_coef=float(source.get("VALUE_COEF", 0.5)),
            kl_coef=float(source.get("KL_COEF", 0.01)),
            entropy_coef=float(source.get("ENTROPY_COEF", 0.0)),
            clip_epsilon=float(source.get("CLIP_EPSILON", 0.2)),
            compliance_coef=float(source.get("COMPLIANCE_COEF", 1.0)),
            advantage_normalization=bool(source.get("ADVANTAGE_NORMALIZATION", True)),
        )

    # ------------------------------------------------------------------
    # Extension dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, hook: str, *args: Any) -> Any:
        """Call the user's hook, returning its value or ``None``.

        The final non-``None`` return wins, which matches ``Signal.send``'s
        convention so a hook and a callback behave predictably.
        """
        fn = self.hooks.get(hook)
        if fn is None:
            return None
        return fn(*args)

    def _dispatch_many(self, hook: str, *args: Any) -> Any:
        """Hooks first, then callbacks; the last non-``None`` value wins."""
        value = self._dispatch(hook, *args)
        for produced in self.callbacks.dispatch(hook, *args):
            value = produced
        return value

    def add_callback(self, callback: Callback) -> "MACTFTrainer":
        self.callbacks.add(callback)
        return self

    def remove_callback(self, name: str) -> bool:
        return self.callbacks.remove(name)

    def request_stop(self, reason: str = "") -> None:
        self.stop_requested = True
        self.stop_reason = str(reason or "stop requested")

    @property
    def active_tree(self) -> PeerTree:
        """The tree rollout, scoring, training, coordination and aggregation walk (M10).

        ``PeerTree.without`` returns the tree itself when nothing is retired, so a run that never
        mutates pays nothing and sees nothing different. The view is rebuilt after a commit
        (``_active_view`` is reset there) and at the top of each epoch.
        """
        view = self._active_view
        if view is None:
            view = self.tree.without(self.freeze.retired())
            self._active_view = view
        return view

    @property
    def active_node_paths(self) -> tuple[str, ...]:
        return tuple(node.path for node in self.active_tree.walk("root-first"))

    def _mark_node(self, path: str, status: str) -> None:
        """Set a node's status on the tree and on the cached view.

        ``without`` shares unchanged node objects, so for almost every node one write is enough.
        An ancestor that lost a retired child is a copy in the view, and a status written only
        to the tree would leave the copy ``active`` for the rest of the epoch, so a failed
        parent would still be trained. Both are written.
        """
        self.tree.mark(path, status)
        view = self._active_view
        if view is not None and view is not self.tree:
            node = view.get(path)
            if node is not None:
                node.status = status

    # -- tree mutation ---------------------------------------------------
    def mutate(
        self, plan: "MutationPlan | Mapping[str, Any] | Sequence[Any]", *, dry_run: bool = False, source: str = "api"
    ) -> MutationReport:
        """Apply a plan atomically (12.5). ``dry_run=True`` simulates, stages, reports, installs nothing."""
        return self.engine.apply(plan, dry_run=dry_run, source=source)

    def preview_mutation(self, plan: "MutationPlan | Mapping[str, Any] | Sequence[Any]") -> MutationReport:
        return self.engine.apply(plan, dry_run=True, source="preview")

    def request_mutation(
        self, plan: "MutationPlan | Mapping[str, Any] | Sequence[Any]", *, source: str = "request"
    ) -> MutationPlan:
        """Queue a plan for the top of the next epoch. Parsed now, so a malformed plan fails here."""
        return self.engine.request(plan, source=source)

    def add_peer(self, path: str, *, reason: str = "", **spec: Any) -> MutationReport:
        return self.mutate(MutationPlan.of(_op_add(path, **spec), label=f"add:{path}", reason=reason))

    def modify_peer(self, path: str, *, reason: str = "", **changes: Any) -> MutationReport:
        return self.mutate(MutationPlan.of(_op_modify(path, **changes), label=f"modify:{path}", reason=reason))

    def retire_peer(self, path: str, *, recursive: bool = False, reason: str = "") -> MutationReport:
        return self.mutate(
            MutationPlan.of(_op_retire(path, recursive=recursive, reason=reason), label=f"retire:{path}", reason=reason)
        )

    def revive_peer(self, path: str, *, recursive: bool = False, reason: str = "") -> MutationReport:
        return self.mutate(
            MutationPlan.of(_op_revive(path, recursive=recursive, reason=reason), label=f"revive:{path}", reason=reason)
        )

    def purge_peer(
        self, path: str, *, recursive: bool = False, force: bool = False, reason: str = ""
    ) -> MutationReport:
        return self.mutate(
            MutationPlan.of(
                _op_purge(path, recursive=recursive, force=force, reason=reason), label=f"purge:{path}", reason=reason
            )
        )

    def mutation_history(self) -> list[dict[str, Any]]:
        """The journal: one entry per committed plan, oldest first, as stored in the checkpoint."""
        return [dict(entry) for entry in self.mutation_journal]

    def topology(self) -> dict[str, Any]:
        """What the tree is now: every node with its status, what is retired, what is tombstoned."""
        self._require_built()
        retired = self.freeze.retired()
        return {
            "topology_epoch": int(self.topology_epoch),
            "digest": structure_digest(self.tree, retired),
            "active": len(self.active_tree),
            "retired": sorted(retired),
            "tombstones": sorted(self._tombstones),
            "journal": len(self.mutation_journal),
            "pending": self.engine.pending(),
            "nodes": [
                {
                    "path": node.path, "role": node.role, "depth": node.depth, "parent": node.parent,
                    "status": node.status, "retired": node.path in retired,
                    "children": list(node.children),
                }
                for node in self.tree.walk("root-first")
            ],
        }

    def mutation_blockers(self) -> list[str]:
        """Reasons this trainer cannot be mutated at all (checked by ``MutationEngine._gate``)."""
        reasons: list[str] = []
        policy = self._trainer_policy if self._trainer_policy is not None else self.distill
        if self._distillation.enabled and not bool(getattr(policy, "supports_tree_mutation", False)):
            reasons.append(
                "DISTILLATION is enabled and CoDistillStep keeps a fixed cohort and fixed teacher "
                "assignments; a policy that can follow a changing tree sets `supports_tree_mutation = True` "
                "and implements `on_tree_mutated(trainer, report)`"
            )
        return reasons

    # -- the watchdog and the health baselines (12.8) ----------------------
    def _prune_node(self, path: str, dead_epochs: int = 0) -> int:
        """``PRUNE_BRANCH`` as a mutation. Returns the number of nodes that left play (0 if none).

        ``TRAINER.MUTATION.WATCHDOG_PRUNE`` decides how: ``purge`` (the default, and what
        ``remove_node`` always meant), ``retire`` (the branch keeps its weights and can be revived),
        or ``off`` (the watchdog only reports). A refusal (protected path, last active root, a
        distillation run) is recorded in ``failures`` and logged; training goes on.
        """
        node = self.tree.get(path)
        if node is None or path in self.freeze.retired():
            return 0
        mode = self.mutation_limits.watchdog_prune
        reason = f"dead branch: no proposals for {int(dead_epochs)} epoch(s)"
        if mode == "off":
            self.failures.append(f"watchdog: {path} is dead ({reason}) and WATCHDOG_PRUNE is off")
            return 0
        if mode == "retire":
            op = _op_retire(path, recursive=True, reason=reason)
        else:
            op = _op_purge(path, recursive=True, force=True, reason=reason)
        plan = MutationPlan.of(op, label=f"watchdog:{mode}:{path}", reason=reason)
        before = len(self.active_tree)
        try:
            self.engine.apply(plan, source="watchdog")
        except MutationError as error:
            if error.code == "E089" and not error.rolled_back:
                raise                       # the trainer is unreliable: the run ends (R9)
            self.failures.append(f"watchdog: could not {mode} {path}: {error}")
            log.error("watchdog could not %s %s: %s", mode, path, error)
            return 0
        return max(0, before - len(self.active_tree))

    def _after_mutation(self, report: MutationReport) -> None:
        """What the health machinery and the per-peer policies learn from a committed plan.

        Called by ``MutationEngine._finish`` before any user callback (12.7.6).
        """
        if report.dry_run or not report.changed:
            return
        # The first epoch measured on the new topology. At the top of epoch E (phase ``idle``) the
        # history holds E results and the next one is E's; in ``observe`` the epoch that just ran
        # is not in the history yet but belongs to the old topology, so the boundary is one later.
        self._loss_baseline = len(self.history) + (1 if self._phase == "observe" else 0)
        # ``_epoch_seconds`` already holds the epoch that just ran when this is called in ``observe``,
        # and not yet the one about to run when it is called in ``idle``; both give ``len``.
        self._rate_baseline = len(self._epoch_seconds)
        grown = tuple(dict.fromkeys((*report.added, *report.rebuilt)))
        if grown and self._initial_rss is not None:
            # New parameters, their gradients and optimizer moments are real, and a baseline
            # taken before them would call them a leak. Memory a purge frees is not credited
            # back: an allocator rarely returns it, so lowering the baseline would invite the
            # opposite false alarm.
            self._initial_rss += self._resident_bytes(grown)
        for policy in (self._trainer_policy, getattr(self._policy, "inner", self._policy)):
            hook = getattr(policy, "on_tree_mutated", None)
            if callable(hook):
                hook(self, report)
                break
        record = getattr(self.logger, "mutation", None)
        if callable(record):
            record(report)

    #: Bytes a parameter costs while training: the weight, its gradient, and two Adam moments.
    _TRAINING_BYTES_PER_PARAMETER_BYTE = 4

    def _resident_bytes(self, paths: Iterable[str]) -> int:
        total = 0
        for path in paths:
            peer = self.peers.get(path)
            module = getattr(getattr(peer, "adapter", None), "module", None)
            if module is None:
                continue
            total += sum(
                int(parameter.numel()) * int(parameter.element_size()) for parameter in module.parameters()
            )
        return total * self._TRAINING_BYTES_PER_PARAMETER_BYTE

    def _mutation_state(self) -> dict[str, Any]:
        """The checkpoint's account of how the tree came to be. ``{}`` for a run that never mutated."""
        pending = self.engine.pending()
        if not (
            self.mutation_journal or pending or self._node_specs or self._node_meta or self._tombstones
        ):
            return {}
        return MutationState(
            journal=tuple(copy.deepcopy(entry) for entry in self.mutation_journal),
            topology_epoch=int(self.topology_epoch),
            base_digest=self._base_digest,
            digest=structure_digest(self.tree, self.freeze.retired()),
            tombstones=tuple(sorted(self._tombstones)),
            specs=copy.deepcopy(self._node_specs),
            metadata=copy.deepcopy(self._node_meta),
            pending=tuple(pending),
        ).as_dict()

    def reconcile_topology(
        self, identity: Mapping[str, Any], tree_state: Mapping[str, Any], *, strict: bool = True
    ) -> list[str]:
        """Called by ``MACTFCheckpoint.restore`` before it compares topologies (12.9)."""
        from .mutation_reconcile import TopologyReconciler

        return TopologyReconciler(self).reconcile(identity, tree_state, strict=strict)

    def record_signal_sharing(self, signals_map: Mapping[str, float], *, epoch: int) -> None:
        """Called by ``CooperativeStrategy``: the cohort saw these signals."""
        self.signal_sharing.append(
            {"epoch": int(epoch), "signals": {str(k): float(v) for k, v in signals_map.items()}}
        )
        del self.signal_sharing[:-64]

    # ------------------------------------------------------------------
    # Views the strategies and the checkpoint read
    # ------------------------------------------------------------------
    @property
    def runtime(self) -> Runtime | None:
        return self._runtime

    @property
    def _built(self) -> bool:
        return self.built

    @property
    def peer_names(self) -> tuple[str, ...]:
        if self.tree is not None:
            return tuple(self.active_tree.root_paths)
        return tuple(self.peers)

    @property
    def kappa_controller(self) -> AdaptiveKappa | None:
        return self.controller

    @property
    def loss_function(self) -> LossFunction:
        if getattr(self, "_loss_function", None) is None:
            self._loss_function = self._resolve_loss()
        return self._loss_function

    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            optimizer_granularity="step",
            supports_amp=bool(self._runtime.amp) if self._runtime is not None else False,
            supports_hierarchy=True,
            supports_peft=self.finetuning is not None,
            supports_aggregation=True,
            deterministic=bool(getattr(self._runtime, "deterministic", False)),
            supports_distillation=self._distillation.enabled
            and importlib.util.find_spec("mactf.distill") is not None,
            distillation_conformance=self._distillation.enabled
            and bool(dict(self._distillation.conformance).get("ENABLED", True)),
        )

    def _require_built(self) -> None:
        if not self.built:
            self.build()

    @staticmethod
    def _rss() -> int | None:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001 - memory probing is best effort
            return None

    def _compliance_coefficient(self) -> float:
        """``TRAINER.COMPLIANCE_COEFFICIENT`` when written, else ``LossConfig.compliance_coef``."""
        trainer = dict(self.settings.get("TRAINER") or {})
        if "COMPLIANCE_COEFFICIENT" in trainer:
            return float(trainer["COMPLIANCE_COEFFICIENT"])
        return float(self._loss_config().compliance_coef)

    def _attach_policy(self) -> None:
        """Decide the one policy this run uses and attach it (document 17 section 7.5).

        Order, highest first: an in-code ``trainer=`` instance, the ``TRAINER.POLICY`` spec,
        distillation (``CoDistillStep``), the default. ``resolve_policy`` applies the unsafe
        flag, the guard and the distillation conflict rule. ``_trainer_policy`` keeps what the
        caller configured (``None`` for the default step); ``_policy`` is what ``train_step``
        calls.
        """
        from ..components.policy_resolver import resolve_policy

        choice = resolve_policy(
            code_instance=self._trainer_policy,
            settings=self.settings,
            distillation=self._distillation,
            slots=self.slots,
            host=self,
            distill_factory=self._build_distillation_step,
        )
        self._trainer_policy = choice.configured
        self._policy = choice.policy
        self._policy_origin = choice.origin
        self._policy_guard = choice.guard

    def _build_distillation_step(self) -> Any:
        """The former body of ``_attach_distillation``, returning the step."""
        from ..distill import CoDistillStep

        policy = CoDistillStep.from_settings(
            self._distillation,
            tree=self.tree,
            optimizers=self.optimizers,
            reference=self.reference,
            runtime=self._runtime,
            settings=self.settings,
        )
        policy.attach(self)
        return policy

    def _validate_functions(self) -> None:
        """``FunctionBook.validate`` against the tree and the policy that now exists."""
        in_code = [
            name
            for name, supplied in (
                ("synthesis", self._synthesis_spec is not None),
                ("rollup", self._hierarchical_scorer is not None),
                ("loss", self._loss_spec is not None),
            )
            if supplied
        ]
        self._function_warnings = self.functions.validate(
            self.node_paths,
            policy=self._policy,
            hierarchy=self._hierarchy is not None,
            loss_block_authored=self._loss_block_authored(),
            in_code=in_code,
        )

    def _loss_block_authored(self) -> bool:
        written = authored_keys(self.settings, "LOSS")
        if written is None:
            # A plain mapping carries no authorship record: any difference from the shipped
            # block is the evidence there is.
            return dict(self.settings.get("LOSS") or {}) != LOSS_DEFAULTS
        return bool(written)

    def _distillation_record(self) -> dict[str, Any]:
        """The JSON-safe resolved config for ``TrainSpec`` (``{}`` when off)."""
        if not self._distillation.enabled:
            return {}
        return _json_safe(self._distillation.as_dict())

    @property
    def distillation(self) -> DistillationConfig:
        """The resolved distillation config (``enabled`` is False for an off run)."""
        return self._distillation

    def _setup_custom_trainer(self) -> None:
        policy = self._policy
        setup = getattr(policy, "setup", None)
        if policy is None or not callable(setup):
            return
        for node in self.tree.walk("root-first"):
            setup(node)

    def resolve_node(self, node: Any) -> tuple[PeerNode | None, Peer]:
        """Public alias of ``_resolve_node``: part of the ``PolicyHost`` surface (7.2)."""
        return self._resolve_node(node)

    def backward(self, loss: torch.Tensor, optimizer: MACTFOptimizer) -> None:
        """Scale for accumulation and AMP, then ``backward``: the sanctioned way (7.2)."""
        self._backward(loss, optimizer)

    def loss_for(self, path: str) -> LossFunction:
        """The loss in force for ``path``: an ``RL.LOSS`` default or override, else ``LOSS``."""
        override = self.functions.get("loss", path)
        return override if override is not None else self.loss_function

    def synthesis_for(self, path: str) -> Any:
        """The synthesis function in force for the parent ``path``: an ``RL.SYNTHESIS``
        override or default, else the one built from ``synthesis=`` (``ConsensusSynthesis``
        when that is unset)."""
        chosen = self.functions.get("synthesis", path)
        return chosen if chosen is not None else self.synthesis_function

    def _resolve_node(self, node: Any) -> tuple[PeerNode | None, Peer]:
        if isinstance(node, PeerNode):
            if node.peer is None:
                raise ContractError(f"node {node.path!r} has no peer")
            return node, node.peer
        if isinstance(node, Peer):
            try:
                found = self.tree.get(node.name)
            except Exception:  # noqa: BLE001 - a Peer outside the tree is allowed
                found = None
            return found, node
        raise ContractError(
            f"train_step needs a PeerNode or a Peer, got {type(node).__name__}"
        )

    def _void(self, path: str) -> bool:
        cards = [card.peers[path] for card in self._epoch_cards if path in card.peers]
        return bool(cards) and all(card.is_void for card in cards)

    def _confidence(self, path: str) -> float:
        total = len(self._epoch_cards) or 1
        present = sum(1 for card in self._epoch_cards if path in card.peers)
        return present / total

    def peers_void(self, members: Sequence[str]) -> bool:
        """True when every member's work was voided this epoch."""
        return bool(members) and all(self._void(path) for path in members)

    @property
    def mean_score(self) -> float:
        if not self._epoch_cards:
            return 0.0
        values = [
            peer_card.final_score
            for path in self.active_tree.root_paths
            for card in self._epoch_cards
            for peer_card in [card.peers.get(path)]
            if peer_card is not None
        ]
        return sum(values) / len(values) if values else 0.0

    # ------------------------------------------------------------------
    # normalize_episode: a pure read
    # ------------------------------------------------------------------
    def normalize_episode(
        self,
        episode: EpisodeScorecard,
        *,
        policy: NormalizationPolicy | None = None,
    ) -> NormalizationReport:
        """Calibrate one scored episode's points without touching trainer state."""
        points: dict[str, float] = {}
        void: dict[str, bool] = {}
        for peer, card in episode.peers.items():
            raw = card.points
            value = None
            if isinstance(raw, Mapping):
                value = raw.get("final_points")
            elif raw is not None:
                value = getattr(raw, "final_points", None)
            if value is None:
                raise ContractError(
                    f"peer {peer!r} has no points on its scorecard; "
                    "COOPETITION.SIGNAL_SOURCE='points' needs the points layer enabled"
                )
            points[str(peer)] = float(value)
            void[str(peer)] = bool(card.is_void)
        normalizer = CohortNormalizer(policy) if policy is not None else self.normalizer
        return normalizer.normalize(points, episode=int(episode.episode), void=void)

    # ------------------------------------------------------------------
    # collect_episode: delegate, rollout, synthesize
    # ------------------------------------------------------------------
    def collect_episode(self, task: Task | None = None) -> Episode:
        """Steps 1-3 of the epoch order: delegate, roll out, synthesize."""
        self._require_built()
        assert self.tree is not None and self.runner is not None
        if task is None:
            task = self.task_source.next(self._epoch)
        episode = Episode(index=len(self.episodes), task=task)
        tree = self.active_tree

        # 1. Delegate, root-first.
        delegation_log: dict[str, dict[str, Any]] = {}
        selected_by_parent: dict[str, tuple[str, ...]] = {}
        if self.delegation_policy is not None:
            for node in tree.walk("root-first"):
                if not node.children:
                    continue
                chosen = tuple(
                    self.delegation_policy.select(
                        node,
                        task,
                        tree=tree,
                        scores=dict(self.signal_by_peer),
                        weights=dict(self._last_weights),
                        epoch=self._epoch,
                    )
                )
                selected_by_parent[node.path] = chosen
                delegation_log[node.path] = {
                    "selected": list(chosen),
                    "starved": [c for c in node.children if c not in chosen],
                }
        else:
            for node in tree.walk("root-first"):
                if node.children:
                    selected_by_parent[node.path] = tuple(node.children)

        # 2. Rollout, leaves-first, with failure isolation per node.
        failed: dict[str, str] = {}
        for node in tree.walk("leaves-first"):
            if node.peer is None or node.status == "failed":
                continue
            try:
                self.runner.run(episode, {node.path: node.peer})
            except Exception as error:  # noqa: BLE001 - isolated, recorded, penalised
                failed[node.path] = f"{type(error).__name__}: {error}"
                self._fail_node(episode, node, failed[node.path])

        # 3. Synthesize, leaves-first.
        if self.synthesis_function is not None:
            for node in tree.walk("leaves-first"):
                if not node.children or node.path in failed:
                    continue
                chosen = selected_by_parent.get(node.path, node.children)
                proposals = {
                    child: episode.participants[child].logits
                    for child in chosen
                    if child in episode.participants
                    and episode.participants[child].logits is not None
                    and child not in failed
                }
                if not proposals:
                    failed[node.path] = "every selected child failed"
                    self._fail_node(episode, node, failed[node.path])
                    continue
                synthesizer = self.synthesis_for(node.path)
                try:
                    synthesized = synthesizer.synthesize(
                        node,
                        proposals,
                        scores=dict(self.signal_by_peer),
                        weights=dict(self._last_weights),
                        task=task,
                    )
                except Exception as error:  # noqa: BLE001
                    failed[node.path] = f"{type(error).__name__}: {error}"
                    self._fail_node(episode, node, failed[node.path])
                    continue
                participant = episode.participants.get(node.path)
                if participant is not None:
                    participant.proposal["synthesized_from"] = sorted(proposals)
                    if (
                        participant.logits is not None
                        and tuple(synthesized.shape) == tuple(participant.logits.shape)
                    ):
                        participant.logits = synthesized.detach()

        # Verifiable rules, recorded by the task judge exactly as ``debate`` does.
        judge = TaskJudge()
        for path, participant in episode.participants.items():
            if participant.logits is None:
                continue
            judge.judge(
                participant.ledger,
                logits=participant.logits,
                targets=task.targets,
                peer=path,
            )

        episode.delegation = delegation_log  # type: ignore[attr-defined]
        episode.failures = dict(failed)  # type: ignore[attr-defined]
        self._pending = episode
        return episode

    def _episode_evidence(self, episode: Episode) -> dict[str, Any]:
        """Per-node evidence for the points layer.

        ``evidence_provider(episode, path)`` supplies rich evidence (an
        ``EpisodeEvidence``, a mapping, or an ``EvaluationSubject``). Without one,
        the only evidence the trainer itself observes is the task judge's verdict,
        so ``task_checks_passed`` is derived from whether R1 was recorded, and
        nothing else is asserted: every other guard then sees its evidence as absent
        and reports it instead of guessing.
        """
        from ..scoring.evidence import EpisodeEvidence

        evidence: dict[str, Any] = {}
        for path, ledger in episode.ledgers.items():
            supplied = (
                self.evidence_provider(episode, path) if self.evidence_provider else None
            )
            if supplied is None:
                supplied = EpisodeEvidence(
                    task_checks_passed="R1" in ledger.triggered_ids()
                    if path in episode.participants
                    else None
                )
            evidence[path] = supplied
        return evidence

    def _fail_node(self, episode: Episode, node: PeerNode, reason: str) -> None:
        """A failed node is marked, penalised, and skipped for the epoch."""
        self._mark_node(node.path, "failed")
        episode.ledger_for(node.path).record(
            "P23", magnitude=1.0, evidence=reason, source="trainer"
        )
        self.failures.append(f"{node.path}: {reason}")

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------
    def score(self) -> EpisodeScorecard:
        """Step 4: score the episode collected for this epoch."""
        self._require_built()
        episode = self._pending
        if episode is None:
            raise ContractError("score() needs collect_episode() to run first")
        scorer = self.scorer
        # Points are always derived beside final_score (they never feed it), from the one
        # fact the trainer observes itself; richer evidence comes from ``evidence_provider``.
        evidence = self._episode_evidence(episode)
        if isinstance(scorer, HierarchicalScorer):
            card = scorer.score_tree(
                self.active_tree, episode.ledgers, episode=episode.index, evidence=evidence
            )
        else:
            card = scorer.score_episode(
                episode.ledgers, episode=episode.index, evidence=evidence
            )
        episode.scorecard = card
        self.scorecards.append(card)
        self.episodes.append(episode)
        self._epoch_cards.append(card)
        self._pending = None

        self._normalization = None
        if self.signal_source == "points":
            report = self.normalize_episode(card)
            self._normalization = report
            self.signal_by_peer = report.signals()
        else:
            totals: dict[str, list[float]] = {}
            for each in self._epoch_cards:
                for peer, peer_card in each.peers.items():
                    totals.setdefault(peer, []).append(peer_card.final_score)
            self.signal_by_peer = {
                peer: sum(values) / len(values) for peer, values in totals.items() if values
            }
        self._dispatch_many("on_score", self, card)
        return card

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------
    def _make_batch(self, node: PeerNode) -> dict[str, Any]:
        episode = self.episodes[-1]
        runtime = self._runtime
        states = runtime.to(episode.task.states)
        targets = episode.task.targets.to(states.device)
        peer_card = self._epoch_cards[-1].peers.get(node.path) if self._epoch_cards else None
        return {
            "states": states,
            "targets": targets,
            "node_path": node.path,
            "signal": float(self.signal_by_peer.get(node.path, 0.0)),
            "void": bool(peer_card.is_void) if peer_card is not None else False,
            "floor": float(peer_card.floor) if peer_card is not None else 0.0,
        }

    def train_step(self, node: Any, batch: Mapping[str, Any]) -> StepResult:
        """One gradient step for one node against its own card."""
        self._require_built()
        tree_node, peer = self._resolve_node(node)
        path = peer.name
        optimizer = self.optimizers.get(path)
        if optimizer is None:
            raise ContractError(f"no optimizer is bound for node {path!r}")
        policy = self._policy
        handler = getattr(policy, "train_step", None)
        if callable(handler):
            raw = handler(tree_node or peer, batch)
        elif callable(policy):
            raw = policy(tree_node or peer, batch, trainer=self)
        else:
            raise ContractError("trainer must be a TrainerProtocol or a callable")
        result = coerce_step_result(raw, node=tree_node or peer)
        if result.loss.requires_grad and not getattr(policy, "self_stepping", False):
            result = self._apply_custom_loss(result, optimizer)
            judge = getattr(policy, "after_step", None)
            if callable(judge):
                # A guarded policy checks the step the trainer just took on its behalf.
                result = judge(tree_node or peer, result)
        if result.metrics and (result.metrics.get("pending") or result.metrics.get("quarantined")):
            # A rendezvous policy has no cohort result yet, and a quarantined step has no
            # loss to record; recording either placeholder would pollute the history.
            return result
        self.global_step += 1
        self.losses.append(result.loss_value)
        grad_norm = result.metrics.get("grad_norm") if result.metrics else None
        if grad_norm is not None:
            self.grad_norms.append(float(grad_norm))
        self._epoch_step_losses.append(result.loss_value)
        entry = self._epoch_node_metrics.setdefault(path, {"loss": [], "grad_norm": []})
        entry["loss"].append(result.loss_value)
        if grad_norm is not None:
            entry["grad_norm"].append(float(grad_norm))
        return result

    def _backward(self, loss: torch.Tensor, optimizer: MACTFOptimizer) -> None:
        scaled = loss / optimizer.grad_accumulation if optimizer.grad_accumulation > 1 else loss
        if optimizer.scaler is not None:
            optimizer.scaler.scale(scaled).backward()
        else:
            scaled.backward()

    def _apply_custom_loss(
        self, result: StepResult, optimizer: MACTFOptimizer
    ) -> StepResult:
        """A custom step that returns a live loss leaves backward and step to us."""
        if getattr(optimizer, "_accumulated", 0) == 0:
            optimizer.zero_grad()
        self._backward(result.loss, optimizer)
        outcome = optimizer.step()
        metrics = dict(result.metrics)
        metrics["lr"] = float(outcome.lr)
        if outcome.grad_norm is not None:
            metrics["grad_norm"] = float(outcome.grad_norm)
        return replace(
            result,
            loss=result.loss.detach(),
            stepped=bool(outcome.stepped),
            metrics=metrics,
        )

    def _stamp_policy_metadata(self) -> None:
        """Write ``metadata["policy"]`` (and ``["distillation"]``) on this epoch's cards.

        Populated by the trainer after the step, never by a scorer: scoring runs before the
        cohort step, so a scorer could only report the previous step. ``metadata["policy"]``
        is written for every policy that implements ``card_metadata`` and for an unsafe one
        (so a card says the step ran unguarded); ``metadata["distillation"]`` is unchanged.
        """
        if not self._epoch_cards:
            return
        policy = self._policy
        recorder = getattr(policy, "card_metadata", None)
        unsafe = bool(getattr(policy, "unsafe", False))
        distill = self.distill
        distill_recorder = getattr(distill, "card_metadata", None)
        if not callable(recorder) and not unsafe and not (
            self._distillation.enabled and callable(distill_recorder)
        ):
            return
        for path, peer_card in self._epoch_cards[-1].peers.items():
            if self._distillation.enabled and callable(distill_recorder):
                peer_card.metadata["distillation"] = distill_recorder(path)
            if callable(recorder) and policy is not distill:
                payload = dict(recorder(path))
                if unsafe:
                    payload["unsafe"] = True
                peer_card.metadata["policy"] = payload
            elif unsafe:
                peer_card.metadata["policy"] = {
                    "name": getattr(policy, "name", type(policy).__name__), "unsafe": True,
                }

    def _apply_pending_swaps(self) -> None:
        """Apply every queued swap an episode boundary permits (functions, then optimizers)."""
        self.slots.episode = self._epoch
        if self.optimizer_plan is not None:
            self.optimizer_plan.episode = self._epoch
        self.functions.apply_pending(Boundary.EPISODE)
        if self.optimizer_plan is not None and self.optimizer_plan.pending_slots():
            self._apply_pending_optimizers()

    def _train_epoch(self) -> int:
        """Step 5: walk leaves-first, one or more updates per node."""
        steps = 0
        episode = self.episodes[-1]
        for node in self.active_tree.walk("leaves-first"):
            if node.peer is None or node.status == "failed" or self.freeze.skips_training(node.path):
                continue
            if node.path not in episode.participants:
                continue
            for _ in range(self.spec.steps_per_epoch):
                batch = self._make_batch(node)
                replacement = self._dispatch_many("on_batch", self, node, batch)
                if replacement is not None:
                    batch = replacement
                result = self.train_step(node, batch)
                replaced = self._dispatch_many("on_step", self, node, result)
                if isinstance(replaced, StepResult):
                    result = replaced
                self.strategy_plan.on_peer_step(self, node, result)
                steps += 1
        return steps

    # ------------------------------------------------------------------
    # update_weights, aggregate, aggregate_group
    # ------------------------------------------------------------------
    def _group_aggregator(self, parent: str | None) -> Aggregator:
        if parent is None:
            return self.aggregator
        cached = self._group_aggregators.get(parent)
        if cached is not None:
            return cached
        controller = self._level_controllers.get(parent)
        if controller is None:
            children = len(self.active_tree[parent].children)
            controller = self.kappa_plan.build(
                parent, dict(self._coopetition_settings()), num_peers=max(1, children)
            )
            self._level_controllers[parent] = controller
            self.controllers[parent] = controller
        aggregator = self.aggregator.clone_with(controller)
        self._group_aggregators[parent] = aggregator
        return aggregator

    def _weights_for(
        self,
        parent: str | None,
        members: Sequence[str],
        aggregator: Aggregator,
        strategy: WeightUpdateStrategy,
        external_eligible: Mapping[str, bool] | None,
    ) -> AggregationReport:
        vectors = {name: self.peers[name].export_parameters() for name in members}
        scores = {name: float(self.signal_by_peer.get(name, 0.0)) for name in members}
        # A frozen peer carries no weight whoever asked (document 17 section 11.4): the freeze
        # set can only shrink the eligible set. ``None`` when nobody is frozen keeps the call
        # exactly what it was.
        frozen = self.freeze.eligible_for(list(members))
        eligible = {
            name: (not self._void(name))
            and bool((external_eligible or {}).get(name, True))
            and bool((frozen or {}).get(name, True))
            for name in members
        }
        confidence = {name: self._confidence(name) for name in members}
        level = 0 if parent is None else int(self.tree[parent].depth)
        controller = aggregator.controller

        def run(chosen: WeightUpdateStrategy) -> AggregationReport:
            if type(chosen) is KappaSoftmaxAverage:
                given: tuple[Any, ...] = (vectors, scores, eligible, confidence)
            else:
                # Custom code gets its own copies: the guard judges its report against
                # ``eligible`` and the fallback re-runs on ``vectors``, so a strategy that
                # edits what it was handed must not be editing the yardstick.
                given = (
                    {name: vector.clone() for name, vector in vectors.items()},
                    dict(scores), dict(eligible), dict(confidence),
                )
            return chosen.apply(
                given[0],
                given[1],
                kappa=float(controller.kappa),
                level=level,
                eligible=given[2],
                confidence=given[3],
            )

        reason: str | None = None
        report: AggregationReport | None = None
        is_default = isinstance(strategy, KappaSoftmaxAverage)
        guard = self.weight_guard
        if guard.covers(strategy, aggregator):
            snapshot = controller.snapshot()
            outcome = guard.run(
                strategy, run,
                names=sorted(members), eligible=eligible,
                floor=aggregator.weight_floor, cap=aggregator.weight_cap,
            )
            if outcome.action == "fallback":
                controller.restore(snapshot)                       # invariant A7
                reason = "; ".join(str(item) for item in outcome.violations)
            else:
                report = outcome.report
        else:
            try:
                report = run(strategy)
            except (ComplianceError, NotImplementedError) as error:
                if is_default:
                    raise
                reason = str(error) or type(error).__name__
            # An intentional reset (``reference_only``) is not a failure (W-G5).
            if (report is not None and report.fallback and not is_default
                    and not traits_of(strategy).may_fallback):
                reason = "it reported fallback"
        if reason is not None:
            name = getattr(strategy, "name", type(strategy).__name__)
            trusted = aggregator if type(aggregator) is Aggregator else aggregator.pristine()
            report = run(KappaSoftmaxAverage(trusted))
            report.adjustments.append(
                f"weight_update {name!r} failed: {reason}; fell back to kappa_softmax"
            )
        assert report is not None
        report.signal_source = self.signal_source
        report.normalization = (
            self._normalization if self.signal_source == "points" else None
        )
        self._last_weights = dict(report.weights) if parent is None else self._last_weights
        return report

    def update_weights(
        self, eligible: Mapping[str, bool] | None = None
    ) -> AggregationReport:
        """Step 6a: the root group's weights, with the documented fallback rule."""
        self._require_built()
        members = tuple(self.active_tree.root_paths)
        strategy = self.weight_plan.strategy_for(ROOT_LABEL, self.aggregator)
        self.weight_strategy = strategy
        report = self._weights_for(None, members, self.aggregator, strategy, eligible)
        self.reports.append(report)
        self._epoch_report = report
        return report

    def _write(
        self,
        aggregator: Aggregator,
        members: Sequence[str],
        report: AggregationReport,
        *,
        reference: Peer | None,
        reference_ema: float,
    ) -> AggregateResult:
        self.write_count += 1
        adapters = {name: self.peers[name].adapter for name in members}
        # Readers are every member; writers are the non-frozen ones (document 17 section 11.4).
        targets = {name: adapter for name, adapter in adapters.items()
                   if not self.freeze.skips_write(name)}
        if not targets:
            report.adjustments.append("every member is frozen; nothing was written")
            log.warning("W032: group %s skipped: every member is frozen", sorted(members))
            first = next(iter(adapters.values()))
            return AggregateResult(parameters=first.export_parameters(), report=report)
        positive = any(weight > 0.0 for weight in report.weights.values())
        if not positive and not report.fallback:
            first = next(iter(adapters.values()))
            return AggregateResult(parameters=first.export_parameters(), report=report)
        if report.fallback and reference is None:
            report.adjustments.append(
                "group fell back with no reference to restore; parameters unchanged"
            )
            first = next(iter(adapters.values()))
            return AggregateResult(parameters=first.export_parameters(), report=report)
        if not report.fallback:
            # The one place parameters are written: whoever produced ``report`` (a strategy
            # that built its own, a legacy ``on_epoch_end`` calling ``aggregate(report)``),
            # a void or frozen peer must not be in the blend.
            kept = self.freeze.eligible_for(list(members)) or {}
            barred = [
                name for name in members
                if float(report.weights.get(name, 0.0)) > 0.0
                and (self._void(name) or not bool(kept.get(name, True)))
            ]
            if barred:
                raise WeightGuardViolation("aggregate", [
                    WeightViolation("U004", name,
                                    f"carries weight {float(report.weights[name]):.6g} but is "
                                    "void or frozen; it is never part of a blend")
                    for name in barred
                ])
        reference_adapter = reference.adapter if reference is not None else None
        custom_blend = type(aggregator).aggregate is not Aggregator.aggregate
        # A custom ``aggregate`` is shown the peers that carry weight and no others: a blend
        # that loops over "every adapter" must not be able to read a void peer's parameters.
        shown = (
            {name: adapter for name, adapter in adapters.items()
             if float(report.weights.get(name, 0.0)) > 0.0}
            if custom_blend and not report.fallback else adapters
        )
        result = aggregator.aggregate(
            shown, report, reference=reference_adapter, device=self._runtime.device
        )
        if custom_blend and not torch.isfinite(torch.as_tensor(result.parameters)).all():
            raise ComplianceError(
                f"aggregator {type(aggregator).__name__} returned non-finite parameters; "
                "nothing was written"
            )
        aggregator.apply(
            targets,
            result,
            reference=reference_adapter,
            reference_ema=reference_ema if reference is not None else 0.0,
            reset_optimizers=False,
            optimizers=None,
        )
        if bool(self._coopetition_settings().get("RESET_OPTIMIZERS", True)):
            for name in targets:
                _clear_optimizer_state(self.optimizers.get(name))
        return result

    def aggregate(
        self, report: AggregationReport | None = None, *, update_reference: bool = True
    ) -> AggregateResult:
        """Step 6b/7: write the root group's blend. ``update_reference=False`` leaves the EMA alone."""
        self._require_built()
        report = report or self._epoch_report or self.update_weights()
        ema = float(self._coopetition_settings().get("REFERENCE_EMA", 0.05))
        result = self._write(
            self.aggregator,
            tuple(self.active_tree.root_paths),
            report,
            reference=self.reference,
            reference_ema=ema if update_reference else 0.0,
        )
        self._dispatch_many("on_aggregate", self, report)
        return result

    def aggregate_group(
        self,
        parent: str | None,
        members: Sequence[str],
        *,
        eligible: Mapping[str, bool] | None = None,
    ) -> AggregationReport:
        """Aggregate one child group. The reference is never touched here."""
        self._require_built()
        aggregator = self._group_aggregator(parent)
        strategy = self.weight_plan.strategy_for(group_label(parent), aggregator)
        report = self._weights_for(parent, tuple(members), aggregator, strategy, eligible)
        self.reports.append(report)
        self._write(
            aggregator, tuple(members), report, reference=None, reference_ema=0.0
        )
        self._dispatch_many("on_aggregate", self, report)
        return report

    def _coordinate(self, epoch: int) -> AggregationReport | None:
        """Step 6: let the strategy plan mix (or not) at the epoch's end."""
        return self.strategy_plan.run(self, epoch, self._tree_view())

    # ------------------------------------------------------------------
    # Weight plan, strategy plan: views, validation and live changes (document 17 sections 10, 11)
    # ------------------------------------------------------------------
    def _group_views(self) -> list[GroupView]:
        coopetition = self._coopetition_settings()
        floor = float(coopetition.get("WEIGHT_FLOOR", 0.02))
        cap = float(coopetition.get("WEIGHT_CAP", 0.6))
        views: list[GroupView] = []
        for parent, members in aggregation_groups(self.active_tree):
            views.append(GroupView(
                label=group_label(parent), parent=parent, members=tuple(members),
                level=0 if parent is None else int(self.tree[parent].depth),
                floor=floor, cap=cap,
            ))
        return views

    def _validate_weight_plan(self, *, initial: bool) -> None:
        views = self._group_views()
        labels = [view.label for view in views]
        notes = self.weight_plan.validate(views, initial=initial)
        notes += self.kappa_plan.validate(labels, initial=initial)
        self.weight_warnings = notes
        self.failures.extend(note for note in notes if note.startswith("W022"))

    def _tree_view(self) -> TreeView:
        return tree_view(self.active_tree, hierarchy=self._hierarchy is not None)

    def _validate_strategy_plan(self, *, initial: bool) -> None:
        view = self._tree_view()
        plan = self.strategy_plan
        notes = plan.validate(view, initial=initial)
        unserved = [
            group.label for group in view.groups
            if not group.is_root
            and plan.select(group.label) == DEFAULT_PATH
            and not plan.reaches(DEFAULT_PATH, plan.default_instance(), group)
        ]
        if unserved and view.hierarchy:
            notes.append(
                f"W035: groups {unserved} are served by no strategy: the default "
                f"{type(plan.default_instance()).__name__} walks the root only; set "
                "TRAINER.PEER_STRATEGY_OVERRIDES or use a strategy with walk='tree'"
            )
        if self.freeze.mode == "legacy" and any(
            isinstance(instance, FrozenPeerStrategy) for instance in plan.live_instances()
        ):
            notes.append(
                "W034: FrozenPeerStrategy under TRAINER.FREEZE_MODE='legacy': frozen peers still take "
                "training steps and receive aggregated parameters; set FREEZE_MODE='strict' to freeze them"
            )
        self.strategy_warnings = notes
        for note in notes:
            if note.startswith(("W034", "W035")):
                log.warning(note)

    def swap_weight_update(
        self,
        glob: str | None,
        spec: Any,
        *,
        boundary: Boundary = Boundary.EPISODE,
        mode: str | None = None,
    ) -> SwapRecord | None:
        """Replace the default strategy (``glob=None``), replace an override, or add one. Now.

        ``spec`` is a name, a mapping, a ``ComponentSpec`` or an in-code instance. Atomic: on
        any failure the old strategy, its per-group state and the cache are untouched.
        """
        self._require_built()
        record = self.weight_plan.set(
            glob, spec, views=self._group_views(), boundary=boundary, mode=mode
        )
        self.weight_strategy = self.weight_plan.strategy_for(ROOT_LABEL, self.aggregator)
        self._validate_weight_plan(initial=False)
        return record

    def remove_weight_override(
        self, glob: str, *, boundary: Boundary = Boundary.EPISODE
    ) -> SwapRecord:
        self._require_built()
        record = self.weight_plan.remove(glob, views=self._group_views(), boundary=boundary)
        self.weight_strategy = self.weight_plan.strategy_for(ROOT_LABEL, self.aggregator)
        self._validate_weight_plan(initial=False)
        return record

    def request_weight_update(self, glob: str | None, spec: Any, *, mode: str | None = None) -> None:
        """Queue a replacement for the next episode boundary; an invalid spec fails *now*."""
        self._require_built()
        self.weight_plan.request(glob, spec, mode=mode)

    def request_remove_weight_override(self, glob: str) -> None:
        self._require_built()
        self.weight_plan.request_remove(glob)

    def _apply_pending_weights(self) -> None:
        for glob, value, mode in self.weight_plan.take_pending():
            try:
                if self.weight_plan.is_removal(value):
                    self.remove_weight_override(str(glob))
                else:
                    self.swap_weight_update(glob, value, mode=mode)
            except Exception as error:  # noqa: BLE001 - a failed queued swap must not end the run
                self.failures.append(f"queued weight_update change {glob!r} failed: {error}")
                log.error("queued weight_update change %r failed: %s", glob, error)

    def legal_report(
        self, report: AggregationReport, parent: str | None, members: Sequence[str]
    ) -> list[str]:
        """Problems that make ``report`` unusable for the group (``Y001``). Empty when legal."""
        aggregator = self._group_aggregator(parent)
        eligible = self.freeze.eligible_for(list(members)) or {}
        flags = {name: (not self._void(name)) and bool(eligible.get(name, True)) for name in members}
        inspector = WeightGuard(mode="fallback", scope="all", tolerance=self.weight_guard.tolerance)
        found = inspector.inspect(
            report, names=sorted(members), eligible=flags,
            floor=aggregator.weight_floor, cap=aggregator.weight_cap,
            traits=WeightTraits(may_fallback=True),
        )
        return [str(item) for item in found]

    def peer_digest(self, path: str) -> str:
        """A digest of one peer's parameters; the guard compares it around a call (``Y005``)."""
        import hashlib

        node = self.tree[path]
        if node.peer is None:
            return "none"
        digest = hashlib.sha256()
        parameters = node.peer.export_parameters()
        if isinstance(parameters, Mapping):
            for name, tensor in sorted(parameters.items()):
                digest.update(str(name).encode("utf-8"))
                digest.update(tensor.detach().cpu().numpy().tobytes())
        else:
            digest.update(parameters.detach().cpu().numpy().tobytes())
        return digest.hexdigest()[:16]

    def swap_peer_strategy(
        self,
        glob: str | None,
        spec: Any,
        *,
        boundary: Boundary = Boundary.EPISODE,
        mode: str | None = None,
    ) -> SwapRecord | None:
        """Replace the default strategy (``glob=None``), replace one override, or add an override. Now.

        ``spec`` is a name, a mapping, a ``ComponentSpec`` or an in-code instance. Atomic: any
        failure leaves the old strategy, its state and the freeze set untouched.
        """
        self._require_built()
        record = self.strategy_plan.set(
            glob, spec, view=self._tree_view(), host=self, boundary=boundary, mode=mode
        )
        self.strategy = self.strategy_plan.default_instance()
        self.failures.extend(self.strategy_plan.drain_errors())
        self._validate_strategy_plan(initial=False)
        return record

    def remove_peer_override(
        self, glob: str, *, boundary: Boundary = Boundary.EPISODE
    ) -> SwapRecord:
        self._require_built()
        record = self.strategy_plan.remove(glob, view=self._tree_view(), host=self, boundary=boundary)
        self.strategy = self.strategy_plan.default_instance()
        self.failures.extend(self.strategy_plan.drain_errors())
        self._validate_strategy_plan(initial=False)
        return record

    def request_peer_strategy(self, glob: str | None, spec: Any, *, mode: str | None = None) -> None:
        """Queue a replacement for the next episode boundary; an invalid spec fails *now*."""
        self._require_built()
        self.strategy_plan.request(glob, spec, mode=mode)

    def request_remove_peer_override(self, glob: str) -> None:
        self._require_built()
        self.strategy_plan.request_remove(glob)

    def _apply_pending_strategies(self) -> None:
        for glob, value, mode in self.strategy_plan.take_pending():
            try:
                if self.strategy_plan.is_removal(value):
                    self.remove_peer_override(str(glob))
                else:
                    self.swap_peer_strategy(glob, value, mode=mode)
            except Exception as error:  # noqa: BLE001 - a failed queued change must not end the run
                self.failures.append(f"queued peer_strategy change {glob!r} failed: {error}")
                log.error("queued peer_strategy change %r failed: %s", glob, error)

    # ------------------------------------------------------------------
    # run_epoch and fit
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> EpochResult:
        """One epoch in the fixed order of document 08 section 5.2."""
        self._require_built()
        if self.engine.poisoned is not None:
            raise self.engine.poisoned              # a failed rollback ended this trainer (R9)
        self._phase = "idle"
        self._epoch = int(epoch)                    # a plan journaled below names this epoch (12.7.4)
        self._mutations_in_cycle = 0
        self.engine.apply_pending(int(epoch))       # queued plans, then this epoch's schedule entries
        started = time.monotonic()
        self._epoch_cards = []
        self._epoch_step_losses = []
        self._epoch_node_metrics = {}
        self._normalization = None
        self._pending = None
        for node in self.tree.walk("root-first"):
            if node.status == "failed":
                self.tree.mark(node.path, "active")
        self.freeze.sync()          # a frozen peer that failed last epoch is frozen again
        self._active_view = None    # an ancestor may hold a stale copy of a status

        self._apply_pending_swaps()
        self._apply_pending_weights()          # 10.8
        self._apply_pending_strategies()       # 11.9
        self._phase = "epoch"
        try:
            return self._run_epoch_body(epoch, started)
        finally:
            self._phase = "idle"

    def _run_epoch_body(self, epoch: int, started: float) -> EpochResult:
        self._dispatch_many("on_epoch_start", self, epoch)
        self.strategy_plan.epoch_start(self, epoch, self._tree_view())
        self.freeze.check_learners(
            node.path for node in self.active_tree.walk("root-first")
            if node.peer is not None and node.status != "failed"
        )                                       # E075 when nobody would learn
        task = self.task_source.next(epoch)
        self.collect_episode(task)
        self.score()
        steps = self._train_epoch()
        self._stamp_policy_metadata()
        report = self._coordinate(epoch)

        losses = self._epoch_step_losses
        metrics: dict[str, dict[str, float]] = {}
        card = self._epoch_cards[-1]
        for path, entry in self._epoch_node_metrics.items():
            row = {"loss": sum(entry["loss"]) / len(entry["loss"])}
            if entry["grad_norm"]:
                row["grad_norm"] = entry["grad_norm"][-1]
            row["signal"] = float(self.signal_by_peer.get(path, 0.0))
            if path in card.peers:
                row["final_score"] = float(card.peers[path].final_score)
            metrics[path] = row
        if report is not None:
            metrics["aggregation"] = {
                "kappa": float(report.kappa),
                "entropy": float(report.entropy),
            }
        duration = time.monotonic() - started
        self._epoch_seconds.append(duration)
        result = EpochResult(
            epoch=int(epoch),
            steps=int(steps),
            mean_loss=sum(losses) / len(losses) if losses else 0.0,
            metrics=metrics,
            completed=True,
        )
        self._phase = "observe"
        if self._policy is not None and callable(getattr(self._policy, "on_epoch_end", None)):
            self._policy.on_epoch_end(self.active_tree, result)
        self._dispatch_many("on_epoch_end", self, result)
        self._observe(result, report)
        self._previous_scores = dict(self.signal_by_peer)
        return result

    def _observe(self, result: EpochResult, report: AggregationReport | None) -> None:
        """Step 8: log, snapshot (no watchdog), then let the watchdog judge."""
        if self.logger is not None:
            fields: dict[str, Any] = {}
            if report is not None:
                fields = {"kappa": report.kappa, "entropy": report.entropy}
            self.logger.epoch(result, **fields)
        every = int(self._training_settings().get("CHECKPOINT_EVERY", 1) or 0)
        writer = self.checkpoint
        if writer is not None and every and (result.epoch + 1) % every == 0:
            writer.save(
                CheckpointContext(
                    trainer=self,
                    episode=result.epoch,
                    global_step=self.global_step,
                    score=self.mean_score,
                    kappa=float(self.controller.kappa) if self.controller else 0.0,
                    entropy=float(self.controller.entropy) if self.controller else 0.0,
                    aggregations=tuple(self.reports),
                    # Document 17 section 14: without this a resumed run rebuilt the
                    # component set of the settings, not the one the file was written with.
                    identity_extra=self._identity_extra(),
                ),
                reason="periodic",
            )
        if self.watchdog is None:
            return
        self._last_health = self.watchdog.tick(self._health_context(result))
        if self.watchdog.should_stop() and self._last_health.action.stops_training:
            self.request_stop(self._last_health.describe())

    @property
    def context(self) -> "HealthContext":
        """The health context of the most recent epoch: what ``watchdog.tick`` consumes."""
        if not self.history:
            raise ContractError("trainer.context needs at least one completed epoch; call fit() first")
        return self._health_context(self.history[-1])

    def _health_context(self, result: EpochResult) -> "HealthContext":
        from .watchdog import HealthContext

        controller = self.controller
        active = self.active_tree
        roots = max(1, len(active.root_paths))
        elapsed = self._epoch_seconds[-1] if self._epoch_seconds else 0.0
        previous_rate = None
        if (
            len(self._epoch_seconds) > 1
            and self._epoch_seconds[-2] > 0
            and len(self._epoch_seconds) - 2 >= self._rate_baseline     # the previous epoch ran on this topology
        ):
            previous_rate = result.steps / self._epoch_seconds[-2]
        escalated = sum(
            len(peer_card.warnings_escalated)
            for card in self._epoch_cards
            for peer_card in card.peers.values()
        )
        dead: dict[str, int] = {}
        in_play: set[str] = set()
        for node in active.walk("root-first"):
            in_play.add(node.path)
            if node.status == "failed":
                dead[node.path] = self._dead_counts.get(node.path, 0) + 1
        self._dead_counts = dict(dead)
        count = 0
        if self.checkpoint is not None:
            try:
                count = len(self.checkpoint.list())
            except Exception:  # noqa: BLE001 - a missing directory is zero checkpoints
                count = 0
        retired = self.freeze.retired()
        return HealthContext(
            epoch=int(result.epoch),
            step=int(self.global_step),
            losses=tuple(item.mean_loss for item in self.history) + (result.mean_loss,),
            grad_norms=tuple(self.grad_norms),
            scores={path: value for path, value in self.signal_by_peer.items() if path in in_play},
            previous_scores=dict(self._previous_scores),
            entropy=float(controller.entropy) if controller else None,
            max_entropy=math.log(roots) if roots > 1 else 0.0,
            kappa=float(controller.kappa) if controller else None,
            aggregation_reports=tuple(self.reports),
            warnings_escalated=int(escalated),
            checkpoint_count=int(count),
            epoch_seconds=tuple(self._epoch_seconds),
            steps_per_second=(result.steps / elapsed) if elapsed > 0 else None,
            previous_steps_per_second=previous_rate,
            rss_bytes=self._rss(),
            initial_rss_bytes=self._initial_rss,
            tree_nodes=len(active),
            initial_tree_nodes=self._initial_nodes,
            dead_branches=dead,
            level_durations=dict(self._level_durations),
            extras={"tree": self.tree},
            loss_baseline=int(self._loss_baseline),
            retired_nodes=len(retired),
            mutations=len(self.mutation_journal),
            topology_epoch=int(self.topology_epoch),
            pruner=self._prune_node,
        )

    def fit(
        self,
        epochs: int | None = None,
        data: Any | None = None,
        *,
        resume: bool = False,
        callbacks: Sequence[Callback] | None = None,
    ) -> TrainResult:
        """The training loop (document 08 section 5.2)."""
        self.build()
        if data is not None:
            if not callable(getattr(data, "next", None)):
                raise ImproperlyConfigured("fit(data=...) needs an object with next(epoch)")
            self.task_source = data
        for callback in callbacks or ():
            self.add_callback(callback)
        total = int(epochs if epochs is not None else self.spec.epochs)
        if total < 0:
            raise ValidationError(f"epochs must be >= 0, got {total}")
        if resume:
            self._resume()
        self.stop_requested = False
        self.stop_reason = ""
        self._started_at = time.monotonic()
        self._closed = False

        results: list[EpochResult] = []
        status, reason = "completed", ""
        try:
            for epoch in range(self.start_epoch, total):
                result = self.run_epoch(epoch)
                results.append(result)
                self.history.append(result)
                if self.stop_requested:
                    health = self._last_health
                    if health is not None and health.action.stops_training:
                        status = health.action.value
                    else:
                        status = "stopped"
                    reason = self.stop_reason
                    break
        finally:
            self.close()
        steps = sum(item.steps for item in results)
        weighted = sum(item.mean_loss * item.steps for item in results)
        controller = self.controller
        return TrainResult(
            epochs=len(results),
            steps=steps,
            mean_loss=(weighted / steps) if steps else 0.0,
            history=tuple(results),
            episodes=tuple(self.scorecards),
            status=status,
            reason=reason,
            metrics={
                "global_step": self.global_step,
                "kappa": float(controller.kappa) if controller else None,
                "entropy": float(controller.entropy) if controller else None,
                "seconds": time.monotonic() - self._started_at,
            },
            component_notes=tuple(self.component_notes),
        )

    def _resume(self) -> ResumeReport:
        if self.checkpoint is None:
            raise ImproperlyConfigured("resume=True needs a checkpoint writer")
        report = self.checkpoint.resume_from_last_checkpoint(
            trainer=self, strict=self.strict
        )
        self._restored = True
        self._dispatch_many("on_resume", self, report)
        resumed = getattr(self.logger, "resume", None)
        if callable(resumed):
            resumed(report)
        return report

    # ------------------------------------------------------------------
    # evaluate and infer
    # ------------------------------------------------------------------
    def evaluate(self, data: Any | None = None) -> Mapping[str, float]:
        """Greedy accuracy and entropy per node. Does not train or advance data."""
        self._require_built()
        if data is not None and hasattr(data, "states") and hasattr(data, "targets"):
            task = data
        elif data is not None:
            task = data.next(self._epoch)
        elif self.episodes:
            task = self.episodes[-1].task
        else:
            task = self.task_source.next(self._epoch)
        states = self._runtime.to(task.states)
        targets = task.targets.to(states.device)
        metrics: dict[str, float] = {}
        accuracies: list[float] = []
        for path, peer in self.peers.items():
            if self.freeze.is_retired(path):
                continue
            adapter = peer.adapter
            was_training = adapter.training
            adapter.eval()
            try:
                with torch.no_grad():
                    logits, _values = adapter(states)
                policy = make_policy(logits)
                if not targets.is_floating_point():
                    accuracy = float(
                        (logits.argmax(dim=-1) == targets).float().mean().item()
                    )
                    metrics[f"{path}/accuracy"] = accuracy
                    accuracies.append(accuracy)
                probs = policy.probs.clamp_min(1e-12)
                metrics[f"{path}/entropy"] = float(
                    (-(probs * probs.log()).sum(dim=-1)).mean().item()
                )
            finally:
                adapter.train(was_training)
        if accuracies:
            metrics["mean_accuracy"] = sum(accuracies) / len(accuracies)
        return metrics

    def infer(self, inputs: Any, *, node_path: str | None = None) -> Any:
        """Hierarchical (or flat) inference through ``run_inference``."""
        self._require_built()
        return run_inference(
            self.active_tree,
            inputs,
            delegation=self.delegation_policy,
            synthesis=self.synthesis_function,
            node_path=node_path,
            seed=int(self._runtime.seed),
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def export_parameters(
        self, *, node_path: str | None = None
    ) -> dict[str, torch.Tensor]:
        """Detached CPU copies keyed ``"<node path>/<parameter name>"``."""
        self._require_built()
        if node_path is not None and node_path not in self.peers:
            raise ContractError(f"no node at path {node_path!r}")
        paths = [node_path] if node_path is not None else list(self.peers)
        payload: dict[str, torch.Tensor] = {}
        for path in paths:
            for name, parameter in self.peers[path].module.named_parameters():
                payload[f"{path}/{name}"] = parameter.detach().cpu().clone()
        return payload

    def import_parameters(
        self, payload: Mapping[str, torch.Tensor], *, strict: bool = True
    ) -> int:
        """Copy tensors into the live parameters. Returns how many were written."""
        self._require_built()
        written = 0
        lookup: dict[str, nn.Parameter] = {}
        for path, peer in self.peers.items():
            for name, parameter in peer.module.named_parameters():
                lookup[f"{path}/{name}"] = parameter
        unknown = sorted(set(payload) - set(lookup))
        if unknown and strict:
            raise ContractError(
                "import_parameters got unknown key(s): " + ", ".join(unknown[:5])
            )
        with torch.no_grad():
            for key, value in payload.items():
                target = lookup.get(key)
                if target is None:
                    continue
                if tuple(value.shape) != tuple(target.shape):
                    raise ContractError(
                        f"{key}: payload shape {tuple(value.shape)} does not match "
                        f"{tuple(target.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))
                written += 1
        if strict:
            missing = sorted(set(lookup) - set(payload))
            if missing:
                raise ContractError(
                    "import_parameters is missing key(s): " + ", ".join(missing[:5])
                )
        return written

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _trainer_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "spec": self.spec.as_dict() if self.spec is not None else {},
            "strategy": self.strategy.state_dict() if self.strategy else {},
            "callbacks": self.callbacks.state_dict(),
            "history": [item.as_dict() for item in self.history],
            "epoch": self._epoch,
            "signal_source": self.signal_source,
            "distillation": self._distillation_state(),
        }
        state.update(self._extension_state())
        return state

    def _extension_state(self) -> dict[str, Any]:
        """The trainer-owned state document 17 added, each key only when it has content.

        Written into ``identity["trainer"]`` by :meth:`_trainer_state` and, on their own, by the
        periodic and watchdog writers (:meth:`_identity_extra`): a resume of a run that swapped
        an optimizer, a weight update or a peer strategy needs these, and a run that did none
        of it writes nothing, so a default run's payload is the one it always wrote (I1).
        """
        state: dict[str, Any] = {}
        policy_state = self._policy_state()
        if policy_state:
            state["policy"] = policy_state
        functions = self.functions.state_dict()
        if functions.get("entries"):
            state["functions"] = functions
        plan = self.optimizer_plan
        if plan is not None and plan.swapped():
            state["optimizer_plan"] = plan.state_dict()
        weights = self.weight_plan
        if weights is not None and (weights.customized() or weights.history()):
            state["weight_plan"] = weights.state_dict()
        strategies = self.strategy_plan
        if strategies is not None and (strategies.customized() or strategies.history()):
            state["strategy_plan"] = strategies.state_dict()
        if self.freeze.paths(RETIRED):
            state["freeze"] = self.freeze.state_dict()
        mutation = self._mutation_state()
        if mutation:
            state["mutation"] = mutation
        return state

    # ------------------------------------------------------------------
    # Component manifest (document 17 section 14)
    # ------------------------------------------------------------------
    def _identity_extra(self) -> dict[str, Any]:
        """Keys a checkpoint adds to ``identity``; ``{}`` for a default run (P1).

        * ``trainer``: the extension state of :meth:`_extension_state`;
        * ``components``: the extensions the run uses (section 14.3).

        * ``engine``: :data:`ENGINE_NAME`, written unconditionally on a built trainer. It is what
          ``E091`` compares; ``Orchestrator`` writes no ``identity`` at all, so an absent key
          reads as "orchestrator".
        """
        extra: dict[str, Any] = {}
        if not self.built:
            return extra
        state = self._extension_state()
        if state:
            extra["trainer"] = state
        manifest = self._component_manifest()
        if manifest:
            extra["components"] = manifest
        extra["engine"] = ENGINE_NAME
        return extra

    def _slot_view(self) -> Any:
        """Every place this run keeps a component, behind ``paths()`` / ``get()``."""
        from ..components.fingerprint import (
            SlotView, aggregator_slots, optimizer_slots, reward_term_slots,
        )
        from ..components.slots import Slot

        extras: list[Any] = []
        extras.extend(optimizer_slots(self.optimizer_plan))
        resolved = self._aggregator_resolved
        if resolved is None and self._aggregator is not None:
            try:
                resolved = self.resolver.adopt(
                    "aggregator", self._aggregator, name=type(self._aggregator).__name__
                )
            except Exception:  # noqa: BLE001 - an instance the contract refuses is not recorded
                resolved = None
        extras.extend(aggregator_slots(resolved))
        chain = tuple(getattr(self.scorer, "chain", ()) or ())
        extras.extend(reward_term_slots(chain))
        held = set(self.slots.paths())
        for stage in chain:
            path = f"scoring.{stage.kind.value}"
            if path in held:
                continue
            try:
                adopted = self.resolver.adopt("scorer", stage, name=type(stage).__name__)
            except Exception:  # noqa: BLE001 - not a component, not recorded
                continue
            extras.append(Slot(path=path, kind_id="scorer", resolved=adopted))
        return SlotView(self.slots, extras)

    def _component_manifest(self) -> dict[str, dict[str, dict[str, Any]]]:
        """The section 14.3 manifest for this trainer's components; ``{}`` when it has none.

        ``slots`` is created in ``__init__``; a stand-in built without it records nothing
        rather than failing.
        """
        if getattr(self, "slots", None) is None:
            return {}
        from ..components.fingerprint import component_manifest

        return component_manifest(self._slot_view())

    def _component_plan(self, stored: Any) -> Any:
        """The ``RestorePlan`` for this payload; ``absent`` when there is nothing recorded."""
        from ..components.restore import ComponentRestorer, RestorePlan

        if getattr(self, "slots", None) is None:
            return RestorePlan(absent=True)
        return ComponentRestorer(self._slot_view()).plan(stored)

    def _policy_state(self) -> dict[str, Any]:
        """The active policy's own state, unless it is the distillation step (kept apart)."""
        policy = self._policy
        inner = getattr(policy, "inner", policy)
        if inner is None or (
            self.distill is not None
            and (inner is self.distill or getattr(inner, "_step", None) is self.distill)
        ):
            return {}
        getter = getattr(policy, "state_dict", None)
        if not callable(getter):
            return {}
        payload = dict(getter())
        # A guard's own counters alone are not worth a key: they are rebuilt from zero.
        if set(payload) == {"inner", "guard"} and not payload["inner"]:
            return {}
        return payload

    def _distillation_state(self) -> dict[str, Any]:
        state = getattr(self.distill, "state_dict", None)
        if not self._distillation.enabled or not callable(state):
            return {}
        return dict(state())

    def state_dict(self) -> dict[str, Any]:
        """A checkpoint payload dict for the current state."""
        self._require_built()
        if self.checkpoint is None:
            raise ImproperlyConfigured("state_dict() needs a checkpoint writer")
        context = CheckpointContext(
            trainer=self,
            episode=self._epoch,
            global_step=self.global_step,
            score=self.mean_score,
            kappa=float(self.controller.kappa) if self.controller else 0.0,
            entropy=float(self.controller.entropy) if self.controller else 0.0,
            aggregations=tuple(self.reports[-64:]),
            identity_extra=self._identity_extra(),
        ).resolve()
        payload = self.checkpoint._collect(
            context, episode=self._epoch, step=self.global_step
        )
        data = _detach_clone(payload.as_dict())
        # ``identity`` already holds ``components`` (section 14); ``trainer`` is the full state,
        # a superset of the extension keys the periodic writer records.
        data["identity"] = {**dict(data["identity"]), "trainer": self._trainer_state()}
        return data

    def load_state_dict(
        self, payload: Mapping[str, Any], *, strict: bool = True
    ) -> ResumeReport:
        """Restore from a payload produced by :meth:`state_dict` or a checkpoint.

        The component verification, the optimizer and weight plans (before the models move)
        and the strategy, freeze, callback, policy and function state (after) are applied by
        :meth:`MACTFCheckpoint.restore` through :meth:`_prepare_restore` and
        :meth:`_finish_restore`, so ``fit(resume=True)`` and this method restore the same things.
        """
        self._require_built()
        if self.checkpoint is None:
            raise ImproperlyConfigured("load_state_dict() needs a checkpoint writer")
        report = self.checkpoint.restore(payload, trainer=self, strict=strict)
        self._restored = True
        self._dispatch_many("on_resume", self, report)
        return report

    @staticmethod
    def _extra_of(stored: Any) -> dict[str, Any]:
        identity = stored.get("identity") if isinstance(stored, Mapping) else stored.identity
        return dict((identity or {}).get("trainer") or {})

    def _prepare_restore(self, stored: Any, *, strict: bool = True) -> None:
        """Everything a restore decides before a single parameter moves (document 17 P3).

        The component manifest is verified first: a missing component is discovered here, not
        after the models were overwritten. Then the signal source, and the optimizer and weight
        plans of the saved run are re-applied, because they decide which optimizers the
        checkpoint's optimizer state is loaded into.
        """
        plan = self._component_plan(stored)
        plan.raise_for_errors()
        self.component_notes = plan.notes()
        extra = self._extra_of(stored)
        recorded = extra.get("signal_source")
        if recorded and recorded != self.signal_source and strict:
            raise ContractError(
                f"checkpoint was written under SIGNAL_SOURCE={recorded!r} but this "
                f"trainer uses {self.signal_source!r}; a resume cannot switch the "
                "temperature policy mid-run"
            )
        saved_plan = extra.get("optimizer_plan")
        if saved_plan and self.optimizer_plan is not None:
            notes = self.optimizer_plan.load_state_dict(saved_plan, strict=self._strict_restore)
            self._optimizer_warnings.extend(notes)
            for path in self.peers:  # the swapped entries may select different optimizers
                entry = self.optimizer_plan.select(path)
                current = self.optimizers.get(path)
                if current is None or getattr(current, "entry_hash", "") != entry.spec_hash:
                    self._install_optimizer(path, self._make_optimizer(path, entry, self._runtime))
                    if entry.frozen:
                        self.freeze.add(path, "optimizer:null")
        saved_weights = extra.get("weight_plan")
        if saved_weights and self.weight_plan is not None:
            weight_notes = self.weight_plan.load_state_dict(saved_weights, strict=self._strict_restore)
            self.weight_strategy = self.weight_plan.strategy_for(ROOT_LABEL, self.aggregator)
            self._validate_weight_plan(initial=False)
            self.weight_warnings.extend(weight_notes)        # after validation, which resets the list

    def _finish_restore(self, stored: Any) -> None:
        """What the trainer keeps outside the checkpoint's own sections, after they are applied."""
        extra = self._extra_of(stored)
        restore_notes: list[str] = []
        saved_freeze = extra.get("freeze")
        if saved_freeze:
            restore_notes.extend(self.freeze.load_state_dict(saved_freeze))
        saved_strategies = extra.get("strategy_plan")
        if saved_strategies and self.strategy_plan is not None:
            before = list(self.strategy_plan.live_instances())
            restore_notes.extend(
                self.strategy_plan.load_state_dict(saved_strategies, strict=self._strict_restore)
            )
            after = list(self.strategy_plan.live_instances())
            for gone in before:
                if not any(gone is kept for kept in after):
                    try:
                        gone.detach(self)
                    except Exception as error:  # noqa: BLE001 - the restore is done; report it
                        log.error("detach of %s raised: %s", type(gone).__name__, error)
            for fresh in after:
                if not any(fresh is old for old in before):
                    fresh.attach(self)
            self.strategy = self.strategy_plan.default_instance()
            self._validate_strategy_plan(initial=False)
            self.strategy_warnings.extend(restore_notes)     # after validation, which resets the list
        elif extra.get("strategy"):
            self.strategy.load_state_dict(extra["strategy"])
        if extra.get("callbacks"):
            self.callbacks.load_state_dict(extra["callbacks"])
        restore = getattr(self.distill, "load_state_dict", None)
        if self._distillation.enabled and callable(restore) and extra.get("distillation"):
            self.distillation_resume = restore(extra["distillation"])
        policy_restore = getattr(self._policy, "load_state_dict", None)
        if extra.get("policy") and callable(policy_restore):
            policy_restore(extra["policy"])
        if extra.get("functions"):
            self._function_warnings.extend(
                self.functions.load_state_dict(extra["functions"], strict=self._strict_restore)
            )

    # ------------------------------------------------------------------
    # Device moves and shutdown
    # ------------------------------------------------------------------
    def to(self, runtime: Runtime) -> "MACTFTrainer":
        """Move every node and the reference, keeping each optimizer's class."""
        self._require_built()
        for peer in self.peers.values():
            peer.adapter = runtime.to(peer.adapter)
        if self.reference is not None:
            self.reference.adapter = runtime.to(self.reference.adapter)
        for optimizer in self.optimizers.values():
            optimizer.move_to(runtime)
        self._runtime = runtime
        self.runner = EpisodeRunner(runtime=runtime)
        return self

    def close(self) -> None:
        """Idempotent shutdown. Flushes the logger; never discards state."""
        if self._closed:
            return
        self._closed = True
        if self.built:
            self._dispatch_many("on_close", self)
        flush = getattr(self.logger, "flush", None)
        if callable(flush):
            flush()


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------
#: Parameter-name prefixes treated as a node's own head under ``shared_trunk``.
_HEAD_PREFIXES = ("head", "policy", "value", "actor", "critic", "classifier", "lm_head", "out")


def _with_shared_trunk(trunk: nn.Module, module: nn.Module) -> nn.Module:
    """Make ``module``'s trunk parameters *be* ``trunk``'s; heads stay its own."""
    shared = dict(trunk.named_parameters())
    for name, parameter in list(module.named_parameters()):
        if name.split(".")[0].startswith(_HEAD_PREFIXES):
            continue
        source = shared.get(name)
        if source is None or tuple(source.shape) != tuple(parameter.shape):
            continue
        owner_path, _, leaf = name.rpartition(".")
        owner = module.get_submodule(owner_path) if owner_path else module
        owner.register_parameter(leaf, source)
    return module


def _json_safe(value: Any) -> Any:
    """A JSON-safe copy: primitives pass, containers recurse, objects become type names."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _checkpoint_block_written(settings: Any) -> bool:
    """Whether the project wrote ``CHECKPOINT`` (a framework default is not a request)."""
    return is_authored(settings, "CHECKPOINT")


def _as_scorers(spec: Any) -> list[Any]:
    """Normalize ``scorers=`` into a list of ``Scorer`` instances."""
    from ..scoring.base import Scorer
    from ..scoring.registry import build_scorer_from_spec

    if isinstance(spec, Scorer):
        return [spec]
    if isinstance(spec, Mapping):
        return [build_scorer_from_spec(spec)]
    resolved: list[Any] = []
    for item in spec:
        resolved.append(item if isinstance(item, Scorer) else build_scorer_from_spec(item))
    return resolved


def _detach_clone(value: Any) -> Any:
    """Deep copy a nested payload, cloning tensors so it no longer aliases live storage."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _detach_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_clone(item) for item in value)
    return value


def _clear_optimizer_state(optimizer: MACTFOptimizer | None) -> None:
    """Drop momentum computed for pre-aggregation weights."""
    if optimizer is not None:
        optimizer.reset_state()


class _CheckpointBridge:
    """Adapts the watchdog's ``HealthContext`` call into a ``CheckpointContext``."""

    def __init__(self, inner: Any, trainer: "MACTFTrainer") -> None:
        self._inner = inner
        self._trainer = trainer

    def save(
        self,
        context: Any,
        *,
        reason: str,
        episode: int | None = None,
        step: int | None = None,
        best: bool = False,
    ) -> Any:
        if isinstance(context, CheckpointContext):
            resolved = context
        else:
            scores = dict(getattr(context, "scores", {}) or {})
            resolved = CheckpointContext(
                trainer=self._trainer,
                episode=int(getattr(context, "epoch", 0)),
                global_step=int(getattr(context, "step", 0)),
                score=(sum(scores.values()) / len(scores)) if scores else 0.0,
                kappa=float(getattr(context, "kappa", None) or 0.0),
                entropy=float(getattr(context, "entropy", None) or 0.0),
                aggregations=tuple(self._trainer.reports),
                identity_extra=self._trainer._identity_extra(),
            )
        return self._inner.save(
            resolved, reason=reason, episode=episode, step=step, best=best
        )

    def resume_from_last_checkpoint(self, **kwargs: Any) -> Any:
        kwargs.setdefault("trainer", self._trainer)
        return self._inner.resume_from_last_checkpoint(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
