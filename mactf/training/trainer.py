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
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from .. import signals
from ..conf import Settings, get_settings
from ..conf_files import authored_keys, is_authored
from ..coopetition.aggregation import (
    AggregateResult,
    AggregationReport,
    Aggregator,
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
    HierarchicalStrategy,
    KappaSoftmaxStrategy,
    PeerStrategy,
    build_peer_strategy,
    deduplicate_tied,
    group_layouts,
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
from ..coopetition.weights import (
    KappaSoftmaxAverage,
    WeightUpdateStrategy,
    build_weight_update,
)
from ..exceptions import (
    ComplianceError,
    ContractError,
    ImproperlyConfigured,
    ValidationError,
)
from ..orchestration.checkpoint_v2 import (
    CheckpointContext,
    MACTFCheckpoint,
    ResumeReport,
)
from ..orchestration.episode import Episode, EpisodeRunner, Task
from ..rl.advantage import penalise_void_advantage
from ..rl.functions.loss import CompositeLoss, LossFunction
from ..rl.loss import LossConfig
from ..rl.policy import make_policy
from ..runtime import Runtime, autocast, build_runtime, seed_everything
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
from .optimizer import MACTFOptimizer, StepOutcome
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

__all__ = ["MACTFTrainer", "TrainSpec"]

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
            if trainer is not None:
                raise ImproperlyConfigured(
                    "trainer=... and distillation=... are mutually exclusive; both "
                    "install the per-step policy. Pass the teachers to distillation= and "
                    "drop trainer=, construct CoDistillStep yourself and pass it as "
                    "trainer=, or pass distillation=False to turn a project-level "
                    "DISTILLATION.ENABLED off for this run."
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
        self.peers = self._build_peers(tree, runtime)

        # Reference policy: one per run, frozen, from the first root.
        first = next(iter(self.peers.values()))
        self.reference = build_reference(first, runtime=runtime)

        # Sibling layouts must match; different branches may not.
        self._validate_group_layouts(tree)

        # Optimizers bind *after* finetuning and freezing.
        self._bind_optimizers(runtime)

        # Temperature, aggregator, strategies.
        # One owner of the temperature: the aggregator's controller *is* the root
        # controller, so the checkpointed kappa is the kappa that was used.
        roots = len(tree.root_paths)
        self.aggregator = self._aggregator or Aggregator.from_settings(
            coopetition, num_peers=max(1, roots)
        )
        if self._kappa is not None:
            self.aggregator.controller = self._kappa
        self.controller = self.aggregator.controller
        levels = self._build_controllers(coopetition)
        levels[None] = self.controller
        self._level_controllers = levels
        self.controllers = {
            ("root" if parent is None else parent): controller
            for parent, controller in levels.items()
        }
        for _parent, members in aggregation_groups(tree):
            feasible_bounds(
                max(1, len(members)),
                floor=float(coopetition.get("WEIGHT_FLOOR", 0.02)),
                cap=float(coopetition.get("WEIGHT_CAP", 0.6)),
            )
        self.weight_strategy = build_weight_update(
            self._weight_update_spec, aggregator=self.aggregator
        )
        self.strategy = build_peer_strategy(
            self._peer_strategy_spec, hierarchy=self._hierarchy is not None
        )
        if self._peer_strategy_spec is None and self._hierarchy is None:
            # The equivalence gate (document 08 section 9) requires a default flat
            # run to aggregate exactly as ``Orchestrator`` does, so the default
            # flat strategy is the one that mixes parameters.
            self.strategy = KappaSoftmaxStrategy()

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
        self._attach_distillation()
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
        self._initial_nodes = len(self.peers)
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
        training = self._training_settings()
        learning_rate = float(training.get("LEARNING_RATE", 1e-3))
        action_dim = int(training.get("ACTION_DIM", 4))
        state_dim = int(training.get("STATE_DIM", 16))
        sharing = str(getattr(self._hierarchy, "sharing", "independent"))
        freeze_base = bool(
            self.finetuning.freeze_base if self.finetuning is not None else True
        )

        peers: dict[str, Peer] = {}
        roots = list(tree.root_paths)
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
                module = _with_shared_trunk(trunk, module)

            module, report = apply_finetuning(
                module, self.finetuning, device=runtime.device
            )
            peer = build_peer(
                module,
                name=node.path,
                runtime=runtime,
                learning_rate=learning_rate,
                action_dim=action_dim,
                state_dim=state_dim,
                weight_decay=float(training.get("WEIGHT_DECAY", 0.0)),
            )
            peer.metadata["depth"] = node.depth
            peer.metadata["sharing"] = sharing
            peer.metadata["finetuning"] = report
            if node.depth == 1 and node.path in roots:
                peer.metadata["role"] = "root"
            node.peer = peer
            peers[node.path] = peer
        if not peers:  # pragma: no cover - the tree always has one node
            raise ImproperlyConfigured("the tree produced no nodes")
        return peers

    def _validate_group_layouts(self, tree: PeerTree) -> None:
        for _parent, members in aggregation_groups(tree):
            if len(members) < 2:
                continue
            group_layouts(self.peers, members)

    def _optimizer_block(self) -> dict[str, Any]:
        """The ``OPTIMIZER`` block when the project wrote it, else ``{}``.

        ``DEFAULTS`` carries an ``OPTIMIZER`` block, so its mere presence proves
        nothing; a legacy project must keep its Adam at ``TRAINING.LEARNING_RATE``.
        When the project wrote the block but left ``LR`` or ``MAX_GRAD_NORM`` out,
        those two defer to ``TRAINING`` rather than to the framework default, so a
        ``TRAINING.LEARNING_RATE`` written in ``trainer.yaml`` is not silently
        replaced by ``OPTIMIZER``'s ``1e-3``.
        """
        if not is_authored(self.settings, "OPTIMIZER"):
            return {}
        block = dict(self.settings.get("OPTIMIZER", {}) or {})
        if not block:
            return {}
        written = authored_keys(self.settings, "OPTIMIZER")
        training = self._training_settings()
        if written is not None:
            if "LR" not in written and "LEARNING_RATE" in training:
                block["LR"] = training["LEARNING_RATE"]
            if "MAX_GRAD_NORM" not in written and "MAX_GRAD_NORM" in training:
                block["MAX_GRAD_NORM"] = training["MAX_GRAD_NORM"]
        return block

    def _bind_optimizers(self, runtime: Runtime) -> None:
        scaler = runtime.grad_scaler
        for path, peer in self.peers.items():
            module = peer.adapter
            spec = self._optimizer_spec
            block = self._optimizer_block()
            # An explicit ``optimizer=`` argument wins over the block, which wins
            # over the inline default: an argument is an override, a block is
            # configuration, a default is a default.
            if spec is not None:
                optimizer = MACTFOptimizer.from_spec(spec)
            elif block:
                optimizer = MACTFOptimizer.from_block(block, node_path=path)
            else:
                optimizer = MACTFOptimizer.from_spec(
                    {
                        "type": "torch",
                        "class": "Adam",
                        "lr": float(self._training_settings().get("LEARNING_RATE", 1e-3)),
                    }
                )
            if self.grad_accumulation != 1:
                optimizer.grad_accumulation = self.grad_accumulation
            elif spec is None and optimizer.grad_accumulation == 1:
                # TRAINER.GRAD_ACCUMULATION arrives as TRAINING.GRAD_ACCUMULATION
                # (folded at load). OPTIMIZER.GRAD_ACCUMULATION, when set, already
                # reached the optimizer through ``from_block`` and is not overridden.
                configured = int(self._training_settings().get("GRAD_ACCUMULATION", 1) or 1)
                if configured > 1:
                    optimizer.grad_accumulation = configured
            if self.max_grad_norm is not None:
                optimizer.max_grad_norm = self.max_grad_norm
            elif optimizer.max_grad_norm is None:
                configured = float(
                    self._training_settings().get("MAX_GRAD_NORM", 0.0) or 0.0
                )
                optimizer.max_grad_norm = configured if configured > 0 else None
            optimizer.scaler = scaler
            optimizer.node_path = path
            optimizer.bind(
                list(module.parameters()),
                named=list(module.named_parameters()),
            )
            self.optimizers[path] = optimizer
            # The facade replaces the legacy Adam on the peer so that
            # ``Peer.state_dict`` (and therefore the checkpoint) carries the
            # optimizer the trainer actually steps.
            peer.optimizer = optimizer  # type: ignore[assignment]
        self._validate_tied_parameters()

    def _validate_tied_parameters(self) -> None:
        """A tied parameter in two groups cannot receive two blended values."""
        owners: dict[int, str] = {}
        for parent, members in aggregation_groups(self.tree):
            group = "root" if parent is None else parent
            seen_in_group: dict[int, str] = {}
            for member in members:
                peer = self.peers.get(member)
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
            controllers[parent] = AdaptiveKappa.from_settings(
                dict(coopetition), num_peers=max(1, len(members))
            )
        return controllers

    def _build_scorer(self) -> Any:
        if self._hierarchical_scorer is not None:
            return self._hierarchical_scorer
        if self._scorers_spec is not None:
            pipeline = build_scorer(self.settings, scorers=_as_scorers(self._scorers_spec))
        else:
            pipeline = build_scorer(self.settings)
        if self._hierarchy is not None:
            return HierarchicalScorer(pipeline)
        return pipeline

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

    def record_signal_sharing(self, signals_map: Mapping[str, float], *, epoch: int) -> None:
        """Called by ``CooperativeStrategy``: the cohort saw these signals."""
        self.signal_sharing.append(
            {"epoch": int(epoch), "signals": {str(k): float(v) for k, v in signals_map.items()}}
        )

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
            return tuple(self.tree.root_paths)
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

    def _attach_distillation(self) -> None:
        """Install ``CoDistillStep`` as the per-step policy when distillation is on.

        Runs before ``_setup_custom_trainer`` so ``setup`` reaches every peer exactly
        once. When off, ``_trainer_policy`` is left exactly as the caller set it.
        """
        config = self._distillation
        if not config.enabled:
            return
        from ..distill import CoDistillStep

        policy = CoDistillStep.from_settings(
            config,
            tree=self.tree,
            optimizers=self.optimizers,
            reference=self.reference,
            runtime=self._runtime,
            settings=self.settings,
        )
        policy.attach(self)
        self._trainer_policy = policy

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
        policy = self._trainer_policy
        setup = getattr(policy, "setup", None)
        if policy is None or not callable(setup):
            return
        for node in self.tree.walk("root-first"):
            setup(node)

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
            for path in self.tree.root_paths
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
        tree = self.tree

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
                try:
                    synthesized = self.synthesis_function.synthesize(
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
        self.tree.mark(node.path, "failed")
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
                self.tree, episode.ledgers, episode=episode.index, evidence=evidence
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
        policy = self._trainer_policy
        if policy is None:
            result = self._builtin_step(peer, optimizer, batch)
        else:
            handler = getattr(policy, "train_step", None)
            if callable(handler):
                raw = handler(tree_node or peer, batch)
            elif callable(policy):
                raw = policy(tree_node or peer, batch, trainer=self)
            else:
                raise ContractError("trainer must be a TrainerProtocol or a callable")
            result = coerce_step_result(raw, node=tree_node or peer)
            if result.loss.requires_grad:
                result = self._apply_custom_loss(result, optimizer)
        if result.metrics and result.metrics.get("pending"):
            # A rendezvous policy has no cohort result yet; recording its placeholder
            # zero would pollute the loss history and the epoch mean.
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

    def _builtin_step(
        self, peer: Peer, optimizer: MACTFOptimizer, batch: Mapping[str, Any]
    ) -> StepResult:
        """The policy-gradient step, i.e. ``local_update`` behind the protocol."""
        runtime = self._runtime
        states = batch["states"]
        adapter = peer.adapter
        adapter.train()
        if getattr(optimizer, "_accumulated", 0) == 0:
            optimizer.zero_grad()
        count = int(states.shape[0])
        fixed: dict[str, torch.Tensor] = {}

        def evaluate() -> tuple[torch.Tensor, Any]:
            with autocast(runtime):
                logits, values = adapter(states)
            policy = make_policy(logits)
            with torch.no_grad():
                ref_logits, _ = self.reference.adapter(states)
            reference_policy = make_policy(ref_logits)
            if "actions" not in fixed:
                # Chosen once so a closure-driven optimizer sees one smooth objective.
                fixed["actions"] = policy.probs.argmax(dim=-1).detach()
            signal = torch.full(
                (count,), float(batch.get("signal", 0.0)),
                device=logits.device, dtype=logits.dtype,
            )
            void = torch.full(
                (count,), bool(batch.get("void", False)),
                device=logits.device, dtype=torch.bool,
            )
            advantages = penalise_void_advantage(signal, void, floor=-1.0)
            loss_batch = {
                "policy": policy,
                "reference": reference_policy,
                "actions": fixed["actions"],
                "advantages": advantages,
                "values": values,
                "value_targets": signal.unsqueeze(-1),
                "compliance_floor": torch.tensor(
                    float(batch.get("floor", 0.0)), device=logits.device
                ),
                "void_mask": void,
            }
            return self.loss_function.compute(loss_batch)

        if optimizer.closure_based:
            total, breakdown = evaluate()
            outcome = optimizer.step(closure=lambda: evaluate()[0])
            total, breakdown = evaluate()
        else:
            total, breakdown = evaluate()
            self._backward(total, optimizer)
            outcome = optimizer.step()
        metrics = {"lr": float(outcome.lr)}
        if outcome.grad_norm is not None:
            metrics["grad_norm"] = float(outcome.grad_norm)
        return StepResult(
            loss=total.detach(),
            breakdown=breakdown,
            stepped=bool(outcome.stepped),
            metrics=metrics,
        )

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

    def _stamp_distillation_metadata(self) -> None:
        """Write ``metadata["distillation"]`` on this epoch's cards, after training.

        Populated by the trainer after the step, never by a scorer: scoring runs
        before the cohort step, so a scorer could only report the previous step.
        """
        policy = self._trainer_policy
        recorder = getattr(policy, "card_metadata", None)
        if not self._distillation.enabled or not callable(recorder) or not self._epoch_cards:
            return
        for path, peer_card in self._epoch_cards[-1].peers.items():
            peer_card.metadata["distillation"] = recorder(path)

    def _train_epoch(self) -> int:
        """Step 5: walk leaves-first, one or more updates per node."""
        steps = 0
        episode = self.episodes[-1]
        for node in self.tree.walk("leaves-first"):
            if node.peer is None or node.status == "failed":
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
                self.strategy.on_peer_step(self, node, result)
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
            children = len(self.tree[parent].children)
            controller = AdaptiveKappa.from_settings(
                dict(self._coopetition_settings()), num_peers=max(1, children)
            )
            self._level_controllers[parent] = controller
            self.controllers[parent] = controller
        base = self.aggregator
        aggregator = Aggregator(
            controller=controller,
            weight_floor=base.weight_floor,
            weight_cap=base.weight_cap,
            confidence_floor=base.confidence_floor,
            trim_fraction=base.trim_fraction,
            disagreement_max=base.disagreement_max,
            fallback_to_reference=base.fallback_to_reference,
            aggregate_buffers=base.aggregate_buffers,
        )
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
        eligible = {
            name: (not self._void(name))
            and bool((external_eligible or {}).get(name, True))
            for name in members
        }
        confidence = {name: self._confidence(name) for name in members}
        level = 0 if parent is None else int(self.tree[parent].depth)
        controller = aggregator.controller

        def run(chosen: WeightUpdateStrategy) -> AggregationReport:
            return chosen.apply(
                vectors,
                scores,
                kappa=float(controller.kappa),
                level=level,
                eligible=eligible,
                confidence=confidence,
            )

        reason: str | None = None
        report: AggregationReport | None = None
        is_default = isinstance(strategy, KappaSoftmaxAverage)
        try:
            report = run(strategy)
        except (ComplianceError, NotImplementedError) as error:
            if is_default:
                raise
            reason = str(error) or type(error).__name__
        if report is not None and report.fallback and not is_default:
            reason = "it reported fallback"
        if reason is not None:
            name = getattr(strategy, "name", type(strategy).__name__)
            report = run(KappaSoftmaxAverage(aggregator))
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
        members = tuple(self.tree.root_paths)
        report = self._weights_for(
            None, members, self.aggregator, self.weight_strategy, eligible
        )
        self.reports.append(report)
        self._epoch_report = report
        return report

    def _group_strategy(self, aggregator: Aggregator) -> WeightUpdateStrategy:
        if isinstance(self.weight_strategy, KappaSoftmaxAverage):
            return KappaSoftmaxAverage(aggregator)
        return self.weight_strategy

    def _write(
        self,
        aggregator: Aggregator,
        members: Sequence[str],
        report: AggregationReport,
        *,
        reference: Peer | None,
        reference_ema: float,
    ) -> AggregateResult:
        adapters = {name: self.peers[name].adapter for name in members}
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
        reference_adapter = reference.adapter if reference is not None else None
        result = aggregator.aggregate(
            adapters, report, reference=reference_adapter, device=self._runtime.device
        )
        aggregator.apply(
            adapters,
            result,
            reference=reference_adapter,
            reference_ema=reference_ema if reference is not None else 0.0,
            reset_optimizers=False,
            optimizers=None,
        )
        if bool(self._coopetition_settings().get("RESET_OPTIMIZERS", True)):
            for name in members:
                _clear_optimizer_state(self.optimizers.get(name))
        return result

    def aggregate(self, report: AggregationReport | None = None) -> AggregateResult:
        """Step 6b/7: write the root group's blend and update the reference EMA."""
        self._require_built()
        report = report or self._epoch_report or self.update_weights()
        ema = float(self._coopetition_settings().get("REFERENCE_EMA", 0.05))
        result = self._write(
            self.aggregator,
            tuple(self.tree.root_paths),
            report,
            reference=self.reference,
            reference_ema=ema,
        )
        self._dispatch_many("on_aggregate", self, report)
        return result

    def aggregate_group(
        self, parent: str | None, members: Sequence[str]
    ) -> AggregationReport:
        """Aggregate one child group. The reference is never touched here."""
        self._require_built()
        aggregator = self._group_aggregator(parent)
        report = self._weights_for(
            parent,
            tuple(members),
            aggregator,
            self._group_strategy(aggregator),
            None,
        )
        self.reports.append(report)
        self._write(
            aggregator, tuple(members), report, reference=None, reference_ema=0.0
        )
        self._dispatch_many("on_aggregate", self, report)
        return report

    def _coordinate(self, epoch: int) -> AggregationReport | None:
        """Step 6: let the configured strategy mix (or not) at the epoch's end."""
        strategy = self.strategy
        self._epoch_report = None
        if isinstance(strategy, HierarchicalStrategy):
            report = self.update_weights()
            strategy.on_epoch_end(self, epoch, report)
        elif isinstance(strategy, (KappaSoftmaxStrategy, FrozenPeerStrategy)):
            # These two drive update_weights and aggregate themselves.
            strategy.on_epoch_end(self, epoch, None)
        elif strategy.should_aggregate():
            report = self.update_weights()
            self.aggregate(report)
            strategy.on_epoch_end(self, epoch, report)
        else:
            strategy.on_epoch_end(self, epoch, None)
        return self._epoch_report

    # ------------------------------------------------------------------
    # run_epoch and fit
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> EpochResult:
        """One epoch in the fixed order of document 08 section 5.2."""
        self._require_built()
        started = time.monotonic()
        self._epoch = int(epoch)
        self._epoch_cards = []
        self._epoch_step_losses = []
        self._epoch_node_metrics = {}
        self._normalization = None
        self._pending = None
        for node in self.tree.walk("root-first"):
            if node.status == "failed":
                self.tree.mark(node.path, "active")

        self._dispatch_many("on_epoch_start", self, epoch)
        self.strategy.on_epoch_start(self, epoch)
        task = self.task_source.next(epoch)
        self.collect_episode(task)
        self.score()
        steps = self._train_epoch()
        self._stamp_distillation_metadata()
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
        if self._trainer_policy is not None and callable(
            getattr(self._trainer_policy, "on_epoch_end", None)
        ):
            self._trainer_policy.on_epoch_end(self.tree, result)
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
        roots = max(1, len(self.tree.root_paths))
        elapsed = self._epoch_seconds[-1] if self._epoch_seconds else 0.0
        previous_rate = None
        if len(self._epoch_seconds) > 1 and self._epoch_seconds[-2] > 0:
            previous_rate = result.steps / self._epoch_seconds[-2]
        escalated = sum(
            len(peer_card.warnings_escalated)
            for card in self._epoch_cards
            for peer_card in card.peers.values()
        )
        dead: dict[str, int] = {}
        for node in self.tree.walk("root-first"):
            if node.status == "failed":
                dead[node.path] = self._dead_counts.get(node.path, 0) + 1
        self._dead_counts = dict(dead)
        count = 0
        if self.checkpoint is not None:
            try:
                count = len(self.checkpoint.list())
            except Exception:  # noqa: BLE001 - a missing directory is zero checkpoints
                count = 0
        return HealthContext(
            epoch=int(result.epoch),
            step=int(self.global_step),
            losses=tuple(item.mean_loss for item in self.history) + (result.mean_loss,),
            grad_norms=tuple(self.grad_norms),
            scores=dict(self.signal_by_peer),
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
            tree_nodes=len(self.peers),
            initial_tree_nodes=self._initial_nodes,
            dead_branches=dead,
            level_durations=dict(self._level_durations),
            extras={"tree": self.tree},
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
            self.tree,
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
        return {
            "spec": self.spec.as_dict() if self.spec is not None else {},
            "strategy": self.strategy.state_dict() if self.strategy else {},
            "callbacks": self.callbacks.state_dict(),
            "history": [item.as_dict() for item in self.history],
            "epoch": self._epoch,
            "signal_source": self.signal_source,
            "distillation": self._distillation_state(),
        }

    def _distillation_state(self) -> dict[str, Any]:
        state = getattr(self._trainer_policy, "state_dict", None)
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
        ).resolve()
        payload = self.checkpoint._collect(
            context, episode=self._epoch, step=self.global_step
        )
        data = _detach_clone(payload.as_dict())
        data["identity"] = {**dict(data["identity"]), "trainer": self._trainer_state()}
        return data

    def load_state_dict(
        self, payload: Mapping[str, Any], *, strict: bool = True
    ) -> ResumeReport:
        """Restore from a payload produced by :meth:`state_dict` or a checkpoint."""
        self._require_built()
        if self.checkpoint is None:
            raise ImproperlyConfigured("load_state_dict() needs a checkpoint writer")
        identity = payload.get("identity") if isinstance(payload, Mapping) else payload.identity
        extra = dict((identity or {}).get("trainer") or {})
        recorded = extra.get("signal_source")
        if recorded and recorded != self.signal_source and strict:
            raise ContractError(
                f"checkpoint was written under SIGNAL_SOURCE={recorded!r} but this "
                f"trainer uses {self.signal_source!r}; a resume cannot switch the "
                "temperature policy mid-run"
            )
        report = self.checkpoint.restore(payload, trainer=self, strict=strict)
        if extra.get("strategy"):
            self.strategy.load_state_dict(extra["strategy"])
        if extra.get("callbacks"):
            self.callbacks.load_state_dict(extra["callbacks"])
        restore = getattr(self._trainer_policy, "load_state_dict", None)
        if self._distillation.enabled and callable(restore) and extra.get("distillation"):
            self.distillation_resume = restore(extra["distillation"])
        self._restored = True
        self._dispatch_many("on_resume", self, report)
        return report

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
    if optimizer is None:
        return
    backend = optimizer.backend
    backends = (
        list(getattr(backend, "_backends", {}).values())
        if hasattr(backend, "_backends")
        else [backend]
    )
    for each in backends:
        native = getattr(each, "_instance", None)
        if native is not None:
            native.state.clear()


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
            )
        return self._inner.save(
            resolved, reason=reason, episode=episode, step=step, best=best
        )

    def resume_from_last_checkpoint(self, **kwargs: Any) -> Any:
        kwargs.setdefault("trainer", self._trainer)
        return self._inner.resume_from_last_checkpoint(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
