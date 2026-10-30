"""``CoDistillStep``: the cohort training step (document 14 section 8).

One shared batch, ``N`` peers, ``M`` frozen teachers. Each peer descends its own
compound loss with its own optimizer; teachers contribute frozen outputs and are
never optimized, never blended, and never counted as aggregation participants.

The class satisfies :class:`~mactf.training.protocol.TrainerProtocol`
structurally. Because that protocol is per-peer and a cohort objective is not, the
per-peer ``train_step`` calls rendezvous: batches are buffered until every peer has
arrived, one cohort step then runs, and its result is returned to every caller.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from .. import signals
from ..coopetition.adapter import PeerAdapter  # noqa: F401  (typing aid for adapters)
from ..coopetition.aggregation import Aggregator
from ..coopetition.peer import build_reference
from ..coopetition.report import AggregationReport
from ..coopetition.tree import node_seed
from ..coopetition.weights import KappaSoftmaxAverage, build_weight_update
from ..conf import get_settings
from ..exceptions import (
    ComplianceError,
    ContractError,
    ImproperlyConfigured,
    ValidationError,
)
from ..rl.advantage import penalise_void_advantage
from ..rl.functions._spec import build_from_kwargs
from ..rl.functions.loss import CompositeLoss
from ..rl.loss import LossBreakdown
from ..rl.policy import greedy_action, make_policy
from ..runtime import Runtime, autocast, build_runtime
from ..training.optimizer import MACTFOptimizer, StepOutcome
from ..training.protocol import EpochResult, StepResult
from .batch import DistillBatch, coerce_batch
from .conformance import (
    CONFORMANCE_KEYS,
    CONFORMANCE_MODES,
    ConformanceVerdict,
    DeclaredAudit,
    TeacherAuditor,
    TrustOnlyAudit,
    TrustProfile,
    TrustState,
)
from .config import DistillationConfig
from .contrastive import ContrastiveReport, contrastive_loss, peer_margin
from .labelmap import LabelMap, build_label_map, shared_axis
from .objective import (
    ObjectiveConfig,
    PeerLoss,
    build_regularizers,
    cohort_signal,
    compound_loss,
    cooperative_term,
    individual_term,
    ramp_factor,
    regularization_term,
)
from .sigma import SIGMA_MODES, SigmaState, sigma_weights
from .state import STATE_HISTORY, DistillationState, DistillResumeReport
from .teacher import Teacher, TeacherOutput, TeacherSpec, build_teacher
from .views import (
    IdentityView,
    OutputBundle,
    OutputView,
    ViewPair,
    build_view,
    peer_bundle,
    teacher_bundle,
)

__all__ = [
    "CONFORMANCE_MODES",
    "CoDistillStep",
    "PreparedPeer",
    "PreparedStep",
]

logger = logging.getLogger(__name__)

_COLLAPSE_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_config(
    config: ObjectiveConfig | Mapping[str, Any] | None, settings: Any
) -> ObjectiveConfig:
    """Explicit argument wins; then the settings block; then dataclass defaults."""
    block = dict(settings.get("DISTILLATION", {}) or {})
    if isinstance(config, ObjectiveConfig):
        return config
    payload: dict[str, Any] = {}
    if isinstance(config, Mapping):
        payload.update(dict(config))
    payload.setdefault("tau", block.get("TAU", 0.07))
    payload.setdefault("tau_min", block.get("TAU_MIN", 0.01))
    payload.setdefault("tau_max", block.get("TAU_MAX", 1.0))
    payload.setdefault("symmetric", block.get("SYMMETRIC", True))
    payload.setdefault("beta", block.get("BETA", 1.0))
    payload.setdefault("sigma_mode", block.get("SIGMA_MODE", "quality_softmax"))
    payload.setdefault("lambda_coop", block.get("LAMBDA_COOP", 0.0))
    payload.setdefault("lambda_ind", block.get("LAMBDA_IND", 1.0))
    payload.setdefault("lambda_reg", block.get("LAMBDA_REG", 0.0))
    payload.setdefault("lambda_cd", block.get("LAMBDA_CD", 1.0))
    payload.setdefault("peer_margin", block.get("PEER_MARGIN", 0.25))
    payload.setdefault("warmup_steps", block.get("WARMUP_STEPS", 0))
    payload.setdefault("ramp_steps", block.get("RAMP_STEPS", 0))
    return build_from_kwargs(ObjectiveConfig, payload)


def _resolve_loss(loss: Any, settings: Any) -> Any:
    """``None`` is the docx composite; a mapping is built from the loss registry."""
    if loss is None:
        return CompositeLoss()
    if isinstance(loss, Mapping):
        from ..rl.functions.registry import build_function

        return build_function(dict(loss), family="loss")
    if hasattr(loss, "compute") and hasattr(loss, "required_fields"):
        return loss
    raise ValidationError(
        f"loss must be a LossFunction, a spec mapping, or None; got {type(loss).__name__}"
    )


def _flat_width(tensor: torch.Tensor) -> int:
    """Width a flattening view sees: the last axis of a 2-D tensor, else the product."""
    if tensor.ndim <= 2:
        return int(tensor.shape[-1])
    width = 1
    for size in tensor.shape[1:]:
        width *= int(size)
    return width


def _kind_of(spec: Mapping[str, Any] | None) -> str:
    payload = spec or {}
    return str(payload.get("kind", payload.get("type", "identity"))).lower()


def _assert_width(view: Any, observed: int, *, side: str) -> None:
    """A resolved view is fixed; a later output of a different width is a bug."""
    expected = getattr(view, "in_dim", None)
    if expected is None and isinstance(view, IdentityView):
        expected = view.dim if view.dim != -1 else None
    if expected is not None and int(expected) != int(observed):
        raise ContractError(
            f"{side} output width changed from {expected} to {observed} after its view "
            "was resolved; a model's output width must be constant for a run"
        )


class _TorchOptimizerAdapter:
    """Lets a bare ``torch.optim.Optimizer`` on ``Peer.optimizer`` drive the step."""

    closure_based = False
    scaler = None
    grad_accumulation = 1
    _accumulated = 0

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> StepOutcome:
        self.optimizer.step()
        return StepOutcome(
            stepped=True, lr=float(self.optimizer.param_groups[0]["lr"]), accumulated=1
        )


def _clear_optimizer_state(optimizer: Any) -> None:
    """Drop momentum computed for pre-blend weights, for any optimizer flavour."""
    if optimizer is None:
        return
    if isinstance(optimizer, _TorchOptimizerAdapter):
        optimizer.optimizer.state.clear()
        return
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.state.clear()
        return
    backend = getattr(optimizer, "backend", None)
    if backend is None:
        return
    backends = (
        list(getattr(backend, "_backends", {}).values())
        if hasattr(backend, "_backends")
        else [backend]
    )
    for each in backends:
        native = getattr(each, "_instance", None)
        if native is not None:
            native.state.clear()


# ---------------------------------------------------------------------------
# step-scoped records
# ---------------------------------------------------------------------------
@dataclass
class PreparedPeer:
    """One peer's forward result and its projected embeddings."""

    path: str
    peer: Any
    logits: torch.Tensor
    values: torch.Tensor | None
    embedding: torch.Tensor | None
    teacher_embeddings: dict[str, torch.Tensor]
    view_pairs: dict[str, ViewPair]
    teacher_admission: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class PreparedStep:
    """Everything one optimizer step needs, computed before any backward."""

    batch: DistillBatch
    peers: dict[str, PreparedPeer]
    teachers: dict[str, TeacherOutput]
    teacher_bundles: dict[str, OutputBundle]
    ramp: float
    step: int
    verdicts: dict[str, ConformanceVerdict] = field(default_factory=dict)
    trust: dict[str, float] = field(default_factory=dict)
    active: tuple[str, ...] = ()
    failed: dict[str, str] = field(default_factory=dict)


class CoDistillStep:
    """TrainerProtocol implementation: one shared batch, N peers, M teachers."""

    def __init__(
        self,
        *,
        teachers: Sequence["Teacher | TeacherSpec | Mapping[str, Any]"],
        loss: Any = None,
        config: ObjectiveConfig | Mapping[str, Any] | None = None,
        views: Mapping[str, Any] | None = None,
        out_dim: int = 64,
        tau: float | None = None,
        beta: float | None = None,
        sigma_mode: str | None = None,
        regularizers: Sequence[Any] | None = None,
        weight_update: Any = None,
        aggregator: Aggregator | None = None,
        kappa: Any = None,
        peer_margin: float | None = None,
        sigma_floor: float = 0.0,
        sigma_cap: float = 1.0,
        sigma_decay: float = 0.9,
        auditors: Mapping[str, "TeacherAuditor | Any"] | None = None,
        trust: TrustState | None = None,
        runtime: Runtime | None = None,
        settings: Any | None = None,
        seed: int | None = None,
        strict: bool = True,
        conformance: Mapping[str, Any] | None = None,
    ) -> None:
        if not teachers:
            raise ImproperlyConfigured(
                "CoDistillStep requires at least one teacher; use the built-in "
                "individual step when no teacher is available"
            )
        self._settings = settings or get_settings()
        self.runtime = runtime or build_runtime(self._settings)
        self.seed = int(seed if seed is not None else self.runtime.seed)
        self.strict = bool(strict)
        self.out_dim = int(out_dim)
        if self.out_dim < 1:
            raise ValidationError(f"out_dim must be >= 1, got {out_dim!r}")

        self.config = _resolve_config(config, self._settings)
        if tau is not None:
            self.config = replace(self.config, tau=float(tau))
        if beta is not None:
            self.config = replace(self.config, beta=float(beta))
        if sigma_mode is not None:
            self.config = replace(self.config, sigma_mode=str(sigma_mode))
        if peer_margin is not None:
            self.config = replace(self.config, peer_margin=float(peer_margin))
        # Re-validate after the overrides: ``replace`` re-runs ``__post_init__``.
        if self.config.sigma_mode not in SIGMA_MODES:
            raise ValidationError(
                f"sigma_mode {self.config.sigma_mode!r} is unknown; accepted: "
                + ", ".join(sorted(SIGMA_MODES))
            )

        self.teachers: dict[str, Teacher] = {}
        for raw in teachers:
            teacher = (
                raw if isinstance(raw, Teacher) else build_teacher(raw, runtime=self.runtime)
            )
            if teacher.spec.name in self.teachers:
                raise ValidationError(
                    f"duplicate teacher name {teacher.spec.name!r}; teacher names must "
                    "be unique"
                )
            self.teachers[teacher.spec.name] = teacher
        self.teacher_names: tuple[str, ...] = tuple(sorted(self.teachers))

        # ---- conformance gate (section 4.6) --------------------------------
        explicit_conformance = dict(conformance or {})
        conformance = dict(self._settings.get("DISTILLATION", {}).get("CONFORMANCE", {}) or {})
        conformance.update(explicit_conformance)
        unknown = sorted(set(conformance) - CONFORMANCE_KEYS)
        if unknown:
            raise ValidationError(
                "DISTILLATION.CONFORMANCE does not accept "
                + ", ".join(repr(key) for key in unknown)
                + "; accepted keys: "
                + ", ".join(sorted(CONFORMANCE_KEYS))
            )
        self._conformance_enabled = bool(conformance.get("ENABLED", True))
        self._conformance = conformance
        self._profiles: dict[str, TrustProfile] = {
            name: self.teachers[name].spec.trust for name in self.teacher_names
        }
        self._auditors = self._resolve_auditors(auditors, conformance)
        self._audit_digests: dict[str, str] = {
            name: getattr(self._auditors[name], "digest", "")
            for name in self.teacher_names
            if name in self._auditors and getattr(self._auditors[name], "digest", "")
        }
        self._trust = trust or TrustState(
            init=float(conformance.get("TRUST_INIT", 1.0)),
            floor=float(conformance.get("TRUST_FLOOR", 0.0)),
            decay=float(conformance.get("TRUST_DECAY", 0.5)),
            restore=float(conformance.get("TRUST_RESTORE", 0.05)),
        )
        self._verify_declared_digests()

        self.views = dict(views or {})
        self._resolved_pairs: dict[str, dict[str, ViewPair]] = {}
        self.label_maps: dict[str, dict[str, LabelMap]] = {}

        self._loss_explicit = loss is not None
        self.loss = _resolve_loss(loss, self._settings)
        self.regularizers_spec = list(regularizers or ())
        self.regularizers: dict[str, list[Any]] = {}

        self.aggregator = aggregator
        self.kappa = kappa
        self.weight_update = build_weight_update(weight_update, aggregator=aggregator)
        self._private_aggregator: Aggregator | None = None

        self.sigma_floor = float(sigma_floor)
        self.sigma_cap = float(sigma_cap)
        self.sigma = SigmaState(decay=sigma_decay)
        self.state = DistillationState(
            enabled=True,
            teacher_names=self.teacher_names,
            teacher_digests={name: self.teachers[name].digest for name in self.teacher_names},
        )
        self._step = 0
        self._peers: dict[str, Any] = {}
        self._optimizers: dict[str, Any] = {}
        self._external_optimizers: Mapping[str, Any] = {}
        self._shared_reference: Any = None
        self._managed = False
        self._buffered: dict[str, DistillBatch] = {}
        self._buffered_meta: dict[str, dict[str, Any]] = {}
        self._view_specs: dict[str, dict[str, Any]] = {}
        self._teacher_view_specs: dict[tuple[str, str], dict[str, Any]] = {}
        self._views: dict[str, OutputView] = {}
        self._episode_counts: dict[str, int] = {}
        self._signal: dict[str, float] = {}
        self._eligible: dict[str, bool] = {}
        self._collapsed: dict[str, bool] = {}
        self.references: dict[str, Any] = {}
        self.last: dict[str, PeerLoss] = {}
        self.last_reports: dict[tuple[str, str], ContrastiveReport] = {}
        self.last_prepared: PreparedStep | None = None
        self.batch: DistillBatch | None = None

    @classmethod
    def from_settings(
        cls,
        config: DistillationConfig,
        *,
        tree: Any = None,
        optimizers: Mapping[str, Any] | None = None,
        reference: Any = None,
        loss: Any = None,
        runtime: Runtime | None = None,
        settings: Any | None = None,
        auditors: Mapping[str, Any] | None = None,
        trust: TrustState | None = None,
        weight_update: Any = None,
        aggregator: Aggregator | None = None,
        kappa: Any = None,
        seed: int | None = None,
    ) -> "CoDistillStep":
        """Expand a resolved :class:`DistillationConfig` into the constructor call.

        A convenience, not a second implementation: every key becomes the keyword
        ``__init__`` already takes. It reads the **already-resolved** config rather
        than re-reading settings, so the section 19.1 precedence is applied once.

        When ``optimizers`` is given the step is *managed*: the trainer owns the
        optimizers, the reference policy, and the aggregation order, so this step
        never blends and steps each peer through the trainer's optimizer (which
        carries accumulation, clipping, and AMP). ``tree`` is used to refuse a peer
        path that collides with a teacher name before any run work starts.
        """
        if not isinstance(config, DistillationConfig):
            raise ValidationError(
                f"from_settings needs a DistillationConfig, got {type(config).__name__}; "
                "resolve the argument with DistillationConfig.from_spec first"
            )
        if not config.enabled:
            raise ImproperlyConfigured(
                "CoDistillStep.from_settings was given a disabled DistillationConfig; "
                "an off run must not construct the step"
            )
        step = cls(
            teachers=list(config.teachers),
            loss=loss,
            config=ObjectiveConfig(**config.objective_kwargs()),
            views=dict(config.views),
            out_dim=config.out_dim,
            regularizers=list(config.regularizers),
            weight_update=weight_update,
            aggregator=aggregator,
            kappa=kappa,
            sigma_floor=config.sigma_floor,
            sigma_cap=config.sigma_cap,
            sigma_decay=config.sigma_decay,
            auditors=auditors,
            trust=trust,
            runtime=runtime,
            settings=settings,
            seed=seed,
            strict=config.strict,
            conformance=config.conformance,
        )
        if tree is not None:
            walker = getattr(tree, "walk", None)
            nodes = list(walker("root-first")) if callable(walker) else []
            clash = sorted(
                {str(getattr(node, "path", "")) for node in nodes} & set(step.teacher_names)
            )
            if clash:
                raise ValidationError(
                    "teacher name(s) "
                    + ", ".join(clash)
                    + " collide with peer path(s); teacher names and peer paths must be "
                    "disjoint"
                )
        if optimizers is not None:
            step._managed = True
            step._external_optimizers = optimizers
        if reference is not None:
            step._shared_reference = reference
        return step

    # -- construction helpers ---------------------------------------------
    def _verify_declared_digests(self) -> None:
        """A declared audit that names a different teacher digest is refused."""
        for name in self.teacher_names:
            auditor = self._auditors.get(name)
            verify = getattr(auditor, "verify_teacher", None)
            if callable(verify):
                verify(self.teachers[name].digest)

    def _resolve_auditors(
        self,
        declared: Mapping[str, Any] | None,
        conformance: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resolve one auditor per teacher, honouring ``CONFORMANCE.MODE``.

        Precedence per teacher: an explicit ``auditors[name]`` entry, then the
        teacher spec's ``audit`` field (or ``AUDIT_ARTIFACTS``), then the mode
        default. ``MODE="audit"`` with nothing resolvable is an
        ``ImproperlyConfigured``, not a silent downgrade to trust. ``MODE="declared"``
        refuses a ``TrustProfile.require_audit`` for the same reason in the other
        direction.

        ``ENABLED = False`` skips only the *fallback* auditor: an explicitly
        supplied auditor is still registered, because a caller who passed one has
        done the work.
        """
        mode = str(conformance.get("MODE", "hybrid"))
        if mode not in CONFORMANCE_MODES:
            raise ValidationError(
                f"DISTILLATION.CONFORMANCE.MODE={mode!r} is unknown; accepted: "
                + ", ".join(sorted(CONFORMANCE_MODES))
            )
        strict_undeclared = bool(conformance.get("STRICT_UNDECLARED", True))
        enabled = bool(conformance.get("ENABLED", True))
        artifacts = dict(conformance.get("AUDIT_ARTIFACTS", {}) or {})
        resolved: dict[str, Any] = {}
        explicit = dict(declared or {})

        for name in self.teacher_names:
            spec = self.teachers[name].spec
            profile = self._profiles[name]
            candidate = explicit.get(name)
            audit_path = spec.audit or artifacts.get(name)
            if candidate is None and audit_path:
                if audit_path == "guard":
                    raise ImproperlyConfigured(
                        f"teacher {name!r} declares audit='guard' but no guard binding "
                        "was supplied; pass auditors={...} with a GuardAudit whose "
                        "evidence_factory reads this teacher's bundle"
                    )
                candidate = DeclaredAudit.from_file(Path(str(audit_path)))

            if candidate is None:
                if not enabled:
                    # Gate off and no auditor of any kind: install nothing, so the
                    # gate admits every row at 1.0 and stamps source="disabled".
                    continue
                if strict_undeclared and not spec.trust.declared:
                    raise ImproperlyConfigured(
                        f"teacher {name!r} declares no trust profile and no auditor, and "
                        "DISTILLATION.CONFORMANCE.STRICT_UNDECLARED is true. Declare "
                        "'trust:' on the spec (a default profile tolerates tiers 3-6 at "
                        "a graduated admission and refuses 0-2), or configure an "
                        "auditor, or set STRICT_UNDECLARED: false to accept the default "
                        "profile."
                    )
                if mode == "audit":
                    raise ImproperlyConfigured(
                        f"DISTILLATION.CONFORMANCE.MODE='audit' but teacher {name!r} has "
                        "no auditor; configure auditors={...}, set the spec's 'audit' "
                        "path, or switch MODE to 'hybrid'"
                    )
                if mode == "declared" and profile.require_audit:
                    raise ImproperlyConfigured(
                        f"teacher {name!r} has require_audit=True but MODE='declared' "
                        "supplies no auditor; switch MODE to 'hybrid' or 'audit'"
                    )
                candidate = TrustOnlyAudit(name=name)
            resolved[name] = candidate
        return resolved

    def _path_of(self, node: Any) -> str:
        """The peer path for a node, matching ``setup``'s keying exactly.

        A ``PeerNode`` carries ``path``; a bare ``Peer`` carries ``name``. Using the
        same expression everywhere keeps ``_peers``, ``_buffered`` and the score
        mapping keyed identically.
        """
        peer = getattr(node, "peer", node)
        path = getattr(node, "path", None)
        if path is None:
            path = getattr(peer, "name", None)
        if path is None:
            raise ContractError(
                f"cannot determine a peer path for {type(node).__name__}; a node must "
                "expose .path or .name"
            )
        return str(path)

    def _emit(self, signal: Any, **payload: Any) -> None:
        signal.send(sender=self, **payload)

    # -- wiring to MACTFTrainer -------------------------------------------
    def attach(self, trainer: Any) -> None:
        """Bind to a ``MACTFTrainer`` so it owns optimizers, reference, and blending.

        Under the trainer the epoch order is fixed by document 08: scoring and the
        configured aggregation strategy run *before* ``on_epoch_end``. A second
        blend here would average parameters that are already identical and reset
        optimizer state a second time, so a managed step never blends. The trainer's
        optimizers (``MACTFOptimizer`` with accumulation, clipping, AMP) replace the
        bare optimizer on a ``Peer``, and its reference policy replaces a private
        deep copy, so the KL baseline is the same one the plain run uses.
        """
        self._managed = True
        self._external_optimizers = getattr(trainer, "optimizers", {}) or {}
        self._shared_reference = getattr(trainer, "reference", None)
        if not self._loss_explicit:
            loss_function = getattr(trainer, "loss_function", None)
            if loss_function is not None:
                self.loss = loss_function

    @property
    def managed(self) -> bool:
        return self._managed

    # -- TrainerProtocol --------------------------------------------------
    def setup(self, node: Any) -> None:
        """Bind views and regularizers for one peer. Idempotent."""
        path = self._path_of(node)
        if path in self._peers:
            return
        if path in self.teachers:
            raise ValidationError(
                f"peer path {path!r} collides with a teacher name; teacher names and "
                "peer paths must be disjoint"
            )
        peer = getattr(node, "peer", node)
        if peer is None:
            raise ContractError(f"node {path!r} has no peer")
        self._peers[path] = peer
        optimizer = self._external_optimizers.get(path) or self._external_optimizers.get(
            getattr(peer, "name", path)
        )
        if optimizer is None:
            candidate = getattr(peer, "optimizer", None)
            if isinstance(candidate, MACTFOptimizer):
                optimizer = candidate
            elif isinstance(candidate, torch.optim.Optimizer):
                optimizer = _TorchOptimizerAdapter(candidate)
        if optimizer is not None:
            self._optimizers[path] = optimizer
        self._bind_views(path, peer)
        # Rebuilt once per newly discovered peer over the accumulated peer set,
        # because a ``kl_to_reference`` regularizer must point at its own peer's
        # frozen reference and cannot be built before that peer is known.
        self.regularizers = build_regularizers(
            self.regularizers_spec,
            peers=self._peers,
            settings={"strict": self.strict},
            references={name: self._reference_for(name) for name in self._peers}
            if self.regularizers_spec
            else None,
        )

    def train_step(self, node: Any, batch: Mapping[str, Any]) -> StepResult:
        """Called once per peer per step. Runs the cohort step on the last peer.

        ``TrainerProtocol`` has no cohort concept, so the honest implementation is a
        rendezvous: buffer each peer's batch, and when every peer has arrived
        execute one cohort step and return the result to all of them. The loss this
        policy minimizes *is* the cohort loss; a per-peer return value would have to
        invent a number, and ``L_i`` for the first peer to arrive is not computable
        yet.

        The buffered batches must be the same batch. That is checked, not assumed.
        """
        path = self._path_of(node)
        if path not in self._peers:
            self.setup(node)
        if path in self._buffered:
            raise ContractError(
                f"peer {path!r} arrived twice before the cohort completed; "
                "CoDistillStep runs one cohort step per arrival set, so the trainer must "
                "give every peer exactly one train_step per cohort step "
                "(set steps_per_epoch=1) and every peer must participate"
            )
        coerced = coerce_batch(batch)
        self._buffered[path] = coerced
        self._buffered_meta[path] = dict(coerced.extras)

        if len(self._buffered) < len(self._peers):
            # Not every peer has arrived. There is no number to report yet, and
            # reporting a real-looking loss would be a lie a caller could act on.
            return StepResult(
                loss=torch.zeros((), dtype=torch.float32),
                breakdown=LossBreakdown(),
                stepped=False,
                metrics={"pending": 1.0, "buffered": float(len(self._buffered))},
            )

        ordered = sorted(self._buffered)
        canonical = self._buffered[ordered[0]]
        for other in ordered:
            candidate = self._buffered[other]
            if candidate.states.shape != canonical.states.shape or not torch.equal(
                candidate.states, canonical.states
            ):
                self._buffered.clear()
                self._buffered_meta.clear()
                raise ContractError(
                    f"peer {other!r} submitted a different batch for the same step; "
                    "every peer and teacher must see one identical states tensor, "
                    "which is invariant I1 of this policy"
                )

        meta = {name: dict(self._buffered_meta[name]) for name in ordered}
        self._buffered.clear()
        self._buffered_meta.clear()

        signal: dict[str, float] | None = None
        eligible: dict[str, bool] | None = None
        cards: dict[str, Any] | None = None
        if all("signal" in meta[name] for name in ordered):
            signal = {name: float(meta[name]["signal"]) for name in ordered}
        if all("void" in meta[name] for name in ordered):
            eligible = {name: not bool(meta[name]["void"]) for name in ordered}
        if any(("void" in meta[name] or "floor" in meta[name]) for name in ordered):
            cards = {
                name: SimpleNamespace(
                    is_void=bool(meta[name].get("void", False)),
                    floor=float(meta[name].get("floor", 0.0)),
                )
                for name in ordered
            }
        return self.step(
            self._peers,
            canonical,
            scorecards=cards,
            signal=signal,
            eligible=eligible,
        )

    def on_epoch_end(self, tree: Any, result: EpochResult) -> None:
        """Blend parameters through the configured strategy, teachers excluded.

        Scoring has already happened before this hook, so the weights use this
        epoch's scores. Under ``MACTFTrainer`` the trainer's own strategy already
        aggregated, so a managed step only reports an incomplete cohort.
        """
        if self._buffered:
            missing = sorted(set(self._peers) - set(self._buffered))
            logger.warning(
                "epoch ended with an incomplete cohort: peer(s) %s did not arrive, so "
                "%d buffered batch(es) were discarded",
                ", ".join(missing),
                len(self._buffered),
            )
            self._buffered.clear()
            self._buffered_meta.clear()
        if self._managed:
            return None
        if not bool(getattr(result, "completed", True)):
            return None
        if not self._peers:
            return None
        report = self.update_weights()
        self.blend(report)
        return None

    # -- views --------------------------------------------------------------
    def _bind_views(self, path: str, peer: Any) -> None:
        """Record one peer's view specs. Resolution happens in ``prepare``."""
        self._view_specs[path] = dict(self.views.get(path) or self.views.get("*") or {})
        for name in self.teacher_names:
            override = self.views.get(f"{path}:{name}")
            spec = override or self.views.get(name) or self.teachers[name].spec.view or {}
            self._teacher_view_specs[(path, name)] = dict(spec)

    def _target_dim(self, spec: Mapping[str, Any]) -> int:
        requested = (spec or {}).get("out_dim")
        return int(requested) if requested else self.out_dim

    def _resolve_view(
        self,
        *,
        side: str,
        spec: Mapping[str, Any],
        observed_dim: int,
        seed: int,
        labels: Sequence[str] | None = None,
        label_map: LabelMap | None = None,
    ) -> OutputView:
        """Build one side's view for an observed width, with the width recorded."""
        kind = _kind_of(spec)
        weightless = "weight" not in (spec or {})
        if (
            weightless
            and kind in {"linear", "fixed_random"}
            and observed_dim == self._target_dim(spec)
        ):
            # A random projection whose input already has the target width would
            # rotate a perfectly good embedding for no reason and lose nothing but
            # comparability. The substitution is visible in ``describe()``.
            return IdentityView(self._target_dim(spec))
        return build_view(
            side=side,
            spec=spec,
            in_dim=observed_dim,
            out_dim=self.out_dim,
            seed=seed,
            labels=labels,
            label_map=label_map,
        )

    @staticmethod
    def _observed(bundle: OutputBundle, spec: Mapping[str, Any]) -> int:
        """The width a view built from ``spec`` will see in ``bundle``."""
        kind = _kind_of(spec)
        payload = spec or {}
        key = payload.get("key")
        if kind == "hidden" and key:
            tensor = bundle.extra.get(str(key))
            if tensor is None:
                raise ContractError(
                    f"hidden view on {bundle.source} {bundle.side!r} found no key {key!r}; "
                    "available: " + (", ".join(sorted(bundle.extra)) or "(none)")
                )
            if tensor.ndim == 3 and str(payload.get("pool", "mean")) == "flat":
                return int(tensor.shape[1] * tensor.shape[2])
            return int(tensor.shape[-1])
        if kind == "identity" and key:
            tensor = bundle.extra.get(str(key))
            if tensor is not None:
                return _flat_width(tensor)
        if kind == "distribution":
            tensor = bundle.extra.get(str(payload.get("key", "logits")))
            if tensor is None:
                tensor = bundle.logits
            if tensor is None:
                raise ContractError(
                    f"distribution view on {bundle.source} {bundle.side!r} found no logits"
                )
            return _flat_width(tensor)
        tensor = bundle.hidden if bundle.hidden is not None else bundle.logits
        if tensor is None:
            raise ContractError(
                f"{bundle.source} {bundle.side!r} produced no tensor to project from"
            )
        return _flat_width(tensor)

    def _distribution_pair(
        self,
        *,
        path: str,
        name: str,
        peer_spec: Mapping[str, Any],
        teacher_spec: Mapping[str, Any],
        peer_out: OutputBundle,
        teacher_out: OutputBundle,
    ) -> ViewPair:
        """A label-aligned pair: both sides are distributions over one shared axis."""
        if _kind_of(peer_spec) != "distribution" or _kind_of(teacher_spec) != "distribution":
            raise ValidationError(
                f"peer {path!r} and teacher {name!r}: a distribution view must be "
                "declared on both sides, because a distribution and an embedding "
                "cannot be compared"
            )
        teacher = self.teachers[name]
        peer_vocab = tuple(
            str(item)
            for item in (peer_spec.get("source_labels") or peer_spec.get("label_names") or ())
        )
        teacher_vocab = tuple(teacher.spec.label_names) or tuple(
            str(item)
            for item in (
                teacher_spec.get("source_labels") or teacher_spec.get("label_names") or ()
            )
        )
        axis = tuple(
            str(item) for item in (peer_spec.get("labels") or teacher_spec.get("labels") or ())
        )
        if not axis:
            if not (peer_vocab and teacher_vocab):
                raise ValidationError(
                    f"distribution pair ({path!r}, {name!r}) needs 'labels' on a view, "
                    "or both vocabularies (peer 'source_labels' and the teacher's "
                    "'label_names') so the shared axis can be computed"
                )
            axis = shared_axis({path: peer_vocab, name: teacher_vocab})
        if not axis:
            raise ValidationError(
                f"peer {path!r} and teacher {name!r} share no label; there is nothing to compare"
            )
        maps: dict[str, LabelMap | None] = {}
        for side, vocab, spec, bundle in (
            ("peer", peer_vocab, peer_spec, peer_out),
            ("teacher", teacher_vocab, teacher_spec, teacher_out),
        ):
            observed = self._observed(bundle, spec)
            if vocab:
                maps[side] = build_label_map(
                    vocab,
                    axis,
                    source_size=observed,
                    aliases=spec.get("aliases"),
                )
            elif observed == len(axis):
                maps[side] = None
            else:
                raise ValidationError(
                    f"{side} {'peer ' + path if side == 'peer' else 'teacher ' + name} "
                    f"has {observed} outputs but the shared axis has {len(axis)} labels "
                    "and no vocabulary was declared to map between them"
                )
        self.label_maps.setdefault(path, {})[name] = maps["teacher"]  # type: ignore[assignment]
        peer_view = build_view(
            side=f"peer:{path}",
            spec=peer_spec,
            in_dim=None,
            out_dim=len(axis),
            seed=node_seed(self.seed, f"view:peer:{path}:{name}"),
            labels=axis,
            label_map=maps["peer"],
        )
        teacher_view = build_view(
            side=f"teacher:{name}",
            spec=teacher_spec,
            in_dim=None,
            out_dim=len(axis),
            seed=node_seed(self.seed, f"view:teacher:{name}:{path}"),
            labels=axis,
            label_map=maps["teacher"],
        )
        self._views[f"peer:{path}:{name}"] = peer_view
        self._views[f"teacher:{name}:{path}"] = teacher_view
        return ViewPair(peer=peer_view, teacher=teacher_view, dim=len(axis))

    def _pair_for(
        self,
        path: str,
        name: str,
        peer_out: OutputBundle,
        teacher_out: OutputBundle,
    ) -> ViewPair:
        """Resolve (and then reuse) the view pair for one ``(peer, teacher)``."""
        peer_spec = self._view_specs.get(path, {})
        teacher_spec = self._teacher_view_specs.get((path, name), {})

        if "distribution" in (_kind_of(peer_spec), _kind_of(teacher_spec)):
            existing_peer = self._views.get(f"peer:{path}:{name}")
            existing_teacher = self._views.get(f"teacher:{name}:{path}")
            if existing_peer is not None and existing_teacher is not None:
                return ViewPair(
                    peer=existing_peer,
                    teacher=existing_teacher,
                    dim=int(getattr(existing_peer, "dim", self.out_dim)),
                )
            return self._distribution_pair(
                path=path,
                name=name,
                peer_spec=peer_spec,
                teacher_spec=teacher_spec,
                peer_out=peer_out,
                teacher_out=teacher_out,
            )

        # Resolved once per peer, then reused for every step and every teacher pair:
        # two pairs landing on two different random matrices would compare the peer
        # to two different geometries at once.
        peer_key = f"peer:{path}"
        peer_observed = self._observed(peer_out, peer_spec)
        peer_view = self._views.get(peer_key)
        if peer_view is None:
            peer_view = self._resolve_view(
                side=peer_key,
                spec=peer_spec,
                observed_dim=peer_observed,
                seed=node_seed(self.seed, f"view:peer:{path}"),
            )
            self._views[peer_key] = peer_view
        else:
            _assert_width(peer_view, peer_observed, side=peer_key)

        # Resolved once per teacher and shared by every peer, so two peers are
        # compared to the teacher in the same geometry. A peer-specific override
        # is the one reason to resolve a teacher view per pair.
        specific = f"{path}:{name}" in self.views
        teacher_key = f"teacher:{name}@{path}" if specific else f"teacher:{name}"
        teacher_observed = self._observed(teacher_out, teacher_spec)
        teacher_view = self._views.get(teacher_key)
        if teacher_view is None:
            teacher_view = self._resolve_view(
                side=teacher_key,
                spec=teacher_spec,
                observed_dim=teacher_observed,
                seed=node_seed(self.seed, f"view:teacher:{name}"),
            )
            self._views[teacher_key] = teacher_view
        else:
            _assert_width(teacher_view, teacher_observed, side=teacher_key)
        return ViewPair(peer=peer_view, teacher=teacher_view, dim=self._target_dim(peer_spec))

    # -- conformance and trust ---------------------------------------------
    def _note_trust_change(self, name: str, previous: float, updated: float) -> None:
        """Emit ``distillation_trust_collapsed`` on the crossing, not on every step."""
        minimum = float(self._conformance.get("TRUST_MIN", 0.05))
        if previous > minimum >= updated:
            self._emit(
                signals.distillation_trust_collapsed,
                teacher=name,
                trust=updated,
                reason="below TRUST_MIN",
                step=self._step,
            )
        elif previous > self._trust.floor and updated <= self._trust.floor:
            self._emit(
                signals.distillation_trust_collapsed,
                teacher=name,
                trust=updated,
                reason="at trust_floor",
                step=self._step,
            )
        elif updated < previous:
            logger.debug("teacher %s trust decayed %.4f -> %.4f", name, previous, updated)

    def _sigma_trust_min(self) -> float:
        """The trust floor below which a teacher leaves a peer's sigma.

        ``CONFORMANCE.TRUST_MIN`` when ``SIGMA_TRUST_MIN`` is true (the default), and
        ``0.0`` otherwise. ``sigma_weights`` tests ``> trust_min``, so a zero trust is
        never live either way.
        """
        if not bool(self._conformance.get("SIGMA_TRUST_MIN", True)):
            return 0.0
        return float(self._conformance.get("TRUST_MIN", 0.05))

    def _conformance_gate(
        self,
        batch: DistillBatch,
        bundles: Mapping[str, OutputBundle],
        names: Sequence[str],
    ) -> dict[str, ConformanceVerdict]:
        """Audit every teacher, isolating an auditor failure like a forward failure.

        An audit that raises is **not** a reason to admit the teacher: its verdict
        becomes an all-zero admission with ``hard_denied=False`` and a ``reasons``
        entry naming the exception. The run continues for the other teachers.
        """
        verdicts: dict[str, ConformanceVerdict] = {}
        enabled = bool(self._conformance_enabled)
        for name in names:
            auditor = self._auditors.get(name)
            profile = self._profiles[name]
            if auditor is None:
                verdicts[name] = ConformanceVerdict(
                    teacher=name,
                    admission=torch.ones((batch.batch,), dtype=torch.float32),
                    trust=float(self._trust.get(name)),
                    audit_mask=torch.zeros((batch.batch,), dtype=torch.bool),
                    source="disabled" if not enabled else "none",
                )
                continue
            try:
                verdict = auditor.audit(bundles[name], batch, profile=profile)
            except Exception as exc:  # noqa: BLE001 - an auditor is user code
                verdict = ConformanceVerdict(
                    teacher=name,
                    admission=torch.zeros((batch.batch,), dtype=torch.float32),
                    trust=float(self._trust.get(name)),
                    audit_mask=torch.ones((batch.batch,), dtype=torch.bool),
                    reasons=(
                        f"auditor {getattr(auditor, 'name', type(auditor).__name__)!r} "
                        f"raised {type(exc).__name__}: {exc}",
                    ),
                    source="audit-error",
                )
                self._emit(
                    signals.distillation_conformance_violation,
                    teacher=name,
                    rules=[],
                    severity="error",
                    masked_fraction=1.0,
                    hard_denied=False,
                    step=self._step,
                    reason=f"auditor raised {type(exc).__name__}",
                )
            # Trust is the gate's own state and enters the objective through
            # ``sigma_weights`` alone. The verdict's ``trust`` is informational and
            # ``admission`` stays exactly as the audit produced it; folding trust
            # into admission too would apply it twice.
            verdict = replace(verdict, trust=float(self._trust.get(name)))
            if verdict.hard_denied:
                self._emit(
                    signals.distillation_conformance_violation,
                    teacher=name,
                    rules=list(verdict.rules),
                    severity=verdict.severity.label,
                    masked_fraction=verdict.masked_fraction,
                    hard_denied=True,
                    step=self._step,
                    reason="non-offsettable rule (I7)",
                )
            elif verdict.masked_fraction > 0.0 or verdict.rules:
                self._emit(
                    signals.distillation_conformance_violation,
                    teacher=name,
                    rules=list(verdict.rules),
                    severity=verdict.severity.label,
                    masked_fraction=verdict.masked_fraction,
                    hard_denied=False,
                    step=self._step,
                    reason="offsettable violation",
                )
            verdicts[name] = verdict
        return verdicts

    def _teacher_forward(
        self, name: str, states: torch.Tensor
    ) -> tuple[TeacherOutput | None, str | None]:
        """One teacher pass; a raise excludes that teacher for the step only."""
        teacher = self.teachers[name]
        try:
            return teacher.forward(states, reset=True), None
        except Exception as exc:  # noqa: BLE001 - a teacher is an external artifact
            reason = f"{type(exc).__name__}: {exc}"
            self._emit(
                signals.distillation_teacher_error,
                teacher=name,
                error=reason,
                action="excluded from this step",
            )
            logger.warning(
                "teacher %s failed during the forward pass; excluding it from this step: %s",
                name,
                reason,
            )
            return None, reason

    # -- prepare ------------------------------------------------------------
    def prepare(
        self,
        nodes: Mapping[str, Any] | Sequence[Any],
        batch: Any,
        *,
        signal: Mapping[str, float] | None = None,
        eligible: Mapping[str, bool] | None = None,
    ) -> PreparedStep:
        """Run every forward pass for one step, with no gradients retained.

        Order is fixed and matters:

        1. Clear teacher caches, so no step can observe a previous step's output.
        2. One teacher pass each, on the shared states, in sorted name order. A
           teacher that raises is excluded for this step.
        3. Audit each teacher and convert the verdict into a per-sample admission.
        4. Update each teacher's trust from its verdict; sigma then reads it.
        5. One peer forward each (no grad), in sorted path order.
        6. Resolve each side's view from the observed width, once, and fix it.
        7. Project both sides for every (peer, teacher) pair.
        """
        self.batch = coerce_batch(batch)
        n = self.batch.batch
        if n < 2:
            raise ValidationError(
                f"batch size {n} is too small; contrastive distillation needs >= 2"
            )
        self._signal = dict(signal or {})
        self._eligible = dict(eligible or {})

        for teacher in self.teachers.values():
            teacher.clear_cache()

        teacher_outputs: dict[str, TeacherOutput] = {}
        teacher_bundles: dict[str, OutputBundle] = {}
        failed: dict[str, str] = {}
        for name in self.teacher_names:
            output, reason = self._teacher_forward(name, self.batch.states)
            if output is None:
                failed[name] = str(reason)
                continue
            teacher_outputs[name] = output
            teacher_bundles[name] = teacher_bundle(output, name=name)
        active = tuple(name for name in self.teacher_names if name in teacher_outputs)

        verdicts = self._conformance_gate(self.batch, teacher_bundles, active)
        # ``observe`` runs after the audit and before sigma, so the weight vector
        # reflects this step's finding immediately. A one-step lag would let a
        # teacher keep a full weight on the very step it fabricated.
        for name in active:
            previous = self._trust.get(name)
            updated = self._trust.observe(name, verdicts[name])
            self._note_trust_change(name, previous, updated)
        trust = self._trust.weights(self.teacher_names)

        peers: dict[str, PreparedPeer] = {}
        for path in sorted(self._peers):
            prepared = self._prepare_peer(path, self.batch, teacher_bundles, active)
            prepared.teacher_admission = {
                name: verdicts[name].gather(n) for name in active
            }
            peers[path] = prepared
        self._assert_view_pairs()
        prepared_step = PreparedStep(
            batch=self.batch,
            peers=peers,
            teachers=teacher_outputs,
            teacher_bundles=teacher_bundles,
            ramp=ramp_factor(
                self._step,
                warmup_steps=self.config.warmup_steps,
                ramp_steps=self.config.ramp_steps,
            ),
            step=self._step,
            verdicts=verdicts,
            trust=trust,
            active=active,
            failed=failed,
        )
        self.last_prepared = prepared_step
        return prepared_step

    def _prepare_peer(
        self,
        path: str,
        batch: DistillBatch,
        teacher_bundles: Mapping[str, OutputBundle],
        active: Sequence[str],
    ) -> PreparedPeer:
        peer = self._peers[path]
        kwargs = batch.kwargs_for(path)
        peer.adapter.train()
        with torch.no_grad(), autocast(self.runtime):
            logits, values = peer.adapter(batch.states, **kwargs)
        peer_out = peer_bundle(logits.detach(), path=path)

        pairs: dict[str, ViewPair] = {}
        teacher_embeddings: dict[str, torch.Tensor] = {}
        primary: torch.Tensor | None = None
        for name in active:
            pair = self._pair_for(path, name, peer_out, teacher_bundles[name])
            pairs[name] = pair
            teacher_embeddings[name] = pair.teacher(teacher_bundles[name])
            if primary is None:
                primary = pair.peer(peer_out)
        if primary is None:
            # No teacher survived: the sibling margin still needs an embedding, and a
            # plain peer view needs no teacher to be resolved.
            spec = self._view_specs.get(path, {})
            if _kind_of(spec) != "distribution":
                key = f"peer:{path}"
                view = self._views.get(key)
                observed = self._observed(peer_out, spec)
                if view is None:
                    view = self._resolve_view(
                        side=key,
                        spec=spec,
                        observed_dim=observed,
                        seed=node_seed(self.seed, f"view:peer:{path}"),
                    )
                    self._views[key] = view
                primary = view(peer_out)
        self._resolved_pairs[path] = pairs
        return PreparedPeer(
            path=path,
            peer=peer,
            logits=logits.detach(),
            values=None if values is None else values.detach(),
            embedding=None if primary is None else primary.detach(),
            teacher_embeddings=teacher_embeddings,
            view_pairs=pairs,
        )

    def _assert_view_pairs(self) -> None:
        """Every peer must have used one peer view across its teacher pairs."""
        for path, pairs in self._resolved_pairs.items():
            if _kind_of(self._view_specs.get(path, {})) == "distribution":
                continue
            distinct = {id(pair.peer) for pair in pairs.values()}
            if len(distinct) > 1:
                raise ContractError(
                    f"peer {path!r} was compared to its teachers through "
                    f"{len(distinct)} different peer views; one geometry per peer is required"
                )

    # -- step ---------------------------------------------------------------
    def step(
        self,
        nodes: Mapping[str, Any] | Sequence[Any],
        batch: Any,
        *,
        scorecards: Mapping[str, Any] | None = None,
        signal: Mapping[str, float] | None = None,
        eligible: Mapping[str, bool] | None = None,
    ) -> StepResult:
        """One cohort step: prepare, per-peer loss, per-peer backward, per-peer step.

        Per-peer ordering is sorted, and each peer's backward happens immediately
        after its own loss is assembled and before the next peer's forward, so no
        two peers' graphs are alive at once.
        """
        for raw_node in nodes.values() if isinstance(nodes, Mapping) else nodes:
            self.setup(raw_node)
        if not self._peers:
            raise ContractError(
                "step() called before setup(); call setup(node) for every peer first, "
                "or pass the peers as `nodes`"
            )

        prepared = self.prepare(nodes, batch, signal=signal, eligible=eligible)
        prepared_signal = self._resolve_signal(signal)
        self._signal = dict(prepared_signal)
        self._eligible = {
            path: bool((eligible or {}).get(path, True)) for path in sorted(self._peers)
        }
        centring = cohort_signal(prepared_signal, eligible=self._eligible)

        self.last = {}
        self.last_reports = {}
        totals: list[float] = []
        stepped_any = False
        outcomes: dict[str, Any] = {}

        for path in sorted(prepared.peers):
            optimizer = self._optimizers.get(path)
            if optimizer is None:
                raise ContractError(
                    f"peer {path!r} has no optimizer; CoDistillStep steps each peer "
                    "through its own optimizer. Give the Peer an optimizer, or attach "
                    "the step to a MACTFTrainer"
                )
            if getattr(optimizer, "closure_based", False):
                raise ContractError(
                    f"peer {path!r} uses a closure-driven optimizer; the compound loss "
                    "is evaluated once per step and cannot be re-evaluated by a closure"
                )
            if int(getattr(optimizer, "_accumulated", 0)) == 0:
                optimizer.zero_grad(set_to_none=True)
            loss = self._peer_loss(
                prepared=prepared,
                entry=prepared.peers[path],
                signal=prepared_signal,
                centring=float(centring.get(path, 0.0)),
                scorecards=scorecards,
            )
            self.last[path] = loss

            accumulation = int(getattr(optimizer, "grad_accumulation", 1) or 1)
            scaled = loss.total / accumulation if accumulation > 1 else loss.total
            scaler = getattr(optimizer, "scaler", None)
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            outcome = optimizer.step()
            outcomes[path] = outcome
            stepped_any = stepped_any or bool(getattr(outcome, "stepped", False))
            totals.append(float(loss.total.detach().item()))

        for path in sorted(self.last):
            for name, value in self.last[path].per_teacher.items():
                # The EMA consumes -L_ij, so a better match is a higher quality.
                self.sigma.update(path, name, value)
            self._episode_counts[path] = self._episode_counts.get(path, 0) + 1

        self._check_sigma_collapse()
        self._step += 1
        self._record_state(prepared)
        mean = math.fsum(totals) / len(totals) if totals else 0.0
        metrics: dict[str, float] = {
            "loss": mean,
            "peers": float(len(totals)),
            "teachers_active": float(len(prepared.active)),
            "ramp": float(prepared.ramp),
        }
        if not prepared.active:
            metrics["individual_only"] = 1.0
        for path in sorted(self.last):
            metrics[f"peer.{path}"] = float(self.last[path].total.detach().item())
            for name in self.teacher_names:
                metrics[f"sigma.{path}.{name}"] = float(self.last[path].sigma.get(name, 0.0))
            if path in outcomes:
                metrics[f"grad_norm.{path}"] = float(
                    getattr(outcomes[path], "grad_norm", None) or 0.0
                )
        self._emit(
            signals.distillation_step,
            step=self._step,
            losses={
                path: float(value.total.detach().item()) for path, value in self.last.items()
            },
            sigma={path: dict(value.sigma) for path, value in self.last.items()},
            metrics=dict(metrics),
        )
        self._log_step(prepared)
        return StepResult(
            loss=torch.tensor(mean, dtype=torch.float32),
            breakdown=self._combined_breakdown(),
            stepped=stepped_any,
            metrics=metrics,
        )

    def _check_sigma_collapse(self) -> None:
        """Emit ``distillation_sigma_collapsed`` on the crossing above 0.9."""
        for path in sorted(self.last):
            sigma = self.last[path].sigma
            if len(sigma) < 2:
                self._collapsed[path] = False
                continue
            top = max(sigma.values())
            collapsed = top > _COLLAPSE_THRESHOLD
            if collapsed and not self._collapsed.get(path, False):
                self._emit(
                    signals.distillation_sigma_collapsed,
                    peer=path,
                    sigma=dict(sigma),
                    max_weight=float(top),
                )
            self._collapsed[path] = collapsed

    def _record_state(self, prepared: PreparedStep) -> None:
        row: dict[str, float] = {"step": float(self._step - 1)}
        for path in sorted(self.last):
            loss = self.last[path]
            row[f"{path}|ind"] = float(loss.ind.detach().item())
            row[f"{path}|cd"] = float(loss.cd.detach().item())
        self.state.step = self._step
        self.state.record(row)

    def _log_step(self, prepared: PreparedStep) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        lines = [
            f"distill step={self._step - 1} ramp={prepared.ramp:.2f} "
            f"peers={len(self.last)} teachers={len(prepared.active)}"
        ]
        for path in sorted(self.last):
            loss = self.last[path]
            sigma = " ".join(f"{k}={v:.3f}" for k, v in sorted(loss.sigma.items()))
            lines.append(
                f"  {path} ind={float(loss.ind.detach().item()):.4f} "
                f"cd={float(loss.cd.detach().item()):.4f} sigma[{sigma}] "
                f"margin={float(loss.margin.detach().item()):.4f} "
                f"total={float(loss.total.detach().item()):.4f}"
            )
        logger.info("\n".join(lines))

    # -- one peer's loss -----------------------------------------------------
    def _peer_loss(
        self,
        *,
        prepared: PreparedStep,
        entry: PreparedPeer,
        signal: Mapping[str, float],
        centring: float,
        scorecards: Mapping[str, Any] | None,
    ) -> PeerLoss:
        """Assemble one peer's compound loss. The only place a peer graph is built."""
        path = entry.path
        peer = entry.peer
        batch = prepared.batch
        kwargs = batch.kwargs_for(path)
        card = (scorecards or {}).get(path)

        # -- individual term: gradient-carrying forward ---------------------
        peer.adapter.train()
        with autocast(self.runtime):
            logits, values = peer.adapter(batch.states, **kwargs)

        policy = make_policy(logits)
        with torch.no_grad():
            reference_logits, _ = self._reference_for(path).adapter(batch.states, **kwargs)
        reference_policy = make_policy(reference_logits)
        actions = greedy_action(policy).detach()

        advantage = self._advantage(
            path=path,
            signal=float(signal.get(path, 0.0)),
            card=card,
            batch=batch,
            device=logits.device,
            dtype=logits.dtype,
        )
        floor = self._compliance_floor(path, scorecards)
        void_mask = self._void_mask(path, batch, card, logits.device)
        signal_tensor = torch.full(
            (batch.batch,), float(signal.get(path, 0.0)), device=logits.device, dtype=logits.dtype
        )
        ind_batch = {
            "policy": policy,
            "reference": reference_policy,
            "actions": actions,
            "advantages": advantage,
            "values": values if values is not None else torch.zeros_like(logits[:, :1]),
            # ``local_update`` targets the value head at the *epoch signal*, not at
            # the advantage; using the advantage would train a different critic.
            "value_targets": signal_tensor.unsqueeze(-1),
            "compliance_floor": torch.tensor(float(floor), device=logits.device),
            "void_mask": void_mask,
        }
        individual, breakdown = individual_term(self.loss, ind_batch)

        # -- contrastive terms ----------------------------------------------
        # The embeddings that carry gradient are rebuilt from the live logits above.
        # Those cached in `prepare` were computed without grad and serve as
        # *siblings* and *hard negatives*, which must not be pushed.
        live_bundle = peer_bundle(logits, path=path)
        live_cache: dict[int, torch.Tensor] = {}

        def live_embedding(view: OutputView) -> torch.Tensor:
            key = id(view)
            if key not in live_cache:
                live_cache[key] = view(live_bundle)
            return live_cache[key]

        own_primary: torch.Tensor | None = None
        own_by_teacher: dict[str, torch.Tensor] = {}
        for name in prepared.active:
            own_by_teacher[name] = live_embedding(entry.view_pairs[name].peer)
            if own_primary is None:
                own_primary = own_by_teacher[name]
        if own_primary is None and entry.embedding is not None:
            plain = self._views.get(f"peer:{path}")
            if plain is not None:
                own_primary = live_embedding(plain)

        siblings = {
            other: prepared.peers[other].embedding
            for other in sorted(prepared.peers)
            if other != path and prepared.peers[other].embedding is not None
        }
        if own_primary is not None and siblings:
            margin = peer_margin(
                own_primary, siblings, margin=self.config.peer_margin, eps=self.config.eps
            )
        else:
            margin = torch.zeros((), device=logits.device, dtype=torch.float32)

        per_teacher: dict[str, torch.Tensor] = {}
        reports: dict[str, ContrastiveReport] = {}
        for name in prepared.active:
            admit = entry.teacher_admission.get(name)
            if admit is not None and not bool((admit > 0).any()):
                # Fully denied: record a zero term without calling the loss. Calling
                # it would raise, and raising would let one discredited teacher
                # stop a cohort whose other teachers are fine.
                zero = torch.zeros((), device=logits.device)
                per_teacher[name] = zero
                reports[name] = ContrastiveReport(
                    loss=zero,
                    positives=zero,
                    negatives=zero,
                    logits_mean=float("nan"),
                    logits_std=0.0,
                    batch=int(batch.batch),
                    temperature=float(self.config.tau),
                    mask_entropy=0.0,
                    masked_rows=int(batch.batch),
                    admitted_rows=0,
                    soft_admission=0.0,
                    clamped=False,
                )
                self.last_reports[(path, name)] = reports[name]
                continue
            hard, hard_admission = self._hard_negatives(
                anchor_path=path, teacher=name, prepared=prepared
            )
            term, report = contrastive_loss(
                own_by_teacher[name],
                entry.teacher_embeddings[name],
                negatives=hard,
                negative_admission=hard_admission,
                tau=self.config.tau,
                tau_min=self.config.tau_min,
                tau_max=self.config.tau_max,
                symmetric=self.config.symmetric,
                detach_positive=self.config.detach_positive,
                admission=admit,
                eps=self.config.eps,
            )
            per_teacher[name] = term
            reports[name] = report
            self.last_reports[(path, name)] = report

        # A teacher that contributed nothing is left out of sigma rather than
        # entered at a zero loss: a zero loss in a softmax is the *best possible*
        # score and would hand a discredited teacher the largest weight. Two things
        # make a teacher contribute nothing and both are tested here, so the two key
        # sets `compound_loss` compares cannot diverge:
        #   * no admitted rows this step, or
        #   * trust at or below the sigma trust minimum.
        trust_min = self._sigma_trust_min()
        live = [
            name
            for name in prepared.active
            if reports[name].admitted_rows != 0
            and float(prepared.trust.get(name, 0.0)) > trust_min
        ]
        contributing = {name: float(reports[name].loss.detach().item()) for name in live}
        if contributing:
            sigma = sigma_weights(
                contributing,
                mode=self.config.sigma_mode,
                beta=self.config.beta,
                priors={name: self.teachers[name].spec.weight for name in contributing},
                quality={name: self.sigma.get(path, name) for name in contributing},
                trust=prepared.trust,
                trust_min=trust_min,
                floor=self.sigma_floor,
                cap=self.sigma_cap,
            )
        else:
            sigma = {}
        contrastive_terms = {name: per_teacher[name] for name in sigma}

        breakdown.extras.update(
            {f"trust.{name}": float(prepared.trust.get(name, 0.0)) for name in self.teacher_names}
        )
        breakdown.extras.update(
            {f"admission.masked.{name}": float(reports[name].masked_rows) for name in reports}
        )
        breakdown.extras.update(
            {
                f"admission.soft.{name}": float(
                    1.0 if reports[name].soft_admission is None else reports[name].soft_admission
                )
                for name in reports
            }
        )

        # -- cooperative and regularizer terms ------------------------------
        coop = cooperative_term(
            policy,
            advantage=advantage,
            centring=centring,
            coefficient=self.config.lambda_coop,
            actions=actions,
        )
        reg, reg_values = regularization_term(
            self.regularizers.get(path, []), peer.module, states=batch.states
        )
        breakdown.extras.update({f"reg.{key}": value for key, value in sorted(reg_values.items())})

        return compound_loss(
            peer=path,
            config=self.config,
            individual=individual,
            cooperative=coop,
            regularization=reg,
            contrastive=contrastive_terms,
            sigma=sigma,
            margin=margin,
            beta=prepared.ramp,
            step=prepared.step,
            breakdown=breakdown,
            reports=reports,
        )

    def _hard_negatives(
        self, *, anchor_path: str, teacher: str, prepared: PreparedStep
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """The cross-teacher hard negatives for one ``(peer, teacher)`` pair.

        The same batch through every *other* active teacher: a peer cannot satisfy one
        teacher by collapsing onto it. Entry ``b`` of a hard negative is about sample
        ``b``, the anchor's own sample, so these are hard negatives and not in-batch
        ones. Each carries its own teacher's admission, so a row that teacher denies
        is not an anti-target either (invariant I8).

        Cross-*peer* separation is deliberately not a denominator entry. A
        denominator entry is satisfiable by collapsing every peer onto the teacher,
        and the sibling hinge (:func:`peer_margin`) is the term that keeps the shared
        space wide.

        A teacher whose pair width differs (a label-aligned pair) cannot be a hard
        negative for this anchor and is skipped.
        """
        entry = prepared.peers[anchor_path]
        width = int(entry.teacher_embeddings[teacher].shape[-1])
        hard: list[torch.Tensor] = []
        admissions: list[torch.Tensor | None] = []
        for name in prepared.active:
            if name == teacher:
                continue
            embedding = entry.teacher_embeddings[name]
            if int(embedding.shape[-1]) != width:
                continue
            hard.append(embedding)
            admissions.append(entry.teacher_admission.get(name))
        return hard, admissions

    # -- advantage, void, floor -----------------------------------------------
    def _advantage(
        self,
        *,
        path: str,
        signal: float,
        card: Any,
        batch: DistillBatch,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Per-sample advantage, built exactly as ``local_update`` builds it."""
        signal_tensor = torch.full((batch.batch,), float(signal), device=device, dtype=dtype)
        void = self._void_mask(path, batch, card, device)
        return penalise_void_advantage(signal_tensor, void, floor=-1.0)

    def _void_mask(
        self, path: str, batch: DistillBatch, card: Any, device: torch.device
    ) -> torch.Tensor:
        """Per-sample void flag: every sample inherits the peer's epoch-level verdict."""
        void = bool(getattr(card, "is_void", False))
        return torch.full((batch.batch,), void, device=device, dtype=torch.bool)

    def _compliance_floor(self, path: str, scorecards: Mapping[str, Any] | None) -> float:
        """The peer's card floor, or ``0.0`` when it has no card."""
        card = (scorecards or {}).get(path)
        floor = getattr(card, "floor", None)
        return float(floor) if floor is not None else 0.0

    def _resolve_signal(self, signal: Mapping[str, float] | None) -> dict[str, float]:
        """One score per peer, defaulting to the last step's negative loss.

        The fallback is legitimate rather than a guess: the compound loss *is* the
        peer's training signal. It is only used when the caller supplied no scores,
        which is the hand-rolled-loop case.
        """
        if signal:
            resolved = {str(k): float(v) for k, v in signal.items()}
        else:
            resolved = {path: -float(loss.total.detach().item()) for path, loss in self.last.items()}
        missing = [path for path in sorted(self._peers) if path not in resolved]
        if missing:
            raise ContractError(
                "no signal for peer(s) " + ", ".join(missing) + "; the cohort term needs a "
                "score for every peer, including ones that did not step. Pass signal={...} "
                "to step() (the first step has no previous loss to fall back on)"
            )
        return resolved

    def _combined_breakdown(self) -> LossBreakdown:
        """One cohort breakdown: the mean of each named component across peers."""
        combined = LossBreakdown()
        if not self.last:
            return combined
        names = sorted(self.last)
        for field_name in ("ind", "coop", "reg", "cd", "margin"):
            combined.extras[field_name] = math.fsum(
                float(getattr(self.last[name], field_name).detach().item()) for name in names
            ) / len(names)
        for name in names:
            combined.extras[f"peer.{name}"] = float(self.last[name].total.detach().item())
        combined.total = math.fsum(
            float(self.last[name].total.detach().item()) for name in names
        ) / len(names)
        return combined

    # -- reference ---------------------------------------------------------------
    def _reference_for(self, path: str) -> Any:
        """The frozen reference for one peer path.

        Order: an explicit ``references`` mapping, the reference the trainer shares,
        the ``*__reference`` node in the tree, then ``build_reference`` of the peer
        itself. Built once per peer and cached: rebuilding per step would re-copy the
        module and silently move the KL baseline between steps.
        """
        if path in self.references:
            return self.references[path]
        if self._shared_reference is not None:
            return self._shared_reference
        candidate = f"{path}__reference"
        if candidate in self._peers:
            return self._peers[candidate]
        reference = build_reference(self._peers[path], runtime=self.runtime)
        self.references[path] = reference
        return reference

    # -- weights and blending ---------------------------------------------------
    def _weights_aggregator(self) -> Aggregator:
        """The aggregator that computes weights, built privately when none was given.

        A privately built one never blends: ``blend`` only writes parameters through
        an aggregator the caller configured.
        """
        if self.aggregator is not None:
            return self.aggregator
        if self._private_aggregator is None:
            coop = dict(self._settings.get("COOPETITION", {}) or {})
            built = Aggregator.from_settings(coop, num_peers=max(1, len(self._peers)))
            if self.kappa is not None:
                built.controller = self.kappa
            self._private_aggregator = built
        return self._private_aggregator

    def _strategy(self) -> Any:
        strategy = self.weight_update
        if isinstance(strategy, KappaSoftmaxAverage) and strategy.aggregator is None:
            return KappaSoftmaxAverage(self._weights_aggregator())
        return strategy

    def update_weights(
        self,
        *,
        scores: Mapping[str, float] | None = None,
        eligible: Mapping[str, bool] | None = None,
    ) -> AggregationReport:
        """Weights over the peer set. Teachers are structurally excluded.

        ``scores`` defaults to the per-peer signals the last step recorded. The
        fallback rule of document 08 applies: a strategy that raises
        ``ComplianceError`` or ``NotImplementedError``, or reports ``fallback``, is
        replaced by ``KappaSoftmaxAverage`` and the reason is appended to
        ``AggregationReport.adjustments``. Degradation is never silent.
        """
        if not self._peers:
            raise ContractError("update_weights called before setup(); there are no peers to weight")
        names = sorted(self._peers)
        resolved = dict(scores if scores is not None else (self._signal or {}))
        missing = [name for name in names if name not in resolved]
        if missing:
            raise ContractError(
                "no score for peer(s) " + ", ".join(missing) + "; every participating "
                "peer needs a score, because a peer with no score would silently receive "
                "weight zero and be dropped from the blend"
            )
        kappa = self._kappa_value()
        vectors = {name: self._peers[name].export_parameters() for name in names}
        floats = {name: float(resolved[name]) for name in names}
        flags = dict(eligible if eligible is not None else self._eligible) or None
        confidence = self._confidence()

        def run(chosen: Any) -> AggregationReport:
            return chosen.apply(
                vectors, floats, kappa=kappa, eligible=flags, confidence=confidence
            )

        strategy = self._strategy()
        is_default = isinstance(strategy, KappaSoftmaxAverage)
        reason: str | None = None
        report: AggregationReport | None = None
        try:
            report = run(strategy)
        except (ComplianceError, NotImplementedError) as error:
            if is_default:
                raise
            reason = str(error) or type(error).__name__
        if report is not None and report.fallback and not is_default:
            reason = "it reported fallback"
        if reason is not None:
            label = getattr(strategy, "name", type(strategy).__name__)
            report = run(KappaSoftmaxAverage(self._weights_aggregator()))
            report.adjustments.append(
                f"weight_update {label!r} failed: {reason}; fell back to kappa_softmax"
            )
            self._emit(signals.distillation_fallback, reason=reason, report=report)
        assert report is not None
        self._assert_peer_only(report)
        self.state.aggregations += 1
        return report

    def _assert_peer_only(self, report: AggregationReport) -> None:
        """The assertion this whole policy exists to make unreachable.

        A teacher reaching the weight vector would mean parameters of a frozen model
        of another architecture were averaged into the peers. ``require_same_layout``
        would catch most of those, but a same-width teacher would slip through.
        """
        weights = set(report.weights)
        peers = set(self._peers)
        leaked = weights & set(self.teachers)
        if leaked:
            raise ComplianceError(
                "aggregation weights name teacher(s) "
                + ", ".join(sorted(leaked))
                + "; teachers are output-only and must never be blended. This is a bug in "
                "the configured WeightUpdateStrategy or in the caller's score mapping."
            )
        if weights != peers:
            raise ComplianceError(
                f"aggregation weights cover {sorted(weights)} but the peer set is "
                f"{sorted(peers)}; the weight vector must be exactly the peer set"
            )
        total = math.fsum(report.weights.values())
        if report.weights and not report.fallback and abs(total - 1.0) > 1e-6 and total != 0.0:
            raise ComplianceError(
                f"aggregation weights sum to {total!r}, not 1; Aggregator.compute_weights "
                "guarantees a normalized vector, so this indicates a custom strategy"
            )

    def _kappa_value(self) -> float:
        """The temperature the report records: the controller's, never recomputed."""
        if self.kappa is not None:
            return float(self.kappa.kappa)
        return float(self._weights_aggregator().controller.kappa)

    def _confidence(self) -> dict[str, float] | None:
        """Per-peer confidence, or ``None`` when nothing distinguishes the peers.

        ``None`` (rather than a dict of ``1.0``) keeps ``Aggregator.compute_weights``
        on its no-gating path, which makes the weights identical to the plain run when
        nothing voided.
        """
        if not self._episode_counts:
            return None
        total = float(max(self._episode_counts.values()))
        if total <= 0.0:
            return None
        fractions = {
            name: float(self._episode_counts.get(name, 0)) / total for name in sorted(self._peers)
        }
        if all(value == 1.0 for value in fractions.values()):
            return None
        return fractions

    def blend(self, report: AggregationReport | None = None) -> AggregationReport:
        """Apply the weight report to every peer, falling back per the documented rule."""
        if report is None:
            report = self.update_weights()
        self._assert_peer_only(report)
        if self.aggregator is None:
            report.adjustments.append("no Aggregator configured; parameters were not blended")
            return report
        adapters = {name: self._peers[name].adapter for name in sorted(self._peers)}
        reference = self._reference_for(sorted(self._peers)[0])
        result = self.aggregator.aggregate(
            adapters, report, reference=reference.adapter, device=self.runtime.device
        )
        coop = dict(self._settings.get("COOPETITION", {}) or {})
        self.aggregator.apply(
            adapters,
            result,
            reference=reference.adapter,
            reference_ema=float(coop.get("REFERENCE_EMA", 0.05)),
            reset_optimizers=False,
            optimizers=None,
        )
        if bool(coop.get("RESET_OPTIMIZERS", True)):
            for name in sorted(self._peers):
                _clear_optimizer_state(self._optimizers.get(name))
        self.state.blends += 1
        return report

    # -- description, checkpoint, resume ----------------------------------------
    def card_metadata(self, path: str) -> dict[str, Any]:
        """The ``PeerScorecard.metadata["distillation"]`` record for one peer.

        The values are the **last step's**, not an epoch mean; the epoch's history
        is in the state's ``loss_history`` and the log records.
        """
        loss = self.last.get(path)
        record: dict[str, Any] = {
            "enabled": True,
            "step": int(self._step),
            "teachers": sorted(self.teacher_names),
        }
        if loss is None:
            return record
        record.update(
            {
                "sigma": {name: round(value, 6) for name, value in sorted(loss.sigma.items())},
                "per_teacher": {
                    name: round(value, 6) for name, value in sorted(loss.per_teacher.items())
                },
                "cd_weighted": round(
                    loss.lambda_cd_effective * float(loss.cd.detach().item()), 6
                ),
                "peer_margin": round(float(loss.margin.detach().item()), 6),
            }
        )
        return record

    def describe(self) -> dict[str, Any]:
        return {
            "type": "co_distill",
            "teachers": {name: self.teachers[name].describe() for name in self.teacher_names},
            "peers": sorted(self._peers),
            "config": asdict(self.config),
            "out_dim": self.out_dim,
            "views": {key: view.describe() for key, view in sorted(self._views.items())},
            "conformance": {
                "enabled": self._conformance_enabled,
                "mode": str(self._conformance.get("MODE", "hybrid")),
                "trust": self._trust.state_dict(),
            },
            "managed": self._managed,
            "step": self._step,
        }

    def state_dict(self) -> dict[str, Any]:
        """Refresh and return the checkpoint section."""
        self.state.enabled = True
        self.state.teacher_names = self.teacher_names
        self.state.teacher_digests = {
            name: self.teachers[name].digest for name in self.teacher_names
        }
        self.state.step = int(self._step)
        self.state.sigma_state = self.sigma.state_dict()
        self.state.view_descriptions = {
            key: view.describe() for key, view in sorted(self._views.items())
        }
        self.state.trust_state = self._trust.state_dict()
        self.state.trust_profiles = {
            name: self._profiles[name].describe() for name in self.teacher_names
        }
        self.state.audit_digests = dict(self._audit_digests)
        self.state.config = asdict(self.config)
        self.state.episode_counts = dict(self._episode_counts)
        return self.state.state_dict()

    def load_state_dict(self, payload: Mapping[str, Any]) -> DistillResumeReport:
        """Restore the section and report every discrepancy.

        A digest mismatch does **not** raise. A run resumed against a re-exported
        teacher is a legitimate operation, and refusing it would force the user to
        delete their checkpoint to continue. What must not happen is the mismatch
        going unnoticed, so it is a reported note.
        """
        restored: list[str] = []
        skipped: list[str] = []
        notes: list[str] = []

        stored = dict(payload.get("teacher_digests") or {})
        for name in self.teacher_names:
            current = self.teachers[name].digest
            previous = stored.get(name)
            if previous is None:
                notes.append(
                    f"teacher {name!r} has no recorded digest; it is new since this checkpoint"
                )
                skipped.append(f"teacher_digest:{name}")
            elif previous != current:
                notes.append(
                    f"teacher {name!r} digest changed ({previous[:12]}... -> {current[:12]}...); "
                    "the contrastive objective is not comparable to the previous segment"
                )
            else:
                restored.append(f"teacher_digest:{name}")
        for name in sorted(set(stored) - set(self.teachers)):
            notes.append(
                f"checkpoint recorded teacher {name!r}, which this run does not configure; "
                "its sigma history is retained but unused"
            )

        if "sigma_state" in payload:
            self.sigma.load_state_dict(dict(payload["sigma_state"]))
            restored.append("sigma_state")
        else:
            skipped.append("sigma_state")
            notes.append("no sigma history; teacher weights restart from the prior")

        self._restore_trust(payload, restored=restored, skipped=skipped, notes=notes)

        # ``step`` is restored so the ramp resumes instead of restarting; a run
        # restarted at step 0 with warmup would silently turn distillation off again.
        self._step = int(payload.get("step", 0))
        counts = dict(payload.get("episode_counts") or {})
        self._episode_counts = {str(name): int(value) for name, value in counts.items()}
        restored.append("episode_counts") if counts else skipped.append("episode_counts")

        self.state = DistillationState(
            enabled=True,
            teacher_names=self.teacher_names,
            teacher_digests={name: self.teachers[name].digest for name in self.teacher_names},
            step=self._step,
            bake=int(payload.get("bake", 0)),
            aggregations=int(payload.get("aggregations", 0)),
            blends=int(payload.get("blends", 0)),
            sigma_state=self.sigma.state_dict(),
            view_descriptions={key: view.describe() for key, view in sorted(self._views.items())},
            trust_state=self._trust.state_dict(),
            trust_profiles={name: self._profiles[name].describe() for name in self.teacher_names},
            audit_digests=dict(self._audit_digests),
            config=asdict(self.config),
            loss_history=[dict(row) for row in (payload.get("loss_history") or ())][-STATE_HISTORY:],
            episode_counts=dict(self._episode_counts),
        )
        return DistillResumeReport(
            section="distillation",
            restored=tuple(restored),
            skipped=tuple(skipped),
            notes=tuple(notes),
        )

    def _restore_trust(
        self,
        payload: Mapping[str, Any],
        *,
        restored: list[str],
        skipped: list[str],
        notes: list[str],
    ) -> None:
        """Restore trust with the one rule that matters: never raise it on resume.

        Trust is applied as ``min(stored, current)``, expressed as "seed from the
        checkpoint, then clamp at the current value" so no branch can raise it. A
        resumed run must not re-earn trust it lost, and a version-1 checkpoint with no
        trust state starts at ``TRUST_INIT`` with a note saying the history was absent
        rather than clean.
        """
        stored_state = dict(payload.get("trust_state") or {})
        if not stored_state:
            skipped.append("trust_state")
            notes.append(
                "no trust history in the checkpoint (format_version < 2); every teacher "
                f"starts at TRUST_INIT={self._trust.init:g}"
            )
        else:
            applier = TrustState(
                init=float(stored_state.get("init", self._trust.init)),
                floor=float(stored_state.get("floor", self._trust.floor)),
                decay=float(stored_state.get("decay", self._trust.decay)),
                restore=float(stored_state.get("restore", self._trust.restore)),
            )
            applier.load_state_dict(stored_state)
            lowered: list[str] = []
            for name in self.teacher_names:
                stored_value = applier.get(name)
                current = self._trust.get(name)
                self._trust.set(name, min(stored_value, current))
                if stored_value < current:
                    lowered.append(f"{name} ({stored_value:.3f} < {current:.3f})")
            if lowered:
                notes.append(
                    "restored trust is lower than the configured start for "
                    + ", ".join(lowered)
                    + "; trust is applied as min(stored, current) so a resumed run "
                    "cannot re-earn trust it lost"
                )
            restored.append("trust_state")

        stored_profiles = dict(payload.get("trust_profiles") or {})
        if not stored_profiles:
            skipped.append("trust_profiles")
        else:
            for name in self.teacher_names:
                previous = dict(stored_profiles.get(name) or {})
                current = self._profiles[name].describe()
                if not previous:
                    notes.append(f"teacher {name!r} has no recorded trust profile")
                elif previous != current:
                    notes.append(
                        f"teacher {name!r} trust profile changed since this checkpoint "
                        f"({previous} -> {current}); admission is not comparable to the "
                        "previous segment"
                    )
            restored.append("trust_profiles")

        stored_audit = dict(payload.get("audit_digests") or {})
        if not stored_audit:
            skipped.append("audit_digests")
        else:
            for name in self.teacher_names:
                previous_digest = stored_audit.get(name)
                current_digest = self._audit_digests.get(name)
                if previous_digest and current_digest and previous_digest != current_digest:
                    notes.append(
                        f"teacher {name!r} conformance artifact changed "
                        f"({previous_digest[:12]}... -> {current_digest[:12]}...)"
                    )
            restored.append("audit_digests")
