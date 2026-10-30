"""Every ``Serializer`` the API publishes (document 02 section 2).

Response serializers are projections of the library's own ``to_dict()`` /
``as_dict()``; request serializers are closed (``unknown_fields = "forbid"``) so a
typo in a submitted spec is an error and not a silently ignored key.
"""

from __future__ import annotations

import copy
from typing import Any

from ...conf import _deep_merge
from ...distill.config import (
    CONFORMANCE_MODES,
    DISTILLATION_DEFAULTS,
    SIGMA_MODE_NAMES,
    validate_block,
)
from ...exceptions import ImproperlyConfigured, ValidationError
from ...http.serializers import (
    CharField,
    Field,
    FieldError,
    ListField,
    ModelSerializer,
    RawField,
    Serializer,
)
from ...scoring.scorecard import EpisodeScorecard, PeerScorecard

_SEVERITIES = ["S0", "S1", "S2", "S3", "S4"]


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
class RuleSerializer(Serializer):
    id: str = Field()
    name: str = Field()
    kind: str = Field()
    tier: int = Field()
    tier_name: str = Field()
    default_severity: str = Field()
    bucket: str | None = Field(allow_null=True, default=None)
    description: str = Field(default="")
    warning_eligible: bool = Field(default=False)
    non_offsettable: bool = Field(default=False)
    omega: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)
    is_reward: bool = Field(default=False)
    intent: str = Field(default="")


class TierSerializer(Serializer):
    tier: int = Field()
    name: str = Field()


class SeveritySerializer(Serializer):
    label: str = Field()
    value: int = Field()
    weight: float = Field()


class CatalogValidationSerializer(Serializer):
    ok: bool = Field()
    problems: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Points and normalization
# ---------------------------------------------------------------------------
class RulePointsSerializer(Serializer):
    unknown_fields = "allow"
    rule_id: str = Field()
    kind: str = Field(choices=["reward", "penalty", "warning"])
    tier: str = Field()
    weight: float = Field()
    severity: str = Field()
    fraction: float = Field()
    occurrences: int = Field(default=1, min_value=1)
    earned: bool = Field(default=True)
    amount: float = Field()
    reason: str | None = Field(allow_null=True, default=None)
    missing: list[str] = Field(default_factory=list)


class DenialSerializer(Serializer):
    unknown_fields = "allow"
    rule_id: str = Field()
    reason: str = Field()
    missing: list[str] = Field(default_factory=list)


class PointsSerializer(Serializer):
    """``PointsBreakdown.as_dict()``. ``warning_notional`` is never subtracted (W1)."""

    unknown_fields = "allow"
    reward: float = Field()
    penalty: float = Field()
    warning_notional: float = Field()
    gross: float = Field()
    final_points: float = Field()
    void: bool = Field()
    saturated: bool = Field(default=False)
    rewards: list[RulePointsSerializer] = Field(default_factory=list)
    penalties: list[RulePointsSerializer] = Field(default_factory=list)
    denials: list[DenialSerializer] = Field(default_factory=list)
    table_version: str = Field()


class NormalizedScoreSerializer(Serializer):
    unknown_fields = "allow"
    peer: str = Field()
    points: float = Field()
    calibrated: float = Field()
    percentile: float = Field()
    rank: int = Field()
    signal: float = Field()
    void: bool = Field(default=False)


class CohortStatsSerializer(Serializer):
    unknown_fields = "allow"
    n: int = Field()
    mean: float = Field()
    std: float = Field()
    minimum: float = Field()
    maximum: float = Field()
    p25: float = Field()
    p50: float = Field()
    p75: float = Field()
    iqr: float = Field()
    spread: float = Field()
    excluded_void: list[str] = Field(default_factory=list)


class NormalizationPolicySerializer(Serializer):
    """``dataclasses.asdict`` over the flat frozen ``NormalizationPolicy``."""

    unknown_fields = "allow"
    enabled: bool = Field()
    target_mean: float = Field()
    shrink: float = Field()
    exclude_void: bool = Field()
    min_cohort: int = Field()
    signal_scale: float = Field()
    clip: tuple[float, float] = Field()


class NormalizationSerializer(Serializer):
    unknown_fields = "allow"
    episode: int = Field()
    policy: NormalizationPolicySerializer = Field()
    stats: CohortStatsSerializer = Field()
    scores: list[NormalizedScoreSerializer] = Field()
    preserves_order: bool = Field()
    before_spread: float = Field()
    after_spread: float = Field()
    mean_shift: float = Field()
    adjustments: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scorecards
# ---------------------------------------------------------------------------
class ContinuousSerializer(Serializer):
    unknown_fields = "allow"
    reward: dict = Field()
    penalty: dict = Field()
    warning: float = Field()
    net: float = Field()
    suppressed_rewards: list[str] = Field(default_factory=list)
    precedence: str | None = Field(allow_null=True, default=None)


class ScorecardSerializer(ModelSerializer):
    model = PeerScorecard
    unknown_fields = "allow"

    peer: str = Field()
    episode: int = Field()
    final_score: float = Field()
    severity: str = Field(choices=_SEVERITIES)
    void: bool = Field()
    continuous: dict = Field()
    floor: float = Field()
    floor_reason: str | None = Field(allow_null=True)
    suppressed: list[dict] = Field()
    non_offsettable: list[str] = Field()
    warnings_issued: list[str] = Field()
    warnings_escalated: list[str] = Field()
    metadata: dict = Field()
    points: PointsSerializer | None = Field(allow_null=True, default=None)


class ScorecardV2Serializer(ScorecardSerializer):
    points: PointsSerializer = Field()
    normalization: NormalizationSerializer | None = Field(allow_null=True, default=None)
    guards_denied: list[DenialSerializer] = Field(default_factory=list)


class EpisodeScorecardSerializer(ModelSerializer):
    model = EpisodeScorecard
    unknown_fields = "allow"
    episode: int = Field()
    peers: dict[str, ScorecardSerializer] = Field()
    metadata: dict = Field(default_factory=dict)


class EvaluationResponseSerializer(ScorecardSerializer):
    """``PeerScorecard.to_dict()`` plus ``explain``, computed on demand and never stored."""

    explain: str = Field(default="")


# ---------------------------------------------------------------------------
# Aggregation, compliance, sandbox, registry rows
# ---------------------------------------------------------------------------
class AggregationSerializer(Serializer):
    unknown_fields = "allow"
    weights: dict[str, float] = Field(default_factory=dict)
    raw_weights: dict[str, float] = Field(default_factory=dict)
    kappa: float | None = Field(allow_null=True, default=None)
    entropy: float | None = Field(allow_null=True, default=None)
    adjustments: list[str] = Field(default_factory=list)
    fallback: bool = Field(default=False)
    gated_peers: list[str] = Field(default_factory=list)
    trimmed_peers: list[str] = Field(default_factory=list)
    order: list[str] = Field(default_factory=list)
    signal_source: str = Field(default="final_score")
    normalization: NormalizationSerializer | None = Field(allow_null=True, default=None)


class ComplianceSerializer(Serializer):
    unknown_fields = "allow"
    value: float = Field()
    reward_component: float = Field()
    offsettable_penalty: float = Field()
    omega: float = Field()
    non_offsettable_penalty: float = Field()
    warning: float = Field()
    floor: float = Field()
    net_before_constraints: float = Field()
    violations: list[str] = Field(default_factory=list)
    reason: str | None = Field(allow_null=True, default=None)
    precedence: str | None = Field(allow_null=True, default=None)


class CapabilitySerializer(Serializer):
    unknown_fields = "allow"
    provider: str = Field()
    isolation: str = Field()
    usable: bool = Field()
    capabilities: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ExecSerializer(Serializer):
    unknown_fields = "allow"
    command: Any = Field()
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    exit_code: int = Field()
    duration: float = Field(default=0.0)
    timed_out: bool = Field(default=False)


class RunSerializer(Serializer):
    unknown_fields = "allow"
    id: str = Field()
    started_at: float | None = Field(allow_null=True, default=None)
    finished_at: float | None = Field(allow_null=True, default=None)
    status: str = Field()
    settings: dict | None = Field(allow_null=True, default=None)
    runtime: dict | None = Field(allow_null=True, default=None)
    sandbox: dict | None = Field(allow_null=True, default=None)
    kappa: float | None = Field(allow_null=True, default=None)
    entropy: float | None = Field(allow_null=True, default=None)
    notes: str | None = Field(allow_null=True, default=None)


class EpisodeSerializer(Serializer):
    unknown_fields = "allow"
    run_id: str = Field()
    episode: int = Field()
    started_at: float | None = Field(allow_null=True, default=None)
    finished_at: float | None = Field(allow_null=True, default=None)
    scorecard: EpisodeScorecardSerializer = Field()


class RuleEventSerializer(Serializer):
    unknown_fields = "allow"
    rule_id: str = Field()
    magnitude: float = Field()
    severity: str | None = Field(allow_null=True, default=None)
    evidence: str = Field(default="")
    source: str = Field(default="")
    created_at: float | None = Field(allow_null=True, default=None)


class RunAcceptedSerializer(Serializer):
    run_id: str = Field()
    status: str = Field()
    created_at: float = Field(default=0.0)
    links: dict[str, str] = Field()


# ---------------------------------------------------------------------------
# Wrapper-layer specs (documents 07-10): closed, round-trip to from_spec/to_spec
# ---------------------------------------------------------------------------
class ScorerSpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(choices=["reward", "penalty", "warning", "hierarchy"])
    order: int | None = Field(allow_null=True, default=None)
    config: dict = Field(default_factory=dict)


class ScorerChainSpecSerializer(Serializer):
    unknown_fields = "forbid"
    scorers: list[ScorerSpecSerializer] = Field(default_factory=list)
    severity_weights: dict[str, float] | None = Field(allow_null=True, default=None)


class LossSpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(
        choices=[
            "policy_gradient", "value_regression", "kl", "entropy", "compliance_floor",
            "void_mask", "composite", "causal_lm", "callable", "chained",
        ]
    )
    policy_coef: float | None = Field(allow_null=True, default=None)
    value_coef: float | None = Field(allow_null=True, default=None)
    entropy_coef: float | None = Field(allow_null=True, default=None)
    kl_coef: float | None = Field(allow_null=True, default=None)
    compliance_coef: float | None = Field(allow_null=True, default=None)
    terms: list[dict] | None = Field(allow_null=True, default=None)
    config: dict = Field(default_factory=dict)


class OptimizerSpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(choices=["torch", "scipy", "callable", "router"])
    class_name: str | None = Field(allow_null=True, default=None)
    lr: float | None = Field(allow_null=True, default=None, min_value=0.0)
    weight_decay: float = Field(default=0.0, min_value=0.0)
    momentum: float | None = Field(allow_null=True, default=None)
    betas: tuple[float, float] | None = Field(allow_null=True, default=None)
    method: str | None = Field(
        allow_null=True,
        default=None,
        choices=["L-BFGS-B", "BFGS", "CG", "Newton-CG", "Nelder-Mead", "Powell", "SLSQP"],
    )
    max_iter: int | None = Field(allow_null=True, default=None)
    tol: float | None = Field(allow_null=True, default=None)
    grad_accumulation: int = Field(default=1, min_value=1)
    max_grad_norm: float | None = Field(allow_null=True, default=None, min_value=0.0)
    scheduler: dict | None = Field(allow_null=True, default=None)
    groups: list[dict] | None = Field(allow_null=True, default=None)
    default: dict | None = Field(allow_null=True, default=None)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if attrs.get("lr") is not None and attrs["lr"] <= 0:
            raise FieldError({"lr": ["must be greater than 0."]})
        if attrs.get("max_grad_norm") is not None and attrs["max_grad_norm"] <= 0:
            raise FieldError({"max_grad_norm": ["must be greater than 0."]})
        return attrs


class OptimizerGroupSerializer(Serializer):
    unknown_fields = "forbid"
    pattern: str = Field()
    optimizer: OptimizerSpecSerializer = Field()


class TargetModulesField(ListField):
    """``"auto"`` collapses to ``[]`` at the boundary (document 02 section 2.7)."""

    def to_internal(self, value: Any) -> list[Any]:
        if isinstance(value, str):
            if value.strip().lower() == "auto":
                return []
            raise FieldError("expected a list of module names or the string 'auto'.")
        return super().to_internal(value)


class FinetuneSpecSerializer(Serializer):
    unknown_fields = "forbid"
    peft_type: str | None = Field(allow_null=True, default=None, choices=["lora", "adalora", "prefix", "qlora"])
    r: int = Field(default=8, min_value=1)
    alpha: int = Field(default=16, min_value=1)
    dropout: float = Field(default=0.05, min_value=0.0, max_value=1.0)
    target_modules = TargetModulesField(child=CharField(), default_factory=list)
    quantization: str | None = Field(allow_null=True, default=None, choices=["4bit", "8bit"])
    quant_type: str = Field(default="nf4", choices=["nf4", "fp4"])
    double_quant: bool = Field(default=True)
    compute_dtype: str = Field(default="bfloat16", choices=["bfloat16", "float16", "float32"])
    freeze_base: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=False)
    modules_to_save: list[str] = Field(default_factory=list)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if float(attrs.get("dropout", 0.0)) >= 1.0:
            raise FieldError({"dropout": ["must be less than 1."]})
        return attrs


class PeerStrategySpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(choices=["independent", "cooperative", "kappa_softmax", "frozen_peer", "hierarchical"])
    frozen: list[str] | None = Field(allow_null=True, default=None)


class WeightUpdateSpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(choices=["kappa_softmax", "uniform", "score_weighted", "reference_only", "none", "explicit"])
    weights: dict[str, float] | None = Field(allow_null=True, default=None)


class LevelSpecSerializer(Serializer):
    unknown_fields = "forbid"
    label: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    count: int | None = Field(allow_null=True, default=None, min_value=1, max_value=16)
    n_count: Any = Field(allow_null=True, default=None)
    model: str | None = Field(allow_null=True, default=None)
    strategy: PeerStrategySpecSerializer | None = Field(allow_null=True, default=None)
    roles: list[str] = Field(default_factory=list)


class HierarchySpecSerializer(Serializer):
    unknown_fields = "forbid"
    levels: list[LevelSpecSerializer] = Field(default_factory=list)
    roots: int = Field(default=1, min_value=1, max_value=256)
    max_depth: int = Field(default=8, min_value=1, max_value=32)
    max_nodes: int = Field(default=256, min_value=1)
    max_children_per_node: int = Field(default=16, min_value=1)
    fanout_policy: str = Field(default="uniform", choices=["uniform", "random", "explicit"])
    sharing: str = Field(default="independent", choices=["independent", "shared_trunk", "tied_weights"])
    attribution: str = Field(default="rollup", choices=["rollup", "own"])
    cross_level_grad: bool = Field(default=False)
    delegation: str = Field(
        default="broadcast", choices=["broadcast", "select_k", "route_by_role", "confidence", "explicit"]
    )
    synthesis: str = Field(
        default="consensus",
        choices=["consensus", "weighted_consensus", "best_by_score", "learned", "concat", "explicit"],
    )
    rollup: str = Field(
        default="kappa_weighted",
        choices=["kappa_weighted", "mean", "minimum", "worst_case", "sum", "explicit"],
    )
    seed: int = Field(default=0)


def _strip_none(value: Any) -> Any:
    """Drop ``None`` entries recursively.

    The serializer fills every unset field with ``None``. For a distillation block an
    unset key must mean "defer to the settings block", so ``None`` cannot be allowed to
    reach the merge as if the caller had written it.
    """
    if isinstance(value, dict):
        return {key: _strip_none(item) for key, item in value.items() if item is not None}
    return value


class DistillationConformanceSerializer(Serializer):
    """``DISTILLATION.CONFORMANCE`` (document 14 section 10.4). Unknown keys are refused."""

    unknown_fields = "forbid"
    ENABLED: bool | None = Field(allow_null=True, default=None)
    MODE: str | None = Field(allow_null=True, default=None, choices=sorted(CONFORMANCE_MODES))
    TRUST_INIT: float | None = Field(allow_null=True, default=None, min_value=0.0, max_value=1.0)
    TRUST_FLOOR: float | None = Field(allow_null=True, default=None, min_value=0.0, max_value=1.0)
    TRUST_DECAY: float | None = Field(allow_null=True, default=None, min_value=0.0, max_value=1.0)
    TRUST_RESTORE: float | None = Field(allow_null=True, default=None, min_value=0.0, max_value=1.0)
    TRUST_MIN: float | None = Field(allow_null=True, default=None, min_value=0.0, max_value=1.0)
    STRICT_UNDECLARED: bool | None = Field(allow_null=True, default=None)
    AUDIT_ARTIFACTS: dict | None = Field(allow_null=True, default=None)
    SIGMA_TRUST_MIN: bool | None = Field(allow_null=True, default=None)


class DistillationSpecSerializer(Serializer):
    """The ``distillation`` block of a trainer spec (document 14 section 19.3).

    ``unknown_fields = "forbid"`` is repeated here on purpose: a nested serializer's
    field policy is its own, and without it ``{"TAUU": 0.05}`` would reach the trainer
    and fail as a server error instead of a ``422`` at the edge. Every field defaults to
    ``None`` so an unset key defers to the settings block rather than overriding it.
    """

    unknown_fields = "forbid"
    ENABLED: bool | None = Field(allow_null=True, default=None)
    TEACHERS: list[dict] | None = Field(allow_null=True, default=None)
    OUT_DIM: int | None = Field(allow_null=True, default=None, min_value=1)
    TAU: float | None = Field(allow_null=True, default=None)
    TAU_MIN: float | None = Field(allow_null=True, default=None)
    TAU_MAX: float | None = Field(allow_null=True, default=None)
    SYMMETRIC: bool | None = Field(allow_null=True, default=None)
    BETA: float | None = Field(allow_null=True, default=None, min_value=0.0)
    SIGMA_MODE: str | None = Field(
        allow_null=True, default=None, choices=sorted(SIGMA_MODE_NAMES)
    )
    SIGMA_FLOOR: float | None = Field(allow_null=True, default=None, min_value=0.0)
    SIGMA_CAP: float | None = Field(allow_null=True, default=None, min_value=0.0, max_value=1.0)
    SIGMA_DECAY: float | None = Field(allow_null=True, default=None, min_value=0.0, max_value=1.0)
    LAMBDA_COOP: float | None = Field(allow_null=True, default=None, min_value=0.0)
    LAMBDA_IND: float | None = Field(allow_null=True, default=None, min_value=0.0)
    LAMBDA_REG: float | None = Field(allow_null=True, default=None, min_value=0.0)
    LAMBDA_CD: float | None = Field(allow_null=True, default=None, min_value=0.0)
    PEER_MARGIN: float | None = Field(allow_null=True, default=None, min_value=0.0, max_value=1.0)
    WARMUP_STEPS: int | None = Field(allow_null=True, default=None, min_value=0)
    RAMP_STEPS: int | None = Field(allow_null=True, default=None, min_value=0)
    REGULARIZERS: list[dict] | None = Field(allow_null=True, default=None)
    VIEWS: dict | None = Field(allow_null=True, default=None)
    STRICT: bool | None = Field(allow_null=True, default=None)
    CONFORMANCE: DistillationConformanceSerializer | None = Field(allow_null=True, default=None)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        # The cross-field rules (sigma floor against teacher count, trust bounds, view
        # kinds) live in one validator shared with ``validate_settings`` and the
        # ``distillation=`` argument. Teacher presence is not checked here: the settings
        # block may supply the teachers, and the trainer constructor decides.
        block = _deep_merge(copy.deepcopy(DISTILLATION_DEFAULTS), _strip_none(attrs))
        try:
            validate_block(block, require_teachers=False)
        except (ImproperlyConfigured, ValidationError) as error:
            raise FieldError(str(error)) from None
        return attrs


class TrainerSpecSerializer(Serializer):
    unknown_fields = "forbid"
    num_peers: int | None = Field(allow_null=True, default=None, min_value=1, max_value=256)
    peer_names: list[str] | None = Field(allow_null=True, default=None)
    hierarchy: HierarchySpecSerializer | None = Field(allow_null=True, default=None)
    optimizer: OptimizerSpecSerializer | None = Field(allow_null=True, default=None)
    loss: LossSpecSerializer | None = Field(allow_null=True, default=None)
    finetuning: FinetuneSpecSerializer | None = Field(allow_null=True, default=None)
    scorers: ScorerChainSpecSerializer | None = Field(allow_null=True, default=None)
    peer_strategy: PeerStrategySpecSerializer | None = Field(allow_null=True, default=None)
    weight_update: WeightUpdateSpecSerializer | None = Field(allow_null=True, default=None)
    grad_accumulation: int = Field(default=1, min_value=1)
    max_grad_norm: float | None = Field(allow_null=True, default=None, min_value=0.0)
    steps_per_epoch: int | None = Field(allow_null=True, default=None, min_value=1)
    distillation: DistillationSpecSerializer | None = Field(allow_null=True, default=None)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if attrs.get("max_grad_norm") is not None and attrs["max_grad_norm"] <= 0:
            raise FieldError({"max_grad_norm": ["must be greater than 0."]})
        if attrs.get("hierarchy") is not None and attrs.get("num_peers") is not None:
            raise FieldError("hierarchy and num_peers are mutually exclusive")
        return attrs


class PeerNodeSerializer(Serializer):
    unknown_fields = "allow"
    path: str = Field()
    name: str = Field()
    kind: str = Field(choices=["peer", "subagent", "miniagent", "microagent", "nanoagent", "generic"])
    depth: int = Field()
    parent: str | None = Field(allow_null=True)
    children: list[str] = Field()
    role: str = Field(default="")
    status: str = Field(choices=["active", "failed", "frozen", "pruned"])
    metadata: dict = Field(default_factory=dict)


class PeerTreeSerializer(Serializer):
    unknown_fields = "allow"
    nodes: list[PeerNodeSerializer] = Field()
    root_paths: list[str] = Field()
    depth: int = Field()
    node_count: int = Field()
    source: str = Field(choices=["live", "checkpoint"])
    max_depth: int = Field()
    max_nodes: int = Field()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class InferenceRequestSerializer(Serializer):
    unknown_fields = "forbid"
    inputs: dict = Field()
    node_path: str | None = Field(allow_null=True, default=None)
    max_depth: int | None = Field(allow_null=True, default=None, min_value=1)
    budget: int | None = Field(allow_null=True, default=None, min_value=1)
    parallel: bool = Field(default=False)
    max_workers: int = Field(default=4, min_value=1, max_value=32)
    seed: int | None = Field(allow_null=True, default=None)
    score: bool = Field(default=False)
    persist_warnings: bool = Field(default=False)


class InferenceNodeResultSerializer(Serializer):
    unknown_fields = "allow"
    path: str = Field()
    depth: int = Field()
    output_shape: list[int] | None = Field(allow_null=True, default=None)
    duration: float = Field()
    selected: list[str] = Field(default_factory=list)
    starved: list[str] = Field(default_factory=list)


class InferenceResultSerializer(Serializer):
    unknown_fields = "allow"
    outputs: dict = Field()
    root_outputs: dict = Field()
    per_node: list[InferenceNodeResultSerializer] = Field()
    batching: list[dict] = Field(default_factory=list)
    cost: dict = Field(default_factory=dict)
    scorecard: EpisodeScorecardSerializer | None = Field(allow_null=True, default=None)
    mode: dict[str, bool] = Field(default_factory=dict)
    duration: float = Field()
    seed: int | None = Field(allow_null=True, default=None)


# ---------------------------------------------------------------------------
# Observability and recovery
# ---------------------------------------------------------------------------
class LogRecordSerializer(Serializer):
    unknown_fields = "allow"
    kind: str = Field(
        choices=["epoch", "step", "loss", "score", "aggregation", "warning", "checkpoint", "health", "resume", "log"]
    )
    run_id: str = Field()
    node_path: str = Field(default="")
    depth: int = Field(default=0)
    epoch: int = Field(default=0)
    step: int = Field(default=0)
    wall_time: float = Field()
    monotonic: float = Field(default=0.0)
    duration: float = Field(default=0.0)
    level: str = Field(default="info", choices=["debug", "info", "warning", "error"])
    message: str = Field(default="")
    schema_version: int = Field(default=1)


class ProbeResultSerializer(Serializer):
    unknown_fields = "allow"
    name: str = Field()
    status: str = Field(choices=["healthy", "degraded", "critical"])
    value: float | None = Field(allow_null=True, default=None)
    threshold: float | None = Field(allow_null=True, default=None)
    detail: str = Field(default="")
    node_path: str = Field(default="")
    metadata: dict = Field(default_factory=dict)


class CheckpointSerializer(Serializer):
    unknown_fields = "allow"
    run_id: str = Field(default="")
    episode: int = Field()
    global_step: int | None = Field(allow_null=True, default=None)
    path: str = Field()
    created_at: float | None = Field(allow_null=True, default=None)
    reason: str = Field(default="")
    size_bytes: int | None = Field(allow_null=True, default=None)
    schema_version: int | None = Field(allow_null=True, default=None)
    verified: bool | None = Field(allow_null=True, default=None)
    best: bool = Field(default=False)
    sections: list[str] = Field(default_factory=list)
    kappa: float | None = Field(allow_null=True, default=None)
    entropy: float | None = Field(allow_null=True, default=None)
    score: float | None = Field(allow_null=True, default=None)


class HealthReportSerializer(Serializer):
    unknown_fields = "allow"
    status: str = Field(choices=["healthy", "degraded", "critical"])
    action: str = Field(choices=["none", "log", "snapshot", "rollback", "pause", "prune_branch", "abort"])
    epoch: int = Field()
    results: list[ProbeResultSerializer] = Field(default_factory=list)
    per_node: dict[str, str] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    snapshot: CheckpointSerializer | None = Field(allow_null=True, default=None)
    duration: float = Field(default=0.0)
    monotonic: float = Field(default=0.0)


class ResumeReportSerializer(Serializer):
    unknown_fields = "allow"
    run_id: str = Field()
    checkpoint: str = Field(default="")
    episode: int = Field(default=0)
    global_step: int = Field(default=0)
    schema_version: int = Field(default=0)
    restored_sections: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    skipped_checkpoints: list[tuple[str, str]] = Field(default_factory=list)
    warnings_restored: int = Field(default=0)
    optimizers_restored: int = Field(default=0)
    bit_identical_expected: bool = Field(default=False)
    duration: float = Field(default=0.0)
    notes: list[str] = Field(default_factory=list)


class ScorerCapabilitySerializer(Serializer):
    unknown_fields = "allow"
    type: str = Field()
    order: int = Field(default=0)
    required_context: list[str] = Field(default_factory=list)
    config_keys: list[str] = Field(default_factory=list)
    description: str = Field(default="")


class TrainerCapabilitiesSerializer(Serializer):
    unknown_fields = "allow"
    optimizer_backends: list[dict] = Field()
    optimizer_granularity: dict[str, str] = Field()
    loss_types: list[str] = Field()
    scorer_types: list[ScorerCapabilitySerializer] = Field()
    finetune_types: list[str] = Field()
    available_extras: dict[str, bool] = Field()
    hierarchy: dict = Field()
    limits: dict = Field()


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class RunCreateRequestSerializer(Serializer):
    unknown_fields = "forbid"
    epochs: int | None = Field(default=None, min_value=1)
    peers: int | None = Field(default=None, min_value=1)
    device: str | None = Field(allow_null=True, default=None)
    seed: int | None = Field(allow_null=True, default=None)
    dry_run: bool = Field(default=False)
    redteam: bool = Field(default=False)
    resume_from: str | None = Field(allow_null=True, default=None)
    run_id: str | None = Field(allow_null=True, default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    task_source: str | None = Field(allow_null=True, default=None)
    tags: dict[str, str] = Field(default_factory=dict)
    trainer: TrainerSpecSerializer | None = Field(allow_null=True, default=None)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        trainer = attrs.get("trainer") or {}
        if attrs.get("peers") is not None and trainer.get("num_peers") is not None:
            raise FieldError("peers and trainer.num_peers are mutually exclusive")
        resume = attrs.get("resume_from")
        if resume is not None and resume != "latest" and not str(resume).isdigit():
            raise FieldError({"resume_from": ['must be "latest" or an episode number.']})
        return attrs


class LedgerEventPayloadSerializer(Serializer):
    unknown_fields = "forbid"
    rule_id: str = Field()
    magnitude: float = Field(default=1.0, min_value=0.0)
    severity: str | None = Field(allow_null=True, default=None, choices=_SEVERITIES)
    evidence: str = Field(min_length=1)
    source: str = Field(default="api")
    tags: list[str] = Field(default_factory=list)


class LedgerPayloadSerializer(Serializer):
    unknown_fields = "forbid"
    peer: str = Field()
    episode: int = Field(default=0)
    events: list[LedgerEventPayloadSerializer] = Field()
    run_id: str | None = Field(allow_null=True, default=None)


class AdjudicationPayloadSerializer(Serializer):
    unknown_fields = "forbid"
    peer: str = Field()
    rule_id: str = Field()
    episode: int = Field()
    decision: str = Field(choices=["upheld", "reversed", "severity_changed"])
    severity: str | None = Field(allow_null=True, default=None, choices=_SEVERITIES)
    rationale: str = Field(min_length=1)
    adjudicator: str = Field(min_length=1)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if attrs["decision"] == "severity_changed" and not attrs.get("severity"):
            raise FieldError({"severity": ["required when decision is severity_changed."]})
        return attrs


class WeightPreviewRequestSerializer(Serializer):
    unknown_fields = "forbid"
    scores: dict[str, float] = Field()
    eligible: dict[str, bool] | None = Field(allow_null=True, default=None)
    confidence: dict[str, float] | None = Field(allow_null=True, default=None)
    advance: bool = Field(default=False)


class SandboxExecRequestSerializer(Serializer):
    unknown_fields = "forbid"
    peer: str = Field()
    command: Any = Field()
    timeout: float | None = Field(allow_null=True, default=None, min_value=0.0)

    def validate_command(self, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value and all(isinstance(part, str) for part in value):
            return value
        raise FieldError("command must be a non-empty string or list of strings.")


class RollbackRequestSerializer(Serializer):
    unknown_fields = "forbid"
    episode: int = Field(min_value=0)


class SnapshotRequestSerializer(Serializer):
    unknown_fields = "forbid"
    reason: str = Field(default="operator")


class ResumeRequestSerializer(Serializer):
    unknown_fields = "forbid"
    epochs: int | None = Field(allow_null=True, default=None, min_value=1)


class PaginationSerializer(Serializer):
    """The query contract shared by every collection route."""

    unknown_fields = "allow"
    limit: int = Field(default=50, min_value=1, max_value=500)
    offset: int = Field(default=0, min_value=0)
