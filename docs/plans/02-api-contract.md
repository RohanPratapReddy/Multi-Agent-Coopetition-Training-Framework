---
title: API Contract
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0, API v1
---

# 02. API Contract

Base path: `/api/v1`. Content type: `application/json` unless stated. Errors:
`application/problem+json` (RFC 9457). All timestamps are Unix epoch seconds as
floats, matching the `runs.started_at` column in
[mactf/db.py](mactf/db.py).

## 1. Integration and entry points

The API is an app in the framework, not a package beside it. There is no `[api]`
extra, no `create_app`, and no third-party HTTP dependency. What exists:

- `mactf/http/` (document 12) — the stdlib WSGI transport: `Request`, `Response`,
  `path`/`include`/`reverse`, `View`, middleware, `WSGIHandler`,
  `get_wsgi_application()`.
- `mactf/contrib/api/` (document 13) — the app: `urlpatterns`, views,
  serializers, auth, checks, and the `serve` command.
- `mactf/client/` (document 03) — the client, returning the framework's own
  dataclasses.

Nothing is installed beyond the framework's existing dependencies, so
`pip install mactf` yields a working server and client exactly as
`pip install django` yields `runserver`.

### 1.1 A project

A project that serves the API sets three settings:

```python
# project/settings.py
INSTALLED_APPS = [
    "mactf.contrib.scoring",
    "mactf.contrib.coopetition",
    "mactf.contrib.sandbox",
    "mactf.contrib.telemetry",
    "mactf.contrib.api",          # <- the service layer
]

ROOT_URLCONF = "project.urls"

MIDDLEWARE = [
    "mactf.http.middleware.RequestIdMiddleware",
    "mactf.http.middleware.ExceptionMiddleware",
    "mactf.contrib.api.auth.AuthMiddleware",
]

API = {"ALLOW_ANONYMOUS": False, "TOKENS": {"...": ["read", "run:write"]}}
```

```python
# project/urls.py
from mactf.http import include, path

urlpatterns = [
    path("api/v1/", include("mactf.contrib.api.urls")),
]
```

The module declares `app_name = "api"`, so `include()` derives the instance
namespace from it and `reverse("api:rule-list")` resolves. Passing
`namespace="v1"` explicitly is supported but renames the prefix, so the client —
which reverses against `api:` — would not find it. Do not pass it.

Removing the `INSTALLED_APPS` entry and the `include()` removes the API. There is
no import error and no dangling command: `mactf run`, `mactf check`, and
`mactf migrate` are untouched (document 13 criterion 47).

`ROOT_URLCONF = None` is legal and means "the framework's default URL conf",
which mounts `mactf.contrib.api.urls` at `API.BASE_PATH`. That is what makes a
bare install serveable without a project, and it is why `mactf serve` works from
a checkout.

### 1.2 Starting a server

One way, with two spellings:

1. `mactf serve --host 127.0.0.1 --port 8000`. The command ships in
   `mactf/contrib/api/management/commands/serve.py` and is discovered through
   `command_packages()` (document 13 section 2.1), so a bare install without the
   app has no `serve`. It builds the application with
   `get_wsgi_application()`, runs `wsgiref.simple_server`, and refuses to start
   when `run_checks()` reports an error unless `--skip-checks` is passed.
2. Any WSGI server the operator prefers:
   `gunicorn project.wsgi:application` where `project/wsgi.py` is
   `application = get_wsgi_application()`.

`python -m mactf.api` and `uvicorn mactf.api.app:app --factory` do not exist and
are not planned. A `[asgi]` extra may later install `uvicorn` and expose
`mactf.http.asgi:asgi_application` (document 12 section 13), and that is the only
sanctioned non-WSGI path.

### 1.3 Application state

`get_wsgi_application()` returns a `WSGIHandler` built for the current settings
object. Its per-application state is a `dict` on the handler:

```python
handler.state = {
    "settings": Settings,
    "registry": Registry,              # from ensure_registry(settings)
    "scorer": ScoringPipeline,         # app-scoped, shared across requests
    "run_manager": RunManager,         # from mactf/orchestration/live.py
    "events": EventBus,
    "principal_resolver": Callable | None,
    "idempotency": _IdempotencyCache,  # LRU, 256 entries, 24h TTL
    "started_at": float,
}
```

Views reach it through a `Request` attribute set by an app middleware, or through
`mactf.contrib.api.deps`:

```python
def get_settings_dep(request: Request) -> Settings: ...
def get_registry_dep(request: Request) -> Registry: ...
def get_scorer_dep(request: Request) -> Scorer: ...
def get_run_manager_dep(request: Request) -> RunManager: ...
```

Resolution order is: `request.app_state` (set by the app's middleware for tests),
then the handler state, then `mactf.conf.get_settings()` /
`mactf.db.ensure_registry(settings)`. Because the handler is reachable through
`get_wsgi_application()` and is cached per settings object, a test that swaps
settings and calls `conf.configure(force=True)` gets a fresh application with a
fresh `RunManager`.

`get_scorer_dep` returns the app-scoped `ScoringPipeline` so warning registries
are shared across requests in one process. Two worker processes therefore hold
separate in-memory warning state, which is exactly why document 04.4 persists
warnings to sqlite; `mactf serve` runs one process and logs a warning when
`--workers` above 1 is requested.

`RunManager` and `EventBus` do not import `mactf.http`. They are L2
(mactf/orchestration/live.py) and are equally usable by the CLI, the client's
in-process transport, and the views.


## 2. Serializers

Every type below is a `Serializer` from `mactf/http/serializers.py` (document 12
section 9), declared in `mactf/contrib/api/serializers.py`. The third column names
the source object and the method that already produces the dict, so
implementation is a projection and nothing else. Every class sets
`unknown_fields = "forbid"` for request bodies and `"allow"` for response bodies,
so a future library field is never silently dropped by an older server, and a
typo in a submitted spec is never silently ignored.

The field names and JSON shapes are **unchanged** from the previous revision of
this document. Only the declaration mechanism changed: `XModel(BaseModel)` became
`XSerializer(Serializer)`, and where the wire shape is exactly a dataclass's
`to_dict()` the class derives from `ModelSerializer` and declares no fields of its
own. `ScorecardV2Model` is now `ScorecardV2Serializer`, `PeerTreeModel` is
`PeerTreeSerializer`, and so on, and document 03's client refers to the framework
objects these serializers project.

The annotated form is used for every declaration below (document 12 section 9.1):
an annotation gives the type, a `Field(...)` gives the constraints.

### 2.1 Catalog serializers

| Serializer | Fields | Source |
|---|---|---|
| `RuleSerializer` | `id, name, kind, tier, tier_name, default_severity, bucket, description, warning_eligible, non_offsettable, omega, tags` | `mactf/rules/catalog.py` `Rule` dataclass; `is_reward`/`intent` computed |
| `TierSerializer` | `tier: int, name: str` | `mactf/rules/catalog.py` `TIERS` |
| `SeveritySerializer` | `label: str, value: int, weight: float` | `mactf/rules/severity.py` `Severity`, `weight_for` |
| `CatalogValidationSerializer` | `ok: bool, problems: list[str]` | `mactf/rules/catalog.py` `validate_catalog()` |

### 2.2 Scorecard serializers

`ScorecardSerializer` derives from `ModelSerializer` with
`model = PeerScorecard` and adds no fields: its representation is
`PeerScorecard.to_dict()`
([mactf/scoring/scorecard.py:79](mactf/scoring/scorecard.py#L79)), which is the
contract. The declared field list exists so `openapi` and the round-trip test have
something to compare against, and it is the list below:

```python
class ScorecardSerializer(ModelSerializer):
    model = PeerScorecard
    unknown_fields = "allow"

    peer: str = Field()
    episode: int = Field()
    final_score: float = Field()
    severity: str = Field(choices=["S0", "S1", "S2", "S3", "S4"])
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
```

`ContinuousSerializer` mirrors `ContinuousScore.as_dict()`. Because
`continuous.py` is the part most likely to evolve, this serializer is declared
`unknown_fields = "allow"` with `RawField()` members for the three nested
breakdowns, and the typed fields are only the ones a client reads:

```python
class ContinuousSerializer(Serializer):
    unknown_fields = "allow"

    reward: dict = Field()        # r_task, r_epi, r_collab, r_align, total, rules, weights
    penalty: dict = Field()       # amounts, raw_total, omega, omega_rules, total, rules
    warning: float = Field()
    net: float = Field()
    suppressed_rewards: list[str] = Field(default_factory=list)
    precedence: str | None = Field(allow_null=True, default=None)
```

`RewardSerializer`, `PenaltySerializer`, and `SuppressedSerializer`
(`rule_id`, `reason`) are direct projections, each a plain `Serializer` with the
keys from the corresponding `as_dict()`. `EpisodeScorecardSerializer` derives from
`ModelSerializer(model=EpisodeScorecard)` with `episode`, `peers: dict[str,
ScorecardSerializer]`, `metadata`.

#### 2.2.1 Points serializers

The 0-100 rubric. Normative definitions are in
[11-points-and-normalization.md](11-points-and-normalization.md) sections 3-7;
this is the wire projection.

```python
class RulePointsSerializer(Serializer):
    unknown_fields = "forbid"
    rule_id: str = Field()
    kind: str = Field(choices=["reward", "penalty", "warning"])
    tier: str = Field()                     # T0 | T1 | T2 | T3 | T4
    weight: float = Field()                 # from the points table
    severity: str = Field()                 # S0..S4, as recorded
    fraction: float = Field()               # severity fraction applied
    occurrences: int = Field(default=1, min_value=1)
    earned: bool = Field(default=True)
    amount: float = Field()                 # points contributed, signed by kind
    reason: str | None = Field(allow_null=True, default=None)
    missing: list[str] = Field(default_factory=list)


class DenialSerializer(Serializer):
    unknown_fields = "forbid"
    rule_id: str = Field()
    reason: str = Field()
    missing: list[str] = Field(default_factory=list)


class PointsSerializer(Serializer):
    """Derives from PointsBreakdown.as_dict(); document 11 section 3."""
    unknown_fields = "allow"
    reward: float = Field()
    penalty: float = Field()
    warning_notional: float = Field()
    gross: float = Field()
    final_points: float = Field()           # 0.0 when void
    void: bool = Field()
    saturated: bool = Field(default=False)
    rewards: list[RulePointsSerializer] = Field(default_factory=list)
    penalties: list[RulePointsSerializer] = Field(default_factory=list)
    denials: list[DenialSerializer] = Field(default_factory=list)
    table_version: str = Field()
```

`warning_notional` is present and is never subtracted, which is the wire-level
expression of W1. A client that sums `reward - penalty + warning_notional` is
computing a number the framework does not define; the field name is chosen so
that mistake is visible in review.

```python
class NormalizedScoreSerializer(Serializer):
    unknown_fields = "forbid"
    peer: str = Field()
    points: float = Field()
    calibrated: float = Field()
    percentile: float = Field()
    rank: int = Field()
    signal: float = Field()
    void: bool = Field(default=False)


class CohortStatsSerializer(Serializer):
    unknown_fields = "forbid"
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


class NormalizationSerializer(Serializer):
    unknown_fields = "forbid"
    episode: int = Field()
    policy: NormalizationPolicySerializer = Field()
    stats: CohortStatsSerializer = Field()
    scores: list[NormalizedScoreSerializer] = Field()
    preserves_order: bool = Field()
    before_spread: float = Field()
    after_spread: float = Field()
    mean_shift: float = Field()
    adjustments: list[str] = Field(default_factory=list)


class NormalizationPolicySerializer(Serializer):
    """dataclasses.asdict over a flat frozen dataclass; no to_dict on the source."""
    unknown_fields = "forbid"
    enabled: bool = Field()
    target_mean: float = Field()
    shrink: float = Field()
    exclude_void: bool = Field()
    min_cohort: int = Field()
    signal_scale: float = Field()
    clip: tuple[float, float] = Field()
```

`ScorecardSerializer` carries one optional field, `points`, which is `None` for an
episode scored before this document. That is what keeps `points` out of the
response without requiring a new version for clients that do not ask for one.

`ScorecardV2Serializer` extends it:

```python
class ScorecardV2Serializer(ScorecardSerializer):
    points: PointsSerializer = Field()                    # non-optional in v2
    normalization: NormalizationSerializer | None = Field(allow_null=True, default=None)
    guards_denied: list[DenialSerializer] = Field(default_factory=list)
```

`guards_denied` is a flattened copy of `points.denials`, present so a client does
not have to reach two levels deep to render "why zero". Duplication is a
deliberate trade: the alternative is every client reimplementing the flattening.

### 2.3 Aggregation serializer

`AggregationSerializer` derives from `ModelSerializer(model=AggregationReport)`,
whose `as_dict()`
([mactf/coopetition/aggregation.py:54](mactf/coopetition/aggregation.py#L54))
supplies `weights`, `raw_weights`, `kappa`, `entropy`, `adjustments`, `fallback`,
`gated_peers`, `trimmed_peers`, `order`. It adds two fields:
`signal_source: str = Field(default="final_score")` and
`normalization: NormalizationSerializer | None = Field(default=None)`.

### 2.4 Compliance serializer

`ComplianceSerializer` derives from `ModelSerializer(model=ComplianceResult)`
([mactf/compliance.py:63](mactf/compliance.py#L63)): `value`,
`reward_component`, `offsettable_penalty`, `omega`, `non_offsettable_penalty`,
`warning`, `floor`, `net_before_constraints`, `violations`, `reason`,
`precedence`.

### 2.5 Sandbox serializers

`CapabilitySerializer` derives from `ModelSerializer(model=CapabilityReport)`
([mactf/sandbox/base.py:94](mactf/sandbox/base.py#L94)): `provider`,
`isolation`, `usable`, `capabilities: dict[str, bool]`, `reasons`, `metadata`.
`ExecSerializer` derives from `ModelSerializer(model=ExecResult)`
([mactf/sandbox/base.py:156](mactf/sandbox/base.py#L156)): `command`, `stdout`,
`stderr`, `exit_code`, `duration`, `timed_out`.

### 2.6 Registry serializers

`RunSerializer` from the `runs` table: `id`, `started_at`, `finished_at`, `status`,
`settings`, `runtime`, `sandbox`, `kappa`, `entropy`, `notes`.
`EpisodeSerializer`: `run_id`, `episode`, `started_at`, `finished_at`, `scorecard`
(nested `EpisodeScorecardSerializer`). Neither is a `ModelSerializer`, because a
table row is a dict rather than a dataclass with a dict protocol; they are plain
`Serializer` subclasses over the row mapping.

### 2.7 Wrapper-layer spec serializers

These carry the configuration of documents 07-10. Every one of them is a closed
serializer (`unknown_fields = "forbid"`) and every one round-trips to the
library's `from_spec` / `to_spec`. A field that exists here but not in the library
is a bug; the reverse is allowed only for library-internal state.

```python
class ScorerSpecSerializer(Serializer):
    """One scorer stage. ``type`` selects the registry entry in document 07 section 10."""
    unknown_fields = "forbid"
    type: str = Field(choices=["reward", "penalty", "warning", "hierarchy"])
    order: int | None = Field(allow_null=True, default=None)
    config: dict = Field(default_factory=dict)


class ScorerChainSpecSerializer(Serializer):
    unknown_fields = "forbid"
    scorers: list[ScorerSpecSerializer] = Field(default_factory=list)
    # Omitted -> the three defaults, which reproduce today's behaviour.
    severity_weights: dict[str, float] | None = Field(allow_null=True, default=None)


class LossSpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(choices=[
        "policy_gradient", "value_regression", "kl", "entropy",
        "compliance_floor", "void_mask", "composite",
        "causal_lm", "callable", "chained",
    ])
    policy_coef: float | None = Field(allow_null=True, default=None)
    value_coef: float | None = Field(allow_null=True, default=None)
    entropy_coef: float | None = Field(allow_null=True, default=None)
    kl_coef: float | None = Field(allow_null=True, default=None)
    compliance_coef: float | None = Field(allow_null=True, default=None)
    terms: list["LossSpecSerializer"] | None = Field(allow_null=True, default=None)
    config: dict = Field(default_factory=dict)


class OptimizerSpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(choices=["torch", "scipy", "callable", "router"])
    # torch
    class_name: str | None = Field(allow_null=True, default=None)   # "AdamW", "SGD", ...
    lr: float | None = Field(allow_null=True, default=None, min_value=0.0,
                             exclusive_min=True)
    weight_decay: float = Field(default=0.0, min_value=0.0)
    momentum: float | None = Field(allow_null=True, default=None)
    betas: tuple[float, float] | None = Field(allow_null=True, default=None)
    # scipy
    method: str | None = Field(allow_null=True, default=None, choices=[
        "L-BFGS-B", "BFGS", "CG", "Newton-CG",
        "Nelder-Mead", "Powell", "SLSQP",
    ])
    max_iter: int | None = Field(allow_null=True, default=None)
    tol: float | None = Field(allow_null=True, default=None)
    # shared
    grad_accumulation: int = Field(default=1, min_value=1)
    max_grad_norm: float | None = Field(allow_null=True, default=None, min_value=0.0,
                                        exclusive_min=True)
    scheduler: dict | None = Field(allow_null=True, default=None)
    groups: list["OptimizerGroupSerializer"] | None = Field(allow_null=True, default=None)
    default: "OptimizerSpecSerializer | None" = Field(allow_null=True, default=None)


class OptimizerGroupSerializer(Serializer):
    unknown_fields = "forbid"
    pattern: str = Field()                  # regex, fullmatch against the param name
    optimizer: OptimizerSpecSerializer = Field()


class FinetuneSpecSerializer(Serializer):
    unknown_fields = "forbid"
    peft_type: str | None = Field(allow_null=True, default=None, choices=[
        "lora", "adalora", "prefix", "qlora",
    ])
    r: int = Field(default=8, min_value=1)
    alpha: int = Field(default=16, min_value=1)
    dropout: float = Field(default=0.05, min_value=0.0, max_value=1.0, exclusive_max=True)
    target_modules: list[str] = Field(default_factory=lambda: [])  # "auto" collapses to []
    quantization: str | None = Field(allow_null=True, default=None, choices=["4bit", "8bit"])
    quant_type: str = Field(default="nf4", choices=["nf4", "fp4"])
    double_quant: bool = Field(default=True)
    compute_dtype: str = Field(default="bfloat16", choices=["bfloat16", "float16", "float32"])
    freeze_base: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=False)
    modules_to_save: list[str] = Field(default_factory=list)


class PeerStrategySpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(choices=[
        "independent", "cooperative", "kappa_softmax", "frozen_peer", "hierarchical",
    ])
    frozen: list[str] | None = Field(allow_null=True, default=None)   # node paths


class WeightUpdateSpecSerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field(choices=[
        "kappa_softmax", "uniform", "score_weighted",
        "reference_only", "none", "explicit",
    ])
    weights: dict[str, float] | None = Field(allow_null=True, default=None)


class LevelSpecSerializer(Serializer):
    unknown_fields = "forbid"
    label: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    count: int | None = Field(allow_null=True, default=None, min_value=1, max_value=16)
    n_count: int | list[int] | None = Field(allow_null=True, default=None)
    model: str | None = Field(allow_null=True, default=None)   # registered model key
    strategy: PeerStrategySpecSerializer | None = Field(allow_null=True, default=None)
    roles: list[str] = Field(default_factory=list)
```

`FinetuneSpecSerializer.target_modules` is `list[str]` with the sentinel `"auto"`
collapsed to `[]` at the boundary: a union of `list[str]` and a single string
cannot be expressed in the JSON schema without an `anyOf` that every client would
have to special-case, and `[]` already means "auto" in the library. The
normalizing function lives in `mactf/contrib/api/serializers.py` and is asserted
by a test that round-trips `{"target_modules": "auto"}` to `{"target_modules": []}`.

```python
class HierarchySpecSerializer(Serializer):
    unknown_fields = "forbid"
    levels: list[LevelSpecSerializer] = Field(default_factory=list)
    roots: int = Field(default=1, min_value=1, max_value=256)
    max_depth: int = Field(default=8, min_value=1, max_value=32)
    max_nodes: int = Field(default=256, min_value=1)
    max_children_per_node: int = Field(default=16, min_value=1)
    fanout_policy: str = Field(default="uniform",
                               choices=["uniform", "random", "explicit"])
    sharing: str = Field(default="independent",
                         choices=["independent", "shared_trunk", "tied_weights"])
    attribution: str = Field(default="rollup", choices=["rollup", "own"])
    cross_level_grad: bool = Field(default=False)
    delegation: str = Field(default="broadcast", choices=[
        "broadcast", "select_k", "route_by_role", "confidence", "explicit",
    ])
    synthesis: str = Field(default="consensus", choices=[
        "consensus", "weighted_consensus", "best_by_score",
        "learned", "concat", "explicit",
    ])
    rollup: str = Field(default="kappa_weighted", choices=[
        "kappa_weighted", "mean", "minimum", "worst_case", "sum", "explicit",
    ])
    seed: int = Field(default=0)


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
    max_grad_norm: float | None = Field(allow_null=True, default=None, min_value=0.0,
                                        exclusive_min=True)
    steps_per_epoch: int | None = Field(allow_null=True, default=None, min_value=1)


class PeerNodeSerializer(Serializer):
    """Derives from PeerNode.describe(); document 10 section 5."""
    unknown_fields = "allow"
    path: str = Field()
    name: str = Field()
    kind: str = Field(choices=[
        "peer", "subagent", "miniagent", "microagent", "nanoagent", "generic",
    ])
    depth: int = Field()
    parent: str | None = Field(allow_null=True)
    children: list[str] = Field()
    role: str = Field()
    status: str = Field(choices=["active", "failed", "frozen", "pruned"])
    metadata: dict = Field(default_factory=dict)


class PeerTreeSerializer(Serializer):
    """Derives from PeerTree.to_dict(); document 10 section 5."""
    unknown_fields = "allow"
    nodes: list[PeerNodeSerializer] = Field()
    root_paths: list[str] = Field()
    depth: int = Field()
    node_count: int = Field()
    source: str = Field(choices=["live", "checkpoint"])
    max_depth: int = Field()
    max_nodes: int = Field()
```

The `Serializer` equivalent of Pydantic's `Literal` is `Field(choices=[...])`,
which validates the same way and publishes an `enum` in the OpenAPI schema (which
`Literal` also did). The open `config` and `metadata` dicts stay `dict` with a
`RawField` value: they are genuinely open payloads, and claiming a narrower type
would be a lie the first time a library adds a key.

### 2.8 Inference serializers

```python
class InferenceRequestSerializer(Serializer):
    unknown_fields = "forbid"
    inputs: dict = Field()
    node_path: str | None = Field(allow_null=True, default=None)     # None -> every root
    max_depth: int | None = Field(allow_null=True, default=None, min_value=1)
    budget: int | None = Field(allow_null=True, default=None, min_value=1)
    parallel: bool = Field(default=False)
    max_workers: int = Field(default=4, min_value=1, max_value=32)
    seed: int | None = Field(allow_null=True, default=None)
    score: bool = Field(default=False)
    persist_warnings: bool = Field(default=False)


class InferenceNodeResultSerializer(Serializer):
    unknown_fields = "forbid"
    path: str = Field()
    depth: int = Field()
    output_shape: list[int] | None = Field(allow_null=True)
    duration: float = Field()
    selected: list[str] = Field()           # children this node delegated to
    starved: list[str] = Field()


class InferenceResultSerializer(Serializer):
    """Derives from InferenceResult.as_dict(); document 10 section 7."""
    unknown_fields = "allow"
    outputs: dict = Field()                 # node path -> serialized output
    root_outputs: dict = Field()
    per_node: list[InferenceNodeResultSerializer] = Field()
    batching: list[dict] = Field()          # one entry per level decision
    cost: dict = Field()                    # invocations, wall_time, per_level
    scorecard: EpisodeScorecardSerializer | None = Field(allow_null=True)
    mode: dict[str, bool] = Field()         # {"warnings_persisted": false, ...}
    duration: float = Field()
    seed: int | None = Field(allow_null=True)
```

### 2.9 Observability and recovery serializers

```python
class LogRecordSerializer(Serializer):
    """LogRecord fields, flattened. Source: mactf/training/logging.py."""
    unknown_fields = "allow"
    kind: str = Field(choices=[
        "epoch", "step", "loss", "score", "aggregation", "warning",
        "checkpoint", "health", "resume", "log",
    ])
    run_id: str = Field()
    node_path: str = Field(default="")
    depth: int = Field(default=0)
    epoch: int = Field(default=0)
    step: int = Field(default=0)
    wall_time: float = Field()
    monotonic: float = Field()
    duration: float = Field(default=0.0)
    level: str = Field(default="info", choices=["debug", "info", "warning", "error"])
    message: str = Field(default="")
    fields: dict = Field(default_factory=dict)
    schema_version: int = Field(default=1)


class ProbeResultSerializer(Serializer):
    unknown_fields = "forbid"
    name: str = Field()
    status: str = Field(choices=["healthy", "degraded", "critical"])
    value: float | None = Field(allow_null=True)
    threshold: float | None = Field(allow_null=True)
    detail: str = Field(default="")
    node_path: str = Field(default="")
    metadata: dict = Field(default_factory=dict)


class HealthReportSerializer(Serializer):
    """Derives from HealthReport.as_dict(); document 09 section 4."""
    unknown_fields = "allow"
    status: str = Field(choices=["healthy", "degraded", "critical"])
    action: str = Field(choices=[
        "none", "log", "snapshot", "rollback", "pause", "prune_branch", "abort",
    ])
    epoch: int = Field()
    results: list[ProbeResultSerializer] = Field()
    per_node: dict[str, str] = Field()
    recommendations: list[str] = Field()
    snapshot: "CheckpointSerializer | None" = Field(allow_null=True)
    duration: float = Field()
    monotonic: float = Field()


class CheckpointSerializer(Serializer):
    """Derives from CheckpointRecord.describe() plus sections; document 09."""
    unknown_fields = "allow"
    run_id: str = Field()
    episode: int = Field()
    global_step: int = Field()
    path: str = Field()
    created_at: float = Field()
    reason: str = Field()
    size_bytes: int = Field()
    schema_version: int = Field()
    verified: bool = Field()
    best: bool = Field(default=False)
    sections: list[str] = Field(default_factory=list)


class ResumeReportSerializer(Serializer):
    """Derives from ResumeReport.as_dict(); document 09 section 6."""
    unknown_fields = "allow"
    run_id: str = Field()
    checkpoint: str = Field()
    episode: int = Field()
    global_step: int = Field()
    schema_version: int = Field()
    restored_sections: list[str] = Field()
    missing_sections: list[str] = Field()
    skipped_checkpoints: list[tuple[str, str]] = Field()
    warnings_restored: int = Field()
    optimizers_restored: int = Field()
    bit_identical_expected: bool = Field()
    duration: float = Field()
    notes: list[str] = Field(default_factory=list)


class TrainerCapabilitiesSerializer(Serializer):
    unknown_fields = "forbid"
    optimizer_backends: list[dict] = Field()
    optimizer_granularity: dict[str, str] = Field()   # backend -> "step" | "epoch"
    loss_types: list[str] = Field()
    scorer_types: list[ScorerCapabilitySerializer] = Field()
    finetune_types: list[str] = Field()
    available_extras: dict[str, bool] = Field()       # scipy, peft, transformers,
                                                      # bitsandbytes, tensorboard, wandb
    hierarchy: dict = Field()                         # max_depth, max_nodes, labels, modes
    limits: dict = Field()


class ScorerCapabilitySerializer(Serializer):
    unknown_fields = "forbid"
    type: str = Field()
    order: int = Field()
    required_context: list[str] = Field()
    config_keys: list[str] = Field()
    description: str = Field()
```

`TrainerCapabilitiesSerializer.available_extras` is not cosmetic. It is how a
client learns that `scipy` is absent **before** it submits a run that would fail
at bind time, and it is the honest counterpart of `P3` in `00` section 4.

### 2.10 Extended request serializers

`RunCreateRequestSerializer` gains exactly one field, because the rest belongs to
a trainer spec:

```python
class RunCreateRequestSerializer(Serializer):
    # ... existing fields unchanged ...
    trainer: TrainerSpecSerializer | None = Field(allow_null=True, default=None)
```

`trainer` is optional. When omitted, the run uses the current defaults, so an
existing client's request is still valid and still produces the current
behaviour. When present, `peers` and `trainer.num_peers` are mutually exclusive
and a request with both returns `422` with
`{"detail": "peers and trainer.num_peers are mutually exclusive"}`.

`RunCreateRequestSerializer.resume_from` keeps its meaning ("latest" or an episode
number), but now routes through `MACTFCheckpoint.resume_from_last_checkpoint`
rather than `CheckpointManager.restore`. Its response gains
`resume: ResumeReportSerializer | None`, so the caller learns what was actually
restored instead of inferring it from the run's status.
`RuleEventSerializer`: `rule_id`, `magnitude`, `severity`, `evidence`, `source`,
`created_at`.
`CheckpointSerializer` also carries `kappa`, `entropy`, and `score` in the
registry-projection form used by `GET /runs/{id}/checkpoints`, so one serializer
serves both the checkpoint file's description and the stored row.

### 2.11 Request serializers

```python
class RunCreateRequestSerializer(Serializer):
    unknown_fields = "forbid"
    epochs: int | None = Field(default=None, min_value=1)
    peers: int | None = Field(default=None, min_value=1)
    device: str | None = Field(allow_null=True, default=None)
    seed: int | None = Field(allow_null=True, default=None)
    dry_run: bool = Field(default=False)
    redteam: bool = Field(default=False)
    resume_from: str | None = Field(allow_null=True, default=None)
    # "latest" or an episode number as string
    run_id: str | None = Field(allow_null=True, default=None,
                               pattern=r"^[A-Za-z0-9_-]{1,64}$")
    task_source: str | None = Field(allow_null=True, default=None)
    # registry key from mactf.data.registry
    tags: dict[str, str] = Field(default_factory=dict)
    trainer: TrainerSpecSerializer | None = Field(allow_null=True, default=None)


class LedgerEventPayloadSerializer(Serializer):
    unknown_fields = "forbid"
    rule_id: str = Field()
    magnitude: float = Field(default=1.0, min_value=0.0)
    severity: str | None = Field(allow_null=True, default=None,
                                 choices=["S0", "S1", "S2", "S3", "S4"])
    evidence: str = Field(min_length=1)      # evidence is mandatory, see 4.3
    source: str = Field(default="api")
    tags: list[str] = Field(default_factory=list)


class LedgerPayloadSerializer(Serializer):
    unknown_fields = "forbid"
    peer: str = Field()
    episode: int = Field(default=0)
    events: list[LedgerEventPayloadSerializer] = Field()
    # Optional: score against a stored run's warning state instead of a fresh one.
    run_id: str | None = Field(allow_null=True, default=None)


class AdjudicationPayloadSerializer(Serializer):
    unknown_fields = "forbid"
    peer: str = Field()
    rule_id: str = Field()
    episode: int = Field()
    decision: str = Field(choices=["upheld", "reversed", "severity_changed"])
    severity: str | None = Field(allow_null=True, default=None,
                                 choices=["S0", "S1", "S2", "S3", "S4"])
    rationale: str = Field(min_length=1)
    adjudicator: str = Field(min_length=1)


class WeightPreviewRequestSerializer(Serializer):
    unknown_fields = "forbid"
    scores: dict[str, float] = Field()
    eligible: dict[str, bool] | None = Field(allow_null=True, default=None)
    confidence: dict[str, float] | None = Field(allow_null=True, default=None)
    advance: bool = Field(default=False)
```

`RunCreateRequestSerializer.device` is a free `str` rather than a
`ChoiceField(["auto","cpu","cuda","mps"])` because `resolve_device` also accepts
`cuda:1`. The value is validated in the view by calling `resolve_device` and
returning `422` with the `ImproperlyConfigured` message otherwise. Narrowing it to
a `ChoiceField` would reject a valid device; validating it in the view keeps the
schema honest about the accepted shape of the string while the library decides
which values are real.

## 3. Endpoints

Routes are declared in `mactf/contrib/api/urls.py` with `path(..., name=...)`.
Every row below lists the method, path, **route name**, auth scope, success code,
and the exact library call that produces the body. "Registry" means a method on
`mactf/db.py:Registry`.

The `name` column is the contract: `reverse(name, **kwargs)` must produce the
path, `mactf.http.openapi` publishes it as the `operationId`, and document 03's
client is written against names rather than hard-coded strings. A name is
kebab-case, stable, and unique within the `v1` namespace.

The `Name` column shows the bare name; the api app declares `app_name = "api"`, so
the reverseable form is namespaced — `reverse("api:rule-list")`,
`reverse("api:run-detail", run_id=...)`. The document 05 worker app declares
`app_name = "worker"` and its names reverse as `reverse("worker:register")`. Every
table below omits the prefix for readability, and the router rejects a duplicate
bare name at import time (`mactf.api.E003`, document 13 section 2.3).

Each table's `Path` column is relative to `API.BASE_PATH` (`api/v1`). The mounted
prefix comes from the project's `include()`, so a deployment that mounts the app
under `/training/api/v1/` changes one string in `urls.py` and nothing else —
`reverse()` follows automatically because it works from the resolver, not from a
constant.

### 3.0 Registry methods the API requires

Read-only today; the API needs paging and filters. These are additive changes to
`Registry` with default arguments, so every existing caller and test is
unaffected:

| Method | Status | Change |
|---|---|---|
| `list_runs(limit=20)` | exists | add `offset: int = 0`; return `(rows, total)` or add `count_runs()` |
| `episodes(run_id)` | exists | add `limit`, `offset` |
| `record_events(run_id, peer, episode, events)` | exists | unchanged |
| `rule_counts(run_id)` | exists | unchanged |
| `checkpoints(run_id)` | exists | unchanged |
| `record_warning`, `expire_warning`, `active_warnings`, `restore_warnings` | new | document 04.4 |
| `record_adjudication`, `adjudications`, `mark_events_reversed` | new | document 04.4 |
| `record_aggregation`, `aggregations` | new | document 04.4 |
| `episodes_with_events(run_id, *, peer, rule_id, episode, severity, source, limit, offset)` | new | powers `GET .../rule-events` without loading every event into Python |

Where a row below says "Registry.list_runs(limit, offset)", read it as this
table's version.

### 3.1 Meta

| Method | Path | Name | Scope | Success | Implementation |
|---|---|---|---|---|---|
| GET | `/health` | `health` | none | 200 | `{"status":"ok","version":mactf.__version__,"uptime":float}`; no settings load, safe for probes |
| GET | `/meta` | `meta` | `read` | 200 | `mactf.__version__`, `VERSION`, `settings.module`, `runtime.describe()`, `len(CATALOG)`, sandbox `CapabilityReport` summary, `api_version="v1"` |
| GET | `/rules` | `rule-list` | `read` | 200 | `list(CATALOG.values())` -> `RuleSerializer`; optional `?kind=reward`/`?tier=1` filters |
| GET | `/rules/{rule_id}` | `rule-detail` | `read` | 200 / 404 | `get_rule(rule_id)`; unknown id -> `RuleViolation` -> 404 |
| GET | `/rules/validation` | `rule-validation` | `read` | 200 | `validate_catalog()` -> `CatalogValidationSerializer` |
| GET | `/severities` | `severity-list` | `read` | 200 | `SEVERITY.WEIGHTS` joined with `Severity` |
| GET | `/tiers` | `tier-list` | `read` | 200 | `TIERS` |
| GET | `/scores/scale` | `score-scale` | `read` | 200 | `PointsTable.to_dict()`: all 40 bindings, severity fractions, `warning_notional`, `reward_total`, `table_version` |
| GET | `/openapi.json` | `openapi` | `read` | 200 | `mactf.http.openapi.schema()`, built from `ROOT_URLCONF` (document 12 section 12) |
| POST | `/system-checks` | `system-checks` | `read` | 200 | `mactf.checks.run_checks()`; returns `ok`, `summary`, `messages[]` with `level`, `id`, `message`, `hint`. Never 500s on a failing check |

`GET /meta` is the endpoint a client uses to decide whether an operation is
possible. It never lies: `sandbox.isolation` is the *effective* level from
`choose_provider`, and `sandbox.reasons` is the provider's own list.

`GET /openapi.json` is a view in the app rather than a framework feature:
`mactf/http/openapi.py` supplies the generator; the app decides to publish it
(document 12 section 12).

### 3.2 Runs

| Method | Path | Name | Scope | Success | Implementation |
|---|---|---|---|---|---|
| POST | `/runs` | `run-list` | `run:write` | 202 + `Location` | `RunManager.start(RunCreateRequestSerializer)` |
| GET | `/runs` | `run-list` | `read` | 200 | `Registry.list_runs(limit=..., offset=...)` |
| GET | `/runs/{run_id}` | `run-detail` | `read` | 200 / 404 | `Registry.get_run(run_id)` |
| POST | `/runs/{run_id}/cancel` | `run-cancel` | `run:write` | 202 / 404 / 409 | `RunManager.cancel(run_id)` |
| GET | `/runs/{run_id}/summary` | `run-summary` | `read` | 200 / 404 | stored `runs` row + latest episode + last aggregation |

`run-list` is one name for two methods, which is what `reverse("api:run-list")`
should mean and what Django does with `ListView`. The route-completeness test
compares `(method, path)` pairs, so a shared name is not a gap.

`POST /runs` behaviour:

1. Validate invariants that would fail later anyway: `device` via
   `resolve_device`, `peers` against the configured swarm size, `resume_from`
   against `Registry.latest_episode` / `CheckpointManager.latest()`.
2. If an `Idempotency-Key` header is present, look it up in
   the handler's `idempotency` state (in-process LRU keyed by
   `(principal, key)`, 24h TTL, 256 entries). On hit, replay the original 202
   with the same body and `Idempotent-Replay: true`.
3. Acquire the concurrency semaphore; on failure `429` with `Retry-After: 5`.
4. `registry.start_run(...)` then `RunManager.start`.
5. Return `202` with `{"run_id": ..., "status": "queued", "links": {"self": ..., "events": ...}}`
   where both links are produced by `reverse("api:run-detail", run_id=...)` and
   `reverse("api:run-events", run_id=...)`, and `Location` is the self link.

`POST /cancel` semantics: sets the cancel event, waits up to
`API.CANCEL_GRACE_SECONDS` (default 10) for the thread to reach an epoch
boundary, returns `202 {"status": "cancelling"}`. A terminal run returns `409`
with `{"current_status": "completed"}` in the problem extensions. There is no
`force` parameter, because killing a thread mid-step corrupts optimizer state;
`RunManager` documents this in the endpoint description.

Run state machine: `queued -> running -> {completed, failed, cancelled, dry_run}`,
plus `running -> paused` and `paused -> running` from the watchdog's `PAUSE`
action, defined and enforced in `mactf/orchestration/live.py:RunManager` and
diagrammed in document 01 section 4. The API adds no transitions of its own. A
`409` is returned for any request whose transition is not in that graph: cancel
on a terminal run, ledger append on a live run, rollback on a live run, config
reload on a live run.

Cancel is the only transition that is requested asynchronously, so its response
status (`cancelling`) is deliberately not a `runs.status` value; the registry
column holds `queued`, `running`, `paused`, `interrupted`, and the four terminal
states `completed`, `failed`, `cancelled`, `dry_run`. A client that needs the
outcome polls the run or reads the `terminal` event.

### 3.3 Run detail

| Method | Path | Name | Scope | Implementation |
|---|---|---|---|---|
| GET | `/runs/{run_id}/episodes` | `run-episode-list` | `read` | `Registry.episodes(run_id)` with `?limit&offset` |
| GET | `/runs/{run_id}/episodes/{episode}` | `run-episode-detail` | `read` | filtered from `Registry.episodes` |
| GET | `/runs/{run_id}/episodes/{episode}/peers/{peer}` | `run-episode-peer` | `read` | nested `scorecard.peers[peer]` |
| GET | `/runs/{run_id}/rule-events` | `run-rule-event-list` | `read` | `SELECT ... FROM rule_events` with `?peer&rule_id&episode&limit&offset` |
| GET | `/runs/{run_id}/rule-counts` | `run-rule-counts` | `read` | `Registry.rule_counts(run_id)` |
| GET | `/runs/{run_id}/aggregations` | `run-aggregation-list` | `read` | new `aggregations` table (04.4) |
| GET | `/runs/{run_id}/checkpoints` | `run-checkpoint-list` | `read` | `Registry.checkpoints(run_id)` |
| POST | `/runs/{run_id}/rollback` | `run-rollback` | `run:write` | `CheckpointManager.restore` + `409` if no checkpoint |
| GET | `/runs/{run_id}/warnings` | `run-warning-list` | `read` | new `warnings` table, current active state per peer/rule |
| GET | `/runs/{run_id}/points` | `run-points` | `read` | per-episode per-peer `PointsSerializer` for the whole run; `?episode&peer` filters |
| GET | `/runs/{run_id}/episodes/{episode}/normalization` | `run-episode-normalization` | `read` | `NormalizationSerializer` for that episode; `404` if the episode has no scorecard, `409` if it was scored with normalization disabled |

`rule-events` must support filtering by `severity` and `source` too; both are
columns in `rule_events`. The endpoint is the audit surface for framework.md
section 12 traceability, so it must never aggregate away the individual events.

`GET .../points` returns one entry per `(episode, peer)` and is the batch form of
the per-peer points in a v2 scorecard. It exists because a dashboard that wants a
points trend across a run would otherwise fetch every scorecard and reach into
`metadata`. When a peer has no points record, the entry is absent rather than
zero: zero is a score, absence is missing data, and conflating them would make
a pre-migration episode look like a failed one.

`GET .../normalization` is the companion the calibration design requires
([11-points-and-normalization.md](11-points-and-normalization.md) section 7):
a `calibrated` number is only meaningful beside the cohort stats that produced
it, so there is no route that returns one without the other. The `409` case is a
real state, not a defensive branch: a run scored with
`COOPETITION.NORMALIZATION.ENABLED=false` has scorecards and no report, and
returning a fabricated single-peer report would misrepresent it.

`POST .../rollback` body: `{"episode": int}`. Rejects with 409 when the run is
live (a checkpoint must not be restored under an active loop) and 404 when the
episode has no checkpoint file. On success returns the restored
`CheckpointSerializer` and emits a `checkpoint_rolled_back` signal.

### 3.4 Live streams

| Method | Path | Name | Scope |
|---|---|---|---|
| GET | `/runs/{run_id}/events` | `run-events` | `read` |

SSE (`text/event-stream`):

```
id: 42
event: epoch
data: {"run_id":"a3ef35594ca7","epoch":3,"score":0.2501,"kappa":1.003,"entropy":1.096,"weights":{"peer_alpha":0.365,...},"duration":0.421}

id: 43
event: phase
data: {"run_id":"...","phase":"aggregation","status":"finished"}

id: 44
event: terminal
data: {"run_id":"...","status":"completed","epochs":3}
```

- Event names are fixed: `run`, `phase`, `epoch`, `warning`, `incident`,
  `checkpoint`, `rollback`, `terminal`, `heartbeat`.
- `id` is a monotonically increasing integer per run, assigned by `EventBus`.
  The ring buffer holds the last `API.EVENT_BUFFER` (default 1024) events.
- Reconnect uses `Last-Event-ID`. The server replays everything after that id
  from the buffer. If the id is older than the buffer, it sends
  `event: stream_lost` with `data: {"oldest": N}` and the client must fall back
  to `GET /runs/{run_id}/episodes`. This is explicit, never silent.
- `heartbeat` every 15s keeps proxies from closing idle streams.
- The stream closes after `terminal`. A client that connects to a finished run
  receives the stored terminal event immediately and the stream closes.
- Implementation: `EventBus.subscribe(run_id, last_id)` returns a subscription
  that blocks on a `threading.Condition` with a heartbeat timeout; the view is a
  generator returning a `StreamingResponse` (document 12 section 11), because the
  run loop is a thread and there is no event loop to bridge to.
- Signals wired in `mactf/orchestration/live.py`: `phase_started`, `phase_finished`,
  `episode_scored`, `episode_peer_scored`, `warning_issued`,
  `warning_escalated`, `warning_expired`, `integrity_incident`,
  `aggregation_completed`, `checkpoint_written`, `checkpoint_rolled_back`. All
  exist in [mactf/signals.py](mactf/signals.py).

There is no WebSocket route. SSE carries every event above over one HTTP
connection, through any proxy that permits streaming, and `mactf client` parses
it with `urllib` alone. A project that needs WebSocket installs a third-party
ASGI package and mounts its own routes; the framework does not ship one
(document 00 section 4 records this as a non-goal, document 12 section 11 as the
transport consequence).

### 3.5 Evaluations and adjudications

This group is what makes document 04.3 observable. Without it, 14 reward rules
and 23 penalty rules have no ingest path.

| Method | Path | Name | Scope | Purpose |
|---|---|---|---|---|
| POST | `/evaluations` | `evaluation-create` | `score:write` | Score a submitted ledger; no run required |
| POST | `/runs/{run_id}/episodes/{episode}/ledger` | `run-episode-ledger` | `score:write` | Append a ledger to a stored episode and rescore that peer |
| GET | `/runs/{run_id}/flags` | `run-flags` | `read` | Flag precision and `P19` state |
| POST | `/runs/{run_id}/adjudications` | `run-adjudication-create` | `score:write` | Record a reversal (W5) or severity change |
| GET | `/runs/{run_id}/adjudications` | `run-adjudication-list` | `read` | Adjudication history |

`POST /evaluations` implementation, in order:

1. `Ledger(peer=payload.peer)`; for each event call `ledger.record(...)`, which
   already validates the rule ID and rejects negative magnitudes
   ([mactf/scoring/ledger.py:62](mactf/scoring/ledger.py#L62)).
2. `scorer = get_scorer_dep(request)` or a fresh `Scorer` when `run_id` is absent
   (a fresh scorer means no warning history leaks between unrelated evaluations).
3. `card = scorer.score_peer(ledger, peer=payload.peer, episode=payload.episode)`.
4. Return `ScorecardSerializer` plus `explain` (the `PeerScorecard.explain()`
   text) for debuggability.

`explain` is the one key not present in `PeerScorecard.to_dict()`. It is not
API-only *state*: it is computed on demand from `explain()` and never stored, so
principle P2 (no API-only state) holds. Documenting it here because
`tests/http/test_serializers.py` would otherwise flag it as a declared field with
no dataclass source; a serializer is allowed to carry derived read-only fields,
and this is the only one in the whole contract.

Response for the canonical P1 case, which is the test in document 06:

```json
{
  "peer": "peer_alpha",
  "episode": 0,
  "final_score": -5.0,
  "void": true,
  "floor": -5.0,
  "floor_reason": "non-offsettable penalty P1 applied as an absolute floor of -5.0000; no reward can reduce it",
  "non_offsettable": ["P1"],
  "continuous": {
    "reward": {"total": 0.0},
    "penalty": {"amounts": {"P1": 5.0}, "raw_total": 5.0, "omega": 1.0, "total": 5.0},
    "warning": 0.0,
    "net": -5.0
  },
  "explain": "peer=peer_alpha episode=0 final=-5.0000\n  reward=0.0000 ..."
}
```

These numbers are verified against the current implementation, not illustrative:
`PENALTY_LAMBDAS["P1"] = 5.0`, magnitude `1.0`, so `raw_total = 5.0` and
`omega = 1.0` because P1 is not in `OMEGA_RULES`. `final_score` is `-5.0`, equal
to the floor, because `constrain` subtracts the non-offsettable amount as an
absolute floor rather than adding it on top of an already-subtracted penalty. The
implementation must not "fix" this; it is `compliance.constrain`'s existing
contract ([mactf/compliance.py:79](mactf/compliance.py#L79)) and the test
asserts the exact values.

`POST .../ledger` differs from `POST /evaluations` only in that it writes the
events through `Registry.record_events` and replaces the stored peer scorecard,
so the run's own detail endpoints reflect the appended evaluation. It returns
`409` when the run is live, because mutating a scorecard under an active
aggregation would make the epoch's weights unexplainable.

`POST .../adjudications` implementation: writes to the `adjudications` table,
calls `WarningRegistry.expunge(peer, rule_id)` on `decision == "reversed"`
(framework.md W5), and for `severity_changed` records the new severity for
future explainability. It never rewrites a historical `rule_events` row; the
audit trail is append-only. A `reversed` adjudication on a non-warning rule
records the decision and marks the referenced events `reversed` in a new
`reversed_at`/`reversed_by` column pair (migration 2).

`GET .../flags` returns:
`{"submitted": int, "accepted": int, "precision": float, "threshold": float, "p19_active": bool, "per_peer": {...}}`.
`P_min` comes from `ENGINE.P_MIN` (default `0.5`); when a peer's precision is
below it, `p19_active` is true for that peer. The key is defined in
[11-points-and-normalization.md](11-points-and-normalization.md) section 9.1
alongside `F_CAP`, because both are the thresholds the R3 and P19 guards read and
defining them in two documents would let them drift.

### 3.6 Peers

| Method | Path | Name | Scope | Implementation |
|---|---|---|---|---|
| GET | `/peers` | `peer-list` | `read` | `_default_modules`-resolved classes or the live swarm of the current run; returns `name`, `class`, `parameter_count`, `layout_hash` |
| GET | `/peers/{name}` | `peer-detail` | `read` | contract details |
| GET | `/peers/{name}/contract` | `peer-contract` | `read` | `PeerAdapter.contract.describe()` plus `input_names`, `outputs_value`, `output_kind`, `notes` |
| POST | `/weights/preview` | `weight-preview` | `read` | `Aggregator.compute_weights(names, scores, eligible, confidence, advance=False)` |

`/weights/preview` is the single most useful debugging endpoint in the system: it
answers "given these scores, what would the swarm's weights be?" without
constructing a run. It must use `advance=False` so a preview never moves kappa.
The aggregator instance is built from `COOPETITION` settings with
`num_peers=len(scores)`; `feasible_bounds` runs first and a `ComplianceError`
becomes `422` with the exact message, because an infeasible floor/cap is a client
error in the preview and a setting error in a run.

### 3.7 Sandbox

| Method | Path | Name | Scope | Implementation |
|---|---|---|---|---|
| GET | `/sandbox/providers` | `sandbox-provider-list` | `read` | `mactf.sandbox.detect.probe_all()` |
| GET | `/sandbox/status` | `sandbox-status` | `read` | `choose_provider(...)` -> `ProviderChoice`, flattened to `provider`, `isolation`, `requested`, `degraded`, `reasons`, `capabilities` |
| POST | `/sandbox/probe` | `sandbox-probe` | `read` | force a fresh probe, bypassing any cache |
| POST | `/sandbox/exec` | `sandbox-exec` | `sandbox:exec` | provider `exec` on a live handle |

`ProviderChoice` is a `detect.py` dataclass, not a `base.py` one, so the
`CapabilitySerializer` response for `/sandbox/status` is a serializer composed of
`ProviderChoice` fields plus the nested `report.as_dict()`. It is the only place
in the contract where a response serializer composes two dataclasses rather than
projecting one; documented here so the implementer does not go looking for a
missing `as_dict()`.

`POST /sandbox/exec` body:
`{"peer": str, "command": str | list[str], "timeout": float | None}`.
It requires a live handle registered by a run that provisioned that peer. With no
handle it returns `409` with a body naming the reason: either no run is active for
that peer, or the provider is `LocalSandbox` (process isolation) which supports
`exec` only for the current user, or QEMU is configured without a serial socket.
With a handle it returns `ExecSerializer` verbatim, including `exit_code: 127` and
the provider's `stderr` text when the guest agent is absent. The endpoint never
converts a failed exec into a 5xx; the exec result is the response.

Scope `sandbox:exec` is separate from `run:write` because executing arbitrary
commands inside a VM is a materially different privilege.

### 3.8 Scorers, trainers, and capabilities

These carry the wrapper layer's own introspection. They are read-only and add no
state.

| Method | Path | Name | Scope | Success | Implementation |
|---|---|---|---|---|---|
| GET | `/scorers` | `scorer-list` | `read` | 200 | `scorer_capabilities()` from document 07 section 10 -> `list[ScorerCapabilitySerializer]` |
| GET | `/trainers/capabilities` | `trainer-capabilities` | `read` | 200 | `TrainerCapabilitiesSerializer`: optimizer backends, `optimizer_granularity`, loss types, scorer types, `available_extras`, hierarchy caps. Backed by `function_capabilities()` plus import probes |
| POST | `/trainers/preview` | `trainer-preview` | `run:write` | 200 / 422 | Builds the tree and the optimizer **without training** and returns the topology, per-node parameter counts, trainable ratio, projected optimizer bytes, and the `TrainerSpec` normalized back to JSON |
| GET | `/trainers/presets` | `trainer-preset-list` | `read` | 200 | Named `TrainerSpec` presets shipped in settings; `[]` when none are configured |

`POST /trainers/preview` is the endpoint that makes the wrapper layer honest. A
user submits a `TrainerSpec`, and before a single gradient step they learn:

- the exact node paths the tree would have (`PeerTreeSerializer`),
- which parameters are trainable per node and which are frozen,
- whether every trainable parameter is claimed by an optimizer route,
- the projected optimizer memory,
- whether the requested extras are installed.

It answers `200` with a preview or `422` with the same validation error the
trainer would raise at `build()`. It never trains, never writes a run row, and
never allocates a checkpoint. A preview that hangs on a 7B model's deep-copy is
the one cost it does have, and the response streams the topology first over SSE
when `?stream=true` is set.

### 3.9 Hierarchy

| Method | Path | Name | Scope | Success | Implementation |
|---|---|---|---|---|---|
| GET | `/runs/{run_id}/peers/tree` | `run-peer-tree` | `read` | 200 / 404 | live tree when the run is in memory, else the newest checkpoint's `tree` section; `source` says which |
| GET | `/runs/{run_id}/peers/{path:path}` | `run-peer-node` | `read` | 200 / 404 | one `PeerNodeSerializer` plus its scorecard for the latest episode |
| POST | `/runs/{run_id}/infer` | `run-infer` | `run:write` | 200 / 404 / 409 / 422 | `run_inference(tree, inputs, ...)` rooted at every root or at `request.node_path` |
| POST | `/peers/{path:path}/infer` | `peer-infer` | `run:write` | 200 / 404 / 409 / 422 | same, with the run resolved from the live handle or `?run_id=` |

A node path contains `/`, so those two routes use the `path` converter
(document 12 section 5.1), which is `mactf.http`'s equivalent of FastAPI's
`{path:path}`. Declaration order matters: `run-peer-tree` must appear before
`run-peer-node` in `urlpatterns`, or the tree route is shadowed by the node route
with `path="tree"`. That ordering is stated in `mactf/contrib/api/urls.py` and
asserted by a test that resolves `/runs/x/peers/tree` and checks the matched view.

The route validates the segments against the tree grammar (document 10 section
3.3). A syntactically valid path that is not in the tree returns `404` with the
nearest existing prefix in the problem body, because a typo in a deep path is
otherwise very hard to diagnose.

`POST /runs/{id}/infer` on a run whose tree cannot be resolved returns `409`
with `{"reason": "run has no loaded tree and no checkpoint"}`. It does not fall
back to a flat single-peer inference, because that would silently answer a
different question.

Budget exhaustion returns `422` with the offending path and the remaining
budget in the problem extensions, matching the `ContractError` message in
document 10 section 7.3. Returning a truncated result is explicitly forbidden.

### 3.10 Observability

| Method | Path | Name | Scope | Success | Implementation |
|---|---|---|---|---|---|
| GET | `/runs/{run_id}/logs/tail` | `run-log-tail` | `read` | 200 / 404 | `MACTFLogger.tail(n, node_path=, kind=)`; `?source=memory\|file`; default `memory` |
| GET | `/runs/{run_id}/watchdog` | `run-watchdog` | `read` | 200 / 404 | last `HealthReportSerializer` plus `history` bounded by `?limit=` (default 50, max 512) |
| POST | `/runs/{run_id}/watchdog/snapshot` | `run-watchdog-snapshot` | `run:write` | 202 / 404 / 409 | `MACTFWatchdog.snapshot(context, reason="operator")` |

`GET /runs/{id}/logs/tail` reads the `MemorySink` deque by default because it is
bounded and cannot grow a response without limit. `?source=file` reads the run's
JSONL from `PATHS["LOGS"]` and supports the same `node_path` and `kind` filters,
tailing from the end. Neither variant reads the sqlite registry, so a tail request
never contends with a training step.

`POST /runs/{id}/watchdog/snapshot` on a live run is allowed and takes effect
between epochs. On a terminal run it is a `409`; on a run with no live handle it
is a `409` with the reason. The response is the `CheckpointSerializer` once the
save completes, not a `202`, because a snapshot the caller cannot see is not
useful.

### 3.11 Checkpoints and resume

| Method | Path | Name | Scope | Success | Implementation |
|---|---|---|---|---|---|
| GET | `/runs/{run_id}/checkpoints` | `run-checkpoint-list` | `read` | 200 | `MACTFCheckpoint.list(valid_only=?)` -> `list[CheckpointSerializer]` |
| GET | `/runs/{run_id}/checkpoints/{episode}` | `run-checkpoint-detail` | `read` | 200 / 404 | `describe(path)` plus `verify(path)` |
| POST | `/runs/{run_id}/checkpoints/{episode}/verify` | `run-checkpoint-verify` | `read` | 200 | `verify(path)` -> `{"ok": bool, "problems": [...], "content_hash": str}` |
| POST | `/runs/{run_id}/resume` | `run-resume` | `run:write` | 202 / 404 / 409 | `resume_from_last_checkpoint()` then continue the same run id |
| POST | `/runs/{run_id}/checkpoints/{episode}/rollback` | `run-checkpoint-rollback` | `run:write` | 202 / 404 / 409 | existing rollback, now through `MACTFCheckpoint` |

`run-checkpoint-list` appears in both section 3.3 and here on purpose: section 3.3
lists the route in its run-detail grouping, this section specifies its
implementation. A route appears once in `urlpatterns`.

`POST /runs/{id}/resume` differs from `POST /runs` with `resume_from`. The former
continues an existing run id that is `paused` or `interrupted`; the latter starts
a **new** run seeded from a checkpoint and records the source episode in
`runs.notes`. Both are needed: one is recovery, one is branching from a
checkpoint. The distinction is in the response — `resume` returns
`ResumeReportSerializer`, a new run returns `RunSerializer` with
`resumed_from: int | None`.

`resume` on a `completed` run returns `409` with
`{"reason": "run is terminal; create a new run with resume_from"}`. Silently
restarting a completed run would make its original scorecard a lie.

### 3.12 Admin

| Method | Path | Name | Scope | Implementation |
|---|---|---|---|---|
| GET | `/admin/capabilities` | `admin-capabilities` | `admin` | everything `/meta` reports plus installed extras, settings module, writable paths from `PATHS` |
| POST | `/admin/config/reload` | `admin-config-reload` | `admin` | `conf.configure(force=True)` then rebuild scorer, aggregator, runner |

`POST /admin/config/reload` refuses with `409` while any run is live: swapping
settings under an active loop would make the epoch's provenance ambiguous. On
success it returns the new `/meta` payload.

## 4. Cross-cutting rules

### 4.1 Errors

`mactf/http/exceptions.py` maps the framework's exceptions to statuses and
renders RFC 9457 problem+json (document 12 section 10). The table below is the
normative client-visible version:

| Exception | Status | `type` |
|---|---|---|
| `RuleViolation` | 422 | `urn:mactf:error:rule-violation` |
| `ComplianceError` | 422 | `urn:mactf:error:compliance` |
| `ImproperlyConfigured` | 500 | `urn:mactf:error:configuration` |
| `ContractError` | 422 | `urn:mactf:error:contract` |
| `SandboxError` and subclasses | 502 | `urn:mactf:error:sandbox` |
| `LookupError` | 404 | `urn:mactf:error:not-found` |
| `CommandError` | 400 | `urn:mactf:error:command` |
| `MACTFError` (base) | 500 | `urn:mactf:error:internal` |
| `ValidationError` (`mactf.http`) | 422 | `urn:mactf:error:validation` |
| `Http404` | 404 | `urn:mactf:error:not-found` |
| `MethodNotAllowed` | 405 | `urn:mactf:error:method-not-allowed` |
| `Conflict` | 409 | `urn:mactf:error:conflict` |
| `RateLimited` | 429 | `urn:mactf:error:rate-limited` |
| `RequestDataTooBig` | 413 | `urn:mactf:error:too-large` |
| `UnsupportedMediaType` | 415 | `urn:mactf:error:unsupported-media-type` |
| any other exception | 500 | `urn:mactf:error:internal` |

Body:

```json
{
  "type": "urn:mactf:error:compliance",
  "title": "ComplianceError",
  "status": 422,
  "detail": "WEIGHT_CAP=0.3 is below 1/4=0.25 ...",
  "instance": "/api/v1/weights/preview",
  "trace_id": "0f3c...",
  "extensions": {}
}
```

`trace_id` is generated per request by `RequestIdMiddleware` and also returned in
the `X-Trace-Id`
response header, so a user can quote it from a UI. Stack traces are logged, never
returned, unless `settings["DEBUG"]` is true.

### 4.2 Pagination

`limit` (default 50, max 500) and `offset` (default 0) query parameters on every
collection endpoint. Responses include:

```json
{"items": [...], "total": 137, "limit": 50, "offset": 0, "has_more": true}
```

No cursor tokens. The registry is sqlite and every collection is small (episodes
per run, events per run); offset paging is correct and simpler. A future
Postgres backend can add cursors without breaking this shape by adding fields.

### 4.3 Evidence is mandatory on ingest

Every `LedgerEventPayload.evidence` has `min_length=1`. framework.md section 2
defines evidence as "a verifiable source, reproducible computation, test result,
or direct observation" and section 9.5.5 says "Evidence-free flags are P4 by
default". Accepting an empty evidence string would let a client create a
penalizable act out of nothing. The API therefore rejects with 422 rather than
guessing.

The points path depends on this rule more heavily than the docx path does,
because `EpisodeEvidence` is built from the same payloads and a guard with no
evidence returns `missing=(...)` rather than `earned=True`
([11-points-and-normalization.md](11-points-and-normalization.md) section 4).
Tightening this validation later would change points without changing
`final_score`; loosening it would let a client earn rubric points with no
evidence. That asymmetry is why the payload rule is stated here as normative
rather than as an input-hygiene convenience.

### 4.3.1 Scorecard version negotiation

Two scorecard representations exist, and a client chooses between them
explicitly. Both are valid responses from the same route.

| Mechanism | Value | Result |
|---|---|---|
| `Accept` header | `application/vnd.mactf.v2+json` | `ScorecardV2Serializer` |
| Query parameter | `?scorecard_version=2` | `ScorecardV2Serializer` |
| Neither | - | `ScorecardSerializer` (v1 shape, plus the optional `points` field) |

Resolution order, and the reason for it:

1. If `Accept` names a vendor media type, it wins. A client that sets a header is
   usually a generated binding, and the header is the part it configured once.
2. Otherwise, a `scorecard_version` query parameter is honoured. Browsers and
   `curl` cannot easily set `Accept` to a vendor type without losing `*/*`
   negotiation, so the query form exists.
3. Otherwise v1.

Precedence matters in exactly one case: a request with
`Accept: application/vnd.mactf.v2+json` and `?scorecard_version=1`. The header
wins, and the response carries
`Warning: 299 - "scorecard_version query ignored; Accept header takes precedence"`.
Silently honouring the query would make the two mechanisms contradictory with no
way for a client to know which applied.

Unsupported versions produce `406` with a problem body that names the supported
values:

```json
{
  "type": "https://mactf.dev/problems/unsupported-scorecard-version",
  "title": "Unsupported scorecard version",
  "status": 406,
  "supported": ["1", "2"],
  "requested": "3"
}
```

Version `1` is the default indefinitely. There is no deprecation of the v1 shape
scheduled, because v1 is what every existing client parses and the new fields are
additive: `points` is present in v1 responses whenever the run was scored by a
version that knows about points, and `null` otherwise. A client that ignores it
loses nothing it previously had. `v2` exists to make `points` non-optional and to
add `normalization` and `guards_denied`, which are the fields a client should not
have to treat as optional when reasoning about calibration.

### 4.4 Auth

`mactf/contrib/api/auth.py`, with the `Principal` value object in
`mactf/http/auth.py` (document 12 section 7) so `AuthMiddleware` can resolve a
principal without importing the app:

```python
# mactf/http/auth.py
@dataclass(frozen=True)
class Principal:
    name: str
    scopes: frozenset[str]
    anonymous: bool = False

# mactf/contrib/api/auth.py
class TokenResolver(Protocol):
    def resolve(self, token: str) -> Principal | None: ...

class StaticTokenResolver:
    """Tokens from API.TOKENS, a JSON object of token -> [scopes]."""
```

Configuration in a new settings key. The authoritative default set is
[13-app-integration.md](13-app-integration.md) section 3; this block is the same
list, shown here because the routes above read it:

```python
API = {
    "ALLOW_ANONYMOUS": True,     # dev default; run:write and admin are never anonymous
    "TOKENS": {},                # {"token": ["read", "run:write", ...]}
    "BASE_PATH": "api/v1",
    "EVENT_BUFFER": 1024,
    "MAX_CONCURRENT_RUNS": 1,
    "CANCEL_GRACE_SECONDS": 10.0,
    "CORS_ORIGINS": [],
    "PRINCIPAL_RESOLVER": None,
    # additive tuning, all with defaults in DEFAULTS (document 13 section 3)
    "IDEMPOTENCY_TTL_SECONDS": 86400,
    "LOG_TAIL_DEFAULT": 100,
    "LOG_TAIL_MAX": 2000,
    "HEALTH_HISTORY_LIMIT": 50,
    "PREVIEW_MAX_NODES": 64,     # /trainers/preview refuses a bigger tree
    "INFERENCE_MAX_WORKERS": 4,
    "INFERENCE_BUDGET_DEFAULT": None,
}
```

Every key above must be present in `DEFAULTS`
([mactf/conf.py:27](mactf/conf.py#L27)); a key absent from it is silently dropped
by `load_settings` (document 13 section 1). `API` is a dict in `DEFAULTS` and
`_deep_merge` applies a project's partial dict over it, so a project may set only
`ALLOW_ANONYMOUS` and keep the rest.

`PREVIEW_MAX_NODES` is a deliberate ceiling: a preview deep-copies modules to
count parameters, so a 256-node tree of 7B models would take minutes and
gigabytes to answer a question the user could answer from the spec. The endpoint
returns `422` naming the limit and suggesting `GET /trainers/capabilities` for
the arithmetic instead.

Scopes: `read`, `run:write`, `score:write`, `sandbox:exec`, `admin`.

Rules:

- `GET /health` never requires auth.
- `ALLOW_ANONYMOUS: True` grants `read` only. It never grants `run:write`,
  `score:write`, `sandbox:exec`, or `admin`.
- Missing token on a scoped route -> `401` with `WWW-Authenticate: Bearer`.
- Insufficient scope -> `403` with `required_scope` in extensions.
- A token with an unknown scope string is a startup error, not a runtime one:
  `mactf.contrib.api.checks` validates the scope vocabulary against the `SCOPES`
  constant and reports `mactf.api.E004`, so `mactf check` fails before a server
  starts.
- Timing: token comparison uses `hmac.compare_digest`.

`ALLOW_ANONYMOUS: True` is the default so that `mactf serve` on localhost works
immediately, matching the framework's existing "works before configuration"
philosophy. Setting it to `False` with an empty `TOKENS` map is reported by
`mactf.api.E001` as an error and `mactf serve` refuses to start, because a locked
door with no key is a configuration mistake.

### 4.5 Headers

| Header | Direction | Meaning |
|---|---|---|
| `X-MACTF-Version` | response | `mactf.__version__` |
| `X-API-Version` | response | `v1` |
| `X-Trace-Id` | both | correlation id; client-supplied values are reused if well-formed |
| `Idempotency-Key` | request | `POST /runs` only, 1-128 chars |
| `Idempotent-Replay` | response | `true` when a replay was served |
| `Retry-After` | response | on 429 |
| `Last-Event-ID` | request | SSE resume |
| `WWW-Authenticate` | response | on 401 |

### 4.6 Rate limiting

Not implemented. The deployment model is a single trusted client or a team
behind one token. Documenting its absence is better than shipping a token bucket
that gives a false sense of protection; section 6 of document 00 records it as a
deliberate non-goal.

## 5. Route table completeness test

`tests/contrib_api/test_contract_completeness.py` reads this document, extracts
every `| METHOD | /path |` row from the endpoint tables, normalizes path
parameters (`{run_id}` -> `/{run_id}`, `{path:path}` -> `/{path}`), and asserts
set equality with the routes produced by `mactf.contrib.api.urls`:

```python
documented = {(method, normalize(path)) for method, path in rows(this_document)}
mounted = {
    (method, pattern.route)
    for pattern in walk(get_resolver())
    for method in pattern.callback.allowed_methods
}
assert documented == mounted
```

It fails in both directions: a route in the app that is not documented, and a
documented route that is not mounted.

A second assertion covers names, because a name is now part of the contract:

```python
assert set(documented_names) == {pattern.name for pattern in walk(get_resolver())}
assert all(reverse(name, **_sample_kwargs(name)) for name in documented_names)
```

This is why the plan's tables are exhaustive rather than illustrative. A missing
row is a test failure, not a documentation nit.

## 6. OpenAPI quality

`mactf.http.openapi.schema()` (document 12 section 12) generates the document from
the URL conf and serializer metadata, so the schema cannot drift from the routes:

- every route has `summary`, `description`, and `tags`
- every non-2xx path is declared with the `Problem` component
- every route's `operationId` is its route `name` — `meta`, `run-detail`,
  `run-events`, and so on. There is no separate naming scheme, which is what makes
  `operationId` and `reverse()` agree by construction rather than by a lookup table
- every route with a request body declares the `Serializer` subclass's schema
- the problem schema is registered once as a component named `Problem`

`tests/http/test_openapi.py` asserts: no route lacks a summary, no `operationId`
is duplicated, every `operationId` equals its route name, and every route
declares a response schema except the stream endpoint (which returns
`StreamingResponse` and is documented as such).

### 6.1 Response schema derivation rule

`P2` in `00` section 4 says every response serializer is a projection of an
existing dataclass's `to_dict()`. The serializer table below is the same set as
section 2, expressed as the derivation each one follows:

| Serializer | Derived from |
|---|---|
| `ScorerCapabilitySerializer` | `Scorer.describe()` |
| `TrainerCapabilitiesSerializer` | `function_capabilities()` + `scorer_capabilities()` + import probes |
| `PeerNodeSerializer` | `PeerNode.describe()` |
| `PeerTreeSerializer` | `PeerTree.to_dict()` plus `source` and the caps from settings |
| `InferenceResultSerializer` | `InferenceResult.as_dict()` |
| `LogRecordSerializer` | `LogRecord` fields, flattened |
| `ProbeResultSerializer` | `ProbeResult` |
| `HealthReportSerializer` | `HealthReport.as_dict()` |
| `CheckpointSerializer` | `CheckpointRecord.describe()` plus `sections` |
| `ResumeReportSerializer` | `ResumeReport.as_dict()` |
| `PointsSerializer` | `PointsBreakdown.as_dict()` |
| `RulePointsSerializer` | `RulePoints.as_dict()` |
| `DenialSerializer` | `DenialRecord.as_dict()` |
| `NormalizationSerializer` | `NormalizationReport.as_dict()` |
| `CohortStatsSerializer` | `CohortStats.as_dict()` |
| `NormalizedScoreSerializer` | `NormalizedScore` fields |
| `NormalizationPolicySerializer` | `NormalizationPolicy` fields (`dataclasses.asdict`) |
| `AggregationSerializer` | `AggregationReport.as_dict()` |
| `ComplianceSerializer` | `ComplianceResult.as_dict()` |
| `CapabilitySerializer` | `CapabilityReport.as_dict()` |
| `ExecSerializer` | `ExecResult.as_dict()` |
| `ScorecardSerializer` | `PeerScorecard.to_dict()` |
| `EpisodeScorecardSerializer` | `EpisodeScorecard.to_dict()` |

Every one of those sources gains a `from_dict` (document 13 section 5), so each
row is a round trip rather than a write-only projection.

`NormalizationPolicySerializer` is the one serializer in this group whose source
has no `to_dict()`, because `NormalizationPolicy` is a frozen dataclass with no
serializer. The projection is `dataclasses.asdict`, which is stable for a flat
dataclass with only scalars, tuples, and booleans. Adding a field to the policy
therefore requires adding it here, and `tests/scoring/test_points_models.py`
asserts the field sets are equal in both directions so the omission is a test
failure rather than a silently dropped setting in an echo response.

`TrainerCapabilitiesSerializer` is the only composed serializer in the contract
and is documented as such here, exactly as `ProviderChoice` is in section 3.7.
