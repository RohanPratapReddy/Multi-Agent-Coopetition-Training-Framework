"""The ``api/v1`` routes (document 02 section 3).

``app_name = "api"`` makes ``include("mactf.contrib.api.urls")`` mount under the
``api:`` namespace, so ``reverse("api:run-detail", run_id=...)`` works wherever the
project mounts it. Do not pass ``namespace=`` to ``include``: the client reverses
against ``api:``.

Ordering matters in exactly two places, both asserted by tests:

* ``rules/validation`` is declared before ``rules/{rule_id}``;
* ``runs/{run_id}/peers/tree`` is declared before ``runs/{run_id}/peers/{path:path}``.
"""

from __future__ import annotations

from ...http.urls import path
from .views.admin import AdminCapabilitiesView, ConfigReloadView
from .views.episodes import (
    AggregationListView,
    EpisodeDetailView,
    EpisodeListView,
    EpisodePeerView,
    NormalizationView,
    PointsView,
    RuleCountsView,
    RuleEventListView,
    WarningListView,
)
from .views.evaluations import AdjudicationView, EpisodeLedgerView, EvaluationCreateView, FlagsView
from .views.hierarchy import PeerInferView, RunInferView, RunPeerNodeView, RunPeerTreeView
from .views.meta import (
    HealthView,
    MetaView,
    OpenApiView,
    RuleDetailView,
    RuleListView,
    RuleValidationView,
    ScoreScaleView,
    SeverityListView,
    SystemChecksView,
    TierListView,
)
from .views.observability import LogTailView, WatchdogSnapshotView, WatchdogView
from .views.peers import PeerContractView, PeerDetailView, PeerListView, WeightPreviewView
from .views.recovery import (
    CheckpointDetailView,
    CheckpointListView,
    CheckpointRollbackView,
    CheckpointVerifyView,
    RunResumeView,
    RunRollbackView,
)
from .views.runs import RunCancelView, RunDetailView, RunListView, RunSummaryView
from .views.sandbox import SandboxExecView, SandboxProbeView, SandboxProviderListView, SandboxStatusView
from .views.scorers import (
    ScorerListView,
    TrainerCapabilitiesView,
    TrainerPresetListView,
    TrainerPreviewView,
)
from .views.streams import RunEventsView

app_name = "api"

urlpatterns = [
    # -- meta (3.1) ---------------------------------------------------------------
    path("health", HealthView.as_view(), name="health"),
    path("meta", MetaView.as_view(), name="meta"),
    path("rules", RuleListView.as_view(), name="rule-list"),
    path("rules/validation", RuleValidationView.as_view(), name="rule-validation"),  # before {rule_id}
    path("rules/{rule_id}", RuleDetailView.as_view(), name="rule-detail"),
    path("severities", SeverityListView.as_view(), name="severity-list"),
    path("tiers", TierListView.as_view(), name="tier-list"),
    path("scores/scale", ScoreScaleView.as_view(), name="score-scale"),
    path("openapi.json", OpenApiView.as_view(), name="openapi"),
    path("system-checks", SystemChecksView.as_view(), name="system-checks"),
    # -- runs (3.2) ---------------------------------------------------------------
    path("runs", RunListView.as_view(), name="run-list"),  # GET lists, POST starts
    path("runs/{run_id}", RunDetailView.as_view(), name="run-detail"),
    path("runs/{run_id}/cancel", RunCancelView.as_view(), name="run-cancel"),
    path("runs/{run_id}/summary", RunSummaryView.as_view(), name="run-summary"),
    # -- run detail (3.3) -----------------------------------------------------------
    path("runs/{run_id}/episodes", EpisodeListView.as_view(), name="run-episode-list"),
    path("runs/{run_id}/episodes/{episode:int}", EpisodeDetailView.as_view(), name="run-episode-detail"),
    path("runs/{run_id}/episodes/{episode:int}/peers/{peer}", EpisodePeerView.as_view(), name="run-episode-peer"),
    path(
        "runs/{run_id}/episodes/{episode:int}/normalization",
        NormalizationView.as_view(),
        name="run-episode-normalization",
    ),
    path("runs/{run_id}/rule-events", RuleEventListView.as_view(), name="run-rule-event-list"),
    path("runs/{run_id}/rule-counts", RuleCountsView.as_view(), name="run-rule-counts"),
    path("runs/{run_id}/aggregations", AggregationListView.as_view(), name="run-aggregation-list"),
    path("runs/{run_id}/checkpoints", CheckpointListView.as_view(), name="run-checkpoint-list"),
    path("runs/{run_id}/rollback", RunRollbackView.as_view(), name="run-rollback"),
    path("runs/{run_id}/warnings", WarningListView.as_view(), name="run-warning-list"),
    path("runs/{run_id}/points", PointsView.as_view(), name="run-points"),
    # -- live stream (3.4) ------------------------------------------------------------
    path("runs/{run_id}/events", RunEventsView.as_view(), name="run-events"),
    # -- evaluations and adjudications (3.5) --------------------------------------------
    path("evaluations", EvaluationCreateView.as_view(), name="evaluation-create"),
    path("runs/{run_id}/episodes/{episode:int}/ledger", EpisodeLedgerView.as_view(), name="run-episode-ledger"),
    path("runs/{run_id}/flags", FlagsView.as_view(), name="run-flags"),
    path("runs/{run_id}/adjudications", AdjudicationView.as_view(), name="run-adjudication-create"),
    path("runs/{run_id}/adjudications", AdjudicationView.as_view(), name="run-adjudication-list"),
    # -- peers (3.6) --------------------------------------------------------------------
    path("peers", PeerListView.as_view(), name="peer-list"),
    path("peers/{name}", PeerDetailView.as_view(), name="peer-detail"),
    path("peers/{name}/contract", PeerContractView.as_view(), name="peer-contract"),
    path("weights/preview", WeightPreviewView.as_view(), name="weight-preview"),
    # -- sandbox (3.7) --------------------------------------------------------------------
    path("sandbox/providers", SandboxProviderListView.as_view(), name="sandbox-provider-list"),
    path("sandbox/status", SandboxStatusView.as_view(), name="sandbox-status"),
    path("sandbox/probe", SandboxProbeView.as_view(), name="sandbox-probe"),
    path("sandbox/exec", SandboxExecView.as_view(), name="sandbox-exec"),
    # -- scorers and trainers (3.8) ----------------------------------------------------------
    path("scorers", ScorerListView.as_view(), name="scorer-list"),
    path("trainers/capabilities", TrainerCapabilitiesView.as_view(), name="trainer-capabilities"),
    path("trainers/preview", TrainerPreviewView.as_view(), name="trainer-preview"),
    path("trainers/presets", TrainerPresetListView.as_view(), name="trainer-preset-list"),
    # -- hierarchy (3.9) ----------------------------------------------------------------------
    path("runs/{run_id}/peers/tree", RunPeerTreeView.as_view(), name="run-peer-tree"),  # before {path:path}
    path("runs/{run_id}/peers/{path:path}", RunPeerNodeView.as_view(), name="run-peer-node"),
    path("runs/{run_id}/infer", RunInferView.as_view(), name="run-infer"),
    path("peers/{path:path}/infer", PeerInferView.as_view(), name="peer-infer"),
    # -- observability (3.10) -------------------------------------------------------------------
    path("runs/{run_id}/logs/tail", LogTailView.as_view(), name="run-log-tail"),
    path("runs/{run_id}/watchdog", WatchdogView.as_view(), name="run-watchdog"),
    path("runs/{run_id}/watchdog/snapshot", WatchdogSnapshotView.as_view(), name="run-watchdog-snapshot"),
    # -- checkpoints and resume (3.11) ----------------------------------------------------------
    path("runs/{run_id}/checkpoints/{episode:int}", CheckpointDetailView.as_view(), name="run-checkpoint-detail"),
    path(
        "runs/{run_id}/checkpoints/{episode:int}/verify",
        CheckpointVerifyView.as_view(),
        name="run-checkpoint-verify",
    ),
    path("runs/{run_id}/resume", RunResumeView.as_view(), name="run-resume"),
    path(
        "runs/{run_id}/checkpoints/{episode:int}/rollback",
        CheckpointRollbackView.as_view(),
        name="run-checkpoint-rollback",
    ),
    # -- admin (3.12) ------------------------------------------------------------------------------
    path("admin/capabilities", AdminCapabilitiesView.as_view(), name="admin-capabilities"),
    path("admin/config/reload", ConfigReloadView.as_view(), name="admin-config-reload"),
]
