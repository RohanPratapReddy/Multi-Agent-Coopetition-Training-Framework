"""The warning stage: the only stateful scorer, and the only one with ``order=5``.

Its ordering is not negotiable. Today ``Scorer.score_peer`` issues warnings
first, collects ``warning_events`` and ``escalated_rules``, then computes the
continuous score with both (``pipeline.py:73-101``). If warnings ran after
penalties, a rule that matured this episode would be scored both as a
warning-only rule (excluded from ``P``) and as a penalty (included), which is
wrong in either direction.

Therefore:

* :meth:`prepare` advances the registry and returns a context already carrying
  ``warning_total``, ``warning_events``, and ``escalated_rules``.
* :meth:`score` is read-only and reports the same numbers plus the
  :class:`~mactf.scoring.base.WarningMutation` tuples that were applied.
* :meth:`prepare` is idempotent per ``(node_path, peer, episode)``, because a
  caller may reasonably re-score an episode after editing its ledger.

Per-node scoping (document 10 section 6) keys registries by ``(node_path,
rule_id)`` so a nano-agent's W4 cannot decay a root peer's W4. In flat mode
``node_path == ""`` and behaviour is byte-identical to before.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..exceptions import ImproperlyConfigured
from ..rules.catalog import ALL_RULES
from .base import (
    ScoreContext,
    Scorer,
    ScorerDiagnostic,
    ScorerOutput,
    ScoringKind,
    WarningMutation,
)
from .warning import EscalationOutcome, WarningPolicy, WarningRegistry

DEFAULT_KNOWN_RULES: frozenset[str] = frozenset(
    rule.id for rule in ALL_RULES if rule.is_penalty
)


class WarningScorer(Scorer):
    """Advances the warning regime and reports what it changed."""

    kind = ScoringKind.WARNING
    order = 5  # before reward and penalty, on purpose
    requires = frozenset({"ledger", "warning_policy"})

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        policy: WarningPolicy | None = None,
        registries: dict[str, WarningRegistry] | None = None,
    ) -> None:
        super().__init__(config=config)
        self.policy = policy or WarningPolicy.from_settings(dict(config or {}).get("WARNINGS", {}))
        self.registries: dict[str, WarningRegistry] = registries if registries is not None else {}
        #: ``(node_path, peer, episode)`` -> prepared context, so a second call for
        #: the same episode cannot double-issue or double-count.
        self._prepared: dict[tuple[str, str, int], ScoreContext] = {}

    # -- registry helpers ------------------------------------------------
    @staticmethod
    def _key(node_path: str, peer: str) -> str:
        return f"{node_path}::{peer}" if node_path else peer

    def registry_for(self, peer: str) -> WarningRegistry:
        """The root-level registry for a peer (``node_path == ""``)."""
        return self.registry_for_node("", peer)

    def registry_for_node(self, node_path: str, peer: str) -> WarningRegistry:
        key = self._key(node_path, peer)
        if key not in self.registries:
            self.registries[key] = WarningRegistry(self.policy)
        return self.registries[key]

    def begin_episode(
        self, peer: str, *, episode: int, node_path: str = ""
    ) -> list[Any]:
        return self.registry_for_node(node_path, peer).begin_episode(peer, episode=episode)

    def reset_episode_cache(self) -> None:
        """Forget prepared contexts so a later call re-runs the regime.

        Called by the pipeline when an episode's ledger changes materially, and
        available to a caller that knows it is re-scoring rather than replaying.
        """
        self._prepared.clear()

    # -- prepare ---------------------------------------------------------
    def prepare(self, ctx: ScoreContext) -> ScoreContext:
        cache_key = (ctx.node_path, ctx.peer, ctx.episode)
        cached = self._prepared.get(cache_key)
        if cached is not None:
            return cached

        registry = self.registry_for_node(ctx.node_path, ctx.peer)
        registry.begin_episode(ctx.peer, episode=ctx.episode)

        issued: list[WarningMutation] = []
        escalated: list[WarningMutation] = []
        warning_events: list[str] = []
        escalated_rules: set[str] = set()
        warning_penalty = 0.0

        for event in ctx.ledger.penalties():
            if not event.rule.warning_eligible:
                continue
            outcome: EscalationOutcome = registry.issue(
                ctx.peer,
                event.rule_id,
                severity=event.resolved_severity(),
                episode=ctx.episode,
                evidence=event.evidence,
                weights=dict(ctx.severity_weights or {}),
            )
            for record in outcome.issued:
                issued.append(
                    WarningMutation(
                        peer=ctx.peer,
                        rule_id=record.rule_id,
                        episode=ctx.episode,
                        action="issue",
                        severity=record.severity,
                        evidence=record.evidence,
                    )
                )
            if outcome.rule_id:
                warning_events.append(outcome.rule_id)
            if outcome.escalated is not None:
                escalated.append(
                    WarningMutation(
                        peer=ctx.peer,
                        rule_id=outcome.rule_id,
                        episode=ctx.episode,
                        action="escalate",
                        severity=outcome.severity,
                        evidence=outcome.escalated.evidence,
                        penalty=outcome.penalty,
                    )
                )
                warning_penalty += outcome.penalty
                escalated_rules.add(outcome.rule_id)

        known = ctx.known_rules or DEFAULT_KNOWN_RULES
        registry.mark_clean_except(ctx.peer, set(warning_events), known_rules=set(known))

        prepared = ctx.evolve(
            warning_total=warning_penalty,
            warning_events=tuple(dict.fromkeys(warning_events)),
            escalated_rules=frozenset(escalated_rules),
            extras={
                **dict(ctx.extras),
                "warning_mutations": tuple(issued + escalated),
            },
        )
        self._prepared[cache_key] = prepared
        return prepared

    # -- score -----------------------------------------------------------
    def score(self, ctx: ScoreContext) -> ScorerOutput:
        mutations: Sequence[WarningMutation] = tuple(ctx.extras.get("warning_mutations", ()))
        issued = tuple(item for item in mutations if item.action == "issue")
        escalated = tuple(item for item in mutations if item.action == "escalate")
        breakdown = {
            "warning_total": float(ctx.warning_total),
            "warning_events": list(ctx.warning_events),
            "escalated_rules": sorted(ctx.escalated_rules),
            "issued": len(issued),
            "escalated": len(escalated),
        }
        return ScorerOutput(
            kind=self.kind,
            amount=float(ctx.warning_total),
            breakdown=breakdown,
            rules=tuple(dict.fromkeys(ctx.warning_events)),
            mutations=tuple(mutations),
            metadata={
                "escalated_rules": tuple(sorted(ctx.escalated_rules)),
                "warning_events": tuple(ctx.warning_events),
            },
        )

    # -- introspection ---------------------------------------------------
    def validate(self, settings: Mapping[str, Any]) -> list[ScorerDiagnostic]:
        """Report every policy problem, not just the first one ``WarningPolicy`` hits.

        ``WarningPolicy.__post_init__`` raises on the first violation, which is
        right for construction but wrong for an API 422: a caller fixing K_ESC
        should not have to discover the W_CAP problem on the next attempt. So the
        raw values are checked here first, and the policy constructor is only
        used as a final consistency gate.
        """
        diagnostics: list[ScorerDiagnostic] = []
        source = dict(settings.get("WARNINGS", settings) or {})

        parsed: dict[str, Any] = {}
        for key, default in (
            ("K_ESC", 3),
            ("N_WIN", 5),
            ("N_DECAY", 3),
            ("W_CAP", 2),
        ):
            raw = source.get(key, default)
            try:
                number = int(raw)
            except (TypeError, ValueError):
                diagnostics.append(
                    ScorerDiagnostic(
                        level="error",
                        code="warnings.invalid_policy",
                        message=f"{key} is not an integer: {raw!r}",
                    )
                )
                continue
            if int(number) != number:
                diagnostics.append(
                    ScorerDiagnostic(
                        level="error",
                        code="warnings.invalid_policy",
                        message=f"{key} must be a whole number: {raw!r}",
                    )
                )
                continue
            parsed[key] = number

        p_warn = source.get("P_WARN", 5.0)
        try:
            p_warn_value = float(p_warn)
        except (TypeError, ValueError):
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="warnings.invalid_policy",
                    message=f"P_WARN is not numeric: {p_warn!r}",
                )
            )
            p_warn_value = None

        k_esc = parsed.get("K_ESC")
        w_cap = parsed.get("W_CAP")
        if k_esc is not None and k_esc < 2:
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="warnings.k_esc_too_small",
                    message=f"K_ESC must be at least 2, got {k_esc}",
                )
            )
        if k_esc is not None and w_cap is not None and not 1 <= w_cap <= k_esc:
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="warnings.w_cap_out_of_range",
                    message=f"W_CAP={w_cap} must be within [1, K_ESC={k_esc}]",
                )
            )
        windows = [parsed[key] for key in ("N_WIN", "N_DECAY") if key in parsed]
        if any(value < 1 for value in windows):
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="warnings.window_too_small",
                    message=(
                        f"N_WIN={parsed.get('N_WIN', 5)} and "
                        f"N_DECAY={parsed.get('N_DECAY', 3)} must be >= 1"
                    ),
                )
            )
        if p_warn_value is not None and p_warn_value < 0.0:
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="warnings.negative_penalty",
                    message=f"P_WARN must be non-negative, got {p_warn_value}",
                )
            )
        if diagnostics:
            return diagnostics

        try:
            WarningPolicy.from_settings(source)
        except ValueError as exc:
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="warnings.invalid_policy",
                    message=str(exc),
                    hint="check K_ESC, W_CAP, N_WIN and N_DECAY",
                )
            )
        return diagnostics

    def assert_valid(self, settings: Mapping[str, Any] | None = None) -> None:
        problems = [
            diagnostic
            for diagnostic in self.validate(settings or {"WARNINGS": {}})
            if diagnostic.level == "error"
        ]
        if problems:
            raise ImproperlyConfigured(
                "; ".join(f"{item.code}: {item.message}" for item in problems)
            )

    # -- state -----------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Serialize every registry, keyed by node path then peer.

        Structure: ``{"class": ..., "policy": {...}, "registries": {node_path:
        {peer: <registry state>}}}``, so a checkpoint restores warning state
        exactly. Expired records travel with it, which is what keeps the audit
        trail and the derived clean streak intact (document 09 section 3.3).
        """
        snapshots: dict[str, dict[str, Any]] = {}
        for key, registry in self.registries.items():
            if "::" in key:
                node_path, _, peer = key.rpartition("::")
            else:
                node_path, peer = "", key
            state = registry.to_state()
            # One peer per registry, so keep only that peer's slice.
            state = {
                "policy": state["policy"],
                "warnings": {peer: state["warnings"].get(peer, {})},
                "clean_streak": (
                    {peer: state["clean_streak"][peer]}
                    if peer in state["clean_streak"]
                    else {}
                ),
            }
            snapshots.setdefault(node_path, {})[peer] = state
        return {
            "class": type(self).__name__,
            "policy": {
                "k_esc": self.policy.k_esc,
                "n_win": self.policy.n_win,
                "n_decay": self.policy.n_decay,
                "w_cap": self.policy.w_cap,
                "p_warn": self.policy.p_warn,
                "escalate": self.policy.escalate,
            },
            "config": dict(self.config),
            "registries": snapshots,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        policy_payload = payload.get("policy")
        if policy_payload:
            self.policy = WarningPolicy(
                k_esc=int(policy_payload.get("k_esc", 3)),
                n_win=int(policy_payload.get("n_win", 5)),
                n_decay=int(policy_payload.get("n_decay", 3)),
                w_cap=int(policy_payload.get("w_cap", 2)),
                p_warn=float(policy_payload.get("p_warn", 5.0)),
                escalate=bool(policy_payload.get("escalate", True)),
            )
        snapshots = payload.get("registries") or {}
        self.registries.clear()
        self._prepared.clear()
        for node_path, peers in snapshots.items():
            for peer, snapshot in (peers or {}).items():
                registry = self.registry_for_node(node_path, peer)
                registry.restore(snapshot)
