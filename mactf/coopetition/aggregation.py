"""Reward-modulated parameter aggregation under framework.md section 10.

The docx defines ``Theta_new = sum_i omega_i * theta_i`` with
``omega = softmax(kappa * R)``. Taken literally that violates framework.md:

* section 10.1.2/3 require a floor and a cap per peer, which a raw softmax does
  not guarantee;
* section 10.1.4 requires participation gating, so a void episode must not
  contribute;
* section 10.1.6 requires bounded influence, so one outlier cannot dominate;
* section 10.1.7 forbids a peer's own weight influencing its own update.

:class:`Aggregator` therefore post-processes the softmax in a fixed, auditable
order and reports every adjustment it makes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from .. import signals
from ..exceptions import ComplianceError
from .report import AggregationReport
from .adapter import PeerAdapter
from .temperature import AdaptiveKappa


@dataclass
class AggregateResult:
    """The new global parameter vector plus its audit trail."""

    parameters: torch.Tensor
    report: AggregationReport
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)

    def describe(self) -> str:
        return self.report.describe()


class Aggregator:
    """Constrained reward-modulated parameter aggregation."""

    def __init__(
        self,
        *,
        controller: AdaptiveKappa,
        weight_floor: float = 0.02,
        weight_cap: float = 0.60,
        confidence_floor: float = 0.10,
        trim_fraction: float = 0.0,
        disagreement_max: float = 0.75,
        fallback_to_reference: bool = True,
        aggregate_buffers: str = "top",
    ) -> None:
        if weight_floor < 0:
            raise ValueError("weight_floor must be >= 0")
        if not 0 < weight_cap <= 1:
            raise ValueError("weight_cap must be in (0, 1]")
        if weight_floor > weight_cap:
            raise ValueError("weight_floor must be <= weight_cap")
        if not 0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        self.controller = controller
        self.weight_floor = float(weight_floor)
        self.weight_cap = float(weight_cap)
        self.confidence_floor = float(confidence_floor)
        self.trim_fraction = float(trim_fraction)
        self.disagreement_max = float(disagreement_max)
        self.fallback_to_reference = bool(fallback_to_reference)
        self.aggregate_buffers = aggregate_buffers

    # -- cloning (document 17 section 10.7) --------------------------------
    def options(self) -> dict[str, Any]:
        """The constructor keywords other than ``controller``.

        A subclass with extra constructor state overrides this, returning
        ``{**super().options(), "my_arg": self.my_arg}``. ``clone_with`` and ``pristine``
        rely on it, and ``verify_aggregator`` checks it round-trips.
        """
        return {
            "weight_floor": self.weight_floor,
            "weight_cap": self.weight_cap,
            "confidence_floor": self.confidence_floor,
            "trim_fraction": self.trim_fraction,
            "disagreement_max": self.disagreement_max,
            "fallback_to_reference": self.fallback_to_reference,
            "aggregate_buffers": self.aggregate_buffers,
        }

    def clone_with(self, controller: AdaptiveKappa) -> "Aggregator":
        """A sibling for another aggregation group: same class, same bounds, own controller."""
        return type(self)(controller=controller, **self.options())

    def pristine(self) -> "Aggregator":
        """The framework's own ``Aggregator`` with this one's bounds and controller.

        What the guard falls back to when a custom ``compute_weights`` is the thing that
        failed: the fallback must not run the code that just misbehaved. Only the base class's
        options are passed (a subclass's extra ones mean nothing to ``Aggregator``).
        """
        return Aggregator(controller=self.controller, **Aggregator.options(self))

    # -- weight construction ---------------------------------------------
    def compute_weights(
        self,
        names: Sequence[str],
        scores: Sequence[float],
        *,
        eligible: Sequence[bool] | None = None,
        confidence: Sequence[float] | None = None,
        advance: bool = True,
    ) -> AggregationReport:
        """Build compliant peer weights from episode scores.

        Order of operations, each recorded in ``report.adjustments``:

        1. Participation gating (ineligible peers get weight 0).
        2. Confidence gating (low-confidence peers are scaled down).
        3. Softmax over ``kappa * score`` using only eligible peers.
        4. Trimmed-mean outlier removal, if configured.
        5. Floor then cap, with a final renormalization.
        6. Disagreement check with fallback.
        """
        names = list(names)
        if len(names) != len(scores):
            raise ComplianceError("names and scores must be the same length")
        if not names:
            raise ComplianceError("cannot aggregate an empty peer set")

        count = len(names)
        eligible_flags = list(eligible) if eligible is not None else [True] * count
        confidences = list(confidence) if confidence is not None else [1.0] * count

        score_tensor = torch.tensor(scores, dtype=torch.float32)
        active_idx = [index for index in range(count) if eligible_flags[index]]
        if not active_idx:
            report = AggregationReport(
                weights={name: 0.0 for name in names},
                raw_weights={name: 0.0 for name in names},
                kappa=self.controller.kappa,
                entropy=0.0,
                adjustments=["no eligible peers; nothing to aggregate"],
                fallback=True,
                gated_peers=tuple(names),
                order=tuple(names),
            )
            return report

        # Operate on the eligible subset so softmax mass is not wasted on voids.
        subset_scores = score_tensor[active_idx]
        if advance:
            subset_weights = self.controller.step(subset_scores)
        else:
            subset_weights = self.controller.softmax_weights(subset_scores)
        kappa = self.controller.kappa
        entropy = self.controller.entropy if advance else self.controller.entropy_of(subset_weights)

        raw = {name: 0.0 for name in names}
        for local_index, peer_index in enumerate(active_idx):
            raw[names[peer_index]] = float(subset_weights[local_index].item())

        weights = dict(raw)
        adjustments: list[str] = []
        gated = tuple(names[i] for i in range(count) if not eligible_flags[i])
        if gated:
            adjustments.append(f"participation gate zeroed {', '.join(gated)}")

        # Confidence gating scales rather than zeroes, so a partially reliable
        # peer still contributes (framework.md 10.1.5).
        for peer_index in active_idx:
            name = names[peer_index]
            conf = max(self.confidence_floor, min(1.0, float(confidences[peer_index])))
            weights[name] *= conf
        if any(conf < 1.0 for conf in confidences):
            adjustments.append("confidence gate scaled low-confidence peers")

        total = sum(weights[name] for name in weights if names.index(name) in active_idx)
        if total <= 0:
            uniform = 1.0 / len(active_idx)
            for peer_index in active_idx:
                weights[names[peer_index]] = uniform
            adjustments.append("degenerate scores; fell back to uniform weights")
        else:
            for peer_index in active_idx:
                name = names[peer_index]
                weights[name] = weights[name] / total

        # Trimmed mean: drop the single most extreme peer when configured.
        trimmed: tuple[str, ...] = ()
        if self.trim_fraction > 0 and len(active_idx) >= 3:
            drop_count = max(1, int(round(len(active_idx) * self.trim_fraction)))
            ranked = sorted(
                ((names[i], weights[names[i]]) for i in active_idx),
                key=lambda item: item[1],
                reverse=True,
            )
            # Remove the most dominant peer, which is the outlier that matters.
            for name, _ in ranked[:drop_count]:
                weights[name] = 0.0
                trimmed += (name,)
            remaining = [
                names[i] for i in active_idx if weights[names[i]] > 0.0
            ]
            adjustments.append(f"trimmed dominant peer(s) {', '.join(trimmed)}")
            subtotal = sum(weights[name] for name in remaining) or 1.0
            for name in remaining:
                weights[name] = weights[name] / subtotal

        # Floor and cap simultaneously. Renormalizing after a naive cap can push a
        # weight straight back above the cap, so the bounds are enforced by
        # projecting onto the bounded simplex instead.
        # Floor and cap apply to every *participating* peer, not only peers that
        # happened to receive softmax mass. Under an extreme temperature a peer can
        # round to 0.0, and framework.md 10.1.2 still requires it to retain the floor.
        participants = [
            names[i] for i in active_idx if names[i] not in set(trimmed)
        ]
        if participants:
            projected = _project_bounded(
                {name: weights[name] for name in participants},
                floor=self.weight_floor,
                cap=self.weight_cap,
            )
            clamped_low = [
                name for name in participants if weights[name] < self.weight_floor - 1e-9
            ]
            clamped_high = [
                name for name in participants if weights[name] > self.weight_cap + 1e-9
            ]
            for name in participants:
                weights[name] = projected[name]
            if clamped_low:
                adjustments.append(f"applied weight floor {self.weight_floor:g}")
            if clamped_high:
                adjustments.append(f"applied weight cap {self.weight_cap:g}")

        total = sum(weights.values())
        if total <= 0:
            raise ComplianceError("aggregation produced an all-zero weight vector")
        weights = {name: value / total for name, value in weights.items()}
        if any(value > self.weight_cap + 1e-6 for value in weights.values()):
            raise ComplianceError(
                f"WEIGHT_CAP={self.weight_cap:g} is infeasible for {len(participants)} "
                "participating peers; increase the cap or add peers"
            )

        # Disagreement check: only a *material* spread should trigger a fallback.
        # The denominator is floored at 1.0 so that small absolute differences on a
        # low-scoring episode (for example 0.28 vs 0.0) never masquerade as peers
        # contradicting each other.
        spread = float(subset_scores.max() - subset_scores.min())
        scale = max(1.0, float(subset_scores.abs().max()))
        disagreement = spread / scale
        fallback = disagreement > self.disagreement_max
        if fallback:
            adjustments.append(
                f"peer disagreement {disagreement:.3f} exceeded {self.disagreement_max:g}"
            )

        report = AggregationReport(
            weights=weights,
            raw_weights=raw,
            kappa=kappa,
            entropy=entropy,
            adjustments=adjustments,
            fallback=fallback and self.fallback_to_reference,
            gated_peers=gated,
            trimmed_peers=trimmed,
            order=tuple(names),
        )
        return report

    # -- parameter aggregation -------------------------------------------
    def aggregate(
        self,
        adapters: dict[str, PeerAdapter],
        report: AggregationReport,
        *,
        reference: PeerAdapter | None = None,
        device: torch.device | str = "cpu",
    ) -> AggregateResult:
        """Compute the weighted parameter sum described by ``report``."""
        if report.fallback:
            if reference is None:
                raise ComplianceError(
                    "aggregation fell back to the reference policy but no reference was supplied"
                )
            params = reference.export_parameters().to(device=device)
            signals.integrity_incident.send(
                sender=self,
                reason="aggregation fallback to reference policy",
                report=report,
            )
            return AggregateResult(parameters=params, report=report)

        vectors: list[tuple[float, torch.Tensor]] = []
        for peer_name, weight in report.weights.items():
            if weight <= 0.0:
                continue
            adapter = adapters.get(peer_name)
            if adapter is None:
                raise ComplianceError(f"no adapter registered for peer {peer_name!r}")
            vectors.append((weight, adapter.export_parameters().to(device=device)))

        if not vectors:
            raise ComplianceError("no peers with positive weight to aggregate")

        lengths = {vector.numel() for _, vector in vectors}
        if len(lengths) != 1:
            raise ComplianceError(
                "peer parameter vectors differ in length; all peers must share a "
                f"parameter layout (saw lengths {sorted(lengths)})"
            )

        accumulator = torch.zeros_like(vectors[0][1])
        for weight, vector in vectors:
            accumulator = accumulator + weight * vector

        buffers: dict[str, torch.Tensor] = {}
        if self.aggregate_buffers == "top" and report.weights:
            top_peer = max(report.weights.items(), key=lambda item: item[1])[0]
            adapter = adapters.get(top_peer)
            if adapter is not None:
                for name, buffer in adapter.module.named_buffers():
                    buffers[name] = buffer.detach().clone()
        return AggregateResult(parameters=accumulator, report=report, buffers=buffers)

    def apply(
        self,
        adapters: dict[str, PeerAdapter],
        result: AggregateResult,
        *,
        reference: PeerAdapter | None = None,
        reference_ema: float = 0.05,
        reset_optimizers: bool = True,
        optimizers: dict[str, Any] | None = None,
    ) -> None:
        """Write aggregated parameters into every peer and update the reference."""
        for peer_name, adapter in adapters.items():
            adapter.import_parameters(result.parameters)
            for buffer_name, buffer in result.buffers.items():
                target = dict(adapter.module.named_buffers()).get(buffer_name)
                if target is not None and target.shape == buffer.shape:
                    target.copy_(buffer)
            if reset_optimizers and optimizers and peer_name in optimizers:
                # Momentum buffers were computed for the pre-aggregation weights.
                optimizers[peer_name].state.clear()

        if reference is not None and reference_ema > 0:
            current = reference.export_parameters()
            blended = (1.0 - reference_ema) * current + reference_ema * result.parameters.to(
                current.device
            )
            reference.import_parameters(blended)

        signals.aggregation_completed.send(sender=self, result=result, adapters=adapters)

    # -- factory ---------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: dict[str, Any], *, num_peers: int) -> "Aggregator":
        controller = AdaptiveKappa.from_settings(settings, num_peers=num_peers)
        return cls(
            controller=controller,
            weight_floor=float(settings.get("WEIGHT_FLOOR", 0.02)),
            weight_cap=float(settings.get("WEIGHT_CAP", 0.60)),
            confidence_floor=float(settings.get("CONFIDENCE_FLOOR", 0.10)),
            trim_fraction=float(settings.get("TRIM_FRACTION", 0.0)),
            disagreement_max=float(settings.get("DISAGREEMENT_MAX", 0.75)),
            fallback_to_reference=bool(settings.get("FALLBACK_TO_REFERENCE", True)),
            aggregate_buffers=str(settings.get("AGGREGATE_BUFFERS", "top")),
        )


def feasible_bounds(num_peers: int, *, floor: float, cap: float) -> tuple[float, float]:
    """Check that floor/cap admit a normalized weight vector for ``num_peers``.

    A floor above ``1/N`` or a cap below ``1/N`` makes normalization impossible.
    """
    target = 1.0 / max(1, num_peers)
    if floor > target:
        raise ComplianceError(
            f"WEIGHT_FLOOR={floor:g} exceeds 1/{num_peers}={target:g}; "
            "the floor would force an overweighted sum"
        )
    if cap < target:
        raise ComplianceError(
            f"WEIGHT_CAP={cap:g} is below 1/{num_peers}={target:g}; "
            "the cap would prevent a valid normalized vector"
        )
    return floor, cap


def _project_bounded(
    weights: dict[str, float], *, floor: float, cap: float, iterations: int = 64
) -> dict[str, float]:
    """Project weights onto ``{w : sum(w)=1, floor <= w <= cap}``.

    Bisection on the dual variable is used because the naive "clamp then
    normalize" cycle can violate the cap again. This keeps both bounds exact,
    which framework.md 10.1.2/3 requires.
    """
    names = list(weights)
    if not names:
        return {}
    if floor * len(names) > 1.0 + 1e-9:
        raise ComplianceError(
            f"floor {floor:g} x {len(names)} peers exceeds 1.0; no normalized vector exists"
        )
    if cap * len(names) < 1.0 - 1e-9:
        raise ComplianceError(
            f"cap {cap:g} x {len(names)} peers is below 1.0; no normalized vector exists"
        )

    def clamp(value: float, shift: float) -> float:
        return min(cap, max(floor, value + shift))

    low, high = -10.0, 10.0
    for _ in range(iterations):
        mid = (low + high) / 2.0
        total = sum(clamp(weights[name], mid) for name in names)
        if total > 1.0:
            high = mid
        else:
            low = mid
    shift = (low + high) / 2.0
    projected = {name: clamp(weights[name], shift) for name in names}
    total = sum(projected.values())
    if total and abs(total - 1.0) > 1e-6:
        # Residual from the box constraint; distribute it where room exists.
        residual = 1.0 - total
        adjustable = [
            name
            for name in names
            if (residual > 0 and projected[name] < cap)
            or (residual < 0 and projected[name] > floor)
        ]
        if adjustable:
            share = residual / len(adjustable)
            for name in adjustable:
                projected[name] = min(cap, max(floor, projected[name] + share))
    return projected


def build_aggregator(
    spec: Any,
    coopetition: dict[str, Any],
    *,
    num_peers: int,
    resolver: Any | None = None,
    trusted: bool = True,
    record: dict[str, Any] | None = None,
) -> Aggregator:
    """``TRAINER.AGGREGATOR`` to an ``Aggregator``.

    ``record``, when given, receives the resolved component under ``"resolved"`` so the
    trainer can put it in the checkpoint's component manifest (document 17 section 14).

    ``None`` is the built-in, built exactly as ``Aggregator.from_settings`` does. A name or
    mapping goes through the ``aggregator`` registry with the controller and every bound
    (``weight_floor``, ``weight_cap``, ...) *injected*, so a spec carries only the class's own
    extra parameters and can never loosen a bound (invariant A3).
    """
    from ..components.resolver import ComponentResolver, ComponentSpec

    if isinstance(spec, Aggregator):
        return spec
    if spec is None:
        return Aggregator.from_settings(dict(coopetition), num_peers=num_peers)
    controller = AdaptiveKappa.from_settings(dict(coopetition), num_peers=num_peers)
    injected = {
        "controller": controller,
        "weight_floor": float(coopetition.get("WEIGHT_FLOOR", 0.02)),
        "weight_cap": float(coopetition.get("WEIGHT_CAP", 0.60)),
        "confidence_floor": float(coopetition.get("CONFIDENCE_FLOOR", 0.10)),
        "trim_fraction": float(coopetition.get("TRIM_FRACTION", 0.0)),
        "disagreement_max": float(coopetition.get("DISAGREEMENT_MAX", 0.75)),
        "fallback_to_reference": bool(coopetition.get("FALLBACK_TO_REFERENCE", True)),
        "aggregate_buffers": str(coopetition.get("AGGREGATE_BUFFERS", "top")),
    }
    parsed = ComponentSpec.parse("aggregator", spec)
    resolved = (resolver or ComponentResolver()).resolve(parsed, trusted=trusted, inject=injected)
    built = resolved.instance
    verify_aggregator(built)
    if record is not None:
        record["resolved"] = resolved
    return built


def verify_aggregator(aggregator: Aggregator) -> None:
    """``E065`` unless ``clone_with`` returns the same class with the same options."""
    from ..exceptions import ImproperlyConfigured

    probe = AdaptiveKappa(num_peers=2)
    try:
        sibling = aggregator.clone_with(probe)
    except Exception as error:  # noqa: BLE001 - reported with the cure
        raise ImproperlyConfigured(
            f"E065: {type(aggregator).__name__}.clone_with() failed ({error}); a subclass with "
            "extra constructor arguments must override options() to return them"
        ) from error
    if type(sibling) is not type(aggregator) or sibling.options() != aggregator.options():
        raise ImproperlyConfigured(
            f"E065: {type(aggregator).__name__}.clone_with() does not reproduce the class and "
            f"options (got {type(sibling).__name__} / {sibling.options()!r}); override options()"
        )
    if sibling.controller is not probe:
        raise ImproperlyConfigured(
            f"E065: {type(aggregator).__name__}.clone_with() must use the controller it is given"
        )


from ..components.registry import HUB  # noqa: E402

HUB.registry("aggregator").register_builtin("default", Aggregator)
