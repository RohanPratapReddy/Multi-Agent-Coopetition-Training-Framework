"""The sigma reduction (document 14 section 6.5 and 6.6).

``sigma_i(j)`` answers "how much should peer ``i`` listen to teacher ``j`` this
step". Trust (from conformance) is a separate multiplier that is applied before
normalisation and before the bounded projection, so a discredited teacher is
removed from the simplex rather than shrunk.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..coopetition.aggregation import _project_bounded
from ..exceptions import ComplianceError, ContractError, ValidationError

__all__ = ["QUALITY_UPDATE", "SIGMA_MODES", "SigmaState", "sigma_weights"]

SIGMA_MODES: frozenset[str] = frozenset({"uniform", "teacher_softmax", "quality_softmax"})

#: ``SigmaState.update`` consumes ``-L_ij``; named so a caller cannot flip the sign.
QUALITY_UPDATE: str = "negative_loss"


def sigma_weights(
    losses: Mapping[str, float],
    *,
    mode: str = "quality_softmax",
    beta: float = 1.0,
    priors: Mapping[str, float] | None = None,
    quality: Mapping[str, float] | None = None,
    trust: Mapping[str, float] | None = None,
    trust_min: float = 0.0,
    floor: float = 0.0,
    cap: float = 1.0,
) -> dict[str, float]:
    """Teacher weights for one peer, summing to 1.

    ``losses`` maps teacher name -> that teacher's current ``L_ij`` for this peer.
    All iteration is over ``sorted()`` names, so the result is bit-identical
    regardless of mapping order.

    Modes:

    ``uniform``          ``sigma_j`` proportional to ``prior_j * trust_j``.
    ``teacher_softmax``  ``sigma_j`` proportional to ``prior_j * trust_j *
                         exp(-beta * L_j)``.
    ``quality_softmax``  ``sigma_j`` proportional to ``prior_j * trust_j *
                         exp(beta * q_j)`` where ``q_j`` is the pair's EMA of
                         ``-L_ij``; slower to react and immune to one noisy batch.

    A teacher whose trust is at or below ``trust_min`` is removed before
    normalisation, so survivors renormalise over a smaller set. The test is a
    strict ``>``, so a trust of exactly zero is never live. Trust is applied
    *before* ``_project_sigma``; applied after, a non-zero ``floor`` would
    resurrect a discredited teacher.

    When every teacher is excluded the result is ``{}``. That is not an error: the
    caller treats an empty sigma as ``L_cd = 0`` for this peer and the step
    proceeds on ``L_ind`` alone.
    """
    if mode not in SIGMA_MODES:
        raise ValidationError(
            f"sigma mode {mode!r} is unknown; accepted: " + ", ".join(sorted(SIGMA_MODES))
        )
    names = sorted(losses)
    if not names:
        raise ValidationError("sigma_weights requires at least one teacher")
    if not 0.0 <= float(floor) <= float(cap):
        raise ValidationError(f"sigma floor {floor!r} must be in [0, cap={cap!r}]")
    if float(cap) > 1.0:
        raise ValidationError(f"sigma cap must be <= 1, got {cap!r}")
    if not 0.0 <= float(trust_min) <= 1.0:
        raise ValidationError(f"sigma trust_min must be in [0, 1], got {trust_min!r}")
    if float(beta) < 0.0:
        raise ValidationError(f"sigma beta must be >= 0, got {beta!r}")

    resolved_priors = {name: float((priors or {}).get(name, 1.0)) for name in names}
    for name, prior in resolved_priors.items():
        if not math.isfinite(prior) or prior <= 0.0:
            raise ValidationError(f"teacher {name!r} prior must be finite and > 0, got {prior!r}")

    resolved_trust: dict[str, float] = {}
    for name in names:
        value = float((trust or {}).get(name, 1.0))
        if not math.isfinite(value):
            raise ContractError(f"teacher {name!r} trust is not finite: {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValidationError(f"teacher {name!r} trust must be in [0, 1], got {value!r}")
        resolved_trust[name] = value

    # Drop untrusted teachers from the box before anything is normalised. This is
    # the ordering that makes TRUST_MIN and SIGMA_FLOOR compose instead of fight.
    live = [name for name in names if resolved_trust[name] > float(trust_min)]
    if not live:
        return {}

    if mode == "uniform":
        raw = {name: resolved_priors[name] * resolved_trust[name] for name in live}
    else:
        scores = (
            {name: -float(losses[name]) for name in names}
            if mode == "teacher_softmax"
            else {name: float((quality or {}).get(name, 0.0)) for name in names}
        )
        for name in live:
            if not math.isfinite(scores[name]):
                raise ContractError(
                    f"sigma score for teacher {name!r} is not finite: {scores[name]!r}"
                )
        # Shift by the maximum over live teachers: softmax is shift-invariant, and
        # without the shift exp(beta * q) overflows for a large beta * q.
        ceiling = max(scores[name] for name in live)
        raw = {
            name: resolved_priors[name]
            * resolved_trust[name]
            * math.exp(float(beta) * (scores[name] - ceiling))
            for name in live
        }

    total = math.fsum(raw[name] for name in live)
    if not math.isfinite(total) or total <= 0.0:
        # A degenerate prior set is a configuration error, not a reason to fall
        # back silently: a uniform fallback would hide a zeroed teacher.
        raise ValidationError("sigma_weights produced a non-positive total; check teacher priors and beta")
    weights = {name: raw[name] / total for name in live}
    return _project_sigma(weights, floor=float(floor), cap=float(cap))


def _project_sigma(weights: Mapping[str, float], *, floor: float, cap: float) -> dict[str, float]:
    """Project onto ``{s : sum(s) = 1, floor <= s <= cap}``.

    The arithmetic is delegated to ``mactf.coopetition.aggregation._project_bounded``
    so a sigma floor of ``0.05`` and a peer ``WEIGHT_FLOOR`` of ``0.05`` mean
    exactly the same thing. An infeasible box raises there; it is translated here
    with a message naming the teacher count, because the caller configured
    teachers, not peers.
    """
    names = sorted(weights)
    if not names:
        return {}
    if floor <= 0.0 and cap >= 1.0:
        total = math.fsum(weights[name] for name in names)
        return {name: weights[name] / total for name in names}
    try:
        projected = _project_bounded({name: float(weights[name]) for name in names}, floor=floor, cap=cap)
    except ComplianceError as exc:
        raise ValidationError(
            f"DISTILLATION.SIGMA_FLOOR={floor:g} / SIGMA_CAP={cap:g} is infeasible "
            f"for {len(names)} teachers: {exc}"
        ) from exc
    total = math.fsum(projected[name] for name in names)
    if total <= 0.0:
        raise ComplianceError(
            "sigma projection produced an all-zero vector; this cannot happen for a "
            "feasible box and indicates a bug in _project_bounded"
        )
    return {name: projected[name] / total for name in names}


class SigmaState:
    """Per-pair running quality: the ``q_j`` that ``quality_softmax`` reads.

    ``q_j`` is an EMA of ``-L_ij``, so higher means "this teacher has recently
    been easier for this peer to match". It is not an absolute measure of teacher
    quality and is never compared across peers.
    """

    def __init__(self, *, decay: float = 0.9, init: float = 0.0) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValidationError(f"SigmaState decay must be in [0, 1), got {decay!r}")
        self.decay = float(decay)
        self._init = float(init)
        self._values: dict[tuple[str, str], float] = {}
        self._updates: dict[tuple[str, str], int] = {}

    def get(self, peer: str, teacher: str) -> float:
        return self._values.get((peer, teacher), self._init)

    def update(self, peer: str, teacher: str, loss: float) -> float:
        if not math.isfinite(float(loss)):
            raise ContractError(
                f"refusing to record a non-finite contrastive loss for ({peer}, {teacher}): {loss!r}"
            )
        key = (peer, teacher)
        previous = self._values.get(key, self._init)
        value = self.decay * previous + (1.0 - self.decay) * (-float(loss))
        self._values[key] = value
        self._updates[key] = self._updates.get(key, 0) + 1
        return value

    def updates(self, peer: str, teacher: str) -> int:
        return self._updates.get((peer, teacher), 0)

    def for_peer(self, peer: str, teachers: Sequence[str]) -> dict[str, float]:
        return {name: self.get(peer, name) for name in sorted(teachers)}

    def describe(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "pairs": {
                f"{peer}|{teacher}": round(value, 6)
                for (peer, teacher), value in sorted(self._values.items())
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "init": self._init,
            "values": {
                f"{peer}|{teacher}": value for (peer, teacher), value in sorted(self._values.items())
            },
            "updates": {
                f"{peer}|{teacher}": count for (peer, teacher), count in sorted(self._updates.items())
            },
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.decay = float(payload.get("decay", self.decay))
        self._init = float(payload.get("init", self._init))
        values: dict[tuple[str, str], float] = {}
        for key, value in dict(payload.get("values") or {}).items():
            peer, separator, teacher = str(key).partition("|")
            if not separator or not peer or not teacher:
                raise ValidationError(f"SigmaState key {key!r} is not 'peer|teacher'")
            values[(peer, teacher)] = float(value)
        self._values = values
        updates: dict[tuple[str, str], int] = {}
        for key, count in dict(payload.get("updates") or {}).items():
            peer, separator, teacher = str(key).partition("|")
            if not separator or not peer or not teacher:
                raise ValidationError(f"SigmaState update key {key!r} is not 'peer|teacher'")
            updates[(peer, teacher)] = int(count)
        self._updates = updates
