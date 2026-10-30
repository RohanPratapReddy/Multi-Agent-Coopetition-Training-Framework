"""The compound objective (document 14 section 7).

``L_i = l_coop * L_coop,i + l_ind * L_ind,i + l_reg * L_reg,i
        + l_cd * ramp * sum_j sigma_ij * L_ij + m * margin_i``

Every term is a tensor built from the peer's own parameters, a frozen teacher
output, or a detached scalar. This module owns the arithmetic that joins them and
nothing else: ``L_ind`` is delegated to the configured ``LossFunction``, ``L_ij``
to :func:`~mactf.distill.contrastive.contrastive_loss`, and ``sigma`` to
:func:`~mactf.distill.sigma.sigma_weights`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ..exceptions import ComplianceError, ContractError, ValidationError
from ..rl.loss import LossBreakdown
from ..rl.policy import PolicyOutput
from .contrastive import ContrastiveReport

__all__ = [
    "IND_REQUIRED",
    "ObjectiveConfig",
    "PeerLoss",
    "build_regularizers",
    "cohort_signal",
    "compound_loss",
    "cooperative_reward_term",
    "cooperative_term",
    "individual_term",
    "ramp_factor",
    "regularization_term",
]


@dataclass(frozen=True)
class ObjectiveConfig:
    """Coefficients and schedule of the compound objective."""

    lambda_coop: float = 0.0
    lambda_ind: float = 1.0
    lambda_reg: float = 0.0
    lambda_cd: float = 1.0
    peer_margin: float = 0.25
    tau: float = 0.07
    tau_min: float = 0.01
    tau_max: float = 1.0
    beta: float = 1.0
    sigma_mode: str = "quality_softmax"
    symmetric: bool = True
    detach_positive: bool = True
    warmup_steps: int = 0
    ramp_steps: int = 0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("lambda_coop", "lambda_ind", "lambda_reg", "lambda_cd"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValidationError(f"{name} must be a finite number >= 0, got {value!r}")
        if self.lambda_ind == 0.0 and self.lambda_cd == 0.0 and self.lambda_coop == 0.0:
            raise ValidationError(
                "all of lambda_coop, lambda_ind, and lambda_cd are zero; the objective "
                "would train nothing but the regularizer"
            )
        if not 0.0 <= float(self.peer_margin) <= 1.0:
            raise ValidationError(f"peer_margin must be in [0, 1], got {self.peer_margin!r}")
        if not float(self.tau) > 0.0:
            raise ValidationError(f"tau must be > 0, got {self.tau!r}")
        if not float(self.tau_min) > 0.0:
            raise ValidationError(f"tau_min must be > 0, got {self.tau_min!r}")
        if float(self.tau_max) < float(self.tau_min):
            raise ValidationError(
                f"tau_max {self.tau_max!r} must be >= tau_min {self.tau_min!r}"
            )
        if float(self.beta) < 0.0:
            raise ValidationError(f"beta must be >= 0, got {self.beta!r}")
        if not float(self.eps) > 0.0:
            raise ValidationError(f"eps must be > 0, got {self.eps!r}")
        if int(self.warmup_steps) < 0 or int(self.ramp_steps) < 0:
            raise ValidationError("warmup_steps and ramp_steps must be >= 0")


# ---------------------------------------------------------------------------
# L_ind
# ---------------------------------------------------------------------------
IND_REQUIRED = frozenset(
    {"policy", "reference", "actions", "advantages", "values", "value_targets"}
)


def individual_term(
    loss_fn: Any,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, LossBreakdown]:
    """Delegate to the configured ``LossFunction``. No arithmetic here.

    ``compliance_floor`` and ``void_mask`` are optional in ``CompositeLoss``: it
    adds a ``ComplianceFloorLoss`` when the former is present and applies
    ``VoidMaskLoss`` when the latter is. ``local_update`` passes both, so the
    distilled path passes both too and the disabled-equivalence gate compares like
    with like.
    """
    missing = sorted(key for key in loss_fn.required_fields if key not in batch)
    if missing:
        raise ContractError(
            "the individual loss is missing required field(s): " + ", ".join(missing)
        )
    total, breakdown = loss_fn.compute(batch)
    if not torch.is_tensor(total):
        raise ContractError(
            f"LossFunction {type(loss_fn).__name__} returned "
            f"{type(total).__name__}, expected a tensor"
        )
    return total, breakdown


# ---------------------------------------------------------------------------
# L_coop
# ---------------------------------------------------------------------------
def cohort_signal(
    signals: Mapping[str, float],
    *,
    eligible: Mapping[str, bool] | None = None,
) -> dict[str, float]:
    """Zero-sum centring of the epoch signals over the eligible peers.

    ``sum(out.values()) == 0`` up to float error, which is what makes the
    competitive half strictly zero-sum: no peer can gain without another losing.

    An ineligible (void) peer is pinned to ``0.0`` rather than centred: it did not
    compete, so giving it the cohort's negative mean would be a benefit it did not
    earn, and giving it the positive mean would be a penalty for a peer that
    already forfeited its credit.
    """
    names = sorted(signals)
    if not names:
        return {}
    flags = {name: bool((eligible or {}).get(name, True)) for name in names}
    active = [name for name in names if flags[name]]
    if not active:
        return {name: 0.0 for name in names}
    mean = math.fsum(float(signals[name]) for name in active) / len(active)
    out = {name: 0.0 for name in names}
    for name in active:
        out[name] = float(signals[name]) - mean
    return out


def cooperative_term(
    policy: PolicyOutput,
    *,
    advantage: torch.Tensor,
    centring: float,
    coefficient: float,
    actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """``-coefficient * centring * mean(advantage * log pi(a))``, advantages detached.

    ``advantage`` is the peer's per-sample advantage already produced by the
    individual path, and ``centring`` is ``c_i`` from :func:`cohort_signal`. The
    advantage is detached because it is a reward: the gradient flows through the
    likelihood, not through the reward estimate (the convention ``policy_loss``
    uses).

    ``actions`` selects ``log pi(a_b | s_b)`` per sample, which is the quantity the
    formula names. The step passes the same actions the individual term used. When
    it is omitted the per-sample likelihood is the mean log-probability over the
    action axis, so a large action space does not swamp the term.

    The term is unclipped by construction. The individual policy term is clipped
    against a reference ratio because it is an importance-weighted surrogate; this
    one is a cohort margin on the likelihood itself, so ``clip_ratio`` does not
    apply.

    With ``coefficient == 0`` or ``centring == 0`` the returned tensor is exactly
    zero and its gradient is exactly zero, so the disabled case is a true no-op.
    """
    log_probs = policy.log_probs
    advantage = advantage.detach().reshape(-1)
    if log_probs.shape[0] != advantage.numel():
        raise ContractError(
            f"cooperative_term: log-probs have {log_probs.shape[0]} rows but the "
            f"advantage has {advantage.numel()} entries"
        )
    scale = -float(coefficient) * float(centring)
    if scale == 0.0:
        # Attached to the graph at zero cost, so the caller can add it to a loss
        # that is later back-propagated without a special case.
        return log_probs.sum() * 0.0
    if actions is None:
        likelihood = log_probs.mean(dim=-1)
    else:
        likelihood = policy.gather(actions)
    per_sample = (likelihood * advantage.to(likelihood.dtype)).mean()
    return scale * per_sample


def cooperative_reward_term(*, awarded: float, coefficient: float) -> torch.Tensor:
    """``-coefficient * awarded``: the pool's grant as a fixed reward.

    A constant has zero gradient, so this term exists for reporting and for a
    caller that wants the objective's value to match the scorecard's accounting.
    The gradient for R10/R11/R16 reaches the parameters through the same
    policy-gradient path as any other reward, via ``advantage`` in the individual
    term; making this term trainable would reward the same event twice.
    """
    return torch.zeros((), dtype=torch.float32) - float(coefficient) * float(awarded)


# ---------------------------------------------------------------------------
# L_reg
# ---------------------------------------------------------------------------
def build_regularizers(
    specs: Sequence[Any] | None,
    *,
    peers: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
    references: Mapping[str, Any] | None = None,
) -> dict[str, list[Any]]:
    """``peer path -> [regularizer, ...]``. Each spec is built per peer.

    A ``kl_to_reference`` regularizer holds a reference *module*, so it cannot be
    shared across peers: peer A's reference is not peer B's. Building one per peer
    is therefore required, not an optimization.

    The reference module is the peer's *frozen reference* (``references[name]``)
    when one is supplied. Pointing the regularizer at the peer's own live module
    would compare a module with itself: the value and the gradient would both be
    exactly zero and the term would be decoration. The live module is used only
    when no reference exists, which makes the term inert rather than wrong, and
    ``settings["strict"]`` turns that case into an error.
    """
    from ..rl.functions.registry import build_function

    strict = bool((settings or {}).get("strict", False))
    out: dict[str, list[Any]] = {name: [] for name in sorted(peers)}
    for raw in specs or ():
        spec = dict(raw) if isinstance(raw, Mapping) else {"type": raw}
        kind = str(spec.get("type", ""))
        for name in sorted(peers):
            payload = dict(spec)
            if kind == "kl_to_reference" and "reference" not in payload:
                reference = (references or {}).get(name)
                if reference is None:
                    if strict:
                        raise ContractError(
                            f"kl_to_reference for peer {name!r} needs a frozen reference "
                            "and none was supplied"
                        )
                    module = peers[name].module
                else:
                    module = getattr(reference, "module", reference)
                payload["reference"] = module
            out[name].append(build_function(payload, family="regularization"))
    return out


def regularization_term(
    regularizers: Sequence[Any],
    module: torch.nn.Module,
    *,
    states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum the configured regularizers for one peer, with per-term values.

    ``states`` is forwarded to a regularizer that evaluates the module on data
    (``kl_to_reference``); parameter-space regularizers do not take it.
    """
    total = None
    values: dict[str, float] = {}
    for index, regularizer in enumerate(regularizers):
        wants_states = getattr(regularizer, "spec_type", "") == "kl_to_reference"
        if wants_states and states is not None:
            component = regularizer(module, states=states)
        else:
            component = regularizer(module)
        if not torch.is_tensor(component) or component.numel() != 1:
            count = component.numel() if torch.is_tensor(component) else "?"
            raise ContractError(
                f"regularizer #{index} ({type(regularizer).__name__}) returned "
                f"{type(component).__name__} with {count} element(s); a "
                "RegularizationFunction must return a scalar tensor"
            )
        component = component.reshape(())
        values[f"{type(regularizer).__name__.lower()}:{index}"] = float(
            component.detach().item()
        )
        total = component if total is None else total + component
    if total is None:
        return torch.zeros((), dtype=torch.float32), {}
    return total, values


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
@dataclass
class PeerLoss:
    """One peer's compound loss and the full derivation behind it."""

    peer: str
    total: torch.Tensor
    ind: torch.Tensor
    coop: torch.Tensor
    reg: torch.Tensor
    cd: torch.Tensor
    sigma: dict[str, float]
    per_teacher: dict[str, float]
    breakdown: LossBreakdown
    cd_reports: dict[str, ContrastiveReport]
    margin: torch.Tensor
    lambda_cd_effective: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "peer": self.peer,
            "total": float(self.total.detach().item()),
            "ind": float(self.ind.detach().item()),
            "coop": float(self.coop.detach().item()),
            "reg": float(self.reg.detach().item()),
            "cd": float(self.cd.detach().item()),
            "margin": float(self.margin.detach().item()),
            "sigma": dict(self.sigma),
            "per_teacher": dict(self.per_teacher),
            "lambda_cd_effective": self.lambda_cd_effective,
        }


def compound_loss(
    *,
    peer: str,
    config: ObjectiveConfig,
    individual: torch.Tensor,
    cooperative: torch.Tensor,
    regularization: torch.Tensor,
    contrastive: Mapping[str, torch.Tensor],
    sigma: Mapping[str, float],
    margin: torch.Tensor,
    beta: float,
    step: int,
    breakdown: LossBreakdown | None = None,
    reports: Mapping[str, ContrastiveReport] | None = None,
) -> PeerLoss:
    """``L = l_coop*coop + l_ind*ind + l_reg*reg + l_cd*ramp*sum_j sigma_j*L_j + m*margin``.

    ``beta`` is the ramp factor from :func:`ramp_factor`: ``lambda_cd`` is scaled by
    it, and no other coefficient is. Distillation ramps in; the individual loss
    never ramps out, because a run that starts by training nothing and then
    switches objectives is not the same run as one that adds a term. The peer
    margin is applied outside the ramp by design; a project that wants it ramped
    leaves ``peer_margin`` at ``0`` and relies on ``lambda_cd``.
    """
    names = sorted(contrastive)
    if names != sorted(sigma):
        raise ContractError(
            f"contrastive teachers {names} and sigma keys {sorted(sigma)} disagree; "
            "a teacher with a loss and no weight (or the reverse) is a bug"
        )
    total_sigma = math.fsum(float(sigma[name]) for name in names)
    if names and abs(total_sigma - 1.0) > 1e-6:
        raise ContractError(
            f"sigma for peer {peer!r} sums to {total_sigma!r}, not 1; "
            "sigma_weights must be used to build it"
        )
    weighted = torch.zeros((), device=individual.device, dtype=individual.dtype)
    for name in names:
        weight = float(sigma[name])
        if weight == 0.0:
            continue
        term = contrastive[name]
        if not torch.is_tensor(term):
            raise ContractError(
                f"contrastive loss for ({peer}, {name}) is "
                f"{type(term).__name__}, expected a tensor"
            )
        weighted = weighted + weight * term.to(dtype=weighted.dtype)
    lambda_cd = config.lambda_cd * float(beta)
    total = (
        config.lambda_coop * cooperative
        + config.lambda_ind * individual
        + config.lambda_reg * regularization.to(device=individual.device)
        + lambda_cd * weighted
        + config.peer_margin * margin.to(device=individual.device)
    )
    if not bool(torch.isfinite(total)):
        raise ComplianceError(
            f"compound loss for peer {peer!r} is not finite at step {step}; "
            "check tau, sigma, and the regularizer coefficients"
        )
    composed = breakdown if breakdown is not None else LossBreakdown()
    composed.total = float(total.detach().item())
    composed.extras.update(
        {
            "coop": float(cooperative.detach().item()),
            "reg": float(regularization.detach().item()),
            "cd": float(weighted.detach().item()),
            "cd_weighted": float((lambda_cd * weighted).detach().item()),
            "peer_margin": float(margin.detach().item()),
            "lambda_cd_effective": float(lambda_cd),
            "sigma_entropy": _sigma_entropy(sigma),
        }
    )
    for name in names:
        composed.extras[f"cd.{name}"] = float(contrastive[name].detach().item())
        composed.extras[f"sigma.{name}"] = float(sigma[name])
    return PeerLoss(
        peer=peer,
        total=total,
        ind=individual,
        coop=cooperative,
        reg=regularization,
        cd=weighted,
        sigma={name: float(sigma[name]) for name in names},
        per_teacher={name: float(contrastive[name].detach().item()) for name in names},
        breakdown=composed,
        cd_reports=dict(reports or {}),
        margin=margin,
        lambda_cd_effective=float(lambda_cd),
    )


def _sigma_entropy(sigma: Mapping[str, float]) -> float:
    total = 0.0
    for name in sorted(sigma):
        value = float(sigma[name])
        if value > 0.0:
            total -= value * math.log(value)
    return float(total)


def ramp_factor(step: int, *, warmup_steps: int, ramp_steps: int) -> float:
    """0 during warmup, then a linear ramp to 1, then 1.

    ``warmup_steps`` is a hard zero: the first N steps train on the individual
    objective only, which lets the peer's representation become non-degenerate
    before it is compared to anything.
    """
    if step < 0:
        raise ValidationError(f"step must be >= 0, got {step!r}")
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    progress = (step - warmup_steps) / float(ramp_steps)
    return float(min(1.0, max(0.0, progress)))
