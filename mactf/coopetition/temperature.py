"""Adaptive temperature (kappa) with entropy-driven momentum.

Implements docx Turn 5::

    omega_i,t = softmax(kappa_t * R_i,t)
    H_t       = -sum_i omega_i,t * ln(omega_i,t)
    Delta_t   = H_t - H_target
    m_t       = beta * m_(t-1) + (1 - beta) * Delta_t
    kappa_t+1 = max(kappa_min, kappa_t + eta_kappa * m_t)

``KAPPA_MAX`` is added as an explicit ceiling so a runaway positive momentum
cannot make the softmax numerically degenerate (one peer at weight 1.0). That
bound exists because framework.md section 10 requires a per-peer floor and cap;
without it the docx temperature alone could violate them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class TemperatureState:
    """Observable state of the adaptive temperature controller."""

    kappa: float
    momentum: float = 0.0
    entropy: float = 0.0
    target_entropy: float = 0.0
    delta: float = 0.0
    step: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"kappa={self.kappa:.4f} H={self.entropy:.4f} "
            f"H_target={self.target_entropy:.4f} delta={self.delta:+.4f} "
            f"m={self.momentum:+.4f}"
        )


class AdaptiveKappa:
    """Entropy-targeting controller for the softmax temperature."""

    def __init__(
        self,
        *,
        num_peers: int,
        kappa_init: float = 1.0,
        kappa_min: float = 0.1,
        kappa_max: float = 10.0,
        target_entropy_ratio: float = 0.8,
        beta: float = 0.9,
        eta_kappa: float = 0.05,
        keep_history: int = 256,
    ) -> None:
        if num_peers < 1:
            raise ValueError("num_peers must be >= 1")
        if kappa_min <= 0:
            raise ValueError("kappa_min must be strictly positive")
        if kappa_max < kappa_min:
            raise ValueError("kappa_max must be >= kappa_min")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if not 0.0 < target_entropy_ratio <= 1.0:
            raise ValueError("target_entropy_ratio must be in (0, 1]")

        self.num_peers = num_peers
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.beta = float(beta)
        self.eta_kappa = float(eta_kappa)
        self.target_ratio = float(target_entropy_ratio)
        self.keep_history = keep_history

        max_entropy = math.log(num_peers) if num_peers > 1 else 0.0
        self.state = TemperatureState(
            kappa=min(max(float(kappa_init), self.kappa_min), self.kappa_max),
            target_entropy=max_entropy * self.target_ratio,
        )

    # -- properties ------------------------------------------------------
    @property
    def kappa(self) -> float:
        return self.state.kappa

    @property
    def target_entropy(self) -> float:
        return self.state.target_entropy

    @property
    def entropy(self) -> float:
        return self.state.entropy

    # -- math ------------------------------------------------------------
    @staticmethod
    def entropy_of(weights: torch.Tensor, *, eps: float = 1e-8) -> float:
        """Shannon entropy of a probability vector, in nats."""
        safe = weights.clamp_min(eps)
        return float(-(safe * safe.log()).sum().item())

    def softmax_weights(self, scores: torch.Tensor, *, kappa: float | None = None) -> torch.Tensor:
        """Compute ``softmax(kappa * scores)`` without updating the controller."""
        temperature = self.kappa if kappa is None else float(kappa)
        return F.softmax(temperature * scores, dim=0)

    def step(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute weights for a new epoch and advance kappa."""
        if scores.ndim != 1:
            scores = scores.reshape(-1)
        if scores.numel() < 1:
            raise ValueError("at least one peer score is required")
        if scores.numel() != self.num_peers:
            # Participation gating can shrink the active set below the swarm size.
            # The entropy target is recomputed for the active arity so kappa keeps
            # tracking a meaningful maximum rather than a stale one.
            self.num_peers = int(scores.numel())
            max_entropy = math.log(self.num_peers) if self.num_peers > 1 else 0.0
            self.state.target_entropy = max_entropy * self.target_ratio

        weights = self.softmax_weights(scores)
        entropy = self.entropy_of(weights)
        delta = entropy - self.state.target_entropy
        momentum = self.beta * self.state.momentum + (1.0 - self.beta) * delta
        updated = min(
            self.kappa_max, max(self.kappa_min, self.state.kappa + self.eta_kappa * momentum)
        )

        self.state.entropy = entropy
        self.state.delta = delta
        self.state.momentum = momentum
        self.state.step += 1
        self.state.kappa = updated

        if self.keep_history:
            self.state.history.append(
                {
                    "step": float(self.state.step),
                    "kappa": updated,
                    "entropy": entropy,
                    "delta": delta,
                    "momentum": momentum,
                }
            )
            if len(self.state.history) > self.keep_history:
                self.state.history.pop(0)
        return weights

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "kappa": self.state.kappa,
            "momentum": self.state.momentum,
            "entropy": self.state.entropy,
            "target_entropy": self.state.target_entropy,
            "step": self.state.step,
            "num_peers": self.num_peers,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "beta": self.beta,
            "eta_kappa": self.eta_kappa,
            "target_ratio": self.target_ratio,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.state.kappa = float(payload.get("kappa", self.state.kappa))
        self.state.momentum = float(payload.get("momentum", 0.0))
        self.state.entropy = float(payload.get("entropy", 0.0))
        self.state.target_entropy = float(
            payload.get("target_entropy", self.state.target_entropy)
        )
        self.state.step = int(payload.get("step", 0))

    @classmethod
    def from_settings(cls, settings: dict[str, Any], *, num_peers: int, ) -> "AdaptiveKappa":
        return cls(
            num_peers=num_peers,
            kappa_init=float(settings.get("KAPPA_INIT", 1.0)),
            kappa_min=float(settings.get("KAPPA_MIN", 0.1)),
            kappa_max=float(settings.get("KAPPA_MAX", 10.0)),
            target_entropy_ratio=float(settings.get("TARGET_ENTROPY_RATIO", 0.8)),
            beta=float(settings.get("BETA_MOMENTUM", 0.9)),
            eta_kappa=float(settings.get("ETA_KAPPA", 0.05)),
        )


def entropy_collapsed(entropy: float, *, max_entropy: float, ratio: float) -> bool:
    """True when group diversity has collapsed below the configured ratio."""
    if max_entropy <= 0:
        return False
    return (entropy / max_entropy) < ratio
