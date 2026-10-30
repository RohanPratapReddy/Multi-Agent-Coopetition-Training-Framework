"""Training policies: the protocol, the host contract and the default step (document 17 section 7).

A *training policy* decides what one optimizer step is for one node. ``DefaultTrainPolicy`` is the
former ``MACTFTrainer._builtin_step`` behind the policy protocol, with four optional function hooks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Protocol, runtime_checkable

import torch

from ..exceptions import ContractError, ValidationError
from ..rl.advantage import penalise_void_advantage
from ..rl.policy import make_policy
from ..runtime import autocast
from .protocol import EpochResult, StepResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..coopetition.peer import Peer
    from ..coopetition.tree import PeerNode, PeerTree
    from .optimizer import MACTFOptimizer

log = logging.getLogger("mactf.training.policy")

#: The advantage a void node is held to, from ``_builtin_step`` (trainer.py:1573).
VOID_ADVANTAGE_FLOOR = -1.0


@runtime_checkable
class PolicyHost(Protocol):
    """What a policy may rely on. ``MACTFTrainer`` implements it; nothing else is promised.

    Everything here is a public attribute or method. A policy that reaches for an
    underscore name has left the contract and needs ``unsafe = True`` (section 7.7).
    """

    tree: "PeerTree"
    reference: "Peer"
    optimizers: Mapping[str, "MACTFOptimizer"]
    settings: Any
    runtime: Any
    functions: Any          # FunctionBook, section 8.4
    slots: Any              # SlotTable, section 4.4
    global_step: int
    distill: Any            # the CoDistillStep when DISTILLATION.ENABLED, else None

    def resolve_node(self, node: Any) -> tuple["PeerNode | None", "Peer"]: ...

    def loss_for(self, path: str) -> Any:
        """The loss function in force for ``path`` (``RL.loss`` override, else ``LOSS``)."""

    def backward(self, loss: torch.Tensor, optimizer: "MACTFOptimizer") -> None:
        """Scale for accumulation and AMP, then ``backward``. The only sanctioned way."""


@runtime_checkable
class TrainerPolicy(Protocol):
    """The full surface. Optional members are marked; absent ones are skipped."""

    #: Registry name; used in cards and checkpoints.
    name: ClassVar[str]
    #: True if the policy bypasses the facade (reads private trainer state, writes
    #: parameters directly). Needs ``TRAINER.ALLOW_UNSAFE_POLICY``. Default False.
    unsafe: ClassVar[bool]
    #: True: the policy calls ``backward`` and steps the optimizer itself (it returns a
    #: detached loss). False: it returns a live loss and the trainer steps for it.
    #: ``None`` means "decided per call"; the guard then snapshots defensively.
    self_stepping: ClassVar[bool | None]
    #: Terms the policy promises to honour; the guard verifies them (section 7.6.4).
    requires_terms: ClassVar[frozenset[str]]
    #: True if the policy delegates to the distillation step when distillation is on.
    composes_distillation: ClassVar[bool]
    #: RL function families the policy reads through `host.functions` (section 8.5.3). A family
    #: configured in `RL` that no active consumer lists is a configuration error (`E041`).
    #: Absent means ``{"loss"}``, which is what a plain callable policy uses.
    consumes_functions: ClassVar[frozenset[str]]

    def train_step(self, node: Any, batch: Mapping[str, Any]) -> StepResult: ...

    # -- optional ----------------------------------------------------------
    def attach(self, host: PolicyHost) -> None: ...
    def setup(self, node: Any) -> None: ...
    def on_epoch_end(self, tree: Any, result: EpochResult) -> None: ...
    def card_metadata(self, path: str) -> dict[str, Any]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, payload: Mapping[str, Any]) -> None: ...
    def describe(self) -> dict[str, Any]: ...


def policy_flag(policy: Any, name: str, default: Any) -> Any:
    """Read an optional class attribute with a default (a plain callable has none)."""
    return getattr(policy, name, default)


def policy_name(policy: Any) -> str:
    explicit = getattr(policy, "name", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    return getattr(policy, "__name__", type(policy).__name__)


@dataclass(frozen=True)
class _Hooks:
    advantage: Any = None
    action: Any = None
    regularization: Any = None
    metric: Any = None


class DefaultTrainPolicy:
    """The policy-gradient step, i.e. ``_builtin_step`` behind the policy protocol."""

    name: ClassVar[str] = "default"
    unsafe: ClassVar[bool] = False
    self_stepping: ClassVar[bool | None] = True
    requires_terms: ClassVar[frozenset[str]] = frozenset()
    composes_distillation: ClassVar[bool] = False
    consumes_functions: ClassVar[frozenset[str]] = frozenset(
        {"loss", "advantage", "action", "regularization", "metric"}
    )

    ADVANTAGE_MASKS: ClassVar[tuple[str, ...]] = ("none", "void", "valid")

    def __init__(
        self,
        *,
        advantage: Any = None,
        action: Any = None,
        regularization: Any = None,
        metric: Any = None,
        advantage_masks: str = "none",
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        if advantage_masks not in self.ADVANTAGE_MASKS:
            raise ValidationError(
                f"advantage_masks must be one of {', '.join(self.ADVANTAGE_MASKS)}; "
                f"got {advantage_masks!r}"
            )
        if not 0.0 <= float(gamma) <= 1.0 or not 0.0 <= float(lam) <= 1.0:
            raise ValidationError("gamma and lam must be in [0, 1]")
        self._spec_hooks = _Hooks(advantage, action, regularization, metric)
        self.advantage_masks = advantage_masks
        self.gamma = float(gamma)
        self.lam = float(lam)
        self._host: PolicyHost | None = None

    # -- lifecycle -----------------------------------------------------------
    def attach(self, host: PolicyHost) -> None:
        self._host = host

    def setup(self, node: Any) -> None:
        """Nothing to prepare: the default step owns no per-node state."""
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "advantage_masks": self.advantage_masks,
            "hooks": sorted(
                key for key, value in vars(self._spec_hooks).items() if value is not None
            ),
        }

    # -- hook resolution: per node, through the FunctionBook ------------------
    def _hook(self, family: str, path: str) -> Any:
        """A constructor-supplied hook wins; otherwise ask the host's FunctionBook.

        ``FunctionBook.get`` returns ``None`` when neither ``RL.<family>`` nor an override
        for ``path`` is configured, which is the default-identity case.
        """
        explicit = getattr(self._spec_hooks, family)
        if explicit is not None:
            return explicit
        host = self._host
        if host is None:
            return None
        return host.functions.get(family, path)

    # -- the step --------------------------------------------------------------
    def train_step(self, node: Any, batch: Mapping[str, Any]) -> StepResult:
        host = self._require_host()
        tree_node, peer = host.resolve_node(node)
        path = peer.name
        optimizer = host.optimizers.get(path)
        if optimizer is None:
            raise ContractError(f"no optimizer is bound for node {path!r}")

        advantage_fn = self._hook("advantage", path)
        action_fn = self._hook("action", path)
        regularizer = self._hook("regularization", path)
        metric_fn = self._hook("metric", path)
        loss_function = host.loss_for(path)
        reference_peer = host.reference
        runtime = host.runtime

        states = batch["states"]
        adapter = peer.adapter
        adapter.train()
        if getattr(optimizer, "_accumulated", 0) == 0:
            optimizer.zero_grad()
        count = int(states.shape[0])
        fixed: dict[str, torch.Tensor] = {}
        seen: dict[str, Any] = {}

        def evaluate() -> tuple[torch.Tensor, Any]:
            with autocast(runtime):
                logits, values = adapter(states)
            policy = make_policy(logits)
            with torch.no_grad():
                ref_logits, _ = reference_peer.adapter(states)
            reference_policy = make_policy(ref_logits)
            if "actions" not in fixed:
                # Chosen once so a closure-driven optimizer sees one smooth objective.
                if action_fn is None:
                    fixed["actions"] = policy.probs.argmax(dim=-1).detach()
                else:
                    chosen = action_fn(logits.detach(), sample=False)
                    fixed["actions"] = torch.as_tensor(chosen, device=logits.device).long()
            signal = torch.full(
                (count,), float(batch.get("signal", 0.0)),
                device=logits.device, dtype=logits.dtype,
            )
            void = torch.full(
                (count,), bool(batch.get("void", False)),
                device=logits.device, dtype=torch.bool,
            )
            if advantage_fn is None:
                advantages = penalise_void_advantage(signal, void, floor=VOID_ADVANTAGE_FLOOR)
            else:
                advantages = self._custom_advantage(advantage_fn, signal, values, void, count)
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
            total, breakdown = loss_function.compute(loss_batch)
            if regularizer is not None:
                penalty = torch.as_tensor(regularizer(adapter.module), device=total.device)
                total = total + penalty.to(total.dtype)
                breakdown.extras["regularization"] = float(penalty.detach().item())
                breakdown.total = float(total.detach().item())
            seen["loss_batch"] = loss_batch
            return total, breakdown

        if optimizer.closure_based:
            total, breakdown = evaluate()
            outcome = optimizer.step(closure=lambda: evaluate()[0])
            total, breakdown = evaluate()
        else:
            total, breakdown = evaluate()
            host.backward(total, optimizer)
            outcome = optimizer.step()
        metrics = {"lr": float(outcome.lr)}
        if outcome.grad_norm is not None:
            metrics["grad_norm"] = float(outcome.grad_norm)
        if metric_fn is not None:
            extra = metric_fn(seen["loss_batch"], total.detach())
            for key, value in dict(extra or {}).items():
                metrics[f"metric.{key}"] = float(value)
        return StepResult(
            loss=total.detach(),
            breakdown=breakdown,
            stepped=bool(outcome.stepped),
            metrics=metrics,
        )

    # -- internals -------------------------------------------------------------
    def _require_host(self) -> PolicyHost:
        if self._host is None:
            raise ContractError(
                "DefaultTrainPolicy.train_step was called before attach(); the trainer "
                "attaches policies in build()"
            )
        return self._host

    def _custom_advantage(
        self,
        advantage_fn: Any,
        signal: torch.Tensor,
        values: torch.Tensor,
        void: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        masks: torch.Tensor | None
        if self.advantage_masks == "void":
            masks = void
        elif self.advantage_masks == "valid":
            masks = ~void
        else:
            masks = None
        produced = advantage_fn(
            signal,
            values.detach().reshape(count),
            masks=masks,
            gamma=self.gamma,
            lam=self.lam,
        )
        raw = getattr(produced, "advantages", produced)
        advantages = torch.as_tensor(raw, device=signal.device, dtype=signal.dtype).reshape(count)
        if not bool(torch.isfinite(advantages).all()):
            raise ContractError(
                f"advantage function {type(advantage_fn).__name__} produced a non-finite value"
            )
        # Rule 1: whatever the function did, a void node is held at or below the floor.
        return penalise_void_advantage(advantages, void, floor=VOID_ADVANTAGE_FLOOR)


def register_builtins() -> None:
    """Register ``default`` (and, through its module, ``co_distill``) as built-in policies.

    Idempotent. Called at import; ``trainer_policy``'s ``builtins_module`` is this module, so
    the registry imports it before the first write and a plugin cannot take either name first.
    """
    from ..components.registry import HUB

    registry = HUB.registry("trainer_policy")
    if registry.find("default") is None:
        registry.register_builtin("default", DefaultTrainPolicy)
    from ..distill import policy as _co_distill  # noqa: F401  (registers ``co_distill``)


__all__ = [
    "DefaultTrainPolicy",
    "PolicyHost",
    "TrainerPolicy",
    "VOID_ADVANTAGE_FLOOR",
    "policy_flag",
    "policy_name",
    "register_builtins",
]

register_builtins()
