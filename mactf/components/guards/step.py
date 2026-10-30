"""StepGuard: the invariants of one policy step (document 17 section 7.6)."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from ...exceptions import StepGuardViolation, ValidationError
from ...rl.loss import LossBreakdown
from ...training.protocol import EpochResult, StepResult, coerce_step_result
from ..kinds import Origin

log = logging.getLogger("mactf.components.step_guard")

MODES: tuple[str, ...] = ("off", "warn", "enforce", "quarantine")
SNAPSHOTS: tuple[str, ...] = ("auto", "always", "never")
VOID_ADVANTAGE_FLOOR = -1.0

#: Codes a ``StepViolation`` may carry; the table in section 7.6.1.
CODES: dict[str, str] = {
    "S001": "non-finite loss",
    "S002": "non-finite or excessive gradient norm",
    "S003": "non-finite parameters after the step",
    "S004": "required term not evidenced",
    "S005": "non-finite loss breakdown",
    "S006": "too many consecutive quarantined steps",
    "S007": "self_stepping declaration contradicted by the result",
    "S008": "positive advantage reported on a void node",
    "S009": "the policy raised",
}


@dataclass(frozen=True)
class StepGuardConfig:
    mode: str = "quarantine"
    snapshot: str = "auto"
    max_consecutive: int = 3
    grad_norm_ceiling: float = 1.0e6
    max_grad_norm: float | None = None
    check_params: bool = True
    void_clamp: bool = True
    require_void_evidence: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValidationError(f"STEP_GUARD.MODE must be one of {', '.join(MODES)}; got {self.mode!r}")
        if self.snapshot not in SNAPSHOTS:
            raise ValidationError(
                f"STEP_GUARD.SNAPSHOT must be one of {', '.join(SNAPSHOTS)}; got {self.snapshot!r}"
            )
        if int(self.max_consecutive) < 1:
            raise ValidationError(f"STEP_GUARD.MAX_CONSECUTIVE must be >= 1, got {self.max_consecutive!r}")
        if not float(self.grad_norm_ceiling) > 0.0:
            raise ValidationError("STEP_GUARD.GRAD_NORM_CEILING must be > 0")
        if self.max_grad_norm is not None and not float(self.max_grad_norm) > 0.0:
            raise ValidationError("STEP_GUARD.MAX_GRAD_NORM must be > 0 when set")

    @classmethod
    def from_settings(cls, block: Mapping[str, Any] | None) -> "StepGuardConfig":
        data = {str(key).upper(): value for key, value in dict(block or {}).items()}
        known = {
            "MODE", "SNAPSHOT", "MAX_CONSECUTIVE", "GRAD_NORM_CEILING", "MAX_GRAD_NORM",
            "CHECK_PARAMS", "VOID_CLAMP", "REQUIRE_VOID_EVIDENCE",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValidationError(
                "unknown TRAINER.STEP_GUARD key(s): " + ", ".join(unknown)
                + "; accepted: " + ", ".join(sorted(known))
            )
        defaults = cls()
        cap = data.get("MAX_GRAD_NORM", defaults.max_grad_norm)
        return cls(
            mode=str(data.get("MODE", defaults.mode)).lower(),
            snapshot=str(data.get("SNAPSHOT", defaults.snapshot)).lower(),
            max_consecutive=int(data.get("MAX_CONSECUTIVE", defaults.max_consecutive)),
            grad_norm_ceiling=float(data.get("GRAD_NORM_CEILING", defaults.grad_norm_ceiling)),
            max_grad_norm=None if cap is None else float(cap),
            check_params=bool(data.get("CHECK_PARAMS", defaults.check_params)),
            void_clamp=bool(data.get("VOID_CLAMP", defaults.void_clamp)),
            require_void_evidence=bool(
                data.get("REQUIRE_VOID_EVIDENCE", defaults.require_void_evidence)
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class StepViolation:
    code: str
    path: str
    step: int
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "path": self.path, "step": self.step, "message": self.message}


@dataclass(frozen=True)
class StepEvent:
    violations: tuple[StepViolation, ...]
    action: str                       # "warned" | "raised" | "quarantined"
    restored: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "restored": self.restored,
            "violations": [violation.as_dict() for violation in self.violations],
        }


# -- required-term evidence -----------------------------------------------------------
TermCheck = Callable[[StepResult, Mapping[str, Any]], str | None]


def _check_compliance_floor(result: StepResult, batch: Mapping[str, Any]) -> str | None:
    """A breached floor (negative ``floor``) must show up as a compliance term."""
    floor = float(batch.get("floor", 0.0))
    if floor >= 0.0:
        return None
    if float(result.breakdown.compliance) > 0.0:
        return None
    return (
        f"the node's floor is {floor:g} (a breach) but breakdown.compliance is "
        f"{result.breakdown.compliance:g}; the loss does not charge the breach"
    )


def _check_void_mask(result: StepResult, batch: Mapping[str, Any]) -> str | None:
    """A void node must show that its mask was honoured (``VoidMaskLoss`` reports it)."""
    if not bool(batch.get("void", False)):
        return None
    if "void_masked" in result.breakdown.extras or "advantage_mean" in result.breakdown.extras:
        return None
    return "the node is void but the result reports neither void_masked nor advantage_mean"


#: Name -> check. ``requires_terms`` entries must be keys here; extend with ``register_term_check``.
TERM_CHECKS: dict[str, TermCheck] = {
    "compliance_floor": _check_compliance_floor,
    "void_mask": _check_void_mask,
}


def register_term_check(name: str, check: TermCheck, *, replace: bool = False) -> None:
    if name in TERM_CHECKS and not replace:
        raise ValidationError(f"term check {name!r} already exists; pass replace=True")
    if not callable(check):
        raise ValidationError(f"term check {name!r} is not callable")
    TERM_CHECKS[name] = check


# -- void probe -------------------------------------------------------------------------
class VoidProbe:
    """Keeps a void node's learning signal from being positive.

    ``apply`` runs *before* the policy and hands it a read-only batch whose ``signal`` is at
    most ``VOID_ADVANTAGE_FLOOR`` when the node is void, so a policy that derives its
    advantage from ``batch["signal"]`` cannot see a reward on a void node. ``check`` runs
    *after* and looks at the one number a loss reports about advantages
    (``breakdown.extras["advantage_mean"]``, written by ``CompositeLoss``): positive on a
    void node is a violation. A policy that reports nothing cannot be checked; that is
    ``REQUIRE_VOID_EVIDENCE``'s job.
    """

    def __init__(self, *, clamp: bool = True, require_evidence: bool = False) -> None:
        self.clamp = bool(clamp)
        self.require_evidence = bool(require_evidence)

    def apply(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        data = dict(batch)
        void = bool(data.get("void", False))
        if void and self.clamp:
            data["signal"] = min(float(data.get("signal", 0.0)), VOID_ADVANTAGE_FLOOR)
        data["void_advantage_ceiling"] = VOID_ADVANTAGE_FLOOR if void else math.inf
        return MappingProxyType(data)

    def check(self, result: StepResult, batch: Mapping[str, Any]) -> str | None:
        if not bool(batch.get("void", False)):
            return None
        reported = result.breakdown.extras.get("advantage_mean")
        if reported is None:
            if self.require_evidence:
                return "the node is void and the policy reported no advantage_mean to verify"
            return None
        if float(reported) > 1e-9:
            return f"the node is void but the policy reported advantage_mean={float(reported):g} > 0"
        return None


# -- snapshot ---------------------------------------------------------------------------
@dataclass
class _Snapshot:
    parameters: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    accumulated: int


def _take_snapshot(module: torch.nn.Module, optimizer: Any) -> _Snapshot:
    return _Snapshot(
        parameters={name: p.detach().clone() for name, p in module.named_parameters()},
        buffers={name: b.detach().clone() for name, b in module.named_buffers()},
        optimizer=copy.deepcopy(optimizer.state_dict()),
        accumulated=int(getattr(optimizer, "_accumulated", 0)),
    )


def _restore_snapshot(module: torch.nn.Module, optimizer: Any, snap: _Snapshot) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(snap.parameters[name])
        for name, buffer in module.named_buffers():
            buffer.copy_(snap.buffers[name])
    optimizer.load_state_dict(snap.optimizer)
    optimizer.zero_grad()
    optimizer._accumulated = snap.accumulated


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all())


def _breakdown_numbers(breakdown: LossBreakdown) -> list[tuple[str, float]]:
    numbers = [
        ("policy", breakdown.policy), ("value", breakdown.value), ("kl", breakdown.kl),
        ("entropy", breakdown.entropy), ("compliance", breakdown.compliance),
        ("total", breakdown.total),
    ]
    numbers.extend((f"extras.{key}", value) for key, value in breakdown.extras.items())
    return numbers


class StepGuard:
    """Runs one policy step inside the invariants of section 7.6.1."""

    def __init__(self, config: StepGuardConfig | None = None) -> None:
        self.config = config or StepGuardConfig()
        self._probe = VoidProbe(
            clamp=self.config.void_clamp, require_evidence=self.config.require_void_evidence
        )
        self._trainer: Any = None
        self._consecutive: dict[str, int] = {}
        #: path -> (snapshot, step) taken after a live-loss policy returned, consumed by after_step
        self._held: dict[str, tuple[_Snapshot | None, int]] = {}
        self.events: list[StepEvent] = []
        self.checked = 0
        self.quarantined = 0
        self.restored = 0

    # -- binding ---------------------------------------------------------------------
    def bind(self, trainer: Any) -> None:
        self._trainer = trainer
        for path, optimizer in dict(trainer.optimizers).items():
            self.rebind(path, optimizer)

    def rebind(self, path: str, optimizer: Any) -> None:
        """Apply ``MAX_GRAD_NORM`` to an optimizer. Called again after any optimizer swap."""
        cap = self.config.max_grad_norm
        if cap is None:
            return
        current = getattr(optimizer, "max_grad_norm", None)
        optimizer.max_grad_norm = float(cap) if not current else min(float(current), float(cap))

    # -- one step ---------------------------------------------------------------------
    def run(self, policy: Any, node: Any, batch: Mapping[str, Any]) -> StepResult:
        cfg = self.config
        trainer = self._trainer
        if cfg.mode == "off" or trainer is None:
            return coerce_step_result(self._invoke(policy, node, batch), node=node)

        tree_node, peer = trainer.resolve_node(node)
        path = peer.name
        optimizer = trainer.optimizers[path]
        step_no = int(getattr(trainer, "global_step", 0))
        declared = getattr(policy, "self_stepping", None)
        snapshot = None
        if cfg.snapshot == "always" or (cfg.snapshot == "auto" and declared is not False):
            snapshot = _take_snapshot(peer.adapter.module, optimizer)

        guarded = self._probe.apply(batch)
        try:
            raw = self._invoke(policy, tree_node or peer, guarded)
            result = coerce_step_result(raw, node=tree_node or peer)
        except Exception as error:  # noqa: BLE001 - S009: restore, then re-raise unchanged
            if snapshot is not None and cfg.mode in ("enforce", "quarantine"):
                _restore_snapshot(peer.adapter.module, optimizer, snapshot)
                self.restored += 1
            self._record(
                (StepViolation("S009", path, step_no, f"{type(error).__name__}: {error}"),),
                action="raised", restored=snapshot is not None,
            )
            raise

        if result.metrics and result.metrics.get("pending"):
            return result  # a rendezvous placeholder carries no step to judge

        self.checked += 1
        violations = self._inspect(policy, result, guarded, peer, snapshot is not None, path, step_no)
        if not violations and result.loss.requires_grad:
            # A live loss: the trainer is about to ``backward`` and ``step`` for the policy.
            # The policy has not touched the parameters (a policy that does is S007), so a
            # snapshot taken now is the pre-step state; ``after_step`` uses it to judge the
            # step the trainer takes. ``warn`` mode keeps no snapshot (nothing is restored)
            # but still holds the entry so ``after_step`` reports S002 / S003.
            held = snapshot
            if (
                held is None
                and cfg.mode in ("enforce", "quarantine")
                and cfg.snapshot != "never"
            ):
                held = _take_snapshot(peer.adapter.module, optimizer)
            self._held[path] = (held, step_no)
        return self._settle(violations, result, peer, optimizer, snapshot, path, step_no)

    def after_step(self, node: Any, result: StepResult) -> StepResult:
        """Judge the step the *trainer* took for a live-loss policy.

        Called by ``MACTFTrainer.train_step`` right after ``_apply_custom_loss``. Checks the
        gradient norm that step produced and that the parameters are still finite; on a
        violation restores the snapshot held by :meth:`run`.
        """
        trainer = self._trainer
        if trainer is None:
            return result
        _, peer = trainer.resolve_node(node)
        path = peer.name
        held = self._held.pop(path, None)
        if held is None:
            return result
        snapshot, step_no = held
        optimizer = trainer.optimizers[path]
        violations = self._inspect_applied(result, peer, path, step_no)
        return self._settle(violations, result, peer, optimizer, snapshot, path, step_no)

    def _inspect_applied(
        self, result: StepResult, peer: Any, path: str, step_no: int
    ) -> tuple[StepViolation, ...]:
        cfg = self.config
        found: list[StepViolation] = []
        grad_norm = result.metrics.get("grad_norm") if result.metrics else None
        if grad_norm is not None:
            norm = float(grad_norm)
            if not math.isfinite(norm) or norm > cfg.grad_norm_ceiling:
                found.append(
                    StepViolation("S002", path, step_no, f"grad_norm {norm!r} exceeds {cfg.grad_norm_ceiling:g}")
                )
        if cfg.check_params:
            module = peer.adapter.module
            bad = [
                name for name, tensor in (*module.named_parameters(), *module.named_buffers())
                if tensor.is_floating_point() and not _finite_tensor(tensor)
            ]
            if bad:
                shown = ", ".join(bad[:3]) + ("..." if len(bad) > 3 else "")
                found.append(
                    StepViolation("S003", path, step_no, f"non-finite parameters after the step: {shown}")
                )
        return tuple(found)

    def _settle(
        self, violations: tuple[StepViolation, ...], result: StepResult, peer: Any,
        optimizer: Any, snapshot: _Snapshot | None, path: str, step_no: int,
    ) -> StepResult:
        """Apply the mode to a set of violations (possibly empty)."""
        cfg = self.config
        if not violations:
            self._consecutive[path] = 0
            return result
        if cfg.mode == "warn":
            self._record(violations, action="warned", restored=False)
            return result

        restored = False
        if snapshot is not None:
            _restore_snapshot(peer.adapter.module, optimizer, snapshot)
            self.restored += 1
            restored = True
        else:
            optimizer.zero_grad()
        self._held.pop(path, None)

        if cfg.mode == "enforce":
            self._record(violations, action="raised", restored=restored)
            raise StepGuardViolation(path, [violation.as_dict() for violation in violations])

        count = self._consecutive.get(path, 0) + 1
        self._consecutive[path] = count
        self.quarantined += 1
        self._record(violations, action="quarantined", restored=restored)
        if count >= cfg.max_consecutive:
            tripped = StepViolation(
                "S006", path, step_no,
                f"{count} consecutive quarantined steps (limit {cfg.max_consecutive})",
            )
            self._record((tripped,), action="raised", restored=restored)
            raise StepGuardViolation(
                path, [violation.as_dict() for violation in (*violations, tripped)]
            )
        return self._neutral(result, violations)

    # -- checks -----------------------------------------------------------------------
    def _inspect(
        self, policy: Any, result: StepResult, batch: Mapping[str, Any], peer: Any,
        snapshotted: bool, path: str, step_no: int,
    ) -> tuple[StepViolation, ...]:
        cfg = self.config
        found: list[StepViolation] = []

        def add(code: str, message: str) -> None:
            found.append(StepViolation(code, path, step_no, message))

        if not _finite_tensor(result.loss):
            add("S001", f"loss is {float(result.loss.detach().float().sum().item())!r}")
        for name, value in _breakdown_numbers(result.breakdown):
            if not math.isfinite(float(value)):
                add("S005", f"breakdown.{name} is {value!r}")
        grad_norm = result.metrics.get("grad_norm") if result.metrics else None
        if grad_norm is not None:
            norm = float(grad_norm)
            if not math.isfinite(norm) or norm > cfg.grad_norm_ceiling:
                add("S002", f"grad_norm {norm!r} exceeds {cfg.grad_norm_ceiling:g}")

        declared = getattr(policy, "self_stepping", None)
        if declared is True and result.loss.requires_grad:
            add("S007", "the policy declares self_stepping=True but returned a loss that requires grad")
        if declared is False and not result.loss.requires_grad:
            add("S007", "the policy declares self_stepping=False but returned a detached loss")

        # S003 is judged wherever the policy may have written a parameter itself, snapshot or
        # not: ``SNAPSHOT=never`` gives up the *undo*, not the detection.
        if cfg.check_params and declared is not False:
            module = peer.adapter.module
            bad = [
                name for name, tensor in (*module.named_parameters(), *module.named_buffers())
                if tensor.is_floating_point() and not _finite_tensor(tensor)
            ]
            if bad:
                shown = ", ".join(bad[:3]) + ("..." if len(bad) > 3 else "")
                add("S003", f"non-finite parameters after the step: {shown}")

        for term in sorted(getattr(policy, "requires_terms", frozenset())):
            check = TERM_CHECKS.get(term)
            if check is None:
                add("S004", f"unknown required term {term!r}; known: {', '.join(sorted(TERM_CHECKS))}")
                continue
            problem = check(result, batch)
            if problem:
                add("S004", f"{term}: {problem}")

        void_problem = self._probe.check(result, batch)
        if void_problem:
            add("S008", void_problem)
        return tuple(found)

    # -- helpers ----------------------------------------------------------------------
    def _invoke(self, policy: Any, node: Any, batch: Mapping[str, Any]) -> Any:
        handler = getattr(policy, "train_step", None)
        if callable(handler):
            return handler(node, batch)
        if callable(policy):
            # The legacy callable form: ``policy(node, batch, trainer=trainer)``.
            return policy(node, batch, trainer=self._trainer)
        raise ValidationError("a trainer policy needs train_step() or must be callable")

    @staticmethod
    def _neutral(result: StepResult, violations: tuple[StepViolation, ...]) -> StepResult:
        """The result a quarantined step reports: zero loss, not stepped, flagged."""
        zero = torch.zeros((), device=result.loss.device, dtype=torch.float32)
        return StepResult(
            loss=zero,
            breakdown=LossBreakdown(extras={"quarantined_codes": float(len(violations))}),
            stepped=False,
            metrics={"quarantined": 1.0},
        )

    def _record(self, violations: tuple[StepViolation, ...], *, action: str, restored: bool) -> None:
        event = StepEvent(violations=violations, action=action, restored=restored)
        self.events.append(event)
        log.warning(
            "step guard %s: %s",
            action, "; ".join(f"{v.code} {v.path}: {v.message}" for v in violations),
        )

    # -- reporting / state -------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "checked": self.checked,
            "quarantined": self.quarantined,
            "restored": self.restored,
            "events": [event.as_dict() for event in self.events[-20:]],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "consecutive": dict(self._consecutive),
            "checked": self.checked,
            "quarantined": self.quarantined,
            "restored": self.restored,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self._consecutive = {str(k): int(v) for k, v in dict(payload.get("consecutive", {})).items()}
        self.checked = int(payload.get("checked", 0))
        self.quarantined = int(payload.get("quarantined", 0))
        self.restored = int(payload.get("restored", 0))


class GuardedPolicy:
    """A policy behind a :class:`StepGuard`. Delegates every optional member."""

    def __init__(self, inner: Any, guard: StepGuard, *, origin: Origin) -> None:
        self._inner = inner
        self._guard = guard
        self.origin = origin

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def guard(self) -> StepGuard:
        return self._guard

    @property
    def name(self) -> str:
        named = getattr(self._inner, "name", None)
        return named if isinstance(named, str) and named else type(self._inner).__name__

    def __getattr__(self, attribute: str) -> Any:
        # Only reached for names not defined above: flags such as ``self_stepping``,
        # ``requires_terms``, ``unsafe`` and the optional protocol members.
        if attribute.startswith("__") or attribute in ("_inner", "_guard"):
            raise AttributeError(attribute)
        return getattr(self._inner, attribute)

    def attach(self, host: Any) -> None:
        self._guard.bind(host)
        attach = getattr(self._inner, "attach", None)
        if callable(attach):
            attach(host)

    def setup(self, node: Any) -> None:
        setup = getattr(self._inner, "setup", None)
        if callable(setup):
            setup(node)

    def train_step(self, node: Any, batch: Mapping[str, Any]) -> StepResult:
        return self._guard.run(self._inner, node, batch)

    def after_step(self, node: Any, result: StepResult) -> StepResult:
        return self._guard.after_step(node, result)

    def on_epoch_end(self, tree: Any, result: EpochResult) -> None:
        handler = getattr(self._inner, "on_epoch_end", None)
        if callable(handler):
            handler(tree, result)

    def card_metadata(self, path: str) -> dict[str, Any]:
        recorder = getattr(self._inner, "card_metadata", None)
        payload = dict(recorder(path)) if callable(recorder) else {}
        payload["step_guard"] = {
            "mode": self._guard.config.mode,
            "quarantined": self._guard.quarantined,
            "checked": self._guard.checked,
        }
        return payload

    def state_dict(self) -> dict[str, Any]:
        state = getattr(self._inner, "state_dict", None)
        return {
            "inner": dict(state()) if callable(state) else {},
            "guard": self._guard.state_dict(),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> Any:
        restore = getattr(self._inner, "load_state_dict", None)
        report = restore(payload.get("inner", {})) if callable(restore) else None
        self._guard.load_state_dict(payload.get("guard", {}))
        return report
