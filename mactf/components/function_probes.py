"""Synthetic-batch probes: verifying that an RL function behaves (document 17 section 8.3)."""

from __future__ import annotations

import contextlib
import dataclasses
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from ..exceptions import ComponentContractError

#: Codes a function may skip through ``probe["skip"]``. Everything else is a standard.
EXEMPTABLE: frozenset[str] = frozenset({"F005", "F006", "F007"})

_LOSS_VOCABULARY: frozenset[str] = frozenset({
    "policy", "reference", "actions", "advantages", "values", "value_targets",
    "compliance_floor", "void_mask", "rewards",
})


@dataclass(frozen=True)
class Finding:
    code: str
    family: str
    name: str
    message: str

    def render(self) -> str:
        return f"{self.code}: {self.message}"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "family": self.family, "name": self.name, "message": self.message}


@dataclass(frozen=True)
class ProbeContext:
    """Sizes and tolerances of the synthetic inputs. Seeded: a probe is reproducible."""

    batch: int = 6
    classes: int = 3
    state_dim: int = 4
    seed: int = 20260101
    rtol: float = 1.0e-5
    atol: float = 1.0e-6

    def generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator

    def adjusted(self, declared: Mapping[str, Any]) -> "ProbeContext":
        sizes = {key: int(declared[key]) for key in ("batch", "classes", "state_dim") if key in declared}
        return replace(self, **sizes) if sizes else self


class _Run:
    """Accumulates the findings of one probe and swallows an entry-method exception."""

    def __init__(self, family: str, name: str, skip: frozenset[str]) -> None:
        self.family = family
        self.name = name
        self.skip = skip
        self.findings: list[Finding] = []

    def add(self, code: str, message: str) -> None:
        if code in self.skip:
            return
        self.findings.append(Finding(code, self.family, self.name, message))

    def call(self, label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        try:
            return True, fn(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - the probe's whole job is to catch this
            self.add("F001", f"{label} raised {type(error).__name__}: {error}")
            return False, None


# -- comparison helpers --------------------------------------------------------------
def _snapshot(value: Any) -> Any:
    """A deep copy of every tensor inside ``value`` (tensors, mappings, sequences, dataclasses)."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_snapshot(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _snapshot(getattr(value, field.name)) for field in dataclasses.fields(value)}
    return value


def _equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and torch.equal(left.detach(), right.detach())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _close(left: Any, right: Any, ctx: ProbeContext) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and torch.allclose(
            left.detach().float(), right.detach().float(), rtol=ctx.rtol, atol=ctx.atol, equal_nan=True
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=ctx.rtol, abs_tol=ctx.atol)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_close(left[key], right[key], ctx) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_close(a, b, ctx) for a, b in zip(left, right))
    return left == right


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach().float()).all())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parameters(instance: Any) -> list[nn.Parameter]:
    accessor = getattr(instance, "parameters", None)
    if not callable(accessor):
        return []
    try:
        return [parameter for parameter in accessor() if isinstance(parameter, nn.Parameter)]
    except TypeError:
        return []


@contextlib.contextmanager
def _evaluating(instance: Any) -> Iterator[None]:
    """Switch a wrapper to evaluation mode for the probe and restore it afterwards."""
    toggle = getattr(instance, "train", None)
    was = getattr(instance, "training", getattr(instance, "_training", None))
    if callable(toggle) and isinstance(was, bool):
        toggle(False)
        try:
            yield
        finally:
            toggle(was)
    else:
        yield


def _check_grad(run: _Run, label: str, instance: Any, output: torch.Tensor, tracked: Sequence[torch.Tensor] = ()) -> None:
    """F005: the output must depend on the parameters (or tracked inputs) it claims to."""
    leaves: list[torch.Tensor] = [*tracked, *_parameters(instance)]
    if not leaves:
        return
    for leaf in leaves:
        leaf.grad = None
    if not output.requires_grad:
        run.add("F005", f"{label} does not require grad although the function owns trainable inputs")
        return
    try:
        output.float().sum().backward(retain_graph=True)
    except RuntimeError as error:
        run.add("F005", f"{label} could not be differentiated: {error}")
        return
    if all(leaf.grad is None for leaf in leaves):
        run.add("F005", f"{label} is not connected to any of its trainable inputs")
    for leaf in leaves:
        leaf.grad = None


def _loss_batch(ctx: ProbeContext) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    """The batch ``DefaultTrainPolicy`` builds (``trainer.py:1575-1585``), at probe size."""
    from ..rl.policy import make_policy

    generator = ctx.generator()
    logits = torch.randn(ctx.batch, ctx.classes, generator=generator, requires_grad=True)
    reference = torch.randn(ctx.batch, ctx.classes, generator=generator)
    values = torch.randn(ctx.batch, 1, generator=generator, requires_grad=True)
    batch = {
        "policy": make_policy(logits),
        "reference": make_policy(reference),
        "actions": torch.randint(0, ctx.classes, (ctx.batch,), generator=generator),
        "advantages": torch.randn(ctx.batch, generator=generator),
        "values": values,
        "value_targets": torch.randn(ctx.batch, 1, generator=generator),
        "compliance_floor": torch.tensor(-0.5),
        "void_mask": torch.zeros(ctx.batch, dtype=torch.bool),
        "rewards": torch.randn(ctx.batch, generator=generator),
    }
    return batch, (logits, values)


def _scalar_result(run: _Run, label: str, result: Any) -> torch.Tensor | None:
    """The shared checks for ``loss.compute`` and ``objective.__call__``: ``(Tensor, extras)``."""
    if not isinstance(result, tuple) or len(result) != 2:
        run.add("F002", f"{label} returned {type(result).__name__}; expected a (tensor, breakdown) pair")
        return None
    total, breakdown = result
    if not torch.is_tensor(total):
        run.add("F002", f"{label} returned {type(total).__name__} as the loss; expected a tensor")
        return None
    if total.numel() != 1:
        run.add("F003", f"{label} loss has shape {tuple(total.shape)}; expected a single element")
        return None
    if not _finite(total):
        run.add("F004", f"{label} loss is not finite ({float(total.detach()):g})")
    as_dict = getattr(breakdown, "as_dict", None)
    values = as_dict() if callable(as_dict) else (dict(breakdown) if isinstance(breakdown, Mapping) else {})
    for key, value in values.items():
        if _is_number(value) and not math.isfinite(float(value)):
            run.add("F004", f"{label} breakdown[{key!r}] is not finite")
    reported = getattr(breakdown, "total", None)
    if _is_number(reported) and math.isfinite(float(total.detach())):
        actual = float(total.detach())
        if abs(float(reported) - actual) > 1.0e-4 * max(1.0, abs(actual)):
            run.add("F008", f"{label} breakdown.total is {float(reported):g} but the loss is {actual:g}")
    return total


# -- the probes -----------------------------------------------------------------------
def _probe_value(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    generator = ctx.generator()
    states = torch.randn(ctx.batch, ctx.state_dim, generator=generator)
    held = states.clone()
    with _evaluating(instance):
        ok, estimate = run.call("__call__(states)", instance, states)
        if not ok:
            return
        values = getattr(estimate, "values", None)
        if not torch.is_tensor(values):
            run.add("F002", f"returned {type(estimate).__name__}; a value function returns a "
                            "ValueEstimate whose .values is a tensor")
            return
        if values.dim() == 0 or values.shape[0] != ctx.batch:
            run.add("F003", f"values has shape {tuple(values.shape)}; the leading dimension must be "
                            f"the batch size {ctx.batch}")
        elif not _finite(values):
            run.add("F004", "values contain a non-finite entry")
        if not torch.equal(states, held):
            run.add("F007", "the function modified the states tensor it was given")
        _check_grad(run, "values", instance, values)
        again_ok, again = run.call("__call__(states) (second call)", instance, states)
        second = getattr(again, "values", None)
        if again_ok and torch.is_tensor(second) and not _close(values, second, ctx):
            run.add("F006", "two calls on the same states returned different values in evaluation mode")


def _probe_advantage(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    generator = ctx.generator()
    rewards = torch.randn(ctx.batch, generator=generator)
    values = torch.randn(ctx.batch, generator=generator)
    held = (rewards.clone(), values.clone())
    first: torch.Tensor | None = None
    for label, masks in (("masks=None", None), ("masks=ones", torch.ones(ctx.batch, dtype=torch.bool))):
        ok, produced = run.call(f"__call__(rewards, values, {label})", instance, rewards, values,
                                masks=masks, gamma=0.99, lam=0.95)
        if not ok:
            return
        raw = getattr(produced, "advantages", produced)
        if not torch.is_tensor(raw):
            run.add("F002", f"returned {type(produced).__name__}; expected an AdvantageBatch or a tensor")
            return
        if raw.numel() != ctx.batch:
            run.add("F003", f"advantages has {raw.numel()} elements; expected one per sample ({ctx.batch})")
            return
        if not _finite(raw):
            run.add("F004", f"advantages contain a non-finite entry ({label})")
        if first is None:
            first = raw
    if not (torch.equal(rewards, held[0]) and torch.equal(values, held[1])):
        run.add("F007", "the function modified the rewards or values tensor it was given")
    ok, again = run.call("__call__ (second call)", instance, rewards, values,
                         masks=None, gamma=0.99, lam=0.95)
    second = getattr(again, "advantages", again)
    if ok and first is not None and torch.is_tensor(second) and not _close(first, second, ctx):
        run.add("F006", "two calls on the same inputs returned different advantages")


def _probe_loss(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    hook = getattr(instance, "probe_batch", None)
    if callable(hook):
        batch, tracked = hook(ctx)
    else:
        required = frozenset(getattr(instance, "required_fields", ()) or ())
        missing = sorted(required - _LOSS_VOCABULARY)
        if missing:
            run.add("F009", f"requires batch keys {missing} that the probe cannot supply; "
                            "define probe_batch(ctx) -> (batch, tracked_tensors) on the loss")
            return
        batch, tracked = _loss_batch(ctx)
    held = _snapshot(batch)
    ok, result = run.call("compute(batch)", instance.compute, batch)
    if not ok:
        return
    total = _scalar_result(run, "compute(batch)", result)
    if total is None:
        return
    if not _equal(held, _snapshot(batch)):
        run.add("F007", "compute() modified a tensor in the batch it was given")
    _check_grad(run, "the loss", instance, total, tracked)
    fresh, _ = _loss_batch(ctx) if not callable(hook) else hook(ctx)
    ok, again = run.call("compute(batch) (second call)", instance.compute, fresh)
    if ok and isinstance(again, tuple) and torch.is_tensor(again[0]) and not _close(total, again[0], ctx):
        run.add("F006", "two calls on identical batches returned different losses")


def _probe_rollup(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    from ..scoring.base import ScorerOutput, ScoringKind

    children = (
        ScorerOutput(kind=ScoringKind.HIERARCHY, amount=1.0, rules=("a",)),
        ScorerOutput(kind=ScoringKind.HIERARCHY, amount=2.0, rules=("b",), void_rules=("P1",)),
        ScorerOutput(kind=ScoringKind.HIERARCHY, amount=0.5, rules=("c",), void_rules=("P20",)),
    )
    before = [child.as_dict() for child in children]
    weights = {"a": 0.2, "b": 0.3, "c": 0.5}
    for label, kwargs in (("no weights", {}), ("weights", {"weights": weights}), ("scores", {"scores": weights})):
        ok, output = run.call(f"rollup(parent, children, {label})", instance.rollup, "root", list(children), **kwargs)
        if not ok:
            return
        if not isinstance(output, ScorerOutput):
            run.add("F002", f"returned {type(output).__name__}; a rollup returns a ScorerOutput")
            return
        if not _is_number(output.amount) or not math.isfinite(float(output.amount)):
            run.add("F004", f"amount is not a finite number ({label})")
        dropped = sorted({"P1", "P20"} - set(output.void_rules))
        if dropped:
            run.add("F010", f"output.void_rules is missing {dropped} from the children ({label}); "
                            "a rollup must carry every child's void set upward")
    if [child.as_dict() for child in children] != before:
        run.add("F007", "the rollup modified a child output")
    ok, empty = run.call("rollup(parent, [])", instance.rollup, "root", [])
    if ok and not isinstance(empty, ScorerOutput):
        run.add("F002", f"an empty child list returned {type(empty).__name__}; expected a ScorerOutput")
    ok, first = run.call("rollup (first)", instance.rollup, "root", list(children))
    ok2, second = run.call("rollup (second)", instance.rollup, "root", list(children))
    if ok and ok2 and isinstance(first, ScorerOutput) and isinstance(second, ScorerOutput) \
            and not _close(float(first.amount), float(second.amount), ctx):
        run.add("F006", "two calls on the same children returned different amounts")


def _probe_return(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    generator = ctx.generator()
    rewards = torch.randn(ctx.batch, generator=generator)
    held = rewards.clone()
    for label, masks in (("masks=None", None), ("masks=ones", torch.ones(ctx.batch, dtype=torch.bool))):
        ok, result = run.call(f"__call__(rewards, gamma=0.9, {label})", instance, rewards, gamma=0.9, masks=masks)
        if not ok:
            return
        if not torch.is_tensor(result):
            run.add("F002", f"returned {type(result).__name__}; expected a tensor")
            return
        if result.shape != rewards.shape:
            run.add("F003", f"returned shape {tuple(result.shape)}; expected {tuple(rewards.shape)}")
            return
        if not _finite(result):
            run.add("F004", f"returns contain a non-finite entry ({label})")
    if not torch.equal(rewards, held):
        run.add("F007", "the function modified the rewards tensor it was given")
    ok, first = run.call("__call__ (first)", instance, rewards, gamma=0.9, masks=None)
    ok2, second = run.call("__call__ (second)", instance, rewards, gamma=0.9, masks=None)
    if ok and ok2 and torch.is_tensor(first) and torch.is_tensor(second) and not _close(first, second, ctx):
        run.add("F006", "two calls on the same rewards returned different returns")


def _probe_reward(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    from types import SimpleNamespace

    card = SimpleNamespace(final_score=0.25)
    episode = SimpleNamespace(scorecard=SimpleNamespace(peers={"probe": card}))
    before = card.final_score
    for peer in ("probe", "absent-peer"):
        ok, value = run.call(f"__call__(episode, peer={peer!r})", instance, episode, peer=peer)
        if not ok:
            return
        if not _is_number(value):
            run.add("F002", f"returned {type(value).__name__}; a reward is a number")
            return
        if not math.isfinite(float(value)):
            run.add("F004", f"reward for {peer!r} is not finite")
    if card.final_score != before:
        run.add("F007", "the function modified the scorecard it was given")
    ok, first = run.call("__call__ (first)", instance, episode, peer="probe")
    ok2, second = run.call("__call__ (second)", instance, episode, peer="probe")
    if ok and ok2 and _is_number(first) and _is_number(second) and not _close(first, second, ctx):
        run.add("F006", "two calls on the same episode returned different rewards")


def _probe_regularization(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    torch.manual_seed(ctx.seed)
    module = nn.Linear(ctx.state_dim, ctx.classes)
    held = {key: value.detach().clone() for key, value in module.state_dict().items()}
    ok, penalty = run.call("__call__(module)", instance, module)
    if not ok:
        return
    if not torch.is_tensor(penalty):
        run.add("F002", f"returned {type(penalty).__name__}; a regularizer returns a tensor")
        return
    if penalty.numel() != 1:
        run.add("F003", f"penalty has shape {tuple(penalty.shape)}; expected a single element")
        return
    if not _finite(penalty):
        run.add("F004", "penalty is not finite")
    if not _equal(held, {key: value.detach() for key, value in module.state_dict().items()}):
        run.add("F007", "the regularizer modified the module's parameters")
    _check_grad(run, "the penalty", _ModuleHandle(module), penalty)


class _ModuleHandle:
    """Lets ``_check_grad`` treat a bare ``nn.Module`` as the owner of trainable inputs."""

    def __init__(self, module: nn.Module) -> None:
        self._module = module

    def parameters(self) -> list[nn.Parameter]:
        return list(self._module.parameters())


def _probe_objective(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    batch, tracked = _loss_batch(ctx)
    held = _snapshot(batch)
    ok, result = run.call("__call__(batch)", instance, batch)
    if not ok:
        return
    total = _scalar_result(run, "__call__(batch)", result)
    if total is None:
        return
    if not _equal(held, _snapshot(batch)):
        run.add("F007", "the objective modified a tensor in the batch it was given")
    _check_grad(run, "the objective", instance, total, tracked)
    fresh, _ = _loss_batch(ctx)
    ok, again = run.call("__call__(batch) (second call)", instance, fresh)
    if ok and isinstance(again, tuple) and torch.is_tensor(again[0]) and not _close(total, again[0], ctx):
        run.add("F006", "two calls on identical batches returned different objectives")


def _probe_gradient(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    generator = ctx.generator()
    torch.manual_seed(ctx.seed)
    module = nn.Linear(ctx.state_dim, ctx.classes)
    module(torch.randn(ctx.batch, ctx.state_dim, generator=generator)).sum().backward()
    parameters = list(module.parameters())
    held = [parameter.detach().clone() for parameter in parameters]
    ok, norm = run.call("__call__(parameters, max_norm=1.0)", instance, parameters, max_norm=1.0, scaler=None)
    if not ok:
        return
    if not _is_number(norm):
        run.add("F002", f"returned {type(norm).__name__}; a gradient function returns the norm as a number")
        return
    if not math.isfinite(float(norm)):
        run.add("F004", "the reported gradient norm is not finite")
    elif float(norm) < 0.0:
        run.add("F003", f"the reported gradient norm {float(norm):g} is negative")
    if not all(torch.equal(parameter.detach(), before) for parameter, before in zip(parameters, held)):
        run.add("F007", "a gradient function changed parameter values; it may only touch .grad")


def _probe_metric(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    batch, _ = _loss_batch(ctx)
    generator = ctx.generator()
    batch["targets"] = torch.randint(0, ctx.classes, (ctx.batch,), generator=generator)
    output = torch.randn(ctx.batch, ctx.classes, generator=generator)
    held = _snapshot(batch)
    ok, produced = run.call("__call__(batch, output)", instance, batch, output)
    if not ok:
        return
    if not isinstance(produced, Mapping):
        run.add("F002", f"returned {type(produced).__name__}; a metric returns a mapping of numbers")
        return
    for key, value in produced.items():
        if not isinstance(key, str):
            run.add("F002", f"metric key {key!r} is not a string")
        elif not _is_number(value):
            run.add("F002", f"metric {key!r} is a {type(value).__name__}; expected a number")
        elif not math.isfinite(float(value)):
            run.add("F004", f"metric {key!r} is not finite")
    if not _equal(held, _snapshot(batch)):
        run.add("F007", "the metric modified a tensor in the batch it was given")
    ok, again = run.call("__call__ (second call)", instance, batch, output)
    if ok and isinstance(again, Mapping) and not _close(dict(produced), dict(again), ctx):
        run.add("F006", "two calls on the same inputs returned different metrics")


def _probe_action(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    generator = ctx.generator()
    logits = torch.randn(ctx.batch, ctx.classes, generator=generator)
    held = logits.clone()
    greedy: torch.Tensor | None = None
    for sample in (False, True):
        ok, actions = run.call(f"__call__(logits, sample={sample})", instance, logits, sample=sample)
        if not ok:
            return
        if not torch.is_tensor(actions):
            run.add("F002", f"returned {type(actions).__name__}; an action function returns a tensor")
            return
        if actions.is_floating_point() or actions.is_complex():
            run.add("F002", f"returned dtype {actions.dtype}; actions are integer class indices")
            return
        if tuple(actions.shape) != (ctx.batch,):
            run.add("F003", f"returned shape {tuple(actions.shape)}; expected ({ctx.batch},)")
            return
        if int(actions.min()) < 0 or int(actions.max()) >= ctx.classes:
            run.add("F003", f"an action lies outside [0, {ctx.classes - 1}]")
        if not sample:
            greedy = actions
    if not torch.equal(logits, held):
        run.add("F007", "the function modified the logits tensor it was given")
    ok, again = run.call("__call__(logits, sample=False) (second call)", instance, logits, sample=False)
    if ok and greedy is not None and torch.is_tensor(again) and not torch.equal(greedy, again):
        run.add("F006", "sample=False is not deterministic")


class _ProbeNode:
    """The few attributes a node-aware synthesis reads (``coopetition/synthesis.py``)."""

    def __init__(self, path: str, children: Sequence[str]) -> None:
        self.path = path
        self.children = tuple(children)
        self.depth = 0
        self.status = "ready"


def _probe_synthesis(instance: Any, ctx: ProbeContext, run: _Run) -> None:
    generator = ctx.generator()
    names = ("a", "b", "c")
    proposals = {name: torch.randn(ctx.batch, ctx.classes, generator=generator) for name in names}
    held = _snapshot(proposals)
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    node_level = callable(getattr(instance, "synthesize", None))
    free_shape = getattr(instance, "output_shape_rule", "same") == "free"

    def invoke(use_weights: bool) -> tuple[bool, Any]:
        if node_level:
            return run.call(
                "synthesize(node, proposals)", instance.synthesize, _ProbeNode("root", names), dict(proposals),
                scores={"a": 1.0, "b": 0.5, "c": 0.1}, weights=weights if use_weights else None, task=None,
            )
        return run.call("__call__(proposals)", instance, dict(proposals), weights=weights if use_weights else None)

    first: Any = None
    for use_weights in (False, True):
        ok, result = invoke(use_weights)
        if not ok:
            return
        if not torch.is_tensor(result):
            run.add("F002", f"returned {type(result).__name__}; a synthesis returns a tensor")
            return
        if not free_shape and tuple(result.shape) != (ctx.batch, ctx.classes):
            run.add("F012", f"returned shape {tuple(result.shape)}; expected {(ctx.batch, ctx.classes)} "
                            "(set output_shape_rule = 'free' if the shape legitimately changes)")
            return
        if free_shape and result.shape[0] != ctx.batch:
            run.add("F003", f"returned leading dimension {result.shape[0]}; expected the batch {ctx.batch}")
            return
        if not _finite(result):
            run.add("F004", "synthesis output contains a non-finite entry")
        if first is None:
            first = result
    if not _equal(held, _snapshot(proposals)):
        run.add("F007", "the synthesis modified a proposal tensor it was given")
    ok, again = invoke(False)
    if ok and torch.is_tensor(again) and torch.is_tensor(first) and not _close(first, again, ctx):
        run.add("F006", "two calls on the same proposals returned different tensors")


PROBES: dict[str, Callable[[Any, ProbeContext, _Run], None]] = {
    "value": _probe_value, "advantage": _probe_advantage, "loss": _probe_loss,
    "rollup": _probe_rollup, "return": _probe_return, "reward": _probe_reward,
    "regularization": _probe_regularization, "objective": _probe_objective,
    "gradient": _probe_gradient, "metric": _probe_metric, "action": _probe_action,
    "synthesis": _probe_synthesis,
}


def verify_function(
    family: str, instance: Any, *, name: str | None = None, ctx: ProbeContext | None = None,
) -> list[Finding]:
    """Run the family's probe on a *built* instance. Empty means it behaves."""
    probe = PROBES.get(family)
    if probe is None:
        raise ComponentContractError(f"rl_function.{family}", name or "?", [f"no probe for family {family!r}"])
    declared = getattr(instance, "probe", None)
    declared = dict(declared) if isinstance(declared, Mapping) else {}
    requested = frozenset(str(code) for code in declared.get("skip", ()) or ())
    label = name or getattr(instance, "name", None) or type(instance).__name__
    run = _Run(family, str(label), requested & EXEMPTABLE)
    for code in sorted(requested - EXEMPTABLE):
        run.findings.append(Finding("F011", family, str(label),
                                    f"probe['skip'] names {code}, which cannot be skipped "
                                    f"(only {sorted(EXEMPTABLE)} can)"))
    probe(instance, (ctx or ProbeContext()).adjusted(declared), run)
    return run.findings


def check_function(
    family: str, instance: Any, *, name: str | None = None, ctx: ProbeContext | None = None,
) -> None:
    """``verify_function`` that raises ``ComponentContractError`` (``E047``) on any finding."""
    findings = verify_function(family, instance, name=name, ctx=ctx)
    if findings:
        raise ComponentContractError(
            f"rl_function.{family}", name or type(instance).__name__,
            [finding.render() for finding in findings], code="E047",
        )
