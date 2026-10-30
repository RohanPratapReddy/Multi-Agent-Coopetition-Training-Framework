"""Parameter-free output views (document 14 section 5).

A teacher is a different model, so its output lives in a different space. A
contrastive loss is only defined once both sides are vectors of the same width
whose distance means the same thing. An ``OutputView`` is that adapter and it is
the only place where heterogeneity is handled.

Every view is a pure function of frozen tensors. A view owns no
``nn.Parameter`` and is never an ``nn.Module``, so it can never become a gradient
path between two peers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

from ..exceptions import ContractError, ValidationError
from .labelmap import LabelMap

if TYPE_CHECKING:  # pragma: no cover
    from .teacher import TeacherOutput

__all__ = [
    "DistributionView",
    "FixedRandomView",
    "HiddenStateView",
    "IdentityView",
    "LinearView",
    "OutputBundle",
    "OutputView",
    "VIEW_KINDS",
    "ViewPair",
    "build_view",
    "build_view_pair",
    "peer_bundle",
    "teacher_bundle",
]


@dataclass
class OutputBundle:
    """What a view may read, for either side."""

    logits: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    extra: Mapping[str, torch.Tensor] = field(default_factory=dict)
    source: str = "peer"  # "peer" | "teacher"
    side: str = "?"  # peer path or teacher name


@runtime_checkable
class OutputView(Protocol):
    """One side's map from an output bundle to a ``(B, d)`` float tensor."""

    dim: int

    def __call__(self, bundle: OutputBundle) -> torch.Tensor: ...

    def describe(self) -> dict[str, Any]: ...


def peer_bundle(logits: torch.Tensor, *, path: str) -> OutputBundle:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return OutputBundle(logits=logits, extra={"logits": logits}, source="peer", side=path)


def teacher_bundle(output: "TeacherOutput", *, name: str) -> OutputBundle:
    return OutputBundle(
        logits=output.logits,
        hidden=output.hidden,
        probs=output.probs,
        extra=dict(output.extra),
        source="teacher",
        side=name,
    )


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor


class IdentityView:
    """The output is already the embedding. Rejects a width mismatch."""

    name = "identity"

    def __init__(self, dim: int | None = None, *, key: str | None = None) -> None:
        self.dim = int(dim) if dim is not None else -1
        self.key = key

    def __call__(self, bundle: OutputBundle) -> torch.Tensor:
        tensor = bundle.extra.get(self.key) if self.key else None
        if tensor is None:
            tensor = bundle.hidden if bundle.hidden is not None else bundle.logits
        if tensor is None:
            raise ContractError(f"IdentityView on {bundle.source} {bundle.side!r} found no tensor")
        tensor = _flatten(tensor)
        if self.dim not in (-1, tensor.shape[-1]):
            raise ContractError(
                f"IdentityView on {bundle.source} {bundle.side!r} expects width "
                f"{self.dim} but the output is {tensor.shape[-1]}; use a projecting view"
            )
        self.dim = int(tensor.shape[-1])
        return tensor

    def describe(self) -> dict[str, Any]:
        return {"view": self.name, "dim": self.dim, "key": self.key}


class LinearView:
    """A frozen linear map ``x @ W + b``.

    ``weight`` may be supplied (a checkpoint carried a projection), or generated
    from ``seed``. The matrix is a plain ``torch.Tensor`` with
    ``requires_grad=False``, never an ``nn.Parameter``.
    """

    name = "linear"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        seed: int | None = None,
        normalize: bool = True,
    ) -> None:
        if in_dim < 1 or out_dim < 1:
            raise ValidationError(f"LinearView needs positive dimensions, got {in_dim} -> {out_dim}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.dim = self.out_dim
        self.normalize = bool(normalize)
        if weight is None:
            if seed is None:
                raise ValidationError("LinearView requires either 'weight' or 'seed'")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            weight = torch.randn(self.in_dim, self.out_dim, generator=generator)
        weight = torch.as_tensor(weight, dtype=torch.float32).detach().clone()
        if tuple(weight.shape) != (self.in_dim, self.out_dim):
            raise ValidationError(
                f"LinearView weight has shape {tuple(weight.shape)}, expected "
                f"{(self.in_dim, self.out_dim)}"
            )
        if self.normalize:
            # One global scale, not per-column: per-column scaling would change
            # the relative geometry of the embedding, which is the one thing the
            # contrastive loss reads.
            scale = float(weight.pow(2).sum().sqrt().clamp_min(1e-8)) / math.sqrt(
                max(1, self.in_dim)
            )
            weight = weight / max(scale, 1e-8)
        self.weight = weight.requires_grad_(False)
        if bias is None:
            self.bias: torch.Tensor | None = None
        else:
            bias = torch.as_tensor(bias, dtype=torch.float32).detach().clone()
            if tuple(bias.shape) not in {(), (self.out_dim,)}:
                raise ValidationError(
                    f"LinearView bias has shape {tuple(bias.shape)}, expected ({self.out_dim},)"
                )
            self.bias = bias.reshape(-1).requires_grad_(False)

    def __call__(self, bundle: OutputBundle) -> torch.Tensor:
        tensor = bundle.hidden if bundle.hidden is not None else bundle.logits
        if tensor is None:
            raise ContractError(f"LinearView on {bundle.source} {bundle.side!r} found no tensor")
        tensor = _flatten(tensor)
        if tensor.shape[-1] != self.in_dim:
            raise ContractError(
                f"LinearView on {bundle.source} {bundle.side!r} expects width "
                f"{self.in_dim} but the output is {tensor.shape[-1]}"
            )
        weight = self.weight.to(device=tensor.device, dtype=tensor.dtype)
        projected = tensor @ weight
        if self.bias is not None:
            projected = projected + self.bias.to(device=tensor.device, dtype=tensor.dtype)
        return projected

    def parameters(self) -> tuple[torch.Tensor, ...]:
        """Present so the ``no trainable parameters`` test is meaningful."""
        return (self.weight,) + ((self.bias,) if self.bias is not None else ())

    def describe(self) -> dict[str, Any]:
        return {
            "view": self.name,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "has_bias": self.bias is not None,
            "normalize": self.normalize,
        }


class FixedRandomView(LinearView):
    """``LinearView`` whose matrix is generated from a recorded seed.

    The seed is ``node_seed(run_seed, f"view:{side}")`` by default, so the map is
    reproducible across a resume and differs per side. A peer and a teacher must
    not share a random projection or the contrast is trivially aligned.
    """

    name = "fixed_random"

    def __init__(self, in_dim: int, out_dim: int, *, seed: int, normalize: bool = True) -> None:
        self.seed = int(seed)
        super().__init__(in_dim, out_dim, seed=seed, normalize=normalize)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "seed": self.seed}


class HiddenStateView:
    """Reads a named hidden tensor, pools the time axis, then projects if needed.

    Sequence models produce ``(B, L, d)``. The contrastive unit is one sample, so
    the time axis is pooled. Mean pooling is the default: a teacher that emits
    per-token states has no single "answer token" contract. ``last`` is available
    for models whose contract really is "the final token".
    """

    name = "hidden"

    def __init__(
        self,
        key: str,
        *,
        out_dim: int | None = None,
        pool: str = "mean",
        seed: int | None = None,
        weight: torch.Tensor | None = None,
    ) -> None:
        if pool not in {"mean", "last", "first", "flat"}:
            raise ValidationError(f"HiddenStateView pool must be mean/last/first/flat, got {pool!r}")
        self.key = key
        self.pool = pool
        self.out_dim = int(out_dim) if out_dim else -1
        self.dim = self.out_dim
        self._inner: LinearView | None = None
        self._seed = seed
        self._weight = weight

    def _select(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim != 3:
            raise ContractError(
                f"HiddenStateView {self.key!r} expects a 2-D or 3-D tensor, got "
                f"{tuple(tensor.shape)}"
            )
        if self.pool == "mean":
            return tensor.mean(dim=1)
        if self.pool == "last":
            return tensor[:, -1, :]
        if self.pool == "first":
            return tensor[:, 0, :]
        return tensor.reshape(tensor.shape[0], -1)

    def __call__(self, bundle: OutputBundle) -> torch.Tensor:
        tensor = bundle.extra.get(self.key)
        if tensor is None:
            raise ContractError(
                f"HiddenStateView {self.key!r} on {bundle.source} {bundle.side!r} found "
                "no such key; available: " + (", ".join(sorted(bundle.extra)) or "(none)")
            )
        pooled = self._select(tensor.float())
        if self.out_dim in (-1, pooled.shape[-1]):
            self.dim = int(pooled.shape[-1])
            return pooled
        if self._inner is None:
            self._inner = LinearView(
                int(pooled.shape[-1]), self.out_dim, weight=self._weight, seed=self._seed
            )
        return self._inner(OutputBundle(logits=pooled, source=bundle.source, side=bundle.side))

    @property
    def in_dim(self) -> int | None:
        """The width of the projection input, once known."""
        return None if self._inner is None else self._inner.in_dim

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return () if self._inner is None else self._inner.parameters()

    def describe(self) -> dict[str, Any]:
        return {"view": self.name, "key": self.key, "pool": self.pool, "out_dim": self.dim}


class DistributionView:
    """Restricts logits to shared labels and returns ``log_softmax(logits / T)``.

    ``T`` is the teacher's own calibration temperature, not the contrastive
    ``tau``. The two must not be confused, so the parameter is ``calibration``.
    """

    name = "distribution"

    def __init__(
        self,
        labels: Sequence[str],
        *,
        calibration: float = 1.0,
        label_map: LabelMap | None = None,
        key: str = "logits",
    ) -> None:
        if calibration <= 0:
            raise ValidationError(f"DistributionView calibration must be > 0, got {calibration!r}")
        self.labels = tuple(str(label) for label in labels)
        self.dim = len(self.labels)
        self.calibration = float(calibration)
        self.key = key
        self.label_map = label_map
        if label_map is not None and len(label_map.axis) != self.dim:
            raise ValidationError(
                f"DistributionView has {self.dim} labels but its LabelMap axis has "
                f"{len(label_map.axis)}"
            )

    def __call__(self, bundle: OutputBundle) -> torch.Tensor:
        logits = bundle.extra.get(self.key)
        if logits is None:
            logits = bundle.logits
        if logits is None:
            raise ContractError(
                f"DistributionView on {bundle.source} {bundle.side!r} found no logits"
            )
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        if self.label_map is not None:
            # ``LabelMap`` returns log-probabilities already, so the calibration
            # fold is skipped: re-applying log_softmax would double-normalise.
            return self.label_map(logits)
        if logits.shape[-1] != self.dim:
            raise ContractError(
                f"DistributionView on {bundle.source} {bundle.side!r} expects "
                f"{self.dim} shared labels but received {logits.shape[-1]}; supply a LabelMap"
            )
        return torch.log_softmax(logits.float() / self.calibration, dim=-1)

    def describe(self) -> dict[str, Any]:
        return {
            "view": self.name,
            "labels": len(self.labels),
            "calibration": self.calibration,
            "mapped": self.label_map is not None,
        }


@dataclass(frozen=True)
class ViewPair:
    """The two sides of one ``(peer, teacher)`` contrast, at one shared width."""

    peer: OutputView
    teacher: OutputView
    dim: int

    def __post_init__(self) -> None:
        peer_dim = int(getattr(self.peer, "dim", -1))
        teacher_dim = int(getattr(self.teacher, "dim", -1))
        target = int(self.dim)
        if target < 1:
            raise ValidationError(f"ViewPair dim must be >= 1, got {self.dim!r}")
        for side, resolved in (("peer", peer_dim), ("teacher", teacher_dim)):
            if resolved not in (-1, target):
                raise ValidationError(
                    f"ViewPair {side} view resolves to width {resolved}, but the pair is "
                    f"declared at width {target}; give both sides the same out_dim, or "
                    "use DistributionView over the same LabelMap"
                )

    def describe(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "peer": self.peer.describe(),
            "teacher": self.teacher.describe(),
        }


VIEW_KINDS: frozenset[str] = frozenset(
    {"identity", "linear", "fixed_random", "hidden", "distribution"}
)


def build_view(
    *,
    side: str,
    spec: Mapping[str, Any] | None,
    in_dim: int | None,
    out_dim: int,
    seed: int,
    labels: Sequence[str] | None = None,
    label_map: LabelMap | None = None,
) -> OutputView:
    """Build one side's view. ``None`` or ``{}`` means identity.

    Only the keys a view reads are consumed. The caller is a config loader, and a
    teacher's ``view`` mapping may also carry keys read elsewhere, so unknown
    *extra* keys are ignored while an unknown ``kind`` fails.
    """
    payload = dict(spec or {})
    kind = str(payload.pop("kind", payload.pop("type", "identity"))).lower()
    if kind not in VIEW_KINDS:
        raise ValidationError(
            f"view kind {kind!r} for {side!r} is unknown; accepted: " + ", ".join(sorted(VIEW_KINDS))
        )
    if kind == "identity":
        return IdentityView(payload.pop("dim", out_dim), key=payload.pop("key", None))
    if kind == "linear":
        if in_dim is None:
            raise ValidationError(
                f"linear view for {side!r} needs in_dim, which is only known after a "
                "forward pass; pass the observed width or use kind='fixed_random'"
            )
        return LinearView(
            int(in_dim),
            int(payload.pop("out_dim", out_dim)),
            weight=payload.pop("weight", None),
            bias=payload.pop("bias", None),
            seed=int(payload.pop("seed", seed)),
            normalize=bool(payload.pop("normalize", True)),
        )
    if kind == "fixed_random":
        if in_dim is None:
            raise ValidationError(
                f"fixed_random view for {side!r} needs in_dim (the observed output width)"
            )
        return FixedRandomView(
            int(in_dim),
            int(payload.pop("out_dim", out_dim)),
            seed=int(payload.pop("seed", seed)),
            normalize=bool(payload.pop("normalize", True)),
        )
    if kind == "hidden":
        key = payload.pop("key", None)
        if not key:
            raise ValidationError(f"hidden view for {side!r} requires 'key'")
        requested = payload.pop("out_dim", out_dim)
        return HiddenStateView(
            str(key),
            out_dim=int(requested) if requested is not None else None,
            pool=str(payload.pop("pool", "mean")),
            seed=int(payload.pop("seed", seed)),
            weight=payload.pop("weight", None),
        )
    resolved = labels or tuple(payload.pop("labels", None) or ())
    if not resolved:
        raise ValidationError(
            f"distribution view for {side!r} requires 'labels' (the shared label names)"
        )
    return DistributionView(
        resolved,
        calibration=float(payload.pop("calibration", 1.0)),
        label_map=label_map,
        key=str(payload.pop("key", "logits")),
    )


def build_view_pair(
    *,
    peer: str,
    teacher: str,
    peer_spec: Mapping[str, Any] | None,
    teacher_spec: Mapping[str, Any] | None,
    peer_in_dim: int | None,
    teacher_in_dim: int | None,
    out_dim: int,
    peer_seed: int,
    teacher_seed: int,
    labels: Sequence[str] | None = None,
    peer_label_map: LabelMap | None = None,
    teacher_label_map: LabelMap | None = None,
) -> ViewPair:
    """Build and width-check both sides of one ``(peer, teacher)`` contrast.

    The two seeds are separate on purpose: a peer and a teacher that shared a
    random projection would make the contrast trivially aligned.
    """
    if peer_seed == teacher_seed:
        raise ValidationError(
            f"peer {peer!r} and teacher {teacher!r} would share a projection seed "
            f"({peer_seed}); each side needs its own"
        )
    peer_view = build_view(
        side=f"peer:{peer}",
        spec=peer_spec,
        in_dim=peer_in_dim,
        out_dim=out_dim,
        seed=peer_seed,
        labels=labels,
        label_map=peer_label_map,
    )
    teacher_view = build_view(
        side=f"teacher:{teacher}",
        spec=teacher_spec,
        in_dim=teacher_in_dim,
        out_dim=out_dim,
        seed=teacher_seed,
        labels=labels,
        label_map=teacher_label_map,
    )
    return ViewPair(peer=peer_view, teacher=teacher_view, dim=out_dim)
