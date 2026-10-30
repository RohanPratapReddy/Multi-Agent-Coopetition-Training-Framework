"""Frozen, output-only teachers (document 14 section 4)."""

from __future__ import annotations

import hashlib
import importlib
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from ..exceptions import ContractError, ImproperlyConfigured, ValidationError
from ..rl.functions._spec import build_from_kwargs
from ..runtime import Runtime, load_checkpoint, resolve_device, resolve_dtype
from .conformance import TrustProfile, build_trust_profile

__all__ = [
    "CallableModule",
    "TEACHER_KINDS",
    "Teacher",
    "TeacherOutput",
    "TeacherSpec",
    "build_teacher",
    "normalize_output",
]

TEACHER_KINDS: frozenset[str] = frozenset({"module", "callable", "state_dict", "transformers"})

#: Extra distribution name per optional dependency a spec may require.
_OPTIONAL_IMPORTS: dict[str, str] = {
    "transformers": "transformers",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
}


@dataclass(frozen=True)
class TeacherSpec:
    """Declarative description of one teacher model.

    ``kind`` selects the loader; the four loaders are the four ways a model
    actually arrives in a project directory.

    * ``module``: ``factory(**kwargs)`` returns an ``nn.Module``.
    * ``callable``: ``factory`` is a function ``states -> output``; ``kwargs`` are
      bound with ``functools.partial``. It is wrapped in a parameter-free module.
    * ``state_dict``: ``arch(**kwargs)`` is built, then ``path`` is loaded into it.
    * ``transformers``: ``from_pretrained(path or factory, **kwargs)``.
    """

    name: str
    kind: str = "module"
    factory: str | None = None
    path: str | None = None
    arch: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    view: Mapping[str, Any] = field(default_factory=dict)
    label_names: tuple[str, ...] = ()
    requires: str | None = None
    device: str | None = None
    dtype: str | None = None
    weight: float = 1.0
    hidden: str | None = None
    trust: Mapping[str, Any] | TrustProfile | None = None
    audit: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValidationError(
                f"teacher name {self.name!r} must be alphanumeric with '_' or '-'"
            )
        if self.kind not in TEACHER_KINDS:
            raise ValidationError(
                f"teacher {self.name!r} has unknown kind {self.kind!r}; accepted: "
                + ", ".join(sorted(TEACHER_KINDS))
            )
        if self.kind in {"module", "callable"} and not self.factory:
            raise ValidationError(f"teacher {self.name!r} of kind {self.kind!r} requires 'factory'")
        if self.kind == "state_dict" and not (self.path and self.arch):
            raise ValidationError(
                f"teacher {self.name!r} of kind 'state_dict' requires both 'path' and 'arch'"
            )
        if self.kind == "transformers" and not (self.path or self.factory):
            raise ValidationError(
                f"teacher {self.name!r} of kind 'transformers' requires 'path' (or 'factory') "
                "naming the pretrained model"
            )
        if not self.weight > 0.0:
            raise ValidationError(f"teacher {self.name!r} weight must be > 0, got {self.weight!r}")
        # Resolve and validate the trust profile here rather than at step time, so
        # a widening attempt fails when the teacher is declared and not hundreds
        # of steps into a run. `TrustProfile.__post_init__` enforces I7, and
        # `build_trust_profile` stamps `declared` from the pre-substitution value
        # so STRICT_UNDECLARED can still see that nothing was written.
        object.__setattr__(
            self, "trust", build_trust_profile(self.trust, declared=self.trust is not None)
        )

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any] | "TeacherSpec") -> "TeacherSpec":
        if isinstance(spec, TeacherSpec):
            return spec
        payload = dict(spec or {})
        name = payload.pop("name", None) or payload.pop("id", None)
        if not name:
            raise ValidationError("a teacher spec requires 'name'")
        labels = payload.get("label_names") or ()
        if isinstance(labels, str):
            labels = (labels,)
        payload["label_names"] = tuple(str(item) for item in labels)
        payload["kwargs"] = dict(payload.get("kwargs") or {})
        payload["view"] = dict(payload.get("view") or {})
        if isinstance(payload.get("trust"), Mapping):
            payload["trust"] = dict(payload["trust"])
        # A TrustProfile passes through; anything else (a bare float) reaches
        # build_trust_profile, which rejects it with a message naming the fix.
        return build_from_kwargs(cls, {**payload, "name": str(name)})

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "factory": self.factory,
            "path": self.path,
            "arch": self.arch,
            "view": dict(self.view),
            "weight": self.weight,
            "requires": self.requires,
            "trust": self.trust.describe() if isinstance(self.trust, TrustProfile) else None,
            "audit": self.audit,
        }


@dataclass
class TeacherOutput:
    """Everything a view may read from a teacher's forward pass."""

    logits: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    extra: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def features(self, key: str | None = None) -> torch.Tensor:
        """The tensor a view consumes when it is not vocabulary-aware."""
        if key is not None:
            candidate = self.extra.get(key)
            if candidate is None:
                raise ContractError(
                    f"teacher output has no key {key!r}; available: "
                    + (", ".join(sorted(self.extra)) or "(none)")
                )
            return candidate
        for candidate in (self.hidden, self.logits, self.probs):
            if candidate is not None:
                return candidate
        raise ContractError("teacher output carries no tensor to project from")


class CallableModule(torch.nn.Module):
    """Adapts a plain ``states -> output`` function to the ``nn.Module`` contract.

    It owns no parameters, so ``freeze()`` and the digest still apply uniformly.
    """

    def __init__(self, function: Callable[..., Any], *, label: str = "callable") -> None:
        super().__init__()
        if not callable(function):
            raise ContractError(f"teacher callable {label!r} is not callable")
        self._function = function
        self._label = label

    def forward(self, states: torch.Tensor) -> Any:
        return self._function(states)

    def extra_repr(self) -> str:
        return f"callable={self._label}"


class Teacher:
    """A frozen, output-only model. It is never an aggregation participant."""

    def __init__(
        self,
        module: torch.nn.Module,
        spec: TeacherSpec,
        *,
        runtime: Runtime,
        hidden_key: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.module = module
        self.spec = spec
        self.runtime = runtime
        self.hidden_key = hidden_key or spec.hidden
        self.device = device if device is not None else runtime.device
        self.dtype = dtype if dtype is not None else runtime.dtype
        self.frozen = False
        self._digest: str | None = None
        self._calls = 0
        self._cache_key: tuple[Any, ...] | None = None
        self._cache: TeacherOutput | None = None

    # -- lifecycle -------------------------------------------------------
    def freeze(self) -> "Teacher":
        """Idempotent: eval mode, no grads, no cached forward."""
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.module.to(device=self.device)
        floating = [p for p in self.module.parameters() if p.is_floating_point()]
        if floating:
            self.module.to(dtype=self.dtype)
        self.frozen = True
        self._cache = None
        self._cache_key = None
        return self

    @property
    def digest(self) -> str:
        """Stable content hash, computed once.

        Covers the class path, every parameter name/shape/dtype, and the raw
        bytes of every parameter and buffer. Used by the resume check so a
        swapped teacher file cannot silently change a resumed run's objective.
        """
        if self._digest is not None:
            return self._digest
        hasher = hashlib.sha256()
        hasher.update(
            f"{type(self.module).__module__}.{type(self.module).__qualname__}".encode()
        )
        for name, parameter in sorted(self.module.named_parameters()):
            _hash_tensor(hasher, name, parameter)
        for name, buffer in sorted(self.module.named_buffers()):
            _hash_tensor(hasher, "buffer:" + name, buffer)
        self._digest = hasher.hexdigest()
        return self._digest

    # -- forward ---------------------------------------------------------
    def __call__(self, states: torch.Tensor, *, reset: bool = False) -> TeacherOutput:
        return self.forward(states, reset=reset)

    def forward(self, states: torch.Tensor, *, reset: bool = False) -> TeacherOutput:
        """One no-grad pass. ``reset=True`` discards the previous cache."""
        if not self.frozen:
            # A caller that forgot freeze() would otherwise train the teacher via
            # the contrastive term, which is the single worst failure mode here.
            raise ContractError(
                f"teacher {self.spec.name!r} was called before freeze(); call "
                "Teacher.freeze() during setup"
            )
        # Invariant I1: within one step there is exactly one batch tensor, so an
        # identity match is a batch match. The version counter additionally
        # catches an in-place edit of a reused buffer.
        key = (int(states.data_ptr()), tuple(states.shape), states.dtype, int(states._version))
        if not reset and self._cache is not None and self._cache_key == key:
            return self._cache
        if states.is_floating_point():
            moved = states.to(device=self.device, dtype=self.dtype)
        else:
            moved = states.to(device=self.device)
        with torch.no_grad():
            raw = self.module(moved)
        output = normalize_output(raw, hidden_key=self.hidden_key, teacher=self.spec.name)
        self._calls += 1
        self._cache = output
        self._cache_key = key
        return output

    def clear_cache(self) -> None:
        self._cache = None
        self._cache_key = None

    def describe(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.module.parameters())
        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        return {
            **self.spec.describe(),
            "class": f"{type(self.module).__module__}.{type(self.module).__qualname__}",
            "parameters": total,
            "trainable_parameters": trainable,
            "frozen": self.frozen,
            "digest": self.digest,
            "calls": self._calls,
        }


def _hash_tensor(hasher: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().to(device="cpu").contiguous()
    hasher.update(name.encode())
    hasher.update(str(tuple(value.shape)).encode())
    hasher.update(str(value.dtype).encode())
    if value.numel():
        hasher.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def normalize_output(
    raw: Any, *, hidden_key: str | None = None, teacher: str = "?"
) -> TeacherOutput:
    """Split an arbitrary module output into logits / hidden / probs.

    Accepts a tensor, a ``(logits, value)`` tuple, a dict, or a ``ModelOutput``-like
    object. The accepted shapes are exactly the ones ``PeerAdapter`` accepts; the
    divergence is that a value head is never synthesized here, because a
    ``PeerAdapter`` registers trainable parameters and a teacher must never own an
    optimizer.
    """

    def first_tensor(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                found = first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, Mapping):
            for key in sorted(value):
                found = first_tensor(value[key])
                if found is not None:
                    return found
        return None

    extra: dict[str, torch.Tensor] = {}
    hidden: torch.Tensor | None

    if torch.is_tensor(raw):
        logits = raw
        hidden = None
    elif isinstance(raw, Mapping):
        logits = raw.get("logits", raw.get("policy_logits", raw.get("action_logits")))
        hidden = raw.get("hidden", raw.get("last_hidden_state"))
        probs = raw.get("probs", raw.get("probabilities"))
        for key in sorted(raw, key=str):
            if torch.is_tensor(raw[key]):
                extra[str(key)] = raw[key]
        if logits is None and hidden is None and probs is None:
            logits = first_tensor(raw)
        if probs is not None:
            extra.setdefault("probs", probs)
    elif isinstance(raw, (tuple, list)):
        if not raw:
            raise ContractError(f"teacher {teacher!r} returned an empty {type(raw).__name__}")
        logits = raw[0] if torch.is_tensor(raw[0]) else first_tensor(raw)
        hidden = raw[1] if len(raw) > 1 and torch.is_tensor(raw[1]) else None
        if len(raw) > 2 and torch.is_tensor(raw[2]):
            extra["hidden3"] = raw[2]
    else:
        logits = getattr(raw, "logits", None)
        hidden = getattr(raw, "last_hidden_state", None)
        probs = getattr(raw, "probs", None)
        if probs is not None:
            extra["probs"] = probs

    if logits is None and hidden is None:
        fallback = first_tensor(raw)
        if fallback is None:
            raise ContractError(
                f"teacher {teacher!r} returned {type(raw).__name__}, which carries no tensor; "
                "return a tensor, a (logits, ...) tuple, a dict with 'logits', or a "
                "ModelOutput-like object"
            )
        logits = fallback

    if logits is not None:
        if logits.ndim == 0:
            logits = logits.reshape(1, 1)
        elif logits.ndim == 1:
            logits = logits.unsqueeze(0)
        extra.setdefault("logits", logits)
    if hidden is not None and hidden.ndim == 1:
        hidden = hidden.unsqueeze(0)
    if hidden is not None:
        extra.setdefault("hidden", hidden)
    if hidden_key is not None:
        chosen = extra.get(hidden_key)
        if chosen is None:
            raise ContractError(
                f"teacher {teacher!r} was configured with hidden key {hidden_key!r} but "
                "its output has no such key; available: " + (", ".join(sorted(extra)) or "(none)")
            )
        hidden = chosen
    return TeacherOutput(logits=logits, hidden=hidden, probs=extra.get("probs"), extra=extra)


def _import_dotted(path: str | None) -> Any:
    """Resolve ``package.module.attr`` or ``package.module:attr``."""
    if not path:
        raise ImproperlyConfigured("a dotted import path is required")
    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ImproperlyConfigured(f"{path!r} is not a dotted path to an attribute")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImproperlyConfigured(f"cannot import {module_name!r} for {path!r}: {exc}") from exc
    target: Any = module
    for part in attribute.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise ImproperlyConfigured(f"{module_name!r} has no attribute {attribute!r}") from exc
    return target


def build_teacher(
    spec: TeacherSpec | Mapping[str, Any],
    *,
    runtime: Runtime,
    module: torch.nn.Module | None = None,
) -> Teacher:
    """Construct and freeze one teacher. Optional extras fail loudly."""
    spec = TeacherSpec.from_spec(spec)
    required = spec.requires or ("transformers" if spec.kind == "transformers" else None)
    if required:
        try:
            importlib.import_module(_OPTIONAL_IMPORTS.get(required, required))
        except ImportError as exc:
            raise ImproperlyConfigured(
                f"teacher {spec.name!r} requires the optional dependency "
                f"{required!r}: pip install 'mactf[{required}]' ({exc})"
            ) from exc

    if module is None:
        module = _load_teacher_module(spec)

    if not isinstance(module, torch.nn.Module):
        raise ContractError(
            f"teacher {spec.name!r} factory returned {type(module).__name__}, expected nn.Module"
        )

    device = resolve_device(spec.device) if spec.device else runtime.device
    dtype = resolve_dtype(spec.dtype, device) if spec.dtype else runtime.dtype
    teacher = Teacher(module, spec, runtime=runtime, hidden_key=spec.hidden, device=device, dtype=dtype)
    teacher.freeze()
    if teacher.module.training:
        raise ContractError(f"teacher {spec.name!r} is still in training mode after freeze()")
    return teacher


def _load_teacher_module(spec: TeacherSpec) -> torch.nn.Module:
    kwargs = dict(spec.kwargs)
    if spec.kind == "module":
        target = _import_dotted(spec.factory)
        produced = target(**kwargs)
        if not isinstance(produced, torch.nn.Module):
            if callable(produced):
                raise ImproperlyConfigured(
                    f"teacher {spec.name!r} factory {spec.factory!r} returned a callable, "
                    "not an nn.Module; use kind='callable' for a plain function"
                )
            raise ContractError(
                f"teacher {spec.name!r} factory {spec.factory!r} returned "
                f"{type(produced).__name__}, expected nn.Module"
            )
        return produced
    if spec.kind == "callable":
        target = _import_dotted(spec.factory)
        function = functools.partial(target, **kwargs) if kwargs else target
        return CallableModule(function, label=str(spec.factory))
    if spec.kind == "state_dict":
        klass = _import_dotted(spec.arch)
        module = klass(**kwargs)
        if not isinstance(module, torch.nn.Module):
            raise ContractError(
                f"teacher {spec.name!r} arch {spec.arch!r} built {type(module).__name__}, "
                "expected nn.Module"
            )
        path = Path(str(spec.path))
        if not path.exists():
            raise ImproperlyConfigured(f"teacher {spec.name!r} checkpoint does not exist: {path}")
        payload = load_checkpoint(path, map_location="cpu")
        state = payload.get("state_dict", payload.get("model", payload))
        if not isinstance(state, Mapping):
            raise ContractError(
                f"teacher {spec.name!r} checkpoint {path} has no 'state_dict' mapping"
            )
        # strict=False is used for its report, not to tolerate a mismatch.
        missing, unexpected = module.load_state_dict(dict(state), strict=False)
        if missing or unexpected:
            raise ContractError(
                f"teacher {spec.name!r} checkpoint does not match {spec.arch!r}; "
                f"missing {len(missing)} key(s), unexpected {len(unexpected)} key(s); "
                f"first missing {list(missing)[:3]}, first unexpected {list(unexpected)[:3]}"
            )
        return module
    if spec.kind == "transformers":
        transformers = importlib.import_module("transformers")
        loader_name = str(kwargs.pop("loader", "AutoModel"))
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            raise ImproperlyConfigured(
                f"teacher {spec.name!r}: transformers has no loader {loader_name!r}"
            )
        return loader.from_pretrained(spec.path or spec.factory, **kwargs)
    raise ImproperlyConfigured(f"teacher {spec.name!r}: unhandled kind {spec.kind!r}")
