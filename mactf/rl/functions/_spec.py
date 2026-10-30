"""Shared ``from_spec`` machinery for the function wrappers (document 07 section 10).

Every wrapper accepts ``{"type": "<registry key>", **kwargs}`` and every spec is
validated against the real constructor signature rather than ``**kwargs``. A typo
must be a ``ValidationError`` at construction, never a silently different
objective trained for four thousand steps.
"""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, ClassVar, Mapping

from ...exceptions import ValidationError


def constructor_keys(cls: type) -> frozenset[str]:
    """The keyword names ``cls.__init__`` accepts, resolved through the MRO.

    ``**kwargs`` is treated as accepting nothing. Every wrapper here declares its
    parameters explicitly, so this keeps validation strict instead of vacuous.
    """
    accepted: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None or init is object.__init__:
            continue
        for name, parameter in signature(init).parameters.items():
            if name == "self":
                continue
            if parameter.kind in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL):
                continue
            accepted.add(name)
        break
    return frozenset(accepted)


def build_from_kwargs(cls: type, kwargs: Mapping[str, Any]) -> Any:
    """Construct ``cls(**kwargs)``, rejecting anything it cannot take."""
    accepted = constructor_keys(cls)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValidationError(
            f"{cls.__name__} does not accept "
            + ", ".join(repr(key) for key in unknown)
            + f"; accepted keys: {', '.join(sorted(accepted)) or '(none)'}"
        )
    return cls(**dict(kwargs))


class SpecMixin:
    """``from_spec`` for every wrapper class.

    ``spec_family`` names the registry table the class belongs to, so
    ``GAEAdvantage.from_spec({"type": "gae"})`` resolves without the caller
    naming the family.
    """

    spec_family: ClassVar[str] = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Any:
        from .registry import build_function

        return build_function(spec, family=cls.spec_family or None)

    def describe(self) -> dict[str, Any]:
        return {
            "type": getattr(self, "spec_type", type(self).__name__),
            "class": type(self).__name__,
        }


__all__ = ["SpecMixin", "build_from_kwargs", "constructor_keys"]
