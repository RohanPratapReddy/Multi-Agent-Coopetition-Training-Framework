"""Constructor reflection, canonical JSON and spec hashing (document 17 section 3.3).

Pure standard library. ``canonical_json`` and ``spec_hash`` are what a checkpoint
fingerprint is built from, so their output is part of a compatibility promise: sorted
keys, no whitespace, floats through ``repr``.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Reflection:
    keys: frozenset[str]
    required: frozenset[str]
    var_keyword: bool


def reflect(factory: Any) -> Reflection:
    """Constructor keyword reflection for a class, function or ``functools.partial``.

    For a class, ``__init__`` signatures are read along the MRO. A base class is
    consulted only while the class before it takes ``**kwargs`` (it forwards), which is
    the rule ``mactf.scoring.base._constructor_keys`` and
    ``mactf.rl.functions._spec.constructor_keys`` already apply, so the three agree.
    ``var_keyword`` is true when the chain is still forwarding ``**kwargs`` after the
    last real ``__init__``: the factory then accepts arbitrary keys.
    """
    bound: frozenset[str] = frozenset()
    target = factory
    if isinstance(target, functools.partial):
        bound = frozenset(target.keywords)
        target = target.func
    keys: set[str] = set()
    required: set[str] = set()
    var_keyword = False
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    if inspect.isclass(target):
        forwarding = False
        for klass in target.__mro__:
            if klass is object:
                continue
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            forwarding = False
            try:
                parameters = list(inspect.signature(init).parameters.values())[1:]
            except (TypeError, ValueError):
                # A C-level ``__init__`` with no signature: nothing can be said about it.
                forwarding = True
                break
            for parameter in parameters:
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    forwarding = True
                elif parameter.kind in named and parameter.name not in keys:
                    keys.add(parameter.name)
                    if parameter.default is inspect.Parameter.empty:
                        required.add(parameter.name)
            if not forwarding:
                break
        var_keyword = forwarding
    else:
        try:
            parameters = list(inspect.signature(target).parameters.values())
        except (TypeError, ValueError):
            parameters = []
            var_keyword = True
        for parameter in parameters:
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                var_keyword = True
            elif parameter.kind in named:
                keys.add(parameter.name)
                if parameter.default is inspect.Parameter.empty:
                    required.add(parameter.name)
    return Reflection(
        keys=frozenset(keys) - bound,
        required=frozenset(required) - bound,
        var_keyword=var_keyword,
    )


def constructor_keys(factory: Any) -> frozenset[str]:
    return reflect(factory).keys


def canonical_json(value: Any) -> str:
    """Stable JSON: sorted keys, no whitespace, floats via ``repr``."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)


def _default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if hasattr(value, "value") and hasattr(value, "name"):  # enum
        return value.value
    return repr(value)


def spec_hash(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(spec)).encode("utf-8")).hexdigest()[:16]


def source_digest(factory: Any) -> str | None:
    """sha256 of the factory's source text, or ``None`` when it is not available
    (built-in extension, REPL, ``exec``). Only computed for non-built-in components.
    """
    target = factory.func if isinstance(factory, functools.partial) else factory
    try:
        text = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
