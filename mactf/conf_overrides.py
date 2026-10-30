"""``--set KEY=VALUE``: command-line settings overrides (document 16, section 5.2).

The grammar is deliberately narrower than YAML, because YAML 1.1 turns ``no`` into
``false``, ``1:30`` into the integer 90, and ``a: b`` into a mapping. A typo can no longer
invent a key no consumer reads, and a value of the wrong type fails before any validator.
"""

from __future__ import annotations

import copy
import difflib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import yaml

from .exceptions import CommandError

_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*")
_INT = re.compile(r"[-+]?\d+")
_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Override:
    path: tuple[str, ...]
    value: Any
    text: str                           # the original KEY=VALUE, for notes and errors


@dataclass(frozen=True)
class Written:
    trainer_keys: set[str]              # TRAINER keys this batch wrote


def parse_value(raw: str) -> Any:
    """Parse the right-hand side of ``KEY=VALUE``."""
    text = raw.strip()
    if text == "":
        return ""                       # KEY= sets the empty string; KEY=null sets null
    if text[0] in "[{\"'":
        try:
            return yaml.safe_load(text)  # flow lists/maps and quoted strings
        except yaml.YAMLError as exc:
            raise CommandError.usage(f"--set value {raw!r} is not valid YAML: {exc}") from exc
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "~"):
        return None
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    return raw                          # anything else is a string, verbatim


def parse_override(text: str) -> Override:
    """Split ``KEY=VALUE`` into a path and a parsed value."""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep:
        raise CommandError.usage(f"--set {text!r}: expected KEY=VALUE, for example --set TRAINING.EPOCHS=2")
    if not _KEY.fullmatch(key):
        raise CommandError.usage(
            f"--set {text!r}: {key!r} is not a dotted key; segments are letters, digits, '_' or '-'"
        )
    path = tuple(key.split("."))
    if path[0] != path[0].upper() or path[0].startswith("_"):
        raise CommandError.usage(
            f"--set {text!r}: the first segment must be an upper-case settings key, got {path[0]!r}"
        )
    return Override(path=path, value=parse_value(raw), text=text)


def _compatible(old: Any, new: Any) -> bool:
    if old is None or new is None:
        return True
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool)
    if isinstance(old, (int, float)):
        return isinstance(new, (int, float))
    if isinstance(old, str):
        return isinstance(new, str)
    if isinstance(old, dict):
        return isinstance(new, dict)
    if isinstance(old, (list, tuple)):
        return isinstance(new, (list, tuple))
    return True


def _suggest(word: str, candidates: Sequence[str]) -> str:
    close = difflib.get_close_matches(word, list(candidates), n=3, cutoff=0.6)
    return f"; did you mean {', '.join(close)}?" if close else ""


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _default_at(defaults: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = defaults
    for segment in path:
        if isinstance(node, Mapping) and segment in node:
            node = node[segment]
        else:
            return _MISSING
    return node


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


def is_open_container(defaults: Mapping[str, Any], container_path: Sequence[str]) -> bool:
    """A container accepts new keys when its default is absent or an empty mapping."""
    default = _default_at(defaults, container_path)
    return default is _MISSING or (isinstance(default, Mapping) and not default)


_is_open = is_open_container   # the private name the rest of this module already uses

#: The leaf names of a component spec (section 5.1). ``name`` is accepted by
#: ``spec_shape_problem`` but is not promoted: a spec is named by ``type`` or ``import``.
_SPEC_LEAVES = frozenset({"type", "import", "params"})


_TRAINER_SPEC_KEYS = frozenset({"PEER_STRATEGY", "WEIGHT_UPDATE", "POLICY", "AGGREGATOR"})
_TRAINER_TABLE_KEYS = frozenset({"PEER_STRATEGY_OVERRIDES", "WEIGHT_UPDATE_OVERRIDES"})


def is_spec_slot(path: Sequence[str]) -> bool:
    """Whether ``path`` addresses a value that is a component spec (a name or a mapping).

    ``TRAINER.WEIGHT_UPDATE``, ``TRAINER.PEER_STRATEGY_OVERRIDES.<glob>``, ``RL.LOSS``,
    ``RL.OVERRIDES.LOSS.<glob>`` and ``SCORING.STAGES.<kind>``.
    """
    head = [str(part) for part in path]
    if len(head) >= 2 and head[0] == "TRAINER":
        if len(head) == 2:
            return head[1] in _TRAINER_SPEC_KEYS
        return len(head) == 3 and head[1] in _TRAINER_TABLE_KEYS
    if head and head[0] == "RL":
        from .components.kinds import RL_FAMILIES

        families = {name.upper() for name in RL_FAMILIES}
        if len(head) == 2:
            return head[1].upper() in families
        return len(head) == 4 and head[1] == "OVERRIDES" and head[2].upper() in families
    return len(head) == 3 and head[:2] == ["SCORING", "STAGES"]


def _promotion(remaining: Sequence[str]) -> dict[str, Any] | None:
    """The mapping a scalar becomes so the remaining path can be written, or ``None``.

    ``("params", "temperature")`` -> ``{"params": {}}``
    ``("type",)``                -> ``{}``
    anything else                -> ``None`` (refused by the caller, as today)
    """
    if not remaining:
        return None
    head = remaining[0].lower()
    if head == "params" and len(remaining) > 1:     # ``params`` as the leaf is a mapping write
        return {"params": {}}
    if len(remaining) == 1 and head in {"type", "import"}:
        return {}
    return None


def apply_overrides(
    values: dict[str, Any], overrides: Sequence[Override], *, defaults: Mapping[str, Any]
) -> Written:
    """Write each override into ``values`` (in order); every failure is a usage error."""
    trainer_keys: set[str] = set()
    top_level = [key for key in values if not str(key).startswith("_")]
    for override in overrides:
        path = override.path
        first = path[0]
        if first not in values or first.startswith("_"):
            raise CommandError.usage(
                f"unknown setting {first!r} in --set {override.text!r}{_suggest(first, top_level)}"
            )
        container: Any = values
        walked: list[str] = []
        promoted_last = False           # the final container was a scalar spec placeholder
        for index, segment in enumerate(path[:-1]):
            promoted_last = False
            if not isinstance(container, dict):
                raise CommandError.usage(
                    f"--set {override.text!r}: {'.'.join(walked)} is not a mapping"
                )
            if segment not in container:
                if _is_open(defaults, walked):
                    container[segment] = {}
                else:
                    raise CommandError.usage(
                        f"{'.'.join(walked) or 'settings'} has no key {segment!r} in "
                        f"--set {override.text!r}{_suggest(segment, list(container))}"
                    )
            child = container[segment]
            walked.append(segment)
            if isinstance(child, (list, tuple)):
                raise CommandError.usage(
                    f"--set {override.text!r}: {'.'.join(walked)} is a list; "
                    f"set the whole list: --set '{'.'.join(walked)}=[...]'"
                )
            if not isinstance(child, dict):
                promoted = _promotion(path[index + 1 :])
                if promoted is None or not isinstance(child, (str, type(None))):
                    # A scalar is only a *spec* placeholder; a float or a bool is a leaf,
                    # and walking into it cannot mean anything.
                    raise CommandError.usage(
                        f"--set {override.text!r}: {'.'.join(walked)} is a "
                        f"{_type_name(child)}, not a mapping"
                    )
                if "params" in promoted and isinstance(child, str) and child:
                    # The name survives: ``--set X.params.k=v`` over ``X: uniform`` is
                    # ``{type: uniform, params: {k: v}}``, never a spec with no name.
                    promoted = {"type": child, **promoted}
                container[segment] = promoted
                child = promoted
                promoted_last = True
            container = child
        last = path[-1]
        if len(path) == 1:
            old = container[last]
        elif last in container:
            old = container[last]
        elif promoted_last or _is_open(defaults, walked):
            old = None
            container[last] = None
        else:
            raise CommandError.usage(
                f"{'.'.join(walked)} has no key {last!r} in --set {override.text!r}"
                f"{_suggest(last, list(container))}"
            )
        spec_slot = is_spec_slot(path) and isinstance(old, (str, dict, type(None)))
        if not (spec_slot and isinstance(override.value, (str, dict))) and not _compatible(old, override.value):
            raise CommandError.usage(
                f"--set {override.text!r}: {'.'.join(path)} is a {_type_name(old)} "
                f"({old!r}); {override.value!r} is a {_type_name(override.value)}"
            )
        container[last] = copy.deepcopy(override.value)
        if first == "TRAINER":
            if len(path) > 1:
                trainer_keys.add(path[1])
            elif isinstance(override.value, dict):
                trainer_keys.update(str(key) for key in override.value)
    return Written(trainer_keys=trainer_keys)
