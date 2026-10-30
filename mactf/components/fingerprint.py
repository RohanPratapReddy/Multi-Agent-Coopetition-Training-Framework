"""The checkpoint's record of which components a run used, and the diff of two records.

Document 17 section 14. The manifest is the *only* place the framework writes a component's
identity into a checkpoint, and ``diff`` is the only place it compares two. Both are pure
functions of a slot table, so both are unit-testable without a trainer, and neither imports
``torch``.

A run's components do not all live in one :class:`~mactf.components.slots.SlotTable`: the
scoring stages, the RL functions, the policy, the weight updates and the peer strategies do,
but the optimizer entries, the aggregator and the terms of a composed reward are held by their
own plans. :class:`SlotView` puts them behind the same two methods (``paths``, ``get``) so the
manifest, the restorer and the CLI read one thing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

from .kinds import Origin

log = logging.getLogger("mactf.components")

#: The shape of one entry. Bumped only if an entry gains a field a reader must understand.
ENTRY_SCHEMA = 1
#: The key the manifest lives under in ``identity``.
IDENTITY_KEY = "components"

#: Attributes on a component's *instance* that force it into the manifest even when it is a
#: built-in with default parameters. ``unsafe`` is how section 7.6 records a policy that runs
#: outside ``StepGuard``; ``record_in_checkpoint`` is the general hook.
RECORD_FLAGS: tuple[str, ...] = ("record_in_checkpoint", "unsafe")


def _is_builtin_class(component: Any) -> bool:
    """Whether a ``runtime`` component is an in-code instance of one of the framework's classes.

    ``SlotTable.adopt`` records every in-code instance with the ``runtime`` origin, including
    ``weight_update=UniformAverage()``; that is not an extension (the class is the framework's)
    and recording it would make a default-equivalent run write different bytes (P1).
    """
    if component.origin is not Origin.RUNTIME:
        return False
    try:
        from .registry import HUB

        return bool(HUB.registry(component.kind).is_builtin_class(component.factory))
    except Exception:  # noqa: BLE001 - an unknown kind has no built-ins
        return False


def _flagged(instance: Any, flag: str) -> bool:
    return getattr(instance, flag, False) is True


def records_in_checkpoint(component: Any, instance: Any = None) -> bool:
    """Whether this slot belongs in a checkpoint's ``components`` manifest.

    Three reasons, in order of frequency:

    1. It is not a built-in: a ``project``, ``plugin`` or ``runtime`` component is *code* that
       the settings cannot reconstruct, so its identity must travel with the weights (I5, I6).
       An in-code instance of a built-in *class* is a built-in.
    2. It replaces a built-in: the origin is ``project`` already in that case, but the
       ``overrides_builtin`` flag is what the resume comparison reports.
    3. Its instance asks to be recorded, or declares itself ``unsafe``.

    A built-in with parameters set in configuration is deliberately **not** recorded: the
    parameters are in ``runs.settings`` and in ``conf/``, and recording them would break P1
    for every run that ever wrote a ``LOSS`` block.
    """
    if bool(component.overrides_builtin):
        return True
    if component.origin is not Origin.BUILTIN and not _is_builtin_class(component):
        return True
    if instance is None:
        return False
    return any(_flagged(instance, flag) for flag in RECORD_FLAGS)


def entry_for(slot: Any) -> dict[str, Any]:
    """One manifest entry for a live slot (``components.slots.Slot`` or anything shaped like it)."""
    resolved = slot.resolved
    component = resolved.component
    payload: dict[str, Any] = {
        "schema": ENTRY_SCHEMA,
        "kind": slot.kind_id,
        "name": component.name,
        "origin": component.origin.value,
        "version": str(component.version),
        "spec_hash": str(resolved.spec_hash),
        "slot": slot.path,
        # The spec's params, verbatim: they are what the project wrote and what a diff shows.
        "params": dict(resolved.spec.params),
    }
    source = component.fingerprint(spec_hash=None).get("source_digest")
    if source:
        payload["source_digest"] = str(source)
    if component.overrides_builtin:
        payload["overrides_builtin"] = True
    if _flagged(getattr(resolved, "instance", None), "unsafe"):
        payload["unsafe"] = True
    return payload


# ---------------------------------------------------------------------------
# One view over every place a run keeps a component
# ---------------------------------------------------------------------------
class SlotView:
    """``SlotTable.paths()`` / ``get()`` over a table *and* the components the plans hold.

    ``extras`` are slot-shaped objects (``path``, ``kind_id``, ``resolved``). A path in both
    resolves to the table's slot. Nothing here builds a component: every slot already holds
    its resolved instance.
    """

    def __init__(self, slots: Any = None, extras: Iterable[Any] = ()) -> None:
        self._slots = slots
        self._extras: dict[str, Any] = {slot.path: slot for slot in extras}

    def paths(self) -> list[str]:
        own = list(self._slots.paths()) if self._slots is not None else []
        return sorted(set(own) | set(self._extras))

    def get(self, path: str) -> Any:
        if self._slots is not None and path in self._slots.paths():
            return self._slots.get(path)
        try:
            return self._extras[path]
        except KeyError:
            from ..exceptions import ComponentMissing

            raise ComponentMissing("slot", path, self.paths()) from None

    def __iter__(self) -> Iterator[Any]:
        for path in self.paths():
            yield self.get(path)


def component_manifest(
    slots: Any, *, kinds: Iterable[str] | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{kind: {slot path: entry}}`` for every slot that must be recorded. ``{}`` otherwise.

    ``slots`` is a :class:`~mactf.components.slots.SlotTable` or a :class:`SlotView`; ``kinds``
    narrows the result (the CLI asks for one kind, a checkpoint always records all of them).
    """
    wanted = None if kinds is None else frozenset(str(kind) for kind in kinds)
    manifest: dict[str, dict[str, dict[str, Any]]] = {}
    for path in slots.paths():
        slot = slots.get(path)
        if wanted is not None and slot.kind_id not in wanted:
            continue
        if not records_in_checkpoint(slot.component, slot.instance):
            continue
        manifest.setdefault(slot.kind_id, {})[path] = entry_for(slot)
    return {kind: dict(sorted(entries.items())) for kind, entries in sorted(manifest.items())}


def recorded_slots(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``[(kind, slot)]`` in kind-then-path order, for a report or a loop."""
    pairs: list[tuple[str, str]] = []
    for kind in sorted(manifest or {}):
        entries = manifest[kind]
        if not isinstance(entries, Mapping):
            continue
        for path in sorted(entries):
            pairs.append((str(kind), str(path)))
    return pairs


# ---------------------------------------------------------------------------
# Comparing two records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Change:
    """One field that differs between a saved entry and a live one."""

    kind: str
    slot: str
    field: str
    saved: Any
    live: Any

    @property
    def structural(self) -> bool:
        """Whether the change means a *different component*, not a different version of it."""
        return self.field in (
            "name", "kind", "origin", "spec_hash", "source_digest", "present", "params",
        )

    def describe(self) -> str:
        if self.field == "present":
            return f"{self.slot}: {'absent now' if self.live is None else 'absent when saved'}"
        return f"{self.slot}: {self.field} {self.saved!r} -> {self.live!r}"


#: Fields compared in this order. ``present`` is synthetic (one side has no entry at all).
_COMPARED: tuple[str, ...] = (
    "kind", "name", "origin", "version", "spec_hash", "source_digest", "overrides_builtin",
)


def diff(
    saved: Mapping[str, Any] | None,
    live: Mapping[str, Any] | None,
) -> list[Change]:
    """Every difference between two manifests of the shape ``component_manifest`` returns.

    A slot present on one side only is one ``present`` change, not seven field changes: the
    message a user needs is "this slot is not configured any more", not the field list of a
    record that no longer exists. A differing ``params`` mapping is reported as one change
    whose values are the mappings, because ``spec_hash`` already moved and a per-key diff
    would repeat the same fact.
    """
    saved_map = _flat(saved)
    live_map = _flat(live)
    changes: list[Change] = []
    for kind, path in sorted(set(saved_map) | set(live_map)):
        before = saved_map.get((kind, path))
        after = live_map.get((kind, path))
        if before is None or after is None:
            changes.append(Change(kind=kind, slot=path, field="present", saved=before, live=after))
            continue
        for field in _COMPARED:
            if before.get(field) != after.get(field):
                changes.append(
                    Change(kind=kind, slot=path, field=field,
                           saved=before.get(field), live=after.get(field))
                )
        if dict(before.get("params") or {}) != dict(after.get("params") or {}):
            changes.append(
                Change(kind=kind, slot=path, field="params",
                       saved=dict(before.get("params") or {}),
                       live=dict(after.get("params") or {}))
            )
    return changes


def _flat(manifest: Mapping[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    """``{(kind, slot): entry}`` from the nested shape."""
    flat: dict[tuple[str, str], dict[str, Any]] = {}
    for kind, entries in dict(manifest or {}).items():
        if not isinstance(entries, Mapping):
            continue
        for path, entry in entries.items():
            if isinstance(entry, Mapping):
                flat[(str(kind), str(path))] = dict(entry)
    return flat


def render(changes: Iterable[Change]) -> str:
    """One line per change, for a log or ``mactf components diff``."""
    return "\n".join(change.describe() for change in changes)


# ---------------------------------------------------------------------------
# Components the plans hold outside the slot table
# ---------------------------------------------------------------------------
def _extra_slot(path: str, kind_id: str, component: Any, instance: Any, spec: Any, spec_hash: str) -> Any:
    from .resolver import Resolved
    from .slots import Slot

    return Slot(
        path=path,
        kind_id=kind_id,
        resolved=Resolved(spec=spec, component=component, instance=instance, spec_hash=spec_hash),
    )


def optimizer_slots(plan: Any) -> list[Any]:
    """Slot-shaped views of an ``OptimizerPlan``'s entries and their schedulers (section 9).

    ``OPTIMIZER`` / ``OPTIMIZER.<glob>`` for the optimizer, ``<slot>:scheduler`` for its
    scheduler. An entry whose backend has no registry record (the plain ``torch.optim``
    classes, ``scipy``, a bare callable) is not a component and has no slot; neither has the
    in-code ``optimizer=`` argument, which is a spec, not a registered class.
    """
    if plan is None:
        return []
    from .registry import HUB
    from .reflect import spec_hash
    from .resolver import ComponentSpec

    slots: list[Any] = []
    for slot_path, entry in sorted(dict(plan.entries).items()):
        if getattr(entry, "kind", "block") == "code":
            continue
        component = HUB.registry("optimizer").find(str(entry.backend))
        if component is not None:
            params = {k: v for k, v in dict(entry.payload).items() if k != "node_path"}
            spec = ComponentSpec(kind_id="optimizer", type=component.name, params=params)
            slots.append(
                _extra_slot(slot_path, "optimizer", component, None, spec, str(entry.spec_hash))
            )
        scheduler = getattr(entry, "scheduler", None)
        if scheduler:
            sched_component = HUB.registry("scheduler").find(str(scheduler[0]))
            if sched_component is not None:
                params = dict(scheduler[1] or {})
                sched_spec = ComponentSpec(
                    kind_id="scheduler", type=sched_component.name, params=params
                )
                slots.append(
                    _extra_slot(
                        f"{slot_path}:scheduler", "scheduler", sched_component, None, sched_spec,
                        spec_hash({"scheduler": [sched_component.name, params]}),
                    )
                )
    return slots


def aggregator_slots(resolved: Any, path: str = "TRAINER.AGGREGATOR") -> list[Any]:
    """The run's aggregator as a slot, from the ``Resolved`` its build recorded (or none)."""
    if resolved is None:
        return []
    from .slots import Slot

    return [Slot(path=path, kind_id="aggregator", resolved=resolved)]


def reward_term_slots(stages: Iterable[Any], prefix: str = "scoring.reward.terms") -> list[Any]:
    """Slot-shaped views of the terms of every ``composed_reward`` stage in ``stages``.

    A stage records what it resolved in ``term_records`` as ``[(label, Resolved)]``; a stage
    without the attribute holds no terms.
    """
    slots: list[Any] = []
    for stage in stages:
        records = getattr(stage, "term_records", None)
        if not records:
            continue
        from .slots import Slot

        for label, resolved in records:
            slots.append(Slot(path=f"{prefix}.{label}", kind_id="reward_term", resolved=resolved))
    return slots
