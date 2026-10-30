"""Reconcile two vocabularies on a shared label axis (document 14 section 5.9)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..exceptions import ContractError, ValidationError

__all__ = ["LabelMap", "build_label_map", "shared_axis"]


class LabelMap:
    """Project one model's logits onto a shared label axis.

    ``index[i]`` names the source position(s) that feed ``axis[i]``. An entry is
    either one integer or a group of integers; when several model labels map to
    the same shared label their *probabilities* are summed. That is the only
    aggregation that preserves a distribution: summing logits or averaging
    probabilities would both change the total mass.

    Labels the axis has but the model does not get zero mass. Labels the model
    has but the axis does not are dropped, and recorded in ``dropped``: dropping
    them without a record is how a silent accuracy bug starts.
    """

    def __init__(
        self,
        index: Sequence[int | Sequence[int]],
        *,
        source_size: int,
        axis: Sequence[str],
        source_labels: Sequence[str] = (),
    ) -> None:
        groups: list[tuple[int, ...]] = []
        for entry in index:
            if isinstance(entry, (int,)) or hasattr(entry, "__index__"):
                groups.append((int(entry),))
            else:
                groups.append(tuple(int(value) for value in entry))
        self.index = tuple(groups)
        self.source_size = int(source_size)
        self.axis = tuple(str(label) for label in axis)
        self.source_labels = tuple(str(label) for label in source_labels)
        if not self.axis:
            raise ValidationError("LabelMap requires a non-empty shared axis")
        if len(self.axis) != len(set(self.axis)):
            raise ValidationError("LabelMap axis labels must be unique")
        for group in self.index:
            if not group:
                raise ValidationError(
                    "LabelMap index entries must name at least one source position"
                )
            if any(value < 0 for value in group):
                raise ValidationError("LabelMap index entries must be >= 0")
            if any(value >= self.source_size for value in group):
                raise ValidationError(
                    f"LabelMap index refers past the source vocabulary ({self.source_size})"
                )
        if len(self.index) != len(self.axis):
            raise ValidationError(
                f"LabelMap index has {len(self.index)} entries for an axis of "
                f"{len(self.axis)} labels"
            )
        used = {value for group in self.index for value in group}
        if len(used) != sum(len(group) for group in self.index):
            raise ValidationError(
                "LabelMap routes one source position to two axis labels; a source "
                "label's mass can only land in one place"
            )
        if self.source_labels:
            if len(self.source_labels) != self.source_size:
                raise ValidationError(
                    f"LabelMap received {len(self.source_labels)} source labels for a "
                    f"vocabulary of {self.source_size}"
                )
            self.dropped = tuple(
                self.source_labels[position]
                for position in range(self.source_size)
                if position not in used
            )
            self.dropped = tuple(sorted(self.dropped))
            covered = {
                self.axis[slot]
                for slot, group in enumerate(self.index)
                if group
            }
            self.missing = tuple(sorted(set(self.axis) - covered))
        else:
            self.dropped = ()
            self.missing = ()

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Aggregate ``logits`` onto the shared axis, in probability space."""
        if logits.shape[-1] != self.source_size:
            raise ContractError(
                f"LabelMap was built for a vocabulary of {self.source_size} but "
                f"received {logits.shape[-1]} logits"
            )
        probs = torch.softmax(logits.float(), dim=-1)
        columns = [probs[:, list(group)].sum(dim=-1) for group in self.index]
        mapped = torch.stack(columns, dim=-1)
        total = mapped.sum(dim=-1, keepdim=True)
        empty = total.reshape(-1) <= 0.0
        if bool(empty.any()):
            # Not an error: a sample whose entire mass sits on dropped labels has
            # no shared evidence, and a uniform vector is the honest encoding of
            # "this model said nothing about the shared axis".
            uniform = torch.full_like(mapped, 1.0 / len(self.axis))
            mapped = torch.where(empty.unsqueeze(-1), uniform, mapped)
            total = mapped.sum(dim=-1, keepdim=True)
        mapped = mapped / total
        return mapped.clamp_min(1e-12).log()

    def describe(self) -> dict[str, Any]:
        return {
            "source_size": self.source_size,
            "axis": len(self.axis),
            "dropped": list(self.dropped),
            "missing": list(self.missing),
        }


def build_label_map(
    source_labels: Sequence[str],
    axis: Sequence[str],
    *,
    source_size: int | None = None,
    aliases: Mapping[str, str] | None = None,
) -> LabelMap:
    """Index map from ``source_labels`` onto ``axis``.

    ``index[i]`` collects the source positions feeding ``axis[i]``: the source
    label of the same name, plus every source label that ``aliases`` maps to it.
    Because a shared axis is an intersection by construction (``shared_axis``),
    every ``axis`` entry resolves; a missing one is a programming error rather
    than a tolerance case.
    """
    axis = tuple(str(label) for label in axis)
    source = tuple(str(label) for label in source_labels)
    positions: dict[str, int] = {}
    for position, label in enumerate(source):
        if label in positions:
            raise ValidationError(
                f"source label {label!r} appears twice; a duplicate would make the "
                "mass of an ambiguous index"
            )
        positions[label] = position

    routed: dict[str, list[int]] = {label: [] for label in axis}
    for label in axis:
        if label in positions:
            routed[label].append(positions[label])
    for alias, target in sorted(dict(aliases or {}).items()):
        alias, target = str(alias), str(target)
        if target not in routed:
            raise ValidationError(
                f"alias {alias!r} points at {target!r}, which is not on the shared axis"
            )
        if alias not in positions:
            raise ValidationError(f"alias {alias!r} is not a label of the source vocabulary")
        if alias == target:
            continue
        if positions[alias] not in routed[target]:
            routed[target].append(positions[alias])

    absent = sorted(label for label, group in routed.items() if not group)
    if absent:
        raise ValidationError(
            f"shared axis labels {absent[:5]} are absent from the source vocabulary; "
            "build the axis with shared_axis() so it is an intersection"
        )
    index = [tuple(sorted(routed[label])) for label in axis]
    size = int(source_size) if source_size is not None else len(source)
    return LabelMap(index, source_size=size, axis=axis, source_labels=source)


def shared_axis(vocabularies: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """The intersection of every vocabulary, in sorted order.

    Intersection rather than union: a label only one model knows cannot be
    compared, and a union axis would hand that model free mass the others can
    never match.
    """
    sets = [set(str(label) for label in labels) for labels in vocabularies.values()]
    if not sets:
        return ()
    return tuple(sorted(set.intersection(*sets)))
