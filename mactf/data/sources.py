"""Task sources: synthetic, JSONL-backed, and mixed.

Every source satisfies :class:`TaskSourceProtocol` and is deterministic for a
fixed ``(seed, epoch)``. Determinism is a scored property in this framework (a
rerun of an identical configuration must reproduce a scorecard), so no source
uses the builtin ``hash`` for anything numeric and none touches the global
random state.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..exceptions import DataError
from .codec import TaskCodec, TaskRecord
from .sampling import batch_slices, content_hash, epoch_indices, seed_from

__all__ = [
    "TaskSourceProtocol",
    "SyntheticTaskSource",
    "JsonlTaskSource",
    "MixedTaskSource",
    "load_records",
]


@runtime_checkable
class TaskSourceProtocol(Protocol):
    """What the orchestrator, bootstrap, and evaluators require of a source."""

    def next(self, epoch: int) -> Any:
        """The next task batch for ``epoch``."""

    def describe(self) -> str:
        ...

    def __len__(self) -> int:
        ...


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------


class SyntheticTaskSource:
    """The framework's original synthetic classification generator.

    The arithmetic here is *identical* to the historical ``TaskSource`` so that
    existing runs and tests produce bit-identical tensors. The seed derivation
    is ``hash((seed, epoch, counter))`` where all three are integers, and Python
    only randomizes ``hash`` for ``str``/``bytes``, so this is already
    cross-process stable. Do not "improve" it.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._counter = 0

    def next(self, epoch: int) -> Any:
        import torch

        from ..orchestration.episode import Task

        self._counter += 1
        local = torch.Generator().manual_seed(
            hash((self.seed, epoch, self._counter)) % (2**31 - 1)
        )
        states = torch.randn(self.batch_size, self.state_dim, generator=local)
        projection = torch.randn(self.state_dim, self.action_dim, generator=local)
        targets = (states @ projection).argmax(dim=-1)
        return Task(
            id=f"epoch{epoch}-task{self._counter}",
            states=states,
            targets=targets,
            prompt="synthetic classification",
            metadata={"source": "synthetic", "epoch": epoch},
        )

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "synthetic", "counter": self._counter, "seed": self.seed}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError(
                f"data position was recorded with seed {state.get('seed')}, "
                f"this source uses {self.seed}"
            )
        self._counter = int(state.get("counter", 0))

    def describe(self) -> str:
        return (
            f"synthetic(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"batch_size={self.batch_size}, seed={self.seed})"
        )

    def __len__(self) -> int:
        # Unbounded by design: it generates a fresh batch per call.
        return 1


# ---------------------------------------------------------------------------
# jsonl
# ---------------------------------------------------------------------------


def load_records(
    paths: Sequence[Path | str],
    *,
    action_dim: int | None = None,
    label_strategy: str = "hash",
) -> list[tuple[TaskRecord, Path, int]]:
    """Read and validate every record from ``paths`` in file order.

    Uniqueness of ``id`` is enforced per file, as the codec contract states; two
    different files may legitimately reuse an id, and the source keeps the file
    name in the task metadata so the two remain distinguishable.
    """
    loaded: list[tuple[TaskRecord, Path, int]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise DataError(f"dataset file not found: {path}", path=path)
        seen: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DataError(
                        f"invalid JSON: {exc.msg} (column {exc.colno})",
                        path=path,
                        line=line_number,
                    ) from exc
                record = TaskCodec.decode(
                    payload, line=line_number, path=path, action_dim=action_dim
                )
                if record.id in seen:
                    raise DataError(
                        f"duplicate record id {record.id!r}",
                        path=path,
                        line=line_number,
                        record_id=record.id,
                        key="id",
                    )
                seen.add(record.id)
                if label_strategy == "explicit" and not record.has_tensors:
                    continue
                loaded.append((record, path, line_number))
    return loaded


def _text_vector(text: str, *, state_dim: int, salt: str) -> list[float]:
    """A deterministic, bounded feature vector for a piece of text.

    Not a learned embedding: it is a stable hashed projection whose only job is
    to give the supervised bootstrap a non-degenerate input per record. Records
    with near-identical text share a direction, which is what makes the label
    learnable rather than pure noise.
    """
    values: list[float] = []
    for index in range(state_dim):
        hashed = content_hash(f"{salt}\x00{index}\x00{text}")
        # Map to [-1, 1) with enough resolution to avoid duplicate rows.
        values.append((hashed / float(1 << 63)) - 1.0)
    return values


class JsonlTaskSource:
    """Deterministic epoch iteration over one or more JSONL files."""

    def __init__(
        self,
        paths: Sequence[Path | str],
        *,
        state_dim: int,
        action_dim: int,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        label_strategy: str = "hash",
    ) -> None:
        if label_strategy not in {"hash", "explicit"}:
            raise ValueError(
                f"unknown label_strategy {label_strategy!r}; expected 'hash' or 'explicit'"
            )
        if state_dim < 1 or action_dim < 1:
            raise ValueError("state_dim and action_dim must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.paths = [Path(path) for path in paths]
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.label_strategy = label_strategy

        self._records = load_records(
            self.paths, action_dim=self.action_dim, label_strategy=label_strategy
        )
        for record, path, line_number in self._records:
            if record.has_tensors and record.state_dim != self.state_dim:
                raise DataError(
                    f"record states have width {record.state_dim} but state_dim is "
                    f"{self.state_dim}",
                    path=path,
                    line=line_number,
                    record_id=record.id,
                    key="states",
                )
        if not self._records:
            raise DataError(
                "no usable records"
                + (
                    " (label_strategy='explicit' skips records without 'targets')"
                    if label_strategy == "explicit"
                    else ""
                ),
                path=self.paths[0] if self.paths else None,
            )
        self._epoch: int | None = None
        self._order: list[int] = []
        self._batches: list[tuple[int, int]] = []
        self._cursor = 0

    # -- batch selection -------------------------------------------------
    def _prepare(self, epoch: int) -> None:
        if self._epoch == epoch:
            return
        self._epoch = epoch
        self._order = epoch_indices(
            len(self._records), epoch=epoch, seed=self.seed, shuffle=self.shuffle
        )
        self._batches = list(batch_slices(self._order, self.batch_size))
        self._cursor = 0

    def _task_for(self, rows: list[int], epoch: int) -> Any:
        import torch

        from ..orchestration.episode import Task

        state_rows: list[list[float]] = []
        targets: list[int] = []
        listed: list[TaskRecord] = []
        synthetic = False
        for position, row in enumerate(rows):
            record, path, line_number = self._records[row]
            listed.append(record)
            if record.has_tensors:
                assert record.states is not None and record.targets is not None
                state_rows.extend(list(state_row) for state_row in record.states)
                targets.extend(record.targets)
            else:
                synthetic = True
                state_rows.append(
                    _text_vector(
                        record.prompt or record.id,
                        state_dim=self.state_dim,
                        salt=path.name,
                    )
                )
                targets.append(
                    content_hash(record.signature) % self.action_dim
                    if self.label_strategy == "hash"
                    else 0
                )

        states = torch.tensor(state_rows, dtype=torch.float32)
        # A record with explicit tensors may carry a different width than the
        # configured state_dim; that is a dataset error, not something to pad.
        if states.shape[1] != self.state_dim:
            raise DataError(
                f"record states have width {states.shape[1]} but state_dim is "
                f"{self.state_dim}",
                path=self._records[rows[0]][1],
                line=self._records[rows[0]][2],
                record_id=listed[0].id,
                key="states",
            )
        target_tensor = torch.tensor(targets, dtype=torch.long)
        first = listed[0]
        first_path, first_line = self._records[rows[0]][1], self._records[rows[0]][2]
        return Task(
            id=f"epoch{epoch}-task{self._cursor}",
            states=states,
            targets=target_tensor,
            prompt=first.prompt,
            metadata={
                "source": first_path.name,
                "record_id": first.id,
                "line": first_line,
                "record_ids": [record.id for record in listed],
                "epoch": epoch,
                "synthetic_label": synthetic,
                "expected": first.expected,
                "stage": first.metadata.get("stage"),
                "expects": list(first.metadata.get("expects", ())),
                "records": [
                    {
                        "id": record.id,
                        "prompt": record.prompt,
                        "expected": record.expected,
                        "evidence": record.evidence,
                        "verified": record.verified,
                        "metadata": dict(record.metadata),
                    }
                    for record in listed
                ],
            },
        )

    def next(self, epoch: int) -> Any:
        self._prepare(epoch)
        start, stop = self._batches[self._cursor % len(self._batches)]
        rows = self._order[start:stop]
        task = self._task_for(rows, epoch)
        self._cursor += 1
        return task

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "jsonl",
            "epoch": self._epoch,
            "cursor": self._cursor,
            "seed": self.seed,
            "records": len(self._records),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("records", len(self._records))) != len(self._records):
            raise DataError(
                f"data position was recorded over {state.get('records')} records "
                f"but this source loaded {len(self._records)}",
                path=self.paths[0] if self.paths else None,
            )
        epoch = state.get("epoch")
        self._epoch = None
        if epoch is not None:
            self._prepare(int(epoch))
        self._cursor = int(state.get("cursor", 0))

    def records(self) -> list[TaskRecord]:
        """Every loaded record, in file order (used by the evaluator fixtures)."""
        return [record for record, _, _ in self._records]

    def describe(self) -> str:
        names = ", ".join(path.name for path in self.paths) or "none"
        return (
            f"jsonl(files=[{names}], records={len(self._records)}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle}, "
            f"label_strategy={self.label_strategy})"
        )

    def __len__(self) -> int:
        """Number of batches per epoch (a short final batch counts)."""
        return max(1, math.ceil(len(self._records) / self.batch_size))


# ---------------------------------------------------------------------------
# mixed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Weighted:
    source: TaskSourceProtocol
    weight: float


class MixedTaskSource:
    """Weighted interleave of several sources.

    Selection is a deterministic proportional schedule, not a random draw, so
    two runs of the same configuration see the same interleave. A source with
    weight 0 is never selected.
    """

    def __init__(self, sources: Sequence[tuple[TaskSourceProtocol, float]]) -> None:
        if not sources:
            raise ValueError("MixedTaskSource requires at least one source")
        entries: list[_Weighted] = []
        for source, weight in sources:
            numeric = float(weight)
            if numeric < 0.0 or math.isnan(numeric) or math.isinf(numeric):
                raise ValueError(f"weight must be a finite non-negative number, got {weight!r}")
            entries.append(_Weighted(source=source, weight=numeric))
        total = sum(entry.weight for entry in entries)
        if total <= 0.0:
            raise ValueError("MixedTaskSource requires at least one positive weight")
        self._entries = tuple(entries)
        self._total = total
        self._cumulative: list[float] = []
        running = 0.0
        for entry in self._entries:
            running += entry.weight
            self._cumulative.append(running / total)
        self._cursor = 0

    def _pick(self, epoch: int, cursor: int) -> TaskSourceProtocol:
        # Deterministic in (epoch, cursor): pick a point in [0, 1) from a stable
        # hash and choose the first cumulative bucket that exceeds it.
        point = (seed_from(epoch, cursor, "mix") % (1 << 32)) / float(1 << 32)
        for index, boundary in enumerate(self._cumulative):
            if point < boundary:
                return self._entries[index].source
        return self._entries[-1].source

    def next(self, epoch: int) -> Any:
        source = self._pick(epoch, self._cursor)
        self._cursor += 1
        return source.next(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "mixed",
            "cursor": self._cursor,
            "sources": [
                entry.source.state_dict() if hasattr(entry.source, "state_dict") else None
                for entry in self._entries
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._cursor = int(state.get("cursor", 0))
        for entry, child in zip(self._entries, state.get("sources", ())):
            if child is not None and hasattr(entry.source, "load_state_dict"):
                entry.source.load_state_dict(child)

    def describe(self) -> str:
        parts = ", ".join(
            f"{entry.source.describe()}@{entry.weight:g}" for entry in self._entries
        )
        return f"mixed([{parts}])"

    def __len__(self) -> int:
        return sum(max(1, len(entry.source)) for entry in self._entries)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def iter_batches(
    source: TaskSourceProtocol, epoch: int, *, limit: int | None = None
) -> Iterator[Any]:
    """Yield consecutive tasks from ``source`` for one epoch."""
    produced = 0
    while limit is None or produced < limit:
        yield source.next(epoch)
        produced += 1


def record_claim_payload(task: Any) -> Mapping[str, Any]:
    """The evidence a runs' task metadata exposes to the evaluators."""
    metadata = getattr(task, "metadata", {}) or {}
    return dict(metadata)
