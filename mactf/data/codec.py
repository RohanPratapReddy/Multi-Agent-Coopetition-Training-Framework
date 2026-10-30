"""Record validation for JSONL task datasets.

A dataset is normative input: a typo in one record changes what a peer is
trained on, so every record is validated strictly and unknown keys are an error
rather than being ignored. Three shapes exist today and this codec accepts all
three by describing them as one record type with optional fields:

``truth.jsonl``
    ``{"id", "claim", "evidence", "verified"}`` - a claim with supporting
    evidence. ``claim`` is folded into ``prompt`` so a peer always has an input.
``ambiguity.jsonl``
    ``{"id", "prompt", "expected"}`` - a prompt with an expected behaviour such
    as ``admitting_ignorance``.
``interaction.jsonl``
    ``{"id", "stage", "peers", "expects"}`` - an interaction fixture naming the
    rule IDs the stage should exercise.

Records may instead carry explicit ``states``/``targets`` tensors, which is how a
user-supplied corpus of precomputed inputs is expressed. Both must be present or
both absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exceptions import DataError

__all__ = ["TaskRecord", "TaskCodec", "CLAIM_KEYS", "INTERACTION_KEYS"]

#: Keys accepted in a record. ``claim`` and ``stage``/``peers``/``expects`` are
#: the two dataset-specific additions; everything else is shared.
CLAIM_KEYS = frozenset({"claim"})
INTERACTION_KEYS = frozenset({"stage", "peers", "expects", "expects_rules"})


@dataclass(frozen=True)
class TaskRecord:
    """One validated dataset record.

    ``states``/``targets`` are optional; when absent the source synthesizes
    tensors from the record's textual content, which is how the shipped
    seed datasets become training data.
    """

    id: str
    prompt: str = ""
    states: tuple[tuple[float, ...], ...] | None = None
    targets: tuple[int, ...] | None = None
    expected: str | None = None
    evidence: str | None = None
    verified: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_tensors(self) -> bool:
        return self.states is not None

    @property
    def signature(self) -> str:
        """The deterministic label signature for a record without targets."""
        return self.expected or self.evidence or self.id

    @property
    def state_dim(self) -> int:
        if not self.states:
            return 0
        return len(self.states[0])

    def describe(self) -> str:
        kind = "tensors" if self.has_tensors else "text"
        return f"record[{self.id}] {kind} prompt={self.prompt[:40]!r}"


class TaskCodec:
    """Decode and encode :class:`TaskRecord` values."""

    ALLOWED = frozenset(
        {
            "id",
            "prompt",
            "states",
            "targets",
            "expected",
            "evidence",
            "verified",
            "metadata",
        }
    ) | CLAIM_KEYS | INTERACTION_KEYS

    # -- decoding --------------------------------------------------------
    @classmethod
    def decode(
        cls,
        raw: Mapping[str, Any],
        *,
        line: int,
        path: Path | str,
        action_dim: int | None = None,
    ) -> TaskRecord:
        """Validate one raw record, raising :class:`DataError` on any problem."""
        if not isinstance(raw, Mapping):
            raise DataError(
                f"record must be a JSON object, got {type(raw).__name__}",
                path=path,
                line=line,
            )

        record_id = raw.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise DataError(
                "record requires a non-empty string 'id'",
                path=path,
                line=line,
                key="id",
            )

        def fail(message: str, key: str | None = None) -> DataError:
            return DataError(
                message, path=path, line=line, record_id=record_id, key=key
            )

        unknown = sorted(set(raw) - cls.ALLOWED)
        if unknown:
            raise fail(
                f"unknown key(s) {', '.join(repr(key) for key in unknown)}; "
                f"allowed keys are {', '.join(sorted(cls.ALLOWED))}",
                unknown[0],
            )

        states, targets = cls._decode_tensors(raw, fail, action_dim=action_dim)

        expected = raw.get("expected")
        if expected is not None and not isinstance(expected, str):
            raise fail("'expected' must be a string", "expected")

        evidence = raw.get("evidence")
        if evidence is not None and not isinstance(evidence, str):
            raise fail("'evidence' must be a string", "evidence")

        verified = raw.get("verified")
        if verified is not None and not isinstance(verified, bool):
            raise fail("'verified' must be a boolean", "verified")

        prompt = raw.get("prompt")
        claim = raw.get("claim")
        if prompt is not None and not isinstance(prompt, str):
            raise fail("'prompt' must be a string", "prompt")
        if claim is not None and not isinstance(claim, str):
            raise fail("'claim' must be a string", "claim")
        if prompt is None:
            prompt = claim or ""

        metadata_raw = raw.get("metadata")
        if metadata_raw is None:
            metadata: dict[str, Any] = {}
        elif isinstance(metadata_raw, Mapping):
            metadata = dict(metadata_raw)
        else:
            raise fail("'metadata' must be an object", "metadata")

        # Dataset-specific keys are preserved in metadata rather than discarded,
        # because they are the evidence the evaluators read.
        if claim is not None:
            metadata.setdefault("claim", claim)
        stage = raw.get("stage")
        if stage is not None:
            if not isinstance(stage, str):
                raise fail("'stage' must be a string", "stage")
            metadata["stage"] = stage
        peers = raw.get("peers")
        if peers is not None:
            if not isinstance(peers, int) or isinstance(peers, bool) or peers < 1:
                raise fail("'peers' must be a positive integer", "peers")
            metadata["peers"] = int(peers)
        expects = raw.get("expects", raw.get("expects_rules"))
        if expects is not None:
            if not isinstance(expects, (list, tuple)) or not all(
                isinstance(item, str) for item in expects
            ):
                raise fail("'expects' must be a list of rule ID strings", "expects")
            metadata["expects"] = list(expects)

        return TaskRecord(
            id=record_id,
            prompt=prompt,
            states=states,
            targets=targets,
            expected=expected,
            evidence=evidence,
            verified=verified,
            metadata=metadata,
        )

    @staticmethod
    def _decode_tensors(
        raw: Mapping[str, Any],
        fail: Any,
        *,
        action_dim: int | None,
    ) -> tuple[tuple[tuple[float, ...], ...] | None, tuple[int, ...] | None]:
        has_states = "states" in raw and raw["states"] is not None
        has_targets = "targets" in raw and raw["targets"] is not None
        if has_states != has_targets:
            raise fail(
                "'states' and 'targets' must both be present or both absent",
                "targets" if has_targets else "states",
            )
        if not has_states:
            return None, None

        raw_states = raw["states"]
        raw_targets = raw["targets"]
        if not isinstance(raw_states, (list, tuple)) or not raw_states:
            raise fail("'states' must be a non-empty list of rows", "states")
        if not isinstance(raw_targets, (list, tuple)):
            raise fail("'targets' must be a list", "targets")
        if len(raw_states) != len(raw_targets):
            raise fail(
                f"'states' has {len(raw_states)} row(s) but 'targets' has "
                f"{len(raw_targets)}; they must match",
                "targets",
            )

        width: int | None = None
        states: list[tuple[float, ...]] = []
        for row_index, row in enumerate(raw_states):
            if not isinstance(row, (list, tuple)) or not row:
                raise fail(
                    f"state row {row_index} must be a non-empty list of numbers",
                    "states",
                )
            values: list[float] = []
            for column in row:
                if isinstance(column, bool) or not isinstance(column, (int, float)):
                    raise fail(
                        f"state row {row_index} contains a non-numeric value {column!r}",
                        "states",
                    )
                number = float(column)
                if math.isnan(number) or math.isinf(number):
                    raise fail(
                        f"state row {row_index} contains a non-finite value {column!r}",
                        "states",
                    )
                values.append(number)
            if width is None:
                width = len(values)
            elif len(values) != width:
                raise fail(
                    f"state row {row_index} has {len(values)} value(s) but row 0 has "
                    f"{width}; every row must have the same length",
                    "states",
                )
            states.append(tuple(values))

        targets: list[int] = []
        for index, value in enumerate(raw_targets):
            if isinstance(value, bool) or not isinstance(value, int):
                raise fail(f"target {index} must be an integer, got {value!r}", "targets")
            if value < 0:
                raise fail(f"target {index} must be non-negative", "targets")
            if action_dim is not None and value >= action_dim:
                raise fail(
                    f"target {index} is {value} but action_dim is {action_dim}",
                    "targets",
                )
            targets.append(int(value))

        return tuple(states), tuple(targets)

    # -- encoding --------------------------------------------------------
    @classmethod
    def encode(cls, record: TaskRecord) -> dict[str, Any]:
        """The inverse of :meth:`decode` for the fields that were set."""
        payload: dict[str, Any] = {"id": record.id}
        if record.prompt:
            payload["prompt"] = record.prompt
        if record.states is not None:
            payload["states"] = [list(row) for row in record.states]
        if record.targets is not None:
            payload["targets"] = list(record.targets)
        if record.expected is not None:
            payload["expected"] = record.expected
        if record.evidence is not None:
            payload["evidence"] = record.evidence
        if record.verified is not None:
            payload["verified"] = record.verified
        if record.metadata:
            payload["metadata"] = dict(record.metadata)
        return payload
