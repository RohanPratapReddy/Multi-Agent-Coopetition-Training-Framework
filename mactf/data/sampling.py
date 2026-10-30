"""Deterministic epoch iteration and batching.

The framework must produce identical batches for a fixed ``(seed, epoch)`` on
every machine and in every process. That rules out the builtin ``hash`` (salted
for ``str``/``bytes`` under ``PYTHONHASHSEED``) and rules out ``random.shuffle``
over the global RNG, which would couple a run's batching to every other consumer
of the random stream.

Both helpers therefore derive their entropy from ``hashlib.blake2b`` over a
canonical byte encoding of the integers involved, and use a small
self-contained PRNG (SplitMix64) so the permutation never depends on library
internals.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence

__all__ = ["content_hash", "seed_from", "epoch_indices", "batch_slices", "permutation"]

_MASK64 = (1 << 64) - 1


def content_hash(value: str, *, digest_size: int = 8) -> int:
    """A stable 64-bit integer for a string, identical across processes.

    Used for synthetic label assignment, where ``hash()`` would vary with
    ``PYTHONHASHSEED`` and silently change the labels between runs.
    """
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=digest_size).digest()
    return int.from_bytes(digest, "big")


def seed_from(*parts: int | str) -> int:
    """Derive a stable 64-bit seed from an ordered tuple of ints and strings."""
    encoded = "\x1f".join(
        f"i:{int(part)}" if isinstance(part, int) and not isinstance(part, bool) else f"s:{part}"
        for part in parts
    )
    return content_hash(encoded)


class _SplitMix64:
    """The SplitMix64 generator: tiny, dependency-free, and reproducible forever."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK64

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        """Uniform integer in ``[0, bound)`` by rejection, with no modulo bias."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        limit = (_MASK64 + 1) - ((_MASK64 + 1) % bound)
        while True:
            candidate = self.next_u64()
            if candidate < limit:
                return candidate % bound


def permutation(count: int, *, seed: int) -> list[int]:
    """A deterministic Fisher-Yates permutation of ``range(count)``."""
    if count < 0:
        raise ValueError("count must be non-negative")
    indices = list(range(count))
    if count <= 1:
        return indices
    generator = _SplitMix64(seed)
    for position in range(count - 1, 0, -1):
        swap = generator.below(position + 1)
        indices[position], indices[swap] = indices[swap], indices[position]
    return indices


def epoch_indices(count: int, *, epoch: int, seed: int, shuffle: bool) -> list[int]:
    """The record order for one epoch.

    Unshuffled order is the file order, which is what a user reading a dataset
    expects. Shuffled order is a pure function of ``(seed, epoch)``, so two
    processes with ``DETERMINISTIC=True`` see the same batches.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if not shuffle:
        return list(range(count))
    return permutation(count, seed=seed_from(seed, epoch, "epoch"))


def batch_slices(indices: Sequence[int], batch_size: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` slices over ``indices``.

    The final batch may be short. It is never padded and never dropped: a
    dataset smaller than ``batch_size`` still yields exactly one batch, which is
    what lets the shipped two-record seed files train at all.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    total = len(indices)
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)
