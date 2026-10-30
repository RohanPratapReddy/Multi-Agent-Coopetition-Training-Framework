"""Parameter vectors on the wire: serialise, compress, checksum, verify (document 05 section 3).

A blob is ``base64( tag || payload )`` where ``tag`` is ``Z`` (zlib level 6) or ``R`` (raw)
and ``payload`` is a ``torch.save`` of ``{name: tensor}``. The checksum is blake2b over the
*uncompressed* ``torch.save`` bytes, so it also covers a corrupt decompression. Every
failure, including a truncated body that no longer base64-decodes, is a
:class:`~mactf.exceptions.ParameterIntegrityError`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import zlib
from collections.abc import Mapping, Sequence

import torch

from ..exceptions import ParameterIntegrityError

__all__ = ["decode_tensors", "encode_tensors", "vector_checksum"]

_COMPRESSED = b"Z"
_RAW = b"R"


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def encode_tensors(payload: Mapping[str, torch.Tensor], *, compress: bool = True) -> tuple[str, str]:
    """``(base64, blake2b hex checksum)`` of a ``torch.save`` blob of ``payload``."""
    buffer = io.BytesIO()
    torch.save({str(name): tensor.detach().cpu().contiguous() for name, tensor in payload.items()}, buffer)
    raw = buffer.getvalue()
    body = _COMPRESSED + zlib.compress(raw, 6) if compress else _RAW + raw
    return base64.b64encode(body).decode("ascii"), _digest(raw)


def decode_tensors(
    b64: str,
    checksum: str,
    *,
    expected: Sequence[tuple[str, tuple[int, ...]]] | None = None,
) -> dict[str, torch.Tensor]:
    """Verify ``checksum`` and, when given, every tensor's name and shape; then load."""
    try:
        body = base64.b64decode(b64.encode("ascii"), validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise ParameterIntegrityError(f"parameter blob is not valid base64: {error}") from error
    if not body:
        raise ParameterIntegrityError("parameter blob is empty")
    tag, data = body[:1], body[1:]
    try:
        if tag == _COMPRESSED:
            raw = zlib.decompress(data)
        elif tag == _RAW:
            raw = data
        else:
            raise ParameterIntegrityError(f"unknown parameter blob encoding {tag!r}")
    except zlib.error as error:
        raise ParameterIntegrityError(f"parameter blob is truncated or corrupt: {error}") from error
    actual = _digest(raw)
    if actual != checksum:
        raise ParameterIntegrityError(
            f"parameter checksum mismatch: expected {checksum[:16]}..., computed {actual[:16]}..."
        )
    try:
        tensors = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - a bad pickle stream is an integrity failure
        raise ParameterIntegrityError(f"parameter blob does not deserialise: {error}") from error
    if not isinstance(tensors, dict):
        raise ParameterIntegrityError("parameter blob does not hold a tensor mapping")
    for name, shape in expected or ():
        if name not in tensors:
            raise ParameterIntegrityError(f"tensor {name!r} is missing from the parameter blob")
        found = tuple(tensors[name].shape)
        if found != tuple(shape):
            raise ParameterIntegrityError(
                f"tensor {name!r} has shape {found}, expected {tuple(shape)}: "
                "the worker is running a different peer layout"
            )
    return dict(tensors)


def vector_checksum(vector: torch.Tensor) -> str:
    """blake2b of a tensor's raw element bytes (dtype and shape are part of the digest)."""
    flat = vector.detach().cpu().contiguous().clone()
    size = flat.numel() * flat.element_size()
    data = bytes(flat.untyped_storage())[:size]
    header = f"{flat.dtype}|{tuple(flat.shape)}|".encode("ascii")
    return _digest(header + data)
