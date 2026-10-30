"""Parameter blobs: checksum, shape and compression guarantees (document 05 section 10)."""

from __future__ import annotations

import base64

import pytest
import torch

from mactf.distributed.transport import decode_tensors, encode_tensors, vector_checksum
from mactf.exceptions import ParameterIntegrityError


def test_roundtrip_compressed_and_raw():
    payload = {"a": torch.randn(7, 3), "b": torch.arange(5, dtype=torch.float32)}
    for compress in (True, False):
        blob, checksum = encode_tensors(payload, compress=compress)
        decoded = decode_tensors(blob, checksum, expected=[("a", (7, 3)), ("b", (5,))])
        assert set(decoded) == {"a", "b"}
        assert torch.equal(decoded["a"], payload["a"])


def test_compression_is_lossless_for_negative_zero_and_inf():
    vector = torch.tensor([-0.0, 0.0, float("inf"), -float("inf"), 1e-45, 3.4e38, 1.5])
    blob, checksum = encode_tensors({"v": vector}, compress=True)
    out = decode_tensors(blob, checksum)["v"]
    assert out.view(torch.int32).tolist() == vector.view(torch.int32).tolist()  # bit-for-bit, so -0.0 survives
    assert torch.signbit(out[0]) and not torch.signbit(out[1])


def test_tampered_payload_fails_the_checksum():
    blob, checksum = encode_tensors({"v": torch.ones(64)}, compress=False)
    raw = bytearray(base64.b64decode(blob))
    raw[-3] ^= 0xFF
    with pytest.raises(ParameterIntegrityError, match="checksum"):
        decode_tensors(base64.b64encode(bytes(raw)).decode(), checksum)


def test_truncated_body_fails_loudly():
    blob, checksum = encode_tensors({"v": torch.randn(500)}, compress=True)
    with pytest.raises(ParameterIntegrityError):
        decode_tensors(blob[: len(blob) // 2], checksum)
    with pytest.raises(ParameterIntegrityError):
        decode_tensors("!!not base64!!", checksum)
    with pytest.raises(ParameterIntegrityError):
        decode_tensors("", checksum)


def test_shape_mismatch_names_both_shapes():
    blob, checksum = encode_tensors({"w": torch.zeros(4, 2)})
    with pytest.raises(ParameterIntegrityError) as caught:
        decode_tensors(blob, checksum, expected=[("w", (2, 4))])
    message = str(caught.value)
    assert "'w'" in message and "(4, 2)" in message and "(2, 4)" in message


def test_missing_tensor_is_named():
    blob, checksum = encode_tensors({"w": torch.zeros(2)})
    with pytest.raises(ParameterIntegrityError, match="'x'"):
        decode_tensors(blob, checksum, expected=[("x", (2,))])


def test_vector_checksum_is_stable_and_sensitive():
    vector = torch.arange(10, dtype=torch.float32)
    assert vector_checksum(vector) == vector_checksum(vector.clone())
    assert vector_checksum(vector) != vector_checksum(vector + 1e-6)
    assert vector_checksum(vector) != vector_checksum(vector.double())
    assert vector_checksum(vector[2:]) == vector_checksum(vector[2:].clone())
