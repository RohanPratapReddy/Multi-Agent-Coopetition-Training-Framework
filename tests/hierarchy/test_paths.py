"""Path grammar (document 10 section 3.3).

The rejection table is the specification: an empty path, a leading or trailing
separator, an empty segment, a character outside the allowed set, and a segment of
exactly ``.`` or ``..`` all fail with the offending path in the message.
"""

from __future__ import annotations

import pytest

from mactf.coopetition.tree import TreeValidationError, validate_path


ACCEPTED = [
    "peer-a",
    "peer-a/sub-1",
    "peer-a/sub-1/mini-2",
    "peer-1",
    "peer_1",
    "peer.1",
    "peer:1",
    "A0/B1/C2",
    "peer-a/micro-1/nano-2",
]

REJECTED = [
    ("", "cannot be empty"),
    ("/peer-a", "leading separator"),
    ("peer-a/", "trailing separator"),
    ("peer-a//sub-1", "empty segment"),
    ("peer a", "outside"),
    ("peer-a/../etc", "traversal"),
    ("..", "parent-directory marker"),
    (".", "parent-directory marker"),
    ("peer-a/sub 1", "outside"),
    ("peer-a/$sub", "outside"),
    ("peer-a/sub-1\t", "outside"),
]


@pytest.mark.parametrize("path", ACCEPTED)
def test_valid_paths_are_accepted(path):
    assert validate_path(path) == path


@pytest.mark.parametrize("path,reason", REJECTED)
def test_invalid_paths_are_rejected_with_the_path(path, reason):
    with pytest.raises(TreeValidationError) as info:
        validate_path(path)
    message = str(info.value)
    assert reason in message
    if path:
        assert repr(path) in message or path in message


def test_a_non_string_path_is_rejected():
    with pytest.raises(TreeValidationError, match="must be a string"):
        validate_path(3)  # type: ignore[arg-type]


def test_a_directory_traversal_is_rejected_because_paths_become_file_paths():
    """The message says why, so the rejection is not mistaken for pedantry."""
    with pytest.raises(TreeValidationError, match="filesystem locations"):
        validate_path("peer-a/../../etc/passwd")


def test_deep_paths_are_accepted():
    path = "/".join(f"level-{index}" for index in range(1, 10))
    assert validate_path(path) == path
