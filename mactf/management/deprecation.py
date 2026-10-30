"""The deprecation mechanism (document 16, section 11.5).

Nothing is deprecated today; this is what makes the first deprecation a table row and a
test instead of a design discussion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RELEASE = re.compile(r"(\d+)\.(\d+)")


def _release(label: str, text: str) -> tuple[int, int]:
    match = _RELEASE.fullmatch(text)
    if match is None:
        raise ValueError(f"{label} {text!r} must be '<major>.<minor>', for example '0.3'")
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class Deprecation:
    surface: str          # exactly what the user typed: "mactf runs list --newest"
    replacement: str      # what to type instead
    since: str            # the first release that warns: "0.3"
    removal: str          # the release that removes it: "0.5"
    reason: str = ""

    def __post_init__(self) -> None:
        since_major, since_minor = _release("since", self.since)
        removal_major, removal_minor = _release("removal", self.removal)
        # At least two minor releases of warning, unless a major version intervenes.
        if not (removal_major > since_major
                or (removal_major, removal_minor) >= (since_major, since_minor + 2)):
            raise ValueError(
                f"{self.surface}: removal {self.removal} gives fewer than two minor releases "
                f"after {self.since}")


#: Every deprecated surface that is still accepted. Empty today (see 11.5).
DEPRECATIONS: tuple[Deprecation, ...] = ()


def deprecation_message(entry: Deprecation) -> str:
    text = (f"{entry.surface} is deprecated since {entry.since} and will be removed in "
            f"{entry.removal}; use {entry.replacement}")
    return f"{text} ({entry.reason})" if entry.reason else text
