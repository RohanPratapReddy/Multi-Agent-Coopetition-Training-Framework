"""The whole coupling between transport and authentication.

``mactf.http`` knows ``Principal(name, scopes, anonymous)`` and ``Denial``. What a
scope means, and where tokens come from, is the app's business
(``mactf.contrib.api.auth``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Principal:
    """Who is calling, and what they may do."""

    name: str = "anonymous"
    scopes: frozenset[str] = frozenset()
    anonymous: bool = True
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def authenticated(
        cls, name: str, scopes: Iterable[str] = (), **metadata: Any
    ) -> "Principal":
        return cls(name=name, scopes=frozenset(scopes), anonymous=False, metadata=dict(metadata))

    def has_scope(self, scope: str) -> bool:
        """A ``*`` scope grants everything; ``area:*`` grants every ``area:`` scope."""
        if "*" in self.scopes or scope in self.scopes:
            return True
        area = scope.split(":", 1)[0]
        return f"{area}:*" in self.scopes

    def has_scopes(self, scopes: Iterable[str]) -> bool:
        return all(self.has_scope(scope) for scope in scopes)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "scopes": sorted(self.scopes), "anonymous": self.anonymous}


ANONYMOUS = Principal()


@dataclass(frozen=True)
class Denial:
    """A resolver's refusal. ``status`` is 401 (who are you) or 403 (not allowed)."""

    status: int = 401
    detail: str = "authentication required"
    headers: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.status not in (401, 403):
            raise ValueError(f"a Denial is 401 or 403, got {self.status}")
