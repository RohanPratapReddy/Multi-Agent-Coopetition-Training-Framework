"""Exception hierarchy for MACTF."""

from __future__ import annotations

from typing import Any


class MACTFError(Exception):
    """Base class for every framework error."""


class ImproperlyConfigured(MACTFError):
    """Settings are missing or contradictory."""


class ValidationError(ImproperlyConfigured):
    """A spec named an unknown type or a keyword the constructor does not accept.

    A subclass of :class:`ImproperlyConfigured` because a bad spec *is* a bad
    configuration: callers that already catch that for settings should not need
    a second branch. The HTTP layer maps it to 422 alongside its own
    ``mactf.http.exceptions.ValidationError``.
    """


class AppRegistryNotReady(MACTFError):
    """An app registry operation was attempted before population."""


class RuleViolation(MACTFError):
    """A ledger event referenced an unknown or incompatible rule."""


class ComplianceError(MACTFError):
    """A score cannot be produced without breaching framework invariants."""


class SandboxError(MACTFError):
    """Base class for sandbox provider failures."""


class SandboxUnavailable(SandboxError):
    """No provider can satisfy the requested isolation level."""


class SandboxLaunchError(SandboxError):
    """A sandbox process failed to start or exited unexpectedly."""


class ContractError(MACTFError):
    """A peer module does not satisfy the peer output contract."""


class CheckError(MACTFError):
    """A system check failed."""


class CommandError(MACTFError):
    """A management command failed in a way the operator can act on.

    ``returncode`` is the process exit code (``mactf.management.exitcodes.ExitCode``);
    the default ``1`` keeps every pre-existing ``raise CommandError(...)`` behaving as before.
    """

    def __init__(self, message: str = "", *, returncode: int = 1) -> None:
        super().__init__(message)
        self.returncode = int(returncode)

    @classmethod
    def usage(cls, message: str) -> "CommandError":
        return cls(message, returncode=2)

    @classmethod
    def not_found(cls, message: str) -> "CommandError":
        return cls(message, returncode=3)

    @classmethod
    def conflict(cls, message: str) -> "CommandError":
        return cls(message, returncode=4)

    @classmethod
    def unavailable(cls, message: str) -> "CommandError":
        return cls(message, returncode=5)


class ResourceNotFound(MACTFError):
    """A named run, episode, checkpoint, rule, or peer does not exist."""


class StateConflict(MACTFError):
    """The target's state forbids the action. ``details`` is machine-readable.

    Named ``StateConflict`` because ``mactf.http.exceptions.Conflict`` (the 409
    response class) already exists and a module that handles both must not alias.
    """

    def __init__(self, message: str = "", **details: object) -> None:
        super().__init__(message)
        self.details = dict(details)


class CapabilityUnavailable(MACTFError):
    """The environment lacks something the action needs."""


class InvalidArgument(MACTFError):
    """A value is unacceptable. ``field`` names the argument or body field, when there is one."""

    def __init__(self, message: str = "", *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class DistributedError(MACTFError):
    """A distributed run cannot proceed: unassigned peers, a bad membership, a lost barrier."""


class ParameterIntegrityError(MACTFError):
    """A parameter blob failed its checksum or its expected shapes (document 05 section 3)."""


class DataError(MACTFError):
    """A dataset record is malformed or inconsistent.

    Carries the file, line, record id, and offending key so a dataset typo is a
    precise failure instead of a changed training run.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Any = None,
        line: int | None = None,
        record_id: str | None = None,
        key: str | None = None,
    ) -> None:
        self.path = path
        self.line = line
        self.record_id = record_id
        self.key = key
        location = ""
        if path is not None:
            location = str(path)
            if line is not None:
                location += f":{line}"
            location += ": "
        detail = location + message
        if record_id:
            detail += f" (record {record_id!r})"
        if key:
            detail += f" [key {key!r}]"
        super().__init__(detail)
