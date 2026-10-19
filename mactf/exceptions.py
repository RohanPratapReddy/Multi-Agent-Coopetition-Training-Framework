"""Exception hierarchy for MACTF."""

from __future__ import annotations


class MACTFError(Exception):
    """Base class for every framework error."""


class ImproperlyConfigured(MACTFError):
    """Settings are missing or contradictory."""


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
    """A management command failed."""
