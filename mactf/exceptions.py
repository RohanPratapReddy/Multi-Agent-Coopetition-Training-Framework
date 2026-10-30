"""Exception hierarchy for MACTF."""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from typing import Any

_CODE_PREFIX = re.compile(r"^(E\d{3}): ")


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


class ComponentError(ValidationError):
    """Base of every error about a *component* (a registered, configurable part).

    ``code`` is the taxonomy code (document 17 section 17); ``kind`` and ``name`` say which
    component, when one is known. A message that already starts with a code (``E012: ...``)
    keeps that code, so one class can carry the sibling codes of its family without a
    subclass per code.
    """

    code = "E005"

    def __init__(
        self,
        message: str,
        *,
        kind: str | None = None,
        name: str | None = None,
        code: str | None = None,
    ) -> None:
        found = _CODE_PREFIX.match(message)
        self.code = code or (found.group(1) if found else type(self).code)
        self.kind = kind
        self.name = name
        super().__init__(message if found else f"{self.code}: {message}")


class ComponentMissing(ComponentError):
    """``E001``: no such component. ``known`` is what *would* have been accepted.

    The same exception names a missing *slot* (``E010``): the slot table's ``get`` raises
    ``ComponentMissing("slot", path, sorted(paths))``.
    """

    code = "E001"

    def __init__(self, kind: str, name: str, known: Sequence[str] = ()) -> None:
        self.known = tuple(known)
        listed = ", ".join(repr(item) for item in self.known) or "none registered"
        # A typo is the usual cause: name the nearest registered name (I8).
        near = difflib.get_close_matches(str(name), [str(item) for item in self.known], n=1)
        hint = f"; did you mean {near[0]!r}?" if near else ""
        super().__init__(
            f"no {kind} named {name!r}; accepted: {listed}{hint}",
            kind=kind,
            name=name,
            code="E010" if kind == "slot" else "E001",
        )


class ComponentCollision(ComponentError):
    """``E002``: the name is taken and ``replace`` was not given."""

    code = "E002"

    def __init__(self, kind: str, name: str, detail: str = "") -> None:
        super().__init__(
            f"{kind} {name!r} is already registered" + (f": {detail}" if detail else ""),
            kind=kind,
            name=name,
        )


class ComponentLocked(ComponentError):
    """``E003``: the component, the slot or the boundary may not be changed.

    ``reason`` is the one-line why (``COMPONENTS.LOCKED``, origin rank, mutability, boundary).
    """

    code = "E003"

    def __init__(self, kind: str, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{kind} {path!r} cannot be changed: {reason}", kind=kind, name=path)


class UnsafePolicyDenied(ComponentLocked):
    """``E009``: a policy declared ``unsafe`` and the operator has not allowed that."""

    code = "E009"

    def __init__(self, name: str, reason: str) -> None:
        # Skip ComponentLocked.__init__ (its message names a *slot*); keep its attributes.
        ComponentError.__init__(
            self,
            f"trainer_policy {name!r} cannot be used: {reason}",
            kind="trainer_policy",
            name=name,
            code="E009",
        )
        self.path = name
        self.reason = reason


class ComponentParamError(ComponentError):
    """``E004``: a spec key the component does not accept, or a value out of range.

    ``problems`` is complete.
    """

    code = "E004"

    def __init__(self, kind: str, name: str, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(f"{kind}:{name}: " + "; ".join(self.problems), kind=kind, name=name)


class ComponentSpecError(ComponentError):
    """``E005`` (or ``E011``/``E012`` by message prefix): the spec is malformed."""

    code = "E005"

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(
            f"{kind} spec: {detail}" if not _CODE_PREFIX.match(detail) else detail, kind=kind
        )


class ComponentImportDenied(ComponentError):
    """``E006``: an ``import:`` spec on an untrusted surface, or a module outside the allow-list."""

    code = "E006"

    def __init__(self, target: str, reason: str) -> None:
        self.target = target
        super().__init__(f"import {target!r} refused: {reason}")


class ComponentContractError(ComponentError):
    """``E007``: the factory or instance does not meet its kind's contract.

    ``problems`` is complete. A probe failure carries its own code (``E047``, ``E056``) by
    message prefix of the first problem.
    """

    code = "E007"

    def __init__(
        self, kind: str, name: str, problems: Sequence[str], *, code: str | None = None
    ) -> None:
        self.problems = tuple(problems)
        text = "; ".join(self.problems)
        super().__init__(
            f"{kind}:{name} breaks its contract: {text}", kind=kind, name=name, code=code
        )


class ComponentBuildError(ComponentError):
    """``E008``: a factory (or a plugin module) raised. The cause is chained, never swallowed."""

    code = "E008"

    def __init__(self, kind: str, name: str, error: BaseException | str) -> None:
        self.cause = error if isinstance(error, BaseException) else None
        text = (
            f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        )
        super().__init__(f"could not build {kind} {name!r}: {text}", kind=kind, name=name)


class ComponentMigrationError(ComponentError):
    """``E048``: a swap's state migration refused or failed; the old component keeps running."""

    code = "E048"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AppRegistryNotReady(MACTFError):
    """An app registry operation was attempted before population."""


class RuleViolation(MACTFError):
    """A ledger event referenced an unknown or incompatible rule."""


class ComplianceError(MACTFError):
    """A score cannot be produced without breaching framework invariants."""


class ComplianceGuardViolation(ComplianceError):
    """The scoring guard refused a stage's output (``C001``-``C012``). ``findings`` is every violation."""

    def __init__(self, findings: Sequence[Mapping[str, object]]) -> None:
        self.findings = tuple(dict(item) for item in findings)
        self.code = str(self.findings[0].get("code", "C001")) if self.findings else "C001"
        listed = "; ".join(
            f"{item.get('code')} {item.get('message', '')}".strip() for item in self.findings
        )
        super().__init__(f"{self.code}: scoring guard refused the output: {listed}")


class StepGuardViolation(ComplianceError):
    """The step guard refused a policy's step (``S001``-``S009``). ``path`` is the node."""

    def __init__(self, path: str, violations: Sequence[Mapping[str, object]]) -> None:
        self.path = path
        self.violations = tuple(dict(item) for item in violations)
        self.code = str(self.violations[0].get("code", "S001")) if self.violations else "S001"
        listed = "; ".join(
            f"{item.get('code')} {item.get('message', '')}".strip() for item in self.violations
        )
        super().__init__(f"{self.code}: step guard refused node {path!r}: {listed}")


class WeightGuardViolation(ComplianceError):
    """``E063``: a weight strategy's output broke a framework invariant under MODE=enforce."""

    code = "E063"

    def __init__(self, strategy: str, violations: Sequence[Any]) -> None:
        self.strategy = strategy
        self.violations = tuple(violations)
        listed = "; ".join(str(item) for item in self.violations)
        super().__init__(f"E063: weight_update {strategy!r} broke the weight contract: {listed}")


class StrategyGuardViolation(ComplianceError):
    """``E073``: a peer strategy broke the coordination contract (hard rule, or ``MODE=enforce``)."""

    code = "E073"

    def __init__(self, strategy: str, violations: Sequence[Any]) -> None:
        self.strategy = strategy
        self.violations = tuple(violations)
        listed = "; ".join(str(item) for item in self.violations)
        super().__init__(
            f"E073: peer_strategy {strategy!r} broke the coordination contract: {listed}"
        )


class LedgerReadOnlyError(ComplianceError):
    """``C007``: a stage or plugin tried to write to the ledger (``record``, ``extend``, ``clear``)."""

    code = "C007"

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"C007: the ledger is read-only for scoring stages; {operation!r} is refused"
        )


class MutationError(ImproperlyConfigured):
    """A tree mutation was refused, or failed and was rolled back (``E080``..``E089``).

    ``code`` is the taxonomy code of the first (most severe) problem; ``findings`` is every
    problem a dry run found, rendered as ``E08x [path]: message``; ``rolled_back`` is true when
    a commit had started and was undone. ``E089`` with ``rolled_back=False`` means the undo
    itself failed and the trainer must be discarded.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        findings: Sequence[str] = (),
        rolled_back: bool = False,
    ) -> None:
        self.code = str(code)
        self.findings = tuple(findings)
        self.rolled_back = bool(rolled_back)
        detail = "; ".join(self.findings)
        super().__init__(f"{self.code}: {message}" + (f" [{detail}]" if detail else ""))


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
