"""The exit-code contract of every ``mactf`` command, and one classifier for exceptions."""

from __future__ import annotations

import enum

from ..exceptions import (
    AppRegistryNotReady,
    CapabilityUnavailable,
    CommandError,
    ImproperlyConfigured,
    InvalidArgument,
    MACTFError,
    ResourceNotFound,
    SandboxUnavailable,
    StateConflict,
)


class ExitCode(enum.IntEnum):
    OK = 0
    #: The operation ran and the answer is "no" or "it failed": a check failed, a
    #: checkpoint did not verify, a run failed, a plan is not applied.
    FAILURE = 1
    #: The invocation is wrong: bad arguments, unknown command, bad settings.
    USAGE = 2
    #: A named thing does not exist: run, episode, checkpoint, rule, peer, key.
    NOT_FOUND = 3
    #: The thing exists but its state forbids the action: run is live, id is taken,
    #: file would be overwritten, schema is behind.
    CONFLICT = 4
    #: The environment cannot do it: no QEMU, no IPython, port in use, extra missing.
    UNAVAILABLE = 5
    #: --timeout elapsed (the GNU `timeout` convention).
    TIMEOUT = 124
    #: A bug in mactf. Always printed with a request for --traceback.
    SOFTWARE = 70
    #: SIGINT or SIGTERM was honoured.
    INTERRUPTED = 130
    #: The reader of stdout went away (| head).
    PIPE = 141


# Exceptions defined in modules that import torch (orchestration.live) are matched by
# class name along the MRO, so the classifier never imports them.
_NOT_FOUND_NAMES = frozenset({"RunNotFound"})
_CONFLICT_NAMES = frozenset({"RunStateError", "RunExists"})


def _mro_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def classify(exc: BaseException) -> tuple[int, str]:
    """The exit code and the one-line stderr message for an exception."""
    message = str(exc)
    if isinstance(exc, KeyboardInterrupt):
        return int(ExitCode.INTERRUPTED), "interrupted"
    if isinstance(exc, BrokenPipeError):
        return int(ExitCode.PIPE), ""
    if isinstance(exc, CommandError):
        return int(exc.returncode), f"CommandError: {message}"
    if isinstance(exc, (ImproperlyConfigured, AppRegistryNotReady)):
        return int(ExitCode.USAGE), f"ImproperlyConfigured: {message}"
    if isinstance(exc, InvalidArgument):
        return int(ExitCode.USAGE), f"error: {message}"
    names = _mro_names(exc)
    if isinstance(exc, ResourceNotFound) or names & _NOT_FOUND_NAMES:
        # LookupError subclasses quote their message (KeyError-style); unwrap it.
        text = exc.args[0] if exc.args and isinstance(exc.args[0], str) else message
        return int(ExitCode.NOT_FOUND), f"NotFound: {text}"
    if isinstance(exc, StateConflict) or names & _CONFLICT_NAMES:
        return int(ExitCode.CONFLICT), f"Conflict: {message}"
    if isinstance(exc, (CapabilityUnavailable, SandboxUnavailable)):
        return int(ExitCode.UNAVAILABLE), f"Unavailable: {message}"
    if isinstance(exc, MACTFError):
        return int(ExitCode.FAILURE), f"{type(exc).__name__}: {message}"
    if isinstance(exc, OSError):
        return int(ExitCode.FAILURE), f"error: {exc.strerror or message}"
    return int(ExitCode.SOFTWARE), f"internal error: {type(exc).__name__}: {message}"
