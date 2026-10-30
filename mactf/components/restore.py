"""Read a checkpoint's ``components`` manifest and say what it means for this run.

Document 17 section 14.5. The restorer never edits a slot: configuration comes from settings
(section 5), and a restore that quietly re-pointed a slot would make ``conf/`` a lie. It
verifies, reports per slot, and raises only when continuing would mean training with the
wrong code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..exceptions import ImproperlyConfigured
from .fingerprint import ENTRY_SCHEMA, IDENTITY_KEY, diff, entry_for, recorded_slots, render

log = logging.getLogger("mactf.components")

#: Statuses that stop a resume in both modes.
FATAL = frozenset({"missing", "malformed", "unsupported"})

#: The note a checkpoint with no manifest produces, once (P5).
ABSENT_NOTE = "W053: this checkpoint records no components; nothing to verify"


@dataclass(frozen=True)
class Finding:
    """What the restorer decided about one slot."""

    kind: str
    slot: str
    status: str
    code: str | None = None
    message: str = ""
    saved: Mapping[str, Any] | None = None
    live: Mapping[str, Any] | None = None

    @property
    def fatal(self) -> bool:
        """Whether this finding stops the resume: the three unusable statuses, and any
        difference under ``COMPONENTS.STRICT_RESTORE`` (``E094``)."""
        return self.status in FATAL or self.code == "E094"

    def describe(self) -> str:
        prefix = f"{self.code}: " if self.code else ""
        return f"{prefix}{self.slot} [{self.status}] {self.message}".rstrip()


@dataclass
class RestorePlan:
    """The findings for one checkpoint, plus the notes a caller appends to its report."""

    findings: list[Finding] = field(default_factory=list)
    #: ``True`` when the checkpoint carried no manifest at all (an older file, or an
    #: all-built-in run). Nothing was verified and nothing may be claimed.
    absent: bool = False
    checked: int = 0

    # -- queries ----------------------------------------------------------
    @property
    def fatal(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.fatal]

    @property
    def warnings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.code and not finding.fatal]

    @property
    def ok(self) -> bool:
        return not self.fatal

    def status_of(self, slot: str) -> str | None:
        for finding in self.findings:
            if finding.slot == slot:
                return finding.status
        return None

    def notes(self) -> list[str]:
        """One line per coded finding; for an absent manifest, the single ``W053``."""
        if self.absent:
            return [ABSENT_NOTE]
        return [finding.describe() for finding in self.findings if finding.code]

    def describe(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.status] = counts.get(finding.status, 0) + 1
        return {
            "absent": self.absent,
            "checked": self.checked,
            "statuses": counts,
            "findings": [
                {
                    "kind": finding.kind,
                    "slot": finding.slot,
                    "status": finding.status,
                    "code": finding.code,
                    "message": finding.message,
                }
                for finding in self.findings
            ],
        }

    # -- raising ----------------------------------------------------------
    def raise_for_errors(self) -> None:
        """Raise for the fatal findings, naming all of them.

        One exception with every fatal slot listed is more useful than the first one: an
        operator who is missing two plugins should install both, not discover them one
        resume at a time.
        """
        fatal = self.fatal
        if not fatal:
            return
        lines = [finding.describe() for finding in fatal]
        raise ImproperlyConfigured(
            lines[0] + ("" if len(lines) == 1 else "\n" + "\n".join(lines[1:]))
        )


class ComponentRestorer:
    """Compare a checkpoint's manifest with the live slot table."""

    def __init__(self, slots: Any, *, policy: Any = None) -> None:
        #: A ``SlotTable`` or a ``fingerprint.SlotView``: anything with ``get(path)``.
        self.slots = slots
        #: ``HubPolicy``; ``strict_restore`` is the only field read, and ``None`` means "read
        #: it from the hub when the plan is built", which keeps this class testable bare.
        self.policy = policy

    # -- planning ---------------------------------------------------------
    def plan(self, payload: Any) -> RestorePlan:
        """Plan against a checkpoint payload, a raw mapping, or a bare manifest.

        ``payload`` is what ``MACTFCheckpoint.load`` returns (a ``CheckpointPayload``), a
        ``torch.load`` result (a mapping with ``identity``), or already the manifest itself --
        the CLI passes the last form.
        """
        manifest = extract_manifest(payload)
        if manifest is None:
            log.info("%s", ABSENT_NOTE)
            return RestorePlan(absent=True)
        strict = self._strict()
        plan = RestorePlan()
        for kind, path in recorded_slots(manifest):
            plan.findings.append(self._check(kind, path, manifest[kind][path], strict=strict))
            plan.checked += 1
        for finding in plan.findings:
            if finding.code:
                (log.error if finding.fatal else log.warning)("%s", finding.describe())
        return plan

    def report(self, plan: RestorePlan) -> str:
        """A human summary, for the engine's ``startup_notes`` and ``mactf runs show``."""
        if plan.absent:
            return ABSENT_NOTE
        if not plan.findings:
            return "components: nothing recorded"
        mismatched = [finding for finding in plan.findings if finding.status == "mismatch"]
        return (
            f"components: {plan.checked} checked, {len(mismatched)} differ, "
            f"{len(plan.fatal)} fatal"
        )

    # -- one slot ---------------------------------------------------------
    def _check(self, kind: str, path: str, entry: Any, *, strict: bool) -> Finding:
        problem = _check_entry(kind, path, entry)
        if problem is not None:
            status, code, message = problem
            return Finding(kind, path, status, code, message, saved=None)

        saved = dict(entry)
        try:
            slot = self.slots.get(path)
        except Exception:  # noqa: BLE001 - ``ComponentMissing`` for a slot is an orphan here
            return Finding(
                kind, path, "orphan", "W052",
                "the run no longer has this slot; the entry is ignored",
                saved=saved, live=None,
            )

        live = entry_for(slot)
        if saved.get("name") != live.get("name") or saved.get("kind") != live.get("kind"):
            if not _is_registered(saved.get("kind"), saved.get("name")) and (
                saved.get("origin") != "runtime"
            ):
                return Finding(
                    kind, path, "missing", "E093",
                    f"needs component {saved.get('name')!r} of kind {saved.get('kind')!r}, which "
                    "is not registered; install the module that provides it "
                    "(COMPONENTS.MODULES or the mactf.components entry point) or resume "
                    "with a different checkpoint",
                    saved=saved, live=live,
                )
            # A ``runtime`` component is an in-code instance: no module can provide it, so
            # its absence is a difference to report (below), not a missing plugin.

        changes = diff({kind: {path: saved}}, {kind: {path: live}})
        if not changes:
            return Finding(kind, path, "ok", None, "", saved=saved, live=live)

        detail = render(changes)
        if strict:
            return Finding(
                kind, path, "mismatch", "E094",
                "differs from the checkpoint and COMPONENTS.STRICT_RESTORE is on:\n" + detail,
                saved=saved, live=live,
            )
        return Finding(
            kind, path, "mismatch", "W052",
            "differs from the checkpoint; this run uses the configured component:\n" + detail,
            saved=saved, live=live,
        )

    def _strict(self) -> bool:
        """``COMPONENTS.STRICT_RESTORE``, read off the effective ``HubPolicy``."""
        policy = self.policy
        if policy is None:
            try:
                from .registry import HUB

                policy = HUB.effective_policy()
            except Exception:  # noqa: BLE001 - a bare unit test has no hub
                return False
        return bool(getattr(policy, "strict_restore", False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_manifest(payload: Any) -> dict[str, dict[str, Any]] | None:
    """The manifest inside a payload, or ``None`` when the payload carries none.

    Four accepted shapes, because the call sites hand this a different object: a
    ``CheckpointPayload`` (``payload.identity``), a ``torch.load`` mapping (``["identity"]``),
    an identity mapping (it has a ``components`` key), and a manifest itself (the CLI's
    ``components diff``: ``{kind: {slot: entry}}``).

    Whatever is not interpretable is returned in a shape ``_check_entry`` reports as ``E095``
    (a non-mapping manifest or kind becomes one ``"?"`` entry holding the offending value),
    never silently dropped: a hand-edited file must not resume as though it recorded nothing.
    """
    if isinstance(payload, Mapping):
        if "identity" in payload and isinstance(payload["identity"], Mapping):
            raw = payload["identity"].get(IDENTITY_KEY)
        elif "identity" in payload:
            return None
        elif IDENTITY_KEY in payload:
            raw = payload[IDENTITY_KEY]
        else:
            raw = payload
    else:
        identity = getattr(payload, "identity", None)
        if not isinstance(identity, Mapping):
            return None
        raw = identity.get(IDENTITY_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return {IDENTITY_KEY: {"?": raw}}
    return {
        str(kind): (dict(entries) if isinstance(entries, Mapping) else {"?": entries})
        for kind, entries in raw.items()
    }


def _check_entry(kind: str, path: str, entry: Any) -> tuple[str, str, str] | None:
    """``(status, code, message)`` for a malformed or unsupported entry, else ``None``."""
    if not isinstance(entry, Mapping):
        return (
            "malformed", "E095",
            f"entry for {kind}/{path} is {type(entry).__name__}, not a mapping",
        )
    missing = [key for key in ("schema", "kind", "name") if key not in entry]
    if missing:
        return (
            "malformed", "E095",
            f"entry for {kind}/{path} lacks {', '.join(missing)}",
        )
    schema = entry["schema"]
    if not isinstance(schema, int) or isinstance(schema, bool):
        return ("malformed", "E095", f"entry for {kind}/{path} has a non-integer schema {schema!r}")
    if schema > ENTRY_SCHEMA or schema < 1:
        return (
            "unsupported", "E096",
            f"entry for {kind}/{path} uses component schema {schema}; this build understands "
            f"{ENTRY_SCHEMA}. Upgrade mactf, or resume with a build that wrote this file",
        )
    for key in ("kind", "name"):
        if not isinstance(entry[key], str) or not entry[key]:
            return (
                "malformed", "E095",
                f"entry for {kind}/{path} has a {key} that is not a non-empty string: "
                f"{entry[key]!r}",
            )
    if entry["kind"] != kind:
        return (
            "malformed", "E095",
            f"entry for {kind}/{path} says it is a {entry['kind']!r} component",
        )
    params = entry.get("params")
    if params is not None and not isinstance(params, Mapping):
        return (
            "malformed", "E095",
            f"entry for {kind}/{path} has params of type {type(params).__name__}, not a mapping",
        )
    return None


def _is_registered(kind: Any, name: Any) -> bool:
    """Whether the registry can still build this name (``E093`` depends on it)."""
    if not isinstance(kind, str) or not isinstance(name, str):
        return False
    try:
        from .registry import HUB

        return HUB.registry(kind).find(name) is not None
    except Exception:  # noqa: BLE001 - an unknown kind means "not registered here"
        return False
