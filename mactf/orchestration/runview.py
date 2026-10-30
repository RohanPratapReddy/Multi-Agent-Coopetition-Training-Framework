"""Pure functions over registry run rows: decoding, id resolution, summaries, liveness.

Document 16, section 7.4. No torch, no HTTP. The API and the CLI both read runs through here
so a run looks the same on both surfaces.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Mapping

from ..exceptions import InvalidArgument, ResourceNotFound, StateConflict
from .recovery import ACTIVE_STATUSES, validate_run_id
from .runlock import RunLock

if TYPE_CHECKING:  # pragma: no cover
    from ..db import Registry

__all__ = [
    "decode_json_columns",
    "decode_run",
    "liveness",
    "resolve_run_id",
    "run_duration",
    "summarize_run",
]


def decode_json_columns(row: Mapping[str, Any], *columns: str) -> dict[str, Any]:
    """A copy of ``row`` with each named JSON text column decoded; non-JSON text stays as is."""
    out = dict(row)
    for column in columns:
        value = out.get(column)
        if isinstance(value, (str, bytes)):
            try:
                out[column] = json.loads(value)
            except ValueError:
                pass
    return out


def decode_run(row: Mapping[str, Any]) -> dict[str, Any]:
    """``decode_json_columns(row, "settings", "runtime", "sandbox")`` (the API's ``run_payload``)."""
    return decode_json_columns(row, "settings", "runtime", "sandbox")


def resolve_run_id(
    registry: "Registry", text: str | None, *, statuses: Collection[str] | None = None
) -> str:
    """The run a user meant by ``text``.

    ``None`` or ``"latest"``: the newest run (restricted to ``statuses`` when given).
    Otherwise ``text`` must pass ``validate_run_id``, then: an exact id wins; else a unique
    prefix wins; an ambiguous prefix is ``InvalidArgument`` listing up to five candidates;
    no match is ``ResourceNotFound``. ``statuses`` restricts exact matches too, so resolving a
    completed run against ``RESUMABLE_STATUSES`` is a ``StateConflict`` naming its status,
    not a silent miss.
    """
    wanted = sorted(statuses) if statuses is not None else None
    if text is None or text == "latest":
        rows = registry.list_runs(limit=1, status=wanted)
        if not rows:
            what = "no runs" if wanted is None else f"no {'/'.join(wanted)} runs"
            raise ResourceNotFound(f"{what} in the registry")
        return str(rows[0]["id"])
    validate_run_id(text)
    exact = registry.get_run(text)
    if exact is not None:
        if wanted is not None and exact["status"] not in wanted:
            raise StateConflict(
                f"run {text} is {exact['status']}; this needs a run that is {' or '.join(wanted)}",
                run_id=text, status=exact["status"])
        return str(exact["id"])
    matches = registry.list_runs(limit=6, id_prefix=text, status=wanted)
    if not matches and wanted is not None:
        elsewhere = registry.list_runs(limit=1, id_prefix=text)
        if elsewhere:
            raise StateConflict(
                f"run {elsewhere[0]['id']} is {elsewhere[0]['status']}; "
                f"this needs a run that is {' or '.join(wanted)}",
                run_id=elsewhere[0]["id"], status=elsewhere[0]["status"])
    if not matches:
        raise ResourceNotFound(f"no run {text!r}")
    if len(matches) > 1:
        listed = ", ".join(str(row["id"]) for row in matches[:5])
        more = "" if len(matches) <= 5 else ", ..."
        raise InvalidArgument(
            f"run prefix {text!r} is ambiguous ({listed}{more}); use more characters",
            field="run")
    return str(matches[0]["id"])


def summarize_run(
    registry: "Registry", run_id: str, *, row: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """``{"run", "latest_episode", "latest_aggregation", "episodes"}``: ``GET /runs/{id}/summary``.

    ``row`` replaces the stored row (the API lays the live status over it). ``latest_episode``
    is the decoded episode row or ``None``; it is one primary-key lookup, not offset arithmetic.
    """
    if row is None:
        row = registry.get_run(run_id)
        if row is None:
            raise ResourceNotFound(f"no run {run_id!r}")
    latest = registry.latest_episode(run_id)
    return {
        "run": decode_run(row),
        "latest_episode": registry.get_episode(run_id, latest) if latest is not None else None,
        "latest_aggregation": registry.latest_aggregation(run_id),
        "episodes": registry.count_episodes(run_id),
    }


def liveness(lock_dir: Path, row: Mapping[str, Any]) -> str:
    """``"live"``: active status and a live lock holder. ``"stale"``: active status, no holder.
    ``"idle"``: any other status."""
    if row.get("status") not in ACTIVE_STATUSES:
        return "idle"
    return "live" if RunLock.holder(lock_dir, str(row["id"])) is not None else "stale"


def run_duration(row: Mapping[str, Any], *, now: float | None = None) -> float | None:
    """``finished_at - started_at``; for an active row ``now - started_at``; ``None`` if unknowable."""
    started = row.get("started_at")
    if started is None:
        return None
    finished = row.get("finished_at")
    if finished is not None:
        return max(0.0, float(finished) - float(started))
    if row.get("status") in ACTIVE_STATUSES:
        return max(0.0, (time.time() if now is None else float(now)) - float(started))
    return None
