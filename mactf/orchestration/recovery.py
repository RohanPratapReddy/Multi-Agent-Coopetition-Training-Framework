"""Checkpoint verification, rollback, branching, pruning, and stale-run reaping.

Document 16, section 7.2. Moved out of ``contrib/api/views/recovery.py`` so the CLI and the
API call one implementation. Module-level imports are the standard library, ``..exceptions``,
``..signals`` and ``.runlock``; ``load_checkpoint`` and ``MACTFCheckpoint`` (both pull in
``torch``) are imported inside the functions that read a checkpoint, so importing this
module is cheap.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .. import signals
from ..conf_files import is_authored
from ..exceptions import InvalidArgument, ResourceNotFound, StateConflict
from .runlock import RunLock

if TYPE_CHECKING:  # pragma: no cover
    from ..db import Registry

#: A run some process is (or claims to be) driving.
ACTIVE_STATUSES = frozenset({"queued", "running", "resuming"})
#: A run that can continue under its own id (RunManager.resume accepts exactly these).
RESUMABLE_STATUSES = frozenset({"paused", "interrupted"})
#: A run whose scorecard is final; restarting it would make that scorecard a lie.
FINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "dry_run"})

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")     # what api/views/base.py already enforces


def validate_run_id(text: str) -> str:
    """``text`` unchanged when it is a legal run id, else ``InvalidArgument``.

    A run id becomes part of checkpoint file names (``<run_id>-epNNNN.pt``), a lock file
    name, and a log directory name, so ``../x``, ``a/b``, ``a b`` and the empty string
    are refused here, once, instead of being trusted three times.
    """
    if not isinstance(text, str) or not RUN_ID_PATTERN.match(text):
        raise InvalidArgument(
            f"invalid run id {text!r}; use 1 to 64 letters, digits, '-' or '_'", field="run_id"
        )
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path, run_id: str) -> dict[str, Any]:
    """``{"ok", "problems", "content_hash"}`` for a v2 or a legacy checkpoint file.

    A missing file, an unreadable file, a v2 manifest that fails ``MACTFCheckpoint.verify``,
    and a legacy payload missing ``run_id``/``episode``/``peers`` are all results
    (``ok: false`` with problems), never exceptions.
    """
    target = Path(path)
    if not target.exists():
        return {"ok": False, "problems": ["file does not exist"], "content_hash": None}
    from ..runtime import load_checkpoint

    try:
        payload = load_checkpoint(target)
    except Exception as exc:  # noqa: BLE001 - an unreadable file is a verification result
        return {"ok": False, "problems": [f"unreadable: {type(exc).__name__}: {exc}"],
                "content_hash": _sha256(target)}
    if isinstance(payload, dict) and ("identity" in payload or "manifest" in payload):
        from .checkpoint_v2 import MACTFCheckpoint

        outcome = MACTFCheckpoint(target.parent, run_id=run_id).verify(target)
        return {"ok": bool(outcome.ok), "problems": [str(p) for p in outcome.problems],
                "content_hash": _sha256(target)}
    problems = [f"missing key {key!r}" for key in ("run_id", "episode", "peers") if key not in payload]
    return {"ok": not problems, "problems": problems, "content_hash": _sha256(target)}


def checkpoint_entry(row: Mapping[str, Any], *, deep: bool = False) -> dict[str, Any]:
    """The documented checkpoint object (keys as the API serializes them); ``deep`` verifies."""
    path = Path(row["path"])
    exists = path.exists()
    entry: dict[str, Any] = {
        "run_id": row["run_id"],
        "episode": int(row["episode"]),
        "path": str(path),
        "created_at": row.get("created_at"),
        "kappa": row.get("kappa"),
        "entropy": row.get("entropy"),
        "score": row.get("score"),
        "size_bytes": path.stat().st_size if exists else None,
        "verified": None,
        "sections": [],
        "best": path.name.endswith("-best.pt"),
        "reason": "",
        "exists": exists,
    }
    if deep and exists:
        outcome = verify_checkpoint(path, row["run_id"])
        entry["verified"] = outcome["ok"]
        entry["problems"] = outcome["problems"]
        try:
            from ..runtime import load_checkpoint

            payload = load_checkpoint(path)
            entry["sections"] = sorted(str(key) for key in payload)
            entry["schema_version"] = payload.get("schema_version", 1)
            entry["global_step"] = (payload.get("identity") or payload).get("global_step")
        except Exception:  # noqa: BLE001 - reported through ``verified``
            pass
    return entry


def find_checkpoint(registry: "Registry", run_id: str, episode: int) -> dict[str, Any]:
    """The registry row for ``episode``; ``ResourceNotFound`` when the run or the row is absent."""
    if registry.get_run(run_id) is None:
        raise ResourceNotFound(f"no run {run_id!r}")
    for row in registry.checkpoints(run_id):
        if int(row["episode"]) == int(episode):
            return row
    raise ResourceNotFound(f"run {run_id!r} has no checkpoint for episode {episode}")


def rollback(
    registry: "Registry", run_id: str, episode: int, *, live: bool, source: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Make ``episode`` the newest checkpoint of a run that nothing is driving.

    Newer checkpoint files are renamed ``*.rolledback`` (never deleted) and their registry
    rows removed; scorecards and rule events stay, because the audit trail is append-only.
    ``live`` is the caller's answer to "is a process driving this run": the API asks its
    RunManager, the CLI asks RunLock. ``source`` ("api", "cli") rides on the
    ``checkpoint_rolled_back`` signal.
    """
    run = registry.get_run(run_id)
    if run is None:
        raise ResourceNotFound(f"no run {run_id!r}")
    if live:
        raise StateConflict("a checkpoint must not be restored under an active loop", run_id=run_id)
    rows = registry.checkpoints(run_id)
    target = next((row for row in rows if int(row["episode"]) == int(episode)), None)
    if target is None or not Path(target["path"]).exists():
        raise ResourceNotFound(f"run {run_id!r} has no checkpoint file for episode {episode}")
    outcome = verify_checkpoint(target["path"], run_id)
    if not outcome["ok"]:
        raise StateConflict("the checkpoint failed verification", problems=outcome["problems"])
    newer = [row for row in rows if int(row["episode"]) > int(episode)]
    if dry_run:
        entry = checkpoint_entry(target, deep=True)
        entry["would_archive"] = [Path(row["path"]).name + ".rolledback"
                                  for row in newer if Path(row["path"]).exists()]
        entry["would_remove_rows"] = len(newer)
        entry["dry_run"] = True
        return entry
    archived: list[str] = []
    for row in newer:
        source_path = Path(row["path"])
        if source_path.exists():
            destination = source_path.with_name(source_path.name + ".rolledback")
            source_path.replace(destination)
            archived.append(destination.name)
    removed = registry.delete_checkpoints_after(run_id, episode)
    if run["status"] != "paused":
        registry.set_run_status(run_id, "interrupted", notes=f"rolled back to episode {episode}")
    signals.checkpoint_rolled_back.send(sender=run_id, run_id=run_id, episode=int(episode),
                                        reason=source)
    entry = checkpoint_entry(target, deep=True)
    entry["archived"] = archived
    entry["rows_removed"] = removed
    return entry


def branch_checkpoint(registry: "Registry", source_run: str, resume_from: str | int) -> tuple[Path, int]:
    """The checkpoint file a NEW run is seeded from: ``"latest"`` or an episode of ``source_run``.

    ``ResourceNotFound`` when the source run is unknown or has no checkpoint file on disk for
    the request. ``InvalidArgument`` when ``resume_from`` is neither ``latest`` nor an integer.
    """
    if registry.get_run(source_run) is None:
        raise ResourceNotFound(f"no run {source_run!r} to resume from")
    rows = registry.checkpoints(source_run)
    if str(resume_from) != "latest":
        try:
            wanted = int(resume_from)
        except (TypeError, ValueError):
            raise InvalidArgument(
                f"resume_from must be 'latest' or an episode number, got {resume_from!r}",
                field="resume_from") from None
        rows = [row for row in rows if int(row["episode"]) == wanted]
    rows = [row for row in rows if Path(row["path"]).exists()]
    if not rows:
        raise ResourceNotFound(
            f"run {source_run!r} has no usable checkpoint for resume_from={resume_from!r}")
    chosen = rows[-1]
    return Path(chosen["path"]), int(chosen["episode"])


def checkpoint_roots(settings: Any) -> tuple[Path, ...]:
    """``CHECKPOINT.DIR`` when the project wrote it, and ``PATHS.CHECKPOINTS``; both resolved.

    The directories a checkpoint file may legitimately live in. This is the same rule
    ``Orchestrator._build_checkpoint_manager`` applies, restated as a function the destructive
    commands use to decide which files they may touch.
    """
    roots: list[Path] = []
    if is_authored(settings, "CHECKPOINT"):
        directory = (settings.get("CHECKPOINT") or {}).get("DIR")
        if directory:
            target = Path(str(directory)).expanduser()
            roots.append((target if target.is_absolute() else settings.path("ROOT") / target).resolve())
    roots.append(settings.path("CHECKPOINTS").resolve())
    return tuple(dict.fromkeys(roots))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def prune(
    registry: "Registry", run_id: str, *, keep: int, keep_best: bool, roots: Sequence[Path],
    live: bool, dry_run: bool = False,
) -> dict[str, Any]:
    """Delete all but the newest ``keep`` checkpoints (and, with ``keep_best``, every ``-best`` one).

    Unlike rollback this *deletes* files, so three guards apply: ``keep`` must be >= 1 (a run
    with no checkpoint cannot resume); a live run is refused; and a file is unlinked only when
    its resolved path is inside one of ``roots`` (a registry row is data, and a doctored row
    must not turn ``prune`` into an arbitrary file delete). A row whose file is already gone is
    removed without complaint; a row whose file cannot be unlinked is kept and reported.
    """
    if keep < 1:
        raise InvalidArgument("--keep must be at least 1", field="keep")
    if registry.get_run(run_id) is None:
        raise ResourceNotFound(f"no run {run_id!r}")
    if live:
        raise StateConflict("a live run's checkpoints must not be pruned", run_id=run_id)
    rows = sorted(registry.checkpoints(run_id), key=lambda row: (int(row["episode"]), int(row["id"])))
    protected = {int(row["id"]) for row in rows[-keep:]}
    if keep_best:
        protected |= {int(row["id"]) for row in rows if Path(row["path"]).name.endswith("-best.pt")}
    resolved_roots = [Path(root).resolve() for root in roots]
    removed: list[dict[str, Any]] = []
    forget: list[int] = []
    freed = 0
    for row in rows:
        if int(row["id"]) in protected:
            continue
        path = Path(row["path"])
        item: dict[str, Any] = {"episode": int(row["episode"]), "path": str(path),
                                "size_bytes": None, "file_deleted": False,
                                "row_removed": False, "problem": None}
        exists = path.exists()
        if exists:
            item["size_bytes"] = path.stat().st_size
            inside = any(_is_within(path.resolve(), root) for root in resolved_roots)
            if not inside:
                item["problem"] = "outside the checkpoint directories; file left alone"
            elif not dry_run:
                try:
                    path.unlink()
                    item["file_deleted"] = True
                    freed += int(item["size_bytes"] or 0)
                except OSError as exc:
                    item["problem"] = f"could not delete: {exc.strerror or exc}"
        if (not exists or item["file_deleted"]) and not dry_run:
            forget.append(int(row["id"]))
            item["row_removed"] = True
        removed.append(item)
    if forget:
        registry.delete_checkpoint_rows(forget)
    if dry_run:
        freed = sum(int(item["size_bytes"] or 0) for item in removed if item["problem"] is None)
    return {
        "run_id": run_id,
        "dry_run": dry_run,
        "kept": sorted(int(row["episode"]) for row in rows if int(row["id"]) in protected),
        "removed": removed,
        "freed_bytes": freed,
    }


def delete_run_artifacts(
    settings: Any, registry: "Registry", run_id: str, *, dry_run: bool = False
) -> dict[str, Any]:
    """Delete a run's checkpoint files (and their ``.rolledback`` archives) and its own log file.

    A checkpoint file is deleted only inside ``checkpoint_roots(settings)``. A log file is
    deleted only when it is the run's own (``<run_id>/events.jsonl`` or ``<run_id>.jsonl``);
    the shared ``LOGGER.JSONL_FILE`` holds other runs and is never touched. Returns
    ``{"run_id", "dry_run", "files", "freed_bytes", "problems"}`` where each file is
    ``{"kind", "path", "size_bytes", "deleted", "problem"}``; ``problems`` lists only the
    files that *should* have been deleted and could not be.
    """
    from ..training.logging import run_log_location

    roots = [Path(root).resolve() for root in checkpoint_roots(settings)]
    candidates: list[tuple[str, Path]] = []
    for row in registry.checkpoints(run_id):
        path = Path(row["path"])
        candidates.append(("checkpoint", path))
        candidates.append(("checkpoint", path.with_name(path.name + ".rolledback")))
    location = run_log_location(settings, run_id)
    if not location.shared:
        candidates.append(("log", location.path))

    files: list[dict[str, Any]] = []
    problems: list[str] = []
    freed = 0
    seen: set[str] = set()
    for kind, path in candidates:
        if str(path) in seen or not path.exists():
            continue
        seen.add(str(path))
        if kind == "checkpoint":
            allowed = any(_is_within(path.resolve(), root) for root in roots)
        else:
            allowed = ((path.name == "events.jsonl" and path.parent.name == run_id)
                       or path.name == f"{run_id}.jsonl")
        item: dict[str, Any] = {"kind": kind, "path": str(path), "size_bytes": path.stat().st_size,
                                "deleted": False, "problem": None}
        if not allowed:
            item["problem"] = "outside the directories a run writes to; file left alone"
        elif dry_run:
            freed += int(item["size_bytes"])
        else:
            try:
                path.unlink()
                item["deleted"] = True
                freed += int(item["size_bytes"])
                if kind == "log" and path.name == "events.jsonl":
                    with contextlib.suppress(OSError):
                        path.parent.rmdir()               # only succeeds when it is empty
            except OSError as exc:
                item["problem"] = f"could not delete: {exc.strerror or exc}"
                problems.append(f"{path}: {item['problem']}")
        files.append(item)
    return {"run_id": run_id, "dry_run": dry_run, "files": files, "freed_bytes": freed,
            "problems": problems}


def reap_stale(registry: "Registry", lock_dir: Path, *, dry_run: bool = False) -> list[dict[str, Any]]:
    """Rewrite ``queued``/``running``/``resuming`` rows no live process holds to ``interrupted``.

    ``RunManager.reap_orphans`` does this for the rows *its own process* is not driving; this
    is the process-independent version, decided by ``RunLock.holder``. Returns one
    ``{"run_id", "was", "reaped"}`` per row examined that has no holder.
    """
    out: list[dict[str, Any]] = []
    for row in registry.runs_with_status(tuple(ACTIVE_STATUSES)):
        if RunLock.holder(lock_dir, row["id"]) is not None:
            continue
        if not dry_run:
            registry.set_run_status(
                row["id"], "interrupted", notes="process ended while the run was active (reaped)"
            )
        out.append({"run_id": row["id"], "was": row["status"], "reaped": not dry_run})
    return out
