"""test resume (split from test_completed_run.py)."""

from __future__ import annotations

from pathlib import Path

import pytest


from api_support import ApiClient, body


from api_support import wait_terminal
from mactf.http.handler import WSGIHandler


@pytest.fixture()
def finished(client):
    created = client.post("/runs", json={"epochs": 2, "run_id": "full"})
    assert created.status_code == 202, created.text
    final = wait_terminal(client, "full")
    assert final["status"] == "completed", final
    return "full"


def crash(app, run_id):
    """Simulate a process restart: forget the live handle and leave an interrupted row."""
    manager = app.state["run_manager"]
    with manager._lock:
        handle = manager._handles.get(run_id)
    if handle is not None:
        # The registry row turns terminal before the worker thread finishes; crashing earlier
        # would let that thread overwrite the "interrupted" row with "completed".
        assert handle.done.wait(15), "the run thread never finished"
    with manager._lock:
        manager._handles.pop(run_id, None)
    app.state["registry"].set_run_status(run_id, "interrupted", notes="simulated crash")


def test_resume_of_terminal_run_is_409(client, finished):
    response = client.post(f"/runs/{finished}/resume")
    assert response.status_code == 409
    assert "terminal" in str(body(response)["extensions"])


def test_interrupted_run_rolls_back_and_resumes(client, app, finished):
    episodes = sorted(row["episode"] for row in body(client.get(f"/runs/{finished}/checkpoints"))["items"])
    crash(app, finished)
    assert body(client.get(f"/runs/{finished}"))["status"] == "interrupted"
    rolled = client.post(f"/runs/{finished}/rollback", json={"episode": episodes[0]})
    assert rolled.status_code == 202, rolled.text
    response = client.post(f"/runs/{finished}/resume", json={})
    assert response.status_code == 202, response.text
    final = wait_terminal(client, finished)
    assert final["status"] == "completed", final
    assert body(client.get(f"/runs/{finished}/episodes"))["total"] >= 2


def _warning(rule_id, episode, *, active=True, expiry=99):
    from mactf.rules.severity import Severity
    from mactf.scoring.warning import WarningRecord

    return WarningRecord(
        rule_id=rule_id,
        episode=episode,
        severity=Severity.parse("S2"),
        evidence="seeded by the resume test",
        expiry_episode=expiry,
        active=active,
    )


@pytest.fixture()
def bare(api_settings):
    """An app whose runs write no checkpoints, so a resume has only the database."""
    api_settings["TRAINING"]["CHECKPOINT_EVERY"] = 0
    handler = WSGIHandler()
    yield handler, ApiClient(handler)
    manager = handler.state.get("run_manager")
    if manager is not None:
        manager.close()


def test_warnings_survive_a_resume_that_has_no_checkpoint(bare):
    app, client = bare
    assert client.post("/runs", json={"epochs": 1, "run_id": "nockpt"}).status_code == 202
    assert wait_terminal(client, "nockpt")["status"] == "completed"
    registry = app.state["registry"]
    # CHECKPOINT_EVERY=0 still checkpoints every epoch, so remove them to leave only the database.
    for row in registry.checkpoints("nockpt"):
        Path(row["path"]).unlink(missing_ok=True)
    registry.delete_checkpoints_after("nockpt", -1)
    assert body(client.get("/runs/nockpt/checkpoints"))["total"] == 0
    registry.record_warning("nockpt", "p0", _warning("P6", 0))
    registry.record_warning("nockpt", "p1", _warning("P11", 0))
    crash(app, "nockpt")

    response = client.post("/runs/nockpt/resume", json={})
    assert response.status_code == 202, response.text
    report = body(response)["resume"]
    assert report is not None
    assert report["checkpoint_episode"] is None
    assert report["warnings_restored"] == 2
    assert report["report"]["warnings_restored"] == 2
    wait_terminal(client, "nockpt")
    assert registry.resume_reports("nockpt")[-1]["warnings_restored"] == 2


def test_a_resume_with_nothing_to_restore_reports_zero(bare):
    app, client = bare
    client.post("/runs", json={"epochs": 1, "run_id": "clean"})
    wait_terminal(client, "clean")
    crash(app, "clean")
    response = client.post("/runs/clean/resume", json={})
    assert response.status_code == 202
    assert body(response)["resume"]["warnings_restored"] == 0
    wait_terminal(client, "clean")


def test_resume_of_an_unknown_run_is_404(client):
    assert client.post("/runs/ghost/resume", json={}).status_code == 404


def test_resume_needs_the_write_scope(client, app, finished):
    crash(app, finished)
    assert client.post(f"/runs/{finished}/resume", json={}, token="read-tok").status_code == 403
    assert client.post(f"/runs/{finished}/resume", json={}).status_code == 202
    wait_terminal(client, finished)
