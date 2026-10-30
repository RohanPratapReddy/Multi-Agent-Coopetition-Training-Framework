"""test serve (split from test_catalog_and_tools.py)."""

from __future__ import annotations

import pytest

import threading


import urllib.request


def test_serve_command_serves_health(api_settings):
    from mactf.contrib.api.management.commands.serve import build_server

    server = build_server("127.0.0.1", 0, workers=2)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/health", timeout=10) as reply:
            assert reply.status == 200
            assert b'"status":"ok"' in reply.read().replace(b" ", b"")
    finally:
        server.shutdown()
        server.server_close()


def test_serve_is_discovered_from_the_app_not_from_a_core_package(api_settings):
    from mactf.management import base

    packages = base.command_packages()
    assert "mactf.contrib.api.management.commands" in packages
    assert "mactf.management.commands" in packages
    command = base.load_command("serve")
    assert command.__module__ == "mactf.contrib.api.management.commands.serve"
    import mactf.management.commands as core

    assert not hasattr(core, "serve")
    from pathlib import Path

    assert not (Path(core.__file__).parent / "serve.py").exists()


def test_serve_check_runs_before_binding_and_blocks_errors(api_settings, capsys):
    from mactf.management import base

    api_settings["API"] = {**api_settings["API"], "ALLOW_ANONYMOUS": False, "TOKENS": {}}
    code = base.execute_from_command_line(["manage.py", "serve", "--port", "0"])
    captured = capsys.readouterr()
    assert code != 0
    assert "mactf.api.E001" in captured.err + captured.out
    assert "system checks failed" in captured.err + captured.out


def test_a_served_socket_streams_events(api_settings):
    import urllib.request

    from mactf.contrib.api.management.commands.serve import build_server

    server = build_server("127.0.0.1", 0, workers=4)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runs",
            data=b'{"dry_run": true, "run_id": "sock"}',
            headers={"Authorization": "Bearer all-tok", "Content-Type": "application/json"},
            method="POST",
        )
        assert urllib.request.urlopen(request, timeout=10).status == 202
        stream = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runs/sock/events",
            headers={"Authorization": "Bearer all-tok"},
        )
        with urllib.request.urlopen(stream, timeout=20) as reply:
            assert "text/event-stream" in reply.headers["Content-Type"]
            text = reply.read().decode("utf-8")
        assert "event: terminal" in text
    finally:
        server.shutdown()
        server.server_close()
