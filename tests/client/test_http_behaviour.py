"""HTTPTransport against a scripted server: retries, idempotency, prefix probing, SSE reconnect."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mactf.client import HTTPTransport, MACTFClient, RateLimited, ServerError, TransportError


class Script:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []
        self.replies: dict[tuple[str, str], list] = {}

    def add(self, method, path, *replies):
        self.replies.setdefault((method, path), []).extend(replies)


@pytest.fixture()
def scripted():
    script = Script()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence
            pass

        def _serve(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            path = self.path.split("?")[0]
            script.calls.append((self.command, path, {k.lower(): v for k, v in self.headers.items()}))
            queue = script.replies.get((self.command, path))
            if not queue:
                status, headers, body = 404, {}, b'{"detail":"nothing scripted"}'
            else:
                status, headers, body = queue.pop(0) if len(queue) > 1 else queue[0]
            if callable(body):
                body = body(self)
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            headers = {"Content-Type": "application/json", **headers}
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_DELETE = do_PUT = _serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield script
    server.shutdown()
    server.server_close()


def _transport(script, sleeps, **kwargs):
    return HTTPTransport(script.url, prefix="/api/v1", sleep=sleeps.append, backoff=0.5, **kwargs)


def test_get_is_retried_on_503_then_succeeds(scripted):
    scripted.add("GET", "/api/v1/health", (503, {}, {"detail": "busy"}), (503, {}, {"detail": "busy"}), (200, {}, {"ok": True}))
    sleeps: list[float] = []
    response = _transport(scripted, sleeps).request("GET", "/api/v1/health")
    assert response.status_code == 200
    assert len([c for c in scripted.calls if c[1] == "/api/v1/health"]) == 3
    assert len(sleeps) == 2 and all(0 <= s <= 30 for s in sleeps)


def test_retry_after_is_honoured_and_limit_is_finite(scripted):
    scripted.add("GET", "/api/v1/meta", (429, {"Retry-After": "7"}, {"detail": "slow down", "type": "urn:mactf:error:rate-limited"}))
    sleeps: list[float] = []
    transport = _transport(scripted, sleeps, max_retries=2)
    response = transport.request("GET", "/api/v1/meta")
    assert response.status_code == 429
    assert sleeps == [7.0, 7.0]
    client = MACTFClient(transport=transport)
    with pytest.raises(RateLimited) as caught:
        client.meta()
    assert caught.value.retry_after == 7.0


def test_post_without_idempotency_key_is_never_retried(scripted):
    scripted.add("POST", "/api/v1/evaluations", (503, {}, {"detail": "busy"}))
    sleeps: list[float] = []
    response = _transport(scripted, sleeps).request("POST", "/api/v1/evaluations", json={"a": 1})
    assert response.status_code == 503
    assert sleeps == []
    assert len(scripted.calls) == 1


def test_post_with_idempotency_key_is_retried_with_the_same_key(scripted):
    scripted.add("POST", "/api/v1/runs", (502, {}, {"detail": "bad gateway"}), (202, {}, {"run_id": "r", "status": "queued", "created_at": 1, "links": {}}))
    sleeps: list[float] = []
    client = MACTFClient(transport=_transport(scripted, sleeps))
    handle = client.runs.create(epochs=1, idempotency_key="abc")
    assert handle.run_id == "r"
    keys = [c[2].get("idempotency-key") for c in scripted.calls]
    assert keys == ["abc", "abc"]


def test_create_generates_an_idempotency_key(scripted):
    scripted.add("POST", "/api/v1/runs", (202, {}, {"run_id": "r", "status": "queued", "created_at": 1, "links": {}}))
    MACTFClient(transport=_transport(scripted, [])).runs.create()
    assert len(scripted.calls[0][2]["idempotency-key"]) >= 16


def test_server_error_maps_and_headers_are_sent(scripted):
    scripted.add("GET", "/api/v1/meta", (500, {}, {"type": "urn:mactf:error:internal", "detail": "boom", "trace_id": "t1"}))
    transport = _transport(scripted, [], token="tok")
    with pytest.raises(ServerError) as caught:
        MACTFClient(transport=transport).meta()
    assert caught.value.trace_id == "t1"
    headers = scripted.calls[0][2]
    assert headers["authorization"] == "Bearer tok"
    assert headers["accept"] == "application/json"
    assert headers["x-mactf-client"]


def test_prefix_probe_prefers_versioned_then_bare(scripted):
    scripted.add("GET", "/health", (200, {}, {"ok": True}))
    transport = HTTPTransport(scripted.url, sleep=lambda s: None)
    assert transport.prefix == ""
    scripted.add("GET", "/meta", (200, {}, {"api_version": "v1"}))
    assert MACTFClient(transport=transport).meta().api_version == "v1"  # /api/v1/meta rebased to /meta


def test_probe_failure_is_a_transport_error(scripted):
    with pytest.raises(TransportError):
        HTTPTransport(scripted.url, sleep=lambda s: None)


def test_connection_refused_is_a_transport_error():
    transport = HTTPTransport("http://127.0.0.1:9", probe=False, max_retries=1, sleep=lambda s: None, timeout=1)
    with pytest.raises(TransportError):
        transport.request("GET", "/api/v1/health")


def _sse(*frames):
    return "".join(frames).encode()


def test_sse_reconnects_with_last_event_id(scripted):
    first = _sse('id: 1\nevent: epoch\ndata: {"n": 1}\n\n', 'id: 2\nevent: epoch\ndata: {"n": 2}\n\n')
    second = _sse('id: 3\nevent: terminal\ndata: {"status": "completed"}\n\n')
    scripted.add(
        "GET", "/api/v1/runs/r/events",
        (200, {"Content-Type": "text/event-stream"}, first),
        (200, {"Content-Type": "text/event-stream"}, second),
    )
    sleeps: list[float] = []
    events = list(MACTFClient(transport=_transport(scripted, sleeps)).runs.stream("r"))
    assert [e.id for e in events] == [1, 2, 3]
    assert events[-1].event == "terminal"
    calls = [c for c in scripted.calls if c[1].endswith("/events")]
    assert "last-event-id" not in calls[0][2]
    assert calls[1][2]["last-event-id"] == "2"


def test_stream_lost_raises_with_oldest(scripted):
    body = _sse('event: stream_lost\ndata: {"oldest": 40}\n\n')
    scripted.add("GET", "/api/v1/runs/r/events", (200, {"Content-Type": "text/event-stream"}, body))
    with pytest.raises(TransportError) as caught:
        list(MACTFClient(transport=_transport(scripted, [])).runs.stream("r", last_event_id=3))
    assert caught.value.extensions["oldest"] == 40


def test_stream_gives_up_after_max_retries(scripted):
    body = _sse('id: 1\nevent: epoch\ndata: {"n": 1}\n\n')
    scripted.add("GET", "/api/v1/runs/r/events", (200, {"Content-Type": "text/event-stream"}, body))
    transport = _transport(scripted, [], max_retries=2)
    events = []
    with pytest.raises(TransportError):
        for event in transport.stream("GET", "/api/v1/runs/r/events"):
            events.append(event)
    assert len(events) >= 1
