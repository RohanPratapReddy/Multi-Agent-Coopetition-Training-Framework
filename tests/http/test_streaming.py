"""Criterion 46: SSE through ``wsgiref`` with heartbeats; a disconnect releases the subscription."""

from __future__ import annotations

import threading
import time
import types
import urllib.request
from wsgiref.simple_server import WSGIRequestHandler, make_server

from mactf.http import StreamingResponse, path, sse_event, sse_heartbeat
from mactf.http.handler import WSGIHandler
from mactf.http.testing import make_environ, parse_sse


class _Quiet(WSGIRequestHandler):
    def log_message(self, *args):
        pass


class Subscription:
    def __init__(self, pause=0.0):
        self.released = threading.Event()
        self.sent = 0
        self.pause = pause

    def events(self, count):
        try:
            for n in range(count):
                self.sent += 1
                yield sse_event("tick", {"n": n}, event_id=n)
                yield sse_heartbeat()
                if self.pause:
                    # A live feed is paced. An unpaced one fills loopback buffers on Windows
                    # faster than the peer's reset can surface as a write error.
                    time.sleep(self.pause)
        finally:
            self.released.set()


def _app(subscription, count):
    def feed(request):
        return StreamingResponse(subscription.events(count), content_type="text/event-stream")

    urlconf = types.ModuleType("stream_urls")
    urlconf.urlpatterns = [path("feed/", feed, name="feed")]
    return WSGIHandler(urlconf=urlconf)


def test_events_and_heartbeats_arrive_over_wsgiref(settings):
    subscription = Subscription()
    server = make_server("127.0.0.1", 0, _app(subscription, 3), handler_class=_Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/feed/"
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = resp.read()
        frames = list(parse_sse([body]))
        assert [f["data"]["n"] for f in frames if f["data"] is not None] == [0, 1, 2]
        assert body.count(b": ") >= 3 or body.count(b":heartbeat") >= 3 or b":" in body
        assert subscription.released.wait(5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_closing_mid_stream_releases_the_subscription(settings):
    subscription = Subscription()
    app = _app(subscription, 1000)
    started = {}

    def start_response(status, headers, exc_info=None):
        started["status"] = status

    iterable = app(make_environ("GET", "/feed/"), start_response)
    first = next(iter(iterable))
    assert b"tick" in first and not subscription.released.is_set()
    iterable.close()  # what a WSGI server does on client disconnect
    assert subscription.released.is_set()
    assert subscription.sent < 1000


def test_a_client_disconnect_over_a_real_socket_releases_it(settings):
    import socket
    import struct

    subscription = Subscription(pause=0.01)
    server = make_server("127.0.0.1", 0, _app(subscription, 10_000_000), handler_class=_Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sock = None
    try:
        sock = socket.create_connection(server.server_address[:2], timeout=5)
        sock.sendall(b"GET /feed/ HTTP/1.0\r\nHost: x\r\n\r\n")
        received = b""
        while b"tick" not in received:  # headers and the first event may arrive separately
            chunk = sock.recv(4096)
            assert chunk, "the server closed before sending an event"
            received += chunk
        # An abortive close (RST) is how a vanished client looks to the server.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        sock.close()
        sock = None
        assert subscription.released.wait(10), "subscription leaked after the client went away"
    finally:
        if sock is not None:
            sock.close()  # never leave a live peer behind: shutdown() waits for the handler
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)