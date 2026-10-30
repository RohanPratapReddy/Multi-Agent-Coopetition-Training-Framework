"""``serve`` - run the API on a bounded, threaded WSGI server.

Document 02 section 5 for the server; document 16 section 8.4 for the command: shutdown is
a ``StopFlag``, a bind failure is classified, and ``--port-file`` tells a script the port.
"""

from __future__ import annotations

import argparse
import errno
import os
import threading
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from .....checks import run_checks
from .....conf import get_settings
from .....exceptions import CommandError
from .....http.handler import get_wsgi_application
from .....management.base import BaseCommand
from .....management.exitcodes import ExitCode
from .....management.interrupt import StopFlag
from .....management.runsupport import positive_int


class BoundedThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """One thread per connection, with at most ``limit`` in flight.

    SSE streams hold a thread for their whole life, so an unbounded pool would let a
    handful of subscribers exhaust the process; excess connections wait in accept().
    """

    daemon_threads = True
    request_queue_size = 64
    limit = 16

    def __init__(self, *args, limit: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(limit or self.limit)

    def process_request(self, request, client_address):  # type: ignore[override]
        self._slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):  # type: ignore[override]
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - the handler logs through ``mactf.http``
        return


def build_server(host: str, port: int, *, workers: int | None = None, quiet: bool = True) -> BoundedThreadingWSGIServer:
    application = get_wsgi_application()
    pool = workers or int((get_settings().get("HTTP") or {}).get("STREAM_WORKERS", 16))

    class _Server(BoundedThreadingWSGIServer):
        limit = pool

    server = make_server(
        host, port, application, server_class=_Server, handler_class=_QuietHandler if quiet else WSGIRequestHandler
    )
    server.application = application  # type: ignore[attr-defined]
    return server  # type: ignore[return-value]


def port_arg(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a port number") from None
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"{text!r} is outside 0-65535")
    return value


def bind_error(host: str, port: int, exc: OSError) -> CommandError:
    """Address in use is a conflict (4); anything else (permission, bad host) is unavailable (5)."""
    message = f"cannot bind {host}:{port}: {exc.strerror or exc}"
    in_use = {errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", errno.EADDRINUSE), 10048}
    if exc.errno in in_use:
        return CommandError.conflict(message + "; choose another --port or stop the other server")
    return CommandError.unavailable(message)


def format_url(host: str, port: int) -> str:
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}/"


class Command(BaseCommand):
    help = "Serve the MACTF API over HTTP (stdlib WSGI; use a production server for exposure)."
    category = "Server"
    output_formats = ("text", "jsonl")
    examples = (
        "mactf serve",
        "mactf serve --host 0.0.0.0 --port 9000",
        "mactf serve --port 0 --port-file .mactf-port --format jsonl",
    )
    arguments = [
        (("--host",), {"default": None, "help": "Address to bind (default HTTP.HOST)."}),
        (("--port",), {"type": port_arg, "default": None,
                       "help": "Port to bind (default HTTP.PORT); 0 asks the OS for a free one."}),
        (("--workers",), {"type": positive_int, "default": 1,
                          "help": "Process count; only 1 is supported."}),
        (("--skip-checks",), {"action": "store_true", "default": False,
                              "help": "Skip the system checks."}),
        (("--verbose-access",), {"action": "store_true", "default": False,
                                 "help": "Print one line per request."}),
        (("--port-file",), {"default": None, "metavar": "PATH",
                            "help": "Write the bound port here once listening; removed on exit."}),
    ]

    def handle(self, **options):
        http = self.need_settings().get("HTTP") or {}
        host = options["host"] or http.get("HOST", "127.0.0.1")
        port = options["port"] if options["port"] is not None else int(http.get("PORT", 8000))
        if options["workers"] > 1:
            self.out.warn("run state (live runs, SSE bus) is process-local, so --workers is ignored; "
                          "serving with a single process")
        if not options["skip_checks"]:
            outcome = run_checks()
            for message in outcome.messages:
                if message.level == "ERROR":
                    self.out.error(str(message))
            if not outcome.ok:
                raise CommandError(f"system checks failed ({outcome.summary()}); fix them or pass --skip-checks")
        try:
            server = build_server(host, port, quiet=not options["verbose_access"])
        except OSError as exc:
            raise bind_error(host, port, exc) from exc

        bound_host, bound_port = server.server_address[:2]
        url = format_url(bound_host, bound_port)
        port_file = Path(options["port_file"]).expanduser() if options["port_file"] else None
        crashed: list[BaseException] = []

        def serve() -> None:
            try:
                server.serve_forever(poll_interval=0.25)
            except BaseException as exc:  # noqa: BLE001 - reported on the main thread
                crashed.append(exc)

        worker = threading.Thread(target=serve, name="mactf-serve", daemon=True)
        interrupted = False
        try:
            if port_file is not None:
                port_file.parent.mkdir(parents=True, exist_ok=True)
                port_file.write_text(f"{bound_port}\n", encoding="utf-8")
            self.out.event({"event": "listening", "host": bound_host, "port": bound_port,
                            "url": url, "pid": os.getpid()},
                           text=f"MACTF API on {url} (Ctrl-C to stop)", level=0)
            with StopFlag(announce=self.out.warn,
                          message="shutting down; press Ctrl-C again to force") as stop:
                worker.start()
                try:
                    # Half-second slices: a blocking Event.wait() is not interruptible on Windows.
                    while worker.is_alive() and not stop.wait(0.5):
                        pass
                except KeyboardInterrupt:
                    interrupted = True
        finally:
            if worker.is_alive():
                server.shutdown()                     # returns once serve_forever has exited
            worker.join(timeout=5)
            server.server_close()
            manager = server.application.state.get("run_manager")  # type: ignore[attr-defined]
            if manager is not None:
                manager.close()
            if port_file is not None:
                port_file.unlink(missing_ok=True)
            self.out.event({"event": "stopped"}, text="stopped", level=0)
        if crashed:
            raise crashed[0]
        return int(ExitCode.INTERRUPTED) if interrupted else int(ExitCode.OK)
