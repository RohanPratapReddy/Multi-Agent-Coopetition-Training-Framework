"""Optional WSGI-to-ASGI bridge. Not imported by ``mactf.http``.

An adapter, not a second concurrency model: the WSGI callable runs in a thread pool,
one worker per request. A deployment that needs native async streaming should use a
WSGI server directly, because SSE over ``wsgiref`` is simpler than SSE over a
thread-hopping bridge. ``uvicorn`` (the ``[asgi]`` extra) is never imported here.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

__all__ = ["asgi_application"]


def asgi_application(
    wsgi: Callable[..., Any] | None = None, *, max_workers: int | None = None
) -> Callable[..., Awaitable[None]]:
    """Wrap ``get_wsgi_application()`` (or ``wsgi``) as an ASGI 3 application."""
    if wsgi is None:
        from .handler import get_wsgi_application

        wsgi = get_wsgi_application()
    if max_workers is None:
        try:
            from ..conf import get_settings

            max_workers = int((get_settings().get("HTTP") or {}).get("STREAM_WORKERS", 16))
        except Exception:  # noqa: BLE001
            max_workers = 16
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mactf-asgi")

    async def application(scope: dict[str, Any], receive: Callable[[], Awaitable[dict[str, Any]]],
                          send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    executor.shutdown(wait=False)
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            raise RuntimeError(f"unsupported ASGI scope type {scope['type']!r}")

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body"):
                break

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        done = object()

        def run() -> None:
            import io

            environ = _environ(scope, bytes(body), io)

            def start_response(status: str, headers: list[tuple[str, str]], exc_info: Any = None):
                loop.call_soon_threadsafe(queue.put_nowait, ("start", status, headers))
                return lambda data: None

            iterable = wsgi(environ, start_response)
            try:
                for chunk in iterable:
                    loop.call_soon_threadsafe(queue.put_nowait, ("body", bytes(chunk)))
            finally:
                close = getattr(iterable, "close", None)
                if callable(close):
                    close()
                loop.call_soon_threadsafe(queue.put_nowait, done)

        future = loop.run_in_executor(executor, run)
        started = False
        while True:
            item = await queue.get()
            if item is done:
                break
            if item[0] == "start":
                status = int(item[1].split(" ", 1)[0])
                headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in item[2]]
                await send({"type": "http.response.start", "status": status, "headers": headers})
                started = True
            else:
                await send({"type": "http.response.body", "body": item[1], "more_body": True})
        await future
        if started:
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    return application


def _environ(scope: dict[str, Any], body: bytes, io: Any) -> dict[str, Any]:
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
    path = scope.get("path", "/")
    environ: dict[str, Any] = {
        "REQUEST_METHOD": scope["method"],
        "SCRIPT_NAME": scope.get("root_path", ""),
        "PATH_INFO": path.encode("utf-8").decode("latin-1"),
        "QUERY_STRING": scope.get("query_string", b"").decode("latin-1"),
        "SERVER_NAME": (scope.get("server") or ("localhost", 80))[0],
        "SERVER_PORT": str((scope.get("server") or ("localhost", 80))[1]),
        "SERVER_PROTOCOL": f"HTTP/{scope.get('http_version', '1.1')}",
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": scope.get("scheme", "http"),
        "wsgi.input": io.BytesIO(body),
        "wsgi.errors": io.StringIO(),
        "wsgi.multithread": True,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
        "CONTENT_LENGTH": str(len(body)),
    }
    if "content-type" in headers:
        environ["CONTENT_TYPE"] = headers.pop("content-type")
    headers.pop("content-length", None)
    for name, value in headers.items():
        environ["HTTP_" + name.upper().replace("-", "_")] = value
    return environ
