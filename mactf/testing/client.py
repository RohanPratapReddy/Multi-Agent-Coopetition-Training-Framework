"""``mactf.testing.Client``: call the WSGI application with no server (document 13 section 7).

The replacement for ``fastapi.testclient.TestClient``. It builds a PEP 3333 environ
and calls :func:`mactf.http.handler.get_wsgi_application`, through the same
``dispatch_wsgi`` the in-process SDK transport uses, so a test and an SDK call for
one request see identical bytes. No server, no port, no thread, no cookies: a test
that needs a token passes it per request.
"""

from __future__ import annotations

import contextlib
import json as _json
from collections.abc import Mapping
from typing import Any

from ..http.middleware import observe_exceptions
from ..http.testing import ClientResponse, active_settings, dispatch_wsgi

__all__ = ["Client", "TestResponse"]


class TestResponse(ClientResponse):
    """A :class:`~mactf.http.testing.ClientResponse` with the attributes a test reads.

    ``status_code``, ``headers``, ``content``, ``json()`` and ``response["Header"]``
    are inherited; the subclass exists so ``mactf.testing`` has one name for it and
    so pytest does not try to collect it (``__test__ = False``).
    """

    __test__ = False


class Client:
    """A request client bound to one URL conf and one settings object."""

    def __init__(
        self,
        *,
        urlconf: str | None = None,
        settings: Any | None = None,
        raise_request_exception: bool = False,
    ) -> None:
        self.urlconf = urlconf
        self.settings = settings
        self.raise_request_exception = raise_request_exception

    # -- verbs ----------------------------------------------------------------
    def get(self, path: str, data: Mapping[str, Any] | None = None, **extra: Any) -> TestResponse:
        return self.request("GET", path, data=data, **extra)

    def head(self, path: str, data: Mapping[str, Any] | None = None, **extra: Any) -> TestResponse:
        return self.request("HEAD", path, data=data, **extra)

    def options(self, path: str, **extra: Any) -> TestResponse:
        return self.request("OPTIONS", path, **extra)

    def post(
        self, path: str, data: Any = None, content_type: str = "application/json", **extra: Any
    ) -> TestResponse:
        return self.request("POST", path, data=data, content_type=content_type, **extra)

    def put(
        self, path: str, data: Any = None, content_type: str = "application/json", **extra: Any
    ) -> TestResponse:
        return self.request("PUT", path, data=data, content_type=content_type, **extra)

    def patch(
        self, path: str, data: Any = None, content_type: str = "application/json", **extra: Any
    ) -> TestResponse:
        return self.request("PATCH", path, data=data, content_type=content_type, **extra)

    def delete(
        self, path: str, data: Any = None, content_type: str = "application/json", **extra: Any
    ) -> TestResponse:
        return self.request("DELETE", path, data=data, content_type=content_type, **extra)

    # -- core -----------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        data: Any = None,
        content_type: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, Any] | str | None = None,
        **extra: Any,
    ) -> TestResponse:
        """Dispatch one request. Unknown keyword arguments become environ entries."""
        method = method.upper()
        kwargs: dict[str, Any] = {"headers": dict(headers or {}), "extra": extra or None}
        if method in ("GET", "HEAD", "OPTIONS", "DELETE") and data is not None and content_type is None:
            query = {**(query if isinstance(query, Mapping) else {}), **dict(data)} if not isinstance(query, str) else query
            data = None
        if data is not None:
            kwargs.update(self._body(data, content_type))
        elif content_type:
            kwargs["content_type"] = content_type
        if query:
            kwargs["query"] = query

        failures: list[BaseException] = []

        def observe(_request: Any, exc: BaseException, status: int) -> None:
            if status >= 500:
                failures.append(exc)

        with self._active():
            from ..http.handler import get_wsgi_application

            application = get_wsgi_application(urlconf=self.urlconf)
            with observe_exceptions(observe) if self.raise_request_exception else contextlib.nullcontext():
                response = dispatch_wsgi(
                    application, method, path, response_class=TestResponse, **kwargs
                )
                if failures:
                    response.close()
                    raise failures[0]
        return response  # type: ignore[return-value]

    @staticmethod
    def _body(data: Any, content_type: str | None) -> dict[str, Any]:
        if isinstance(data, (bytes, bytearray)):
            return {"body": bytes(data), "content_type": content_type or "application/octet-stream"}
        if isinstance(data, str):
            return {"body": data, "content_type": content_type or "text/plain"}
        if content_type and "json" not in content_type:
            if "x-www-form-urlencoded" in content_type:
                return {"data": data, "content_type": content_type}
            return {"body": _json.dumps(data), "content_type": content_type}
        return {"json": data, "content_type": content_type or "application/json"}

    def _active(self):
        """``settings`` current for one request, restored after: nothing leaks between tests."""
        return active_settings(self.settings)
