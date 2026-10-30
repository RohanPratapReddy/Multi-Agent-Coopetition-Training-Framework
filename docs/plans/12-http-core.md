---
title: HTTP Core
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0
owner: TBD
last_reviewed: 2026-10-21
---

# 12. HTTP Core

`mactf/http/` is the framework's transport layer. It turns a WSGI `environ` into
a `Request`, resolves a URL through the project's `ROOT_URLCONF`, calls a view,
and turns the returned `Response` back into a WSGI response. It does not know what
a peer, a scorecard, or a run is.

This layer exists for the same reason `django.core.handlers` exists: without it,
every capability the framework serves would have to be re-expressed in whichever
HTTP library happened to be installed. With it, the service layer (document 02)
is a set of views and serializers, and the client (document 03) is a set of
requests against a URL conf — both usable from a plain `pip install mactf`.

## 1. Why it is in the framework

Three facts decide the shape of this document.

1. `mactf/apps.py` already models `INSTALLED_APPS`, `ready()`, and peer/provider
   registries ([mactf/apps.py](mactf/apps.py)). The service layer should be an
   app in that registry, not a package that imports the framework and starts a
   server of its own.
2. `mactf/checks.py` already models system checks
   ([mactf/checks.py:67](mactf/checks.py#L67)). An app that cannot register a
   check is a second-class app.
3. The framework's own dataclasses already have `to_dict()`. A wire model written
   against a second serialization library duplicates a projection that exists.

So `mactf/http/` is **stdlib only**: `json`, `urllib.parse`, `re`, `wsgiref`,
`http`, `email.utils`, `dataclasses`. No FastAPI, no Starlette, no pydantic, no
httpx, no uvicorn. `mactf serve` runs on `wsgiref.simple_server`, which is exactly
what `django-admin runserver` does by default.

### 1.1 What is explicitly not here

| Not here | Where it lives |
|---|---|
| Route definitions for `/api/v1/...` | `mactf/contrib/api/urls.py`, document 02 section 3 |
| Bearer-token scope table | `mactf/contrib/api/auth.py`, document 02 section 4.4 |
| `RunManager`, `EventBus` | `mactf/orchestration/live.py`, L2, document 01 section 3 |
| Scoring, aggregation, checkpoints | L2 modules, called by views |
| The client that speaks this protocol | `mactf/client/`, document 03 |

The last row is the point of the whole split: `mactf/client` imports
`mactf/http` (for `Request`, `Response`, and dispatch) and never the reverse.

## 2. Module layout

```
mactf/http/
  __init__.py        Request, Response, JsonResponse, StreamingResponse,
                     HttpError family, path, include, reverse, resolve,
                     get_wsgi_application  (the public surface)
  request.py         Request, QueryDict, parse_qs wrapper, header access
  response.py        Response, JsonResponse, StreamingResponse, ProblemResponse,
                     FileResponse, STATUS_TEXT, status constants
  urls.py            URLPattern, URLResolver, path, re_path, include,
                     resolve, reverse, clear_url_caches, get_resolver
  converters.py      path conversion: str, int, slug, uuid, path
  views.py           View, as_view, method dispatch, require_* decorators
  middleware.py      Middleware protocol, RequestIdMiddleware,
                     ExceptionMiddleware, CorsMiddleware, AuthMiddleware
  handler.py         WSGIHandler, get_wsgi_application, request/response bridge
  serializers.py     Field family, Serializer, ModelSerializer, ValidationError
  exceptions.py      HttpError and subclasses, problem+json rendering
  openapi.py         schema generation from a URL conf plus serializer metadata
  asgi.py            optional WSGI-to-ASGI adapter (documented, not required)
  testing.py         dispatch_wsgi and request/response helpers, shared by
                     InProcessTransport and mactf.testing.Client
```

Every module is importable on its own. `mactf/http/__init__.py` imports all of
them eagerly, and a test asserts that importing the package imports no
third-party distribution (document 06, `stdlib-only` job).

## 3. `Request`

`Request` wraps a WSGI `environ` and lazily decodes what a view asks for. The
body is never parsed until a view calls `json()` or `POST`.

```python
class Request:
    method: str                       # upper-cased
    path: str                         # PATH_INFO, percent-decoded, no query
    path_info: str                    # raw PATH_INFO
    script_name: str                  # SCRIPT_NAME, "" when mounted at root
    GET: QueryDict                    # parse_qs(QUERY_STRING, keep_blank_values=True)
    POST: QueryDict                   # form-encoded body, empty for JSON
    headers: Headers                  # case-insensitive read-only mapping
    body: bytes                       # read once, cached
    content_type: str                 # "" when absent
    META: Mapping[str, Any]           # the environ, read-only view
    resolver_match: ResolverMatch | None
    principal: Principal | None       # set by AuthMiddleware, None if anonymous
    request_id: str                   # set by RequestIdMiddleware
    started_at: float                 # time.monotonic() at handler entry
```

Rules:

- `method` is upper-cased at construction. `Request.method` is never `None`;
  a missing `REQUEST_METHOD` is a `400` produced by the handler, not an
  `AttributeError` produced by a view.
- `headers` is case-insensitive and multi-value aware, backed by
  `email.message.Message`. `headers["Content-Type"]` and
  `headers.get("content-type")` are the same lookup. `headers.getlist(name)`
  preserves repeats, which is what `Accept` negotiation needs.
- `body` reads `wsgi.input` once and caches the bytes. It refuses a body larger
  than `HTTP.MAX_BODY_BYTES` (default 8 MiB) with `RequestDataTooBig` (413)
  **before** reading, using `CONTENT_LENGTH` when present. A request with no
  `CONTENT_LENGTH` and a `Transfer-Encoding: chunked` header is read through the
  stream and capped while reading, so a lying client cannot exhaust memory.
- `json()` decodes UTF-8 and returns the parsed value. It raises
  `ParseError` (400) on malformed JSON and `UnsupportedMediaType` (415) when
  `Content-Type` is neither `application/json` nor `application/*+json`. It is
  called at most once; the parsed value is cached.
- `GET` is a `QueryDict`: repeated keys return the **last** value through
  `__getitem__`, and `getlist(key)` returns all of them. That is Django's rule
  and it is what makes `?limit=10&limit=20` deterministic instead of
  order-dependent.
- `Request` has no `user` attribute and no authentication logic. `principal` is
  whatever `AuthMiddleware` assigned, or `None`. That is the entire coupling
  between transport and auth.

`QueryDict` is immutable by default (`mutable=False`). A view that needs to
rewrite query parameters constructs a new one; nothing in the framework mutates a
request in place, which is what keeps a request reusable by the in-process
transport.

## 4. `Response`

```python
class Response:
    status_code: int = 200
    headers: dict[str, str]
    content: bytes = b""
    streaming: bool = False

    def __init__(self, content=b"", status=200, *, headers=None,
                 content_type="text/plain; charset=utf-8") -> None: ...
    def __setitem__(self, name: str, value: str) -> None: ...
    def __getitem__(self, name: str) -> str: ...
    def set_cookie(self, key, value="", *, max_age=None, path="/",
                   domain=None, secure=False, httponly=False,
                   samesite="Lax") -> None: ...
    def delete_cookie(self, key, *, path="/", domain=None) -> None: ...
    def close(self) -> None: ...        # closes a streaming iterator
```

| Class | Body |
|---|---|
| `JsonResponse(data, status=200, *, headers=None)` | UTF-8 JSON, `application/json`; `json.dumps(..., ensure_ascii=False, separators=(",", ":"))` |
| `ProblemResponse(...)` | `application/problem+json`, built from an `HttpError`; see section 10 |
| `StreamingResponse(iterator, status=200, *, headers=None, content_type="text/event-stream")` | yields `bytes`; the handler forwards each chunk |
| `FileResponse(path, *, content_type=None, filename=None)` | `wsgi.file_wrapper` when the server provides it, else a 64 KiB read loop |

`status` is a module of integer constants (`status.OK = 200`,
`status.UNPROCESSABLE_ENTITY = 422`, ...) plus `STATUS_TEXT: dict[int, str]`.
The set is exactly the codes document 02 uses; a code the framework never returns
is not defined, so a view cannot silently emit `418`.

Three rules:

- `content` is always `bytes`. A view that returns `str` is a programming error
  caught by `Response.__init__` with `TypeError`, not encoded implicitly. Implicit
  encoding is how a `str` with a lone surrogate becomes a 500 inside the server.
- A `StreamingResponse` sets `Transfer-Encoding: chunked` only when the server
  declares `wsgi.file_wrapper` absent **and** no `Content-Length` was set. Under
  `wsgiref.simple_server` the server handles framing; the handler passes
  `start_response` the status and headers once and then yields.
- `close()` is called by the handler in a `finally`. A generator that raises after
  the first chunk terminates the stream; the handler logs it and does **not**
  attempt to rewrite the status, because the status line is already sent.

## 5. URL configuration

`ROOT_URLCONF` is a dotted module path in settings. A project's `urls.py` is a
module with a `urlpatterns` list, exactly like Django's.

```python
# project/urls.py
from mactf.http import include, path

urlpatterns = [
    path("api/v1/", include("mactf.contrib.api.urls")),
]
```

### 5.1 Patterns and resolvers

```python
def path(route: str, view, *, name: str | None = None,
         kwargs: Mapping[str, Any] | None = None) -> URLPattern: ...
def re_path(regex: str, view, *, name: str | None = None,
            kwargs: Mapping[str, Any] | None = None) -> URLPattern: ...
def include(arg, *, namespace: str | None = None,
            app_name: str | None = None) -> tuple[list[URLPattern], str, str]: ...

class URLPattern:
    route: str
    name: str | None
    callback: Callable | View
    default_kwargs: Mapping[str, Any]
    pattern: re.Pattern[str]

class URLResolver:
    pattern: re.Pattern[str]
    url_patterns: list[URLPattern | URLResolver]
    namespace: str | None
    app_name: str | None

    def resolve(self, path: str) -> ResolverMatch: ...
    def reverse(self, name: str, *args, **kwargs) -> str: ...
```

`include()` accepts a module, a module path string, or a `(patterns, app_name)`
tuple. It returns a three-tuple so `urlpatterns` can hold it directly, matching
Django's shape and therefore Django's mental model:

```python
urlpatterns = [
    path("api/v1/", include("mactf.contrib.api.urls")),
]
```

The third argument is the namespace. Omitting it derives the namespace from the
included module's `app_name` (here `"api"`), which is what makes
`reverse("api:rule-list")` work. Passing `namespace="v1"` overrides it and is
legal, but every `reverse()` call in documents 02 and 03 uses `api:`, so a project
that overrides the namespace must also be the one calling `reverse()`; the client
does not support it.

Route converters are declared inline in the route string and compiled by
`converters.py`:

| Converter | Regex | `to_python` |
|---|---|---|
| `str` (default) | `[^/]+` | `str` |
| `int` | `[0-9]+` | `int` |
| `slug` | `[-a-zA-Z0-9_]+` | `str` |
| `uuid` | canonical 8-4-4-4-12 | `uuid.UUID` |
| `path` | `.*` | `str` (may contain `/`) |

`{run_id}` therefore yields a `str`, `{episode:int}` yields an `int`, and a
trailing `{subpath:path}` captures a remainder. A route that declares the same
converter name twice is an `ImproperlyConfigured` raised at import of the URL
conf, not at request time.

### 5.2 `resolve()` and `reverse()`

```python
def resolve(path: str) -> ResolverMatch: ...          # Resolver404 when no match
def reverse(name: str, *args, **kwargs) -> str: ...   # NoReverseMatch otherwise

@dataclass(frozen=True)
class ResolverMatch:
    func: Callable
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    url_name: str
    route: str                    # the pattern text, for logs and OpenAPI
    namespaces: tuple[str, ...]
    app_name: str | None
```

`reverse` accepts two spellings, which differ only in lookups. A **bare** name
(`"rule-list"`) is resolved without namespaces and raises `NoReverseMatch` if the
pattern lives inside a namespace. A **fully qualified** name
(`"api:rule-list"`, or `"v1:api:rule-list"` for nested namespaces) resolves the
namespace chain from left to right. The framework's own docs and the client use
the qualified form, because a project may mount the same app twice under two
prefixes and only the qualified name is unambiguous.

`ROOT_URLCONF = None` selects `mactf.http.urls.default_urlpatterns`, which is a
one-entry conf: `path("", include("mactf.contrib.api.urls"))` when the app is
installed, and an empty `urlpatterns` when it is not. It is what lets a checkout
run `mactf serve` with no project, and it is why `include()` deriving the
namespace from `app_name` matters — the default conf has no explicit namespace
either.

- `resolve` strips `SCRIPT_NAME`, tries the patterns in declaration order, and
  returns the first match. It does not fall back, rank, or reorder; a URL conf is
  ordered, and a conflicting route is a bug the developer sees immediately.
- `reverse` walks the resolver tree looking for `name`, fills the converters from
  positional args or keyword args, and percent-encodes each value with
  `urllib.parse.quote(value, safe="")` — except the `path` converter, which uses
  `safe="/"`. A missing or extra argument raises `NoReverseMatch` naming the
  pattern and the expected converter list.
- Names are unique per namespace. `resolve` returns the namespace tuple so a view
  can call `reverse(f"{':'.join(match.namespaces)}:run-detail", ...)`. Duplicate
  names inside one namespace raise `ImproperlyConfigured` when the conf is first
  loaded; duplicates across namespaces are legal and are what namespaces are for.
- `get_resolver(urlconf=None)` loads and caches the URL conf **per settings
  object identity**. `conf.configure(force=True)` and `conf.reset()` clear the
  cache through `clear_url_caches()`, which is the only supported way to pick up a
  changed `ROOT_URLCONF` in a long-lived process.

### 5.3 Trailing slashes

`APPEND_SLASH` (default `True`) makes the resolver retry once with a trailing
slash appended when the initial match fails and the path does not end in `/`.
On a hit it returns `301` with `Location` set to the slashed path, preserving the
query string. It never rewrites a `POST`: a `POST` to a slashed-only route
returns `301` and the client must re-issue, because silently forwarding a body
across a redirect is how a body is delivered twice.

`APPEND_SLASH=False` disables the retry and the redirect entirely. A project that
declares all routes without trailing slashes sets it.

## 6. Views

Two kinds of view are accepted anywhere a view is accepted: a function
`def view(request, **kwargs) -> Response` and a `View` subclass.

```python
class View:
    http_method_names = ["get", "post", "put", "patch", "delete", "head", "options"]

    @classmethod
    def as_view(cls, **initkwargs) -> Callable[..., Response]: ...

    def dispatch(self, request: Request, *args, **kwargs) -> Response:
        handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
        return handler(request, *args, **kwargs)

    def http_method_not_allowed(self, request, *args, **kwargs) -> Response:
        return Response(status=405, headers={"Allow": self.allowed_methods})
```

- `as_view()` rejects an `initkwargs` key that shadows an existing class
  attribute with `TypeError`, and marks the resulting callable with
  `view_class`, `view_initkwargs`, and `__name__`. `mactf.http.openapi` and the
  route-completeness test in document 02 section 5 both read `view_class`.
- `HEAD` is served by `GET` when a view does not define `head`, with the body
  dropped by the handler after the response is built. The other direction is not
  implemented: a view that defines only `head` does not answer `GET`.
- `OPTIONS` returns `200` with `Allow` and `Content-Length: 0`. It is never
  cached, because a route's methods can change between deployments.
- `405` always carries `Allow`. A `405` without `Allow` is the one header omission
  a client cannot recover from, so the handler asserts it and replaces the
  response with a problem body if a view returns `405` without it.
- `csrf_exempt`, `require_http_methods`, `require_GET`, and `require_POST` are
  decorators over a function view. They do not exist for `View` subclasses,
  because `http_method_names` already expresses the same restriction.

A view returns a `Response` or, for convenience, a `dict`/`list`, which
`View.dispatch` converts through `JsonResponse`. Returning `None` is an error
(`ImproperlyConfigured`) rather than an empty `200`, because a view that forgets
`return` must not look like success.

## 7. Middleware

A middleware is any callable following this protocol:

```python
class Middleware(Protocol):
    def __init__(self, get_response: Callable[[Request], Response]) -> None: ...
    def __call__(self, request: Request) -> Response: ...
```

`MIDDLEWARE` is an ordered list of dotted paths. The handler builds it once:

```python
handler = None
for path in reversed(settings["MIDDLEWARE"]):
    handler = import_string(path)(handler or self._get_response)
```

Request phase runs top-down, response phase bottom-up — the same order as
Django, and the order document 02's error table assumes: `ExceptionMiddleware`
must be above anything that can raise, so it can convert an exception into a
problem body.

| Middleware | Behaviour |
|---|---|
| `RequestIdMiddleware` | assigns `request.request_id` from `X-Request-Id` when it is a valid token, else `uuid4().hex`; sets `X-Request-Id` and `X-Trace-Id` on the response (the same value, one name per audience) |
| `ExceptionMiddleware` | catches `HttpError` and `MACTFError`, converts through section 10, logs with the trace id, and returns problem+json; with `DEBUG=True` the traceback is appended as a `debug` extension |
| `CorsMiddleware` | answers `OPTIONS` preflight from `API.CORS_ORIGINS`, echoes `Access-Control-Request-Headers`, and sets `Vary: Origin`; a wildcard is never combined with credentials |
| `AuthMiddleware` | resolves `request.principal` through `API.PRINCIPAL_RESOLVER` (a dotted path) and rejects with `401` or `403` when the resolver returns a denial; with no resolver configured every request is anonymous, which is what `DEBUG` projects want |

`AuthMiddleware` ships in `mactf/http/` but the token table and scope vocabulary
it uses ship in `mactf/contrib/api/auth.py`. The middleware knows
`Principal(name, scopes, anonymous)` from `mactf/http/auth.py`; what a scope means
is the app's business. Removing `mactf.contrib.api` from `INSTALLED_APPS` leaves
the middleware importable and inert.

`Middleware` instances are constructed once per application, not per request.
A middleware that needs per-request state attaches it to the `Request`.

## 8. The WSGI handler

```python
def get_wsgi_application() -> Callable[..., Any]:
    """Return the WSGI callable for the current settings object."""

class WSGIHandler:
    def __init__(self, *, urlconf: str | None = None) -> None: ...
    def __call__(self, environ: dict[str, Any],
                 start_response: Callable[..., Any]) -> Iterable[bytes]: ...
```

`__call__` does, in order:

1. Build a `Request` from `environ`. A malformed `environ` (no
   `REQUEST_METHOD`, no `PATH_INFO`) produces `400` without invoking middleware.
2. Resolve `request.path` through `get_resolver(urlconf)`. A `Resolver404` becomes
   `Http404` and is handled by the middleware chain, so a project that installs
   a custom 404 handler sees it — resolution failures go through the same funnel
   as view failures.
3. Call the composed middleware chain with `request`. When the chain is empty,
   call the resolved view directly.
4. Convert the `Response` into `(status_line, headers, body)`, call
   `start_response(status_line, headers)`, and return the body iterable.
5. Call `response.close()` and log the request line, status, and duration at
   `INFO` with `request_id`.

Details that are specified because they are the ones implementations get wrong:

- **Headers are native strings.** WSGI requires `list[tuple[str, str]]` with
  Latin-1 encodable values. A header value outside Latin-1 is percent-encoded by
  the handler and a warning is logged; it is not silently dropped.
- **`Content-Length` is set from `len(response.content)`** for a non-streaming
  response unless the view set it explicitly. For a streaming response it is not
  set; `wsgiref` composes the transfer framing.
- **`HEAD` responses send headers and zero body**, with the `Content-Length` of
  the equivalent `GET` preserved.
- **A view that raises is caught.** `ExceptionMiddleware` is the normal path;
  if it is absent from `MIDDLEWARE`, the handler still catches `HttpError` and
  returns problem+json, and catches everything else and returns a bare `500`
  problem with no detail. The framework never lets an exception escape a WSGI
  callable, because `wsgiref` answers that with an HTML traceback.
- **`SCRIPT_NAME` is honoured.** A deployment behind a reverse proxy that mounts
  the app at `/mactf` sets `SCRIPT_NAME=/mactf`; `reverse()` prefixes it, and
  `resolve()` strips it. The API therefore does not need a `BASE_PATH` rewrite
  hook, and `API.BASE_PATH` stays a documentation string for clients.
- **No event loop.** Everything in the handler is synchronous. A streaming
  response is a generator; SSE works because the server iterates it. This is the
  single largest difference from the previous FastAPI plan and it is deliberate:
  the run loop is a thread, not a coroutine (document 01 section 3), so an
  asyncio request path would have to bridge back to threads for every write.

`clear_url_caches()` is called by `conf.reset()` and by the test client between
cases, so a test that reloads `ROOT_URLCONF` does not observe a stale resolver.

### 8.1 `mactf serve`

The command lives in `mactf/contrib/api/management/commands/serve.py` and is
discovered because of the app-command discovery in document 13. It is
`wsgiref.simple_server` with `WSGIRequestHandler` overridden only to route its
log lines through `mactf.telemetry`:

```python
def handle(self, *args, **options):
    application = get_wsgi_application()
    server = make_server(options["host"], options["port"], application)
    self.stdout.write(f"Serving on http://{options['host']}:{options['port']}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
```

`--workers` above 1 logs a warning that in-memory warning state is per process
(document 01 section 7), and the command still starts one process. There is no
reloader: `--reload` is not offered, because a development reloader that forks
mid-run corrupts an in-flight training thread. A developer who wants auto-reload
runs their own server in front of `get_wsgi_application()`.

## 9. Serializers

A serializer is a declarative projection between a Python object and JSON. It
replaces the Pydantic models of the previous plan with the same field lists and
the same wire shapes.

```python
class Serializer:
    #: `"forbid"` rejects unknown keys on input (request bodies),
    #: `"allow"` copies them through (response bodies from a newer library).
    unknown_fields: Literal["forbid", "allow"] = "allow"

    def __init__(self, instance: Any = None, *, data: Any = None,
                 many: bool = False) -> None: ...

    def is_valid(self, *, raise_exception: bool = False) -> bool: ...
    @property
    def errors(self) -> dict[str, list[str]]: ...
    @property
    def validated_data(self) -> dict[str, Any]: ...
    def save(self) -> Any: ...          # only for ModelSerializer
    def to_representation(self) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]: ...            # alias for responses

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Serializer: ...
    @classmethod
    def build(cls, obj: Any) -> Serializer: ...
```

Fields:

| Field | Python | Input coercion |
|---|---|---|
| `CharField(max_length=None, min_length=None, pattern=None)` | `str` | rejects non-str |
| `IntField(min_value=None, max_value=None)` | `int` | accepts `"7"`; rejects `7.5` and `True` |
| `FloatField(min_value=None, max_value=None)` | `float` | accepts int and numeric str |
| `BoolField()` | `bool` | accepts `true/false/1/0`, not `"yes"` |
| `ChoiceField(choices, *, case_sensitive=True)` | `str` | rejects a value outside `choices` |
| `ListField(child, allow_empty=True)` | `list` | validates each element |
| `DictField(value_field=None)` | `dict` | validates values when a field is given |
| `NestedField(serializer, many=False)` | `Serializer` | recurses and reports errors under the nested key |
| `RawField()` | any JSON value | passes through unchanged |

`IntField` rejecting `True` is not pedantry: `bool` is a subclass of `int` in
Python, so accepting it means `{"episode": true}` silently becomes episode 1, and
a JSON typo becomes a scored episode.

### 9.1 The annotated declaration form

A serializer may declare fields either as explicit field objects or as annotated
attributes carrying a `Field` metadata object:

```python
# explicit form
class RuleSerializer(Serializer):
    id = CharField(max_length=16)
    tier = IntField(min_value=0, max_value=6)

# annotated form, identical schema and behaviour
class RuleSerializer(Serializer):
    id: str = Field(max_length=16)
    tier: int = Field(min_value=0, max_value=6)
```

`Serializer.__init_subclass__` resolves both forms into one internal field map:
the annotation supplies the field class (`str` -> `CharField`, `int` ->
`IntField`, `float` -> `FloatField`, `bool` -> `BoolField`, `list[X]` ->
`ListField`, a `Serializer` subclass -> `NestedField`), and the `Field` object
supplies `default`, `default_factory`, `min_value`, `max_value`, `min_length`,
`max_length`, `pattern`, `choices`, `read_only`, `allow_null`, and `help_text`.
`get_type_hints` is evaluated once, lazily, at the first `is_valid()` or
`to_representation()` call, so a forward reference does not force an import at
declaration time.

The two forms are the same object graph: `tests/http/test_serializers.py` asserts
that a serializer declared each way produces an identical `openapi` field schema
and identical validation errors. Document 02 uses the annotated form throughout
because a 40-field contract reads better as annotations; the explicit form is used
where the field class matters and no annotation could express it (a
`RawField` for an open `metadata` mapping, or a `DictField` with a typed value).

Validation rules:

- Every field declares `required`, `default` (or `None` plus
  `allow_null=False`), `help_text`, and `read_only`. `mactf.http.openapi` reads
  those four attributes to build the schema, so a field without `help_text`
  produces a schema without a description, which is a review failure.
- `is_valid()` collects **all** field errors, not the first. `errors` is
  `{"field": ["message", ...], "nested": {"child": [...]}}`.
- `is_valid(raise_exception=True)` raises `ValidationError` with
  `errors` attached; `ExceptionMiddleware` renders it as a `422` problem whose
  `extensions["errors"]` is the dict, which is the shape document 02 section 4.1
  publishes.
- `unknown_fields="forbid"` rejects an unknown key with
  `{"non_field_errors": ["unexpected field: 'foo'"]}`. This replaces Pydantic's
  `extra="forbid"` and preserves its behaviour exactly: a typo in a spec is a
  `422`, never a silently ignored key.
- `unknown_fields="allow"` copies unknown keys into the representation
  untouched. It is the default because a response built from a newer library must
  not lose a field an older server does not know about.

### 9.2 `ModelSerializer`

```python
class ModelSerializer(Serializer):
    """Binds a framework dataclass through its own to_dict/from_dict."""

    model: type                    # e.g. PeerScorecard

    def to_representation(self) -> dict[str, Any]:
        return self.instance.to_dict()

    @classmethod
    def build(cls, obj: Any) -> "ModelSerializer":
        return cls(instance=obj)
```

`ModelSerializer` is the whole reason there is no parallel model family. The
framework class is the single source of truth:

- `to_dict()` already exists on `PeerScorecard`, `EpisodeScorecard`,
  `AggregationReport`, `ComplianceResult`, `CapabilityReport`, `ExecResult`, and
  the points/report types (document 02 section 6.1's table).
- `from_dict()` is added to the same classes by document 13 section 5. That is
  what lets the client (document 03) return framework objects instead of its own
  dataclasses, and what lets `ScorecardModel` be a real serializer with a real
  round trip rather than a write-only projection.

A `ModelSerializer` subclass exists only when the wire shape differs from
`to_dict()` — `ScorecardV2Serializer` adds `points`, `normalization`, and
`guards_denied`. In that case the subclass declares the extra fields and
`to_representation` calls `super().to_representation()`, then adds them. The
field list in document 02 remains the contract; the serializer is the mechanism.

`ScorecardSerializer` declares its fields explicitly even though `to_dict()` could
produce them, because the field declarations are what OpenAPI publishes and what
`tests/http/test_serializers.py` compares against the dataclass in both
directions. A field present in one and absent in the other is a test failure; that
is the enforcement clause for principle P2 in document 00.

## 10. Errors

```python
class HttpError(Exception):
    status: int = 500
    title: str = "Internal Server Error"
    type_uri: str = "urn:mactf:error:internal"

class BadRequest(HttpError): status = 400; type_uri = "urn:mactf:error:bad-request"
class NotAuthenticated(HttpError): status = 401
class PermissionDenied(HttpError): status = 403
class Http404(HttpError): status = 404
class MethodNotAllowed(HttpError): status = 405
class Conflict(HttpError): status = 409
class RequestDataTooBig(HttpError): status = 413
class UnsupportedMediaType(HttpError): status = 415
class ValidationError(HttpError): status = 422
class RuleViolation(HttpError): status = 422; type_uri = "urn:mactf:error:rule-violation"
class RateLimited(HttpError): status = 429            # exposes retry_after
class ServerError(HttpError): status = 500
```

`mactf.exceptions`' classes are not re-declared; `mactf/http/exceptions.py`
registers a translation table that `ExceptionMiddleware` consults:

| Framework exception | Status | `type` |
|---|---|---|
| `RuleViolation` | 422 | `urn:mactf:error:rule-violation` |
| `ComplianceError` | 422 | `urn:mactf:error:compliance` |
| `ImproperlyConfigured` | 500 | `urn:mactf:error:configuration` |
| `ContractError` | 422 | `urn:mactf:error:contract` |
| `SandboxError` and subclasses | 502 | `urn:mactf:error:sandbox` |
| `LookupError` | 404 | `urn:mactf:error:not-found` |
| `CommandError` | 400 | `urn:mactf:error:command` |
| `MACTFError` (base) | 500 | `urn:mactf:error:internal` |
| `ValidationError` (http) | 422 | `urn:mactf:error:validation` |
| `Http404` | 404 | `urn:mactf:error:not-found` |
| anything else | 500 | `urn:mactf:error:internal` |

This table is the normative one; document 02 section 4.1 publishes it to clients
and section 10 of this document defines the mechanism. The registry is a
`dict[type[BaseException], Handler]` that an app may extend in `ready()` with
`register_handler(exception_type, handler)` — that is how
`mactf/contrib/api` maps its own `Principal` denials without `mactf/http` knowing
about tokens.

Problem body (RFC 9457):

```json
{
  "type": "urn:mactf:error:compliance",
  "title": "ComplianceError",
  "status": 422,
  "detail": "WEIGHT_CAP=0.3 is below 1/4=0.25 ...",
  "instance": "/api/v1/weights/preview",
  "trace_id": "0f3c1d2e4a5b4c6d8e9f0a1b2c3d4e5f",
  "extensions": {}
}
```

- `instance` is `request.path`, never the full URL, so a host header cannot
  leak into a persisted error.
- `trace_id` is `request.request_id`, also returned as `X-Trace-Id`.
- `extensions` carries structured detail: `errors` for validation, `current_status`
  for a run conflict, `oldest` for a lost event stream. The key is always present
  and always an object, so a client never branches on its existence.
- Stack traces are logged at `ERROR` with the trace id and never serialized,
  except the `debug` extension when `DEBUG` is true.
- A response that is already streaming cannot be replaced by a problem body; the
  handler logs the error and closes the stream. Document 02's SSE section states
  this from the client's side.

## 11. Server-sent events over WSGI

SSE is a `StreamingResponse` with `content_type="text/event-stream"` and three
headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no` (so nginx does not
buffer it), and `Connection: keep-alive`.

A generator yields already-framed bytes:

```
event: epoch\n
id: 42\n
data: {"run_id":"..."}\n
\n
```

Framing is the emit helper's job, not each view's:

```python
def sse_event(event: str, data: Mapping[str, Any], *, event_id: int | None = None) -> bytes: ...
def sse_heartbeat(*, comment: str = "keepalive") -> bytes: ...
```

Two consequences of not having an event loop:

- The generator blocks on `EventBus.subscribe(...)`'s condition variable with a
  timeout equal to the heartbeat interval. A blocking read is correct here: the
  request thread has nothing else to do, and a thread parked on a condition costs
  nothing compared with a busy poll.
- Disconnect is detected on the next write: `start_response` raises
  `BrokenPipeError`/`ConnectionResetError` from the server, the handler closes the
  generator, and the subscription is released in a `finally`. There is no
  `request.is_disconnected()` coroutine, and none is needed.

WebSocket is **not** in this layer and not in the framework. Django core has no
WebSocket support either; the equivalent is a separate package. SSE carries every
event document 02 defines, over one connection, through every proxy that permits
streaming HTTP. Document 00 section 4 records this as a non-goal, and document 02
section 3.4 no longer publishes a `WS` route.

## 12. OpenAPI without a framework

`mactf/http/openapi.py` builds a schema dict from three inputs: the resolved URL
conf, each view's `view_class`/`__doc__`/serializer attributes, and the serializer
field metadata.

```python
def schema(*, title: str = "MACTF", version: str | None = None,
           base_path: str = "api/v1") -> dict[str, Any]: ...
```

What it guarantees:

- one path entry per `URLPattern` reachable from `ROOT_URLCONF`, with the
  converter-aware `{param}` template rather than the regex;
- `operationId` equal to the route `name` (document 02 section 6), so the value is
  stable and matches what `reverse()` accepts;
- request and response bodies from `Serializer` field metadata, including
  `required`, `type`, `format`, `enum` for `ChoiceField`, and `description` from
  `help_text`;
- a `Problem` component referenced by every declared non-2xx response;
- `summary` and `tags` from class attributes on the view
  (`summary`, `tags`, `description`), with a documented fallback to the view's
  docstring first line.

What it does not do: it does not validate that the schema matches reality, and it
does not infer a response model from a view's return statement. The route table
test (document 02 section 5) remains the completeness gate; the OpenAPI test
checks that every operation has an `operationId`, a summary, and a declared
response.

Serving the schema is a view in `mactf/contrib/api/views/meta.py`, not a
`mactf/http` feature. `mactf/http` provides the generator; the app decides
whether to publish it.

## 13. Optional ASGI adapter

`mactf/http/asgi.py` contains a WSGI-to-ASGI bridge for deployments that require
an ASGI server:

```python
def asgi_application() -> Callable[..., Awaitable[None]]:
    """Wrap ``get_wsgi_application()`` as an ASGI 3 application."""
```

It is documented as **optional**: it requires an ASGI server (`uvicorn` or
`hypercorn`) to be installed by the deployer, is not imported by
`mactf.http.__init__`, and is covered by one test that skips when no ASGI server
is present. The bridge runs the WSGI callable in a `ThreadPoolExecutor` with one
worker per request, which is honest about what it is: an adapter, not a second
concurrency model. A deployment that needs native async streaming should use the
WSGI server directly, because SSE over `wsgiref` is simpler than SSE over a
thread-hopping ASGI bridge.

The `[asgi]` extra in `pyproject.toml` (document 06) installs `uvicorn` only. No
module in `mactf/http` imports it; only `asgi.py` does, lazily, inside
`asgi_application()`.

## 14. Layering rules

`mactf/http/` is layer L3. It may import:

- the standard library;
- `mactf.exceptions` (L1), because the translation table is a mapping over those
  types;
- nothing else from `mactf`.

It must **not** import `mactf.orchestration`, `mactf.training`, `mactf.scoring`,
`mactf.coopetition`, `mactf.rules`, `mactf.sandbox`, `mactf.contrib`, or
`mactf.client`. This is enforced by `tests/http/test_layering.py`, which parses
imports in every module under `mactf/http/` and fails with the offending module
path.

Two corollaries that are easy to violate by accident:

- `mactf/http/__init__.py` must not import `mactf/http/asgi.py`. A package
  `__init__` that imports an optional adapter makes the adapter mandatory.
- No module in `mactf/http/` may read a setting other than through
  `mactf.conf.get_settings()`, and none may cache a settings object at module
  level. The handler and the resolver are cached *per settings object identity*,
  which is what makes `conf.configure(force=True)` in a test work.

### 14.1 Settings this layer reads

These keys must be present in `DEFAULTS` ([mactf/conf.py:27](mactf/conf.py#L27)),
or `load_settings` drops a project's override silently
([mactf/conf.py:346](mactf/conf.py#L346)). Document 13 section 3 adds them.

| Key | Default | Used by |
|---|---|---|
| `ROOT_URLCONF` | `None` | `get_resolver`; `None` means the framework's default URL conf |
| `MIDDLEWARE` | `["mactf.http.middleware.RequestIdMiddleware", "mactf.http.middleware.ExceptionMiddleware"]` | `WSGIHandler` |
| `APPEND_SLASH` | `True` | resolver |
| `DEBUG` | `False` | `ExceptionMiddleware` detail level |
| `HTTP.MAX_BODY_BYTES` | `8388608` | `Request.body` |
| `HTTP.STREAM_WORKERS` | `16` | `mactf serve` worker pool, one per SSE stream |
| `HTTP.SSE_HEARTBEAT` | `15.0` | `StreamingResponse` keep-alive comment frame |
| `API.PRINCIPAL_RESOLVER` | `None` | `AuthMiddleware` |
| `API.CORS_ORIGINS` | `[]` | `CorsMiddleware` |

## 15. Tests

`tests/http/` covers the layer without a socket and without a project:

| File | Asserts |
|---|---|
| `test_request.py` | header case-insensitivity, `getlist`, body cap, JSON decoding, `ParseError` and `UnsupportedMediaType` |
| `test_response.py` | bytes-only content, `HEAD` body suppression, `Content-Length`, cookie attributes |
| `test_urls.py` | converters, ordering, `reverse` for every converter, `NoReverseMatch` messages, duplicate-name detection, `APPEND_SLASH` redirect and its `POST` refusal |
| `test_resolver_cache.py` | a second `get_resolver()` returns the same object; `conf.reset()` produces a new one |
| `test_views.py` | `as_view` rejection of shadowing kwargs, `405` with `Allow`, `OPTIONS`, `dict` return conversion, `None` return error |
| `test_middleware.py` | ordering, request id propagation to `X-Trace-Id`, exception conversion, CORS preflight |
| `test_streaming.py` | an SSE generator streams through the handler, heartbeats appear at the interval, a disconnect closes the generator and releases its thread |
| `test_handler.py` | malformed `environ` yields `400`; an escaping exception yields a bare `500` problem and never a traceback; `SCRIPT_NAME` prefixing round-trips through `reverse` |
| `test_serializers.py` | every serializer's field set equals its source dataclass's `to_dict()` keys, in both directions; `forbid` rejects a typo; `allow` copies it; int/float/bool coercion rules |
| `test_openapi.py` | every operation has a summary, a unique `operationId` equal to its route name, and a `Problem` response declared |
| `test_layering.py` | the import rules of section 14 |
| `test_stdlib_only.py` | `sys.modules` after importing `mactf.http` contains no `fastapi`, `starlette`, `pydantic`, `httpx`, or `uvicorn` |

`test_serializers.py` is the load-bearing one. It is what makes "the field lists
and JSON shapes are unchanged" (document 02 section 2) a checked property rather
than a promise kept by review.

## 16. Acceptance criteria for this document

| # | Criterion |
|---|---|
| 39 | `import mactf.http` succeeds in an interpreter with no third-party HTTP package installed |
| 40 | A route declared with `path(..., name="x")` can be produced by `reverse("x", ...)` and consumed by `resolve()`, for all five converters |
| 41 | `get_wsgi_application()` is callable by `wsgiref.simple_server` and by `mactf.testing.Client` without modification |
| 42 | An unhandled exception in a view produces a `500` problem body with a trace id, and no traceback in the body unless `DEBUG` |
| 43 | Removing `mactf.contrib.api` from `INSTALLED_APPS` leaves every module in `mactf/http` importable and every test in `tests/http/` passing |
| 44 | No module under `mactf/http/` imports an engine module, verified by parsing imports |
| 45 | `mactf.http.openapi.schema()` emits an operation for every route in `ROOT_URLCONF` with `operationId == route name` |
| 46 | SSE streams through `wsgiref` with heartbeats, and a closed connection releases the subscription without leaking a thread |

Criteria 39-46 are the transport half of the release gate in document 06; the
app-integration half is 47-54 in document 13.
