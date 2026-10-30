---
title: Python Client
status: approved-for-implementation
version: 2.0.0
applies_to: mactf >= 0.1.0
owner: TBD
last_reviewed: 2026-10-21
---

# 03. Python Client

`mactf.client` is the framework's client, and it is part of the framework for the
same reason `mactf.http` is (document 12): every object it returns already
exists in `mactf/`, and every route it calls is declared in
`mactf/contrib/api/urls.py`. A separate distribution would have to re-declare
both.

## 1. Goal

One client API, two transports:

- `InProcessTransport` builds a WSGI request and calls
  `get_wsgi_application()` directly. No socket, but the full path: resolver,
  middleware, auth, view, serializer, and JSON round trip.
- `HTTPTransport` speaks the v1 protocol to a running `mactf serve` over
  `urllib.request`.

The same script works against either. That is the client's entire value, and it is
enforced by one parametrized suite (section 9).

Both return the framework's own classes. There is no `mactf/client/models.py`,
because `PeerScorecard`, `EpisodeScorecard`, `AggregationReport`,
`CapabilityReport`, `PointsBreakdown`, and `NormalizationReport` are the models.

## 2. Package layout

```
mactf/client/
  __init__.py       MACTFClient, Transport, Page, RunHandle, Event
  client.py         MACTFClient and namespace facades
  base.py           Transport protocol, RunHandle, Event, Page
  transport.py      InProcessTransport, HTTPTransport
  errors.py         MACTFClientError hierarchy
  pagination.py     Page iterator over a collection
  specs.py          dict builders for TrainerSpec and friends
```

Dependencies: **none beyond the framework**. `transport.py` imports
`urllib.request`, `urllib.error`, `urllib.parse`, `json`, and `uuid` from the
standard library, plus `mactf.http` for `Request`, `Response`, and
`get_wsgi_application()`.

`mactf/client/` may import `mactf.http` and `mactf.exceptions`, and nothing else
from the framework at module scope. In particular it must not import
`mactf.contrib.api`, `mactf.orchestration`, `mactf.training`, `mactf.scoring`, or
`torch` eagerly.

It does need the framework's dataclasses to construct return values, and it gets
them by a deferred import inside the parse helper rather than at module scope:

```python
# mactf/client/transport.py
def _scorecard(payload: dict) -> Any:
    from mactf.scoring.scorecard import PeerScorecard   # deferred, one import
    return PeerScorecard.from_dict(payload)
```

Two obligations follow, and both are tests. `tests/client/test_lazy_import.py`
asserts `sys.modules` after `import mactf.client` contains no `torch`, no
`mactf.training`, and no `mactf.contrib`. `tests/client/test_layering.py` asserts
no module-scope import of an engine package, which is the same rule document 01
section 1 states and criterion 39 checks for `mactf/http`.

That last rule is what makes a dashboard install viable: `pip install mactf`
pulls torch because the trainer needs it, but a client module that imports
`torch` would make merely *referring* to `mactf.client` load a 2 GB package in a
process that only wants to read a scorecard. The dataclasses in
`mactf/scoring/scorecard.py` import `ContinuousScore`, which imports no torch;
`tests/client/test_lazy_import.py` pins that.

## 3. Envelopes

Three small types exist only in the client, because they are transport shapes
rather than domain objects.

```python
# mactf/client/base.py
@dataclass(frozen=True)
class RunHandle:
    run_id: str
    status: str                  # queued | running | cancelled | failed | dry_run
    created_at: float
    links: Mapping[str, str]

@dataclass(frozen=True)
class Event:
    id: int | None
    event: str                   # run | phase | epoch | warning | incident |
                                 # checkpoint | rollback | terminal | heartbeat
    data: Mapping[str, Any]

@dataclass(frozen=True)
class Page(Generic[T]):
    items: tuple[T, ...]
    total: int
    limit: int
    offset: int
    has_more: bool

    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...
```

`Page.__iter__` yields `items`; it does not re-fetch. `pagination.py` provides
`iter_pages(transport, path, **filters)` for a caller that wants to exhaust a
collection, and it is deliberately separate from `Page` so that a `Page` cannot
accidentally issue a request when iterated.

## 4. `Transport`

```python
# mactf/client/base.py
class Transport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response: ...

    def stream(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[Event]: ...
```

`path` is a **route path**, never a route name. The client builds paths with
`reverse()` from `mactf.http`, so document 02's `name` column is the contract the
client is written against — a renamed route is a `NoReverseMatch` at build time
rather than a 404 at runtime. Names are namespaced: the api app declares
`app_name = "api"`, so every client call site uses `reverse("api:<name>", ...)`:

```python
# mactf/client/client.py
from mactf.http import reverse

class RunsAPI:
    def get(self, run_id: str) -> Run:
        response = self._transport.request(
            "GET", reverse("api:run-detail", run_id=run_id)
        )
        return Run.from_dict(_ok(response))
```

`reverse()` on the `HTTPTransport` path resolves against the **local**
`ROOT_URLCONF`. That is the one assumption the client makes about the server it
talks to: the deployment serving `/api/v1` was built from a compatible version.
Document 02 section 4.5's `X-API-Version` header exists so a mismatch is detected
on the first response rather than inferred from a 404.

A project that mounts the api app at a non-default prefix still works unchanged,
because the prefix lives in the project's `urls.py` and `reverse()` reads the
resolver. What does *not* work is a client whose own `ROOT_URLCONF` omits the app:
`reverse()` then raises before a socket is opened, which is the intended failure —
loud, local, and at the first call.

`request()` returns a `mactf.http.Response`, not a `requests`-like object, so
there is exactly one response type in the framework. `stream()` yields `Event`
objects, having parsed SSE frames; a transport that cannot stream raises
`TransportError`, and `HTTPTransport` declines only when the server returns a
non-`text/event-stream` content type.

### 4.1 `InProcessTransport`

```python
class InProcessTransport:
    def __init__(
        self,
        application: Callable[..., Any] | None = None,
        *,
        settings: Settings | None = None,
    ) -> None: ...
```

- `application` defaults to `get_wsgi_application()`, called lazily on the first
  request. Passing one is how a test pins a specific URL conf.
- `request()` builds an `environ` and calls the application:

  ```python
  environ = {
      "REQUEST_METHOD": method,
      "PATH_INFO": parsed.path,
      "QUERY_STRING": urlencode(params or {}, doseq=True),
      "SCRIPT_NAME": "",
      "SERVER_NAME": "localhost",
      "SERVER_PORT": "80",
      "SERVER_PROTOCOL": "HTTP/1.1",
      "wsgi.url_scheme": "http",
      "wsgi.input": io.BytesIO(body),
      "wsgi.errors": io.StringIO(),
      "CONTENT_TYPE": "application/json",
      "CONTENT_LENGTH": str(len(body)),
      "HTTP_ACCEPT": "application/json",
  }
  ```

  `dispatch_wsgi(application, environ)` in `mactf/http/testing.py` calls
  `start_response`, collects the body, closes the iterable, and returns a
  `Response`. `HTTPTransport` and `mactf.testing.Client` use the same helper, so
  the three cannot disagree about how a response is read.
- `stream()` iterates the application's body iterable and parses SSE frames. It
  is genuinely incremental: a generator response is consumed chunk by chunk, so a
  long-lived stream in a notebook yields events as they happen rather than after
  the run ends.
- `Principal` is whatever `AuthMiddleware` resolves. With no token in the
  environment, `API.ALLOW_ANONYMOUS` decides, exactly as over a socket.
- **No serialization shortcut.** The response is bytes parsed by
  `Serializer.from_dict`, so an in-process call exercises every serializer that a
  socket call exercises. Document 13 section 8 states the equivalence obligation
  and the sentinel test that keeps this true.

### 4.2 `HTTPTransport`

```python
class HTTPTransport:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        headers: Mapping[str, str] | None = None,
    ) -> None: ...
```

Behaviour, all of it specified because each item is a bug an implementation gets
wrong:

- **`urllib.request` only.** `Request` from `urllib.request`, `urlopen`, and
  `HTTPError` for non-2xx. `HTTPError` is read into a `Response` and then
  converted to a client exception (section 5), rather than being surfaced as an
  `URLError`.
- **Base URL normalization.** Strip a trailing `/`. Resolve the prefix once at
  construction by probing `GET {base}/api/v1/health` and, on failure,
  `GET {base}/health`. Cache the winner. Both failing raises `TransportError`
  naming both attempted URLs. A client that guesses wrong on every call is harder
  to diagnose than one that fails at construction.
- **Auth.** `Authorization: Bearer <token>` when a token is given.
- **Default headers.** `Accept: application/json` and
  `X-MACTF-Client: <mactf.__version__>`. An `Accept` naming the v2 vendor type is
  set by `ScoresAPI.scorecard(version=2)` only, so a caller that never asks for v2
  never triggers it.
- **Retries.** On `429`, `502`, `503`, `504`, and `urllib.error.URLError`:
  `max_retries` attempts, delay `backoff * 2**attempt` with full jitter, capped at
  30 s, honouring `Retry-After` (seconds or HTTP-date) when present. Never on a
  `POST` without an `Idempotency-Key`; `create_run` generates a `uuid4` key
  itself, so a retry of `POST /runs` is safe by default and a retry of
  `/evaluations` is never attempted.
- **Streaming.** `urlopen` returns a file-like object; `stream()` reads lines,
  assembles `id:`/`event:`/`data:` fields until a blank line, and yields an
  `Event`. On a dropped connection it reconnects with `Last-Event-ID` set to the
  last id seen, up to `max_retries`, then raises `TransportError`. A
  `stream_lost` frame raises `TransportError` carrying `oldest`, because silently
  skipping events is the one failure a stream client must not have.
- **Timeouts.** `timeout` applies per request, not per retry sequence. A streaming
  read uses `timeout` as a socket timeout and therefore assumes the server sends a
  heartbeat within it (document 02 section 3.4 fixes 15 s, so the default 30 s is
  sufficient).
- **No connection pooling configuration.** `urllib` opens a connection per
  request. Stating that is more honest than pretending to tune a pool the
  standard library does not expose.

There is no `AsyncHTTPTransport`. There is no async server to talk to (document
12 section 8), so an async client would be a thread pool behind an `async def`. A
caller in an async context runs `MACTFClient` in `asyncio.to_thread` and gets
exactly that, without a second code path to keep correct.

## 5. Errors

```python
# mactf/client/errors.py
class MACTFClientError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        type_uri: str | None = None,
        trace_id: str | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> None: ...

class ValidationError(MACTFClientError): ...        # 422
class NotFound(MACTFClientError): ...               # 404
class Conflict(MACTFClientError): ...               # 409
class Unauthorized(MACTFClientError): ...           # 401
class Forbidden(MACTFClientError): ...              # 403
class RateLimited(MACTFClientError):
    retry_after: float | None                       # 429

class ServerError(MACTFClientError): ...             # 5xx
class CapabilityUnavailable(MACTFClientError): ...   # 502 urn:mactf:error:sandbox
class TransportError(MACTFClientError): ...          # no connection, timeout, bad frame
```

`HTTPTransport` maps a problem body by `status` first and `type` second:

| Status | `type` | Exception |
|---|---|---|
| 401 | any | `Unauthorized` |
| 403 | any | `Forbidden` |
| 404 | `urn:mactf:error:not-found` | `NotFound` |
| 409 | `urn:mactf:error:conflict` | `Conflict` |
| 413 | `urn:mactf:error:too-large` | `ValidationError` |
| 415 | `urn:mactf:error:unsupported-media-type` | `ValidationError` |
| 422 | any | `ValidationError` |
| 429 | `urn:mactf:error:rate-limited` | `RateLimited` |
| 502 | `urn:mactf:error:sandbox` | `CapabilityUnavailable` |
| other 5xx | any | `ServerError` |

`InProcessTransport` maps identically, because what it receives is the same
problem+json the handler produced. There is no second translation layer and no
catching of library exceptions in the client: `RuleViolation` raised in a view is
a `422` problem by the time the transport sees it, so the client never imports
`mactf.exceptions`.

That is a deliberate reversal of the `LocalBackend` design in the previous
revision, which caught library exceptions directly. Catching them meant the local
path could raise a `ValidationError` for a condition the remote path turned into a
`500`, and the parity suite could not see the difference. Going through the WSGI
application removes the category.

`CapabilityUnavailable` deserves its own class because it is the one error a
caller can act on without a bug report: it means "this host cannot do that", and
`extensions["reasons"]` carries the provider's own list verbatim.

## 6. `MACTFClient`

```python
class MACTFClient:
    def __init__(
        self,
        *,
        transport: Transport | None = None,
        settings: Settings | None = None,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None: ...

    runs: RunsAPI
    scoring: ScoringAPI
    rules: RulesAPI
    peers: PeersAPI
    sandbox: SandboxAPI
    trainers: TrainersAPI
    scores: ScoresAPI
    observability: ObservabilityAPI
```

Construction rules, in order:

1. `transport` given -> use it. This is how a test injects a recording transport
   and how a caller reuses one client across cases.
2. `base_url` given -> `HTTPTransport(base_url, token=token, timeout=timeout)`.
3. Neither -> `InProcessTransport(settings=settings)`.

Local is the default because the common case is a script in the same process as
the engine, and because it requires no configuration at all. A caller that wants
a server says so.

`settings` is used only to build the in-process application. The client does not
mutate global settings: `mactf.conf.configure` is the CLI's job. A test that
wants a specific configuration constructs a `Settings` and passes it.

### 6.1 Namespaces

Namespaces are thin facades over the transport. Any method needing logic beyond
forwarding belongs on `MACTFClient`, not on a namespace.

```python
class RunsAPI:
    def create(self, request: Mapping[str, Any]) -> RunHandle: ...
    def list(self, *, limit: int = 50, offset: int = 0) -> Page[Run]: ...
    def get(self, run_id: str) -> Run: ...
    def cancel(self, run_id: str) -> RunHandle: ...
    def summary(self, run_id: str) -> Mapping[str, Any]: ...
    def episodes(self, run_id: str, *, limit: int = 50, offset: int = 0) -> Page[Episode]: ...
    def episode(self, run_id: str, episode: int) -> Episode: ...
    def scorecard(self, run_id: str, episode: int, peer: str, *,
                  version: int = 1) -> PeerScorecard: ...
    def rule_events(self, run_id: str, **filters: Any) -> Page[RuleEvent]: ...
    def aggregations(self, run_id: str) -> list[AggregationReport]: ...
    def checkpoints(self, run_id: str) -> list[CheckpointRecord]: ...
    def rollback(self, run_id: str, episode: int) -> CheckpointRecord: ...
    def warnings(self, run_id: str) -> Mapping[str, Any]: ...
    def stream(self, run_id: str, *, last_event_id: int | None = None) -> Iterator[Event]: ...
    def tree(self, run_id: str) -> PeerTree: ...
    def node(self, run_id: str, path: str) -> PeerNode: ...
    def infer(self, run_id: str, inputs: Mapping[str, Any], **kwargs: Any) -> InferenceResult: ...
```

```python
class ScoringAPI:
    def evaluate(self, payload: Mapping[str, Any]) -> PeerScorecard: ...
    def append_ledger(self, run_id: str, episode: int,
                      payload: Mapping[str, Any]) -> PeerScorecard: ...
    def adjudicate(self, run_id: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def adjudications(self, run_id: str) -> list[Mapping[str, Any]]: ...
    def flags(self, run_id: str) -> Mapping[str, Any]: ...

class RulesAPI:
    def list(self, *, kind: str | None = None, tier: int | None = None) -> list[Rule]: ...
    def get(self, rule_id: str) -> Rule: ...
    def validation(self) -> CatalogValidation: ...
    def severities(self) -> list[Severity]: ...
    def tiers(self) -> list[Tier]: ...

class PeersAPI:
    def list(self) -> list[Mapping[str, Any]]: ...
    def get(self, name: str) -> Mapping[str, Any]: ...
    def contract(self, name: str) -> Mapping[str, Any]: ...
    def preview_weights(self, payload: Mapping[str, Any]) -> AggregationReport: ...

class SandboxAPI:
    def providers(self) -> list[CapabilityReport]: ...
    def status(self) -> CapabilityReport: ...
    def probe(self) -> CapabilityReport: ...
    def exec(self, peer: str, command: str | list[str], *,
             timeout: float | None = None) -> ExecResult: ...

class TrainersAPI:
    def capabilities(self) -> TrainerCapabilities: ...
    def scorers(self) -> list[ScorerCapability]: ...
    def preview(self, spec: Mapping[str, Any]) -> TrainerPreview: ...
    def presets(self) -> list[Mapping[str, Any]]: ...

class ObservabilityAPI:
    def logs(self, run_id: str, *, tail: int = 100,
             node_path: str | None = None, kind: str | None = None,
             source: str = "memory") -> list[LogRecord]: ...
    def health(self, run_id: str, *, limit: int = 50) -> HealthReport: ...
    def snapshot(self, run_id: str, *, reason: str = "operator") -> CheckpointRecord: ...
    def verify_checkpoint(self, run_id: str, episode: int) -> VerificationResult: ...
    def resume(self, run_id: str, *, checkpoint: int | None = None,
               allow_partial: bool = False) -> ResumeReport: ...
```

Return types are framework classes or narrow client envelopes. Where the value is
genuinely open — `summary`, `flags`, `adjudicate`, `adjudications`, `peers.list`,
`peers.get`, `peers.contract`, `trainers.presets` — the return type is a
`Mapping`, because the API does not constrain the shape and a dataclass would
invent one.

Four names in that signature list are neither engine classes nor transport
envelopes, so they are defined here rather than left to the reader:

| Name | What it is | Why not the engine class |
|---|---|---|
| `Run` | the `runs` row: `id`, `status`, `started_at`, `finished_at`, `epochs`, `config`, `summary` | `mactf.db.RunRecord` is the storage row with `settings`/`runtime` blobs and no `summary`; `Run` is the API projection of it, and importing `mactf.db` into the client would pull sqlite schema knowledge into a remote process |
| `Episode` | the episode row: `run_id`, `episode`, `started_at`, `finished_at`, `scorecard` | `mactf.orchestration.episode.Episode` is the in-flight training object with peers and task state. It is not serializable and not what the route returns |
| `RuleEvent` | one ledger event as stored: `rule_id`, `peer`, `severity`, `amount`, `evidence`, `created_at` | it is a join of `rule_events` and its severity resolution, which no existing dataclass holds |
| `Rule` | the catalog entry | this one **is** `mactf.rules.catalog.Rule`, reused directly |

`Run`, `Episode`, and `RuleEvent` live in `mactf/client/base.py`, next to
`RunHandle`, `Event`, and `Page`, and each is a frozen dataclass with
`from_dict`/`to_dict`. They are transport shapes, not domain models, and the
distinction matters: a `Run` from the client and a `RunRecord` from the registry
can disagree about a field the other does not have, and nothing should quietly
convert one into the other.

`runs.tree` returns `mactf.coopetition.tree.PeerTree` via its existing
`from_dict`. That is why document 13 section 5 lists `PeerTree` as already having
both halves: `to_dict` is at
[mactf/coopetition/tree.py](mactf/coopetition/tree.py) and `from_dict` is already
used by the checkpoint layer.

### 6.2 `ScoresAPI`

The points and calibration facade:

```python
class ScoresAPI:
    def scale(self) -> PointsTable: ...
    def points(self, run_id: str, *, episode: int | None = None,
               peer: str | None = None) -> list[Mapping[str, Any]]: ...
    def normalization(self, run_id: str, episode: int) -> NormalizationReport: ...
    def scorecard(self, run_id: str, episode: int, peer: str, *,
                  version: int = 2) -> PeerScorecard: ...
```

`ScoresAPI.scorecard` is the only namespace method that takes a version, and it is
where the client's v2 negotiation lives. `version=1` sends
`Accept: application/json` and returns the classic shape with `points` possibly
`None`; `version=2` sends `Accept: application/vnd.mactf.v2+json` and raises
`ValidationError` if the response has no `points`. A server that predates v2
therefore fails loudly at the client boundary instead of handing back a
`PeerScorecard` whose `points` the caller then has to null-check.

`points` returns `list[Mapping[str, Any]]` rather than `list[PointsBreakdown]`,
and that exception needs its reason. The route returns one row per
`(episode, peer)`, and a `PointsBreakdown` alone does not say which episode and
peer it belongs to. The alternatives were a wrapper dataclass
(`PointEntry(episode, peer, points)`) or `PointsBreakdown` gaining two fields it
does not conceptually own. Neither is worth it for a batch listing whose natural
consumer is dataframe construction. The typed path is
`scorecard(..., version=2).points`, which is unambiguously scoped. Section 9.1
asserts that `points(...)[i]["final_points"]` equals the corresponding
scorecard's `points.final_points`, so the loose return type cannot drift from the
typed one.

### 6.3 Convenience on the client

Three methods that add value and are implemented once on `MACTFClient`, not on a
transport:

```python
def run_to_completion(
    self,
    request: Mapping[str, Any],
    *,
    on_event: Callable[[Event], None] | None = None,
    timeout: float | None = None,
) -> Run:
    handle = self.runs.create(request)
    for event in self.runs.stream(handle.run_id):
        if on_event is not None:
            on_event(event)
        if event.event == "terminal":
            break
    return self.runs.get(handle.run_id)

def score(
    self,
    peer: str,
    events: Sequence[Mapping[str, Any]],
    *,
    episode: int = 0,
) -> PeerScorecard:
    return self.scoring.evaluate({"peer": peer, "episode": episode,
                                  "events": list(events)})

def wait_for(self, run_id: str, *, timeout: float | None = None) -> Run:
    """Poll until a terminal status, with a linear backoff capped at 2s."""
```

`run_to_completion` is the documented idiom for "start and wait". There is no
`run_and_wait` on a transport, because a transport that could block would diverge
between the two implementations; streaming is the primitive both can honour.

## 7. What the client does not do

| Omitted | Reason |
|---|---|
| A parallel model family | The framework classes are the models. `from_dict` closes the round trip; a second family would drift (document 00 P2) |
| `httpx` | `urllib.request` is stdlib and sufficient for a synchronous client |
| An async client | No async server. `asyncio.to_thread(client.runs.get, run_id)` is the documented form |
| Auto-generated stubs from OpenAPI | The protocol is hand-written so `InProcessTransport` can satisfy it. A generated HTTP client cannot |
| A CLI | `mactf` already is the CLI and stays canonical |
| Client-side scoring | A second rubric is the failure mode document 00 P10 forbids |
| Recomputing points | `PointsBreakdown.explain()` renders what the server returned. It never recomputes a weight or a fraction; that would be a second rubric that silently disagrees after a table upgrade |
| A `NormalizationPolicy` constructor | The policy is echo-only (section 8). A client that could build one would expect to send it, and calibration is a server-side cohort operation, not a per-request parameter |
| Constructing `MACTFOptimizer` or `Scorer` | The client sends specs; it does not instantiate library objects. Importing `torch` in a client is a dependency `mactf.client` does not take |
| A pandas export | Not every user has pandas, and the transform is three lines for users who do |
| Retry on non-idempotent writes | Adjudications and ledger appends are not safely replayable; the caller decides |
| `{"type": "callable", "module": ...}` over HTTP | The server must not import a client-supplied module path; the server rejects it and the client raises `ValidationError`. `InProcessTransport` permits it because the module is already in the caller's process |

The last row is the only behavioural asymmetry between the transports, and it is
tested: `tests/client/test_parity.py::test_callable_spec_rejected_remotely`
asserts the in-process path accepts a callable spec that the HTTP path rejects.
Document 13 section 8 states the equivalence obligation and lists this as one of
two allowed exceptions; the other is the `MAX_CONCURRENT_RUNS` semaphore, which
only the server enforces.

## 8. Specs are dicts, not dataclasses

`TrainerSpec`, `ScorerChainSpec`, `LossSpec`, `OptimizerSpec`, `FinetuneSpec`,
and `HierarchySpec` are **not** duplicated as client dataclasses.
`create_run` and `trainers.preview` accept a plain `Mapping[str, Any]`, and
`mactf/client/specs.py` provides thin builders that return dicts:

```python
from mactf.client.specs import trainer, lora, adamw, consensus

spec = trainer(
    num_peers=3,
    optimizer=adamw(lr=3e-4),
    finetuning=lora(r=8, alpha=16, target_modules="auto"),
    hierarchy=consensus(levels=[("subagent", 2), ("miniagent", 3)]),
)
client.runs.create({"epochs": 3, "trainer": spec})
```

Two reasons:

1. The server is the authority on which keys are valid, and it validates with
   `unknown_fields="forbid"`. A client-side dataclass would either duplicate that
   vocabulary or drift from it, and a drifted copy is the failure mode where a
   valid server option is unreachable from the client.
2. The client and the server must accept byte-identical JSON. A dataclass layer
   between them is one more place for a field name to differ, and
   `tests/client/test_spec_roundtrip.py` compares a captured request body against
   the server's normalized spec.

`mactf/client/specs.py` imports nothing from the server, so a builder cannot
become a second validation layer. It returns plain dicts; the server's serializers
do all validation.

Reports are dataclasses because callers write `report.bit_identical_expected`, not
`report["..."]`. Specs are dicts because callers build them once and send them.
The asymmetry is deliberate.

## 9. Parity requirement

`tests/client/test_parity.py` runs one parametrized suite over
`[InProcessTransport(), HTTPTransport(base_url=server.url)]`, where the HTTP
fixture starts `wsgiref.simple_server` on an ephemeral port in a thread and shuts
it down in a fixture finalizer. Each test asserts identical **framework objects**,
not merely "no exception".

| Test | Assertion |
|---|---|
| `test_meta_matches` | both return the same `api_version`, `VERSION`, and `len(CATALOG)` |
| `test_create_run` | both return a `RunHandle` with a 12-hex-char `run_id` and status `queued` or `running` |
| `test_run_completes` | both reach `completed`; the final `Run.kappa` and `entropy` match to 1e-9 for the same seed |
| `test_episodes_match` | both return the same number of episodes; every `final_score` matches to 1e-9 |
| `test_scorecard_matches` | every field of `PeerScorecard` is equal, including `floor_reason` and the `explain` text |
| `test_rules_match` | 40 rules, same ids, same tiers, same severities |
| `test_evaluation_p1_voids` | both return `void=True`, `final_score=-5.0` |
| `test_preview_weights_match` | identical weights to 1e-9, and `kappa` unchanged by the preview in both |
| `test_cancel` | both end `cancelled`, both reject a second cancel with `Conflict` |
| `test_not_found` | both raise `NotFound` for an unknown run |
| `test_validation_error` | both raise `ValidationError` for an unknown rule id |
| `test_spec_roundtrip` | a `TrainerSpec` dict sent by both transports produces the same normalized spec in `TrainerPreview.spec` |
| `test_preview_matches_build` | the tree, trainable counts, and `optimizer_bytes` in `TrainerPreview` equal what a real `build()` produces, without training |
| `test_tree_matches` | both return the same `PeerTree`: same `nodes` paths, same `root_paths`, same `depth`, same `source` |
| `test_inference_matches` | both return the same `root_outputs` to 1e-9 for the same seed, and the same `batching` decisions |
| `test_logs_ordering` | both return records in the same `monotonic` order with identical `kind`/`epoch`/`node_path` sequences |
| `test_fallback_reported` | a `weight_update` that forces the fallback reports `fallback=True` and the same `adjustments` text in both |
| `test_resume_report_matches` | both return a `ResumeReport` with identical `restored_sections`, `missing_sections`, and `bit_identical_expected` |
| `test_health_matches` | both return the same `status`, `action`, and `per_node` for the same epoch |
| `test_capabilities_match` | `available_extras` and `optimizer_granularity` agree, because both probe the same interpreter |
| `test_serializers_are_exercised` | the in-process path calls `to_representation`, proven by a sentinel serializer replaced in a test |

Tolerances: floats compared with `abs(a-b) <= 1e-9` for values that come from the
same seed and device. The suite pins `DEVICE=cpu`, `SEED=1337`,
`DETERMINISTIC=True`, `EPOCHS=1`, and a two-peer swarm, so non-determinism is not
available as an excuse.

`test_serializers_are_exercised` is the load-bearing one. The tempting
optimization is to have `InProcessTransport` hand back the view's object directly,
which would pass every equality test while silently stopping all serializer
coverage. A sentinel fails the moment that shortcut appears.

### 9.1 Points parity

The points surface gets its own rows, because a rubric that differs between
transports would be worse than no rubric at all: a notebook experiment on the
in-process transport would not predict the scored results from a served run.

| Test | Assertion |
|---|---|
| `test_scale_matches` | `scores.scale()` is identical field by field on both, and `rewards` sums to `100.0` in both |
| `test_points_match` | for the same evaluated payload, both return identical `PointsBreakdown`, including every `RulePoints` and `DenialRecord`, field by field |
| `test_points_denials_match` | a payload that trips a guard produces the same `denials` in both, in the same order |
| `test_void_pin_matches` | both return `final_points == 0.0` and `void is True` for a P1 payload, while both still return the same non-zero `final_score` |
| `test_normalization_matches` | `normalization` for a four-peer episode agrees on `calibrated`, `percentile`, `rank`, and `signal` to `1e-9` |
| `test_normalization_stats_match` | `CohortStats` agrees on every field including `excluded_void` |
| `test_v2_negotiation_both_transports` | `scores.scorecard(..., version=2)` returns a `PeerScorecard` with `points` non-`None` on both, and `version=1` permits `None` |
| `test_run_points_agrees_with_scorecard` | the row for `(episode, peer)` in `points(...)` equals that peer's `scorecard(...).points` |
| `test_scale_stable_across_runs` | `scores.scale()` is identical before and after a run, so the table is deployment state and not run state |

`test_void_pin_matches` is the one that protects the entire design. It asserts the
non-interference property from
[07-scorers-and-functions.md](07-scorers-and-functions.md) section 11 from the
outside: `final_score` and `final_points` can and do disagree, and both transports
agree about *how* they disagree.

### 9.2 Error parity

Four cases per transport, asserting the identical exception class and the
identical `type_uri`, `status`, `trace_id` presence, and `extensions` keys:

| Case | Expected |
|---|---|
| Unknown rule id on `POST /evaluations` | `ValidationError`, `urn:mactf:error:rule-violation` |
| Unknown run id on `runs.get` | `NotFound`, `urn:mactf:error:not-found` |
| Cancel on a `completed` run | `Conflict`, extension `current_status` |
| Sandbox exec with no handle | `Conflict`, extension `reason` |

The HTTP transport produces these by parsing a problem body; the in-process
transport receives the same body. If the two ever differ, the problem renderer
and the client parser disagree, which is exactly what this table exists to
detect.

## 10. Why the client is in core

Three arguments, in order of weight:

1. **It returns framework objects.** A separate distribution cannot import
   `mactf.scoring.PeerScorecard` without depending on `mactf` anyway, so the split
   buys nothing and costs a version-skew constraint.
2. **It resolves routes with `mactf.http.reverse`.** The route names are the
   framework's contract (document 02 section 3), and `reverse` is in
   `mactf/http/urls.py`. A separate distribution would have to hard-code paths or
   ship its own resolver.
3. **It must be installable without torch.** That is a property of which modules
   the client imports, not of which distribution it lives in.
   `tests/client/test_lazy_import.py` pins the import set, and `mactf/scoring/
   scorecard.py`'s dataclasses already avoid torch. Moving the package out would
   not change that test; it would only change the install instructions.

The one thing a separate distribution would genuinely give — an independent
release cadence — is the thing the framework's single-version policy
([06-verification-and-release.md](06-verification-and-release.md) section 8)
deliberately does not offer.
