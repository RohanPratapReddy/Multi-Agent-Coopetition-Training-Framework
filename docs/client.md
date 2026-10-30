# Client

`mactf.client.MACTFClient` talks to the [HTTP API](api.md) and returns the framework's own
classes (`PeerScorecard`, `Run`, `Episode`, ...), not dictionaries. It has two transports with
identical behaviour and identical exceptions:

| Transport | Spelling | Needs |
|---|---|---|
| `HTTPTransport` | `transport="http", base_url=...` | a running `manage.py serve`; standard-library `urllib` only |
| `InProcessTransport` | `transport="in_process"` | nothing; dispatches through `ROOT_URLCONF` with no socket |

`HTTPTransport` probes `/api/v1/health` when it is constructed, so a wrong URL fails at
once with `TransportError`. Errors map to one hierarchy rooted at `MACTFClientError`
(`ValidationError`, `NotFound`, `Conflict`, `Unauthorized`, `Forbidden`, `RateLimited`, `ServerError`, `CapabilityUnavailable`, `TransportError`), and
each keeps the server's `trace_id`.

Namespaces: `runs`, `scoring`, `scores`, `rules`, `peers`, `sandbox`, `trainers`,
`observability`, plus `health()`, `meta()`, `openapi()`, `wait_for()`, `run_to_completion()` and
`score()` on the client itself.

## Run a swarm and read the result

```python
from mactf.client import MACTFClient

client = MACTFClient(transport="in_process")
run = client.run_to_completion(epochs=2, seed=1337, device="cpu")
for episode in client.runs.iter_episodes(run.id):
    for peer, card in episode.peers.items():
        print(episode.epoch, peer, round(card.final_score, 3))
```

`runs.create(options=None, /, **keywords)` accepts a mapping, keywords, or both; an unknown
key is a `ValidationError` before anything is sent. Pass `idempotency_key=` to make a retried
create return the original run.

## Stream epochs

```python
run = client.runs.create(epochs=5, device="cpu")
for event in client.runs.stream(run.id):
    if event.type == "epoch":
        print(event.epoch, event.data["score"])
```

The iterator reconnects with `Last-Event-ID`. If the server has dropped events it can no
longer replay, it raises `TransportError` rather than skipping them.

## Score a ledger without running anything

```python
card = client.score("peer-a", [("P1", "fabricated a citation", 1.0)])
print(card.final_score, card.is_void)
```

`client.scoring.append_ledger(...)` adds events to a stored episode, and
`client.scoring.adjudicate(...)` records a human ruling.

## Read points

`client.scores.scale()` returns the 0-100 rubric, `points(run_id, episode, peer)` the breakdown
(including denials), `normalization(run_id, episode)` the cohort report, and
`scorecard(run_id, episode, peer, version=2)` the v2 scorecard. Asking for v2 on a run that was
scored without points raises `Conflict`.

## Preview weights and trainers

```python
report = client.peers.preview_weights({"a": 20.0, "b": 80.0})   # an AggregationReport, no run
spec = client.trainers.preview({"optimizer": "adamw", "lr": 1e-4})
```

A `trainer` spec on `runs.create` is validated and stored with the run; the orchestrator does
not execute it. Training itself is done in-process with `MACTFTrainer`, see [training.md](training.md).
