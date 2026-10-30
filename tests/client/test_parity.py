"""Criteria 5 and 53: both transports return identical framework objects and identical exceptions."""

from __future__ import annotations

import dataclasses

import pytest

from mactf.client import (
    Conflict,
    MACTFClientError,
    HTTPTransport,
    InProcessTransport,
    MACTFClient,
    NotFound,
    Unauthorized,
    ValidationError,
)


@pytest.fixture()
def pair(api_settings, server):
    """One app, two clients."""
    in_process = MACTFClient(transport=InProcessTransport(token="all-tok"))
    over_http = MACTFClient(transport=HTTPTransport(server, token="all-tok", timeout=15, backoff=0.01))
    run = over_http.run_to_completion(run_id="par", epochs=2, timeout=120)
    assert run.status == "completed"
    return in_process, over_http


def _same(left, right):
    assert type(left) is type(right), (type(left), type(right))
    if dataclasses.is_dataclass(left):
        assert left == right
    elif hasattr(left, "to_dict"):
        assert left.to_dict() == right.to_dict()
    elif hasattr(left, "as_dict"):
        assert left.as_dict() == right.as_dict()
    else:
        assert left == right


def test_runs_are_the_same_objects(pair):
    a, b = pair
    _same(a.runs.get("par"), b.runs.get("par"))
    assert [r.id for r in a.runs.iter()] == [r.id for r in b.runs.iter()]
    _same(a.runs.episodes("par")[0], b.runs.episodes("par")[0])
    assert a.runs.episodes("par").total == b.runs.episodes("par").total


def test_scorecards_are_the_same_framework_objects(pair):
    from mactf.scoring import PeerScorecard

    a, b = pair
    left = a.runs.scorecard("par", 0, "p0", version=1)
    right = b.runs.scorecard("par", 0, "p0", version=1)
    assert isinstance(left, PeerScorecard) and isinstance(right, PeerScorecard)
    assert left.to_dict() == right.to_dict()


def test_evaluation_is_identical_and_matches_the_engine(pair):
    from mactf.scoring import PeerScorecard

    a, b = pair
    events = [("R1", "answered correctly", 0.5), ("P1", "fabricated a citation", 1.0)]
    left, right = a.score("x", events), b.score("x", events)
    assert left.to_dict() == right.to_dict()
    assert isinstance(left, PeerScorecard) and left.is_void


def test_catalog_and_tables_are_identical(pair):
    a, b = pair
    assert [r.to_dict() for r in a.rules.list()] == [r.to_dict() for r in b.rules.list()]
    assert a.scores.scale().to_dict() == b.scores.scale().to_dict()
    assert a.meta().to_dict() == b.meta().to_dict()


def test_aggregations_and_tree_are_identical(pair):
    a, b = pair
    left = [x.to_dict() if hasattr(x, "to_dict") else x for x in a.runs.aggregations("par")]
    right = [x.to_dict() if hasattr(x, "to_dict") else x for x in b.runs.aggregations("par")]
    assert left == right
    assert type(a.runs.tree("par")) is type(b.runs.tree("par"))


@pytest.mark.parametrize(
    "call, expected",
    [
        (lambda c: c.runs.get("nope"), NotFound),
        (lambda c: c.rules.get("ZZ9"), NotFound),
        (lambda c: c.scoring.evaluate("a", [{"rule_id": "NOPE", "evidence": "x"}]), ValidationError),
        (lambda c: c.runs.create(epochs=-3), ValidationError),
        (lambda c: c.runs.scorecard("par", 99, "p0", version=1), NotFound),
        (lambda c: c.runs.cancel("par"), Conflict),
    ],
)
def test_errors_are_the_same_class_with_the_same_status(pair, call, expected):
    seen = []
    for client in pair:
        with pytest.raises(expected) as caught:
            call(client)
        seen.append((type(caught.value), caught.value.status, caught.value.type_uri))
    assert seen[0] == seen[1]
    assert issubclass(seen[0][0], MACTFClientError)


def test_auth_errors_are_the_same_class(api_settings, server):
    clients = [
        MACTFClient(transport=InProcessTransport()),
        MACTFClient(transport=HTTPTransport(server, timeout=10)),
    ]
    for client in clients:
        with pytest.raises(Unauthorized):
            client.score("a", [("R1", "x")])
    reader = [
        MACTFClient(transport=InProcessTransport(token="read-tok")),
        MACTFClient(transport=HTTPTransport(server, token="read-tok", timeout=10)),
    ]
    from mactf.client import Forbidden

    for client in reader:
        with pytest.raises(Forbidden):
            client.runs.create(epochs=1)


def test_streams_replay_the_same_events(pair):
    a, b = pair
    left = [(e.id, e.event, e.data) for e in a.runs.stream("par")]
    right = [(e.id, e.event, e.data) for e in b.runs.stream("par")]
    assert left and left == right
    assert [e[0] for e in left] == sorted(e[0] for e in left)

