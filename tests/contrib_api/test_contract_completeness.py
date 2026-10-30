"""Document 02: every documented route exists, with its name, method and scope."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mactf.http import urls as http_urls
from mactf.http.urls import URLPattern, URLResolver, reverse

DOC = Path(__file__).resolve().parents[2] / "docs" / "plans" / "02-api-contract.md"
ROW = re.compile(r"^\|\s*([A-Z]+)\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*(?:\||$)")


def documented():
    rows = []
    for line in DOC.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match:
            rows.append(match.groups())
    return rows


def mounted():
    from mactf.contrib.api import urls as api_urls

    found = []
    for entry in api_urls.urlpatterns:
        view = entry.callback
        cls = getattr(view, "view_class", None)
        methods = [m.upper() for m in cls.http_method_names if hasattr(cls, m) and m not in ("options", "head")]
        found.append((entry.name, entry.route, methods, cls))
    return found


def _norm(route: str) -> str:
    return re.sub(r"\{(\w+)(:\w+)?\}", lambda m: "{%s}" % m.group(1), route.lstrip("/"))


def test_document_has_the_expected_number_of_rows():
    assert len(documented()) >= 55


def test_every_documented_method_and_path_is_mounted():
    have = set()
    for name, route, methods, _ in mounted():
        for method in methods:
            have.add((method, _norm(route)))
    missing = [(m, p) for m, p, *_ in documented() if (m, _norm(p)) not in have]
    assert not missing, missing


def test_every_documented_route_name_exists():
    names = {name for name, *_ in mounted()}
    missing = sorted({row[2] for row in documented()} - names)
    assert not missing, missing


def test_nothing_is_mounted_that_the_document_does_not_describe():
    wanted = {(m, _norm(p)) for m, p, *_ in documented()}
    extra = []
    for name, route, methods, _ in mounted():
        for method in methods:
            if (method, _norm(route)) not in wanted:
                extra.append((method, route))
    assert not extra, extra


def test_ordering_constraints():
    routes = [_norm(route) for _, route, _, _ in mounted()]
    assert routes.index("rules/validation") < routes.index("rules/{rule_id}")
    assert routes.index("runs/{run_id}/peers/tree") < routes.index("runs/{run_id}/peers/{path}")


def test_write_routes_declare_a_write_scope(api_settings):
    scopes = {}
    for name, route, methods, cls in mounted():
        for method in methods:
            scopes[(method, _norm(route))] = (getattr(cls, "scopes", {}) or {}).get(method.lower())
    for method, path, name, scope, _status in documented():
        wanted = scope.strip("` ")
        if wanted == "none":
            continue
        declared = scopes.get((method, _norm(path)))
        assert declared == wanted or (declared is None and wanted == "read"), (method, path, declared, wanted)


@pytest.mark.parametrize("name,kwargs,expected", [
    ("health", {}, "/api/v1/health"),
    ("run-detail", {"run_id": "r1"}, "/api/v1/runs/r1"),
    ("run-episode-detail", {"run_id": "r1", "episode": 3}, "/api/v1/runs/r1/episodes/3"),
    ("run-peer-node", {"run_id": "r1", "path": "a/b"}, "/api/v1/runs/r1/peers/a/b"),
    ("run-adjudication-list", {"run_id": "r1"}, "/api/v1/runs/r1/adjudications"),
    ("run-adjudication-create", {"run_id": "r1"}, "/api/v1/runs/r1/adjudications"),
])
def test_reverse(api_settings, name, kwargs, expected):
    assert reverse(f"api:{name}", **kwargs) == expected
