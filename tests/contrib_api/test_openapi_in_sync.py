"""The generated route table of docs/api.md and the live OpenAPI document agree with the URL conf."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from api_support import body

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_api_docs", ROOT / "scripts" / "gen_api_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openapi_in_sync(gen, client):
    """docs/api.md is what ``scripts/gen_api_docs.py`` writes, and /openapi.json covers every route."""
    from mactf.contrib.api.urls import urlpatterns

    committed = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    assert gen.render(committed, gen.document()) == committed, "docs/api.md is stale: python scripts/gen_api_docs.py"
    live = body(client.get("/openapi.json"))
    ids = {op["operationId"] for item in live["paths"].values() for op in item.values()}
    names = {entry.name for entry in urlpatterns if getattr(entry, "name", None)}
    assert names <= ids, sorted(names - ids)
    assert {"component-list", "component-validate", "component-detail", "trainer-validate",
            "run-components", "run-mutation-create"} <= ids
    for name in ("component-list", "run-mutation-create"):
        assert f"`{name}`" in committed


def test_the_component_routes_describe_their_bodies(client):
    live = body(client.get("/openapi.json"))
    patch = live["paths"]["/api/v1/runs/{run_id}/components"]["patch"]
    assert patch["operationId"] == "run-components"
    schema = body(client.get("/openapi.json"))["paths"]["/api/v1/components/validate"]["post"]
    assert "requestBody" in schema
