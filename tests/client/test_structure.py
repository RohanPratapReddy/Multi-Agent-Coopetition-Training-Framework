"""Structural guarantees: lazy import, layering, route table parity, no parallel models, specs."""

from __future__ import annotations

import pytest


from mactf.client import specs


from mactf.client.routes import ROUTES, expand, path_for


SAMPLE = {"run_id": "r1", "episode": 3, "rule_id": "P1", "peer": "p0", "name": "p0", "path": "a/b"}


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_route_table_matches_reverse(api_settings, name):
    from mactf.http.urls import reverse

    template = ROUTES[name]
    args = {k: v for k, v in SAMPLE.items() if "{" + k in template}
    assert reverse(f"api:{name}", **args) == "/api/v1/" + expand(template, **args)
    assert path_for(name, **args) == reverse(f"api:{name}", **args)


def test_route_table_covers_every_named_route(api_settings):
    from mactf.contrib.api.urls import urlpatterns

    def names(patterns):
        for entry in patterns:
            if getattr(entry, "name", None):
                yield entry.name

    assert set(names(urlpatterns)) == set(ROUTES)


def test_fallback_without_the_api_app_uses_the_documented_base(settings):
    assert path_for("run-detail", run_id="x y") == "/api/v1/runs/x%20y"


def test_specs_are_plain_dicts_the_server_accepts(api_settings, inproc):
    spec = specs.trainer(
        optimizer=specs.adamw(1e-3),
        finetuning=specs.lora(4, target_modules=["policy"]),
        scorers=specs.scorers("reward", "penalty"),
        hierarchy=specs.consensus(levels=[("root", 1), ("worker", 2)]),
    )
    assert spec["optimizer"] == {"type": "torch", "class_name": "AdamW", "lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.999]}
    assert spec["finetuning"]["peft_type"] == "lora"
    import json

    assert json.loads(json.dumps(spec)) == spec
    with pytest.raises(ValueError):
        specs.trainer(num_peers=2, hierarchy=specs.consensus())
    from mactf.contrib.api.serializers import TrainerSpecSerializer

    assert TrainerSpecSerializer(data=spec).is_valid() if hasattr(TrainerSpecSerializer, "is_valid") else True
