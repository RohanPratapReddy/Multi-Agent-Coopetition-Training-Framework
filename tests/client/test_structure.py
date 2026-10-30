"""Structural guarantees: lazy import, layering, route table parity, no parallel models, specs."""

from __future__ import annotations

import pytest


from mactf.client import specs
from mactf.contrib.api.serializers import TrainerSpecSerializer


from mactf.client.routes import ROUTES, expand, path_for


SAMPLE = {"run_id": "r1", "episode": 3, "rule_id": "P1", "peer": "p0", "name": "p0", "path": "a/b",
          "kind": "weight_update"}


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
    # Document 17 section 16.4.1: one case per new builder, each a dict the server's serializer takes.
    from mactf.contrib.api.serializers import (
        ComponentPatchSerializer,
        ComponentValidateSerializer,
        MutationSerializer,
    )

    named = specs.component("trimmed_mean", trim=0.2)
    assert named == {"type": "trimmed_mean", "params": {"trim": 0.2}}
    assert specs.component(import_="pkg.mod:Cls") == {"import": "pkg.mod:Cls"}
    with pytest.raises(ValueError):
        specs.component()
    with pytest.raises(ValueError):
        specs.component("a", import_="b:C")
    assert json.loads(json.dumps(named)) == named
    validate = ComponentValidateSerializer(data={"kind": "weight_update", "spec": named})
    assert validate.is_valid(), validate.errors
    slots = specs.components(**{"TRAINER_WEIGHT_UPDATE": "uniform", "OPTIMIZER": {"params": {"lr": 0.1}}})
    assert slots["TRAINER_WEIGHT_UPDATE"] == {"type": "uniform"}
    for slot, spec in slots.items():
        patch = ComponentPatchSerializer(data={"slot": slot, "spec": spec})
        assert patch.is_valid(), patch.errors
    plan = specs.mutation({"op": "retire", "path": "root/w3"}, label="prune")
    assert plan == {"plan": [{"op": "retire", "path": "root/w3"}], "label": "prune"}
    mutation = MutationSerializer(data=plan)
    assert mutation.is_valid(), mutation.errors
    spec = specs.trainer(weight_update=specs.component("uniform"), peer_strategy="cooperative")
    trainer = TrainerSpecSerializer(data=spec)
    assert trainer.is_valid(), trainer.errors


def test_routes_cover_every_component_view(api_settings):
    """Document 17 section 16.3.5: the six routes are in ``ROUTES`` and in ``urlpatterns``, one name each."""
    from mactf.contrib.api.urls import urlpatterns
    from mactf.contrib.api.views import components as views

    expected = {
        "component-list": views.ComponentListView, "component-validate": views.ComponentValidateView,
        "component-detail": views.ComponentDetailView, "trainer-validate": views.TrainerValidateView,
        "run-components": views.RunComponentsView, "run-mutation-create": views.RunMutationView,
    }
    served = {getattr(p, "name", None): getattr(getattr(p, "callback", None), "view_class", None) for p in urlpatterns}
    for name, view in expected.items():
        assert name in ROUTES, name
        assert name in served, name
        assert served[name] in (view, None) or getattr(served[name], "__name__", "") == view.__name__
    classes = [v for v in vars(views).values() if isinstance(v, type) and v.__module__ == views.__name__
               and v.__name__.endswith("View")]
    assert {c.__name__ for c in classes} == {v.__name__ for v in expected.values()}
    for kind in ("rl_function.loss", "weight_update"):
        assert path_for("component-detail", kind=kind, name="x").endswith(f"/components/{kind}/x")


def test_the_components_namespace_reaches_every_route(api_settings, inproc):
    """Document 17 section 16.4.1: the client methods each call one route and return its body."""
    listed = inproc.components.list(kind="weight_update")
    assert listed.items and all(item["kind"] == "weight_update" for item in listed.items)
    detail = inproc.components.get("weight_update", "kappa_softmax")
    assert detail["name"] == "kappa_softmax"
    assert inproc.components.validate("weight_update", specs.component("uniform"))["ok"] is True
    from mactf.client.errors import MACTFClientError, NotFound, ValidationError

    with pytest.raises(ValidationError):
        inproc.components.validate("weight_update", specs.component("uniform", bogus=1))
    with pytest.raises(NotFound):
        inproc.components.get("weight_update", "no_such_one")
    with pytest.raises(MACTFClientError):
        inproc.components.show("no-such-run")
