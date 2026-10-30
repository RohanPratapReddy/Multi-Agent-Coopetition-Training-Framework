"""Document 17 section 16.3: the component and mutation routes, through the WSGI test client."""

from __future__ import annotations

import time

import pytest

from api_support import ApiClient, body
from mactf.components import HUB, overlay, register_weight_update
from mactf.contrib.api import urls as api_urls
from mactf.contrib.api.serializers import (
    ComponentChoiceField,
    ComponentValidateSerializer,
    TrainerSpecSerializer,
)
from mactf.http import resolve
from mactf.http.handler import WSGIHandler


# -- a real, live trainer-engine run -----------------------------------------------------------
@pytest.fixture()
def live(api_settings):
    """A trainer-engine run that is still running (400 epochs of three tiny peers)."""
    api_settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    api_settings["TRAINING"]["EPOCHS"] = 400
    api_settings["TRAINER"]["MUTATION"] = {**api_settings["TRAINER"].get("MUTATION", {}), "ENABLED": True}
    api_settings["PEERS"] = [*api_settings["PEERS"], {"name": "p2", "class": "ApiTinyPeer", "kwargs": {}}]
    handler = WSGIHandler()
    client = ApiClient(handler)
    assert client.post("/runs", json={"epochs": 400, "run_id": "lv"}).status_code == 202
    manager = handler.state["run_manager"]
    deadline = time.time() + 40
    while time.time() < deadline:
        handle = manager.get_handle("lv")
        trainer = getattr(handle.orchestrator, "trainer", None) if handle else None
        if trainer is not None and getattr(trainer, "built", False) and handle.epochs_completed >= 1:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the run never started")
    yield client, manager
    client.post("/runs/lv/cancel")
    manager.close()


def _slot(payload, path):
    return next(item for item in payload["slots"] if item["slot"] == path)


def _wait(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not reached")


# -- routing ----------------------------------------------------------------------------------
def test_literal_routes_precede_parameterised(api_settings):
    """``components/validate`` resolves to the validate view, never to ``component-detail``."""
    names = [entry.name for entry in api_urls.urlpatterns]
    assert names.index("component-validate") < names.index("component-detail")
    assert resolve("/api/v1/components/validate").url_name == "component-validate"
    assert resolve("/api/v1/components/weight_update/uniform").url_name == "component-detail"
    assert resolve("/api/v1/components/rl_function.loss/policy_gradient").kwargs["kind"] == "rl_function.loss"


# -- the stateless routes -----------------------------------------------------------------------
def test_list_filters_and_paginates(client):
    everything = body(client.get("/components?limit=500"))
    assert everything["total"] > 20
    kinds = {row["kind"] for row in everything["items"]}
    assert {"weight_update", "scorer", "peer_strategy"} <= kinds
    only = body(client.get("/components?kind=weight_update"))
    assert only["items"] and {row["kind"] for row in only["items"]} == {"weight_update"}
    assert {"kappa_softmax", "uniform"} <= {row["name"] for row in only["items"]}
    assert body(client.get("/components?kind=weight_update&origin=project"))["total"] == 0
    assert body(client.get("/components?limit=2"))["has_more"] is True
    assert client.get("/components?locked=maybe").status_code == 422


def test_detail_describes_one_component_and_404s_with_the_accepted_names(client):
    detail = body(client.get("/components/weight_update/uniform"))
    assert detail["name"] == "uniform" and detail["kind"] == "weight_update"
    missing = client.get("/components/weight_update/unifrom")
    assert missing.status_code == 404
    problem = body(missing)
    assert problem["extensions"]["code"] == "E001" and "uniform" in problem["extensions"]["known"]
    assert "did you mean 'uniform'" in problem["detail"]
    assert client.get("/components/no_such_kind/x").status_code == 404


def test_validate_accepts_a_good_spec_and_reports_every_bad_one(client):
    good = client.post("/components/validate", json={"kind": "weight_update", "spec": {"type": "uniform"}})
    assert good.status_code == 200
    assert body(good)["ok"] is True and body(good)["spec_hash"]
    bad = client.post("/components/validate",
                      json={"kind": "weight_update", "spec": {"type": "uniform", "params": {"trimm": 0.2}}})
    assert bad.status_code == 422
    problem = body(bad)["extensions"]
    assert problem["code"] == "E004" and problem["findings"][0]["severity"] == "error"
    assert "trimm" in problem["errors"]["spec"][0]


def test_validate_is_untrusted_about_imports(client):
    response = client.post(
        "/components/validate", json={"kind": "weight_update", "spec": {"import": "os.path:join"}},
    )
    assert response.status_code == 422
    assert body(response)["extensions"]["code"] == "E006"


def test_validate_unknown_name_is_404_and_a_patchless_spec_is_422(client):
    missing = client.post("/components/validate", json={"kind": "weight_update", "spec": {"type": "nope"}})
    assert missing.status_code == 404 and body(missing)["extensions"]["code"] == "E001"
    empty = client.post("/components/validate", json={"kind": "weight_update", "spec": {"params": {}}})
    assert empty.status_code == 422
    assert body(empty)["extensions"]["code"] == "E005"
    assert client.post("/components/validate", json={"kind": "bogus", "spec": {"type": "x"}}).status_code == 422


def test_validate_probe_runs_the_authors_probe(client):
    ok = client.post("/components/validate",
                     json={"kind": "weight_update", "spec": {"type": "uniform"}, "probe": True})
    assert ok.status_code == 200
    scheduler = client.post(
        "/components/validate",
        json={"kind": "scheduler", "probe": True,
              "spec": {"type": "warmup_cosine", "params": {"warmup_steps": 20, "total_steps": 10}}},
    )
    assert scheduler.status_code == 422
    assert body(scheduler)["extensions"]["code"] == "E008"


def test_trainer_validate_reports_all_problems_at_once(client):
    fine = client.post("/trainers/validate", json={"weight_update": {"type": "uniform"}, "num_peers": 2})
    assert fine.status_code == 200 and body(fine) == {"ok": True, "findings": []}
    bad = client.post("/trainers/validate", json={
        "weight_update": {"type": "uniform", "params": {"zz": 1}},
        "peer_strategy": {"import": "os.path:join"},
        "optimizer": {"type": "torch", "class_name": "NoSuchOptimizer", "lr": 0.1},
    })
    assert bad.status_code == 422
    findings = body(bad)["findings"]
    assert {item["code"] for item in findings} >= {"E004", "E006"}
    assert len(findings) >= 3
    # a name the registry does not know is refused by the serializer, before the view
    assert client.post("/trainers/validate", json={"weight_update": "nope"}).status_code == 422


def test_trainer_spec_names_come_from_the_live_registry(client):
    """A strategy registered after the serializer class was built is accepted (16.3.2)."""
    assert client.post("/trainers/validate", json={"weight_update": "tmp_uniform"}).status_code == 422
    from mactf.coopetition.weights import UniformAverage

    with overlay():
        register_weight_update("tmp_uniform", scope="run")(type("TmpUniform", (UniformAverage,), {}))
        assert client.post("/trainers/validate", json={"weight_update": "tmp_uniform"}).status_code == 200
        enum = TrainerSpecSerializer.schema()["properties"]["weight_update"]["oneOf"][0]["enum"]
        assert "tmp_uniform" in enum
    assert client.post("/trainers/validate", json={"weight_update": "tmp_uniform"}).status_code == 422


def test_old_trainer_bodies_still_validate(client):
    for spec in ({"peer_strategy": {"type": "frozen_peer", "frozen": ["p0"]}},
                 {"weight_update": {"type": "explicit", "weights": {"p0": 0.5, "p1": 0.5}}},
                 {"peer_strategy": "cooperative", "weight_update": "kappa_softmax"}):
        assert client.post("/trainers/validate", json=spec).status_code == 200, spec


def test_choice_field_refreshes():
    """A component registered after the field was built appears after ``refresh()``."""
    from mactf.coopetition.weights import UniformAverage

    field = ComponentChoiceField("weight_update")
    assert "late_uniform" not in field.choices
    with overlay():
        register_weight_update("late_uniform", scope="run")(type("LateUniform", (UniformAverage,), {}))
        assert "late_uniform" not in field.choices          # built before the registration
        field.refresh()
        assert "late_uniform" in field.choices
        assert field.to_internal("late_uniform") == "late_uniform"
        assert ComponentValidateSerializer(data={"kind": "weight_update", "spec": {"type": "x"}}).is_valid()
    field.refresh()
    assert "late_uniform" not in field.choices
    assert "late_uniform" not in HUB.registry("weight_update").names()


# -- a live run ---------------------------------------------------------------------------------
def test_show_a_live_runs_components(live):
    client, _manager = live
    shown = body(client.get("/runs/lv/components"))
    assert shown["source"] == "live" and shown["run_id"] == "lv"
    weight = _slot(shown, "TRAINER.WEIGHT_UPDATE")
    assert weight["name"] == "kappa_softmax" and weight["kind"] == "weight_update"
    assert weight["mutability"] == "episode" and weight["spec_hash"]
    assert _slot(shown, "TRAINER.POLICY")["mutability"] == "static"
    assert {"scoring.reward", "scoring.penalty", "scoring.warning"} <= {item["slot"] for item in shown["slots"]}
    assert client.get("/runs/nosuch/components").status_code == 404


def test_patch_queues_a_swap_that_lands_at_the_next_boundary(live):
    client, _manager = live
    queued = client.request("PATCH", "/runs/lv/components",
                            json={"slot": "TRAINER.WEIGHT_UPDATE", "spec": {"type": "uniform"}})
    assert queued.status_code == 202
    answer = body(queued)
    assert answer["slot"] == "TRAINER.WEIGHT_UPDATE" and answer["boundary"] == "episode"
    assert answer["spec_hash"] and answer["applies_at"] and answer["queued_at"]

    def swapped():
        current = body(client.get("/runs/lv/components"))
        return current if _slot(current, "TRAINER.WEIGHT_UPDATE")["name"] == "uniform" else None

    current = _wait(swapped)
    record = [row for row in current["history"] if row["slot"] == "TRAINER.WEIGHT_UPDATE"][-1]
    assert (record["from_name"], record["to_name"]) == ("kappa_softmax", "uniform")
    assert _slot(current, "TRAINER.WEIGHT_UPDATE")["spec_hash"] == answer["spec_hash"]


def test_patch_with_only_params_is_merged_over_the_current_spec(live):
    client, _manager = live
    first = client.request("PATCH", "/runs/lv/components",
                           json={"slot": "TRAINER.PEER_STRATEGY", "spec": {"type": "independent"}})
    assert first.status_code == 202
    _wait(lambda: _slot(body(client.get("/runs/lv/components")), "TRAINER.PEER_STRATEGY")["name"] == "independent")
    bad = client.request("PATCH", "/runs/lv/components",
                         json={"slot": "TRAINER.PEER_STRATEGY", "spec": {"params": {"nope": 1}}})
    assert bad.status_code == 422 and body(bad)["extensions"]["code"] == "E004"
    assert body(client.get("/runs/lv/components"))["pending"] == {}          # nothing was queued


def test_patch_refusals_carry_their_codes(live):
    client, _manager = live
    missing = client.request("PATCH", "/runs/lv/components", json={"slot": "NOPE", "spec": {"type": "uniform"}})
    assert missing.status_code == 404 and body(missing)["extensions"]["code"] == "E010"
    static = client.request("PATCH", "/runs/lv/components",
                            json={"slot": "TRAINER.POLICY", "spec": {"type": "default"}})
    assert static.status_code == 409 and body(static)["extensions"]["code"] == "E003"
    unknown = client.request("PATCH", "/runs/lv/components",
                             json={"slot": "TRAINER.WEIGHT_UPDATE", "spec": {"type": "nope"}})
    assert unknown.status_code == 404 and body(unknown)["extensions"]["code"] == "E001"
    denied = client.request("PATCH", "/runs/lv/components",
                            json={"slot": "TRAINER.WEIGHT_UPDATE", "spec": {"import": "os.path:join"}})
    assert denied.status_code == 422 and body(denied)["extensions"]["code"] == "E006"
    stepwise = client.request("PATCH", "/runs/lv/components", json={
        "slot": "TRAINER.WEIGHT_UPDATE", "spec": {"type": "uniform"}, "boundary": "step"})
    assert stepwise.status_code == 409 and body(stepwise)["extensions"]["code"] == "E003"
    assert client.request("PATCH", "/runs/lv/components", json={"slot": "X"}).status_code == 422


def test_patch_needs_the_write_scope_and_is_idempotent(live):
    client, _manager = live
    payload = {"slot": "TRAINER.WEIGHT_UPDATE", "spec": {"type": "uniform"}}
    assert client.request("PATCH", "/runs/lv/components", token="read-tok", json=payload).status_code == 403
    first = client.request("PATCH", "/runs/lv/components", json=payload, headers={"Idempotency-Key": "k1"})
    again = client.request("PATCH", "/runs/lv/components", json=payload, headers={"Idempotency-Key": "k1"})
    assert first.status_code == again.status_code == 202
    assert again.headers.get("Idempotent-Replay") == "true"
    other = client.request("PATCH", "/runs/lv/components",
                           json={**payload, "spec": {"type": "kappa_softmax"}}, headers={"Idempotency-Key": "k1"})
    assert other.status_code == 409


def test_mutations_simulate_then_queue_and_commit(live):
    client, manager = live
    plan = [{"op": "retire", "path": "p2", "reason": "diverged"}]
    dry = client.post("/runs/lv/mutations", json={"dry_run": True, "plan": plan})
    assert dry.status_code == 200, body(dry)
    report = body(dry)
    assert report["ok"] is True and report["changed"] is True and report["hard"] == []
    assert report["active_paths"] == ["p0", "p1"] and report["retired"] == ["p2"]
    # a dry run queues nothing
    trainer = manager.get_handle("lv").orchestrator.trainer
    assert trainer.engine.pending() == [] or all(p["plan"].get("label") != "prune" for p in trainer.engine.pending())

    bad = client.post("/runs/lv/mutations", json={"dry_run": True, "plan": [{"op": "retire", "path": "ghost"}]})
    assert bad.status_code == 422 and body(bad)["hard"][0]["code"] == "E081"
    refused = client.post("/runs/lv/mutations", json={"plan": [{"op": "retire", "path": "ghost"}]})
    assert refused.status_code == 422 and body(refused)["extensions"]["code"] == "E081"
    malformed = client.post("/runs/lv/mutations", json={"plan": [{"op": "explode", "path": "p1"}]})
    assert malformed.status_code == 422 and body(malformed)["extensions"]["code"] == "E080"

    queued = client.post("/runs/lv/mutations", json={"plan": plan, "label": "prune"})
    assert queued.status_code == 202
    assert body(queued)["label"] == "prune" and body(queued)["digest"]

    def retired():
        tree = body(client.get("/runs/lv/peers/tree"))
        return any(n["path"] == "p2" and n.get("status") != "active" for n in tree["nodes"])

    _wait(retired)
    assert trainer.mutation_journal[-1]["plan"]["label"] == "prune"


def test_status_table(live, client):
    """200, 202, 400, 404, 409, 422 are each reachable (a route returns 201 nowhere in this family)."""
    live_client, _manager = live
    assert live_client.get("/components").status_code == 200
    assert live_client.request(
        "PATCH", "/runs/lv/components", json={"slot": "TRAINER.WEIGHT_UPDATE", "spec": {"type": "uniform"}},
    ).status_code == 202
    assert live_client.post("/components/validate", body=b"{not json",
                            headers={"Content-Type": "application/json"}).status_code == 400
    assert live_client.get("/components/weight_update/zzz").status_code == 404
    assert live_client.request(
        "PATCH", "/runs/lv/components", json={"slot": "TRAINER.POLICY", "spec": {"type": "default"}},
    ).status_code == 409
    assert live_client.post("/components/validate", json={"kind": "weight_update", "spec": {}}).status_code == 422
    assert live_client.post("/components/validate", token=None,
                            json={"kind": "weight_update", "spec": {"type": "uniform"}}).status_code == 200


def test_a_run_that_is_not_live_is_a_409(client):
    created = client.post("/runs", json={"epochs": 2, "run_id": "done"})
    assert created.status_code == 202
    from api_support import wait_terminal

    assert wait_terminal(client, "done")["status"] == "completed"
    patched = client.request("PATCH", "/runs/done/components",
                             json={"slot": "TRAINER.WEIGHT_UPDATE", "spec": {"type": "uniform"}})
    assert patched.status_code == 409
    assert body(patched)["extensions"]["run_id"] == "done"
    mutated = client.post("/runs/done/mutations", json={"dry_run": True, "plan": [{"op": "retire", "path": "p1"}]})
    assert mutated.status_code == 409
    shown = body(client.get("/runs/done/components"))
    assert shown["source"] == "checkpoint" and shown["pending"] == {}
    assert client.request("PATCH", "/runs/ghost/components",
                          json={"slot": "A", "spec": {"type": "u"}}).status_code == 404


# -- the RunManager side both the API and the CLI use ----------------------------------------------
def test_manager_for_cli_returns_the_registered_manager(live):
    from mactf.orchestration import live as live_module

    _client, manager = live
    live_module.set_cli_manager(manager)
    try:
        assert live_module.manager_for_cli(manager.settings, manager.registry) is manager
    finally:
        live_module.set_cli_manager(None)
    fresh = live_module.manager_for_cli(manager.settings, manager.registry)
    assert fresh is not manager
    fresh.close()


def test_a_manager_without_the_run_refuses_with_the_right_error(live):
    from mactf.exceptions import StateConflict  # noqa: F401
    from mactf.orchestration.live import RunManager, RunNotFound, RunStateError

    _client, manager = live
    other = RunManager(manager.settings, registry=manager.registry)
    try:
        with pytest.raises(RunNotFound):
            other.components("ghost")
        with pytest.raises(RunNotFound):
            other.request_components("ghost", "TRAINER.WEIGHT_UPDATE", {"type": "uniform"})
        with pytest.raises(RunStateError):                       # the row exists, nobody drives it here
            other.request_components("lv", "TRAINER.WEIGHT_UPDATE", {"type": "uniform"})
        with pytest.raises(RunStateError):
            other.request_mutation("lv", [{"op": "retire", "path": "p2"}])
        with pytest.raises(RunStateError):
            other.simulate_mutation("lv", [{"op": "retire", "path": "p2"}])
    finally:
        other.close()


def test_slot_families_dispatch_to_their_plans(live):
    _client, manager = live
    trainer = manager.get_handle("lv").orchestrator.trainer
    queued = manager.request_components("lv", "TRAINER.WEIGHT_UPDATE_OVERRIDES.p*", {"type": "uniform"})
    assert queued["slot"] == "TRAINER.WEIGHT_UPDATE_OVERRIDES.p*"
    _wait(lambda: "TRAINER.WEIGHT_UPDATE_OVERRIDES.p*" in trainer.slots.paths())
    shown = manager.components("lv")
    assert _slot(shown, "TRAINER.WEIGHT_UPDATE_OVERRIDES.p*")["name"] == "uniform"
    # a patch on the override merges over the override's own spec
    manager.request_components("lv", "TRAINER.WEIGHT_UPDATE_OVERRIDES.p*", {"type": "kappa_softmax"})
    _wait(lambda: _slot(manager.components("lv"), "TRAINER.WEIGHT_UPDATE_OVERRIDES.p*")["name"] == "kappa_softmax")
    # a brand-new override with only params has nothing to patch
    from mactf.exceptions import ComponentSpecError

    with pytest.raises(ComponentSpecError):
        manager.request_components("lv", "TRAINER.WEIGHT_UPDATE_OVERRIDES.q*", {"params": {}})
    # an optimizer block, as an operator writes it
    queued = manager.request_components("lv", "OPTIMIZER", {"TYPE": "torch", "CLASS": "SGD", "LR": 0.01})
    assert queued["slot"] == "OPTIMIZER"
    assert "OPTIMIZER" in manager.components("lv")["pending"] or _optimizer_swapped(manager)
    _wait(lambda: _optimizer_swapped(manager))


def _optimizer_swapped(manager):
    shown = manager.components("lv")
    return any(row["slot"].startswith("OPTIMIZER") for row in shown["history"])
