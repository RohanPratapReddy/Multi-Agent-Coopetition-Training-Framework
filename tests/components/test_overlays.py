"""``Hub``, overlays and per-run activation (document 17 section 3.5)."""

from __future__ import annotations

import asyncio
import contextvars
import threading

import pytest

from component_helpers import OtherPolicy, Policy
from mactf.components import HUB, Hub, HubPolicy, Origin, overlay
from mactf.exceptions import (
    ComponentCollision,
    ComponentLocked,
    ComponentMissing,
    ValidationError,
)


@pytest.fixture
def policies(hub):
    return hub.registry("trainer_policy")


def test_an_overlay_registration_is_visible_inside_and_gone_outside(hub, policies):
    policies.register("base", Policy)
    with hub.scoped("run-1"):
        policies.register("mine", OtherPolicy, scope="run")
        assert policies.names() == ["base", "mine"]
        assert policies.get("mine").factory is OtherPolicy
        assert policies._entries.keys() == {"base"}  # the base layer was not touched
    assert policies.names() == ["base"]
    with pytest.raises(ComponentMissing):
        policies.get("mine")


def test_scope_run_without_an_active_overlay_is_refused(policies):
    with pytest.raises(ComponentLocked, match="no run scope is active"):
        policies.register("x", Policy, scope="run")
    with pytest.raises(ComponentLocked, match="no run scope is active"):
        policies.unregister("x", scope="run")


def test_the_innermost_overlay_wins(hub, policies):
    policies.register("p", Policy)
    with hub.scoped("outer"):
        policies.register("p", OtherPolicy, scope="run", replace=True)
        with hub.scoped("inner"):
            class Inner(Policy):
                pass

            policies.register("p", Inner, scope="run", replace=True)
            assert policies.get("p").factory is Inner
        assert policies.get("p").factory is OtherPolicy
    assert policies.get("p").factory is Policy


def test_hiding_a_name_in_an_overlay_leaves_the_base_alone(hub, policies):
    policies.register("p", Policy, origin=Origin.RUNTIME)
    with hub.scoped():
        removed = policies.unregister("p", scope="run")
        assert removed.name == "p"
        assert not policies.has("p") and policies.names() == []
        with pytest.raises(ComponentMissing):
            policies.get("p")
        # registering it again inside the scope un-hides it
        policies.register("p", OtherPolicy, scope="run")
        assert policies.get("p").factory is OtherPolicy
    assert policies.get("p").factory is Policy


def test_registering_over_a_visible_base_name_in_an_overlay_needs_replace(hub, policies):
    policies.register("p", Policy)
    with hub.scoped():
        with pytest.raises(ComponentCollision):
            policies.register("p", OtherPolicy, scope="run")


def test_an_overlay_may_not_hide_a_builtin_without_the_flag(hub, policies):
    policies.register_builtin("shipped", Policy)
    with hub.scoped():
        with pytest.raises(ComponentLocked, match="ALLOW_BUILTIN_OVERRIDE"):
            policies.unregister("shipped", scope="run")
    with hub.scoped(policy=HubPolicy(allow_builtin_override=True)):
        policies.unregister("shipped", scope="run")
        assert not policies.has("shipped")
    assert policies.has("shipped")


def test_a_frozen_registry_still_takes_overlay_writes(hub, policies):
    policies.register("p", Policy)
    policies.freeze()
    with hub.scoped():
        policies.register("q", Policy, scope="run")
        assert policies.names() == ["p", "q"]
        policies.unregister("p", scope="run")
        assert policies.names() == ["q"]
    assert policies.names() == ["p"]


def test_a_scoped_policy_applies_only_inside(hub, policies):
    assert hub.effective_policy() == HubPolicy()
    with hub.scoped(policy=HubPolicy(locked=frozenset({"trainer_policy:no*"}))):
        with pytest.raises(ComponentLocked):
            policies.register("nope", Policy, scope="run")
        policies.register("fine", Policy, scope="run")
    assert hub.effective_policy() == HubPolicy()


def test_current_overlay_and_active_overlays(hub):
    assert hub.current_overlay() is None and hub.active_overlays() == ()
    with hub.scoped("a") as outer:
        assert hub.current_overlay() is outer
        with hub.scoped("b") as inner:
            assert hub.active_overlays() == (outer, inner) and hub.current_overlay() is inner
        assert hub.current_overlay() is outer
    assert hub.current_overlay() is None
    assert outer.label == "a" and outer.is_empty()


def test_the_overlay_is_popped_even_when_the_body_raises(hub, policies):
    with pytest.raises(RuntimeError):
        with hub.scoped():
            policies.register("p", Policy, scope="run")
            raise RuntimeError("boom")
    assert hub.active_overlays() == () and policies.names() == []


def test_a_plain_thread_starts_with_an_empty_context_and_copy_context_carries_it(hub, policies):
    seen: dict[str, list[str]] = {}

    def worker(label):
        seen[label] = policies.names()

    with hub.scoped():
        policies.register("mine", Policy, scope="run")
        plain = threading.Thread(target=worker, args=("plain",))
        carried = threading.Thread(
            target=contextvars.copy_context().run, args=(worker, "carried")
        )
        for thread in (plain, carried):
            thread.start()
            thread.join()
    assert seen == {"plain": [], "carried": ["mine"]}


def test_activate_re_enters_a_run_overlay_in_a_worker_thread(hub, policies):
    seen = {}
    with hub.scoped() as run_overlay:
        policies.register("mine", Policy, scope="run")

    def worker():
        token = hub.activate(run_overlay)
        try:
            seen["inside"] = policies.names()
        finally:
            hub.deactivate(token)
        seen["after"] = policies.names()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen == {"inside": ["mine"], "after": []}


def test_asyncio_tasks_inherit_the_overlay_of_their_creator(hub, policies):
    async def main():
        with hub.scoped():
            policies.register("t", Policy, scope="run")

            async def child():
                return policies.names()

            return await asyncio.gather(asyncio.create_task(child()), child())

    assert asyncio.run(main()) == [["t"], ["t"]]


def test_overlays_in_two_threads_are_independent(hub, policies):
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def run(label, cls):
        with hub.scoped(label):
            policies.register("loss", cls, scope="run")
            barrier.wait(timeout=10)  # both overlays are populated at this moment
            results[label] = policies.get("loss").qualname
            barrier.wait(timeout=10)

    threads = [
        threading.Thread(target=run, args=("a", Policy)),
        threading.Thread(target=run, args=("b", OtherPolicy)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {"a": "Policy", "b": "OtherPolicy"}
    assert policies.names() == []


def test_default_origin_is_project_and_loading_scopes_another(hub, policies):
    assert hub.default_origin() is Origin.PROJECT
    with hub.loading(Origin.PLUGIN):
        assert hub.default_origin() is Origin.PLUGIN
        assert policies.register("a", Policy).origin is Origin.PLUGIN
        with hub.loading(Origin.RUNTIME):
            assert policies.register("b", Policy).origin is Origin.RUNTIME
        assert hub.default_origin() is Origin.PLUGIN
    assert policies.register("c", Policy).origin is Origin.PROJECT
    assert policies.register("d", Policy, origin=Origin.RUNTIME).origin is Origin.RUNTIME


def test_unknown_kind_is_a_validation_error(hub):
    with pytest.raises(ValidationError, match="unknown component kind 'nope'"):
        hub.registry("nope")


def test_hub_kinds_snapshot_and_freeze_all(hub):
    assert len(hub.kinds()) == 21
    hub.registry("trainer_policy").register("p", Policy)
    hub.registry("trainer_policy").freeze()
    hub.freeze_all()
    assert all(hub.registry(kind).frozen for kind in hub.kinds())
    hub.unfreeze_all()
    assert not any(hub.registry(kind).frozen for kind in hub.kinds())


def test_snapshot_lists_every_visible_component(hub):
    hub.registry("trainer_policy").register("p", Policy)
    snapshot = hub.snapshot()
    assert snapshot["trainer_policy"]["p"]["class"] == "Policy"
    assert set(snapshot) == set(hub.kinds())
    with hub.scoped():
        hub.registry("trainer_policy").register("q", Policy, scope="run")
        assert set(hub.snapshot()["trainer_policy"]) == {"p", "q"}
    assert set(hub.snapshot()["trainer_policy"]) == {"p"}


def test_the_public_overlay_helper_is_the_process_hub_scope():
    with overlay("x") as scope:
        assert HUB.current_overlay() is scope and scope.label == "x"
    assert HUB.current_overlay() is None
    assert isinstance(HUB, Hub)


def test_the_component_overlay_fixture_checks_for_leaks(component_overlay):
    HUB.registry("trainer_policy").register("fixture_probe", Policy, scope="run")
    assert HUB.registry("trainer_policy").has("fixture_probe")
