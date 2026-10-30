"""Per-run isolation (I9) and built-in immutability (I10), document 17 section 19.3.3 rows 16-17."""

from __future__ import annotations

import contextvars
import logging
import threading

import pytest

from component_helpers import OtherPolicy, Policy
from mactf.components import HUB, ComponentResolver, ComponentSpec, HubPolicy, Origin, overlay
from mactf.exceptions import ComponentCollision, ComponentLocked


def _loss(tag: str):
    """A loss class whose ``compute`` returns ``tag``: what each run registers as ``huber``."""

    class Loss:
        required_fields = frozenset({"policy"})

        def __init__(self, *, scale: float = 1.0) -> None:
            self.scale = scale

        def compute(self, batch, **kwargs):
            return tag

    Loss.__name__ = Loss.__qualname__ = f"Loss_{tag}"
    return Loss


def test_overlay_isolated_between_runs():
    """Two runs in one process each register a different ``loss``; neither sees the other's.

    The runs are real threads started the way ``RunManager._run`` starts them (a copied
    context), registering into the real process hub with ``scope="run"``. Both are inside
    their overlay at the same moment (a barrier), build the component through the
    resolver, and the process registry is identical before and after (I9).
    """
    registry = HUB.registry("rl_function.loss")
    before = HUB.snapshot()
    barrier = threading.Barrier(2)
    seen: dict[str, dict] = {}
    errors: list[BaseException] = []

    def run(tag: str) -> None:
        try:
            with overlay(f"run-{tag}"):
                registry.register("huber", _loss(tag), scope="run")
                barrier.wait(timeout=20)
                resolved = ComponentResolver().resolve(
                    ComponentSpec.parse("rl_function.loss", {"type": "huber", "scale": 2.0}),
                    trusted=True,
                )
                seen[tag] = {
                    "computes": resolved.instance.compute({}),
                    "cls": registry.get("huber").qualname,
                    "visible": "huber" in registry.names(),
                    "scale": resolved.instance.scale,
                }
                barrier.wait(timeout=20)  # neither run leaves before the other has looked
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
            barrier.abort()

    threads = [
        threading.Thread(target=contextvars.copy_context().run, args=(run, tag))
        for tag in ("alpha", "beta")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert seen["alpha"]["computes"] == "alpha" and seen["beta"]["computes"] == "beta"
    assert seen["alpha"]["cls"] == "Loss_alpha" and seen["beta"]["cls"] == "Loss_beta"
    assert seen["alpha"]["visible"] and seen["beta"]["visible"]
    assert seen["alpha"]["scale"] == seen["beta"]["scale"] == 2.0
    # the main thread never saw either, and the process registry is unchanged
    assert "huber" not in registry.names()
    assert HUB.snapshot() == before
    from mactf.rl.functions.registry import FUNCTION_TYPES

    assert "huber" not in FUNCTION_TYPES["loss"]  # the legacy view shows the base layer only


def test_a_run_overlay_never_reaches_the_legacy_table():
    from mactf.rl.functions.registry import ROLLUP_TYPES

    class Rollup:
        def rollup(self, parent, children, *, weights=None, scores=None):
            return parent

    names_before = sorted(ROLLUP_TYPES)
    with overlay():
        HUB.registry("rl_function.rollup").register("zz_private", Rollup, scope="run")
        assert "zz_private" in HUB.registry("rl_function.rollup").names()
        assert "zz_private" not in ROLLUP_TYPES
    assert sorted(ROLLUP_TYPES) == names_before


# -- I10 -------------------------------------------------------------------------------
def _builtin_hub(hub):
    registry = hub.registry("trainer_policy")
    registry.register_builtin("shipped", Policy)
    return registry


def test_builtin_replacement_needs_the_flag_and_logs_w001(hub, policy_for, caplog):
    registry = _builtin_hub(hub)
    # without the flag: refused, nothing changes, nothing logged as W001
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        with pytest.raises(ComponentCollision, match="needs COMPONENTS.ALLOW_BUILTIN_OVERRIDE"):
            registry.register("shipped", OtherPolicy, replace=True)
    assert registry.get("shipped").factory is Policy
    assert registry.get("shipped").origin is Origin.BUILTIN
    assert not [r for r in caplog.records if r.getMessage().startswith("W001")]
    # replace=False on a taken built-in is a plain collision, flag or no flag
    hub.configure(policy_for(allow_builtin_override=True))
    with pytest.raises(ComponentCollision, match="already registered"):
        registry.register("shipped", OtherPolicy)
    # with the flag: succeeds, is recorded as an override, and logs W001 exactly once
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        component = registry.register("shipped", OtherPolicy, replace=True)
    w001 = [r for r in caplog.records if r.getMessage().startswith("W001")]
    assert len(w001) == 1 and "trainer_policy:shipped" in w001[0].getMessage()
    assert w001[0].levelno == logging.WARNING
    assert component.overrides_builtin and registry.get("shipped").factory is OtherPolicy
    assert registry.get("shipped").fingerprint()["overrides_builtin"] is True


def test_builtin_replacement_on_the_real_registries_through_a_run_scope(caplog):
    """The same rule on a real registry (``peer_strategy:independent``), inside a run scope."""
    from mactf.coopetition.strategy import PEER_STRATEGIES, IndependentStrategy

    class Mine(IndependentStrategy):
        name = "independent"

    registry = HUB.registry("peer_strategy")
    with overlay():
        with pytest.raises(ComponentCollision, match="ALLOW_BUILTIN_OVERRIDE"):
            registry.register("independent", Mine, replace=True, scope="run")
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        with overlay(policy=HubPolicy(allow_builtin_override=True)):
            registry.register("independent", Mine, replace=True, scope="run")
            assert registry.get("independent").factory is Mine
            assert registry.get("independent").overrides_builtin
    assert any(r.getMessage().startswith("W001 peer_strategy:independent") for r in caplog.records)
    assert registry.get("independent").factory is IndependentStrategy
    assert PEER_STRATEGIES["independent"] is IndependentStrategy


def test_builtin_removal_needs_the_flag_too(hub, policy_for):
    registry = _builtin_hub(hub)
    with pytest.raises(ComponentLocked, match="removing it needs COMPONENTS.ALLOW_BUILTIN_OVERRIDE"):
        registry.unregister("shipped")
    assert registry.has("shipped")
    hub.configure(policy_for(allow_builtin_override=True))
    registry.unregister("shipped")
    assert not registry.has("shipped")
