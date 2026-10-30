"""``mactf.components.known``: static names for the torch-free settings validators (section 15.4)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from mactf.components import known
from mactf.components.known import (
    BUILTIN_NAMES,
    component_ref_problem,
    is_dotted_name,
    known_names,
    may_be_extended,
    spec_shape_problem,
)


def test_the_table_matches_the_live_registries():
    """``known.BUILTIN_NAMES[kind] == frozenset(legacy table)`` for every kind it lists."""
    from mactf.coopetition.strategy import PEER_STRATEGIES
    from mactf.coopetition.weights import WEIGHT_STRATEGIES
    from mactf.scoring.registry import SCORER_TYPES
    from mactf.components import HUB

    live = {"scorer": SCORER_TYPES, "peer_strategy": PEER_STRATEGIES, "weight_update": WEIGHT_STRATEGIES}
    # ``reward_term`` has no legacy dict (document 17 section 6.3): its registry is the table.
    live["reward_term"] = {name: None for name in HUB.registry("reward_term").names()}
    # ``trainer_policy`` joins with document 17 section 7: ``default`` and ``co_distill``.
    import mactf.training.policy  # noqa: F401  (registers the built-in policies)

    live["trainer_policy"] = {name: None for name in HUB.registry("trainer_policy").names()}
    # ``aggregator`` joins with document 17 section 10.7: the one built-in, ``default``.
    import mactf.coopetition.aggregation  # noqa: F401  (registers it)

    live["aggregator"] = {name: None for name in HUB.registry("aggregator").names()}
    assert set(BUILTIN_NAMES) == set(live)
    for kind_id, table in live.items():
        shipped = {name for name, cls in table.items()
                   if __import__("mactf.components", fromlist=["HUB"]).HUB.registry(kind_id).get(name).origin.value == "builtin"}
        assert BUILTIN_NAMES[kind_id] == frozenset(shipped), kind_id


def test_conf_mirrors_are_views_of_the_table():
    from mactf import conf

    assert conf._PEER_STRATEGY_NAMES is BUILTIN_NAMES["peer_strategy"]
    assert conf._WEIGHT_STRATEGY_NAMES is BUILTIN_NAMES["weight_update"]


def test_every_kind_is_covered_or_explicitly_none():
    for kind_id in ("scorer", "peer_strategy", "weight_update", "reward_term", "trainer_policy"):
        assert known_names(kind_id) is not None
    for kind_id in ("rl_function.loss", "optimizer", "scheduler"):
        assert known_names(kind_id) is None


def test_lowercase_kinds_match_what_the_registry_enforces():
    from mactf.components.registry import _LOWERCASE_KINDS

    assert known.LOWERCASE_KINDS == _LOWERCASE_KINDS


def test_known_names_is_the_union_with_the_registry(component_overlay):
    from mactf.components import HUB

    assert known_names("peer_strategy") == BUILTIN_NAMES["peer_strategy"]
    # a global registration is visible; an overlay one is too (names() sees overlays)
    from mactf.coopetition.strategy import IndependentStrategy

    class Mine(IndependentStrategy):
        name = "zz_known_probe"

    HUB.registry("peer_strategy").register("zz_known_probe", Mine, scope="run")
    assert "zz_known_probe" in known_names("peer_strategy")


def test_reading_names_never_imports_the_builtins_of_an_untouched_registry():
    code = (
        "import sys\n"
        "from mactf.components import known\n"
        "assert known.known_names('weight_update') == known.BUILTIN_NAMES['weight_update']\n"
        "assert known.registry_names('scorer') == frozenset()\n"
        "assert 'torch' not in sys.modules\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize(
    "raw, problem",
    [
        (None, None), ("x", None), ({"type": "x"}, None), ({"name": "x"}, None),
        ({"import": "a.b:C"}, None), ({"type": "x", "params": {}}, None),
        ("", "an empty string is not a component name"),
        ("  ", "an empty string is not a component name"),
        ({}, "needs 'type'"), ({"params": {}}, "needs 'type'"),
        ({"type": "x", "params": 3}, "'params' must be a mapping"),
        (3, "not int"), (["x"], "not list"),
    ],
)
def test_a_mapping_spec_needs_type_or_import(raw, problem):
    found = spec_shape_problem(raw)
    assert (found is None) if problem is None else (problem in found)


class _Settings(dict):
    pass


@pytest.fixture
def no_plugins(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None, **k: [])


@pytest.mark.parametrize("name", ["kappa_softmax", "UNIFORM", " none ", "Explicit"])
def test_a_shipped_name_is_accepted_for_weight_update(name, no_plugins):
    assert component_ref_problem("W", name, kind_id="weight_update", settings=_Settings()) is None


def test_an_unknown_name_is_refused_without_modules(no_plugins):
    problem = component_ref_problem("TRAINER.WEIGHT_UPDATE", "trimmed_mean", kind_id="weight_update",
                                    settings=_Settings())
    assert problem == "TRAINER.WEIGHT_UPDATE must be one of explicit, kappa_softmax, none, reference_only, score_weighted, uniform"


def test_an_unknown_name_is_deferred_with_modules(no_plugins):
    settings = _Settings(COMPONENTS={"MODULES": ["acme.strategies"]})
    assert component_ref_problem("W", "trimmed_mean", kind_id="weight_update", settings=settings) is None


def test_an_unknown_name_is_deferred_with_an_entry_point(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None, **k: [object()])
    assert component_ref_problem("W", "trimmed_mean", kind_id="weight_update", settings=_Settings()) is None
    # ... unless plugins are switched off
    off = _Settings(COMPONENTS={"PLUGINS": "off"})
    assert component_ref_problem("W", "trimmed_mean", kind_id="weight_update", settings=off) is not None
    none = _Settings(COMPONENTS={"PLUGINS": []})
    assert component_ref_problem("W", "trimmed_mean", kind_id="weight_update", settings=none) is not None


def test_a_scorer_registered_by_a_plugin_is_accepted(component_overlay, no_plugins):
    from mactf.components import HUB
    from mactf.scoring.reward_scorer import RewardScorer

    class Stage(RewardScorer):
        pass

    assert component_ref_problem("S", "zz_stage", kind_id="scorer", settings=_Settings()) is not None
    HUB.registry("scorer").register("zz_stage", Stage, scope="run")
    assert component_ref_problem("S", "zz_stage", kind_id="scorer", settings=_Settings()) is None


def test_mapping_specs_check_the_type_and_skip_import_only(no_plugins):
    s = _Settings()
    assert component_ref_problem("W", {"type": "uniform"}, kind_id="weight_update", settings=s) is None
    assert component_ref_problem("W", {"type": "nope"}, kind_id="weight_update", settings=s) \
        .startswith("W must be one of")
    assert component_ref_problem("W", {"import": "a.b:C"}, kind_id="weight_update", settings=s) is None
    assert "needs 'type'" in component_ref_problem("W", {}, kind_id="weight_update", settings=s)
    assert component_ref_problem("W", {"type": " "}, kind_id="weight_update", settings=s) \
        .endswith("an empty component name")


def test_a_kind_that_is_not_name_checked_accepts_any_shape_valid_reference(no_plugins):
    assert component_ref_problem("O", "whatever", kind_id="optimizer", settings=_Settings()) is None
    assert component_ref_problem("O", 3, kind_id="optimizer", settings=_Settings()) is not None


@pytest.mark.parametrize(
    "text, ok",
    [("acme", True), ("acme.components", True), ("_a.b_2.c3", True), ("./components.py", False),
     ("acme/scoring", False), ("acme.", False), ("acme..x", False), (".acme", False), ("1acme", False),
     ("acme-scoring", False), ("", False), (None, False), (3, False), ("acme.scoring.py", True)],
)
def test_dotted_names(text, ok):
    assert is_dotted_name(text) is ok


def test_may_be_extended_forms(no_plugins):
    assert may_be_extended({"MODULES": ["x"]}) is True
    assert may_be_extended({"MODULES": []}) is False
    assert may_be_extended(None) is False and may_be_extended({}) is False
    assert may_be_extended({"MODULES": ["x"], "PLUGINS": "off"}) is True


def test_every_persistence_code_the_restorer_emits_is_in_the_table():
    """Section 14.11.1: the codes the restorer can put on a finding are exactly the documented ones."""
    import re
    from pathlib import Path

    from mactf.components import restore
    from mactf.components.known import PERSISTENCE_CODES

    emitted = set(re.findall(r'"([EW]\d{3})"', Path(restore.__file__).read_text()))
    emitted |= set(re.findall(r"\b([EW]053)\b", Path(restore.__file__).read_text()))
    assert emitted == set(PERSISTENCE_CODES), emitted ^ set(PERSISTENCE_CODES)
