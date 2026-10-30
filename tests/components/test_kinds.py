"""Kinds, origins, mutability, boundaries (document 17 section 3.2). Unit: no torch."""

from __future__ import annotations

import pytest

from mactf.components import (
    KIND_SPECS,
    RL_FAMILIES,
    Boundary,
    ComponentKind,
    Mutability,
    Origin,
    all_kind_ids,
    parse_kind_id,
)
from mactf.components.kinds import resolve_kind
from mactf.exceptions import ValidationError


def test_origin_ranks_order_who_may_replace_whom():
    ranks = [Origin.BUILTIN.rank, Origin.PLUGIN.rank, Origin.PROJECT.rank, Origin.RUNTIME.rank]
    assert ranks == [0, 1, 2, 3]
    assert [o.value for o in Origin] == ["builtin", "plugin", "project", "runtime"]


def test_mutability_and_boundary_share_one_ordering():
    assert [m.rank for m in (Mutability.STATIC, Mutability.EPISODE, Mutability.STEP)] == [0, 1, 2]
    assert [b.rank for b in (Boundary.RUN_START, Boundary.EPISODE, Boundary.STEP)] == [0, 1, 2]
    for mutability, boundary in zip(Mutability, (Boundary.RUN_START, Boundary.EPISODE, Boundary.STEP)):
        assert mutability.rank == boundary.rank


def test_twelve_rl_families_in_capability_order():
    assert RL_FAMILIES == (
        "value", "advantage", "loss", "rollup", "return", "reward",
        "regularization", "objective", "gradient", "metric", "action", "synthesis",
    )


def test_kind_ids_are_nine_plain_kinds_and_twelve_families():
    ids = all_kind_ids()
    assert len(ids) == 9 + 12 == len(set(ids))
    assert "rl_function" not in ids
    assert set(ids) == set(KIND_SPECS)
    assert all(f"rl_function.{family}" in ids for family in RL_FAMILIES)


@pytest.mark.parametrize("kind_id", all_kind_ids())
def test_parse_kind_id_inverts_kind_id(kind_id):
    kind, family = parse_kind_id(kind_id)
    assert kind.kind_id(family) == kind_id
    assert resolve_kind(kind_id) == kind_id


@pytest.mark.parametrize(
    "call",
    [
        lambda: ComponentKind.RL_FUNCTION.kind_id(),
        lambda: ComponentKind.RL_FUNCTION.kind_id("nonsense"),
        lambda: ComponentKind.SCORER.kind_id("loss"),
        lambda: parse_kind_id("nonsense"),
        lambda: parse_kind_id("rl_function"),
        lambda: parse_kind_id("rl_function.nonsense"),
        lambda: parse_kind_id("scorer.loss"),
        lambda: resolve_kind("nonsense"),
    ],
)
def test_bad_kinds_are_validation_errors(call):
    with pytest.raises(ValidationError):
        call()


def test_unknown_family_message_is_the_historical_one():
    with pytest.raises(ValidationError, match="unknown function family 'x'; accepted families"):
        ComponentKind.RL_FUNCTION.kind_id("x")


def test_resolve_kind_accepts_enum_name_and_full_id():
    assert resolve_kind(ComponentKind.SCORER) == "scorer"
    assert resolve_kind("scorer") == "scorer"
    assert resolve_kind("rl_function", "loss") == "rl_function.loss"
    assert resolve_kind(ComponentKind.RL_FUNCTION, "loss") == "rl_function.loss"
    assert resolve_kind("rl_function.loss") == "rl_function.loss"


#: Section 3.2's table of default mutability per kind.
DEFAULT_MUTABILITY = {
    "scorer": "episode", "reward_term": "episode",
    "rl_function.loss": "step", "rl_function.regularization": "step",
    "rl_function.gradient": "step", "rl_function.metric": "step",
    "rl_function.advantage": "episode", "rl_function.value": "episode",
    "rl_function.return": "episode", "rl_function.reward": "episode",
    "rl_function.rollup": "episode", "rl_function.objective": "episode",
    "rl_function.action": "episode", "rl_function.synthesis": "episode",
    "optimizer": "episode", "scheduler": "episode",
    "weight_update": "episode", "peer_strategy": "episode",
    "aggregator": "static", "trainer_policy": "static", "peer_factory": "episode",
}


def test_default_mutability_matches_the_section_3_2_table():
    assert set(DEFAULT_MUTABILITY) == set(all_kind_ids())
    for kind_id, expected in DEFAULT_MUTABILITY.items():
        assert KIND_SPECS[kind_id].default_mutability.value == expected, kind_id


def test_slot_root_names_for_locking():
    assert KIND_SPECS["scorer"].slot == "scorer"
    assert KIND_SPECS["rl_function.loss"].slot == "rl_function.loss"
    assert KIND_SPECS["weight_update"].slot == "weight_update"


def test_legacy_tables_are_named_for_the_kinds_that_have_one():
    assert KIND_SPECS["scorer"].legacy_table == "mactf.scoring.registry.SCORER_TYPES"
    assert KIND_SPECS["peer_strategy"].legacy_table.endswith("PEER_STRATEGIES")
    assert KIND_SPECS["weight_update"].legacy_table.endswith("WEIGHT_STRATEGIES")
    assert KIND_SPECS["rl_function.loss"].legacy_table.endswith("FUNCTION_TYPES['loss']")
    assert KIND_SPECS["optimizer"].legacy_table is None


def test_importing_the_package_pulls_in_no_heavy_module():
    import subprocess
    import sys

    code = (
        "import sys, mactf.components as c\n"
        "bad = sorted(m for m in ('torch','transformers','peft','scipy','numpy') if m in sys.modules)\n"
        "assert not bad, bad\n"
        "assert not any(m.startswith('mactf.scoring') or m.startswith('mactf.coopetition') "
        "or m.startswith('mactf.rl') or m.startswith('mactf.training') for m in sys.modules), "
        "[m for m in sys.modules if m.startswith('mactf.')]\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
