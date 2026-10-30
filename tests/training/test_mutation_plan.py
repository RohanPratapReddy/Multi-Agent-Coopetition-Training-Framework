"""The pure half of tree mutation (document 17 section 12.4, 12.11.3): plans, specs, digests.

Nothing here builds a model; ``test_the_pure_half_imports_no_torch`` proves it can't.
"""

import subprocess
import sys

import pytest

from mactf.exceptions import MutationError
from mactf.training.mutation import (
    BUILD_TOKEN, MutationLimits, MutationPlan, Op, OpKind, PeerSpec, REFERENCE_TOKEN, add_peer, modify_peer,
    purge_peer, retire_peer, revive_peer,
)


def test_a_plan_round_trips_through_its_canonical_form():
    plan = MutationPlan.of(add_peer("peer-4", like="peer-1"), label="grow")
    again = MutationPlan.parse(plan.canonical())
    assert again.digest() == plan.digest()
    assert again.canonical() == plan.canonical()


def test_the_digest_ignores_key_order():
    a = MutationPlan.parse({"ops": [{"op": "add", "path": "p", "like": "q", "role": "worker"}]})
    b = MutationPlan.parse({"ops": [{"op": "add", "role": "worker", "like": "q", "path": "p"}]})
    assert a.digest() == b.digest()


def test_keys_are_accepted_in_any_case():
    upper = MutationPlan.parse({"OPS": [{"OP": "retire", "PATH": "p", "RECURSIVE": True}], "LABEL": "x"})
    lower = MutationPlan.parse({"ops": [{"op": "retire", "path": "p", "recursive": True}], "label": "x"})
    assert upper.digest() == lower.digest()


@pytest.mark.parametrize(
    "raw, code",
    [
        ({"ops": []}, "E080"),                                            # empty plan
        ({"ops": [{"op": "explode", "path": "p"}]}, "E080"),              # unknown verb
        ({"ops": [{"op": "add", "path": ""}]}, "E080"),                   # empty path
        ({"ops": [{"op": "add", "path": "p", "recursive": True}]}, "E080"),  # key not of `add`
        ({"ops": [{"op": "modify", "path": "p"}]}, "E080"),               # modify needs `set`
        ({"ops": [{"op": "retire", "path": "p", "force": True}]}, "E080"),# force is purge only
        ({"ops": [{"op": "add", "path": "p", "role": "Root"}]}, "E080"),  # role grammar
        ({"ops": [{"op": "add", "path": "p", "init_from": "@nowhere"}]}, "E080"),
        ({"ops": [{"op": "modify", "path": "p", "set": {"role": "x", "rebuild": {}}}]}, "E080"),
        ({"ops": [{"op": "add", "path": "p"}], "extra": 1}, "E080"),      # unknown plan key
        ({"ops": "retire p"}, "E080"),                                    # ops must be a list
        ({"ops": [{"op": "add", "path": "p", "like": "@reference"}]}, "E080"),
        ({"ops": [{"op": "add", "path": "p", "factory": "@elsewhere"}]}, "E080"),
        ({"ops": [{"op": "add", "path": "p", "factory": 7}]}, "E080"),   # not a name or a mapping
        ({"ops": [{"op": "add", "path": "p", "metadata": [1]}]}, "E080"),
        ({"ops": [{"op": "add", "path": "p", "metadata": {"x": float("nan")}}]}, "E080"),
        ({"ops": [{"op": "add", "path": "p", "optimizer": {}}]}, "E080"),
        ({"ops": [{"op": "add", "path": "p", "bogus": 1}]}, "E080"),
        ({"ops": [{"op": "modify", "path": "p", "set": {"nonsense": 1}}]}, "E080"),
        ({"ops": [{"op": "modify", "path": "p", "set": {"init_from": "x", "rebuild": {"like": "y"}}}]}, "E080"),
        ({"ops": [{"op": "modify", "path": "p", "set": {"rebuild": {"role": "x"}}}]}, "E080"),
        ({"ops": [{"op": "retire", "path": "p", "oops": 1}]}, "E080"),
        ({"ops": [42]}, "E080"),
    ],
)
def test_a_malformed_plan_names_its_code(raw, code):
    with pytest.raises(MutationError) as caught:
        MutationPlan.parse(raw)
    assert caught.value.code == code
    assert caught.value.findings == ()


def test_a_spec_that_names_the_wrong_path_is_refused():
    with pytest.raises(MutationError) as caught:
        PeerSpec.parse({"path": "peer-5"}, path="peer-4")
    assert "names" in str(caught.value) and caught.value.code == "E080"


def test_the_tokens_are_the_documented_ones():
    spec = PeerSpec.parse({"path": "p", "factory": BUILD_TOKEN, "init_from": REFERENCE_TOKEN})
    assert spec.factory == "@build" and spec.init_from == "@reference"
    modify = Op.parse({"op": "modify", "path": "p", "set": {"init_from": "@reference"}})
    assert modify.changes == {"init_from": "@reference"}


def test_a_plan_is_data_that_survives_json():
    import json

    plan = MutationPlan.of(
        add_peer("a/x", like="a", factory={"type": "perturbed_clone", "params": {"std": 0.1}},
                 metadata={"k": [1, 2]}, optimizer={"LR": 0.1}),
        modify_peer("a", metadata={"gone": None, "kept": 1}),
        retire_peer("b", recursive=True, reason="why"),
        revive_peer("b", recursive=True),
        purge_peer("c", recursive=True, force=True),
        label="mixed", reason="r",
    )
    again = MutationPlan.parse(json.loads(json.dumps(plan.canonical())))
    assert again.digest() == plan.digest()
    assert [op.kind for op in again.ops] == [OpKind.ADD, OpKind.MODIFY, OpKind.RETIRE, OpKind.REVIVE, OpKind.PURGE]
    assert again.paths() == ("a/x", "a", "b", "c")


def test_a_callable_factory_cannot_be_journaled():
    with pytest.raises(MutationError, match="register_peer_factory"):
        add_peer("p", factory=lambda spec: None)


def test_a_modify_may_not_change_the_path_or_carry_unknown_keys():
    with pytest.raises(MutationError):
        Op.parse({"op": "modify", "path": "p", "role": "x"})          # changes belong under `set`
    assert Op.parse({"op": "modify", "path": "p", "set": {"role": "x"}}).changes == {"role": "x"}


def test_removing_the_reference_is_not_expressible():
    """M5: the reference is not a node, so no operation can address it by its token."""
    for verb in ("retire", "revive", "purge"):
        op = Op.parse({"op": verb, "path": REFERENCE_TOKEN})
        assert op.path == REFERENCE_TOKEN       # parses as a path; the simulator refuses it (E081)


def test_dry_run_and_apply_delegate_to_the_trainer():
    class Fake:
        def mutate(self, plan, **kwargs):
            return (plan, kwargs)

    plan = MutationPlan.of(retire_peer("p"))
    assert plan.apply(Fake()) == (plan, {})
    assert plan.dry_run(Fake()) == (plan, {"dry_run": True})


def test_the_pure_half_imports_no_torch():
    code = (
        "import sys\n"
        "import mactf.training.mutation, mactf.training.mutation_reconcile, mactf.testing.tree\n"
        "assert 'torch' not in sys.modules, [m for m in sys.modules if m.startswith('torch')][:3]\n"
    )
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr


def test_the_root_label_matches_the_weight_plan():
    from mactf.components.weight_plan import ROOT_LABEL
    from mactf.training import mutation

    assert mutation.ROOT == ROOT_LABEL
    assert mutation.label_of(None) == ROOT_LABEL and mutation.label_of("a") == "a"
