"""``test_guard_property_mutation_plans`` (document 17 section 19.7.1): the simulator and the engine agree."""

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from mactf.exceptions import MutationError
from mactf.training.mutation import MutationPlan
from mutation_support import mutable_trainer, mutation_settings, snapshot

PATHS = ["peer-1", "peer-2", "peer-3", "peer-4", "peer-1/a", "peer-1/b", "peer-2/a", "ghost", "peer-9/x"]


def op_strategy():
    path = st.sampled_from(PATHS)
    flags = st.booleans()
    return st.one_of(
        st.builds(lambda p, like: {"op": "add", "path": p, **({"like": like} if like else {})},
                  path, st.sampled_from([None, "peer-1", "peer-2", "ghost"])),
        st.builds(lambda p, r: {"op": "retire", "path": p, "recursive": r}, path, flags),
        st.builds(lambda p, r: {"op": "revive", "path": p, "recursive": r}, path, flags),
        st.builds(lambda p, r, f: {"op": "purge", "path": p, "recursive": r, "force": f}, path, flags, flags),
        st.builds(lambda p, m: {"op": "modify", "path": p, "set": {"metadata": {"k": m}}}, path, st.integers(0, 3)),
    )


@hyp_settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ops=st.lists(op_strategy(), min_size=1, max_size=5))
def test_guard_property_mutation_plans(ops, settings, tmp_path_factory):
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path_factory.mktemp("prop"), peers=4)
    plan = MutationPlan.parse({"ops": ops})
    simulation = trainer.engine.simulator().run(plan)
    before = snapshot(trainer)
    if simulation.hard:
        with pytest.raises(MutationError) as caught:
            trainer.mutate(plan)
        assert caught.value.code == simulation.hard[0].code           # the simulator names the code the engine raises
        assert snapshot(trainer) == before
    else:
        try:
            trainer.mutate(plan)                                       # a plan the simulator accepts commits...
        except MutationError as error:
            # ...unless staging finds what only a model can (layouts, ties, factories): never a half run
            assert error.code in {"E085", "E087"} and snapshot(trainer) == before
        else:
            assert trainer.path_state.audit() == []
            assert sorted(n.path for n in trainer.tree.walk("root-first")) == sorted(
                n.path for n in simulation.tree.walk("root-first")
            )
