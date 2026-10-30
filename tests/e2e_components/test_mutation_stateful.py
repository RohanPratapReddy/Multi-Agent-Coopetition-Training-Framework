"""Stateful fuzz of tree mutation: sequences of plans and epochs on one live trainer.

After every plan the simulator and the real engine must agree (same verdict, same error code, same
resulting structure digest); a refused plan leaves nothing behind; after every epoch the path state
audits clean and the parameters are finite. At the end the last checkpoint restores into a *fresh*
trainer with the same topology and bit-identical parameters.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from mactf.exceptions import MutationError
from mactf.training import MACTFTrainer
from mactf.training.mutation import MutationPlan
from mutation_support import Tiny, mutable_trainer, mutation_settings, parameters

PATHS = ["peer-1", "peer-2", "peer-3", "peer-4", "peer-5", "peer-1/a", "peer-1/b", "peer-2/a", "peer-4/a", "ghost"]


def _op():
    path = st.sampled_from(PATHS)
    flag = st.booleans()
    return st.one_of(
        st.builds(lambda p, like: {"op": "add", "path": p, "like": like}, path, st.sampled_from(["peer-1", "peer-2", "peer-3"])),
        st.builds(lambda p, r: {"op": "retire", "path": p, "recursive": r}, path, flag),
        st.builds(lambda p, r: {"op": "revive", "path": p, "recursive": r}, path, flag),
        st.builds(lambda p, r, f: {"op": "purge", "path": p, "recursive": r, "force": f}, path, flag, flag),
        st.builds(lambda p, m: {"op": "modify", "path": p, "set": {"metadata": {"k": m}}}, path, st.integers(0, 3)),
    )


STEP = st.one_of(st.just("epoch"), st.lists(_op(), min_size=1, max_size=3))


def _finite(trainer) -> bool:
    return all(bool(torch.isfinite(v).all()) for v in parameters(trainer).values())


@hyp_settings(max_examples=60, deadline=None, database=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(steps=st.lists(STEP, min_size=2, max_size=7))
def test_simulator_and_engine_agree_over_a_run(steps, settings, tmp_path_factory):
    mutation_settings(settings)
    base = tmp_path_factory.mktemp("stateful")
    trainer = mutable_trainer(settings, base, peers=4)
    epoch = 0
    for step in steps:
        if step == "epoch":
            trainer.run_epoch(epoch)
            epoch += 1
            assert trainer.path_state.audit() == []
            assert _finite(trainer)
            continue
        plan = MutationPlan.parse({"ops": step})
        simulation = trainer.engine.simulator().run(plan)
        before = (trainer.topology()["digest"], {k: v.clone() for k, v in parameters(trainer).items()}, len(trainer.mutation_journal))
        if simulation.hard:
            with pytest.raises(MutationError) as caught:
                trainer.mutate(plan)
            assert caught.value.code in {simulation.hard[0].code, "E088"}   # E088: the per-epoch plan budget is the engine's gate
            after = (trainer.topology()["digest"], parameters(trainer), len(trainer.mutation_journal))
            assert after[0] == before[0] and after[2] == before[2]
            assert all(torch.equal(after[1][k], before[1][k]) for k in before[1])
            continue
        try:
            report = trainer.mutate(plan)
        except MutationError as error:
            assert error.code in {"E085", "E087", "E088"}   # a layout or tie problem only a model shows; the epoch's plan budget
            assert trainer.topology()["digest"] == before[0] and len(trainer.mutation_journal) == before[2]
            continue
        assert report.after_digest == simulation.after_digest
        assert report.before_digest == simulation.before_digest
        assert trainer.topology()["digest"] == simulation.after_digest
        assert sorted(n.path for n in trainer.tree.walk("root-first")) == sorted(n.path for n in simulation.tree.walk("root-first"))
        assert sorted(trainer.freeze.retired()) == sorted(simulation.retired)
        assert trainer.path_state.audit() == []
    # the run stays trainable, and its last checkpoint restores into a fresh trainer
    trainer.run_epoch(epoch)
    assert _finite(trainer)
    record = trainer.checkpoint.latest()
    payload = trainer.checkpoint.load(record.path)
    final = parameters(trainer)
    digest = trainer.topology()["digest"]
    torch.manual_seed(0)
    fresh = MACTFTrainer(model=Tiny(), num_peers=4, settings=settings, checkpoint_dir=base / "fresh")
    fresh.run_id = trainer.run_id
    fresh.build()
    fresh.load_state_dict(payload)
    assert fresh.topology()["digest"] == digest
    restored = parameters(fresh)
    assert set(restored) == set(final)
    assert all(torch.equal(restored[k], final[k]) for k in final)
    assert fresh.path_state.audit() == []
