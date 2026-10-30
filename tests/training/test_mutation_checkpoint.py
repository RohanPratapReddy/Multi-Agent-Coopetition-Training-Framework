"""A mutated run in a checkpoint: journal, reconciliation, rollback (document 17 section 12.9, rows 51-54)."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mactf.conf import Settings
from mactf.exceptions import MutationError
from mactf.orchestration.checkpoint_v2 import CheckpointPayload
from mactf.training.mutation import MutationPlan, add_peer, purge_peer, retire_peer
from mactf.training.mutation_reconcile import MutationState, Target, plan_reconciliation
from mutation_support import Tiny, mutable_trainer, mutation_settings, parameters

HERE = Path(__file__).parent
SCHEDULE = [
    {"AT_EPOCH": 1, "LABEL": "grow", "OPS": [
        {"op": "add", "path": "peer-4", "like": "peer-1", "factory": {"type": "perturbed_clone", "params": {"std": 0.02}},
         "metadata": {"origin": "schedule"}},
        {"op": "retire", "path": "peer-3", "reason": "converged"},
    ]},
]
RUN_ID = "mut-run"


def _config(settings):
    mutation_settings(settings, SCHEDULE=copy.deepcopy(SCHEDULE))
    settings["TRAINING"]["EPOCHS"] = 4
    return settings


def _fresh(settings, tmp_path, *, seed=7):
    torch.manual_seed(seed)
    from mactf.training import MACTFTrainer

    trainer = MACTFTrainer(model=Tiny(), num_peers=3, settings=settings, checkpoint_dir=tmp_path / "ck")
    trainer.run_id = RUN_ID
    return trainer.build()


def _same_parameters(left, right):
    assert set(left) == set(right)
    for path in left:
        assert torch.equal(left[path], right[path]), path


def test_a_mutated_run_resumes_in_a_new_process(settings, tmp_path):
    """Resume in a *fresh interpreter* continues bit-identically to a run that never stopped (19.7.2)."""
    _config(settings)
    # the run that never stopped
    reference = _fresh(settings, tmp_path / "ref")
    reference.fit(epochs=4)
    # the run that stops after epoch 1 (the plan committed at the top of epoch 1 is in its checkpoint)
    first = _fresh(settings, tmp_path / "split")
    first.fit(epochs=2)
    assert first.mutation_journal and sorted(first.peers) == ["peer-1", "peer-2", "peer-3", "peer-4"]
    stored = first.checkpoint.load(first.checkpoint.latest().path)
    assert stored.identity["trainer"]["mutation"]["journal"][0]["source"] == "schedule"
    first.close()

    job = {
        "settings": settings.as_dict(), "seed": 7, "peers": 3, "run_id": RUN_ID, "epochs": 4,
        "checkpoints": str(tmp_path / "split" / "ck"), "out": str(tmp_path / "child.json"),
    }
    (tmp_path / "job.json").write_text(json.dumps(job, default=str))
    run = subprocess.run(
        [sys.executable, str(HERE / "mutation_resume_child.py"), str(tmp_path / "job.json")],
        capture_output=True, text=True, timeout=300, cwd=str(tmp_path),
    )
    assert run.returncode == 0, run.stderr[-3000:]
    child = json.loads((tmp_path / "child.json").read_text())

    # the new process built the *initial* tree and reproduced the stored one before loading a tensor
    assert child["peers_after_build"] == ["peer-1", "peer-2", "peer-3"] and child["journal_after_build"] == 0
    assert child["start_epoch"] == 2 and child["epochs_run"] == 2
    assert child["peers"] == ["peer-1", "peer-2", "peer-3", "peer-4"] and child["retired"] == ["peer-3"]
    assert child["audit"] == [] and child["pending"] == []
    assert [entry["source"] for entry in child["journal"]] == ["schedule"]       # the plan did not run twice
    assert child["journal"] == reference.mutation_journal
    assert child["topology"]["digest"] == reference.topology()["digest"]
    for path, values in child["params"].items():
        assert torch.equal(torch.tensor(values), reference.peers[path].export_parameters().detach()), path


def test_a_resumed_trainer_reproduces_the_structure_before_it_loads_weights(settings, tmp_path):
    _config(settings)
    first = _fresh(settings, tmp_path / "a")
    first.fit(epochs=2)
    first.close()
    payload = first.checkpoint.load(first.checkpoint.latest().path)
    second = _fresh(settings, tmp_path / "a", seed=99)             # a different init: the weights must come from the file
    assert sorted(second.peers) == ["peer-1", "peer-2", "peer-3"]
    report = second.load_state_dict(payload)
    assert sorted(second.peers) == ["peer-1", "peer-2", "peer-3", "peer-4"]
    assert second.freeze.retired() == frozenset({"peer-3"}) and second.tree["peer-3"].status == "pruned"
    assert second.tree["peer-4"].metadata == {"origin": "schedule"}
    assert second.topology_epoch == first.topology_epoch and second.mutation_journal == first.mutation_journal
    _same_parameters(parameters(second), parameters(first))
    assert any("topology reconciled" in note for note in report.notes)
    assert second.path_state.audit() == []
    # and it keeps training: the retired peer is still out, the new one is in
    second.run_epoch(2)
    assert "peer-4" in second.scorecards[-1].peers and "peer-3" not in second.scorecards[-1].peers


def test_a_rollback_past_a_purge_brings_the_peer_back(settings, tmp_path):
    """12.9.7 case 3: the live tree lost a branch; the snapshot's weights come back on the rebuilt node."""
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    trainer.mutate(MutationPlan.of(add_peer("peer-1/a", like="peer-1"), add_peer("peer-1/b", like="peer-1")))
    trainer.run_epoch(0)
    snapshot_params = parameters(trainer)
    snap = trainer.checkpoint.load(trainer.checkpoint.latest().path)
    trainer.run_epoch(1)                                            # the run moves on...
    trainer.mutate(MutationPlan.of(retire_peer("peer-1", recursive=True), purge_peer("peer-1", recursive=True)))
    assert "peer-1" not in trainer.tree and trainer.mutation_journal[-1]["plan"]["ops"][-1]["op"] == "purge"
    report = trainer.load_state_dict(snap)                          # ...and the watchdog's rollback restores a snapshot
    assert sorted(trainer.peers) == sorted(snapshot_params)
    assert any("reconciled" in note and "added 3" in note for note in report.notes)
    _same_parameters(parameters(trainer), snapshot_params)
    assert trainer.path_state.audit() == []
    assert trainer.mutation_journal == snap.identity["trainer"]["mutation"]["journal"]
    trainer.run_epoch(2)                                            # the restored branch trains


def test_a_rollback_past_an_add_a_retire_and_a_modify(settings, tmp_path):
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    trainer.run_epoch(0)
    base = trainer.checkpoint.load(trainer.checkpoint.latest().path)            # no mutation state at all
    base_params = parameters(trainer)
    trainer.mutate(MutationPlan.of(add_peer("peer-4", like="peer-1", metadata={"k": 1}), retire_peer("peer-3")))
    trainer.modify_peer("peer-2", role="lead", metadata={"m": 2})
    trainer.run_epoch(1)
    assert "mutation" in trainer.checkpoint.load(trainer.checkpoint.latest().path).identity.get("trainer", {})
    trainer.load_state_dict(base)                                               # case 8: undo every mutation
    assert sorted(trainer.peers) == ["peer-1", "peer-2", "peer-3"] and trainer.freeze.retired() == frozenset()
    assert trainer.tree["peer-2"].role == "worker" and trainer.tree["peer-2"].metadata == {}
    assert trainer.mutation_journal == [] and trainer.topology_epoch == 0 and trainer._tombstones == set()
    assert trainer._node_specs == {} and trainer._node_meta == {}
    _same_parameters(parameters(trainer), base_params)
    assert trainer.path_state.audit() == []


def test_a_tampered_journal_is_e089(settings, tmp_path):
    _config(settings)
    first = _fresh(settings, tmp_path / "a")
    first.fit(epochs=2)
    first.close()
    payload = first.checkpoint.load(first.checkpoint.latest().path)
    tree_digest = first.topology()["digest"]
    state = MutationState.parse(payload.identity["trainer"]["mutation"])
    assert state.verify_chain(tree_digest) == []

    bad = state.as_dict()
    bad["journal"][0]["after"] = "0" * 16
    problems = MutationState.parse(bad).verify_chain(tree_digest)
    assert problems and "journal ends at structure" in problems[-1] or any("starts from" in p or "ends at" in p for p in problems)
    bad = state.as_dict()
    bad["journal"][0]["plan"]["ops"].pop()                                       # the plan no longer hashes to its digest
    assert any("hashes to" in p for p in MutationState.parse(bad).verify_chain(tree_digest))
    bad = state.as_dict()
    bad["journal"][0]["seq"] = 5
    assert any("not contiguous" in p for p in MutationState.parse(bad).verify_chain(tree_digest))

    # strict: E089 and nothing changes
    second = _fresh(settings, tmp_path / "b", seed=99)
    tampered = copy.deepcopy(payload.as_dict())
    tampered["identity"]["trainer"]["mutation"]["journal"][0]["before"] = "f" * 16
    before = (sorted(second.peers), parameters(second))
    with pytest.raises(MutationError) as caught:
        second.load_state_dict(tampered)
    assert caught.value.code == "E089" and "STRICT_REPLAY" in str(caught.value)
    assert sorted(second.peers) == before[0] and second.mutation_journal == []
    _same_parameters(parameters(second), before[1])
    # STRICT_REPLAY: false tolerates it with W045 and trusts the tree section
    mutation_settings(settings, SCHEDULE=copy.deepcopy(SCHEDULE), STRICT_REPLAY=False)
    third = _fresh(settings, tmp_path / "c", seed=99)
    report = third.load_state_dict(tampered, strict=True)
    assert any("W045" in note for note in report.notes) and sorted(third.peers) == sorted(first.peers)


@pytest.mark.parametrize("damage", ["version", "unknown_key", "journal_not_a_list", "specs_not_a_mapping", "bad_epoch", "entry_missing"])
def test_a_malformed_mutation_state_is_e089(damage):
    state = MutationState(journal=(), topology_epoch=0).as_dict()
    if damage == "version":
        state["version"] = 99
    elif damage == "unknown_key":
        state["surprise"] = 1
    elif damage == "journal_not_a_list":
        state["journal"] = "plan"
    elif damage == "specs_not_a_mapping":
        state["specs"] = {"p": "not a mapping"}
    elif damage == "bad_epoch":
        state["topology_epoch"] = -1
    else:
        state["journal"] = [{"seq": 0}]
    with pytest.raises(MutationError) as caught:
        MutationState.parse(state)
    assert caught.value.code == "E089"


def test_a_journal_with_a_user_mutation_needs_mutation_enabled(settings, tmp_path):
    _config(settings)
    first = _fresh(settings, tmp_path / "a")
    first.fit(epochs=2)
    first.close()
    payload = first.checkpoint.load(first.checkpoint.latest().path)
    settings["TRAINER"]["MUTATION"] = {"ENABLED": False}
    second = _fresh(settings, tmp_path / "b", seed=99)
    with pytest.raises(MutationError) as caught:
        second.load_state_dict(payload)
    assert caught.value.code == "E080" and "ENABLED" in str(caught.value)
    assert sorted(second.peers) == ["peer-1", "peer-2", "peer-3"]


def test_an_unmutated_run_writes_no_trainer_identity(settings, tmp_path):
    """I1: ``_mutation_state() == {}`` and the identity section is the pre-mutation one (plus ``engine``)."""
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    trainer.run_epoch(0)
    assert trainer._mutation_state() == {}
    assert trainer._identity_extra() == {"engine": "trainer"}
    stored = trainer.checkpoint.load(trainer.checkpoint.latest().path)
    assert "trainer" not in stored.identity and "components" not in stored.identity
    assert set(stored.identity) == {
        "run_id", "episode", "global_step", "schema_version", "created_at", "torch_version", "python_version",
        "mactf_version", "hostname", "command_line", "device", "dtype", "amp", "mode", "score", "engine",
    }
    # `enabled` alone writes nothing either; the first plan is what adds the key
    trainer.add_peer("peer-4", like="peer-1")
    assert "mutation" in trainer._identity_extra()["trainer"]


def test_a_queued_plan_travels_in_the_checkpoint(settings, tmp_path):
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=3, build=False)
    trainer.run_id = RUN_ID
    trainer.build()
    trainer.request_mutation(MutationPlan.of(add_peer("peer-4", like="peer-1")), source="callback")
    state = trainer._mutation_state()
    assert state["pending"][0]["source"] == "callback"
    fresh = mutable_trainer(settings, tmp_path / "b", peers=3, seed=4, build=False)
    fresh.run_id = RUN_ID
    fresh.build()
    fresh.run_epoch(0)
    payload = trainer.state_dict()
    fresh.load_state_dict(payload)
    assert [row["source"] for row in fresh.engine.pending()] == ["callback"]
    fresh.run_epoch(1)                                                          # case 14: it runs at the top of the resumed epoch
    assert "peer-4" in fresh.peers and fresh.mutation_journal[-1]["source"] == "callback"


def test_the_planner_is_a_pure_function(settings, tmp_path):
    """plan_reconciliation: the operations, in an order every one of them is valid at its turn."""
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    trainer.mutate(MutationPlan.of(add_peer("peer-4", like="peer-1"), retire_peer("peer-3")))
    target_state = trainer.tree.state_dict()
    # from a fresh three-peer trainer's point of view
    other = mutable_trainer(settings, tmp_path / "o", peers=3)
    target = Target.build(target_state, MutationState.parse(trainer._mutation_state()), other._build_meta)
    plan = plan_reconciliation(other.tree, other.freeze.retired(), other._node_specs, target)
    assert [op.kind.value for op in plan.plan.ops] == ["add", "retire"]
    assert plan.added == ("peer-4",) and plan.retired == ("peer-3",) and plan.summary() == "added 1, retired 1"
    # the other direction
    back = Target.build(other.tree.state_dict(), None, other._build_meta)
    undo = plan_reconciliation(trainer.tree, trainer.freeze.retired(), trainer._node_specs, back)
    assert [op.kind.value for op in undo.plan.ops] == ["revive", "purge"]
    assert plan_reconciliation(other.tree, (), {}, back).empty and plan_reconciliation(other.tree, (), {}, back).summary() == "nothing"
