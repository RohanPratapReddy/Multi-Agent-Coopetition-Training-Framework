"""I1, default identity (document 17 section 19.4): the P0 gate.

The fixtures under ``fixtures/default_identity/`` were recorded by
``default_identity.py`` against the UNMODIFIED framework, before any production line of
document 17 existed (rule 2 of section 20.1). A default run - no ``COMPONENTS``
configuration, no registration - must reproduce them bitwise: parameters, loss, kappa,
entropy, weights, scorecards, signal, the RNG state after the run and the content hash of
the epoch-3 checkpoint.
"""

from __future__ import annotations

import json

import pytest

import default_identity as recorder


@pytest.fixture(scope="module")
def fixtures():
    return recorder.load_fixtures()


@pytest.fixture(scope="module")
def replay():
    """One replay of the scenario, with the component machinery watched.

    Section 19.4 condition 4: a default run does no new work on the hot path. No guard
    exists yet, so the watched calls are the registry's ``build`` and ``register``: a
    default run builds and registers nothing through the component model.
    """
    import torch

    from mactf.components.registry import ComponentRegistry

    calls = {"build": 0, "register": 0}
    originals = {
        "build": ComponentRegistry.build,
        "register": ComponentRegistry.register,
    }

    def counting(name):
        def wrapper(self, *args, **kwargs):
            calls[name] += 1
            return originals[name](self, *args, **kwargs)

        return wrapper

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ComponentRegistry, "build", counting("build"))
        patch.setattr(ComponentRegistry, "register", counting("register"))
        payload = recorder.record_all()
    payload["calls"] = calls
    assert torch.is_grad_enabled()
    return payload


@pytest.fixture(autouse=True)
def _same_torch(fixtures):
    import torch

    recorded = fixtures["run"]["torch"]
    current = torch.__version__.split("+")[0].rsplit(".", 1)[0]
    if recorded != current:
        pytest.skip(
            f"fixtures were recorded with torch {recorded}.x, this is {current}.x: bitwise "
            "equality across torch versions is not promised"
        )


def test_default_identity(replay, fixtures):
    problems = recorder.diff(replay, fixtures)
    assert problems == [], "\n".join(problems)


def test_a_default_run_uses_no_component_machinery(replay):
    assert replay["calls"] == {"build": 0, "register": 0}


def test_both_legs_are_compared(replay, fixtures):
    # the Orchestrator leg and the MACTFTrainer leg are both recorded and both replayed
    assert set(fixtures["scalars"]) == {"orchestrator", "trainer"}
    assert len(replay["orchestrator"]["epochs"]) == len(replay["trainer"]["epochs"]) == 3


#: The identity keys of a trainer checkpoint before P5 (the recorded set, 15 keys).
PRE_ENGINE_IDENTITY_KEYS = (
    "amp", "command_line", "created_at", "device", "dtype", "episode", "global_step", "hostname",
    "mactf_version", "mode", "python_version", "run_id", "schema_version", "score", "torch_version",
)


def test_the_engine_key_is_the_only_identity_change(replay, fixtures):
    """P5 (document 17 section 14.8): ``identity["engine"]`` is the one re-recorded fact.

    The fixture's key set was edited by exactly that key (see the P5 implementation notes); every
    section digest and the content hash in it are the ones recorded on the unmodified tree, so the
    sixteen-section comparison above is still a comparison against pre-change bytes.
    """
    keys = fixtures["scalars"]["trainer"]["checkpoint_identity_keys"]
    assert set(keys) - set(PRE_ENGINE_IDENTITY_KEYS) == {"engine"}
    assert set(PRE_ENGINE_IDENTITY_KEYS) <= set(keys)
    assert replay["trainer"]["checkpoint_identity_keys"] == keys
    sections = fixtures["scalars"]["trainer"]["checkpoint_sections"]
    assert replay["trainer"]["checkpoint_sections"] == sections
    assert replay["orchestrator"].get("checkpoint_identity_keys") is None   # a v1 file has no identity at all


def test_the_checkpoint_identity_has_no_components_key(replay, fixtures):
    """Section 14.3: a default run's checkpoint identity gains nothing."""
    keys = replay["trainer"]["checkpoint_identity_keys"]
    assert "components" not in keys
    assert keys == fixtures["scalars"]["trainer"]["checkpoint_identity_keys"]


def test_the_checkpoint_content_hash_is_the_recorded_one(replay, fixtures):
    assert replay["trainer"]["checkpoint_content_hash"] == fixtures["checkpoint"]
    assert fixtures["checkpoint"].startswith("sha256:")


def test_the_fixtures_are_not_vacuous(fixtures):
    """A fixture that compared nothing would pass for any code."""
    scalars = fixtures["scalars"]["orchestrator"]["epochs"]
    assert len(scalars) == 3
    assert all(0.0 < entry["entropy"] for entry in scalars)
    weights = [entry["weights"] for entry in scalars]
    assert all(abs(sum(w.values()) - 1.0) < 1e-9 for w in weights)
    assert any(key.endswith("params/peer-1") for key in fixtures["arrays"])
    cards = fixtures["scorecards"]["orchestrator"]
    assert len(cards) == 3 and all(card for card in cards)
    assert json.dumps(cards)  # canonical JSON round-trips


def test_a_perturbed_run_is_detected(replay, fixtures):
    """The comparison must fail when a bit moves, or the gate proves nothing."""
    import copy

    tampered = copy.deepcopy(replay)
    tampered["orchestrator"]["epochs"][1]["params"]["peer-1"][0] += 1e-12
    assert any("parameters of peer-1 differ" in line for line in recorder.diff(tampered, fixtures))
    tampered = copy.deepcopy(replay)
    tampered["trainer"]["epochs"][2]["kappa"] += 1e-12
    assert any("kappa" in line for line in recorder.diff(tampered, fixtures))
    tampered = copy.deepcopy(replay)
    tampered["trainer"]["checkpoint_content_hash"] = "sha256:00"
    assert any("checkpoint_content_hash" in line for line in recorder.diff(tampered, fixtures))
