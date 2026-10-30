"""Teachers: loaders, freezing (I2), digest, caching, output normalisation."""

from __future__ import annotations

import pytest
import torch

from mactf.distill.conformance import TrustProfile
from mactf.distill.teacher import (
    CallableModule,
    Teacher,
    TeacherOutput,
    TeacherSpec,
    build_teacher,
    normalize_output,
)
from mactf.exceptions import ContractError, ImproperlyConfigured, ValidationError


def test_spec_validation():
    with pytest.raises(ValidationError, match="alphanumeric"):
        TeacherSpec(name="bad name", factory="torch.nn.Linear")
    with pytest.raises(ValidationError, match="unknown kind"):
        TeacherSpec(name="t", kind="zip", factory="x")
    with pytest.raises(ValidationError, match="requires 'factory'"):
        TeacherSpec(name="t", kind="module")
    with pytest.raises(ValidationError, match="'path' and 'arch'"):
        TeacherSpec(name="t", kind="state_dict", path="x")
    with pytest.raises(ValidationError, match="pretrained"):
        TeacherSpec(name="t", kind="transformers")
    with pytest.raises(ValidationError, match="weight"):
        TeacherSpec(name="t", factory="x", weight=0.0)


def test_spec_resolves_an_undeclared_trust_profile_fail_closed():
    spec = TeacherSpec(name="t", factory="torch.nn.Linear")
    assert isinstance(spec.trust, TrustProfile)
    assert spec.trust.declared is False
    declared = TeacherSpec(name="t", factory="torch.nn.Linear", trust={"trust_init": 0.5})
    assert declared.trust.declared is True
    assert declared.trust.trust_init == 0.5


def test_spec_refuses_a_widening_trust_profile():
    with pytest.raises(ValidationError):
        TeacherSpec(name="t", factory="torch.nn.Linear", trust={"tiers": [1, 3]})


def test_from_spec_requires_a_name_and_accepts_id():
    with pytest.raises(ValidationError, match="requires 'name'"):
        TeacherSpec.from_spec({"factory": "torch.nn.Linear"})
    spec = TeacherSpec.from_spec({"id": "alpha", "factory": "torch.nn.Linear", "label_names": "a"})
    assert spec.name == "alpha"
    assert spec.label_names == ("a",)
    assert TeacherSpec.from_spec(spec) is spec


def test_module_teacher_is_frozen_and_inert(tiny_teacher, states):
    """Invariant I2: a teacher never accumulates a gradient."""
    assert tiny_teacher.frozen
    assert not tiny_teacher.module.training
    assert all(not p.requires_grad for p in tiny_teacher.module.parameters())
    output = tiny_teacher(states)
    assert output.logits.shape == (6, 3)
    assert not output.logits.requires_grad
    assert all(p.grad is None for p in tiny_teacher.module.parameters())


def test_forward_before_freeze_is_refused(tiny_spec, runtime):
    teacher = Teacher(torch.nn.Linear(4, 3), tiny_spec, runtime=runtime)
    with pytest.raises(ContractError, match="before freeze"):
        teacher(torch.zeros(2, 4))


def test_cache_hits_on_the_same_tensor_and_invalidates_on_edit(tiny_teacher, states):
    first = tiny_teacher(states)
    assert tiny_teacher(states) is first
    assert tiny_teacher.describe()["calls"] == 1
    states.add_(1.0)  # in-place edit bumps the version counter
    assert tiny_teacher(states) is not first
    assert tiny_teacher(states, reset=True) is not first
    tiny_teacher.clear_cache()
    assert tiny_teacher.describe()["calls"] == 3


def test_digest_is_stable_and_changes_with_weights(tiny_teacher):
    digest = tiny_teacher.digest
    assert digest == tiny_teacher.digest
    other = Teacher(
        torch.nn.Linear(4, 3), tiny_teacher.spec, runtime=tiny_teacher.runtime
    ).freeze()
    assert other.digest != digest


def test_callable_teacher_has_no_parameters(runtime, states):
    spec = TeacherSpec(name="fn", kind="callable", factory="torch.tanh")
    teacher = build_teacher(spec, runtime=runtime)
    assert isinstance(teacher.module, CallableModule)
    assert list(teacher.module.parameters()) == []
    assert torch.allclose(teacher(states).logits, torch.tanh(states))


def test_state_dict_teacher_round_trips_and_rejects_mismatch(tmp_path, runtime, states):
    source = torch.nn.Linear(4, 3)
    path = tmp_path / "teacher.pt"
    torch.save({"state_dict": source.state_dict()}, path)
    spec = TeacherSpec(
        name="sd",
        kind="state_dict",
        arch="torch.nn.Linear",
        path=str(path),
        kwargs={"in_features": 4, "out_features": 3},
    )
    teacher = build_teacher(spec, runtime=runtime)
    assert torch.allclose(teacher(states).logits, source(states), atol=1e-6)

    bad = TeacherSpec(
        name="sd",
        kind="state_dict",
        arch="torch.nn.Linear",
        path=str(path),
        kwargs={"in_features": 4, "out_features": 5},
    )
    with pytest.raises((RuntimeError, ContractError)):
        build_teacher(bad, runtime=runtime)

    missing = TeacherSpec(
        name="sd",
        kind="state_dict",
        arch="torch.nn.Linear",
        path=str(tmp_path / "nope.pt"),
        kwargs={"in_features": 4, "out_features": 3},
    )
    with pytest.raises(ImproperlyConfigured, match="does not exist"):
        build_teacher(missing, runtime=runtime)


def test_loader_errors_name_the_problem(runtime):
    with pytest.raises(ImproperlyConfigured, match="cannot import"):
        build_teacher(TeacherSpec(name="t", factory="no_such_pkg.thing"), runtime=runtime)
    with pytest.raises(ImproperlyConfigured, match="no attribute"):
        build_teacher(TeacherSpec(name="t", factory="torch.no_such_attr"), runtime=runtime)
    with pytest.raises(ContractError, match="expected nn.Module"):
        build_teacher(TeacherSpec(name="t", factory="builtins.dict"), runtime=runtime)


def test_missing_optional_dependency_fails_loudly(runtime):
    spec = TeacherSpec(name="t", factory="torch.nn.Linear", requires="definitely_not_installed_pkg")
    with pytest.raises(ImproperlyConfigured, match="optional dependency"):
        build_teacher(spec, runtime=runtime)


def test_a_supplied_module_is_used_and_must_be_a_module(tiny_spec, runtime):
    module = torch.nn.Linear(4, 3)
    assert build_teacher(tiny_spec, runtime=runtime, module=module).module is module
    with pytest.raises(ContractError, match="expected nn.Module"):
        build_teacher(tiny_spec, runtime=runtime, module="nope")  # type: ignore[arg-type]


def test_normalize_output_shapes():
    t = torch.randn(2, 3)
    assert normalize_output(t).logits is t
    from_tuple = normalize_output((t, torch.randn(2, 5)))
    assert from_tuple.logits is t and from_tuple.hidden.shape == (2, 5)
    from_dict = normalize_output({"logits": t, "last_hidden_state": torch.randn(2, 5), "probs": t})
    assert from_dict.hidden is not None and from_dict.probs is t

    class _Out:
        logits = t
        last_hidden_state = None

    assert normalize_output(_Out()).logits is t
    vector = normalize_output(torch.randn(3))
    assert vector.logits.shape == (1, 3)


def test_normalize_output_hidden_key_and_failures():
    t = torch.randn(2, 3)
    chosen = normalize_output({"logits": t, "emb": torch.randn(2, 7)}, hidden_key="emb")
    assert chosen.hidden.shape == (2, 7)
    with pytest.raises(ContractError, match="hidden key"):
        normalize_output(t, hidden_key="absent")
    with pytest.raises(ContractError, match="empty"):
        normalize_output(())
    with pytest.raises(ContractError, match="no tensor"):
        normalize_output("text")


def test_teacher_output_features_prefers_hidden():
    logits, hidden = torch.zeros(2, 3), torch.ones(2, 4)
    assert TeacherOutput(logits=logits, hidden=hidden).features() is hidden
    assert TeacherOutput(logits=logits).features() is logits
    with pytest.raises(ContractError):
        TeacherOutput().features()
    with pytest.raises(ContractError, match="no key"):
        TeacherOutput(logits=logits).features("zzz")


def test_transformers_teacher_loads_a_real_model(tmp_path, runtime, states):
    try:
        import transformers

        _ = transformers.BertModel  # lazy module: touch it so a broken install surfaces here
    except (ImportError, OSError, RuntimeError) as exc:  # a broken optional install is an environment fact
        pytest.skip(f"transformers is not importable here: {exc}")
    config = transformers.BertConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
    )
    transformers.BertModel(config).save_pretrained(tmp_path)
    spec = TeacherSpec(name="bert", kind="transformers", path=str(tmp_path))
    teacher = build_teacher(spec, runtime=runtime)
    ids = torch.randint(0, 16, (3, 5))
    output = teacher(ids)
    assert output.hidden.shape == (3, 5, 8)
    assert all(not p.requires_grad for p in teacher.module.parameters())
    with pytest.raises(ImproperlyConfigured, match="no loader"):
        build_teacher(
            TeacherSpec(name="bert", kind="transformers", path=str(tmp_path), kwargs={"loader": "Nope"}),
            runtime=runtime,
        )
