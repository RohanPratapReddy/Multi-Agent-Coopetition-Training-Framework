"""Output views: widths, frozen projections, pooling, view pairs, batches."""

from __future__ import annotations

import pytest
import torch

from mactf.distill.batch import DistillBatch, coerce_batch
from mactf.distill.labelmap import build_label_map
from mactf.distill.views import (
    DistributionView,
    FixedRandomView,
    HiddenStateView,
    IdentityView,
    LinearView,
    OutputBundle,
    ViewPair,
    build_view,
    build_view_pair,
    peer_bundle,
)
from mactf.exceptions import ContractError, ValidationError


def _bundle(logits=None, hidden=None, **extra):
    return OutputBundle(logits=logits, hidden=hidden, extra=extra, source="teacher", side="t")


def test_identity_view_passes_through_and_rejects_width_mismatch():
    view = IdentityView(4)
    x = torch.randn(3, 4)
    assert torch.equal(view(_bundle(logits=x)), x)
    with pytest.raises(ContractError, match="expects width"):
        view(_bundle(logits=torch.randn(3, 5)))
    open_view = IdentityView()
    open_view(_bundle(logits=x))
    assert open_view.dim == 4


def test_identity_view_prefers_hidden_and_flattens():
    hidden = torch.randn(2, 3, 2)
    out = IdentityView()(_bundle(logits=torch.zeros(2, 9), hidden=hidden))
    assert out.shape == (2, 6)
    with pytest.raises(ContractError):
        IdentityView()(_bundle())


def test_linear_view_has_no_trainable_parameters_and_is_deterministic():
    a = LinearView(5, 3, seed=7)
    b = LinearView(5, 3, seed=7)
    assert all(not p.requires_grad for p in a.parameters())
    assert torch.equal(a.weight, b.weight)
    x = torch.randn(4, 5)
    assert a(_bundle(logits=x)).shape == (4, 3)
    assert not a(_bundle(logits=x)).requires_grad


def test_linear_view_gradient_reaches_the_input_only():
    view = LinearView(3, 2, seed=1)
    x = torch.randn(2, 3, requires_grad=True)
    view(_bundle(logits=x)).sum().backward()
    assert x.grad is not None
    assert view.weight.grad is None


def test_linear_view_validation():
    with pytest.raises(ValidationError):
        LinearView(0, 2, seed=1)
    with pytest.raises(ValidationError, match="'weight' or 'seed'"):
        LinearView(2, 2)
    with pytest.raises(ValidationError):
        LinearView(2, 2, weight=torch.zeros(3, 3))
    with pytest.raises(ValidationError):
        LinearView(2, 2, seed=1, bias=torch.zeros(5))
    with pytest.raises(ContractError):
        LinearView(2, 2, seed=1)(_bundle(logits=torch.zeros(1, 3)))


def test_fixed_random_view_records_its_seed_and_differs_per_seed():
    a, b = FixedRandomView(4, 3, seed=1), FixedRandomView(4, 3, seed=2)
    assert a.describe()["seed"] == 1
    assert not torch.equal(a.weight, b.weight)


def test_hidden_view_pools_and_projects():
    seq = torch.randn(2, 5, 4)
    mean = HiddenStateView("h", pool="mean")(_bundle(h=seq))
    assert torch.allclose(mean, seq.mean(dim=1))
    assert torch.equal(HiddenStateView("h", pool="last")(_bundle(h=seq)), seq[:, -1])
    assert torch.equal(HiddenStateView("h", pool="first")(_bundle(h=seq)), seq[:, 0])
    assert HiddenStateView("h", pool="flat")(_bundle(h=seq)).shape == (2, 20)
    projected = HiddenStateView("h", out_dim=3, seed=4)
    assert projected(_bundle(h=seq)).shape == (2, 3)
    assert projected.in_dim == 4
    with pytest.raises(ContractError, match="no such key"):
        HiddenStateView("missing")(_bundle(h=seq))
    with pytest.raises(ValidationError):
        HiddenStateView("h", pool="bogus")
    with pytest.raises(ContractError):
        HiddenStateView("h")(_bundle(h=torch.zeros(2)))


def test_distribution_view_is_log_softmax_with_calibration():
    logits = torch.randn(3, 4)
    out = DistributionView(["a", "b", "c", "d"], calibration=2.0)(_bundle(logits=logits))
    assert torch.allclose(out, torch.log_softmax(logits / 2.0, dim=-1), atol=1e-6)
    with pytest.raises(ContractError, match="supply a LabelMap"):
        DistributionView(["a", "b"])(_bundle(logits=logits))
    with pytest.raises(ValidationError):
        DistributionView(["a"], calibration=0.0)


def test_distribution_view_with_label_map_matches_axis_width():
    mapping = build_label_map(["a", "b", "c"], ["a", "c"])
    view = DistributionView(["a", "c"], label_map=mapping)
    assert view(_bundle(logits=torch.randn(2, 3))).shape == (2, 2)
    with pytest.raises(ValidationError):
        DistributionView(["a", "b", "c"], label_map=mapping)


def test_peer_bundle_lifts_a_vector_to_a_batch():
    bundle = peer_bundle(torch.zeros(4), path="p")
    assert bundle.logits.shape == (1, 4)
    assert bundle.source == "peer"


def test_view_pair_checks_widths():
    ViewPair(peer=IdentityView(4), teacher=IdentityView(4), dim=4)
    with pytest.raises(ValidationError, match="same out_dim"):
        ViewPair(peer=IdentityView(4), teacher=IdentityView(5), dim=4)
    with pytest.raises(ValidationError):
        ViewPair(peer=IdentityView(4), teacher=IdentityView(4), dim=0)


def test_build_view_kinds_and_errors():
    assert isinstance(build_view(side="s", spec=None, in_dim=None, out_dim=4, seed=0), IdentityView)
    assert isinstance(
        build_view(side="s", spec={"kind": "fixed_random"}, in_dim=6, out_dim=4, seed=0),
        FixedRandomView,
    )
    assert isinstance(
        build_view(side="s", spec={"kind": "linear", "out_dim": 3}, in_dim=6, out_dim=4, seed=0),
        LinearView,
    )
    with pytest.raises(ValidationError, match="unknown"):
        build_view(side="s", spec={"kind": "zzz"}, in_dim=1, out_dim=1, seed=0)
    with pytest.raises(ValidationError, match="needs in_dim"):
        build_view(side="s", spec={"kind": "linear"}, in_dim=None, out_dim=4, seed=0)
    with pytest.raises(ValidationError, match="requires 'key'"):
        build_view(side="s", spec={"kind": "hidden"}, in_dim=None, out_dim=4, seed=0)
    with pytest.raises(ValidationError, match="requires 'labels'"):
        build_view(side="s", spec={"kind": "distribution"}, in_dim=None, out_dim=4, seed=0)


def test_build_view_pair_rejects_a_shared_seed_and_builds_matched_widths():
    kwargs = dict(
        peer="p",
        teacher="t",
        peer_spec={"kind": "fixed_random"},
        teacher_spec={"kind": "fixed_random"},
        peer_in_dim=6,
        teacher_in_dim=9,
        out_dim=5,
    )
    with pytest.raises(ValidationError, match="own"):
        build_view_pair(peer_seed=1, teacher_seed=1, **kwargs)
    pair = build_view_pair(peer_seed=1, teacher_seed=2, **kwargs)
    assert pair.dim == 5
    assert pair.peer(peer_bundle(torch.randn(3, 6), path="p")).shape == (3, 5)
    assert pair.describe()["dim"] == 5


def test_distill_batch_contract():
    with pytest.raises(ValidationError, match="at least 2"):
        DistillBatch(states=torch.zeros(1, 3))
    with pytest.raises(ContractError):
        DistillBatch(states=[1, 2])
    with pytest.raises(ContractError):
        DistillBatch(states=torch.tensor(1.0))
    batch = DistillBatch(states=torch.zeros(3, 2), peer_inputs={"p": {"mask": 1}})
    assert batch.batch == 3
    assert batch.kwargs_for("p") == {"mask": 1}
    assert batch.kwargs_for("other") == {}
    with pytest.raises(ContractError, match="shared"):
        DistillBatch(states=torch.zeros(3, 2), peer_inputs={"p": {"states": 1}}).kwargs_for("p")


def test_batch_fingerprint_binds_dtype_and_values():
    a = DistillBatch(states=torch.zeros(2, 2))
    assert a.fingerprint == DistillBatch(states=torch.zeros(2, 2)).fingerprint
    assert a.fingerprint != DistillBatch(states=torch.zeros(2, 2, dtype=torch.float64)).fingerprint
    assert a.fingerprint != DistillBatch(states=torch.ones(2, 2)).fingerprint


def test_coerce_batch_forms():
    tensor = torch.zeros(2, 2)
    assert coerce_batch(tensor).batch == 2
    mapped = coerce_batch({"states": tensor, "peer_inputs": {"p": {"k": 1}}, "note": "x"})
    assert mapped.extras == {"note": "x"}
    existing = DistillBatch(states=tensor)
    assert coerce_batch(existing) is existing
    with pytest.raises(ContractError, match="'states' key"):
        coerce_batch({"other": 1})
    with pytest.raises(ContractError):
        coerce_batch({"states": tensor, "peer_inputs": [1]})
    with pytest.raises(ContractError, match="cannot read"):
        coerce_batch("nope")
