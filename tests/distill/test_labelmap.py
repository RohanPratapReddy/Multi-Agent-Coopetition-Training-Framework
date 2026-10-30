"""Label maps: probability-space aggregation, dropped labels, axis construction."""

from __future__ import annotations

import pytest
import torch

from mactf.distill.labelmap import LabelMap, build_label_map, shared_axis
from mactf.exceptions import ContractError, ValidationError


def test_shared_axis_is_the_sorted_intersection():
    axis = shared_axis({"a": ["x", "y", "z"], "b": ["z", "y", "w"]})
    assert axis == ("y", "z")
    assert shared_axis({}) == ()


def test_identity_map_returns_log_softmax():
    labels = ["a", "b", "c"]
    mapping = build_label_map(labels, labels)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = mapping(logits)
    assert torch.allclose(out, torch.log_softmax(logits, dim=-1), atol=1e-6)


def test_dropped_labels_are_recorded_and_mass_renormalised():
    mapping = build_label_map(["a", "b", "c"], ["a", "c"])
    assert mapping.dropped == ("b",)
    out = mapping(torch.tensor([[0.0, 5.0, 0.0]]))
    probs = out.exp()
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(probs[0, 0]) == pytest.approx(0.5, abs=1e-6)


def test_aliases_sum_probabilities_not_logits():
    mapping = build_label_map(["car", "auto", "boat"], ["car", "boat"], aliases={"auto": "car"})
    logits = torch.log(torch.tensor([[0.3, 0.3, 0.4]]))
    probs = mapping(logits).exp()
    assert float(probs[0, 0]) == pytest.approx(0.6, abs=1e-6)
    assert float(probs[0, 1]) == pytest.approx(0.4, abs=1e-6)


def test_all_mass_on_dropped_labels_gives_a_uniform_row():
    mapping = build_label_map(["a", "b", "c"], ["a", "b"])
    out = mapping(torch.tensor([[-1e4, -1e4, 50.0]]))
    assert torch.allclose(out.exp(), torch.full((1, 2), 0.5), atol=1e-6)


def test_wrong_width_raises():
    mapping = build_label_map(["a", "b"], ["a", "b"])
    with pytest.raises(ContractError):
        mapping(torch.zeros(1, 3))


def test_builder_validation():
    with pytest.raises(ValidationError, match="appears twice"):
        build_label_map(["a", "a"], ["a"])
    with pytest.raises(ValidationError, match="absent"):
        build_label_map(["a"], ["a", "b"])
    with pytest.raises(ValidationError, match="not on the shared axis"):
        build_label_map(["a", "b"], ["a"], aliases={"b": "zzz"})
    with pytest.raises(ValidationError, match="not a label"):
        build_label_map(["a"], ["a"], aliases={"q": "a"})


def test_constructor_validation():
    with pytest.raises(ValidationError):
        LabelMap([0], source_size=2, axis=[])
    with pytest.raises(ValidationError):
        LabelMap([0, 1], source_size=2, axis=["a", "a"])
    with pytest.raises(ValidationError):
        LabelMap([5], source_size=2, axis=["a"])
    with pytest.raises(ValidationError, match="two axis labels"):
        LabelMap([0, 0], source_size=2, axis=["a", "b"])
    with pytest.raises(ValidationError):
        LabelMap([0], source_size=2, axis=["a", "b"])
    with pytest.raises(ValidationError):
        LabelMap([-1], source_size=2, axis=["a"])


def test_describe_reports_the_shape():
    mapping = build_label_map(["a", "b", "c"], ["a", "c"])
    assert mapping.describe() == {"source_size": 3, "axis": 2, "dropped": ["b"], "missing": []}
