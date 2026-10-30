"""Shared fixtures for the contrastive-distillation tests."""

from __future__ import annotations

import pytest
import torch

from mactf.distill.teacher import TeacherSpec, build_teacher
from mactf.runtime import Runtime


@pytest.fixture
def runtime() -> Runtime:
    return Runtime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        amp=False,
        seed=0,
        deterministic=False,
    )


@pytest.fixture
def states() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(6, 4, generator=generator)


@pytest.fixture
def tiny_spec() -> TeacherSpec:
    """A real ``nn.Linear`` teacher built through the declarative loader."""
    return TeacherSpec(
        name="tiny",
        kind="module",
        factory="torch.nn.Linear",
        kwargs={"in_features": 4, "out_features": 3},
    )


@pytest.fixture
def tiny_teacher(tiny_spec, runtime):
    torch.manual_seed(5)
    return build_teacher(tiny_spec, runtime=runtime)
