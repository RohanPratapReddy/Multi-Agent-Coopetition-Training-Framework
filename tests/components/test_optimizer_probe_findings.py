"""One test per optimizer probe finding O001..O006 (document 17 sections 9.2.2 and 9.10.2; invariant O4)."""

from __future__ import annotations

import pytest
import torch

from mactf.components import register_optimizer
from mactf.exceptions import ImproperlyConfigured
from mactf.training.optimizer_plan import compile_entry, verify_entry


class _Sgd(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-group["lr"])


class _BadBind(_Sgd):
    def __init__(self, params, lr=1e-2):
        raise RuntimeError("cannot bind")


class _Unpicklable(_Sgd):
    def state_dict(self):
        state = super().state_dict()
        state["hook"] = lambda: None
        return state


class _NoReload(_Sgd):
    def load_state_dict(self, state):
        raise RuntimeError("will not reload")


class _NanRate(_Sgd):
    def step(self, closure=None):
        for group in self.param_groups:
            group["lr"] = float("nan")


def _entry(name: str):
    return compile_entry("OPTIMIZER", {"TYPE": name, "LR": 1e-2}, glob=None, horizon=4, written=None)


@pytest.mark.parametrize(
    "cls, code",
    [(_BadBind, "O001"), (_Unpicklable, "O004"), (_NoReload, "O005"), (_NanRate, "O006")],
)
def test_each_probe_finding_is_reported_under_e056(component_overlay, cls, code):
    register_optimizer(cls.__name__, cls, scope="run")
    with pytest.raises(ImproperlyConfigured, match=rf"E056.*{code}"):
        verify_entry(_entry(cls.__name__))


def test_a_well_behaved_optimizer_passes_the_probe(component_overlay):
    register_optimizer("OkSgd", _Sgd, scope="run")
    assert verify_entry(_entry("OkSgd")) is None
