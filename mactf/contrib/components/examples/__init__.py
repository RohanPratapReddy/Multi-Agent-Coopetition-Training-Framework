"""The eight worked examples of document 17 section 18, as importable modules.

Importing this package registers nothing and imports no torch: ``EXAMPLES`` is an index. A
module registers its component when *it* is imported, which is what ``COMPONENTS.MODULES``
asks for (I5: registration is an explicit opt-in). Inside a per-run overlay
(:func:`mactf.components.overlay`) an example registers into that overlay (scope ``run``),
so a test or a tool can load one without touching the process registry.

Each entry names the module, the kind, the settings fragment that turns it on (a YAML file
under ``conf/``), the registered names, and the *misbehaving* variant that the guard row of
the section 18 table is about (a class in the same module, never registered by the module).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["EXAMPLES", "CONF_DIR", "PLANS_DIR", "registration_scope", "conf_path", "load_conf", "plan_path"]

_HERE = Path(__file__).resolve().parent
CONF_DIR = _HERE / "conf"
PLANS_DIR = _HERE / "plans"
_PACKAGE = "mactf.contrib.components.examples"


def registration_scope() -> str:
    """``"run"`` while a per-run overlay is active, else ``"global"``."""
    from ....components import HUB

    return "run" if HUB.active_overlays() else "global"


def conf_path(number: str) -> Path:
    """The settings fragment of example ``18.N``."""
    return CONF_DIR / f"{number.replace('.', '_')}.yaml"


def plan_path(name: str) -> Path:
    return PLANS_DIR / f"{name}.yaml"


def load_conf(number: str) -> dict[str, Any]:
    """The fragment of example ``18.N`` as a mapping of settings blocks."""
    import yaml

    with conf_path(number).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


#: ``"18.N" -> entry``. ``misbehaves`` is ``(class name in the module, code the guard raises)``.
EXAMPLES: dict[str, dict[str, Any]] = {
    "18.1": {
        "title": "A reward stage that tempers credit",
        "module": f"{_PACKAGE}.tempered_reward", "kind": "scorer", "names": ("tempered_reward",),
        "conf": "18_1.yaml", "misbehaves": ("InflatedReward", "C008"),
    },
    "18.2": {
        "title": "A reward term: diminishing returns",
        "module": f"{_PACKAGE}.sqrt_magnitude", "kind": "reward_term", "names": ("sqrt_magnitude",),
        "conf": "18_2.yaml", "misbehaves": ("NegativeTerm", "ComplianceError"),
    },
    "18.3": {
        "title": "A weight-update strategy: per-group, with state",
        "module": f"{_PACKAGE}.ema_softmax", "kind": "weight_update", "names": ("ema_softmax",),
        "conf": "18_3.yaml", "misbehaves": ("SumsToThree", "E063"),
    },
    "18.4": {
        "title": "A peer strategy: mix only every N epochs",
        "module": f"{_PACKAGE}.every_n", "kind": "peer_strategy", "names": ("every_n",),
        "conf": "18_4.yaml", "misbehaves": ("LegacyEveryN", "E072"),
    },
    "18.5": {
        "title": "An optimizer and a scheduler: Lion with a step-down",
        "module": f"{_PACKAGE}.lion", "kind": "optimizer", "kinds": ("optimizer", "scheduler"),
        "names": ("Lion", "step_then_flat"),
        "conf": "18_5.yaml", "misbehaves": ("drop_at=0", "E008"),
    },
    "18.6": {
        "title": "An RL function: a Huber value loss",
        "module": f"{_PACKAGE}.huber_value", "kind": "rl_function.loss", "names": ("huber_value",),
        "conf": "18_6.yaml", "misbehaves": ("NonFiniteHuber", "E047"),
    },
    "18.7": {
        "title": "A training policy",
        "module": f"{_PACKAGE}.anchored_policy", "kind": "trainer_policy", "names": ("anchored",),
        "conf": "18_7.yaml", "misbehaves": ("Forgetful", "S004"),
    },
    "18.8": {
        "title": "Retire, revive and purge a peer",
        "module": None, "kind": "peer tree", "names": (),
        "conf": "18_8.yaml", "plans": ("plan-retire", "plan-revive", "plan-purge"),
        "misbehaves": ("plan-purge", "E082"),
    },
}
