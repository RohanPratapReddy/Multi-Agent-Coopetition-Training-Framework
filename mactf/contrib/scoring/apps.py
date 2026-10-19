"""The scoring contrib app."""

from __future__ import annotations

from ...apps import AppConfig
from ...checks import ERROR, WARNING, CheckMessage
from ...rules.catalog import validate_catalog


class ScoringConfig(AppConfig):
    name = "mactf.contrib.scoring"
    label = "mactf_scoring"
    verbose_name = "MACTF Scoring"

    def ready(self) -> None:
        from . import systems  # noqa: F401  (registers checks)


def check_rule_catalog() -> list[CheckMessage]:
    problems = validate_catalog()
    return [
        CheckMessage(
            level=ERROR,
            message=problem,
            hint="The catalog must mirror docs/framework.md R1-R17 and P1-P23.",
            id="mactf.scoring.E001",
        )
        for problem in problems
    ]


def check_settings_consistency() -> list[CheckMessage]:
    from ...conf import get_settings
    from ...rules.catalog import CATALOG

    settings = get_settings()
    messages: list[CheckMessage] = []
    lambdas = settings["SCORING"]["PENALTY_LAMBDAS"]
    missing = sorted(rule_id for rule_id in CATALOG if CATALOG[rule_id].is_penalty and rule_id not in lambdas)
    if missing:
        messages.append(
            CheckMessage(
                level=WARNING,
                message=f"penalties without a configured lambda use DEFAULT_LAMBDA: {missing}",
                id="mactf.scoring.W001",
            )
        )
    reserved = {"P1", "P2", "P17", "P20", "P21", "P22"}
    weakened = reserved - set(settings["SCORING"]["NON_OFFSETTABLE"])
    if weakened:
        messages.append(
            CheckMessage(
                level=ERROR,
                message=f"non-offsettable rules were removed from settings: {sorted(weakened)}",
                hint="framework.md 6.1 forbids allowing rewards to offset these penalties.",
                id="mactf.scoring.E002",
            )
        )
    return messages
