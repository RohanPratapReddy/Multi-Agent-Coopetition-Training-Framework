"""The coopetition contrib app."""

from __future__ import annotations

from ...apps import AppConfig
from ...checks import WARNING, CheckMessage


class CoopetitionConfig(AppConfig):
    name = "mactf.contrib.coopetition"
    label = "mactf_coopetition"
    verbose_name = "MACTF Coopetition"

    def ready(self) -> None:
        from . import systems  # noqa: F401


def check_aggregation_bounds() -> list[CheckMessage]:
    from ...conf import get_settings
    from ...coopetition.aggregation import feasible_bounds
    from ...exceptions import ComplianceError

    settings = get_settings()
    peers = settings.get("PEERS") or []
    count = max(2, len(peers) or 2)
    coop = settings["COOPETITION"]
    messages: list[CheckMessage] = []
    try:
        feasible_bounds(
            count,
            floor=float(coop["WEIGHT_FLOOR"]),
            cap=float(coop["WEIGHT_CAP"]),
        )
    except ComplianceError as exc:
        messages.append(
            CheckMessage(
                level=WARNING,
                message=str(exc),
                hint="Adjust WEIGHT_FLOOR/WEIGHT_CAP so 1/N lies inside the band.",
                id="mactf.coopetition.W001",
            )
        )
    if int(coop["KAPPA_MAX"]) <= int(coop["KAPPA_MIN"]):
        messages.append(
            CheckMessage(
                level=WARNING,
                message="KAPPA_MAX must exceed KAPPA_MIN for adaptive temperature to move",
                id="mactf.coopetition.W002",
            )
        )
    return messages
