"""The sandbox contrib app."""

from __future__ import annotations

from ...apps import AppConfig
from ...checks import INFO, WARNING, CheckMessage


class SandboxConfig(AppConfig):
    name = "mactf.contrib.sandbox"
    label = "mactf_sandbox"
    verbose_name = "MACTF Sandbox"

    def ready(self) -> None:
        from . import systems  # noqa: F401


def check_isolation() -> list[CheckMessage]:
    from ...conf import get_settings
    from ...sandbox.detect import probe_all

    settings = get_settings()
    sandbox = settings["SANDBOX"]
    reports = probe_all()
    usable = [report for report in reports if report.usable]
    messages: list[CheckMessage] = []
    if not usable:
        messages.append(
            CheckMessage(
                level=WARNING,
                message="no usable sandbox provider; training will run without isolation",
                hint="Install QEMU for hardware isolation, or accept the local provider.",
                id="mactf.sandbox.W001",
            )
        )
        return messages

    best = max(usable, key=lambda report: int(report.isolation))
    wanted = str(sandbox.get("ISOLATION", "hardware")).lower()
    if wanted in {"hardware", "3"} and best.isolation.label != "hardware":
        messages.append(
            CheckMessage(
                level=WARNING,
                message=(
                    f"SANDBOX.ISOLATION requests hardware but the best available is "
                    f"{best.isolation.label}: {'; '.join(best.reasons) or 'no reason given'}"
                ),
                hint="Run `manage.py sandbox install`, or set ALLOW_FALLBACK explicitly.",
                id="mactf.sandbox.W002",
            )
        )
    if best.isolation.label == "hardware":
        messages.append(
            CheckMessage(
                level=INFO,
                message=f"hardware isolation available via {best.provider}",
                id="mactf.sandbox.I001",
            )
        )
    return messages
