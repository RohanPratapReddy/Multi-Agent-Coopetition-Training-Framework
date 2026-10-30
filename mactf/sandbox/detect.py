"""Provider detection and selection.

Selection is deterministic and auditable: probes are ranked by isolation
strength, then by usability, and the chosen report is always returned alongside
the provider so a run can record exactly what boundary it ran behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..exceptions import SandboxUnavailable
from .base import (
    CapabilityReport,
    IsolationLevel,
    SandboxProvider,
    SandboxSpec,
    ensure_hardware,
)


@dataclass
class ProviderChoice:
    provider: SandboxProvider
    report: CapabilityReport
    requested: str
    degraded: bool = False
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        mode = "degraded" if self.degraded else "requested"
        return (
            f"provider={self.report.provider} ({self.report.isolation.label}) "
            f"requested={self.requested} ({mode})"
        )


def candidate_providers() -> list[SandboxProvider]:
    """Instantiate every built-in provider, cheapest probe last."""
    from .local import LocalSandbox
    from .qemu import QemuSandbox

    return [QemuSandbox(), LocalSandbox()]


def probe_all(
    providers: Iterable[SandboxProvider] | None = None,
) -> list[CapabilityReport]:
    reports: list[CapabilityReport] = []
    for provider in providers or candidate_providers():
        try:
            reports.append(provider.probe())
        except Exception as exc:  # noqa: BLE001 - a broken probe must not abort selection
            reports.append(
                CapabilityReport(
                    provider=provider.name,
                    isolation=provider.isolation,
                    usable=False,
                    reasons=[f"probe raised {exc!r}"],
                )
            )
    return reports


def choose_provider(
    *,
    requested: str = "auto",
    isolation: IsolationLevel | str | None = None,
    settings: dict[str, Any] | None = None,
    providers: Iterable[SandboxProvider] | None = None,
    require_hardware: bool | None = None,
    allow_fallback: bool | None = None,
) -> ProviderChoice:
    """Pick a provider that satisfies the requested isolation."""
    settings = settings or {}
    require_hardware = (
        bool(settings.get("REQUIRE_HARDWARE", False)) if require_hardware is None else require_hardware
    )
    allow_fallback = (
        bool(settings.get("ALLOW_FALLBACK", True)) if allow_fallback is None else allow_fallback
    )
    target = IsolationLevel.parse(isolation or settings.get("ISOLATION", "hardware"))

    instances = {provider.name: provider for provider in (providers or candidate_providers())}
    reports = probe_all(instances.values())
    by_name = {report.provider: report for report in reports}

    if requested and requested.lower() != "auto":
        if requested.lower() not in instances:
            raise SandboxUnavailable(
                f"requested provider {requested!r} is not registered; "
                f"available: {', '.join(sorted(instances))}"
            )
        report = by_name[requested.lower()]
        if not report.usable:
            if not allow_fallback:
                raise SandboxUnavailable(
                    f"provider {requested!r} is not usable: {'; '.join(report.reasons)}"
                )
            choice = _fallback(instances, by_name, target, allow_fallback)
            choice.requested = requested
            choice.reasons.extend(report.reasons)
            return choice
        choice = ProviderChoice(
            provider=instances[requested.lower()],
            report=report,
            requested=requested,
        )
        if target >= IsolationLevel.HARDWARE:
            ensure_hardware(report, allow_fallback=allow_fallback and not require_hardware)
        return choice

    # Rank: sufficient isolation first, then stronger isolation, then usable.
    ranked = sorted(
        reports,
        key=lambda report: (
            0 if report.isolation >= target else 1,
            -int(report.isolation),
            0 if report.usable else 1,
            report.provider,
        ),
    )
    for report in ranked:
        if report.usable and report.isolation >= target:
            return ProviderChoice(
                provider=instances[report.provider],
                report=report,
                requested=requested,
            )
    return _fallback(instances, by_name, target, allow_fallback)


def _fallback(
    instances: dict[str, SandboxProvider],
    reports: dict[str, CapabilityReport],
    target: IsolationLevel,
    allow_fallback: bool,
) -> ProviderChoice:
    usable = [report for report in reports.values() if report.usable]
    if not usable:
        reasons = "; ".join(
            f"{report.provider}: {'; '.join(report.reasons) or 'unusable'}"
            for report in reports.values()
        )
        raise SandboxUnavailable(f"no usable sandbox provider: {reasons}")
    if target >= IsolationLevel.HARDWARE and not allow_fallback:
        raise SandboxUnavailable(
            "hardware isolation requested but unavailable: "
            + "; ".join(
                f"{report.provider}: {'; '.join(report.reasons) or 'unusable'}"
                for report in reports.values()
                if report.isolation >= IsolationLevel.HARDWARE
            )
        )
    best = sorted(usable, key=lambda report: (-int(report.isolation), report.provider))[0]
    return ProviderChoice(
        provider=instances[best.provider],
        report=best,
        requested="auto",
        degraded=best.isolation < target,
        reasons=list(best.reasons),
    )


def build_spec(
    peer: str,
    *,
    settings: dict[str, Any],
    root: Path,
    serial_socket: Path | None = None,
) -> SandboxSpec:
    """Translate SANDBOX settings into a :class:`SandboxSpec` for one peer."""

    def resolve(template: str) -> Path:
        candidate = Path(str(template).format(peer=peer))
        return candidate if candidate.is_absolute() else (root / candidate)

    socket_template = settings.get("SERIAL_SOCKET")
    socket_path = None
    if serial_socket is not None:
        socket_path = serial_socket
    elif socket_template:
        socket_path = resolve(str(socket_template))

    return SandboxSpec(
        peer=peer,
        isolation=IsolationLevel.parse(settings.get("ISOLATION", "hardware")),
        memory_mb=int(settings.get("MEMORY_MB", 4096)),
        vcpus=int(settings.get("VCPUS", 2)),
        network=str(settings.get("NETWORK", "none")),
        snapshot=bool(settings.get("SNAPSHOT", True)),
        base_image=resolve(str(settings.get("BASE_IMAGE", "var/images/base.qcow2"))),
        work_image=resolve(str(settings.get("WORK_IMAGE", "var/images/peer-{peer}.qcow2"))),
        accel=str(settings.get("ACCEL", "auto")),
        boot_timeout=float(settings.get("BOOT_TIMEOUT", 180)),
        command_timeout=float(settings.get("COMMAND_TIMEOUT", 300)),
        labels={"managed_by": "mactf"},
    )
