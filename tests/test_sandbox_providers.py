"""Sandbox capability detection and degradation tests."""

from __future__ import annotations

import pytest

from mactf.exceptions import SandboxUnavailable
from mactf.sandbox.base import Capability, CapabilityReport, IsolationLevel, SandboxSpec
from mactf.sandbox.channel import Frame, ProtocolError
from mactf.sandbox.detect import build_spec, choose_provider, probe_all
from mactf.sandbox.local import LocalSandbox
from mactf.sandbox.qemu import QemuSandbox, _accel_flag


# -- base types ----------------------------------------------------------
def test_isolation_level_ordering():
    assert IsolationLevel.HARDWARE > IsolationLevel.CONTAINER > IsolationLevel.PROCESS


def test_isolation_parse_accepts_names_and_ints():
    assert IsolationLevel.parse("hardware") is IsolationLevel.HARDWARE
    assert IsolationLevel.parse(2) is IsolationLevel.CONTAINER
    with pytest.raises(ValueError):
        IsolationLevel.parse("quantum")


def test_capability_report_require_raises():
    report = CapabilityReport(
        provider="x",
        isolation=IsolationLevel.PROCESS,
        usable=True,
        capabilities=[Capability("qemu", False, "missing")],
    )
    with pytest.raises(Exception):
        report.require("qemu")


# -- probing -------------------------------------------------------------
def test_probe_all_never_raises():
    reports = probe_all()
    assert reports
    assert all(report.provider for report in reports)


def test_local_provider_probes_usable():
    report = LocalSandbox().probe()
    assert report.provider == "local"
    assert report.isolation is IsolationLevel.PROCESS
    assert report.usable
    assert any("hardware" in reason for reason in report.reasons)


def test_qemu_probe_reports_missing_binaries_without_raising():
    report = QemuSandbox().probe()
    assert report.provider == "qemu"
    assert report.isolation is IsolationLevel.HARDWARE
    # On a host without QEMU this is unusable, and the reason is recorded.
    if not report.usable:
        assert report.reasons


# -- selection -----------------------------------------------------------
def test_auto_selection_degrades_with_recorded_reason():
    choice = choose_provider(requested="auto", isolation="hardware", allow_fallback=True)
    assert choice.provider is not None
    if choice.report.isolation < IsolationLevel.HARDWARE:
        assert choice.degraded
        assert choice.reasons


def test_hardware_without_fallback_refuses_degradation(monkeypatch):
    # Force a host with no hardware provider by making QEMU report unusable.
    monkeypatch.setattr(
        QemuSandbox,
        "probe",
        lambda self: CapabilityReport(
            provider="qemu",
            isolation=IsolationLevel.HARDWARE,
            usable=False,
            reasons=["simulated missing qemu"],
        ),
    )
    with pytest.raises(SandboxUnavailable):
        choose_provider(requested="auto", isolation="hardware", allow_fallback=False)


def test_unknown_provider_raises():
    with pytest.raises(SandboxUnavailable, match="not registered"):
        choose_provider(requested="firecracker")


def test_local_provider_executes_and_isolates_env(tmp_path):
    provider = LocalSandbox(workdir=tmp_path)
    spec = SandboxSpec(peer="test", network="none")
    handle = provider.start(provider.provision(spec))
    result = provider.exec(handle, ["python", "-c", "import os; print(os.environ.get('MACTF_SANDBOX'))"])
    assert result.ok
    assert "local" in result.stdout


def test_local_provider_reports_timeout(tmp_path):
    provider = LocalSandbox(workdir=tmp_path)
    spec = SandboxSpec(peer="test", command_timeout=1)
    handle = provider.start(provider.provision(spec))
    result = provider.exec(handle, ["python", "-c", "import time; time.sleep(5)"], timeout=1)
    assert result.timed_out
    assert result.exit_code == 124


# -- qemu command construction ------------------------------------------
def test_accel_flag_rejects_unknown():
    with pytest.raises(Exception):
        _accel_flag("magic")


def test_build_spec_uses_settings_templates(tmp_path):
    spec = build_spec(
        "alpha",
        settings={
            "ISOLATION": "hardware",
            "MEMORY_MB": 2048,
            "VCPUS": 4,
            "NETWORK": "none",
            "BASE_IMAGE": "images/base.qcow2",
            "WORK_IMAGE": "images/{peer}.qcow2",
            "SERIAL_SOCKET": "run/{peer}.sock",
        },
        root=tmp_path,
    )
    assert spec.peer == "alpha"
    assert spec.memory_mb == 2048
    assert spec.work_image is not None and "alpha" in spec.work_image.name
    assert spec.network == "none"


def test_qemu_provision_fails_closed_on_missing_base(tmp_path):
    provider = QemuSandbox()
    spec = SandboxSpec(
        peer="alpha",
        base_image=tmp_path / "missing.qcow2",
        work_image=tmp_path / "work.qcow2",
    )
    with pytest.raises(Exception, match="base image not found"):
        provider.provision(spec)


# -- wire protocol -------------------------------------------------------
def test_frame_roundtrip():
    frame = Frame(kind="exec", payload={"command": "echo hi"}, id=1)
    decoded = Frame.decode(frame.encode())
    assert decoded.kind == "exec"
    assert decoded.payload == {"command": "echo hi"}
    assert decoded.id == 1


def test_frame_decode_rejects_malformed():
    with pytest.raises(ProtocolError):
        Frame.decode(b"not json")
    with pytest.raises(ProtocolError):
        Frame.decode(b'{"payload": {}}')
