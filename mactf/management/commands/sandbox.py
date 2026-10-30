"""``sandbox`` - probe, provision, install, launch, and clean isolation backends. Section 8.10."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..exitcodes import ExitCode
from ..group import GroupCommand, Sub
from ..runsupport import declined
from ...exceptions import CapabilityUnavailable, CommandError
from ...sandbox.base import IsolationLevel
from ...sandbox.detect import build_spec, choose_provider, probe_all

LEVELS = ["none", "process", "container", "hardware"]

BASE_IMAGE_HINT = (
    "Download a cloud image, for example:\n"
    "  https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img"
)


def install_command() -> tuple[list[str], list[str]]:
    """``(argv, follow_up_lines)`` to install QEMU here; ``CapabilityUnavailable`` if it cannot be done."""
    if sys.platform.startswith("win"):
        winget = shutil.which("winget")
        if not winget:
            raise CapabilityUnavailable(
                "winget is not available. Install QEMU from https://qemu.weilnetz.de/ and add it to PATH.")
        return (
            [winget, "install", "-e", "--id", "SoftwareFreedomConservancy.QEMU",
             "--accept-package-agreements", "--accept-source-agreements"],
            ["QEMU installed. WHPX acceleration needs the Windows Hypervisor Platform:",
             "  (admin) dism /online /enable-feature /featurename:HypervisorPlatform /all",
             "  then reboot and re-run `{prog} sandbox status`"],
        )
    if sys.platform.startswith("linux"):
        managers = (
            ("apt-get", ["apt-get", "install", "-y", "qemu-system-x86", "qemu-utils"]),
            ("dnf", ["dnf", "install", "-y", "qemu-kvm", "qemu-img"]),
            ("pacman", ["pacman", "-S", "--noconfirm", "qemu-full"]),
        )
        for binary, argv in managers:
            if not shutil.which(binary):
                continue
            if hasattr(os, "geteuid") and os.geteuid() != 0:
                sudo = shutil.which("sudo")
                if sudo:
                    argv = [sudo, *argv]
            return argv, ["Verify KVM access with: ls -l /dev/kvm"]
        raise CapabilityUnavailable("no supported package manager found (apt/dnf/pacman)")
    raise CapabilityUnavailable(
        "automatic install is supported on Windows (winget) and Linux only; "
        "install QEMU manually on this platform")


def _resolve(template: Any, root: Path) -> Path | None:
    if not template:
        return None
    path = Path(str(template))
    return (path if path.is_absolute() else root / path).resolve()


def _pattern(template: str, root: Path) -> str:
    pieces = [glob.escape(piece) for piece in template.split("{peer}")]
    joined = "*".join(pieces)
    if Path(template.replace("{peer}", "x")).is_absolute():
        return joined
    return os.path.join(glob.escape(str(root)), joined)


def clean_targets(settings: Any) -> list[Path]:
    """Files the sandbox created: ``WORK_IMAGE`` and ``SERIAL_SOCKET`` with ``{peer}`` as a wildcard.

    The base image is never a target, even if a template were to match it.
    """
    sandbox = settings["SANDBOX"]
    root = settings.path("ROOT")
    base = _resolve(sandbox.get("BASE_IMAGE"), root)
    found: dict[str, Path] = {}
    for key in ("WORK_IMAGE", "SERIAL_SOCKET"):
        template = sandbox.get(key)
        if not template:
            continue
        for match in glob.glob(_pattern(str(template), root)):
            path = Path(match)
            if path.is_file() or path.is_socket() or path.is_symlink():
                if base is None or path.resolve() != base:
                    found[str(path)] = path
    return sorted(found.values())


_CHOICE_ARGS = (
    (("--peer",), {"default": "swarm", "metavar": "NAME", "completer": "peer_name",
                       "help": "Peer to build the spec for."}),
    (("--isolation",), {"default": None, "choices": LEVELS,
                        "help": "Isolation level; default is SANDBOX.ISOLATION."}),
    (("--no-fallback",), {"action": "store_true", "default": False,
                          "help": "Refuse a weaker provider than requested."}),
)


class Command(GroupCommand):
    help = "Manage the QEMU hardware sandbox and inspect isolation capability."
    category = "Project"
    output_formats = ("text",)
    default_subcommand = "status"
    examples = ("mactf sandbox", "mactf sandbox --require hardware", "mactf sandbox install --dry-run",
                "mactf sandbox clean --dry-run")
    # The default verb's own arguments are declared on the group too, so ``mactf sandbox
    # --require hardware`` parses; ``--require`` is read with ``options.get``.
    arguments = [
        (("--require",), {"choices": LEVELS, "default": None,
                          "help": "Exit 5 when the best usable isolation is below this level."}),
    ]
    subcommands = (
        Sub("status", "Probe every provider and report the best usable isolation.",
            # SUPPRESS: ``sandbox --require hardware status`` keeps the group-level value.
            arguments=((("--require",), {"choices": LEVELS, "default": argparse.SUPPRESS,
                                         "help": "Exit 5 when the best usable isolation is below LEVEL."}),),
            output_formats=("text", "json"),
            examples=("mactf sandbox status", "mactf sandbox status --require hardware")),
        Sub("install", "Install QEMU with the platform's package manager.",
            arguments=((("--force",), {"action": "store_true", "default": False,
                                       "help": "Run the installer even when QEMU is present."}),
                       (("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "Print the command without running it."})),
            output_formats=("text",), destructive=True,
            examples=("mactf sandbox install --dry-run",)),
        Sub("provision", "Create the peer's overlay disk and spec.",
            arguments=_CHOICE_ARGS, output_formats=("text", "json"),
            examples=("mactf sandbox provision --peer alpha", "mactf sandbox provision --isolation hardware")),
        Sub("launch", "Provision and start the peer's sandbox.",
            arguments=_CHOICE_ARGS, output_formats=("text", "json"),
            examples=("mactf sandbox launch --peer alpha", "mactf sandbox launch --no-fallback")),
        Sub("clean", "Remove sandbox overlay disks and sockets.",
            arguments=((("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "List what would be removed."}),),
            output_formats=("text", "json"), destructive=True,
            examples=("mactf sandbox clean --dry-run",)),
    )

    # -- status ---------------------------------------------------------------
    def sub_status(self, **options):
        reports = probe_all()
        usable = [report for report in reports if report.usable]
        best = max((report.isolation for report in usable), default=IsolationLevel.NONE)
        # json is the bare array of reports, as it has always been.
        self.out.emit([report.as_dict() for report in reports],
                      text=lambda: self._status_text(reports, best))
        required = options.get("require")
        if required and best < IsolationLevel.parse(required):
            self.out.error(f"best usable isolation is {best.label}; {required} required")
            return int(ExitCode.UNAVAILABLE)
        return int(ExitCode.OK)

    def _status_text(self, reports, best: IsolationLevel) -> None:
        for report in reports:
            self.out.result(report.describe())
        has_usable = any(report.usable for report in reports)
        self.out.result(f"best usable isolation: {best.label if has_usable else 'none'}")

    # -- install --------------------------------------------------------------
    def sub_install(self, **options):
        existing = shutil.which("qemu-system-x86_64") or shutil.which("qemu-system-x86_64.exe")
        if existing and not options["force"]:
            self.out.result(f"QEMU is already installed at {existing}; pass --force to run the installer anyway")
            return 0
        argv, follow_up = install_command()
        shown = " ".join(argv)
        if options["dry_run"]:
            self.out.result(f"would run: {shown}")
            return 0
        if not self.confirm(f"Run `{shown}` to install QEMU (may require elevation)?"):
            return declined(self)
        self.out.result(f"Installing QEMU: {shown}")
        completed = subprocess.run(argv, check=False)          # noqa: S603 - argv built above from fixed names
        if completed.returncode != 0:
            raise CommandError(f"{Path(argv[0]).name} exited with {completed.returncode}")
        for line in follow_up:
            self.out.result(line.format(prog=self.prog))
        return 0

    # -- provision and launch -------------------------------------------------
    def _choose(self, options):
        sandbox = self.settings["SANDBOX"]
        choice = choose_provider(
            requested=str(sandbox.get("PROVIDER", "auto")),
            isolation=options["isolation"] or sandbox.get("ISOLATION"),
            settings=sandbox,
            allow_fallback=False if options["no_fallback"] else None,     # None: settings decide
        )
        spec = build_spec(options["peer"], settings=sandbox, root=self.settings.path("ROOT"))
        return choice, spec

    def sub_provision(self, **options):
        choice, spec = self._choose(options)
        if not choice.report.usable:
            raise CapabilityUnavailable("selected provider is not usable: " + "; ".join(choice.report.reasons))
        if spec.base_image is None or not Path(spec.base_image).exists():
            raise CommandError(f"base image missing: {spec.base_image}\n{BASE_IMAGE_HINT}")
        handle = choice.provider.provision(spec)
        payload = {"provider": choice.report.provider, "choice": choice.describe(),
                   "spec": spec.describe(), "handle": handle.describe()}

        def text() -> None:
            self.out.result(choice.describe())
            self.out.result(f"spec: {spec.describe()}")
            self.out.result(f"provisioned: {handle.describe()}")
        self.out.emit(payload, text=text)
        return 0

    def sub_launch(self, **options):
        choice, spec = self._choose(options)
        handle = choice.provider.provision(spec)
        handle = choice.provider.start(handle)
        payload = {"provider": choice.report.provider, "choice": choice.describe(),
                   "spec": spec.describe(), "handle": handle.describe(),
                   "pid": handle.pid, "endpoints": handle.endpoints}

        def text() -> None:
            self.out.result(f"launching: {handle.describe()}")
            self.out.result(f"ready: pid={handle.pid} endpoints={handle.endpoints}")
            self.out.result(f"use `{self.prog} sandbox clean` to remove overlay disks")
        self.out.emit(payload, text=text)
        return 0

    # -- clean ----------------------------------------------------------------
    def sub_clean(self, **options):
        targets = clean_targets(self.settings)
        if options.get("dry_run"):
            return self._clean_report(targets, [], dry_run=True)
        if targets and not self.confirm(f"Remove {len(targets)} sandbox artifact(s)?"):
            return declined(self)
        removed: list[Path] = []
        failed: list[dict[str, str]] = []
        for path in targets:
            try:
                path.unlink()
                removed.append(path)
            except OSError as exc:
                failed.append({"path": str(path), "error": exc.strerror or str(exc)})
        return self._clean_report(removed, failed, dry_run=False)

    def _clean_report(self, files: Sequence[Path], failed: Sequence[Mapping[str, str]], *,
                      dry_run: bool) -> int:
        payload = {"dry_run": dry_run, "count": len(files), "removed": [str(path) for path in files],
                   "failed": list(failed)}

        def text() -> None:
            self.out.result(f"{'would remove' if dry_run else 'removed'} {len(files)} sandbox artifact(s)")
            for path in files:
                self.out.info(f"  {path}")
            for item in failed:
                self.out.error(f"could not remove {item['path']}: {item['error']}")
        self.out.emit(payload, text=text)
        return int(ExitCode.FAILURE) if failed else int(ExitCode.OK)
