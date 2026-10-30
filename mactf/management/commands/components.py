"""``mactf components``: list, inspect, validate, scaffold and diff components (document 17 section 16.1)."""

from __future__ import annotations

import argparse
import json
import keyword
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ...exceptions import (
    CommandError,
    ComponentError,
    ComponentMissing,
    ResourceNotFound,
)
from ..group import GroupCommand, Sub
from ..output import Column
from ..structured import read_structured

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PARAM = re.compile(r"([A-Za-z_][A-Za-z0-9_.-]*)=(.*)")

#: A file with one of these suffixes is read as a manifest; any other file is a checkpoint.
_TEXT_SUFFIXES = (".json", ".yaml", ".yml")


def param_pair(text: str) -> tuple[str, Any]:
    """``K=V`` with ``V`` parsed as JSON when it is, else kept as a string."""
    match = _PARAM.fullmatch(text)
    if match is None:
        raise argparse.ArgumentTypeError(f"{text!r} is not KEY=VALUE")
    key, raw = match.groups()
    try:
        return key, json.loads(raw)
    except ValueError:
        return key, raw


#: ``kind`` for scaffold: the kinds a person writes code for (peers have ``startpeer``).
SCAFFOLD_KINDS = (
    "scorer", "reward_term", "weight_update", "peer_strategy", "rl_function", "optimizer",
    "scheduler", "trainer_policy",
)


class Command(GroupCommand):
    help = "List, inspect, validate, scaffold and diff the pluggable training components."
    category = "Swarm"
    output_formats = ("text", "json", "yaml")
    subcommands = (
        Sub("list", "List every visible component, optionally of one kind.", aliases=("ls",),
            arguments=((("--kind",), {"default": None, "metavar": "KIND", "completer": "component_kind",
                                      "help": "Only this kind, e.g. scorer or rl_function.loss."}),
                       (("--origin",), {"default": None, "choices": ("builtin", "plugin", "project", "runtime"),
                                        "help": "Only components registered by this origin."}),
                       (("--locked",), {"action": "store_true", "default": False,
                                        "help": "Only components matched by COMPONENTS.LOCKED."})),
            examples=("mactf components list", "mactf components list --kind rl_function.loss --origin project",
                      "mactf components list --format json")),
        Sub("inspect", "Show one component: origin, mutability, parameters, contract and fingerprint.",
            arguments=((("kind",), {"completer": "component_kind"}),
                       (("name",), {"completer": "component_name"})),
            examples=("mactf components inspect weight_update kappa_softmax",
                      "mactf components inspect rl_function.loss policy_gradient --format yaml")),
        Sub("validate", "Check a component spec against its contract without starting a run.",
            arguments=((("--kind",), {"required": True, "metavar": "KIND", "completer": "component_kind"}),
                       (("--name",), {"default": None, "metavar": "NAME", "completer": "component_name",
                                      "help": "The registered name; omit it when --import is given."}),
                       (("--import", "--import-path"), {"dest": "import_path", "default": None,
                                                        "metavar": "MODULE:ATTR",
                                                        "help": "Check an importable factory instead of a name."}),
                       (("--param",), {"action": "append", "type": param_pair, "default": None,
                                       "metavar": "KEY=VALUE",
                                       "help": "A parameter; repeat it. VALUE is JSON when it parses."}),
                       (("--spec-file",), {"default": None, "metavar": "FILE", "completer": "file",
                                           "help": "A JSON or YAML spec; '-' reads standard input."}),
                       (("--probe",), {"action": "store_true", "default": False,
                                       "help": "Also build a throwaway instance and run the author probe."})),
            examples=("mactf components validate --kind weight_update --name trimmed_mean --param trim=0.2",
                      "mactf components validate --kind rl_function.loss --import myproj.losses:Huber --probe")),
        Sub("scaffold", "Write a ready-to-edit component module for a kind.", destructive=True,
            arguments=((("kind",), {"choices": SCAFFOLD_KINDS}),
                       (("name",), {"help": "The registered name; also the module name (snake_case)."}),
                       (("--dir",), {"default": None, "metavar": "PATH", "completer": "directory",
                                     "help": "Base directory; defaults to <project>/components."}),
                       (("--force",), {"action": "store_true", "default": False,
                                       "help": "Overwrite an existing file."}),
                       (("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "Print what would be written and write nothing."})),
            examples=("mactf components scaffold weight_update my_trimmed_mean",
                      "mactf components scaffold scorer carbon_cost --dir src/mine --dry-run")),
        Sub("diff", "Compare two component manifests: runs, checkpoints, or manifest files.",
            arguments=((("left",), {"metavar": "A", "completer": "run_id",
                                    "help": "A run id, a checkpoint file, or a manifest file."}),
                       (("right",), {"metavar": "B", "completer": "run_id",
                                     "help": "The other side; 'live' means the current project settings."}),
                       (("--structural",), {"action": "store_true", "default": False,
                                            "help": "Hide changes that cannot alter behaviour (a version bump)."})),
            examples=("mactf components diff run-0007 run-0009", "mactf components diff run-0007 live --structural")),
    )

    # -- loading ---------------------------------------------------------
    def _loaded(self) -> Any:
        """The settings, with every module ``COMPONENTS`` names imported and every built-in registered."""
        from ...components import ensure_components_loaded
        from ...components.known import import_builtins

        settings = self.need_settings()
        import_builtins()
        ensure_components_loaded(settings)
        return settings

    @staticmethod
    def _kind(kind_id: str) -> str:
        from ...components import all_kind_ids

        if kind_id not in all_kind_ids():
            raise ResourceNotFound(f"no component kind {kind_id!r}; kinds: {sorted(all_kind_ids())}")
        return kind_id

    # -- list / inspect ----------------------------------------------------
    def sub_list(self, **options):
        from ...components import HUB

        self._loaded()
        kind = options.get("kind")
        if kind:
            self._kind(kind)
        rows = []
        for kind_id, entries in HUB.snapshot().items():
            if kind and kind != kind_id:
                continue
            for name, info in entries.items():
                if options.get("origin") and info["origin"] != options["origin"]:
                    continue
                if options.get("locked") and not info.get("locked"):
                    continue
                rows.append({"kind": kind_id, "name": name, **{k: v for k, v in info.items()
                                                                if k not in ("kind", "name")}})
        payload = {"count": len(rows), "items": rows}
        self.out.emit(payload, text=lambda: self.out.table(rows, (
            Column("kind", "KIND"), Column("name", "NAME"), Column("origin", "ORIGIN"),
            Column("mutability", "MUTABLE"),
            Column("locked", "LOCKED", fmt=lambda value: "yes" if value else "no"),
            Column("fingerprint", "FINGERPRINT", fmt=lambda value: str(value)[:12])), empty="no components"))
        return 0

    def sub_inspect(self, **options):
        from ...components import HUB
        from ...components.kinds import KIND_SPECS

        self._loaded()
        kind_id = self._kind(options["kind"])
        try:
            component = HUB.registry(kind_id).get(options["name"])
        except ComponentMissing as exc:
            raise ResourceNotFound(str(exc)) from exc
        problems = KIND_SPECS[kind_id].contract(component.factory)
        payload = {**component.describe(), "locked": HUB.effective_policy().is_locked(kind_id, component.name),
                   "fingerprint": component.fingerprint_text(),
                   "contract": "ok" if not problems else problems,
                   "unsafe": bool(getattr(component.factory, "unsafe", False))}
        self.out.emit(payload, text=lambda: self._inspect_text(payload))
        return 0

    def _inspect_text(self, payload: dict[str, Any]) -> None:
        params = payload.get("params") or {}
        pairs = [(key, payload[key]) for key in ("kind", "name", "origin", "mutability", "version", "class",
                                                 "module", "description") if payload.get(key) not in (None, "")]
        pairs.append(("params", ", ".join(f"{key} ({spec})" for key, spec in params.items())
                      or ", ".join(payload.get("kwargs") or []) or "none"))
        pairs.append(("contract", payload["contract"] if payload["contract"] == "ok"
                      else "; ".join(payload["contract"])))
        pairs += [("locked", "yes" if payload["locked"] else "no"), ("unsafe", "yes" if payload["unsafe"] else "no"),
                  ("fingerprint", payload["fingerprint"])]
        self.out.kv(pairs)

    # -- validate -----------------------------------------------------------
    def sub_validate(self, **options):
        from ...components import HUB, ComponentResolver, ComponentSpec
        from ...components.chain_validation import finding_of, injected_offer, probe_instance

        settings = self._loaded()
        kind_id = self._kind(options["kind"])
        raw = self._raw_spec(options)
        findings: list[dict[str, Any]] = []
        try:
            spec = ComponentSpec.parse(kind_id, raw)
            resolver = ComponentResolver()
            # A CLI user is the operator: the import surface is trusted (5.4).
            component = resolver.validate(spec, trusted=True, inject=injected_offer(kind_id))
        except ComponentMissing as exc:
            raise ResourceNotFound(str(exc)) from exc
        except ComponentError as exc:                   # spec, param, contract and import errors
            findings.append(finding_of(exc))
        else:
            findings.extend(self._advisories(component, kind_id, settings))
            if options["probe"]:
                findings.extend(probe_instance(resolver, spec))
        failed = any(item["severity"] == "error" for item in findings)
        payload = {"kind": kind_id, "spec": raw, "ok": not failed, "findings": findings}
        self.out.emit(payload, text=lambda: self.out.result(self._render_findings(payload)))
        return 1 if failed else 0

    def _raw_spec(self, options: dict[str, Any]) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        if options.get("spec_file"):
            raw = dict(read_structured(self, options["spec_file"]))
        if options.get("name"):
            raw["type"] = options["name"]
        if options.get("import_path"):
            raw["import"] = options["import_path"]
        if options.get("param"):
            raw.setdefault("params", {}).update(dict(options["param"]))
        if not raw:
            raise CommandError.usage("give --name, --import or --spec-file")
        return raw

    @staticmethod
    def _advisories(component: Any, kind_id: str, settings: Any) -> list[dict[str, Any]]:
        """Warnings that are not failures: ``W004`` for an unsafe policy (7.7)."""
        found = []
        if getattr(component.factory, "unsafe", False):
            found.append({"code": "W004", "severity": "warning",
                          "message": f"{kind_id}:{component.name} is declared unsafe; it needs "
                                     "TRAINER.ALLOW_UNSAFE_POLICY to run"})
        return found

    @staticmethod
    def _render_findings(payload: dict[str, Any]) -> str:
        head = "ok" if payload["ok"] else "invalid"
        lines = [f"{payload['kind']}: {head}"]
        lines += [f"  {item['severity'].upper():7} {item['code']}  {item['message']}" for item in payload["findings"]]
        return "\n".join(lines)

    # -- scaffold -----------------------------------------------------------
    def sub_scaffold(self, **options):
        from ...components.scaffold import render_scaffold

        name = options["name"]
        if not _IDENT.fullmatch(name) or keyword.iskeyword(name):
            raise CommandError.usage(f"{name!r} cannot be a module name; use letters, digits and underscores")
        base = Path(options["dir"]).expanduser() if options["dir"] else self.project_root() / "components"
        target = (base / f"{name}.py").resolve()
        root = base.resolve()
        if root != target and root not in target.parents:
            raise CommandError.usage(f"{name!r} would write outside {root}")
        body = render_scaffold(options["kind"], name)
        payload = {"path": str(target), "kind": options["kind"], "name": name,
                   "bytes": len(body.encode("utf-8")), "dry_run": bool(options["dry_run"])}
        if options["dry_run"]:
            self.out.emit(payload, text=lambda: self.out.result(f"would write {target}\n\n{body}"))
            return 0
        if target.exists() and not options["force"]:
            raise CommandError.conflict(f"{target} exists; pass --force to overwrite")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8", newline="\n")
        self.out.emit(payload, text=lambda: self.out.result(
            f"wrote {target}\nadd {name!r}'s module to COMPONENTS.MODULES to load it"))
        return 0

    def project_root(self) -> Path:
        """The project directory: ``PATHS.ROOT`` of the loaded settings."""
        return self.need_settings().path("ROOT")

    # -- diff ---------------------------------------------------------------
    def sub_diff(self, **options):
        from ...components.fingerprint import diff, render

        self._loaded()
        left, right = self._manifest(options["left"]), self._manifest(options["right"])
        changes = diff(left, right)
        if options["structural"]:
            changes = [change for change in changes if change.structural]
        payload = {"left": options["left"], "right": options["right"], "same": not changes,
                   "changes": [{"kind": c.kind, "slot": c.slot, "field": c.field, "saved": c.saved,
                                "live": c.live, "structural": c.structural} for c in changes]}
        self.out.emit(payload, text=lambda: self.out.result(render(changes) or "no differences"))
        return 0 if not changes else 1

    def _manifest(self, argument: str) -> dict[str, Any]:
        """A manifest from, in order: the word ``live``, a file, a checkpoint, a run id."""
        if argument == "live":
            return self._live_manifest()
        path = Path(argument).expanduser()
        if path.is_file():
            return self._file_manifest(path)
        from ...orchestration.runview import resolve_run_id

        registry = self.registry_for_read()
        if registry is not None:
            try:
                run_id = resolve_run_id(registry, argument)
            except (ResourceNotFound, LookupError):
                run_id = None
            if run_id is not None:
                return self._run_manifest(registry, run_id)
        raise ResourceNotFound(f"{argument!r} is not 'live', a manifest or checkpoint file, or a run id")

    def _file_manifest(self, path: Path) -> dict[str, Any]:
        if path.suffix.lower() in _TEXT_SUFFIXES:
            data = read_structured(self, str(path))
        else:
            from ...runtime import load_checkpoint

            try:
                data = load_checkpoint(path)
            except Exception as exc:  # noqa: BLE001 - any unreadable checkpoint is the same answer
                raise CommandError.usage(f"{path}: not a readable checkpoint: {exc}") from exc
        return self._as_manifest(data, str(path))

    @staticmethod
    def _as_manifest(data: Any, where: str) -> dict[str, Any]:
        identity = data.get("identity") if isinstance(data, dict) else None
        if isinstance(identity, dict):
            data = identity.get("components") or {}
        elif isinstance(data, dict) and isinstance(data.get("components"), dict):
            data = data["components"]
        if not isinstance(data, dict) or not all(isinstance(entries, dict) for entries in data.values()):
            raise CommandError.usage(f"{where}: expected a manifest of {{kind: {{slot: entry}}}}")
        return data

    @staticmethod
    def _run_manifest(registry: Any, run_id: str) -> dict[str, Any]:
        from ...runtime import load_checkpoint

        for row in reversed(registry.checkpoints(run_id)):
            path = Path(row["path"])
            if not path.exists():
                continue
            try:
                identity = (load_checkpoint(path) or {}).get("identity") or {}
            except Exception:  # noqa: BLE001 - try the next-newest file
                continue
            if identity.get("components"):
                return dict(identity["components"])
        return dict(registry.components(run_id) or {})

    def _live_manifest(self) -> dict[str, Any]:
        """The manifest the current settings would start a run with (what is not a built-in)."""
        from ...components import ComponentResolver, ComponentSpec
        from ...components.chain_validation import injected_offer
        from ...components.fingerprint import entry_for, records_in_checkpoint
        from ...components.known import configured_specs

        resolver = ComponentResolver()
        manifest: dict[str, dict[str, Any]] = {}
        for slot, kind_id, raw in configured_specs(self.need_settings()):
            try:
                spec = ComponentSpec.parse(kind_id, raw)
                component = resolver.validate(spec, trusted=True, inject=injected_offer(kind_id))
            except ComponentError:
                continue                                  # `components validate` reports it
            if not records_in_checkpoint(component):
                continue
            resolved = SimpleNamespace(component=component, spec=spec, spec_hash=spec.hash(), instance=None)
            manifest.setdefault(kind_id, {})[slot] = entry_for(
                SimpleNamespace(path=slot, kind_id=kind_id, resolved=resolved))
        return {kind: dict(sorted(entries.items())) for kind, entries in sorted(manifest.items())}
