"""mactf config: the effective settings, and where each value came from. Section 8.15."""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any, Collection, Mapping, Sequence

from ...conf import iter_leaves, load_settings
from ...conf_files import resolve_conf_dir
from ...exceptions import CommandError, ResourceNotFound
from ..exitcodes import ExitCode
from ..group import GroupCommand, Sub
from ..output import Column, clean

MASK = "********"
_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_KEY = (("key",), {"metavar": "KEY", "completer": "setting_key",
                   "help": "A dotted path such as TRAINING.EPOCHS."})


def secret_fragments(settings: Any) -> tuple[str, ...]:
    """Name fragments that mark a value secret: the logger's own redaction list."""
    from ...training.logging import DEFAULT_REDACT
    logger = settings.get("LOGGER")
    configured = logger.get("REDACT") if isinstance(logger, Mapping) else None
    return tuple(dict.fromkeys(
        str(item).lower() for item in (*DEFAULT_REDACT, *(configured or ())) if str(item)))


def _is_secret(name: str, fragments: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in fragments)


def redact(value: Any, fragments: Sequence[str], *, name: str = "") -> Any:
    """Mask secrets in a settings tree (rules 1 and 2 of section 8.15)."""
    secret = _is_secret(name, fragments)
    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        for index, (key, item) in enumerate(value.items(), start=1):
            out[f"<redacted {index}>" if secret else key] = redact(
                item, fragments, name=name if secret else str(key))
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item, fragments, name=name) for item in value]
    if secret and isinstance(value, str) and value:
        return MASK
    return value


def strip_private(tree: Mapping[str, Any]) -> dict[str, Any]:
    return {key: strip_private(item) if isinstance(item, Mapping) else item
            for key, item in tree.items() if not str(key).startswith("_")}


def select_leaves(tree: Mapping[str, Any], keep: Collection[str]) -> dict[str, Any]:
    """A tree holding only the leaves whose dotted path is in ``keep``; paths are never re-split."""
    out: dict[str, Any] = {}
    for path, value in iter_leaves(tree):
        if ".".join(path) in keep:
            node = out
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = value
    return out


def lookup(tree: Any, key: str) -> tuple[tuple[str, ...], Any]:
    """``(canonical path, value)`` for a dotted ``key``; ``ResourceNotFound`` names the nearest keys."""
    segments = key.split(".")
    for segment in segments:
        if not _SEGMENT.fullmatch(segment) or segment.startswith("_"):
            raise CommandError.usage(f"{key!r} is not a setting path; use dotted names such as TRAINING.EPOCHS")
    current, walked = tree, []
    for segment in segments:
        where = ".".join(walked) or "the settings"
        if not isinstance(current, Mapping):
            raise ResourceNotFound(f"{where} is a value, not a table; it has no key {segment!r}")
        if segment in current:
            match = segment
        else:
            folded = [name for name in current if str(name).lower() == segment.lower()]
            match = folded[0] if len(folded) == 1 else None          # ambiguous case-folding is no match
        if match is None:
            close = difflib.get_close_matches(segment.upper(), [str(k) for k in current], n=3, cutoff=0.5)
            hint = f"; did you mean {', '.join('.'.join([*walked, name]) for name in close)}?" if close else ""
            raise ResourceNotFound(f"{where} has no key {segment!r}{hint}")
        walked.append(str(match))
        current = current[match]
    return tuple(walked), current


def origin_for(origins: Mapping[str, str], path: Sequence[str]) -> str:
    """The origin of a leaf; a masked or synthetic path falls back to its nearest known ancestor."""
    for size in range(len(path), 0, -1):
        dotted = ".".join(path[:size])
        if dotted in origins:
            return origins[dotted]
        prefix = dotted + "."
        for candidate, origin in origins.items():
            if candidate.startswith(prefix):
                return origin
    return "unknown"


def plain(value: Any) -> str:
    """One value as a script wants it: strings bare, JSON for everything else."""
    if isinstance(value, str):
        return value
    return json.dumps(clean(value), ensure_ascii=False)


_KEY_OPTIONAL = (_KEY[0], {**_KEY[1], "nargs": "?", "default": None})


class Command(GroupCommand):
    help = "Show the effective settings and where each value came from."
    category = "Project"
    requires_settings = False
    output_formats = ("text", "json", "yaml")
    subcommands = (
        Sub("show", "Print the effective settings (or one table of them).",
            arguments=(_KEY_OPTIONAL,
                       (("--origin",), {"action": "store_true", "default": False,
                                        "help": "Say which layer set each value."}),
                       (("--changed",), {"action": "store_true", "default": False,
                                         "help": "Only values that differ from the framework defaults."}),
                       (("--reveal",), {"action": "store_true", "default": False,
                                        "help": "Do not mask secrets."})),
            examples=("mactf config show TRAINING", "mactf config show --changed --origin",
                      "mactf --set TRAINING.EPOCHS=2 config show TRAINING.EPOCHS --origin")),
        Sub("get", "Print one value, bare, for use in scripts.",
            arguments=(_KEY, (("--origin",), {"action": "store_true", "default": False,
                                              "help": "Print the originating layer after a tab."}),
                       (("--reveal",), {"action": "store_true", "default": False,
                                        "help": "Do not mask secrets."})),
            examples=("EPOCHS=$(mactf config get TRAINING.EPOCHS)",)),
        Sub("files", "List the layers settings were built from, lowest first.", arguments=(),
            examples=("mactf --config prod.yaml config files",)),
        Sub("validate", "Load and validate the settings; exit 2 if they do not load.",
            arguments=((("--strict",), {"action": "store_true", "default": False,
                                        "help": "Treat configuration notes as errors."}),),
            examples=("mactf --settings myproject.settings config validate",
                      "mactf --config overrides.yaml config validate --strict")),
    )

    def _traced(self, settings: Any) -> Any:
        if not settings.module:
            raise CommandError.usage("provenance needs a settings module; these settings were built in code")
        boot = self.bootstrap
        return load_settings(settings.module, overrides=boot.overrides if boot else (),
                             config_files=boot.config if boot else (), trace=True)

    def _view(self, settings: Any, *, key: str | None, origin: bool, changed: bool, reveal: bool):
        source = self._traced(settings) if (origin or changed) else settings
        origins = dict(source["_ORIGINS"]) if source is not settings else None
        tree = strip_private(source.as_dict())
        if changed:
            tree = select_leaves(tree, {path for path, layer in origins.items() if layer != "default"})
        if reveal:
            self.out.warn("--reveal: secrets are shown in clear text")
        else:
            tree = redact(tree, secret_fragments(settings))
        path, value = ((), tree) if key is None else lookup(tree, key)
        shown = None if origins is None else {".".join(leaf): origin_for(origins, leaf)
                                              for leaf, _ in iter_leaves(value, path)}
        return path, value, shown

    def sub_show(self, **options):
        settings = self.need_settings()
        path, value, origins = self._view(settings, key=options["key"], origin=options["origin"],
                                          changed=options["changed"], reveal=options["reveal"])
        payload = {"settings_module": settings.module, "key": ".".join(path) or None,
                   "value": value, "origins": origins}

        def text() -> None:
            if origins is None:
                if isinstance(value, Mapping):
                    import yaml
                    self.out.result(yaml.safe_dump(clean(value), sort_keys=False, allow_unicode=True,
                                                   default_flow_style=False).rstrip() or "{}")
                else:
                    self.out.result(plain(value))
                return
            rows = [(".".join(leaf), json.dumps(clean(item), ensure_ascii=False), origins[".".join(leaf)])
                    for leaf, item in iter_leaves(value, path)]
            width = min(max((len(f"{name} = {shown}") for name, shown, _ in rows), default=0), 72)
            for name, shown, origin in rows:
                self.out.result(f"{f'{name} = {shown}':<{width}}  # {origin}")
            if not rows:
                self.out.result("nothing differs from the defaults" if options["changed"] else "(empty)")
        self.out.emit(payload, text=text)
        return 0

    def sub_get(self, **options):
        settings = self.need_settings()
        path, value, origins = self._view(settings, key=options["key"], origin=options["origin"],
                                          changed=False, reveal=options["reveal"])
        if origins is not None and len(origins) != 1:
            raise CommandError.usage(
                f"--origin needs one value, and {options['key']} holds {len(origins)}; "
                f"use `{self.prog} config show {options['key']} --origin`")
        origin = next(iter(origins.values())) if origins else None
        payload = {"settings_module": settings.module, "key": ".".join(path), "value": value, "origin": origin}
        self.out.emit(payload, text=lambda: self.out.result(
            plain(value) + (f"\t{origin}" if origin is not None else "")))
        return 0

    def sub_files(self, **options):
        settings = self.need_settings()
        traced = self._traced(settings)
        layers = [dict(layer) for layer in traced["_LAYERS"]]
        conf_dir = resolve_conf_dir(settings, settings.path("ROOT"))
        payload = {"settings_module": settings.module, "conf_dir": str(conf_dir) if conf_dir else None,
                   "layers": layers}

        def text() -> None:
            self.out.table(layers, [Column("label", "LAYER"), Column("leaves", "VALUES", align="r"),
                                    Column("path", "FILE")], empty="no layers")
            if conf_dir:
                self.out.info(f"conf directory: {conf_dir}")
        self.out.emit(payload, text=text)
        return 0

    def sub_validate(self, **options):
        error = self.settings_error
        errors = [str(error)] if error is not None else []
        notes = [] if error is not None else [str(note) for note in (self.settings.get("_CONF_NOTES") or [])]
        if options["strict"]:
            errors.extend(f"note (strict): {note}" for note in notes)
        module = (self.bootstrap.settings if self.bootstrap else None) or os.environ.get("MACTF_SETTINGS_MODULE")
        payload = {"ok": not errors, "settings_module": module, "errors": errors, "notes": notes}

        def text() -> None:
            for line in errors:
                self.out.error(line)
            for line in notes:
                self.out.info(f"note: {line}")
            self.out.result(f"settings {module or '(framework defaults)'}: "
                            + ("valid" if not errors else f"{len(errors)} problem(s)"))
        self.out.emit(payload, text=text)
        return int(ExitCode.OK if not errors else ExitCode.USAGE)
