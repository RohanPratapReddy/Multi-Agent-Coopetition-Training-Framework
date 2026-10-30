"""mactf peers: the configured peers, their contracts, and weight previews. Section 8.17."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from pathlib import Path
from typing import Any

from ...exceptions import CapabilityUnavailable, CommandError, ComplianceError, MACTFError, ResourceNotFound
from ..group import GroupCommand, Sub
from ..output import Column

_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def _pair(kind: str, convert):
    def parse(text: str) -> tuple[str, Any]:
        name, separator, raw = text.partition("=")
        name = name.strip()
        if not separator or not _NAME.fullmatch(name):
            raise argparse.ArgumentTypeError(f"{text!r} is not NAME=VALUE (a {kind} for a named peer)")
        try:
            return name, convert(raw.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{text!r}: {exc}") from None
    return parse


def _finite(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("must be finite")
    return value


def _unit(raw: str) -> float:
    value = _finite(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError("must be between 0 and 1")
    return value


def _boolean(raw: str) -> bool:
    lowered = raw.lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise ValueError("must be true or false")


score_pair, confidence_pair, eligible_pair = _pair("score", _finite), _pair("confidence", _unit), _pair("flag", _boolean)


class Command(GroupCommand):
    help = "List the configured peers, describe a peer's contract, preview aggregation weights."
    category = "Swarm"
    output_formats = ("text", "json", "yaml")
    subcommands = (
        Sub("list", "List the configured peers.", aliases=("ls",), arguments=(),
            examples=("mactf peers list",)),
        Sub("show", "Describe one peer and its input/output contract.",
            arguments=((("name",), {"completer": "peer_name"}),), examples=("mactf peers show alpha",)),
        Sub("contract", "Print only a peer's input/output contract.",
            arguments=((("name",), {"completer": "peer_name"}),), examples=("mactf peers contract alpha",)),
        Sub("weights", "What would the swarm's weights be for these scores? Changes nothing.",
            arguments=((("--score",), {"action": "append", "type": score_pair, "default": None,
                                       "metavar": "NAME=VALUE", "completer": "peer_assignment"}),
                       (("--eligible",), {"action": "append", "type": eligible_pair, "default": None,
                                          "metavar": "NAME=BOOL", "completer": "peer_assignment",
                                          "help": "Default true for a peer not named."}),
                       (("--confidence",), {"action": "append", "type": confidence_pair, "default": None,
                                            "metavar": "NAME=0..1", "completer": "peer_assignment",
                                            "help": "Default 1.0 for a peer not named."}),
                       (("--scores-file",), {"default": None, "metavar": "FILE", "completer": "file",
                                             "help": "JSON or YAML with scores/eligible/confidence "
                                                     "(the API body); '-' reads standard input."}),
                       (("--advance",), {"action": "store_true", "default": False,
                                         "help": "Step the throwaway temperature controller once."})),
            examples=("mactf peers weights --score alpha=0.8 --score beta=0.6 --score gamma=0.2",
                      "mactf peers weights --scores-file scores.json --eligible gamma=false --format json")),
    )

    def _swarm(self) -> dict[str, Any]:
        from ...coopetition.introspect import build_swarm
        settings = self.need_settings()
        try:
            return build_swarm(settings)
        except ImportError as exc:
            raise CapabilityUnavailable(f"building peers needs torch: {exc}") from exc
        except (CommandError, MACTFError):
            raise
        except Exception as exc:  # noqa: BLE001 - the project's peer code failed, which is not our bug
            raise CommandError(f"could not build the configured peers: {type(exc).__name__}: {exc}") from exc

    def _find(self, name: str) -> Any:
        adapters = self._swarm()
        if name not in adapters:
            close = difflib.get_close_matches(name, list(adapters), n=3, cutoff=0.5)
            raise ResourceNotFound(f"no peer {name!r}; known peers: {sorted(adapters)}"
                                   + (f" (did you mean {', '.join(close)}?)" if close else ""))
        return adapters[name]

    def sub_list(self, **options):
        from ...coopetition.introspect import describe_peer
        items = [describe_peer(name, adapter, "settings") for name, adapter in self._swarm().items()]
        payload = {"source": "settings", "count": len(items), "items": items}
        self.out.emit(payload, text=lambda: self.out.table(items, (
            Column("name", "PEER"), Column("class", "CLASS"),
            Column("parameter_count", "PARAMS", align="r", fmt=lambda value: f"{value:,}"),
            Column("layout_hash", "LAYOUT")), empty="no peers configured"))
        return 0

    def sub_show(self, **options):
        from ...coopetition.introspect import describe_contract, describe_peer
        adapter = self._find(options["name"])
        payload = {**describe_peer(options["name"], adapter, "settings"), **describe_contract(adapter)}
        self.out.emit(payload, text=lambda: self._text(payload))
        return 0

    def sub_contract(self, **options):
        from ...coopetition.introspect import describe_contract
        payload = {"source": "settings", **describe_contract(self._find(options["name"]))}
        self.out.emit(payload, text=lambda: self._text(payload))
        return 0

    def _text(self, payload: dict[str, Any]) -> None:
        pairs = [(key, ", ".join(map(str, value)) if isinstance(value, list) else value)
                 for key, value in payload.items() if key != "notes"]
        self.out.kv(pairs)
        for note in payload.get("notes", []):
            self.out.result(f"  note: {note}")

    def sub_weights(self, **options):
        from ...coopetition.introspect import preview_weights
        body = self._read_body(options["scores_file"])
        scores = {**body["scores"], **dict(options["score"] or ())}
        if not scores:
            raise CommandError.usage("give at least one --score NAME=VALUE, or --scores-file")
        eligible = {**(body["eligible"] or {}), **dict(options["eligible"] or ())}
        confidence = {**(body["confidence"] or {}), **dict(options["confidence"] or ())}
        stray = sorted(({*eligible} | {*confidence}) - {*scores})
        if stray:
            raise CommandError.usage(f"eligible/confidence name peer(s) with no score: {', '.join(stray)}")
        eligible = {name: bool(eligible.get(name, True)) for name in scores} if eligible else None
        confidence = {name: float(confidence.get(name, 1.0)) for name in scores} if confidence else None
        advance = options["advance"] or body["advance"]
        try:
            report = preview_weights(self.need_settings(), scores, eligible=eligible, confidence=confidence,
                                     advance=advance)
        except ComplianceError as exc:
            raise CommandError.usage(f"these settings cannot weight {len(scores)} peer(s): {exc}") from exc
        payload = {"source": "settings",
                   "input": {"scores": scores, "eligible": eligible, "confidence": confidence,
                             "advance": advance},
                   "report": report}
        self.out.emit(payload, text=lambda: self._weights_text(payload))
        return 0

    def _read_body(self, source: str | None) -> dict[str, Any]:
        empty = {"scores": {}, "eligible": None, "confidence": None, "advance": False}
        if source is None:
            return empty
        import yaml
        try:
            raw = self.stdin.read() if source == "-" else Path(source).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise ResourceNotFound(f"{source}: {exc.strerror or exc}") from exc
        try:
            data = yaml.safe_load(raw) if source.lower().endswith((".yaml", ".yml")) else json.loads(raw)
        except (ValueError, yaml.YAMLError) as exc:
            raise CommandError.usage(f"{source}: not valid JSON/YAML: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("scores"), dict):
            raise CommandError.usage(f"{source}: expected an object with a 'scores' object")
        unknown = sorted(set(data) - set(empty))
        if unknown:
            raise CommandError.usage(f"{source}: unknown field(s) {', '.join(map(str, unknown))}; "
                                     "allowed: scores, eligible, confidence, advance")
        try:
            return {"scores": {str(k): _finite(str(v)) for k, v in data["scores"].items()},
                    "eligible": {str(k): bool(v) for k, v in (data.get("eligible") or {}).items()} or None,
                    "confidence": {str(k): _unit(str(v)) for k, v in (data.get("confidence") or {}).items()} or None,
                    "advance": bool(data.get("advance", False))}
        except (TypeError, ValueError) as exc:
            raise CommandError.usage(f"{source}: scores and confidence must be numbers "
                                     f"(confidence within 0..1): {exc}") from exc

    def _weights_text(self, payload: dict[str, Any]) -> None:
        report, given = payload["report"], payload["input"]
        rows = [{"peer": name, "score": given["scores"][name],
                 "eligible": (given["eligible"] or {}).get(name, True),
                 "confidence": (given["confidence"] or {}).get(name, 1.0),
                 "raw": report["raw_weights"].get(name), "weight": report["weights"].get(name)}
                for name in (report["order"] or list(given["scores"]))]
        self.out.table(rows, (Column("peer", "PEER"), Column("score", "SCORE", align="r", fmt=lambda v: f"{v:.4f}"),
                              Column("eligible", "ELIGIBLE", fmt=lambda v: "yes" if v else "no"),
                              Column("confidence", "CONF", align="r", fmt=lambda v: f"{v:.2f}"),
                              Column("raw", "RAW", align="r", fmt=lambda v: f"{v:.4f}"),
                              Column("weight", "WEIGHT", align="r", fmt=lambda v: f"{v:.4f}")))
        self.out.result(f"kappa {report['kappa']:.4f}  entropy {report['entropy']:.4f}"
                        + ("  FALLBACK to the reference weights" if report["fallback"] else ""))
        for label, names in (("gated", report["gated_peers"]), ("trimmed", report["trimmed_peers"])):
            if names:
                self.out.result(f"{label}: {', '.join(names)}")
        for line in report["adjustments"]:
            self.out.result(f"  {line}")
