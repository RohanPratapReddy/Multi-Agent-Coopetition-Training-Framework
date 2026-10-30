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
from ..structured import read_structured

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
        Sub("mutate", "Simulate a peer-tree mutation plan, and optionally queue it on a run.",
            arguments=((("plan",), {"metavar": "FILE", "completer": "file",
                                    "help": "A MutationPlan as JSON or YAML; '-' reads standard input."}),
                       (("--run",), {"default": None, "metavar": "RUN_ID", "completer": "run_id",
                                     "help": "Simulate against this run's live tree instead of the settings."}),
                       (("--apply",), {"action": "store_true", "default": False,
                                       "help": "Queue the plan on --run for the next epoch boundary."}),
                       (("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "Simulate only, even with --apply; queue nothing."})),
            examples=("mactf peers mutate plan.yaml",
                      "mactf peers mutate plan.yaml --run run-0007 --apply")),
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

    # -- mutate ---------------------------------------------------------------
    def sub_mutate(self, **options):
        if options["apply"] and not options["run"]:
            raise CommandError.usage("--apply needs --run: a plan is queued on a run, not on the settings")
        raw = read_structured(self, options["plan"], root=(dict, list))
        if options["run"] is None:
            report = self._simulate_on_settings(raw)
            return self._mutation_result(report, queued=None)
        from ...orchestration.live import manager_for_cli
        from ...orchestration.runview import resolve_run_id

        registry = self.registry_for_read()
        if registry is None:
            raise ResourceNotFound("no runs recorded")
        run_id = resolve_run_id(registry, options["run"])
        manager = manager_for_cli(self.need_settings(), registry)
        report = manager.simulate_mutation(run_id, raw).as_dict()      # RunStateError -> exit 4
        queued = None
        if options["apply"] and not options["dry_run"] and report["ok"]:
            queued = manager.request_mutation(run_id, raw, source="cli")
        return self._mutation_result(report, queued=queued)

    def _simulate_on_settings(self, raw: Any) -> dict[str, Any]:
        """Run a plan against a flat tree of the configured peers: nothing is built, nothing trains."""
        from ...coopetition.aggregation import feasible_bounds
        from ...coopetition.tree import PeerTree
        from ...orchestration.live import MutationSimulation
        from ...training.mutation import MutationLimits, MutationPlan, TopologySimulator

        settings = self.need_settings()
        names = [str(entry["name"]) for entry in (settings.get("PEERS") or [])
                 if isinstance(entry, dict) and entry.get("name")]
        if not names:
            raise CommandError.usage("the settings configure no peers; pass --run to simulate on a run's tree")
        coopetition = settings.get("COOPETITION") or {}
        floor, cap = float(coopetition.get("WEIGHT_FLOOR", 0.02)), float(coopetition.get("WEIGHT_CAP", 0.6))
        tree = PeerTree.flat(names)
        limits = MutationLimits.from_settings((settings.get("TRAINER") or {}).get("MUTATION"),
                                              initial_roots=len(names), initial_nodes=len(names))
        plan = MutationPlan.parse(raw)
        simulator = TopologySimulator(tree, retired=(), tombstones=(), limits=limits,
                                      feasible=lambda count: feasible_bounds(count, floor=floor, cap=cap))
        return MutationSimulation(plan, simulator.run(plan)).as_dict()

    def _mutation_result(self, report: dict[str, Any], *, queued: dict[str, Any] | None) -> int:
        payload = {**report, "queued": queued}
        self.out.emit(payload, text=lambda: self._mutation_text(payload))
        return 0 if report["ok"] else 1

    def _mutation_text(self, payload: dict[str, Any]) -> None:
        label = payload.get("label") or "plan"
        digest = str(payload.get("digest", ""))[:12]
        self.out.result(f"plan {label} (digest {digest}): {'applicable' if payload['ok'] else 'refused'}")
        for item in payload["hard"]:
            self.out.result(f"  {item['code']:7} {item.get('path') or ''} {item['message']}".rstrip())
        for item in payload["soft"]:
            self.out.result(f"  {item['code']:7} {item.get('path') or ''} {item['message']}".rstrip())
        for verb, key in (("add", "added"), ("retire", "retired"), ("revive", "revived"),
                          ("purge", "purged"), ("modify", "modified")):
            for path in payload.get(key) or []:
                self.out.result(f"  ok      {verb} {path}")
        if payload["queued"]:
            self.out.result(f"queued {label} for the next epoch boundary (digest {digest})")
