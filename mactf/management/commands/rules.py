"""mactf rules: the reward and penalty catalog. Section 8.16."""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import textwrap
from typing import Any

from ...exceptions import ResourceNotFound
from ...rules import ALL_RULES, CATALOG, TIERS, Rule, validate_catalog
from ...rules.severity import Severity
from ..exitcodes import ExitCode
from ..group import GroupCommand, Sub
from ..output import Column

BUCKETS = ("r_task", "r_epi", "r_collab", "r_align", "penalty")
_INDEX = {rule.id: position for position, rule in enumerate(ALL_RULES)}


def severity_arg(text: str) -> Severity:
    lowered = text.strip().lower()
    for member in Severity:
        if lowered == member.name.lower():
            return member
    found = re.fullmatch(r"s?([0-4])", lowered)
    if found:
        return Severity(int(found.group(1)))
    raise argparse.ArgumentTypeError(
        f"{text!r} is not a severity; use S0-S4 or none, minor, moderate, major, severe")


def tier_arg(text: str) -> int:
    lowered = text.strip().lower()
    found = re.fullmatch(r"t?([0-9]+)", lowered)
    if found and int(found.group(1)) in TIERS:
        return int(found.group(1))
    for number, name in TIERS.items():
        if lowered == name:
            return number
    raise argparse.ArgumentTypeError(
        f"{text!r} is not a tier; use T0-T{max(TIERS)} or one of {', '.join(TIERS.values())}")


def rule_dict(rule: Rule) -> dict[str, Any]:
    return {"id": rule.id, "name": rule.name, "kind": rule.kind, "tier": rule.tier,
            "tier_name": rule.tier_name, "severity": rule.default_severity.label,
            "severity_name": rule.default_severity.name.lower(), "bucket": rule.bucket,
            "description": rule.description, "warning_eligible": rule.warning_eligible,
            "non_offsettable": rule.non_offsettable, "omega": rule.omega, "tags": sorted(rule.tags)}


def flags_of(rule: Rule) -> str:
    return ",".join(label for label, on in (("warn", rule.warning_eligible),
                                            ("non-offsettable", rule.non_offsettable),
                                            ("omega", rule.omega)) if on) or "-"


def matches(rule: Rule, options: dict[str, Any]) -> bool:
    if options["kind"] and rule.kind != options["kind"]:
        return False
    if options["tier"] and rule.tier not in options["tier"]:
        return False
    if options["severity"] and rule.default_severity not in options["severity"]:
        return False
    if options["bucket"] and rule.bucket != options["bucket"]:
        return False
    if options["tag"] and not {tag.lower() for tag in options["tag"]} <= {tag.lower() for tag in rule.tags}:
        return False                                           # every tag asked for
    if options["warning_eligible"] and not rule.warning_eligible:
        return False
    if options["non_offsettable"] and not rule.non_offsettable:
        return False
    if options["omega"] and not rule.omega:
        return False
    needle = (options["search"] or "").lower()
    return not needle or needle in f"{rule.id} {rule.name} {rule.description}".lower()


SORTS = {
    "id": lambda rule: _INDEX[rule.id],
    "tier": lambda rule: (rule.tier, _INDEX[rule.id]),
    "severity": lambda rule: (-int(rule.default_severity), _INDEX[rule.id]),   # worst first
    "name": lambda rule: (rule.name.lower(), _INDEX[rule.id]),
}

RULE_COLUMNS = (
    Column("id", "ID"), Column("kind", "KIND"),
    Column("tier", "TIER", fmt=lambda value: f"T{value}"), Column("severity", "SEV"),
    Column("bucket", "BUCKET"), Column("flags", "FLAGS"), Column("name", "NAME"),
)


class Command(GroupCommand):
    help = "List and explain the reward and penalty rules."
    category = "Swarm"
    requires_settings = False
    output_formats = ("text", "json", "yaml")
    subcommands = (
        Sub("list", "List rules, filtered.", aliases=("ls",),
            arguments=((("--kind",), {"choices": ["reward", "penalty"], "default": None}),
                       (("--tier",), {"action": "append", "type": tier_arg, "default": None, "metavar": "TIER"}),
                       (("--severity",), {"action": "append", "type": severity_arg, "default": None,
                                          "metavar": "SEVERITY"}),
                       (("--bucket",), {"choices": list(BUCKETS), "default": None}),
                       (("--tag",), {"action": "append", "default": None, "completer": "rule_tag"}),
                       (("--warning-eligible",), {"action": "store_true", "default": False,
                                                  "help": "Only rules whose first S1 occurrence is a warning."}),
                       (("--non-offsettable",), {"action": "store_true", "default": False,
                                                 "help": "Only penalties no reward can offset."}),
                       (("--omega",), {"action": "store_true", "default": False,
                                       "help": "Only rules that trigger the Omega amplifier."}),
                       (("--search",), {"default": None, "help": "Substring of id, name, or description."}),
                       (("--sort",), {"choices": list(SORTS), "default": "id"})),
            examples=("mactf rules list --kind penalty --severity major --severity severe",
                      "mactf rules list --tag honesty --format json",
                      "mactf rules list --non-offsettable")),
        Sub("show", "Describe one or more rules.",
            arguments=((("rule",), {"nargs": "+", "completer": "rule_id", "metavar": "RULE",
                                    "help": "Rule ids such as R2 or P20 (case-insensitive)."}),),
            examples=("mactf rules show P20 P21", "mactf rules show r2 --format json")),
        Sub("tiers", "List the precedence tiers and the rules in each.", arguments=(),
            examples=("mactf rules tiers",)),
        Sub("validate", "Check the catalog's own integrity; exit 1 if it is broken.", arguments=(),
            examples=("mactf rules validate",)),
    )

    def sub_list(self, **options):
        rules = sorted((rule for rule in ALL_RULES if matches(rule, options)), key=SORTS[options["sort"]])
        items = [{**rule_dict(rule), "flags": flags_of(rule)} for rule in rules]
        payload = {"count": len(items), "total": len(ALL_RULES), "items": [rule_dict(rule) for rule in rules]}
        self.out.emit(payload, text=lambda: self.out.table(items, RULE_COLUMNS, empty="no rules match"))
        return 0

    def sub_show(self, **options):
        wanted = [name.upper() for name in options["rule"]]
        unknown = [name for name in dict.fromkeys(wanted) if name not in CATALOG]
        if unknown:
            hints = []
            for name in unknown:
                close = difflib.get_close_matches(name, list(CATALOG), n=3, cutoff=0.5)
                hints.append(f"{name!r}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
            raise ResourceNotFound("unknown rule " + ", ".join(hints)
                                   + "; rewards are R1-R17 and penalties P1-P23")
        rules = [CATALOG[name] for name in dict.fromkeys(wanted)]
        payload = {"count": len(rules), "items": [rule_dict(rule) for rule in rules]}

        def text() -> None:
            width = min(shutil.get_terminal_size((100, 24)).columns, 100) - 4
            for position, rule in enumerate(rules):
                if position:
                    self.out.result("")
                self.out.result(f"{rule.id}  {rule.name}")
                self.out.kv([("kind", rule.kind), ("tier", f"T{rule.tier} {rule.tier_name}"),
                             ("severity", f"{rule.default_severity.label} {rule.default_severity.name.lower()}"),
                             ("bucket", rule.bucket), ("flags", flags_of(rule)),
                             ("tags", ", ".join(sorted(rule.tags)) or "-")])
                self.out.result(textwrap.indent(textwrap.fill(rule.description, width=width), "  "))
        self.out.emit(payload, text=text)
        return 0

    def sub_tiers(self, **options):
        items = []
        for number, name in TIERS.items():
            members = [rule for rule in ALL_RULES if rule.tier == number]
            items.append({"tier": number, "name": name, "precedence": number + 1,
                          "rewards": sum(rule.is_reward for rule in members),
                          "penalties": sum(rule.is_penalty for rule in members),
                          "rules": [rule.id for rule in members]})
        self.out.emit({"count": len(items), "items": items}, text=lambda: self.out.table(
            items, [Column("tier", "TIER", fmt=lambda value: f"T{value}"), Column("name", "NAME"),
                    Column("rewards", "REWARDS", align="r"), Column("penalties", "PENALTIES", align="r"),
                    Column("rules", "RULES", fmt=lambda value: " ".join(value))]))
        return 0

    def sub_validate(self, **options):
        problems = validate_catalog()
        payload = {"ok": not problems, "rules": len(ALL_RULES), "problems": problems}

        def text() -> None:
            for problem in problems:
                self.out.error(problem)
            self.out.result("catalog ok" if not problems else f"{len(problems)} catalog problem(s)")
        self.out.emit(payload, text=text)
        return int(ExitCode.FAILURE if problems else ExitCode.OK)
