import pytest
import yaml

from mactf.management.commands.rules import severity_arg, tier_arg
from mactf.rules import ALL_RULES, CATALOG
from mactf.testing.cli import invoke


def run(*argv):
    return invoke(["rules", *argv])


def ids(out):
    return [item["id"] for item in out.json()["items"]]


def test_list_all_in_catalog_order():
    out = run("list", "--format", "json")
    assert out.code == 0
    assert ids(out) == [rule.id for rule in ALL_RULES]
    assert out.json()["count"] == out.json()["total"] == len(ALL_RULES)


def test_filters_combine():
    out = run("list", "--kind", "penalty", "--severity", "major", "--severity", "severe", "--format", "json")
    assert out.code == 0
    assert ids(out) and all(CATALOG[i].kind == "penalty" and CATALOG[i].default_severity >= 3 for i in ids(out))
    assert set(ids(run("list", "--tier", "T0", "--omega", "--format", "json"))) == {"P20"}


def test_tag_requires_every_tag_asked_for():
    both = ids(run("list", "--tag", "honesty", "--tag", "epistemic", "--format", "json"))
    one = ids(run("list", "--tag", "honesty", "--format", "json"))
    assert set(both) <= set(one) and "R2" in both


def test_boolean_flags_and_search():
    assert all(CATALOG[i].non_offsettable for i in ids(run("list", "--non-offsettable", "--format", "json")))
    assert all(CATALOG[i].warning_eligible for i in ids(run("list", "--warning-eligible", "--format", "json")))
    assert ids(run("list", "--search", "sandbox escape", "--format", "json")) == ["P20"]


def test_sort_orders_are_deterministic():
    by_severity = [CATALOG[i].default_severity for i in ids(run("list", "--sort", "severity", "--format", "json"))]
    assert by_severity == sorted(by_severity, reverse=True)
    names = [CATALOG[i].name.lower() for i in ids(run("list", "--sort", "name", "--format", "json"))]
    assert names == sorted(names)
    assert run("list", "--sort", "tier").stdout == run("list", "--sort", "tier").stdout


def test_empty_result_is_success():
    out = run("list", "--tag", "no-such-tag")
    assert out.code == 0 and "no rules match" in out.stdout


def test_bad_severity_and_tier_are_usage_errors():
    assert run("list", "--severity", "bogus").code == 2
    assert run("list", "--tier", "T9").code == 2


def test_show_several_ids_case_insensitively():
    out = run("show", "r2", "P20", "p20", "--format", "json")
    assert out.code == 0
    assert ids(out) == ["R2", "P20"]                          # duplicates collapse


def test_show_names_every_unknown_id():
    out = run("show", "R2", "P99", "X")
    assert out.code == 3
    assert "'P99'" in out.stderr and "'X'" in out.stderr


def test_tiers_account_for_every_rule():
    out = run("tiers", "--format", "json")
    assert out.code == 0
    listed = [rule for item in out.json()["items"] for rule in item["rules"]]
    assert sorted(listed) == sorted(CATALOG)
    assert [item["precedence"] for item in out.json()["items"]] == list(range(1, len(out.json()["items"]) + 1))


def test_validate_ok_on_the_shipped_catalog():
    out = run("validate", "--format", "json")
    assert out.code == 0 and out.json() == {"ok": True, "rules": len(ALL_RULES), "problems": []}


def test_validate_fails_on_a_broken_catalog(monkeypatch):
    from mactf.management.commands import rules

    monkeypatch.setattr(rules, "validate_catalog", lambda: ["P1 has no description"])
    out = run("validate")
    assert out.code == 1 and "P1 has no description" in out.stderr


def test_yaml_format_and_alias():
    out = run("ls", "--kind", "reward", "--format", "yaml")
    assert out.code == 0
    assert yaml.safe_load(out.stdout)["count"] == sum(rule.is_reward for rule in ALL_RULES)


def test_no_settings_are_needed():
    assert invoke(["rules", "list", "--settings", "no_such_module"], env={"MACTF_NO_DISCOVER": "1"}).code == 0


def test_argument_parsers():
    assert severity_arg("S3") == severity_arg("major") == severity_arg("3")
    assert tier_arg("t1") == tier_arg("1") == tier_arg("honesty_and_non_deception")
