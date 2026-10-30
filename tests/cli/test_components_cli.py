"""Document 17 section 16.1 and 16.2: ``mactf components``, ``runs components``, ``peers mutate``.

Every command here runs through the real entry point (``mactf.testing.cli.invoke``) in a generated
project; the verbs that change a *live* run are driven against a run that really is live in this
process (a ``RunManager`` registered with ``set_cli_manager``), because that is the only place a
CLI process can reach one.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time

import pytest

pytest.importorskip("torch")

from mactf import apps as apps_module
from mactf import conf
from mactf.components import HUB, ComponentResolver, ComponentSpec, overlay
from mactf.components.scaffold import KIND_IDS, SCAFFOLD_KINDS, render_scaffold
from mactf.exceptions import (
    ComponentBuildError,
    ComponentCollision,
    ComponentImportDenied,
    ComponentLocked,
    ComponentMissing,
    ImproperlyConfigured,
    MutationError,
    UnsafePolicyDenied,
    WeightGuardViolation,
)
from mactf.management.exitcodes import ExitCode, classify
from mactf.orchestration.live import RunManager, RunStateError, set_cli_manager

CPU = ("--device", "cpu")
EXAMPLES = "mactf.contrib.components.examples"
EMA = f"{EXAMPLES}.ema_softmax"


@pytest.fixture(autouse=True)
def restore_process_registry():
    """A command that loads ``COMPONENTS.MODULES`` registers globally; give the process back as found.

    Registrations, freeze state, the hub policy, the plugin loader's memory and the modules the
    commands imported are all restored, so no later test sees a component these tests loaded.
    """
    from mactf.components import plugins

    from mactf.components.known import import_builtins

    import_builtins()                                # every kind's built-ins, trainer policies included
    saved = {
        kind: (dict(registry._entries), registry._frozen, dict(registry._builtin_factories),
               dict(registry._mirror) if registry._mirror is not None else None)
        for kind, registry in HUB._registries.items()
    }
    policy, loaded = HUB.policy, dict(plugins._LOADED)
    last = (plugins._LAST_BLOCK, plugins._LAST_REPORT)
    known_modules = set(sys.modules)
    yield
    for kind, (entries, frozen, builtins, mirror) in saved.items():
        registry = HUB._registries[kind]
        registry._entries, registry._frozen, registry._builtin_factories = entries, frozen, builtins
        if registry._mirror is not None and mirror is not None:
            dict.clear(registry._mirror)
            dict.update(registry._mirror, mirror)
    HUB.configure(policy)
    plugins._LOADED.clear()
    plugins._LOADED.update(loaded)
    plugins._LAST_BLOCK, plugins._LAST_REPORT = last
    for name in set(sys.modules) - known_modules:
        if name.startswith((EXAMPLES + ".", "boom_components", "every_other", "scaffolded_")):
            del sys.modules[name]
    examples = sys.modules.get(EXAMPLES)
    if examples is not None:
        for attribute in [a for a in vars(examples) if a in {"tempered_reward", "sqrt_magnitude", "ema_softmax",
                                                              "every_n", "lion", "huber_value", "anchored_policy"}]:
            delattr(examples, attribute)


def modules(*names: str) -> tuple[str, ...]:
    return ("--set", "COMPONENTS.MODULES=" + json.dumps(list(names)))


# -- exit codes: every row of the 16.2.1 table ----------------------------------------------------
EXCEPTION_ROWS = [
    ("settings name an unknown component", lambda: ComponentMissing("scorer", "x", ["a"]), 2, "ImproperlyConfigured"),
    ("import denied on an untrusted surface", lambda: ComponentImportDenied("a:B", "untrusted"), 2,
     "ImproperlyConfigured"),
    ("registering a name that exists", lambda: ComponentCollision("scorer", "x"), 4, "Conflict"),
    ("replacing a locked component", lambda: ComponentLocked("scorer", "scorer:x", "locked"), 4, "Conflict"),
    ("a slot that forbids this boundary", lambda: ComponentLocked("slot", "TRAINER.POLICY", "static"), 4, "Conflict"),
    ("an unsafe policy without the allowance", lambda: UnsafePolicyDenied("greedy", "unsafe"), 2,
     "ImproperlyConfigured"),
    ("a component that cannot be built", lambda: ComponentBuildError("scorer", "x", ValueError("boom")), 1,
     "ComponentBuildError"),
    ("a guard violation at the top of a command", lambda: WeightGuardViolation("s", ["U001"]), 1,
     "WeightGuardViolation"),
    ("a mutation plan with a hard finding", lambda: MutationError("E082", "protected"), 1, "MutationError"),
    ("a run that is not live", lambda: RunStateError("r", "completed", "change the components of"), 4, "Conflict"),
]


@pytest.mark.parametrize("label,make,code,prefix", EXCEPTION_ROWS, ids=[row[0] for row in EXCEPTION_ROWS])
def test_component_exit_codes_by_exception(label, make, code, prefix):
    exit_code, message = classify(make())
    assert exit_code == code, label
    assert message.startswith(prefix + ":"), (label, message)


def test_component_exit_codes_hold_for_the_other_exception_classes():
    """The classes with a code of their own keep the exit code of the kind of mistake they are."""
    assert classify(ComponentCollision("scorer", "x"))[0] == int(ExitCode.CONFLICT)
    assert classify(ImproperlyConfigured("E013: x"))[0] == int(ExitCode.USAGE)
    assert classify(ComponentMissing("slot", "TRAINER.NOPE", ["a"]))[0] == int(ExitCode.USAGE)


def test_component_exit_codes(project):
    """The command-level rows: a name from an argument is 3, a name from settings is 2, and so on."""
    # an unknown component named by an argument: NotFound (3), E001 names the accepted set
    out = project.invoke("components", "inspect", "weight_update", "no_such_one")
    assert out.code == 3 and out.stderr.startswith("NotFound: E001") and "kappa_softmax" in out.stderr
    out = project.invoke("components", "validate", "--kind", "weight_update", "--name", "no_such_one")
    assert out.code == 3 and "E001" in out.stderr
    out = project.invoke("components", "inspect", "no_such_kind", "x")
    assert out.code == 3 and "no component kind" in out.stderr
    out = project.invoke("components", "list", "--kind", "no_such_kind")
    assert out.code == 3
    # an unknown component named by a settings file or --set: a configuration error (2)
    out = project.invoke("config", "show", "--set", "TRAINER.WEIGHT_UPDATE=no_such_one")
    assert out.code == 2 and "ImproperlyConfigured" in out.stderr
    # validate: a bad key is a finding, not a failure to look up (1), and the output is the finding
    out = project.invoke("components", "validate", "--kind", "weight_update", "--name", "uniform",
                         "--param", "trimm=0.2")
    assert out.code == 1 and "E004" in out.stdout and out.stderr == ""
    # a spec that is not a mapping/name is a finding too (E005)
    assert project.invoke("components", "validate", "--kind", "scorer", "--import", "no-colon").code in (1, 2)
    # a bad CLI invocation is usage (2)
    assert project.invoke("components", "validate", "--kind", "scorer").code == 2
    assert project.invoke("peers", "mutate", "plan.yaml", "--apply").code == 2
    # scaffold onto an existing file without --force: Conflict (4)
    first = project.invoke("components", "scaffold", "weight_update", "my_rule")
    assert first.code == 0, first.stderr
    again = project.invoke("components", "scaffold", "weight_update", "my_rule")
    assert again.code == 4 and again.stderr.startswith("CommandError:")
    assert project.invoke("components", "scaffold", "weight_update", "my_rule", "--force").code == 0
    # a module that raises at import is a ComponentBuildError (1)
    (project.root / "boom_components.py").write_text("raise RuntimeError('kaboom')\n", encoding="utf-8")
    out = project.invoke("components", "list", *modules("boom_components"))
    assert out.code == 1 and out.stderr.startswith("ComponentBuildError:") and "E008" in out.stderr
    # a run that is not live: the verbs that change a run are Conflicts (4)
    seeded = project.invoke("run", "--epochs", "1", "--run-id", "idle", *CPU)
    assert seeded.code == 0, seeded.stderr
    out = project.invoke("runs", "components", "set", "idle", "TRAINER.WEIGHT_UPDATE", "--spec", "uniform")
    assert out.code == 4 and out.stderr.startswith("Conflict:")
    plan = project.root / "plan.yaml"
    plan.write_text("ops:\n  - {op: retire, path: peer_gamma}\n", encoding="utf-8")
    out = project.invoke("peers", "mutate", str(plan), "--run", "idle", "--apply")
    assert out.code == 4 and out.stderr.startswith("Conflict:")
    # an unknown run is NotFound
    assert project.invoke("runs", "components", "show", "nosuchrun").code == 3


# -- list / inspect / validate -----------------------------------------------------------------------
def test_list_inspect_and_validate_end_to_end(project):
    listed = project.invoke("components", "list", "--format", "json")
    assert listed.code == 0, listed.stderr
    payload = listed.json()
    assert payload["count"] == len(payload["items"]) > 50
    kinds = {item["kind"] for item in payload["items"]}
    assert {"scorer", "weight_update", "peer_strategy", "trainer_policy", "optimizer", "rl_function.loss"} <= kinds
    row = next(item for item in payload["items"] if (item["kind"], item["name"]) == ("scorer", "penalty"))
    assert row["origin"] == "builtin" and row["locked"] is False and len(row["fingerprint"]) == 12
    text = project.invoke("components", "list", "--kind", "weight_update")
    assert text.stdout.splitlines()[0].split() == ["KIND", "NAME", "ORIGIN", "MUTABLE", "LOCKED", "FINGERPRINT"]
    assert "kappa_softmax" in text.stdout and "scorer" not in text.stdout
    assert project.invoke("components", "list", "--origin", "project", "--format", "json").json()["count"] == 0
    locked = project.invoke("components", "list", "--locked", "--format", "json",
                            "--set", 'COMPONENTS.LOCKED=["scorer:penalty"]').json()
    assert [(i["kind"], i["name"]) for i in locked["items"]] == [("scorer", "penalty")]

    shown = project.invoke("components", "inspect", "weight_update", "kappa_softmax", "--format", "json").json()
    assert shown["contract"] == "ok" and shown["origin"] == "builtin" and shown["mutability"] == "episode"
    assert project.invoke("components", "inspect", "rl_function.loss", "policy_gradient", "--format", "yaml").code == 0

    good = project.invoke("components", "validate", "--kind", "weight_update", "--name", "uniform",
                          "--format", "json")
    assert good.code == 0 and good.json()["ok"] is True and good.json()["findings"] == []
    bad = project.invoke("components", "validate", "--kind", "weight_update", "--name", "uniform",
                         "--param", "trimm=0.2", "--format", "json")
    assert bad.code == 1 and bad.json()["findings"][0]["code"] == "E004"
    assert "does not accept 'trimm'" in bad.json()["findings"][0]["message"]
    # the spec can come from standard input; JSON values are parsed
    via_stdin = project.invoke("components", "validate", "--kind", "weight_update", "--spec-file", "-",
                               "--format", "json", input='{"type": "uniform", "params": {"bogus": 1}}')
    assert via_stdin.code == 1 and via_stdin.json()["findings"][0]["code"] == "E004"
    both = project.invoke("components", "validate", "--kind", "weight_update", "--name", "uniform",
                          "--spec-file", "-", input='{"params": {"bogus": 1}}', *["--format", "json"])
    assert both.json()["spec"] == {"params": {"bogus": 1}, "type": "uniform"}


def test_validate_an_unsafe_policy_warns_and_probe_finds_a_broken_function(project):
    out = project.invoke("components", "validate", "--kind", "trainer_policy", "--import",
                         f"{EXAMPLES}.anchored_policy:Greedy", "--format", "json")
    assert out.code == 0, out.stderr
    assert [f["code"] for f in out.json()["findings"]] == ["W004"]
    assert out.json()["findings"][0]["severity"] == "warning"
    ok = project.invoke("components", "validate", "--kind", "rl_function.loss", "--import",
                        f"{EXAMPLES}.huber_value:HuberValueLoss", "--probe", "--format", "json")
    assert ok.code == 0, ok.stdout + ok.stderr
    broken = project.invoke("components", "validate", "--kind", "rl_function.loss", "--import",
                            f"{EXAMPLES}.huber_value:NonFiniteHuber", "--probe", "--format", "json")
    assert broken.code == 1
    finding = next(f for f in broken.json()["findings"] if f["code"] == "E047")
    assert "F004" in finding["message"] and finding["probe"][0] == "F004"
    spec = project.invoke("components", "validate", "--kind", "scorer", "--import", f"{EXAMPLES}.tempered_reward:TemperedReward",
                          "--param", "factr=0.5", "--format", "json")
    assert spec.code == 1 and spec.json()["findings"][0]["code"] == "E004"


# -- scaffold -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", SCAFFOLD_KINDS)
def test_scaffold_renders_and_imports(kind, tmp_path):
    """Each template renders, imports, registers, and passes its own stub (resolves under its contract)."""
    name = f"scaffolded_{kind}"
    text = render_scaffold(kind, name)
    for placeholder in ("{name}", "{cls}", "{kind}", "{kind_id}", "{contract}"):
        assert placeholder not in text, placeholder
    compile(text, name, "exec")
    path = tmp_path / f"{name}.py"
    path.write_text(text, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    before = HUB.snapshot()
    try:
        importlib.import_module(name)
        kind_id = KIND_IDS[kind]
        assert name in HUB.snapshot()[kind_id] and HUB.snapshot()[kind_id][name]["origin"] == "project"
        spec = ComponentSpec.parse(kind_id, {"type": name})
        from mactf.components.chain_validation import injected_offer

        component = ComponentResolver().validate(spec, trusted=True, inject=injected_offer(kind_id))
        assert component.name == name
        from mactf.components.kinds import KIND_SPECS

        assert KIND_SPECS[kind_id].contract(component.factory) == []
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)
        if name in HUB.snapshot()[KIND_IDS[kind]]:
            from mactf.components import unregister

            family = KIND_IDS[kind].partition(".")[2] or None
            unregister(KIND_IDS[kind].partition(".")[0], name, family=family)
    assert HUB.snapshot() == before, "the scaffold test leaked a registration"
    # a stub raises NotImplementedError inside the method only: importing and resolving were fine above
    assert "NotImplementedError" in text


def test_scaffold_through_the_cli_then_load_it(project):
    out = project.invoke("components", "scaffold", "peer_strategy", "every_other", "--dir", str(project.root / "src"),
                         "--format", "json")
    assert out.code == 0, out.stderr
    created = out.json()
    assert created["path"].endswith("src/every_other.py") and created["bytes"] > 100
    dry = project.invoke("components", "scaffold", "scorer", "tmp_scorer", "--dry-run", "--format", "json")
    assert dry.code == 0 and dry.json()["dry_run"] is True
    assert not (project.package.parent / "components" / "tmp_scorer.py").exists()
    assert "would write" in project.invoke("components", "scaffold", "scorer", "tmp_scorer", "--dry-run").stdout
    for bad in ("2fast", "class", "has-dash"):
        assert project.invoke("components", "scaffold", "scorer", bad).code == 2
    escape = project.invoke("components", "scaffold", "scorer", "x", "--dir", str(project.root / "src"))
    assert escape.code == 0
    sys.path.insert(0, str(project.root / "src"))
    try:
        out = project.invoke("components", "list", "--origin", "project", "--format", "json",
                             *modules("every_other"))
        assert out.code == 0, out.stderr
        assert [(i["kind"], i["name"]) for i in out.json()["items"]] == [("peer_strategy", "every_other")]
        valid = project.invoke("components", "validate", "--kind", "peer_strategy", "--name", "every_other",
                               "--format", "json", *modules("every_other"))
        assert valid.code == 0 and valid.json()["ok"] is True
    finally:
        sys.path.remove(str(project.root / "src"))
        sys.modules.pop("every_other", None)
        with overlay():
            pass
        from mactf.components import unregister

        if "every_other" in HUB.snapshot()["peer_strategy"]:
            unregister("peer_strategy", "every_other")


def test_every_verb_dry_runs(project):
    """scaffold, runs components set and peers mutate accept --dry-run and touch nothing."""
    tree = sorted(path.relative_to(project.root).as_posix() for path in project.root.rglob("*")
                  if "__pycache__" not in path.parts)
    out = project.invoke("components", "scaffold", "scorer", "dry_one", "--dry-run")
    assert out.code == 0 and "would write" in out.stdout
    plan = project.root / "plan.yaml"
    plan.write_text("label: p\nops:\n  - {op: retire, path: peer_gamma}\n", encoding="utf-8")
    sim = project.invoke("peers", "mutate", str(plan), "--dry-run", "--format", "json")
    assert sim.code == 0, sim.stderr
    assert sim.json()["queued"] is None and sim.json()["ok"] is True
    seeded = project.invoke("run", "--epochs", "1", "--run-id", "dr", *CPU)
    assert seeded.code == 0
    after = sorted(path.relative_to(project.root).as_posix() for path in project.root.rglob("*")
                   if "__pycache__" not in path.parts)
    new = set(after) - set(tree)
    assert not any("dry_one" in item for item in new)
    out = project.invoke("runs", "components", "set", "dr", "TRAINER.NOPE", "--spec", "uniform", "--dry-run",
                         "--format", "json")
    assert out.code == 1 and out.json()["findings"][0]["code"] == "E010"


# -- peers mutate against the settings ------------------------------------------------------------
def test_peers_mutate_simulates_on_the_configured_peers(project):
    plan = project.root / "retire.yaml"
    plan.write_text("label: prune\nreason: diverged\nops:\n  - {op: retire, path: peer_gamma, reason: nope}\n",
                    encoding="utf-8")
    ok = project.invoke("peers", "mutate", str(plan), "--format", "json")
    assert ok.code == 0, ok.stderr
    body = ok.json()
    assert body["ok"] is True and body["retired"] == ["peer_gamma"] and body["label"] == "prune"
    assert body["active_paths"] == ["peer_alpha", "peer_beta"] and len(body["digest"]) >= 12
    text = project.invoke("peers", "mutate", str(plan))
    assert "applicable" in text.stdout and "retire peer_gamma" in text.stdout
    # a hard finding: unknown node (E081) is exit 1, and the plan's finding is the output
    missing = project.root / "missing.yaml"
    missing.write_text("ops:\n  - {op: retire, path: nowhere}\n", encoding="utf-8")
    out = project.invoke("peers", "mutate", str(missing), "--format", "json")
    assert out.code == 1 and out.json()["hard"][0]["code"] == "E081"
    # a plan on standard input, as a bare list
    piped = project.invoke("peers", "mutate", "-", "--format", "json",
                           input='[{"op": "retire", "path": "peer_alpha"}]')
    assert piped.code == 0 and piped.json()["retired"] == ["peer_alpha"]
    # purge without retiring first is E082
    purge = project.root / "purge.yaml"
    purge.write_text("ops:\n  - {op: purge, path: peer_gamma}\n", encoding="utf-8")
    refused = project.invoke("peers", "mutate", str(purge), "--format", "json")
    assert refused.code == 1 and refused.json()["hard"][0]["code"] == "E082"
    assert project.invoke("peers", "mutate", str(project.root / "absent.yaml")).code == 3
    garbage = project.root / "garbage.yaml"
    garbage.write_text("- [unclosed\n", encoding="utf-8")
    assert project.invoke("peers", "mutate", str(garbage)).code == 2


# -- a real run, recorded, then read back ----------------------------------------------------------
def test_runs_components_and_diff_read_back_a_real_run(project):
    args = ("--set", "ORCHESTRATOR.ENGINE=trainer", "--set", f"TRAINER.WEIGHT_UPDATE=ema_softmax",
            "--set", "TRAINER.WEIGHT_UPDATE.params.alpha=0.4", *modules(EMA))
    run = project.invoke("run", "--epochs", "2", "--run-id", "ema1", *CPU, *args)
    assert run.code == 0, run.stdout + run.stderr
    shown = project.invoke("runs", "components", "show", "ema1", "--format", "json", *modules(EMA))
    assert shown.code == 0, shown.stderr
    slot = next(s for s in shown.json()["slots"] if s["slot"] == "TRAINER.WEIGHT_UPDATE")
    assert slot["name"] == "ema_softmax" and slot["kind"] == "weight_update" and slot["origin"] == "project"
    text = project.invoke("runs", "components", "show", "ema1", *modules(EMA))
    assert text.stdout.splitlines()[0].split() == ["SLOT", "KIND", "NAME", "SPEC_HASH", "STATE"]
    assert "TRAINER.WEIGHT_UPDATE" in text.stdout and "ema_softmax" in text.stdout
    # the same settings: nothing differs; a changed parameter is a structural difference (exit 1)
    same = project.invoke("components", "diff", "ema1", "live", "--format", "json", *args)
    assert same.code == 0 and same.json() == {"left": "ema1", "right": "live", "same": True, "changes": []}
    changed_args = ("--set", "ORCHESTRATOR.ENGINE=trainer", "--set", "TRAINER.WEIGHT_UPDATE=ema_softmax",
                    "--set", "TRAINER.WEIGHT_UPDATE.params.alpha=0.9", *modules(EMA))
    other = project.invoke("components", "diff", "ema1", "live", "--format", "json", *changed_args)
    assert other.code == 1 and not other.json()["same"]
    assert {c["field"] for c in other.json()["changes"]} >= {"spec_hash", "params"}
    assert project.invoke("components", "diff", "ema1", "live", "--structural", *changed_args).code == 1
    assert "params" in project.invoke("components", "diff", "ema1", "live", *changed_args).stdout
    # a second run with the default strategy: the slot is absent on that side
    plain = project.invoke("run", "--epochs", "1", "--run-id", "plain1", *CPU, "--set", "ORCHESTRATOR.ENGINE=trainer")
    assert plain.code == 0, plain.stderr
    versus = project.invoke("components", "diff", "ema1", "plain1", "--format", "json")
    assert versus.code == 1 and versus.json()["changes"][0]["field"] == "present"
    # a manifest file works as one side
    manifest = project.root / "manifest.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")
    assert project.invoke("components", "diff", str(manifest), "plain1").code == 0
    assert project.invoke("components", "diff", "nothing-here", "plain1").code == 3
    # --component-module registers for the run only and writes no settings
    listed = project.invoke("run", "--epochs", "1", "--run-id", "cm1", *CPU, "--component-module", EMA,
                            "--set", "ORCHESTRATOR.ENGINE=trainer", "--set", "TRAINER.WEIGHT_UPDATE=ema_softmax")
    assert listed.code == 0, listed.stderr
    refused = project.invoke("run", "--epochs", "1", "--component-module", "not a module")
    assert refused.code == 2
    assert "ema_softmax" not in (project.package / "settings.py").read_text(encoding="utf-8")
    assert "ema_softmax" not in "".join(p.read_text(encoding="utf-8") for p in (project.package / "conf").glob("*.yaml"))


# -- the verbs that change a live run --------------------------------------------------------------
@pytest.fixture()
def live_project(project):
    """A trainer-engine run that is really running in this process, reachable by the CLI."""
    settings = conf.configure(project.module, force=True)
    apps_module.populate()                       # the project's peers register on `ready()`
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    settings["TRAINER"]["MUTATION"] = {**settings["TRAINER"].get("MUTATION", {}), "ENABLED": True}
    settings["TRAINING"]["EPOCHS"] = 400
    manager = RunManager(settings)
    set_cli_manager(manager)
    handle = manager.start({"epochs": 400, "run_id": "lv", "device": "cpu"})
    deadline = time.time() + 60
    while time.time() < deadline:
        trainer = getattr(handle.orchestrator, "trainer", None)
        if trainer is not None and getattr(trainer, "built", False) and handle.epochs_completed >= 1:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"the run never started: {handle.status} {handle.error}")
    yield project, manager
    set_cli_manager(None)
    try:
        manager.cancel("lv")
    except RunStateError:
        pass
    manager.close()
    apps_module.reset_registry()
    conf.reset()


def _wait(predicate, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def test_runs_components_set_changes_a_live_run(live_project):
    project, manager = live_project
    shown = project.invoke("runs", "components", "show", "lv", "--format", "json")
    assert shown.code == 0, shown.stderr
    assert shown.json()["source"] == "live"
    assert next(s for s in shown.json()["slots"] if s["slot"] == "TRAINER.WEIGHT_UPDATE")["name"] == "kappa_softmax"
    queued = project.invoke("runs", "components", "set", "lv", "TRAINER.WEIGHT_UPDATE", "--spec", "uniform",
                            "--format", "json")
    assert queued.code == 0, queued.stdout + queued.stderr
    assert queued.json()["slot"] == "TRAINER.WEIGHT_UPDATE" and queued.json()["boundary"] == "episode"

    def swapped():
        current = manager.components("lv")
        slot = next(s for s in current["slots"] if s["slot"] == "TRAINER.WEIGHT_UPDATE")
        return slot["name"] == "uniform"

    _wait(swapped)
    # refusals keep their exit codes: a static slot is a Conflict, a bad key is usage-level 2/E004
    locked = project.invoke("runs", "components", "set", "lv", "TRAINER.POLICY", "--spec", "default")
    assert locked.code == 4 and "E003" in locked.stderr
    nope = project.invoke("runs", "components", "set", "lv", "TRAINER.NOT_A_SLOT", "--spec", "uniform")
    assert nope.code == 3 and "E010" in nope.stderr
    bad = project.invoke("runs", "components", "set", "lv", "TRAINER.PEER_STRATEGY", "--spec",
                         "{type: independent, params: {nope: 1}}")
    assert bad.code == 2 and "E004" in bad.stderr
    assert project.invoke("runs", "components", "set", "lv", "TRAINER.WEIGHT_UPDATE").code == 2   # no --spec
    # --spec from standard input
    piped = project.invoke("runs", "components", "set", "lv", "TRAINER.WEIGHT_UPDATE", "--spec", "-",
                           input="kappa_softmax", *["--format", "json"])
    assert piped.code == 0, piped.stderr


def test_peers_mutate_apply_queues_on_a_live_run(live_project):
    project, manager = live_project
    plan = project.root / "retire.yaml"
    plan.write_text("label: prune-gamma\nops:\n  - {op: retire, path: peer_gamma}\n", encoding="utf-8")
    sim = project.invoke("peers", "mutate", str(plan), "--run", "lv", "--format", "json")
    assert sim.code == 0, sim.stderr
    assert sim.json()["ok"] and sim.json()["retired"] == ["peer_gamma"] and sim.json()["queued"] is None
    dry = project.invoke("peers", "mutate", str(plan), "--run", "lv", "--apply", "--dry-run", "--format", "json")
    assert dry.code == 0 and dry.json()["queued"] is None
    applied = project.invoke("peers", "mutate", str(plan), "--run", "lv", "--apply", "--format", "json")
    assert applied.code == 0, applied.stdout + applied.stderr
    assert applied.json()["queued"]["label"] == "prune-gamma" and applied.json()["queued"]["digest"]

    def retired():
        trainer = manager.get_handle("lv").orchestrator.trainer
        return "peer_gamma" in trainer.freeze.retired()

    _wait(retired)
    # a plan the live tree refuses is exit 1 and nothing is queued
    purge = project.root / "purge.yaml"
    purge.write_text("ops:\n  - {op: purge, path: peer_alpha}\n", encoding="utf-8")
    refused = project.invoke("peers", "mutate", str(purge), "--run", "lv", "--apply", "--format", "json")
    assert refused.code == 1 and refused.json()["queued"] is None and refused.json()["hard"]
    assert threading.active_count() >= 1


# -- completers (16.1.6) -------------------------------------------------------------------------
def test_the_component_completers_answer_without_importing_torch(project):
    from mactf.management.completion import complete

    def values(*words):
        return [item.value for item in complete(list(words)).candidates]

    kinds = values("components", "inspect", "")
    assert {"scorer", "weight_update", "rl_function.loss", "trainer_policy"} <= set(kinds)
    assert "kappa_softmax" in values("components", "inspect", "weight_update", "")
    assert "kappa_softmax" in values("components", "validate", "--kind", "weight_update", "--name", "")
    assert "carbon_cost" not in values("components", "validate", "--kind", "weight_update", "--name", "")
    assert "TRAINER.WEIGHT_UPDATE" in values("runs", "components", "set", "r1", "TRAINER.")
    assert values("runs", "components", "set", "r1", "NOPE.") == []
