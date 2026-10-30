"""The exception classes of section 17.1 (the part the P0 code raises, and the rest of the block)."""

from __future__ import annotations

import pytest

from mactf import exceptions as ex

COMPONENT_ERRORS = [
    (ex.ComponentMissing, "E001", lambda: ex.ComponentMissing("scorer", "x", ["a"])),
    (ex.ComponentCollision, "E002", lambda: ex.ComponentCollision("scorer", "x", "d")),
    (ex.ComponentLocked, "E003", lambda: ex.ComponentLocked("scorer", "x", "why")),
    (ex.ComponentParamError, "E004", lambda: ex.ComponentParamError("scorer", "x", ["p"])),
    (ex.ComponentSpecError, "E005", lambda: ex.ComponentSpecError("scorer", "bad")),
    (ex.ComponentImportDenied, "E006", lambda: ex.ComponentImportDenied("a:B", "no")),
    (ex.ComponentContractError, "E007", lambda: ex.ComponentContractError("scorer", "x", ["p"])),
    (ex.ComponentBuildError, "E008", lambda: ex.ComponentBuildError("scorer", "x", ValueError("v"))),
    (ex.UnsafePolicyDenied, "E009", lambda: ex.UnsafePolicyDenied("p", "unsafe")),
    (ex.ComponentMigrationError, "E048", lambda: ex.ComponentMigrationError("nope")),
]


@pytest.mark.parametrize("cls, code, make", COMPONENT_ERRORS, ids=[c[0].__name__ for c in COMPONENT_ERRORS])
def test_every_component_error_has_its_code_in_class_attribute_and_message(cls, code, make):
    error = make()
    assert cls.code == code and error.code == code
    assert str(error).startswith(f"{code}: ")
    assert isinstance(error, ex.ComponentError)


def test_bases_are_compatible_with_existing_handlers():
    for cls, _code, make in COMPONENT_ERRORS:
        error = make()
        assert isinstance(error, ex.ValidationError)
        assert isinstance(error, ex.ImproperlyConfigured)
        assert isinstance(error, ex.MACTFError)
        assert not isinstance(error, ex.ComplianceError)
    assert issubclass(ex.UnsafePolicyDenied, ex.ComponentLocked)


def test_guard_violations_are_compliance_errors_and_never_configuration_errors():
    for error in (
        ex.ComplianceGuardViolation([{"code": "C003", "message": "m"}]),
        ex.StepGuardViolation("root/a", [{"code": "S002", "message": "m"}]),
        ex.WeightGuardViolation("s", ["u"]),
        ex.StrategyGuardViolation("s", ["u"]),
        ex.LedgerReadOnlyError("record"),
    ):
        assert isinstance(error, ex.ComplianceError)
        assert not isinstance(error, ex.ImproperlyConfigured)


def test_guard_violation_codes():
    assert ex.ComplianceGuardViolation([{"code": "C003"}, {"code": "C005"}]).code == "C003"
    assert ex.ComplianceGuardViolation([]).code == "C001"
    assert "C003" in str(ex.ComplianceGuardViolation([{"code": "C003", "message": "floor"}]))
    assert ex.StepGuardViolation("p", [{"code": "S004"}]).code == "S004"
    assert ex.WeightGuardViolation("s", ["U001"]).code == "E063"
    assert ex.StrategyGuardViolation("s", ["V1"]).code == "E073"
    assert ex.LedgerReadOnlyError("clear").code == "C007"


def test_a_message_that_starts_with_a_code_keeps_it():
    assert ex.ComponentSpecError("weight_update", "E012: both flat and params").code == "E012"
    assert str(ex.ComponentSpecError("weight_update", "E011: x")).startswith("E011: x")
    assert ex.ComponentSpecError("k", "plain").code == "E005"
    assert str(ex.ComponentSpecError("k", "plain")) == "E005: k spec: plain"
    assert ex.ComponentError("E013: boom").code == "E013"
    assert ex.ComponentError("boom", code="E014").code == "E014"
    assert ex.ComponentError("boom").code == "E005"


def test_a_missing_slot_is_e010_and_a_missing_component_is_e001():
    slot = ex.ComponentMissing("slot", "SCORING.CHAIN.x", ["a", "b"])
    assert slot.code == "E010" and slot.known == ("a", "b")
    assert ex.ComponentMissing("scorer", "x").code == "E001"
    assert "none registered" in str(ex.ComponentMissing("scorer", "x"))


def test_a_missing_component_suggests_the_nearest_name():
    error = ex.ComponentMissing("weight_update", "kapa_softmax", ["kappa_softmax", "uniform"])
    assert "did you mean 'kappa_softmax'?" in str(error)
    assert "did you mean" not in str(ex.ComponentMissing("weight_update", "zzzzzz", ["uniform"]))


def test_params_and_contract_errors_keep_every_problem():
    error = ex.ComponentParamError("scorer", "penalty", ["a bad", "b bad"])
    assert error.problems == ("a bad", "b bad") and "a bad; b bad" in str(error)
    assert error.kind == "scorer" and error.name == "penalty"
    contract = ex.ComponentContractError("optimizer", "lion", ["p1", "p2"])
    assert contract.problems == ("p1", "p2") and "breaks its contract" in str(contract)


def test_a_build_error_keeps_its_cause():
    cause = ValueError("bad value")
    error = ex.ComponentBuildError("scorer", "x", cause)
    assert error.cause is cause and "ValueError: bad value" in str(error)
    assert ex.ComponentBuildError("scorer", "x", "text").cause is None


def test_locked_error_carries_path_and_reason():
    error = ex.ComponentLocked("scorer", "penalty", "locked by COMPONENTS.LOCKED")
    assert error.path == "penalty" and error.reason == "locked by COMPONENTS.LOCKED"
    unsafe = ex.UnsafePolicyDenied("risky", "unsafe = True")
    assert unsafe.code == "E009" and unsafe.path == "risky" and "risky" in str(unsafe)


def test_mutation_error_carries_findings_and_rollback_state():
    error = ex.MutationError("E085", "refused", findings=["E085 [root/a]: x"], rolled_back=True)
    assert error.code == "E085" and error.rolled_back and error.findings == ("E085 [root/a]: x",)
    assert str(error).startswith("E085: refused [E085 [root/a]: x]")
    assert isinstance(error, ex.ImproperlyConfigured)
    assert not ex.MutationError("E089", "x").rolled_back


def test_every_exception_has_a_code_and_a_status():
    """17.1.3: each class carries a code and the API status the table gives it (via ``_translate``)."""
    from mactf.contrib.api.views.base import _translate

    wanted = [
        (ex.ComponentMissing("scorer", "x", ["a"]), 404, "E001"),
        (ex.ComponentMissing("slot", "TRAINER.X", ["a"]), 404, "E010"),
        (ex.ComponentCollision("scorer", "x"), 409, "E002"),
        (ex.ComponentLocked("scorer", "x", "why"), 409, "E003"),
        (ex.UnsafePolicyDenied("p", "unsafe"), 422, "E009"),
        (ex.ComponentParamError("scorer", "x", ["p"]), 422, "E004"),
        (ex.ComponentSpecError("scorer", "bad"), 422, "E005"),
        (ex.ComponentSpecError("scorer", "E011: injected"), 422, "E011"),
        (ex.ComponentImportDenied("a:B", "no"), 422, "E006"),
        (ex.ComponentContractError("scorer", "x", ["p"]), 422, "E007"),
        (ex.ComponentBuildError("scorer", "x", ValueError("v")), 500, "E008"),
        (ex.ComponentMigrationError("nope"), 409, "E048"),
        (ex.ComponentMigrationError("E049: strict"), 409, "E049"),
        (ex.MutationError("E082", "protected"), 422, "E082"),
    ]
    for error, status, code in wanted:
        http = _translate(error)
        assert http.status == status, (type(error).__name__, http)
        assert http.extensions["code"] == code and error.code == code
