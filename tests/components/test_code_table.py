"""Document 17 section 17.2: the code table and the source tree agree, in both directions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mactf.components import codes
from mactf.components.codes import CODE_TABLE, FAMILIES, RESERVED

ROOT = Path(codes.__file__).resolve().parents[1]          # the mactf/ package
TOKEN = re.compile(r"(?<![A-Za-z0-9_])([EWCF]\d{3})(?![A-Za-z0-9_])")


NOQA = re.compile(r"#\s*noqa\b[^\n]*")


def _scan() -> dict[str, set[str]]:
    """``code -> files (relative) that mention it``, ``codes.py`` and templates excluded."""
    found: dict[str, set[str]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        if path == Path(codes.__file__).resolve():
            continue
        text = NOQA.sub("", path.read_text(encoding="utf-8"))      # flake8 codes are not ours
        for match in TOKEN.finditer(text):
            found.setdefault(match.group(1), set()).add(path.relative_to(ROOT).as_posix())
    return found


def test_code_table_is_complete():
    """Both directions: no code used in mactf/ is missing from the table, and no row is
    raised nowhere (unless RESERVED says why)."""
    used = _scan()
    unknown = sorted(set(used) - set(CODE_TABLE))
    assert not unknown, {code: sorted(used[code]) for code in unknown}
    dead = sorted(set(CODE_TABLE) - set(used) - set(RESERVED))
    assert not dead, f"in the table but raised nowhere (raise it, or reserve it with a reason): {dead}"


def test_every_reserved_code_is_in_the_table_and_has_a_reason():
    assert set(RESERVED) <= set(CODE_TABLE)
    assert all(reason.strip() for reason in RESERVED.values())
    assert not set(RESERVED) & set(_scan()), "a reserved code that is raised is not reserved"


def test_the_table_is_well_formed():
    assert len(CODE_TABLE) == 127
    for code, row in CODE_TABLE.items():
        assert row.code == code
        assert row.severity == {"E": "error", "W": "warning", "C": "violation", "F": "finding"}[code[0]]
        assert re.fullmatch(r"\d+(\.\d+)?", row.section)
        assert row.raised_by.strip() and row.meaning.strip()
    assert set(FAMILIES) == {"S", "O", "U", "Y"}


@pytest.mark.parametrize("code,low,high", [
    ("E", 1, 96), ("W", 1, 53), ("C", 1, 12), ("F", 1, 12),
])
def test_codes_stay_inside_the_allocated_ranges(code, low, high):
    numbers = [int(c[1:]) for c in CODE_TABLE if c[0] == code]
    assert low <= min(numbers) and max(numbers) <= high
    if code == "E":
        gaps = {n for n in numbers if 19 <= n <= 39 or n in (58, 59)}
        assert not gaps, f"reserved gaps were used: {sorted(gaps)}"


def test_the_table_matches_the_live_registries():
    """Each registry's own family tables agree with 17.2.2."""
    from mactf.components.known import PERSISTENCE_CODES

    assert set(PERSISTENCE_CODES) <= set(CODE_TABLE)
    # the exception classes carry codes that exist in the table
    from mactf import exceptions

    for name in dir(exceptions):
        cls = getattr(exceptions, name)
        code = getattr(cls, "code", None)
        if isinstance(cls, type) and isinstance(code, str) and re.fullmatch(r"[EWCF]\d{3}", code):
            assert code in CODE_TABLE, f"{name}.code = {code}"
    # the RL, optimizer, weight and strategy modules' own code constants
    from mactf.components.guards import compliance, rollup

    for module in (compliance, rollup):
        for code in getattr(module, "CODES", {}) or {}:
            if re.fullmatch(r"[EWCF]\d{3}", str(code)):
                assert code in CODE_TABLE, (module.__name__, code)
