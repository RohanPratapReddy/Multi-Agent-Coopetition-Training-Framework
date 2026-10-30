"""Document 17 section 19.3.3 row 19: every ``mactf/...`` path the shipped docs cite exists."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ("docs/components.md", "docs/api.md", "docs/cli.md")
CITED = re.compile(r"`((?:mactf|tests|scripts|docs)/[A-Za-z0-9_./*-]+)`")


@pytest.mark.parametrize("doc", SHIPPED)
def test_cited_paths_exist(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = []
    for match in CITED.finditer(text):
        cited = match.group(1).rstrip(".")
        if "*" in cited:
            if not list(ROOT.glob(cited)):
                missing.append(cited)
        elif not (ROOT / cited).exists():
            missing.append(cited)
    assert not missing, f"{doc} cites paths that do not exist: {sorted(set(missing))}"


def test_the_guide_cites_something():
    assert len(CITED.findall((ROOT / "docs/components.md").read_text(encoding="utf-8"))) >= 4
