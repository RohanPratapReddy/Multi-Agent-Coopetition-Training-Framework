import json
import os
from pathlib import Path

from mactf.testing.cli import invoke

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("MACTF_UPDATE_GOLDEN") == "1"


def _check(path: Path, actual: str) -> None:
    if UPDATE:
        path.write_text(actual, encoding="utf-8", newline="\n")
    expected = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert actual == expected, (
        f"{path.name} changed; review the diff, then MACTF_UPDATE_GOLDEN=1 pytest {__file__}")


def test_help_json_is_stable():
    payload = invoke(["help", "--format", "json"]).json()
    payload["version"] = "<version>"
    _check(GOLDEN / "help.json", json.dumps(payload, indent=2, sort_keys=False) + "\n")


def test_overview_text_is_stable():
    out = invoke(["help"], env={"NO_COLOR": "1", "COLUMNS": "80"})
    assert out.code == 0
    _check(GOLDEN / "overview.txt", out.stdout)


def test_help_json_schema_is_additive_only():
    """Every key of the committed golden is still present; new keys are allowed (8.19)."""
    golden = json.loads((GOLDEN / "help.json").read_text(encoding="utf-8"))
    live = invoke(["help", "--format", "json"]).json()
    assert live["schema"] == golden["schema"]

    def keys(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield prefix + "." + key
                yield from keys(value, prefix + "." + key)
        elif isinstance(node, list) and node:
            yield from keys(node[0], prefix + "[]")

    assert set(keys(golden)) <= set(keys(live))
