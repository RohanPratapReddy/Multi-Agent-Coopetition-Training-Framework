"""test lazy import (split from test_structure.py)."""

from __future__ import annotations

import subprocess


import sys


def test_importing_the_client_loads_no_engine():
    code = (
        "import sys, mactf.client\n"
        "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith(('mactf.training', 'mactf.contrib', "
        "'mactf.coopetition', 'mactf.scoring', 'mactf.orchestration')))\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
