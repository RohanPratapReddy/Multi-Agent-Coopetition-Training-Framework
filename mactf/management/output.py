"""Output for management commands: text, tables, JSON, JSONL, YAML, and colour.

Document 16, section 6.3. In text mode ``result`` and ``info`` go to stdout and
``debug``/``warn``/``error`` to stderr. In a machine mode (``json``, ``jsonl``, ``yaml``)
stdout carries *only* the payload; narrative is dropped below verbosity 2 and goes to
stderr at or above it.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

MACHINE_FORMATS = ("json", "jsonl", "yaml")


@dataclass(frozen=True)
class Column:
    key: str
    header: str
    align: str = "l"                          # "l" or "r"
    max_width: int | None = None
    fmt: Callable[[Any], str] | None = None   # None: "-" for None, else str()


def clean(value: Any) -> Any:
    """Make any value representable as JSON. Total: it never raises."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None     # NaN/Inf are not JSON
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((clean(item) for item in value), key=repr)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return clean(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return clean(dataclasses.asdict(value))
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        try:
            return clean(as_dict())
        except Exception:  # noqa: BLE001 - clean() is total
            return str(value)
    item = getattr(value, "item", None)                     # numpy / torch scalars
    if callable(item):
        try:
            return clean(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _enable_vt(stream: TextIO) -> bool:
    """True off Windows; on Windows switch the console to virtual-terminal mode."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetConsoleMode.restype = wintypes.BOOL
        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetConsoleMode.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:  # noqa: BLE001 - no colour is the safe answer
        return False


def resolve_color(
    stream: TextIO, *, no_color: bool = False, force_color: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """One colour decision per stream."""
    environ = os.environ if environ is None else environ
    if no_color:
        return False
    if force_color:
        return True
    if environ.get("NO_COLOR"):                              # no-color.org: any non-empty value
        return False
    if environ.get("FORCE_COLOR") not in (None, "", "0"):
        return True
    if environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    try:
        tty = bool(isatty and isatty())
    except (ValueError, OSError):
        tty = False
    return tty and _enable_vt(stream)


class Style:
    """ANSI styling that degrades to identity when disabled."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled and text else text

    def ok(self, text: str) -> str:
        return self._wrap("32", text)

    def warn(self, text: str) -> str:
        return self._wrap("33", text)

    def err(self, text: str) -> str:
        return self._wrap("31", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)


def _ellipsis(stream: TextIO) -> str:
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "…".encode(encoding)
        return "…"
    except (UnicodeEncodeError, LookupError):
        return "..."


class Printer:
    """The one object a command writes through."""

    def __init__(
        self, stdout: TextIO, stderr: TextIO, *, fmt: str = "text", verbosity: int = 1,
        color: bool | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.fmt = fmt
        self.verbosity = verbosity
        self.machine = fmt in MACHINE_FORMATS
        out_color = resolve_color(stdout) if color is None else bool(color)
        err_color = resolve_color(stderr) if color is None else bool(color)
        self.style = Style(out_color and not self.machine)
        self.err_style = Style(err_color)
        self._json_written = False

    # -- narrative -------------------------------------------------------
    def _write(self, stream: TextIO, text: str) -> None:
        stream.write(text + "\n")

    def result(self, text: str = "") -> None:
        """The command's answer; printed at any verbosity."""
        if not self.machine:
            self._write(self.stdout, text)
        elif self.verbosity >= 2:
            self._write(self.stderr, text)

    def info(self, text: str = "") -> None:
        """Ordinary narrative (verbosity >= 1)."""
        if self.verbosity < 1:
            return
        if not self.machine:
            self._write(self.stdout, text)
        elif self.verbosity >= 2:
            self._write(self.stderr, text)

    def debug(self, text: str = "") -> None:
        if self.verbosity >= 2:
            self._write(self.stderr, text)

    # -- diagnostics -----------------------------------------------------
    def warn(self, text: str) -> None:
        self._write(self.stderr, self.err_style.warn(f"warning: {text}"))

    def error(self, text: str) -> None:
        self._write(self.stderr, self.err_style.err(f"error: {text}"))

    # -- structure -------------------------------------------------------
    def kv(self, pairs: Sequence[tuple[str, Any]], *, indent: int = 2) -> None:
        """Aligned ``key  value`` lines (text mode only)."""
        if self.machine or not pairs:
            return
        width = max(len(str(key)) for key, _ in pairs)
        pad = " " * indent
        for key, value in pairs:
            shown = "-" if value is None else str(value)
            self._write(self.stdout, f"{pad}{self.style.dim(str(key).ljust(width))}  {shown}")

    def table(
        self, rows: Sequence[Mapping[str, Any]], columns: Sequence[Column], *, empty: str = "(none)"
    ) -> None:
        if self.machine:
            return
        if not rows:
            self._write(self.stdout, self.style.dim(empty))
            return
        ellipsis = _ellipsis(self.stdout)
        cells: list[list[str]] = []
        for row in rows:
            line = []
            for column in columns:
                value = row.get(column.key)
                if value is None:
                    text = "-"                       # a formatter never has to handle None
                elif column.fmt is not None:
                    text = column.fmt(value)
                else:
                    text = str(value)
                line.append(" ".join(str(text).split()) if "\n" in str(text) else str(text))
            cells.append(line)
        widths = [
            min(max([len(column.header), *(len(line[i]) for line in cells)]),
                column.max_width or 10 ** 9)
            for i, column in enumerate(columns)
        ]
        tty = False
        try:
            tty = bool(self.stdout.isatty())
        except (AttributeError, ValueError, OSError):
            tty = False
        if tty:
            limit = shutil.get_terminal_size((100, 24)).columns
            gaps = 2 * (len(columns) - 1)
            flexible = [i for i, column in enumerate(columns) if column.max_width is None]
            while flexible and sum(widths) + gaps > limit:
                widest = max(flexible, key=lambda i: widths[i])
                if widths[widest] <= 12:
                    break
                widths[widest] -= 1

        def fit(text: str, width: int) -> str:
            if len(text) <= width:
                return text
            return text[: max(0, width - len(ellipsis))] + ellipsis

        def render(values: Sequence[str]) -> str:
            parts = []
            for i, column in enumerate(columns):
                shown = fit(values[i], widths[i])
                parts.append(shown.rjust(widths[i]) if column.align == "r" else shown.ljust(widths[i]))
            return "  ".join(parts).rstrip()

        self._write(self.stdout, self.style.bold(render([c.header for c in columns])))
        for line in cells:
            self._write(self.stdout, render(line))

    # -- machine payloads ------------------------------------------------
    def json(self, payload: Any) -> None:
        if self._json_written:
            raise RuntimeError("Printer.json() called twice; stdout carries exactly one document")
        self._json_written = True
        self.stdout.write(
            json.dumps(clean(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

    def jsonl(self, record: Any) -> None:
        self.stdout.write(
            json.dumps(clean(record), separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
        )
        self.stdout.flush()

    def yaml(self, payload: Any) -> None:
        import yaml

        self.stdout.write(
            yaml.safe_dump(clean(payload), sort_keys=False, allow_unicode=True, default_flow_style=False)
        )

    def event(self, record: Mapping[str, Any], *, text: str | None = None, level: int = 1) -> None:
        """One line of a stream.

        ``jsonl``: ``record`` as one compact line, always. ``text``: ``text`` through
        ``info`` (``result`` when ``level`` is 0), when given. ``json`` / ``yaml`` print
        exactly one final document, so they print nothing here.
        """
        if self.fmt == "jsonl":
            self.jsonl(record)
        elif self.fmt == "text" and text is not None:
            if level <= 0:
                self.result(text)
            else:
                self.info(text)

    def emit(self, payload: Any, *, text: Callable[[], None]) -> None:
        if self.fmt == "json":
            self.json(payload)
        elif self.fmt == "jsonl":
            self.jsonl(payload)
        elif self.fmt == "yaml":
            self.yaml(payload)
        else:
            text()


def format_bytes(count: int) -> str:
    """0 B, 1023 B, 1.5 KiB, 12.0 MiB, 3.2 GiB, 1.0 TiB."""
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")        # the loop returns at "TiB"


def silence_stdout() -> None:
    """Point stdout at the null device so the interpreter's exit flush cannot raise again."""
    import sys

    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass   # stdout has no file descriptor (a captured stream): nothing to redirect
