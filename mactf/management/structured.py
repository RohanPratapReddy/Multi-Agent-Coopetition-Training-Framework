"""Read a JSON or YAML document from a file, standard input or an inline string.

Shared by the verbs that take a spec or a plan (``components validate --spec-file``,
``runs components set --spec``, ``peers mutate PLAN``): one reader, one set of error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import CommandError, ResourceNotFound


def read_structured(command: Any, source: str, *, inline: bool = False, root: type = dict) -> Any:
    """The parsed document ``source`` names.

    ``-`` reads ``command.stdin``. With ``inline`` a string that is not an existing file is
    parsed as the document itself, so ``--spec '{params: {temperature: 2}}'`` works. ``root`` is
    the type the top level must have (``dict``, or ``(dict, list)`` for a plan).
    """
    import yaml

    if source == "-":
        raw, label = command.stdin.read(), "standard input"
    else:
        path = Path(source).expanduser()
        if inline and not path.is_file():
            raw, label = source, "the inline document"
        else:
            try:
                raw, label = path.read_text(encoding="utf-8"), str(path)
            except OSError as exc:
                raise ResourceNotFound(f"{source}: {exc.strerror or exc}") from exc
    try:
        data = yaml.safe_load(raw)            # JSON is YAML
    except yaml.YAMLError as exc:
        raise CommandError.usage(f"{label}: not valid JSON or YAML: {exc}") from exc
    if data is None:
        data = root() if isinstance(root, type) else {}
    if not isinstance(data, root):
        wanted = " or ".join(t.__name__ for t in (root if isinstance(root, tuple) else (root,)))
        raise CommandError.usage(f"{label}: expected a {wanted} at the top level, got {type(data).__name__}")
    return data
