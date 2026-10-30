"""``startproject`` - scaffold a new MACTF project.

The equivalent of ``django-admin startproject``: a distribution install has no
settings module, so this command writes a runnable project with a settings
module, a peer package, YAML overlays, and a ``manage.py``.

Templates live in ``mactf/templates/project`` as shipped data files. They are
plain text with a literal ``{{ project_name }}`` token, replaced with
:meth:`str.replace` rather than ``str.format`` so Python dict braces in
``settings.py`` survive untouched.
"""

from __future__ import annotations

import keyword
from pathlib import Path

from ..base import BaseCommand
from ...conf_files import CONF_LAYOUT, CONF_SUBDIR
from ...exceptions import CommandError

TEMPLATE_PACKAGE = "mactf.templates.project"
#: Package holding the ``conf/*.yaml.tpl`` masters (document 15).
CONF_TEMPLATE_PACKAGE = "mactf.templates.conf"

_LAYOUT: list[tuple[str, str]] = [
    ("manage.py.tpl", "manage.py"),
    ("package_init.py.tpl", "__init__.py"),
    ("settings.py.tpl", "settings.py"),
    ("peers/__init__.py.tpl", "peers/__init__.py"),
]
# Generated from CONF_LAYOUT so the files the loader reads and the files the
# scaffolder writes are one list. Entries under ``conf/`` are read from
# ``conf_template_root()``, every other entry from ``template_root()``.
_LAYOUT.extend(
    (f"{CONF_SUBDIR}/{entry.template_name}", f"{CONF_SUBDIR}/{entry.filename}")
    for entry in CONF_LAYOUT
)

# (template file name, destination path relative to the project directory)
LAYOUT: tuple[tuple[str, str], ...] = tuple(_LAYOUT)

TOKEN = "{{ project_name }}"

#: Lines added to the generated ``.gitignore`` beyond the blanket ``var/``, so a
#: project that un-ignores one subdirectory never commits runs or teacher weights.
_GITIGNORE = (
    "var/\nvar/runs/\nvar/teachers/\nvar/audits/\nvar/cache/\nvar/state/\n"
    "__pycache__/\n*.py[cod]\n.pytest_cache/\n*.egg-info/\ndist/\nbuild/\n"
)


def template_root() -> Path:
    """Directory holding the shipped project templates."""
    from importlib import resources

    return Path(resources.files(TEMPLATE_PACKAGE))


def conf_template_root() -> Path:
    """Directory holding the shipped ``conf/*.yaml.tpl`` masters."""
    from importlib import resources

    return Path(resources.files(CONF_TEMPLATE_PACKAGE))


def template_source(template_name: str) -> tuple[Path, str]:
    """The file for a LAYOUT template name, and the package it must ship in."""
    prefix = f"{CONF_SUBDIR}/"
    if template_name.startswith(prefix):
        return conf_template_root() / template_name[len(prefix):], CONF_TEMPLATE_PACKAGE
    return template_root() / template_name, TEMPLATE_PACKAGE


def render(text: str, project_name: str) -> str:
    """Substitute the project name without interpreting other braces."""
    return text.replace(TOKEN, project_name)


def validate_project_name(name: str) -> str:
    """Reject names that cannot be imported as a package."""
    if not name:
        raise CommandError.usage("a project name is required")
    if not name.isidentifier():
        raise CommandError.usage(
            f"{name!r} is not a valid Python identifier; use letters, digits, and underscores"
        )
    if keyword.iskeyword(name):
        raise CommandError.usage(f"{name!r} is a Python keyword; choose another name")
    return name


def write_project(project_name: str, *, target_root: Path, force: bool = False,
                  dry_run: bool = False) -> list[Path]:
    """The files written (or, with ``dry_run``, that would be).

    Everything is read and rendered before anything is written, so a template missing from
    an incomplete install cannot leave a half-written project behind.
    """
    validate_project_name(project_name)
    package_dir = target_root / project_name
    if (package_dir / "settings.py").exists() and not force:
        raise CommandError.conflict(
            f"{package_dir} already exists; pass --force to overwrite the generated files"
        )

    rendered: list[tuple[Path, str]] = []
    for template_name, relative in LAYOUT:                      # 1. read and render all of it
        source, package = template_source(template_name)
        if not source.is_file():  # pragma: no cover - packaging regression guard
            raise CommandError(
                f"template {template_name!r} is missing from {package}; "
                "the installation is incomplete"
            )
        # conf/*.yaml is rendered with the same substitution: YAML has no Python braces
        # to preserve, and only settings and a few comments carry the name.
        rendered.append((package_dir / relative, render(source.read_text(encoding="utf-8"), project_name)))
    # Keep generated runtime state out of a future wheel/sdist and out of git.
    ignore = package_dir / ".gitignore"
    if not ignore.exists() or force:
        rendered.append((ignore, _GITIGNORE))

    if dry_run:                                                 # 2. stop here, having touched nothing
        return [destination for destination, _ in rendered]
    for destination, text in rendered:                          # 3. only now write
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
    return [destination for destination, _ in rendered]


class Command(BaseCommand):
    help = "Scaffold a new MACTF project (settings, peers, configs, manage.py)."
    category = "Scaffold"
    requires_settings = False
    output_formats = ("text", "json")
    examples = ("mactf startproject myswarm", "mactf startproject myswarm --dir D:/work --dry-run")
    arguments = [
        (("name",), {"help": "Project package name, for example `myswarm`."}),
        (("--dir",), {"default": None, "metavar": "PATH", "completer": "directory",
                      "help": "Parent directory; defaults to the working directory."}),
        (("--force",), {"action": "store_true", "default": False,
                        "help": "Overwrite generated files that already exist."}),
        (("--dry-run",), {"action": "store_true", "default": False,
                          "help": "List the files that would be written; write nothing."}),
    ]

    def handle(self, **options):
        name = validate_project_name(options["name"].strip())
        target_root = Path(options["dir"] or Path.cwd()).expanduser().resolve()
        written = write_project(name, target_root=target_root, force=options["force"],
                                dry_run=options["dry_run"])
        created = [path.relative_to(target_root).as_posix() for path in written]
        steps = [f"cd {name}", "edit conf/trainer.yaml to set epochs, batch size, and the task",
                 "python manage.py check", "python manage.py migrate", "python manage.py run --epochs 3"]
        payload = {"project": name, "root": str(target_root), "directory": str(target_root / name),
                   "dry_run": options["dry_run"], "created": created, "next_steps": steps}

        def text() -> None:
            verb = "would create" if options["dry_run"] else "created"
            self.out.result(f"{verb} project {name!r} in {target_root}")
            for item in created:
                self.out.result(f"  {Path(item)}")      # the platform's own separator in text
            if not options["dry_run"]:
                self.out.result("")
                self.out.result("next steps:")
                for step in steps:
                    self.out.result(f"  {step}")
        self.out.emit(payload, text=text)
        return 0


