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
from ...exceptions import CommandError

TEMPLATE_PACKAGE = "mactf.templates.project"

# (template file name, destination path relative to the project directory)
LAYOUT: tuple[tuple[str, str], ...] = (
    ("manage.py.tpl", "manage.py"),
    ("package_init.py.tpl", "__init__.py"),
    ("settings.py.tpl", "settings.py"),
    ("peers/__init__.py.tpl", "peers/__init__.py"),
    ("configs/swarm.yaml.tpl", "configs/swarm.yaml"),
    ("configs/scoring.yaml.tpl", "configs/scoring.yaml"),
    ("configs/sandbox.yaml.tpl", "configs/sandbox.yaml"),
)

TOKEN = "{{ project_name }}"


def template_root() -> Path:
    """Directory holding the shipped templates."""
    from importlib import resources

    return Path(resources.files(TEMPLATE_PACKAGE))


def render(text: str, project_name: str) -> str:
    """Substitute the project name without interpreting other braces."""
    return text.replace(TOKEN, project_name)


def validate_project_name(name: str) -> str:
    """Reject names that cannot be imported as a package."""
    if not name:
        raise CommandError("a project name is required")
    if not name.isidentifier():
        raise CommandError(
            f"{name!r} is not a valid Python identifier; use letters, digits, and underscores"
        )
    if keyword.iskeyword(name):
        raise CommandError(f"{name!r} is a Python keyword; choose another name")
    return name


def write_project(project_name: str, *, target_root: Path, force: bool = False) -> list[Path]:
    """Create the project tree and return the files written."""
    validate_project_name(project_name)
    package_dir = target_root / project_name
    settings_path = package_dir / "settings.py"
    if settings_path.exists() and not force:
        raise CommandError(
            f"{package_dir} already exists; pass --force to overwrite the generated files"
        )

    root = template_root()
    written: list[Path] = []
    for template_name, relative in LAYOUT:
        source = root / template_name
        if not source.is_file():  # pragma: no cover - packaging regression guard
            raise CommandError(
                f"template {template_name!r} is missing from {TEMPLATE_PACKAGE}; "
                "the installation is incomplete"
            )
        destination = package_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            render(source.read_text(encoding="utf-8"), project_name),
            encoding="utf-8",
        )
        written.append(destination)

    # Keep generated runtime state out of a future wheel/sdist and out of git.
    ignore = package_dir / ".gitignore"
    if not ignore.exists() or force:
        ignore.write_text(
            "var/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n*.egg-info/\ndist/\nbuild/\n",
            encoding="utf-8",
        )
        written.append(ignore)
    return written


class Command(BaseCommand):
    help = "Scaffold a new MACTF project (settings, peers, configs, manage.py)."
    arguments = [
        (("name",), {"help": "Project package name, for example `myswarm`."}),
        (("--dir",), {"default": None, "help": "Parent directory; defaults to the working directory."}),
        (("--force",), {"action": "store_true", "default": False}),
    ]

    def handle(self, **options):
        project_name = validate_project_name(options["name"].strip())
        target_root = Path(options.get("dir") or Path.cwd()).expanduser().resolve()
        target_root.mkdir(parents=True, exist_ok=True)

        written = write_project(
            project_name,
            target_root=target_root,
            force=bool(options.get("force")),
        )
        self.success(f"created project {project_name!r} in {target_root}")
        for path in written:
            self.success(f"  {path.relative_to(target_root)}")
        self.success("")
        self.success("next steps:")
        self.success(f"  cd {project_name}")
        self.success("  python manage.py check")
        self.success("  python manage.py migrate")
        self.success("  python manage.py run --epochs 3")
        return 0
