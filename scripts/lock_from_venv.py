from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def main() -> int:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    runtime_roots = [
        canonicalize_name(Requirement(requirement).name)
        for requirement in pyproject["project"]["dependencies"]
    ]
    dev_roots = runtime_roots + [
        canonicalize_name(Requirement(requirement).name)
        for requirement in pyproject["project"]["optional-dependencies"]["dev"]
    ]

    installed = {canonicalize_name(dist.metadata["Name"]): dist for dist in metadata.distributions()}
    runtime_names, runtime_missing = _dependency_closure(runtime_roots, installed)
    dev_names, dev_missing = _dependency_closure(dev_roots, installed)

    _write_lock(PROJECT_ROOT / "requirements.lock", runtime_names, installed, "runtime dependencies")
    dev_scope = "runtime + test/packaging dependencies"
    if dev_missing:
        dev_scope += f"; missing from current venv: {', '.join(sorted(dev_missing))}"
    _write_lock(PROJECT_ROOT / "requirements-dev.lock", dev_names, installed, dev_scope)

    for name in sorted(runtime_missing | dev_missing):
        print(f"warning: {name!r} is declared but not installed in the active environment")
    return 0


def _dependency_closure(
    roots: list[str],
    installed: dict[str, metadata.Distribution],
) -> tuple[set[str], set[str]]:
    env = default_environment()
    seen: set[str] = set()
    missing: set[str] = set()
    stack = list(roots)

    while stack:
        name = canonicalize_name(stack.pop())
        if name in seen:
            continue
        dist = installed.get(name)
        if dist is None:
            missing.add(name)
            continue
        seen.add(name)
        for requirement_text in dist.requires or []:
            requirement = Requirement(requirement_text)
            if requirement.marker is not None and not requirement.marker.evaluate(env):
                continue
            dep_name = canonicalize_name(requirement.name)
            if dep_name not in seen:
                stack.append(dep_name)

    return seen, missing


def _write_lock(
    path: Path,
    names: set[str],
    installed: dict[str, metadata.Distribution],
    scope: str,
) -> None:
    lines = [
        "# This file is generated from the project .venv metadata.",
        "# Update it after dependency changes with: python scripts/lock_from_venv.py",
        f"# Scope: {scope}",
        "",
    ]
    for name in sorted(names):
        dist = installed[name]
        normalized = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        lines.append(f"{normalized}=={dist.version}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
