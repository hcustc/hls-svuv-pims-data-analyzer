from __future__ import annotations

import ast
from pathlib import Path

import bl03u_masstool


def test_package_version_uses_release_version_source() -> None:
    version_file = Path(bl03u_masstool.__file__).with_name("_version.py")
    module = ast.parse(version_file.read_text(encoding="utf-8"))
    assigned_version = next(
        node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
        and isinstance(node.value, ast.Constant)
    )

    assert bl03u_masstool.__version__ == assigned_version == "0.2.3"
