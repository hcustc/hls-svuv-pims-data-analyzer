from __future__ import annotations

import ast
from pathlib import Path


FRONTEND_TEST_ROOT = Path(__file__).parent / "frontends"


def _imports_pyqt(module: ast.Module) -> bool:
    return any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name == "PyQt6" or alias.name.startswith("PyQt6.") for alias in node.names)
            if isinstance(node, ast.Import)
            else bool(node.module and (node.module == "PyQt6" or node.module.startswith("PyQt6.")))
        )
        for node in ast.walk(module)
    )


def _has_gui_marker(module: ast.Module) -> bool:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "gui"
            and isinstance(value.value, ast.Attribute)
            and value.value.attr == "mark"
        ):
            return True
    return False


def _has_guarded_pyqt_import(module: ast.Module) -> bool:
    for node in ast.walk(module):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = any(
            isinstance(handler.type, ast.Name) and handler.type.id == "ImportError"
            for handler in node.handlers
        )
        if not catches_import_error:
            continue
        guarded_module = ast.Module(body=node.body, type_ignores=[])
        if _imports_pyqt(guarded_module):
            return True
    return False


def _has_module_level_skip(module: ast.Module) -> bool:
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "skip"
            and isinstance(function.value, ast.Name)
            and function.value.id == "pytest"
        ):
            continue
        if any(
            keyword.arg == "allow_module_level"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        ):
            return True
    return False


def test_gui_test_modules_are_safe_to_collect_without_native_qt_libraries():
    violations: list[str] = []
    for path in sorted(FRONTEND_TEST_ROOT.rglob("test_*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not _imports_pyqt(module):
            continue
        missing = []
        if not _has_gui_marker(module):
            missing.append("pytest.mark.gui")
        if not _has_guarded_pyqt_import(module):
            missing.append("PyQt6 import guarded by ImportError")
        if not _has_module_level_skip(module):
            missing.append("pytest.skip(..., allow_module_level=True)")
        if missing:
            violations.append(f"{path.relative_to(FRONTEND_TEST_ROOT)}: {', '.join(missing)}")

    assert not violations, "Unsafe GUI test collection:\n" + "\n".join(violations)
