from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.project_settings import ProjectSettingsManager
from bl03u_masstool.frontends.pyqt_app.normalization.widget import FunctionDefaultsWidget
from bl03u_masstool.frontends.pyqt_app.spectrum.workbench import MainWindow


@pytest.fixture(autouse=True)
def isolated_project_settings(tmp_path):
    manager = ProjectSettingsManager()
    manager.set_project_path(tmp_path / "active_project")
    try:
        yield
    finally:
        manager.clear_project_path()


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_function_defaults_apply_to_project_settings(qapp):
    widget = FunctionDefaultsWidget()
    try:
        ps = ProjectSettings(peak_algorithm="legacy", detection_min_idx=111)
        widget.set_project_settings(ps)

        index = widget.peak_algorithm_combo.findData("cwt")
        assert index >= 0
        widget.peak_algorithm_combo.setCurrentIndex(index)
        widget.peak_detection_min_idx_edit.setValue(4321)
        widget.peak_min_intensity_edit.setValue(12.5)

        widget.apply_to_settings(ps)

        assert ps.peak_algorithm == "cwt"
        assert ps.detection_min_idx == 4321
        assert ps.min_intensity == pytest.approx(12.5)
    finally:
        widget.deleteLater()


def test_save_and_apply_uses_selected_parent_directory_for_new_project(qapp, tmp_path, monkeypatch):
    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir()

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    try:
        window.project_name_edit.setText("test")
        window.project_system_edit.setText("test")
        window.project_description_edit.setText("test")
        window.project_output_dir_edit.setText(str(downloads_dir))

        window.save_and_apply_project_settings()

        project_dir = downloads_dir / "test"
        assert project_dir.exists()
        assert (project_dir / "config" / "project.yaml").exists()
        assert window.project_output_dir_edit.text() == str(project_dir)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()
