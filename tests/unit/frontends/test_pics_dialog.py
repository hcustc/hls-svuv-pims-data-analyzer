from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.pie_analysis import save_species_database_sqlite
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.frontends.pyqt_app.core_tools.dialog import CoreToolsDialog
from bl03u_masstool.frontends.pyqt_app.pics.dialog import PICSCalculatorDialog
from bl03u_masstool.frontends.pyqt_app.pics.import_widget import PICSImportWidget
from bl03u_masstool.frontends.pyqt_app.spectrum.workbench import MainWindow


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_species_page_splits_reference_and_import_tools_into_inner_tabs(qapp):
    dialog = PICSCalculatorDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
            "1. 物种与参考",
            "2. PIE 信号",
            "3. 计算设置",
            "4. 结果",
        ]
        assert dialog.species_tabs.count() == 2
        assert [dialog.species_tabs.tabText(i) for i in range(2)] == ["基础参数", "NO截面库"]

        basics_tab = dialog.species_tabs.widget(0)
        reference_tab = dialog.species_tabs.widget(1)

        assert basics_tab.findChildren(QtWidgets.QTableWidget) == []
        assert dialog.no_cs_table in reference_tab.findChildren(QtWidgets.QTableWidget)
        assert not dialog.btn_export_csv.isEnabled()
        assert not dialog.btn_export_database.isEnabled()
    finally:
        dialog.deleteLater()


def test_species_continue_validates_and_advances(qapp):
    dialog = PICSCalculatorDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.species_continue_button.click()
        assert dialog.tabs.currentIndex() == 0
        assert dialog.status_label.property("status") == "error"

        dialog.txt_new_name.setText("test species")
        dialog.txt_new_formula.setText("CH4")
        dialog.spin_new_mz.setValue(16)
        dialog.species_continue_button.click()

        assert dialog.tabs.currentIndex() == 1
        assert dialog.new_species_mz == 16
        assert dialog.status_label.property("status") == "success"
    finally:
        dialog.deleteLater()


def test_pics_import_widget_initializes(qapp):
    widget = PICSImportWidget()
    try:
        assert widget is not None
        buttons = widget.findChildren(QtWidgets.QPushButton)
        assert any("选择" in button.text() and "文件" in button.text() for button in buttons)
        # Check that preview table exists
        tables = widget.findChildren(QtWidgets.QTableWidget)
        assert len(tables) > 0
        assert not widget.confirm_import_button.isEnabled()
        assert [widget.import_mode_combo.itemData(i) for i in range(widget.import_mode_combo.count())] == [
            "upsert",
            "append",
        ]
    finally:
        widget.deleteLater()


def test_pics_calculator_and_import_use_project_database(qapp, tmp_path):
    database_path = tmp_path / "project.sqlite"
    save_species_database_sqlite(
        [
            {
                "mz": 30,
                "species": "ProjectNO",
                "ie": 9.2,
                "energies": np.array([10.0, 11.0]),
                "cross_sections": np.array([1.0, 2.0]),
            }
        ],
        database_path,
    )
    settings = ProjectSettings(pics_database_path=str(database_path))
    calculator = PICSCalculatorDialog(Calibration(), None)
    importer = PICSImportWidget()
    try:
        calculator.set_project_settings(settings)
        importer.set_project_settings(settings)

        assert calculator._loaded_database_path == str(database_path)
        assert any(item["species"] == "ProjectNO" for item in calculator.database)
        assert importer.database_path() == database_path
    finally:
        calculator.deleteLater()
        importer.deleteLater()


def test_pics_import_preview_requires_explicit_confirmation(qapp):
    widget = PICSImportWidget()
    records = [{"species": "NO", "mz": 30, "ie": 9.264, "energies": [10.0, 11.0], "cross_sections": [1.0, 2.0]}]
    try:
        widget._show_preview(records)

        assert widget.preview_table.rowCount() == 1
        assert widget.preview_table.item(0, 0).text() == "NO"
        assert widget.preview_table.item(0, 3).text() == "10.000–11.000"
        assert widget.confirm_import_button.isEnabled()
        assert widget.result_label.text() == "请检查预览后确认导入"
    finally:
        widget.deleteLater()


def test_workspace_pics_import_switches_to_import_page(qapp):
    window = MainWindow()
    try:
        window.switch_workspace_page("pics")
        assert window.workspace_stack.currentWidget() is window.pics_page

        window.switch_workspace_page("pics_import")

        assert window.workspace_stack.currentWidget() is window.pics_import_page
        assert window.workspace_stack.currentWidget() is not window.pics_page
        assert window.page_buttons["pics_import"].isChecked()
    finally:
        window.deleteLater()


def test_pics_import_navigation_and_completion_refresh_calculator(qapp, monkeypatch):
    window = MainWindow()
    refreshed = []
    try:
        monkeypatch.setattr(window.pics_page, "refresh_database", lambda: refreshed.append(True))
        window.switch_workspace_page("pics_import")
        window.pics_import_page.open_calculator_button.click()
        assert window.workspace_stack.currentWidget() is window.pics_page

        window.pics_import_page.import_completed.emit({"inserted_species": 1})
        assert refreshed == [True]
    finally:
        window.deleteLater()


def test_core_tools_dialog_has_distinct_pics_import_tab(qapp):
    dialog = CoreToolsDialog(Calibration(a=0.0, b=1.0, c=0.0), initial_tab="pics_import")
    try:
        labels = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]

        assert "PICS计算" in labels
        assert "PICS导入" in labels
        assert dialog.tabs.currentWidget().__class__ is PICSImportWidget
    finally:
        dialog.deleteLater()
