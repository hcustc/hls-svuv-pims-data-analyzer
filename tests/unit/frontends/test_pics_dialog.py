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

from bl03u_masstool.core.calibration import Calibration
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
        assert dialog.species_tabs.count() == 2
        assert [dialog.species_tabs.tabText(i) for i in range(2)] == ["基础参数", "NO截面库"]

        basics_tab = dialog.species_tabs.widget(0)
        reference_tab = dialog.species_tabs.widget(1)

        assert basics_tab.findChildren(QtWidgets.QTableWidget) == []
        assert dialog.no_cs_table in reference_tab.findChildren(QtWidgets.QTableWidget)
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


def test_core_tools_dialog_has_distinct_pics_import_tab(qapp):
    dialog = CoreToolsDialog(Calibration(a=0.0, b=1.0, c=0.0), initial_tab="pics_import")
    try:
        labels = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]

        assert "PICS计算" in labels
        assert "PICS导入" in labels
        assert dialog.tabs.currentWidget().__class__ is PICSImportWidget
    finally:
        dialog.deleteLater()
