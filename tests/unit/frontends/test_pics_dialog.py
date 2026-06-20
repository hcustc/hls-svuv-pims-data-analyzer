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
from bl03u_masstool.frontends.pyqt_app.pics.dialog import PICSCalculatorDialog
from bl03u_masstool.frontends.pyqt_app.pics.import_widget import PICSImportWidget


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
        assert any("导入" in button.text() for button in buttons)
    finally:
        widget.deleteLater()
