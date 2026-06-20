from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

pytestmark = pytest.mark.gui

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.frontends.pyqt_app.pics.dialog import PICSCalculatorDialog


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_species_page_splits_reference_and_import_tools_into_inner_tabs(qapp):
    dialog = PICSCalculatorDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        assert dialog.species_tabs.count() == 3
        assert [dialog.species_tabs.tabText(i) for i in range(3)] == ["基础参数", "NO截面库", "外部导入"]

        basics_tab = dialog.species_tabs.widget(0)
        reference_tab = dialog.species_tabs.widget(1)
        import_tab = dialog.species_tabs.widget(2)

        assert basics_tab.findChildren(QtWidgets.QTableWidget) == []
        assert dialog.no_cs_table in reference_tab.findChildren(QtWidgets.QTableWidget)
        assert any(button.text() == "从文件导入 PICS 到数据库…" for button in import_tab.findChildren(QtWidgets.QPushButton))
    finally:
        dialog.deleteLater()
