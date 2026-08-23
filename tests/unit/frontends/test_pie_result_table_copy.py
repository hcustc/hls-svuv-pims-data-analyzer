from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtTest, QtWidgets
except ImportError as exc:
    pytest.skip(
        f"PyQt6 display libraries not available: {exc}",
        allow_module_level=True,
    )

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog
from bl03u_masstool.frontends.pyqt_app.pie.result_display_widget import (
    CopyableTableWidget,
)


pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _table_with_curve_data() -> CopyableTableWidget:
    table = CopyableTableWidget()
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(["能量(eV)", "实验值"])
    table.setRowCount(2)
    for row, values in enumerate((("7.5998", "0.000000"), ("7.6500", "0.000247"))):
        for column, value in enumerate(values):
            table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
    return table


def test_curve_table_copies_selected_cells_with_standard_shortcut(qapp):
    table = _table_with_curve_data()
    table.show()
    table.setFocus()
    table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, 1, 1), True)
    qapp.processEvents()
    qapp.clipboard().setText("unchanged")

    QtTest.QTest.keySequence(
        table,
        QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy),
    )
    qapp.processEvents()

    assert qapp.clipboard().text() == (
        "7.5998\t0.000000\n"
        "7.6500\t0.000247"
    )
    table.deleteLater()


def test_curve_table_copies_all_cells_with_headers(qapp):
    table = _table_with_curve_data()
    qapp.clipboard().setText("unchanged")

    table.copy_all_to_clipboard()

    assert qapp.clipboard().text() == (
        "能量(eV)\t实验值\n"
        "7.5998\t0.000000\n"
        "7.6500\t0.000247"
    )
    table.deleteLater()


def test_curve_table_context_menu_can_copy_all(qapp):
    table = _table_with_curve_data()
    qapp.clipboard().setText("unchanged")
    copy_all_action = next(
        action for action in table.actions() if action.text() == "复制全部"
    )

    copy_all_action.trigger()

    assert qapp.clipboard().text() == (
        "能量(eV)\t实验值\n"
        "7.5998\t0.000000\n"
        "7.6500\t0.000247"
    )
    table.deleteLater()


def test_result_detail_popup_copy_all_button_copies_current_tab(qapp):
    dialog = PIESpeciesFitDialog(Calibration(a=0.0, b=1.0, c=0.0))
    dialog.current_mz = 92.03
    dialog.result_display_widget.update_curve_data(
        {
            "energies": [7.5998],
            "experimental": [0.000247],
            "total_fit": [0.1],
            "residual": [-0.099753],
        }
    )
    dialog.show()
    qapp.clipboard().setText("unchanged")
    observed = {"copy_button": False}

    def use_popup() -> None:
        popup = next(
            widget
            for widget in qapp.topLevelWidgets()
            if isinstance(widget, QtWidgets.QDialog)
            and widget is not dialog
            and widget.windowTitle().startswith("详细数据 -")
        )
        copy_button = next(
            (
                button
                for button in popup.findChildren(QtWidgets.QPushButton)
                if button.text() == "复制全部"
            ),
            None,
        )
        observed["copy_button"] = copy_button is not None
        if copy_button is not None:
            copy_button.click()
        popup.accept()

    QtCore.QTimer.singleShot(0, use_popup)
    dialog.show_result_detail_btn.click()

    assert observed["copy_button"]
    assert qapp.clipboard().text() == (
        "能量(eV)\t实验值\t拟合值\t残差\n"
        "7.5998\t0.000247\t0.100000\t-0.099753"
    )
    dialog.deleteLater()
