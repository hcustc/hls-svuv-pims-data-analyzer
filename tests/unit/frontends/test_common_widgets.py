from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin, ElidedLabel


pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _FailingTable(QtWidgets.QTableWidget):
    def setItem(self, row, column, item):
        del row, column, item
        raise RuntimeError("test insertion failure")


def test_elided_label_sets_tooltip_for_initial_text(qapp):
    label = ElidedLabel("完整参数摘要")
    try:
        assert label.toolTip() == "完整参数摘要"
    finally:
        label.deleteLater()


def test_elided_label_exposes_full_text_when_rendered_text_is_elided(qapp):
    full_text = "项目：A deliberately long project name"
    label = ElidedLabel(full_text)
    try:
        label.resize(24, label.sizeHint().height())
        label.show()
        qapp.processEvents()

        assert label.text() != full_text
        assert label.fullText() == full_text
    finally:
        label.deleteLater()


def test_dataframe_table_restores_widget_state_after_error(qapp):
    table = _FailingTable()
    table.blockSignals(False)
    table.setUpdatesEnabled(True)

    with pytest.raises(RuntimeError, match="insertion failure"):
        DataFrameTableMixin().set_dataframe(table, pd.DataFrame({"value": [1]}))

    assert table.updatesEnabled()
    assert not table.signalsBlocked()
    table.deleteLater()
