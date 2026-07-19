from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin


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


def test_dataframe_table_restores_widget_state_after_error(qapp):
    table = _FailingTable()
    table.blockSignals(False)
    table.setUpdatesEnabled(True)

    with pytest.raises(RuntimeError, match="insertion failure"):
        DataFrameTableMixin().set_dataframe(table, pd.DataFrame({"value": [1]}))

    assert table.updatesEnabled()
    assert not table.signalsBlocked()
    table.deleteLater()
