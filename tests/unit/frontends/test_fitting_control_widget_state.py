"""Candidate-state and compact-layout tests for FittingControlWidget."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtCore, QtTest, QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.frontends.pyqt_app.pie.fitting_control_widget import FittingControlWidget
from bl03u_masstool.frontends.pyqt_app.theme import apply_application_theme


@pytest.fixture
def qapp():
    """Qt应用实例"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def widget(qapp):
    """FittingControlWidget实例"""
    return FittingControlWidget()


class TestLockedSpeciesState:
    def test_get_locked_species_returns_copy(self, widget):
        widget.lock_candidate('O2')
        widget.lock_candidate('N2')

        locked = widget.get_locked_species()
        locked.append('Ar')

        assert locked == ['O2', 'N2', 'Ar']
        assert widget.get_locked_species() == ['O2', 'N2']

    def test_set_locked_species_replaces_state(self, widget):
        widget.set_locked_species(['O2', 'N2', 'Ar'])
        assert widget._locked_species == ['O2', 'N2', 'Ar']

    def test_clear_ui_clears_locked_species(self, widget):
        widget.lock_candidate('O2')
        widget.lock_candidate('N2')

        widget.clear_ui()

        assert widget.get_locked_species() == []


class TestEdgeCases:
    """边界情况测试"""

    def test_add_duplicate_species(self, widget):
        widget.lock_candidate('O2')
        widget.lock_candidate('O2')

        assert widget.get_locked_species().count('O2') == 1

    def test_unlock_nonexistent_species(self, widget):
        """验证解锁不存在的物种的处理"""
        # 应该不抛出异常
        widget.unlock_candidate('NonExistent')
        assert widget._locked_species == []

    def test_set_locked_species_with_empty_list(self, widget):
        widget.lock_candidate('O2')
        widget.set_locked_species([])

        assert widget.get_locked_species() == []


def test_candidate_table_displays_ie_and_missing_query_status(widget):
    widget.populate_unified_species_table(
        30,
        [
            {
                "id": 1,
                "mz": 30,
                "species": "Nitric oxide",
                "formula": "NO",
                "smiles": "[N]=O",
                "ie": 9.2642,
                "energies": [9.0, 10.0],
                "cross_sections": [0.0, 1.0],
            },
            {
                "id": 2,
                "mz": 30,
                "species": "Unknown isomer",
                "ie": None,
                "energies": [9.0, 10.0],
                "cross_sections": [0.0, 1.0],
            },
        ],
        [],
    )

    assert widget.species_table.columnCount() == 3
    assert [
        widget.species_table.horizontalHeaderItem(column).text()
        for column in range(widget.species_table.columnCount())
    ] == ["启用", "候选物种", "IE (eV)"]
    assert widget.species_table.item(0, 2).text() == "9.2642"
    assert widget.species_table.item(1, 2).text() == "待查询"
    identity = widget.species_table.cellWidget(0, 1)
    assert identity.findChild(QtWidgets.QLabel, "SpeciesFormulaLabel").text() == "NO"
    assert widget.findChild(QtWidgets.QFrame, "SpeciesPreviewCard") is None

    widget.update_ionization_energy_for_ids(
        {2},
        value=None,
        status="not_found",
        source="NIST WebBook",
        message="未返回匹配物种",
    )
    assert widget.species_table.item(1, 2).text() == "未查到"
    assert widget.ie_query_btn.isEnabled()


def test_candidate_readiness_and_summary_follow_enabled_rows(widget):
    widget.populate_unified_species_table(
        30,
        [
            {"id": 1, "mz": 30, "species": "NO", "energies": [9.0, 10.0], "cross_sections": [0.0, 1.0]},
            {"id": 2, "mz": 30, "species": "N2", "energies": [9.0, 10.0], "cross_sections": [0.0, 1.0]},
        ],
        [],
    )

    ready, reason = widget.get_fit_readiness()
    assert ready is True
    assert reason == ""
    assert "候选 2" in widget.candidate_summary_label.text()
    assert "已选 2" in widget.candidate_summary_label.text()

    widget._set_all_rows_checked(False)

    ready, reason = widget.get_fit_readiness()
    assert ready is False
    assert "至少启用一个" in reason
    assert "已选 0" in widget.candidate_summary_label.text()


def test_candidate_panel_is_nnls_only_and_preview_guard_confirmation(widget):
    widget.populate_unified_species_table(
        30,
        [{"id": 1, "mz": 30, "species": "NO", "energies": [9.0, 10.0], "cross_sections": [0.0, 1.0]}],
        [],
    )
    assert not any(
        isinstance(
            widget.species_table.cellWidget(0, column),
            QtWidgets.QDoubleSpinBox,
        )
        for column in range(widget.species_table.columnCount())
    )
    assert widget.fit_mode_label.text() == "自动拟合"
    assert widget.fit_mode_label.isHidden()

    model = {
        "r_squared": 0.98,
        "species": [{"species": "NO", "coefficient": 1.25, "contribution_percent": 100.0}],
    }
    widget.show_fit_result(model, "PREVIEW")
    assert not widget.confirm_btn.isEnabled()
    assert widget.result_status_label.text() == "实时预览"
    assert widget.result_contribution_label.text() == "贡献：NO 100.0%"
    detail_requests = []
    widget.result_details_requested.connect(lambda: detail_requests.append(True))
    widget.result_details_btn.click()
    assert detail_requests == [True]

    widget.show_fit_result(model, "COMPLETED")
    assert widget.confirm_btn.isEnabled()


def test_candidate_hover_and_compact_fit_summary(widget, qapp):
    widget.resize(430, 740)
    candidates = [
        {
            "id": 1,
            "mz": 142,
            "species": "1-Methylnaphthalene",
            "formula": "C11H10",
            "smiles": "Cc1cccc2ccccc12",
        },
        {
            "id": 2,
            "mz": 142,
            "species": "2-Methylnaphthalene",
            "formula": "C11H10",
            "smiles": "Cc1ccc2ccccc2c1",
        },
    ]
    widget.populate_unified_species_table(142, candidates, [])
    widget.show()
    qapp.processEvents()

    first_identity = widget.species_table.cellWidget(0, 1)
    QtTest.QTest.mouseMove(widget, QtCore.QPoint(5, 5))
    QtTest.QTest.mouseMove(first_identity, QtCore.QPoint(20, 20))
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while not widget.species_hover_card.isVisible() and deadline.elapsed() < 1_000:
        QtTest.QTest.qWait(20)
    assert widget.species_hover_card.isVisible()
    assert widget.species_hover_card.name_label.text() == "1-Methylnaphthalene"
    assert widget.species_hover_card.formula_label.text() == "分子式：C11H10"
    assert not widget.species_hover_card.structure_label.pixmap().isNull()
    QtWidgets.QApplication.sendEvent(
        first_identity,
        QtCore.QEvent(QtCore.QEvent.Type.Leave),
    )
    qapp.processEvents()
    assert not widget.species_hover_card.isVisible()

    QtTest.QTest.mouseClick(
        widget.species_table.cellWidget(1, 1),
        QtCore.Qt.MouseButton.LeftButton,
    )
    qapp.processEvents()
    assert widget.species_table.currentRow() == 1

    widget.show_fit_result(
        {
            "r_squared": 0.99,
            "species": [
                {
                    **candidates[0],
                    "coefficient": 1.2,
                    "contribution_percent": 60.0,
                },
                {
                    **candidates[1],
                    "coefficient": 0.8,
                    "contribution_percent": 40.0,
                },
            ],
        }
    )
    qapp.processEvents()
    assert widget.result_contribution_label.text() == (
        "贡献：1-Methylnaphthalene 60.0% · "
        "2-Methylnaphthalene 40.0%"
    )
    assert "系数 1.2" in widget.result_contribution_label.toolTip()
    assert not hasattr(widget, "fit_result_table")


def test_formula_only_candidate_has_hover_fallback(widget, qapp):
    widget.populate_unified_species_table(
        18,
        [{"id": 1, "mz": 18, "species": "Water", "formula": "H2O", "smiles": ""}],
        [],
    )
    widget.show()
    qapp.processEvents()

    identity = widget.species_table.cellWidget(0, 1)
    identity.hover_entered.emit(identity.mapToGlobal(QtCore.QPoint(20, 20)))
    QtTest.QTest.qWait(275)
    assert widget.species_hover_card.formula_label.text() == "分子式：H2O"
    assert widget.species_hover_card.structure_label.text() == "H2O\n暂无二维结构"
    assert "未提供 SMILES" in widget.species_hover_card.hint_label.text()


def test_candidate_table_controls_stay_inside_row_at_narrow_width(widget, qapp):
    apply_application_theme(qapp)
    widget.resize(420, 760)
    widget.populate_unified_species_table(
        15,
        [
            {
                "id": 1,
                "mz": 15,
                "species": "Methyl radical",
                "formula": "CH3",
                "smiles": "[CH3]",
                "ie": 9.839,
            }
        ],
        [],
    )
    widget.show()
    qapp.processEvents()

    table = widget.species_table
    assert table.horizontalHeaderItem(1).text() == "候选物种"
    assert table.rowHeight(0) == 46
    assert sum(table.columnWidth(index) for index in (0, 2)) == 124
    assert widget.species_hover_card.parent() is widget
    assert widget.species_hover_card.isWindow()
