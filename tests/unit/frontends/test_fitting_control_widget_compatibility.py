"""
向后兼容性测试：验证_force_species属性别名与get_force_species等旧API的正确行为

目标：确保
1. _force_species属性别名正确指向_locked_species
2. get_force_species()等旧API委托到新方法
3. 旧代码的读写行为与新代码一致
"""

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


class TestLockedSpeciesPropertyAlias:
    """测试_force_species属性别名"""

    def test_force_species_is_readonly_alias(self, widget):
        """验证_force_species是_locked_species的只读别名"""
        # 通过新API添加
        widget.lock_candidate('O2')

        # 通过旧属性别名读取
        assert 'O2' in widget._force_species
        assert widget._force_species == widget._locked_species

    def test_force_species_reflects_locked_species_changes(self, widget):
        """验证_force_species实时反映_locked_species的变化"""
        widget._locked_species.append('N2')
        widget._locked_species.append('O2')

        # 旧属性应该看到相同的列表
        assert widget._force_species == ['N2', 'O2']

    def test_force_species_is_readonly(self, widget):
        """验证_force_species属性不能被直接赋值"""
        with pytest.raises(AttributeError):
            widget._force_species = ['O2']


class TestOldAPIBackwardCompatibility:
    """测试旧API的向后兼容性"""

    def test_get_force_species_returns_locked_species(self, widget):
        """验证get_force_species()返回_locked_species的内容"""
        widget.lock_candidate('O2')
        widget.lock_candidate('N2')

        force_list = widget.get_force_species()
        assert force_list == ['O2', 'N2']
        assert force_list == widget.get_locked_species()

    def test_set_force_species_updates_locked_species(self, widget):
        """验证set_force_species()更新_locked_species"""
        widget.set_force_species(['O2', 'N2', 'Ar'])

        assert widget._locked_species == ['O2', 'N2', 'Ar']
        assert widget.get_force_species() == ['O2', 'N2', 'Ar']

    def test_add_force_species_delegates_to_lock_candidate(self, widget):
        """验证add_force_species()委托到lock_candidate()"""
        widget.add_force_species('O2')

        assert 'O2' in widget._locked_species
        assert 'O2' in widget.get_force_species()

    def test_multiple_add_force_species_calls(self, widget):
        """验证多次调用add_force_species()的累积效果"""
        widget.add_force_species('O2')
        widget.add_force_species('N2')
        widget.add_force_species('Ar')

        assert widget.get_force_species() == ['O2', 'N2', 'Ar']
        assert len(widget._locked_species) == 3


class TestStateConsistency:
    """测试新旧API的状态一致性"""

    def test_locked_species_and_force_species_stay_in_sync(self, widget):
        """验证_locked_species和_force_species始终保持一致"""
        # 通过新API添加
        widget.lock_candidate('O2')
        assert widget._locked_species == widget._force_species

        # 通过新API移除
        widget.unlock_candidate('O2')
        assert widget._locked_species == widget._force_species

        # 通过旧API添加
        widget.add_force_species('N2')
        assert widget._locked_species == widget._force_species

    def test_clear_ui_clears_locked_species(self, widget):
        """验证clear_ui()正确清空_locked_species"""
        widget.lock_candidate('O2')
        widget.lock_candidate('N2')

        widget.clear_ui()

        assert widget._locked_species == []
        assert widget.get_force_species() == []


class TestEdgeCases:
    """边界情况测试"""

    def test_add_duplicate_species(self, widget):
        """验证重复添加同一物种的处理"""
        widget.add_force_species('O2')
        widget.add_force_species('O2')  # 重复添加

        # 应该只有一个O2
        assert widget.get_force_species().count('O2') == 1

    def test_unlock_nonexistent_species(self, widget):
        """验证解锁不存在的物种的处理"""
        # 应该不抛出异常
        widget.unlock_candidate('NonExistent')
        assert widget._locked_species == []

    def test_set_force_species_with_empty_list(self, widget):
        """验证设置空列表的处理"""
        widget.add_force_species('O2')
        widget.set_force_species([])

        assert widget._locked_species == []
        assert widget.get_force_species() == []


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

    assert widget.species_table.columnCount() == 5
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
    assert "2 个候选" in widget.candidate_summary_label.text()
    assert "2 个启用" in widget.candidate_summary_label.text()

    widget._set_all_rows_checked(False)

    ready, reason = widget.get_fit_readiness()
    assert ready is False
    assert "至少启用一个" in reason
    assert "0 个启用" in widget.candidate_summary_label.text()


def test_manual_coefficient_mode_and_preview_guard_confirmation(widget):
    widget.populate_unified_species_table(
        30,
        [{"id": 1, "mz": 30, "species": "NO", "energies": [9.0, 10.0], "cross_sections": [0.0, 1.0]}],
        [],
    )
    coefficient = widget.species_table.cellWidget(0, 3)
    coefficient.setValue(1.25)
    assert widget.fit_mode_label.text() == "手动系数"

    model = {
        "r_squared": 0.98,
        "species": [{"species": "NO", "coefficient": 1.25, "contribution_percent": 100.0}],
    }
    widget.show_fit_result(model, "PREVIEW")
    assert not widget.confirm_btn.isEnabled()
    assert "参数预览" in widget.result_status_label.text()

    widget.show_fit_result(model, "COMPLETED")
    assert widget.confirm_btn.isEnabled()


def test_candidate_and_fit_result_share_hover_structure_card(widget, qapp):
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
    QtTest.QTest.qWait(275)
    assert widget.species_hover_card.isVisible()
    assert widget.species_hover_card.name_label.text() == "1-Methylnaphthalene"
    assert widget.species_hover_card.formula_label.text() == "分子式：C11H10"
    assert not widget.species_hover_card.structure_label.pixmap().isNull()
    QtTest.QTest.mouseMove(widget, QtCore.QPoint(5, 5))
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
    assert widget.fit_result_table.item(0, 0).text() == ""
    assert widget.fit_result_table.item(0, 0).data(
        QtCore.Qt.ItemDataRole.UserRole
    ) == "1-Methylnaphthalene\nC11H10"
    result_identity = widget.fit_result_table.cellWidget(1, 0)
    labels = result_identity.findChildren(QtWidgets.QLabel)
    assert [label.text() for label in labels] == [
        "2-Methylnaphthalene",
        "C11H10",
    ]

    QtTest.QTest.mouseMove(widget, QtCore.QPoint(5, 5))
    QtTest.QTest.mouseMove(result_identity, QtCore.QPoint(20, 20))
    QtTest.QTest.qWait(275)
    assert widget.species_hover_card.name_label.text() == "2-Methylnaphthalene"
    assert not widget.species_hover_card.structure_label.pixmap().isNull()


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


def test_candidate_table_editor_stays_inside_row_at_narrow_width(widget, qapp):
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
    coefficient = table.cellWidget(0, 3)
    assert isinstance(coefficient, QtWidgets.QDoubleSpinBox)
    assert table.horizontalHeaderItem(1).text() == "物种"
    assert table.rowHeight(0) == 46
    assert coefficient.height() <= 24
    assert coefficient.y() >= 0
    assert coefficient.y() + coefficient.height() <= table.rowHeight(0)
    assert sum(table.columnWidth(index) for index in (0, 2, 3, 4)) == 264
    assert widget.species_hover_card.parent() is widget
    assert widget.species_hover_card.isWindow()
