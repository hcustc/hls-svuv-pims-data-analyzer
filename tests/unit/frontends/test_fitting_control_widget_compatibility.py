"""
向后兼容性测试：验证_force_species属性别名与get_force_species等旧API的正确行为

目标：确保
1. _force_species属性别名正确指向_locked_species
2. get_force_species()等旧API委托到新方法
3. 旧代码的读写行为与新代码一致
"""

import pytest
from PyQt6 import QtWidgets
from bl03u_masstool.frontends.pyqt_app.pie.fitting_control_widget import FittingControlWidget


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
