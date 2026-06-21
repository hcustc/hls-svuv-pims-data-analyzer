"""
Phase 1 Tests for PIE Dialog: per-m/z configuration preservation and concurrency guards.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets, QtCore
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def pie_dialog(qapp):
    """Create a PIE dialog instance for testing."""
    dialog = PIESpeciesFitDialog(Calibration(a=0.0, b=1.0, c=0.0))
    dialog.show()
    yield dialog
    dialog.deleteLater()


class TestPerM_zConfiguration:
    """Test per-m/z configuration preservation."""

    def test_per_mz_config_structure_initialized(self, pie_dialog):
        """Test that per_mz_config storage is initialized."""
        assert hasattr(pie_dialog, 'per_mz_config')
        assert isinstance(pie_dialog.per_mz_config, dict)
        assert len(pie_dialog.per_mz_config) == 0

    def test_save_and_restore_candidate_selection(self, pie_dialog):
        """Test saving and restoring candidate species selection."""
        # Setup mock data
        pie_dialog.curves = {
            46: {
                'mz': 46,
                'energies': [10.0, 11.0, 12.0],
                'intensities': [1.0, 2.0, 1.5],
                'rows': None
            }
        }

        # Populate candidate table with mock species
        pie_dialog.fitting_control_widget._candidate_data = [
            {'id': 1, 'species': 'NO', 'mz': 46},
            {'id': 2, 'species': 'N2O', 'mz': 46},
            {'id': 3, 'species': 'CO2', 'mz': 46},
        ]
        pie_dialog.candidate_table.setRowCount(3)

        # Set current mz and populate candidate table checkboxes
        pie_dialog.current_mz = 46
        for row in range(3):
            check_widget = QtWidgets.QWidget()
            check_layout = QtWidgets.QHBoxLayout(check_widget)
            check_layout.setContentsMargins(0, 0, 0, 0)
            chk = QtWidgets.QCheckBox()
            chk.setChecked(row < 2)  # Select first two
            check_layout.addWidget(chk)
            pie_dialog.candidate_table.setCellWidget(row, 0, check_widget)

        # Save configuration
        pie_dialog._save_current_mz_config()

        # Verify saved configuration
        assert 46 in pie_dialog.per_mz_config
        config = pie_dialog.per_mz_config[46]
        assert len(config['selected_species']) == 2
        assert config['selected_species'][0]['species'] == 'NO'
        assert config['selected_species'][1]['species'] == 'N2O'

    def test_restore_mz_config_restores_selection(self, pie_dialog):
        """Test that restored config restores candidate selection."""
        # Setup mock species
        pie_dialog.fitting_control_widget._candidate_data = [
            {'id': 1, 'species': 'NO', 'mz': 46},
            {'id': 2, 'species': 'N2O', 'mz': 46},
        ]

        # Create checkboxes in candidate table
        pie_dialog.candidate_table.setRowCount(2)
        for row in range(2):
            check_widget = QtWidgets.QWidget()
            check_layout = QtWidgets.QHBoxLayout(check_widget)
            check_layout.setContentsMargins(0, 0, 0, 0)
            chk = QtWidgets.QCheckBox()
            check_layout.addWidget(chk)
            pie_dialog.candidate_table.setCellWidget(row, 0, check_widget)

        # Manually set per_mz_config
        pie_dialog.per_mz_config[46] = {
            'selected_species': [
                {'id': 1, 'species': 'NO', 'mz': 46},
            ],
            'mode': 'auto',
            'coefficients': {},
            'locked_ids': [],
        }

        # Restore config
        pie_dialog._restore_mz_config(46)

        # Verify restored selection
        check_widget_0 = pie_dialog.candidate_table.cellWidget(0, 0)
        check_0 = check_widget_0.findChild(QtWidgets.QCheckBox)
        check_widget_1 = pie_dialog.candidate_table.cellWidget(1, 0)
        check_1 = check_widget_1.findChild(QtWidgets.QCheckBox)

        assert check_0.isChecked()
        assert not check_1.isChecked()

    def test_restore_mz_config_initializes_default_if_no_history(self, pie_dialog):
        """Test that restore initializes default config if no history."""
        pie_dialog.fitting_control_widget._candidate_data = [
            {'id': 1, 'species': 'NO', 'mz': 46},
        ]

        pie_dialog.candidate_table.setRowCount(1)
        check_widget = QtWidgets.QWidget()
        check_layout = QtWidgets.QHBoxLayout(check_widget)
        check_layout.setContentsMargins(0, 0, 0, 0)
        chk = QtWidgets.QCheckBox()
        check_layout.addWidget(chk)
        pie_dialog.candidate_table.setCellWidget(0, 0, check_widget)

        # No config for m/z 46, should initialize default
        pie_dialog._restore_mz_config(46)

        # Default should select all (per user requirement)
        check_widget_0 = pie_dialog.candidate_table.cellWidget(0, 0)
        check_0 = check_widget_0.findChild(QtWidgets.QCheckBox)
        assert check_0.isChecked()


class TestConcurrencyGuards:
    """Test concurrency prevention with _busy flag."""

    def test_fit_current_curve_respects_busy_flag(self, pie_dialog):
        """Test that fit_current_curve returns early if _busy is True."""
        pie_dialog._busy = True
        pie_dialog.current_mz = 46
        pie_dialog.curves = {46: {'mz': 46, 'energies': [10.0], 'intensities': [1.0]}}

        # Call fit_current_curve with _busy=True should return immediately
        result = pie_dialog.fit_current_curve()

        assert result is None  # Early return
        assert pie_dialog._busy is True

    def test_fit_all_curves_respects_busy_flag(self, pie_dialog):
        """Test that fit_all_curves returns early if _busy is True."""
        pie_dialog._busy = True
        pie_dialog.curves = {46: {'mz': 46, 'energies': [10.0], 'intensities': [1.0]}}

        # Call fit_all_curves with _busy=True should return immediately
        result = pie_dialog.fit_all_curves()

        assert result is None  # Early return
        assert pie_dialog._busy is True

    def test_refit_selected_curves_respects_busy_flag(self, pie_dialog):
        """Test that refit_selected_curves returns early if _busy is True."""
        pie_dialog._busy = True

        # Even if items are selected, should return early
        result = pie_dialog.refit_selected_curves()

        assert result is None  # Early return
        assert pie_dialog._busy is True


class TestUIChanges:
    """Test UI structure changes from Phase 1."""

    def test_candidate_apply_button_removed(self, pie_dialog):
        """Test that candidate_apply_btn has been removed."""
        # The button should not exist in dialog attributes
        assert not hasattr(pie_dialog, 'candidate_apply_btn')

    def test_refit_selected_button_renamed(self, pie_dialog):
        """Test that refit_selected_button has correct new name."""
        assert pie_dialog.refit_selected_button.text() == "拟合选中曲线"

    def test_more_actions_menu_exists(self, pie_dialog):
        """Test that more_actions menu button exists."""
        assert hasattr(pie_dialog, 'more_actions_btn')
        assert hasattr(pie_dialog, 'more_actions_menu')
        assert pie_dialog.more_actions_btn.text() == "更多操作"

    def test_more_actions_menu_has_correct_items(self, pie_dialog):
        """Test that more_actions menu contains expected items."""
        menu = pie_dialog.more_actions_menu
        actions = menu.actions()

        # Should have: 清除拟合, separator, 穷举优选, separator, 重新加载...
        action_texts = [a.text() for a in actions if not a.isSeparator()]

        assert "清除拟合" in action_texts
        assert "穷举优选" in action_texts
        assert "重新加载PICS截面数据库" in action_texts


class TestFitSelectedCurvesWithPerM_zConfig:
    """Test fit_selected_curves_sync using per-m/z configurations."""

    def test_fit_selected_curves_uses_per_mz_config(self, pie_dialog):
        """Test that fit_selected_curves_sync uses per-m/z config when available."""
        # Setup mock database
        pie_dialog.database = [
            {
                'id': 1,
                'species': 'NO',
                'mz': 46,
                'ionization_energy': 9.26,
                'energies': np.array([10.0, 11.0, 12.0]),
                'cross_sections': np.array([0.1, 0.2, 0.15]),
            },
            {
                'id': 2,
                'species': 'N2O',
                'mz': 46,
                'ionization_energy': 12.89,
                'energies': np.array([10.0, 11.0, 12.0]),
                'cross_sections': np.array([0.05, 0.1, 0.08]),
            },
        ]

        # Setup curves
        pie_dialog.curves = {
            46: {
                'mz': 46,
                'energies': np.array([10.0, 11.0, 12.0]),
                'intensities': np.array([1.0, 2.0, 1.5]),
            }
        }

        # Setup per-m/z config for m/z 46 with NO only
        pie_dialog.per_mz_config[46] = {
            'selected_species': [
                {
                    'id': 1,
                    'species': 'NO',
                    'mz': 46,
                    'ionization_energy': 9.26,
                    'energies': np.array([10.0, 11.0, 12.0]),
                    'cross_sections': np.array([0.1, 0.2, 0.15]),
                },
            ],
            'mode': 'auto',
            'coefficients': {},
            'locked_ids': [],
        }

        # Call fit_selected_curves_sync
        results = pie_dialog.fit_selected_curves_sync([46])

        # Should have result for m/z 46
        assert 46 in results
        assert results[46]['success'] is True


class TestMenuIntegration:
    """Test integration of menu and buttons."""

    def test_update_action_state_handles_menu_button(self, pie_dialog):
        """Test that _update_action_state updates menu button state."""
        pie_dialog.curves = {46: {'mz': 46, 'energies': [10.0], 'intensities': [1.0]}}
        pie_dialog._busy = False

        pie_dialog._update_action_state()

        # Menu button should be enabled when not busy
        assert pie_dialog.more_actions_btn.isEnabled()

    def test_update_action_state_disables_menu_when_busy(self, pie_dialog):
        """Test that menu button is disabled when busy."""
        pie_dialog.curves = {46: {'mz': 46, 'energies': [10.0], 'intensities': [1.0]}}
        pie_dialog._busy = True

        pie_dialog._update_action_state()

        # Menu button should be disabled when busy
        assert not pie_dialog.more_actions_btn.isEnabled()


class TestConfigurationOnMZSwitch:
    """Test configuration save/restore on m/z switching."""

    def test_save_config_on_mz_switch(self, pie_dialog):
        """Test that config is saved before switching m/z."""
        # Setup curves and candidates
        pie_dialog.curves = {
            46: {'mz': 46, 'energies': [10.0], 'intensities': [1.0], 'rows': None},
            47: {'mz': 47, 'energies': [10.0], 'intensities': [1.0], 'rows': None},
        }

        pie_dialog.fitting_control_widget._candidate_data = [
            {'id': 1, 'species': 'NO', 'mz': 46},
        ]

        # Manually set current_mz and candidate table
        pie_dialog.current_mz = 46
        pie_dialog.candidate_table.setRowCount(1)
        check_widget = QtWidgets.QWidget()
        check_layout = QtWidgets.QHBoxLayout(check_widget)
        check_layout.setContentsMargins(0, 0, 0, 0)
        chk = QtWidgets.QCheckBox()
        chk.setChecked(True)
        check_layout.addWidget(chk)
        pie_dialog.candidate_table.setCellWidget(0, 0, check_widget)

        # Simulate switching to m/z 47 (this would happen in on_mz_selected)
        pie_dialog._save_current_mz_config()

        # Config should be saved for m/z 46
        assert 46 in pie_dialog.per_mz_config
        assert len(pie_dialog.per_mz_config[46]['selected_species']) == 1
