from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.frontends.pyqt_app.mole_fraction.dialog import MoleFractionDialog


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_manual_peak_ranges_integrate_fixed_bounds(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        data = np.zeros(24)
        data[5:8] = [2.0, 10.0, 3.0]
        data[12:15] = [1.0, 4.0, 2.0]

        peaks = dialog._detect_and_integrate_peaks(data, {7: (5, 7), 14: (12, 14)})

        assert [peak["mz_rounded"] for peak in peaks] == [7, 14]
        assert peaks[0]["index"] == 6
        assert peaks[0]["left_idx"] == 5
        assert peaks[0]["right_idx"] == 7
        assert peaks[0]["integral"] == pytest.approx(15.0)
    finally:
        dialog.deleteLater()


def test_data_loading_uses_switchable_data_views(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        # Verify parallel card layout with expandable tables
        assert hasattr(dialog, "ts_table_group")
        assert hasattr(dialog, "pie_table_group")
        assert hasattr(dialog, "ts_data_table")
        assert hasattr(dialog, "pie_species_table")
        # Tables should be in separate GroupBoxes (different parents)
        assert dialog.ts_data_table.parent() != dialog.pie_species_table.parent()
        # GroupBoxes should be checkable (expandable) and initially unchecked
        assert dialog.ts_table_group.isCheckable()
        assert dialog.pie_table_group.isCheckable()
        assert not dialog.ts_table_group.isChecked()
        assert not dialog.pie_table_group.isChecked()
    finally:
        dialog.deleteLater()


def test_auto_mf_table_marks_parent_references_after_products(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.spin_parent_mz.setValue(112)
        dialog.spin_parent_energy.setValue(8.0)
        dialog.parent_mf_by_energy = {8.0: {650.0: 0.02}, 9.5: {650.0: 0.018}}
        dialog.parent_config_by_energy = {
            8.0: {"mz": 112, "species_name": "1,1-dimethyl cyclohexane"},
            9.5: {"mz": 112, "species_name": "1,1-dimethyl cyclohexane"},
        }
        dialog.all_species_mf = {
            (112, "1,1-dimethyl cyclohexane", 8.0): {650.0: 0.02},
            (15, "Methyl radical", 9.5): {650.0: 0.001},
            (112, "1,1-dimethyl cyclohexane", 9.5): {650.0: 0.018},
        }

        dialog._update_auto_mf_table()

        assert dialog.combo_auto_plot_scope.currentData() == "products"
        assert dialog.auto_mf_table.item(0, 0).text() == "产物"
        assert dialog.auto_mf_table.item(0, 1).text() == "15"
        assert dialog.auto_mf_table.item(1, 0).text() == "母体参考"
        assert dialog._auto_result_counts() == (2, 1)
        assert dialog._auto_plot_keys_for_scope() == [(15, "Methyl radical", 9.5)]
    finally:
        dialog.deleteLater()


def test_results_summary_table_uses_readable_long_format(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.all_species_mf = {
            (15 + idx, f"Species {idx}", 8.0 + idx): {650.0: 0.001 * idx, 700.0: 0.002 * idx}
            for idx in range(1, 12)
        }

        dialog._update_results_table()

        assert dialog.results_table.columnCount() == 6
        assert dialog.results_table.rowCount() == 22
        assert [
            dialog.results_table.horizontalHeaderItem(col).text()
            for col in range(dialog.results_table.columnCount())
        ] == ["类型", "m/z", "物种/结果", "光子能量(eV)", "温度(°C)", "摩尔分数"]
        assert dialog.results_table.item(0, 2).text().startswith("Species")
        assert "11 条结果" in dialog.lbl_results_status.text()
    finally:
        dialog.deleteLater()


def test_recompute_peak_info_refreshes_loaded_scan_with_manual_ranges(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        first = np.zeros(24)
        first[5:8] = [1.0, 6.0, 2.0]
        second = np.zeros(24)
        second[5:8] = [2.0, 8.0, 3.0]
        dialog.temperature_scan_data = {
            10.0: {
                100: {"avg_data": first.tolist(), "avg_io": 100.0, "filenames": "100.txt", "repeat_count": 1},
                200: {"avg_data": second.tolist(), "avg_io": 100.0, "filenames": "200.txt", "repeat_count": 1},
            }
        }
        dialog.available_energies = [10.0]
        dialog.pie_species_data = [{"mz": 7, "species": "ManualSpecies"}]
        dialog.peak_ranges = {7: (5, 7)}

        peak_count = dialog._recompute_peak_info()

        assert peak_count == 2
        assert dialog.ts_data_table.rowCount() == 2
        assert "ManualSpecies" in dialog.ts_data_table.item(0, 5).text()
        for info in dialog.temperature_scan_data[10.0].values():
            assert info["peaks_info"][0]["left_idx"] == 5
            assert info["peaks_info"][0]["right_idx"] == 7
            assert info["matched_species"] == ["ManualSpecies"]
    finally:
        dialog.deleteLater()
