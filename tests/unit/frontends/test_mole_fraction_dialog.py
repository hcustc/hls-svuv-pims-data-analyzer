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
from bl03u_masstool.core.project_settings import ProjectSettings
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
        assert hasattr(dialog, "btn_load_project_ts_folder")
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


def _write_temperature_scan_file(path, *, temperature=650, io=100.0, peak_index=20):
    data = [0.0] * 140
    data[peak_index - 1:peak_index + 2] = [4.0, 20.0, 5.0]
    header = [
        f"Temperature: {temperature} C",
        f"IO: {io}",
        "#",
        "#",
        "#",
        "#",
        "#",
        "#",
        "#",
        "#",
    ]
    path.write_text("\n".join(header + [str(value) for value in data]), encoding="utf-8")


def test_project_temperature_scan_folder_loads_energy_subfolders(qapp, tmp_path, monkeypatch):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)
    try:
        root = tmp_path / "temperature_root"
        low = root / "8.0eV"
        high = root / "9.5 eV"
        low.mkdir(parents=True)
        high.mkdir(parents=True)
        _write_temperature_scan_file(low / "sample_650C_a.txt", temperature=650, io=50.0)
        _write_temperature_scan_file(low / "sample_700C_a.txt", temperature=700, io=60.0)
        _write_temperature_scan_file(high / "sample_650C_a.txt", temperature=650, io=70.0)

        dialog.set_project_settings(ProjectSettings(temperature_scan_folder=str(root)))

        assert dialog.btn_load_project_ts_folder.isEnabled()

        dialog._load_project_temperature_scan_folder()

        assert dialog.available_energies == [8.0, 9.5]
        assert sorted(dialog.temperature_scan_data[8.0]) == [650, 700]
        assert sorted(dialog.temperature_scan_data[9.5]) == [650]
        assert dialog.combo_energy_select.count() == 3
        assert dialog.lbl_energy_count.text() == "共 2 个能量点"
        assert "项目原始目录" in dialog.lbl_ts_folder.text()
    finally:
        dialog.deleteLater()


def test_low_energy_reference_tab_follows_parent_mole_fraction(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        tab_texts = [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())]

        assert tab_texts[1] == "2. 母体摩尔分数"
        assert tab_texts[2] == "3. 低能参考物种"
        assert all("多能量母体配置" not in text for text in tab_texts)
    finally:
        dialog.deleteLater()


def test_apply_low_energy_reference_keeps_only_overrides(qapp, monkeypatch):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)
    try:
        dialog.spin_parent_mz.setValue(112)
        dialog.pie_species_data = [
            {"mz": 15, "species": "Methyl radical", "ie": 9.84},
            {"mz": 112, "species": "Parent", "ie": 9.0},
        ]
        dialog.available_energies = [8.0, 9.5]
        dialog._refresh_energy_parent_table()

        mz_combo = dialog.energy_parent_table.cellWidget(0, 1)
        species_combo = dialog.energy_parent_table.cellWidget(0, 2)
        assert mz_combo is not None
        assert species_combo is not None

        dialog._set_combo_current_data(mz_combo, 15)
        dialog._on_mz_changed_for_parent_config(0, 8.0)
        species_combo = dialog.energy_parent_table.cellWidget(0, 2)
        assert species_combo is not None
        dialog._set_combo_current_data(species_combo, "Methyl radical")
        dialog._on_species_changed_for_parent_config(8.0, mz_combo, species_combo)

        dialog._apply_energy_parent_config()

        assert dialog.energy_parent_config == {
            8.0: {"mz": 15, "species_name": "Methyl radical"}
        }
    finally:
        dialog.deleteLater()


def test_low_energy_reference_cache_is_derived_from_main_parent(qapp, monkeypatch):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.spin_parent_mz.setValue(112)
        dialog.spin_parent_energy.setValue(9.5)
        dialog.spin_parent_t0.setValue(650)
        dialog.parent_mf_results = {650.0: 0.02}
        dialog.expansion_coefficients = {650.0: 1.0}
        dialog.available_energies = [8.0, 9.5]
        dialog.temperature_scan_data = {
            8.0: {
                650.0: {
                    "precomputed_signals": {15: 10.0},
                    "peaks_info": [],
                    "avg_data": [],
                    "avg_io": 1.0,
                    "filenames": "8eV.txt",
                    "repeat_count": 1,
                }
            },
            9.5: {
                650.0: {
                    "precomputed_signals": {112: 100.0},
                    "peaks_info": [],
                    "avg_data": [],
                    "avg_io": 1.0,
                    "filenames": "9.5eV.txt",
                    "repeat_count": 1,
                }
            },
        }
        dialog.pie_species_data = [
            {"mz": 15, "species": "Methyl radical", "ie": 9.84},
            {"mz": 112, "species": "Parent", "ie": 9.0},
        ]
        dialog.energy_parent_config = {
            8.0: {"mz": 15, "species_name": "Methyl radical"}
        }
        captured = {}

        def fake_product_mf(
            mz,
            species,
            ref_mz,
            ref_mw,
            ref_energy,
            ref_signal_data,
            ref_mf_at_tm,
            signal_data,
            calc_energy=None,
            ref_species_name=None,
        ):
            captured.update(
                {
                    "mz": mz,
                    "species": species,
                    "ref_mz": ref_mz,
                    "ref_energy": ref_energy,
                    "ref_signal_data": ref_signal_data,
                    "ref_mf_at_tm": ref_mf_at_tm,
                    "signal_data": signal_data,
                    "calc_energy": calc_energy,
                }
            )
            return {650.0: 0.003}

        monkeypatch.setattr(dialog, "_calc_product_mf_auto", fake_product_mf)

        dialog._recalculate_parent_mf_by_energy()

        assert dialog.parent_mf_by_energy == {8.0: {650.0: 0.003}}
        assert captured["mz"] == 15
        assert captured["ref_mz"] == 112
        assert captured["ref_energy"] == pytest.approx(9.5)
        assert captured["ref_signal_data"] == {650.0: 100.0}
        assert captured["ref_mf_at_tm"] == pytest.approx(0.02)
        assert captured["signal_data"] == {650.0: 10.0}
        assert captured["calc_energy"] == pytest.approx(8.0)
    finally:
        dialog.deleteLater()


def test_signal_extraction_prefers_raw_scan_over_precomputed_result(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.available_energies = [10.0]
        dialog.temperature_scan_data = {
            10.0: {
                650.0: {
                    "precomputed_signals": {84: 999.0},
                    "peaks_info": [{"mz_rounded": 84, "left_idx": 2, "right_idx": 4}],
                    "avg_data": [0.0, 0.0, 2.0, 4.0, 6.0, 0.0],
                    "avg_io": 2.0,
                    "filenames": "raw.txt",
                    "repeat_count": 1,
                }
            }
        }

        assert dialog._get_signal_from_scan_data(84, 10.0) == {650.0: pytest.approx(6.0)}
    finally:
        dialog.deleteLater()


def test_reference_parent_for_species_prefers_closest_ionized_low_energy_reference(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.spin_parent_mz.setValue(112)
        dialog.spin_parent_energy.setValue(10.0)
        dialog.spin_parent_t0.setValue(650)
        dialog.parent_mf_results = {650.0: 0.02}
        dialog.parent_mf_by_energy = {8.0: {650.0: 0.003}}
        dialog.parent_signal_by_energy = {8.0: {650.0: 10.0}}
        dialog.parent_config_by_energy = {8.0: {"mz": 15, "species_name": "LowRef"}}
        dialog.database = [
            {"mz": 112, "species": "MainParent", "ie": 9.5},
            {"mz": 15, "species": "LowRef", "ie": 8.0},
        ]
        dialog.available_energies = [8.0, 9.0, 10.0]
        dialog.temperature_scan_data = {
            9.0: {
                650.0: {
                    "precomputed_signals": {15: 12.0, 112: 0.0},
                    "peaks_info": [],
                    "avg_data": [],
                    "avg_io": 1.0,
                    "filenames": "9eV.csv",
                    "repeat_count": 1,
                }
            }
        }

        ref = dialog._reference_parent_for_species(9.0, 8.6)

        assert ref is not None
        ref_mz, _, ref_energy, ref_signal, ref_mf_at_tm, ref_species_name = ref
        assert ref_mz == 15
        assert ref_energy == pytest.approx(9.0)
        assert ref_signal == {650.0: pytest.approx(12.0)}
        assert ref_mf_at_tm == pytest.approx(0.003)
        assert ref_species_name == "LowRef"
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
