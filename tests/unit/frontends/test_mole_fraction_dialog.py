from __future__ import annotations

from copy import deepcopy
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
from bl03u_masstool.core.pie_analysis import save_species_database_sqlite
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


def test_manual_peak_ranges_keep_same_nominal_mz_peaks_separate(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        data = np.zeros(24)
        data[5] = 2.0
        data[12] = 100.0

        peaks = dialog._detect_and_integrate_peaks(
            data,
            {
                227.587: (5, 5),
                228.023: (12, 12),
            },
        )

        assert [peak["mz_rounded"] for peak in peaks] == [228, 228]
        assert [peak["curve_key"] for peak in peaks] == pytest.approx(
            [227.587, 228.023]
        )
        assert not any(peak["overlapped"] for peak in peaks)
    finally:
        dialog.deleteLater()


def test_temperature_and_pie_import_preserve_precise_mz_collision(
    qapp,
    tmp_path,
):
    temperature_result = tmp_path / "temperature_collision.csv"
    temperature_result.write_text(
        "\n".join(
            [
                "temperature,mz,mz_rounded,temperature_curve_key,"
                "temperature_peak_track,area,photon_energy,file",
                "650,227.587,228,227.587,0,2,10,a.txt",
                "650,228.023,228,228.023,1,100,10,a.txt",
                "750,227.587,228,227.587,0,3,10,b.txt",
                "750,228.023,228,228.023,1,120,10,b.txt",
            ]
        ),
        encoding="utf-8",
    )
    pie_result = tmp_path / "pie_collision.csv"
    pie_result.write_text(
        "\n".join(
            [
                "精确m/z,质量数,物种名称,电离能(eV),贡献比例(%),R²",
                "227.587,228,Noise,9.0,1.0,0.1",
                "228.023,228,Product,10.0,99.0,0.999",
            ]
        ),
        encoding="utf-8",
    )

    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog._load_temperature_result_file(
            temperature_result,
            show_message=False,
            register_artifact=False,
        )
        signals = dialog.temperature_scan_data[10.0][650.0][
            "precomputed_signals"
        ]

        assert signals == pytest.approx({227.587: 2.0, 228.023: 100.0})
        assert dialog._get_signal_from_scan_data(227.587, 10.0) == pytest.approx(
            {650.0: 2.0, 750.0: 3.0}
        )
        assert dialog._get_signal_from_scan_data(228.023, 10.0) == pytest.approx(
            {650.0: 100.0, 750.0: 120.0}
        )
        assert dialog._get_signal_from_scan_data(228, 10.0) == {}

        dialog._load_pie_results(
            pie_result,
            show_message=False,
            register_artifact=False,
        )

        assert {
            record["curve_key"]: record["species"]
            for record in dialog.pie_species_data
        } == {227.587: "Noise", 228.023: "Product"}
        species_names = {
            name for name, _ie in dialog._species_options_for_mz(228.023)
        }
        assert "Product" in species_names
        assert "Noise" not in species_names
    finally:
        dialog.deleteLater()


def test_mass_discrimination_exponent_falls_back_to_legacy_settings(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.project_settings = None
        dialog.settings.mass_disc_exponent = 0.42

        assert dialog._mass_disc_exponent == pytest.approx(0.42)
    finally:
        dialog.deleteLater()


def test_project_switch_replaces_mole_fraction_runtime_parameters(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        first = ProjectSettings(
            project_name="First",
            cal_a=7.1e-7,
            cal_b=2.3e-4,
            cal_c=0.45,
            light_source="beam_current",
            mf_mass_disc_exponent=0.61,
            mf_reference_temperature=650,
            expansion_factors={650.0: 1.2},
        )
        dialog.set_project_settings(first)

        assert dialog.calibration == first.to_calibration()
        assert dialog.normalization_settings.light_source == "beam_current"
        assert dialog.settings.mass_disc_exponent == pytest.approx(0.61)
        assert dialog.spin_parent_t0.value() == 650
        assert dialog.expansion_coefficients == {650.0: 1.2}

        second = ProjectSettings(project_name="Second", expansion_factors={})
        dialog.set_project_settings(second)

        assert dialog.calibration == second.to_calibration()
        assert dialog.spin_parent_t0.value() == 550
        assert dialog.expansion_coefficients == {}
    finally:
        dialog.deleteLater()


def _mole_fraction_project(tmp_path, name, *, signals=(100.0, 50.0), initial_mf=0.1):
    project_dir = tmp_path / name
    project_dir.mkdir()
    temperature_result = project_dir / "temperature.csv"
    temperature_result.write_text(
        "temperature,mz,area,photon_energy,file\n"
        f"650,28,{signals[0]},10,a.txt\n750,28,{signals[1]},10,b.txt\n",
        encoding="utf-8",
    )
    pie_result = project_dir / "pie.csv"
    pie_result.write_text(
        "质量数,物种名称,电离能(eV),贡献比例(%),R²\n"
        f"28,Parent {name},9.0,100,1.0\n",
        encoding="utf-8",
    )
    return ProjectSettings(
        project_name=name,
        output_dir=str(project_dir),
        temperature_scan_result_file=str(temperature_result),
        pie_identification_result_file=str(pie_result),
        mf_parent_mz=28,
        mf_reference_temperature=650,
        mf_photon_energy=10,
        mf_parent_initial_mf=initial_mf,
        expansion_factors={650.0: 1.0, 750.0: 1.0},
    )


@pytest.mark.parametrize("new_source", ["empty", "missing", "invalid"])
def test_project_switch_cannot_calculate_or_export_previous_project_data(
    qapp, tmp_path, monkeypatch, new_source,
):
    warnings = []
    save_requests = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        lambda *args: (save_requests.append(args) or "", ""),
    )
    first = _mole_fraction_project(tmp_path, "A")
    second = ProjectSettings(output_dir=str(tmp_path / "B"), mf_parent_mz=28)
    if new_source != "empty":
        second.temperature_scan_result_file = str(tmp_path / "unavailable.csv")
        second.pie_identification_result_file = str(tmp_path / "unavailable-pie.csv")
    if new_source == "invalid":
        (tmp_path / "unavailable.csv").write_text("not,a,spectrum\n", encoding="utf-8")
        (tmp_path / "unavailable-pie.csv").write_text("not,a,fit\n", encoding="utf-8")

    dialog = MoleFractionDialog(Calibration(), None)
    try:
        dialog.set_project_settings(first)
        dialog.btn_confirm_parent_mz.click()
        dialog.btn_calc_parent.click()
        assert dialog.parent_result_table.rowCount() == 2
        assert dialog.results_table.rowCount() >= 2
        assert dialog.mf_plot_widget.figure.axes[0].lines
        auto_calculate = next(
            button for button in dialog.findChildren(QtWidgets.QPushButton)
            if button.text() == "开始计算"
        )
        auto_calculate.click()
        dialog.combo_auto_plot_scope.setCurrentIndex(
            dialog.combo_auto_plot_scope.findData("all")
        )
        assert dialog.auto_mf_table.rowCount() >= 1
        assert dialog.auto_mf_plot_widget.figure.axes[0].lines

        dialog.set_project_settings(second)

        assert dialog.ts_data_table.rowCount() == 0
        assert dialog.pie_species_table.rowCount() == 0
        assert dialog.combo_energy_select.count() == 1
        assert dialog.energy_parent_table.rowCount() == 0
        assert dialog.parent_result_table.rowCount() == 0
        assert dialog.results_table.rowCount() == 0
        assert dialog.mf_series_list.count() == 0
        assert not dialog.mf_plot_widget.figure.axes[0].lines
        assert dialog.auto_mf_table.rowCount() == 0
        assert not dialog.auto_mf_plot_widget.figure.axes[0].lines
        assert dialog.lbl_auto_status.text() == "未计算"
        dialog.btn_confirm_parent_mz.click()
        assert not dialog.btn_calc_parent.isEnabled()
        dialog.btn_calc_parent.click()
        export = next(button for button in dialog.findChildren(QtWidgets.QPushButton)
                      if button.text() == "导出结果 (Excel/CSV)")
        export.click()
        assert "没有计算结果可导出" in warnings
        assert save_requests == []
    finally:
        dialog.deleteLater()


def test_leaving_project_does_not_restore_its_registered_inputs(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    project = _mole_fraction_project(tmp_path, "A")
    dialog = MoleFractionDialog(Calibration(), None)
    try:
        dialog.set_project_settings(project)
        dialog.btn_confirm_parent_mz.click()
        dialog.btn_calc_parent.click()
        assert dialog.parent_result_table.rowCount() == 2

        # The caller can still hold the former settings when it removes scope.
        dialog.set_project_settings(project, activate_project_scope=False)

        assert dialog.ts_data_table.rowCount() == 0
        assert dialog.pie_species_table.rowCount() == 0
        assert dialog.parent_result_table.rowCount() == 0
        assert dialog.results_table.rowCount() == 0
        assert not dialog.btn_load_project_ts_result.isEnabled()
        assert not dialog.btn_load_project_pie_result.isEnabled()
        assert not dialog.btn_calc_parent.isEnabled()
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("shared_inputs", [False, True])
def test_project_switch_reloads_inputs_and_computes_only_new_project_results(
    qapp, tmp_path, monkeypatch, shared_inputs,
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    first = _mole_fraction_project(tmp_path, "A")
    second = _mole_fraction_project(tmp_path, "B", signals=(20.0, 40.0), initial_mf=0.2)
    if shared_inputs:
        second.temperature_scan_result_file = first.temperature_scan_result_file
        second.pie_identification_result_file = first.pie_identification_result_file
    dialog = MoleFractionDialog(Calibration(), None)
    try:
        dialog.set_project_settings(first)
        dialog.btn_confirm_parent_mz.click()
        dialog.btn_calc_parent.click()
        assert dialog.parent_result_table.item(1, 3).text() == "0.050000"

        dialog.set_project_settings(second)

        assert dialog.ts_data_table.rowCount() == 2
        assert dialog.pie_species_table.rowCount() == 1
        assert dialog.parent_result_table.rowCount() == 0
        assert dialog.results_table.rowCount() == 0
        dialog.btn_confirm_parent_mz.click()
        dialog.btn_calc_parent.click()
        values = [float(dialog.parent_result_table.item(row, 3).text()) for row in range(2)]
        assert values == pytest.approx([0.2, 0.1] if shared_inputs else [0.2, 0.4])
    finally:
        dialog.deleteLater()


def test_same_project_sync_preserves_loaded_inputs_and_results(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    project = _mole_fraction_project(tmp_path, "A")
    dialog = MoleFractionDialog(Calibration(), None)
    try:
        dialog.set_project_settings(project)
        dialog.btn_confirm_parent_mz.click()
        dialog.btn_calc_parent.click()
        updated = deepcopy(project)
        updated.project_name = "Renamed"
        updated.pie_energy_decimals = 3
        updated.output_dir = str(tmp_path / "alias" / ".." / "A")

        dialog.set_project_settings(updated)

        assert dialog.ts_data_table.rowCount() == 2
        assert dialog.pie_species_table.rowCount() == 1
        assert dialog.parent_result_table.rowCount() == 2
        assert dialog.parent_result_table.item(1, 3).text() == "0.050000"
        assert dialog.results_table.rowCount() >= 2
        assert dialog.mf_plot_widget.figure.axes[0].lines
    finally:
        dialog.deleteLater()


def test_project_settings_loads_project_pics_database(qapp, tmp_path):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        db_path = tmp_path / "project_species.sqlite"
        save_species_database_sqlite(
            [
                {
                    "mz": 15,
                    "species": "ProjectSpecies",
                    "ie": 9.8,
                    "smiles": "",
                    "energies": np.array([10.0]),
                    "cross_sections": np.array([1.5]),
                }
            ],
            db_path,
        )

        dialog.set_project_settings(ProjectSettings(pics_database_path=str(db_path), mf_parent_mz=15))

        assert dialog._loaded_database_path == str(db_path)
        assert any(record["species"] == "ProjectSpecies" for record in dialog.database)
        assert 15 in dialog.mz_index
    finally:
        dialog.deleteLater()


def test_switching_or_closing_project_drops_previous_pics_records(qapp, tmp_path):
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    for path, name, mz in (
        (first_path, "OnlyInFirstProject", 151),
        (second_path, "OnlyInSecondProject", 152),
    ):
        save_species_database_sqlite(
            [
                {
                    "mz": mz,
                    "species": name,
                    "ie": 9.8,
                    "smiles": "",
                    "energies": np.array([10.0]),
                    "cross_sections": np.array([1.5]),
                }
            ],
            path,
        )

    dialog = MoleFractionDialog(Calibration(), None)
    try:
        dialog.set_project_settings(ProjectSettings(pics_database_path=str(first_path)))
        assert any(item["species"] == "OnlyInFirstProject" for item in dialog.database)

        dialog.set_project_settings(ProjectSettings(pics_database_path=str(second_path)))
        names = {item["species"] for item in dialog.database}
        assert "OnlyInFirstProject" not in names
        assert "OnlyInSecondProject" in names

        dialog.set_project_settings(ProjectSettings())
        names = {item["species"] for item in dialog.database}
        assert "OnlyInFirstProject" not in names
        assert "OnlyInSecondProject" not in names
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
        assert dialog.available_energies == []

        dialog._load_project_temperature_scan_folder()

        assert dialog.available_energies == [8.0, 9.5]
        assert "项目原始目录" in dialog.lbl_ts_folder.text()

        dialog._load_project_temperature_scan_folder()

        assert dialog.available_energies == [8.0, 9.5]
        assert sorted(dialog.temperature_scan_data[8.0]) == [650, 700]
        assert sorted(dialog.temperature_scan_data[9.5]) == [650]
        assert dialog.combo_energy_select.count() == 3
        assert dialog.lbl_energy_count.text() == "共 2 个能量点"
        assert "项目原始目录" in dialog.lbl_ts_folder.text()
    finally:
        dialog.deleteLater()


def test_project_settings_auto_restores_mole_fraction_inputs(qapp, tmp_path):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        temperature_result = tmp_path / "temperature_result.csv"
        temperature_result.write_text(
            "\n".join(
                [
                    "temperature,mz,area,photon_energy,file,io,left_bound,right_bound",
                    "650,15,10.0,10.0,sample_650.txt,50,19,21",
                    "750,15,20.0,10.0,sample_750.txt,60,19,21",
                ]
            ),
            encoding="utf-8",
        )
        pie_result = tmp_path / "pie_result.csv"
        pie_result.write_text(
            "\n".join(
                [
                    "质量数,物种名称,电离能(eV),贡献比例(%),R²",
                    "15,Methyl radical,9.84,92.5,0.998",
                ]
            ),
            encoding="utf-8",
        )

        dialog.set_project_settings(
            ProjectSettings(
                temperature_scan_result_file=str(temperature_result),
                pie_identification_result_file=str(pie_result),
                mf_parent_mz=15,
                mf_photon_energy=10.0,
                mf_reference_temperature=650,
            )
        )

        assert dialog.available_energies == [10.0]
        assert sorted(dialog.temperature_scan_data[10.0]) == [650.0, 750.0]
        assert dialog.temperature_scan_data[10.0][650.0]["precomputed_signals"][15] == pytest.approx(10.0)
        assert dialog.pie_species_data == [
            {
                "mz": 15,
                "species": "Methyl radical",
                "ie": pytest.approx(9.84),
                "contribution": pytest.approx(92.5),
                "r_squared": pytest.approx(0.998),
            }
        ]
        assert dialog.pie_species_table.rowCount() == 1
        assert "已从项目自动恢复" in dialog.status_label.text()
        assert "温度扫描结果 1 个能量" in dialog.status_label.text()
        assert "PIE结果 1 条" in dialog.status_label.text()
    finally:
        dialog.deleteLater()


def test_stale_legacy_parent_mz_128_is_cleared_when_data_lacks_128(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.spin_parent_mz.setValue(128)
        dialog.temperature_scan_data = {
            10.0: {
                650.0: {
                    "precomputed_signals": {112: 100.0},
                    "peaks_info": [{"mz_rounded": 112}],
                }
            }
        }
        dialog.pie_species_data = [{"mz": 112, "species": "Parent"}]

        assert dialog._clear_stale_legacy_parent_mz()
        assert dialog.spin_parent_mz.value() == 0
    finally:
        dialog.deleteLater()


def test_project_parent_mz_is_shown_as_unconfirmed_project_preset(qapp, tmp_path):
    temperature_result = tmp_path / "temperature.csv"
    temperature_result.write_text(
        "temperature,mz,area,photon_energy,file\n"
        "650,128,100,10,a.txt\n650,112,50,10,a.txt\n",
        encoding="utf-8",
    )
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.set_project_settings(ProjectSettings(
            output_dir=str(tmp_path),
            temperature_scan_result_file=str(temperature_result),
            mf_parent_mz=128,
        ))

        assert dialog.spin_parent_mz.value() == 128
        assert "项目预设 m/z 128" in dialog.lbl_parent_mz_status.text()
        assert "请确认" in dialog.lbl_parent_mz_status.text()
        assert dialog.btn_calc_parent.isEnabled() is False

        dialog.btn_confirm_parent_mz.click()

        assert dialog._parent_mz_confirmed is True
        assert "已确认 m/z 128" in dialog.lbl_parent_mz_status.text()
        assert dialog.btn_calc_parent.isEnabled() is True
    finally:
        dialog.deleteLater()


def test_manual_parent_mz_change_requires_confirmation(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.temperature_scan_data = {
            10.0: {
                650.0: {
                    "precomputed_signals": {112: 50.0},
                    "peaks_info": [{"mz_rounded": 112}],
                }
            }
        }

        dialog.spin_parent_mz.setValue(112)

        assert dialog._parent_mz_confirmed is False
        assert "请确认" in dialog.lbl_parent_mz_status.text()
        assert dialog.btn_calc_parent.isEnabled() is False

        dialog._confirm_parent_mz()

        assert dialog._parent_mz_confirmed is True
        assert dialog.btn_calc_parent.isEnabled() is True
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
        assert dialog.energy_parent_table.rowHeight(0) == 32
        assert mz_combo.height() <= 24
        assert species_combo.height() <= 24

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


def test_low_energy_reference_cache_uses_configured_energy_signal(qapp, monkeypatch):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.spin_parent_mz.setValue(112)
        dialog.spin_parent_energy.setValue(10.0)
        dialog.spin_parent_t0.setValue(650)
        dialog.parent_mf_results = {650.0: 0.02}
        dialog.available_energies = [8.0, 10.0]
        dialog.temperature_scan_data = {
            8.0: {
                650.0: {
                    "precomputed_signals": {92: 8.0},
                    "peaks_info": [],
                    "avg_data": [],
                    "avg_io": 1.0,
                    "filenames": "8eV.txt",
                    "repeat_count": 1,
                }
            },
            10.0: {
                650.0: {
                    "precomputed_signals": {92: 10.0, 112: 100.0},
                    "peaks_info": [],
                    "avg_data": [],
                    "avg_io": 1.0,
                    "filenames": "10eV.txt",
                    "repeat_count": 1,
                }
            },
        }
        dialog.energy_parent_config = {8.0: {"mz": 92, "species_name": "LowRef"}}
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
            captured["signal_data"] = signal_data
            captured["calc_energy"] = calc_energy
            return {650.0: 0.003}

        monkeypatch.setattr(dialog, "_calc_product_mf_auto", fake_product_mf)

        dialog._recalculate_parent_mf_by_energy()

        assert captured["signal_data"] == {650.0: 8.0}
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


def test_dialog_product_mf_uses_max_reference_signal_temperature_when_t0_missing(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.project_settings = ProjectSettings(mf_mass_disc_exponent=0.0)
        dialog.spin_parent_t0.setValue(550)
        dialog.expansion_coefficients = {650.0: 1.0, 750.0: 1.0}
        dialog.database = [
            {"mz": 18, "species": "Water", "energies": np.array([12.0]), "cross_sections": np.array([2.0])},
            {"mz": 30, "species": "NO", "energies": np.array([12.0]), "cross_sections": np.array([4.0])},
        ]
        dialog.mz_index = {18: [0], 30: [1]}

        mf = dialog._calc_product_mf_from_signal(
            18,
            {"mz": 18, "species": "Water", "ie": 11.0},
            12.0,
            {650.0: 50.0, 750.0: 80.0},
            30,
            30.0,
            12.0,
            {650.0: 1000.0, 750.0: 800.0},
            0.01,
            ref_species_name="NO",
        )

        assert mf is not None
        assert mf[650.0] == pytest.approx(0.00125)
        assert mf[750.0] == pytest.approx(0.002)
    finally:
        dialog.deleteLater()


def test_dialog_product_mf_uses_energy_specific_expansion_coefficients(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.project_settings = ProjectSettings(mf_mass_disc_exponent=0.0)
        dialog.spin_parent_t0.setValue(650)
        dialog.expansion_coefficients = {
            14.6: {650.0: 2.0, 750.0: 2.0},
            14.8: {650.0: 10.0, 750.0: 5.0},
        }
        dialog.database = [
            {"mz": 18, "species": "Water", "energies": np.array([14.8]), "cross_sections": np.array([2.0])},
            {"mz": 30, "species": "NO", "energies": np.array([14.6]), "cross_sections": np.array([4.0])},
        ]
        dialog.mz_index = {18: [0], 30: [1]}

        mf = dialog._calc_product_mf_from_signal(
            18,
            {"mz": 18, "species": "Water", "ie": 12.0},
            14.8,
            {750.0: 100.0},
            30,
            30.0,
            14.6,
            {650.0: 1000.0},
            0.01,
            ref_species_name="NO",
        )

        assert mf is not None
        assert mf[750.0] == pytest.approx(0.0008)
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


def test_results_summary_uses_auto_product_results_only(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.spin_parent_mz.setValue(112)
        dialog.spin_parent_energy.setValue(8.0)
        dialog.parent_mf_results = {650.0: 0.02}
        dialog.parent_mf_by_energy = {8.0: {650.0: 0.02}}
        dialog.parent_config_by_energy = {
            8.0: {"mz": 112, "species_name": "1,1-dimethyl cyclohexane"},
        }
        dialog.all_species_mf = {
            (112, "1,1-dimethyl cyclohexane", 8.0): {650.0: 0.02},
            (15, "Methyl radical", 9.5): {650.0: 0.001, 700.0: 0.0015},
        }

        dialog._update_results_table()

        assert dialog.results_table.rowCount() == 2
        assert dialog.results_table.item(0, 0).text() == "产物"
        assert dialog.results_table.item(0, 1).text() == "15"
        assert dialog.results_table.item(0, 2).text() == "Methyl radical"
        assert "1 条结果" in dialog.lbl_results_status.text()
    finally:
        dialog.deleteLater()


def test_results_summary_follows_auto_plot_scope(qapp):
    dialog = MoleFractionDialog(Calibration(a=0.0, b=1.0, c=0.0), None)
    try:
        dialog.spin_parent_mz.setValue(112)
        dialog.spin_parent_energy.setValue(8.0)
        dialog.parent_config_by_energy = {
            8.0: {"mz": 112, "species_name": "1,1-dimethyl cyclohexane"},
        }
        dialog.all_species_mf = {
            (112, "1,1-dimethyl cyclohexane", 8.0): {650.0: 0.02},
            (15, "Methyl radical", 9.5): {650.0: 0.001},
        }
        dialog._set_combo_current_data(dialog.combo_auto_plot_scope, "all")

        dialog._update_results_table()

        assert dialog.results_table.rowCount() == 2
        assert dialog.results_table.item(0, 0).text() == "母体参考"
        assert dialog.results_table.item(1, 0).text() == "产物"
        assert "2 条结果" in dialog.lbl_results_status.text()
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
