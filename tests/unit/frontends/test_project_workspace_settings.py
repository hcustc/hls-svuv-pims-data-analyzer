from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.project_settings import ProjectSettingsManager
from bl03u_masstool.core.project_settings import load_project_settings
from bl03u_masstool.core.project_settings import save_project_settings
from bl03u_masstool.core.project_lifecycle import project_root
from bl03u_masstool.frontends.pyqt_app.normalization.widget import FunctionDefaultsWidget
from bl03u_masstool.frontends.pyqt_app.spectrum.workbench import MainWindow


@pytest.fixture(autouse=True)
def isolated_project_settings(tmp_path):
    manager = ProjectSettingsManager()
    manager.set_project_path(tmp_path / "active_project")
    try:
        yield
    finally:
        manager.clear_project_path()


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_calibration_spinboxes_accept_command_v_paste(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersWidget

    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
        widget.show()
        qapp.processEvents()

        cases = [
            (widget.calibration_a_edit, "3.66334E-07", 3.66334e-7),
            (widget.calibration_b_edit, "0.000637719", 0.000637719),
            (widget.calibration_c_edit, "0.272489072", 0.272489072),
        ]
        for spin, pasted_text, expected in cases:
            spin.setValue(999.0)
            spin.setFocus()
            spin.lineEdit().setFocus()
            spin.lineEdit().setCursorPosition(len(spin.lineEdit().text()))
            qapp.processEvents()
            QtWidgets.QApplication.clipboard().setText(pasted_text)
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_V,
                QtCore.Qt.KeyboardModifier.MetaModifier,
                "v",
            )
            QtWidgets.QApplication.sendEvent(spin.lineEdit(), event)

            assert event.isAccepted()
            assert spin.value() == pytest.approx(expected)
    finally:
        widget.deleteLater()


def test_kr_energy_selection_updates_displayed_lambda_values(qapp, monkeypatch):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersWidget
    import bl03u_masstool.frontends.pyqt_app.normalization.widget as normalization_widget

    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
        monkeypatch.setattr(normalization_widget, "save_normalization_settings", lambda settings: Path("/tmp/normalization.yaml"))
        df = pd.DataFrame(
            [
                {"photon_energy": 14.6, "temperature": 400.0, "kr_signal": 10.0, "expansion_lambda": 1.0},
                {"photon_energy": 14.6, "temperature": 800.0, "kr_signal": 20.0, "expansion_lambda": 2.0},
                {"photon_energy": 14.7, "temperature": 400.0, "kr_signal": 10.0, "expansion_lambda": 1.0},
                {"photon_energy": 14.7, "temperature": 800.0, "kr_signal": 40.0, "expansion_lambda": 4.0},
            ]
        )

        widget.on_kr_factors_ready(df)
        assert widget.factor_table.item(1, 2).text() == "3.000000"

        energy_index = widget.energy_combo.findData(14.6)
        assert energy_index >= 0
        widget.energy_combo.setCurrentIndex(energy_index)

        assert widget.factor_table.item(1, 2).text() == "2.000000"
        assert widget.factor_table.item(1, 1).text() == "20.0000"
        assert widget.settings.expansion_factors[14.6][800.0] == pytest.approx(2.0)
        assert widget.settings.expansion_factors[14.7][800.0] == pytest.approx(4.0)
    finally:
        widget.deleteLater()


def test_function_defaults_apply_to_project_settings(qapp):
    widget = FunctionDefaultsWidget()
    try:
        ps = ProjectSettings(peak_algorithm="legacy", detection_min_idx=111)
        widget.set_project_settings(ps)

        index = widget.peak_algorithm_combo.findData("cwt")
        assert index >= 0
        widget.peak_algorithm_combo.setCurrentIndex(index)
        widget.peak_detection_min_idx_edit.setValue(4321)
        widget.peak_min_intensity_edit.setValue(12.5)

        widget.apply_to_settings(ps)

        assert ps.peak_algorithm == "cwt"
        assert ps.detection_min_idx == 4321
        assert ps.min_intensity == pytest.approx(12.5)
    finally:
        widget.deleteLater()


def test_common_parameters_apply_to_project_settings(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersWidget

    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
        ps = ProjectSettings()
        widget.set_project_settings(ps)

        light_idx = widget.light_source_combo.findData("beam_current")
        pie_idx = widget.pie_photon_mode_combo.findData("off")
        preset_idx = widget.mf_md_preset_combo.findData("30 Torr (Catalysis)")
        assert light_idx >= 0
        assert pie_idx >= 0
        assert preset_idx >= 0

        widget.light_source_combo.setCurrentIndex(light_idx)
        widget.pie_photon_mode_combo.setCurrentIndex(pie_idx)
        widget.mass_discrimination_edit.setValue(0.42)
        widget.calibration_a_edit.setValue(1.23e-7)
        widget.calibration_b_edit.setValue(2.34e-4)
        widget.calibration_c_edit.setValue(0.56)
        widget.mf_md_preset_combo.setCurrentIndex(preset_idx)
        widget.mf_mass_disc_exponent_edit.setValue(0.75148)
        widget.kr_folder_edit.setText("/tmp/kr")
        widget.kr_peak_mode_manual_radio.setChecked(True)
        widget.kr_peak_file_edit.setText("/tmp/kr_peak.csv")
        widget.element_checks["C"].setChecked(True)
        widget.element_checks["H"].setChecked(True)
        widget.element_checks["O"].setChecked(False)

        widget.apply_to_settings(ps)

        assert ps.light_source == "beam_current"
        assert ps.pie_photon_mode == "off"
        assert ps.mass_discrimination == pytest.approx(0.42)
        assert ps.cal_a == pytest.approx(1.23e-7)
        assert ps.cal_b == pytest.approx(2.34e-4)
        assert ps.cal_c == pytest.approx(0.56)
        assert ps.mf_md_preset == "30 Torr (Catalysis)"
        assert ps.mf_mass_disc_exponent == pytest.approx(0.75148)
        assert ps.kr_calibration_folder == "/tmp/kr"
        assert ps.kr_calibration_peak_file == "/tmp/kr_peak.csv"
        assert "C" in ps.selected_elements
        assert "H" in ps.selected_elements
        assert "O" not in ps.selected_elements
    finally:
        widget.deleteLater()


def test_save_and_apply_uses_selected_parent_directory_for_new_project(qapp, tmp_path, monkeypatch):
    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir()

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    try:
        window.project_name_edit.setText("test")
        window.project_system_edit.setText("test")
        window.project_description_edit.setText("test")
        window.project_output_dir_edit.setText(str(downloads_dir))

        window.save_and_apply_project_settings()

        project_dir = downloads_dir / "test"
        assert project_dir.exists()
        assert (project_dir / "config" / "project.yaml").exists()
        assert window.project_output_dir_edit.text() == str(project_dir)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_normalize_project_output_dir_resolves_relative_project_directory(qapp):
    window = MainWindow()
    try:
        ps = ProjectSettings(project_name="test", system="test", output_dir="test")

        normalized = window._normalize_project_output_dir(ps)

        expected_dir = project_root(ProjectSettings(output_dir="test"))
        assert Path(normalized.output_dir) == expected_dir
        assert Path(window.project_output_dir_edit.text()) == expected_dir
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_open_project_restores_data_sources_and_applies_to_tools(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Test"
    single_file = project_dir / "raw_data" / "single_spectrum" / "single.txt"
    sum_folder = project_dir / "raw_data" / "sum_spectrum"
    temperature_folder = project_dir / "raw_data" / "temperature_scan"
    pie_folder = project_dir / "raw_data" / "pie_scan"
    manual_peak = project_dir / "raw_data" / "manual_peaks" / "peaks.csv"

    for folder in (single_file.parent, sum_folder, temperature_folder, pie_folder, manual_peak.parent):
        folder.mkdir(parents=True, exist_ok=True)
    single_file.write_text("tof intensity\n1 2\n", encoding="utf-8")
    (sum_folder / "sum.txt").write_text("tof intensity\n1 2\n", encoding="utf-8")
    (temperature_folder / "temp.txt").write_text("tof intensity\n1 2\n", encoding="utf-8")
    (pie_folder / "pie.txt").write_text("tof intensity\n1 2\n", encoding="utf-8")
    manual_peak.write_text("mz,start,end\n44,1,2\n", encoding="utf-8")

    ps = ProjectSettings(
        project_name="Opened Project",
        system="C6H6",
        output_dir=str(project_dir),
        single_spectrum_file=str(single_file),
        sum_spectrum_folder=str(sum_folder),
        temperature_scan_folder=str(temperature_folder),
        pie_scan_folder=str(pie_folder),
        manual_peak_file=str(manual_peak),
        cal_a=9.1e-7,
        cal_b=8.2e-4,
        cal_c=0.73,
        light_source="beam_current",
        pie_photon_mode="off",
        mass_discrimination=0.55,
        mf_md_preset="30 Torr (Catalysis)",
        mf_mass_disc_exponent=0.75148,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(project_dir))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    try:
        window.open_project()

        assert window.project_single_file_edit.text() == str(single_file)
        assert window.project_sum_folder_edit.text() == str(sum_folder)
        assert window.project_temperature_folder_edit.text() == str(temperature_folder)
        assert window.project_pie_folder_edit.text() == str(pie_folder)
        assert window.project_manual_peak_edit.text() == str(manual_peak)
        assert window.lineEdit.text() == str(single_file)
        assert window.folder_path.text() == str(sum_folder)
        assert window.spectrum_source_scope == "project"
        assert window.lineEdit.isReadOnly()
        assert not window.singleBrowseButton.isEnabled()
        assert window.project_common_parameters_widget.show_actions is False
        assert window.project_common_parameters_widget.action_bar.isHidden()
        assert window.current_calibration().a == pytest.approx(9.1e-7)
        assert window.normalization_settings.light_source == "beam_current"
        assert window.normalization_settings.pie_photon_mode == "off"
        assert window.normalization_settings.mass_discrimination == pytest.approx(0.55)
        assert window.project_common_parameters_widget.mass_discrimination_edit.value() == pytest.approx(0.55)
        assert window.project_common_parameters_widget.mf_md_preset_combo.currentText() == "30 Torr (Catalysis)"
        assert window.project_common_parameters_widget.mf_mass_disc_exponent_edit.value() == pytest.approx(0.75148)
        assert window.temperature_page.project_settings.temperature_scan_folder == str(temperature_folder)
        assert window.temperature_page.normalization_settings.mass_discrimination == pytest.approx(0.55)
        assert window.temperature_page.calibration.a == pytest.approx(9.1e-7)
        assert window.pie_page.project_settings.pie_scan_folder == str(pie_folder)
        assert window.pie_page.normalization_settings.mass_discrimination == pytest.approx(0.55)
        assert window.mole_fraction_page.project_settings.mf_md_preset == "30 Torr (Catalysis)"
        assert window.datasource_row_status_labels["single_spectrum"].text().startswith("✓")
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_spectrum_workbench_custom_source_does_not_overwrite_project_source(qapp, tmp_path):
    project_dir = tmp_path / "Project_Source"
    project_single = project_dir / "raw_data" / "single_spectrum" / "project.txt"
    custom_single = tmp_path / "custom.txt"
    project_single.parent.mkdir(parents=True)
    project_single.write_text("tof intensity\n1 2\n", encoding="utf-8")
    custom_single.write_text("tof intensity\n3 4\n", encoding="utf-8")

    ps = ProjectSettings(
        project_name="Source Test",
        system="C6H6",
        output_dir=str(project_dir),
        single_spectrum_file=str(project_single),
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)
        window._apply_settings_to_tools(ps)

        assert window.spectrum_source_scope == "project"
        assert window.lineEdit.text() == str(project_single)

        window.set_spectrum_source_scope("custom")
        window.lineEdit.setText(str(custom_single))
        window._remember_custom_spectrum_paths()

        replacement = project_dir / "raw_data" / "single_spectrum" / "replacement.txt"
        replacement.write_text("tof intensity\n5 6\n", encoding="utf-8")
        window.project_single_file_edit.setText(str(replacement))
        window._auto_save_datasource()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.single_spectrum_file == str(replacement)
        assert window.project_single_file_edit.text() == str(replacement)
        assert window.lineEdit.text() == str(custom_single)
        assert window.spectrum_source_scope == "custom"
        assert not window.lineEdit.isReadOnly()
        assert window.singleBrowseButton.isEnabled()

        window.set_spectrum_source_scope("project")
        assert window.lineEdit.text() == str(replacement)
        assert window.lineEdit.isReadOnly()
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_save_and_apply_persists_common_parameters_to_project_file(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Params"
    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    try:
        window.project_name_edit.setText("Project Params")
        window.project_system_edit.setText("C6H6")
        window.project_output_dir_edit.setText(str(project_dir))

        light_idx = window.project_common_parameters_widget.light_source_combo.findData("beam_current")
        pie_idx = window.project_common_parameters_widget.pie_photon_mode_combo.findData("off")
        preset_idx = window.project_common_parameters_widget.mf_md_preset_combo.findData("150 Torr (Combustion)")
        assert light_idx >= 0
        assert pie_idx >= 0
        assert preset_idx >= 0

        window.project_common_parameters_widget.light_source_combo.setCurrentIndex(light_idx)
        window.project_common_parameters_widget.pie_photon_mode_combo.setCurrentIndex(pie_idx)
        window.project_common_parameters_widget.mass_discrimination_edit.setValue(0.33)
        window.project_common_parameters_widget.calibration_a_edit.setValue(4.56e-7)
        window.project_common_parameters_widget.calibration_b_edit.setValue(7.89e-4)
        window.project_common_parameters_widget.calibration_c_edit.setValue(0.12)
        window.project_common_parameters_widget.mf_md_preset_combo.setCurrentIndex(preset_idx)
        window.project_common_parameters_widget.mf_mass_disc_exponent_edit.setValue(0.76155)

        window.save_and_apply_project_settings()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.light_source == "beam_current"
        assert saved.pie_photon_mode == "off"
        assert saved.mass_discrimination == pytest.approx(0.33)
        assert saved.cal_a == pytest.approx(4.56e-7)
        assert saved.cal_b == pytest.approx(7.89e-4)
        assert saved.cal_c == pytest.approx(0.12)
        assert saved.mf_md_preset == "150 Torr (Combustion)"
        assert saved.mf_mass_disc_exponent == pytest.approx(0.76155)
        assert window.normalization_settings.mass_discrimination == pytest.approx(0.33)
        assert window.current_calibration().a == pytest.approx(4.56e-7)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_import_finished_registers_data_source_and_refreshes_tools(qapp, tmp_path):
    project_dir = tmp_path / "Project_Test"
    imported_dir = project_dir / "raw_data" / "temperature_scan" / "温度扫描"
    imported_dir.mkdir(parents=True)
    (imported_dir / "temp.txt").write_text("tof intensity\n1 2\n", encoding="utf-8")

    ps = ProjectSettings(project_name="Import Test", system="C6H6", output_dir=str(project_dir))
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)

        window._on_import_finished(
            {
                "success": True,
                "field_name": "temperature_scan_folder",
                "destination": str(imported_dir),
                "label": "温度扫描目录",
                "source_key": "temperature_scan",
            }
        )

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.temperature_scan_folder == str(imported_dir)
        assert window.project_temperature_folder_edit.text() == str(imported_dir)
        assert window.temperature_page.project_settings.temperature_scan_folder == str(imported_dir)
        assert window.datasource_row_status_labels["temperature_scan"].text().startswith("✓")
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()
