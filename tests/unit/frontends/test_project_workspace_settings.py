from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.config import PeakDetectionConfig
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.project_settings import ProjectSettingsManager
from bl03u_masstool.core.project_settings import load_project_settings
from bl03u_masstool.core.project_settings import save_project_settings
from bl03u_masstool.core.project_lifecycle import project_root
from bl03u_masstool.frontends.pyqt_app.normalization.widget import FunctionDefaultsWidget
from bl03u_masstool.frontends.pyqt_app.spectrum.workbench import MainWindow


@pytest.fixture(autouse=True)
def isolated_project_settings(tmp_path, monkeypatch):
    import bl03u_masstool.core.config as core_config
    import bl03u_masstool.frontends.pyqt_app.normalization.widget as normalization_widget
    import bl03u_masstool.frontends.pyqt_app.spectrum.workspace_pages as workspace_pages

    config_dir = tmp_path / "runtime_config"
    config_dir.mkdir()

    def fake_save_calibration_config(calibration, path=None):
        return config_dir / "calibration.yaml"

    def fake_save_peak_detection_config(config, path=None):
        return config_dir / "peak_detection.yaml"

    monkeypatch.setattr(core_config, "save_calibration_config", fake_save_calibration_config)
    monkeypatch.setattr(core_config, "save_peak_detection_config", fake_save_peak_detection_config)
    monkeypatch.setattr(normalization_widget, "save_calibration_config", fake_save_calibration_config)
    monkeypatch.setattr(normalization_widget, "save_peak_detection_config", fake_save_peak_detection_config)
    monkeypatch.setattr(workspace_pages, "save_peak_detection_config", fake_save_peak_detection_config)

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
        peak_source_index = widget.temp_peak_source_combo.findData("manual")
        reference_index = widget.temp_reference_mode_combo.findData("individual")
        merge_index = widget.pie_merge_method_combo.findData("mean")
        assert peak_source_index >= 0
        assert reference_index >= 0
        assert merge_index >= 0
        widget.temp_peak_source_combo.setCurrentIndex(peak_source_index)
        widget.temp_reference_mode_combo.setCurrentIndex(reference_index)
        widget.temp_prefer_gaussian_check.setChecked(False)
        widget.temp_kr_mz_edit.setValue(86)
        widget.temp_curve_class_change_threshold_edit.setValue(0.35)
        widget.temp_curve_class_peak_fraction_edit.setValue(0.72)
        widget.pie_energy_decimals_edit.setValue(3)
        widget.pie_recursive_check.setChecked(False)
        widget.pie_prefer_gaussian_check.setChecked(False)
        widget.pie_multi_folder_check.setChecked(True)
        widget.pie_merge_method_combo.setCurrentIndex(merge_index)
        widget.pics_no_mz_edit.setValue(31)
        widget.pics_no_formula_edit.setText("15NO")
        widget.pics_no_mf_edit.setValue(0.02)
        widget.pics_new_species_mf_edit.setValue(0.004)
        widget.mf_parent_mz_edit.setValue(130)
        widget.mf_parent_initial_mf_edit.setValue(0.006)
        widget.mf_photon_energy_edit.setValue(11.2)
        widget.mf_reference_temperature_edit.setValue(575)

        widget.apply_to_settings(ps)

        assert ps.peak_algorithm == "cwt"
        assert ps.detection_min_idx == 4321
        assert ps.min_intensity == pytest.approx(12.5)
        assert ps.temp_peak_source == "manual"
        assert ps.temp_reference_mode == "individual"
        assert ps.temp_prefer_gaussian is False
        assert ps.temp_kr_mz == 86
        assert ps.temp_curve_class_change_threshold == pytest.approx(0.35)
        assert ps.temp_curve_class_peak_fraction == pytest.approx(0.72)
        assert ps.pie_energy_decimals == 3
        assert ps.pie_recursive is False
        assert ps.pie_prefer_gaussian is False
        assert ps.pie_multi_folder_mode is True
        assert ps.pie_merge_method == "mean"
        assert ps.pics_no_mz == 31
        assert ps.pics_no_formula == "15NO"
        assert ps.pics_no_mf == pytest.approx(0.02)
        assert ps.pics_new_species_mf == pytest.approx(0.004)
        assert ps.mf_parent_mz == 130
        assert ps.mf_parent_initial_mf == pytest.approx(0.006)
        assert ps.mf_photon_energy == pytest.approx(11.2)
        assert ps.mf_reference_temperature == pytest.approx(575.0)
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


def test_calibration_sync_persists_through_multiple_round_trips(qapp, tmp_path):
    """用户场景：修改项目管理参数→切到质谱→切回→再次修改→切到质谱，确保参数不丢失"""
    project_dir = tmp_path / "Calibration_Sync_Test"
    project_dir.mkdir(parents=True)
    (project_dir / "raw_data").mkdir()

    ps = ProjectSettings(
        project_name="Cal Sync Test",
        system="Test",
        output_dir=str(project_dir),
        cal_a=1.1e-7,
        cal_b=2.2e-4,
        cal_c=3.3,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        # 模拟打开项目
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)

        # Round 0: 初始状态 — 验证加载的定标参数
        window._apply_project_runtime_settings(ps)
        cal0 = window.current_calibration()
        assert cal0.a == pytest.approx(1.1e-7)
        assert cal0.b == pytest.approx(2.2e-4)
        assert cal0.c == pytest.approx(3.3)

        # Round 1: 模拟用户在项目管理页面修改定标参数
        widget = window.project_common_parameters_widget
        widget.calibration_a_edit.setValue(9.9e-7)
        widget.calibration_b_edit.setValue(8.8e-4)
        widget.calibration_c_edit.setValue(7.7)

        # 触发 editingFinished (模拟 spinbox 失去焦点)
        widget.calibration_a_edit.editingFinished.emit()
        widget.calibration_b_edit.editingFinished.emit()
        widget.calibration_c_edit.editingFinished.emit()

        # 模拟切到质谱页面
        window.switch_workspace_page("spectrum")
        cal1 = window.current_calibration()
        assert cal1.a == pytest.approx(9.9e-7), f"Round 1: expected a=9.9e-7, got {cal1.a}"
        assert cal1.b == pytest.approx(8.8e-4), f"Round 1: expected b=8.8e-4, got {cal1.b}"
        assert cal1.c == pytest.approx(7.7), f"Round 1: expected c=7.7, got {cal1.c}"

        # 验证 project.yaml 已更新
        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.cal_a == pytest.approx(9.9e-7)
        assert saved.cal_b == pytest.approx(8.8e-4)
        assert saved.cal_c == pytest.approx(7.7)

        # Round 2: 切回项目管理页面
        window.switch_workspace_page("project")
        cal2 = window.current_calibration()
        assert cal2.a == pytest.approx(9.9e-7), f"Round 2: expected a=9.9e-7, got {cal2.a}"
        assert cal2.b == pytest.approx(8.8e-4), f"Round 2: expected b=8.8e-4, got {cal2.b}"
        assert cal2.c == pytest.approx(7.7), f"Round 2: expected c=7.7, got {cal2.c}"

        # 再次修改定标参数
        widget.calibration_a_edit.setValue(5.5e-7)
        widget.calibration_b_edit.setValue(4.4e-4)
        widget.calibration_c_edit.setValue(3.3)

        # 触发 editingFinished
        widget.calibration_a_edit.editingFinished.emit()
        widget.calibration_b_edit.editingFinished.emit()
        widget.calibration_c_edit.editingFinished.emit()

        # 切到质谱页面
        window.switch_workspace_page("spectrum")
        cal3 = window.current_calibration()
        assert cal3.a == pytest.approx(5.5e-7), f"Round 3: expected a=5.5e-7, got {cal3.a}"
        assert cal3.b == pytest.approx(4.4e-4), f"Round 3: expected b=4.4e-4, got {cal3.b}"
        assert cal3.c == pytest.approx(3.3), f"Round 3: expected c=3.3, got {cal3.c}"

        # 验证 project.yaml 再次更新
        saved2 = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved2.cal_a == pytest.approx(5.5e-7)
        assert saved2.cal_b == pytest.approx(4.4e-4)
        assert saved2.cal_c == pytest.approx(3.3)

        # 切回项目管理页面验证 UI 不被覆盖
        window.switch_workspace_page("project")
        assert widget.calibration_a_edit.value() == pytest.approx(5.5e-7)
        assert widget.calibration_b_edit.value() == pytest.approx(4.4e-4)
        assert widget.calibration_c_edit.value() == pytest.approx(3.3)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_loaded_spectrum_is_reprojected_after_second_project_calibration_edit(qapp, tmp_path, monkeypatch):
    """Regression: loaded spectrum coordinates must follow the second project calibration edit."""
    import bl03u_masstool.frontends.pyqt_app.spectrum.workbench as workbench_module

    project_dir = tmp_path / "Calibration_Reproject_Test"
    project_dir.mkdir(parents=True)

    ps = ProjectSettings(
        project_name="Cal Reproject Test",
        system="Test",
        output_dir=str(project_dir),
        cal_a=0.0,
        cal_b=1.0,
        cal_c=0.0,
        peak_algorithm="legacy",
        detection_min_idx=0,
        threshold_end=0.5,
        min_intensity=1.0,
        nearby_peak_window=3,
        duplicate_window=3,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    peak_config = PeakDetectionConfig(
        algorithm="legacy",
        detection_min_idx=0,
        threshold_end=0.5,
        min_intensity=1.0,
        nearby_peak_window=3,
        duplicate_window=3,
        gaussian_window_max=8,
        boundary_padding=0,
    )
    monkeypatch.setattr(workbench_module, "load_peak_detection_config", lambda: peak_config)

    def fail_warning(*args, **kwargs):
        pytest.fail(f"Unexpected warning dialog: {args[2] if len(args) > 2 else args}")

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", fail_warning)

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)
        window._load_project_settings_to_parameter_widgets(ps)
        window._apply_project_runtime_settings(ps)

        tof = np.arange(100.0, 220.0)
        peak_index = 35
        peak_time = tof[peak_index]
        intensity = np.exp(-0.5 * ((np.arange(tof.size) - peak_index) / 2.0) ** 2) * 100.0
        window.current_time_offset = float(tof[0])
        window.setup_plots(tof, intensity)

        window.auto_find_peaks()
        assert window.peakData.rowCount() == 1
        assert float(window.peakData.item(0, 1).text()) == pytest.approx(peak_time, abs=0.02)
        assert float(window.peakData.item(0, 2).text()) == pytest.approx(peak_time, abs=0.02)
        assert window.current_plot_axis_x[peak_index] == pytest.approx(peak_time)

        window.switch_workspace_page("project")
        widget = window.project_common_parameters_widget
        widget.calibration_a_edit.setValue(0.0)
        widget.calibration_b_edit.setValue(2.0)
        widget.calibration_c_edit.setValue(5.0)
        widget.calibration_a_edit.editingFinished.emit()
        widget.calibration_b_edit.editingFinished.emit()
        widget.calibration_c_edit.editingFinished.emit()

        window.switch_workspace_page("spectrum")
        expected_mz = peak_time * 2.0 + 5.0
        assert window.current_plot_axis_x[peak_index] == pytest.approx(expected_mz)

        window.auto_find_peaks()
        assert window.peakData.rowCount() == 1
        assert float(window.peakData.item(0, 1).text()) == pytest.approx(peak_time, abs=0.02)
        assert float(window.peakData.item(0, 2).text()) == pytest.approx(expected_mz, abs=0.02)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_project_page_edits_sync_when_switching_to_tools(qapp, tmp_path, monkeypatch):
    import bl03u_masstool.frontends.pyqt_app.spectrum.workbench as workbench_module

    project_dir = tmp_path / "Project_Live_Sync"
    project_dir.mkdir(parents=True)

    ps = ProjectSettings(
        project_name="Live Sync",
        system="C6H6",
        output_dir=str(project_dir),
        peak_algorithm="legacy",
        detection_min_idx=111,
        min_intensity=3.0,
        mass_discrimination=1.0,
        light_source="io",
        pie_photon_mode="first",
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    calls = []

    def fake_detect_peaks_by_algorithm(y_data, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(workbench_module, "detect_peaks_by_algorithm", fake_detect_peaks_by_algorithm)

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)
        window._load_project_settings_to_parameter_widgets(ps)
        window._apply_project_runtime_settings(ps)
        window.workspace_stack.setCurrentWidget(window.project_page)

        algorithm_index = window.project_function_defaults_widget.peak_algorithm_combo.findData("cwt")
        assert algorithm_index >= 0
        window.project_function_defaults_widget.peak_algorithm_combo.setCurrentIndex(algorithm_index)
        window.project_function_defaults_widget.peak_detection_min_idx_edit.setValue(2222)
        window.project_function_defaults_widget.peak_min_intensity_edit.setValue(12.5)

        light_index = window.project_common_parameters_widget.light_source_combo.findData("beam_current")
        pie_index = window.project_common_parameters_widget.pie_photon_mode_combo.findData("off")
        assert light_index >= 0
        assert pie_index >= 0
        window.project_common_parameters_widget.light_source_combo.setCurrentIndex(light_index)
        window.project_common_parameters_widget.pie_photon_mode_combo.setCurrentIndex(pie_index)
        window.project_common_parameters_widget.mass_discrimination_edit.setValue(0.42)

        window.switch_workspace_page("spectrum")

        active = window.project_settings_manager.get()
        assert active.peak_algorithm == "cwt"
        assert active.detection_min_idx == 2222
        assert active.min_intensity == pytest.approx(12.5)
        assert active.light_source == "beam_current"
        assert active.pie_photon_mode == "off"
        assert active.mass_discrimination == pytest.approx(0.42)

        assert window.current_peak_detection_config().algorithm == "cwt"
        assert window.current_peak_detection_config().detection_min_idx == 2222
        assert window.normalization_settings.light_source == "beam_current"
        assert window.temperature_page.project_settings.peak_algorithm == "cwt"
        assert window.temperature_page.normalization_settings.mass_discrimination == pytest.approx(0.42)
        assert window.pie_page.project_settings.peak_algorithm == "cwt"
        assert window.pie_page.normalization_settings.pie_photon_mode == "off"

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.peak_algorithm == "cwt"
        assert saved.detection_min_idx == 2222
        assert saved.light_source == "beam_current"
        assert saved.mass_discrimination == pytest.approx(0.42)

        tof = np.arange(100.0, 140.0)
        intensity = np.zeros_like(tof)
        window.current_time_offset = float(tof[0])
        window.setup_plots(tof, intensity)
        window.auto_find_peaks()
        assert calls
        assert calls[-1]["algorithm"] == "cwt"
        assert calls[-1]["detection_min_idx"] == 2222
        assert calls[-1]["min_intensity"] == pytest.approx(12.5)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_function_default_edits_sync_when_switching_to_tools(qapp, tmp_path):
    project_dir = tmp_path / "Project_Function_Default_Sync"
    project_dir.mkdir(parents=True)

    ps = ProjectSettings(
        project_name="Function Default Sync",
        system="C6H6",
        output_dir=str(project_dir),
        pie_energy_decimals=1,
        pie_recursive=True,
        pie_prefer_gaussian=True,
        pie_multi_folder_mode=False,
        pie_merge_method="low_energy_dominant",
        temp_peak_source="auto",
        temp_reference_mode="sum",
        temp_prefer_gaussian=True,
        temp_kr_mz=84,
        pics_no_mz=30,
        pics_no_formula="NO",
        pics_no_mf=0.01,
        pics_new_species_mf=0.002,
        mf_parent_mz=0,
        mf_parent_initial_mf=0.002,
        mf_photon_energy=10.0,
        mf_reference_temperature=None,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)
        window._load_project_settings_to_parameter_widgets(ps)
        window._apply_project_runtime_settings(ps)
        window.workspace_stack.setCurrentWidget(window.project_page)

        widget = window.project_function_defaults_widget
        window.project_tabs.setCurrentWidget(widget)
        peak_source_index = widget.temp_peak_source_combo.findData("manual")
        reference_index = widget.temp_reference_mode_combo.findData("individual")
        merge_index = widget.pie_merge_method_combo.findData("first_segment_dominant")
        assert peak_source_index >= 0
        assert reference_index >= 0
        assert merge_index >= 0
        widget.temp_peak_source_combo.setCurrentIndex(peak_source_index)
        widget.temp_reference_mode_combo.setCurrentIndex(reference_index)
        widget.temp_prefer_gaussian_check.setChecked(False)
        widget.temp_kr_mz_edit.setValue(86)
        widget.temp_curve_class_change_threshold_edit.setValue(0.31)
        widget.temp_curve_class_peak_fraction_edit.setValue(0.73)
        widget.pie_energy_decimals_edit.setValue(4)
        widget.pie_recursive_check.setChecked(False)
        widget.pie_prefer_gaussian_check.setChecked(False)
        widget.pie_multi_folder_check.setChecked(True)
        widget.pie_merge_method_combo.setCurrentIndex(merge_index)
        widget.pics_no_mz_edit.setValue(31)
        widget.pics_no_formula_edit.setText("15NO")
        widget.pics_no_mf_edit.setValue(0.021)
        widget.pics_new_species_mf_edit.setValue(0.004)
        widget.mf_parent_mz_edit.setValue(130)
        widget.mf_parent_initial_mf_edit.setValue(0.006)
        widget.mf_photon_energy_edit.setValue(11.25)
        widget.mf_reference_temperature_edit.setValue(575)

        # Switching within the project page should also collect edits so
        # project-page tools such as Kr calculation do not read stale defaults.
        window.project_tabs.setCurrentWidget(window.project_common_parameters_widget)
        assert window.project_settings_manager.get().pie_energy_decimals == 4
        assert window.project_settings_manager.get().temp_kr_mz == 86

        # Re-clicking the current Project tab should collect in-memory edits,
        # not reload the older project.yaml over the UI values.
        window.switch_workspace_page("project")
        assert widget.pie_energy_decimals_edit.value() == 4
        assert widget.temp_kr_mz_edit.value() == 86

        window.switch_workspace_page("pie")

        active = window.project_settings_manager.get()
        assert active.temp_peak_source == "manual"
        assert active.temp_reference_mode == "individual"
        assert active.temp_prefer_gaussian is False
        assert active.temp_kr_mz == 86
        assert active.temp_curve_class_change_threshold == pytest.approx(0.31)
        assert active.temp_curve_class_peak_fraction == pytest.approx(0.73)
        assert active.pie_energy_decimals == 4
        assert active.pie_recursive is False
        assert active.pie_prefer_gaussian is False
        assert active.pie_multi_folder_mode is True
        assert active.pie_merge_method == "first_segment_dominant"
        assert active.pics_no_mz == 31
        assert active.pics_no_formula == "15NO"
        assert active.pics_no_mf == pytest.approx(0.021)
        assert active.pics_new_species_mf == pytest.approx(0.004)
        assert active.mf_parent_mz == 130
        assert active.mf_parent_initial_mf == pytest.approx(0.006)
        assert active.mf_photon_energy == pytest.approx(11.25)
        assert active.mf_reference_temperature == pytest.approx(575.0)

        assert window.temperature_page.project_settings.temp_reference_mode == "individual"
        assert window.temperature_page.project_settings.temp_kr_mz == 86
        assert window.pie_page.project_settings.pie_energy_decimals == 4
        assert window.pie_page.use_multi_folders.isChecked()
        assert window.pie_page.merge_method_combo.currentData() == "first_segment_dominant"
        assert window.pics_page.spin_no_mz.value() == 31
        assert window.pics_page.txt_no_formula.text() == "15NO"
        assert window.pics_page.double_no_mf.value() == pytest.approx(0.021)
        assert window.mole_fraction_page.project_settings.mf_parent_mz == 130

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.pie_energy_decimals == 4
        assert saved.temp_reference_mode == "individual"
        assert saved.pics_no_formula == "15NO"
        assert saved.mf_photon_energy == pytest.approx(11.25)
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
