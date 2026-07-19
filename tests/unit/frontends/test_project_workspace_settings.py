from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

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
from bl03u_masstool.frontends.pyqt_app.temporary_analysis_settings import (
    TemporaryAnalysisSettingsDialog,
)


def _wait_for_project_materialization(qapp, window: MainWindow, timeout_ms: int = 5000) -> None:
    timer = QtCore.QElapsedTimer()
    timer.start()
    while (
        hasattr(window, "_materialize_worker_manager")
        and window._materialize_worker_manager.is_running()
        and timer.elapsed() < timeout_ms
    ):
        qapp.processEvents()
        QtCore.QThread.msleep(5)
    qapp.processEvents()
    assert not (
        hasattr(window, "_materialize_worker_manager")
        and window._materialize_worker_manager.is_running()
    )


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


def test_project_data_source_ui_round_trips_multiple_pie_folders(qapp, tmp_path):
    low_folder = tmp_path / "pie_low"
    high_folder = tmp_path / "pie_high"
    low_folder.mkdir()
    high_folder.mkdir()
    ps = ProjectSettings(
        project_name="Multi PIE Project",
        output_dir=str(tmp_path / "project"),
        pie_scan_folder=str(low_folder),
        pie_scan_folders=[str(low_folder), str(high_folder)],
        pie_multi_folder_mode=True,
    )
    window = MainWindow()
    try:
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)

        assert window.project_pie_folders_list.count() == 2
        assert window.project_pie_folder_edit.text() == str(low_folder)

        collected = window._collect_project_settings_from_ui()
        assert collected.pie_scan_folder == str(low_folder)
        assert collected.pie_scan_folders == [str(low_folder), str(high_folder)]
        assert collected.pie_multi_folder_mode is True
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_project_data_source_adds_pie_segment_folder_without_mode_switch(
    qapp,
    tmp_path,
    monkeypatch,
):
    low_folder = tmp_path / "pie_low"
    high_folder = tmp_path / "pie_high"
    low_folder.mkdir()
    high_folder.mkdir()
    window = MainWindow()
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(high_folder),
    )
    monkeypatch.setattr(window, "_auto_save_datasource", lambda: None)
    try:
        window._set_project_pie_folders([str(low_folder)])

        window.add_project_pie_folder()

        assert window._project_pie_folders_from_ui() == [
            str(low_folder),
            str(high_folder),
        ]
        assert not hasattr(window.project_function_defaults_widget, "pie_multi_folder_check")
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_pie_manage_segments_action_opens_project_data_tab(qapp):
    window = MainWindow()
    try:
        window.switch_workspace_page("pie")
        window.project_tabs.setCurrentWidget(window.project_common_parameters_widget)

        window.pie_page._open_project_settings()

        assert window.workspace_stack.currentWidget() is window.project_page
        assert window.project_tabs.currentWidget() is window.project_identity_page
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


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


def test_project_calibration_round_trip_preserves_fitted_precision(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersWidget

    calibration = Calibration(
        a=3.66334123456789e-7,
        b=0.000637719123456789,
        c=0.2724890721234567,
    )
    settings = ProjectSettings(
        cal_a=calibration.a,
        cal_b=calibration.b,
        cal_c=calibration.c,
    )
    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
        widget.set_project_settings(settings)
        restored = ProjectSettings()
        widget.apply_to_settings(restored)

        assert restored.cal_a == pytest.approx(calibration.a, rel=1e-12)
        assert restored.cal_b == pytest.approx(calibration.b, rel=1e-14)
        assert restored.cal_c == pytest.approx(calibration.c, rel=1e-14)
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
        ps = ProjectSettings(
            peak_algorithm="legacy",
            detection_min_idx=111,
            pie_scan_folders=["pie-low", "pie-high"],
        )
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
        temp_integration_index = widget.temp_integration_method_combo.findData("baseline")
        assert temp_integration_index >= 0
        widget.temp_integration_method_combo.setCurrentIndex(temp_integration_index)
        widget.temp_curve_class_change_threshold_edit.setValue(0.35)
        widget.temp_curve_class_peak_fraction_edit.setValue(0.72)
        widget.pie_energy_decimals_edit.setValue(3)
        widget.pie_recursive_check.setChecked(False)
        integration_index = widget.pie_integration_method_combo.findData("baseline")
        assert integration_index >= 0
        widget.pie_integration_method_combo.setCurrentIndex(integration_index)
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
        assert ps.temp_integration_method == "baseline"
        assert ps.temp_curve_class_change_threshold == pytest.approx(0.35)
        assert ps.temp_curve_class_peak_fraction == pytest.approx(0.72)
        assert ps.pie_energy_decimals == 3
        assert ps.pie_recursive is False
        assert ps.pie_prefer_gaussian is False
        assert ps.pie_integration_method == "baseline"
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


def test_function_defaults_routes_users_to_corresponding_function_page(qapp):
    widget = FunctionDefaultsWidget()
    requested_pages: list[str] = []
    widget.navigate_requested.connect(requested_pages.append)
    try:
        widget.tabs.setCurrentIndex(2)
        assert widget.open_function_page_button.text() == "打开PIE 拟合页"

        widget.open_function_page_button.click()

        assert requested_pages == ["pie"]
    finally:
        widget.deleteLater()


def test_temporary_settings_dialog_scopes_function_pages_for_pie(qapp):
    dialog = TemporaryAnalysisSettingsDialog(
        ProjectSettings(),
        initial_tab="function",
        visible_function_pages=("spectrum", "pie"),
        initial_function_page="pie",
    )
    try:
        assert dialog.tabs.currentWidget() is dialog.function_widget
        assert [
            dialog.function_widget.tabs.tabText(index)
            for index in range(dialog.function_widget.tabs.count())
        ] == ["寻峰与积分", "PIE 分析"]
        assert dialog.function_widget.tabs.currentIndex() == 1

        dialog.function_widget.tabs.setCurrentIndex(1)
        assert dialog.function_widget.open_function_page_button.text() == "打开PIE 拟合页"
    finally:
        dialog.deleteLater()


def test_project_parameter_pages_do_not_show_internal_ownership_banners(qapp):
    window = MainWindow()
    widget = FunctionDefaultsWidget()
    try:
        assert window.project_page.findChild(QtWidgets.QFrame, "ProjectBoundaryBanner") is None
        assert widget.findChild(QtWidgets.QFrame, "ProjectBoundaryBanner") is None
        assert "参数管理原则" not in " ".join(label.text() for label in window.project_page.findChildren(QtWidgets.QLabel))
    finally:
        widget.deleteLater()
        window.deleteLater()


def test_spectrum_source_path_edit_has_bottom_painting_clearance(qapp):
    window = MainWindow()
    try:
        window.resize(1600, 900)
        window.show()
        qapp.processEvents()

        for mode_button, path_edit in (
            (window.singleModeButton, window.lineEdit),
            (window.sumModeButton, window.folder_path),
        ):
            mode_button.click()
            qapp.processEvents()
            assert window.sourceStack.height() >= 34
            assert path_edit.height() <= 30
            assert path_edit.geometry().bottom() < window.sourceStack.contentsRect().bottom()
    finally:
        window.close()
        window.deleteLater()


def test_project_page_omits_workflow_progress_card(qapp):
    window = MainWindow()
    try:
        assert not hasattr(window, "project_workflow_card")
        assert not hasattr(window, "project_stage_buttons")
        assert not hasattr(window, "project_continue_button")
        assert [
            window.project_tabs.tabText(index)
            for index in range(window.project_tabs.count())
        ] == ["项目与数据", "共享参数", "分析默认值"]
        labels = " ".join(label.text() for label in window.project_identity_page.findChildren(QtWidgets.QLabel))
        assert "项目工作流" not in labels
        assert "建议下一步" not in labels
    finally:
        window.deleteLater()


def test_pie_project_parameter_menu_opens_pie_analysis_defaults(qapp):
    window = MainWindow()
    try:
        window.pie_page.set_project_settings(
            ProjectSettings(project_name="UI test", output_dir="output"),
            activate_project_scope=True,
        )
        window.switch_workspace_page("pie")

        assert window.pie_page.common_params_button.text() == "项目参数"
        assert window.pie_page.common_params_action.text() == "分析默认值…"
        assert window.pie_page.edit_project_action.text() == "项目与数据…"

        window.pie_page.common_params_action.trigger()

        assert window.workspace_stack.currentWidget() is window.project_page
        assert window.project_tabs.currentWidget() is window.project_function_defaults_widget
        assert window.project_function_defaults_widget.tabs.tabText(
            window.project_function_defaults_widget.tabs.currentIndex()
        ) == "PIE 分析"
    finally:
        window.deleteLater()


def test_temperature_project_parameter_menu_opens_temperature_defaults(qapp):
    window = MainWindow()
    try:
        window.temperature_page.set_project_settings(
            ProjectSettings(project_name="UI test", output_dir="output"),
            activate_project_scope=True,
        )
        window.switch_workspace_page("temperature")

        assert window.temperature_page.common_params_button.text() == "项目参数"
        assert window.temperature_page.common_params_action.text() == "分析默认值…"
        assert window.temperature_page.edit_project_action.text() == "项目与数据…"

        window.temperature_page.common_params_action.trigger()

        assert window.workspace_stack.currentWidget() is window.project_page
        assert window.project_tabs.currentWidget() is window.project_function_defaults_widget
        assert window.project_function_defaults_widget.tabs.tabText(
            window.project_function_defaults_widget.tabs.currentIndex()
        ) == "温度扫描"
    finally:
        window.deleteLater()


def test_temperature_project_parameters_fall_back_outside_workspace(qapp, monkeypatch):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature import dialog as temperature_dialog_module
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    opened = []

    class FakeCommonParametersDialog:
        def __init__(self, settings, calibration, parent):
            opened.append((settings, calibration, parent))

        def exec(self):
            return 0

    reloaded_calibration = Calibration(a=1.0, b=2.0, c=3.0)
    monkeypatch.setattr(
        temperature_dialog_module,
        "CommonParametersDialog",
        FakeCommonParametersDialog,
    )
    monkeypatch.setattr(
        temperature_dialog_module,
        "load_calibration_config",
        lambda: reloaded_calibration,
    )

    widget = TemperatureScanDialog(Calibration())
    try:
        widget.set_project_settings(
            ProjectSettings(project_name="Standalone", output_dir="output"),
            activate_project_scope=True,
        )
        widget.open_common_parameters()

        assert len(opened) == 1
        assert opened[0][2] is widget
        assert widget.calibration == reloaded_calibration
    finally:
        widget.deleteLater()


def test_refresh_project_lifecycle_is_compatible_without_workflow_ui(qapp, tmp_path):
    window = MainWindow()
    result_file = tmp_path / "temperature_result.csv"
    result_file.write_text("mz,temperature,signal\n29,650,1\n", encoding="utf-8")
    try:
        ps = ProjectSettings(
            project_name="Legacy Workflow",
            system="Test",
            output_dir=str(tmp_path / "Legacy_Workflow"),
            temperature_scan_result_file=str(result_file),
        )
        window.project_settings_manager.set(ps)
        window.refresh_project_lifecycle(ps)

        assert not hasattr(window, "project_stage_buttons")
        assert not hasattr(window, "project_continue_button")
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_common_parameters_apply_to_project_settings(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersWidget

    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
        ps = ProjectSettings()
        widget.set_project_settings(ps)

        light_idx = widget.light_source_combo.findData("beam_current")
        preset_idx = widget.mf_md_preset_combo.findData("30 Torr (Catalysis)")
        assert light_idx >= 0
        assert preset_idx >= 0
        assert not hasattr(widget, "pie_photon_mode_combo")
        assert not hasattr(widget, "temperature_photon_check")
        assert not hasattr(widget, "temperature_kr_check")

        widget.light_source_combo.setCurrentIndex(light_idx)
        widget.settings.expansion_factors = {400.0: 1.0, 800.0: 1.2}
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
        assert ps.pie_photon_mode == "none"
        assert ps.temperature_photon_normalize is True
        assert ps.temperature_kr_correct is False
        assert ps.mass_discrimination == pytest.approx(1.0)
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


def test_temperature_page_owns_analysis_switches(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    scan_dir = tmp_path / "temperature_scan"
    scan_dir.mkdir()
    (scan_dir / "C24110904-0000.txt").write_text("1 2\n", encoding="utf-8")
    settings = NormalizationSettings(light_source="beam_current", expansion_factors={})
    widget = TemperatureScanDialog(Calibration(), settings)
    try:
        assert widget._sidebar.isHidden()
        assert widget.curve_stats.isHidden()
        assert widget.preview_data_button.isHidden()
        assert widget.export_button.isHidden()
        assert widget.export_plot_button.isHidden()
        ps = ProjectSettings(
            temperature_scan_folder=str(scan_dir),
            temperature_photon_normalize=False,
            temperature_kr_correct=True,
            temp_replicate_mode="sum",
            light_source="beam_current",
        )
        widget.set_project_settings(ps, activate_project_scope=True)
        assert widget.temperature_photon_check.isChecked() is False
        assert widget.temperature_kr_check.isChecked() is True
        assert widget.replicate_enabled_check.isChecked() is True
        assert widget.replicate_mode_combo.currentData() == "sum"
        assert "Beam Current" in widget.light_source_status_label.text()

        widget.run_analysis()

        assert ps.temperature_photon_normalize is False
        assert ps.temperature_kr_correct is False
        assert ps.temp_replicate_mode == "sum"
        assert "退回" in widget.inline_status_text.text()
    finally:
        if widget.worker is not None and widget.worker.isRunning():
            widget.worker.wait(1000)
        widget.deleteLater()


def test_temperature_project_scope_uses_project_calibration(qapp, tmp_path, monkeypatch):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature import dialog as temperature_dialog_module
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    project_calibration = Calibration(a=8.21e-7, b=4.56e-4, c=0.789)
    ps = ProjectSettings(
        project_name="Calibrated Temperature",
        output_dir=str(tmp_path / "project"),
        temperature_scan_folder=str(tmp_path / "temperature_scan"),
        cal_a=project_calibration.a,
        cal_b=project_calibration.b,
        cal_c=project_calibration.c,
        light_source="beam_current",
        temperature_kr_correct=True,
        expansion_factors={500.0: 1.25},
    )
    widget = TemperatureScanDialog(Calibration(a=0.0, b=1.0, c=0.0))
    captured = {}

    def fake_analyze_temperature_folder(_folder, **kwargs):
        captured["calibration"] = kwargs["calibration"]
        return pd.DataFrame()

    monkeypatch.setattr(
        temperature_dialog_module,
        "analyze_temperature_folder",
        fake_analyze_temperature_folder,
    )
    try:
        widget.set_project_settings(ps, activate_project_scope=True)
        params = widget._analysis_parameters()
        widget._analyze_temperature_folder_with_params(str(tmp_path), params)

        assert widget.calibration == project_calibration
        assert widget.normalization_settings.light_source == "beam_current"
        assert widget.normalization_settings.expansion_factors == {500.0: 1.25}
        assert captured["calibration"] == project_calibration
        assert widget._analysis_cache_parameters(params)["calibration"] == {
            "a": project_calibration.a,
            "b": project_calibration.b,
            "c": project_calibration.c,
        }
        assert params["cache_dir"] == tmp_path / "project" / "analysis" / "temperature_scan" / "cache"
    finally:
        widget.deleteLater()


def test_temperature_ignores_analysis_result_from_previous_project(qapp, monkeypatch):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    widget = TemperatureScanDialog(Calibration())
    applied = []
    monkeypatch.setattr(widget, "on_analysis_complete", lambda result: applied.append(result))
    try:
        widget._analysis_request_id = 7

        widget._on_analysis_result(6, {"stale": True})
        widget._on_analysis_result(7, {"current": True})

        assert applied == [{"current": True}]
    finally:
        widget.deleteLater()


def test_temperature_parameters_follow_selected_data_source(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    project_calibration = Calibration(a=1.25e-6, b=0.0025, c=-0.75)
    (tmp_path / "project_data").mkdir()
    ps = ProjectSettings(
        project_name="Temporary parameter source",
        output_dir=str(tmp_path / "project"),
        temperature_scan_folder=str(tmp_path / "project_data"),
        cal_a=project_calibration.a,
        cal_b=project_calibration.b,
        cal_c=project_calibration.c,
        temp_integration_method="baseline",
        min_intensity=12.0,
    )
    widget = TemperatureScanDialog(Calibration(a=0.0, b=1.0, c=0.0))
    try:
        widget.set_project_settings(ps, activate_project_scope=True)

        project_params = widget._analysis_parameters()
        assert widget.temperature_source_scope == "project"
        assert project_params["calibration"] == project_calibration
        assert project_params["integration_method"] == "baseline"
        assert project_params["min_intensity"] == 12.0

        widget.set_temperature_source_scope("temporary")
        assert widget.temporary_settings_source == "default"
        assert widget.temporary_params_title_label.text() == "临时数据参数"
        assert widget.temporary_params_label.text() == "默认参数，可修改"
        assert widget.temporary_params_button.text() == "编辑参数…"
        assert not hasattr(widget, "temporary_params_reload_button")
        assert not hasattr(widget, "copy_project_params_action")
        assert widget._analysis_parameters()["integration_method"] == ProjectSettings().temp_integration_method

        params = widget._analysis_parameters()
        assert widget.temporary_params_hint.text() == "只影响本次临时分析，不会写回项目"
        assert params["calibration"] != project_calibration
        assert params["integration_method"] == ProjectSettings().temp_integration_method
        assert params["min_intensity"] == ProjectSettings().min_intensity
        assert params["cache_dir"] != tmp_path / "project" / "analysis" / "temperature_scan" / "cache"

        widget.temporary_settings.cal_a = 9.0
        widget.temporary_settings.temp_integration_method = "gaussian"
        assert ps.cal_a == project_calibration.a
        assert ps.temp_integration_method == "baseline"

        widget.temporary_settings_modified = True
        widget._refresh_temperature_source_controls()
        assert widget.temporary_settings_source == "default"
        assert widget.temporary_params_label.text() == "本次已修改"
    finally:
        widget.deleteLater()


def test_temperature_page_lists_energy_subfolders_for_project_source(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    root = tmp_path / "temperature_scan"
    low = root / "8eV"
    high = root / "9.5eV"
    range_energy = root / "14.6-14.8eV"
    low.mkdir(parents=True)
    high.mkdir(parents=True)
    range_energy.mkdir(parents=True)
    (low / "C24110901-0000.txt").write_text("1\n2\n", encoding="utf-8")
    (high / "C24110902-0000.txt").write_text("1\n2\n", encoding="utf-8")
    (range_energy / "C24110903-0000.txt").write_text("1\n2\n", encoding="utf-8")

    widget = TemperatureScanDialog(Calibration())
    try:
        ps = ProjectSettings(
            project_name="Energy Source",
            output_dir=str(tmp_path / "project"),
            temperature_scan_folder=str(root),
        )
        widget.set_project_settings(ps, activate_project_scope=True)

        assert widget.temperature_source_scope == "project"
        assert widget.scan_folder_combo.count() == 4
        assert [widget.scan_folder_combo.itemText(i) for i in range(widget.scan_folder_combo.count())] == [
            "全部能量 (3)",
            "8eV",
            "9.5eV",
            "14.6-14.8eV",
        ]
        assert widget._current_temperature_folder() == str(root)
        assert widget._selected_analysis_folders() == [
            (8.0, str(low)),
            (9.5, str(high)),
            (14.6, str(range_energy)),
        ]
        assert widget.run_button.isEnabled()
        assert not hasattr(widget, "workflow_stage_label")
        assert not hasattr(widget, "interval_mz_spin")
        assert not hasattr(widget, "energy_interval_button")

        widget.scan_folder_combo.setCurrentIndex(2)
        assert widget._current_temperature_folder() == str(high)
        assert "9.5eV" in widget.summary_data_label.text()
    finally:
        widget.deleteLater()


def test_temperature_curve_detail_table_shows_calculation_row_labels(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    widget = TemperatureScanDialog(Calibration())
    try:
        df = pd.DataFrame(
            {
                650.0: ["范围累加", 26.0, 0.3428, 1.0, 0.3428],
                750.0: ["范围累加", 26.0, 0.3467, 1.036516, 0.3345],
            },
            index=["积分方式", "原始积分", "IO归一化", "λ(T)", "最终强度"],
        )
        widget._set_curve_detail_table(df)

        labels = [
            widget.curve_table.verticalHeaderItem(row).text()
            for row in range(widget.curve_table.rowCount())
        ]
        assert labels == ["积分方式", "原始积分", "IO归一化", "λ(T)", "最终强度"]
        assert not widget.curve_table.verticalHeader().isHidden()
        assert widget.curve_table.item(4, 0).font().bold()
    finally:
        widget.deleteLater()


def test_temperature_energy_interval_table_marks_product_like_rows(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.temperature_scan import identify_product_energy_intervals
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    widget = TemperatureScanDialog(Calibration())
    try:
        result = identify_product_energy_intervals(
            [
                {
                    "energy": 8.0,
                    "folder": "8eV",
                    "mz": 70,
                    "curve_class": "unclassified",
                    "curve_class_label": "暂未区分",
                    "curve_class_reason": "趋势不满足规则",
                    "points": 10,
                    "max_signal": 0.2,
                    "temperature_range": "650-1000 °C",
                },
                {
                    "energy": 8.7,
                    "folder": "8.7eV",
                    "mz": 70,
                    "curve_class": "formation",
                    "curve_class_label": "生成(升高)",
                    "curve_class_reason": "高温端信号明显高于低温端",
                    "points": 10,
                    "max_signal": 12.5,
                    "temperature_range": "650-1000 °C",
                },
            ]
        )
        widget._populate_energy_interval_table(result)

        assert widget.energy_interval_table.rowCount() == 2
        assert widget.energy_interval_table.item(0, 2).text() == "否"
        assert widget.energy_interval_table.item(1, 2).text() == "是"
        assert widget.energy_interval_table.item(1, 2).font().bold()
    finally:
        widget.deleteLater()


def test_temperature_selected_mz_updates_multi_energy_plot_and_interval_card(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    widget = TemperatureScanDialog(Calibration())
    try:
        def curve(mz: int, curve_class: str, label: str, areas: list[float]) -> dict:
            rows = pd.DataFrame(
                {
                    "temperature": [650.0, 750.0, 850.0],
                    "file": ["a.txt", "b.txt", "c.txt"],
                    "mz": [float(mz)] * 3,
                    "area": areas,
                    "raw_area": areas,
                    "photon_normalized_area": areas,
                    "expansion_lambda": [1.0, 1.0, 1.0],
                    "integration_method": ["sum_counts"] * 3,
                    "species": [""] * 3,
                }
            )
            return {
                "mz": mz,
                "species": "",
                "temperatures": [650.0, 750.0, 850.0],
                "areas": areas,
                "curve_class": curve_class,
                "curve_class_label": label,
                "curve_class_reason": "测试",
                "rows": rows,
            }

        widget.energy_results = [
            {"energy": 8.0, "folder": "/tmp/8eV", "folder_label": "8eV", "curves": {70: curve(70, "formation", "生成(升高)", [1, 2, 3])}},
            {"energy": 8.7, "folder": "/tmp/8.7eV", "folder_label": "8.7eV", "curves": {70: curve(70, "intermediate", "中间体(先升后降低)", [1, 3, 1])}},
            {"energy": 9.0, "folder": "/tmp/9eV", "folder_label": "9eV", "curves": {70: curve(70, "unclassified", "暂未区分", [1, 1, 1])}},
        ]
        widget.result_df = pd.concat(
            [item["curves"][70]["rows"] for item in widget.energy_results],
            ignore_index=True,
        )
        widget.curves = widget._build_display_curves()
        widget.populate_mz_list()

        assert widget.current_mz == 70
        assert not widget._sidebar.isHidden()
        assert not widget.curve_stats.isHidden()
        assert not widget.sidebar_toggle_btn.isHidden()
        assert not widget.export_button.isHidden()
        assert widget.export_plot_button.isHidden()
        assert widget.export_result_action.isEnabled()
        assert widget.export_plot_action.isEnabled()
        assert "8.00-8.70 eV" in widget.current_interval_label.text()
        assert widget.energy_interval_table.rowCount() == 3
        assert widget.energy_interval_table.item(0, 2).text() == "是"
        assert widget.energy_interval_table.item(2, 2).text() == "否"
        assert "各能量" in widget.plot_widget.axes.get_title()
        assert [line.get_color() for line in widget.plot_widget.axes.lines[:3]] == [
            "#0C5DA5",
            "#00B945",
            "#FF9500",
        ]
        assert [line.get_linestyle() for line in widget.plot_widget.axes.lines[:3]] == [
            "-",
            "--",
            "-.",
        ]
    finally:
        widget.deleteLater()


def test_temperature_energy_combo_filters_cached_curves_without_rerun(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    root = tmp_path / "temperature_scan"
    low = root / "8eV"
    high = root / "9eV"
    low.mkdir(parents=True)
    high.mkdir(parents=True)
    (low / "a.txt").write_text("1\n2\n", encoding="utf-8")
    (high / "b.txt").write_text("1\n2\n", encoding="utf-8")

    def curve(mz: int, energy: float, areas: list[float]) -> dict:
        rows = pd.DataFrame(
            {
                "temperature": [650.0, 750.0, 850.0],
                "file": [f"{energy}-a.txt", f"{energy}-b.txt", f"{energy}-c.txt"],
                "mz": [float(mz)] * 3,
                "area": areas,
                "raw_area": areas,
                "photon_normalized_area": areas,
                "expansion_lambda": [1.0, 1.0, 1.0],
                "integration_method": ["sum_counts"] * 3,
                "species": [""] * 3,
            }
        )
        return {
            "mz": mz,
            "species": "",
            "temperatures": [650.0, 750.0, 850.0],
            "areas": areas,
            "curve_class": "formation",
            "curve_class_label": "生成(升高)",
            "curve_class_reason": "测试",
            "rows": rows,
        }

    widget = TemperatureScanDialog(Calibration())
    try:
        ps = ProjectSettings(
            project_name="Energy Switch",
            output_dir=str(tmp_path / "project"),
            temperature_scan_folder=str(root),
        )
        widget.set_project_settings(ps, activate_project_scope=True)
        widget.energy_results = [
            {"energy": 8.0, "folder": str(low), "folder_label": "8eV", "curves": {70: curve(70, 8.0, [1.0, 2.0, 3.0])}},
            {"energy": 9.0, "folder": str(high), "folder_label": "9eV", "curves": {70: curve(70, 9.0, [10.0, 20.0, 30.0])}},
        ]
        widget.result_df = pd.concat([item["curves"][70]["rows"] for item in widget.energy_results], ignore_index=True)
        widget._apply_energy_view_selection(preferred_mz=70)

        assert widget.current_mz == 70
        assert "各能量" in widget.plot_widget.axes.get_title()
        assert widget.curve_table.rowCount() == 6

        widget.scan_folder_combo.setCurrentIndex(widget.scan_folder_combo.findData(str(low)))
        qapp.processEvents()

        assert widget.current_mz == 70
        assert widget.plot_widget.axes.get_title() == "m/z 70 温度响应曲线"
        assert widget.curve_table.columnCount() == 3
        assert widget.curve_table.horizontalHeaderItem(0).text() == "650.0"
        assert "8.00-9.00 eV" in widget.current_interval_label.text()

        widget.scan_folder_combo.setCurrentIndex(widget.scan_folder_combo.findData(widget.ALL_ENERGY_FOLDERS))
        qapp.processEvents()

        assert "各能量" in widget.plot_widget.axes.get_title()
        assert widget.curve_table.rowCount() == 6
    finally:
        widget.deleteLater()


def test_temperature_analysis_cache_reuses_integrated_results(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    root = tmp_path / "temperature_scan"
    low = root / "8eV"
    low.mkdir(parents=True)
    (low / "a.txt").write_text("1\n2\n", encoding="utf-8")

    widget = TemperatureScanDialog(Calibration())
    try:
        ps = ProjectSettings(
            project_name="Cache Project",
            output_dir=str(tmp_path / "project"),
            temperature_scan_folder=str(root),
            temperature_photon_normalize=False,
            temperature_kr_correct=False,
        )
        widget.set_project_settings(ps, activate_project_scope=True)
        folders = [(8.0, str(low))]
        params = widget._analysis_parameters()
        calls = {"count": 0}

        def fake_analyze(folder, params):
            calls["count"] += 1
            return pd.DataFrame(
                {
                    "temperature": [650.0, 750.0, 850.0],
                    "file": ["a.txt", "b.txt", "c.txt"],
                    "mz": [70.0, 70.0, 70.0],
                    "area": [1.0, 2.0, 3.0],
                    "raw_area": [1.0, 2.0, 3.0],
                    "photon_normalized_area": [1.0, 2.0, 3.0],
                    "expansion_lambda": [1.0, 1.0, 1.0],
                    "integration_method": ["sum_counts"] * 3,
                    "species": [""] * 3,
                    "curve_class": ["formation"] * 3,
                    "curve_class_label": ["生成(升高)"] * 3,
                    "curve_class_reason": ["测试"] * 3,
                    "reference_temperature": [850.0, 850.0, 850.0],
                }
            )

        widget._analyze_temperature_folder_with_params = fake_analyze
        first = widget._analyze_temperature_folders_with_cache(folders, params)
        assert first["from_cache"] is False
        assert calls["count"] == 1
        assert (Path(ps.output_dir) / "analysis" / "temperature_scan" / "cache").is_dir()

        def fail_analyze(folder, params):
            raise AssertionError("cache miss caused recomputation")

        widget._analyze_temperature_folder_with_params = fail_analyze
        second = widget._analyze_temperature_folders_with_cache(folders, params)

        assert second["from_cache"] is True
        assert second["result_df"].shape == first["result_df"].shape
        assert second["energy_results"][0]["folder_label"] == "8eV"
        assert 70 in second["energy_results"][0]["curves"]
    finally:
        widget.deleteLater()


def test_temperature_multi_folder_analysis_reuses_curves_and_preserves_order(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.temperature_scan import build_temperature_curves
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    folders = []
    for energy in (8.0, 9.0, 10.0):
        folder = tmp_path / f"{energy:.1f}eV"
        folder.mkdir()
        folders.append((energy, str(folder)))

    widget = TemperatureScanDialog(Calibration())
    try:
        analysis_order = []

        def fake_analyze(folder, _params):
            energy = float(Path(folder).name.removesuffix("eV"))
            analysis_order.append(energy)
            result = pd.DataFrame(
                {
                    "temperature": [650.0, 750.0],
                    "file": ["a.txt", "b.txt"],
                    "mz": [70.0, 70.0],
                    "area": [energy, energy + 1.0],
                    "curve_class": ["formation", "formation"],
                    "curve_class_label": ["生成(升高)", "生成(升高)"],
                    "curve_class_reason": ["测试", "测试"],
                    "reference_temperature": [750.0, 750.0],
                }
            )
            result.attrs["_bl03u_temperature_curves"] = build_temperature_curves(result)
            return result

        widget._analyze_temperature_folder_with_params = fake_analyze
        progress_updates = []
        output = widget._analyze_temperature_folders_with_params(
            folders,
            {},
            progress_callback=lambda value, message: progress_updates.append((value, message)),
        )

        assert analysis_order == [8.0, 9.0, 10.0]
        assert progress_updates[0] == (10, "正在处理 8.0eV（1/3）")
        assert progress_updates[-1] == (86, "正在汇总不同能量的温度曲线…")
        assert [value for value, _message in progress_updates] == sorted(
            value for value, _message in progress_updates
        )
        assert [item["energy"] for item in output["energy_results"]] == [8.0, 9.0, 10.0]
        assert [item["folder_label"] for item in output["energy_results"]] == [
            "8.0eV",
            "9.0eV",
            "10.0eV",
        ]
        for item in output["energy_results"]:
            curve_rows = item["curves"][70]["rows"]
            assert curve_rows["scan_folder"].nunique() == 1
            assert curve_rows["scan_energy"].iloc[0] == item["energy"]
    finally:
        widget.deleteLater()


def test_temperature_project_open_autoloads_cached_curves(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    root = tmp_path / "temperature_scan"
    low = root / "8eV"
    low.mkdir(parents=True)
    (low / "a.txt").write_text("1\n2\n", encoding="utf-8")
    ps = ProjectSettings(
        project_name="Autoload Cache",
        output_dir=str(tmp_path / "project"),
        temperature_scan_folder=str(root),
        temperature_photon_normalize=False,
        temperature_kr_correct=False,
    )
    folders = [(8.0, str(low))]

    writer = TemperatureScanDialog(Calibration())
    try:
        writer.set_project_settings(ps, activate_project_scope=True)
        params = writer._analysis_parameters()

        def fake_analyze(folder, params):
            return pd.DataFrame(
                {
                    "temperature": [650.0, 750.0, 850.0],
                    "file": ["a.txt", "b.txt", "c.txt"],
                    "mz": [70.0, 70.0, 70.0],
                    "area": [1.0, 2.0, 3.0],
                    "raw_area": [1.0, 2.0, 3.0],
                    "photon_normalized_area": [1.0, 2.0, 3.0],
                    "expansion_lambda": [1.0, 1.0, 1.0],
                    "integration_method": ["sum_counts"] * 3,
                    "species": [""] * 3,
                    "curve_class": ["formation"] * 3,
                    "curve_class_label": ["生成(升高)"] * 3,
                    "curve_class_reason": ["测试"] * 3,
                    "reference_temperature": [850.0, 850.0, 850.0],
                }
            )

        writer._analyze_temperature_folder_with_params = fake_analyze
        writer._analyze_temperature_folders_with_cache(folders, params)
    finally:
        writer.deleteLater()

    reader = TemperatureScanDialog(Calibration())
    try:
        reader.set_project_settings(ps, activate_project_scope=True)
        deadline = QtCore.QDeadlineTimer(3000)
        while reader._autoload_worker is not None and reader._autoload_worker.isRunning() and not deadline.hasExpired():
            qapp.processEvents()
            QtCore.QThread.msleep(10)
        qapp.processEvents()

        assert not reader.result_df.empty
        assert 70 in reader.curves
        assert reader.current_mz == 70
        assert "已自动载入" in reader.inline_status_text.text() or "缓存" in reader.inline_status_text.text()
    finally:
        if reader._autoload_worker is not None and reader._autoload_worker.isRunning():
            reader._autoload_worker.wait(1000)
        reader.deleteLater()


def test_temperature_project_cache_autoload_ignores_stale_project_result(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    old_root = tmp_path / "old_temperature"
    new_root = tmp_path / "new_temperature"
    old_energy = old_root / "8eV"
    new_energy = new_root / "9eV"
    old_energy.mkdir(parents=True)
    new_energy.mkdir(parents=True)
    (old_energy / "old.txt").write_text("1\n2\n", encoding="utf-8")
    (new_energy / "new.txt").write_text("1\n2\n", encoding="utf-8")

    old_ps = ProjectSettings(
        project_name="Old Project",
        output_dir=str(tmp_path / "old_project"),
        temperature_scan_folder=str(old_root),
    )
    new_ps = ProjectSettings(
        project_name="New Project",
        output_dir=str(tmp_path / "new_project"),
        temperature_scan_folder=str(new_root),
    )

    widget = TemperatureScanDialog(Calibration())
    try:
        widget.set_project_settings(old_ps, activate_project_scope=True)
        old_folders = widget._generation_analysis_folders()
        old_token = widget._project_temperature_cache_token(old_folders, "old-cache")

        widget.set_project_settings(new_ps, activate_project_scope=True)
        stale_result = {
            "result_df": pd.DataFrame(
                {
                    "temperature": [650.0],
                    "mz": [70.0],
                    "area": [1.0],
                    "raw_area": [1.0],
                    "photon_normalized_area": [1.0],
                    "expansion_lambda": [1.0],
                    "integration_method": ["sum_counts"],
                }
            ),
            "energy_results": [],
        }
        widget._on_project_temperature_cache_loaded(stale_result, old_token)

        assert widget.result_df.empty
        assert widget.curves == {}
        assert "New Project" in widget.summary_project_label.text()
    finally:
        widget.deleteLater()


def test_pie_page_owns_photon_mode_switch(qapp, tmp_path, monkeypatch):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog

    pie_dir = tmp_path / "pie_scan"
    pie_dir.mkdir()
    (pie_dir / "8.0eV.txt").write_text("1 2\n", encoding="utf-8")

    widget = PIESpeciesFitDialog(Calibration(), NormalizationSettings(light_source="beam_current"))
    try:
        ps = ProjectSettings(
            project_name="PIE Switch",
            output_dir=str(tmp_path / "project"),
            pie_scan_folder=str(pie_dir),
            pie_photon_mode="first",
        )
        manager = ProjectSettingsManager()
        manager.set_project_path(tmp_path / "project")
        manager.set(ps)
        widget.set_project_settings(ps, activate_project_scope=True)

        assert widget.photon_correction_check.isChecked() is True

        monkeypatch.setattr(widget, "set_busy", lambda *args, **kwargs: None)

        class _SignalStub:
            def connect(self, *args, **kwargs):
                pass

        class DummyWorker:
            def __init__(self, *args, **kwargs):
                self.finished_with_result = _SignalStub()
                self.failed = _SignalStub()
                self.finished = _SignalStub()

            def start(self):
                pass

        monkeypatch.setattr(
            "bl03u_masstool.frontends.pyqt_app.pie.dialog.WorkerThread",
            DummyWorker,
        )

        widget.run_analysis()

        saved = load_project_settings(tmp_path / "project" / "config" / "project.yaml")
        assert ps.pie_photon_mode == "none"
        assert widget.normalization_settings.pie_photon_mode == "none"
        assert saved.pie_photon_mode == "none"

        widget.photon_correction_check.setChecked(False)
        widget.run_analysis()

        saved = load_project_settings(tmp_path / "project" / "config" / "project.yaml")
        assert ps.pie_photon_mode == "off"
        assert widget.normalization_settings.pie_photon_mode == "off"
        assert saved.pie_photon_mode == "off"
    finally:
        ProjectSettingsManager().clear_project_path()
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
        _wait_for_project_materialization(qapp, window)

        project_dir = downloads_dir / "test"
        assert project_dir.exists()
        assert (project_dir / "config" / "project.yaml").exists()
        assert window.project_output_dir_edit.text() == str(project_dir)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_project_storage_field_explains_parent_and_root_semantics(qapp):
    window = MainWindow()
    try:
        placeholder = window.project_output_dir_edit.placeholderText()

        assert "父目录" in placeholder
        assert "项目根目录" in placeholder
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_empty_state_placeholders_do_not_use_specific_system_examples(qapp):
    window = MainWindow()
    try:
        placeholders = [
            window.project_name_edit.placeholderText(),
            window.project_system_edit.placeholderText(),
            window.ionization_page.query_edit.placeholderText(),
            window.isotope_page.formula_edit.placeholderText(),
            window.pics_page.txt_new_name.placeholderText(),
            window.pics_page.txt_new_formula.placeholderText(),
        ]
        disallowed = ["C6F11O2H", "C6H6", "C6H5ClO", "Benzene", "SVUV-PIMS", "methane", "methyl"]

        for placeholder in placeholders:
            for token in disallowed:
                assert token not in placeholder
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_ie_lookup_hides_unavailable_prediction_model_ui(qapp):
    window = MainWindow()
    try:
        group_titles = [group.title() for group in window.ionization_page.findChildren(QtWidgets.QGroupBox)]

        assert "IE预测模型输出" not in group_titles
        assert not hasattr(window.ionization_page, "model_prediction_table")
        assert "查询结果" in group_titles
        assert "IE测定记录" in group_titles
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_new_project_uses_parent_directory_even_if_parent_contains_old_project_files(qapp, tmp_path, monkeypatch):
    downloads_dir = tmp_path / "Downloads"
    (downloads_dir / "config").mkdir(parents=True)
    (downloads_dir / "config" / "project.yaml").write_text("project:\n  name: old\n", encoding="utf-8")
    (downloads_dir / "analysis").mkdir()

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    try:
        window.new_project()
        window.project_name_edit.setText("hhh")
        window.project_system_edit.setText("hhh")
        window.project_description_edit.setText("hhh")
        window.project_output_dir_edit.setText(str(downloads_dir))

        window.save_and_apply_project_settings()
        _wait_for_project_materialization(qapp, window)

        project_dir = downloads_dir / "hhh"
        assert project_dir.exists()
        assert (project_dir / "analysis").exists()
        assert (project_dir / "config" / "project.yaml").exists()
        assert window.project_output_dir_edit.text() == str(project_dir)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_save_new_project_ignores_stale_opened_parent_project_root(qapp, tmp_path, monkeypatch):
    downloads_dir = tmp_path / "Downloads"
    (downloads_dir / "config").mkdir(parents=True)
    save_project_settings(
        ProjectSettings(project_name="old", system="old", output_dir=str(downloads_dir)),
        downloads_dir / "config" / "project.yaml",
    )
    (downloads_dir / ".bl03u_project").write_text("old\n", encoding="utf-8")

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: pytest.fail(str(args)))
    try:
        # Simulate the real failure mode: a previous bad save made Downloads the
        # active project root, then the user creates project "aaa" using Downloads
        # as the storage location.
        window.project_settings_manager.set_project_path(downloads_dir)
        window.new_project()
        window.project_name_edit.setText("aaa")
        window.project_system_edit.setText("aaa")
        window.project_description_edit.setText("aaa")
        window.project_output_dir_edit.setText(str(downloads_dir))

        window.save_and_apply_project_settings()
        _wait_for_project_materialization(qapp, window)

        project_dir = downloads_dir / "aaa"
        assert (project_dir / "config" / "project.yaml").exists()
        assert (project_dir / "analysis").exists()
        assert window.project_output_dir_edit.text() == str(project_dir)

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.output_dir == str(project_dir)
        assert window.project_settings_manager.get_project_config_path() == project_dir / "config" / "project.yaml"
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_save_project_name_into_active_stale_parent_creates_child_project(qapp, tmp_path, monkeypatch):
    downloads_dir = tmp_path / "Downloads"
    (downloads_dir / "config").mkdir(parents=True)
    save_project_settings(
        ProjectSettings(project_name="old", system="old", output_dir=str(downloads_dir)),
        downloads_dir / "config" / "project.yaml",
    )

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: pytest.fail(str(args)))
    try:
        window.project_settings_manager.set_project_path(downloads_dir)
        window.project_name_edit.setText("aaa")
        window.project_system_edit.setText("aaa")
        window.project_description_edit.setText("aaa")
        window.project_output_dir_edit.setText(str(downloads_dir))

        window.save_and_apply_project_settings()
        _wait_for_project_materialization(qapp, window)

        project_dir = downloads_dir / "aaa"
        assert (project_dir / "analysis").exists()
        assert (project_dir / "config" / "project.yaml").exists()
        assert window.project_output_dir_edit.text() == str(project_dir)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_save_and_apply_materializes_project_data_sources(qapp, tmp_path, monkeypatch):
    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir()
    temp_source = tmp_path / "external" / "温度扫描"
    pie_source = tmp_path / "external" / "PIE"
    kr_source = tmp_path / "external" / "Kr定标"
    kr_peak_source = tmp_path / "external" / "kr_peaks.csv"
    (temp_source / "8.0eV").mkdir(parents=True)
    pie_source.mkdir(parents=True)
    kr_source.mkdir(parents=True)
    (temp_source / "8.0eV" / "650K.txt").write_text("temp", encoding="utf-8")
    (pie_source / "8.0eV.txt").write_text("pie", encoding="utf-8")
    (kr_source / "Kr_650K.txt").write_text("kr", encoding="utf-8")
    kr_peak_source.write_text("mz,start,end\n84,1,2\n", encoding="utf-8")

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: pytest.fail(str(args)))
    try:
        window.project_name_edit.setText("managed")
        window.project_system_edit.setText("C6H6")
        window.project_output_dir_edit.setText(str(downloads_dir))
        window.project_temperature_folder_edit.setText(str(temp_source))
        window.project_pie_folder_edit.setText(str(pie_source))
        window.project_common_parameters_widget.kr_folder_edit.setText(str(kr_source))
        window.project_common_parameters_widget.kr_peak_mode_manual_radio.setChecked(True)
        window.project_common_parameters_widget.kr_peak_file_edit.setText(str(kr_peak_source))

        window.save_and_apply_project_settings()
        _wait_for_project_materialization(qapp, window)

        project_dir = downloads_dir / "managed"
        managed_temp = project_dir / "raw_data" / "temperature_scan" / "温度扫描"
        managed_pie = project_dir / "raw_data" / "pie_scan" / "PIE"
        managed_kr = project_dir / "raw_data" / "kr_calibration" / "Kr定标"
        managed_kr_peak = project_dir / "analysis" / "spectrum" / "kr_manual_peaks" / "kr_peaks.csv"
        saved = load_project_settings(project_dir / "config" / "project.yaml")

        assert Path(saved.temperature_scan_folder) == managed_temp
        assert Path(saved.pie_scan_folder) == managed_pie
        assert Path(saved.kr_calibration_folder) == managed_kr
        assert Path(saved.kr_calibration_peak_file) == managed_kr_peak
        assert window.project_temperature_folder_edit.text() == str(managed_temp)
        assert window.project_pie_folder_edit.text() == str(managed_pie)
        assert window.project_common_parameters_widget.kr_folder_edit.text() == str(managed_kr)
        assert window.project_common_parameters_widget.kr_peak_file_edit.text() == str(managed_kr_peak)
        assert not managed_temp.is_symlink()
        assert (managed_temp / "8.0eV" / "650K.txt").read_text(encoding="utf-8") == "temp"
        assert (managed_pie / "8.0eV.txt").read_text(encoding="utf-8") == "pie"
        assert (managed_kr / "Kr_650K.txt").read_text(encoding="utf-8") == "kr"
        assert managed_kr_peak.read_text(encoding="utf-8").startswith("mz,start,end")
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

        assert window.project_temperature_folder_edit.text() == str(temperature_folder)
        assert window.project_pie_folder_edit.text() == str(pie_folder)
        assert window.project_manual_peak_edit.text() == str(manual_peak)
        assert window.spectrum_source_scope == "project"
        assert window.projectSourceButton.text() == "项目数据"
        assert window.customSourceButton.text() == "临时数据"
        assert window.lineEdit.isReadOnly()
        assert window.lineEdit.text() == str(single_file)
        assert window.folder_path.text() == str(sum_folder)
        assert window.singleBrowseButton.isEnabled()
        assert window.project_common_parameters_widget.show_actions is False
        assert window.project_common_parameters_widget.action_bar.isHidden()
        assert window.current_calibration().a == pytest.approx(9.1e-7)
        assert window.normalization_settings.light_source == "beam_current"
        assert window.normalization_settings.pie_photon_mode == "off"
        assert window.normalization_settings.mass_discrimination == pytest.approx(1.0)
        assert not hasattr(window.project_common_parameters_widget, "mass_discrimination_edit")
        assert window.project_common_parameters_widget.mf_md_preset_combo.currentText() == "30 Torr (Catalysis)"
        assert window.project_common_parameters_widget.mf_mass_disc_exponent_edit.value() == pytest.approx(0.75148)
        assert window.temperature_page.project_settings.temperature_scan_folder == str(temperature_folder)
        assert window.temperature_page.temperature_source_scope == "project"
        assert window.temperature_page.project_source_button.text() == "项目数据"
        assert window.temperature_page.temporary_source_button.text() == "临时数据"
        assert window.temperature_page.select_folder_button.isHidden()
        assert window.temperature_page.normalization_settings.mass_discrimination == pytest.approx(1.0)
        assert window.temperature_page.calibration.a == pytest.approx(9.1e-7)
        assert window.pie_page.project_settings.pie_scan_folder == str(pie_folder)
        assert window.pie_page.pie_source_scope == "project"
        assert window.pie_page.project_source_button.text() == "项目数据"
        assert window.pie_page.temporary_source_button.text() == "临时数据"
        assert not window.pie_page.select_folder_button.isHidden()
        assert window.pie_page.select_folder_button.text() == "管理能段..."
        assert window.pie_page.temporary_segments_button.isHidden()
        assert window.pie_page.normalization_settings.mass_discrimination == pytest.approx(1.0)
        assert window.mole_fraction_page.project_settings.mf_md_preset == "30 Torr (Catalysis)"
        assert window.datasource_row_status_labels["temperature_scan"].text().startswith("✓")
        assert window.datasource_row_status_labels["pie_scan"].text().startswith("✓")
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_close_project_returns_related_tools_to_temporary_data_scope(qapp, tmp_path, monkeypatch):
    from bl03u_masstool.core.mole_fraction import MoleFractionSettings
    from bl03u_masstool.frontends.pyqt_app.mole_fraction import dialog as mole_fraction_module
    from bl03u_masstool.frontends.pyqt_app.pics import dialog as pics_module

    project_dir = tmp_path / "Opened_Project"
    single_file = project_dir / "raw_data" / "single.txt"
    sum_folder = project_dir / "raw_data" / "sum"
    temperature_folder = project_dir / "raw_data" / "temperature_scan"
    pie_folder = project_dir / "raw_data" / "pie_scan"
    for folder in (sum_folder, temperature_folder, pie_folder):
        folder.mkdir(parents=True)
    single_file.parent.mkdir(parents=True, exist_ok=True)
    single_file.write_text("tof intensity\n1 2\n", encoding="utf-8")
    ps = ProjectSettings(
        project_name="Opened Project",
        system="C6H6",
        output_dir=str(project_dir),
        single_spectrum_file=str(single_file),
        sum_spectrum_folder=str(sum_folder),
        temperature_scan_folder=str(temperature_folder),
        pie_scan_folder=str(pie_folder),
        mf_mass_disc_exponent=0.91,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    standalone_mf = MoleFractionSettings(mass_disc_exponent=0.42)
    monkeypatch.setattr(mole_fraction_module, "load_mole_fraction_settings", lambda: standalone_mf)
    monkeypatch.setattr(pics_module, "load_mole_fraction_settings", lambda: standalone_mf)

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(project_dir))
    try:
        window.open_project()

        assert window.project_settings_manager.has_project_path()
        assert window.spectrum_source_scope == "project"
        assert window.temperature_page.temperature_source_scope == "project"
        assert window.pie_page.pie_source_scope == "project"

        window.close_current_project()

        assert not window.project_settings_manager.has_project_path()
        assert window.project_name_edit.text() == ""
        assert window.project_system_edit.text() == ""
        assert window.project_status_label.text() == "当前项目: 未打开项目"
        assert not window.project_close_button.isEnabled()
        assert window.spectrum_source_scope == "custom"
        assert not window.lineEdit.isReadOnly()
        assert window.singleBrowseButton.isEnabled()
        assert window.temperature_page.temperature_source_scope == "temporary"
        assert not window.temperature_page.select_folder_button.isHidden()
        assert window.pie_page.pie_source_scope == "temporary"
        assert not window.pie_page.select_folder_button.isHidden()
        assert window.temperature_page.project_settings is None
        assert window.pie_page.project_settings is None
        assert window.mole_fraction_page.project_settings is None
        assert window.pics_page.project_settings is None
        assert window.mole_fraction_page._mass_disc_exponent == pytest.approx(0.42)
        assert window.pics_page.spin_md_exponent.value() == pytest.approx(0.42)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_temperature_temporary_source_survives_project_parameter_sync(qapp, tmp_path):
    project_dir = tmp_path / "Project_Temp_Source"
    project_temp = project_dir / "raw_data" / "temperature_scan"
    temporary_temp = tmp_path / "temporary_temperature"
    updated_project_temp = project_dir / "raw_data" / "temperature_scan_updated"
    for folder in (project_temp, temporary_temp, updated_project_temp):
        folder.mkdir(parents=True)

    ps = ProjectSettings(
        project_name="Temp Source",
        output_dir=str(project_dir),
        temperature_scan_folder=str(project_temp),
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)
        window._load_project_settings_to_parameter_widgets(ps)
        window._apply_settings_to_tools(ps)

        assert window.temperature_page.temperature_source_scope == "project"
        assert window.temperature_page.folder_edit.text() == str(project_temp)

        window.temperature_page.set_temperature_source_scope("temporary")
        window.temperature_page.folder_edit.setText(str(temporary_temp))
        ps.temperature_scan_folder = str(updated_project_temp)
        window._sync_project_settings_to_tool_pages(ps)

        assert window.temperature_page.temperature_source_scope == "temporary"
        assert window.temperature_page.folder_edit.text() == str(temporary_temp)
        assert not window.temperature_page.select_folder_button.isHidden()
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_spectrum_workbench_custom_source_does_not_overwrite_legacy_project_source_fields(qapp, tmp_path):
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
        window.set_spectrum_source_scope("custom")
        window.lineEdit.setText(str(custom_single))
        window._remember_custom_spectrum_paths()

        window._auto_save_datasource()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.single_spectrum_file == str(project_single)
        assert window.lineEdit.text() == str(custom_single)
        assert window.spectrum_source_scope == "custom"
        assert not window.lineEdit.isReadOnly()
        assert window.singleBrowseButton.isEnabled()
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_spectrum_workbench_project_source_can_select_from_project_scan_folder(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Source_Select"
    temp_dir = project_dir / "raw_data" / "temperature_scan" / "temp"
    source_file = temp_dir / "650K.txt"
    temp_dir.mkdir(parents=True)
    source_file.write_text("tof intensity\n1 2\n", encoding="utf-8")

    ps = ProjectSettings(
        project_name="Source Select",
        system="C6H6",
        output_dir=str(project_dir),
        temperature_scan_folder=str(temp_dir),
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(source_file), ""),
        )

        window.set_spectrum_source_scope("project")
        assert window.singleBrowseButton.isEnabled()

        window.choose_single_source()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert window.lineEdit.text() == str(source_file)
        assert saved.single_spectrum_file == str(source_file)
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
        preset_idx = window.project_common_parameters_widget.mf_md_preset_combo.findData("150 Torr (Combustion)")
        assert light_idx >= 0
        assert preset_idx >= 0
        assert not hasattr(window.project_common_parameters_widget, "pie_photon_mode_combo")
        assert not hasattr(window.project_common_parameters_widget, "temperature_photon_check")
        assert not hasattr(window.project_common_parameters_widget, "temperature_kr_check")

        window.project_common_parameters_widget.light_source_combo.setCurrentIndex(light_idx)
        window.project_common_parameters_widget.calibration_a_edit.setValue(4.56e-7)
        window.project_common_parameters_widget.calibration_b_edit.setValue(7.89e-4)
        window.project_common_parameters_widget.calibration_c_edit.setValue(0.12)
        window.project_common_parameters_widget.mf_md_preset_combo.setCurrentIndex(preset_idx)
        window.project_common_parameters_widget.mf_mass_disc_exponent_edit.setValue(0.76155)

        window.save_and_apply_project_settings()
        _wait_for_project_materialization(qapp, window)

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        saved_yaml = yaml.safe_load((project_dir / "config" / "project.yaml").read_text(encoding="utf-8"))
        assert saved.light_source == "beam_current"
        assert saved.pie_photon_mode == "none"
        assert saved.mass_discrimination == pytest.approx(1.0)
        assert "mass_discrimination" not in saved_yaml["general_parameters"]["normalization"]
        assert saved.cal_a == pytest.approx(4.56e-7)
        assert saved.cal_b == pytest.approx(7.89e-4)
        assert saved.cal_c == pytest.approx(0.12)
        assert saved.mf_md_preset == "150 Torr (Combustion)"
        assert saved.mf_mass_disc_exponent == pytest.approx(0.76155)
        assert window.normalization_settings.mass_discrimination == pytest.approx(1.0)
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
        pie_photon_mode="none",
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
        assert light_index >= 0
        window.project_common_parameters_widget.light_source_combo.setCurrentIndex(light_index)

        window.switch_workspace_page("spectrum")

        active = window.project_settings_manager.get()
        assert active.peak_algorithm == "cwt"
        assert active.detection_min_idx == 2222
        assert active.min_intensity == pytest.approx(12.5)
        assert active.light_source == "beam_current"
        assert active.pie_photon_mode == "none"
        assert active.mass_discrimination == pytest.approx(1.0)

        assert window.current_peak_detection_config().algorithm == "cwt"
        assert window.current_peak_detection_config().detection_min_idx == 2222
        assert window.normalization_settings.light_source == "beam_current"
        assert window.temperature_page.project_settings.peak_algorithm == "cwt"
        assert window.temperature_page.normalization_settings.mass_discrimination == pytest.approx(1.0)
        assert window.pie_page.project_settings.peak_algorithm == "cwt"
        assert window.pie_page.normalization_settings.pie_photon_mode == "none"

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        saved_yaml = yaml.safe_load((project_dir / "config" / "project.yaml").read_text(encoding="utf-8"))
        assert saved.peak_algorithm == "cwt"
        assert saved.detection_min_idx == 2222
        assert saved.light_source == "beam_current"
        assert saved.pie_photon_mode == "none"
        assert saved.mass_discrimination == pytest.approx(1.0)
        assert "mass_discrimination" not in saved_yaml["general_parameters"]["normalization"]

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


def test_workbench_peak_detection_preset_is_temporary(qapp, tmp_path):
    project_dir = tmp_path / "Project_Peak_Preset"
    project_dir.mkdir(parents=True)

    ps = ProjectSettings(
        project_name="Peak Preset",
        system="C6H6",
        output_dir=str(project_dir),
        peak_algorithm="ensemble",
        min_intensity=3.0,
        prominence_ratio=0.005,
        vote_threshold=0.667,
        min_intensity_for_single_vote=5.0,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)

        preset_index = window.peakDetectionPreset.findData("less_noise")
        assert preset_index >= 0
        window.peakDetectionPreset.setCurrentIndex(preset_index)

        original = window.current_peak_detection_config()
        adjusted = window._apply_peak_detection_preset(original)

        assert adjusted is not original
        assert adjusted.min_intensity_for_single_vote == pytest.approx(50.0)
        assert adjusted.prominence_ratio == pytest.approx(0.01)
        assert window.project_settings_manager.get().min_intensity_for_single_vote == pytest.approx(5.0)
        assert window.project_settings_manager.get().prominence_ratio == pytest.approx(0.005)

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.min_intensity_for_single_vote == pytest.approx(5.0)
        assert saved.prominence_ratio == pytest.approx(0.005)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_peak_commands_use_compact_action_menus(qapp):
    window = MainWindow()
    try:
        assert window.savePeakdata.isHidden()
        assert window.clearPeaksButton.isHidden()
        assert window.deleteSelectedPeakButton.isHidden()
        assert window.updatePeakRangeButton.isHidden()
        assert window.openProjectPeaksButton.isHidden()
        assert window.publishProjectPeaksButton.isHidden()

        assert window.peakDataMenuButton.menu() is not None
        assert [action.text() for action in window.peakDataMenuButton.menu().actions()] == [
            "导出卡峰范围…",
            "清空峰值数据",
        ]
        assert window.peakNavigationPanel.layout().count() == 4
        assert [
            window.peakNavigationPanel.layout().itemAt(index).widget().text()
            for index in range(window.peakNavigationPanel.layout().count())
        ] == ["上一峰", "下一峰", "添加峰", "编辑峰"]
        assert [action.text() for action in window.peakEditMenuButton.menu().actions()] == [
            "用框选区域更新当前峰",
            "删除选中的峰",
        ]
        assert window.peakProjectActionPanel.layout().count() == 2
        assert window.peakProjectActionPanel.layout().itemAt(0).widget() is window.peakProjectStateLabel
        assert window.peakProjectActionPanel.layout().itemAt(1).widget() is window.peakProjectMenuButton
        assert [
            window.peakData.horizontalHeaderItem(column).text()
            for column in range(window.peakData.columnCount())
        ] == ["物种", "TOF", "m/z", "强度", "左边界", "右边界"]
        assert not window.publishProjectPeaksAction.isEnabled()
        assert not window.updatePeakRangeAction.isEnabled()
        assert not window.deleteSelectedPeakAction.isEnabled()
    finally:
        window.deleteLater()


def test_workbench_delete_selected_peaks_removes_multiple_rows(qapp, monkeypatch):
    window = MainWindow()
    try:
        rows = [
            ["A", "100.0", "10.0", "50.0", "98.0", "102.0"],
            ["B", "200.0", "20.0", "60.0", "198.0", "202.0"],
            ["C", "300.0", "30.0", "70.0", "298.0", "302.0"],
        ]
        window.peakData.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                window.peakData.setItem(row, column, QtWidgets.QTableWidgetItem(value))

        window.update_plot = lambda: None
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
        )

        window.peakData.clearSelection()
        selection_model = window.peakData.selectionModel()
        for row in (0, 2):
            index = window.peakData.model().index(row, 0)
            selection_model.select(
                index,
                QtCore.QItemSelectionModel.SelectionFlag.Select
                | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
        assert window._selected_peak_rows() == [0, 2]

        window.delete_selected_peaks()

        assert window.peakData.rowCount() == 1
        assert window.peakData.item(0, 0).text() == "B"
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_publish_peak_ranges_registers_project_manual_file(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Publish_Peaks"
    project_dir.mkdir(parents=True)
    ps = ProjectSettings(
        project_name="Publish Peaks",
        system="C6H6",
        output_dir=str(project_dir),
        temp_peak_source="auto",
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)

        source_file = project_dir / "raw_data" / "single.txt"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("tof intensity\n96 1\n100 20\n104 1\n196 1\n200 15\n204 1\n", encoding="utf-8")
        window.tabWidget.setCurrentIndex(0)
        window.lineEdit.setText(str(source_file))
        window.x_axis_mode = "tof"
        window.current_time_offset = 0.0
        window.setup_plots(np.array([96.0, 100.0, 104.0, 196.0, 200.0, 204.0]), np.array([1.0, 20.0, 1.0, 1.0, 15.0, 1.0]))

        rows = [
            ["CH4", "100.2", "16.03", "1200.0", "96.0", "104.0"],
            ["C2H4", "200.4", "28.05", "900.0", "196.0", "204.0"],
        ]
        window.peakData.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                window.peakData.setItem(row, column, QtWidgets.QTableWidgetItem(value))

        window.publish_peak_ranges_to_project()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        saved_path = Path(saved.manual_peak_file)
        manifest_path = saved_path.with_suffix(".manifest.yaml")
        assert saved.temp_peak_source == "manual"
        assert saved_path == project_dir / "analysis" / "spectrum" / "manual_peaks" / "manual_peak_ranges.csv"
        assert saved_path.exists()
        assert manifest_path.exists()

        df = pd.read_csv(saved_path)
        assert list(df.columns) == ["label", "peak_index", "mz", "left_bound", "right_bound"]
        assert df.loc[0, "label"] == "CH4"
        assert df.loc[0, "peak_index"] == 100
        assert df.loc[0, "left_bound"] == 96
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert manifest["peak_file"] == "manual_peak_ranges.csv"
        assert manifest["source"]["mode"] == "single"
        assert manifest["source"]["path"] == str(source_file)
        assert window.project_manual_peak_edit.text() == str(saved_path)
        assert window._peak_table_dirty is False
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_auto_find_does_not_overwrite_project_peak_file(qapp, tmp_path, monkeypatch):
    import bl03u_masstool.frontends.pyqt_app.spectrum.workbench as workbench_module

    project_dir = tmp_path / "Project_Auto_Does_Not_Overwrite"
    manual_peak_file = project_dir / "analysis" / "spectrum" / "manual_peaks" / "manual_peak_ranges.csv"
    manual_peak_file.parent.mkdir(parents=True)
    manual_peak_file.write_text("label,peak_index,mz,left_bound,right_bound\nOld,10,10,8,12\n", encoding="utf-8")

    ps = ProjectSettings(
        project_name="Auto Preserve",
        system="C6H6",
        output_dir=str(project_dir),
        manual_peak_file=str(manual_peak_file),
        temp_peak_source="manual",
        peak_algorithm="legacy",
        detection_min_idx=0,
        min_intensity=1.0,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        monkeypatch.setattr(
            workbench_module,
            "detect_peaks_by_algorithm",
            lambda *args, **kwargs: [],
        )

        tof = np.arange(20.0)
        intensity = np.ones_like(tof)
        window.current_time_offset = 0.0
        window.setup_plots(tof, intensity)
        window.auto_find_peaks()

        assert manual_peak_file.read_text(encoding="utf-8") == "label,peak_index,mz,left_bound,right_bound\nOld,10,10,8,12\n"
        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.manual_peak_file == str(manual_peak_file)
        assert saved.temp_peak_source == "manual"
        assert window.peakData.rowCount() == 0
        assert window._peak_table_dirty is True
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_publish_peak_ranges_requires_reopenable_spectrum_source(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Publish_Requires_Source"
    project_dir.mkdir(parents=True)
    ps = ProjectSettings(
        project_name="Publish Requires Source",
        system="C6H6",
        output_dir=str(project_dir),
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append((args[1], args[2])),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: pytest.fail("保存前置条件不满足时不应出现覆盖确认"),
    )

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window.peakData.setRowCount(1)
        for column, value in enumerate(["Peak", "101.0", "28.0", "42.0", "99.0", "103.0"]):
            window.peakData.setItem(0, column, QtWidgets.QTableWidgetItem(value))

        window.publish_peak_ranges_to_project()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.manual_peak_file == ""
        assert not (project_dir / "analysis" / "spectrum" / "manual_peaks" / "manual_peak_ranges.csv").exists()
        assert warnings
        assert "请先打开原始谱图" in warnings[0][1]
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_open_project_peak_ranges_rejects_incomplete_project_artifact(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Incomplete_Peaks"
    project_dir.mkdir(parents=True)
    peak_file = project_dir / "peak_ranges.csv"
    peak_file.write_text(
        "\ufeffSpecies,飞行时间,质量数 (m/z),强度,左边界,右边界\n"
        "Unknown,101.0,28.0,42.0,99.0,103.0\n",
        encoding="utf-8",
    )
    ps = ProjectSettings(
        project_name="Incomplete Peaks",
        system="C6H6",
        output_dir=str(project_dir),
        manual_peak_file=str(peak_file),
        temp_peak_source="manual",
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append((args[1], args[2])),
    )

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)

        window.open_project_peak_ranges()

        assert window._valid_peak_rows() == []
        assert warnings
        assert warnings[0][0] == "项目卡峰不完整"
        assert "manifest 未写入" in warnings[0][1]
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_open_project_peak_ranges_restores_manifest_source(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Reopen_Peaks"
    source_dir = project_dir / "raw_data" / "sum_spectrum"
    source_dir.mkdir(parents=True)
    spectrum_lines_a = ["header\n"] * 10 + [f"{idx} {1.0 + (idx == 4101) * 4.0}\n" for idx in range(1, 4104)]
    spectrum_lines_b = ["header\n"] * 10 + [f"{idx} {2.0 + (idx == 4101) * 5.0}\n" for idx in range(1, 4104)]
    (source_dir / "a.txt").write_text("".join(spectrum_lines_a), encoding="utf-8")
    (source_dir / "b.txt").write_text("".join(spectrum_lines_b), encoding="utf-8")

    ps = ProjectSettings(
        project_name="Reopen Peaks",
        system="C6H6",
        output_dir=str(project_dir),
        sum_spectrum_folder=str(source_dir),
        temp_peak_source="auto",
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

        window.tabWidget.setCurrentIndex(1)
        window.folder_path.setText(str(source_dir))
        window.x_axis_mode = "tof"
        window.current_time_offset = 4001.0
        window.setup_plots(np.array([4001.0, 4002.0, 4003.0]), np.array([3.0, 12.0, 3.0]), title="累加质谱图")
        window.peakData.setRowCount(1)
        for column, value in enumerate(["Peak", "4002.0", "4002.0", "12.0", "4001.0", "4003.0"]):
            window.peakData.setItem(0, column, QtWidgets.QTableWidgetItem(value))

        window.publish_peak_ranges_to_project()
        saved = load_project_settings(project_dir / "config" / "project.yaml")

        window.clear_peak_data()
        window.folder_path.clear()
        window.open_project_peak_ranges()

        assert window.folder_path.text() == str(source_dir)
        assert window.peakData.rowCount() == 1
        assert window.peakData.item(0, 0).text() == "Peak"
        assert window.peakData.item(0, 4).text() == "4001.00"
        assert Path(saved.manual_peak_file).with_suffix(".manifest.yaml").exists()
        assert window._peak_table_dirty is False
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_next_peak_does_not_modify_existing_bounds(qapp):
    window = MainWindow()
    try:
        window.x_axis_mode = "tof"
        tof = np.arange(7001.0, 24163.0)
        intensity = np.zeros_like(tof)
        intensity[7159] = 970.0
        intensity[8505] = 52.0
        window.current_time_offset = 0.0
        window.setup_plots(tof, intensity)

        rows = [
            ["Unknown", "14160.00", "83.11", "970.00", "14149.00", "14170.00"],
            ["Unknown", "15505.80", "98.64", "52.00", "7001.00", "24162.00"],
        ]
        window.peakData.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                window.peakData.setItem(row, column, QtWidgets.QTableWidgetItem(value))

        window._select_peak_row(0)
        window.select_next_peak()

        selected_left, selected_right = window.selection_region.getRegion()
        assert window.peakData.currentRow() == 1
        assert selected_left == pytest.approx(7001.0)
        assert selected_right == pytest.approx(24162.0)
        assert window.peakData.item(1, 4).text() == "7001.00"
        assert window.peakData.item(1, 5).text() == "24162.00"
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_peak_navigation_y_scale_prioritizes_selected_peak(qapp):
    window = MainWindow()
    try:
        window.x_axis_mode = "tof"
        tof = np.arange(1000.0, 1301.0)
        intensity = np.full_like(tof, 5.0)
        intensity[50] = 5000.0
        intensity[150] = 100.0
        window.current_time_offset = 0.0
        window.setup_plots(tof, intensity)

        rows = [
            ["Unknown", "1050.00", "1050.00", "5000.00", "1048.00", "1052.00"],
            ["Unknown", "1150.00", "1150.00", "100.00", "1148.00", "1152.00"],
        ]
        window.peakData.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                window.peakData.setItem(row, column, QtWidgets.QTableWidgetItem(value))

        window._select_peak_row(0)
        window.select_next_peak()

        _, y_range = window.p2.viewRange()
        assert window.peakData.currentRow() == 1
        assert y_range[1] < 1000.0
        assert y_range[1] > 100.0
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_function_default_edits_sync_when_switching_to_tools(qapp, tmp_path):
    project_dir = tmp_path / "Project_Function_Default_Sync"
    project_dir.mkdir(parents=True)
    pie_low = tmp_path / "pie_low"
    pie_high = tmp_path / "pie_high"
    pie_low.mkdir()
    pie_high.mkdir()

    ps = ProjectSettings(
        project_name="Function Default Sync",
        system="C6H6",
        output_dir=str(project_dir),
        pie_scan_folder=str(pie_low),
        pie_scan_folders=[str(pie_low), str(pie_high)],
        pie_energy_decimals=1,
        pie_recursive=True,
        pie_prefer_gaussian=True,
        pie_integration_method="gaussian",
        pie_multi_folder_mode=False,
        pie_merge_method="low_energy_dominant",
        temp_peak_source="auto",
        temp_reference_mode="sum",
        temp_prefer_gaussian=True,
        temp_integration_method="gaussian",
        kr_mz=84,
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
        temp_replicate_index = widget.temp_replicate_mode_combo.findData("sum")
        pie_replicate_index = widget.pie_replicate_mode_combo.findData("sum")
        assert peak_source_index >= 0
        assert reference_index >= 0
        assert merge_index >= 0
        assert temp_replicate_index >= 0
        assert pie_replicate_index >= 0
        widget.temp_peak_source_combo.setCurrentIndex(peak_source_index)
        widget.temp_reference_mode_combo.setCurrentIndex(reference_index)
        temp_integration_index = widget.temp_integration_method_combo.findData("baseline")
        assert temp_integration_index >= 0
        widget.temp_integration_method_combo.setCurrentIndex(temp_integration_index)
        widget.temp_curve_class_change_threshold_edit.setValue(0.31)
        widget.temp_curve_class_peak_fraction_edit.setValue(0.73)
        widget.temp_replicate_mode_combo.setCurrentIndex(temp_replicate_index)
        widget.pie_energy_decimals_edit.setValue(4)
        widget.pie_recursive_check.setChecked(False)
        integration_index = widget.pie_integration_method_combo.findData("baseline")
        assert integration_index >= 0
        widget.pie_integration_method_combo.setCurrentIndex(integration_index)
        widget.pie_merge_method_combo.setCurrentIndex(merge_index)
        widget.pie_replicate_mode_combo.setCurrentIndex(pie_replicate_index)
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
        assert window.project_settings_manager.get().kr_mz == 84

        # Re-clicking the current Project tab should collect in-memory edits,
        # not reload the older project.yaml over the UI values.
        window.switch_workspace_page("project")
        assert widget.pie_energy_decimals_edit.value() == 4

        window.switch_workspace_page("pie")

        active = window.project_settings_manager.get()
        assert active.temp_peak_source == "manual"
        assert active.temp_reference_mode == "individual"
        assert active.temp_prefer_gaussian is False
        assert active.temp_integration_method == "baseline"
        assert active.kr_mz == 84
        assert active.temp_curve_class_change_threshold == pytest.approx(0.31)
        assert active.temp_curve_class_peak_fraction == pytest.approx(0.73)
        assert active.temp_replicate_mode == "sum"
        assert active.pie_energy_decimals == 4
        assert active.pie_recursive is False
        assert active.pie_prefer_gaussian is False
        assert active.pie_integration_method == "baseline"
        assert active.pie_multi_folder_mode is True
        assert active.pie_merge_method == "first_segment_dominant"
        assert active.pie_replicate_mode == "sum"
        assert active.pics_no_mz == 31
        assert active.pics_no_formula == "15NO"
        assert active.pics_no_mf == pytest.approx(0.021)
        assert active.pics_new_species_mf == pytest.approx(0.004)
        assert active.mf_parent_mz == 130
        assert active.mf_parent_initial_mf == pytest.approx(0.006)
        assert active.mf_photon_energy == pytest.approx(11.25)
        assert active.mf_reference_temperature == pytest.approx(575.0)

        assert window.temperature_page.project_settings.temp_reference_mode == "individual"
        assert window.temperature_page.project_settings.kr_mz == 84
        assert window.pie_page.project_settings.pie_energy_decimals == 4
        assert not hasattr(window.pie_page, "use_multi_folders")
        assert not hasattr(window.pie_page, "merge_method_combo")
        assert window.pie_page.project_settings.pie_merge_method == "first_segment_dominant"
        assert window.pie_page.folder_edit.text() == "项目管理已登记 2 个 PIE 能段目录"
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


def test_import_finished_appends_pie_segment_without_overwriting_existing(qapp, tmp_path):
    project_dir = tmp_path / "Project_Multi_PIE"
    low = project_dir / "raw_data" / "pie_scan" / "low"
    high = project_dir / "raw_data" / "pie_scan" / "high"
    low.mkdir(parents=True)
    high.mkdir(parents=True)
    ps = ProjectSettings(
        project_name="Multi PIE",
        output_dir=str(project_dir),
        pie_scan_folder=str(low),
        pie_scan_folders=[str(low)],
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)

        window._on_import_finished(
            {
                "success": True,
                "field_name": "pie_scan_folder",
                "destination": str(high),
                "label": "PIE 能段目录",
                "source_key": "pie_scan",
            }
        )

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.pie_scan_folders == [str(low), str(high)]
        assert saved.pie_multi_folder_mode is True
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_project_settings_autosave_failure_is_visible(qapp, tmp_path, monkeypatch, caplog):
    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(tmp_path / "project.yaml")
        monkeypatch.setattr(
            window.project_settings_manager,
            "save",
            lambda: (_ for _ in ()).throw(OSError("disk full")),
        )

        with caplog.at_level("ERROR"):
            window._sync_project_page_edits_to_runtime(
                save_project=True,
                sync_tools=False,
            )

        assert "自动保存失败" in window.statusbar.currentMessage()
        assert "Failed to save project settings" in caplog.text
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_import_finished_registers_manual_peak_file(qapp, tmp_path):
    project_dir = tmp_path / "Project_Manual_Import"
    imported_peak = project_dir / "analysis" / "spectrum" / "manual_peaks" / "peaks.csv"
    imported_peak.parent.mkdir(parents=True)
    imported_peak.write_text("mz,start,end\n28,10,12\n", encoding="utf-8")

    ps = ProjectSettings(project_name="Manual Import", system="C6H6", output_dir=str(project_dir))
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)

        window._on_import_finished(
            {
                "success": True,
                "field_name": "manual_peak_file",
                "destination": str(imported_peak),
                "label": "手动卡峰文件",
                "source_key": "manual_peak",
                "mode": "copy",
            }
        )

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.manual_peak_file == str(imported_peak)
        assert window.project_manual_peak_edit.text() == str(imported_peak)
        assert window.datasource_row_status_labels["manual_peak"].text().startswith("✓")
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_project_manual_peak_button_imports_legacy_file(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Manual_Button"
    legacy_peak = tmp_path / "legacy" / "old_peak_ranges.csv"
    legacy_peak.parent.mkdir(parents=True)
    legacy_peak.write_text("label,peak_index,mz,left_bound,right_bound\nA,100,28,98,102\n", encoding="utf-8")

    ps = ProjectSettings(project_name="Manual Button", system="C6H6", output_dir=str(project_dir))
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(legacy_peak), ""),
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: pytest.fail(str(args)))
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)

        window.import_project_manual_peak_file()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        imported = project_dir / "analysis" / "spectrum" / "manual_peaks" / legacy_peak.name
        assert Path(saved.manual_peak_file) == imported
        assert imported.read_text(encoding="utf-8") == legacy_peak.read_text(encoding="utf-8")
        assert window.project_manual_peak_edit.text() == str(imported)
        assert window.datasource_row_status_labels["manual_peak"].text().startswith("✓")
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()
