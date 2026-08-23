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

from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.project_settings import ProjectSettingsManager
from bl03u_masstool.core.project_settings import load_factory_project_settings
from bl03u_masstool.core.project_settings import load_project_settings
from bl03u_masstool.core.project_settings import save_project_settings
from bl03u_masstool.core.project_lifecycle import project_root
from bl03u_masstool.core.peak_sets import (
    create_peak_set,
    import_peak_set,
    list_peak_sets,
    verify_peak_set,
)
from bl03u_masstool.core.peak_range_assessment import (
    PeakRangeAssessment,
    PeakRangeFitStatus,
)
from bl03u_masstool.core.peak_ranges import PeakRange
from bl03u_masstool.frontends.pyqt_app.common.widgets import AnalysisEmptyState
from bl03u_masstool.frontends.pyqt_app.normalization.widget import FunctionDefaultsWidget
from bl03u_masstool.frontends.pyqt_app.spectrum.workbench import MainWindow
from bl03u_masstool.frontends.pyqt_app.temporary_analysis_settings import (
    ANALYSIS_PARAMETER_FIELDS,
    AnalysisSettingsDialog,
    TemporaryAnalysisSettingsDialog,
    TemporarySharedParametersDialog,
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

    config_dir = tmp_path / "runtime_config"
    config_dir.mkdir()

    def fake_save_calibration_config(calibration, path=None):
        return config_dir / "calibration.yaml"

    def fake_save_peak_detection_config(config, path=None):
        return config_dir / "peak_detection.yaml"

    monkeypatch.setattr(core_config, "save_calibration_config", fake_save_calibration_config)
    monkeypatch.setattr(core_config, "save_peak_detection_config", fake_save_peak_detection_config)

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
    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
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


@pytest.mark.parametrize(
    "dependency",
    (
        "light_source",
        "cal_a",
        "cal_b",
        "cal_c",
        "kr_folder",
        "kr_peak_file",
        "kr_mz",
    ),
)
def test_common_parameters_invalidates_kr_factors_when_inputs_change(
    qapp,
    dependency,
):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import (
        CommonParametersWidget,
    )

    settings = ProjectSettings(
        cal_a=1.0,
        cal_b=2.0,
        cal_c=3.0,
        light_source="io",
        kr_calibration_folder="/tmp/kr-original",
        kr_calibration_peak_file="/tmp/kr-original.csv",
        kr_mz=84,
        expansion_factors={400.0: 1.0, 800.0: 1.2},
    )
    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
        widget.set_project_settings(settings)
        assert widget.factor_table.rowCount() == 2

        if dependency == "light_source":
            widget.light_source_combo.setCurrentIndex(
                widget.light_source_combo.findData("beam_current")
            )
        elif dependency == "cal_a":
            widget.calibration_a_edit.setValue(1.5)
        elif dependency == "cal_b":
            widget.calibration_b_edit.setValue(2.5)
        elif dependency == "cal_c":
            widget.calibration_c_edit.setValue(3.5)
        elif dependency == "kr_folder":
            widget.kr_folder_edit.setText("/tmp/kr-updated")
        elif dependency == "kr_peak_file":
            widget.kr_peak_file_edit.setText("/tmp/kr-updated.csv")
        elif dependency == "kr_mz":
            widget.kr_mz_combo.setCurrentText("86")

        widget.apply_to_settings(settings)

        assert settings.expansion_factors == {}
        assert widget.settings.expansion_factors == {}
        assert widget.factor_table.rowCount() == 0
        assert "重新计算 λ(T)" in widget.status_label.text()
    finally:
        widget.deleteLater()


def test_common_parameters_preserves_kr_factors_when_inputs_are_unchanged(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import (
        CommonParametersWidget,
    )

    factors = {400.0: 1.0, 800.0: 1.2}
    settings = ProjectSettings(
        cal_a=1.0,
        cal_b=2.0,
        cal_c=3.0,
        light_source="io",
        kr_calibration_folder="/tmp/kr",
        kr_mz=84,
        expansion_factors=factors,
    )
    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
        widget.set_project_settings(settings)
        widget.apply_to_settings(settings)

        assert settings.expansion_factors == factors
        assert widget.factor_table.rowCount() == 2
    finally:
        widget.deleteLater()


def test_embedded_project_calibration_change_invalidates_kr_factors(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import (
        CommonParametersWidget,
    )

    original = ProjectSettings(
        cal_a=1.0,
        cal_b=2.0,
        cal_c=3.0,
        light_source="io",
        kr_calibration_folder="/tmp/kr",
        kr_mz=84,
        expansion_factors={400.0: 1.0},
    )
    edited = ProjectSettings(
        cal_a=1.5,
        cal_b=2.0,
        cal_c=3.0,
        light_source="io",
        kr_calibration_folder="/tmp/kr",
        kr_mz=84,
        expansion_factors={400.0: 1.0},
    )
    widget = CommonParametersWidget(
        NormalizationSettings(),
        Calibration(),
        None,
        show_calibration=False,
        show_kr_expansion=False,
    )
    try:
        widget.set_project_settings(original)
        widget.apply_to_settings(edited)

        assert edited.expansion_factors == {}
    finally:
        widget.deleteLater()


def test_recomputed_kr_factors_are_bound_to_current_inputs(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.normalization.widget import (
        CommonParametersWidget,
    )

    settings = ProjectSettings(
        kr_calibration_folder="/tmp/kr-original",
        kr_mz=84,
        expansion_factors={400.0: 1.0},
    )
    widget = CommonParametersWidget(NormalizationSettings(), Calibration(), None)
    try:
        widget.set_project_settings(settings)
        widget.kr_mz_combo.setCurrentText("86")
        widget.apply_to_settings(settings)
        assert settings.expansion_factors == {}

        widget.on_kr_factors_ready(
            pd.DataFrame(
                [
                    {
                        "temperature": 400.0,
                        "kr_signal": 10.0,
                        "expansion_lambda": 1.0,
                    },
                    {
                        "temperature": 800.0,
                        "kr_signal": 20.0,
                        "expansion_lambda": 2.0,
                    },
                ]
            )
        )
        widget.apply_to_settings(settings)
        assert settings.expansion_factors == {400.0: 1.0, 800.0: 2.0}

        widget.kr_folder_edit.setText("/tmp/kr-updated")
        widget.apply_to_settings(settings)
        assert settings.expansion_factors == {}
    finally:
        widget.deleteLater()


def test_function_defaults_apply_to_project_settings(qapp):
    widget = FunctionDefaultsWidget()
    try:
        ps = ProjectSettings(
            peak_algorithm="legacy",
            detection_min_idx=111,
            pie_scan_folders=["pie-low", "pie-high"],
            expansion_factors={500.0: 1.0},
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
        widget.temperature_photon_check.setChecked(False)
        widget.temperature_kr_check.setChecked(True)
        temp_integration_index = widget.temp_integration_method_combo.findData("baseline")
        assert temp_integration_index >= 0
        widget.temp_integration_method_combo.setCurrentIndex(temp_integration_index)
        widget.temp_curve_class_change_threshold_edit.setValue(0.35)
        widget.temp_curve_class_peak_fraction_edit.setValue(0.72)
        widget.pie_energy_decimals_edit.setValue(3)
        widget.pie_recursive_check.setChecked(False)
        widget.pie_photon_mode_combo.setCurrentIndex(
            widget.pie_photon_mode_combo.findData("first")
        )
        widget.pie_time_normalize_check.setChecked(False)
        integration_index = widget.pie_integration_method_combo.findData("baseline")
        assert integration_index >= 0
        widget.pie_integration_method_combo.setCurrentIndex(integration_index)
        widget.pie_merge_method_combo.setCurrentIndex(merge_index)
        widget.pics_no_mz_edit.setValue(31)
        widget.pics_no_formula_edit.setText("15NO")
        widget.pics_no_mf_edit.setValue(0.02)
        widget.pics_new_species_mf_edit.setValue(0.004)
        widget.pics_mass_disc_exponent_edit.setValue(0.67)
        widget.element_checks["C"].setChecked(True)
        widget.element_checks["H"].setChecked(True)
        widget.element_checks["O"].setChecked(False)
        preset_idx = widget.mf_md_preset_combo.findData(
            "30 Torr (Catalysis)"
        )
        assert preset_idx >= 0
        widget.mf_md_preset_combo.setCurrentIndex(preset_idx)
        widget.mf_mass_disc_exponent_edit.setValue(0.75148)
        widget.mf_parent_mz_edit.setValue(130)
        widget.mf_parent_initial_mf_edit.setValue(0.006)
        widget.mf_photon_energy_edit.setValue(11.2)
        widget.mf_reference_temperature_edit.setValue(575)

        widget.apply_to_settings(ps)

        assert ps.peak_algorithm == "cwt"
        assert ps.detection_min_idx == 4321
        assert ps.min_intensity == pytest.approx(12.5)
        # Peak source is derived by the project save service from the current
        # project baseline; the hidden compatibility control cannot overwrite it.
        assert ps.temp_peak_source == "auto"
        assert ps.temp_reference_mode == "individual"
        assert ps.temperature_photon_normalize is False
        assert ps.temperature_kr_correct is True
        assert ps.temp_prefer_gaussian is False
        assert ps.temp_integration_method == "baseline"
        assert ps.temp_curve_class_change_threshold == pytest.approx(0.35)
        assert ps.temp_curve_class_peak_fraction == pytest.approx(0.72)
        assert ps.pie_energy_decimals == 3
        assert ps.pie_recursive is False
        assert ps.pie_photon_mode == "first"
        assert ps.pie_time_normalize is False
        assert ps.pie_prefer_gaussian is False
        assert ps.pie_integration_method == "baseline"
        assert ps.pie_multi_folder_mode is True
        assert ps.pie_merge_method == "mean"
        assert ps.pics_no_mz == 31
        assert ps.pics_no_formula == "15NO"
        assert ps.pics_no_mf == pytest.approx(0.02)
        assert ps.pics_new_species_mf == pytest.approx(0.004)
        assert ps.pics_mass_disc_exponent == pytest.approx(0.67)
        assert "C" in ps.selected_elements
        assert "H" in ps.selected_elements
        assert "O" not in ps.selected_elements
        assert ps.mf_md_preset == "30 Torr (Catalysis)"
        assert ps.mf_mass_disc_exponent == pytest.approx(0.75148)
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


def test_project_function_defaults_keep_integration_methods_independent(qapp):
    widget = FunctionDefaultsWidget()
    try:
        settings = ProjectSettings(
            temp_integration_method="sum_counts",
            pie_integration_method="baseline",
        )
        widget.set_project_settings(settings)

        assert widget.tabs.tabText(0) == "寻峰与积分"
        assert widget.temp_integration_method_combo.currentData() == "sum_counts"
        assert widget.pie_integration_method_combo.currentData() == "baseline"
        assert (
            widget.temp_integration_method_combo
            is not widget.pie_integration_method_combo
        )

        widget.temp_integration_method_combo.setCurrentIndex(
            widget.temp_integration_method_combo.findData("gaussian")
        )
        widget.apply_to_settings(settings)

        assert settings.temp_integration_method == "gaussian"
        assert settings.pie_integration_method == "baseline"
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
        assert [
            dialog.function_widget.tabs.tabText(index)
            for index in range(dialog.function_widget.tabs.count())
        ] == ["PIE 分析"]
        assert dialog.function_widget.tabs.currentIndex() == 0
        assert not hasattr(dialog, "common_widget")
        assert dialog.scope == "temporary"
        assert dialog.page == "pie"
        assert dialog.function_widget.open_function_page_button.text() == "打开PIE 拟合页"
        assert dialog.function_widget.open_function_page_button.isHidden()
    finally:
        dialog.deleteLater()


def test_analysis_settings_dialog_applies_only_page_whitelist(qapp):
    original = ProjectSettings(
        project_name="Keep me",
        light_source="beam_current",
        temperature_photon_normalize=True,
        temp_integration_method="sum_counts",
        pie_energy_decimals=2,
    )
    dialog = AnalysisSettingsDialog(
        original,
        scope="project",
        page="temperature",
    )
    try:
        dialog.function_widget.temperature_photon_check.setChecked(False)
        integration_index = (
            dialog.function_widget.temp_integration_method_combo.findData("baseline")
        )
        dialog.function_widget.temp_integration_method_combo.setCurrentIndex(
            integration_index
        )
        dialog._settings.project_name = "Must not leak"
        dialog._settings.light_source = "io"
        dialog._settings.pie_energy_decimals = 6

        dialog.accept()
        edited = dialog.settings()

        assert edited.temperature_photon_normalize is False
        assert edited.temp_integration_method == "baseline"
        assert edited.temp_prefer_gaussian is False
        assert edited.project_name == "Keep me"
        assert edited.light_source == "beam_current"
        assert edited.pie_energy_decimals == 2
        assert set(ANALYSIS_PARAMETER_FIELDS["temperature"]) >= {
            "temperature_photon_normalize",
            "temp_integration_method",
        }
    finally:
        dialog.deleteLater()


def test_analysis_settings_dialog_cancel_keeps_snapshot_unchanged(qapp):
    original = ProjectSettings(pie_energy_decimals=2, pie_recursive=True)
    dialog = AnalysisSettingsDialog(
        original,
        scope="temporary",
        page="pie",
    )
    try:
        dialog.function_widget.pie_energy_decimals_edit.setValue(5)
        dialog.function_widget.pie_recursive_check.setChecked(False)
        dialog.reject()

        assert dialog.settings().pie_energy_decimals == 2
        assert dialog.settings().pie_recursive is True
        assert original.pie_energy_decimals == 2
        assert original.pie_recursive is True
    finally:
        dialog.deleteLater()


def test_temporary_shared_parameters_edit_without_project_navigation(
    qapp,
    monkeypatch,
):
    original = ProjectSettings(
        cal_a=1.0,
        cal_b=2.0,
        cal_c=3.0,
        light_source="io",
        expansion_factors={},
        temp_integration_method="sum_counts",
    )
    project_navigation_requests: list[bool] = []

    def accept_shared_edits(shared_dialog):
        common = shared_dialog.common_widget
        common.calibration_a_edit.setValue(4.0)
        common.calibration_b_edit.setValue(5.0)
        common.calibration_c_edit.setValue(6.0)
        common.light_source_combo.setCurrentIndex(
            common.light_source_combo.findData("beam_current")
        )
        common.settings.expansion_factors = {650.0: 1.25}
        shared_dialog.accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        TemporarySharedParametersDialog,
        "exec",
        accept_shared_edits,
    )
    dialog = AnalysisSettingsDialog(
        original,
        scope="temporary",
        page="temperature",
    )
    dialog.shared_parameters_requested.connect(
        lambda: project_navigation_requests.append(True)
    )
    try:
        integration_index = (
            dialog.function_widget.temp_integration_method_combo.findData("baseline")
        )
        dialog.function_widget.temp_integration_method_combo.setCurrentIndex(
            integration_index
        )

        assert dialog.shared_parameters_group.title() == "通用参数"
        assert dialog.open_shared_parameters_button.text() == "编辑通用参数…"
        assert "卡峰集" not in dialog.shared_parameters_label.text()

        dialog.open_shared_parameters_button.click()

        assert project_navigation_requests == []
        assert "Beam Current" in dialog.shared_parameters_label.text()
        assert "Kr：可用" in dialog.shared_parameters_label.text()

        dialog.accept()
        edited = dialog.settings()

        assert edited.cal_a == pytest.approx(4.0)
        assert edited.cal_b == pytest.approx(5.0)
        assert edited.cal_c == pytest.approx(6.0)
        assert edited.light_source == "beam_current"
        assert edited.expansion_factors == {650.0: 1.25}
        assert edited.temp_integration_method == "baseline"
        assert original.cal_a == pytest.approx(1.0)
        assert original.light_source == "io"
        assert original.expansion_factors == {}
    finally:
        dialog.deleteLater()


def test_project_shared_parameters_still_request_project_management(qapp):
    requests: list[bool] = []
    dialog = AnalysisSettingsDialog(
        ProjectSettings(),
        scope="project",
        page="temperature",
    )
    dialog.shared_parameters_requested.connect(lambda: requests.append(True))
    try:
        assert dialog.shared_parameters_group.title() == "通用参数（项目）"
        assert (
            dialog.open_shared_parameters_button.text()
            == "前往项目管理修改"
        )

        dialog.open_shared_parameters_button.click()

        assert requests == [True]
    finally:
        dialog.deleteLater()


def test_temporary_shared_parameter_change_disables_stale_kr_correction(
    qapp,
    monkeypatch,
):
    original = ProjectSettings(
        kr_mz=84,
        expansion_factors={500.0: 1.2},
        temperature_kr_correct=True,
    )

    def accept_changed_kr_input(shared_dialog):
        shared_dialog.common_widget.kr_mz_combo.setCurrentText("86")
        shared_dialog.accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        TemporarySharedParametersDialog,
        "exec",
        accept_changed_kr_input,
    )
    dialog = AnalysisSettingsDialog(
        original,
        scope="temporary",
        page="temperature",
    )
    try:
        assert dialog.function_widget.temperature_kr_check.isChecked()

        dialog.open_shared_parameters_button.click()

        assert "Kr：不可用" in dialog.shared_parameters_label.text()
        assert not dialog.function_widget.temperature_kr_check.isEnabled()
        assert not dialog.function_widget.temperature_kr_check.isChecked()

        dialog.accept()
        edited = dialog.settings()

        assert edited.kr_mz == 86
        assert edited.expansion_factors == {}
        assert edited.temperature_kr_correct is False
        assert original.kr_mz == 84
        assert original.expansion_factors == {500.0: 1.2}
        assert original.temperature_kr_correct is True
    finally:
        dialog.deleteLater()


def test_temporary_temperature_settings_dialog_imports_peak_file_for_session(
    qapp,
    tmp_path,
    monkeypatch,
):
    peak_file = tmp_path / "temporary_peak_ranges.csv"
    peak_file.write_text(
        "label,peak_index,mz,left_bound,right_bound\n"
        "CH4,100,16.03,96,104\n",
        encoding="utf-8",
    )
    original = ProjectSettings(
        manual_peak_file="",
        active_peak_set_id="",
        temp_peak_source="auto",
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(peak_file), "卡峰文件 (*.csv)"),
    )

    dialog = AnalysisSettingsDialog(
        original,
        scope="temporary",
        page="temperature",
    )
    try:
        assert dialog.import_temporary_peak_button.text() == "导入卡峰文件…"
        assert "自动寻峰" in dialog.temporary_peak_status_label.text()

        dialog.import_temporary_peak_button.click()

        assert dialog.temporary_peak_file_edit.text() == str(peak_file.resolve())
        assert "1 个卡峰" in dialog.temporary_peak_status_label.text()

        dialog.accept()
        edited = dialog.settings()

        assert edited.manual_peak_file == str(peak_file.resolve())
        assert edited.active_peak_set_id == ""
        assert edited.temp_peak_source == "manual"
        assert original.manual_peak_file == ""
        assert original.temp_peak_source == "auto"
    finally:
        dialog.deleteLater()


def test_temporary_temperature_settings_can_return_to_automatic_peaks(
    qapp,
    tmp_path,
):
    peak_file = tmp_path / "temporary_peak_ranges.csv"
    peak_file.write_text(
        "label,peak_index,mz,left_bound,right_bound\n"
        "CH4,100,16.03,96,104\n",
        encoding="utf-8",
    )
    original = ProjectSettings(
        manual_peak_file=str(peak_file),
        active_peak_set_id="project-peak-set",
        temp_peak_source="manual",
    )
    dialog = AnalysisSettingsDialog(
        original,
        scope="temporary",
        page="temperature",
    )
    try:
        dialog.clear_temporary_peak_button.click()
        dialog.accept()
        edited = dialog.settings()

        assert edited.manual_peak_file == ""
        assert edited.active_peak_set_id == ""
        assert edited.temp_peak_source == "auto"
        assert original.manual_peak_file == str(peak_file)
        assert original.active_peak_set_id == "project-peak-set"
    finally:
        dialog.deleteLater()


def test_analysis_settings_dialog_controls_kr_and_pie_merge_availability(qapp):
    temperature_dialog = AnalysisSettingsDialog(
        ProjectSettings(
            temperature_kr_correct=True,
            expansion_factors={},
        ),
        scope="project",
        page="temperature",
    )
    pie_dialog = AnalysisSettingsDialog(
        ProjectSettings(pie_scan_folders=["one"]),
        scope="project",
        page="pie",
        pie_multi_segment=False,
    )
    try:
        assert not temperature_dialog.function_widget.temperature_kr_check.isEnabled()
        assert not temperature_dialog.function_widget.temperature_kr_check.isChecked()
        assert not pie_dialog.function_widget.pie_merge_method_combo.isEnabled()

        pie_dialog.function_widget.set_pie_multi_segment_state(True)
        assert pie_dialog.function_widget.pie_merge_method_combo.isEnabled()
    finally:
        temperature_dialog.deleteLater()
        pie_dialog.deleteLater()


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
        ] == ["项目与数据", "定标与卡峰", "项目分析参数"]
        assert [
            window.project_analysis_tabs.tabText(index)
            for index in range(window.project_analysis_tabs.count())
        ] == [
            "寻峰与积分",
            "温度扫描",
            "PIE 分析",
            "PICS 计算",
            "摩尔分数",
        ]
        assert (
            window.project_analysis_save_button.text()
            == "保存并应用项目参数"
        )
        assert "提示保存、放弃或取消" in (
            window.project_analysis_unsaved_hint.text()
        )
        assert not hasattr(
            window.project_common_parameters_widget,
            "mf_md_preset_combo",
        )
        assert hasattr(
            window.project_function_defaults_widget,
            "mf_md_preset_combo",
        )
        assert hasattr(
            window.project_function_defaults_widget,
            "element_checks",
        )
        labels = " ".join(label.text() for label in window.project_identity_page.findChildren(QtWidgets.QLabel))
        assert "项目工作流" not in labels
        assert "建议下一步" not in labels
    finally:
        window.deleteLater()


def test_project_pages_keep_short_content_top_aligned(qapp):
    window = MainWindow()
    try:
        window.resize(1600, 960)
        window.show()
        window.switch_workspace_page("project")
        window.project_tabs.setCurrentWidget(window.project_analysis_page)
        window.project_analysis_tabs.setCurrentIndex(0)
        qapp.processEvents()

        common_page = window.project_common_analysis_page
        groups = {
            group.title(): group
            for group in common_page.findChildren(QtWidgets.QGroupBox)
        }
        group_positions = {
            title: group.mapTo(common_page, QtCore.QPoint(0, 0)).y()
            for title, group in groups.items()
        }

        assert window.project_function_defaults_widget.isVisibleTo(window)
        assert window.project_common_parameters_widget.embedded is True
        assert not window.project_common_parameters_widget.findChildren(
            QtWidgets.QScrollArea
        )
        assert groups["归一化参数"].isVisibleTo(window)
        assert groups["自动寻峰参数"].isVisibleTo(window)
        assert (
            group_positions["归一化参数"]
            < group_positions["自动寻峰参数"]
        )
        assert group_positions["归一化参数"] < 40

        window.project_tabs.setCurrentWidget(window.project_identity_page)
        qapp.processEvents()
        assert (
            window.datasource_card.sizePolicy().verticalPolicy()
            == QtWidgets.QSizePolicy.Policy.Fixed
        )
        assert (
            window.datasource_card.geometry().bottom()
            < window.project_identity_page.height()
        )
    finally:
        window.close()
        window.deleteLater()


def test_leaving_project_page_can_discard_draft_without_auto_save(
    qapp,
    tmp_path,
    monkeypatch,
):
    project_dir = tmp_path / "Explicit_Save_Project"
    window = MainWindow()
    try:
        window.new_project()
        window.project_name_edit.setText("Explicit Save")
        window.project_output_dir_edit.setText(str(project_dir))
        assert window.save_project() is True
        window.workspace_stack.setCurrentWidget(window.project_page)

        light_index = (
            window.project_common_parameters_widget.light_source_combo.findData(
                "beam_current"
            )
        )
        window.project_common_parameters_widget.light_source_combo.setCurrentIndex(
            light_index
        )
        prompts: list[bool] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _box: (
                prompts.append(True)
                or QtWidgets.QMessageBox.StandardButton.Discard
            ),
        )

        window.switch_workspace_page("spectrum")

        saved = load_project_settings(
            project_dir / "config" / "project.yaml"
        )
        assert prompts == [True]
        assert saved.light_source == "io"
        assert (
            window.project_common_parameters_widget.light_source_combo.currentData()
            == "io"
        )
        assert window.workspace_stack.currentWidget() is window.spectrum_page
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_hidden_curve_pages_defer_cache_loading_until_opened(qapp, monkeypatch):
    window = MainWindow()
    try:
        ps = ProjectSettings(project_name="Lazy cache", output_dir="output")
        temperature_calls: list[dict] = []
        pie_calls: list[dict] = []
        temperature_loads: list[bool] = []
        pie_loads: list[bool] = []
        monkeypatch.setattr(
            window.temperature_page,
            "set_project_settings",
            lambda _ps, **kwargs: temperature_calls.append(kwargs),
        )
        monkeypatch.setattr(
            window.pie_page,
            "set_project_settings",
            lambda _ps, **kwargs: pie_calls.append(kwargs),
        )
        monkeypatch.setattr(
            window.temperature_page,
            "ensure_project_cache_loaded",
            lambda: temperature_loads.append(True) or True,
        )
        monkeypatch.setattr(
            window.pie_page,
            "ensure_project_cache_loaded",
            lambda: pie_loads.append(True) or True,
        )

        window.workspace_stack.setCurrentWidget(window.spectrum_page)
        window._sync_project_settings_to_tool_pages(ps, activate_project_scope=True)

        assert temperature_calls[-1]["load_cached_results"] is False
        assert pie_calls[-1]["load_cached_results"] is False
        assert not temperature_loads
        assert not pie_loads

        window.switch_workspace_page("temperature")
        assert temperature_loads == [True]
        assert not pie_loads
    finally:
        window.deleteLater()


def test_pie_project_parameter_menu_opens_pie_analysis_dialog(qapp, monkeypatch):
    window = MainWindow()
    opened: list[tuple[str, str]] = []
    monkeypatch.setattr(
        AnalysisSettingsDialog,
        "exec",
        lambda dialog: (
            opened.append((dialog.scope, dialog.page))
            or QtWidgets.QDialog.DialogCode.Rejected
        ),
    )
    try:
        window.pie_page.set_project_settings(
            ProjectSettings(project_name="UI test", output_dir="output"),
            activate_project_scope=True,
        )
        window.switch_workspace_page("pie")

        assert window.pie_page.common_params_button.text() == "编辑项目参数…"
        assert window.pie_page.common_params_action.text() == "编辑项目参数…"
        assert window.pie_page.edit_project_action.text() == "修改通用参数…"

        window.pie_page.common_params_action.trigger()

        assert opened == [("project", "pie")]
        assert window.workspace_stack.currentWidget() is window.pie_page
    finally:
        window.deleteLater()


def test_temperature_project_parameter_menu_opens_temperature_dialog(qapp, monkeypatch):
    window = MainWindow()
    opened: list[tuple[str, str]] = []
    monkeypatch.setattr(
        AnalysisSettingsDialog,
        "exec",
        lambda dialog: (
            opened.append((dialog.scope, dialog.page))
            or QtWidgets.QDialog.DialogCode.Rejected
        ),
    )
    try:
        window.temperature_page.set_project_settings(
            ProjectSettings(project_name="UI test", output_dir="output"),
            activate_project_scope=True,
        )
        window.switch_workspace_page("temperature")

        assert window.temperature_page.common_params_button.text() == "编辑项目参数…"
        assert window.temperature_page.common_params_action.text() == "编辑项目参数…"
        assert window.temperature_page.edit_project_action.text() == "修改通用参数…"

        window.temperature_page.common_params_action.trigger()

        assert opened == [("project", "temperature")]
        assert window.workspace_stack.currentWidget() is window.temperature_page
    finally:
        window.deleteLater()


def test_temperature_project_dialog_saves_whitelist_and_syncs_pages(
    qapp, tmp_path, monkeypatch
):
    project_dir = tmp_path / "dialog_project"
    window = MainWindow()

    def accept_with_edits(dialog):
        dialog.function_widget.temperature_photon_check.setChecked(False)
        dialog.function_widget.temp_integration_method_combo.setCurrentIndex(
            dialog.function_widget.temp_integration_method_combo.findData("baseline")
        )
        dialog._settings.light_source = "io"
        dialog._settings.project_name = "Must not overwrite"
        dialog.accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(AnalysisSettingsDialog, "exec", accept_with_edits)
    try:
        window.new_project()
        window.project_name_edit.setText("Dialog project")
        window.project_system_edit.setText("C2H4")
        window.project_output_dir_edit.setText(str(project_dir))
        assert window.save_project() is True
        committed = window.project_settings_manager.snapshot()
        committed.light_source = "beam_current"
        committed = window.project_settings_manager.replace_and_save(committed)
        window._set_project_draft(committed, committed=True)
        window._apply_settings_to_tools(committed)
        window.pie_page.set_pie_source_scope("temporary")

        window.switch_workspace_page("temperature")
        window.temperature_page.common_params_action.trigger()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.temperature_photon_normalize is False
        assert saved.temp_integration_method == "baseline"
        assert saved.project_name == "Dialog project"
        assert saved.light_source == "beam_current"
        assert window.temperature_page.project_settings.temp_integration_method == "baseline"
        assert (
            window.project_function_defaults_widget.temp_integration_method_combo.currentData()
            == "baseline"
        )
        assert window.pie_page.pie_source_scope == "temporary"
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_temperature_project_parameters_open_card_outside_workspace(qapp, monkeypatch):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    opened = []
    monkeypatch.setattr(
        AnalysisSettingsDialog,
        "exec",
        lambda dialog: (
            opened.append((dialog.scope, dialog.page, dialog.parent()))
            or QtWidgets.QDialog.DialogCode.Rejected
        ),
    )

    widget = TemperatureScanDialog(Calibration())
    try:
        widget.set_project_settings(
            ProjectSettings(project_name="Standalone", output_dir="output"),
            activate_project_scope=True,
        )
        widget.open_common_parameters()

        assert opened == [("project", "temperature", widget)]
    finally:
        widget.deleteLater()


def test_temperature_temporary_parameter_change_invalidates_active_view(
    qapp, monkeypatch
):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    def accept_with_edits(dialog):
        dialog.function_widget.temp_integration_method_combo.setCurrentIndex(
            dialog.function_widget.temp_integration_method_combo.findData("baseline")
        )
        dialog.accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(AnalysisSettingsDialog, "exec", accept_with_edits)
    widget = TemperatureScanDialog(Calibration())
    try:
        widget.result_df = pd.DataFrame({"mz": [28.0], "temperature": [700.0]})
        widget.curves = {28.0: {"temperatures": [700.0], "areas": [1.0]}}
        widget.current_mz = 28.0

        widget.open_common_parameters()

        assert widget.temporary_settings.temp_integration_method == "baseline"
        assert widget.result_df.empty
        assert widget.curves == {}
        assert widget.current_mz is None
        assert "临时参数已应用" in widget.inline_status_text.text()
    finally:
        widget.deleteLater()


def test_temperature_temporary_shared_parameter_change_invalidates_active_view(
    qapp,
    monkeypatch,
):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import (
        TemperatureScanDialog,
    )

    def accept_with_shared_edit(dialog):
        dialog._settings.cal_a = 7.5
        dialog.accept()
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(AnalysisSettingsDialog, "exec", accept_with_shared_edit)
    widget = TemperatureScanDialog(Calibration())
    try:
        widget.result_df = pd.DataFrame({"mz": [28.0], "temperature": [700.0]})
        widget.curves = {28.0: {"temperatures": [700.0], "areas": [1.0]}}
        widget.current_mz = 28.0

        widget.open_common_parameters()

        assert widget.temporary_settings.cal_a == pytest.approx(7.5)
        assert widget.result_df.empty
        assert widget.curves == {}
        assert widget.current_mz is None
        assert "临时参数已应用" in widget.inline_status_text.text()
    finally:
        widget.deleteLater()


def test_project_page_has_no_legacy_lifecycle_ui_or_hook(qapp):
    window = MainWindow()
    try:
        assert not hasattr(window, "project_stage_buttons")
        assert not hasattr(window, "project_continue_button")
        assert not hasattr(window, "refresh_project_lifecycle")
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
        assert light_idx >= 0
        assert not hasattr(widget, "mf_md_preset_combo")
        assert not hasattr(widget, "mf_mass_disc_exponent_edit")
        assert not hasattr(widget, "element_checks")
        assert not hasattr(widget, "pie_photon_mode_combo")
        assert not hasattr(widget, "pie_time_normalize_check")
        assert not hasattr(widget, "temperature_photon_check")
        assert not hasattr(widget, "temperature_kr_check")

        widget.light_source_combo.setCurrentIndex(light_idx)
        widget.settings.expansion_factors = {400.0: 1.0, 800.0: 1.2}
        widget.calibration_a_edit.setValue(1.23e-7)
        widget.calibration_b_edit.setValue(2.34e-4)
        widget.calibration_c_edit.setValue(0.56)
        widget.kr_folder_edit.setText("/tmp/kr")
        widget.kr_peak_mode_manual_radio.setChecked(True)
        widget.kr_peak_file_edit.setText("/tmp/kr_peak.csv")
        widget.apply_to_settings(ps)

        assert ps.light_source == "beam_current"
        assert ps.pie_photon_mode == "none"
        assert ps.temperature_photon_normalize is True
        assert ps.temperature_kr_correct is False
        assert ps.mass_discrimination == pytest.approx(1.0)
        assert ps.cal_a == pytest.approx(1.23e-7)
        assert ps.cal_b == pytest.approx(2.34e-4)
        assert ps.cal_c == pytest.approx(0.56)
        assert ps.kr_calibration_folder == "/tmp/kr"
        assert ps.kr_calibration_peak_file == "/tmp/kr_peak.csv"
    finally:
        widget.deleteLater()


def test_temperature_page_owns_analysis_switches(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    scan_dir = tmp_path / "temperature_scan"
    scan_dir.mkdir()
    (scan_dir / "C24110904-0000.txt").write_text("1 2\n", encoding="utf-8")
    project_dir = tmp_path / "project"
    peak_set = create_peak_set(
        project_dir,
        content=b"label,peak_index,mz,left_bound,right_bound\nA,1,1,0,2\n",
        extension=".csv",
        label="test",
        origin="imported",
    )
    settings = NormalizationSettings(light_source="beam_current", expansion_factors={})
    widget = TemperatureScanDialog(
        Calibration(a=0.0, b=1.0, c=0.0),
        settings,
    )
    try:
        assert widget._sidebar.isHidden()
        assert widget.curve_stats.isHidden()
        assert widget.preview_data_button.isHidden()
        assert widget.export_button.isHidden()
        assert widget.export_plot_button.isHidden()
        ps = ProjectSettings(
            output_dir=str(project_dir),
            temperature_scan_folder=str(scan_dir),
            active_peak_set_id=peak_set.peak_set_id,
            manual_peak_file=str(verify_peak_set(project_dir, peak_set)),
            temperature_photon_normalize=False,
            temperature_kr_correct=True,
            temp_replicate_mode="sum",
            light_source="beam_current",
            cal_a=0.0,
            cal_b=1.0,
            cal_c=0.0,
        )
        widget.set_project_settings(ps, activate_project_scope=True)
        assert widget.temperature_photon_check.isChecked() is False
        assert widget.temperature_kr_check.isChecked() is True
        assert widget.replicate_enabled_check.isChecked() is True
        assert widget.replicate_mode_combo.currentData() == "sum"
        assert "Beam Current" in widget.light_source_status_label.text()

        widget.run_analysis()

        assert ps.temperature_photon_normalize is False
        assert ps.temperature_kr_correct is True
        assert widget.project_settings is not ps
        assert widget.project_settings.temperature_kr_correct is True
        assert widget.project_settings.temp_replicate_mode == "sum"
        assert "无法生成曲线" in widget.inline_status_text.text()
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
        assert widget.temporary_settings_source == "project"
        assert widget.temporary_params_title_label.text() == "临时参数 · 不写入项目"
        assert widget.temporary_params_label.text() == "项目参数副本"
        assert widget.temporary_params_button.text() == "编辑临时参数…"
        assert not hasattr(widget, "temporary_params_reload_button")
        assert not hasattr(widget, "copy_project_params_action")
        assert widget._analysis_parameters()["integration_method"] == "baseline"

        params = widget._analysis_parameters()
        assert widget.temporary_params_hint.text() == "只影响本次临时分析，不会写回项目"
        assert params["calibration"] == project_calibration
        assert params["integration_method"] == "baseline"
        assert params["min_intensity"] == 12.0
        assert params["cache_dir"] != tmp_path / "project" / "analysis" / "temperature_scan" / "cache"

        widget.temporary_settings.cal_a = 9.0
        widget.temporary_settings.temp_integration_method = "gaussian"
        assert ps.cal_a == project_calibration.a
        assert ps.temp_integration_method == "baseline"

        widget.temporary_settings_modified = True
        widget._refresh_temperature_source_controls()
        assert widget.temporary_settings_source == "project"
        assert widget.temporary_params_label.text() == "本次已修改"
    finally:
        widget.deleteLater()


def test_temperature_temporary_scope_forwards_imported_peak_file(
    qapp,
    tmp_path,
    monkeypatch,
):
    import bl03u_masstool.frontends.pyqt_app.temperature.dialog as temperature_module
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import (
        TemperatureScanDialog,
    )

    peak_file = tmp_path / "temporary_peak_ranges.csv"
    peak_file.write_text(
        "label,peak_index,mz,left_bound,right_bound\n"
        "CH4,100,16.03,96,104\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_analyze(folder, **kwargs):
        captured["folder"] = folder
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(
        temperature_module,
        "analyze_temperature_folder",
        fake_analyze,
    )

    widget = TemperatureScanDialog(Calibration())
    try:
        widget.set_temperature_source_scope("temporary")
        settings = widget._ensure_temporary_settings()
        settings.manual_peak_file = str(peak_file)
        settings.active_peak_set_id = ""
        settings.temp_peak_source = "manual"

        params = widget._analysis_parameters()

        assert params["effective_peak_source"] == "manual"
        assert params["manual_peak_path"] == str(peak_file.resolve())
        assert widget._validate_analysis_parameters(params) is True

        widget._analyze_temperature_folder_with_params("temporary-scan", params)

        assert captured["folder"] == "temporary-scan"
        assert captured["manual_peak_path"] == str(peak_file.resolve())
    finally:
        widget.deleteLater()


def test_temperature_temporary_project_snapshot_uses_registered_peak_set(
    qapp,
    tmp_path,
):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import (
        TemperatureScanDialog,
    )

    project_dir = tmp_path / "project"
    scan_dir = tmp_path / "temperature_scan"
    scan_dir.mkdir()
    peak_set = create_peak_set(
        project_dir,
        content=(
            b"label,peak_index,mz,left_bound,right_bound\n"
            b"CH4,100,16.03,96,104\n"
        ),
        extension=".csv",
        label="project peaks",
        origin="imported",
    )
    peak_path = verify_peak_set(project_dir, peak_set)
    settings = ProjectSettings(
        project_name="Temporary project snapshot",
        output_dir=str(project_dir),
        temperature_scan_folder=str(scan_dir),
        manual_peak_file=str(peak_path),
        active_peak_set_id=peak_set.peak_set_id,
        temp_peak_source="manual",
    )

    widget = TemperatureScanDialog(Calibration())
    try:
        widget.set_project_settings(settings, activate_project_scope=True)
        widget.set_temperature_source_scope("temporary")

        params = widget._analysis_parameters()

        assert params["effective_peak_source"] == "manual"
        assert params["manual_peak_path"] == str(peak_path)
        assert params["peak_set_origin"] == "imported"
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
            "全部光子能量 (3)",
            "8eV",
            "9.5eV",
            "14.6-14.8eV",
        ]
        assert widget._current_temperature_folder() == str(root)
        assert widget._selected_view_folders() == [
            (8.0, str(low)),
            (9.5, str(high)),
            (14.6, str(range_energy)),
        ]
        assert widget._generation_analysis_folders() == [
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
        assert widget._selected_view_folders() == [(9.5, str(high))]
        assert widget._generation_analysis_folders() == [(9.5, str(high))]
        assert "9.5eV" in widget.summary_data_label.text()
        assert "请选择本次要处理的光子能量" in widget.scan_folder_combo.toolTip()
        assert "选择“全部光子能量”" in widget.scan_folder_combo.toolTip()
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


def test_temperature_display_keeps_close_precise_peaks_separate_across_energies(qapp):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    def curve(exact_mz: float, areas: list[float]) -> dict:
        rows = pd.DataFrame(
            {
                "temperature": [650.0, 750.0, 850.0],
                "file": ["a.txt", "b.txt", "c.txt"],
                "mz": [exact_mz] * 3,
                "area": areas,
                "raw_area": areas,
                "photon_normalized_area": areas,
                "expansion_lambda": [1.0, 1.0, 1.0],
                "integration_method": ["sum_counts"] * 3,
                "species": [""] * 3,
            }
        )
        return {
            "mz": exact_mz,
            "mz_rounded": 228,
            "mz_exact_mean": exact_mz,
            "curve_key": exact_mz,
            "has_nominal_collision": True,
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
        low_8 = curve(228.020, [1.0, 2.0, 3.0])
        high_8 = curve(228.100, [10.0, 20.0, 30.0])
        low_9 = curve(228.022, [4.0, 5.0, 6.0])
        high_9 = curve(228.098, [40.0, 50.0, 60.0])
        widget.energy_results = [
            {
                "energy": 8.0,
                "folder": "/tmp/8eV",
                "folder_label": "8eV",
                "curves": {228.020: low_8, 228.100: high_8},
            },
            {
                "energy": 9.0,
                "folder": "/tmp/9eV",
                "folder_label": "9eV",
                "curves": {228.022: low_9, 228.098: high_9},
            },
        ]
        widget.result_df = pd.concat(
            [curve["rows"] for curve in (low_8, high_8, low_9, high_9)],
            ignore_index=True,
        )

        widget.curves = widget._build_display_curves()
        curve_keys = sorted(widget.curves, key=float)

        assert len(curve_keys) == 2
        assert curve_keys == pytest.approx([228.021, 228.099])
        assert [
            len(widget.curves[curve_key]["energy_curves"])
            for curve_key in curve_keys
        ] == [2, 2]
        assert widget.curves[curve_keys[0]]["areas"] == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
        assert widget.curves[curve_keys[1]]["areas"] == [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
        ]
        assert widget._mz_text_for_key(curve_keys[0]) == "228.021000"
        assert widget._mz_text_for_key(curve_keys[0]) != widget._mz_text_for_key(
            curve_keys[1]
        )

        widget.populate_mz_list()
        assert widget.current_mz in curve_keys
        assert widget.energy_interval_table.rowCount() == 2
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
        widget.set_project_settings(
            ps,
            activate_project_scope=True,
            load_cached_results=False,
        )
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
        assert widget.plot_widget.axes.get_title() == "m/z 70.000000 温度响应曲线"
        assert widget.curve_table.columnCount() == 3
        assert widget.curve_table.horizontalHeaderItem(0).text() == "650.0"
        assert widget.curve_table.verticalHeaderItem(0).text() == "精确m/z"
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
        assert first["analysis_provenance"]["cache_key"]
        assert first["analysis_provenance"]["peak_source"] == "sum"
        assert first["analysis_provenance"]["manual_peak_file"] is None
        assert calls["count"] == 1
        assert (Path(ps.output_dir) / "analysis" / "temperature_scan" / "cache").is_dir()

        def fail_analyze(folder, params):
            raise AssertionError("cache miss caused recomputation")

        widget._analyze_temperature_folder_with_params = fail_analyze
        second = widget._analyze_temperature_folders_with_cache(folders, params)

        assert second["from_cache"] is True
        assert second["analysis_provenance"]["cache_key"] == first["analysis_provenance"]["cache_key"]
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
    project_dir = tmp_path / "project"
    peak_set = create_peak_set(
        project_dir,
        content=b"label,peak_index,mz,left_bound,right_bound\nA,1,70,0,2\n",
        extension=".csv",
        label="test",
        origin="imported",
    )
    ps = ProjectSettings(
        project_name="Autoload Cache",
        output_dir=str(project_dir),
        temperature_scan_folder=str(root),
        active_peak_set_id=peak_set.peak_set_id,
        manual_peak_file=str(verify_peak_set(project_dir, peak_set)),
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

        assert reader.result_df.empty
        assert 70 in reader.curves
        assert reader.curves[70]["areas"] == [1.0, 2.0, 3.0]
        assert reader.curve_database_dataset_id is not None
        assert reader.current_mz == 70
        assert any(
            marker in reader.inline_status_text.text()
            for marker in ("已自动载入", "缓存", "SQLite")
        )
        request_id = reader._autoload_request_id
        worker = reader._autoload_worker
        assert reader.ensure_project_cache_loaded() is True
        assert reader._autoload_request_id == request_id
        assert reader._autoload_worker is worker
    finally:
        if reader._autoload_worker is not None and reader._autoload_worker.isRunning():
            reader._autoload_worker.wait(1000)
        reader.deleteLater()


@pytest.mark.parametrize(
    ("storage_mode", "expects_database", "expects_lazy_mapping"),
    [
        ("legacy", False, False),
        ("shadow", True, False),
        ("sqlite", True, True),
    ],
)
def test_temperature_project_curve_storage_modes(
    qapp,
    tmp_path,
    storage_mode,
    expects_database,
    expects_lazy_mapping,
):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.curve_database import project_curve_database_path
    from bl03u_masstool.core.curve_repository import RepositoryCurveMapping
    from bl03u_masstool.core.temperature_scan import build_temperature_curves
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

    source = tmp_path / f"temperature-{storage_mode}"
    source.mkdir()
    settings = ProjectSettings(
        project_name=f"Temperature {storage_mode}",
        output_dir=str(tmp_path / f"project-{storage_mode}"),
        temperature_scan_folder=str(source),
        curve_storage_mode=storage_mode,
    )
    rows = pd.DataFrame(
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
        }
    )
    dialog = TemperatureScanDialog(Calibration())
    try:
        dialog.set_project_settings(
            settings,
            activate_project_scope=True,
            load_cached_results=False,
        )
        dialog.on_analysis_complete(
            {
                "result_df": rows,
                "energy_results": [
                    {
                        "energy": 10.0,
                        "folder": str(source),
                        "folder_label": source.name,
                        "result_df": rows,
                        "curves": build_temperature_curves(rows),
                    }
                ],
                "from_cache": True,
                "analysis_provenance": {
                    "cache_key": f"mode-{storage_mode}",
                    "peak_source": "manual_peak_file",
                },
            }
        )

        assert project_curve_database_path(settings).exists() is expects_database
        assert isinstance(dialog.curves, RepositoryCurveMapping) is expects_lazy_mapping
        assert dialog.result_df.empty is expects_lazy_mapping
        if storage_mode == "sqlite":
            assert dialog.curve_source_label.text().startswith(
                "当前曲线：项目 SQLite · 有效"
            )
            assert "有效" in dialog.curve_source_label.text()
            assert dialog.curve_source_label.property("sourceState") == "valid"
        else:
            assert "项目内存" in dialog.curve_source_label.text()
    finally:
        dialog.deleteLater()


def test_temperature_sqlite_list_does_not_eagerly_load_all_curves(
    qapp,
    tmp_path,
):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.curve_database import store_curve_dataset_version
    from bl03u_masstool.core.curve_repository import (
        RepositoryCurveMapping,
        SQLiteCurveRepository,
    )
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import (
        TemperatureScanDialog,
    )

    curves = {}
    for index in range(12):
        exact_mz = 50.0 + index / 100
        curve_class = "formation" if index % 2 == 0 else "consumption"
        rows = pd.DataFrame(
            {
                "temperature": [650.0, 750.0, 850.0],
                "scan_energy": [10.0] * 3,
                "mz": [exact_mz] * 3,
                "mz_rounded": [round(exact_mz)] * 3,
                "area": [1.0, 2.0, 3.0],
                "raw_area": [1.0, 2.0, 3.0],
                "photon_normalized_area": [1.0, 2.0, 3.0],
                "expansion_lambda": [1.0, 1.0, 1.0],
                "integration_method": ["sum_counts"] * 3,
                "curve_class": [curve_class] * 3,
                "curve_class_label": [
                    "生成(升高)" if curve_class == "formation" else "消耗(降低)"
                ]
                * 3,
                "curve_class_reason": ["测试"] * 3,
            }
        )
        curves[exact_mz] = {
            "mz": exact_mz,
            "mz_rounded": round(exact_mz),
            "temperatures": rows["temperature"].tolist(),
            "areas": rows["area"].tolist(),
            "curve_class": curve_class,
            "rows": rows,
        }

    database_path = tmp_path / "temperature.sqlite"
    dataset_id = store_curve_dataset_version(
        database_path,
        curve_type="temperature",
        curves=curves,
        dataset_group="temperature:project",
        analysis_key="lazy-list",
        name="Lazy temperature list",
    )
    repository = SQLiteCurveRepository(database_path, dataset_id, cache_size=4)
    mapping = RepositoryCurveMapping(repository)
    original_load_channel = repository.load_channel
    loaded_channel_ids = []

    def counted_load_channel(channel_id):
        loaded_channel_ids.append(channel_id)
        return original_load_channel(channel_id)

    repository.load_channel = counted_load_channel
    dialog = TemperatureScanDialog(Calibration())
    try:
        dialog.curve_repository = repository
        dialog.curves = mapping
        dialog.populate_mz_list()

        # Selecting the first visible item may load that one curve for plotting;
        # building and grouping the list must not load all twelve payloads.
        assert len(set(loaded_channel_ids)) <= 1
        assert dialog.mz_list.topLevelItemCount() == 2
    finally:
        dialog.deleteLater()


def test_temperature_sqlite_energy_selector_filters_lazy_curve(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.curve_database import store_curve_dataset_version
    from bl03u_masstool.core.curve_repository import (
        RepositoryCurveMapping,
        SQLiteCurveRepository,
    )
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import (
        TemperatureScanDialog,
    )

    root = tmp_path / "temperature"
    energy_8 = root / "8.0eV"
    energy_11 = root / "11.0eV"
    energy_8.mkdir(parents=True)
    energy_11.mkdir()
    (energy_8 / "scan.txt").write_text("1\n", encoding="utf-8")
    (energy_11 / "scan.txt").write_text("1\n", encoding="utf-8")
    exact_mz = 97.09
    rows = pd.DataFrame(
        {
            "temperature": [300.0, 400.0, 300.0, 400.0],
            "scan_energy": [8.0, 8.0, 11.0, 11.0],
            "photon_energy": [8.0, 8.0, 11.0, 11.0],
            "scan_folder": ["8.0eV", "8.0eV", "11.0eV", "11.0eV"],
            "mz": [exact_mz] * 4,
            "mz_rounded": [97] * 4,
            "area": [1.0, 2.0, 10.0, 20.0],
            "raw_area": [1.0, 2.0, 10.0, 20.0],
            "photon_normalized_area": [1.0, 2.0, 10.0, 20.0],
            "expansion_lambda": [1.0] * 4,
            "integration_method": ["sum_counts"] * 4,
            "curve_class": ["formation"] * 4,
            "curve_class_label": ["生成(升高)"] * 4,
            "curve_class_reason": ["测试"] * 4,
        }
    )
    database_path = tmp_path / "temperature.sqlite"
    dataset_id = store_curve_dataset_version(
        database_path,
        curve_type="temperature",
        curves={
            exact_mz: {
                "mz": exact_mz,
                "mz_rounded": 97,
                "temperatures": rows["temperature"].tolist(),
                "areas": rows["area"].tolist(),
                "curve_class": "formation",
                "rows": rows,
            }
        },
        dataset_group="temperature:project",
        analysis_key="energy-filter",
        name="Energy-filter temperature list",
    )
    repository = SQLiteCurveRepository(database_path, dataset_id, cache_size=4)
    dialog = TemperatureScanDialog(Calibration())
    try:
        dialog.set_project_settings(
            ProjectSettings(temperature_scan_folder=str(root)),
            activate_project_scope=True,
            load_cached_results=False,
        )
        dialog.curve_repository = repository
        dialog.curves = RepositoryCurveMapping(repository)
        dialog._refresh_temperature_folder_options()
        dialog.populate_mz_list(preferred_mz=exact_mz)

        all_curve = dialog._repository_curve_for_energy_selection(
            dialog.curves[exact_mz]
        )
        assert len(all_curve["energy_curves"]) == 2
        assert all_curve["areas"] == [1.0, 2.0, 10.0, 20.0]

        selected_index = dialog.scan_folder_combo.findData(str(energy_11))
        assert selected_index >= 0
        dialog.scan_folder_combo.setCurrentIndex(selected_index)
        selected_curve = dialog._repository_curve_for_energy_selection(
            dialog.curves[exact_mz]
        )

        assert selected_curve["temperatures"] == [300.0, 400.0]
        assert selected_curve["areas"] == [10.0, 20.0]
        assert selected_curve["energy_curves"] == []
        assert "2 个点" in dialog.current_curve_metric_label.text()
        assert "当前能量 11.0eV" in dialog.summary_label.text()
        current_item = dialog.find_curve_tree_item(exact_mz)
        assert current_item is not None
        assert "当前 11.0eV" in current_item.text(0)
        assert "2能量/4点" not in current_item.text(0)
    finally:
        dialog.deleteLater()


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


def test_temperature_sqlite_cache_hit_is_not_invalidated_before_activation(
    qapp,
    tmp_path,
    monkeypatch,
):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.frontends.pyqt_app.temperature.dialog import (
        TemperatureScanDialog,
    )

    widget = TemperatureScanDialog(Calibration())
    activated = []
    token = {"request_id": 1}
    try:
        widget.set_project_settings(
            ProjectSettings(output_dir=str(tmp_path)),
            activate_project_scope=True,
            load_cached_results=False,
        )
        widget._autoload_cache_token = token
        monkeypatch.setattr(
            widget,
            "_discard_stale_temperature_curves",
            lambda _cache_key: pytest.fail(
                "A verified SQLite cache hit must not be invalidated"
            ),
        )
        monkeypatch.setattr(
            widget,
            "_activate_sqlite_temperature_dataset",
            lambda dataset_id, **kwargs: activated.append(dataset_id),
        )

        widget._on_project_temperature_cache_loaded(
            {
                "cache_key": "analysis-key",
                "sqlite_dataset_id": 42,
                "dataset_metadata": {},
            },
            token,
        )

        assert activated == [42]
    finally:
        widget.deleteLater()


def test_pie_run_uses_project_photon_mode_not_hidden_compatibility_control(
    qapp, tmp_path, monkeypatch
):
    from bl03u_masstool.core.calibration import Calibration
    from bl03u_masstool.core.normalization import NormalizationSettings
    from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog

    pie_dir = tmp_path / "pie_scan"
    pie_dir.mkdir()
    (pie_dir / "8.0eV.txt").write_text("1 2\n", encoding="utf-8")
    project_dir = tmp_path / "project"
    peak_set = create_peak_set(
        project_dir,
        content=b"label,peak_index,mz,left_bound,right_bound\nA,1,1,0,2\n",
        extension=".csv",
        label="test",
        origin="imported",
    )

    widget = PIESpeciesFitDialog(
        Calibration(a=0.0, b=1.0, c=0.0),
        NormalizationSettings(light_source="beam_current"),
    )
    try:
        ps = ProjectSettings(
            project_name="PIE Switch",
            output_dir=str(project_dir),
            pie_scan_folder=str(pie_dir),
            pie_photon_mode="none",
            active_peak_set_id=peak_set.peak_set_id,
            manual_peak_file=str(verify_peak_set(project_dir, peak_set)),
            cal_a=0.0,
            cal_b=1.0,
            cal_c=0.0,
        )
        manager = ProjectSettingsManager()
        manager.set_project_path(project_dir)
        manager.replace_and_save(ps)
        widget.set_project_settings(ps, activate_project_scope=True)
        widget.photon_mode_combo.setCurrentIndex(
            widget.photon_mode_combo.findData("first")
        )

        assert widget.photon_correction_check.isChecked() is True
        assert widget._current_photon_mode() == "none"
        assert widget._pie_cache_static_parameters(None)["photon_mode"] == "none"

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
    finally:
        ProjectSettingsManager().clear_project_path()
        widget.deleteLater()


def test_save_project_uses_selected_parent_directory_for_new_project(qapp, tmp_path, monkeypatch):
    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir()

    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    try:
        window.new_project()
        window.project_name_edit.setText("test")
        window.project_system_edit.setText("test")
        window.project_description_edit.setText("test")
        window.project_output_dir_edit.setText(str(downloads_dir))

        assert window.save_project() is True

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


def test_branded_empty_state_preserves_foreground_controls(qapp):
    state = AnalysisEmptyState(
        title="空状态标题",
        description="空状态说明",
        action_text="选择数据",
    )
    clicked: list[bool] = []
    state.browse_requested.connect(lambda: clicked.append(True))
    try:
        assert not state._watermark.isNull()
        assert state.title_label.text() == "空状态标题"
        assert state.description_label.text() == "空状态说明"
        assert state.primary_button.text() == "选择数据"

        state.primary_button.click()

        assert clicked == [True]
    finally:
        state.deleteLater()


def test_spectrum_empty_state_hides_side_panel_until_spectrum_is_ready(qapp):
    window = MainWindow()
    try:
        assert window._spectrum_empty_state.title_label.text() == "尚未加载质谱数据"
        assert not window._spectrum_empty_state.isHidden()
        assert window.widget_3.isHidden()

        window.setup_plots(
            np.array([100.0, 101.0, 102.0]),
            np.array([1.0, 10.0, 2.0]),
        )
        qapp.processEvents()

        assert window._spectrum_empty_state.isHidden()
        assert not window.widget_3.isHidden()

        window.clear_widgets()
        qapp.processEvents()

        assert not window._spectrum_empty_state.isHidden()
        assert window.widget_3.isHidden()
        assert window.graph_layout.indexOf(window._spectrum_empty_state) >= 0
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_temperature_and_pie_reuse_branded_empty_state_without_replacing_controls(qapp):
    window = MainWindow()
    try:
        temperature_state = window.temperature_page._empty_state
        pie_state = window.pie_page._empty_state

        assert temperature_state.title_label.text() == "尚未生成温度曲线"
        assert temperature_state.primary_button.text() == "选择数据"
        assert not temperature_state._watermark.isNull()
        assert pie_state.title_label.text() == "尚未生成 PIE 曲线"
        assert pie_state.primary_button.text() == "选择数据"
        assert not pie_state._watermark.isNull()
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

        assert window.save_project() is True

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

        assert window.save_project() is True

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
        window.new_project()
        window.project_name_edit.setText("aaa")
        window.project_system_edit.setText("aaa")
        window.project_description_edit.setText("aaa")
        window.project_output_dir_edit.setText(str(downloads_dir))

        assert window.save_project() is True

        project_dir = downloads_dir / "aaa"
        assert (project_dir / "analysis").exists()
        assert (project_dir / "config" / "project.yaml").exists()
        assert window.project_output_dir_edit.text() == str(project_dir)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_save_project_does_not_materialize_project_data_sources(qapp, tmp_path, monkeypatch):
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
        window.new_project()
        window.project_name_edit.setText("managed")
        window.project_system_edit.setText("C6H6")
        window.project_output_dir_edit.setText(str(downloads_dir))
        window.project_temperature_folder_edit.setText(str(temp_source))
        window.project_pie_folder_edit.setText(str(pie_source))
        assert window.save_project() is True

        project_dir = downloads_dir / "managed"
        saved = load_project_settings(project_dir / "config" / "project.yaml")

        assert saved.temperature_scan_folder == ""
        assert saved.pie_scan_folder == ""
        assert saved.kr_calibration_folder == ""
        assert saved.kr_calibration_peak_file == ""
        for destination in (
            project_dir / "raw_data" / "temperature_scan",
            project_dir / "raw_data" / "pie_scan",
            project_dir / "raw_data" / "kr_calibration",
        ):
            assert not destination.exists() or not any(destination.iterdir())
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
        saved = load_project_settings(project_dir / "config" / "project.yaml")
        migrated_peak = Path(window.project_manual_peak_edit.text())
        assert saved.active_peak_set_id
        assert migrated_peak.parent == (
            project_dir
            / "analysis"
            / "spectrum"
            / "manual_peaks"
            / "peak_sets"
        )
        assert migrated_peak.read_bytes() == manual_peak.read_bytes()
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
        assert (
            window.project_function_defaults_widget.mf_md_preset_combo.currentText()
            == "30 Torr (Catalysis)"
        )
        assert (
            window.project_function_defaults_widget.mf_mass_disc_exponent_edit.value()
            == pytest.approx(0.75148)
        )
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
        factory_exponent = load_factory_project_settings().mf_mass_disc_exponent
        assert window.mole_fraction_page._mass_disc_exponent == pytest.approx(factory_exponent)
        assert window.pics_page.spin_md_exponent.value() == pytest.approx(factory_exponent)
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


def test_spectrum_workbench_ad_hoc_source_is_session_only(qapp, tmp_path, monkeypatch):
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
        assert saved.single_spectrum_file == ""
        assert window.lineEdit.isReadOnly()
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_save_and_apply_persists_common_parameters_to_project_file(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Params"
    window = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    try:
        window.new_project()
        window.project_name_edit.setText("Project Params")
        window.project_system_edit.setText("C6H6")
        window.project_output_dir_edit.setText(str(project_dir))
        assert window.save_project() is True

        light_idx = window.project_common_parameters_widget.light_source_combo.findData("beam_current")
        preset_idx = (
            window.project_function_defaults_widget.mf_md_preset_combo.findData(
                "150 Torr (Combustion)"
            )
        )
        assert light_idx >= 0
        assert preset_idx >= 0
        assert not hasattr(window.project_common_parameters_widget, "pie_photon_mode_combo")
        assert not hasattr(window.project_common_parameters_widget, "temperature_photon_check")
        assert not hasattr(window.project_common_parameters_widget, "temperature_kr_check")

        window.project_common_parameters_widget.light_source_combo.setCurrentIndex(light_idx)
        window.project_calibration_edits[0].setValue(4.56e-7)
        window.project_calibration_edits[1].setValue(7.89e-4)
        window.project_calibration_edits[2].setValue(0.12)
        window.project_function_defaults_widget.mf_md_preset_combo.setCurrentIndex(
            preset_idx
        )
        window.project_function_defaults_widget.mf_mass_disc_exponent_edit.setValue(
            0.76155
        )

        window.save_and_apply_project_settings()
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
        window._set_project_draft(ps, committed=True)
        window.set_spectrum_source_scope("project", apply_project=False)

        # Round 0: 初始状态 — 验证加载的定标参数
        window._apply_project_runtime_settings(ps)
        cal0 = window.current_calibration()
        assert cal0.a == pytest.approx(1.1e-7)
        assert cal0.b == pytest.approx(2.2e-4)
        assert cal0.c == pytest.approx(3.3)

        # Round 1: 模拟用户在项目管理页面修改定标参数
        edits = window.project_calibration_edits
        edits[0].setValue(9.9e-7)
        edits[1].setValue(8.8e-4)
        edits[2].setValue(7.7)

        # 触发 editingFinished (模拟 spinbox 失去焦点)
        for edit in edits:
            edit.editingFinished.emit()

        # 失焦与切换页签都不保存；显式统一事务后才更新运行时。
        unchanged = load_project_settings(project_dir / "config" / "project.yaml")
        assert unchanged.cal_a == pytest.approx(1.1e-7)
        assert window.save_and_apply_project_settings() is True
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
        edits[0].setValue(5.5e-7)
        edits[1].setValue(4.4e-4)
        edits[2].setValue(3.3)

        # 触发 editingFinished
        for edit in edits:
            edit.editingFinished.emit()

        assert window.save_and_apply_project_settings() is True
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
        assert edits[0].value() == pytest.approx(5.5e-7)
        assert edits[1].value() == pytest.approx(4.4e-4)
        assert edits[2].value() == pytest.approx(3.3)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_project_calibration_points_load_and_save_with_coefficients(qapp, tmp_path):
    project_dir = tmp_path / "Calibration_Points_Project"
    initial_points = [
        {"tof": 10.0, "mz": 2.0},
        {"tof": 20.0, "mz": 5.0},
        {"tof": 30.0, "mz": 10.0},
    ]
    ps = ProjectSettings(
        project_name="Calibration points",
        output_dir=str(project_dir),
        cal_a=0.01,
        cal_b=0.0,
        cal_c=1.0,
        calibration_points=initial_points,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        loaded = window.project_settings_manager.get()
        window.set_spectrum_source_scope("project", apply_project=False)
        window._apply_project_runtime_settings(loaded)

        assert window.region.rowCount() == 4
        assert window.region.item(0, 0).text() == "10"
        assert window.region.item(2, 1).text() == "10"
        assert window.region.item(3, 0).text() == ""

        updated_points = [(11.0, 3.0), (21.0, 6.0), (31.0, 11.0)]
        window.lineEdit_4.setText("0.02")
        window.lineEdit_5.setText("0.1")
        window.lineEdit_6.setText("0.5")
        window._sync_calibration_to_project_settings(updated_points)

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.cal_a == pytest.approx(0.01)

        window._persist_project_calibration(
            window.current_calibration(),
            updated_points,
        )
        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.cal_a == pytest.approx(0.02)
        assert saved.cal_b == pytest.approx(0.1)
        assert saved.cal_c == pytest.approx(0.5)
        assert saved.calibration_points == [
            {"tof": 11.0, "mz": 3.0},
            {"tof": 21.0, "mz": 6.0},
            {"tof": 31.0, "mz": 11.0},
        ]
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_temporary_spectrum_calibration_is_independent_from_project(qapp, tmp_path):
    project_dir = tmp_path / "Temporary_Calibration_Project"
    points = [
        {"tof": 10.0, "mz": 2.0},
        {"tof": 20.0, "mz": 5.0},
        {"tof": 30.0, "mz": 10.0},
    ]
    ps = ProjectSettings(
        project_name="Temporary calibration",
        output_dir=str(project_dir),
        cal_a=0.01,
        cal_b=0.0,
        cal_c=1.0,
        calibration_points=points,
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        loaded = window.project_settings_manager.get()
        window._apply_project_runtime_settings(loaded)
        assert window.spectrum_source_scope == "custom"
        assert window.commonParamsButton.text() == "临时参数"

        window.lineEdit_4.setText("0.02")
        window.lineEdit_5.setText("0.1")
        window.lineEdit_6.setText("0.5")
        assert window._sync_calibration_to_project_settings()

        unchanged = load_project_settings(project_dir / "config" / "project.yaml")
        assert unchanged.cal_a == pytest.approx(0.01)
        assert unchanged.cal_b == pytest.approx(0.0)
        assert unchanged.cal_c == pytest.approx(1.0)

        window.set_spectrum_source_scope("project", apply_project=False)
        assert window.commonParamsButton.text() == "项目参数"
        assert window.current_calibration().a == pytest.approx(0.01)
        assert window.current_calibration().b == pytest.approx(0.0)
        assert window.current_calibration().c == pytest.approx(1.0)

        window.set_spectrum_source_scope("custom", apply_project=False)
        assert window.commonParamsButton.text() == "临时参数"
        assert window.current_calibration().a == pytest.approx(0.02)
        assert window.current_calibration().b == pytest.approx(0.1)
        assert window.current_calibration().c == pytest.approx(0.5)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_calibration_point_table_grows_from_trailing_row_and_deletes(qapp, monkeypatch):
    window = MainWindow()
    try:
        assert not hasattr(window, "addCalibrationRowButton")
        assert "双击末行新增" in window.calibrationTableHint.text()
        window._set_calibration_points_table(
            [(float(index), float(index + 10)) for index in range(1, 8)]
        )
        assert window.region.rowCount() == 8

        window.region.item(7, 0).setText("8")
        window.region.item(7, 1).setText("18")
        assert window.region.rowCount() == 9
        assert window.region.item(8, 0).text() == ""

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        window.region.selectRow(7)
        window.delete_selected_calibration_points()

        assert window.region.rowCount() == 8
        assert window._calibration_points_from_table()[-1] == (7.0, 17.0)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_calibration_table_pastes_two_columns_and_expands(qapp):
    window = MainWindow()
    try:
        window._set_calibration_points_table([(10.0, 2.0)])
        window.region.setCurrentCell(1, 0)
        QtWidgets.QApplication.clipboard().setText(
            "20\t5\n30\t10\n40\t17"
        )

        window.paste_calibration_points()

        assert window._calibration_points_from_table() == [
            (10.0, 2.0),
            (20.0, 5.0),
            (30.0, 10.0),
            (40.0, 17.0),
        ]
        assert window.region.rowCount() == 5
        assert window.region.item(4, 0).text() == ""
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_double_clicking_region_adds_calibration_point_on_calibration_tab(
    qapp,
    monkeypatch,
):
    window = MainWindow()
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("50.0", True),
    )
    try:
        tof = np.arange(100.0, 121.0)
        intensity = np.zeros(tof.size)
        intensity[8] = 100.0
        window.setup_plots(tof, intensity)
        window.peakResult.setCurrentWidget(window.tab)

        window.selection_region.calibrationDoubleClicked.emit()

        assert (108.0, 50.0) in window._calibration_points_from_table()
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_calibration_calculate_previews_before_project_update(qapp, tmp_path):
    project_dir = tmp_path / "Calibration_Preview_Project"
    original = ProjectSettings(
        project_name="Calibration preview",
        output_dir=str(project_dir),
        cal_a=0.0,
        cal_b=1.0,
        cal_c=0.0,
        calibration_points=[
            {"tof": 1.0, "mz": 1.0},
            {"tof": 2.0, "mz": 2.0},
            {"tof": 3.0, "mz": 3.0},
        ],
    )
    save_project_settings(original, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.set_spectrum_source_scope("project", apply_project=False)
        window._apply_project_runtime_settings(window.project_settings_manager.get())
        fit_points = [
            (10.0, 3.0),
            (20.0, 7.0),
            (30.0, 13.0),
            (40.0, 21.0),
        ]
        window._set_calibration_points_table(fit_points)

        window.calculate()

        unchanged = load_project_settings(project_dir / "config" / "project.yaml")
        assert unchanged.cal_a == pytest.approx(0.0)
        assert unchanged.cal_b == pytest.approx(1.0)
        assert window._calibration_candidate is not None
        assert "尚未应用" in window.label_4.text()

        window.apply_calibration_candidate_to_project()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.cal_a == pytest.approx(0.01)
        assert saved.cal_b == pytest.approx(0.1)
        assert saved.cal_c == pytest.approx(1.0)
        assert saved.calibration_points == [
            {"tof": tof, "mz": mz}
            for tof, mz in fit_points
        ]
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_project_calibration_apply_refreshes_loaded_custom_spectrum(qapp, tmp_path):
    from bl03u_masstool.core.calibration import Calibration

    project_dir = tmp_path / "Calibration_Custom_Refresh"
    original = ProjectSettings(
        project_name="Custom calibration refresh",
        output_dir=str(project_dir),
        cal_a=0.0,
        cal_b=1.0,
        cal_c=0.0,
        calibration_points=[
            {"tof": 1.0, "mz": 1.0},
            {"tof": 2.0, "mz": 2.0},
            {"tof": 3.0, "mz": 3.0},
        ],
    )
    save_project_settings(original, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.set_spectrum_source_scope("custom", apply_project=False)
        window._activate_spectrum_calibration(
            Calibration(a=0.0, b=1.0, c=0.0),
            [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        )
        tof = np.arange(1.0, 8.0)
        window.setup_plots(tof, np.arange(tof.size, dtype=float))
        candidate = Calibration(a=0.0, b=2.0, c=5.0)
        window._set_calibration_candidate(
            {
                "calibration": candidate,
                "points": [(1.0, 7.0), (2.0, 9.0), (3.0, 11.0)],
                "r2": 1.0,
            }
        )

        window.apply_calibration_candidate_to_project()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.cal_b == pytest.approx(2.0)
        assert saved.cal_c == pytest.approx(5.0)
        assert window.current_plot_axis_x.tolist() == pytest.approx(
            (2.0 * tof + 5.0).tolist()
        )
        temporary_settings, _ = window._ensure_temporary_spectrum_settings()
        assert temporary_settings.cal_b == pytest.approx(2.0)
        assert temporary_settings.cal_c == pytest.approx(5.0)
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_loaded_spectrum_axis_updates_after_second_project_calibration_edit(qapp, tmp_path, monkeypatch):
    """Regression: the displayed m/z axis follows each project calibration edit."""
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

    def fail_warning(*args, **kwargs):
        pytest.fail(f"Unexpected warning dialog: {args[2] if len(args) > 2 else args}")

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", fail_warning)

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)
        window._load_project_settings_to_parameter_widgets(ps)
        window.set_spectrum_source_scope("project", apply_project=False)
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
        edits = window.project_calibration_edits
        edits[0].setValue(0.0)
        edits[1].setValue(2.0)
        edits[2].setValue(5.0)
        for edit in edits:
            edit.editingFinished.emit()
        assert window.save_and_apply_project_settings() is True

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


def test_project_page_edits_apply_only_after_unified_save(qapp, tmp_path, monkeypatch):
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
        window._set_project_draft(ps, committed=True)
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

        window._on_project_tab_changed(2)
        unchanged = load_project_settings(project_dir / "config" / "project.yaml")
        assert unchanged.peak_algorithm == "legacy"
        assert window.project_settings_manager.snapshot().peak_algorithm == "legacy"

        assert window.save_and_apply_project_settings() is True
        window.switch_workspace_page("spectrum")

        active = window.project_settings_manager.snapshot()
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
        assert not hasattr(window, "openProjectPeaksButton")
        assert not hasattr(window, "publishProjectPeaksButton")

        assert window.peakDataMenuButton.menu() is not None
        assert [action.text() for action in window.peakDataMenuButton.menu().actions()] == [
            "预览外部卡峰范围…",
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
        assert window.peakProjectActionPanel.layout().itemAt(1).widget() is window.saveProjectPeaksButton
        assert window.saveProjectPeaksButton.text() == "保存卡峰范围"
        assert not hasattr(window, "peakProjectMenuButton")
        assert [
            window.peakData.horizontalHeaderItem(column).text()
            for column in range(window.peakData.columnCount())
        ] == ["物种", "TOF", "m/z", "强度", "左边界", "右边界"]
        assert not window.saveProjectPeaksButton.isEnabled()
        assert not window.updatePeakRangeAction.isEnabled()
        assert not window.deleteSelectedPeakAction.isEnabled()
    finally:
        window.deleteLater()


def test_workbench_peak_assessment_controls_do_not_compete_with_summary(qapp):
    window = MainWindow()
    try:
        window.resize(1180, 700)
        window.setup_plots(
            np.array([100.0, 101.0, 102.0]),
            np.array([1.0, 3.0, 1.0]),
        )
        window.peakRangeAssessmentLabel.setText(
            "适配检查 · 共 299 个：适用 162，需复核 62，不匹配 73，无法评价 2"
        )
        window.peakRangeAssessmentPanel.show()
        window.show()
        window.main_splitter.setSizes([720, 460])
        qapp.processEvents()

        summary_rect = window.peakRangeAssessmentLabel.geometry()
        display_top = window.peakRangeDisplayModeCombo.mapTo(
            window.peakRangeAssessmentPanel,
            QtCore.QPoint(0, 0),
        ).y()
        review_top = window.peakRangeReviewOnlyCheck.mapTo(
            window.peakRangeAssessmentPanel,
            QtCore.QPoint(0, 0),
        ).y()

        assert summary_rect.bottom() < min(display_top, review_top)
    finally:
        window.deleteLater()


def test_workbench_side_panel_preserves_a_readable_minimum_width(qapp):
    window = MainWindow()
    try:
        window.resize(1180, 700)
        window.setup_plots(
            np.array([100.0, 101.0, 102.0]),
            np.array([1.0, 3.0, 1.0]),
        )
        window.show()
        window.main_splitter.setSizes([2000, 1])
        qapp.processEvents()

        assert window.widget_3.width() >= 420
    finally:
        window.deleteLater()


def test_workbench_peak_table_keeps_numeric_values_readable(qapp):
    window = MainWindow()
    try:
        window.resize(1180, 700)
        window.setup_plots(
            np.array([100.0, 101.0, 102.0]),
            np.array([1.0, 3.0, 1.0]),
        )
        values = [
            "Unknown species",
            "29983.50",
            "347.88898",
            "4.00",
            "29983.00",
            "29984.00",
        ]
        window.peakData.setRowCount(1)
        for column, value in enumerate(values):
            window.peakData.setItem(0, column, QtWidgets.QTableWidgetItem(value))
        window.show()
        window.main_splitter.setSizes([2000, 420])
        qapp.processEvents()

        metrics = window.peakData.fontMetrics()
        for column in range(1, window.peakData.columnCount()):
            minimum_text_width = metrics.horizontalAdvance(values[column]) + 6
            assert window.peakData.columnWidth(column) >= minimum_text_width
    finally:
        window.deleteLater()


def test_workbench_side_panel_groups_peak_commands_by_task(qapp):
    window = MainWindow()
    try:
        window.setup_plots(
            np.array([100.0, 101.0, 102.0]),
            np.array([1.0, 3.0, 1.0]),
        )
        window.show()
        qapp.processEvents()

        assert window.peakWorkflowTitle.text() == "卡峰处理"
        assert window.peakWorkflowTitle.isVisible()
        assert window.peakNavigationTitle.text() == "峰操作"
        assert window.peakNavigationTitle.isVisible()
    finally:
        window.deleteLater()


def test_workbench_current_peak_summary_sits_above_spectrum_plot(qapp):
    window = MainWindow()
    try:
        window.resize(1280, 720)
        window.setup_plots(
            np.array([100.0, 101.0, 102.0]),
            np.array([1.0, 3.0, 1.0]),
        )
        window.show()
        qapp.processEvents()

        summary_rect = window.peakSummaryPanel.geometry()
        plot_rect = window.graph_layout.geometry()

        assert window.peakSummaryPanel.parentWidget() is window.widget_2
        assert summary_rect.bottom() < plot_rect.top()
        assert window.peakSummaryPanel.height() <= 44
    finally:
        window.deleteLater()


def test_workbench_side_panel_keeps_actions_above_peak_table(qapp):
    window = MainWindow()
    try:
        window.setup_plots(
            np.array([100.0, 101.0, 102.0]),
            np.array([1.0, 3.0, 1.0]),
        )
        window.peakRangeAssessmentPanel.show()
        window.show()
        qapp.processEvents()

        centers = [
            widget.geometry().center().y()
            for widget in (
                window.peakWorkflowPanel,
                window.peakRangeAssessmentPanel,
                window.peakNavigationSection,
                window.peakProjectActionPanel,
                window.peakResult,
            )
        ]

        assert centers == sorted(centers)
    finally:
        window.deleteLater()


def test_workbench_previews_external_peak_ranges_without_activating_project(
    qapp,
    tmp_path,
    monkeypatch,
):
    peak_file = tmp_path / "candidate.csv"
    peak_file.write_text(
        "label,peak_index,mz,left_bound,right_bound\n"
        "centered,100,100,90,110\n"
        "shifted,135,140,132,148\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(peak_file), "Peak Files"),
    )

    window = MainWindow()
    try:
        saved_display_preference = QtCore.QSettings(
            "BL03U", "MassSpecTool"
        ).value("spectrum/peak_range_display_mode")
        window.x_axis_mode = "tof"
        window.lineEdit_4.setText("0")
        window.lineEdit_5.setText("1")
        window.lineEdit_6.setText("0")
        x = np.arange(50.0, 171.0)
        y = (
            3.0
            + 100.0 * np.exp(-0.5 * ((x - 100.0) / 3.0) ** 2)
            + 60.0 * np.exp(-0.5 * ((x - 140.0) / 2.5) ** 2)
        )
        window.setup_plots(x, y)
        before = window.project_settings_manager.snapshot()

        window.preview_peak_ranges_from_file()
        qapp.processEvents()

        after = window.project_settings_manager.snapshot()
        assert after.active_peak_set_id == before.active_peak_set_id
        assert after.manual_peak_file == before.manual_peak_file
        assert window._peak_range_preview_path == str(peak_file)
        assert window.peakData.rowCount() == 2
        assert window.peakData.item(0, 3).text() == "103.00"
        assert "适用 1" in window.peakRangeAssessmentLabel.text()
        assert "需复核 1" in window.peakRangeAssessmentLabel.text()
        assert not window.peakRangeAssessmentPanel.isHidden()
        assert len(window._peak_range_overlay_items) == 1
        assert window._peak_range_marker_item is not None
        assert len(window._peak_range_marker_item.points()) == 2
        assert "适用" in window.peakData.item(0, 0).toolTip()
        assert "偏离文件峰位" in window.peakData.item(1, 0).toolTip()
        assert window._peak_table_dirty
        assert window._peak_candidate_origin == "imported_preview"
        assert window.peakRangeDisplayModeCombo.currentData() == "all"
        assert (
            QtCore.QSettings("BL03U", "MassSpecTool").value(
                "spectrum/peak_range_display_mode"
            )
            == saved_display_preference
        )

        exported = window._peak_ranges_export_dataframe()
        assert exported.loc[1, "peak_index"] == 135
        assert exported.loc[1, "left_bound"] == 132

        window.peakRangeReviewOnlyCheck.setChecked(True)
        assert window.peakData.isRowHidden(0)
        assert not window.peakData.isRowHidden(1)
        assert window.peakData.currentRow() == 1
        assert len(window._peak_range_overlay_items) == 1
        assert window._peak_range_overlay_items[0]._table_row == 1
        assert window._peak_range_marker_item is not None
        assert len(window._peak_range_marker_item.points()) == 1
    finally:
        window.deleteLater()


def test_editing_previewed_peak_clears_stale_assessment(qapp, tmp_path, monkeypatch):
    peak_file = tmp_path / "candidate.csv"
    peak_file.write_text(
        "label,peak_index,mz,left_bound,right_bound\nP,100,100,90,110\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(peak_file), "Peak Files"),
    )

    window = MainWindow()
    try:
        window.x_axis_mode = "tof"
        window.lineEdit_4.setText("0")
        window.lineEdit_5.setText("1")
        window.lineEdit_6.setText("0")
        x = np.arange(50.0, 151.0)
        y = 3.0 + 100.0 * np.exp(-0.5 * ((x - 100.0) / 3.0) ** 2)
        window.setup_plots(x, y)
        window.preview_peak_ranges_from_file()

        window.peakData.item(0, 4).setText("91")
        qapp.processEvents()

        assert window._peak_range_assessments == []
        assert window._preview_peak_ranges == []
        assert window._peak_range_overlay_items == []
        assert window.peakRangeAssessmentPanel.isHidden()
        assert window.peakData.item(0, 0).toolTip() == ""
    finally:
        window.deleteLater()


def test_peak_range_overview_does_not_render_every_assessment_as_full_height_band(qapp):
    window = MainWindow()
    try:
        window.x_axis_mode = "tof"
        x = np.arange(0.0, 4000.0)
        y = 5.0 + np.sin(x / 25.0)
        window.setup_plots(x, y)

        peak_ranges = [
            PeakRange(
                mz=float(index),
                peak_index=index,
                left_bound=index - 3,
                right_bound=index + 3,
                label=f"P{row}",
            )
            for row, index in enumerate(range(10, 3000, 10))
        ]
        assessments = [
            PeakRangeAssessment(
                peak_range=peak_range,
                status=PeakRangeFitStatus.FIT,
                reasons=("匹配",),
                actual_peak_x=float(peak_range.peak_index),
                actual_mz=float(peak_range.mz),
                peak_intensity=6.0,
            )
            for peak_range in peak_ranges
        ]
        window._load_peak_records_into_table(
            [
                {
                    "label": peak_range.label,
                    "peak_index": peak_range.peak_index,
                    "mz": peak_range.mz,
                    "intensity": 6.0,
                    "left_bound": peak_range.left_bound,
                    "right_bound": peak_range.right_bound,
                }
                for peak_range in peak_ranges
            ]
        )
        window._preview_peak_ranges = peak_ranges
        window._set_peak_range_display_mode("all", persist=False)
        window._set_peak_range_assessments(assessments)
        window._select_peak_row(0)
        window.update_plot()

        assert len(assessments) == 299
        assert len(window._peak_range_overlay_items) <= 1
        assert window._peak_range_marker_item is not None
        assert len(window._peak_range_marker_item.points()) == 299
    finally:
        window.deleteLater()


def test_peak_range_display_mode_defaults_to_selected_and_remembers_manual_choice(qapp):
    settings = QtCore.QSettings("BL03U", "MassSpecTool")
    key = "spectrum/peak_range_display_mode"
    previous_value = settings.value(key)
    settings.remove(key)
    settings.sync()

    first_window = MainWindow()
    second_window = None
    try:
        assert first_window.peakRangeDisplayModeCombo.currentData() == "selected"
        first_window.x_axis_mode = "tof"
        x = np.arange(50.0, 151.0)
        y = 3.0 + 100.0 * np.exp(-0.5 * ((x - 100.0) / 3.0) ** 2)
        first_window.setup_plots(x, y)
        peak_range = PeakRange(100.0, 100, 90, 110, "P")
        assessment = PeakRangeAssessment(
            peak_range=peak_range,
            status=PeakRangeFitStatus.FIT,
            reasons=("匹配",),
            actual_peak_x=100.0,
            actual_mz=100.0,
            peak_intensity=103.0,
        )
        first_window._load_peak_records_into_table(
            [
                {
                    "label": "P",
                    "peak_index": 100,
                    "mz": 100.0,
                    "intensity": 103.0,
                    "left_bound": 90,
                    "right_bound": 110,
                }
            ]
        )
        first_window._preview_peak_ranges = [peak_range]
        first_window._set_peak_range_assessments([assessment])
        first_window._select_peak_row(0)
        first_window.update_plot()

        assert len(first_window._peak_range_overlay_items) == 1
        assert first_window._peak_range_marker_item is None

        all_index = first_window.peakRangeDisplayModeCombo.findData("all")
        first_window.peakRangeDisplayModeCombo.setCurrentIndex(all_index)
        assert first_window._peak_range_marker_item is not None

        hidden_index = first_window.peakRangeDisplayModeCombo.findData("hidden")
        first_window.peakRangeDisplayModeCombo.setCurrentIndex(hidden_index)
        assert first_window._peak_range_overlay_items == []
        assert first_window._peak_range_marker_item is None

        settings.sync()
        second_window = MainWindow()
        assert second_window.peakRangeDisplayModeCombo.currentData() == "hidden"
    finally:
        first_window.deleteLater()
        if second_window is not None:
            second_window.deleteLater()
        if previous_value is None:
            settings.remove(key)
        else:
            settings.setValue(key, previous_value)
        settings.sync()


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


def test_workbench_save_peak_ranges_registers_project_manual_file(qapp, tmp_path, monkeypatch):
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

        window.save_peak_ranges_to_project()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        saved_path = Path(saved.manual_peak_file)
        manifest_path = saved_path.with_suffix(".manifest.yaml")
        assert saved.temp_peak_source == "manual"
        assert saved.active_peak_set_id
        assert saved_path.parent == (
            project_dir / "analysis" / "spectrum" / "manual_peaks" / "peak_sets"
        )
        assert saved_path.exists()
        assert manifest_path.exists()

        df = pd.read_csv(saved_path)
        assert list(df.columns) == ["label", "peak_index", "mz", "left_bound", "right_bound"]
        assert df.loc[0, "label"] == "CH4"
        assert df.loc[0, "peak_index"] == 100
        assert df.loc[0, "left_bound"] == 96
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert manifest["peak_file"] == saved_path.name
        assert manifest["peak_set_id"] == saved.active_peak_set_id
        assert manifest["approved"] is True
        assert manifest["source"]["mode"] == "single"
        assert manifest["source"]["path"] == str(source_file)
        assert window.project_manual_peak_edit.text() == str(saved_path)
        assert window._peak_table_dirty is False

        first_content = saved_path.read_bytes()
        window.peakData.item(0, 4).setText("95.0")
        window.save_peak_ranges_to_project()
        updated = load_project_settings(project_dir / "config" / "project.yaml")
        records = list_peak_sets(project_dir)
        assert len(records) == 2
        assert updated.active_peak_set_id != saved.active_peak_set_id
        assert Path(updated.manual_peak_file) != saved_path
        assert saved_path.read_bytes() == first_content
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_peak_save_change_summary_reports_table_diff(tmp_path):
    current = tmp_path / "approved.csv"
    pd.DataFrame(
        [
            {"label": "A", "mz": 10.0, "left_bound": 8, "right_bound": 12},
            {"label": "B", "mz": 20.0, "left_bound": 18, "right_bound": 22},
        ]
    ).to_csv(current, index=False)
    candidate = pd.DataFrame(
        [
            {"label": "A", "mz": 10.0, "left_bound": 7, "right_bound": 13},
            {"label": "C", "mz": 30.0, "left_bound": 28, "right_bound": 32},
        ]
    )

    summary = MainWindow._peak_range_change_summary(candidate, str(current))

    assert "新增 1" in summary
    assert "删除 1" in summary
    assert "边界修改 1" in summary


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


def test_workbench_save_peak_ranges_requires_reopenable_spectrum_source(qapp, tmp_path, monkeypatch):
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

        window.save_peak_ranges_to_project()

        saved = load_project_settings(project_dir / "config" / "project.yaml")
        assert saved.manual_peak_file == ""
        assert not (project_dir / "analysis" / "spectrum" / "manual_peaks" / "manual_peak_ranges.csv").exists()
        assert warnings
        assert "请先打开原始谱图" in warnings[0][1]
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_workbench_open_project_peak_ranges_accepts_import_without_spectrum_manifest(
    qapp,
    tmp_path,
    monkeypatch,
):
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
        temp_peak_source="manual",
    )
    peak_set = import_peak_set(project_dir, peak_file)
    ps.active_peak_set_id = peak_set.peak_set_id
    ps.manual_peak_file = str(verify_peak_set(project_dir, peak_set))
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

        assert window.open_project_peak_ranges()

        assert window._valid_peak_rows() == [0]
        assert window.peakData.item(0, 0).text() == "Unknown"
        assert window.widget_3.isHidden()
        assert (
            window._spectrum_empty_state.title_label.text()
            == "项目卡峰范围已加载"
        )
        assert warnings
        assert warnings[0][0] == "谱图未恢复"
        assert "未关联谱图" in warnings[0][1]
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_open_project_automatically_loads_current_peak_ranges(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "Project_Auto_Load_Peaks"
    peak_set = create_peak_set(
        project_dir,
        content=b"label,peak_index,mz,left_bound,right_bound\nCH4,100,16,98,102\n",
        extension=".csv",
        label="current",
        origin="spectrum_workbench",
    )
    ps = ProjectSettings(
        project_name="Auto load peaks",
        output_dir=str(project_dir),
        manual_peak_file=peak_set.peak_file,
        active_peak_set_id=peak_set.peak_set_id,
        temp_peak_source="manual",
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(project_dir),
    )

    window = MainWindow()
    try:
        window.open_project()

        assert window._valid_peak_rows() == [0]
        assert window.peakData.item(0, 0).text() == "CH4"
        assert window.peakData.item(0, 4).text() == "98.00"
        assert window._peak_table_dirty is False
        assert window.spectrum_source_scope == "project"
        assert window.peakProjectStateLabel.text() == "已保存为当前项目卡峰范围"
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

        window.save_peak_ranges_to_project()
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
        assert not window.peakRangeAssessmentPanel.isHidden()
        expected_overlay_count = (
            0 if window.peakRangeDisplayModeCombo.currentData() == "hidden" else 1
        )
        assert len(window._peak_range_overlay_items) == expected_overlay_count
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
        window._set_project_draft(ps, committed=True)
        window._apply_project_runtime_settings(ps)
        window.workspace_stack.setCurrentWidget(window.project_page)

        widget = window.project_function_defaults_widget
        window.project_tabs.setCurrentWidget(window.project_analysis_page)
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

        # Switching tabs updates only the shared draft, not the manager or disk.
        window._on_project_tab_changed(1)
        assert window._project_draft.pie_energy_decimals == 4
        assert window.project_settings_manager.snapshot().pie_energy_decimals == 1
        assert load_project_settings(
            project_dir / "config" / "project.yaml"
        ).pie_energy_decimals == 1
        assert widget.pie_energy_decimals_edit.value() == 4

        assert window.save_and_apply_project_settings() is True
        window.switch_workspace_page("pie")

        active = window.project_settings_manager.snapshot()
        assert active.temp_peak_source == "auto"
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


def test_import_finished_does_not_write_after_project_is_closed(
    qapp,
    tmp_path,
    monkeypatch,
):
    window = MainWindow()
    try:
        window.project_settings_manager.clear_project_path()
        window._import_project_config_path = (
            tmp_path / "closed-project" / "config" / "project.yaml"
        )
        monkeypatch.setattr(
            window.project_settings_manager,
            "snapshot",
            lambda: pytest.fail("closed project must not produce an import snapshot"),
        )
        monkeypatch.setattr(
            "bl03u_masstool.frontends.pyqt_app.spectrum.workspace_pages.import_peak_set",
            lambda *args, **kwargs: pytest.fail(
                "closed project must not create a peak set"
            ),
        )

        window._on_import_finished(
            {
                "success": True,
                "field_name": "manual_peak_file",
                "destination": str(tmp_path / "imported-peaks.csv"),
                "label": "手动卡峰文件",
                "source_key": "manual_peak",
                "mode": "copy",
            }
        )

        assert "原项目已关闭或切换" in window.statusbar.currentMessage()
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_import_finished_reports_project_registration_failure(
    qapp,
    tmp_path,
    monkeypatch,
):
    project_dir = tmp_path / "Import_Save_Failure"
    ps = ProjectSettings(project_name="Import failure", output_dir=str(project_dir))
    save_project_settings(ps, project_dir / "config" / "project.yaml")
    messages = []

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        monkeypatch.setattr(
            window.project_settings_manager,
            "replace_and_save",
            lambda _settings: (_ for _ in ()).throw(OSError("disk full")),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "critical",
            lambda *args, **kwargs: messages.append(str(args[2])),
        )

        window._on_import_finished(
            {
                "success": True,
                "field_name": "temperature_scan_folder",
                "destination": str(project_dir / "raw_data" / "temperature_scan"),
                "label": "温度扫描目录",
                "source_key": "temperature_scan",
                "mode": "copy",
            }
        )

        assert messages
        assert "disk full" in messages[0]
        assert window.statusbar.currentMessage() == "导入结果未能登记到项目"
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


def test_project_save_failure_preserves_committed_file_and_draft(qapp, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    original = ProjectSettings(project_name="Committed", output_dir=str(project_dir))
    save_project_settings(original, project_dir / "config" / "project.yaml")
    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window._set_project_draft(original, committed=True)
        window.project_name_edit.setText("Draft survives")
        monkeypatch.setattr(
            window.project_settings_manager,
            "replace_and_save",
            lambda _settings: (_ for _ in ()).throw(OSError("disk full")),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "critical",
            lambda *args, **kwargs: None,
        )

        assert window.save_and_apply_project_settings() is False
        assert load_project_settings(
            project_dir / "config" / "project.yaml"
        ).project_name == "Committed"
        assert window.project_name_edit.text() == "Draft survives"
        assert window._project_draft.project_name == "Draft survives"
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
        approved_path = Path(saved.manual_peak_file)
        assert saved.active_peak_set_id
        assert approved_path.parent == (
            project_dir / "analysis" / "spectrum" / "manual_peaks" / "peak_sets"
        )
        assert approved_path.read_bytes() == imported_peak.read_bytes()
        assert window.project_manual_peak_edit.text() == str(approved_path)
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
        imported = Path(saved.manual_peak_file)
        assert saved.active_peak_set_id
        assert imported.parent == (
            project_dir / "analysis" / "spectrum" / "manual_peaks" / "peak_sets"
        )
        assert Path(saved.manual_peak_file) == imported
        assert imported.read_text(encoding="utf-8") == legacy_peak.read_text(encoding="utf-8")
        assert window.project_manual_peak_edit.text() == str(imported)
        assert window.datasource_row_status_labels["manual_peak"].text().startswith("✓")
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()


def test_project_management_exposes_only_current_peak_ranges(qapp, tmp_path):
    project_dir = tmp_path / "Project_Peak_Set_Switch"
    first = create_peak_set(
        project_dir,
        content=b"label,peak_index,mz,left_bound,right_bound\nA,100,28,98,102\n",
        extension=".csv",
        label="first",
        origin="spectrum_workbench",
    )
    second = create_peak_set(
        project_dir,
        content=b"label,peak_index,mz,left_bound,right_bound\nA,100,28,97,103\n",
        extension=".csv",
        label="second",
        origin="spectrum_workbench",
    )
    ps = ProjectSettings(
        project_name="Peak switch",
        output_dir=str(project_dir),
        manual_peak_file=first.peak_file,
        active_peak_set_id=first.peak_set_id,
        temp_peak_source="manual",
    )
    save_project_settings(ps, project_dir / "config" / "project.yaml")

    window = MainWindow()
    try:
        window.project_settings_manager.set_project_path(project_dir)
        window.project_settings_manager.set(ps)
        window._read_project_settings_to_ui(ps)

        assert not hasattr(window, "project_peak_set_combo")
        assert not hasattr(window, "project_peak_set_activate_button")
        assert not hasattr(window, "activate_selected_project_peak_set")
        assert window.project_manual_peak_edit.text() == first.peak_file
        assert window.datasource_row_status_labels["manual_peak"].text() == "✓ 当前"
        assert verify_peak_set(project_dir, first).exists()
        assert verify_peak_set(project_dir, second).exists()
    finally:
        window.project_settings_manager.clear_project_path()
        window.deleteLater()
