from __future__ import annotations

import os
import time

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtGui, QtWidgets
except ImportError as exc:
    pytest.skip(f"PyQt6 display libraries not available: {exc}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.curve_database import (
    get_curve_dataset_state_read_only,
    mark_curve_datasets_stale,
    replace_species_assignments,
    store_curve_dataset,
    store_curve_dataset_version,
)
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.frontends.pyqt_app.isotope_correction.dialog import IsotopeCorrectionDialog
from bl03u_masstool.frontends.pyqt_app.spectrum.workbench import MainWindow
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _temperature_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature": [300, 300, 300, 400, 400, 400],
            "mz": [226.0, 228.0, 230.0, 226.0, 228.0, 230.0],
            "area": [136.0, 80.0, 27.0, 200.0, 120.0, 35.0],
        }
    )


def _wait_for_project_load(qapp, widget: IsotopeCorrectionDialog) -> None:
    worker = widget._project_load_worker
    assert worker is not None
    deadline = time.monotonic() + 3.0
    while worker.isRunning() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert worker.wait(1_000)
    qapp.processEvents()


def test_dialog_loads_temperature_result_and_calculates_without_conclusion(qapp):
    widget = IsotopeCorrectionDialog()
    try:
        assert not widget.calculate_button.isEnabled()
        widget.load_dataframe(_temperature_result(), source_type="temperature", source_label="memory")
        assert not widget.calculate_button.isEnabled()
        widget.mz_min_spin.setValue(226)
        widget.mz_max_spin.setValue(230)
        widget.formula_edit.setText("C11H8Cl2O")
        widget.parent_mz_spin.setValue(226)
        widget.fraction_spin.setValue(100)
        widget.add_hypothesis()
        assert widget.calculate_button.isEnabled()
        widget.calculate_correction()

        assert widget.correction_result is not None
        assert widget.result_mz_combo.count() == 3
        assert widget.curve_table_widget.rowCount() == 6
        visible_text = " ".join(label.text() for label in widget.findChildren(QtWidgets.QLabel))
        assert "自动结论" not in visible_text
        assert "物种鉴定" in visible_text
    finally:
        widget.deleteLater()


def test_dialog_max_compatible_mode_keeps_curve_residual_nonnegative(qapp):
    widget = IsotopeCorrectionDialog()
    try:
        widget.load_dataframe(_temperature_result(), source_type="temperature")
        widget.mz_min_spin.setValue(226)
        widget.mz_max_spin.setValue(230)
        widget.formula_edit.setText("C11H8Cl2O")
        widget.parent_mz_spin.setValue(226)
        widget.add_hypothesis()
        widget.mode_combo.setCurrentIndex(1)
        widget.calculate_correction()

        assert widget.correction_result is not None
        assert widget.correction_result.mode == "max_compatible"
        assert (widget.correction_result.curve_table["residual"] >= -1e-9).all()
    finally:
        widget.deleteLater()


def test_temperature_result_filters_one_photon_energy_before_correction(qapp):
    data = pd.concat(
        [
            _temperature_result().assign(photon_energy=10.0),
            _temperature_result().assign(
                photon_energy=11.0,
                area=lambda frame: frame["area"] * 10,
            ),
        ],
        ignore_index=True,
    )
    widget = IsotopeCorrectionDialog()
    try:
        widget.load_dataframe(data, source_type="temperature")
        assert widget.energy_filter_combo.count() == 2
        widget.energy_filter_combo.setCurrentIndex(1)
        widget.mz_min_spin.setValue(226)
        widget.mz_max_spin.setValue(230)
        widget.formula_edit.setText("C11H8Cl2O")
        widget.parent_mz_spin.setValue(226)
        widget.add_hypothesis()
        widget.calculate_correction()

        assert widget.correction_result is not None
        selected = widget.correction_result.curve_table
        observed_226 = selected.loc[
            (selected["temperature"] == 300.0) & (selected["mz"] == 226),
            "observed",
        ].iloc[0]
        assert observed_226 == pytest.approx(1360.0)
    finally:
        widget.deleteLater()


def test_temperature_energy_filter_groups_float_jitter_and_preserves_full_curve(qapp):
    data = pd.DataFrame(
        {
            "temperature": [300, 400, 300, 400],
            "mz": [226.0, 226.0, 226.0, 226.0],
            "area": [10.0, 20.0, 30.0, 40.0],
            "photon_energy": [10.9996, 11.0002, 11.9996, 12.0001],
        }
    )
    widget = IsotopeCorrectionDialog()
    try:
        widget.load_dataframe(data, source_type="temperature")

        assert widget.energy_filter_combo.count() == 2
        assert widget.energy_filter_combo.itemData(0) == pytest.approx(11.0)
        assert "2 个温度点" in widget.energy_filter_combo.itemText(0)
        selected = widget._selected_input_data()
        assert selected["temperature"].tolist() == [300, 400]
    finally:
        widget.deleteLater()


def test_formula_drives_parent_mass_and_focuses_isotope_mass_range(qapp):
    widget = IsotopeCorrectionDialog()
    try:
        widget.load_dataframe(_temperature_result(), source_type="temperature")
        assert widget.parent_mz_spin.value() == 226

        widget.formula_edit.setText("C18H12")
        assert widget.parent_mz_spin.value() == 228
        widget.add_hypothesis()

        assert widget.hypotheses[0].parent_mz == 228
        assert widget.mz_min_spin.value() == 228
        assert widget.mz_max_spin.value() == 232
    finally:
        widget.deleteLater()


def test_parent_mass_dropdown_lists_loaded_nominal_channels(qapp):
    widget = IsotopeCorrectionDialog()
    try:
        widget.load_dataframe(_temperature_result(), source_type="temperature")

        assert [
            widget.parent_mz_combo.itemData(index)
            for index in range(widget.parent_mz_combo.count())
        ] == [None, 226, 228, 230]
        assert [
            widget.parent_mz_combo.itemText(index)
            for index in range(widget.parent_mz_combo.count())
        ] == ["选择 m/z", "m/z 226", "m/z 228", "m/z 230"]
        widget.parent_mz_combo.setCurrentIndex(
            widget.parent_mz_combo.findData(226)
        )

        assert widget.parent_mz_spin.value() == 226
        assert widget.mz_min_spin.value() == 226
        assert widget.mz_max_spin.value() == 230
    finally:
        widget.deleteLater()


def test_isotope_page_keeps_result_plot_visible_at_compact_height(qapp):
    widget = IsotopeCorrectionDialog()
    try:
        widget.resize(1_600, 700)
        widget.load_dataframe(_temperature_result(), source_type="temperature")
        widget.show()
        qapp.processEvents()

        assert widget.height() == 700
        assert widget.result_tabs.isVisible()
        assert widget.plot_widget.height() >= 180
        assert sum(widget.page_splitter.sizes()) <= widget.height()
    finally:
        widget.close()
        widget.deleteLater()


def test_selected_parent_mass_rejects_mismatched_formula(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2])),
    )
    widget = IsotopeCorrectionDialog()
    try:
        widget.load_dataframe(_temperature_result(), source_type="temperature")
        widget.parent_mz_combo.setCurrentIndex(
            widget.parent_mz_combo.findData(226)
        )
        widget.formula_edit.setText("C18H12")
        widget.add_hypothesis()

        assert widget.hypotheses == []
        assert widget.parent_mz_combo.currentData() == 226
        assert "名义质量为 228" in warnings[0]
        assert "所选母峰 m/z 226" in warnings[0]
    finally:
        widget.deleteLater()


def test_dialog_uses_project_result_paths_without_auto_loading(qapp, tmp_path):
    temperature_path = tmp_path / "temperature.csv"
    _temperature_result().to_csv(temperature_path, index=False)
    widget = IsotopeCorrectionDialog()
    try:
        widget.set_project_settings(
            ProjectSettings(temperature_scan_result_file=str(temperature_path)),
            activate_project_scope=True,
        )

        assert widget.project_source_button.isEnabled()
        assert widget.input_df.empty
        widget.load_project_result()
        assert not widget.input_df.empty
    finally:
        widget.deleteLater()


def test_dialog_uses_registered_pie_curve_file_not_identification_report(qapp, tmp_path):
    pie_curve_path = tmp_path / "pie_curves.csv"
    pd.DataFrame(
        {
            "energy": [10.0, 10.0, 11.0, 11.0],
            "mz": [226.0, 228.0, 226.0, 228.0],
            "normalized_intensity": [100.0, 65.0, 150.0, 98.0],
        }
    ).to_csv(pie_curve_path, index=False)
    widget = IsotopeCorrectionDialog()
    try:
        widget.source_type_combo.setCurrentIndex(1)
        widget.set_project_settings(
            ProjectSettings(
                pie_curve_result_file=str(pie_curve_path),
                pie_identification_result_file=str(tmp_path / "identification.xlsx"),
            ),
            activate_project_scope=True,
        )
        widget.load_project_result()

        assert widget.source_path == str(pie_curve_path)
        assert widget.axis_column == "energy"
        assert widget.signal_column_combo.currentData() == "normalized_intensity"
    finally:
        widget.deleteLater()


def test_dialog_prefers_project_curve_database_over_export_file(qapp, tmp_path):
    database_path = tmp_path / "curve_data.sqlite"
    export_path = tmp_path / "stale.csv"
    _temperature_result().assign(area=1.0).to_csv(export_path, index=False)
    curves = {
        226.01: {
            "mz": 226.01,
            "mz_rounded": 226,
            "rows": pd.DataFrame(
                {
                    "temperature": [300.0, 400.0],
                    "area": [136.0, 200.0],
                    "photon_energy": [11.0, 11.0],
                }
            ),
        }
    }
    store_curve_dataset_version(
        database_path,
        curve_type="temperature",
        curves=curves,
        dataset_group="temperature:project",
        analysis_key="temperature-analysis-v1",
        name="项目温度曲线",
    )
    pie_dataset_id = store_curve_dataset_version(
        database_path,
        curve_type="pie",
        curves={
            226.01: {
                "mz": 226.01,
                "mz_rounded": 226,
                "rows": pd.DataFrame(
                    {
                        "energy": [9.0, 10.0],
                        "normalized_intensity": [0.0, 1.0],
                    }
                ),
            }
        },
        dataset_group="pie:project",
        analysis_key="pie-analysis-v1",
        name="项目PIE曲线",
    )
    replace_species_assignments(
        database_path,
        dataset_id=pie_dataset_id,
        exact_mz=226.01,
        assignments=[
            {
                "species": "candidate",
                "formula": "C11H8Cl2O",
            }
        ],
        assignment_status="candidate",
    )
    widget = IsotopeCorrectionDialog()
    try:
        widget.set_project_settings(
            ProjectSettings(
                curve_database_path=str(database_path),
                temperature_scan_result_file=str(export_path),
            ),
            activate_project_scope=True,
        )
        widget.load_project_result()
        _wait_for_project_load(qapp, widget)

        assert "curve_data.sqlite" in widget.source_path
        assert widget.input_df["area"].tolist() == [136.0, 200.0]
        assert widget.input_df["mz"].unique().tolist() == pytest.approx([226.01])
        assert not widget.database_candidate_combo.isEnabled()
        widget.parent_mz_combo.setCurrentIndex(
            widget.parent_mz_combo.findData(226)
        )
        assert widget.database_candidate_combo.isEnabled()
        widget.database_candidate_combo.setCurrentIndex(1)
        assert widget.formula_edit.text() == "C11H8Cl2O"
        assert widget.parent_mz_spin.value() == 226
        assert widget.hypotheses == []
    finally:
        widget.deleteLater()


def test_dialog_ignores_legacy_batches_when_selecting_project_result(qapp, tmp_path):
    database_path = tmp_path / "curve_data.sqlite"
    store_curve_dataset(
        database_path,
        curve_type="temperature",
        curves={
            225.01: {
                "mz": 225.01,
                "mz_rounded": 225,
                "rows": pd.DataFrame(
                    {"temperature": [300.0], "area": [999.0]}
                ),
            }
        },
        dataset_key="temperature:legacy-batch:42",
        name="旧批次",
    )
    current_result_id = store_curve_dataset(
        database_path,
        curve_type="temperature",
        curves={
            226.01: {
                "mz": 226.01,
                "mz_rounded": 226,
                "rows": pd.DataFrame(
                    {"temperature": [300.0], "area": [136.0]}
                ),
            }
        },
        dataset_key="temperature:project",
        name="项目温度结果",
    )
    widget = IsotopeCorrectionDialog()
    try:
        widget.set_project_settings(
            ProjectSettings(curve_database_path=str(database_path)),
            activate_project_scope=True,
        )

        selected_path, datasets, current = widget._project_curve_selection()

        assert selected_path == database_path
        assert [dataset.dataset_id for dataset in datasets] == [current_result_id]
        assert current is not None
        assert current.dataset_id == current_result_id
        assert widget.project_dataset_combo.count() == 1
        assert not widget.project_dataset_combo.isVisible()
    finally:
        worker = widget._project_load_worker
        if worker is not None and worker.isRunning():
            worker.wait(2_000)
        widget.deleteLater()


def test_dialog_refreshes_and_auto_loads_new_valid_sqlite_dataset(qapp, tmp_path):
    database_path = tmp_path / "curve_data.sqlite"
    result_id = store_curve_dataset(
        database_path,
        curve_type="temperature",
        curves={
            226.01: {
                "mz": 226.01,
                "mz_rounded": 226,
                "rows": pd.DataFrame({"temperature": [300.0], "area": [1.0]}),
            }
        },
        dataset_key="temperature:project",
        name="旧温度曲线",
    )
    mark_curve_datasets_stale(
        database_path,
        curve_type="temperature",
        reason="卡峰文件已变化",
    )
    widget = IsotopeCorrectionDialog()
    try:
        widget.set_project_settings(
            ProjectSettings(curve_database_path=str(database_path)),
            activate_project_scope=True,
        )
        assert widget.project_source_button.text() == "项目无对应结果"

        valid_result_id = store_curve_dataset(
            database_path,
            curve_type="temperature",
            curves={
                226.02: {
                    "mz": 226.02,
                    "mz_rounded": 226,
                    "rows": pd.DataFrame(
                        {"temperature": [300.0, 400.0], "area": [136.0, 200.0]}
                    ),
                }
            },
            dataset_key="temperature:project",
            name="新温度曲线",
        )
        assert valid_result_id == result_id

        assert widget.ensure_project_source_loaded()
        _wait_for_project_load(qapp, widget)

        assert widget.input_df["area"].tolist() == [136.0, 200.0]
        assert widget.loaded_database_path == str(database_path)
        assert widget.project_source_button.text() == "当前使用项目结果"
        assert not widget.project_source_button.isEnabled()
        assert "项目 SQLite" in widget.source_path
    finally:
        worker = widget._project_load_worker
        if worker is not None and worker.isRunning():
            worker.wait(2_000)
        widget.deleteLater()


def test_dialog_does_not_replace_explicit_file_when_page_is_revisited(qapp, tmp_path):
    database_path = tmp_path / "curve_data.sqlite"
    store_curve_dataset(
        database_path,
        curve_type="temperature",
        curves={
            226.01: {
                "mz": 226.01,
                "mz_rounded": 226,
                "rows": pd.DataFrame(
                    {"temperature": [300.0], "area": [136.0]}
                ),
            }
        },
        dataset_key="temperature:project",
        name="项目温度曲线",
    )
    explicit_path = tmp_path / "explicit.csv"
    explicit = _temperature_result().assign(area=7.0)
    explicit.to_csv(explicit_path, index=False)
    widget = IsotopeCorrectionDialog()
    try:
        widget.set_project_settings(
            ProjectSettings(curve_database_path=str(database_path)),
            activate_project_scope=True,
        )
        widget.load_result_file(explicit_path)

        assert widget.ensure_project_source_loaded()

        assert widget._project_load_worker is None
        assert widget.source_path == str(explicit_path)
        assert widget.input_df["area"].eq(7.0).all()
    finally:
        widget.deleteLater()


def test_dialog_does_not_offer_stale_project_result_for_review(qapp, tmp_path):
    database_path = tmp_path / "curve_data.sqlite"
    store_curve_dataset(
        database_path,
        curve_type="temperature",
        curves={
            226.01: {
                "mz": 226.01,
                "mz_rounded": 226,
                "rows": pd.DataFrame(
                    {"temperature": [300.0], "area": [136.0]}
                ),
            }
        },
        dataset_key="temperature:project",
        name="项目温度曲线",
    )
    mark_curve_datasets_stale(
        database_path,
        curve_type="temperature",
        reason="卡峰文件已变化",
    )
    widget = IsotopeCorrectionDialog()
    try:
        widget.set_project_settings(
            ProjectSettings(curve_database_path=str(database_path)),
            activate_project_scope=True,
        )

        assert not widget.project_source_button.isEnabled()
        assert widget.project_source_button.text() == "项目无对应结果"
        assert not widget.project_dataset_combo.isVisible()
    finally:
        widget.deleteLater()


def test_dialog_cannot_load_stale_project_result(
    qapp,
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "curve_data.sqlite"
    dataset_id = store_curve_dataset(
        database_path,
        curve_type="temperature",
        curves={
            226.01: {
                "mz": 226.01,
                "mz_rounded": 226,
                "rows": pd.DataFrame(
                    {"temperature": [300.0, 400.0], "area": [136.0, 200.0]}
                ),
            }
        },
        dataset_key="temperature:project",
        name="项目温度曲线",
    )
    mark_curve_datasets_stale(
        database_path,
        curve_type="temperature",
        reason="卡峰文件已变化",
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )
    widget = IsotopeCorrectionDialog()
    try:
        widget.set_project_settings(
            ProjectSettings(curve_database_path=str(database_path)),
            activate_project_scope=True,
        )

        widget.load_project_result()

        assert widget._project_load_worker is None
        assert widget.input_df.empty
        state = get_curve_dataset_state_read_only(database_path, dataset_id)
        assert state is not None
        assert state.validity_status == "stale"
        assert not state.is_current
    finally:
        worker = widget._project_load_worker
        if worker is not None and worker.isRunning():
            worker.wait(2_000)
        widget.deleteLater()


def test_dialog_exports_curves_patterns_components_and_parameters(qapp, tmp_path, monkeypatch):
    output_path = tmp_path / "temperature_isotope_corrected.xlsx"
    widget = IsotopeCorrectionDialog()
    try:
        widget.load_dataframe(_temperature_result(), source_type="temperature", source_label="memory")
        widget.mz_min_spin.setValue(226)
        widget.mz_max_spin.setValue(230)
        widget.formula_edit.setText("C11H8Cl2O")
        widget.parent_mz_spin.setValue(226)
        widget.add_hypothesis()
        widget.calculate_correction()
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(output_path), "Excel Files (*.xlsx)"),
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)

        widget.export_result()

        assert output_path.is_file()
        workbook = pd.ExcelFile(output_path)
        assert set(workbook.sheet_names) == {
            "corrected_curves",
            "isotope_patterns",
                "source_curves",
                "components",
                "numerical_diagnostics",
                "coefficient_sensitivity",
                "parameters",
        }
    finally:
        widget.deleteLater()


def test_workspace_exposes_isotope_correction_as_independent_page(qapp):
    window = MainWindow()
    try:
        assert window.page_buttons["isotope_correction"].text() == "同位素贡献校正"
        assert window.isotope_correction_page is not window.isotope_page
        assert "PIE" in window.page_buttons["isotope_correction"].toolTip()
    finally:
        window.deleteLater()


def test_main_window_does_not_close_while_worker_thread_is_running(qapp):
    window = MainWindow()
    worker = WorkerThread(lambda: time.sleep(0.15), window)
    try:
        worker.start()
        close_event = QtGui.QCloseEvent()
        window.closeEvent(close_event)

        assert not close_event.isAccepted()
        assert window._close_waiting_for_workers is True

        assert worker.wait(2_000)
        final_event = QtGui.QCloseEvent()
        window.closeEvent(final_event)
        assert final_event.isAccepted()
    finally:
        if worker.isRunning():
            worker.wait(2_000)
        window.deleteLater()
