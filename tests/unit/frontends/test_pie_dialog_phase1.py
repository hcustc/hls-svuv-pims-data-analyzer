"""
Phase 1 Tests for PIE Dialog: per-m/z configuration preservation and concurrency guards.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets, QtCore
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.nist_webbook import (
    NistCompoundIonization,
    NistIonizationEnergy,
    NistWebBookResult,
)
from bl03u_masstool.core.pie_analysis import PieSegmentSummary
from bl03u_masstool.core.pie_state import PieStateManager
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.project_lifecycle import project_root
from bl03u_masstool.frontends.pyqt_app.pie import dialog as pie_dialog_module
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


def test_pie_workspace_columns_share_contiguous_splitter_boundaries(pie_dialog, qapp):
    pie_dialog.curves = {
        30: {"mz": 30, "energies": np.array([9.0, 10.0]), "intensities": np.array([0.0, 1.0])}
    }
    pie_dialog._update_action_state()
    pie_dialog.resize(1180, 720)
    qapp.processEvents()

    splitter = pie_dialog.findChild(QtWidgets.QSplitter, "MainSplitter")
    assert splitter is not None
    assert splitter.count() == 3
    assert splitter.handleWidth() == 7

    for index in (1, 2):
        previous = splitter.widget(index - 1).geometry()
        handle = splitter.handle(index).geometry()
        following = splitter.widget(index).geometry()
        assert previous.right() + 1 == handle.left()
        assert handle.right() + 1 == following.left()


def test_fit_result_selection_signal_is_connected_once(pie_dialog):
    assert pie_dialog.fit_table.receivers(
        pie_dialog.fit_table.itemSelectionChanged
    ) == 1


def test_pie_generation_progress_uses_center_plot_workspace(pie_dialog, qapp):
    pie_dialog._begin_analysis_progress("正在读取 PIE 数据…")
    pie_dialog._on_analysis_progress(35, "正在识别并积分质谱峰…")
    qapp.processEvents()

    assert pie_dialog._plot_stack.currentWidget() is pie_dialog._progress_state
    assert pie_dialog._progress_state.progress_bar.value() == 35
    assert "积分" in pie_dialog._progress_state.detail_label.text()

    pie_dialog._restore_analysis_workspace()
    assert pie_dialog._plot_stack.currentWidget() is pie_dialog._empty_state


def test_project_open_restores_cached_pie_curves_without_reanalysis(qapp, tmp_path):
    raw_dir = tmp_path / "raw" / "pie"
    raw_dir.mkdir(parents=True)
    (raw_dir / "10.0eV.txt").write_text("cached source fingerprint", encoding="utf-8")
    project_dir = tmp_path / "Project_PIE_Cache"
    ps = ProjectSettings(
        project_name="PIE Cache",
        system="Test",
        output_dir=str(project_dir),
        pie_scan_folder=str(raw_dir),
        pie_scan_folders=[str(raw_dir)],
    )
    analysis_df = pd.DataFrame(
        {
            "energy": [9.0, 10.0],
            "mz_rounded": [28, 28],
            "mz": [28.01, 28.02],
            "normalized_intensity": [1.0, 2.0],
            "raw_area": [10.0, 20.0],
            "photon_normalized_intensity": [1.0, 2.0],
            "species": ["CO", "CO"],
            "integration_method": ["sum_counts", "sum_counts"],
            "file_count": [1, 1],
            "io": [1.0, 1.0],
            "replicate_mode": ["off", "off"],
        }
    )

    writer = PIESpeciesFitDialog(Calibration(a=0.0, b=1.0, c=0.0))
    writer.project_settings = ps
    writer.project_dir = str(project_root(ps))
    writer.set_pie_source_scope("project", apply_project=False)
    writer.calibration = ps.to_calibration()
    writer.photon_correction_check.setChecked(ps.pie_photon_mode != "off")
    writer.analysis_df = analysis_df
    cache_key = writer._pie_cache_key([str(raw_dir)], None)
    writer._save_pie_analysis_cache(
        cache_key,
        {"folder_count": 1, "folders": [str(raw_dir)], "source_scope": "project"},
    )
    writer.deleteLater()

    restored = PIESpeciesFitDialog(Calibration(a=0.0, b=1.0, c=0.0))
    try:
        restored.set_project_settings(ps, activate_project_scope=True)
        timer = QtCore.QElapsedTimer()
        timer.start()
        while timer.elapsed() < 3000 and not restored.curves:
            qapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)

        assert 28 in restored.curves
        assert restored.analysis_df["normalized_intensity"].tolist() == [1.0, 2.0]
        assert restored._last_analysis_source_info["from_cache"] is True
        assert "自动载入" in restored.status_label.text()
        (raw_dir / "10.0eV.txt").write_text(
            "source data changed; cached curves must be invalidated",
            encoding="utf-8",
        )
        assert restored._pie_cache_key([str(raw_dir)], None) != cache_key
    finally:
        worker = restored._autoload_worker
        if worker is not None and worker.isRunning():
            worker.wait(3000)
        restored.deleteLater()


def test_project_state_restore_rebuilds_live_fit_model(pie_dialog, tmp_path):
    project_dir = tmp_path / "Project_PIE_State"
    ps = ProjectSettings(project_name="PIE State", output_dir=str(project_dir))
    curves = {
        28: {
            "mz": 28,
            "energies": [9.0, 10.0],
            "intensities": [1.0, 2.0],
            "species": "CO",
            "rows": pd.DataFrame(),
        }
    }
    database = [
        {
            "id": 1,
            "species": "CO",
            "ionization_energy": 9.0,
            "cross_sections": [1.0, 2.0],
        }
    ]
    config = {
        28: {
            "selected_species": [{"id": 1, "species": "CO"}],
            "coefficients": {1: 1.0},
            "locked_ids": [],
        }
    }
    config_hash = pie_dialog._compute_config_hash(
        pie_dialog._build_per_mz_config_payload(config[28])
    )
    fit_results = {
        28: {
            "success": True,
            "model": {
                "r_squared": 0.99,
                "rmse": 0.01,
                "mae": 0.01,
                "species": [
                    {
                        "id": 1,
                        "species": "CO",
                        "formula": "CO",
                        "smiles": "[C-]#[O+]",
                        "coefficient": 1.0,
                        "contribution_percent": 100.0,
                        "ie": 9.0,
                    }
                ],
                "fitted_curve": [1.0, 2.0],
                "component_curves": [[1.0], [2.0]],
            },
            "fit_config_hash": config_hash,
            "global_config_hash": pie_dialog.global_solver_config["config_hash"],
        }
    }
    manager = PieStateManager(str(project_dir))
    success, error = manager.save_state(
        curves,
        database,
        pie_dialog.calibration,
        config,
        fit_results,
        pie_dialog.global_solver_config["config_hash"],
    )
    assert success, error

    pie_dialog.project_settings = ps
    pie_dialog.project_dir = str(project_dir)
    pie_dialog.set_pie_source_scope("project", apply_project=False)
    pie_dialog.curves = curves
    pie_dialog.database = database
    pie_dialog._restore_pie_project_state()

    restored = pie_dialog.all_fit_results[28]
    assert restored["model"]["fitted"] == [1.0, 2.0]
    assert restored["model"]["species"][0]["component_intensities"] == [1.0, 2.0]
    assert restored["model"]["species"][0]["formula"] == "CO"
    assert restored["model"]["species"][0]["smiles"] == "[C-]#[O+]"
    assert pie_dialog._derive_result_status(
        restored,
        pie_dialog._get_per_mz_config_hash(28),
        pie_dialog.global_solver_config["config_hash"],
    ) == "COMPLETED"


def test_ie_query_batch_keeps_missing_species_nonfatal(monkeypatch):
    nitric_oxide = NistCompoundIonization(
        nist_id="C10102439",
        name="Nitric oxide",
        formula="NO",
        cas_rn="10102-43-9",
        url=None,
        ion_energetics_url=None,
        evaluated_ie=NistIonizationEnergy(9.2642, source="evaluated"),
    )

    class FakeClient:
        def query_ionization_energy(self, query, *, search_type="auto"):
            if query == "Nitric oxide":
                return NistWebBookResult(
                    query=query,
                    search_type=search_type,
                    requested_url="https://example.test/no",
                    compounds=(nitric_oxide,),
                    selected_compound=nitric_oxide,
                    message="命中",
                )
            return NistWebBookResult(
                query=query,
                search_type=search_type,
                requested_url="https://example.test/missing",
                message="未找到",
            )

    monkeypatch.setattr(pie_dialog_module, "default_nist_webbook_client", lambda: FakeClient())
    results = PIESpeciesFitDialog._query_ie_requests_sync([
        {"cache_key": "no", "species": "Nitric oxide", "formula": "NO", "mz": 30},
        {"cache_key": "missing", "species": "Unknown species", "formula": "", "mz": 30},
    ])

    assert results[0]["status"] == "available"
    assert results[0]["value"] == 9.2642
    assert results[0]["source"] == "NIST WebBook"
    assert results[1]["status"] == "not_found"
    assert results[1]["value"] is None


def test_ie_lookup_result_updates_candidate_and_existing_fit(pie_dialog, monkeypatch):
    pie_dialog.database = [
        {
            "id": 72,
            "mz": 30,
            "species": "Nitric oxide",
            "formula": "NO",
            "ie": None,
            "ionization_energy": None,
            "energies": np.array([9.0, 10.0, 11.0]),
            "cross_sections": np.array([0.0, 1.0, 2.0]),
        }
    ]
    pie_dialog.current_mz = 30
    monkeypatch.setattr(pie_dialog, "_start_ie_lookup_worker", lambda: None)
    pie_dialog._populate_candidate_table(30)
    assert pie_dialog.species_table.item(0, 2).text() == "查询中…"

    pie_dialog.current_fit = {
        "species": [
            {
                "ids": [72],
                "mz": 30,
                "species": "Nitric oxide",
                "formula": "NO",
                "ie": None,
            }
        ]
    }
    pie_dialog._apply_ie_lookup_result({
        "cache_key": pie_dialog._species_ie_cache_key(pie_dialog.database[0]),
        "value": 9.2642,
        "status": "available",
        "source": "NIST WebBook",
        "message": "evaluated IE",
    })

    assert pie_dialog.database[0]["ie"] == 9.2642
    assert pie_dialog.current_fit["species"][0]["ie"] == 9.2642
    assert pie_dialog.species_table.item(0, 2).text() == "9.2642"


def test_pie_source_scope_defaults_to_temporary_without_project(pie_dialog):
    assert pie_dialog.pie_source_scope == "temporary"
    assert pie_dialog.temporary_source_button.isChecked()
    assert not pie_dialog.folder_edit.isReadOnly()


def test_project_scope_applies_project_source_and_restores_temporary_path(pie_dialog, tmp_path):
    temporary_folder = tmp_path / "temporary_pie"
    project_folder = tmp_path / "project_pie"
    temporary_folder.mkdir()
    project_folder.mkdir()

    pie_dialog.folder_edit.setText(str(temporary_folder))
    ps = ProjectSettings(
        project_name="Project PIE",
        output_dir=str(tmp_path / "project"),
        pie_scan_folder=str(project_folder),
    )

    pie_dialog.set_project_settings(ps, activate_project_scope=True)

    assert pie_dialog.pie_source_scope == "project"
    assert pie_dialog.project_source_button.isChecked()
    assert pie_dialog.folder_edit.text() == str(project_folder)
    assert pie_dialog.folder_edit.isReadOnly()

    pie_dialog.set_pie_source_scope("temporary")

    assert pie_dialog.pie_source_scope == "temporary"
    assert pie_dialog.temporary_source_button.isChecked()
    assert pie_dialog.folder_edit.text() == str(temporary_folder)
    assert not pie_dialog.folder_edit.isReadOnly()


def test_pie_project_scope_uses_project_calibration(pie_dialog, tmp_path):
    project_calibration = Calibration(a=7.65e-7, b=3.21e-4, c=0.654)
    project_folder = tmp_path / "project_pie"
    project_folder.mkdir()
    ps = ProjectSettings(
        project_name="Calibrated PIE",
        output_dir=str(tmp_path / "project"),
        pie_scan_folder=str(project_folder),
        cal_a=project_calibration.a,
        cal_b=project_calibration.b,
        cal_c=project_calibration.c,
        light_source="beam_current",
        expansion_factors={450.0: 1.1},
    )

    pie_dialog.set_project_settings(ps, activate_project_scope=True)

    assert pie_dialog.calibration == project_calibration
    assert pie_dialog.normalization_settings.light_source == "beam_current"
    assert pie_dialog.normalization_settings.expansion_factors == {450.0: 1.1}
    assert pie_dialog._pie_cache_parameters([str(project_folder)], None)["calibration"] == {
        "a": project_calibration.a,
        "b": project_calibration.b,
        "c": project_calibration.c,
    }


def test_pie_ignores_analysis_result_from_previous_project(pie_dialog, monkeypatch):
    applied = []
    monkeypatch.setattr(pie_dialog, "on_analysis_complete", lambda result: applied.append(result))
    pie_dialog._analysis_request_id = 9

    pie_dialog._on_analysis_result(8, ("stale",))
    pie_dialog._on_analysis_result(9, ("current",))

    assert applied == [("current",)]


def test_switching_from_project_scope_persists_dirty_results(pie_dialog, tmp_path, monkeypatch):
    project_folder = tmp_path / "project_pie"
    project_folder.mkdir()
    ps = ProjectSettings(
        project_name="Project PIE",
        output_dir=str(tmp_path / "project"),
        pie_scan_folder=str(project_folder),
    )
    pie_dialog.set_project_settings(ps, activate_project_scope=True)
    calls = []
    monkeypatch.setattr(
        pie_dialog,
        "persist_pie_project_state",
        lambda: (calls.append(True) or True, None),
    )
    pie_dialog.pie_state_dirty = True

    pie_dialog.set_pie_source_scope("temporary")

    assert calls == [True]
    assert pie_dialog.pie_source_scope == "temporary"


def test_failed_persistence_keeps_project_scope_and_results(pie_dialog, tmp_path, monkeypatch):
    project_folder = tmp_path / "project_pie"
    project_folder.mkdir()
    ps = ProjectSettings(
        project_name="Project PIE",
        output_dir=str(tmp_path / "project"),
        pie_scan_folder=str(project_folder),
    )
    pie_dialog.set_project_settings(ps, activate_project_scope=True)
    pie_dialog.all_fit_results = {28: {"success": True}}
    pie_dialog.pie_state_dirty = True
    warnings = []
    monkeypatch.setattr(
        pie_dialog,
        "persist_pie_project_state",
        lambda: (False, "disk full"),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    pie_dialog.set_pie_source_scope("temporary")

    assert pie_dialog.pie_source_scope == "project"
    assert pie_dialog.all_fit_results == {28: {"success": True}}
    assert warnings


def test_unreadable_pie_folder_fingerprint_does_not_abort(pie_dialog, tmp_path, monkeypatch):
    folder = tmp_path / "pie"
    folder.mkdir()

    def deny_access(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", deny_access)
    result = pie_dialog._pie_folder_fingerprints([str(folder)], recursive=False)

    assert result[0]["unreadable"] is True
    assert result[0]["files"] == []


def test_project_scope_restores_all_pie_segment_folders(pie_dialog, tmp_path):
    low_folder = tmp_path / "pie_low"
    high_folder = tmp_path / "pie_high"
    low_folder.mkdir()
    high_folder.mkdir()
    ps = ProjectSettings(
        project_name="Project Multi PIE",
        output_dir=str(tmp_path / "project"),
        pie_scan_folder=str(low_folder),
        pie_scan_folders=[str(low_folder), str(high_folder)],
        pie_multi_folder_mode=True,
        pie_merge_method="low_energy_dominant",
    )

    pie_dialog.set_project_settings(ps, activate_project_scope=True)

    assert pie_dialog.pie_source_scope == "project"
    assert pie_dialog.project_source_button.text() == "项目数据"
    assert not hasattr(pie_dialog, "use_multi_folders")
    assert not hasattr(pie_dialog, "multi_folder_section")
    assert pie_dialog.folder_edit.text() == "项目管理已登记 2 个 PIE 能段目录"
    assert pie_dialog.folder_edit.toolTip().splitlines() == [str(low_folder), str(high_folder)]
    assert pie_dialog.select_folder_button.text() == "管理能段..."
    assert not pie_dialog.select_folder_button.isHidden()
    assert pie_dialog.temporary_segments_button.isHidden()
    assert pie_dialog._has_analysis_source()
    folders, merge_method = pie_dialog._analysis_source_selection()
    assert folders == [str(low_folder), str(high_folder)]
    assert merge_method == "low_energy_dominant"

    pie_dialog.set_pie_source_scope("temporary")

    assert pie_dialog.select_folder_button.text() == "浏览..."
    assert not pie_dialog.select_folder_button.isHidden()
    assert not pie_dialog.temporary_segments_button.isHidden()


def test_temporary_scope_selects_discovered_segments_inside_pie_page(
    pie_dialog, tmp_path
):
    root = tmp_path / "PIE-total"
    low_folder = root / "PIE-low"
    high_folder = root / "PIE-high"
    low_folder.mkdir(parents=True)
    high_folder.mkdir()
    pie_dialog.folder_edit.setText(str(root))
    pie_dialog._temporary_segment_root = str(root)
    pie_dialog._temporary_segment_summaries = [
        PieSegmentSummary(low_folder, 23, 7.0, 8.1),
        PieSegmentSummary(high_folder, 61, 8.0, 11.0),
    ]
    pie_dialog._set_temporary_segment_selection(
        [str(low_folder), str(high_folder)]
    )

    assert pie_dialog.temporary_segments_button.text() == "能段: 2/2..."
    assert "7.000–8.100 eV" in pie_dialog.temporary_segments_button.toolTip()
    assert "8.000–11.000 eV" in pie_dialog.temporary_segments_button.toolTip()
    folders, merge_method = pie_dialog._analysis_source_selection()
    assert folders == [str(low_folder), str(high_folder)]
    assert merge_method == "low_energy_dominant"

    pie_dialog._set_temporary_segment_selection([str(low_folder)])

    assert pie_dialog.temporary_segments_button.text() == "能段: 1/2..."
    folders, merge_method = pie_dialog._analysis_source_selection()
    assert folders == [str(low_folder)]
    assert merge_method is None


def test_project_settings_sync_respects_temporary_source_scope(pie_dialog, tmp_path):
    temporary_folder = tmp_path / "temporary_pie"
    project_folder = tmp_path / "project_pie"
    updated_project_folder = tmp_path / "updated_project_pie"
    for folder in (temporary_folder, project_folder, updated_project_folder):
        folder.mkdir()

    ps = ProjectSettings(
        project_name="Project PIE",
        output_dir=str(tmp_path / "project"),
        pie_scan_folder=str(project_folder),
    )
    pie_dialog.folder_edit.setText(str(temporary_folder))
    pie_dialog.set_project_settings(ps, activate_project_scope=True)
    pie_dialog.set_pie_source_scope("temporary")

    ps.pie_scan_folder = str(updated_project_folder)
    pie_dialog.set_project_settings(ps)

    assert pie_dialog.pie_source_scope == "temporary"
    assert pie_dialog.folder_edit.text() == str(temporary_folder)


def test_temporary_pie_parameters_are_an_isolated_project_snapshot(pie_dialog, tmp_path):
    project_folder = tmp_path / "project_pie"
    project_folder.mkdir()
    ps = ProjectSettings(
        project_name="PIE temporary parameters",
        output_dir=str(tmp_path / "project"),
        pie_scan_folder=str(project_folder),
        cal_a=3.5e-6,
        cal_b=0.004,
        cal_c=-1.25,
        pie_energy_decimals=3,
        pie_integration_method="baseline",
        min_intensity=17.0,
    )
    pie_dialog.set_project_settings(ps, activate_project_scope=True)
    pie_dialog.set_pie_source_scope("temporary")

    effective = pie_dialog._effective_analysis_settings()
    assert pie_dialog.temporary_settings_source == "project"
    assert pie_dialog.temporary_params_panel.title() == "临时数据参数"
    assert pie_dialog.temporary_params_label.text() == "当前来源：项目参数副本"
    assert pie_dialog.temporary_params_button.text() == "编辑本次参数…"
    assert pie_dialog.copy_project_params_action.isEnabled()
    assert pie_dialog.common_params_action.text() == "编辑本次临时参数…"
    assert effective is not ps
    assert effective.to_calibration() == ps.to_calibration()
    assert effective.pie_energy_decimals == 3
    assert effective.pie_integration_method == "baseline"
    assert effective.to_peak_detection_config().min_intensity == 17.0
    assert pie_dialog._pie_cache_dir() != tmp_path / "project" / "analysis" / "pie" / "cache"

    effective.cal_a = 8.0
    effective.pie_energy_decimals = 5
    assert ps.cal_a == 3.5e-6
    assert ps.pie_energy_decimals == 3


def test_temporary_parameter_menu_disables_project_copy_without_project(pie_dialog):
    assert pie_dialog.pie_source_scope == "temporary"
    assert pie_dialog.temporary_params_label.text() == "当前来源：全局参数副本"
    assert pie_dialog.temporary_params_button.text() == "编辑本次参数…"
    assert not pie_dialog.copy_project_params_action.isEnabled()
    assert pie_dialog.common_params_action.text() == "编辑本次临时参数…"
    assert pie_dialog.common_params_button.isHidden()

    pie_dialog._select_temporary_settings_source("default")
    assert pie_dialog.temporary_settings_source == "default"
    assert pie_dialog.temporary_params_label.text() == "当前来源：程序默认参数"


def test_analysis_and_fit_actions_follow_real_prerequisites(pie_dialog, tmp_path):
    assert not pie_dialog.analyze_button.isEnabled()
    assert not pie_dialog.fit_button.isEnabled()
    assert pie_dialog.left_panel.isHidden()
    assert pie_dialog.fitting_control_widget.isHidden()
    assert pie_dialog.stats_bar.isHidden()
    assert pie_dialog.export_button.isHidden()
    assert pie_dialog.export_plot_button.isHidden()

    source = tmp_path / "pie"
    source.mkdir()
    pie_dialog.folder_edit.setText(str(source))
    assert pie_dialog.analyze_button.isEnabled()

    pie_dialog.curves = {
        30: {
            "mz": 30,
            "energies": np.array([9.0, 10.0]),
            "intensities": np.array([0.0, 1.0]),
            "rows": pd.DataFrame({
                "energy": [9.0, 10.0],
                "integration_method": ["sum_counts", "sum_counts"],
                "raw_area": [0.0, 1.0],
                "photon_normalized_intensity": [0.0, 1.0],
                "normalized_intensity": [0.0, 1.0],
            }),
        }
    }
    pie_dialog.database = [
        {
            "id": 1,
            "mz": 30,
            "species": "NO",
            "ie": 9.264,
            "ionization_energy": 9.264,
            "energies": np.array([9.0, 10.0]),
            "cross_sections": np.array([0.0, 1.0]),
        }
    ]
    pie_dialog.populate_mz_list()
    pie_dialog.mz_list.setCurrentRow(0)
    assert pie_dialog.fit_button.isEnabled()
    assert not pie_dialog.left_panel.isHidden()
    assert not pie_dialog.fitting_control_widget.isHidden()
    assert not pie_dialog.stats_bar.isHidden()
    assert not pie_dialog.export_button.isHidden()
    assert pie_dialog.export_plot_button.isHidden()
    assert pie_dialog.export_curve_action.isEnabled()
    assert pie_dialog.export_plot_action.isEnabled()
    assert pie_dialog.fitting_control_widget.candidate_controls_widget.isVisible()

    pie_dialog.fitting_control_widget._set_all_rows_checked(False)
    assert not pie_dialog.fit_button.isEnabled()
    assert "至少启用一个" in pie_dialog.fit_button.toolTip()


def test_selected_fit_jobs_keep_candidates_scoped_to_each_mz(pie_dialog):
    pie_dialog.current_mz = None
    pie_dialog.curves = {
        30: {"mz": 30, "energies": [9.0, 10.0], "intensities": [0.0, 1.0]},
        44: {"mz": 44, "energies": [9.0, 10.0], "intensities": [0.0, 1.0]},
    }
    pie_dialog.database = [
        {"id": 1, "mz": 30, "species": "NO", "energies": [9.0, 10.0], "cross_sections": [0.0, 1.0]},
        {"id": 2, "mz": 44, "species": "CO2", "energies": [9.0, 10.0], "cross_sections": [0.0, 1.0]},
    ]

    jobs = pie_dialog._build_selected_fit_jobs([30, 44])

    assert [item["species"] for item in jobs[30]["selected_species"]] == ["NO"]
    assert [item["species"] for item in jobs[44]["selected_species"]] == ["CO2"]


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

    def test_refit_selected_button_removed(self, pie_dialog):
        """Test that refit_selected_button has been moved to more_actions menu."""
        # Button no longer exists as standalone attribute - moved to menu
        assert not hasattr(pie_dialog, 'refit_selected_button')

    def test_more_actions_menu_exists(self, pie_dialog):
        """Test that more_actions menu button exists."""
        assert hasattr(pie_dialog, 'more_actions_btn')
        assert hasattr(pie_dialog, 'more_actions_menu')




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
