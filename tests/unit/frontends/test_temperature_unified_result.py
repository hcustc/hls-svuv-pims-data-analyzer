from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets
except ImportError as exc:
    pytest.skip(
        f"PyQt6 display libraries not available: {exc}",
        allow_module_level=True,
    )

pytestmark = pytest.mark.gui

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.curve_database import (
    list_curve_datasets_read_only,
    project_curve_database_path,
)
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.temperature_scan import build_temperature_curves
from bl03u_masstool.frontends.pyqt_app.temperature.dialog import (
    TemperatureScanDialog,
)


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _energy_result(
    folder: Path,
    energy: float,
    areas: list[float],
) -> dict:
    rows = pd.DataFrame(
        {
            "temperature": [650.0, 750.0, 850.0],
            "file": [f"{energy}-a.txt", f"{energy}-b.txt", f"{energy}-c.txt"],
            "mz": [70.0, 70.0, 70.0],
            "area": areas,
            "raw_area": areas,
            "photon_normalized_area": areas,
            "expansion_lambda": [1.0, 1.0, 1.0],
            "integration_method": ["sum_counts"] * 3,
            "species": [""] * 3,
            "scan_energy": [energy] * 3,
            "scan_folder": [folder.name] * 3,
        }
    )
    return {
        "energy": energy,
        "folder": str(folder),
        "folder_label": folder.name,
        "result_df": rows,
        "curves": build_temperature_curves(rows),
    }


def test_energy_selector_limits_generation_to_selected_energy(
    qapp,
    tmp_path,
):
    root = tmp_path / "temperature"
    energy_11 = root / "11.0eV"
    energy_12 = root / "12.0eV"
    energy_11.mkdir(parents=True)
    energy_12.mkdir()
    (energy_11 / "scan.txt").write_text("1\n", encoding="utf-8")
    (energy_12 / "scan.txt").write_text("1\n", encoding="utf-8")

    dialog = TemperatureScanDialog(Calibration())
    try:
        dialog.folder_edit.setText(str(root))
        index = dialog.scan_folder_combo.findData(str(energy_11))
        assert index >= 0
        dialog.scan_folder_combo.setCurrentIndex(index)

        assert dialog.scan_folder_label.text() == "光子能量"
        assert not hasattr(dialog, "curve_batch_button")
        assert dialog._selected_view_folders() == [(11.0, str(energy_11))]
        assert dialog._generation_analysis_folders() == [(11.0, str(energy_11))]

        all_index = dialog.scan_folder_combo.findData(dialog.ALL_ENERGY_FOLDERS)
        dialog.scan_folder_combo.setCurrentIndex(all_index)
        assert dialog._generation_analysis_folders() == [
            (11.0, str(energy_11)),
            (12.0, str(energy_12)),
        ]
        assert dialog.body_splitter.handleWidth() == 7
        assert dialog._sidebar.minimumWidth() == 220
        assert dialog._sidebar.maximumWidth() == 360
        assert dialog.source_panel.layout().count() == 2
        assert dialog.source_panel.sizeHint().height() <= 120
    finally:
        dialog.deleteLater()


def test_project_temperature_all_energy_result_keeps_immutable_versions(
    qapp,
    tmp_path,
):
    root = tmp_path / "temperature"
    energy_11 = root / "11.0eV"
    energy_12 = root / "12.0eV"
    energy_11.mkdir(parents=True)
    energy_12.mkdir()
    (energy_11 / "scan.txt").write_text("1\n", encoding="utf-8")
    (energy_12 / "scan.txt").write_text("1\n", encoding="utf-8")
    settings = ProjectSettings(
        project_name="Unified temperature result",
        output_dir=str(tmp_path / "project"),
        temperature_scan_folder=str(root),
        curve_storage_mode="sqlite",
    )
    dialog = TemperatureScanDialog(Calibration())
    try:
        dialog.set_project_settings(
            settings,
            activate_project_scope=True,
            load_cached_results=False,
        )
        assert (
            dialog.scan_folder_combo.currentData()
            == dialog.ALL_ENERGY_FOLDERS
        )

        first_results = [
            _energy_result(energy_11, 11.0, [1.0, 2.0, 3.0]),
            _energy_result(energy_12, 12.0, [10.0, 20.0, 30.0]),
        ]
        dialog.on_analysis_complete(
            {
                "result_df": pd.concat(
                    [item["result_df"] for item in first_results],
                    ignore_index=True,
                ),
                "energy_results": first_results,
                "analysis_provenance": {"cache_key": "unified-v1"},
            }
        )

        database_path = project_curve_database_path(settings)
        datasets = list_curve_datasets_read_only(
            database_path,
            curve_type="temperature",
            include_stale=True,
        )
        assert len(datasets) == 1
        assert datasets[0].dataset_group == dialog.TEMPERATURE_DATASET_GROUP
        assert datasets[0].metadata["energy_folders"] == ["11.0eV", "12.0eV"]
        stored_rows = dialog.curves[70.0]["rows"]
        assert set(stored_rows["scan_folder"]) == {"11.0eV", "12.0eV"}
        first_dataset_id = datasets[0].dataset_id

        second_results = [
            _energy_result(energy_11, 11.0, [2.0, 4.0, 6.0]),
            _energy_result(energy_12, 12.0, [20.0, 40.0, 60.0]),
        ]
        dialog.on_analysis_complete(
            {
                "result_df": pd.concat(
                    [item["result_df"] for item in second_results],
                    ignore_index=True,
                ),
                "energy_results": second_results,
                "analysis_provenance": {"cache_key": "unified-v2"},
            }
        )

        datasets = list_curve_datasets_read_only(
            database_path,
            curve_type="temperature",
            include_stale=True,
        )
        assert len(datasets) == 2
        assert first_dataset_id in {
            dataset.dataset_id for dataset in datasets
        }
        assert dialog.curve_database_dataset_id != first_dataset_id
        assert [
            dataset.dataset_id for dataset in datasets if dataset.is_current
        ] == [dialog.curve_database_dataset_id]
    finally:
        dialog.deleteLater()
