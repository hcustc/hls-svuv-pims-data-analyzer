from __future__ import annotations

import zipfile

from bl03u_masstool.core.project_lifecycle import (
    build_project_stage_statuses,
    collect_project_files,
    create_project_snapshot,
    DATA_SOURCE_FILE_COUNT_LIMIT,
    ensure_project_structure,
    export_project_archive,
    import_initial_project_data,
    next_project_stage,
    validate_data_source,
)
from bl03u_masstool.core.project_settings import ProjectSettings


def test_project_lifecycle_creates_structure_and_imports_initial_data(tmp_path):
    single_file = tmp_path / "source" / "single.txt"
    single_file.parent.mkdir()
    single_file.write_text("tof intensity\n1 2\n", encoding="utf-8")

    settings = ProjectSettings(
        project_name="C6F11O2H",
        system="C6F11O2H",
        output_dir=str(tmp_path / "Project_C6F11O2H"),
    )

    directories = ensure_project_structure(settings)
    assert directories["raw_data"].exists()
    assert directories["pie_analysis"].exists()
    assert directories["pie_analysis"] == tmp_path / "Project_C6F11O2H" / "analysis" / "pie"
    assert not (tmp_path / "Project_C6F11O2H" / "final_report").exists()

    results = import_initial_project_data(
        settings,
        {
            "single_spectrum": single_file,
        },
    )

    assert len(results) == 1
    assert settings.single_spectrum_file.endswith("raw_data/single_spectrum/single.txt")
    linked_file = tmp_path / "Project_C6F11O2H" / "raw_data" / "single_spectrum" / "single.txt"
    assert linked_file.is_symlink()
    assert linked_file.resolve() == single_file.resolve()
    manifest = tmp_path / "Project_C6F11O2H" / "raw_data" / "_sources.yaml"
    assert manifest.exists()

    statuses = {status.key: status for status in build_project_stage_statuses(settings)}
    assert statuses["project_setup"].completed is True
    assert statuses["raw_data"].completed is True
    assert next_project_stage(settings).key == "calibration"


def test_project_lifecycle_collects_registered_outputs_and_exports_archive(tmp_path):
    project_root = tmp_path / "Project_C6F11O2H"
    temp_result = project_root / "analysis" / "temperature_scan" / "temperature.xlsx"
    temp_result.parent.mkdir(parents=True)
    temp_result.write_text("temperature result", encoding="utf-8")
    external_pie = tmp_path / "external" / "pie.xlsx"
    external_pie.parent.mkdir()
    external_pie.write_text("pie result", encoding="utf-8")

    settings = ProjectSettings(
        project_name="C6F11O2H",
        system="C6F11O2H",
        output_dir=str(project_root),
        temperature_scan_result_file=str(temp_result),
        pie_identification_result_file=str(external_pie),
    )

    records = collect_project_files(settings)
    record_names = {record.path.name for record in records}
    assert "temperature.xlsx" in record_names
    assert "pie.xlsx" in record_names
    assert any(record.registered for record in records if record.path.name == "pie.xlsx")

    archive_path = export_project_archive(settings, tmp_path / "exports" / "project.zip")
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert f"{project_root.name}/project_state.yaml" in names
    assert f"{project_root.name}/analysis/temperature_scan/temperature.xlsx" in names
    assert f"{project_root.name}/_registered_external/pie_identification_result_file/pie.xlsx" in names

    snapshot_path = create_project_snapshot(settings, "after pie")
    assert snapshot_path.exists()
    assert snapshot_path.parent == project_root / "versions"


def test_data_source_validation_caps_large_directory_counts(tmp_path):
    source_dir = tmp_path / "large_source"
    source_dir.mkdir()
    for index in range(DATA_SOURCE_FILE_COUNT_LIMIT + 3):
        (source_dir / f"{index}.txt").write_text("data", encoding="utf-8")

    settings = ProjectSettings(sum_spectrum_folder=str(source_dir))

    record = validate_data_source(settings, "sum_spectrum")

    assert record.is_valid is True
    assert record.file_count == DATA_SOURCE_FILE_COUNT_LIMIT
    assert record.detail == f"至少 {DATA_SOURCE_FILE_COUNT_LIMIT} 个文件"
