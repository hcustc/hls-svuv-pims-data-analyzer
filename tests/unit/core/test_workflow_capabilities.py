"""Test workflow capability analysis for project-owned raw scan folders."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from bl03u_masstool.core.project_lifecycle import (
    DataSourceValidationStatus,
    ProjectSettings,
    WorkflowProfile,
    analyze_workflow_capabilities,
    ensure_project_structure,
    get_data_source_validation_status,
    validate_all_data_sources,
)


def _folder(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "file.txt").write_text("data", encoding="utf-8")
    return path


class TestWorkflowCapabilityAnalysis:
    def test_spectrum_only_still_reflects_workbench_single_or_sum_source(self):
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(output_dir=tmpdir, project_name="test", system="test")
            ensure_project_structure(settings)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.SPECTRUM_ONLY not in result.available_workflows

            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data", encoding="utf-8")
            settings.single_spectrum_file = str(single_file)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.SPECTRUM_ONLY in result.available_workflows

    def test_temperature_scan_requires_only_temperature_scan_folder(self):
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(output_dir=tmpdir, project_name="test", system="test")
            ensure_project_structure(settings)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.TEMPERATURE_SCAN not in result.available_workflows

            settings.temperature_scan_folder = str(_folder(Path(tmpdir) / "temp_scan"))

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.TEMPERATURE_SCAN in result.available_workflows
            assert WorkflowProfile.FULL_ANALYSIS in result.available_workflows

    def test_pie_analysis_requires_only_pie_scan_folder(self):
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(output_dir=tmpdir, project_name="test", system="test")
            ensure_project_structure(settings)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.PIE_ANALYSIS not in result.available_workflows

            settings.pie_scan_folder = str(_folder(Path(tmpdir) / "pie_scan"))

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.PIE_ANALYSIS in result.available_workflows
            assert WorkflowProfile.FULL_ANALYSIS in result.available_workflows

    def test_temperature_and_pie_are_project_raw_sources_independent_of_workbench_source(self):
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(output_dir=tmpdir, project_name="test", system="test")
            ensure_project_structure(settings)
            settings.temperature_scan_folder = str(_folder(Path(tmpdir) / "temp_scan"))
            settings.pie_scan_folder = str(_folder(Path(tmpdir) / "pie_scan"))

            result = analyze_workflow_capabilities(settings)

            assert WorkflowProfile.SPECTRUM_ONLY not in result.available_workflows
            assert WorkflowProfile.TEMPERATURE_SCAN in result.available_workflows
            assert WorkflowProfile.PIE_ANALYSIS in result.available_workflows
            assert WorkflowProfile.FULL_ANALYSIS in result.available_workflows

    def test_recommended_next_step_prioritizes_raw_scan_folders(self):
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(output_dir=tmpdir, project_name="test", system="test")
            ensure_project_structure(settings)

            result = analyze_workflow_capabilities(settings)

            assert result.recommended_next_step
            assert "温度扫描" in result.recommended_next_step or "PIE" in result.recommended_next_step

    def test_data_source_status_keeps_workbench_sources_but_project_status_uses_scan_folders(self):
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(output_dir=tmpdir, project_name="test", system="test")
            ensure_project_structure(settings)
            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data", encoding="utf-8")
            settings.single_spectrum_file = str(single_file)

            records = validate_all_data_sources(settings)
            result = analyze_workflow_capabilities(settings, records)

            assert result.data_source_status["single_spectrum"] is True
            assert result.data_source_status["temperature_scan"] is False
            assert get_data_source_validation_status(settings, records) == DataSourceValidationStatus.UNCONFIGURED

            settings.temperature_scan_folder = str(_folder(Path(tmpdir) / "temp_scan"))
            records = validate_all_data_sources(settings)
            assert get_data_source_validation_status(settings, records) == DataSourceValidationStatus.PARTIAL

            settings.pie_scan_folder = str(_folder(Path(tmpdir) / "pie_scan"))
            records = validate_all_data_sources(settings)
            assert get_data_source_validation_status(settings, records) == DataSourceValidationStatus.COMPLETE
