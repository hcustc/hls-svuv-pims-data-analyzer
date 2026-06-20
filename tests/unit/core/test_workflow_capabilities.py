"""Test workflow capability analysis and flexible data source requirements."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bl03u_masstool.core.project_lifecycle import (
    ProjectSettings,
    WorkflowProfile,
    analyze_workflow_capabilities,
    ensure_project_structure,
)


class TestWorkflowCapabilityAnalysis:
    """Test workflow capability analysis system."""

    def test_spectrum_only_requires_single_or_sum(self):
        """Test that SPECTRUM_ONLY requires either single or sum spectrum."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Initially no spectrum data
            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.SPECTRUM_ONLY not in result.available_workflows

            # Add single spectrum
            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data")
            settings.single_spectrum_file = str(single_file)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.SPECTRUM_ONLY in result.available_workflows

    def test_temperature_requires_spectrum_and_temp_data(self):
        """Test that TEMPERATURE_SCAN requires spectrum + temperature folder."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Only spectrum - temperature should not be available
            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data")
            settings.single_spectrum_file = str(single_file)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.TEMPERATURE_SCAN not in result.available_workflows

            # Add temperature folder
            temp_dir = Path(tmpdir) / "temp_scan"
            temp_dir.mkdir()
            (temp_dir / "file.txt").write_text("data")
            settings.temperature_scan_folder = str(temp_dir)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.TEMPERATURE_SCAN in result.available_workflows

    def test_pie_requires_spectrum_and_pie_data(self):
        """Test that PIE_ANALYSIS requires spectrum + PIE folder."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Only spectrum - PIE should not be available
            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data")
            settings.single_spectrum_file = str(single_file)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.PIE_ANALYSIS not in result.available_workflows

            # Add PIE folder
            pie_dir = Path(tmpdir) / "pie_scan"
            pie_dir.mkdir()
            (pie_dir / "file.txt").write_text("data")
            settings.pie_scan_folder = str(pie_dir)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.PIE_ANALYSIS in result.available_workflows

    def test_sum_spectrum_alternative_to_single(self):
        """Test that sum spectrum can replace single spectrum."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Use sum spectrum instead of single
            sum_dir = Path(tmpdir) / "sum_spectra"
            sum_dir.mkdir()
            (sum_dir / "sum.ms").write_text("data")
            settings.sum_spectrum_folder = str(sum_dir)

            # Temperature folder
            temp_dir = Path(tmpdir) / "temp_scan"
            temp_dir.mkdir()
            (temp_dir / "file.txt").write_text("data")
            settings.temperature_scan_folder = str(temp_dir)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.TEMPERATURE_SCAN in result.available_workflows

    def test_no_spectrum_blocks_all_workflows(self):
        """Test that missing spectrum blocks all workflows."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Add temperature and PIE but no spectrum
            temp_dir = Path(tmpdir) / "temp_scan"
            temp_dir.mkdir()
            (temp_dir / "file.txt").write_text("data")
            settings.temperature_scan_folder = str(temp_dir)

            pie_dir = Path(tmpdir) / "pie_scan"
            pie_dir.mkdir()
            (pie_dir / "file.txt").write_text("data")
            settings.pie_scan_folder = str(pie_dir)

            result = analyze_workflow_capabilities(settings)
            assert len(result.available_workflows) == 0
            assert WorkflowProfile.SPECTRUM_ONLY not in result.available_workflows
            assert WorkflowProfile.TEMPERATURE_SCAN not in result.available_workflows
            assert WorkflowProfile.PIE_ANALYSIS not in result.available_workflows

    def test_full_analysis_requires_spectrum_and_at_least_one_analysis(self):
        """Test that FULL_ANALYSIS allows either temperature or PIE."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Only spectrum
            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data")
            settings.single_spectrum_file = str(single_file)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.FULL_ANALYSIS not in result.available_workflows

            # Add temperature only
            temp_dir = Path(tmpdir) / "temp_scan"
            temp_dir.mkdir()
            (temp_dir / "file.txt").write_text("data")
            settings.temperature_scan_folder = str(temp_dir)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.FULL_ANALYSIS in result.available_workflows

            # Replace temperature with PIE
            settings.temperature_scan_folder = ""
            pie_dir = Path(tmpdir) / "pie_scan"
            pie_dir.mkdir()
            (pie_dir / "file.txt").write_text("data")
            settings.pie_scan_folder = str(pie_dir)

            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.FULL_ANALYSIS in result.available_workflows

            # Both temperature and PIE
            settings.temperature_scan_folder = str(temp_dir)
            result = analyze_workflow_capabilities(settings)
            assert WorkflowProfile.FULL_ANALYSIS in result.available_workflows

    def test_missing_sources_are_reported(self):
        """Test that missing sources are reported in unavailable workflows."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            result = analyze_workflow_capabilities(settings)

            # All workflows should be unavailable
            assert len(result.unavailable_workflows) > 0

            # Check that missing sources are reported
            for workflow, missing_str in result.unavailable_workflows.items():
                assert len(missing_str) > 0

    def test_recommended_next_step_provided(self):
        """Test that recommended next step is provided."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            result = analyze_workflow_capabilities(settings)
            assert len(result.recommended_next_step) > 0

            # When workflows available
            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data")
            settings.single_spectrum_file = str(single_file)

            result = analyze_workflow_capabilities(settings)
            assert "可执行" in result.recommended_next_step or "工作流" in result.recommended_next_step

    def test_data_source_status_collected(self):
        """Test that individual data source status is collected."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Add some sources
            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data")
            settings.single_spectrum_file = str(single_file)

            result = analyze_workflow_capabilities(settings)

            # Check status collected
            assert "single_spectrum" in result.data_source_status
            assert result.data_source_status["single_spectrum"] is True
            assert result.data_source_status.get("sum_spectrum") is False

    def test_multiple_workflows_available_simultaneously(self):
        """Test that multiple workflows can be available at the same time."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Add spectrum + both temperature and PIE
            single_file = Path(tmpdir) / "single.ms"
            single_file.write_text("data")
            settings.single_spectrum_file = str(single_file)

            temp_dir = Path(tmpdir) / "temp_scan"
            temp_dir.mkdir()
            (temp_dir / "file.txt").write_text("data")
            settings.temperature_scan_folder = str(temp_dir)

            pie_dir = Path(tmpdir) / "pie_scan"
            pie_dir.mkdir()
            (pie_dir / "file.txt").write_text("data")
            settings.pie_scan_folder = str(pie_dir)

            result = analyze_workflow_capabilities(settings)

            # All four workflows should be available
            assert WorkflowProfile.SPECTRUM_ONLY in result.available_workflows
            assert WorkflowProfile.TEMPERATURE_SCAN in result.available_workflows
            assert WorkflowProfile.PIE_ANALYSIS in result.available_workflows
            assert WorkflowProfile.FULL_ANALYSIS in result.available_workflows
            assert len(result.available_workflows) == 4

    def test_missing_both_spectrum_sources(self):
        """Test workflow status when both single and sum spectrum are missing."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            # Add analysis data but no spectrum
            temp_dir = Path(tmpdir) / "temp_scan"
            temp_dir.mkdir()
            (temp_dir / "file.txt").write_text("data")
            settings.temperature_scan_folder = str(temp_dir)

            result = analyze_workflow_capabilities(settings)

            # All workflows should be unavailable
            assert len(result.available_workflows) == 0

            # Check that both spectrum sources are mentioned as missing
            all_missing = " ".join(str(v) for v in result.unavailable_workflows.values())
            assert "单谱" in all_missing or "累计" in all_missing or "谱" in all_missing
