"""Test artifact management and scanning functionality."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bl03u_masstool.core.project_lifecycle import (
    ArtifactCategory,
    ArtifactRecord,
    ArtifactStatus,
    ProjectSettings,
    ensure_project_structure,
    project_root,
    scan_project_artifacts,
)


class TestArtifactRecord:
    """Test ArtifactRecord dataclass."""

    def test_artifact_record_creation(self):
        """Test creating an artifact record."""
        record = ArtifactRecord(
            artifact_type="temperature_scan_result",
            category="temperature_scan",
            path="/path/to/result.csv",
            generation_time="2026-06-20T12:00:00",
            size_bytes=1024,
            source_module="TemperatureModule",
            status=ArtifactStatus.VALID.value,
        )
        assert record.artifact_type == "temperature_scan_result"
        assert record.category == "temperature_scan"
        assert record.status == "有效"
        assert record.size_bytes == 1024

    def test_artifact_record_with_detail(self):
        """Test artifact record with detail message."""
        record = ArtifactRecord(
            artifact_type="pie_identification_result",
            category="pie",
            path="/path/to/pie_result.xlsx",
            generation_time="",
            size_bytes=0,
            source_module="PIEModule",
            status=ArtifactStatus.MISSING.value,
            detail="文件不存在或已删除",
        )
        assert record.status == "缺失"
        assert record.detail == "文件不存在或已删除"


class TestArtifactCategory:
    """Test ArtifactCategory enum."""

    def test_artifact_categories_exist(self):
        """Test that all required categories exist."""
        categories = list(ArtifactCategory)
        category_keys = {c.key for c in categories}
        expected_keys = {
            "intermediate",
            "temperature_scan",
            "pie",
            "mole_fraction",
            "pics",
            "reports",
            "snapshots",
        }
        assert category_keys == expected_keys

    def test_artifact_category_properties(self):
        """Test category properties."""
        category = ArtifactCategory.TEMPERATURE
        assert category.key == "temperature_scan"
        assert category.label == "温度扫描"
        assert category.directory_key == "temperature_scan"


class TestArtifactStatus:
    """Test ArtifactStatus enum."""

    def test_artifact_status_values(self):
        """Test artifact status values."""
        assert ArtifactStatus.VALID.value == "有效"
        assert ArtifactStatus.MISSING.value == "缺失"
        assert ArtifactStatus.EXPIRED.value == "已过期"
        assert ArtifactStatus.INCOMPLETE.value == "不完整"


class TestScanProjectArtifacts:
    """Test scan_project_artifacts function."""

    def test_scan_empty_project(self):
        """Test scanning artifacts in empty project."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)
            artifacts = scan_project_artifacts(settings)
            # Should have no artifacts since no result files are registered
            assert len(artifacts) == 0

    def test_scan_project_with_missing_artifacts(self):
        """Test scanning project with missing registered artifacts."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
                temperature_scan_result_file="/nonexistent/path/temp_result.csv",
            )
            ensure_project_structure(settings)
            artifacts = scan_project_artifacts(settings)
            assert len(artifacts) == 1
            assert artifacts[0].artifact_type == "temperature_scan_result"
            assert artifacts[0].status == ArtifactStatus.MISSING.value
            assert artifacts[0].detail == "文件不存在或已删除"

    def test_scan_project_with_valid_artifacts(self):
        """Test scanning project with valid artifacts."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Create a test result file
            result_file = Path(tmpdir) / "temperature_result.csv"
            result_file.write_text("m/z,intensity\n100,1000\n200,2000\n")

            settings.temperature_scan_result_file = str(result_file)
            artifacts = scan_project_artifacts(settings)

            assert len(artifacts) == 1
            artifact = artifacts[0]
            assert artifact.artifact_type == "temperature_scan_result"
            assert artifact.status == ArtifactStatus.VALID.value
            assert artifact.size_bytes > 0
            assert artifact.generation_time != ""
            assert artifact.path == str(result_file)

    def test_scan_multiple_artifact_types(self):
        """Test scanning multiple artifact types."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Create multiple result files
            temp_result = Path(tmpdir) / "temp_result.csv"
            pie_result = Path(tmpdir) / "pie_result.xlsx"
            mf_result = Path(tmpdir) / "mf_result.xlsx"

            temp_result.write_text("temperature_scan_result")
            pie_result.write_text("pie_identification_result")
            mf_result.write_text("mole_fraction_result")

            settings.temperature_scan_result_file = str(temp_result)
            settings.pie_identification_result_file = str(pie_result)
            settings.mole_fraction_result_file = str(mf_result)

            artifacts = scan_project_artifacts(settings)

            assert len(artifacts) == 3
            artifact_types = {a.artifact_type for a in artifacts}
            assert artifact_types == {
                "temperature_scan_result",
                "pie_identification_result",
                "mole_fraction_result",
            }

            # All should be valid
            for artifact in artifacts:
                assert artifact.status == ArtifactStatus.VALID.value

    def test_scan_project_with_snapshots(self):
        """Test scanning project snapshots."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Create a snapshot file
            versions_dir = project_root(settings) / "versions"
            versions_dir.mkdir(parents=True, exist_ok=True)
            snapshot_file = versions_dir / "20260620_120000_backup.zip"
            snapshot_file.write_text("snapshot_data")

            artifacts = scan_project_artifacts(settings)

            # Should only have the snapshot
            assert len(artifacts) == 1
            artifact = artifacts[0]
            assert artifact.artifact_type == "snapshot"
            assert artifact.category == "snapshots"
            assert artifact.status == ArtifactStatus.VALID.value
            assert artifact.source_module == "ProjectManager"

    def test_scan_artifact_categorization(self):
        """Test artifact categorization."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Create result files
            temp_result = Path(tmpdir) / "temp_result.csv"
            pie_result = Path(tmpdir) / "pie_result.xlsx"
            mf_result = Path(tmpdir) / "mf_result.xlsx"
            manual_peak = Path(tmpdir) / "manual_peak.yaml"

            for f in [temp_result, pie_result, mf_result, manual_peak]:
                f.write_text("test_data")

            settings.temperature_scan_result_file = str(temp_result)
            settings.pie_identification_result_file = str(pie_result)
            settings.mole_fraction_result_file = str(mf_result)
            settings.manual_peak_file = str(manual_peak)

            artifacts = scan_project_artifacts(settings)

            # Verify categorization
            categories = {a.category for a in artifacts}
            assert "temperature_scan" in categories
            assert "pie" in categories
            assert "mole_fraction" in categories
            assert "intermediate" in categories

    def test_scan_mixed_valid_and_invalid(self):
        """Test scanning mix of valid and invalid artifacts."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Create one valid file
            temp_result = Path(tmpdir) / "temp_result.csv"
            temp_result.write_text("data")

            # Point to nonexistent files
            settings.temperature_scan_result_file = str(temp_result)
            settings.pie_identification_result_file = "/nonexistent/pie_result.xlsx"
            settings.mole_fraction_result_file = "/nonexistent/mf_result.xlsx"

            artifacts = scan_project_artifacts(settings)

            assert len(artifacts) == 3
            valid_count = sum(1 for a in artifacts if a.status == ArtifactStatus.VALID.value)
            missing_count = sum(1 for a in artifacts if a.status == ArtifactStatus.MISSING.value)

            assert valid_count == 1
            assert missing_count == 2
