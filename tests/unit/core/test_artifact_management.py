"""Test artifact management and scanning functionality."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import pytest

from bl03u_masstool.core.project_lifecycle import (
    ArtifactCategory,
    ArtifactRecord,
    ArtifactStatus,
    ProjectSettings,
    ensure_project_structure,
    export_project_archive,
    import_project_source,
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
            "isotope_correction",
            "mole_fraction",
            "pics",
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
            assert len(artifacts) >= 1
            assert any(a.artifact_type == "temperature_scan_result" for a in artifacts)
            assert any(a.status == ArtifactStatus.MISSING.value for a in artifacts)

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

            assert len(artifacts) >= 1
            temp_artifacts = [a for a in artifacts if a.artifact_type == "temperature_scan_result"]
            assert len(temp_artifacts) == 1

            artifact = temp_artifacts[0]
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

            registered_types = {a.artifact_type for a in artifacts if not "unregistered" in a.artifact_type}
            assert "temperature_scan_result" in registered_types
            assert "pie_identification_result" in registered_types
            assert "mole_fraction_result" in registered_types

            # All should be valid
            for artifact in artifacts:
                if not "unregistered" in artifact.artifact_type:
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

            # Should have the snapshot
            snapshot_artifacts = [a for a in artifacts if a.artifact_type == "snapshot"]
            assert len(snapshot_artifacts) > 0

            artifact = snapshot_artifacts[0]
            assert artifact.category == "snapshots"
            assert artifact.status == ArtifactStatus.VALID.value
            assert artifact.source_module == "ProjectManager"

    def test_scan_includes_unregistered_directory_files(self):
        """Test that scan includes unregistered files from project directories."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Create an unregistered file in spectrum_analysis directory
            spectrum_dir = project_root(settings) / "analysis" / "spectrum"
            unregistered_file = spectrum_dir / "some_result.txt"
            unregistered_file.write_text("unregistered content")

            artifacts = scan_project_artifacts(settings)

            # Should include the unregistered file
            unregistered = [a for a in artifacts if "unregistered" in a.artifact_type]
            assert len(unregistered) > 0
            assert any(str(unregistered_file) in a.path for a in unregistered)

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

            valid_count = sum(1 for a in artifacts if a.status == ArtifactStatus.VALID.value)
            missing_count = sum(1 for a in artifacts if a.status == ArtifactStatus.MISSING.value)

            assert valid_count >= 1
            assert missing_count >= 2


class TestProjectInitializationMarker:
    """Test project initialization marker file creation and validation."""

    def test_marker_file_created_on_structure_init(self):
        """Test that marker file is created when project structure is initialized."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            marker_path = project_root(settings) / ".bl03u_project"
            assert marker_path.exists()
            assert marker_path.is_file()

    def test_marker_file_persists(self):
        """Test that marker file persists after structure initialization."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)
            marker_path = project_root(settings) / ".bl03u_project"
            first_mtime = marker_path.stat().st_mtime

            # Initialize again - marker should not be recreated
            ensure_project_structure(settings)
            second_mtime = marker_path.stat().st_mtime

            # Timestamp should be the same (file not rewritten)
            assert first_mtime == second_mtime


class TestImportSafety:
    """Test import_project_source safety validations."""

    def test_import_rejects_self_containment(self):
        """Test that importing a directory into itself is rejected."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Try to import project root as data source
            root = project_root(settings)

            with pytest.raises(ValueError, match="destination is inside source"):
                import_project_source(settings, root, "sum_spectrum")

    def test_import_normal_external_source_succeeds(self):
        """Test that importing from external directory links it into raw_data."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Create external data source
            external_dir = Path(tmpdir) / "external_data"
            external_dir.mkdir()
            (external_dir / "test.txt").write_text("test data")

            # This should succeed
            result = import_project_source(settings, external_dir, "sum_spectrum")
            assert result.mode == "link"
            assert result.destination.is_symlink()
            assert result.destination.resolve() == external_dir.resolve()
            assert (result.destination / "test.txt").exists()

    def test_copy_mode_still_copies_external_source(self):
        """Test explicit copy mode for archival workflows."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            external_dir = Path(tmpdir) / "external_data"
            external_dir.mkdir()
            (external_dir / "test.txt").write_text("test data")

            result = import_project_source(settings, external_dir, "sum_spectrum", mode="copy")
            assert result.mode == "copy"
            assert not result.destination.is_symlink()
            assert result.destination.exists()
            assert (result.destination / "test.txt").exists()


class TestSnapshotNoRecursion:
    """Test that snapshots don't include previous snapshots."""

    def test_repeated_snapshot_excludes_previous(self):
        """Test that creating a second snapshot doesn't include the first."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test_project",
                system="test_system",
            )
            ensure_project_structure(settings)

            # Create a data file
            data_file = project_root(settings) / "raw_data" / "test.txt"
            data_file.write_text("test data")

            # Create first snapshot
            first_snapshot = export_project_archive(
                settings,
                project_root(settings) / "versions" / "snapshot1.zip"
            )
            first_size = first_snapshot.stat().st_size

            # Create second snapshot
            second_snapshot = export_project_archive(
                settings,
                project_root(settings) / "versions" / "snapshot2.zip"
            )
            second_size = second_snapshot.stat().st_size

            # Second snapshot should not be significantly larger (shouldn't include first)
            # Allow some overhead but should be similar size
            assert second_size < first_size * 1.2  # Less than 20% larger

            # Verify second snapshot doesn't contain first snapshot
            with zipfile.ZipFile(second_snapshot, "r") as zf:
                filenames = zf.namelist()
                assert "snapshot1.zip" not in filenames
                # Ensure versions/*.zip files are excluded
                zip_files = [f for f in filenames if f.endswith(".zip")]
                assert len(zip_files) == 0
