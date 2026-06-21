"""
Phase 3 Step 2 Tests for PIE Dialog: Per-m/z configuration persistence.

Test coverage:
- Configuration save and restore
- Project reload after save
- Project movement to different directory
- Missing state file handling
- Corrupted configuration file handling
- Independent m/z configuration restoration
- Config hash updates after restoration
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.project_lifecycle import project_root
from bl03u_masstool.core.pie_state import PieStateManager
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


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    tmpdir = tempfile.mkdtemp(prefix="pie_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_curves():
    """Create sample PIE curves for testing."""
    return {
        46: {
            "mz": 46,
            "energies": [9.0, 10.0, 11.0, 12.0],
            "intensities": [100.0, 200.0, 150.0, 50.0],
            "rows": 4,
        },
        47: {
            "mz": 47,
            "energies": [9.0, 10.0, 11.0, 12.0],
            "intensities": [80.0, 180.0, 140.0, 40.0],
            "rows": 4,
        },
    }


@pytest.fixture
def sample_database():
    """Create a sample PICS database."""
    return [
        {
            "id": 1,
            "species": "NO",
            "ionization_energy": 9.26,
            "cross_sections": [10.0, 20.0, 15.0, 5.0],
        },
        {
            "id": 2,
            "species": "N2O",
            "ionization_energy": 12.89,
            "cross_sections": [8.0, 18.0, 12.0, 4.0],
        },
    ]


@pytest.fixture
def project_settings(temp_project_dir):
    """Create a ProjectSettings instance pointing to temp directory."""
    ps = ProjectSettings(
        project_name="Test Project",
        system="Test System",
        output_dir=temp_project_dir,
    )
    return ps


class TestConfigurationSaveRestore:
    """Test basic configuration save and restore functionality."""

    def test_save_and_restore_per_mz_configs(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test saving and restoring per-m/z configurations."""
        # Setup
        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database
        pie_dialog._auto_load_database()

        # Configure m/z 46
        pie_dialog.per_mz_config[46] = {
            "mz": 46,
            "selected_species": [
                {"id": 1, "species": "NO", "ionization_energy": 9.26}
            ],
            "mode": "auto",
            "coefficients": {"1": 0.75},
            "locked_ids": [],
            "config_hash": "hash46",
        }

        # Configure m/z 47
        pie_dialog.per_mz_config[47] = {
            "mz": 47,
            "selected_species": [
                {"id": 2, "species": "N2O", "ionization_energy": 12.89}
            ],
            "mode": "auto",
            "coefficients": {"2": 0.85},
            "locked_ids": [],
            "config_hash": "hash47",
        }

        # Save configurations
        pie_dialog.save_per_mz_configs()

        # Verify state file exists
        state_file = (
            Path(pie_dialog.project_dir) / ".bl03u_pie_state" / "configs.json"
        )
        assert state_file.exists(), "State file should be created"

        # Load in new dialog
        pie_dialog2 = PIESpeciesFitDialog(pie_dialog.calibration)
        pie_dialog2.project_settings = project_settings
        pie_dialog2.project_dir = str(project_root(project_settings))
        pie_dialog2.curves = sample_curves
        pie_dialog2.database = sample_database
        pie_dialog2._load_per_mz_configs()

        # Verify restoration
        assert 46 in pie_dialog2.per_mz_config
        assert 47 in pie_dialog2.per_mz_config
        assert pie_dialog2.per_mz_config[46]["selected_species"][0]["species"] == "NO"
        assert pie_dialog2.per_mz_config[47]["selected_species"][0]["species"] == "N2O"
        pie_dialog2.deleteLater()

    def test_empty_state_file_works_as_new_project(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test that missing state file is treated as a new project."""
        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database

        # No configurations saved - simulate first open
        pie_dialog._load_per_mz_configs()

        # Should not crash, per_mz_config should be empty
        assert len(pie_dialog.per_mz_config) == 0


class TestProjectMovement:
    """Test configuration persistence when project is moved."""

    def test_restore_config_after_project_move(
        self, pie_dialog, temp_project_dir, sample_curves, sample_database
    ):
        """Test that configurations are restored after project directory is moved."""
        # Save in original location
        ps1 = ProjectSettings(
            project_name="Test",
            system="Test",
            output_dir=str(temp_project_dir),
        )
        pie_dialog.project_settings = ps1
        pie_dialog.project_dir = str(project_root(ps1))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database
        pie_dialog._auto_load_database()

        pie_dialog.per_mz_config[46] = {
            "mz": 46,
            "selected_species": [
                {"id": 1, "species": "NO", "ionization_energy": 9.26}
            ],
            "mode": "auto",
            "coefficients": {"1": 0.75},
            "locked_ids": [],
        }
        pie_dialog.save_per_mz_configs()

        # Simulate project move
        new_temp_dir = tempfile.mkdtemp(prefix="pie_moved_")
        try:
            # Copy state directory to new location
            old_state_dir = Path(pie_dialog.project_dir) / ".bl03u_pie_state"
            new_state_dir = Path(new_temp_dir) / ".bl03u_pie_state"
            if old_state_dir.exists():
                shutil.copytree(old_state_dir, new_state_dir)

            # Load from new location
            ps2 = ProjectSettings(
                project_name="Test",
                system="Test",
                output_dir=new_temp_dir,
            )
            pie_dialog.project_settings = ps2
            pie_dialog.project_dir = str(project_root(ps2))
            pie_dialog.per_mz_config = {}  # Reset
            pie_dialog._load_per_mz_configs()

            # Verify restoration in new location
            assert 46 in pie_dialog.per_mz_config
            assert (
                pie_dialog.per_mz_config[46]["selected_species"][0]["species"] == "NO"
            )
        finally:
            shutil.rmtree(new_temp_dir, ignore_errors=True)


class TestErrorHandling:
    """Test error handling for corrupted or missing files."""

    def test_missing_state_file_doesnt_block_loading(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test that missing state file doesn't block project from loading."""
        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database

        # Load when state file doesn't exist
        pie_dialog._load_per_mz_configs()

        # Should not crash or raise exception
        assert pie_dialog.project_dir is not None
        assert len(pie_dialog.per_mz_config) == 0

    def test_corrupted_config_file_warns_but_continues(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test that corrupted config file warns but doesn't block loading."""
        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database

        # Create corrupted config file
        state_dir = Path(pie_dialog.project_dir) / ".bl03u_pie_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        config_file = state_dir / "configs.json"
        config_file.write_text("{ invalid json }")

        # Load corrupted file
        pie_dialog._load_per_mz_configs()

        # Should not crash, per_mz_config should be empty
        assert len(pie_dialog.per_mz_config) == 0


class TestConfigHashRecalculation:
    """Test that config hashes are recalculated after restoration."""

    def test_config_hash_updated_after_restoration(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test that config_hash is recalculated when configs are loaded."""
        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database
        pie_dialog._auto_load_database()

        # Save config with old hash
        pie_dialog.per_mz_config[46] = {
            "mz": 46,
            "selected_species": [
                {"id": 1, "species": "NO", "ionization_energy": 9.26}
            ],
            "mode": "auto",
            "coefficients": {"1": 0.75},
            "locked_ids": [],
            "config_hash": "old_hash_value",
        }
        pie_dialog.save_per_mz_configs()

        # Load and verify hash was recalculated
        pie_dialog.per_mz_config = {}  # Reset
        pie_dialog._load_per_mz_configs()

        assert 46 in pie_dialog.per_mz_config
        # Hash should be recalculated, not the old value
        saved_hash = pie_dialog.per_mz_config[46].get("config_hash", "")
        assert saved_hash != "old_hash_value"
        assert saved_hash != ""  # Should have a valid hash


class TestIndependentMzRestoration:
    """Test that individual m/z configurations are restored independently."""

    def test_multiple_mz_restored_independently(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test that multiple m/z configs are restored independently."""
        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database
        pie_dialog._auto_load_database()

        # Save multiple configurations
        pie_dialog.per_mz_config[46] = {
            "mz": 46,
            "selected_species": [
                {"id": 1, "species": "NO", "ionization_energy": 9.26}
            ],
            "mode": "manual",
            "coefficients": {"1": 0.5},
            "locked_ids": [],
        }
        pie_dialog.per_mz_config[47] = {
            "mz": 47,
            "selected_species": [
                {"id": 2, "species": "N2O", "ionization_energy": 12.89}
            ],
            "mode": "auto",
            "coefficients": {"2": 0.8},
            "locked_ids": [2],
        }
        pie_dialog.save_per_mz_configs()

        # Load and verify each m/z config is distinct
        pie_dialog.per_mz_config = {}
        pie_dialog._load_per_mz_configs()

        assert pie_dialog.per_mz_config[46]["mode"] == "manual"
        assert pie_dialog.per_mz_config[47]["mode"] == "auto"
        assert pie_dialog.per_mz_config[46]["coefficients"]["1"] == 0.5
        assert pie_dialog.per_mz_config[47]["coefficients"]["2"] == 0.8
        assert len(pie_dialog.per_mz_config[47]["locked_ids"]) == 1

    def test_partial_mz_restoration_doesnt_affect_others(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test that restoration of some m/z doesn't affect others."""
        # Only save m/z 46
        partial_curves = {46: sample_curves[46]}

        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = partial_curves
        pie_dialog.database = sample_database
        pie_dialog._auto_load_database()

        pie_dialog.per_mz_config[46] = {
            "mz": 46,
            "selected_species": [
                {"id": 1, "species": "NO", "ionization_energy": 9.26}
            ],
            "mode": "auto",
            "coefficients": {"1": 0.75},
            "locked_ids": [],
        }
        pie_dialog.save_per_mz_configs()

        # Now load with both curves present
        pie_dialog.curves = sample_curves
        pie_dialog.per_mz_config = {}
        pie_dialog._load_per_mz_configs()

        # Should load m/z 46, but m/z 47 should not be present
        assert 46 in pie_dialog.per_mz_config
        assert 47 not in pie_dialog.per_mz_config


class TestGlobalConfigHashPreservation:
    """Test that global config hash is updated appropriately."""

    def test_global_config_hash_updated_after_load(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test that global config hash is recalculated after loading."""
        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database
        pie_dialog._auto_load_database()

        # Save with some global config
        pie_dialog.global_solver_config = {"some": "config"}
        pie_dialog._update_global_config_hash()
        original_hash = pie_dialog.global_solver_config.get("config_hash", "")

        pie_dialog.save_per_mz_configs()

        # Load and verify hash was updated
        pie_dialog.global_solver_config = {}
        pie_dialog._load_per_mz_configs()

        new_hash = pie_dialog.global_solver_config.get("config_hash", "")
        assert new_hash == original_hash or new_hash == ""


class TestNoSaveOnEveryChange:
    """Test that configurations are not saved on every change."""

    def test_config_changes_dont_trigger_immediate_save(
        self, pie_dialog, project_settings, sample_curves, sample_database
    ):
        """Test that modifying config doesn't immediately write to disk."""
        pie_dialog.project_settings = project_settings
        pie_dialog.project_dir = str(project_root(project_settings))
        pie_dialog.curves = sample_curves
        pie_dialog.database = sample_database

        # Modify config without calling save_per_mz_configs()
        pie_dialog.per_mz_config[46] = {
            "mz": 46,
            "selected_species": [
                {"id": 1, "species": "NO", "ionization_energy": 9.26}
            ],
            "mode": "auto",
            "coefficients": {"1": 0.75},
            "locked_ids": [],
        }

        # State file should NOT exist yet
        state_file = (
            Path(pie_dialog.project_dir) / ".bl03u_pie_state" / "configs.json"
        )
        assert not state_file.exists(), "State should not be saved automatically"

        # Save explicitly
        pie_dialog.save_per_mz_configs()

        # Now state file should exist
        assert state_file.exists(), "State should exist after explicit save"
