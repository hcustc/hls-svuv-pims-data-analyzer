"""
Phase 3 Step 3: PIE Fitting Result Persistence Tests

Comprehensive tests for:
- Result persistence and restoration
- Validity derivation from fingerprints
- Array persistence and validation
- Project-level result management
- Fine-grained PICS fingerprinting
"""
import os
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np

from bl03u_masstool.core.pie_state import (
    PieStateManager,
    compute_project_fingerprint,
    compute_database_fingerprint,
    compute_per_mz_pics_fingerprint,
)


@pytest.fixture
def sample_curves() -> Dict[int, Dict]:
    """Create sample m/z curves."""
    return {
        46: {
            "energies": [10.0, 11.0, 12.0],
            "intensities": [1.0, 2.0, 1.5],
            "species": "NO",
        },
        28: {
            "energies": [10.0, 11.0, 12.0],
            "intensities": [3.0, 4.0, 3.5],
            "species": "CO",
        },
    }


@pytest.fixture
def sample_database() -> list[dict]:
    """Create sample PICS database."""
    return [
        {
            "id": 1,
            "species": "NO",
            "ionization_energy": 9.26,
            "cross_sections": [0.5, 0.6, 0.7],
        },
        {
            "id": 2,
            "species": "CO",
            "ionization_energy": 14.01,
            "cross_sections": [0.4, 0.5, 0.6],
        },
        {
            "id": 5,
            "species": "N2",
            "ionization_energy": 15.6,
            "cross_sections": [0.2, 0.3, 0.4],
        },
    ]


@pytest.fixture
def sample_calibration():
    """Create sample calibration object."""
    class Calibration:
        def __init__(self):
            self.a = 1.0
            self.b = 1.0
            self.c = 0.0

    return Calibration()


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestResultPersistence:
    """Test complete result persistence and restoration."""

    def test_save_and_restore_complete_result(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test saving and restoring a complete fitting result with metrics and arrays."""
        manager = PieStateManager(temp_project_dir)

        # Create fitting result with metrics and components
        fit_result = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "rmse": 0.05,
                    "mae": 0.03,
                    "species": [
                        {"id": 1, "species": "NO", "coefficient": 0.73, "contribution_percent": 73.0, "ie": 9.26},
                        {"id": 5, "species": "N2", "coefficient": 0.27, "contribution_percent": 27.0, "ie": 15.6},
                    ],
                    "fitted_curve": [1.05, 2.05, 1.55],
                    "component_curves": [[0.73, 0.32], [0.78, 0.78], [0.71, 0.84]],
                },
                "components": [
                    {"id": 1, "species": "NO", "coefficient": 0.73, "contribution_percent": 73.0, "ie": 9.26},
                    {"id": 5, "species": "N2", "coefficient": 0.27, "contribution_percent": 27.0, "ie": 15.6},
                ],
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46_v1",
                "global_config_hash": "hash_global_v1",
            }
        }

        # Save state
        success, error = manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, fit_result, "hash_global_v1"
        )
        assert success, f"Save failed: {error}"

        # Load state
        loaded = manager.load_state(sample_curves, sample_database, sample_calibration)
        assert loaded["success"]
        assert "46" in loaded["results"]

        result = loaded["results"]["46"]
        assert result["success"] is True
        assert result["metrics"]["r_squared"] == 0.95
        assert result["metrics"]["rmse"] == 0.05
        assert len(result["components"]) == 2
        assert result["components"][0]["species"] == "NO"
        assert "experimental" in result["arrays"]
        assert "total_fit" in result["arrays"]

        # Load arrays
        arrays = manager.load_mz_arrays(46)
        assert "energies" in arrays
        assert "experimental" in arrays
        assert "total_fit" in arrays
        assert "residual" in arrays
        assert "components" in arrays

    def test_save_and_restore_multiple_results(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test saving and restoring multiple m/z results."""
        manager = PieStateManager(temp_project_dir)

        fit_results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.05, 2.05, 1.55],
                    "component_curves": [[0.8], [0.85], [0.8]],
                },
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            },
            28: {
                "success": True,
                "model": {
                    "r_squared": 0.92,
                    "species": [{"id": 2, "species": "CO", "coefficient": 0.9, "contribution_percent": 90.0}],
                    "fitted_curve": [3.05, 4.05, 3.55],
                    "component_curves": [[0.9], [0.95], [0.9]],
                },
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_28",
                "global_config_hash": "hash_global",
            },
        }

        success, error = manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, fit_results, "hash_global"
        )
        assert success

        loaded = manager.load_state(sample_curves, sample_database, sample_calibration)
        assert "46" in loaded["results"]
        assert "28" in loaded["results"]
        assert loaded["results"]["46"]["success"] is True
        assert loaded["results"]["28"]["success"] is True

    def test_configs_and_results_in_same_transaction(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that configs and results are saved in atomic transaction."""
        manager = PieStateManager(temp_project_dir)

        configs = {46: {"mode": "fit", "candidates": [1, 2]}}
        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[0.8], [0.85], [0.8]],
                },
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        success, error = manager.save_state(
            sample_curves, sample_database, sample_calibration, configs, results, "hash_global"
        )
        assert success

        loaded = manager.load_state(sample_curves, sample_database, sample_calibration)
        assert "46" in loaded["configs"]
        assert "46" in loaded["results"]
        assert loaded["configs"]["46"]["mode"] == "fit"
        assert loaded["results"]["46"]["success"] is True

    def test_failed_fit_saved_with_error_info(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that failed fits are saved with error information."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": False,
                "error": "Fitting algorithm did not converge",
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        success, error = manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )
        assert success

        loaded = manager.load_state(sample_curves, sample_database, sample_calibration)
        assert loaded["results"]["46"]["success"] is False
        assert loaded["results"]["46"]["error"] == "Fitting algorithm did not converge"


class TestValidityDerivation:
    """Test fingerprint-based validity derivation."""

    def test_derived_status_completed_when_hashes_match(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that status is COMPLETED when all fingerprints match."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[0.8], [0.85], [0.8]],
                },
                "components": [{"species_id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )

        loaded = manager.load_state(sample_curves, sample_database, sample_calibration)
        assert loaded["results"]["46"]["_status"] == "COMPLETED"

    def test_derived_status_obsolete_when_experimental_data_changes(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that status is OBSOLETE when experimental data changes."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[0.8], [0.85], [0.8]],
                },
                "components": [{"species_id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )

        # Change experimental data
        modified_curves = sample_curves.copy()
        modified_curves[46] = modified_curves[46].copy()
        modified_curves[46]["intensities"] = [2.0, 3.0, 2.5]  # Changed

        loaded = manager.load_state(modified_curves, sample_database, sample_calibration)
        assert loaded["results"]["46"]["_status"] == "OBSOLETE"
        assert loaded["results"]["46"]["_obsolete_reason"] == "experimental_data_changed"

    def test_derived_status_obsolete_when_pics_data_changes(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that status is OBSOLETE when used PICS species data changes."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[0.8], [0.85], [0.8]],
                },
                "components": [{"species_id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )

        # Change PICS cross sections for species 1 (which is used)
        modified_db = [item.copy() for item in sample_database]
        modified_db[0] = modified_db[0].copy()
        modified_db[0]["cross_sections"] = [0.6, 0.7, 0.8]  # Changed

        loaded = manager.load_state(sample_curves, modified_db, sample_calibration)
        assert loaded["results"]["46"]["_status"] == "OBSOLETE"
        assert loaded["results"]["46"]["_obsolete_reason"] == "pics_data_changed"

    def test_unrelated_species_change_does_not_invalidate(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that changes to unrelated species don't invalidate per-m/z result."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[0.8], [0.85], [0.8]],
                },
                "components": [{"species_id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )

        # Change species 5 (N2) which is NOT used in the m/z 46 fit
        modified_db = [item.copy() for item in sample_database]
        modified_db[2] = modified_db[2].copy()  # Species 5
        modified_db[2]["cross_sections"] = [0.9, 1.0, 1.1]  # Changed

        loaded = manager.load_state(sample_curves, modified_db, sample_calibration)
        # Should still be COMPLETED because species 5 is not used
        assert loaded["results"]["46"]["_status"] == "COMPLETED"


class TestArrayPersistence:
    """Test array persistence and validation."""

    def test_arrays_persisted_and_loaded_correctly(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that all arrays are correctly persisted and loaded."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.05, 2.05, 1.55],
                    "component_curves": [[0.8, 0.25], [0.85, 0.2], [0.8, 0.25]],
                },
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )

        arrays = manager.load_mz_arrays(46)
        assert arrays["energies"].shape == (3,)
        assert arrays["experimental"].shape == (3,)
        assert arrays["total_fit"].shape == (3,)
        assert arrays["residual"].shape == (3,)
        assert arrays["components"].shape == (3, 2)

        # Verify values
        np.testing.assert_array_almost_equal(
            arrays["experimental"], np.array([1.0, 2.0, 1.5])
        )
        np.testing.assert_array_almost_equal(
            arrays["total_fit"], np.array([1.05, 2.05, 1.55])
        )

    def test_component_curves_shape_matches_species_count(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that component curves shape is (n_energy, n_species)."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [
                        {"id": 1, "species": "NO", "coefficient": 0.6, "contribution_percent": 60.0},
                        {"id": 5, "species": "N2", "coefficient": 0.4, "contribution_percent": 40.0},
                    ],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[0.6, 0.4], [0.65, 0.4], [0.6, 0.4]],
                },
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )

        arrays = manager.load_mz_arrays(46)
        # Should be (n_energy=3, n_species=2)
        assert arrays["components"].shape == (3, 2)

    def test_empty_arrays_handled_gracefully(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that missing optional arrays are handled gracefully."""
        manager = PieStateManager(temp_project_dir)

        # Result with no component curves
        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    # Missing component_curves
                },
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        success, error = manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )
        assert success

        arrays = manager.load_mz_arrays(46)
        # Should still have energies and experimental data
        assert "energies" in arrays
        assert "experimental" in arrays


class TestProjectManagement:
    """Test result management across project operations."""

    def test_dirty_flag_set_on_fit_success(self):
        """Test that dirty flag is set when fitting completes successfully."""
        # This test requires PIE dialog setup
        # Simplified check: dirty flag management is tested in dialog tests
        pass

    def test_multiple_saves_without_garbage_files(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that multiple saves don't leave garbage files."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[0.8], [0.85], [0.8]],
                },
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        # Save multiple times
        for i in range(3):
            success, _ = manager.save_state(
                sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
            )
            assert success

        # Check that temp and backup dirs are cleaned up
        assert not os.path.exists(manager.temp_dir)
        assert not os.path.exists(manager.backup_dir)

        # State dir should exist and contain only expected files
        assert os.path.exists(manager.state_dir)
        files = os.listdir(manager.state_dir)
        assert "manifest.json" in files
        assert "configs.json" in files
        assert "results.json" in files
        assert "arrays" in files

    def test_empty_results_can_be_saved_and_restored(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that empty result set is handled correctly."""
        manager = PieStateManager(temp_project_dir)

        # Save with empty results
        success, _ = manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, {}, "hash_global"
        )
        assert success

        # Load and verify
        loaded = manager.load_state(sample_curves, sample_database, sample_calibration)
        assert loaded["success"]
        assert len(loaded["results"]) == 0


class TestFinegrainedPicsFingerprint:
    """Test fine-grained per-m/z PICS fingerprinting."""

    def test_fine_grained_pics_fingerprint_per_species(self, sample_database):
        """Test that per-m/z fingerprints only cover used species."""
        used_species_ids = [1, 5]  # NO and N2
        mz = 46

        fp = compute_per_mz_pics_fingerprint(mz, used_species_ids, sample_database)

        assert fp["mz"] == 46
        assert fp["used_species_ids"] == [1, 5]
        assert fp["species_count"] == 2
        assert isinstance(fp["cs_combined_hash"], str)
        assert len(fp["cs_combined_hash"]) == 16  # SHA256 truncated

    def test_unrelated_species_change_does_not_change_fingerprint(self, sample_database):
        """Test that changing unrelated species doesn't change fingerprint."""
        used_species_ids = [1]  # Only NO

        fp1 = compute_per_mz_pics_fingerprint(46, used_species_ids, sample_database)

        # Change species 5 (not in used list)
        modified_db = [item.copy() for item in sample_database]
        modified_db[2]["cross_sections"] = [0.9, 1.0, 1.1]

        fp2 = compute_per_mz_pics_fingerprint(46, used_species_ids, modified_db)

        # Fingerprint should be unchanged
        assert fp1["cs_combined_hash"] == fp2["cs_combined_hash"]

    def test_used_species_change_changes_fingerprint(self, sample_database):
        """Test that changing used species changes fingerprint."""
        used_species_ids = [1]  # Only NO

        fp1 = compute_per_mz_pics_fingerprint(46, used_species_ids, sample_database)

        # Change species 1 (which IS in used list)
        modified_db = [item.copy() for item in sample_database]
        modified_db[0]["cross_sections"] = [0.9, 1.0, 1.1]  # Changed

        fp2 = compute_per_mz_pics_fingerprint(46, used_species_ids, modified_db)

        # Fingerprint should be different
        assert fp1["cs_combined_hash"] != fp2["cs_combined_hash"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_result_with_zero_contribution_species(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test handling of species with zero contribution."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [
                        {"id": 1, "species": "NO", "coefficient": 1.0, "contribution_percent": 100.0},
                        {"id": 5, "species": "N2", "coefficient": 0.0, "contribution_percent": 0.0},
                    ],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                },
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        success, _ = manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )
        assert success

        loaded = manager.load_state(sample_curves, sample_database, sample_calibration)
        assert loaded["results"]["46"]["components"][1]["contribution_percent"] == 0.0

    def test_very_small_differences_detected_in_fingerprints(
        self, temp_project_dir, sample_curves, sample_database, sample_calibration
    ):
        """Test that very small data changes are detected by fingerprints."""
        manager = PieStateManager(temp_project_dir)

        results = {
            46: {
                "success": True,
                "model": {
                    "r_squared": 0.95,
                    "species": [{"id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                    "fitted_curve": [1.0, 2.0, 1.5],
                    "component_curves": [[0.8], [0.85], [0.8]],
                },
                "components": [{"species_id": 1, "species": "NO", "coefficient": 0.8, "contribution_percent": 80.0}],
                "fit_timestamp": time.time(),
                "fit_config_hash": "hash_46",
                "global_config_hash": "hash_global",
            }
        }

        manager.save_state(
            sample_curves, sample_database, sample_calibration, {}, results, "hash_global"
        )

        # Make very small change to intensity (1e-6)
        modified_curves = sample_curves.copy()
        modified_curves[46] = modified_curves[46].copy()
        modified_curves[46]["intensities"] = [1.0 + 1e-6, 2.0, 1.5]

        loaded = manager.load_state(modified_curves, sample_database, sample_calibration)
        # Should detect the change
        assert loaded["results"]["46"]["_status"] == "OBSOLETE"
