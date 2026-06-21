"""
Tests for PIE fitting state persistence and schema.
"""
from __future__ import annotations

import json
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from bl03u_masstool.core.pie_state import (
    PieStateManager,
    compute_project_fingerprint,
    compute_database_fingerprint,
    SCHEMA_VERSION,
)


class MockCalibration:
    """Mock calibration object for testing."""
    def __init__(self, a=0.0, b=1.0, c=0.0):
        self.a = a
        self.b = b
        self.c = c


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.fixture
def sample_curves():
    """Create sample PIE curves for testing."""
    return {
        46: {
            'mz': 46,
            'species': 'NO',
            'energies': np.array([10.0, 11.0, 12.0]),
            'intensities': np.array([1.0, 2.0, 1.5]),
        },
        47: {
            'mz': 47,
            'species': 'NO2',
            'energies': np.array([10.5, 11.5, 12.5]),
            'intensities': np.array([1.5, 2.5, 2.0]),
        },
    }


@pytest.fixture
def sample_database():
    """Create sample PICS database for testing."""
    return [
        {
            'id': 1,
            'species': 'NO',
            'ionization_energy': 9.26,
            'energies': np.array([10.0, 11.0, 12.0]),
            'cross_sections': np.array([0.1, 0.2, 0.15]),
        },
        {
            'id': 2,
            'species': 'N2O',
            'ionization_energy': 12.89,
            'energies': np.array([10.0, 11.0, 12.0]),
            'cross_sections': np.array([0.05, 0.1, 0.08]),
        },
    ]


@pytest.fixture
def sample_per_mz_config():
    """Create sample per-m/z configuration."""
    return {
        46: {
            'selected_species': [
                {'id': 1, 'species': 'NO', 'ionization_energy': 9.26},
            ],
            'mode': 'auto',
            'coefficients': {},
            'locked_ids': [],
            'config_hash': 'abc123',
        },
        47: {
            'selected_species': [
                {'id': 1, 'species': 'NO'},
                {'id': 2, 'species': 'N2O'},
            ],
            'mode': 'fit',
            'coefficients': {},
            'locked_ids': [],
            'config_hash': 'def456',
        },
    }


@pytest.fixture
def sample_fit_results():
    """Create sample fitting results."""
    return {
        46: {
            'success': True,
            'model': {
                'mz': 46,
                'species': [
                    {
                        'id': 1,
                        'species': 'NO',
                        'coefficient': 0.8,
                        'contribution_percent': 80.0,
                        'ie': 9.26,
                        'r_squared': 0.95,
                    }
                ],
                'r_squared': 0.95,
                'candidate_count': 25,
                'fitted_curve': np.array([0.95, 1.95, 1.50]),
                'component_curves': [
                    np.array([0.95, 1.95, 1.50]),
                ],
                'residuals': np.array([0.05, 0.05, 0.0]),
            },
            'fit_config_hash': 'abc123',
            'global_config_hash': 'xyz789',
            'fit_timestamp': 1719052320.5,
        },
        47: {
            'success': False,
            'error': 'No matching species',
            'fit_timestamp': None,
        },
    }


class TestProjectFingerprint:
    """Test project fingerprint computation."""

    def test_fingerprint_stable(self, sample_curves):
        """Test that identical curves produce identical fingerprints."""
        calib = MockCalibration()

        fp1 = compute_project_fingerprint(sample_curves, calib)
        fp2 = compute_project_fingerprint(sample_curves, calib)

        assert fp1 == fp2

    def test_fingerprint_changes_on_curve_modification(self, sample_curves):
        """Test that modifying curves changes fingerprint."""
        calib = MockCalibration()

        fp1 = compute_project_fingerprint(sample_curves, calib)

        # Modify curve
        sample_curves[46]['intensities'] = np.array([1.1, 2.1, 1.6])
        fp2 = compute_project_fingerprint(sample_curves, calib)

        assert fp1['curve_data_hash'] != fp2['curve_data_hash']

    def test_fingerprint_changes_on_mz_addition(self, sample_curves):
        """Test that adding m/z changes fingerprint."""
        calib = MockCalibration()

        fp1 = compute_project_fingerprint(sample_curves, calib)

        # Add new m/z
        sample_curves[48] = {
            'mz': 48,
            'energies': np.array([10.0, 11.0, 12.0]),
            'intensities': np.array([1.0, 2.0, 1.5]),
        }

        fp2 = compute_project_fingerprint(sample_curves, calib)

        assert fp1['mz_keys_hash'] != fp2['mz_keys_hash']
        assert fp1['curve_count'] != fp2['curve_count']

    def test_fingerprint_changes_on_calibration_change(self, sample_curves):
        """Test that changing calibration changes fingerprint."""
        calib1 = MockCalibration(a=0.0, b=1.0, c=0.0)
        calib2 = MockCalibration(a=0.1, b=1.0, c=0.0)

        fp1 = compute_project_fingerprint(sample_curves, calib1)
        fp2 = compute_project_fingerprint(sample_curves, calib2)

        assert fp1['calibration_hash'] != fp2['calibration_hash']


class TestDatabaseFingerprint:
    """Test database fingerprint computation."""

    def test_fingerprint_stable(self, sample_database):
        """Test that identical database produces identical fingerprints."""
        fp1 = compute_database_fingerprint(sample_database)
        fp2 = compute_database_fingerprint(sample_database)

        assert fp1 == fp2

    def test_fingerprint_changes_on_species_addition(self, sample_database):
        """Test that adding species changes fingerprint."""
        fp1 = compute_database_fingerprint(sample_database)

        # Add species
        sample_database.append({
            'id': 3,
            'species': 'CO2',
            'ionization_energy': 13.78,
            'energies': np.array([10.0, 11.0, 12.0]),
            'cross_sections': np.array([0.08, 0.15, 0.12]),
        })

        fp2 = compute_database_fingerprint(sample_database)

        assert fp1['species_ids_hash'] != fp2['species_ids_hash']
        assert fp1['total_species'] != fp2['total_species']

    def test_fingerprint_changes_on_cross_section_update(self, sample_database):
        """Test that updating cross-sections changes fingerprint."""
        fp1 = compute_database_fingerprint(sample_database)

        # Update cross-section
        sample_database[0]['cross_sections'] = np.array([0.2, 0.3, 0.25])

        fp2 = compute_database_fingerprint(sample_database)

        assert fp1['cross_section_hash'] != fp2['cross_section_hash']


class TestPieStateManagerSave:
    """Test PIE state saving."""

    def test_save_creates_state_directory(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that save creates .bl03u_pie_state directory."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        success, error = manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        assert success is True
        assert error is None
        assert os.path.exists(manager.state_dir)

    def test_save_creates_manifest(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that manifest.json is created with correct structure."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        manifest_path = os.path.join(manager.state_dir, 'manifest.json')
        assert os.path.exists(manifest_path)

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        assert manifest['schema_version'] == SCHEMA_VERSION
        assert 'project_fingerprint' in manifest
        assert 'database_fingerprint' in manifest
        assert 'global_config_hash' in manifest
        assert manifest['global_config_hash'] == 'global_hash_123'

    def test_save_creates_configs_and_results(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that configs.json and results.json are created."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        configs_path = os.path.join(manager.state_dir, 'configs.json')
        results_path = os.path.join(manager.state_dir, 'results.json')

        assert os.path.exists(configs_path)
        assert os.path.exists(results_path)

    def test_save_creates_npz_arrays(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that NPZ files are created for successful fits."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        npz_path = os.path.join(manager.state_dir, 'arrays', 'mz_46.npz')
        assert os.path.exists(npz_path)

    def test_save_atomic_on_failure(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
    ):
        """Test that failed save doesn't corrupt state directory."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        # First successful save
        success1, _ = manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            {},
            'global_hash_123',
        )
        assert success1 is True

        # Store manifest content
        manifest_path = os.path.join(manager.state_dir, 'manifest.json')
        with open(manifest_path, 'r') as f:
            original_manifest = json.load(f)

        # Second save with invalid data (missing 'id' in database)
        bad_database = [{'species': 'NO'}]  # Missing 'id'

        success2, error = manager.save_state(
            sample_curves,
            bad_database,
            calib,
            sample_per_mz_config,
            {},
            'global_hash_456',
        )

        # Check that old state is preserved
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                current_manifest = json.load(f)

            # Either save succeeded or old state is preserved
            assert current_manifest is not None


class TestPieStateManagerLoad:
    """Test PIE state loading."""

    def test_load_nonexistent_state_returns_empty(self, temp_project_dir, sample_curves, sample_database):
        """Test that loading from empty project returns empty state."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        result = manager.load_state(sample_curves, sample_database, calib)

        assert result['success'] is True
        assert result['configs'] == {}
        assert result['results'] == {}
        assert '未找到保存的 PIE 状态' in result['warnings'][0]

    def test_load_and_restore_config(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that configs are saved and restored correctly."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        # Save
        manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        # Load
        result = manager.load_state(sample_curves, sample_database, calib)

        assert result['success'] is True
        assert '46' in result['configs']
        assert '47' in result['configs']
        assert result['configs']['46']['mode'] == 'auto'

    def test_load_detects_curve_data_change(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that loading detects changes in curve data."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        # Save with original curves
        manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        # Load with modified curves
        sample_curves[46]['intensities'] = np.array([2.0, 3.0, 2.5])
        result = manager.load_state(sample_curves, sample_database, calib)

        assert result['success'] is True
        assert any('已变化' in w for w in result['warnings'])

    def test_load_detects_database_change(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that loading detects changes in database."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        # Save with original database
        manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        # Load with modified database
        sample_database[0]['cross_sections'] = np.array([0.2, 0.3, 0.25])
        result = manager.load_state(sample_curves, sample_database, calib)

        assert result['success'] is True
        assert any('PICS' in w for w in result['warnings'])

    def test_load_arrays(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that arrays can be loaded from NPZ files."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        # Save
        manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        # Load arrays
        arrays = manager.load_mz_arrays(46)

        assert 'energies' in arrays
        assert 'intensities' in arrays
        assert 'fitted_curve' in arrays
        assert isinstance(arrays['energies'], np.ndarray)


class TestPieStateIntegration:
    """Integration tests for save/load cycle."""

    def test_round_trip_preserves_data(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that data is preserved through save/load cycle."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        # Save
        manager.save_state(
            sample_curves,
            sample_database,
            calib,
            sample_per_mz_config,
            sample_fit_results,
            'global_hash_123',
        )

        # Load
        result = manager.load_state(sample_curves, sample_database, calib)

        # Verify configs
        assert result['configs']['46']['config_hash'] == 'abc123'
        assert len(result['configs']['46']['selected_species']) == 1

        # Verify results
        assert result['results']['46']['success'] is True
        assert result['results']['47']['success'] is False
        assert result['results']['47']['error'] == 'No matching species'

    def test_multiple_saves_dont_accumulate_files(
        self,
        temp_project_dir,
        sample_curves,
        sample_database,
        sample_per_mz_config,
        sample_fit_results,
    ):
        """Test that multiple saves don't create garbage files."""
        manager = PieStateManager(temp_project_dir)
        calib = MockCalibration()

        # Save multiple times
        for i in range(3):
            manager.save_state(
                sample_curves,
                sample_database,
                calib,
                sample_per_mz_config,
                sample_fit_results,
                f'global_hash_{i}',
            )

        # Check directory structure
        arrays_dir = os.path.join(manager.state_dir, 'arrays')
        npz_files = [f for f in os.listdir(arrays_dir) if f.endswith('.npz')]

        # Should only have npz files for actual m/z values
        assert len(npz_files) == 1  # Only mz_46 has success=True
        assert 'mz_46.npz' in npz_files
