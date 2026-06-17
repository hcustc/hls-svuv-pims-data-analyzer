"""Regression tests for ensemble algorithm integration in CLI.

Tests the complete chain:
- _peak_kwargs() extracts config → detect_peaks_by_algorithm()
- analyze_pie_folder() passes algorithm parameter → detect_peaks_by_algorithm()
- analyze_temperature_folder() passes algorithm parameter → detect_peaks_by_algorithm()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.config import PeakDetectionConfig, load_peak_detection_config
from bl03u_masstool.core.peak_detection import Peak, detect_peaks_by_algorithm
from bl03u_masstool.core.pie_analysis import analyze_pie_folder
from bl03u_masstool.core.temperature_scan import analyze_temperature_folder
from bl03u_masstool.core.peak_benchmark import build_default_synthetic_benchmark
from bl03u_masstool.core.ensemble_optimization import extract_ensemble_parameters, optimize_ensemble_parameters
from bl03u_masstool.scripts.bl03u_cli import _peak_kwargs


class TestPeakKwargsExtraction:
    """Test that _peak_kwargs() includes algorithm and ensemble parameters."""

    def test_peak_kwargs_includes_algorithm(self, tmp_path):
        """_peak_kwargs() should include the algorithm parameter."""
        kwargs = _peak_kwargs()
        assert "algorithm" in kwargs
        assert isinstance(kwargs["algorithm"], str)

    def test_peak_kwargs_includes_ensemble_parameters(self, tmp_path):
        """_peak_kwargs() should include all ensemble-specific parameters."""
        kwargs = _peak_kwargs()
        assert "vote_threshold" in kwargs
        assert "min_intensity_for_single_vote" in kwargs
        assert "mz_tolerance" in kwargs
        assert isinstance(kwargs["vote_threshold"], float)
        assert isinstance(kwargs["min_intensity_for_single_vote"], float)
        assert isinstance(kwargs["mz_tolerance"], float)

    def test_peak_kwargs_includes_all_algorithm_parameters(self, tmp_path):
        """_peak_kwargs() should include parameters for all algorithms."""
        kwargs = _peak_kwargs()
        # Legacy parameters
        assert "threshold_end" in kwargs
        assert "min_intensity" in kwargs
        assert "nearby_peak_window" in kwargs
        # Prominence parameters
        assert "prominence_ratio" in kwargs
        assert "baseline_window" in kwargs
        # All CWT parameters should also be included for dispatcher
        expected_keys = {
            "algorithm",
            "threshold_end",
            "min_intensity",
            "detection_min_idx",
            "nearby_peak_window",
            "duplicate_window",
            "weak_tail_early_window",
            "weak_tail_late_window",
            "weak_tail_ratio",
            "gaussian_window_max",
            "gaussian_boundary_scale",
            "boundary_padding",
            "prominence_ratio",
            "smoothing_window",
            "smoothing_poly_order",
            "baseline_window",
            "baseline_percentile",
            "min_peak_width",
            "max_peak_width",
            "vote_threshold",
            "min_intensity_for_single_vote",
            "mz_tolerance",
        }
        assert set(kwargs.keys()) == expected_keys


class TestDetectPeaksByAlgorithmDispatcher:
    """Test that dispatcher routes to correct algorithm including ensemble."""

    def test_dispatcher_accepts_ensemble_algorithm(self):
        """dispatch should accept 'ensemble' as a valid algorithm."""
        y_data = np.array([0.0] * 100)
        y_data[50] = 10.0
        calibration = Calibration()

        # Should not raise ValueError
        peaks = detect_peaks_by_algorithm(
            y_data,
            algorithm="ensemble",
            calibration=calibration,
            vote_threshold=0.667,
            min_intensity_for_single_vote=5.0,
            mz_tolerance=0.2,
        )
        assert isinstance(peaks, list)

    def test_dispatcher_passes_ensemble_parameters(self):
        """dispatch should pass ensemble parameters to detect_peaks_ensemble."""
        y_data = np.array([0.0] * 100)
        y_data[50] = 10.0
        calibration = Calibration()

        # Test with non-default ensemble parameters
        peaks = detect_peaks_by_algorithm(
            y_data,
            algorithm="ensemble",
            calibration=calibration,
            vote_threshold=0.9,  # Non-default
            min_intensity_for_single_vote=10.0,  # Non-default
            mz_tolerance=0.5,  # Non-default
        )
        assert isinstance(peaks, list)

    def test_dispatcher_legacy_algorithm_still_works(self):
        """dispatch should still work with legacy algorithm."""
        y_data = np.array([0.0] * 100)
        y_data[50] = 10.0
        calibration = Calibration()

        peaks = detect_peaks_by_algorithm(
            y_data,
            algorithm="legacy",
            calibration=calibration,
        )
        assert isinstance(peaks, list)

    def test_dispatcher_prominence_algorithm_still_works(self):
        """dispatch should still work with prominence algorithm."""
        y_data = np.array([0.0] * 100)
        y_data[50] = 10.0
        calibration = Calibration()

        peaks = detect_peaks_by_algorithm(
            y_data,
            algorithm="prominence",
            calibration=calibration,
        )
        assert isinstance(peaks, list)


class TestAnalysisFunctionsAlgorithmIntegration:
    """Test that analyze_pie_folder and analyze_temperature_folder use dispatcher."""

    def _create_test_spectrum(self, path, y_values, metadata_lines=None):
        """Helper to create a test spectrum file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if metadata_lines is None:
            metadata_lines = [
                "Energy:11.0 eV",
                "IO:10 nA",
                "Beam Current:1mA",
                "Undulator Offset:0mm",
                "Time:1 s",
                "Burner Position:0 mm",
                "Temperature:300 C",
                "DIFF PRESSURE:1Pa",
                "ION PRESSURE:1Pa",
                "TOF PRESSURE:1Pa",
            ]
        content = "\n".join(metadata_lines + [str(float(v)) for v in y_values])
        path.write_text(content, encoding="utf-8")

    def test_pie_folder_accepts_algorithm_parameter(self, tmp_path):
        """analyze_pie_folder should accept algorithm parameter."""
        y = [0.0] * 100
        y[50] = 10.0
        self._create_test_spectrum(tmp_path / "11.0eV" / "spectrum.txt", y)
        self._create_test_spectrum(tmp_path / "12.0eV" / "spectrum.txt", y)

        # Should accept algorithm parameter without error
        df = analyze_pie_folder(
            tmp_path,
            algorithm="legacy",
            detection_min_idx=0,
            nearby_peak_window=5,
        )
        assert df is not None

    def test_pie_folder_accepts_ensemble_parameters(self, tmp_path):
        """analyze_pie_folder should accept ensemble parameters."""
        y = [0.0] * 100
        y[50] = 10.0
        self._create_test_spectrum(tmp_path / "11.0eV" / "spectrum.txt", y)
        self._create_test_spectrum(tmp_path / "12.0eV" / "spectrum.txt", y)

        # Should accept ensemble parameters
        df = analyze_pie_folder(
            tmp_path,
            algorithm="ensemble",
            detection_min_idx=0,
            nearby_peak_window=5,
            vote_threshold=0.9,
            min_intensity_for_single_vote=10.0,
            mz_tolerance=0.5,
        )
        assert df is not None

    def test_temperature_folder_accepts_algorithm_parameter(self, tmp_path):
        """analyze_temperature_folder should accept algorithm parameter."""
        y = [0.0] * 100
        y[50] = 10.0
        self._create_test_spectrum(
            tmp_path / "400.txt",
            y,
            metadata_lines=[
                "Energy:12.0 eV",
                "IO:10 nA",
                "Beam Current:1mA",
                "Undulator Offset:0mm",
                "Time:1 s",
                "Burner Position:0 mm",
                "Temperature:400 C",
                "DIFF PRESSURE:1Pa",
                "ION PRESSURE:1Pa",
                "TOF PRESSURE:1Pa",
            ],
        )
        self._create_test_spectrum(
            tmp_path / "500.txt",
            y,
            metadata_lines=[
                "Energy:12.0 eV",
                "IO:10 nA",
                "Beam Current:1mA",
                "Undulator Offset:0mm",
                "Time:1 s",
                "Burner Position:0 mm",
                "Temperature:500 C",
                "DIFF PRESSURE:1Pa",
                "ION PRESSURE:1Pa",
                "TOF PRESSURE:1Pa",
            ],
        )

        # Should accept algorithm parameter
        df = analyze_temperature_folder(
            tmp_path,
            algorithm="legacy",
            detection_min_idx=0,
            nearby_peak_window=5,
        )
        assert df is not None

    def test_temperature_folder_accepts_ensemble_parameters(self, tmp_path):
        """analyze_temperature_folder should accept ensemble parameters."""
        y = [0.0] * 100
        y[50] = 10.0
        self._create_test_spectrum(
            tmp_path / "400.txt",
            y,
            metadata_lines=[
                "Energy:12.0 eV",
                "IO:10 nA",
                "Beam Current:1mA",
                "Undulator Offset:0mm",
                "Time:1 s",
                "Burner Position:0 mm",
                "Temperature:400 C",
                "DIFF PRESSURE:1Pa",
                "ION PRESSURE:1Pa",
                "TOF PRESSURE:1Pa",
            ],
        )
        self._create_test_spectrum(
            tmp_path / "500.txt",
            y,
            metadata_lines=[
                "Energy:12.0 eV",
                "IO:10 nA",
                "Beam Current:1mA",
                "Undulator Offset:0mm",
                "Time:1 s",
                "Burner Position:0 mm",
                "Temperature:500 C",
                "DIFF PRESSURE:1Pa",
                "ION PRESSURE:1Pa",
                "TOF PRESSURE:1Pa",
            ],
        )

        # Should accept ensemble parameters
        df = analyze_temperature_folder(
            tmp_path,
            algorithm="ensemble",
            detection_min_idx=0,
            nearby_peak_window=5,
            vote_threshold=0.9,
            min_intensity_for_single_vote=10.0,
            mz_tolerance=0.5,
        )
        assert df is not None


class TestConfigurationPropagation:
    """Test that configuration changes propagate through the chain."""

    def test_config_algorithm_affects_peak_kwargs(self):
        """Loading config should propagate algorithm choice to _peak_kwargs."""
        config = load_peak_detection_config()
        kwargs = _peak_kwargs()

        assert kwargs["algorithm"] == config.algorithm

    def test_config_ensemble_parameters_affect_peak_kwargs(self):
        """Loading config should propagate ensemble parameters to _peak_kwargs."""
        config = load_peak_detection_config()
        kwargs = _peak_kwargs()

        assert kwargs["vote_threshold"] == config.vote_threshold
        assert kwargs["min_intensity_for_single_vote"] == config.min_intensity_for_single_vote
        assert kwargs["mz_tolerance"] == config.mz_tolerance


def test_ensemble_optimization_smoke():
    background = np.zeros(300, dtype=float)
    cases = build_default_synthetic_benchmark(background, calibration=Calibration(a=0, b=1, c=0))

    result = optimize_ensemble_parameters(
        cases=cases,
        calibration=Calibration(a=0, b=1, c=0),
        n_trials=2,
        initial_random=1,
        random_state=1,
        quick_mode=True,
    )
    params = extract_ensemble_parameters(result)

    assert len(result.trials) == 2
    assert "vote_threshold" in params
    assert "mz_tolerance" in params
