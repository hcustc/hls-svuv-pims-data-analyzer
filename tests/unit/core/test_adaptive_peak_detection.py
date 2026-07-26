from dataclasses import replace

import numpy as np
import pytest

from bl03u_masstool.core.adaptive_peak_detection import (
    detect_peaks_adaptive,
    estimate_signal_scale,
    normalize_signal_scale,
)
from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.config import PeakDetectionConfig
from bl03u_masstool.core.peak_detection import detect_peaks_by_algorithm


def _test_peak_config() -> PeakDetectionConfig:
    return replace(
        PeakDetectionConfig(),
        detection_min_idx=0,
        threshold_end=0.5,
        min_intensity=2.0,
        nearby_peak_window=8,
        duplicate_window=5,
        weak_tail_early_window=15,
        weak_tail_late_window=10,
        gaussian_window_max=10,
        smoothing_window=3,
        baseline_window=21,
        min_peak_width=1,
        max_peak_width=15,
    )


def test_signal_scale_is_homogeneous():
    y = np.array([0.0, 1.0, 2.0, 4.0, 8.0])

    assert estimate_signal_scale(y * 17.0) == pytest.approx(estimate_signal_scale(y) * 17.0)
    normalized, scale = normalize_signal_scale(y * 17.0)
    expected, expected_scale = normalize_signal_scale(y)

    assert scale == pytest.approx(expected_scale * 17.0)
    assert normalized == pytest.approx(expected)


def test_adaptive_detection_is_invariant_to_intensity_scale():
    y = np.zeros(120, dtype=float)
    y[38:45] = [1.0, 4.0, 9.0, 20.0, 9.0, 4.0, 1.0]
    y[78:85] = [1.0, 3.0, 7.0, 14.0, 7.0, 3.0, 1.0]
    calibration = Calibration(a=0, b=1, c=0)

    original = detect_peaks_adaptive(
        y,
        calibration=calibration,
        peak_config=_test_peak_config(),
        time_offset=1.0,
    )
    scaled = detect_peaks_adaptive(
        y * 0.125,
        calibration=calibration,
        peak_config=_test_peak_config(),
        time_offset=1.0,
    )

    assert [peak.time for peak in original] == pytest.approx([peak.time for peak in scaled])
    assert [peak.intensity for peak in original] == pytest.approx(
        [peak.intensity * 8.0 for peak in scaled]
    )


def test_adaptive_algorithm_is_available_through_dispatcher():
    y = np.zeros(120, dtype=float)
    y[58:65] = [1.0, 4.0, 9.0, 20.0, 9.0, 4.0, 1.0]
    config = _test_peak_config()

    peaks = detect_peaks_by_algorithm(
        y,
        algorithm="adaptive",
        calibration=Calibration(a=0, b=1, c=0),
        detection_min_idx=config.detection_min_idx,
        threshold_end=config.threshold_end,
        min_intensity=config.min_intensity,
        nearby_peak_window=config.nearby_peak_window,
        duplicate_window=config.duplicate_window,
        weak_tail_early_window=config.weak_tail_early_window,
        weak_tail_late_window=config.weak_tail_late_window,
        gaussian_window_max=config.gaussian_window_max,
        smoothing_window=config.smoothing_window,
        baseline_window=config.baseline_window,
        min_peak_width=config.min_peak_width,
        max_peak_width=config.max_peak_width,
        time_offset=1.0,
    )

    assert len(peaks) == 1
    assert peaks[0].time == pytest.approx(62.0, abs=0.5)
