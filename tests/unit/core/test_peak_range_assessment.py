from __future__ import annotations

import numpy as np

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.peak_range_assessment import (
    PeakRangeFitStatus,
    assess_peak_ranges,
)
from bl03u_masstool.core.peak_ranges import PeakRange, load_peak_ranges


LINEAR_CALIBRATION = Calibration(a=0.0, b=1.0, c=0.0)


def test_canonical_peak_index_column_round_trips(tmp_path):
    path = tmp_path / "peaks.csv"
    path.write_text(
        "label,peak_index,mz,left_bound,right_bound\nP,94,100,90,110\n",
        encoding="utf-8",
    )

    loaded = load_peak_ranges(path, calibration=LINEAR_CALIBRATION)

    assert loaded[0].peak_index == 94


def _gaussian_spectrum(center: float = 100.0):
    x = np.arange(50.0, 151.0)
    y = 3.0 + 100.0 * np.exp(-0.5 * ((x - center) / 3.0) ** 2)
    return x, y


def test_assessment_accepts_centered_peak_with_baseline_boundaries():
    x, y = _gaussian_spectrum()
    peak_range = PeakRange(
        mz=100.0,
        peak_index=100,
        left_bound=90,
        right_bound=110,
        label="P",
    )

    result = assess_peak_ranges(
        [peak_range],
        x,
        y,
        calibration=LINEAR_CALIBRATION,
    )[0]

    assert result.status is PeakRangeFitStatus.FIT
    assert result.actual_peak_x == 100.0
    assert result.actual_mz == 100.0
    assert result.center_offset_fraction == 0.0
    assert result.edge_ratio is not None and result.edge_ratio < 0.1


def test_assessment_marks_shifted_declared_center_for_review():
    x, y = _gaussian_spectrum()
    peak_range = PeakRange(
        mz=100.0,
        peak_index=94,
        left_bound=90,
        right_bound=110,
        label="shifted",
    )

    result = assess_peak_ranges(
        [peak_range],
        x,
        y,
        calibration=LINEAR_CALIBRATION,
    )[0]

    assert result.status is PeakRangeFitStatus.REVIEW
    assert result.center_offset_fraction == 0.3
    assert any("偏离文件峰位" in reason for reason in result.reasons)


def test_assessment_rejects_flat_window_and_reports_out_of_range():
    x = np.arange(50.0, 151.0)
    y = np.ones_like(x)
    flat = PeakRange(mz=100.0, peak_index=100, left_bound=95, right_bound=105)
    outside = PeakRange(mz=200.0, peak_index=200, left_bound=195, right_bound=205)

    flat_result, outside_result = assess_peak_ranges(
        [flat, outside],
        x,
        y,
        calibration=LINEAR_CALIBRATION,
    )

    assert flat_result.status is PeakRangeFitStatus.MISMATCH
    assert any("没有可确认" in reason for reason in flat_result.reasons)
    assert outside_result.status is PeakRangeFitStatus.UNAVAILABLE
    assert outside_result.actual_peak_x is None


def test_assessment_uses_spectrum_coordinates_after_workbench_trim():
    x = np.arange(4001.0, 4101.0)
    y = 2.0 + 80.0 * np.exp(-0.5 * ((x - 4050.0) / 2.5) ** 2)
    peak_range = PeakRange(
        mz=4050.0,
        peak_index=4050,
        left_bound=4042,
        right_bound=4058,
    )

    result = assess_peak_ranges(
        [peak_range],
        x,
        y,
        calibration=LINEAR_CALIBRATION,
    )[0]

    assert result.status is PeakRangeFitStatus.FIT
    assert result.actual_peak_x == 4050.0


def test_assessment_marks_materially_overlapping_ranges_for_review():
    x, y = _gaussian_spectrum()
    ranges = [
        PeakRange(mz=100.0, peak_index=100, left_bound=90, right_bound=110, label="A"),
        PeakRange(mz=100.0, peak_index=100, left_bound=108, right_bound=128, label="B"),
    ]

    results = assess_peak_ranges(
        ranges,
        x,
        y,
        calibration=LINEAR_CALIBRATION,
    )

    assert all("与相邻卡峰范围重叠" in result.reasons for result in results)
    assert results[0].status is PeakRangeFitStatus.REVIEW
