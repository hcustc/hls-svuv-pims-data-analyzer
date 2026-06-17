from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any

import numpy as np

from .calibration import Calibration
from .peak_detection import Peak
from .spectrum_io import read_spectrum


@dataclass(frozen=True)
class SyntheticPeakCase:
    name: str
    description: str
    y: np.ndarray
    injected_peaks: list[Peak]


def read_benchmark_background(path: str) -> np.ndarray:
    """Read a spectrum file as background for synthetic peak benchmarks."""
    spectrum = read_spectrum(path, header_lines=None, trim_start=0)
    return np.asarray(spectrum.y, dtype=float)


def _inject_gaussian_peak(
    y: np.ndarray,
    *,
    center: int,
    amplitude: float,
    width: float,
    calibration: Calibration,
) -> Peak:
    x = np.arange(y.size, dtype=float)
    y += amplitude * np.exp(-0.5 * ((x - float(center)) / float(width)) ** 2)
    left = max(0, int(round(center - 3 * width)))
    right = min(y.size - 1, int(round(center + 3 * width)))
    return Peak(
        index=int(center),
        time=float(center),
        mz=float(calibration.tof_to_mz(center)),
        intensity=float(amplitude),
        fwhm=float(2.354820045 * width),
        left_bound=left,
        right_bound=right,
        is_auto=False,
        species="Synthetic",
    )


def build_default_synthetic_benchmark(
    background: Iterable[float],
    *,
    calibration: Calibration = Calibration(),
) -> list[SyntheticPeakCase]:
    """Build a small synthetic benchmark from a measured background trace."""
    base = np.asarray(list(background), dtype=float)
    if base.size < 256:
        base = np.pad(base, (0, 256 - base.size), mode="constant")
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    noise_floor = float(np.percentile(np.abs(base), 95)) if base.size else 1.0
    amplitude = max(noise_floor * 8.0, 20.0)
    n = base.size

    cases: list[SyntheticPeakCase] = [
        SyntheticPeakCase(
            name="blank_background",
            description="Measured background without injected peaks",
            y=base.copy(),
            injected_peaks=[],
        )
    ]

    single = base.copy()
    single_truth = [
        _inject_gaussian_peak(
            single,
            center=max(20, int(n * 0.45)),
            amplitude=amplitude,
            width=4.0,
            calibration=calibration,
        )
    ]
    cases.append(SyntheticPeakCase("single_peak", "One isolated injected peak", single, single_truth))

    double = base.copy()
    double_truth = [
        _inject_gaussian_peak(
            double,
            center=max(20, int(n * 0.35)),
            amplitude=amplitude * 0.8,
            width=3.0,
            calibration=calibration,
        ),
        _inject_gaussian_peak(
            double,
            center=max(40, int(n * 0.68)),
            amplitude=amplitude * 1.2,
            width=7.0,
            calibration=calibration,
        ),
    ]
    cases.append(SyntheticPeakCase("mixed_widths", "Two injected peaks with different widths", double, double_truth))

    shoulder = base.copy()
    center = max(30, int(n * 0.55))
    shoulder_truth = [
        _inject_gaussian_peak(
            shoulder,
            center=center,
            amplitude=amplitude,
            width=5.0,
            calibration=calibration,
        ),
        _inject_gaussian_peak(
            shoulder,
            center=min(n - 30, center + 18),
            amplitude=amplitude * 0.55,
            width=5.0,
            calibration=calibration,
        ),
    ]
    cases.append(SyntheticPeakCase("nearby_peaks", "Two partially separated injected peaks", shoulder, shoulder_truth))
    return cases


def _match_peaks(
    detected_peaks: Iterable[Peak],
    truth_peaks: Iterable[Peak],
    *,
    mz_tolerance: float = 0.35,
) -> tuple[list[tuple[Peak, Peak]], list[Peak], list[Peak]]:
    """Greedily match detected peaks to truth peaks by nearest m/z."""
    remaining_truth = list(truth_peaks)
    matches: list[tuple[Peak, Peak]] = []
    false_peaks: list[Peak] = []

    for detected in sorted(detected_peaks, key=lambda peak: peak.mz):
        if not remaining_truth:
            false_peaks.append(detected)
            continue
        nearest = min(remaining_truth, key=lambda truth: abs(truth.mz - detected.mz))
        if abs(nearest.mz - detected.mz) <= mz_tolerance:
            matches.append((detected, nearest))
            remaining_truth.remove(nearest)
        else:
            false_peaks.append(detected)

    return matches, remaining_truth, false_peaks


def evaluate_synthetic_peak_benchmark(
    cases: Iterable[SyntheticPeakCase],
    detector: Callable[[np.ndarray], list[Peak]],
    *,
    mz_tolerance: float = 0.35,
) -> dict[str, Any]:
    """Evaluate a detector callable against synthetic benchmark cases."""
    rows = []
    total_tp = 0
    total_fp = 0
    total_truth = 0
    for case in cases:
        detected = detector(case.y)
        matches, missed, false_peaks = _match_peaks(detected, case.injected_peaks, mz_tolerance=mz_tolerance)
        rows.append(
            {
                "case": case.name,
                "truth_count": len(case.injected_peaks),
                "detected_count": len(detected),
                "true_positive_count": len(matches),
                "missed_count": len(missed),
                "false_positive_count": len(false_peaks),
            }
        )
        total_tp += len(matches)
        total_fp += len(false_peaks)
        total_truth += len(case.injected_peaks)
    return {
        "total_recall": total_tp / total_truth if total_truth else 1.0,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_truth": total_truth,
        "case_results": rows,
    }
