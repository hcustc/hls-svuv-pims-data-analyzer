from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from .calibration import Calibration


@dataclass
class GaussianFit:
    amplitude: float
    mean: float
    std_dev: float
    fwhm: float
    baseline: float


@dataclass
class Peak:
    index: int
    time: float
    mz: float
    intensity: float
    fwhm: float
    left_bound: int
    right_bound: int
    is_auto: bool = True
    gaussian_params: GaussianFit | None = None
    species: str = "Unknown"

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.gaussian_params is not None:
            data["gaussian_params"] = asdict(self.gaussian_params)
        return data


def gaussian(x, amplitude: float, mean: float, std_dev: float, baseline: float):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev**2)) + baseline


def fit_gaussian(y_data: Iterable[float], center_idx: int, window_size: int = 20) -> GaussianFit | None:
    data = np.asarray(list(y_data), dtype=float)
    if data.size == 0:
        return None
    start_idx = max(0, int(center_idx) - int(window_size))
    end_idx = min(len(data), int(center_idx) + int(window_size) + 1)
    x = np.arange(start_idx, end_idx, dtype=float)
    y = data[start_idx:end_idx]
    if len(x) < 5:
        return None
    try:
        initial = [float(np.max(y)), float(center_idx), 3.0, float(np.min(y))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(gaussian, x, y, p0=initial, maxfev=10000)
        amplitude, mean, std_dev, baseline = [float(v) for v in params]
        if std_dev <= 0:
            return None
        fwhm = float(2 * np.sqrt(2 * np.log(2)) * std_dev)
        return GaussianFit(amplitude=amplitude, mean=mean, std_dev=std_dev, fwhm=fwhm, baseline=baseline)
    except Exception:
        return None


def detect_peaks_in_range(
    y_data: Iterable[float],
    *,
    calibration: Calibration = Calibration(),
    start_idx: int = 0,
    end_idx: int | None = None,
    detection_min_idx: int = 3000,
    time_offset: float = 0.0,
    threshold_end: float = 2,
    min_intensity: float = 3,
    nearby_peak_window: int = 30,
    duplicate_window: int = 20,
    weak_tail_early_window: int = 90,
    weak_tail_late_window: int = 50,
    weak_tail_ratio: float = 5,
    gaussian_window_max: int = 30,
    gaussian_boundary_scale: float = 1.5,
    boundary_padding: int = 2,
) -> list[Peak]:
    """Detect peaks using the BL03U local-maxima and Gaussian-boundary logic."""
    data = np.asarray(list(y_data), dtype=float)
    if data.size == 0 or np.max(data) == 0:
        return []

    start_idx = max(0, int(start_idx))
    end_idx = len(data) if end_idx is None else min(int(end_idx), len(data))
    detection_start = max(int(detection_min_idx), start_idx)
    if detection_start >= end_idx:
        return []
    nearby_peak_window = max(0, int(nearby_peak_window))
    duplicate_window = max(0, int(duplicate_window))
    weak_tail_early_window = max(0, int(weak_tail_early_window))
    weak_tail_late_window = max(0, int(weak_tail_late_window))
    weak_tail_ratio = max(float(weak_tail_ratio), 0.000001)
    gaussian_window_max = max(3, int(gaussian_window_max))
    gaussian_boundary_scale = max(float(gaussian_boundary_scale), 0.1)
    boundary_padding = max(0, int(boundary_padding))

    local_max_indices: list[int] = []
    for idx in range(detection_start, end_idx - 1):
        if data[idx] > min_intensity and data[idx] >= data[idx - 1] and data[idx] >= data[idx + 1]:
            local_max_indices.append(idx)
    local_max_indices.sort(key=lambda item: -data[item])

    used_indices: set[int] = set()
    final_indices: list[int] = []
    for max_idx in local_max_indices:
        if max_idx in used_indices:
            continue
        peak_val = data[max_idx]
        has_higher_nearby = any(
            other != max_idx and other in final_indices and data[other] > peak_val
            for other in range(
                max(detection_start, max_idx - nearby_peak_window),
                min(end_idx, max_idx + nearby_peak_window + 1),
            )
        )
        if has_higher_nearby:
            continue

        weak_tail_range = weak_tail_early_window if max_idx <= 15000 else weak_tail_late_window
        if any(
            max_idx > prev
            and max_idx - prev <= weak_tail_range
            and peak_val * weak_tail_ratio <= data[prev]
            for prev in final_indices
        ):
            continue

        final_indices.append(max_idx)
        for used in range(
            max(detection_start, max_idx - duplicate_window),
            min(end_idx, max_idx + duplicate_window + 1),
        ):
            used_indices.add(used)

    peaks: list[Peak] = []
    for max_idx in final_indices:
        peak_val = float(data[max_idx])
        left = max_idx
        while left > detection_start and data[left] >= threshold_end:
            left -= 1
        right = max_idx
        while right < end_idx - 1 and data[right] >= threshold_end:
            right += 1

        fit = fit_gaussian(data, max_idx, min(gaussian_window_max, right - left + 5))
        if fit is not None:
            center = fit.mean
            fwhm = fit.fwhm
            left = int(center - fwhm * gaussian_boundary_scale)
            right = int(center + fwhm * gaussian_boundary_scale)
            left = max(detection_start, min(left, max_idx))
            right = min(end_idx - 1, max(right, max_idx))
            tof_time = center
        else:
            tof_time = float(max_idx)

        left = max(detection_start, left - boundary_padding)
        right = min(end_idx - 1, right + boundary_padding)
        left = min(left, max_idx)
        right = max(right, max_idx)
        peaks.append(
            Peak(
                index=int(round(tof_time)),
                time=float(tof_time + time_offset),
                mz=float(calibration.tof_to_mz(tof_time + time_offset)),
                intensity=peak_val,
                fwhm=float(right - left),
                left_bound=int(left),
                right_bound=int(right),
                is_auto=True,
                gaussian_params=fit,
            )
        )

    return sorted(peaks, key=lambda peak: peak.time)


def add_manual_peak(
    y_data: Iterable[float],
    left_idx: int,
    right_idx: int,
    *,
    calibration: Calibration = Calibration(),
    time_offset: float = 0.0,
) -> Peak | None:
    data = np.asarray(list(y_data), dtype=float)
    if data.size == 0:
        return None
    left_idx = max(0, int(left_idx))
    right_idx = min(len(data) - 1, int(right_idx))
    if left_idx >= right_idx:
        return None
    region = data[left_idx : right_idx + 1]
    max_idx = int(np.argmax(region) + left_idx)
    fit = fit_gaussian(data, max_idx)
    return Peak(
        index=max_idx,
        time=float(max_idx + time_offset),
        mz=float(calibration.tof_to_mz(max_idx + time_offset)),
        intensity=float(np.max(region)),
        fwhm=float(fit.fwhm if fit else 0),
        left_bound=left_idx,
        right_bound=right_idx,
        is_auto=False,
        gaussian_params=fit,
    )


def peaks_to_dataframe(peaks: Iterable[Peak]) -> pd.DataFrame:
    rows = [
        {
            "Species": peak.species,
            "flight_time": peak.time,
            "mz": peak.mz,
            "intensity": peak.intensity,
            "fwhm": peak.fwhm,
            "left_bound": peak.left_bound,
            "right_bound": peak.right_bound,
            "type": "auto" if peak.is_auto else "manual",
        }
        for peak in peaks
    ]
    return pd.DataFrame(rows)
