from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter

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


def fit_gaussian(
    y_data: Iterable[float],
    center_idx: int,
    window_size: int = 20,
    *,
    constrained: bool = True,
) -> GaussianFit | None:
    """Fit a local Gaussian without allowing the optimizer to leave the peak.

    The old unconstrained fit could return negative amplitudes, extremely wide
    peaks, or centers hundreds of samples away from ``center_idx`` on sloping
    baselines.  Those values then contaminated calibration, clustering, and
    integration.  The bounds below intentionally describe only a *local* peak.
    """
    data = np.asarray(y_data, dtype=float)
    if data.size == 0:
        return None
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    center_idx = max(0, min(int(center_idx), data.size - 1))
    window_size = max(2, int(window_size))
    start_idx = max(0, int(center_idx) - int(window_size))
    end_idx = min(len(data), int(center_idx) + int(window_size) + 1)
    x = np.arange(start_idx, end_idx, dtype=float)
    y = data[start_idx:end_idx]
    if len(x) < 5:
        return None
    if not constrained:
        # Historical BL03U behavior retained only for the explicit legacy
        # detector so old projects and published integrations remain
        # reproducible.  New detectors use the validated local fit below.
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
    edge_count = max(1, min(5, len(y) // 4))
    edge_values = np.concatenate((y[:edge_count], y[-edge_count:]))
    baseline_initial = float(np.median(edge_values))
    amplitude_initial = float(np.max(y) - baseline_initial)
    data_range = float(np.ptp(y))
    if not np.isfinite(amplitude_initial) or amplitude_initial <= max(np.finfo(float).eps, data_range * 1e-6):
        return None
    local_max_pos = int(np.argmax(y))
    support_level = baseline_initial + 0.10 * amplitude_initial
    if (
        0 < local_max_pos < len(y) - 1
        and y[local_max_pos - 1] < support_level
        and y[local_max_pos + 1] < support_level
    ):
        return None

    half_height = baseline_initial + amplitude_initial * 0.5
    above_half = np.flatnonzero(y >= half_height)
    if above_half.size >= 2:
        std_initial = max(0.5, float(above_half[-1] - above_half[0]) / 2.354820045)
    else:
        std_initial = min(3.0, max(0.5, window_size / 3.0))

    max_center_shift = max(1.5, min(float(window_size) * 0.5, 8.0))
    mean_lower = max(float(start_idx), float(center_idx) - max_center_shift)
    mean_upper = min(float(end_idx - 1), float(center_idx) + max_center_shift)
    std_upper = max(1.0, min(float(window_size), float(len(y) - 1) / 2.0))
    baseline_margin = max(data_range, abs(baseline_initial) * 0.5, 1.0)
    try:
        initial = [amplitude_initial, float(center_idx), min(std_initial, std_upper), baseline_initial]
        lower_bounds = [0.0, mean_lower, 0.35, float(np.min(y) - baseline_margin)]
        upper_bounds = [
            max(amplitude_initial * 5.0, data_range * 5.0, 1.0),
            mean_upper,
            std_upper,
            float(np.max(y) + baseline_margin),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                gaussian,
                x,
                y,
                p0=initial,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
            )
        amplitude, mean, std_dev, baseline = [float(v) for v in params]
        if amplitude <= 0 or std_dev <= 0 or abs(mean - center_idx) > max_center_shift + 1e-6:
            return None
        fitted = gaussian(x, amplitude, mean, std_dev, baseline)
        residual_error = float(np.sum((y - fitted) ** 2))
        flat_error = float(np.sum((y - np.mean(y)) ** 2))
        if not np.all(np.isfinite(fitted)) or flat_error <= 0 or residual_error >= flat_error * 0.98:
            return None
        fwhm = float(2 * np.sqrt(2 * np.log(2)) * std_dev)
        if fwhm > 2.0 * window_size:
            return None
        return GaussianFit(amplitude=amplitude, mean=mean, std_dev=std_dev, fwhm=fwhm, baseline=baseline)
    except Exception:
        return None


def _valid_savgol_window(data_size: int, window_size: int, poly_order: int) -> int | None:
    window = max(int(window_size), int(poly_order) + 2)
    if window % 2 == 0:
        window += 1
    if window > data_size:
        window = data_size if data_size % 2 == 1 else data_size - 1
    if window <= poly_order or window < 3:
        return None
    return window


def _smooth_for_peak_detection(data: np.ndarray, window_size: int, poly_order: int) -> np.ndarray:
    window = _valid_savgol_window(data.size, window_size, poly_order)
    if window is None:
        return data
    return savgol_filter(data, window, poly_order)


def _rolling_percentile_baseline(data: np.ndarray, window_size: int, percentile: float) -> np.ndarray:
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    percentile = max(0.0, min(float(percentile), 100.0))
    return percentile_filter(data, percentile=percentile, size=window_size, mode="nearest")


def _estimate_noise_sigma(data: np.ndarray) -> float:
    """Robustly estimate point noise while being insensitive to broad peaks."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0
    differences = np.diff(values)
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    return mad / (0.67448975 * np.sqrt(2.0)) if mad > 0 else 0.0


def _sanitize_peak_bounds(
    left: int,
    right: int,
    center_idx: int,
    lower_idx: int,
    upper_idx: int,
    *,
    max_width: int,
    fallback_half_width: int,
) -> tuple[int, int]:
    """Keep peak bounds local when an absolute threshold never reaches baseline.

    BL03U spectra can have a positive baseline across the whole acquisition.
    In that case expanding left/right until ``y < threshold_end`` may walk to
    the entire detection interval for a tiny local maximum. Such a full-span
    bound is not a valid integration window for a single mass peak.
    """
    right_limit = max(int(lower_idx), int(upper_idx) - 1)
    center = max(int(lower_idx), min(int(center_idx), right_limit))
    left = int(left)
    right = int(right)
    if right < left:
        left, right = right, left

    left = max(int(lower_idx), min(left, right_limit))
    right = max(int(lower_idx), min(right, right_limit))
    max_width = max(1, int(max_width))
    if left <= center <= right and 0 < right - left <= max_width:
        return left, right

    half_width = max(1, int(fallback_half_width))
    left = max(int(lower_idx), center - half_width)
    right = min(right_limit, center + half_width)
    left = min(left, center)
    right = max(right, center)
    return left, right


def detect_peaks_prominence(
    y_data: Iterable[float],
    *,
    calibration: Calibration = Calibration(),
    start_idx: int = 0,
    end_idx: int | None = None,
    detection_min_idx: int = 3000,
    time_offset: float = 0.0,
    threshold_end: float = 2,
    min_intensity: float = 3,
    duplicate_window: int = 20,
    gaussian_window_max: int = 30,
    gaussian_boundary_scale: float = 1.5,
    boundary_padding: int = 2,
    prominence_ratio: float = 0.005,
    smoothing_window: int = 5,
    smoothing_poly_order: int = 2,
    baseline_window: int = 301,
    baseline_percentile: float = 5.0,
    min_peak_width: int = 1,
    max_peak_width: int = 80,
) -> list[Peak]:
    """Detect peaks after baseline correction using scipy prominence and width filters."""
    data = np.asarray(y_data, dtype=float)
    if data.size == 0:
        return []
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(data) <= 0:
        return []

    start_idx = max(0, int(start_idx))
    end_idx = len(data) if end_idx is None else min(int(end_idx), len(data))
    detection_start = max(int(detection_min_idx), start_idx)
    if detection_start >= end_idx:
        return []

    duplicate_window = max(1, int(duplicate_window))
    gaussian_window_max = max(3, int(gaussian_window_max))
    gaussian_boundary_scale = max(float(gaussian_boundary_scale), 0.1)
    boundary_padding = max(0, int(boundary_padding))
    min_peak_width = max(1, int(min_peak_width))
    max_peak_width = max(min_peak_width, int(max_peak_width))

    segment = data[detection_start:end_idx]
    smoothed = _smooth_for_peak_detection(segment, smoothing_window, smoothing_poly_order)
    baseline = _rolling_percentile_baseline(smoothed, baseline_window, baseline_percentile)
    corrected = np.maximum(smoothed - baseline, 0.0)
    corrected_max = float(np.max(corrected))
    if corrected_max <= 0:
        return []

    noise_sigma = _estimate_noise_sigma(segment - smoothed)
    if noise_sigma <= 0:
        noise_sigma = _estimate_noise_sigma(corrected)
    # A rolling high percentile supplies a local scale.  Unlike a fraction of
    # the spectrum-wide maximum, it does not hide a weak peak merely because a
    # much stronger mass exists elsewhere in the acquisition.
    local_reference = percentile_filter(
        corrected,
        percentile=95.0,
        size=max(11, int(baseline_window) | 1),
        mode="nearest",
    )
    prominence_threshold = np.maximum(
        max(0.0, 3.0 * noise_sigma),
        max(0.0, float(prominence_ratio)) * local_reference,
    )

    peak_indices, _ = find_peaks(
        corrected,
        height=max(0.0, float(min_intensity)),
        prominence=prominence_threshold,
        distance=duplicate_window,
        width=(min_peak_width, max_peak_width),
    )
    if len(peak_indices) == 0:
        return []

    widths, _, left_ips, right_ips = peak_widths(corrected, peak_indices, rel_height=0.5)
    corrected_full = np.zeros_like(data)
    corrected_full[detection_start:end_idx] = corrected
    peaks: list[Peak] = []

    for peak_idx, width, left_ip, right_ip in zip(peak_indices, widths, left_ips, right_ips):
        max_idx = int(peak_idx) + detection_start
        left = int(np.floor(left_ip)) + detection_start
        right = int(np.ceil(right_ip)) + detection_start
        fallback_half_width = max(min_peak_width, min(max_peak_width, int(np.ceil(width)) + boundary_padding))
        max_boundary_width = max(
            max_peak_width * 4,
            gaussian_window_max * 8,
            fallback_half_width * 4,
        )

        left_local = int(peak_idx)
        while left_local > 0 and corrected[left_local] >= threshold_end:
            left_local -= 1
        right_local = int(peak_idx)
        while right_local < corrected.size - 1 and corrected[right_local] >= threshold_end:
            right_local += 1
        left = min(left, left_local + detection_start)
        right = max(right, right_local + detection_start)

        fit = fit_gaussian(corrected_full, max_idx, min(gaussian_window_max, int(width) + 5))
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
        left, right = _sanitize_peak_bounds(
            left,
            right,
            max_idx,
            detection_start,
            end_idx,
            max_width=max_boundary_width,
            fallback_half_width=fallback_half_width,
        )
        peak_time = float(tof_time + time_offset)
        peaks.append(
            Peak(
                index=int(round(tof_time)),
                time=peak_time,
                mz=float(calibration.tof_to_mz(peak_time)),
                intensity=float(data[max_idx]),
                fwhm=float(right - left),
                left_bound=int(left),
                right_bound=int(right),
                is_auto=True,
                gaussian_params=fit,
            )
        )

    return sorted(peaks, key=lambda peak: peak.time)


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
    weak_tail_cutoff_idx: int = 15000,
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

        weak_tail_range = weak_tail_early_window if max_idx <= weak_tail_cutoff_idx else weak_tail_late_window
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

        fit = fit_gaussian(
            data,
            max_idx,
            min(gaussian_window_max, right - left + 5),
            constrained=False,
        )
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
        max_boundary_width = max(
            nearby_peak_window * 4,
            duplicate_window * 4,
            gaussian_window_max * 8,
            25,
        )
        fallback_half_width = max(3, duplicate_window, gaussian_window_max)
        left, right = _sanitize_peak_bounds(
            left,
            right,
            max_idx,
            detection_start,
            end_idx,
            max_width=max_boundary_width,
            fallback_half_width=fallback_half_width,
        )
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


def detect_peaks_by_algorithm(
    y_data: Iterable[float],
    *,
    algorithm: str = "legacy",
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
    prominence_ratio: float = 0.005,
    smoothing_window: int = 5,
    smoothing_poly_order: int = 2,
    baseline_window: int = 301,
    baseline_percentile: float = 5.0,
    min_peak_width: int = 1,
    max_peak_width: int = 80,
    cwt_snr_threshold: float = 0.02,
    cwt_wavelet_max_width: int = 30,
    weak_tail_cutoff_idx: int = 15000,
    # Ensemble parameters
    vote_threshold: float = 0.667,
    min_intensity_for_single_vote: float = 5.0,
    mz_tolerance: float = 0.2,
) -> list[Peak]:
    """Dispatch BL03U peak detection to legacy, prominence, CWT, or ensemble algorithms."""
    algorithm = (algorithm or "legacy").lower()
    if algorithm == "legacy":
        return detect_peaks_in_range(
            y_data,
            calibration=calibration,
            start_idx=start_idx,
            end_idx=end_idx,
            detection_min_idx=detection_min_idx,
            time_offset=time_offset,
            threshold_end=threshold_end,
            min_intensity=min_intensity,
            nearby_peak_window=nearby_peak_window,
            duplicate_window=duplicate_window,
            weak_tail_early_window=weak_tail_early_window,
            weak_tail_late_window=weak_tail_late_window,
            weak_tail_ratio=weak_tail_ratio,
            gaussian_window_max=gaussian_window_max,
            gaussian_boundary_scale=gaussian_boundary_scale,
            boundary_padding=boundary_padding,
            weak_tail_cutoff_idx=weak_tail_cutoff_idx,
        )
    if algorithm == "prominence":
        return detect_peaks_prominence(
            y_data,
            calibration=calibration,
            start_idx=start_idx,
            end_idx=end_idx,
            detection_min_idx=detection_min_idx,
            time_offset=time_offset,
            threshold_end=threshold_end,
            min_intensity=min_intensity,
            duplicate_window=duplicate_window,
            gaussian_window_max=gaussian_window_max,
            gaussian_boundary_scale=gaussian_boundary_scale,
            boundary_padding=boundary_padding,
            prominence_ratio=prominence_ratio,
            smoothing_window=smoothing_window,
            smoothing_poly_order=smoothing_poly_order,
            baseline_window=baseline_window,
            baseline_percentile=baseline_percentile,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
        )
    if algorithm == "cwt":
        from .cwt_peak_detection import CwtPeakDetectionConfig, detect_peaks_cwt

        data = np.asarray(list(y_data), dtype=float)
        start_idx = max(0, int(start_idx))
        end_idx = len(data) if end_idx is None else min(int(end_idx), len(data))
        detection_start = max(int(detection_min_idx), start_idx)
        if detection_start >= end_idx:
            return []
        segment = data[detection_start:end_idx]
        min_peak_width = max(1, int(min_peak_width))
        max_peak_width = max(min_peak_width, int(max_peak_width))
        window_size = max(3, int(smoothing_window))
        wavelet_max_width = max(2, int(cwt_wavelet_max_width))
        config = CwtPeakDetectionConfig(
            window_size=window_size,
            poly_order=max(1, int(smoothing_poly_order)),
            snr_threshold=max(0.0, float(cwt_snr_threshold)),
            prominence_ratio=max(0.0, float(prominence_ratio)),
            min_peak_distance=max(1, int(duplicate_window)),
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            wavelet_widths=tuple(range(1, wavelet_max_width + 1)),
            baseline_percentile=baseline_percentile,
            baseline_window_factor=max(1, int(baseline_window / window_size)),
        )
        return detect_peaks_cwt(
            segment,
            calibration=calibration,
            start_idx=detection_start,
            time_offset=time_offset,
            config=config,
        )
    if algorithm == "ensemble":
        return detect_peaks_ensemble(
            y_data,
            calibration=calibration,
            start_idx=start_idx,
            end_idx=end_idx,
            detection_min_idx=detection_min_idx,
            time_offset=time_offset,
            threshold_end=threshold_end,
            min_intensity=min_intensity,
            nearby_peak_window=nearby_peak_window,
            duplicate_window=duplicate_window,
            weak_tail_early_window=weak_tail_early_window,
            weak_tail_late_window=weak_tail_late_window,
            weak_tail_ratio=weak_tail_ratio,
            gaussian_window_max=gaussian_window_max,
            gaussian_boundary_scale=gaussian_boundary_scale,
            boundary_padding=boundary_padding,
            weak_tail_cutoff_idx=weak_tail_cutoff_idx,
            prominence_ratio=prominence_ratio,
            smoothing_window=smoothing_window,
            smoothing_poly_order=smoothing_poly_order,
            baseline_window=baseline_window,
            baseline_percentile=baseline_percentile,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            cwt_snr_threshold=cwt_snr_threshold,
            cwt_wavelet_max_width=cwt_wavelet_max_width,
            vote_threshold=vote_threshold,
            min_intensity_for_single_vote=min_intensity_for_single_vote,
            mz_tolerance=mz_tolerance,
        )
    raise ValueError("algorithm must be 'legacy', 'prominence', 'cwt', or 'ensemble'")


def _cluster_peaks_by_mz(
    peaks: list[Peak],
    tolerance: float = 0.2,
) -> list[list[Peak]]:
    """按m/z将峰聚类，同一聚类内的峰距离<=tolerance。"""
    if not peaks:
        return []

    sorted_peaks = sorted(peaks, key=lambda p: p.mz)
    clusters: list[list[Peak]] = []
    current_cluster = [sorted_peaks[0]]

    for peak in sorted_peaks[1:]:
        if peak.mz - current_cluster[0].mz <= tolerance:
            current_cluster.append(peak)
        else:
            clusters.append(current_cluster)
            current_cluster = [peak]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def _cluster_algorithm_peaks_by_mz(
    peaks: list[tuple[str, Peak]],
    tolerance: float = 0.2,
) -> list[list[tuple[str, Peak]]]:
    """Cluster algorithm-labelled peaks by m/z."""
    if not peaks:
        return []

    sorted_peaks = sorted(peaks, key=lambda item: item[1].mz)
    clusters: list[list[tuple[str, Peak]]] = []
    current_cluster = [sorted_peaks[0]]

    for item in sorted_peaks[1:]:
        # Compare with the cluster's first member, not only the previous peak.
        # This prevents chain-link merging such as 10.0, 10.15, 10.30 with a
        # tolerance of 0.2, where the endpoints are different physical peaks.
        if item[1].mz - current_cluster[0][1].mz <= tolerance:
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def _merge_cluster_peaks(cluster: list[Peak]) -> Peak:
    """Merge repeated estimates of one peak using a robust consensus center."""
    if not cluster:
        raise ValueError("cluster must not be empty")

    if len(cluster) == 1:
        return cluster[0]

    # All algorithms sample the same raw intensity, so intensity weighting does
    # not represent confidence.  The median is stable when one fit is shifted.
    consensus_mz = float(np.median([p.mz for p in cluster]))
    consensus_time = float(np.median([p.time for p in cluster]))

    # intensity取最大值
    max_intensity = max(p.intensity for p in cluster)

    # Prefer a validated Gaussian closest to consensus for its integration
    # bounds; otherwise use the closest detector estimate.
    max_peak = min(
        cluster,
        key=lambda p: (
            p.gaussian_params is None,
            abs(p.time - consensus_time),
            -p.intensity,
        ),
    )

    return Peak(
        index=max_peak.index,
        time=consensus_time,
        mz=consensus_mz,
        intensity=max_intensity,
        fwhm=max_peak.fwhm,
        left_bound=max_peak.left_bound,
        right_bound=max_peak.right_bound,
        is_auto=True,
        gaussian_params=max_peak.gaussian_params,
        species=max_peak.species,
    )


def _single_vote_peak_is_plausible(
    data: np.ndarray,
    peak: Peak,
    *,
    min_intensity: float,
    local_window: int,
) -> bool:
    """Reject isolated one-sample artifacts from the permissive 1-vote path."""
    if peak.intensity < min_intensity or data.size < 3:
        return False
    center = max(1, min(int(round(peak.index)), data.size - 2))
    radius = max(3, min(int(local_window), 15))
    left = max(0, center - radius)
    right = min(data.size, center + radius + 1)
    region = data[left:right]
    if region.size < 3:
        return False
    baseline = float(np.percentile(region, 20.0))
    signal = float(data[center] - baseline)
    noise = _estimate_noise_sigma(region)
    if signal <= max(0.0, 3.0 * noise):
        return False
    left_min = float(np.min(data[left : center + 1]))
    right_min = float(np.min(data[center:right]))
    local_prominence = float(data[center] - max(left_min, right_min))
    if local_prominence < max(3.0 * noise, 0.10 * signal):
        return False
    support_level = baseline + 0.12 * signal
    return bool(data[center - 1] >= support_level and data[center + 1] >= support_level)


def _is_consistent_impulse_artifact(data: np.ndarray, peak: Peak, *, local_window: int) -> bool:
    """Identify a one-sample detector/electronic spike at a voted peak."""
    if data.size < 3:
        return False
    center = max(1, min(int(round(peak.index)), data.size - 2))
    # Do not interfere with mocked/externally supplied Peak objects whose
    # reported intensity clearly does not describe this data array.
    if not np.isclose(data[center], peak.intensity, rtol=0.2, atol=1e-9):
        return False
    radius = max(3, min(int(local_window), 15))
    region = data[max(0, center - radius) : min(data.size, center + radius + 1)]
    baseline = float(np.percentile(region, 20.0))
    signal = float(data[center] - baseline)
    if signal <= 0:
        return False
    support_level = baseline + 0.12 * signal
    return bool(data[center - 1] < support_level and data[center + 1] < support_level)


def detect_peaks_ensemble(
    y_data: Iterable[float],
    *,
    calibration: Calibration = Calibration(),
    start_idx: int = 0,
    end_idx: int | None = None,
    detection_min_idx: int = 3000,
    time_offset: float = 0.0,
    # 融合参数
    use_legacy: bool = True,
    use_prominence: bool = True,
    use_cwt: bool = True,
    mz_tolerance: float = 0.2,
    vote_threshold: float = 0.667,
    min_intensity_for_single_vote: float = 5.0,
    # Legacy参数（recall-focused）
    threshold_end: float = 1.0,
    min_intensity: float = 1.0,
    nearby_peak_window: int = 30,
    duplicate_window: int = 20,
    weak_tail_early_window: int = 120,
    weak_tail_late_window: int = 80,
    weak_tail_ratio: float = 3.0,
    weak_tail_cutoff_idx: int = 15000,
    gaussian_window_max: int = 30,
    gaussian_boundary_scale: float = 1.5,
    boundary_padding: int = 2,
    # Prominence参数（recall-focused）
    prominence_ratio: float = 0.002,
    smoothing_window: int = 7,
    smoothing_poly_order: int = 2,
    baseline_window: int = 301,
    baseline_percentile: float = 5.0,
    min_peak_width: int = 1,
    max_peak_width: int = 100,
    # CWT参数（recall-focused）
    cwt_snr_threshold: float = 0.01,
    cwt_wavelet_max_width: int = 60,
) -> list[Peak]:
    """检测峰的融合方法：结合Legacy、Prominence、CWT三个算法的优势。

    通过投票规则最大化峰召回率（recall）：
    - 2/3算法同意 → 高置信峰
    - 1/3算法 + 强度>阈值 → 低置信峰

    Args:
        y_data: 光谱强度数据
        calibration: 质量标度校准对象
        start_idx, end_idx: 检测范围
        detection_min_idx: 检测起始索引（TOF时间阈值）
        time_offset: 时间偏移
        use_legacy, use_prominence, use_cwt: 启用哪些算法
        mz_tolerance: m/z聚类容差（默认0.2）
        vote_threshold: 投票阈值（默认0.667 = 2/3）
        min_intensity_for_single_vote: 单个算法检出时的强度下界
        [其他参数]: 三个算法各自的参数（recall-focused初始值）

    Returns:
        融合后的峰列表，按time排序
    """
    data = np.asarray(y_data, dtype=float)
    if data.size == 0:
        return []
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(data) <= 0:
        return []

    # 统计启用的算法数
    enabled_algorithms = sum([use_legacy, use_prominence, use_cwt])
    if enabled_algorithms == 0:
        return []

    # 分别运行各算法
    peaks_by_algo: dict[str, list[Peak]] = {}

    if use_legacy:
        peaks_by_algo["legacy"] = detect_peaks_in_range(
            data,
            calibration=calibration,
            start_idx=start_idx,
            end_idx=end_idx,
            detection_min_idx=detection_min_idx,
            time_offset=time_offset,
            threshold_end=threshold_end,
            min_intensity=min_intensity,
            nearby_peak_window=nearby_peak_window,
            duplicate_window=duplicate_window,
            weak_tail_early_window=weak_tail_early_window,
            weak_tail_late_window=weak_tail_late_window,
            weak_tail_ratio=weak_tail_ratio,
            gaussian_window_max=gaussian_window_max,
            gaussian_boundary_scale=gaussian_boundary_scale,
            boundary_padding=boundary_padding,
            weak_tail_cutoff_idx=weak_tail_cutoff_idx,
        )

    if use_prominence:
        peaks_by_algo["prominence"] = detect_peaks_prominence(
            data,
            calibration=calibration,
            start_idx=start_idx,
            end_idx=end_idx,
            detection_min_idx=detection_min_idx,
            time_offset=time_offset,
            threshold_end=threshold_end,
            min_intensity=min_intensity,
            duplicate_window=duplicate_window,
            gaussian_window_max=gaussian_window_max,
            gaussian_boundary_scale=gaussian_boundary_scale,
            boundary_padding=boundary_padding,
            prominence_ratio=prominence_ratio,
            smoothing_window=smoothing_window,
            smoothing_poly_order=smoothing_poly_order,
            baseline_window=baseline_window,
            baseline_percentile=baseline_percentile,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
        )

    if use_cwt:
        peaks_by_algo["cwt"] = detect_peaks_by_algorithm(
            data,
            algorithm="cwt",
            calibration=calibration,
            start_idx=start_idx,
            end_idx=end_idx,
            detection_min_idx=detection_min_idx,
            time_offset=time_offset,
            cwt_snr_threshold=cwt_snr_threshold,
            cwt_wavelet_max_width=cwt_wavelet_max_width,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            smoothing_window=smoothing_window,
            smoothing_poly_order=smoothing_poly_order,
            baseline_window=baseline_window,
            baseline_percentile=baseline_percentile,
            prominence_ratio=prominence_ratio,
            duplicate_window=duplicate_window,
        )

    # 合并所有峰并保留算法来源，投票必须按不同算法计数。
    all_peaks: list[tuple[str, Peak]] = []
    for algorithm_name, peaks in peaks_by_algo.items():
        all_peaks.extend((algorithm_name, peak) for peak in peaks)

    if not all_peaks:
        return []

    # 按m/z聚类
    clusters = _cluster_algorithm_peaks_by_mz(all_peaks, tolerance=mz_tolerance)

    # 投票规则应用。Treat UI/default values like 0.667 and 0.334 as rounded
    # representations of 2/3 and 1/3 instead of requiring an impossible extra vote.
    vote_threshold_value = max(0.0, min(float(vote_threshold), 1.0))
    vote_threshold_tolerance = 1e-3
    final_peaks: list[Peak] = []

    for cluster in clusters:
        cluster_peaks = [peak for _algorithm_name, peak in cluster]
        vote_count = len({algorithm_name for algorithm_name, _peak in cluster})

        # 规则1：足够的投票 → 保留
        if (vote_count / enabled_algorithms) + vote_threshold_tolerance >= vote_threshold_value:
            merged = _merge_cluster_peaks(cluster_peaks)
            if not _is_consistent_impulse_artifact(data, merged, local_window=duplicate_window):
                final_peaks.append(merged)
        # 规则2：单个算法检出但强度足够 → 保留
        elif vote_count == 1:
            peak = max(cluster_peaks, key=lambda item: item.intensity)
            if _single_vote_peak_is_plausible(
                data,
                peak,
                min_intensity=float(min_intensity_for_single_vote),
                local_window=duplicate_window,
            ):
                final_peaks.append(peak)

    return sorted(final_peaks, key=lambda peak: peak.time)


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
