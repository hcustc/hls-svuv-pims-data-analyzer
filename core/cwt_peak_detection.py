from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .calibration import Calibration
from .peak_detection import Peak, fit_gaussian

try:
    import pywt
except Exception:  # pragma: no cover - handled when CWT is requested.
    pywt = None


@dataclass(frozen=True)
class CwtPeakDetectionConfig:
    window_size: int = 3
    poly_order: int = 2
    snr_threshold: float = 0.02
    prominence_ratio: float = 0.005
    min_peak_distance: int = 10
    min_peak_width: int = 3
    max_peak_width: int = 50
    wavelet_widths: Sequence[int] = field(default_factory=lambda: tuple(range(1, 30)))
    wavelet: str = "mexh"
    baseline_method: str = "rolling_percentile"
    baseline_percentile: float = 5.0
    baseline_window_factor: int = 10


def _valid_savgol_window(data_size: int, window_size: int, poly_order: int) -> int | None:
    window = max(int(window_size), int(poly_order) + 2)
    if window % 2 == 0:
        window += 1
    if window > data_size:
        window = data_size if data_size % 2 == 1 else data_size - 1
    if window <= poly_order or window < 3:
        return None
    return window


def smooth_data(
    y_data: Iterable[float],
    *,
    window_size: int = 3,
    poly_order: int = 2,
) -> np.ndarray:
    data = np.asarray(list(y_data), dtype=float)
    window = _valid_savgol_window(data.size, window_size, poly_order)
    if window is None:
        return data
    return savgol_filter(data, window, poly_order)


def estimate_noise(y_data: Iterable[float]) -> float:
    data = np.asarray(list(y_data), dtype=float)
    if data.size < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(data))) * 1.4826)


def correct_baseline(
    y_data: Iterable[float],
    *,
    method: str = "rolling_percentile",
    window_size: int = 30,
    percentile: float = 5.0,
) -> np.ndarray:
    data = np.asarray(list(y_data), dtype=float)
    if data.size == 0:
        return data
    if method == "min":
        corrected = data - float(np.min(data))
    elif method == "median":
        corrected = data - float(np.median(data))
    elif method == "rolling_percentile":
        window_size = max(3, int(window_size))
        pad_width = window_size // 2
        padded = np.pad(data, pad_width, mode="edge")
        baseline = np.empty_like(data)
        for idx in range(data.size):
            baseline[idx] = np.percentile(padded[idx : idx + window_size], percentile)
        corrected = data - baseline
    else:
        raise ValueError(f"Unsupported baseline correction method: {method}")
    return np.maximum(corrected, 0)


def calculate_cwt(
    y_data: Iterable[float],
    widths: Sequence[int],
    *,
    wavelet: str = "mexh",
) -> np.ndarray:
    if pywt is None:
        raise ImportError("PyWavelets is required for CWT peak detection. Install package 'PyWavelets'.")
    data = np.asarray(list(y_data), dtype=float)
    cwt_matrix, _ = pywt.cwt(data, widths, wavelet)
    return cwt_matrix


def detect_peaks_cwt(
    y_data: Iterable[float],
    *,
    calibration: Calibration = Calibration(),
    start_idx: int = 0,
    time_values: Iterable[float] | None = None,
    time_offset: float = 0.0,
    config: CwtPeakDetectionConfig | None = None,
) -> list[Peak]:
    """Detect mass-spectrum peaks using continuous wavelet transform."""
    config = config or CwtPeakDetectionConfig()
    raw = np.asarray(list(y_data), dtype=float)
    if raw.size == 0:
        return []
    if time_values is None:
        time_axis = np.arange(int(start_idx), int(start_idx) + raw.size, dtype=float)
    else:
        time_axis = np.asarray(list(time_values), dtype=float)
        if time_axis.size != raw.size:
            raise ValueError("time_values and y_data must have the same length.")

    smoothed = smooth_data(raw, window_size=config.window_size, poly_order=config.poly_order)
    corrected = correct_baseline(
        smoothed,
        method=config.baseline_method,
        window_size=config.window_size * config.baseline_window_factor,
        percentile=config.baseline_percentile,
    )
    if np.max(corrected) <= 0:
        return []

    cwt_matrix = calculate_cwt(corrected, config.wavelet_widths, wavelet=config.wavelet)
    max_cwt = np.max(cwt_matrix, axis=0)
    if np.max(max_cwt) <= 0:
        return []

    noise_level = estimate_noise(corrected)
    peaks, _ = find_peaks(
        max_cwt,
        height=config.snr_threshold * noise_level,
        prominence=config.prominence_ratio * float(np.max(max_cwt)),
        distance=config.min_peak_distance,
        width=(config.min_peak_width, config.max_peak_width),
    )
    if len(peaks) == 0:
        return []

    widths, _, left_ips, right_ips = peak_widths(max_cwt, peaks, rel_height=0.5)
    peak_results: list[Peak] = []
    sample_positions = np.arange(raw.size, dtype=float)

    for peak_idx, width, left_ip, right_ip in zip(peaks, widths, left_ips, right_ips):
        left_pos = max(0.0, float(left_ip))
        right_pos = min(float(raw.size - 1), float(right_ip))
        if right_pos - left_pos < config.min_peak_width or right_pos - left_pos > config.max_peak_width:
            continue

        fit = fit_gaussian(corrected, int(peak_idx), max(config.min_peak_width, int(width)))
        center_pos = fit.mean if fit is not None else float(peak_idx)
        center_time = float(np.interp(center_pos, sample_positions, time_axis) + time_offset)
        left_bound = int(round(np.interp(left_pos, sample_positions, time_axis)))
        right_bound = int(round(np.interp(right_pos, sample_positions, time_axis)))
        peak_results.append(
            Peak(
                index=int(round(center_time)),
                time=center_time,
                mz=float(calibration.tof_to_mz(center_time)),
                intensity=float(raw[int(peak_idx)]),
                fwhm=float(right_bound - left_bound),
                left_bound=left_bound,
                right_bound=right_bound,
                is_auto=True,
                gaussian_params=fit,
            )
        )

    return sorted(peak_results, key=lambda peak: peak.time)
