"""Scale-invariant, high-recall candidate peak detection.

The detector is intentionally simple: normalize the signal unit, then combine
the validated legacy and prominence detectors. CWT is excluded because it did
not add any annotated peaks on the 11 eV validation dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .calibration import Calibration
from .config import PeakDetectionConfig
from .peak_detection import Peak, detect_peaks_ensemble


@dataclass(frozen=True)
class AdaptivePeakDetectionConfig:
    normalization_quantile: float = 25.0
    minimum_scale: float = 1e-12


def estimate_signal_scale(
    y_data: Iterable[float],
    *,
    quantile: float = 25.0,
    minimum_scale: float = 1e-12,
) -> float:
    """Estimate a homogeneous low-signal unit.

    A positive lower quantile preserves weak peaks and scales linearly when the
    whole spectrum is multiplied. For integer-count spectra it normally equals
    one count; for an averaged spectrum it recovers the corresponding fraction
    of a count.
    """
    data = np.asarray(y_data, dtype=float)
    finite_positive = data[np.isfinite(data) & (data > 0)]
    if finite_positive.size == 0:
        return 1.0
    quantile = max(0.0, min(float(quantile), 100.0))
    scale = float(np.percentile(finite_positive, quantile))
    if not np.isfinite(scale) or scale <= minimum_scale:
        scale = float(np.min(finite_positive))
    return max(scale, float(minimum_scale))


def normalize_signal_scale(
    y_data: Iterable[float],
    *,
    quantile: float = 25.0,
    minimum_scale: float = 1e-12,
) -> tuple[np.ndarray, float]:
    data = np.asarray(y_data, dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    scale = estimate_signal_scale(data, quantile=quantile, minimum_scale=minimum_scale)
    return data / scale, scale


def _ensemble_kwargs(config: PeakDetectionConfig) -> dict[str, float | int | bool]:
    return {
        "detection_min_idx": config.detection_min_idx,
        "threshold_end": config.threshold_end,
        "min_intensity": config.min_intensity,
        "nearby_peak_window": config.nearby_peak_window,
        "duplicate_window": config.duplicate_window,
        "weak_tail_early_window": config.weak_tail_early_window,
        "weak_tail_late_window": config.weak_tail_late_window,
        "weak_tail_ratio": config.weak_tail_ratio,
        "weak_tail_cutoff_idx": config.weak_tail_cutoff_idx,
        "gaussian_window_max": config.gaussian_window_max,
        "gaussian_boundary_scale": config.gaussian_boundary_scale,
        "boundary_padding": config.boundary_padding,
        "prominence_ratio": config.prominence_ratio,
        "smoothing_window": config.smoothing_window,
        "smoothing_poly_order": config.smoothing_poly_order,
        "baseline_window": config.baseline_window,
        "baseline_percentile": config.baseline_percentile,
        "min_peak_width": config.min_peak_width,
        "max_peak_width": config.max_peak_width,
        "vote_threshold": 1.0,
        "min_intensity_for_single_vote": config.min_intensity_for_single_vote,
        "mz_tolerance": config.mz_tolerance,
        "use_legacy": True,
        "use_prominence": True,
        "use_cwt": False,
    }


def detect_peaks_adaptive(
    y_data: Iterable[float],
    *,
    calibration: Calibration = Calibration(),
    peak_config: PeakDetectionConfig | None = None,
    adaptive_config: AdaptivePeakDetectionConfig | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
    time_offset: float = 0.0,
) -> list[Peak]:
    """Return a scale-invariant, high-recall candidate peak list."""
    peak_config = peak_config or PeakDetectionConfig()
    adaptive_config = adaptive_config or AdaptivePeakDetectionConfig()
    normalized, scale = normalize_signal_scale(
        y_data,
        quantile=adaptive_config.normalization_quantile,
        minimum_scale=adaptive_config.minimum_scale,
    )
    peaks = detect_peaks_ensemble(
        normalized,
        calibration=calibration,
        start_idx=start_idx,
        end_idx=end_idx,
        time_offset=time_offset,
        **_ensemble_kwargs(peak_config),
    )
    # The detector ran in normalized units; the workbench should still display
    # intensities and Gaussian amplitudes in the source spectrum's units.
    for peak in peaks:
        peak.intensity *= scale
        if peak.gaussian_params is not None:
            peak.gaussian_params.amplitude *= scale
            peak.gaussian_params.baseline *= scale
    return peaks
