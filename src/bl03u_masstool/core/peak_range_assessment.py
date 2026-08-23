from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Iterable

import numpy as np

from .calibration import Calibration
from .peak_ranges import PeakRange


class PeakRangeFitStatus(StrEnum):
    """Suitability of one imported integration window for a spectrum."""

    FIT = "fit"
    REVIEW = "review"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PeakRangeAssessment:
    peak_range: PeakRange
    status: PeakRangeFitStatus
    reasons: tuple[str, ...]
    actual_peak_x: float | None = None
    actual_mz: float | None = None
    peak_intensity: float | None = None
    center_offset_fraction: float | None = None
    edge_ratio: float | None = None
    snr: float | None = None
    mz_error: float | None = None


_STATUS_PRIORITY = {
    PeakRangeFitStatus.FIT: 0,
    PeakRangeFitStatus.REVIEW: 1,
    PeakRangeFitStatus.UNAVAILABLE: 2,
    PeakRangeFitStatus.MISMATCH: 3,
}


def assess_peak_ranges(
    ranges: Iterable[PeakRange],
    spectrum_x: Iterable[float],
    spectrum_y: Iterable[float],
    *,
    calibration: Calibration = Calibration(),
) -> list[PeakRangeAssessment]:
    """Assess imported peak windows against the currently displayed spectrum.

    ``PeakRange`` bounds use the full-spectrum TOF/sample coordinate, whereas a
    workbench spectrum may have its leading samples trimmed.  Matching through
    ``spectrum_x`` keeps those two coordinate systems aligned.
    """

    range_list = list(ranges)
    x_values = np.asarray(list(spectrum_x), dtype=float)
    y_values = np.asarray(list(spectrum_y), dtype=float)
    point_count = min(x_values.size, y_values.size)
    x_values = x_values[:point_count]
    y_values = y_values[:point_count]

    if point_count == 0:
        return [
            PeakRangeAssessment(
                peak_range=item,
                status=PeakRangeFitStatus.UNAVAILABLE,
                reasons=("当前谱图没有可评价的数据",),
            )
            for item in range_list
        ]

    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite):
        return [
            PeakRangeAssessment(
                peak_range=item,
                status=PeakRangeFitStatus.UNAVAILABLE,
                reasons=("当前谱图没有有效数值",),
            )
            for item in range_list
        ]

    x_values = x_values[finite]
    y_values = y_values[finite]
    order = np.argsort(x_values, kind="stable")
    x_values = x_values[order]
    y_values = y_values[order]

    assessments = [
        _assess_one(item, x_values, y_values, calibration=calibration)
        for item in range_list
    ]
    return _mark_overlapping_ranges(assessments)


def _assess_one(
    item: PeakRange,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    calibration: Calibration,
) -> PeakRangeAssessment:
    left = float(min(item.left_bound, item.right_bound))
    right = float(max(item.left_bound, item.right_bound))
    width = max(right - left, 1.0)
    data_left = float(x_values[0])
    data_right = float(x_values[-1])

    if right < data_left or left > data_right:
        return PeakRangeAssessment(
            peak_range=item,
            status=PeakRangeFitStatus.UNAVAILABLE,
            reasons=(f"范围超出当前谱图（{data_left:g}–{data_right:g}）",),
        )

    window_mask = (x_values >= left) & (x_values <= right)
    window_indices = np.flatnonzero(window_mask)
    if window_indices.size < 3:
        return PeakRangeAssessment(
            peak_range=item,
            status=PeakRangeFitStatus.UNAVAILABLE,
            reasons=("范围内数据点不足，无法评价",),
        )

    reasons: list[str] = []
    status = PeakRangeFitStatus.FIT
    if left < data_left or right > data_right:
        status = PeakRangeFitStatus.MISMATCH
        reasons.append("范围被当前谱图边界截断")

    first = int(window_indices[0])
    last = int(window_indices[-1])
    segment_y = y_values[first : last + 1]
    local_offset = int(np.argmax(segment_y))
    peak_pos = first + local_offset
    actual_peak_x = float(x_values[peak_pos])
    peak_intensity = float(y_values[peak_pos])

    context_left = max(data_left, left - width)
    context_right = min(data_right, right + width)
    context_mask = (x_values >= context_left) & (x_values <= context_right)
    context_x = x_values[context_mask]
    context_y = y_values[context_mask]
    baseline = _robust_baseline(context_y)
    signal = peak_intensity - baseline
    noise = _noise_sigma(context_y)
    snr = float(signal / noise) if noise > 0 else (float("inf") if signal > 0 else 0.0)

    center_offset = abs(actual_peak_x - float(item.peak_index)) / width
    edge_count = max(1, min(5, segment_y.size // 5 or 1))
    edge_level = max(
        float(np.median(segment_y[:edge_count])),
        float(np.median(segment_y[-edge_count:])),
    )
    edge_ratio = float(max(0.0, edge_level - baseline) / signal) if signal > 0 else float("inf")

    actual_mz = float(calibration.tof_to_mz(actual_peak_x))
    mz_error = abs(actual_mz - float(item.mz))
    mz_span = abs(
        float(calibration.tof_to_mz(right))
        - float(calibration.tof_to_mz(left))
    )
    mz_tolerance = max(0.05, mz_span * 0.35)

    if context_y.size:
        context_peak_pos = int(np.argmax(context_y))
        context_peak_x = float(context_x[context_peak_pos])
        context_peak_signal = float(context_y[context_peak_pos] - baseline)
        if (
            not left <= context_peak_x <= right
            and context_peak_signal > max(signal * 1.15, noise * 5.0)
        ):
            status = _worse(status, PeakRangeFitStatus.MISMATCH)
            reasons.append("相邻更强峰位于卡峰范围之外")

    if signal <= 0 or snr < 2.0:
        status = _worse(status, PeakRangeFitStatus.MISMATCH)
        reasons.append("范围内没有可确认的峰信号")
    elif snr < 5.0:
        status = _worse(status, PeakRangeFitStatus.REVIEW)
        reasons.append(f"峰信噪比较低（S/N≈{snr:.1f}）")

    if center_offset > 0.45:
        status = _worse(status, PeakRangeFitStatus.MISMATCH)
        reasons.append(f"实际峰顶偏离文件峰位 {center_offset:.0%}")
    elif center_offset > 0.20:
        status = _worse(status, PeakRangeFitStatus.REVIEW)
        reasons.append(f"实际峰顶偏离文件峰位 {center_offset:.0%}")

    if edge_ratio > 0.65:
        status = _worse(status, PeakRangeFitStatus.MISMATCH)
        reasons.append(f"范围边界仍处于峰高的 {edge_ratio:.0%}")
    elif edge_ratio > 0.35:
        status = _worse(status, PeakRangeFitStatus.REVIEW)
        reasons.append(f"范围边界仍处于峰高的 {edge_ratio:.0%}")

    if mz_error > mz_tolerance * 3.0:
        status = _worse(status, PeakRangeFitStatus.MISMATCH)
        reasons.append(f"当前定标下 m/z 偏差 {mz_error:.4g}")
    elif mz_error > mz_tolerance:
        status = _worse(status, PeakRangeFitStatus.REVIEW)
        reasons.append(f"当前定标下 m/z 偏差 {mz_error:.4g}")

    if not reasons:
        reasons.append("峰顶、边界和当前定标均匹配")

    return PeakRangeAssessment(
        peak_range=item,
        status=status,
        reasons=tuple(reasons),
        actual_peak_x=actual_peak_x,
        actual_mz=actual_mz,
        peak_intensity=peak_intensity,
        center_offset_fraction=center_offset,
        edge_ratio=edge_ratio,
        snr=snr,
        mz_error=mz_error,
    )


def _robust_baseline(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    ordered = np.sort(values)
    count = max(1, int(np.ceil(ordered.size * 0.3)))
    return float(np.median(ordered[:count]))


def _noise_sigma(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    differences = np.diff(values)
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    return max(0.0, 1.4826 * mad / np.sqrt(2.0))


def _worse(
    current: PeakRangeFitStatus,
    candidate: PeakRangeFitStatus,
) -> PeakRangeFitStatus:
    return candidate if _STATUS_PRIORITY[candidate] > _STATUS_PRIORITY[current] else current


def _mark_overlapping_ranges(
    assessments: list[PeakRangeAssessment],
) -> list[PeakRangeAssessment]:
    if len(assessments) < 2:
        return assessments
    updated = list(assessments)
    ordered = sorted(
        enumerate(assessments),
        key=lambda entry: (
            entry[1].peak_range.left_bound,
            entry[1].peak_range.right_bound,
        ),
    )
    overlapping_indices: set[int] = set()
    for ordered_index, (original_index, assessment) in enumerate(ordered[:-1]):
        current = assessment.peak_range
        for next_original_index, next_assessment in ordered[ordered_index + 1 :]:
            following = next_assessment.peak_range
            if following.left_bound >= current.right_bound:
                break
            overlap = min(current.right_bound, following.right_bound) - max(
                current.left_bound, following.left_bound
            )
            if overlap <= 0:
                continue
            minimum_width = max(
                1,
                min(
                    current.right_bound - current.left_bound,
                    following.right_bound - following.left_bound,
                ),
            )
            if overlap / minimum_width >= 0.10:
                overlapping_indices.update((original_index, next_original_index))

    for index in overlapping_indices:
        item = updated[index]
        updated[index] = replace(
            item,
            status=_worse(item.status, PeakRangeFitStatus.REVIEW),
            reasons=item.reasons + ("与相邻卡峰范围重叠",),
        )
    return updated
