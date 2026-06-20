from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .calibration import Calibration
from .integration import integrate_peak
from .normalization import extract_light_intensity
from .peak_ranges import load_peak_ranges, peak_ranges_to_peaks
from .peak_detection import detect_peaks_in_range, detect_peaks_by_algorithm
from .spectrum_io import Spectrum, list_spectrum_files, read_spectrum


TEMPERATURE_CURVE_CLASS_LABELS = {
    "formation": "生成(升高)",
    "consumption": "消耗(减少)",
    "intermediate": "中间体(先升后降低)",
    "unclassified": "暂未区分",
}


def extract_temperature(metadata_lines: list[str], fallback: float) -> float:
    for line in metadata_lines:
        match = re.search(r"Temperature[:\s]*([\d.]+)", line, re.IGNORECASE)
        if match:
            return float(match.group(1))
    for line in metadata_lines:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", line)
        if match:
            return float(match.group(0))
    return fallback


def extract_photon_energy(metadata_lines: list[str], fallback: float | None = None) -> float:
    for line in metadata_lines:
        if "energy" in line.lower():
            match = re.search(r"[-+]?\d+(?:\.\d+)?", line)
            if match:
                return float(match.group(0))
    if fallback is not None:
        return float(fallback)
    raise ValueError("cannot extract photon energy")


def extract_io_current(metadata_lines: list[str], fallback: float = 1.0) -> float:
    return extract_light_intensity(metadata_lines, "io", fallback)


def analyze_temperature_folder(
    folder: str | Path,
    *,
    calibration: Calibration = Calibration(),
    algorithm: str = "legacy",
    threshold_end: float = 2,
    min_intensity: float = 3,
    prefer_gaussian: bool = True,
    reference_mode: str = "sum",
    detection_min_idx: int = 3000,
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
    manual_peak_path: str | Path | None = None,
    photon_normalize: bool = False,
    kr_correct: bool = False,
    kr_mz: int = 84,
    mass_discrimination: float = 1.0,
    light_source: str = "io",
    expansion_factors: dict[float, float] | None = None,
    vote_threshold: float = 0.667,
    min_intensity_for_single_vote: float = 5.0,
    mz_tolerance: float = 0.2,
    cwt_snr_threshold: float = 0.02,
    cwt_wavelet_max_width: int = 30,
    weak_tail_cutoff_idx: int = 15000,
) -> pd.DataFrame:
    files = list_spectrum_files(folder, (".txt",))
    spectra = []
    for idx, path in enumerate(files):
        spectrum = read_spectrum(path, header_lines=10, trim_start=0)
        temperature = extract_temperature(spectrum.metadata_lines, idx * 10.0)
        photon_energy = extract_photon_energy(spectrum.metadata_lines, fallback=0.0)
        io_current = extract_light_intensity(spectrum.metadata_lines, light_source)
        spectra.append((temperature, path, spectrum, io_current, photon_energy))
    if not spectra:
        return pd.DataFrame(
            columns=[
                "temperature",
                "file",
                "io",
                "photon_energy",
                "light_source",
                "reference_temperature",
                "reference_source",
                "mz",
                "mz_rounded",
                "species",
                "raw_area",
                "photon_normalized_area",
                "expansion_lambda",
                "normalized_area",
                "area",
                "left_bound",
                "right_bound",
                "curve_class",
                "curve_class_label",
                "curve_class_reason",
            ]
        )

    if manual_peak_path:
        reference_peaks = peak_ranges_to_peaks(load_peak_ranges(manual_peak_path, calibration=calibration))
        ref_temperature = max(item[0] for item in spectra)
        reference_source = Path(manual_peak_path).name
    else:
        ref_temperature, ref_spectrum, reference_source = _build_reference_spectrum(spectra, reference_mode)
        reference_peaks = detect_peaks_by_algorithm(
            ref_spectrum.y,
            algorithm=algorithm,
            calibration=calibration,
            start_idx=0,
            end_idx=len(ref_spectrum.y) - 1,
            detection_min_idx=detection_min_idx,
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
            prominence_ratio=prominence_ratio,
            smoothing_window=smoothing_window,
            smoothing_poly_order=smoothing_poly_order,
            baseline_window=baseline_window,
            baseline_percentile=baseline_percentile,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            vote_threshold=vote_threshold,
            min_intensity_for_single_vote=min_intensity_for_single_vote,
            mz_tolerance=mz_tolerance,
            cwt_snr_threshold=cwt_snr_threshold,
            cwt_wavelet_max_width=cwt_wavelet_max_width,
            weak_tail_cutoff_idx=weak_tail_cutoff_idx,
        )

    rows = []
    for temperature, path, spectrum, io_current, photon_energy in spectra:
        for peak in reference_peaks:
            raw_area = integrate_peak(spectrum.y, peak, prefer_gaussian=prefer_gaussian)
            photon_area = raw_area / io_current if photon_normalize and io_current > 0 else raw_area
            rows.append(
                {
                    "temperature": temperature,
                    "file": path.name,
                    "io": io_current,
                    "photon_energy": photon_energy,
                    "light_source": light_source,
                    "reference_temperature": ref_temperature,
                    "reference_source": reference_source,
                    "mz": peak.mz,
                    "mz_rounded": int(round(peak.mz)),
                    "species": peak.species,
                    "raw_area": raw_area,
                    "photon_normalized_area": photon_area,
                    "expansion_lambda": 1.0,
                    "normalized_area": photon_area,
                    "area": photon_area,
                    "left_bound": peak.left_bound,
                    "right_bound": peak.right_bound,
                }
            )
    result = pd.DataFrame(rows)
    normalized = _apply_temperature_normalization(
        result,
        kr_correct=kr_correct,
        kr_mz=kr_mz,
        mass_discrimination=mass_discrimination,
        expansion_factors=expansion_factors,
    )
    return annotate_temperature_curve_groups(normalized)


def _build_reference_spectrum(spectra: list[tuple[float, Path, Spectrum, float, float]], reference_mode: str) -> tuple[float, Spectrum, str]:
    if reference_mode == "max_temperature":
        ref_temperature, path, ref_spectrum, _, _ = max(spectra, key=lambda item: item[0])
        return ref_temperature, ref_spectrum, path.name

    if reference_mode != "sum":
        raise ValueError("reference_mode must be 'sum' or 'max_temperature'")

    min_len = min(len(item[2].y) for item in spectra)
    if min_len == 0:
        raise ValueError("spectrum files contain no numeric data")
    y_sum = np.sum([item[2].y[:min_len] for item in spectra], axis=0)
    ref_temperature = max(item[0] for item in spectra)
    return (
        ref_temperature,
        Spectrum(
            x=np.arange(1, min_len + 1, dtype=float),
            y=y_sum,
            metadata_lines=[],
            path=None,
        ),
        "sum",
    )


def _apply_temperature_normalization(
    result: pd.DataFrame,
    *,
    kr_correct: bool,
    kr_mz: int,
    mass_discrimination: float,
    expansion_factors: dict[float, float] | None = None,
) -> pd.DataFrame:
    if result.empty:
        return result
    normalized = result["photon_normalized_area"].astype(float).copy()
    result = result.copy()

    if kr_correct:
        if expansion_factors:
            result["expansion_lambda"] = _map_expansion_factors(result["temperature"], expansion_factors)
        else:
            kr_rows = result[result["mz_rounded"].astype(int) == int(kr_mz)]
            if kr_rows.empty:
                raise ValueError(f"Kr correction requested but m/z {kr_mz} is not present in peak ranges")
            kr_by_temperature = kr_rows.groupby("temperature")["photon_normalized_area"].sum().sort_index()
            positive_kr = kr_by_temperature[kr_by_temperature > 0]
            if positive_kr.empty:
                raise ValueError("Kr correction requested but all Kr signals are zero")
            t0 = float(positive_kr.index.min())
            kr_ref = float(kr_by_temperature.loc[t0])
            if kr_ref <= 0:
                raise ValueError("Kr reference signal is zero; cannot compute expansion correction")
            lambda_by_temperature = kr_by_temperature / kr_ref
            result["expansion_lambda"] = result["temperature"].map(lambda_by_temperature).fillna(1.0).astype(float)
        lambda_values = result["expansion_lambda"].replace(0, np.nan)
        normalized = normalized / lambda_values
    else:
        result["expansion_lambda"] = 1.0

    denominator = float(mass_discrimination)
    if denominator <= 0:
        raise ValueError("mass_discrimination must be positive")
    result["normalized_area"] = (normalized / denominator).fillna(0.0)
    result["area"] = result["normalized_area"]
    return result


def _map_expansion_factors(temperatures: pd.Series, expansion_factors: dict[float, float]) -> pd.Series:
    factor_temperatures = np.array(sorted(float(key) for key in expansion_factors), dtype=float)
    factor_values = np.array([float(expansion_factors[temperature]) for temperature in factor_temperatures], dtype=float)
    mapped = []
    for temperature in temperatures.astype(float):
        nearest = int(np.argmin(np.abs(factor_temperatures - temperature)))
        mapped.append(factor_values[nearest])
    return pd.Series(mapped, index=temperatures.index, dtype=float)


def classify_temperature_curve(
    temperatures,
    areas,
    *,
    min_points: int = 3,
    relative_change_threshold: float = 0.25,
    endpoint_peak_fraction: float = 0.65,
) -> dict:
    """Classify temperature curves by their coarse thermal trend."""
    x_values = np.asarray(temperatures, dtype=float)
    y_values = np.asarray(areas, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = np.maximum(y_values[valid], 0.0)
    if x_values.size < min_points:
        return _curve_class_result("unclassified", "有效温度点不足")

    order = np.argsort(x_values)
    y_values = y_values[order]
    if y_values.size >= 3:
        y_smooth = (
            pd.Series(y_values)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
    else:
        y_smooth = y_values

    peak = float(np.max(y_values))
    trough = float(np.min(y_values))
    if peak <= 0:
        return _curve_class_result("unclassified", "曲线无正信号")
    dynamic_range = peak - trough
    if dynamic_range <= max(1e-12, peak * 0.05):
        return _curve_class_result("unclassified", "曲线变化幅度过小")

    edge_count = min(2, y_smooth.size)
    low_temp_signal = float(np.median(y_smooth[:edge_count]))
    high_temp_signal = float(np.median(y_smooth[-edge_count:]))
    peak_index = int(np.argmax(y_values))
    normalized_delta = (high_temp_signal - low_temp_signal) / peak
    first_is_peak_like = low_temp_signal >= endpoint_peak_fraction * peak
    last_is_peak_like = high_temp_signal >= endpoint_peak_fraction * peak
    peak_is_internal = 0 < peak_index < y_smooth.size - 1

    if (
        peak_is_internal
        and (peak - low_temp_signal) / peak >= relative_change_threshold
        and (peak - high_temp_signal) / peak >= relative_change_threshold
    ):
        return _curve_class_result("intermediate", "中间温度信号最高且两端明显降低")
    if normalized_delta >= relative_change_threshold and last_is_peak_like:
        return _curve_class_result("formation", "高温端信号明显高于低温端")
    if normalized_delta <= -relative_change_threshold and first_is_peak_like:
        return _curve_class_result("consumption", "低温端信号明显高于高温端")
    return _curve_class_result("unclassified", "趋势不满足升高、降低或中间体规则")


def _curve_class_result(curve_class: str, reason: str) -> dict:
    return {
        "curve_class": curve_class,
        "curve_class_label": TEMPERATURE_CURVE_CLASS_LABELS[curve_class],
        "curve_class_reason": reason,
    }


def annotate_temperature_curve_groups(result_df: pd.DataFrame) -> pd.DataFrame:
    """Add curve classification columns to temperature scan rows."""
    result = result_df.copy()
    if result.empty:
        for column in ("curve_class", "curve_class_label", "curve_class_reason"):
            if column not in result:
                result[column] = []
        return result

    curves = build_temperature_curves(result)
    class_by_mz = {
        mz: (
            curve["curve_class"],
            curve["curve_class_label"],
            curve["curve_class_reason"],
        )
        for mz, curve in curves.items()
    }
    mz_rounded = result["mz"].round().astype(int)
    result["curve_class"] = mz_rounded.map(lambda mz: class_by_mz.get(int(mz), ("unclassified", "", ""))[0])
    result["curve_class_label"] = mz_rounded.map(
        lambda mz: class_by_mz.get(int(mz), ("unclassified", TEMPERATURE_CURVE_CLASS_LABELS["unclassified"], ""))[1]
    )
    result["curve_class_reason"] = mz_rounded.map(
        lambda mz: class_by_mz.get(int(mz), ("unclassified", "", ""))[2]
    )
    return result


def compute_kr_expansion_factors(
    folder: str | Path,
    *,
    calibration: Calibration = Calibration(),
    kr_mz: int = 84,
    manual_peak_path: str | Path | None = None,
    light_source: str = "io",
    threshold_end: float = 2,
    min_intensity: float = 3,
    prefer_gaussian: bool = True,
    reference_mode: str = "sum",
    detection_min_idx: int = 3000,
    nearby_peak_window: int = 30,
    duplicate_window: int = 20,
    weak_tail_early_window: int = 90,
    weak_tail_late_window: int = 50,
    weak_tail_ratio: float = 5,
    gaussian_window_max: int = 30,
    gaussian_boundary_scale: float = 1.5,
    boundary_padding: int = 2,
) -> pd.DataFrame:
    result = analyze_temperature_folder(
        folder,
        calibration=calibration,
        threshold_end=threshold_end,
        min_intensity=min_intensity,
        prefer_gaussian=prefer_gaussian,
        reference_mode=reference_mode,
        detection_min_idx=detection_min_idx,
        nearby_peak_window=nearby_peak_window,
        duplicate_window=duplicate_window,
        weak_tail_early_window=weak_tail_early_window,
        weak_tail_late_window=weak_tail_late_window,
        weak_tail_ratio=weak_tail_ratio,
        gaussian_window_max=gaussian_window_max,
        gaussian_boundary_scale=gaussian_boundary_scale,
        boundary_padding=boundary_padding,
        manual_peak_path=manual_peak_path,
        photon_normalize=True,
        kr_correct=False,
        kr_mz=kr_mz,
        light_source=light_source,
    )
    kr_rows = result[result["mz_rounded"].astype(int) == int(kr_mz)]
    if kr_rows.empty:
        raise ValueError(f"Kr m/z {kr_mz} was not found in the calibration folder")
    kr_by_temperature = kr_rows.groupby("temperature")["photon_normalized_area"].sum().sort_index()
    positive_kr = kr_by_temperature[kr_by_temperature > 0]
    if positive_kr.empty:
        raise ValueError("all Kr calibration signals are zero")
    t0 = float(positive_kr.index.min())
    kr_ref = float(kr_by_temperature.loc[t0])
    if kr_ref <= 0:
        raise ValueError("Kr reference signal is zero; cannot compute expansion correction")
    lambda_by_temperature = kr_by_temperature / kr_ref
    return pd.DataFrame(
        {
            "temperature": kr_by_temperature.index.astype(float),
            "kr_signal": kr_by_temperature.astype(float).values,
            "expansion_lambda": lambda_by_temperature.astype(float).values,
            "reference_temperature": t0,
        }
    )


def build_temperature_curves(result_df: pd.DataFrame) -> dict[int, dict]:
    """Convert temperature scan rows into m/z keyed curve objects."""
    curves: dict[int, dict] = {}
    if result_df.empty:
        return curves

    working = result_df.copy()
    working["mz_rounded"] = working["mz"].round().astype(int)
    if "raw_area" not in working:
        working["raw_area"] = working["area"]
    if "photon_normalized_area" not in working:
        working["photon_normalized_area"] = working["area"]
    if "expansion_lambda" not in working:
        working["expansion_lambda"] = 1.0
    if "species" not in working:
        working["species"] = ""
    if "photon_energy" not in working:
        working["photon_energy"] = np.nan
    for mz, group in working.groupby("mz_rounded"):
        ordered = (
            group.groupby("temperature", as_index=False)
            .agg(
                area=("area", "sum"),
                raw_area=("raw_area", "sum"),
                photon_normalized_area=("photon_normalized_area", "sum"),
                expansion_lambda=("expansion_lambda", "first"),
                species=("species", "first"),
                photon_energy=("photon_energy", "first"),
                mz=("mz", "mean"),
                file_count=("file", "nunique"),
                reference_temperature=("reference_temperature", "first"),
            )
            .sort_values("temperature")
        )
        classification = classify_temperature_curve(ordered["temperature"], ordered["area"])
        curves[int(mz)] = {
            "mz": int(mz),
            "mz_exact_mean": float(ordered["mz"].mean()),
            "species": str(ordered["species"].iloc[0]) if "species" in ordered else "",
            "temperatures": ordered["temperature"].astype(float).tolist(),
            "areas": ordered["area"].astype(float).tolist(),
            "curve_class": classification["curve_class"],
            "curve_class_label": classification["curve_class_label"],
            "curve_class_reason": classification["curve_class_reason"],
            "rows": ordered,
        }
    return curves
