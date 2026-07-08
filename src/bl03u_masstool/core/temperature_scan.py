from __future__ import annotations

import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .calibration import Calibration
from .integration import integrate_peak_with_method
from .normalization import extract_light_intensity
from .peak_ranges import load_peak_ranges, peak_ranges_to_peaks
from .peak_detection import detect_peaks_by_algorithm
from .spectrum_io import Spectrum, find_filename_replicate_groups, filename_replicate_key, list_spectrum_files, read_spectrum


logger = logging.getLogger(__name__)


TEMPERATURE_CURVE_CLASS_LABELS = {
    "formation": "生成(升高)",
    "consumption": "消耗(减少)",
    "intermediate": "中间体(先升后降低)",
    "unclassified": "暂未区分",
}


def group_energies_by_tolerance(energies: list[float], tolerance: float = 0.01) -> dict[float, list[float]]:
    """按误差容忍度将能量值分组。

    Args:
        energies: 所有能量值列表
        tolerance: 误差容忍度（eV）

    Returns:
        {代表能量: [该组的所有能量值]}
        代表能量为该组的平均值，按升序排列
    """
    if not energies:
        return {}

    sorted_energies = sorted(set(energies))  # 去重并排序
    groups = {}

    for energy in sorted_energies:
        # 检查是否与已有的组匹配
        matched = False
        for group_key in groups:
            if abs(energy - group_key) <= tolerance:
                groups[group_key].append(energy)
                matched = True
                break

        # 如果没有匹配的组，创建新组
        if not matched:
            groups[energy] = [energy]

    # 重新计算每组的平均能量作为key
    result = {}
    for energies_in_group in groups.values():
        avg_energy = float(np.mean(energies_in_group))
        result[avg_energy] = energies_in_group

    return result


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


def _summarize_integration_methods(values) -> str:
    methods = sorted({str(value) for value in values if str(value)})
    if not methods:
        return ""
    if len(methods) == 1:
        return methods[0]
    return "mixed"


def _normalize_integration_method(value: str | None, *, prefer_gaussian: bool | None = None) -> str:
    if prefer_gaussian:
        return "gaussian"
    method = str(value or "").strip()
    if method in {"sum_counts", "baseline", "gaussian"}:
        return method
    return "sum_counts"


def analyze_temperature_folder(
    folder: str | Path,
    *,
    calibration: Calibration = Calibration(),
    algorithm: str = "legacy",
    threshold_end: float = 2,
    min_intensity: float = 3,
    prefer_gaussian: bool = True,
    integration_method: str = "sum_counts",
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
    temp_curve_class_change_threshold: float = 0.25,
    temp_curve_class_peak_fraction: float = 0.65,
    replicate_mode: str = "off",
) -> pd.DataFrame:
    files = _iter_temperature_files(folder, (".txt",))
    replicate_mode = "sum" if replicate_mode == "sum" else "mean" if replicate_mode == "mean" else "off"
    filename_replicates = find_filename_replicate_groups(files) if replicate_mode != "off" else {}
    spectra = []
    for idx, path in enumerate(files):
        spectrum = read_spectrum(path, header_lines=10, trim_start=0)
        temperature = extract_temperature(spectrum.metadata_lines, idx * 10.0)
        photon_energy = extract_photon_energy(spectrum.metadata_lines, fallback=0.0)
        io_current = extract_light_intensity(spectrum.metadata_lines, light_source)
        repeat_key = filename_replicate_key(path)
        uses_filename_grouping = repeat_key is not None and repeat_key in filename_replicates
        spectra.append((temperature, path, spectrum, io_current, photon_energy, uses_filename_grouping))
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
                "integration_method",
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

    configured_integration_method = _normalize_integration_method(
        integration_method,
        prefer_gaussian=prefer_gaussian,
    )
    rows = []
    for temperature, path, spectrum, io_current, photon_energy, uses_filename_grouping in spectra:
        for peak in reference_peaks:
            raw_area, actual_integration_method = integrate_peak_with_method(
                spectrum.y,
                peak,
                prefer_gaussian=prefer_gaussian,
                integration_method=configured_integration_method,
            )
            photon_area = raw_area / io_current if photon_normalize and io_current > 0 else raw_area
            rows.append(
                {
                    "temperature": temperature,
                    "file": path.name,
                    "io": io_current,
                    "photon_energy": photon_energy,
                    "light_source": light_source,
                    "replicate_mode": replicate_mode,
                    "replicate_grouping": "filename" if uses_filename_grouping else "temperature",
                    "replicate_warning": "",
                    "reference_temperature": ref_temperature,
                    "reference_source": reference_source,
                    "mz": peak.mz,
                    "mz_rounded": int(round(peak.mz)),
                    "species": peak.species,
                    "raw_area": raw_area,
                    "integration_method": actual_integration_method,
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
    if not normalized.empty:
        normalized["file_count"] = normalized.groupby("temperature")["file"].transform("nunique")
        fallback_mask = (
            (normalized["replicate_mode"].astype(str) != "off")
            &
            (normalized["file_count"] > 1)
            & (normalized["replicate_grouping"].astype(str) != "filename")
        )
        normalized.loc[fallback_mask, "replicate_warning"] = (
            "未识别到文件名末尾采集序号，已退回按温度分组的旧逻辑处理重复文件。"
        )
    return annotate_temperature_curve_groups(
        normalized,
        relative_change_threshold=temp_curve_class_change_threshold,
        endpoint_peak_fraction=temp_curve_class_peak_fraction,
    )


def _iter_temperature_files(folder: str | Path, suffixes: tuple[str, ...]) -> list[Path]:
    direct_files = list_spectrum_files(folder, suffixes)
    if direct_files:
        return direct_files

    root = Path(folder)
    nested_files: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        nested_files.extend(
            path
            for path in sorted(child.iterdir())
            if path.is_file() and path.suffix.lower() in suffixes
        )
    return sorted(nested_files)


def _build_reference_spectrum(spectra: list[tuple], reference_mode: str) -> tuple[float, Spectrum, str]:
    if reference_mode == "max_temperature":
        ref_temperature, path, ref_spectrum, *_ = max(spectra, key=lambda item: item[0])
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
    expansion_factors: dict | None = None,
) -> pd.DataFrame:
    if result.empty:
        return result
    normalized = result["photon_normalized_area"].astype(float).copy()
    result = result.copy()

    if kr_correct:
        if expansion_factors:
            energies = result["photon_energy"] if "photon_energy" in result.columns else None
            result["expansion_lambda"] = _map_expansion_factors(
                result["temperature"],
                expansion_factors,
                energies=energies,
            )
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


def _map_expansion_factors(
    temperatures: pd.Series,
    expansion_factors: dict,
    *,
    energies: pd.Series | None = None,
) -> pd.Series:
    factors = _coerce_expansion_factor_map(expansion_factors)
    if not factors:
        return pd.Series([1.0] * len(temperatures), index=temperatures.index, dtype=float)

    multi_energy_factors = {
        float(key): value
        for key, value in factors.items()
        if isinstance(value, dict)
    }
    if multi_energy_factors:
        energy_keys = np.array(sorted(float(key) for key in multi_energy_factors), dtype=float)
        energy_values = (
            pd.Series([np.nan] * len(temperatures), index=temperatures.index, dtype=float)
            if energies is None
            else pd.to_numeric(energies, errors="coerce")
        )
        mapped = []
        for index, temperature in temperatures.astype(float).items():
            energy = (
                float(energy_values.loc[index])
                if index in energy_values.index and pd.notna(energy_values.loc[index])
                else np.nan
            )
            if np.isfinite(energy):
                nearest_energy = float(energy_keys[int(np.argmin(np.abs(energy_keys - energy)))])
            else:
                nearest_energy = float(energy_keys[0])
            mapped.append(
                _nearest_temperature_factor(
                    float(temperature),
                    multi_energy_factors.get(nearest_energy, {}),
                )
            )
        return pd.Series(mapped, index=temperatures.index, dtype=float)

    return pd.Series(
        [_nearest_temperature_factor(float(temperature), factors) for temperature in temperatures.astype(float)],
        index=temperatures.index,
        dtype=float,
    )


def _coerce_expansion_factor_map(expansion_factors: dict) -> dict:
    coerced: dict[float, float | dict[float, float]] = {}
    for key, value in (expansion_factors or {}).items():
        float_key = float(key)
        if isinstance(value, dict):
            nested = {float(temp): float(lam) for temp, lam in value.items()}
            if nested:
                coerced[float_key] = nested
        else:
            coerced[float_key] = float(value)
    return coerced


def _nearest_temperature_factor(temperature: float, factors: dict[float, float]) -> float:
    if not factors:
        return 1.0
    factor_temperatures = np.array(sorted(float(key) for key in factors), dtype=float)
    factor_values = np.array([float(factors[temperature]) for temperature in factor_temperatures], dtype=float)
    nearest = int(np.argmin(np.abs(factor_temperatures - float(temperature))))
    return float(factor_values[nearest])


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


def annotate_temperature_curve_groups(
    result_df: pd.DataFrame,
    *,
    relative_change_threshold: float = 0.25,
    endpoint_peak_fraction: float = 0.65,
) -> pd.DataFrame:
    """Add curve classification columns to temperature scan rows."""
    result = result_df.copy()
    if result.empty:
        for column in ("curve_class", "curve_class_label", "curve_class_reason"):
            if column not in result:
                result[column] = []
        return result

    curves = build_temperature_curves(
        result,
        relative_change_threshold=relative_change_threshold,
        endpoint_peak_fraction=endpoint_peak_fraction,
    )
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
    integration_method: str = "sum_counts",
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
    use_highest_energy: bool = False,
) -> pd.DataFrame:
    """计算Kr膨胀系数，自动处理多能量温度扫描数据。

    Args:
        use_highest_energy: 是否只返回最高能量组数据。默认 False，会按能量分别计算 λ(T)。
    """
    result = analyze_temperature_folder(
        folder,
        calibration=calibration,
        threshold_end=threshold_end,
        min_intensity=min_intensity,
        prefer_gaussian=prefer_gaussian,
        integration_method=integration_method,
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
    kr_rows = result[result["mz_rounded"].astype(int) == int(kr_mz)].copy()
    if kr_rows.empty:
        raise ValueError(f"Kr m/z {kr_mz} was not found in the calibration folder")

    return _build_kr_expansion_factor_table(kr_rows, use_highest_energy=use_highest_energy)


def _build_kr_expansion_factor_table(kr_rows: pd.DataFrame, *, use_highest_energy: bool = False) -> pd.DataFrame:
    kr_rows = kr_rows.copy()
    kr_rows["temperature"] = pd.to_numeric(kr_rows["temperature"], errors="coerce")
    kr_rows["photon_normalized_area"] = pd.to_numeric(kr_rows["photon_normalized_area"], errors="coerce")
    kr_rows = kr_rows.dropna(subset=["temperature", "photon_normalized_area"])

    if kr_rows.empty:
        raise ValueError("all Kr calibration signals are zero")

    if "photon_energy" in kr_rows.columns:
        energy_values = pd.to_numeric(kr_rows["photon_energy"], errors="coerce")
        valid_energy_values = sorted(set(float(value) for value in energy_values.dropna() if float(value) > 0))
    else:
        valid_energy_values = []

    if valid_energy_values:
        energy_groups = group_energies_by_tolerance(valid_energy_values, tolerance=0.01)
        kr_rows["photon_energy"] = [
            _nearest_energy_group(float(value), energy_groups)
            if pd.notna(value) and float(value) > 0
            else np.nan
            for value in pd.to_numeric(kr_rows["photon_energy"], errors="coerce")
        ]
        kr_rows = kr_rows.dropna(subset=["photon_energy"])
        if kr_rows.empty:
            raise ValueError("all Kr calibration signals are zero")
        if use_highest_energy:
            kr_rows = kr_rows[kr_rows["photon_energy"] == float(max(energy_groups.keys()))]

        grouped = (
            kr_rows.groupby(["photon_energy", "temperature"], as_index=False)["photon_normalized_area"]
            .sum()
            .sort_values(["photon_energy", "temperature"])
        )
        rows = []
        for energy, energy_group in grouped.groupby("photon_energy", sort=True):
            kr_by_temperature = energy_group.set_index("temperature")["photon_normalized_area"].sort_index()
            rows.extend(_kr_expansion_rows_for_signal_series(kr_by_temperature, photon_energy=float(energy)))
        return pd.DataFrame(rows)

    kr_by_temperature = kr_rows.groupby("temperature")["photon_normalized_area"].sum().sort_index()
    return pd.DataFrame(_kr_expansion_rows_for_signal_series(kr_by_temperature))


def _nearest_energy_group(energy: float, energy_groups: dict[float, list[float]]) -> float:
    energy_centers = np.array(sorted(float(key) for key in energy_groups), dtype=float)
    return float(energy_centers[int(np.argmin(np.abs(energy_centers - float(energy))))])


def _kr_expansion_rows_for_signal_series(
    kr_by_temperature: pd.Series,
    *,
    photon_energy: float | None = None,
) -> list[dict]:
    kr_by_temperature = kr_by_temperature.astype(float).sort_index()
    positive_kr = kr_by_temperature[kr_by_temperature > 0]
    if positive_kr.empty:
        raise ValueError("all Kr calibration signals are zero")

    reference_temperature = float(positive_kr.index.min())
    reference_signal = float(kr_by_temperature.loc[reference_temperature])
    if reference_signal <= 0:
        raise ValueError("Kr reference signal is zero; cannot compute expansion correction")

    rows = []
    for temperature, signal in kr_by_temperature.items():
        row = {
            "temperature": float(temperature),
            "kr_signal": float(signal),
            "expansion_lambda": float(signal) / reference_signal,
            "reference_temperature": reference_temperature,
        }
        if photon_energy is not None:
            row["photon_energy"] = float(photon_energy)
            row["reference_energy"] = float(photon_energy)
        rows.append(row)
    return rows



def build_temperature_curves(
    result_df: pd.DataFrame,
    *,
    relative_change_threshold: float = 0.25,
    endpoint_peak_fraction: float = 0.65,
) -> dict[int, dict]:
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
    if "integration_method" not in working:
        working["integration_method"] = ""
    if "photon_energy" not in working:
        working["photon_energy"] = np.nan
    if "replicate_mode" not in working:
        working["replicate_mode"] = "sum"
    if "replicate_grouping" not in working:
        working["replicate_grouping"] = "temperature"
    if "replicate_warning" not in working:
        working["replicate_warning"] = ""
    replicate_mode = str(
        working["replicate_mode"].dropna().iloc[0]
        if not working["replicate_mode"].dropna().empty
        else "off"
    )
    area_agg = "mean" if replicate_mode != "sum" else "sum"
    for mz, group in working.groupby("mz_rounded"):
        group = group.copy()
        group["file_count"] = group.groupby("temperature")["file"].transform("nunique")
        fallback_replicates = pd.Series(dtype=int)
        if replicate_mode != "off":
            fallback_replicates = (
                group.groupby("temperature")["file"]
                .nunique()
                .loc[lambda counts: counts > 1]
            )
        if replicate_mode != "off" and not fallback_replicates.empty:
            for temperature in fallback_replicates.index:
                mask = (
                    (group["temperature"] == temperature)
                    & (group["replicate_grouping"].astype(str) != "filename")
                )
                if mask.any():
                    group.loc[mask, "replicate_warning"] = (
                        "未识别到文件名末尾采集序号，已退回按温度分组的旧逻辑处理重复文件。"
                    )
        if replicate_mode == "off":
            ordered = group.sort_values(["temperature", "file"]).reset_index(drop=True)
        else:
            ordered = (
                group.groupby("temperature", as_index=False)
                .agg(
                    area=("area", area_agg),
                    raw_area=("raw_area", area_agg),
                    integration_method=("integration_method", _summarize_integration_methods),
                    photon_normalized_area=("photon_normalized_area", area_agg),
                    expansion_lambda=("expansion_lambda", "first"),
                    species=("species", "first"),
                    photon_energy=("photon_energy", "first"),
                    mz=("mz", "mean"),
                    file_count=("file", "nunique"),
                    replicate_mode=("replicate_mode", "first"),
                    replicate_grouping=("replicate_grouping", lambda values: "filename" if (values.astype(str) == "filename").any() else "temperature"),
                    replicate_warning=("replicate_warning", lambda values: "; ".join(sorted({str(v) for v in values if str(v)}))),
                    reference_temperature=("reference_temperature", "first"),
                )
                .sort_values("temperature")
            )
        classification = classify_temperature_curve(
            ordered["temperature"],
            ordered["area"],
            relative_change_threshold=relative_change_threshold,
            endpoint_peak_fraction=endpoint_peak_fraction,
        )
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
