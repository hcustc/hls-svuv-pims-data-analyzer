from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .calibration import Calibration
from .integration import integrate_peak
from .normalization import extract_light_intensity
from .peak_ranges import load_peak_ranges, peak_ranges_to_peaks
from .peak_detection import detect_peaks_in_range
from .spectrum_io import Spectrum, list_spectrum_files, read_spectrum


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
    manual_peak_path: str | Path | None = None,
    photon_normalize: bool = False,
    kr_correct: bool = False,
    kr_mz: int = 84,
    mass_discrimination: float = 1.0,
    light_source: str = "io",
    expansion_factors: dict[float, float] | None = None,
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
            ]
        )

    if manual_peak_path:
        reference_peaks = peak_ranges_to_peaks(load_peak_ranges(manual_peak_path, calibration=calibration))
        ref_temperature = max(item[0] for item in spectra)
        reference_source = Path(manual_peak_path).name
    else:
        ref_temperature, ref_spectrum, reference_source = _build_reference_spectrum(spectra, reference_mode)
        reference_peaks = detect_peaks_in_range(
            ref_spectrum.y,
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
    return _apply_temperature_normalization(
        result,
        kr_correct=kr_correct,
        kr_mz=kr_mz,
        mass_discrimination=mass_discrimination,
        expansion_factors=expansion_factors,
    )


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
        curves[int(mz)] = {
            "mz": int(mz),
            "mz_exact_mean": float(ordered["mz"].mean()),
            "species": str(ordered["species"].iloc[0]) if "species" in ordered else "",
            "temperatures": ordered["temperature"].astype(float).tolist(),
            "areas": ordered["area"].astype(float).tolist(),
            "rows": ordered,
        }
    return curves
