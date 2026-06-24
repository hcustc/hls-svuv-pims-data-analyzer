from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import openpyxl
import pandas as pd

try:
    _np_trapezoid = np.trapezoid
except AttributeError:
    _np_trapezoid = np.trapz

from .config import load_peak_integration_config
from .peak_detection import Peak, fit_gaussian
from .spectrum_io import extract_header_numbers, list_spectrum_files, read_spectrum


def baseline_corrected_area(y_data: Iterable[float], left: int, right: int) -> float:
    data = np.asarray(list(y_data), dtype=float)
    left = max(0, int(left))
    right = min(len(data) - 1, int(right))
    if left > right or data.size == 0:
        return 0.0
    segment = data[left : right + 1]
    if segment.size == 0:
        return 0.0
    return float(_np_trapezoid(segment - np.min(segment)))


def gaussian_area(amplitude: float, fwhm: float) -> float:
    return float(max(0.0, amplitude * fwhm * np.sqrt(np.pi / (4 * np.log(2)))))


def integrate_peak(y_data: Iterable[float], peak: Peak, *, prefer_gaussian: bool = True) -> float:
    if prefer_gaussian:
        window_size = max(5, min(30, int(peak.right_bound) - int(peak.left_bound) + 5))
        fit = fit_gaussian(y_data, int(round(peak.index)), window_size)
        if fit is not None:
            area = gaussian_area(fit.amplitude, fit.fwhm)
            if area > 0:
                return area
    return baseline_corrected_area(y_data, peak.left_bound, peak.right_bound)


def integrate_peaks(y_data: Iterable[float], peaks: Iterable[Peak], *, prefer_gaussian: bool = True) -> list[dict]:
    return [
        {
            "mz": peak.mz,
            "time": peak.time,
            "left_bound": peak.left_bound,
            "right_bound": peak.right_bound,
            "area": integrate_peak(y_data, peak, prefer_gaussian=prefer_gaussian),
        }
        for peak in peaks
    ]


def load_peak_config_excel(path: str | Path, sheet_name: str = "卡峰配置") -> dict:
    workbook = openpyxl.load_workbook(path, data_only=True)
    sheet = workbook[sheet_name]
    config: dict = {}
    for row in range(2, sheet.max_row + 1):
        species = sheet.cell(column=1, row=row).value
        if species is None:
            continue
        config.setdefault(species, {})
        for col in range(3, 6):
            key = sheet.cell(column=col, row=1).value
            config[species][key] = sheet.cell(column=col, row=row).value
    return config


def load_peak_config_yaml(path: str | Path | None = None) -> dict:
    config: dict = {}
    for peak in load_peak_integration_config(path):
        species = peak["formula"]
        config[species] = {
            "M/Z": peak["mz"],
            "Peak": peak["peak"],
            "Start": peak["start"],
            "End": peak["end"],
        }
    return config


def load_peak_config(path: str | Path | None = None) -> dict:
    if path is None:
        return load_peak_config_yaml()
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_peak_config_yaml(path)
    return load_peak_config_excel(path)


def integrate_by_config(y_data: Iterable[float], config: dict, *, mode: str = "area") -> dict:
    data = np.asarray(list(y_data), dtype=float)
    result = {}
    for species, info in config.items():
        start = int(info["Start"])
        end = int(info["End"])
        segment = data[start : end + 1]
        if segment.size == 0:
            result[species] = 0.0
        elif mode == "height":
            result[species] = float(np.max(segment))
        else:
            result[species] = float(_np_trapezoid(segment, dx=0.01))
    return result


def integrate_folder_by_config(
    folder: str | Path,
    config: dict,
    *,
    suffixes: tuple[str, ...] = (".txt",),
    normalize_header_index: int | None = 1,
) -> pd.DataFrame:
    rows = []
    for path in list_spectrum_files(folder, suffixes):
        spectrum = read_spectrum(path, header_lines=10, trim_start=0)
        values = integrate_by_config(spectrum.y, config)
        if normalize_header_index is not None:
            header_numbers = extract_header_numbers(path)
            if len(header_numbers) > normalize_header_index and header_numbers[normalize_header_index] != 0:
                values = {key: value / header_numbers[normalize_header_index] for key, value in values.items()}
        rows.append({"file": path.name, **values})
    return pd.DataFrame(rows)
