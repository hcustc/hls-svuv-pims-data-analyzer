from __future__ import annotations

from typing import Iterable

import numpy as np

from .peak_detection import Peak
from .spectrum_io import read_bl03u_txt, sum_spectra


def spectrum_plot_data(path: str, *, trim_start: int = 4000) -> dict:
    spectrum = read_bl03u_txt(path, trim_start=trim_start)
    return {"x": spectrum.x.tolist(), "y": spectrum.y.tolist(), "path": spectrum.path}


def summed_spectrum_plot_data(folder: str, *, trim_start: int = 4000) -> dict:
    spectrum = sum_spectra(folder, trim_start=trim_start)
    return {"x": spectrum.x.tolist(), "y": spectrum.y.tolist(), "path": spectrum.path}


def peak_annotations(peaks: Iterable[Peak]) -> list[dict]:
    return [
        {
            "x": peak.time,
            "y": peak.intensity,
            "label": f"{peak.mz:.2f}",
            "left": peak.left_bound,
            "right": peak.right_bound,
        }
        for peak in peaks
    ]


def curve_data(x_values: Iterable[float], y_values: Iterable[float], *, normalize: bool = False) -> dict:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    if normalize and y.size and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return {"x": x.tolist(), "y": y.tolist()}

