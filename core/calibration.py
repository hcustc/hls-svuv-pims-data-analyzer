from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """Quadratic TOF to m/z calibration: mz = c + b * tof + a * tof^2."""

    a: float = 3.66334e-7
    b: float = 0.000637719
    c: float = 0.272489072

    def tof_to_mz(self, tof):
        return tof_to_mz(tof, self.a, self.b, self.c)

    def mz_to_tof(self, mz: float) -> float:
        return mz_to_tof(mz, self.a, self.b, self.c)


def tof_to_mz(tof, a: float, b: float, c: float):
    """Convert flight time to m/z using mz = c + b * tof + a * tof^2."""
    tof_array = np.asarray(tof, dtype=float)
    result = c + b * tof_array + a * tof_array * tof_array
    return float(result) if np.ndim(result) == 0 else result


def mz_to_tof(mz: float, a: float, b: float, c: float) -> float:
    """Solve the positive TOF root for mz = c + b * tof + a * tof^2."""
    if abs(a) < 1e-20:
        if abs(b) < 1e-20:
            raise ValueError("invalid calibration: both a and b are zero")
        return (mz - c) / b

    discriminant = b * b - 4 * a * (c - mz)
    if discriminant < 0:
        raise ValueError("calibration equation has no real TOF root")
    root = np.sqrt(discriminant)
    candidates = [(-b + root) / (2 * a), (-b - root) / (2 * a)]
    positive = [value for value in candidates if value >= 0]
    if not positive:
        raise ValueError("calibration equation has no positive TOF root")
    return float(min(positive))


def fit_quadratic_calibration(points: Iterable[Sequence[float]]) -> Calibration:
    """Fit a quadratic calibration from (tof, mz) points."""
    clean_points = [(float(tof), float(mz)) for tof, mz in points]
    if len(clean_points) < 3:
        raise ValueError("at least three calibration points are required")

    x = np.array([item[0] for item in clean_points], dtype=float)
    y = np.array([item[1] for item in clean_points], dtype=float)
    design = np.column_stack([x**2, x, np.ones(len(x))])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return Calibration(a=float(coeffs[0]), b=float(coeffs[1]), c=float(coeffs[2]))


def score_quadratic_calibration(points: Iterable[Sequence[float]], calibration: Calibration) -> float:
    """Return R^2 for a calibration against (tof, mz) points."""
    clean_points = [(float(tof), float(mz)) for tof, mz in points]
    if not clean_points:
        return 0.0
    x = np.array([item[0] for item in clean_points], dtype=float)
    y = np.array([item[1] for item in clean_points], dtype=float)
    pred = calibration.tof_to_mz(x)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 0.0
    ss_res = float(np.sum((y - pred) ** 2))
    return 1 - ss_res / ss_tot

