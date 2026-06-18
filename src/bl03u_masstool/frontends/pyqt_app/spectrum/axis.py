from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from bl03u_masstool.core.calibration import Calibration


class SpectrumBottomAxis(pg.AxisItem):
    """Bottom axis formatter for TOF or true m/z plot coordinates."""

    def __init__(self, orientation: str = "bottom"):
        super().__init__(orientation=orientation)
        self.display_mode = "mz"
        self.calibration = Calibration()

    def set_display_mode(self, mode: str, calibration: Calibration) -> None:
        self.display_mode = mode if mode in {"tof", "mz"} else "mz"
        self.calibration = calibration
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if self.display_mode != "mz":
            return super().tickStrings(values, scale, spacing)

        mz_values = np.atleast_1d(np.asarray(values, dtype=float))
        finite_mz_values = mz_values[np.isfinite(mz_values)]
        mz_spacing = float(abs(spacing)) if np.isfinite(spacing) and spacing > 0 else None
        if mz_spacing is None and finite_mz_values.size > 1:
            sorted_values = np.sort(finite_mz_values)
            deltas = np.diff(sorted_values)
            deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
            if deltas.size:
                mz_spacing = float(np.min(deltas))
        return [self._format_tick(float(value), mz_spacing) for value in mz_values]

    @staticmethod
    def _format_tick(value: float, spacing: float | None = None) -> str:
        if not np.isfinite(value):
            return ""
        if spacing is None or not np.isfinite(spacing) or spacing <= 0:
            decimals = 4
        else:
            decimals = int(np.clip(np.ceil(-np.log10(spacing)) + 1, 3, 6))
        return f"{value:.{decimals}f}"
