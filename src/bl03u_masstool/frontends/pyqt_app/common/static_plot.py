from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PyQt6 import QtCore, QtWidgets

_mpl_config_dir = Path(tempfile.gettempdir()) / "bl03u_masstool_matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
_font_cache_dir = _mpl_config_dir / "font-cache"
_font_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_font_cache_dir))

try:
    from matplotlib import rcParams
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - exercised only when the optional runtime is absent
    rcParams = None
    FigureCanvas = None
    Figure = None

from bl03u_masstool.frontends.pyqt_app.theme import get_plot_theme


def _mpl_color(color):
    if isinstance(color, tuple) and color and max(color) > 1:
        return tuple(channel / 255 for channel in color)
    return color


class StaticCurvePlot(QtWidgets.QWidget):
    """Matplotlib canvas for publication-style trend curves in low-interaction pages."""

    def __init__(self, xlabel: str, ylabel: str, *, min_height: int | None = None, parent=None):
        super().__init__(parent)
        self._plot_theme = get_plot_theme()
        self._xlabel = xlabel
        self._ylabel = ylabel
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder: QtWidgets.QLabel | None = None
        if Figure is None or FigureCanvas is None or rcParams is None:
            self.figure = None
            self.canvas = None
            self.axes = None
            self._placeholder = QtWidgets.QLabel("未安装 matplotlib，无法显示高质量曲线图")
            self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setObjectName("ProjectHint")
            if min_height is not None:
                self.setMinimumHeight(min_height)
                self._placeholder.setMinimumHeight(min_height)
            layout.addWidget(self._placeholder)
            return

        rcParams["font.sans-serif"] = [
            "PingFang SC",
            "Microsoft YaHei UI",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        rcParams["axes.unicode_minus"] = False

        self.figure = Figure(figsize=(7.2, 4.2), dpi=110, facecolor=self._plot_theme.background, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        if min_height is not None:
            self.setMinimumHeight(min_height)
            self.canvas.setMinimumHeight(min_height)
        layout.addWidget(self.canvas)

        self.axes = self.figure.add_subplot(111)
        self.clear_plot()

    def clear_plot(self, *, title: str = "", xlabel: str | None = None, ylabel: str | None = None) -> None:
        if self.axes is None or self.canvas is None:
            if self._placeholder is not None:
                self._placeholder.setText(title or "未安装 matplotlib，无法显示高质量曲线图")
            return
        self.axes.clear()
        self._xlabel = xlabel or self._xlabel
        self._ylabel = ylabel or self._ylabel
        self._style_axes()
        self.axes.set_title(title, loc="center", fontsize=12, fontweight="bold", color="#111827", pad=12)
        self.canvas.draw_idle()

    def show_empty(self, message: str, *, title: str = "") -> None:
        if self.axes is None or self.canvas is None:
            if self._placeholder is not None:
                self._placeholder.setText(message)
            return
        self.clear_plot(title=title)
        self.axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.axes.transAxes,
            fontsize=11,
            color="#64748b",
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.canvas.draw_idle()

    def plot_series(
        self,
        x_values: Iterable[float],
        y_values: Iterable[float],
        *,
        label: str | None = None,
        color="#2563eb",
        linewidth: float = 2.2,
        marker: str | None = "o",
        markersize: float = 5.5,
        linestyle: str = "-",
        alpha: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(list(x_values), dtype=float)
        y_arr = np.asarray(list(y_values), dtype=float)
        count = min(x_arr.size, y_arr.size)
        x_arr = x_arr[:count]
        y_arr = y_arr[:count]
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[valid]
        y_arr = y_arr[valid]
        if self.axes is None:
            return x_arr, y_arr
        if x_arr.size == 0:
            return x_arr, y_arr
        self.axes.plot(
            x_arr,
            y_arr,
            label=label,
            color=_mpl_color(color),
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            linestyle=linestyle,
            alpha=alpha,
            markeredgewidth=0.8,
            markeredgecolor="white",
        )
        return x_arr, y_arr

    def apply_data_limits(
        self,
        x_arrays: Iterable[Iterable[float]],
        y_arrays: Iterable[Iterable[float]],
        *,
        x_pad_min: float = 0.2,
        y_pad_min: float = 0.05,
        y_floor: float | None = None,
    ) -> None:
        if self.axes is None:
            return
        x_data = self._finite_concat(x_arrays)
        y_data = self._finite_concat(y_arrays)
        if x_data.size == 0 or y_data.size == 0:
            return
        x_min = float(np.min(x_data))
        x_max = float(np.max(x_data))
        y_min = float(np.min(y_data))
        y_max = float(np.max(y_data))
        if x_min == x_max:
            x_min -= x_pad_min
            x_max += x_pad_min
        if y_min == y_max:
            y_min -= y_pad_min
            y_max += y_pad_min
        # Apply 8% margins (between 5-10% as specified)
        x_pad = max(x_pad_min, (x_max - x_min) * 0.08)
        y_pad = max(y_pad_min, (y_max - y_min) * 0.08)
        upper_y = y_max + y_pad
        lower_y = y_min - y_pad
        # Special handling when y_min is 0: add modest negative margin (20% of padding)
        if y_min == 0.0:
            lower_y = -y_pad * 0.2
        elif y_floor is not None:
            lower_y = max(y_floor, lower_y)
        self.axes.set_xlim(x_min - x_pad, x_max + x_pad)
        self.axes.set_ylim(lower_y, upper_y)

    def finish(self, *, legend: bool = False, legend_loc: str = "best", legend_bbox_to_anchor: tuple[float, float] | None = None) -> None:
        if self.axes is None or self.canvas is None:
            return
        if legend:
            handles, labels = self.axes.get_legend_handles_labels()
            if labels:
                column_count = 2 if len(labels) > 8 else 1
                legend_kwargs = {
                    "loc": legend_loc,
                    "fontsize": 8,
                    "frameon": True,
                    "ncol": column_count,
                    "labelspacing": 0.35,
                    "handlelength": 1.8,
                    "borderpad": 0.5,
                }
                if legend_bbox_to_anchor is not None:
                    legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
                legend_obj = self.axes.legend(**legend_kwargs)
                legend_obj.get_frame().set_facecolor("#ffffff")
                legend_obj.get_frame().set_edgecolor("#d8dee8")
                legend_obj.get_frame().set_alpha(0.92)
        self.canvas.draw_idle()

    def save_plot(self, filepath: str | Path) -> bool:
        """Save figure to file. Returns True if successful."""
        if self.figure is None:
            return False
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(
                str(filepath),
                dpi=300,
                bbox_inches="tight",
                facecolor=self._plot_theme.background,
            )
            return True
        except Exception:
            return False

    def _style_axes(self) -> None:
        self.axes.set_facecolor(self._plot_theme.background)
        self.axes.set_xlabel(self._xlabel, color="#334155", labelpad=8)
        self.axes.set_ylabel(self._ylabel, color="#334155", labelpad=8)
        # Lighten grid: only major grid, reduce opacity
        self.axes.grid(True, which="major", color=self._plot_theme.grid, linewidth=0.6, alpha=0.35)
        self.axes.grid(False, which="minor")
        self.axes.tick_params(colors="#475569", labelsize=9)
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["left"].set_color("#cbd5e1")
        self.axes.spines["bottom"].set_color("#cbd5e1")

    @staticmethod
    def _finite_concat(arrays: Iterable[Iterable[float]]) -> np.ndarray:
        finite_arrays = []
        for values in arrays:
            arr = np.asarray(list(values), dtype=float)
            if arr.size:
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    finite_arrays.append(arr)
        if not finite_arrays:
            return np.array([], dtype=float)
        return np.concatenate(finite_arrays)
