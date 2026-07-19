from __future__ import annotations

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PyQt6 import QtCore, QtWidgets

_mpl_config_dir = Path(tempfile.gettempdir()) / "bl03u_masstool_matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
_font_cache_dir = _mpl_config_dir / "font-cache"
_font_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_font_cache_dir))

try:
    from matplotlib import font_manager, rcParams, rc_context, rc_params_from_file
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - exercised only when the optional runtime is absent
    font_manager = None
    rcParams = None
    rc_context = None
    rc_params_from_file = None
    FigureCanvas = None
    Figure = None

from bl03u_masstool.core.runtime_paths import default_resource_path
from bl03u_masstool.frontends.pyqt_app.common.plot_spec import (
    CurveSeries,
    PlotProfile,
    ScientificPlotSpec,
    SeriesRole,
)
from bl03u_masstool.frontends.pyqt_app.theme import get_plot_theme


def _mpl_color(color):
    if isinstance(color, tuple) and color and max(color) > 1:
        return tuple(channel / 255 for channel in color)
    return color


_CJK_FONT_FAMILIES = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "SimSun",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]

_PROFILE_FILENAMES = {
    PlotProfile.SCREEN: "bl03u-screen.mplstyle",
    PlotProfile.PUBLICATION: "bl03u-publication.mplstyle",
    PlotProfile.MONOCHROME: "bl03u-monochrome.mplstyle",
}

_SCIENCE_COLOR_CYCLE = (
    "#0C5DA5",
    "#00B945",
    "#FF9500",
    "#FF2C00",
    "#845B97",
    "#474747",
    "#9E9E9E",
)

_COMPONENT_COLORS = (
    "#00B945",
    "#FF9500",
    "#845B97",
    "#474747",
    "#9E9E9E",
)

_COMPONENT_LINESTYLES: tuple[Any, ...] = (
    "--",
    "-.",
    ":",
    (0, (5, 2, 1, 2)),
    (0, (3, 1, 1, 1, 1, 1)),
)

_COMPARISON_LINESTYLES: tuple[Any, ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 2, 1, 2)),
    (0, (3, 1, 1, 1, 1, 1)),
)


@lru_cache(maxsize=len(_PROFILE_FILENAMES))
def load_plot_profile(profile: PlotProfile | str) -> dict[str, Any]:
    """Load a packaged Matplotlib profile without mutating global rcParams."""
    resolved = PlotProfile(profile)
    if rc_params_from_file is None:
        return {}
    style_path = default_resource_path(Path("mplstyles") / _PROFILE_FILENAMES[resolved])
    if not style_path.is_file():
        raise FileNotFoundError(f"Matplotlib style profile not found: {style_path}")
    return dict(rc_params_from_file(style_path, use_default_template=False))


def configure_matplotlib_fonts(
    *,
    platform_name: str | None = None,
    windows_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Select a real CJK-capable Matplotlib font, including in frozen Windows builds."""
    if rcParams is None or font_manager is None:
        return ""

    platform_name = platform_name or sys.platform
    if platform_name == "win32":
        font_root = Path(windows_dir or os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        # PyInstaller launches can inherit an incomplete/stale Matplotlib font cache.
        # Explicit registration makes the system fonts available for this process.
        for filename in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"):
            font_path = font_root / filename
            if font_path.is_file():
                try:
                    font_manager.fontManager.addfont(str(font_path))
                except (OSError, RuntimeError):
                    continue

    installed_families = {entry.name for entry in font_manager.fontManager.ttflist}
    selected_family = next(
        (family for family in _CJK_FONT_FAMILIES if family in installed_families),
        "DejaVu Sans",
    )
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [
        selected_family,
        *[family for family in _CJK_FONT_FAMILIES if family != selected_family],
        "DejaVu Sans",
    ]
    rcParams["axes.unicode_minus"] = False
    return selected_family


class StaticCurvePlot(QtWidgets.QWidget):
    """Matplotlib canvas for publication-style trend curves in low-interaction pages."""

    def __init__(self, xlabel: str, ylabel: str, *, min_height: int | None = None, parent=None):
        super().__init__(parent)
        self._plot_theme = get_plot_theme()
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._last_spec: ScientificPlotSpec | None = None
        self._last_profile = PlotProfile.SCREEN
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

        configure_matplotlib_fonts()

        screen_profile = load_plot_profile(PlotProfile.SCREEN)
        with rc_context(screen_profile):
            self.figure = Figure(
                figsize=(7.2, 4.2),
                dpi=110,
                facecolor=self._plot_theme.background,
                constrained_layout=True,
            )
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        if min_height is not None:
            self.setMinimumHeight(min_height)
            self.canvas.setMinimumHeight(min_height)
        layout.addWidget(self.canvas)

        with rc_context(screen_profile):
            self.axes = self.figure.add_subplot(111)
            self.clear_plot()

    def render_spec(
        self,
        spec: ScientificPlotSpec,
        *,
        profile: PlotProfile | str = PlotProfile.SCREEN,
        draw: bool = True,
    ) -> None:
        """Render a scientific plot specification while preserving point order."""
        if self.axes is None or self.canvas is None or rc_context is None:
            if self._placeholder is not None:
                self._placeholder.setText(spec.title or "无法显示曲线图")
            return

        resolved_profile = PlotProfile(profile)
        self._last_spec = spec
        self._last_profile = resolved_profile
        self._xlabel = spec.xlabel
        self._ylabel = spec.ylabel

        x_arrays: list[np.ndarray] = []
        y_arrays: list[np.ndarray] = []
        component_index = 0
        comparison_index = 0
        with rc_context(load_plot_profile(resolved_profile)):
            self.axes.clear()
            self._style_axes()
            self.axes.set_title(spec.title)

            for series in spec.series:
                if (
                    series.role is SeriesRole.COMPONENT
                    and series.contribution_percent is not None
                    and series.contribution_percent <= spec.component_visibility_threshold
                ):
                    continue

                x_arr, y_arr = self._finite_series_values(series)
                if x_arr.size == 0:
                    continue
                x_arrays.append(x_arr)
                y_arrays.append(y_arr)

                if series.role is SeriesRole.COMPONENT:
                    style_index = component_index
                    component_index += 1
                elif series.role is SeriesRole.COMPARISON:
                    style_index = comparison_index
                    comparison_index += 1
                else:
                    style_index = 0

                style = self._series_style(series, style_index, resolved_profile)
                self.axes.plot(
                    x_arr,
                    y_arr,
                    label=series.label or None,
                    **style,
                )

            if spec.xlim is None or spec.ylim is None:
                self.apply_data_limits(x_arrays, y_arrays)
            if spec.xlim is not None:
                self.axes.set_xlim(*spec.xlim)
            if spec.ylim is not None:
                self.axes.set_ylim(*spec.ylim)

            if spec.show_zero_line:
                self.axes.axhline(
                    0.0,
                    color="#64748b" if resolved_profile is PlotProfile.SCREEN else "0.35",
                    linewidth=0.9,
                    linestyle=(0, (4, 4)),
                    alpha=0.65,
                    zorder=0,
                )

            if spec.show_legend:
                handles, labels = self.axes.get_legend_handles_labels()
                if labels:
                    self.axes.legend(
                        handles,
                        labels,
                        loc=spec.legend_loc,
                        ncol=2 if len(labels) > 8 else 1,
                    )

            if draw:
                self.canvas.draw_idle()

    def clear_plot(self, *, title: str = "", xlabel: str | None = None, ylabel: str | None = None) -> None:
        if self.axes is None or self.canvas is None:
            if self._placeholder is not None:
                self._placeholder.setText(title or "未安装 matplotlib，无法显示高质量曲线图")
            return
        with rc_context(load_plot_profile(PlotProfile.SCREEN)):
            self.axes.clear()
            self._xlabel = xlabel or self._xlabel
            self._ylabel = ylabel or self._ylabel
            self._style_axes()
            self.axes.set_title(title)
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
                    "borderpad": 0.8,  # Increased from 0.5 to 0.8 for 10-15px padding
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
        previous_spec = self._last_spec
        previous_profile = self._last_profile
        previous_size = self.figure.get_size_inches().copy()
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            profile = load_plot_profile(PlotProfile.PUBLICATION)
            if previous_spec is not None:
                self.render_spec(previous_spec, profile=PlotProfile.PUBLICATION, draw=False)
            publication_size = profile.get("figure.figsize")
            if publication_size:
                self.figure.set_size_inches(*publication_size, forward=False)
            with rc_context(profile):
                self.figure.savefig(
                    str(filepath),
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=self._plot_theme.background,
                )
            return True
        except Exception:
            return False
        finally:
            self.figure.set_size_inches(*previous_size, forward=False)
            if previous_spec is not None:
                self.render_spec(previous_spec, profile=previous_profile)

    def _style_axes(self) -> None:
        self.axes.set_facecolor(self._plot_theme.background)
        self.axes.set_xlabel(self._xlabel)
        self.axes.set_ylabel(self._ylabel)
        self.axes.grid(bool(rcParams["axes.grid"]), which="major")
        self.axes.grid(False, which="minor")
        self.axes.spines["top"].set_visible(bool(rcParams["xtick.top"]))
        self.axes.spines["right"].set_visible(bool(rcParams["ytick.right"]))

    @staticmethod
    def _finite_series_values(series: CurveSeries) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(list(series.x), dtype=float)
        y_arr = np.asarray(list(series.y), dtype=float)
        count = min(x_arr.size, y_arr.size)
        x_arr = x_arr[:count]
        y_arr = y_arr[:count]
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        return x_arr[valid], y_arr[valid]

    def _series_style(
        self,
        series: CurveSeries,
        index: int,
        profile: PlotProfile,
    ) -> dict[str, Any]:
        monochrome = profile is PlotProfile.MONOCHROME
        publication = profile is not PlotProfile.SCREEN
        if series.role is SeriesRole.EXPERIMENTAL:
            return {
                "color": "0.12" if monochrome else _SCIENCE_COLOR_CYCLE[0],
                "linewidth": 0.75 if publication else 0.85,
                "marker": "o",
                "markersize": 3.0 if publication else 4.0,
                "markeredgewidth": 0.45 if publication else 0.55,
                "markeredgecolor": "white",
                "zorder": 5,
            }
        if series.role is SeriesRole.TOTAL_FIT:
            return {
                "color": "0.05" if monochrome else _SCIENCE_COLOR_CYCLE[3],
                "linewidth": 1.3 if publication else 1.7,
                "marker": None,
                "linestyle": "-",
                "zorder": 6,
            }
        if series.role is SeriesRole.COMPONENT:
            return {
                "color": (
                    str(min(0.28 + index * 0.12, 0.72))
                    if monochrome
                    else (series.color or _COMPONENT_COLORS[index % len(_COMPONENT_COLORS)])
                ),
                "linewidth": 0.8 if publication else 0.95,
                "marker": None,
                "linestyle": _COMPONENT_LINESTYLES[index % len(_COMPONENT_LINESTYLES)],
                "alpha": 0.92,
                "zorder": 3,
            }
        if series.role is SeriesRole.RESIDUAL:
            return {
                "color": "0.20" if monochrome else _SCIENCE_COLOR_CYCLE[5],
                "linewidth": 0.8 if publication else 0.95,
                "marker": "o",
                "markersize": 2.8 if publication else 3.6,
                "linestyle": "-",
                "zorder": 4,
            }
        if series.role is SeriesRole.COMPARISON:
            return {
                "color": (
                    str(min(0.15 + index * 0.10, 0.75))
                    if monochrome
                    else (series.color or _SCIENCE_COLOR_CYCLE[index % len(_SCIENCE_COLOR_CYCLE)])
                ),
                "linewidth": 1.0 if publication else 1.25,
                "marker": "o",
                "markersize": 2.8 if publication else 3.6,
                "linestyle": _COMPARISON_LINESTYLES[index % len(_COMPARISON_LINESTYLES)],
                "markeredgewidth": 0.65,
                "markeredgecolor": "white",
                "zorder": 3,
            }
        return {
            "color": "0.15" if monochrome else (series.color or _SCIENCE_COLOR_CYCLE[0]),
            "linewidth": 1.0 if publication else 1.4,
            "marker": "o",
            "markersize": 3.0 if publication else 4.0,
            "markeredgewidth": 0.45 if publication else 0.55,
            "markeredgecolor": "white",
            "zorder": 4,
        }

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
