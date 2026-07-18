from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

from bl03u_masstool.core.calibration import Calibration, tof_to_mz
from bl03u_masstool.core.config import (
    PeakDetectionConfig,
    load_calibration_config,
    load_peak_detection_config,
    save_calibration_config,
    save_peak_detection_config,
    species_database_path,
)
from bl03u_masstool.core.isotope import (
    calculate_isotope_distribution,
    formula_monoisotopic_mass,
    formula_nominal_mass,
    generate_formula_candidates,
    parse_element_count_ranges,
    parse_formula,
)
from bl03u_masstool.core.nist_webbook import default_nist_webbook_client
from bl03u_masstool.core.output_paths import ensure_output_dir
from bl03u_masstool.core.pie_analysis import analyze_pie_folder, build_pie_curves, identify_species_for_mz_with_curve, load_species_database, analyze_multiple_pie_folders, merge_pie_segments
from bl03u_masstool.core.pics_calculator import calc_pics_single_energy
from bl03u_masstool.core.elements import get_all_elements_from_database, filter_species_by_elements, COMMON_ELEMENTS, parse_formula as parse_formula_elements, get_elements_from_formula
from bl03u_masstool.core.normalization import NormalizationSettings, load_normalization_settings, save_normalization_settings
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.temperature_scan import (
    TEMPERATURE_CURVE_CLASS_LABELS,
    analyze_temperature_folder,
    build_temperature_curves,
    compute_kr_expansion_factors,
)
from bl03u_masstool.core.mole_fraction import (
    MASS_DISCRIMINATION_PRESETS,
    MoleFractionSettings,
    calc_expansion_coefficients,
    calc_isomeric_separation,
    calc_mass_discrimination,
    calc_parent_mole_fraction,
    calc_product_mole_fraction,
    compute_all_mole_fractions,
    extract_signal_from_temperature_curves,
    get_expansion_coefficient,
    load_mole_fraction_settings,
    save_mole_fraction_settings,
)
from bl03u_masstool.frontends.pyqt_app.theme import get_plot_theme
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - only used when optional plotting is unavailable
    pg = None


class StateGlyph(QtWidgets.QWidget):
    """Small painted status glyph that avoids platform-dependent text symbols."""

    def __init__(self, kind: str = "candidate", parent=None):
        super().__init__(parent)
        self._kind = kind
        self.setFixedSize(64, 64)
        self.setAccessibleName("状态图标")

    def set_kind(self, kind: str) -> None:
        self._kind = str(kind or "candidate")
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        center = QtCore.QPointF(self.width() / 2, self.height() / 2)
        warning = self._kind == "warning"
        accent = QtGui.QColor("#d97706" if warning else "#2563eb")
        soft = QtGui.QColor("#fff7ed" if warning else "#eff6ff")
        painter.setPen(QtGui.QPen(QtGui.QColor("#fed7aa" if warning else "#bfdbfe"), 1.2))
        painter.setBrush(soft)
        painter.drawEllipse(center, 29, 29)

        if warning:
            triangle = QtGui.QPolygonF(
                [
                    QtCore.QPointF(32, 17),
                    QtCore.QPointF(48, 45),
                    QtCore.QPointF(16, 45),
                ]
            )
            painter.setPen(QtGui.QPen(accent, 2.2, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPolygon(triangle)
            painter.drawLine(QtCore.QPointF(32, 26), QtCore.QPointF(32, 36))
            painter.drawPoint(QtCore.QPointF(32, 40))
            return

        painter.setPen(QtGui.QPen(accent, 2.2, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
        for y, width in ((23, 19), (32, 25), (41, 16)):
            painter.drawEllipse(QtCore.QPointF(20, y), 2.2, 2.2)
            painter.drawLine(QtCore.QPointF(27, y), QtCore.QPointF(27 + width, y))


class CurvePreviewCanvas(QtWidgets.QWidget):
    """Painted curve preview for analysis empty states."""

    def __init__(self, variant: str = "pie", parent=None):
        super().__init__(parent)
        self._variant = variant
        self.setMinimumHeight(150)
        self.setMaximumHeight(190)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.setAccessibleName("分析曲线预览示意图")

    def _normalized_points(self) -> list[tuple[float, float]]:
        if self._variant == "temperature":
            return [
                (0.04, 0.78),
                (0.16, 0.74),
                (0.28, 0.64),
                (0.40, 0.48),
                (0.52, 0.27),
                (0.64, 0.18),
                (0.76, 0.34),
                (0.88, 0.57),
                (0.96, 0.66),
            ]
        return [
            (0.04, 0.84),
            (0.16, 0.83),
            (0.28, 0.79),
            (0.40, 0.68),
            (0.52, 0.61),
            (0.64, 0.45),
            (0.76, 0.36),
            (0.88, 0.17),
            (0.96, 0.09),
        ]

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        outer = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(QtGui.QPen(QtGui.QColor("#dbe4f0"), 1.0))
        painter.setBrush(QtGui.QColor("#f8fafc"))
        painter.drawRoundedRect(outer, 12, 12)

        chart = outer.adjusted(34, 20, -24, -25)
        grid_pen = QtGui.QPen(QtGui.QColor("#e2e8f0"), 1.0, QtCore.Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for index in range(1, 5):
            x = chart.left() + chart.width() * index / 5
            painter.drawLine(QtCore.QPointF(x, chart.top()), QtCore.QPointF(x, chart.bottom()))
        for index in range(1, 4):
            y = chart.top() + chart.height() * index / 4
            painter.drawLine(QtCore.QPointF(chart.left(), y), QtCore.QPointF(chart.right(), y))

        axis_pen = QtGui.QPen(QtGui.QColor("#94a3b8"), 1.2)
        painter.setPen(axis_pen)
        painter.drawLine(chart.bottomLeft(), chart.bottomRight())
        painter.drawLine(chart.bottomLeft(), chart.topLeft())

        points = [
            QtCore.QPointF(
                chart.left() + x * chart.width(),
                chart.top() + y * chart.height(),
            )
            for x, y in self._normalized_points()
        ]
        accent = QtGui.QColor("#0f766e" if self._variant == "temperature" else "#2563eb")
        fill = QtGui.QColor(accent)
        fill.setAlpha(28)
        area = QtGui.QPolygonF(
            [chart.bottomLeft(), *points, chart.bottomRight()]
        )
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPolygon(area)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(
            QtGui.QPen(
                accent,
                2.6,
                QtCore.Qt.PenStyle.SolidLine,
                QtCore.Qt.PenCapStyle.RoundCap,
                QtCore.Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawPolyline(QtGui.QPolygonF(points))
        painter.setBrush(QtGui.QColor("#ffffff"))
        for point in points:
            painter.drawEllipse(point, 3.2, 3.2)


class AnalysisEmptyState(QtWidgets.QFrame):
    """Reusable, visually structured empty state for analysis workspaces."""

    browse_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        variant: str,
        eyebrow: str,
        title: str,
        description: str,
        steps: tuple[str, str, str],
        action_text: str = "选择数据目录",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("AnalysisEmptyState")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)

        card = QtWidgets.QFrame()
        card.setObjectName("EmptyStateCard")
        card.setMinimumWidth(420)
        card.setMaximumWidth(720)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(10)

        eyebrow_label = QtWidgets.QLabel(eyebrow)
        eyebrow_label.setObjectName("EmptyStateEyebrow")
        eyebrow_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(eyebrow_label)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("EmptyStateTitle")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title_label)

        description_label = QtWidgets.QLabel(description)
        description_label.setObjectName("EmptyStateSubtitle")
        description_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        card_layout.addWidget(description_label)

        card_layout.addWidget(CurvePreviewCanvas(variant, card))

        steps_layout = QtWidgets.QHBoxLayout()
        steps_layout.setSpacing(8)
        for index, step in enumerate(steps, start=1):
            step_frame = QtWidgets.QFrame()
            step_frame.setObjectName("EmptyStateStep")
            step_layout = QtWidgets.QHBoxLayout(step_frame)
            step_layout.setContentsMargins(9, 7, 9, 7)
            step_layout.setSpacing(7)
            number = QtWidgets.QLabel(str(index))
            number.setObjectName("EmptyStateStepNumber")
            number.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            step_layout.addWidget(number)
            label = QtWidgets.QLabel(step)
            label.setObjectName("EmptyStateStepLabel")
            label.setWordWrap(True)
            step_layout.addWidget(label, stretch=1)
            steps_layout.addWidget(step_frame, stretch=1)
        card_layout.addLayout(steps_layout)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch()
        self.primary_button = QtWidgets.QPushButton(action_text)
        self.primary_button.setObjectName("EmptyStateAction")
        self.primary_button.clicked.connect(self.browse_requested.emit)
        self.primary_button.setAccessibleDescription("从当前分析页面选择数据目录")
        footer.addWidget(self.primary_button)
        hint = QtWidgets.QLabel("已有数据源时可直接使用上方“生成曲线”")
        hint.setObjectName("EmptyStateFooterHint")
        footer.addWidget(hint)
        footer.addStretch()
        card_layout.addLayout(footer)

        outer.addStretch()
        outer.addWidget(card, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        outer.addStretch()

class DataFrameTableMixin:
    def set_dataframe(self, table: QtWidgets.QTableWidget, df: pd.DataFrame) -> None:
        table.blockSignals(True)
        table.setUpdatesEnabled(False)
        table.clearContents()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, value in enumerate(row):
                table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem("" if pd.isna(value) else str(value)))
        table.clearSelection()
        table.setCurrentItem(None)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.resizeColumnsToContents()
        table.setUpdatesEnabled(True)
        table.blockSignals(False)


def combo_set_data(combo: QtWidgets.QComboBox, value: object) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            combo.setCurrentIndex(index)
            return


class FlowLayout(QtWidgets.QLayout):
    """Custom flow layout: arranges widgets left-to-right, wrapping to next row."""

    def __init__(self, parent=None, margin: int = 0, spacing: int = 4):
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._h_spacing = spacing
        self._v_spacing = spacing
        if margin >= 0:
            self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def addWidget(self, w: QtWidgets.QWidget) -> None:
        super().addWidget(w)

    def expandingDirections(self) -> QtCore.Qt.Orientation:
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QtCore.QRect(0, 0, width, 0), apply_geometry=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, apply_geometry=True)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect: QtCore.QRect, apply_geometry: bool) -> int:
        margins = self.contentsMargins()
        x = rect.x() + margins.left()
        y = rect.y() + margins.top()
        line_height = 0
        max_width = rect.width() - margins.left() - margins.right()

        for item in self._items:
            widget = item.widget()
            if widget and not widget.isVisible():
                continue
            size_hint = item.sizeHint()
            next_x = x + size_hint.width() + self._h_spacing
            if next_x - self._h_spacing > max_width and line_height > 0:
                x = rect.x() + margins.left()
                y += line_height + self._v_spacing
                line_height = 0
            if apply_geometry:
                item.setGeometry(QtCore.QRect(x, y, size_hint.width(), size_hint.height()))
            x += size_hint.width() + self._h_spacing
            line_height = max(line_height, size_hint.height())
        return y + line_height + margins.bottom() - rect.y()
