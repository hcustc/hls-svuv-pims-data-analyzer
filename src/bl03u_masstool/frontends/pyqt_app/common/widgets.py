from __future__ import annotations

import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets


class ElidedLabel(QtWidgets.QLabel):
    """Single-line label that elides long text while preserving it in a tooltip."""

    def __init__(self, text: str = "", parent=None):
        super().__init__("", parent)
        self._full_text = ""
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._update_elided_text()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        available = max(0, self.contentsRect().width() - 2)
        visible = self.fontMetrics().elidedText(
            self._full_text,
            QtCore.Qt.TextElideMode.ElideRight,
            available,
        )
        super().setText(visible)


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


class AnalysisEmptyState(QtWidgets.QFrame):
    """Compact empty state with one clear next action."""

    browse_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        title: str,
        description: str,
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
        card.setMinimumWidth(460)
        card.setMaximumWidth(560)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(28, 24, 28, 24)
        card_layout.setSpacing(12)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("EmptyStateTitle")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title_label)

        description_label = QtWidgets.QLabel(description)
        description_label.setObjectName("EmptyStateSubtitle")
        description_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        card_layout.addWidget(description_label)

        self.primary_button = QtWidgets.QPushButton(action_text)
        self.primary_button.setObjectName("EmptyStateAction")
        self.primary_button.clicked.connect(self.browse_requested.emit)
        self.primary_button.setAccessibleDescription("从当前分析页面选择数据目录")
        card_layout.addWidget(
            self.primary_button,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        outer.addStretch()
        outer.addWidget(card, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        outer.addStretch()


class AnalysisProgressState(QtWidgets.QFrame):
    """Centered analysis progress shown in the plot workspace."""

    def __init__(self, *, title: str = "正在处理数据", parent=None):
        super().__init__(parent)
        self.setObjectName("AnalysisProgressState")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setAccessibleName("数据处理进度")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)

        card = QtWidgets.QFrame()
        card.setObjectName("ProgressStateCard")
        card.setMinimumWidth(420)
        card.setMaximumWidth(540)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(28, 24, 28, 24)
        card_layout.setSpacing(12)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("ProgressStateTitle")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.title_label)

        self.detail_label = QtWidgets.QLabel("正在准备分析…")
        self.detail_label.setObjectName("ProgressStateDetail")
        self.detail_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setWordWrap(True)
        card_layout.addWidget(self.detail_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("AnalysisProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAccessibleName("分析完成百分比")
        card_layout.addWidget(self.progress_bar)

        outer.addStretch()
        outer.addWidget(card, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        outer.addStretch()

    def start(self, *, title: str, detail: str) -> None:
        self.title_label.setText(title)
        self.set_progress(0, detail)

    def set_progress(self, value: int, detail: str = "") -> None:
        progress = max(0, min(100, int(value)))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{progress}%")
        if detail:
            self.detail_label.setText(detail)


class DataFrameTableMixin:
    def set_dataframe(self, table: QtWidgets.QTableWidget, df: pd.DataFrame) -> None:
        updates_enabled = table.updatesEnabled()
        signals_blocked = table.signalsBlocked()
        table.blockSignals(True)
        table.setUpdatesEnabled(False)
        try:
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
        finally:
            table.setUpdatesEnabled(updates_enabled)
            table.blockSignals(signals_blocked)


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
