from __future__ import annotations

from PyQt6 import QtWidgets


APP_QSS = """
QWidget {
    font-family: "PingFang SC", "Helvetica Neue", "Microsoft YaHei UI", Arial, sans-serif;
    font-size: 10pt;
    color: #1f2937;
    background: #f6f8fb;
}

QMainWindow, QDialog {
    background: #f6f8fb;
}

QLabel,
QCheckBox {
    background: transparent;
}

QWidget#ControlBar,
QWidget#PlotPanel,
QWidget#SidePanel {
    background: #ffffff;
    border: 1px solid #d8dee8;
    border-radius: 6px;
}

QFrame#ProjectCard {
    background: #ffffff;
    border: 1px solid #d8dee8;
    border-radius: 6px;
}

QLabel#ProjectTitle {
    background: transparent;
    color: #111827;
    font-size: 11pt;
    font-weight: 700;
}

QLabel#ProjectStatus {
    background: transparent;
    color: #475569;
    font-weight: 600;
}

QLabel#ProjectParamSummary {
    background: #f8fafc;
    border: 1px solid #d8dee8;
    border-radius: 5px;
    color: #334155;
    padding: 4px 7px;
}

QWidget#PageNav {
    background: transparent;
    border: 0;
}

QWidget#ProjectPage {
    background: transparent;
}

QStackedWidget#WorkspaceStack {
    background: transparent;
    border: 0;
}

QWidget#CalibrationPanel {
    background: transparent;
}

QWidget#SourcePanel,
QStackedWidget#SourceStack,
QWidget#ModePanel,
QWidget#ModeSegment,
QWidget#ArrowGroup {
    background: transparent;
    border: 0;
}

QLabel#ModeTitle {
    background: transparent;
    color: #475569;
    font-weight: 600;
}

QWidget#PeakSummary {
    background: #f8fafc;
    border: 1px solid #d8dee8;
    border-radius: 6px;
}

QLabel#ReadoutLabel {
    background: transparent;
    color: #475569;
    font-weight: 600;
}

QLabel#ReadoutValue {
    background: #ffffff;
    border: 1px solid #d8dee8;
    border-radius: 4px;
    color: #111827;
    font-weight: 600;
    padding: 3px 6px;
}

QGroupBox {
    border: 1px solid #d8dee8;
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
}

QFrame#PanelFrame {
    border: 1px solid #d8dee8;
    border-radius: 6px;
    padding: 8px;
}

QPushButton {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
    border-radius: 5px;
    padding: 4px 10px;
    min-height: 20px;
}

QPushButton:hover {
    background: #1d4ed8;
}

QPushButton:pressed {
    background: #1e40af;
}

QPushButton:disabled {
    background: #cbd5e1;
    border-color: #cbd5e1;
    color: #64748b;
}

QPushButton#WorkflowButton {
    background: #ffffff;
    color: #1d4ed8;
    border-color: #bfdbfe;
    padding: 3px 8px;
    min-height: 20px;
}

QPushButton#WorkflowButton:hover {
    background: #eff6ff;
    border-color: #2563eb;
}

QPushButton#BrowseButton {
    background: #ffffff;
    color: #1d4ed8;
    border-color: #bfdbfe;
    padding: 4px 9px;
    min-height: 20px;
}

QPushButton#BrowseButton:hover {
    background: #eff6ff;
    border-color: #2563eb;
}

QToolButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 3px;
    min-height: 20px;
}

QToolButton:hover {
    background: #eef2ff;
    border-color: #2563eb;
}

QToolButton:checked {
    background: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
}

QToolButton#ArrowButton {
    padding: 0;
    min-height: 26px;
}

QToolButton#ModeToggle {
    background: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 0;
    min-height: 24px;
    font-weight: 600;
}

QToolButton#ModeToggle:hover {
    background: #eff6ff;
    border-color: #93c5fd;
}

QToolButton#ModeToggle:checked {
    background: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
}

QToolButton#PageCard {
    background: #ffffff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 4px 14px;
    min-height: 26px;
    font-weight: 700;
}

QToolButton#PageCard:hover {
    background: #eff6ff;
    border-color: #2563eb;
}

QToolButton#PageCard:checked {
    background: #2563eb;
    color: #ffffff;
    border-color: #1d4ed8;
}

QRadioButton {
    background: transparent;
    spacing: 6px;
}

QRadioButton::indicator {
    width: 14px;
    height: 14px;
}

QCheckBox#InlineCheck {
    background: transparent;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 4px 7px;
    min-height: 20px;
    selection-background-color: #bfdbfe;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #2563eb;
}

QTableWidget, QListWidget {
    background: #ffffff;
    alternate-background-color: #f8fafc;
    border: 1px solid #d8dee8;
    border-radius: 5px;
    gridline-color: #e5e7eb;
}

QHeaderView::section {
    background: #eef2f7;
    color: #111827;
    border: 0;
    border-right: 1px solid #d8dee8;
    border-bottom: 1px solid #d8dee8;
    padding: 4px;
    font-weight: 600;
}

QTabWidget::pane {
    border: 1px solid #d8dee8;
    border-radius: 6px;
    background: #ffffff;
    padding: 0;
}

QTabBar::tab {
    background: #eef2f7;
    border: 1px solid #d8dee8;
    border-bottom: 0;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 5px 11px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #111827;
}

QSplitter::handle {
    background: #e5e7eb;
}
"""


def apply_application_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
