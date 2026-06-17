from __future__ import annotations

from dataclasses import dataclass

from PyQt6 import QtWidgets


@dataclass(frozen=True)
class ThemeTokens:
    app_bg: str = "#f6f8fb"
    panel_bg: str = "#ffffff"
    panel_subtle: str = "#f8fafc"
    panel_alt: str = "#eef2f7"
    border: str = "#d8dee8"
    border_light: str = "#e2e8f0"
    border_focus: str = "#2563eb"
    text_primary: str = "#111827"
    text_body: str = "#1f2937"
    text_secondary: str = "#475569"
    text_muted: str = "#64748b"
    text_inverse: str = "#ffffff"
    primary: str = "#2563eb"
    primary_hover: str = "#1d4ed8"
    primary_pressed: str = "#1e40af"
    primary_soft: str = "#eff6ff"
    primary_softer: str = "#dbeafe"
    primary_border: str = "#93c5fd"
    success: str = "#16a34a"
    success_hover: str = "#15803d"
    warning: str = "#b45309"
    warning_bg: str = "#fefce8"
    warning_border: str = "#fde047"
    warning_hover: str = "#fef9c3"
    danger: str = "#dc2626"
    disabled_bg: str = "#cbd5e1"
    disabled_text: str = "#64748b"
    gridline: str = "#e5e7eb"
    selection: str = "#bfdbfe"


@dataclass(frozen=True)
class PlotTheme:
    background: str = "#ffffff"
    foreground: str = "#475569"
    grid: str = "#cbd5e1"
    spectrum_curve: str = "#2563eb"
    fit_curve: str = "#dc2626"
    secondary_curve: str = "#f97316"
    hover_line: str = "#94a3b8"
    region_brush: tuple[int, int, int, int] = (100, 116, 139, 45)


LIGHT_THEME = ThemeTokens()
PLOT_THEME = PlotTheme()


def get_plot_theme() -> PlotTheme:
    return PLOT_THEME


def _build_qss(t: ThemeTokens) -> str:
    return f"""
QWidget {{
    font-family: "PingFang SC", "Helvetica Neue", "Microsoft YaHei UI", Arial, sans-serif;
    font-size: 10pt;
    color: {t.text_body};
    background: {t.app_bg};
}}

QMainWindow, QDialog {{
    background: {t.app_bg};
}}

QLabel,
QCheckBox {{
    background: transparent;
}}

QWidget#ControlBar {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 8px;
}}

QWidget#PlotPanel,
QWidget#SidePanel {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 6px;
}}

QFrame#ProjectCard {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 6px;
}}

QFrame#ProjectHero {{
    background: {t.panel_bg};
    border: 1px solid {t.selection};
    border-radius: 8px;
}}

QWidget#ProjectActionBar {{
    background: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 6px;
}}

QLabel#ProjectHint,
QLabel#HintLabel {{
    background: transparent;
    color: {t.text_muted};
}}

QLabel#ProjectDirtyStatus {{
    background: transparent;
    color: {t.warning};
    font-weight: 600;
}}

QLabel#InlineStatusLabel {{
    background: transparent;
    color: {t.text_muted};
}}

QLabel#InlineStatusLabel[status="error"] {{
    color: {t.danger};
}}

QLabel#InlineStatusLabel[status="success"] {{
    color: {t.success};
}}

QLabel#ProjectTitle {{
    background: transparent;
    color: {t.text_primary};
    font-size: 11pt;
    font-weight: 700;
}}

QLabel#ProjectStatus {{
    background: transparent;
    color: {t.text_secondary};
    font-weight: 600;
}}

QLabel#ProjectParamSummary {{
    background: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 5px;
    color: #334155;
    padding: 4px 7px;
}}

QWidget#PageNav {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 8px;
}}

QWidget#ProjectPage {{
    background: transparent;
}}

QStackedWidget#WorkspaceStack {{
    background: transparent;
    border: 0;
}}

QWidget#CalibrationPanel {{
    background: {t.panel_subtle};
    border: 1px solid {t.border_light};
    border-radius: 6px;
}}

QWidget#SpectrumToolbarBody,
QWidget#SourcePanel,
QStackedWidget#SourceStack,
QWidget#ModePanel,
QWidget#ModeSegment,
QWidget#ArrowGroup {{
    background: transparent;
    border: 0;
}}

QLabel#ToolbarSectionTitle {{
    background: transparent;
    color: #334155;
    font-weight: 700;
    padding: 0 2px;
}}

QWidget#ToolbarGroup {{
    background: {t.panel_bg};
    border: 1px solid {t.border_light};
    border-radius: 6px;
}}

QLabel#ModeTitle {{
    background: transparent;
    color: {t.text_secondary};
    font-weight: 600;
}}

QWidget#PeakSummary,
QFrame#StatsBar {{
    background: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 6px;
}}

QFrame#StatsBar {{
    padding: 4px 10px;
}}

QLabel#StatsTitle {{
    background: transparent;
    color: {t.text_secondary};
    font-weight: 600;
}}

QLabel#ReadoutLabel {{
    background: transparent;
    color: {t.text_secondary};
    font-weight: 600;
}}

QLabel#ReadoutValue {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 4px;
    color: {t.text_primary};
    font-weight: 600;
    padding: 3px 6px;
}}

QGroupBox {{
    border: 1px solid {t.border};
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
}}

QFrame#PanelFrame {{
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 8px;
}}

QPushButton {{
    background: {t.primary};
    color: {t.text_inverse};
    border: 1px solid {t.primary_hover};
    border-radius: 5px;
    padding: 4px 10px;
    min-height: 20px;
}}

QPushButton:hover {{
    background: {t.primary_hover};
}}

QPushButton:pressed {{
    background: {t.primary_pressed};
}}

QPushButton:disabled {{
    background: {t.disabled_bg};
    border-color: {t.disabled_bg};
    color: {t.disabled_text};
}}

QPushButton#WorkflowButton,
QPushButton#BrowseButton {{
    background: {t.panel_bg};
    color: {t.primary_hover};
    border-color: {t.selection};
    padding: 4px 9px;
    min-height: 20px;
}}

QPushButton#WorkflowButton {{
    padding: 3px 8px;
}}

QPushButton#WorkflowButton:hover,
QPushButton#BrowseButton:hover {{
    background: {t.primary_soft};
    border-color: {t.primary};
}}

QPushButton#PrimaryToolbarButton {{
    background: {t.primary};
    color: {t.text_inverse};
    border-color: {t.primary_hover};
    padding: 4px 12px;
    min-height: 20px;
    font-weight: 600;
}}

QPushButton#PrimaryToolbarButton:hover {{
    background: {t.primary_hover};
}}

QPushButton#ExportButton {{
    background: {t.success};
    color: {t.text_inverse};
    border: 1px solid {t.success_hover};
    border-radius: 5px;
    padding: 4px 10px;
}}

QPushButton#ExportButton:hover {{
    background: {t.success_hover};
}}

QPushButton#WarningButton {{
    background: {t.warning_bg};
    color: #a16207;
    border: 1px solid {t.warning_border};
    border-radius: 5px;
}}

QPushButton#WarningButton:hover {{
    background: {t.warning_hover};
    border-color: #eab308;
}}

QToolButton {{
    background: {t.panel_bg};
    border: 1px solid {t.disabled_bg};
    border-radius: 5px;
    padding: 3px;
    min-height: 20px;
}}

QToolButton:hover {{
    background: #eef2ff;
    border-color: {t.primary};
}}

QToolButton:checked {{
    background: {t.primary};
    border-color: {t.primary_hover};
    color: {t.text_inverse};
}}

QToolButton#ArrowButton {{
    padding: 0;
    min-height: 26px;
}}

QToolButton#ModeToggle {{
    background: {t.panel_bg};
    color: {t.text_body};
    border: 1px solid {t.disabled_bg};
    border-radius: 5px;
    padding: 0;
    min-height: 24px;
    font-weight: 600;
}}

QToolButton#ModeToggle:hover {{
    background: {t.primary_soft};
    border-color: {t.primary_border};
}}

QToolButton#ModeToggle:checked {{
    background: {t.primary};
    border-color: {t.primary_hover};
    color: {t.text_inverse};
}}

QToolButton#WorkspaceTab,
QToolButton#PageCard {{
    background: transparent;
    color: {t.text_secondary};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 12px;
    min-height: 24px;
    font-weight: 700;
}}

QToolButton#WorkspaceTab:hover,
QToolButton#PageCard:hover {{
    background: {t.primary_soft};
    color: {t.primary_hover};
    border-color: {t.primary_softer};
}}

QToolButton#WorkspaceTab:checked,
QToolButton#PageCard:checked {{
    background: {t.primary_softer};
    color: {t.primary_hover};
    border-color: {t.primary_border};
}}

QRadioButton {{
    background: transparent;
    spacing: 6px;
}}

QRadioButton::indicator {{
    width: 14px;
    height: 14px;
}}

QCheckBox#InlineCheck {{
    background: transparent;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {t.panel_bg};
    border: 1px solid {t.disabled_bg};
    border-radius: 5px;
    padding: 4px 7px;
    min-height: 20px;
    selection-background-color: {t.selection};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {t.border_focus};
}}

QTableWidget, QListWidget {{
    background: {t.panel_bg};
    alternate-background-color: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 5px;
    gridline-color: {t.gridline};
}}

QHeaderView::section {{
    background: {t.panel_alt};
    color: {t.text_primary};
    border: 0;
    border-right: 1px solid {t.border};
    border-bottom: 1px solid {t.border};
    padding: 4px;
    font-weight: 600;
}}

QTabWidget::pane {{
    border: 1px solid {t.border};
    border-radius: 6px;
    background: {t.panel_bg};
    padding: 0;
}}

QTabWidget#ProjectTabs::pane,
QTabWidget#PeakResultTabs::pane {{
    border: 1px solid {t.border};
    border-radius: 6px;
    background: {t.panel_bg};
    padding: 0;
}}

QTabWidget#SourceModeStateTabs::pane {{
    border: 0;
    background: transparent;
}}

QTabBar::tab {{
    background: {t.panel_alt};
    border: 1px solid {t.border};
    border-bottom: 0;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 5px 11px;
    margin-right: 2px;
}}

QTabWidget#ProjectTabs QTabBar::tab,
QTabWidget#PeakResultTabs QTabBar::tab {{
    background: {t.panel_alt};
    border: 1px solid {t.border};
    border-bottom: 0;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 5px 11px;
    margin-right: 2px;
}}

QTabBar::tab:selected,
QTabWidget#ProjectTabs QTabBar::tab:selected,
QTabWidget#PeakResultTabs QTabBar::tab:selected {{
    background: {t.panel_bg};
    color: {t.text_primary};
}}

QSplitter::handle {{
    background: {t.gridline};
}}
"""


APP_QSS = _build_qss(LIGHT_THEME)


def apply_application_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
