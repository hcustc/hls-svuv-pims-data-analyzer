from __future__ import annotations

from dataclasses import dataclass

from PyQt6 import QtGui, QtWidgets


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

QToolTip {{
    background: {t.text_primary};
    color: {t.text_inverse};
    border: 1px solid {t.text_primary};
    border-radius: 4px;
    padding: 4px 7px;
}}

QStatusBar {{
    background: {t.panel_bg};
    color: {t.text_secondary};
    border-top: 1px solid {t.border};
}}

QStatusBar::item {{ border: 0; }}

QLabel,
QCheckBox {{
    background: transparent;
}}

QWidget#ControlBar {{
    background: {t.panel_bg};
    border: 1px solid {t.border_light};
    border-radius: 10px;
}}

QWidget#PlotPanel {{
    background: {t.panel_bg};
    border: 1px solid {t.border_light};
    border-radius: 10px;
}}

QWidget#SidePanel {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 10px;
}}

QWidget#FittingConfigPanel,
QWidget#ResultDisplayPanel {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 10px;
}}

QFrame#PanelHeader,
QFrame#ResultPanelHeader {{
    background: {t.panel_alt};
    border: 0;
    border-bottom: 1px solid {t.border_light};
    border-radius: 4px;
}}

QFrame#PanelSeparator {{
    background: {t.border_light};
    border: 0;
    max-height: 1px;
}}

QLabel#PieMzBadge,
QLabel#FitModeBadge {{
    background: {t.primary_soft};
    border: 1px solid {t.primary_border};
    border-radius: 4px;
    color: {t.primary_hover};
    font-weight: 600;
    padding: 2px 6px;
}}

QLabel#FitModeBadge[mode="manual"] {{
    background: {t.warning_bg};
    border-color: {t.warning_border};
    color: {t.warning};
}}

QLabel#CandidateSummary,
QLabel#ResultMetrics {{
    background: transparent;
    color: {t.text_secondary};
    font-weight: 600;
}}

QFrame#SpeciesHoverCard {{
    background: {t.panel_bg};
    border: 1px solid {t.primary_border};
    border-radius: 8px;
}}

QLabel#HoverStructureCanvas {{
    background: {t.panel_subtle};
    border: 1px solid {t.border_light};
    border-radius: 6px;
    color: {t.text_muted};
    font-size: 9pt;
}}

QLabel#SpeciesHoverKicker {{
    color: {t.text_muted};
    font-size: 9pt;
    font-weight: 600;
}}

QLabel#SpeciesHoverName {{
    color: {t.text_primary};
    font-size: 11pt;
    font-weight: 600;
}}

QLabel#SpeciesHoverFormulaBadge {{
    background: {t.primary_soft};
    border: 1px solid {t.primary_border};
    border-radius: 4px;
    color: {t.primary_hover};
    font-weight: 600;
    padding: 2px 6px;
}}

QLabel#SpeciesHoverMeta,
QLabel#SpeciesHoverHint,
QLabel#SpeciesFormulaLabel {{
    color: {t.text_muted};
    font-size: 9pt;
}}

QLabel#PieWorkflowStage,
QLabel#PieSelectionSummary {{
    background: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 4px;
    color: {t.text_secondary};
    font-weight: 600;
    padding: 2px 7px;
}}

QLabel#PieWorkflowStage[status="active"] {{
    background: {t.primary_soft};
    border-color: {t.primary_border};
    color: {t.primary_hover};
}}

QLabel#PieWorkflowStage[status="busy"] {{
    background: #eef2ff;
    border-color: #a5b4fc;
    color: #4338ca;
}}

QLabel#PieWorkflowStage[status="warning"] {{
    background: {t.warning_bg};
    border-color: {t.warning_border};
    color: {t.warning};
}}

QLabel#PieWorkflowStage[status="complete"] {{
    background: #f0fdf4;
    border-color: #86efac;
    color: {t.success};
}}

QLabel#PieEmptyIcon {{
    background: transparent;
    color: {t.text_muted};
    font-size: 26px;
}}

QLabel#PieEmptyTitle {{
    background: transparent;
    color: {t.text_primary};
    font-size: 12pt;
    font-weight: 700;
}}

QFrame#AnalysisEmptyState,
QFrame#AnalysisProgressState {{
    background: {t.panel_subtle};
    border: 0;
    border-radius: 8px;
}}

QFrame#EmptyStateCard,
QFrame#ProgressStateCard {{
    background: transparent;
    border: 0;
}}

QLabel#EmptyStateTitle,
QLabel#ProgressStateTitle {{
    background: transparent;
    color: {t.text_primary};
    font-size: 14pt;
    font-weight: 700;
}}

QLabel#EmptyStateSubtitle,
QLabel#ProgressStateDetail {{
    background: transparent;
    color: {t.text_muted};
    font-size: 10pt;
}}

QProgressBar#AnalysisProgressBar {{
    min-height: 16px;
    max-height: 16px;
    font-size: 8pt;
}}

QPushButton#EmptyStateAction {{
    background: {t.primary};
    border: 1px solid {t.primary};
    border-radius: 6px;
    color: {t.text_inverse};
    font-weight: 700;
    padding: 5px 14px;
}}

QPushButton#EmptyStateAction:hover {{
    background: {t.primary_hover};
    border-color: {t.primary_hover};
    color: {t.text_inverse};
}}

QLabel#ResultStatus {{
    background: transparent;
    font-weight: 600;
}}

QLabel#ResultStatus[status="preview"] {{ color: #7c3aed; }}
QLabel#ResultStatus[status="completed"] {{ color: {t.primary_hover}; }}
QLabel#ResultStatus[status="obsolete"] {{ color: {t.warning}; }}
QLabel#ResultStatus[status="failed"] {{ color: {t.danger}; }}
QLabel#ResultStatus[status="confirmed"] {{ color: {t.success}; font-weight: 700; }}

QPushButton#LockCandidateButton {{
    background: {t.panel_alt};
    border: 1px solid {t.border};
    border-radius: 3px;
    color: {t.text_muted};
    padding: 0;
    min-height: 20px;
}}

QPushButton#LockCandidateButton:hover {{
    background: {t.panel_subtle};
    border-color: {t.text_muted};
}}

QPushButton#LockCandidateButton[locked="true"] {{
    background: #fbbf24;
    border-color: #f59e0b;
    color: #92400e;
    font-weight: 700;
}}

QPushButton#IconButton {{
    background: transparent;
    border: 1px solid transparent;
    color: {t.text_muted};
    padding: 0;
    min-height: 20px;
}}

QPushButton#IconButton:hover {{
    background: #fef2f2;
    border-color: #fecaca;
    color: {t.danger};
}}

QPushButton#ResultDetailLink {{
    background: transparent;
    border: 0;
    color: #0369a1;
    padding: 1px 4px;
    min-height: 16px;
    text-decoration: underline;
}}

QPushButton#ResultDetailLink:hover {{
    background: transparent;
    color: #0284c7;
}}

QWidget#TempToolbar {{
    background: {t.panel_bg};
    border-bottom: 1px solid {t.border};
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

QLabel#InlineStatusLabel[status="busy"] {{
    color: {t.primary_hover};
    font-weight: 600;
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

QLabel#ProjectBoundaryTitle {{
    background: transparent;
    color: {t.primary_hover};
    font-weight: 700;
}}

QFrame#ParentSelectionBanner {{
    background: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 6px;
}}

QLabel#ParentSelectionStatus {{
    background: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 5px;
    color: {t.text_secondary};
    padding: 4px 7px;
}}

QLabel#ParentSelectionStatus[selectionState="complete"] {{
    background: #ecfdf5;
    border-color: #a7f3d0;
    color: #047857;
}}

QLabel#ParentSelectionStatus[selectionState="active"] {{
    background: {t.primary_soft};
    border-color: {t.primary_border};
    color: {t.primary_hover};
}}

QLabel#ParentSelectionStatus[selectionState="warning"] {{
    background: {t.warning_bg};
    border-color: {t.warning_border};
    color: #a16207;
}}

QWidget#WorkspaceShell {{
    background: transparent;
}}

QWidget#PageNav {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 8px;
    min-height: 44px;
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

QWidget#ContextHeader {{
    background: transparent;
    border: 0;
}}

QWidget#SourceInputRow {{
    background: {t.panel_bg};
    border: 1px solid {t.border_light};
    border-radius: 6px;
}}

QWidget#SettingsStrip {{
    background: {t.panel_subtle};
    border: 1px solid {t.border_light};
    border-radius: 6px;
}}

QWidget#StatusActionStrip {{
    background: transparent;
    border: 0;
    border-left: 1px solid {t.border_light};
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

QLabel#ContextValue {{
    background: transparent;
    border: 0;
    color: {t.text_secondary};
    font-weight: 600;
    padding: 1px 2px;
}}

QLabel#CurveSourceBadge {{
    background: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 5px;
    color: {t.text_secondary};
    font-weight: 600;
    padding: 3px 7px;
}}

QLabel#CurveSourceBadge[sourceState="valid"] {{
    background: #f0fdf4;
    border-color: #86efac;
    color: #166534;
}}

QLabel#CurveSourceBadge[sourceState="memory"] {{
    background: {t.primary_soft};
    border-color: {t.primary_border};
    color: {t.primary_hover};
}}

QLabel#CurveSourceBadge[sourceState="warning"] {{
    background: {t.warning_bg};
    border-color: {t.warning_border};
    color: {t.warning};
}}

QGroupBox {{
    border: 1px solid {t.border};
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
    background: {t.panel_bg};
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {t.text_primary};
    background: {t.panel_bg};
}}

QWidget#InlineParameterPanel {{
    border: 1px solid {t.border_light};
    border-radius: 6px;
    background: {t.panel_bg};
    color: {t.text_secondary};
}}

QFrame#PanelFrame {{
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 8px;
}}

QPushButton {{
    background: {t.panel_bg};
    color: {t.text_body};
    border: 1px solid {t.border};
    border-radius: 5px;
    padding: 4px 10px;
    min-height: 20px;
}}

QPushButton:hover {{
    background: {t.primary_soft};
    color: {t.primary_hover};
    border-color: {t.primary_border};
}}

QPushButton:pressed {{
    background: {t.primary_softer};
}}

QPushButton:focus,
QToolButton:focus {{
    border: 2px solid {t.border_focus};
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

QPushButton#WorkflowButton:disabled,
QPushButton#BrowseButton:disabled,
QPushButton#PrimaryToolbarButton:disabled,
QPushButton#ExportButton:disabled,
QPushButton#WarningButton:disabled {{
    background: {t.disabled_bg};
    border-color: {t.disabled_bg};
    color: {t.disabled_text};
}}

QPushButton#PrimaryToolbarButton {{
    background: {t.primary};
    color: {t.text_inverse};
    border-color: {t.primary_hover};
    padding: 4px 12px;
    min-height: 20px;
    font-weight: 600;
}}

QPushButton#PrimaryButton {{
    background: {t.primary};
    color: {t.text_inverse};
    border-color: {t.primary_hover};
    padding: 4px 12px;
    font-weight: 700;
}}

QPushButton#PrimaryButton:hover {{
    background: {t.primary_hover};
}}

QPushButton#PrimaryButton:disabled {{
    background: {t.disabled_bg};
    border-color: {t.disabled_bg};
    color: {t.disabled_text};
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

QToolButton#CommandMenuButton {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 5px;
    color: {t.text_secondary};
    font-weight: 600;
    padding: 3px 10px;
    min-height: 22px;
}}

QToolButton#CommandMenuButton:hover {{
    background: {t.panel_subtle};
    border-color: {t.primary_border};
    color: {t.primary_hover};
}}

QToolButton#CommandMenuButton:disabled {{
    background: {t.disabled_bg};
    border-color: {t.disabled_bg};
    color: {t.disabled_text};
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

QToolButton#WorkspaceTab {{
    background: transparent;
    color: {t.text_secondary};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 26px;
    font-weight: 700;
    text-align: center;
}}

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

QToolButton#WorkspaceTab:checked {{
    background: {t.primary_softer};
    color: {t.primary_hover};
    border-color: {t.primary_border};
}}

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
    background: #ffffff;
}}

QSpinBox#TableCellEditor,
QDoubleSpinBox#TableCellEditor,
QComboBox#TableCellEditor {{
    border-radius: 3px;
    padding: 1px 5px;
    min-height: 0;
}}

QLineEdit#CurveSearch {{
    background: {t.panel_subtle};
}}

QTableWidget, QListWidget, QTreeWidget {{
    background: {t.panel_bg};
    alternate-background-color: {t.panel_subtle};
    border: 1px solid {t.border};
    border-radius: 5px;
    gridline-color: {t.gridline};
    selection-background-color: {t.selection};
    selection-color: {t.text_primary};
    outline: 0;
}}

QTableWidget::item {{ padding: 4px 6px; }}
QTableWidget::item:selected {{ background: {t.selection}; color: {t.text_primary}; }}

QTreeWidget::item, QListWidget::item {{
    padding: 4px 6px;
}}

QTreeWidget::item:selected, QListWidget::item:selected {{
    background: {t.selection};
    color: {t.text_primary};
}}

QWidget#TagCloud {{
    background: transparent;
}}

QFrame#ForceTag {{
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 4px;
    padding: 0;
}}

QLabel#ForceTagText {{
    background: transparent;
    color: #3730a3;
    font-size: 9pt;
}}

QPushButton#TagCloseButton {{
    background: transparent;
    border: 0;
    color: #6366f1;
    font-size: 8pt;
    padding: 0;
    min-height: 14px;
}}

QPushButton#TagCloseButton:hover {{
    color: {t.danger};
    font-weight: 700;
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

QScrollArea {{ border: 0; background: transparent; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #cbd5e1; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #94a3b8; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #cbd5e1; border-radius: 4px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: #94a3b8; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}

QProgressBar {{
    background: {t.panel_alt};
    border: 1px solid {t.border};
    border-radius: 5px;
    text-align: center;
    color: {t.text_primary};
    min-height: 18px;
}}

QProgressBar::chunk {{ background: {t.primary}; border-radius: 4px; }}

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
    font-weight: 600;
}}

QTabBar::tab:hover:!selected {{
    background: {t.primary_soft};
    color: {t.primary_hover};
}}

QFrame#NavSeparator {{
    background: {t.border};
    min-width: 1px;
    max-width: 1px;
    margin: 7px 8px;
}}

QSplitter#MainSplitter {{
    background: {t.panel_bg};
    border: 1px solid {t.border};
    border-radius: 10px;
}}

/* PIE uses one continuous three-column workbench. Repeating rounded cards on
   each column creates double borders and wide pinched gutters at the splitters. */
QSplitter#MainSplitter QWidget#SidePanel,
QSplitter#MainSplitter QWidget#PlotPanel,
QSplitter#MainSplitter QWidget#FittingConfigPanel,
QSplitter#MainSplitter QWidget#ResultDisplayPanel {{
    border: 0;
    border-radius: 0;
}}

QSplitter#MainSplitter::handle:horizontal {{
    background: {t.panel_bg};
    border: 0;
    border-left: 1px solid {t.border_light};
    width: 7px;
}}

QSplitter#MainSplitter::handle:horizontal:hover {{
    background: {t.primary_soft};
    border-left-color: {t.primary_border};
}}
"""


APP_QSS = _build_qss(LIGHT_THEME)


def apply_application_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font.setPointSizeF(10.0)
    app.setFont(font)
    app.setStyleSheet(APP_QSS)
