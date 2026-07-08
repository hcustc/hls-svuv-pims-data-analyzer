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
from bl03u_masstool.core.project_settings import ProjectSettings, ProjectSettingsManager
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
    parse_expansion_factors_from_result,
    save_mole_fraction_settings,
)
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread
from bl03u_masstool.frontends.pyqt_app.project_artifacts import record_project_artifact

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin
from bl03u_masstool.frontends.pyqt_app.common.static_plot import StaticCurvePlot

class TemperatureScanDialog(QtWidgets.QWidget, DataFrameTableMixin):
    # 曲线分类颜色映射
    CURVE_CLASS_COLORS = {
        "formation": "#10b981",      # 绿色 - 生成(升高)
        "consumption": "#ef4444",    # 红色 - 消耗(减少)
        "intermediate": "#f59e0b",   # 橙色 - 中间体(先升后降)
        "unclassified": "#6b7280",   # 灰色 - 暂未区分
    }

    def __init__(self, calibration: Calibration, normalization_settings: NormalizationSettings | None = None, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.normalization_settings = normalization_settings or NormalizationSettings()
        self.project_settings: ProjectSettings | None = None
        self.temperature_source_scope = "temporary"
        self._temporary_temperature_folder = ""
        self.peak_detection = load_peak_detection_config()
        self.result_df = pd.DataFrame()
        self.curves: dict[int, dict] = {}
        self.current_mz: int | None = None
        self.worker: WorkerThread | None = None
        self.setWindowTitle("温度扫描分析")
        self.resize(1280, 800)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ─────────────────────────────────────────────────────────
        toolbar = QtWidgets.QWidget()
        toolbar.setObjectName("TempToolbar")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(6)

        # Project info (read-only chip)
        self.summary_project_label = QtWidgets.QLabel("项目: ---")
        self.summary_system_label = QtWidgets.QLabel("体系: ---")
        self.summary_data_label = QtWidgets.QLabel("数据源: ---")
        for lbl in (self.summary_project_label, self.summary_system_label, self.summary_data_label):
            lbl.setObjectName("ReadoutValue")
        toolbar_layout.addWidget(self.summary_project_label)
        toolbar_layout.addWidget(self.summary_system_label)
        toolbar_layout.addWidget(self.summary_data_label, stretch=1)

        # Action buttons
        self.run_button = QtWidgets.QPushButton("开始分析")
        self.run_button.setObjectName("WorkflowButton")
        self.run_button.setToolTip("开始分析温度扫描数据")
        self.run_button.clicked.connect(self.run_analysis)

        self.export_button = QtWidgets.QPushButton("导出结果")
        self.export_button.setObjectName("ExportButton")
        self.export_button.setToolTip("导出分析结果为CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)

        self.export_plot_button = QtWidgets.QPushButton("导出图表")
        self.export_plot_button.setObjectName("ExportButton")
        self.export_plot_button.setToolTip("导出当前曲线图表为PNG/PDF")
        self.export_plot_button.setEnabled(False)
        self.export_plot_button.clicked.connect(self.export_plot)

        self.preview_data_button = QtWidgets.QPushButton("预览数据")
        self.preview_data_button.setObjectName("ExportButton")
        self.preview_data_button.setToolTip("预览全部积分结果")
        self.preview_data_button.setEnabled(False)
        self.preview_data_button.clicked.connect(self.show_data_preview)

        self.summary_open_project_btn = QtWidgets.QPushButton("项目管理")
        self.summary_open_project_btn.setObjectName("BrowseButton")
        self.summary_open_project_btn.setToolTip("在项目管理中修改数据源、寻峰、归一化和温度扫描默认参数")
        self.summary_open_project_btn.clicked.connect(self._open_project_settings)

        # Sidebar toggle
        self.sidebar_toggle_btn = QtWidgets.QPushButton("◀ 曲线")
        self.sidebar_toggle_btn.setObjectName("BrowseButton")
        self.sidebar_toggle_btn.setToolTip("展开/折叠曲线浏览")
        self.sidebar_toggle_btn.setCheckable(True)
        self.sidebar_toggle_btn.setChecked(True)
        self.sidebar_toggle_btn.clicked.connect(self._toggle_sidebar)

        for btn in (self.run_button, self.export_button, self.export_plot_button,
                    self.preview_data_button, self.summary_open_project_btn, self.sidebar_toggle_btn):
            toolbar_layout.addWidget(btn)

        # Inline status
        self.inline_status_icon = QtWidgets.QLabel("")
        self.inline_status_icon.setFixedWidth(20)
        self.inline_status_text = QtWidgets.QLabel("就绪")
        self.inline_status_text.setObjectName("InlineStatusLabel")
        self.inline_retry_button = QtWidgets.QPushButton("重试")
        self.inline_retry_button.setMaximumWidth(52)
        self.inline_retry_button.hide()
        self.inline_action_hint = QtWidgets.QLabel('在项目管理确认数据源和参数后，点击"开始分析"')
        self.inline_action_hint.setObjectName("ProjectHint")
        toolbar_layout.addWidget(self.inline_status_icon)
        toolbar_layout.addWidget(self.inline_status_text)
        toolbar_layout.addWidget(self.inline_action_hint)
        toolbar_layout.addWidget(self.inline_retry_button)
        root.addWidget(toolbar)

        # thin separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setObjectName("NavSeparator")
        root.addWidget(sep)

        # ── Temperature Scan Parameters ────────────────────────────────────
        params_bar = QtWidgets.QWidget()
        params_layout = QtWidgets.QHBoxLayout(params_bar)
        params_layout.setContentsMargins(8, 4, 8, 4)
        params_layout.setSpacing(12)

        params_label = QtWidgets.QLabel("温度扫描参数：")
        params_label.setObjectName("ReadoutLabel")
        params_layout.addWidget(params_label)

        self.project_source_button = QtWidgets.QToolButton()
        self.project_source_button.setText("项目数据")
        self.project_source_button.setObjectName("ModeToggle")
        self.project_source_button.setCheckable(True)
        self.project_source_button.setToolTip("使用项目管理中登记的温度扫描数据源")
        self.temporary_source_button = QtWidgets.QToolButton()
        self.temporary_source_button.setText("临时数据")
        self.temporary_source_button.setObjectName("ModeToggle")
        self.temporary_source_button.setCheckable(True)
        self.temporary_source_button.setToolTip("只为本次温度扫描分析选择数据，不写回项目配置")
        self.source_scope_group = QtWidgets.QButtonGroup(self)
        self.source_scope_group.setExclusive(True)
        self.source_scope_group.addButton(self.project_source_button, 0)
        self.source_scope_group.addButton(self.temporary_source_button, 1)
        self.source_scope_group.idClicked.connect(
            lambda button_id: self.set_temperature_source_scope("project" if button_id == 0 else "temporary")
        )
        source_scope_panel = QtWidgets.QWidget()
        source_scope_panel.setObjectName("ModeSegment")
        source_scope_layout = QtWidgets.QHBoxLayout(source_scope_panel)
        source_scope_layout.setContentsMargins(0, 0, 0, 0)
        source_scope_layout.setSpacing(3)
        source_scope_layout.addWidget(self.project_source_button)
        source_scope_layout.addWidget(self.temporary_source_button)
        params_layout.addWidget(source_scope_panel)

        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("选择温度扫描数据文件夹")
        self.folder_edit.textChanged.connect(self._remember_temporary_source)
        params_layout.addWidget(self.folder_edit, stretch=1)

        self.select_folder_button = QtWidgets.QPushButton("浏览...")
        self.select_folder_button.setObjectName("BrowseButton")
        self.select_folder_button.setToolTip("选择温度扫描数据文件夹")
        self.select_folder_button.clicked.connect(self.select_folder)
        params_layout.addWidget(self.select_folder_button)

        self.temperature_photon_check = QtWidgets.QCheckBox("光强归一化")
        self.temperature_photon_check.setToolTip("使用项目管理中设置的光强来源归一化温度扫描信号")
        self.temperature_kr_check = QtWidgets.QCheckBox("Kr 校正")
        self.temperature_kr_check.setToolTip("使用项目管理中计算的 Kr 膨胀系数 λ(T) 校正温度扫描信号")
        self.temperature_kr_check.toggled.connect(self._on_temperature_kr_toggled)
        self.light_source_status_label = QtWidgets.QLabel("光强来源: IO")
        self.light_source_status_label.setObjectName("ReadoutValue")
        self.integration_method_label = QtWidgets.QLabel("积分方式: 范围累加")
        self.integration_method_label.setObjectName("ReadoutValue")
        self.integration_method_label.setToolTip("温度扫描积分方式由项目管理 -> 功能默认参数 -> 温度扫描设置")
        params_layout.addWidget(self.temperature_photon_check)
        params_layout.addWidget(self.temperature_kr_check)
        params_layout.addWidget(self.light_source_status_label)
        params_layout.addWidget(self.integration_method_label)
        self.replicate_enabled_check = QtWidgets.QCheckBox("合并重复采集")
        self.replicate_enabled_check.setToolTip("仅在确认为同一条件多次采集时开启；关闭时每个文件按自身条件处理")
        self.replicate_enabled_check.toggled.connect(self._on_replicate_enabled_changed)
        params_layout.addWidget(self.replicate_enabled_check)
        self.replicate_mode_combo = QtWidgets.QComboBox()
        self.replicate_mode_combo.addItem("平均", "mean")
        self.replicate_mode_combo.addItem("累加", "sum")
        self.replicate_mode_combo.setToolTip("开启合并重复采集后，同组重复采集的聚合方式")
        self.replicate_mode_combo.setEnabled(False)
        params_layout.addWidget(self.replicate_mode_combo)
        params_layout.addStretch(1)

        root.addWidget(params_bar)
        self.set_temperature_source_scope("temporary", restore_saved=False)

        # ── Body: sidebar + main area ────────────────────────────────────────
        body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        body_splitter.setObjectName("MainSplitter")
        root.addWidget(body_splitter, stretch=1)

        # ── Left sidebar ─────────────────────────────────────────────────────
        self._sidebar = QtWidgets.QWidget()
        self._sidebar.setObjectName("SidePanel")
        self._sidebar.setFixedWidth(240)
        sidebar_layout = QtWidgets.QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(8)

        # Curve filter
        curve_browser_title = QtWidgets.QLabel("结果曲线")
        curve_browser_title.setObjectName("ReadoutLabel")
        sidebar_layout.addWidget(curve_browser_title)

        self.summary_label = QtWidgets.QLabel("未生成温度曲线")
        self.summary_label.setObjectName("ProjectHint")
        self.summary_label.setWordWrap(True)
        sidebar_layout.addWidget(self.summary_label)

        self.curve_filter_edit = QtWidgets.QLineEdit()
        self.curve_filter_edit.setObjectName("CurveSearch")
        self.curve_filter_edit.setPlaceholderText("搜索 m/z / 物种 / 分类")
        self.curve_filter_edit.textChanged.connect(self.populate_mz_list)
        sidebar_layout.addWidget(self.curve_filter_edit)

        filter_row = QtWidgets.QWidget()
        filter_layout = QtWidgets.QVBoxLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(4)
        disp_row = QtWidgets.QHBoxLayout()
        disp_row.addWidget(QtWidgets.QLabel("显示"))
        self.curve_display_combo = QtWidgets.QComboBox()
        self.curve_display_combo.addItem("按类别分组", "grouped")
        self.curve_display_combo.addItem("按m/z排序", "mz")
        self.curve_display_combo.currentIndexChanged.connect(self.populate_mz_list)
        disp_row.addWidget(self.curve_display_combo, stretch=1)
        filter_layout.addLayout(disp_row)
        filt_row = QtWidgets.QHBoxLayout()
        filt_row.addWidget(QtWidgets.QLabel("筛选"))
        self.curve_group_combo = QtWidgets.QComboBox()
        self.curve_group_combo.addItem("全部", "all")
        for key in ("formation", "consumption", "intermediate", "unclassified"):
            self.curve_group_combo.addItem(TEMPERATURE_CURVE_CLASS_LABELS[key], key)
        self.curve_group_combo.currentIndexChanged.connect(self.populate_mz_list)
        filt_row.addWidget(self.curve_group_combo, stretch=1)
        filter_layout.addLayout(filt_row)
        sidebar_layout.addWidget(filter_row)

        self.group_summary_label = QtWidgets.QLabel("")
        self.group_summary_label.setObjectName("ProjectHint")
        self.group_summary_label.setWordWrap(True)
        sidebar_layout.addWidget(self.group_summary_label)

        self.mz_list = QtWidgets.QTreeWidget()
        self.mz_list.setHeaderHidden(True)
        self.mz_list.setRootIsDecorated(True)
        self.mz_list.setUniformRowHeights(True)
        self.mz_list.currentItemChanged.connect(self.on_mz_selected)
        sidebar_layout.addWidget(self.mz_list, stretch=1)

        body_splitter.addWidget(self._sidebar)

        # ── Right main area ──────────────────────────────────────────────────
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)
        body_splitter.addWidget(right_widget)
        body_splitter.setSizes([240, 1040])
        body_splitter.setCollapsible(0, True)
        body_splitter.setCollapsible(1, False)

        # Plot area with empty-state overlay
        self._plot_container = QtWidgets.QWidget()
        self._plot_container.setObjectName("PlotPanel")
        plot_stack = QtWidgets.QStackedLayout(self._plot_container)

        # Empty state widget
        self._empty_state = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(self._empty_state)
        empty_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_icon = QtWidgets.QLabel("[数据]")
        empty_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_icon.setStyleSheet("font-size: 48px;")
        empty_msg = QtWidgets.QLabel('尚未生成温度扫描曲线\n\n在项目管理确认数据源和参数后，点击"开始分析"')
        empty_msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_msg.setObjectName("ProjectHint")
        empty_msg.setWordWrap(True)
        empty_layout.addWidget(empty_icon)
        empty_layout.addWidget(empty_msg)
        plot_stack.addWidget(self._empty_state)

        self.plot_widget = StaticCurvePlot("温度 / °C", "归一化信号")
        plot_stack.addWidget(self.plot_widget)

        plot_stack.setCurrentIndex(0)  # show empty state initially
        right_layout.addWidget(self._plot_container, stretch=3)

        curve_stats = QtWidgets.QFrame()
        curve_stats.setObjectName("StatsBar")
        curve_stats_layout = QtWidgets.QHBoxLayout(curve_stats)
        curve_stats_layout.setContentsMargins(10, 2, 10, 2)
        curve_stats_layout.setSpacing(12)
        curve_stats_title = QtWidgets.QLabel("当前曲线")
        curve_stats_title.setObjectName("StatsTitle")
        self.current_curve_label = QtWidgets.QLabel("未选择")
        self.current_curve_label.setObjectName("HintLabel")
        self.current_curve_metric_label = QtWidgets.QLabel("生成曲线后可在左侧选择 m/z")
        self.current_curve_metric_label.setObjectName("HintLabel")
        curve_stats_layout.addWidget(curve_stats_title)
        curve_stats_layout.addWidget(self.current_curve_label)
        curve_stats_layout.addWidget(self.current_curve_metric_label, stretch=1)

        # Toggle table visibility button
        self.toggle_table_button = QtWidgets.QPushButton("查看数据")
        self.toggle_table_button.setCheckable(True)
        self.toggle_table_button.setToolTip("点击显示/隐藏下方数据表格")
        self.toggle_table_button.setFixedWidth(90)
        self.toggle_table_button.setFixedHeight(26)
        self.toggle_table_button.setObjectName("BrowseButton")
        self.table_visible = False  # Initially hidden
        self.toggle_table_button.clicked.connect(self._toggle_table_visibility)
        curve_stats_layout.addWidget(self.toggle_table_button)
        right_layout.addWidget(curve_stats)

        # Bottom tabs (can be toggled) - initially hidden
        self.detail_tabs = QtWidgets.QTabWidget()
        self.detail_tabs.setObjectName("PeakResultTabs")
        self.detail_tabs.setVisible(False)  # Start hidden
        self.curve_table = QtWidgets.QTableWidget()
        self.curve_table.setWordWrap(False)
        self.curve_table.setAlternatingRowColors(True)
        self.curve_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.detail_tabs.addTab(self.curve_table, "当前曲线")
        right_layout.addWidget(self.detail_tabs, stretch=2)

    def _open_project_settings(self):
        """跳转到项目管理页面。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("project")

    def _toggle_sidebar(self, checked: bool) -> None:
        self._sidebar.setVisible(checked)
        self.sidebar_toggle_btn.setText("◀ 曲线" if checked else "▶ 曲线")

    def _toggle_table_visibility(self) -> None:
        """Toggle the visibility of the data tables and adjust layout."""
        self.table_visible = not self.table_visible
        self.detail_tabs.setVisible(self.table_visible)
        self.toggle_table_button.setText("隐藏数据" if self.table_visible else "查看数据")
        # Force layout recalculation to adjust the plot area
        parent = self.detail_tabs.parentWidget()
        if parent and parent.layout():
            parent.layout().invalidate()
            parent.layout().activate()
            parent.update()

    def _show_plot(self) -> None:
        """Switch plot container from empty state to the actual plot."""
        stack = self._plot_container.layout()
        if stack is not None and stack.count() > 1:
            stack.setCurrentIndex(1)

    def _has_project_scope(self) -> bool:
        return self.temperature_source_scope == "project"

    def _has_project_context(self) -> bool:
        if self.project_settings is None:
            return False
        return bool(
            self.project_settings.project_name
            or self.project_settings.temperature_scan_folder
            or (self.project_settings.output_dir and self.project_settings.output_dir != "output")
        )

    def _remember_temporary_source(self, _text: str | None = None) -> None:
        if self.temperature_source_scope != "temporary":
            return
        self._temporary_temperature_folder = self.folder_edit.text().strip()

    def set_temperature_source_scope(
        self,
        scope: str,
        *,
        restore_saved: bool = True,
        apply_project: bool = True,
    ) -> None:
        scope = "project" if scope == "project" else "temporary"
        previous_scope = getattr(self, "temperature_source_scope", "temporary")
        if previous_scope == "temporary" and scope == "project":
            self._remember_temporary_source()
        self.temperature_source_scope = scope
        if hasattr(self, "project_source_button"):
            self.project_source_button.setChecked(scope == "project")
        if hasattr(self, "temporary_source_button"):
            self.temporary_source_button.setChecked(scope == "temporary")
        if scope == "project":
            if apply_project and self.project_settings is not None:
                self.folder_edit.setText(self.project_settings.temperature_scan_folder or "")
        elif previous_scope == "project" and restore_saved:
            self.folder_edit.setText(self._temporary_temperature_folder)
        self._refresh_temperature_source_controls()

    def _refresh_temperature_source_controls(self) -> None:
        use_project = self._has_project_scope()
        has_project_path = bool(self.project_settings and self.project_settings.temperature_scan_folder)
        if hasattr(self, "project_source_button"):
            self.project_source_button.setChecked(use_project)
            self.project_source_button.setEnabled(self._has_project_context())
        if hasattr(self, "temporary_source_button"):
            self.temporary_source_button.setChecked(not use_project)
        self.folder_edit.setReadOnly(use_project)
        self.folder_edit.setClearButtonEnabled(not use_project)
        self.folder_edit.setPlaceholderText(
            "项目未登记温度扫描数据源" if use_project else "选择温度扫描数据文件夹"
        )
        self.select_folder_button.setText("浏览...")
        self.select_folder_button.setVisible(not use_project)
        if use_project and not has_project_path:
            self._show_inline_empty("请先在项目管理中配置温度扫描文件夹")

    def select_folder(self) -> None:
        if self._has_project_scope():
            self._open_project_settings()
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择温度扫描数据文件夹")
        if folder:
            self.folder_edit.setText(folder)

    def _current_temperature_folder(self) -> str:
        if self._has_project_scope():
            return (self.project_settings.temperature_scan_folder if self.project_settings else "").strip()
        return self.folder_edit.text().strip()

    def _refresh_normalization_status(self) -> None:
        ps = self.project_settings or ProjectSettings()
        light_source = "IO" if self.normalization_settings.light_source == "io" else "Beam Current"
        self.temperature_photon_check.blockSignals(True)
        self.temperature_photon_check.setChecked(bool(ps.temperature_photon_normalize))
        self.temperature_photon_check.blockSignals(False)
        self.temperature_kr_check.blockSignals(True)
        self.temperature_kr_check.setChecked(bool(ps.temperature_kr_correct))
        self.temperature_kr_check.blockSignals(False)
        self.light_source_status_label.setText(f"光强来源: {light_source}")
        self.integration_method_label.setText(f"积分方式: {self._integration_method_label(ps.temp_integration_method)}")

    @staticmethod
    def _integration_method_label(method: str | None) -> str:
        return {
            "sum_counts": "范围累加",
            "baseline": "扣基线积分",
            "gaussian": "高斯",
            "mixed": "混合",
        }.get(str(method or "sum_counts"), "范围累加")

    def _on_temperature_kr_toggled(self, checked: bool) -> None:
        if not checked or self.normalization_settings.expansion_factors:
            return
        self.temperature_kr_check.blockSignals(True)
        self.temperature_kr_check.setChecked(False)
        self.temperature_kr_check.blockSignals(False)
        self._show_inline_error("尚未计算 Kr 膨胀系数 λ(T)，已退回为不使用 Kr 校正。请先到项目管理 -> 通用参数计算。")

    def set_project_settings(self, ps: ProjectSettings, *, activate_project_scope: bool | None = None) -> None:
        """Keep project context visible while project-owned parameters stay in 项目管理."""
        has_project_context = bool(
            ps.project_name
            or ps.temperature_scan_folder
            or (ps.output_dir and ps.output_dir != "output")
        )
        if activate_project_scope is None:
            activate_project_scope = self._has_project_scope() and has_project_context
        self.project_settings = ps

        if hasattr(self, "summary_project_label"):
            project_name = ps.project_name or "---"
            system = ps.system or "---"
            self.summary_project_label.setText(f"项目: {project_name}")
            self.summary_system_label.setText(f"体系: {system}")

        if activate_project_scope:
            self.set_temperature_source_scope("project", restore_saved=False, apply_project=False)
        elif self._has_project_scope() and not has_project_context:
            self.set_temperature_source_scope("temporary")

        if self._has_project_scope():
            self.folder_edit.setText(ps.temperature_scan_folder or "")
        data_path = self._current_temperature_folder() or "---"

        # 加载温度扫描参数
        if hasattr(self, "replicate_mode_combo"):
            mode = ps.temp_replicate_mode if ps.temp_replicate_mode in {"mean", "sum"} else "off"
            self.replicate_enabled_check.setChecked(mode != "off")
            idx = self.replicate_mode_combo.findData(mode if mode != "off" else "mean")
            if idx >= 0:
                self.replicate_mode_combo.setCurrentIndex(idx)
            self._on_replicate_enabled_changed(mode != "off")
            self.summary_data_label.setText(
                f"{'项目温扫' if self._has_project_scope() else '临时温扫'}: {data_path}"
            )
        self._refresh_normalization_status()

        self._refresh_temperature_source_controls()
        if self._current_temperature_folder():
            self._show_inline_empty('项目参数已同步，点击"开始分析"')
        else:
            self._show_inline_empty(
                "请先在项目管理中配置温度扫描文件夹"
                if self._has_project_scope()
                else "请选择温度扫描数据文件夹"
            )

    def _show_inline_error(self, msg: str, retry_callback=None):
        self.inline_status_icon.setText("\u26a0\ufe0f")
        self.inline_status_text.setText(msg)
        self._set_inline_status("error")
        self.inline_action_hint.hide()
        if retry_callback:
            self.inline_retry_button.show()
            try:
                self.inline_retry_button.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            self.inline_retry_button.clicked.connect(retry_callback)
        else:
            self.inline_retry_button.hide()

    def _show_inline_success(self, msg: str):
        self.inline_status_icon.setText("\u2705")
        self.inline_status_text.setText(msg)
        self._set_inline_status("success")
        self.inline_action_hint.hide()
        self.inline_retry_button.hide()
        QtCore.QTimer.singleShot(5000, self._clear_inline_status)

    def _show_inline_empty(self, msg: str = ""):
        self.inline_status_icon.setText("")
        self.inline_status_text.setText("就绪")
        self._set_inline_status("")
        if msg:
            self.inline_action_hint.setText(msg)
            self.inline_action_hint.show()
        else:
            self.inline_action_hint.hide()
        self.inline_retry_button.hide()

    def _clear_inline_status(self):
        self.inline_status_icon.setText("")
        self.inline_status_text.setText("就绪")
        self._set_inline_status("")
        self.inline_action_hint.hide()
        self.inline_retry_button.hide()

    def _set_inline_status(self, status: str) -> None:
        self.inline_status_text.setProperty("status", status)
        self.inline_status_text.style().unpolish(self.inline_status_text)
        self.inline_status_text.style().polish(self.inline_status_text)

    def open_common_parameters(self):
        self._open_project_settings()

    def run_analysis(self):
        # Get parameters from project settings (single source of truth)
        ps = self.project_settings or ProjectSettings()

        # 温度扫描页保存本功能的分析开关；项目管理只维护所需参数/资源。
        if ps:
            ps.temp_replicate_mode = self._current_replicate_mode()
            ps.temperature_photon_normalize = self.temperature_photon_check.isChecked()
            ps.temperature_kr_correct = self.temperature_kr_check.isChecked()

        folder = self._current_temperature_folder()
        if not folder:
            self._show_inline_error(
                "请在项目管理中配置温度扫描文件夹"
                if self._has_project_scope()
                else "请选择温度扫描数据文件夹"
            )
            return

        # If folder has no direct .txt files but has subdirectories, ask user to pick a subfolder
        folder_path = Path(folder)
        has_direct_txt = any(p.is_file() and p.suffix.lower() == ".txt" for p in folder_path.iterdir())
        if not has_direct_txt and any(p.is_dir() for p in folder_path.iterdir()):
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self, "选择包含 .txt 文件的子文件夹", folder
            )
            if not selected:
                return
            folder = selected

        peak_config = ps.to_peak_detection_config() if self.project_settings else load_peak_detection_config()
        threshold_end = peak_config.threshold_end
        min_intensity = peak_config.min_intensity
        reference_mode = ps.temp_reference_mode
        integration_method = ps.temp_integration_method
        prefer_gaussian = integration_method == "gaussian"

        # Auto-switch to manual peak detection if peak file is set
        effective_peak_source = ps.temp_peak_source
        if self._has_project_scope() and ps.manual_peak_file and ps.temp_peak_source == "auto":
            effective_peak_source = "manual"

        # Get manual peak file if using manual peak detection
        manual_peak_path = None
        if self._has_project_scope() and effective_peak_source == "manual":
            manual_peak_path = ps.manual_peak_file
            if not manual_peak_path:
                self._show_inline_error("请在项目管理中配置手动卡峰文件")
                return

        settings = self.normalization_settings
        photon_normalize = ps.temperature_photon_normalize
        kr_correct = ps.temperature_kr_correct
        kr_mz = ps.kr_mz
        mass_discrimination = 1.0
        light_source = settings.light_source
        expansion_factors = settings.expansion_factors if kr_correct else None
        if kr_correct and not expansion_factors:
            self._show_inline_error(
                "已勾选 Kr 校正，但尚未计算 λ(T)，本次分析已退回。请先到 项目管理 -> 通用参数 计算 Kr 膨胀系数。"
            )
            self.temperature_kr_check.blockSignals(True)
            self.temperature_kr_check.setChecked(False)
            self.temperature_kr_check.blockSignals(False)
            ps.temperature_kr_correct = False
            if self._has_project_scope():
                self._save_project_switches()
            return
        self._save_project_switches()
        self.set_busy(True, "正在分析温度扫描数据...")
        self.worker = WorkerThread(
            lambda: analyze_temperature_folder(
                folder,
                calibration=self.calibration,
                algorithm=peak_config.algorithm,
                threshold_end=threshold_end,
                min_intensity=min_intensity,
                detection_min_idx=peak_config.detection_min_idx,
                nearby_peak_window=peak_config.nearby_peak_window,
                duplicate_window=peak_config.duplicate_window,
                weak_tail_early_window=peak_config.weak_tail_early_window,
                weak_tail_late_window=peak_config.weak_tail_late_window,
                weak_tail_ratio=peak_config.weak_tail_ratio,
                gaussian_window_max=peak_config.gaussian_window_max,
                gaussian_boundary_scale=peak_config.gaussian_boundary_scale,
                boundary_padding=peak_config.boundary_padding,
                prominence_ratio=peak_config.prominence_ratio,
                smoothing_window=peak_config.smoothing_window,
                smoothing_poly_order=peak_config.smoothing_poly_order,
                baseline_window=peak_config.baseline_window,
                baseline_percentile=peak_config.baseline_percentile,
                min_peak_width=peak_config.min_peak_width,
                max_peak_width=peak_config.max_peak_width,
                prefer_gaussian=prefer_gaussian,
                integration_method=integration_method,
                reference_mode=reference_mode,
                manual_peak_path=manual_peak_path,
                photon_normalize=photon_normalize,
                kr_correct=kr_correct,
                kr_mz=kr_mz,
                mass_discrimination=mass_discrimination,
                light_source=light_source,
                expansion_factors=expansion_factors,
                vote_threshold=peak_config.vote_threshold,
                min_intensity_for_single_vote=peak_config.min_intensity_for_single_vote,
                mz_tolerance=peak_config.mz_tolerance,
                cwt_snr_threshold=peak_config.cwt_snr_threshold,
                cwt_wavelet_max_width=peak_config.cwt_wavelet_max_width,
                weak_tail_cutoff_idx=peak_config.weak_tail_cutoff_idx,
                temp_curve_class_change_threshold=ps.temp_curve_class_change_threshold,
                temp_curve_class_peak_fraction=ps.temp_curve_class_peak_fraction,
                replicate_mode=self._current_replicate_mode(),
            ),
            self,
        )
        self.worker.finished_with_result.connect(self.on_analysis_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def compute_kr_expansion(self):
        # Get parameters from project settings (single source of truth)
        ps = self.project_settings or ProjectSettings()
        folder = self._current_temperature_folder()
        if not folder:
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                "请在项目管理中配置温度扫描文件夹"
                if self._has_project_scope()
                else "请选择温度扫描数据文件夹",
            )
            return

        peak_config = ps.to_peak_detection_config() if self.project_settings else load_peak_detection_config()
        threshold_end = peak_config.threshold_end
        min_intensity = peak_config.min_intensity
        reference_mode = ps.temp_reference_mode
        integration_method = ps.temp_integration_method
        prefer_gaussian = integration_method == "gaussian"

        # Auto-switch to manual peak detection if peak file is set
        effective_peak_source = ps.temp_peak_source
        if self._has_project_scope() and ps.manual_peak_file and ps.temp_peak_source == "auto":
            effective_peak_source = "manual"

        # Get manual peak file if using manual peak detection
        manual_peak_path = None
        if self._has_project_scope() and effective_peak_source == "manual":
            manual_peak_path = ps.manual_peak_file
            if not manual_peak_path:
                QtWidgets.QMessageBox.warning(self, "提示", "请在项目管理中配置手动卡峰文件")
                return

        settings = self.normalization_settings
        light_source = settings.light_source
        kr_mz = ps.kr_mz
        self.set_busy(True, "正在计算 Kr 膨胀系数...")
        self.worker = WorkerThread(
            lambda: compute_kr_expansion_factors(
                folder,
                calibration=self.calibration,
                kr_mz=kr_mz,
                manual_peak_path=manual_peak_path,
                light_source=light_source,
                threshold_end=threshold_end,
                min_intensity=min_intensity,
                prefer_gaussian=prefer_gaussian,
                integration_method=integration_method,
                reference_mode=reference_mode,
                detection_min_idx=peak_config.detection_min_idx,
                nearby_peak_window=peak_config.nearby_peak_window,
                duplicate_window=peak_config.duplicate_window,
                weak_tail_early_window=peak_config.weak_tail_early_window,
                weak_tail_late_window=peak_config.weak_tail_late_window,
                weak_tail_ratio=peak_config.weak_tail_ratio,
                gaussian_window_max=peak_config.gaussian_window_max,
                gaussian_boundary_scale=peak_config.gaussian_boundary_scale,
                boundary_padding=peak_config.boundary_padding,
            ),
            self,
        )
        self.worker.finished_with_result.connect(self.on_kr_compute_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def on_kr_compute_complete(self, result: pd.DataFrame) -> None:
        settings = self.normalization_settings
        expansion_factors = parse_expansion_factors_from_result(result)
        settings.expansion_factors = expansion_factors
        save_normalization_settings(settings)
        if self._has_project_scope() and self.project_settings is not None:
            self.project_settings.expansion_factors = dict(expansion_factors)
            manager = ProjectSettingsManager()
            manager.set(self.project_settings)
            manager.save()
        self._refresh_normalization_status()

        if "photon_energy" in result.columns and result["photon_energy"].nunique(dropna=True) > 1:
            point_text = (
                f"{result['photon_energy'].nunique(dropna=True)} 个能量 × "
                f"{result['temperature'].nunique()} 个温度点"
            )
        else:
            point_text = f"{len(expansion_factors)} 个温度点"
        QtWidgets.QMessageBox.information(
            self, "完成",
            "成功计算 Kr 膨胀系数！\n"
            f"参考温度: {result['reference_temperature'].iloc[0]:.1f}°C\n"
            f"共 {point_text}\n\n"
            "如需在本次温度扫描中使用，请勾选顶部的 Kr 校正后重新分析。",
        )
        if self.worker is not None:
            try:
                self.worker.finished.disconnect()
            except (RuntimeError, TypeError):
                pass

    def _save_project_switches(self) -> None:
        if not self._has_project_scope() or self.project_settings is None:
            return
        manager = ProjectSettingsManager()
        if manager.has_project_path():
            manager.set(self.project_settings)
            manager.save()

    def _on_replicate_enabled_changed(self, checked: bool) -> None:
        self.replicate_mode_combo.setEnabled(checked)

    def _current_replicate_mode(self) -> str:
        if not self.replicate_enabled_check.isChecked():
            return "off"
        return str(self.replicate_mode_combo.currentData() or "mean")

    def set_busy(self, busy: bool, message: str) -> None:
        self.run_button.setDisabled(busy)
        self.summary_open_project_btn.setDisabled(busy)
        # export only available when there are results and not busy
        self.export_button.setEnabled(not busy and not self.result_df.empty)
        if busy:
            self.inline_status_icon.setText("")
            self.inline_status_text.setText(message)
            self._set_inline_status("busy")
            self.inline_action_hint.hide()
            self.inline_retry_button.hide()

    def on_analysis_complete(self, result: object) -> None:
        self.result_df = result
        self.curves = build_temperature_curves(self.result_df)
        self.populate_mz_list()
        temperature_count = self.result_df["temperature"].nunique() if not self.result_df.empty else 0
        replicate_note = self._replicate_status_text(self.result_df)
        integration_note = self._integration_status_text(self.result_df)
        self.summary_label.setText(
            f"{len(self.curves)} 条m/z曲线 | {temperature_count} 个温度点"
            + (f" | {replicate_note}" if replicate_note else "")
            + (f" | {integration_note}" if integration_note else "")
        )
        self.update_group_summary()
        self._show_plot()  # reveal plot, hide empty state
        self.export_button.setEnabled(True)
        self.export_plot_button.setEnabled(True)
        self.preview_data_button.setEnabled(True)
        success = f"已完成分析，生成 {len(self.result_df)} 行温度扫描结果"
        if replicate_note:
            success = f"{success}；{replicate_note}"
        if integration_note:
            success = f"{success}；{integration_note}"
        self._show_inline_success(success)

    @staticmethod
    def _replicate_status_text(df: pd.DataFrame) -> str:
        if df.empty or "file_count" not in df:
            return ""
        max_count = int(pd.to_numeric(df["file_count"], errors="coerce").fillna(1).max())
        if max_count <= 1:
            return ""
        warnings = []
        if "replicate_warning" in df:
            warnings = sorted({str(value) for value in df["replicate_warning"].dropna() if str(value)})
        if warnings:
            return warnings[0]
        mode = str(df["replicate_mode"].dropna().iloc[0]) if "replicate_mode" in df and not df["replicate_mode"].dropna().empty else "mean"
        if mode == "off":
            return ""
        mode_text = "平均" if mode == "mean" else "累加"
        grouping = str(df["replicate_grouping"].dropna().iloc[0]) if "replicate_grouping" in df and not df["replicate_grouping"].dropna().empty else ""
        if grouping == "filename":
            return f"按文件名识别重复采集，已{mode_text}"
        return f"检测到重复采集，已按旧逻辑{mode_text}"

    @staticmethod
    def _integration_status_text(df: pd.DataFrame) -> str:
        if df.empty or "integration_method" not in df:
            return ""
        methods = df["integration_method"].dropna().astype(str)
        gaussian_count = int((methods == "gaussian").sum())
        baseline_count = int((methods == "baseline").sum())
        sum_counts_count = int((methods == "sum_counts").sum())
        mixed_count = int((methods == "mixed").sum())
        parts = []
        if sum_counts_count:
            parts.append(f"范围累加 {sum_counts_count} 点")
        if gaussian_count:
            parts.append(f"高斯 {gaussian_count} 点")
        if baseline_count:
            parts.append(f"扣基线积分 {baseline_count} 点")
        if mixed_count:
            parts.append(f"混合 {mixed_count} 点")
        return "积分方式: " + "，".join(parts) if parts else ""

    def on_analysis_failed(self, message: str) -> None:
        self._show_inline_error(f"分析失败: {message}", lambda: self.run_analysis())

    def populate_mz_list(self):
        self.mz_list.clear()
        selected_group = self.curve_group_combo.currentData() if hasattr(self, "curve_group_combo") else "all"
        display_mode = self.curve_display_combo.currentData() if hasattr(self, "curve_display_combo") else "grouped"
        if display_mode == "mz":
            self.populate_mz_list_flat(selected_group)
        else:
            self.populate_mz_list_grouped(selected_group)
        first_item = self.first_curve_tree_item()
        if first_item is not None:
            self.mz_list.setCurrentItem(first_item)
        else:
            self.on_mz_selected(None)

    def populate_mz_list_grouped(self, selected_group: str):
        for group_key in ("formation", "consumption", "intermediate", "unclassified"):
            if selected_group != "all" and selected_group != group_key:
                continue
            items = [
                (mz, self.curves[mz])
                for mz in sorted(self.curves)
                if self.curves[mz].get("curve_class", "unclassified") == group_key
                and self.curve_matches_filter(mz, self.curves[mz])
            ]
            if not items:
                continue
            parent = QtWidgets.QTreeWidgetItem([f"{TEMPERATURE_CURVE_CLASS_LABELS[group_key]} ({len(items)}条)"])
            parent.setData(0, QtCore.Qt.ItemDataRole.UserRole, None)
            parent.setFlags(parent.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            # 设置分类组的颜色
            color = self.CURVE_CLASS_COLORS.get(group_key, "#6b7280")
            parent.setForeground(0, QtGui.QColor(color))
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            self.mz_list.addTopLevelItem(parent)
            for mz, curve in items:
                child = QtWidgets.QTreeWidgetItem([self.curve_tree_label(mz, curve, include_group=False)])
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, mz)
                # 设置每条曲线的颜色
                child.setForeground(0, QtGui.QColor(color))
                parent.addChild(child)
            parent.setExpanded(True)

    def populate_mz_list_flat(self, selected_group: str):
        for mz in sorted(self.curves):
            curve = self.curves[mz]
            if selected_group != "all" and curve.get("curve_class") != selected_group:
                continue
            if not self.curve_matches_filter(mz, curve):
                continue
            item = QtWidgets.QTreeWidgetItem([self.curve_tree_label(mz, curve, include_group=True)])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, mz)
            # 设置曲线颜色
            group_key = curve.get("curve_class", "unclassified")
            color = self.CURVE_CLASS_COLORS.get(group_key, "#6b7280")
            item.setForeground(0, QtGui.QColor(color))
            self.mz_list.addTopLevelItem(item)

    def curve_matches_filter(self, mz: int, curve: dict) -> bool:
        query = self.curve_filter_edit.text().strip().lower() if hasattr(self, "curve_filter_edit") else ""
        if not query:
            return True
        haystack = " ".join(
            str(value)
            for value in (
                mz,
                curve.get("species", ""),
                curve.get("curve_class", ""),
                curve.get("curve_class_label", ""),
            )
        ).lower()
        return query in haystack

    def curve_tree_label(self, mz: int, curve: dict, *, include_group: bool) -> str:
        label = curve.get("species") or ""
        suffix = f" {label}" if label and label != "Unknown" else ""
        base = f"{mz}{suffix}  ({len(curve['temperatures'])}点)"
        if include_group:
            class_label = curve.get("curve_class_label", TEMPERATURE_CURVE_CLASS_LABELS["unclassified"])
            return f"[{class_label}] {base}"
        return base

    def first_curve_tree_item(self):
        for index in range(self.mz_list.topLevelItemCount()):
            item = self.mz_list.topLevelItem(index)
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) is not None:
                return item
            if item.childCount() > 0:
                return item.child(0)
        return None

    def update_group_summary(self):
        counts = {key: 0 for key in TEMPERATURE_CURVE_CLASS_LABELS}
        for curve in self.curves.values():
            counts[curve.get("curve_class", "unclassified")] = counts.get(curve.get("curve_class", "unclassified"), 0) + 1
        parts = [
            f"{TEMPERATURE_CURVE_CLASS_LABELS[key]} {counts.get(key, 0)}"
            for key in ("formation", "consumption", "intermediate", "unclassified")
        ]
        self.group_summary_label.setText(" | ".join(parts))

    def on_mz_selected(self, current, previous=None):
        if current is None:
            self.current_mz = None
            self.curve_table.clear()
            self.curve_table.setRowCount(0)
            self.curve_table.setColumnCount(0)
            self.current_curve_label.setText("未选择")
            self.current_curve_metric_label.setText("调整筛选或重新生成曲线")
            if self.plot_widget is not None:
                self.plot_widget.clear_plot(title="未选择温度曲线")
            return
        mz_value = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if mz_value is None:
            if current.childCount() > 0:
                self.mz_list.setCurrentItem(current.child(0))
            return
        self.current_mz = int(mz_value)
        curve = self.curves[self.current_mz]
        rows = curve["rows"].copy()
        integration_methods = rows.get("integration_method", pd.Series([""] * len(rows))).astype(str).map(
            {"sum_counts": "范围累加", "gaussian": "高斯", "baseline": "扣基线积分", "mixed": "混合"}
        ).fillna("")
        curve_df = pd.DataFrame(
            {
                "温度(C)": np.round(rows["temperature"].astype(float), 1),
                "积分方式": integration_methods,
                "原始积分": np.round(rows["raw_area"].astype(float), 4),
                "IO归一化": np.round(rows["photon_normalized_area"].astype(float), 4),
                "λ(T)": np.round(rows["expansion_lambda"].astype(float), 6),
                "最终强度": np.round(rows["area"].astype(float), 4),
            }
        )
        # 转置表格，使温度点成为列头
        curve_df_transposed = curve_df.set_index("温度(C)").T
        self.set_dataframe(self.curve_table, curve_df_transposed)
        class_label = curve.get("curve_class_label", "")
        self.current_curve_label.setText(f"m/z {self.current_mz}")

        # 改进状态栏：添加温度范围和最大值信息
        temperatures = np.asarray(curve["temperatures"], dtype=float)
        areas = np.asarray(curve["areas"], dtype=float)
        valid = np.isfinite(temperatures) & np.isfinite(areas)
        temperatures = temperatures[valid]
        areas = areas[valid]

        if len(temperatures) > 0:
            t_min = float(np.nanmin(temperatures))
            t_max = float(np.nanmax(temperatures))
            max_idx = int(np.nanargmax(areas))
            max_value = float(areas[max_idx])
            t_at_max = float(temperatures[max_idx])
            status_text = (
                f"{class_label} | {len(curve['temperatures'])} 个点 | "
                f"{t_min:.0f}–{t_max:.0f} °C | 最大值 {max_value:.2f} @ {t_at_max:.0f} °C"
            )
        else:
            status_text = f"{class_label} | {len(curve['temperatures'])} 个温度点"

        self.current_curve_metric_label.setText(status_text)
        self.update_plot(curve)

    def update_plot(self, curve: dict):
        if self.plot_widget is None:
            return
        x_values = np.asarray(curve["temperatures"], dtype=float)
        y_values = np.asarray(curve["areas"], dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid]
        y_values = y_values[valid]

        # 简化标题：仅显示对象和图表类型，分类信息在状态栏
        title = f"m/z {curve['mz']} 温度响应曲线"
        # 改为中文坐标轴标签
        self.plot_widget.clear_plot(title=title, xlabel="温度 / °C", ylabel="归一化信号")

        if x_values.size == 0:
            self.plot_widget.show_empty("无有效数据", title=f"m/z {curve['mz']} 温度曲线")
            return

        # 根据分类获取颜色
        group_key = curve.get("curve_class", "unclassified")
        color = self.CURVE_CLASS_COLORS.get(group_key, "#6b7280")

        # 改进曲线绘制：线宽 2.2、标记点 6.0、白色描边（已内置），避免粘连
        plot_x, plot_y = self.plot_widget.plot_series(
            x_values,
            y_values,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=6.0,
        )

        # 优化纵轴范围：减少顶部无效留白
        x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
        y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1e-6)

        x_pad = 0.025 * x_span  # 2.5% 边距
        y_pad = max(0.08 * y_span, 0.03 * max(abs(y_max), 1.0))  # 8% 或最小 3%

        # 手动设置数据范围避免留白过大
        if self.plot_widget.axes is not None:
            self.plot_widget.axes.set_xlim(x_min - x_pad, x_max + x_pad)
            self.plot_widget.axes.set_ylim(y_min - y_pad, y_max + y_pad)

            # 添加零基线（如果有物理意义）
            if y_min < 0 < y_max:
                self.plot_widget.axes.axhline(
                    0.0,
                    color="#94A3B8",
                    linewidth=1.0,
                    linestyle=(0, (4, 4)),
                    alpha=0.55,
                    zorder=0,
                )

        self.plot_widget.finish()

    def run_analysis_sync(
        self,
        folder: str,
        threshold_end: float,
        min_intensity: float,
        *,
        prefer_gaussian: bool = True,
        integration_method: str = "sum_counts",
        reference_mode: str = "sum",
        manual_peak_path: str | None = None,
        photon_normalize: bool = False,
        kr_correct: bool = False,
        mass_discrimination: float = 1.0,
        light_source: str = "io",
        expansion_factors: dict[float, float] | None = None,
        temp_curve_class_change_threshold: float = 0.25,
        temp_curve_class_peak_fraction: float = 0.65,
        replicate_mode: str = "off",
    ) -> pd.DataFrame:
        peak_config = (
            self.project_settings.to_peak_detection_config()
            if self.project_settings
            else load_peak_detection_config()
        )
        return analyze_temperature_folder(
            folder,
            calibration=self.calibration,
            threshold_end=threshold_end,
            min_intensity=min_intensity,
            detection_min_idx=peak_config.detection_min_idx,
            nearby_peak_window=peak_config.nearby_peak_window,
            duplicate_window=peak_config.duplicate_window,
            weak_tail_early_window=peak_config.weak_tail_early_window,
            weak_tail_late_window=peak_config.weak_tail_late_window,
            weak_tail_ratio=peak_config.weak_tail_ratio,
            gaussian_window_max=peak_config.gaussian_window_max,
            gaussian_boundary_scale=peak_config.gaussian_boundary_scale,
            boundary_padding=peak_config.boundary_padding,
            prefer_gaussian=prefer_gaussian,
            integration_method=integration_method,
            reference_mode=reference_mode,
            manual_peak_path=manual_peak_path,
            photon_normalize=photon_normalize,
            kr_correct=kr_correct,
            mass_discrimination=mass_discrimination,
            light_source=light_source,
            expansion_factors=expansion_factors,
            temp_curve_class_change_threshold=temp_curve_class_change_threshold,
            temp_curve_class_peak_fraction=temp_curve_class_peak_fraction,
            replicate_mode=replicate_mode,
        )

    def export_result(self):
        if self.result_df.empty:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可导出的结果")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出温度扫描结果",
            str(ensure_output_dir("exports", "temperature") / "temperature_scan.xlsx"),
            "Excel Files (*.xlsx);;CSV Files (*.csv)",
        )
        if not path:
            return
        if path.endswith(".xlsx"):
            self.result_df.to_excel(path, index=False)
        else:
            self.result_df.to_csv(path, index=False, encoding="utf-8-sig")
        if self._has_project_scope():
            record_project_artifact(
                self,
                "temperature_scan_result_file",
                path,
                message="温度扫描结果已登记到项目管理",
            )
            message = "温度扫描结果已导出并登记到项目管理。"
        else:
            message = "温度扫描结果已导出。当前为临时数据模式，结果未登记到项目管理。"
        QtWidgets.QMessageBox.information(self, "成功", message)

    def export_plot(self):
        if self.plot_widget is None or self.plot_widget.figure is None:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可导出的图表")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出温度扫描曲线图",
            str(ensure_output_dir("exports", "temperature") / "temperature_scan_plot.png"),
            "PNG Images (*.png);;PDF Files (*.pdf)",
        )
        if not path:
            return
        if self.plot_widget.save_plot(path):
            if self._has_project_scope():
                record_project_artifact(
                    self,
                    "temperature_scan_plot_file",
                    path,
                    message="温度扫描曲线图已登记到项目管理",
                )
            QtWidgets.QMessageBox.information(self, "成功", f"曲线图已导出：{path}")
        else:
            QtWidgets.QMessageBox.warning(self, "错误", "导出图表失败")

    def show_data_preview(self):
        """打开独立窗口预览全部积分结果表格"""
        if self.result_df.empty:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可预览的数据")
            return

        # 创建非模态预览窗口
        preview_win = QtWidgets.QMainWindow()
        preview_win.setWindowTitle("温度扫描数据预览")
        preview_win.resize(1000, 600)

        # 创建表格
        preview_table = QtWidgets.QTableWidget()
        preview_table.setWordWrap(False)
        preview_table.setAlternatingRowColors(True)
        preview_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.set_dataframe(preview_table, self.result_df)

        # 设置为中心widget
        preview_win.setCentralWidget(preview_table)

        # 非模态显示
        preview_win.show()
        # 保持窗口引用，防止被垃圾回收
        if not hasattr(self, '_preview_windows'):
            self._preview_windows = []
        self._preview_windows.append(preview_win)
