from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re

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
    identify_product_energy_intervals,
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
from bl03u_masstool.frontends.pyqt_app.temporary_analysis_settings import (
    TEMPORARY_SETTINGS_SOURCES,
    TemporaryAnalysisSettingsDialog,
    build_temporary_settings,
)

from bl03u_masstool.frontends.pyqt_app.common.widgets import (
    AnalysisEmptyState,
    AnalysisProgressState,
    DataFrameTableMixin,
)
from bl03u_masstool.frontends.pyqt_app.common.plot_spec import (
    CurveSeries,
    ScientificPlotSpec,
    SeriesRole,
)
from bl03u_masstool.frontends.pyqt_app.common.static_plot import StaticCurvePlot


logger = logging.getLogger(__name__)


class TemperatureScanDialog(QtWidgets.QWidget, DataFrameTableMixin):
    ALL_ENERGY_FOLDERS = "__all_energy_folders__"
    CACHE_VERSION = 2

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
        self._global_calibration = deepcopy(self.calibration)
        self._global_normalization_settings = deepcopy(self.normalization_settings)
        self.project_settings: ProjectSettings | None = None
        self.temporary_settings: ProjectSettings | None = None
        self.temporary_settings_source = "global"
        self.temporary_settings_modified = False
        self._temporary_settings_source_explicit = False
        self.temperature_source_scope = "temporary"
        self._temporary_temperature_folder = ""
        self.peak_detection = load_peak_detection_config()
        self.result_df = pd.DataFrame()
        self.curves: dict[int, dict] = {}
        self.energy_results: list[dict] = []
        self.current_mz: int | None = None
        self.worker: WorkerThread | None = None
        self._analysis_request_id = 0
        self.energy_interval_result: dict | None = None
        self._busy = False
        self._selected_temperature_folder = ""
        self._autoload_worker: WorkerThread | None = None
        self._autoload_cache_token: dict | None = None
        self._autoload_request_id = 0
        self._folder_options_cache: dict[tuple[str, int | None], list[tuple[str, str]]] = {}
        self.setWindowTitle("温度扫描分析")
        self.resize(1280, 800)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Compact workflow/data source bar ────────────────────────────────
        source_panel = QtWidgets.QWidget()
        source_panel.setObjectName("ControlBar")
        data_layout = QtWidgets.QVBoxLayout(source_panel)
        data_layout.setContentsMargins(8, 8, 8, 8)
        data_layout.setSpacing(6)

        self.summary_bar = QtWidgets.QWidget()
        summary_layout = QtWidgets.QHBoxLayout(self.summary_bar)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(6)
        self.summary_project_label = QtWidgets.QLabel("项目: ---")
        self.summary_system_label = QtWidgets.QLabel("体系: ---")
        self.summary_data_label = QtWidgets.QLabel("数据源: ---")
        for lbl in (self.summary_project_label, self.summary_system_label, self.summary_data_label):
            lbl.setObjectName("ContextValue")
            lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_layout.addWidget(self.summary_project_label)
        summary_layout.addWidget(self.summary_system_label)
        summary_layout.addWidget(self.summary_data_label, stretch=1)
        self.workflow_stage_label = QtWidgets.QLabel("步骤 1/3 · 选择温度扫描数据")
        self.workflow_stage_label.setObjectName("PieWorkflowStage")
        summary_layout.addWidget(self.workflow_stage_label)
        data_layout.addWidget(self.summary_bar)

        source_row = QtWidgets.QHBoxLayout()
        source_row.setSpacing(6)
        source_label = QtWidgets.QLabel("温扫数据")
        source_label.setObjectName("ReadoutLabel")
        source_row.addWidget(source_label)

        self.project_source_button = QtWidgets.QToolButton()
        self.project_source_button.setText("项目数据")
        self.project_source_button.setObjectName("ModeToggle")
        self.project_source_button.setCheckable(True)
        self.project_source_button.setMinimumWidth(66)
        self.project_source_button.setToolTip("使用项目管理中登记的温度扫描数据源")
        self.temporary_source_button = QtWidgets.QToolButton()
        self.temporary_source_button.setText("临时数据")
        self.temporary_source_button.setObjectName("ModeToggle")
        self.temporary_source_button.setCheckable(True)
        self.temporary_source_button.setMinimumWidth(66)
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
        source_scope_panel.setMinimumWidth(136)
        source_scope_layout = QtWidgets.QHBoxLayout(source_scope_panel)
        source_scope_layout.setContentsMargins(0, 0, 0, 0)
        source_scope_layout.setSpacing(3)
        source_scope_layout.addWidget(self.project_source_button)
        source_scope_layout.addWidget(self.temporary_source_button)
        source_row.addWidget(source_scope_panel)

        self.temporary_params_label = QtWidgets.QLabel("当前来源：全局参数副本")
        self.temporary_params_label.setObjectName("ContextValue")
        # Kept as a non-layout compatibility control for tests and callers that
        # select a source by data.  The visible UI uses an action menu whose
        # wording makes it explicit that a session snapshot is being rebuilt.
        self.temporary_params_combo = QtWidgets.QComboBox()
        for source, label in TEMPORARY_SETTINGS_SOURCES:
            self.temporary_params_combo.addItem(label, source)
        self.temporary_params_combo.setCurrentIndex(
            self.temporary_params_combo.findData(self.temporary_settings_source)
        )
        self.temporary_params_combo.hide()
        self.temporary_params_combo.currentIndexChanged.connect(self._on_temporary_settings_source_changed)
        self.temporary_params_button = QtWidgets.QPushButton("编辑本次参数…")
        self.temporary_params_button.setObjectName("BrowseButton")
        self.temporary_params_button.clicked.connect(self._edit_temporary_settings)
        self.temporary_params_reload_button = QtWidgets.QToolButton()
        self.temporary_params_reload_button.setText("重新载入")
        self.temporary_params_reload_button.setObjectName("CommandMenuButton")
        self.temporary_params_reload_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.temporary_params_menu = QtWidgets.QMenu(self.temporary_params_reload_button)
        self.copy_project_params_action = self.temporary_params_menu.addAction("从当前项目复制")
        self.copy_global_params_action = self.temporary_params_menu.addAction("从全局配置复制")
        self.restore_default_params_action = self.temporary_params_menu.addAction("恢复程序默认值")
        self.copy_project_params_action.triggered.connect(
            lambda: self._select_temporary_settings_source("project")
        )
        self.copy_global_params_action.triggered.connect(
            lambda: self._select_temporary_settings_source("global")
        )
        self.restore_default_params_action.triggered.connect(
            lambda: self._select_temporary_settings_source("default")
        )
        self.temporary_params_reload_button.setMenu(self.temporary_params_menu)

        self.temporary_params_panel = QtWidgets.QGroupBox("临时数据参数")
        temporary_params_layout = QtWidgets.QHBoxLayout(self.temporary_params_panel)
        temporary_params_layout.setContentsMargins(10, 8, 10, 8)
        temporary_params_layout.setSpacing(10)
        temporary_params_layout.addWidget(self.temporary_params_label)
        temporary_params_layout.addWidget(self.temporary_params_button)
        temporary_params_layout.addWidget(self.temporary_params_reload_button)
        self.temporary_params_hint = QtWidgets.QLabel("仅用于当前临时数据，不会修改项目或全局配置")
        self.temporary_params_hint.setObjectName("HintLabel")
        temporary_params_layout.addWidget(self.temporary_params_hint)
        temporary_params_layout.addStretch(1)

        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("选择温度扫描数据文件夹")
        self.folder_edit.textChanged.connect(self._on_temperature_source_path_changed)
        source_row.addWidget(self.folder_edit, stretch=1)

        self.select_folder_button = QtWidgets.QPushButton("浏览...")
        self.select_folder_button.setObjectName("BrowseButton")
        self.select_folder_button.setToolTip("选择温度扫描数据文件夹")
        self.select_folder_button.clicked.connect(self.select_folder)
        source_row.addWidget(self.select_folder_button)

        self.scan_folder_label = QtWidgets.QLabel("能量范围")
        self.scan_folder_label.setObjectName("ReadoutLabel")
        self.scan_folder_combo = QtWidgets.QComboBox()
        self.scan_folder_combo.setMinimumWidth(170)
        self.scan_folder_combo.setToolTip("选择要查看的能量范围；生成曲线时会汇总项目中所有有效能量目录")
        self.scan_folder_combo.currentIndexChanged.connect(self._on_temperature_folder_option_changed)
        source_row.addWidget(self.scan_folder_label)
        source_row.addWidget(self.scan_folder_combo)

        self.run_button = QtWidgets.QPushButton("生成 / 刷新曲线")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.setToolTip("按当前数据源和分析开关生成温度扫描曲线")
        self.run_button.clicked.connect(self.run_analysis)

        self.summary_open_project_btn = QtWidgets.QPushButton("项目管理")
        self.summary_open_project_btn.setObjectName("BrowseButton")
        self.summary_open_project_btn.setToolTip("在项目管理中修改数据源、寻峰、归一化和温度扫描默认参数")
        self.summary_open_project_btn.clicked.connect(self._open_project_settings)
        source_row.addWidget(self.summary_open_project_btn)
        source_row.addWidget(self.run_button)
        data_layout.addLayout(source_row)
        data_layout.addWidget(self.temporary_params_panel)

        analysis_options_row = QtWidgets.QHBoxLayout()
        analysis_options_row.setSpacing(8)
        analysis_label = QtWidgets.QLabel("本次分析")
        analysis_label.setObjectName("ReadoutLabel")
        analysis_options_row.addWidget(analysis_label)
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
        analysis_options_row.addWidget(self.temperature_photon_check)
        analysis_options_row.addWidget(self.temperature_kr_check)
        analysis_options_row.addWidget(self.light_source_status_label)
        analysis_options_row.addWidget(self.integration_method_label)
        self.replicate_enabled_check = QtWidgets.QCheckBox("合并重复采集")
        self.replicate_enabled_check.setToolTip("仅在确认为同一条件多次采集时开启；关闭时每个文件按自身条件处理")
        self.replicate_enabled_check.toggled.connect(self._on_replicate_enabled_changed)
        analysis_options_row.addWidget(self.replicate_enabled_check)
        self.replicate_mode_combo = QtWidgets.QComboBox()
        self.replicate_mode_combo.addItem("平均", "mean")
        self.replicate_mode_combo.addItem("累加", "sum")
        self.replicate_mode_combo.setToolTip("开启合并重复采集后，同组重复采集的聚合方式")
        self.replicate_mode_combo.setEnabled(False)
        analysis_options_row.addWidget(self.replicate_mode_combo)
        analysis_options_row.addStretch(1)

        result_label = QtWidgets.QLabel("状态")
        result_label.setObjectName("ReadoutLabel")
        self.inline_status_icon = QtWidgets.QLabel("")
        self.inline_status_icon.setFixedWidth(12)
        self.inline_status_text = QtWidgets.QLabel("就绪")
        self.inline_status_text.setObjectName("InlineStatusLabel")
        self.inline_status_text.setMaximumWidth(300)
        self.inline_status_text.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.inline_retry_button = QtWidgets.QPushButton("重试")
        self.inline_retry_button.setObjectName("BrowseButton")
        self.inline_retry_button.setMaximumWidth(52)
        self.inline_retry_button.hide()
        self.inline_action_hint = QtWidgets.QLabel('确认数据源后点击"生成曲线"')
        self.inline_action_hint.setObjectName("ProjectHint")

        self.export_button = QtWidgets.QToolButton()
        self.export_button.setText("结果操作")
        self.export_button.setObjectName("CommandMenuButton")
        self.export_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.result_menu = QtWidgets.QMenu(self.export_button)
        self.preview_data_action = self.result_menu.addAction("预览全部积分数据")
        self.preview_data_action.triggered.connect(self.show_data_preview)
        self.result_menu.addSeparator()
        self.export_result_action = self.result_menu.addAction("导出分析结果…")
        self.export_result_action.triggered.connect(self.export_result)
        self.export_plot_action = self.result_menu.addAction("导出当前图表…")
        self.export_plot_action.triggered.connect(self.export_plot)
        self.export_button.setMenu(self.result_menu)
        self.export_button.setToolTip("预览或导出温度扫描结果")

        # Compatibility controls retain stable attributes for existing callers;
        # their commands now live in the compact result menu above.
        self.export_plot_button = QtWidgets.QPushButton("导出图表")
        self.export_plot_button.setObjectName("ExportButton")
        self.export_plot_button.setToolTip("导出当前曲线图表为 PNG/PDF")
        self.export_plot_button.clicked.connect(self.export_plot)
        self.export_plot_button.hide()
        self.preview_data_button = QtWidgets.QPushButton("预览数据")
        self.preview_data_button.setObjectName("BrowseButton")
        self.preview_data_button.setToolTip("预览全部积分结果")
        self.preview_data_button.clicked.connect(self.show_data_preview)
        self.preview_data_button.hide()
        self.sidebar_toggle_btn = QtWidgets.QPushButton("隐藏列表")
        self.sidebar_toggle_btn.setObjectName("BrowseButton")
        self.sidebar_toggle_btn.setToolTip("展开/折叠曲线浏览")
        self.sidebar_toggle_btn.setCheckable(True)
        self.sidebar_toggle_btn.setChecked(True)
        self.sidebar_toggle_btn.clicked.connect(self._toggle_sidebar)
        analysis_options_row.addWidget(result_label)
        analysis_options_row.addWidget(self.inline_status_icon)
        analysis_options_row.addWidget(self.inline_status_text)
        analysis_options_row.addWidget(self.inline_action_hint)
        analysis_options_row.addWidget(self.inline_retry_button)
        analysis_options_row.addWidget(self.sidebar_toggle_btn)
        analysis_options_row.addWidget(self.export_button)
        data_layout.addLayout(analysis_options_row)
        root.addWidget(source_panel)
        self.set_temperature_source_scope("temporary", restore_saved=False)

        # ── Body: sidebar + main area ────────────────────────────────────────
        body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        body_splitter.setObjectName("MainSplitter")
        self.body_splitter = body_splitter
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
        self.summary_label.setMinimumHeight(38)
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
        self.group_summary_label.setMinimumHeight(36)
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
        self._plot_stack = plot_stack

        # Empty state widget: one concise explanation and one next action.
        self._empty_state = AnalysisEmptyState(
            title="尚未生成温度曲线",
            description="选择温度扫描目录并生成曲线后，可继续浏览 m/z 和比较不同能量。",
            action_text="选择数据",
        )
        self._empty_state.browse_requested.connect(self.select_folder)
        plot_stack.addWidget(self._empty_state)

        self._progress_state = AnalysisProgressState(title="正在生成温度曲线")
        plot_stack.addWidget(self._progress_state)

        self.plot_widget = StaticCurvePlot("温度 / °C", "归一化信号")
        plot_stack.addWidget(self.plot_widget)

        plot_stack.setCurrentWidget(self._empty_state)
        right_layout.addWidget(self._plot_container, stretch=3)

        curve_stats = QtWidgets.QFrame()
        curve_stats.setObjectName("StatsBar")
        self.curve_stats = curve_stats
        curve_stats_layout = QtWidgets.QHBoxLayout(curve_stats)
        curve_stats_layout.setContentsMargins(10, 2, 10, 2)
        curve_stats_layout.setSpacing(12)
        curve_stats_title = QtWidgets.QLabel("当前曲线")
        curve_stats_title.setObjectName("StatsTitle")
        self.current_curve_label = QtWidgets.QLabel("未选择")
        self.current_curve_label.setObjectName("HintLabel")
        self.current_curve_metric_label = QtWidgets.QLabel("生成曲线后可在左侧选择 m/z")
        self.current_curve_metric_label.setObjectName("HintLabel")
        self.current_interval_label = QtWidgets.QLabel("电离区间: --")
        self.current_interval_label.setObjectName("ReadoutValue")
        self.current_interval_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        curve_stats_layout.addWidget(curve_stats_title)
        curve_stats_layout.addWidget(self.current_curve_label)
        curve_stats_layout.addWidget(self.current_curve_metric_label, stretch=1)
        curve_stats_layout.addWidget(self.current_interval_label)

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
        self.energy_interval_table = QtWidgets.QTableWidget()
        self.energy_interval_table.setWordWrap(False)
        self.energy_interval_table.setAlternatingRowColors(True)
        self.energy_interval_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.detail_tabs.addTab(self.energy_interval_table, "电离区间")
        right_layout.addWidget(self.detail_tabs, stretch=2)
        self._update_action_state()

    def _open_project_settings(self):
        """跳转到项目管理页面。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("project")

    def _toggle_sidebar(self, checked: bool) -> None:
        self._sidebar.setVisible(checked)
        self.sidebar_toggle_btn.setText("隐藏列表" if checked else "曲线列表")

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
        self._plot_stack.setCurrentWidget(self.plot_widget)

    def _show_empty_plot(self) -> None:
        self._plot_stack.setCurrentWidget(self._empty_state)

    def _begin_analysis_progress(self, detail: str) -> None:
        self._progress_state.start(title="正在生成温度曲线", detail=detail)
        self._plot_stack.setCurrentWidget(self._progress_state)

    def _on_analysis_progress(self, value: int, detail: str) -> None:
        self._progress_state.set_progress(value, detail)

    def _restore_analysis_workspace(self) -> None:
        if self._plot_stack.currentWidget() is not self._progress_state:
            return
        if self.curves:
            self._show_plot()
        else:
            self._show_empty_plot()

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

    def _ensure_temporary_settings(self) -> ProjectSettings:
        if self.temporary_settings is None:
            source = self.temporary_settings_source
            if source == "project" and self.project_settings is None:
                source = "global"
                self.temporary_settings_source = source
            self.temporary_settings = self._build_temporary_settings_snapshot(source)
            self.temporary_settings_modified = False
        return self.temporary_settings

    def _build_temporary_settings_snapshot(self, source: str) -> ProjectSettings:
        return build_temporary_settings(
            source,
            self.project_settings,
            runtime_calibration=self._global_calibration,
            runtime_normalization=self._global_normalization_settings,
        )

    def _effective_analysis_settings(self) -> ProjectSettings:
        if self._has_project_scope() and self.project_settings is not None:
            return self.project_settings
        return self._ensure_temporary_settings()

    def _on_temporary_settings_source_changed(self, _index: int) -> None:
        if not hasattr(self, "temporary_params_combo"):
            return
        source = str(self.temporary_params_combo.currentData() or "default")
        self._reset_temporary_settings_source(source)

    def _select_temporary_settings_source(self, source: str) -> None:
        if source == "project" and not self._has_project_context():
            return
        blocker = QtCore.QSignalBlocker(self.temporary_params_combo)
        self.temporary_params_combo.setCurrentIndex(self.temporary_params_combo.findData(source))
        del blocker
        self._reset_temporary_settings_source(source)

    def _reset_temporary_settings_source(self, source: str) -> None:
        if source == "project" and not self._has_project_context():
            source = "global"
        self.temporary_settings_source = source
        self._temporary_settings_source_explicit = True
        self.temporary_settings = self._build_temporary_settings_snapshot(source)
        self.temporary_settings_modified = False
        self._apply_temporary_settings_to_controls()
        self._refresh_temperature_source_controls()

    def _edit_temporary_settings(self) -> None:
        dialog = TemporaryAnalysisSettingsDialog(
            self._ensure_temporary_settings(),
            self.temporary_settings_source,
            self,
            initial_tab="function",
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.temporary_settings = dialog.settings()
        self.temporary_settings_modified = True
        self._apply_temporary_settings_to_controls()

    def _apply_temporary_settings_to_controls(self) -> None:
        if self._has_project_scope():
            return
        ps = self._ensure_temporary_settings()
        mode = ps.temp_replicate_mode if ps.temp_replicate_mode in {"mean", "sum"} else "off"
        self.replicate_enabled_check.setChecked(mode != "off")
        index = self.replicate_mode_combo.findData(mode if mode != "off" else "mean")
        if index >= 0:
            self.replicate_mode_combo.setCurrentIndex(index)
        self._refresh_normalization_status()

    def _remember_temporary_source(self, _text: str | None = None) -> None:
        if self.temperature_source_scope != "temporary":
            return
        self._temporary_temperature_folder = self.folder_edit.text().strip()

    def _on_temperature_source_path_changed(self, text: str | None = None) -> None:
        self._remember_temporary_source(text)
        self._refresh_temperature_folder_options()
        self._update_action_state()

    def _on_temperature_folder_option_changed(self, _index: int) -> None:
        previous_mz = self.current_mz
        self._selected_temperature_folder = str(self.scan_folder_combo.currentData() or "")
        self._refresh_temperature_source_summary()
        if self.energy_results:
            self._apply_energy_view_selection(preferred_mz=previous_mz)
        self._update_action_state()

    @staticmethod
    def _folder_has_spectrum_files(folder: str | Path) -> bool:
        path = Path(folder)
        if not path.is_dir():
            return False
        return any(child.is_file() and child.suffix.lower() == ".txt" for child in path.iterdir())

    @staticmethod
    def _energy_sort_key(path: Path) -> tuple[int, float, str]:
        if "ev" in path.name.lower():
            match = re.search(r"\d+(?:\.\d+)?", path.name)
            if match:
                return (0, float(match.group(0)), path.name)
        return (1, 0.0, path.name)

    @staticmethod
    def _energy_from_folder_name(folder: str | Path) -> float | None:
        name = Path(folder).name
        if "ev" not in name.lower():
            return None
        match = re.search(r"\d+(?:\.\d+)?", name)
        return float(match.group(0)) if match else None

    def _temperature_folder_options(self, root_folder: str | Path) -> list[tuple[str, str]]:
        root = Path(root_folder)
        if not root.is_dir():
            return []
        try:
            cache_key = (str(root.resolve()), root.stat().st_mtime_ns)
        except OSError:
            cache_key = (str(root), None)
        cached = self._folder_options_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        if self._folder_has_spectrum_files(root):
            options = [(f"{root.name} (当前文件夹)", str(root))]
            self._folder_options_cache = {cache_key: options}
            return list(options)
        options = [
            (child.name, str(child))
            for child in sorted(
                (item for item in root.iterdir() if item.is_dir() and self._folder_has_spectrum_files(item)),
                key=self._energy_sort_key,
            )
        ]
        self._folder_options_cache = {cache_key: options}
        return list(options)

    def _refresh_temperature_folder_options(self) -> None:
        if not hasattr(self, "scan_folder_combo"):
            return
        root_folder = self._current_temperature_root_folder()
        previous = self._selected_temperature_folder
        folder_options = self._temperature_folder_options(root_folder) if root_folder else []
        energy_options = [
            (label, path)
            for label, path in folder_options
            if self._energy_from_folder_name(path) is not None
        ]
        # Energy-named folders are the user-facing analysis batches. Raw backup
        # folders may also contain .txt files but should not appear as runnable
        # choices when valid energy folders are available.
        options = list(energy_options or folder_options)
        if len(energy_options) > 1:
            options.insert(0, (f"全部能量 ({len(energy_options)})", self.ALL_ENERGY_FOLDERS))

        self.scan_folder_combo.blockSignals(True)
        self.scan_folder_combo.clear()
        for label, path in options:
            self.scan_folder_combo.addItem(label, path)
        if options:
            selected_index = 0
            if previous and previous != self.ALL_ENERGY_FOLDERS:
                for index, (_label, path) in enumerate(options):
                    if path == previous:
                        selected_index = index
                        break
            elif previous == self.ALL_ENERGY_FOLDERS:
                selected_index = 0
            self.scan_folder_combo.setCurrentIndex(selected_index)
            self._selected_temperature_folder = str(self.scan_folder_combo.itemData(selected_index) or "")
        else:
            self._selected_temperature_folder = ""
        self.scan_folder_combo.blockSignals(False)

        has_multiple_options = len(energy_options) > 1
        has_root = bool(root_folder)
        self.scan_folder_label.setVisible(has_root)
        self.scan_folder_combo.setVisible(has_root)
        self.scan_folder_combo.setEnabled(bool(options) and not getattr(self, "_busy", False))
        if has_root and not folder_options:
            self.scan_folder_combo.addItem("未找到 .txt 数据", "")
        self.scan_folder_combo.setToolTip(
            f"当前数据源下发现 {len(folder_options)} 个可分析能量文件夹；默认生成全部能量曲线。"
            if has_multiple_options
            else "当前数据源将直接用于生成温度曲线。"
        )
        self._refresh_temperature_source_summary()

    def set_temperature_source_scope(
        self,
        scope: str,
        *,
        restore_saved: bool = True,
        apply_project: bool = True,
        refresh: bool = True,
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
        if scope == "temporary":
            self._ensure_temporary_settings()
            self._apply_temporary_settings_to_controls()
        if refresh:
            self._refresh_temperature_source_controls()
            self._refresh_temperature_folder_options()
            self._update_action_state()

    def _refresh_temperature_source_controls(self) -> None:
        use_project = self._has_project_scope()
        has_project_path = bool(self.project_settings and self.project_settings.temperature_scan_folder)
        busy = getattr(self, "_busy", False)
        if hasattr(self, "project_source_button"):
            self.project_source_button.setChecked(use_project)
            self.project_source_button.setEnabled(self._has_project_context() and not busy)
        if hasattr(self, "temporary_source_button"):
            self.temporary_source_button.setChecked(not use_project)
            self.temporary_source_button.setEnabled(not busy)
        self.folder_edit.setReadOnly(use_project)
        self.folder_edit.setEnabled(not busy or use_project)
        self.folder_edit.setClearButtonEnabled(not use_project)
        self.folder_edit.setPlaceholderText(
            "项目未登记温度扫描数据源" if use_project else "选择温度扫描数据文件夹"
        )
        self.select_folder_button.setText("浏览...")
        self.select_folder_button.setVisible(not use_project)
        self.select_folder_button.setEnabled(not busy)
        self.summary_open_project_btn.setVisible(use_project)
        self.temporary_params_panel.setVisible(not use_project)
        self.temporary_params_combo.setVisible(False)
        self.temporary_params_button.setEnabled(not busy)
        self.temporary_params_reload_button.setEnabled(not busy)
        project_index = self.temporary_params_combo.findData("project")
        if project_index >= 0:
            model_item = self.temporary_params_combo.model().item(project_index)
            if model_item is not None:
                model_item.setEnabled(self._has_project_context())
        self.copy_project_params_action.setEnabled(self._has_project_context())
        source_text = {
            "project": "项目参数副本",
            "global": "全局参数副本",
            "default": "程序默认参数",
        }.get(self.temporary_settings_source, "程序默认参数")
        if self.temporary_settings_modified:
            source_text += " · 已修改"
        self.temporary_params_label.setText(f"当前来源：{source_text}")
        self.temporary_params_button.setText("编辑本次参数…")
        self.temporary_params_button.setToolTip(
            "打开完整参数编辑器；所有修改仅对当前临时数据会话生效"
        )
        self.temporary_params_reload_button.setToolTip("丢弃本次修改，并从指定来源重新创建参数副本")
        if use_project and not has_project_path:
            self._show_inline_empty("请先在项目管理中配置温度扫描文件夹")

    def select_folder(self) -> None:
        if self._has_project_scope():
            self._open_project_settings()
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择温度扫描数据文件夹")
        if folder:
            self.folder_edit.setText(folder)

    def _current_temperature_root_folder(self) -> str:
        if self._has_project_scope():
            return (self.project_settings.temperature_scan_folder if self.project_settings else "").strip()
        return self.folder_edit.text().strip()

    def _current_temperature_folder(self) -> str:
        if self._selected_temperature_folder == self.ALL_ENERGY_FOLDERS:
            return self._current_temperature_root_folder()
        return self._selected_temperature_folder or self._current_temperature_root_folder()

    def _selected_analysis_folders(self) -> list[tuple[float | None, str]]:
        root_folder = self._current_temperature_root_folder()
        selected = self._selected_temperature_folder
        if selected == self.ALL_ENERGY_FOLDERS:
            return self._analysis_folders_for_energy_interval()
        folder = self._current_temperature_folder()
        if not folder:
            return []
        energy = self._energy_from_folder_name(folder)
        return [(energy, folder)]

    def _generation_analysis_folders(self) -> list[tuple[float | None, str]]:
        energy_folders = self._analysis_folders_for_energy_interval()
        if len(energy_folders) > 1:
            return energy_folders
        return self._selected_analysis_folders()

    def _visible_energy_results(self) -> list[dict]:
        if not self.energy_results:
            return []
        selected = self._selected_temperature_folder
        if not selected or selected == self.ALL_ENERGY_FOLDERS:
            return list(self.energy_results)
        return [
            item
            for item in self.energy_results
            if str(item.get("folder", "")) == selected
        ]

    def _refresh_temperature_source_summary(self) -> None:
        if not hasattr(self, "summary_data_label"):
            return
        root_path = self._current_temperature_root_folder()
        selected_path = self._current_temperature_folder()
        if not root_path:
            data_text = "---"
        elif self._selected_temperature_folder == self.ALL_ENERGY_FOLDERS:
            count = len(self._analysis_folders_for_energy_interval())
            data_text = f"{Path(root_path).name} / 全部能量({count})"
        elif selected_path and selected_path != root_path:
            data_text = f"{Path(root_path).name} / {Path(selected_path).name}"
        else:
            data_text = root_path
        self.summary_data_label.setText(
            f"{'项目温扫' if self._has_project_scope() else '临时温扫'}: {data_text}"
        )

    def _refresh_normalization_status(self) -> None:
        ps = self._effective_analysis_settings()
        normalization = ps.to_normalization_settings()
        light_source = "IO" if normalization.light_source == "io" else "Beam Current"
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
        if not checked or self._effective_analysis_settings().expansion_factors:
            return
        self.temperature_kr_check.blockSignals(True)
        self.temperature_kr_check.setChecked(False)
        self.temperature_kr_check.blockSignals(False)
        self._show_inline_error("尚未计算 Kr 膨胀系数 λ(T)，已退回为不使用 Kr 校正。请先到项目管理 -> 通用参数计算。")

    def set_project_settings(self, ps: ProjectSettings, *, activate_project_scope: bool | None = None) -> None:
        """Keep project context visible while project-owned parameters stay in 项目管理."""
        self._analysis_request_id += 1
        has_project_context = bool(
            ps.project_name
            or ps.temperature_scan_folder
            or (ps.output_dir and ps.output_dir != "output")
        )
        if activate_project_scope is None:
            activate_project_scope = self._has_project_scope() and has_project_context
        self.project_settings = ps
        if has_project_context and not self._temporary_settings_source_explicit:
            self.temporary_settings_source = "project"
            self.temporary_settings = self._build_temporary_settings_snapshot("project")
            blocker = QtCore.QSignalBlocker(self.temporary_params_combo)
            self.temporary_params_combo.setCurrentIndex(self.temporary_params_combo.findData("project"))
            del blocker
        elif (
            not has_project_context
            and activate_project_scope is False
            and (not self._temporary_settings_source_explicit or self.temporary_settings_source == "project")
        ):
            self.temporary_settings_source = "global"
            self._temporary_settings_source_explicit = False
            self.temporary_settings_modified = False
            self.temporary_settings = self._build_temporary_settings_snapshot("global")
            blocker = QtCore.QSignalBlocker(self.temporary_params_combo)
            self.temporary_params_combo.setCurrentIndex(self.temporary_params_combo.findData("global"))
            del blocker
        elif self.temporary_settings_source == "project" and not self.temporary_settings_modified:
            self.temporary_settings = self._build_temporary_settings_snapshot("project")

        if hasattr(self, "summary_project_label"):
            project_name = ps.project_name or "---"
            system = ps.system or "---"
            self.summary_project_label.setText(f"项目: {project_name}")
            self.summary_system_label.setText(f"体系: {system}")

        if activate_project_scope:
            self.set_temperature_source_scope(
                "project",
                restore_saved=False,
                apply_project=False,
                refresh=False,
            )
        elif self._has_project_scope() and not has_project_context:
            self.set_temperature_source_scope("temporary", refresh=False)

        # A project-owned temperature analysis must use the calibration stored
        # in that project, not the global/default object passed at construction.
        # Do this before cache lookup so the cache fingerprint uses it as well.
        if self._has_project_scope():
            self.calibration = ps.to_calibration()
            self.normalization_settings = ps.to_normalization_settings()
            self.peak_detection = ps.to_peak_detection_config()

        if self._has_project_scope():
            blocker = QtCore.QSignalBlocker(self.folder_edit)
            self.folder_edit.setText(ps.temperature_scan_folder or "")
            del blocker
        # 加载温度扫描参数
        if hasattr(self, "replicate_mode_combo"):
            mode = ps.temp_replicate_mode if ps.temp_replicate_mode in {"mean", "sum"} else "off"
            self.replicate_enabled_check.setChecked(mode != "off")
            idx = self.replicate_mode_combo.findData(mode if mode != "off" else "mean")
            if idx >= 0:
                self.replicate_mode_combo.setCurrentIndex(idx)
            self._on_replicate_enabled_changed(mode != "off")
        self._refresh_normalization_status()

        if not self._has_project_scope():
            self._apply_temporary_settings_to_controls()

        self._refresh_temperature_source_controls()
        self._refresh_temperature_folder_options()
        if not self._current_temperature_folder():
            self._show_inline_empty(
                "请先在项目管理中配置温度扫描文件夹"
                if self._has_project_scope()
                else "请选择温度扫描数据文件夹"
            )
        elif not self._try_load_project_temperature_cache_async():
            self._show_inline_empty('项目参数已同步，点击"生成曲线"')
        self._update_action_state()
        if not has_project_context and activate_project_scope is False:
            self.project_settings = None

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
        self._update_action_state()

    def open_common_parameters(self):
        if not self._has_project_scope():
            self._edit_temporary_settings()
            return
        self._open_project_settings()

    def _analysis_parameters(self) -> dict:
        ps = self._effective_analysis_settings()
        peak_config = ps.to_peak_detection_config()
        normalization = ps.to_normalization_settings()
        integration_method = ps.temp_integration_method
        kr_correct = bool(ps.temperature_kr_correct)
        effective_peak_source = ps.temp_peak_source
        if self._has_project_scope() and ps.manual_peak_file and ps.temp_peak_source == "auto":
            effective_peak_source = "manual"
        manual_peak_path = None
        if self._has_project_scope() and effective_peak_source == "manual":
            manual_peak_path = ps.manual_peak_file

        return {
            "project_settings": ps,
            "calibration": ps.to_calibration(),
            "cache_dir": self._temperature_cache_dir(),
            "peak_config": peak_config,
            "threshold_end": peak_config.threshold_end,
            "min_intensity": peak_config.min_intensity,
            "reference_mode": ps.temp_reference_mode,
            "integration_method": integration_method,
            "prefer_gaussian": integration_method == "gaussian",
            "manual_peak_path": manual_peak_path,
            "effective_peak_source": effective_peak_source,
            "photon_normalize": bool(ps.temperature_photon_normalize),
            "kr_correct": kr_correct,
            "kr_mz": ps.kr_mz,
            "mass_discrimination": 1.0,
            "light_source": normalization.light_source,
            "expansion_factors": normalization.expansion_factors if kr_correct else None,
            "replicate_mode": self._current_replicate_mode(),
        }

    def _validate_analysis_parameters(self, params: dict) -> bool:
        if (
            self._has_project_scope()
            and params.get("effective_peak_source") == "manual"
            and not params.get("manual_peak_path")
        ):
            self._show_inline_error("请在项目管理中配置手动卡峰文件")
            return False
        if params.get("kr_correct") and not params.get("expansion_factors"):
            self._show_inline_error(
                "已勾选 Kr 校正，但尚未计算 λ(T)，本次分析已退回。请先到 项目管理 -> 通用参数 计算 Kr 膨胀系数。"
            )
            return False
        return True

    def _analysis_folders_for_energy_interval(self) -> list[tuple[float, str]]:
        root_folder = self._current_temperature_root_folder()
        if not root_folder:
            return []
        folders: list[tuple[float, str]] = []
        for _label, folder in self._temperature_folder_options(root_folder):
            energy = self._energy_from_folder_name(folder)
            if energy is None:
                continue
            folders.append((energy, folder))
        if not folders and self._folder_has_spectrum_files(root_folder):
            energy = self._energy_from_folder_name(root_folder)
            if energy is not None:
                folders.append((energy, root_folder))
        return sorted(folders, key=lambda item: item[0])

    def _analyze_temperature_folder_with_params(self, folder: str, params: dict) -> pd.DataFrame:
        peak_config = params["peak_config"]
        ps = params["project_settings"]
        return analyze_temperature_folder(
            folder,
            calibration=params["calibration"],
            algorithm=peak_config.algorithm,
            threshold_end=params["threshold_end"],
            min_intensity=params["min_intensity"],
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
            prefer_gaussian=params["prefer_gaussian"],
            integration_method=params["integration_method"],
            reference_mode=params["reference_mode"],
            manual_peak_path=params["manual_peak_path"],
            photon_normalize=params["photon_normalize"],
            kr_correct=params["kr_correct"],
            kr_mz=params["kr_mz"],
            mass_discrimination=params["mass_discrimination"],
            light_source=params["light_source"],
            expansion_factors=params["expansion_factors"],
            vote_threshold=peak_config.vote_threshold,
            min_intensity_for_single_vote=peak_config.min_intensity_for_single_vote,
            mz_tolerance=peak_config.mz_tolerance,
            cwt_snr_threshold=peak_config.cwt_snr_threshold,
            cwt_wavelet_max_width=peak_config.cwt_wavelet_max_width,
            weak_tail_cutoff_idx=peak_config.weak_tail_cutoff_idx,
            temp_curve_class_change_threshold=ps.temp_curve_class_change_threshold,
            temp_curve_class_peak_fraction=ps.temp_curve_class_peak_fraction,
            replicate_mode=params["replicate_mode"],
            cache_curves=True,
        )

    def _analyze_temperature_folders_with_params(
        self,
        folders: list[tuple[float | None, str]],
        params: dict,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict:
        energy_results: list[dict] = []
        frames: list[pd.DataFrame] = []

        # Peak fitting relies on SciPy optimizers whose warning/solver state is
        # not stable when several energy folders are fitted in Python threads.
        # Keep folder analysis deterministic; the safe speedup below comes from
        # reusing each folder's already-built curves instead of grouping twice.
        folder_count = max(1, len(folders))
        for index, (energy, folder) in enumerate(folders):
            folder_label = Path(folder).name
            if progress_callback is not None:
                progress_callback(
                    10 + round(72 * index / folder_count),
                    f"正在处理 {folder_label}（{index + 1}/{folder_count}）",
                )
            result_df = self._analyze_temperature_folder_with_params(folder, params)
            cached_curves = result_df.attrs.pop(
                "_bl03u_temperature_curves",
                None,
            )
            scan_energy = float(energy) if energy is not None else np.nan
            if not result_df.empty:
                result_df = result_df.copy()
                result_df["scan_folder"] = folder_label
                result_df["scan_energy"] = scan_energy
                frames.append(result_df)
                if isinstance(cached_curves, dict):
                    # Match the previous GUI-built curve rows exactly: those rows
                    # were created after scan provenance columns were attached.
                    for curve in cached_curves.values():
                        curve_rows = curve.get("rows")
                        if not isinstance(curve_rows, pd.DataFrame):
                            continue
                        curve_rows = curve_rows.copy()
                        curve_rows["scan_folder"] = folder_label
                        curve_rows["scan_energy"] = scan_energy
                        curve["rows"] = curve_rows
            energy_results.append(
                {
                    "energy": float(energy) if energy is not None else self._infer_energy_from_result(result_df),
                    "folder": folder,
                    "folder_label": folder_label,
                    "result_df": result_df,
                    "curves": (
                        cached_curves
                        if isinstance(cached_curves, dict)
                        else build_temperature_curves(result_df)
                    ),
                }
            )
            if progress_callback is not None:
                progress_callback(
                    10 + round(72 * (index + 1) / folder_count),
                    f"已完成 {folder_label}（{index + 1}/{folder_count}）",
                )
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if progress_callback is not None:
            progress_callback(86, "正在汇总不同能量的温度曲线…")
        return {"result_df": combined, "energy_results": energy_results}

    def _analyze_temperature_folders_with_cache(
        self,
        folders: list[tuple[float | None, str]],
        params: dict,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict:
        if progress_callback is not None:
            progress_callback(5, "正在检查已有分析缓存…")
        cache_key = self._temperature_cache_key(folders, params)
        cache_dir = params.get("cache_dir")
        cached = self._load_temperature_analysis_cache(cache_key, cache_dir=cache_dir)
        if cached is not None:
            if progress_callback is not None:
                progress_callback(100, "缓存结果已载入，正在显示曲线…")
            cached["from_cache"] = True
            return cached

        result = self._analyze_temperature_folders_with_params(
            folders,
            params,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback(92, "正在保存分析结果缓存…")
        self._save_temperature_analysis_cache(cache_key, result, cache_dir=cache_dir)
        result["from_cache"] = False
        if progress_callback is not None:
            progress_callback(100, "温度曲线已生成，正在更新界面…")
        return result

    def _try_load_project_temperature_cache_async(self) -> bool:
        if not self._has_project_scope() or self.project_settings is None:
            return False
        folders = self._generation_analysis_folders()
        if not folders:
            return False
        try:
            params = self._analysis_parameters()
        except Exception:
            return False
        if (
            self._has_project_scope()
            and params.get("effective_peak_source") == "manual"
            and not params.get("manual_peak_path")
        ):
            return False
        if params.get("kr_correct") and not params.get("expansion_factors"):
            return False

        self._autoload_request_id += 1
        token = self._project_temperature_cache_token(folders, self._autoload_request_id)
        self._autoload_cache_token = token
        self._show_inline_empty("正在检查项目温度扫描缓存...")
        self._autoload_worker = WorkerThread(
            lambda: self._load_project_temperature_cache(folders, params),
            self,
        )
        self._autoload_worker.finished_with_result.connect(
            lambda result, expected_token=token: self._on_project_temperature_cache_loaded(result, expected_token)
        )
        self._autoload_worker.failed.connect(
            lambda _message, expected_token=token: self._on_project_temperature_cache_failed(expected_token)
        )
        self._autoload_worker.start()
        return True

    def _project_temperature_cache_token(
        self,
        folders: list[tuple[float | None, str]],
        request_id: object,
    ) -> dict:
        ps = self.project_settings or ProjectSettings()
        return {
            "request_id": request_id,
            "output_dir": str(ps.output_dir or ""),
            "temperature_scan_folder": str(ps.temperature_scan_folder or ""),
            "folders": [(float(energy) if energy is not None else None, str(folder)) for energy, folder in folders],
        }

    def _is_current_project_temperature_cache_token(self, token: dict | None) -> bool:
        if not token or not self._has_project_scope() or self.project_settings is None:
            return False
        return token == self._autoload_cache_token

    def _load_project_temperature_cache(
        self,
        folders: list[tuple[float | None, str]],
        params: dict,
    ) -> dict:
        cache_key = self._temperature_cache_key(folders, params)
        return {
            "cache_key": cache_key,
            "cached_result": self._load_temperature_analysis_cache(cache_key),
        }

    def _on_project_temperature_cache_failed(self, expected_token: dict | None = None) -> None:
        if not self._is_current_project_temperature_cache_token(expected_token):
            return
        self._show_inline_empty('项目参数已同步，点击"生成曲线"')

    def _on_project_temperature_cache_loaded(self, result: object, expected_token: dict | None = None) -> None:
        if not self._is_current_project_temperature_cache_token(expected_token):
            logger.debug("Ignored stale temperature-scan project cache autoload result.")
            return
        if isinstance(result, dict) and "cached_result" in result:
            result = result.get("cached_result")
        if not result:
            self._show_inline_empty('项目参数已同步，点击"生成曲线"')
            self._update_action_state()
            return
        if isinstance(result, dict):
            result["from_cache"] = True
            self.on_analysis_complete(result)
            self._show_inline_success("已自动载入项目缓存的温度扫描曲线")
        else:
            self._show_inline_empty('项目参数已同步，点击"生成曲线"')
        self._update_action_state()

    def _temperature_cache_dir(self) -> Path:
        if self._has_project_scope() and self.project_settings is not None and self.project_settings.output_dir:
            root = Path(self.project_settings.output_dir)
            return root / "analysis" / "temperature_scan" / "cache"
        else:
            return ensure_output_dir("analysis", "temperature_scan") / "cache"

    def _temperature_cache_key(self, folders: list[tuple[float | None, str]], params: dict) -> str:
        payload = {
            "version": self.CACHE_VERSION,
            "folders": self._folder_fingerprints(folders),
            "parameters": self._analysis_cache_parameters(params),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def _folder_fingerprints(self, folders: list[tuple[float | None, str]]) -> list[dict]:
        fingerprints: list[dict] = []
        for energy, folder in folders:
            folder_path = Path(folder)
            files = [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in sorted(folder_path.iterdir())
                if path.is_file() and path.suffix.lower() == ".txt"
            ]
            fingerprints.append(
                {
                    "energy": float(energy) if energy is not None else None,
                    "folder": str(folder_path),
                    "folder_mtime_ns": folder_path.stat().st_mtime_ns if folder_path.exists() else None,
                    "files": files,
                }
            )
        return fingerprints

    def _analysis_cache_parameters(self, params: dict) -> dict:
        peak_config = params["peak_config"]
        ps = params["project_settings"]
        manual_peak_path = params.get("manual_peak_path")
        manual_peak_fingerprint = None
        if manual_peak_path:
            manual_path = Path(manual_peak_path)
            if manual_path.exists():
                manual_peak_fingerprint = {
                    "path": str(manual_path),
                    "size": manual_path.stat().st_size,
                    "mtime_ns": manual_path.stat().st_mtime_ns,
                }
            else:
                manual_peak_fingerprint = {"path": str(manual_path), "missing": True}
        calibration = params["calibration"]
        return {
            "calibration": {
                "a": float(calibration.a),
                "b": float(calibration.b),
                "c": float(calibration.c),
            },
            "peak_config": asdict(peak_config),
            "reference_mode": params["reference_mode"],
            "integration_method": params["integration_method"],
            "prefer_gaussian": params["prefer_gaussian"],
            "manual_peak_path": manual_peak_path,
            "manual_peak_fingerprint": manual_peak_fingerprint,
            "photon_normalize": params["photon_normalize"],
            "kr_correct": params["kr_correct"],
            "kr_mz": params["kr_mz"],
            "mass_discrimination": params["mass_discrimination"],
            "light_source": params["light_source"],
            "expansion_factors": params["expansion_factors"],
            "replicate_mode": params["replicate_mode"],
            "temp_curve_class_change_threshold": ps.temp_curve_class_change_threshold,
            "temp_curve_class_peak_fraction": ps.temp_curve_class_peak_fraction,
        }

    def _cache_paths(
        self,
        cache_key: str,
        *,
        cache_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        scoped_cache_dir = (cache_dir or self._temperature_cache_dir()) / cache_key
        return scoped_cache_dir / "manifest.json", scoped_cache_dir / "results.csv"

    def _save_temperature_analysis_cache(
        self,
        cache_key: str,
        result: dict,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        result_df = result.get("result_df", pd.DataFrame())
        if not isinstance(result_df, pd.DataFrame) or result_df.empty:
            return
        manifest_path, result_path = self._cache_paths(cache_key, cache_dir=cache_dir)
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
            energy_items = [
                {
                    "energy": item.get("energy"),
                    "folder": item.get("folder"),
                    "folder_label": item.get("folder_label"),
                }
                for item in result.get("energy_results", [])
            ]
            manifest = {
                "version": self.CACHE_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "result_file": result_path.name,
                "row_count": int(len(result_df)),
                "energy_results": energy_items,
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Cache writes are an optimization; analysis results remain valid if this fails.
            logger.exception("Failed to save temperature-scan analysis cache at %s", manifest_path.parent)
            return

    def _load_temperature_analysis_cache(
        self,
        cache_key: str,
        *,
        cache_dir: Path | None = None,
    ) -> dict | None:
        manifest_path, result_path = self._cache_paths(cache_key, cache_dir=cache_dir)
        if not manifest_path.exists() or not result_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("version", -1)) != self.CACHE_VERSION:
                return None
            result_df = pd.read_csv(result_path)
            if not isinstance(result_df, pd.DataFrame):
                return None
            for column in ("species", "integration_method", "replicate_grouping", "replicate_warning", "scan_folder"):
                if column in result_df:
                    result_df[column] = result_df[column].fillna("").astype(str)
            energy_results = self._energy_results_from_cached_dataframe(result_df, manifest.get("energy_results", []))
            return {"result_df": result_df, "energy_results": energy_results}
        except Exception:
            return None

    def _energy_results_from_cached_dataframe(self, result_df: pd.DataFrame, energy_items: list[dict]) -> list[dict]:
        energy_results: list[dict] = []
        for item in energy_items:
            folder_label = str(item.get("folder_label") or Path(str(item.get("folder", ""))).name)
            energy = item.get("energy")
            if "scan_folder" in result_df:
                group_df = result_df[result_df["scan_folder"].astype(str) == folder_label].copy()
            else:
                group_df = result_df.copy()
            if group_df.empty:
                continue
            energy_results.append(
                {
                    "energy": float(energy) if energy is not None and pd.notna(energy) else self._infer_energy_from_result(group_df),
                    "folder": item.get("folder", ""),
                    "folder_label": folder_label,
                    "result_df": group_df,
                    "curves": build_temperature_curves(group_df),
                }
            )
        if not energy_results and not result_df.empty:
            energy_results.append(
                {
                    "energy": self._infer_energy_from_result(result_df),
                    "folder": self._current_temperature_folder(),
                    "folder_label": Path(self._current_temperature_folder()).name,
                    "result_df": result_df,
                    "curves": build_temperature_curves(result_df),
                }
            )
        return energy_results

    @staticmethod
    def _infer_energy_from_result(result_df: pd.DataFrame) -> float | None:
        if result_df.empty or "photon_energy" not in result_df:
            return None
        values = pd.to_numeric(result_df["photon_energy"], errors="coerce").dropna()
        values = values[values > 0]
        if values.empty:
            return None
        unique_values = sorted({round(float(value), 6) for value in values})
        return float(unique_values[0]) if len(unique_values) == 1 else None

    def _energy_interval_summary_for_mz(self, target_mz: int) -> dict:
        rows: list[dict] = []
        for item in self.energy_results:
            energy = item.get("energy")
            if energy is None or not np.isfinite(float(energy)):
                continue
            curve = item.get("curves", {}).get(int(target_mz))
            if curve is None:
                rows.append(
                    {
                        "energy": float(energy),
                        "folder": item.get("folder_label", Path(str(item.get("folder", ""))).name),
                        "mz": int(target_mz),
                        "curve_class": "missing",
                        "curve_class_label": "未检出",
                        "curve_class_reason": "当前能量未生成该 m/z 曲线",
                        "points": 0,
                        "max_signal": 0.0,
                        "temperature_range": "",
                    }
                )
                continue
            rows.append(self._energy_curve_summary_row(target_mz, float(energy), str(item.get("folder_label", "")), curve))

        summary = identify_product_energy_intervals(rows)
        summary["target_mz"] = int(target_mz)
        return summary

    @staticmethod
    def _energy_curve_summary_row(target_mz: int, energy: float, folder_label: str, curve: dict) -> dict:
        temperatures = np.asarray(curve.get("temperatures", []), dtype=float)
        areas = np.asarray(curve.get("areas", []), dtype=float)
        valid = np.isfinite(temperatures) & np.isfinite(areas)
        temperatures = temperatures[valid]
        areas = areas[valid]
        if temperatures.size and areas.size:
            max_signal = float(np.nanmax(areas))
            temperature_range = f"{float(np.nanmin(temperatures)):.0f}-{float(np.nanmax(temperatures)):.0f} °C"
        else:
            max_signal = 0.0
            temperature_range = ""
        return {
            "energy": float(energy),
            "folder": folder_label,
            "mz": int(target_mz),
            "curve_class": curve.get("curve_class", "unclassified"),
            "curve_class_label": curve.get("curve_class_label", TEMPERATURE_CURVE_CLASS_LABELS["unclassified"]),
            "curve_class_reason": curve.get("curve_class_reason", ""),
            "points": len(curve.get("temperatures", [])),
            "max_signal": max_signal,
            "temperature_range": temperature_range,
        }

    @staticmethod
    def _format_energy_intervals(summary: dict) -> str:
        intervals = summary.get("intervals", [])
        if not intervals:
            return "未识别到产物型区间"
        return "；".join(
            f"{item['start']:.2f}-{item['end']:.2f} eV"
            if abs(float(item["start"]) - float(item["end"])) > 1e-9
            else f"{item['start']:.2f} eV"
            for item in intervals
        )

    def _populate_energy_interval_table(self, result: dict) -> None:
        rows = result.get("rows", pd.DataFrame())
        if not isinstance(rows, pd.DataFrame):
            rows = pd.DataFrame(rows)
        display = rows.copy()
        if display.empty:
            self.energy_interval_table.clear()
            self.energy_interval_table.setRowCount(0)
            self.energy_interval_table.setColumnCount(0)
            return
        display["产物型"] = display["is_product_like"].map(lambda value: "是" if bool(value) else "否")
        display_df = pd.DataFrame(
            {
                "能量(eV)": display["energy"].map(lambda value: f"{float(value):.2f}"),
                "线形": display["curve_class_label"],
                "产物型": display["产物型"],
                "最高信号": display["max_signal"].map(lambda value: f"{float(value):.4g}" if pd.notna(value) else ""),
                "温度范围": display["temperature_range"],
                "说明": display["curve_class_reason"],
                "文件夹": display["folder"],
            }
        )
        self.set_dataframe(self.energy_interval_table, display_df)
        for row_index, is_product in enumerate(display["is_product_like"].tolist()):
            if not is_product:
                continue
            for column in range(self.energy_interval_table.columnCount()):
                item = self.energy_interval_table.item(row_index, column)
                if item is None:
                    continue
                item.setBackground(QtGui.QBrush(QtGui.QColor("#ecfdf5")))
                if column == 2:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#047857")))

    def run_analysis(self):
        # Get parameters from project settings (single source of truth)
        ps = self._effective_analysis_settings()

        # 温度扫描页保存本功能的分析开关；项目管理只维护所需参数/资源。
        if ps:
            ps.temp_replicate_mode = self._current_replicate_mode()
            ps.temperature_photon_normalize = self.temperature_photon_check.isChecked()
            ps.temperature_kr_correct = self.temperature_kr_check.isChecked()

        folders = self._generation_analysis_folders()
        if not folders:
            self._show_inline_error(
                "请在项目管理中配置温度扫描文件夹"
                if self._has_project_scope()
                else "请选择温度扫描数据文件夹"
            )
            return

        missing_folders = [folder for _energy, folder in folders if not self._folder_has_spectrum_files(folder)]
        if missing_folders:
            self._show_inline_error("当前扫描批次未找到 .txt 质谱文件，请切换扫描批次或检查数据源")
            return

        params = self._analysis_parameters()
        if not self._validate_analysis_parameters(params):
            if params.get("kr_correct") and not params.get("expansion_factors"):
                self.temperature_kr_check.blockSignals(True)
                self.temperature_kr_check.setChecked(False)
                self.temperature_kr_check.blockSignals(False)
                ps.temperature_kr_correct = False
                if self._has_project_scope():
                    self._save_project_switches()
            return
        self._save_project_switches()
        if len(folders) > 1:
            message = f"正在分析 {len(folders)} 个能量文件夹..."
        else:
            message = "正在分析温度扫描数据..."
        self.set_busy(True, message)
        self._begin_analysis_progress(message)
        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        worker = WorkerThread(
            lambda: self._analyze_temperature_folders_with_cache(
                folders,
                params,
                progress_callback=worker.report_progress,
            ),
            self,
        )
        self.worker = worker
        if hasattr(worker, "progress"):
            worker.progress.connect(self._on_analysis_progress)
        worker.finished_with_result.connect(
            lambda result, token=request_id: self._on_analysis_result(token, result)
        )
        worker.failed.connect(self.on_analysis_failed)
        worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        worker.start()

    def _on_analysis_result(self, request_id: int, result: object) -> None:
        """Ignore results produced for a project context that is no longer active."""
        if request_id != self._analysis_request_id:
            return
        self.on_analysis_complete(result)

    def compute_kr_expansion(self):
        # Get parameters from project settings (single source of truth)
        ps = self._effective_analysis_settings()
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

        peak_config = ps.to_peak_detection_config()
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

        settings = ps.to_normalization_settings()
        light_source = settings.light_source
        kr_mz = ps.kr_mz
        self.set_busy(True, "正在计算 Kr 膨胀系数...")
        self.worker = WorkerThread(
            lambda: compute_kr_expansion_factors(
                folder,
                calibration=ps.to_calibration(),
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
        ps = self._effective_analysis_settings()
        settings = ps.to_normalization_settings()
        expansion_factors = parse_expansion_factors_from_result(result)
        settings.expansion_factors = expansion_factors
        ps.expansion_factors = dict(expansion_factors)
        self.normalization_settings = settings
        if self._has_project_scope():
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
        self._update_action_state()

    def _current_replicate_mode(self) -> str:
        if not self.replicate_enabled_check.isChecked():
            return "off"
        return str(self.replicate_mode_combo.currentData() or "mean")

    def set_busy(self, busy: bool, message: str) -> None:
        self._busy = busy
        self.summary_open_project_btn.setEnabled(not busy)
        self.mz_list.setEnabled(not busy)
        self.curve_filter_edit.setEnabled(not busy)
        self.curve_display_combo.setEnabled(not busy)
        self.curve_group_combo.setEnabled(not busy)
        self._refresh_temperature_source_controls()
        self._refresh_temperature_folder_options()
        if busy:
            self.inline_status_icon.setText("")
            self.inline_status_text.setText(message)
            self._set_inline_status("busy")
            self.inline_action_hint.hide()
            self.inline_retry_button.hide()
        else:
            self._update_action_state()

    def _has_analysis_source(self) -> bool:
        return bool(self._selected_analysis_folders())

    def _update_action_state(self) -> None:
        if not hasattr(self, "run_button"):
            return
        busy = getattr(self, "_busy", False)
        has_source = bool(self._selected_analysis_folders())
        has_results = not self.result_df.empty
        has_curves = bool(self.curves)
        has_selection = self.current_mz is not None and self.current_mz in self.curves

        self.run_button.setEnabled(has_source and not busy)
        self.run_button.setText("刷新曲线" if has_curves else "生成曲线")
        self.export_button.setEnabled(has_results and not busy)
        self.export_plot_button.setEnabled(has_curves and has_selection and not busy)
        self.preview_data_button.setEnabled(has_results and not busy)
        self.preview_data_action.setEnabled(has_results and not busy)
        self.export_result_action.setEnabled(has_results and not busy)
        self.export_plot_action.setEnabled(has_curves and has_selection and not busy)
        if hasattr(self, "toggle_table_button"):
            self.toggle_table_button.setEnabled((has_selection or self.energy_interval_result is not None) and not busy)
        self.scan_folder_combo.setEnabled(has_source and self.scan_folder_combo.count() > 1 and not busy)
        if hasattr(self, "_sidebar"):
            self._sidebar.setVisible(has_curves and self.sidebar_toggle_btn.isChecked())
            self.curve_stats.setVisible(has_curves)
            self.sidebar_toggle_btn.setVisible(has_curves)
            self.export_button.setVisible(has_results)
            self.preview_data_button.setVisible(False)
            self.export_plot_button.setVisible(False)

        if busy:
            workflow_text = "处理中 · 请稍候"
            workflow_status = "busy"
        elif not has_source:
            workflow_text = "步骤 1/3 · 选择温度扫描数据"
            workflow_status = "pending"
        elif not has_curves:
            workflow_text = "步骤 2/3 · 生成温度曲线"
            workflow_status = "active"
        elif not has_selection:
            workflow_text = "步骤 3/3 · 选择 m/z 曲线"
            workflow_status = "pending"
        else:
            workflow_text = "结果就绪 · 可查看或导出"
            workflow_status = "complete"
        self.workflow_stage_label.setText(workflow_text)
        self.workflow_stage_label.setProperty("status", workflow_status)
        self.workflow_stage_label.style().unpolish(self.workflow_stage_label)
        self.workflow_stage_label.style().polish(self.workflow_stage_label)

    def on_analysis_complete(self, result: object) -> None:
        from_cache = False
        if isinstance(result, dict):
            self.result_df = result.get("result_df", pd.DataFrame())
            self.energy_results = list(result.get("energy_results", []))
            from_cache = bool(result.get("from_cache", False))
        else:
            self.result_df = result
            self.energy_results = [
                {
                    "energy": self._infer_energy_from_result(self.result_df),
                    "folder": self._current_temperature_folder(),
                    "folder_label": Path(self._current_temperature_folder()).name,
                    "result_df": self.result_df,
                    "curves": build_temperature_curves(self.result_df),
                }
            ]
        self.curves = self._build_display_curves()
        self.energy_interval_result = None
        self.populate_mz_list(preferred_mz=self.current_mz)
        temperature_count = self.result_df["temperature"].nunique() if not self.result_df.empty else 0
        energy_count = len([item for item in self.energy_results if item.get("curves")])
        replicate_note = self._replicate_status_text(self.result_df)
        integration_note = self._integration_status_text(self.result_df)
        self.summary_label.setText(
            f"{len(self.curves)} 条 m/z 曲线\n{energy_count} 个能量 · {temperature_count} 个温度点"
        )
        detail_notes = [note for note in (replicate_note, integration_note) if note]
        self.summary_label.setToolTip("；".join(detail_notes))
        self.update_group_summary()
        if self.curves:
            self._show_plot()  # reveal plot, hide empty state
        else:
            self._show_empty_plot()
        success = (
            f"已从缓存读取 {len(self.result_df)} 行温度扫描结果"
            if from_cache
            else f"已完成分析，生成 {len(self.result_df)} 行温度扫描结果"
        )
        if energy_count > 1:
            success = f"{success}；已汇总 {energy_count} 个能量文件夹"
        if replicate_note:
            success = f"{success}；{replicate_note}"
        if integration_note:
            success = f"{success}；{integration_note}"
        self._show_inline_success(success)
        self._update_action_state()

    def _build_display_curves(self, energy_results: list[dict] | None = None) -> dict[int, dict]:
        energy_results = self._visible_energy_results() if energy_results is None else energy_results
        if not energy_results:
            return build_temperature_curves(self.result_df)

        display_curves: dict[int, dict] = {}
        all_mz = sorted(
            {
                int(mz)
                for item in energy_results
                for mz in item.get("curves", {}).keys()
            }
        )
        for mz in all_mz:
            energy_curves = [
                (item, item.get("curves", {}).get(mz))
                for item in energy_results
                if item.get("curves", {}).get(mz) is not None
            ]
            if not energy_curves:
                continue
            class_key, class_label = self._summarize_mz_class([curve for _item, curve in energy_curves])
            rows = []
            temperatures: list[float] = []
            areas: list[float] = []
            for item, curve in energy_curves:
                curve_rows = curve["rows"].copy()
                curve_rows["scan_energy"] = item.get("energy")
                curve_rows["scan_folder"] = item.get("folder_label", "")
                rows.append(curve_rows)
                temperatures.extend(float(value) for value in curve.get("temperatures", []))
                areas.extend(float(value) for value in curve.get("areas", []))
            combined_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            display_curves[mz] = {
                "mz": int(mz),
                "species": str(energy_curves[0][1].get("species", "")),
                "temperatures": temperatures,
                "areas": areas,
                "curve_class": class_key,
                "curve_class_label": class_label,
                "curve_class_reason": "跨能量汇总分类",
                "rows": combined_rows,
                "energy_curves": energy_curves,
            }
        return display_curves

    def _apply_energy_view_selection(self, *, preferred_mz: int | None = None) -> None:
        if not self.energy_results:
            return
        self.curves = self._build_display_curves()
        self.energy_interval_result = None
        self.populate_mz_list(preferred_mz=preferred_mz)
        selected_count = len(self._visible_energy_results())
        total_count = len(self.energy_results)
        temperature_count = self.result_df["temperature"].nunique() if not self.result_df.empty else 0
        if selected_count == total_count:
            prefix = f"{len(self.curves)} 条m/z曲线 | {total_count} 个能量"
        else:
            prefix = f"{len(self.curves)} 条m/z曲线 | 当前 {selected_count} 个能量 / 共 {total_count} 个能量"
        self.summary_label.setText(f"{prefix} | {temperature_count} 个温度点")
        self.update_group_summary()

    @staticmethod
    def _summarize_mz_class(curves: list[dict]) -> tuple[str, str]:
        classes = [curve.get("curve_class", "unclassified") for curve in curves]
        if "formation" in classes:
            key = "formation"
        elif "intermediate" in classes:
            key = "intermediate"
        elif "consumption" in classes:
            key = "consumption"
        else:
            key = "unclassified"
        return key, TEMPERATURE_CURVE_CLASS_LABELS[key]

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
        self._restore_analysis_workspace()
        self._show_inline_error(f"分析失败: {message}", lambda: self.run_analysis())
        self._update_action_state()

    def populate_mz_list(self, *args, preferred_mz: int | None = None):
        if args and preferred_mz is None and isinstance(args[0], int) and args[0] in self.curves:
            preferred_mz = int(args[0])
        self.mz_list.clear()
        selected_group = self.curve_group_combo.currentData() if hasattr(self, "curve_group_combo") else "all"
        display_mode = self.curve_display_combo.currentData() if hasattr(self, "curve_display_combo") else "grouped"
        if display_mode == "mz":
            self.populate_mz_list_flat(selected_group)
        else:
            self.populate_mz_list_grouped(selected_group)
        item = self.find_curve_tree_item(preferred_mz) if preferred_mz is not None else None
        if item is None:
            item = self.first_curve_tree_item()
        if item is not None:
            self.mz_list.setCurrentItem(item)
        else:
            self.on_mz_selected(None)
        self._update_action_state()

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
        energy_count = len(curve.get("energy_curves", []))
        point_count = len(curve.get("temperatures", []))
        count_text = f"{energy_count}能量/{point_count}点" if energy_count > 1 else f"{point_count}点"
        base = f"{mz}{suffix}  ({count_text})"
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

    def find_curve_tree_item(self, mz: int | None):
        if mz is None:
            return None
        for index in range(self.mz_list.topLevelItemCount()):
            item = self.mz_list.topLevelItem(index)
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == int(mz):
                return item
            for child_index in range(item.childCount()):
                child = item.child(child_index)
                if child.data(0, QtCore.Qt.ItemDataRole.UserRole) == int(mz):
                    return child
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
            self.energy_interval_table.clear()
            self.energy_interval_table.setRowCount(0)
            self.energy_interval_table.setColumnCount(0)
            self.energy_interval_result = None
            self.current_curve_label.setText("未选择")
            self.current_curve_metric_label.setText("调整筛选或重新生成曲线")
            self.current_interval_label.setText("电离区间: --")
            if self.plot_widget is not None:
                self.plot_widget.clear_plot(title="未选择温度曲线")
            self._update_action_state()
            return
        mz_value = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if mz_value is None:
            if current.childCount() > 0:
                self.mz_list.setCurrentItem(current.child(0))
            return
        self.current_mz = int(mz_value)
        curve = self.curves[self.current_mz]
        self.energy_interval_result = self._energy_interval_summary_for_mz(self.current_mz)
        self._populate_energy_interval_table(self.energy_interval_result)
        rows = curve["rows"].copy()
        integration_methods = rows.get("integration_method", pd.Series([""] * len(rows))).astype(str).map(
            {"sum_counts": "范围累加", "gaussian": "高斯", "baseline": "扣基线积分", "mixed": "混合"}
        ).fillna("")
        if len(curve.get("energy_curves", [])) > 1:
            curve_df = pd.DataFrame(
                {
                    "能量(eV)": pd.to_numeric(rows.get("scan_energy", pd.Series([np.nan] * len(rows))), errors="coerce").map(
                        lambda value: f"{float(value):.2f}" if pd.notna(value) else ""
                    ),
                    "文件夹": rows.get("scan_folder", pd.Series([""] * len(rows))).astype(str),
                    "温度(C)": np.round(rows["temperature"].astype(float), 1),
                    "积分方式": integration_methods,
                    "原始积分": np.round(rows["raw_area"].astype(float), 4),
                    "IO归一化": np.round(rows["photon_normalized_area"].astype(float), 4),
                    "λ(T)": np.round(rows["expansion_lambda"].astype(float), 6),
                    "最终强度": np.round(rows["area"].astype(float), 4),
                }
            )
            self.set_dataframe(self.curve_table, curve_df)
            self.curve_table.verticalHeader().setVisible(False)
        else:
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
            curve_df_transposed = curve_df.set_index("温度(C)").T
            self._set_curve_detail_table(curve_df_transposed)
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
            energy_count = len(curve.get("energy_curves", []))
            if energy_count > 1:
                status_text = f"{class_label} | {energy_count} 个能量 | {len(curve['temperatures'])} 个点 | 最大值 {max_value:.2f}"
        else:
            status_text = f"{class_label} | {len(curve['temperatures'])} 个温度点"

        self.current_curve_metric_label.setText(status_text)
        self.current_interval_label.setText(f"电离区间: {self._format_energy_intervals(self.energy_interval_result)}")
        self.update_plot(curve)
        self._update_action_state()

    def _set_curve_detail_table(self, df: pd.DataFrame) -> None:
        """Render one selected m/z curve with explicit calculation-step row labels."""
        self.set_dataframe(self.curve_table, df)
        row_labels = [str(index) for index in df.index]
        self.curve_table.setVerticalHeaderLabels(row_labels)
        self.curve_table.verticalHeader().setVisible(True)
        self.curve_table.verticalHeader().setMinimumWidth(96)
        self.curve_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.curve_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        final_row = row_labels.index("最终强度") if "最终强度" in row_labels else -1
        if final_row >= 0:
            header_item = self.curve_table.verticalHeaderItem(final_row)
            if header_item is not None:
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                header_item.setForeground(QtGui.QBrush(QtGui.QColor("#047857")))
            for column in range(self.curve_table.columnCount()):
                item = self.curve_table.item(final_row, column)
                if item is None:
                    continue
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setForeground(QtGui.QBrush(QtGui.QColor("#047857")))
                item.setBackground(QtGui.QBrush(QtGui.QColor("#ecfdf5")))

    def update_plot(self, curve: dict):
        if self.plot_widget is None:
            return
        if len(curve.get("energy_curves", [])) > 1:
            self.update_multi_energy_plot(curve)
            return

        x_values = np.asarray(curve["temperatures"], dtype=float)
        y_values = np.asarray(curve["areas"], dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid]
        y_values = y_values[valid]

        title = f"m/z {curve['mz']} 温度响应曲线"
        if x_values.size == 0:
            self.plot_widget.show_empty("无有效数据", title=f"m/z {curve['mz']} 温度曲线")
            return

        # 根据分类获取颜色
        group_key = curve.get("curve_class", "unclassified")
        color = self.CURVE_CLASS_COLORS.get(group_key, "#6b7280")

        x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
        y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1e-6)

        x_pad = 0.025 * x_span  # 2.5% 边距
        y_pad = max(0.08 * y_span, 0.03 * max(abs(y_max), 1.0))  # 8% 或最小 3%

        self.plot_widget.render_spec(
            ScientificPlotSpec(
                title=title,
                xlabel="温度 / °C",
                ylabel="归一化信号",
                series=(
                    CurveSeries(
                        key=f"temperature-{curve['mz']}",
                        role=SeriesRole.MEASUREMENT,
                        x=x_values,
                        y=y_values,
                        color=color,
                    ),
                ),
                xlim=(x_min - x_pad, x_max + x_pad),
                ylim=(y_min - y_pad, y_max + y_pad),
                show_zero_line=y_min < 0 < y_max,
            )
        )

    def update_multi_energy_plot(self, curve: dict) -> None:
        target_mz = int(curve.get("mz", 0))
        title = f"m/z {target_mz} 各能量温度响应曲线"

        x_arrays: list[np.ndarray] = []
        y_arrays: list[np.ndarray] = []
        series: list[CurveSeries] = []
        energy_curves = sorted(
            curve.get("energy_curves", []),
            key=lambda pair: (
                float(pair[0].get("energy")) if pair[0].get("energy") is not None else float("inf"),
                str(pair[0].get("folder_label", "")),
            ),
        )
        for index, (item, energy_curve) in enumerate(energy_curves):
            energy = item.get("energy")
            label = (
                f"{float(energy):.2f} eV"
                if energy is not None and np.isfinite(float(energy))
                else str(item.get("folder_label", ""))
            )
            x_values = np.asarray(energy_curve.get("temperatures", []), dtype=float)
            y_values = np.asarray(energy_curve.get("areas", []), dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            x_values = x_values[valid]
            y_values = y_values[valid]
            if x_values.size == 0 or y_values.size == 0:
                continue
            series.append(
                CurveSeries(
                    key=f"energy-{index}",
                    role=SeriesRole.COMPARISON,
                    x=x_values,
                    y=y_values,
                    label=label,
                )
            )
            x_arrays.append(x_values)
            y_arrays.append(y_values)

        if not series:
            self.plot_widget.show_empty("无有效数据", title=title)
            return

        all_x = np.concatenate(x_arrays)
        all_y = np.concatenate(y_arrays)
        x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
        y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1e-6)
        x_pad = max(8.0, x_span * 0.08)
        y_pad = max(0.02, y_span * 0.08)
        self.plot_widget.render_spec(
            ScientificPlotSpec(
                title=title,
                xlabel="温度 / °C",
                ylabel="归一化信号",
                series=tuple(series),
                show_legend=True,
                legend_loc="upper right",
                xlim=(x_min - x_pad, x_max + x_pad),
                ylim=(y_min - y_pad, y_max + y_pad),
                show_zero_line=y_min < 0 < y_max,
            )
        )

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
        ps = self._effective_analysis_settings()
        peak_config = ps.to_peak_detection_config()
        return analyze_temperature_folder(
            folder,
            calibration=ps.to_calibration(),
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
