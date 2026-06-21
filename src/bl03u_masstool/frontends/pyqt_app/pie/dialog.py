from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

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
from bl03u_masstool.core.pie_state import PieStateManager
from bl03u_masstool.core.pics_calculator import calc_pics_single_energy
from bl03u_masstool.core.elements import get_all_elements_from_database, filter_species_by_elements, COMMON_ELEMENTS, parse_formula as parse_formula_elements, get_elements_from_formula
from bl03u_masstool.core.normalization import NormalizationSettings, load_normalization_settings, save_normalization_settings
from bl03u_masstool.core.project_lifecycle import project_root
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
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread
from bl03u_masstool.frontends.pyqt_app.project_artifacts import record_project_artifact

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin, FlowLayout
from bl03u_masstool.frontends.pyqt_app.common.static_plot import StaticCurvePlot
from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersDialog
from bl03u_masstool.frontends.pyqt_app.pie.fitting_control_widget import FittingControlWidget
from bl03u_masstool.frontends.pyqt_app.pie.result_display_widget import ResultDisplayWidget

def _run_exhaustive_fit(
    candidates: list[dict],
    energies,
    intensities,
) -> list[dict]:
    """穷举候选物种所有组合，按R²降序排列。"""
    from itertools import combinations as _combinations

    from bl03u_masstool.core.pie_analysis import fit_species_combination_with_curve

    ranked: list[dict] = []
    n = len(candidates)
    for r in range(1, n + 1):
        for combo_indices in _combinations(range(n), r):
            combo = [candidates[i] for i in combo_indices]
            model = fit_species_combination_with_curve(combo, energies, intensities)
            r2 = model.get("r_squared", 0.0)
            if r2 <= 0:
                continue
            names = [s.get("species", "?") for s in combo]
            ranked.append({
                "species_names": names,
                "species_count": len(combo),
                "r_squared": r2,
                "rmse": model.get("rmse", 0.0),
                "model": model,
            })
    ranked.sort(key=lambda x: x["r_squared"], reverse=True)
    return ranked


class PIESpeciesFitDialog(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, calibration: Calibration, normalization_settings: NormalizationSettings | None = None, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.normalization_settings = normalization_settings or NormalizationSettings()
        self.project_settings: ProjectSettings | None = None
        self.project_dir: str | None = None  # Phase 3: Project directory for state persistence
        self.pie_state_dirty = False  # Phase 3 Step 2: dirty flag for unsaved config changes
        self.peak_detection = load_peak_detection_config()
        self.database: list[dict] = []
        self.analysis_df = pd.DataFrame()
        self.curves: dict[int, dict] = {}
        self.current_mz: int | None = None
        self.current_fit: dict | None = None
        self.all_fit_results: dict[int, dict] = {}  # 保存所有拟合结果
        self.worker: WorkerThread | None = None
        self.pie_folders: list[str] = []
        self._busy = False
        # Per-m/z 配置存储（Phase 1）+ 版本管理（Phase 2）
        self.per_mz_config: dict[int, dict] = {}
        # Phase 2: 全局求解配置版本管理
        # 注意：当前拟合函数无可配置的全局求解参数，仅保留接口供未来扩展
        self.global_solver_config = {}
        self._update_global_config_hash()
        self.setWindowTitle("PIE物种拟合")
        self.resize(1380, 850)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Auto-load PICS database (built-in, not user-selectable)
        self._auto_load_database()

        self.use_multi_folders = QtWidgets.QToolButton()
        self.use_multi_folders.setText("多文件夹")
        self.use_multi_folders.setCheckable(True)
        self.use_multi_folders.setChecked(False)
        self.use_multi_folders.setObjectName("ModeToggle")
        self.use_multi_folders.toggled.connect(self.toggle_multi_folder_mode)

        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("选择包含PIE扫描质谱文件的文件夹")
        self.select_folder_button = QtWidgets.QPushButton("浏览...")
        self.select_folder_button.setToolTip("选择PIE扫描文件夹")
        self.select_folder_button.clicked.connect(self.select_folder)

        self.folder_list = QtWidgets.QListWidget()
        self.folder_list.setMaximumHeight(100)
        self.add_folder_button = QtWidgets.QPushButton("添加文件夹")
        self.add_folder_button.clicked.connect(self.add_folder)
        self.remove_folder_button = QtWidgets.QPushButton("移除选中")
        self.remove_folder_button.clicked.connect(self.remove_folder)
        self.clear_folders_button = QtWidgets.QPushButton("清空列表")
        self.clear_folders_button.clicked.connect(self.clear_folders)

        self.merge_method_combo = QtWidgets.QComboBox()
        self.merge_method_combo.addItem("低能段为主 (推荐)", "low_energy_dominant")
        self.merge_method_combo.addItem("第一组为主", "first_segment_dominant")
        self.merge_method_combo.addItem("简单拼接 (不缩放)", "mean")

        self.analyze_button = QtWidgets.QPushButton("生成曲线")
        self.analyze_button.setObjectName("WorkflowButton")
        self.analyze_button.setToolTip("生成PIE曲线")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.export_button = QtWidgets.QPushButton("导出曲线")
        self.export_button.setObjectName("ExportButton")
        self.export_button.setToolTip("导出PIE曲线数据")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_curve_data)
        self.export_plot_button = QtWidgets.QPushButton("导出图表")
        self.export_plot_button.setObjectName("ExportButton")
        self.export_plot_button.setToolTip("导出当前PIE曲线图表为PNG/PDF")
        self.export_plot_button.setEnabled(False)
        self.export_plot_button.clicked.connect(self.export_plot)
        self.common_params_button = QtWidgets.QPushButton("参数")
        self.common_params_button.setObjectName("BrowseButton")
        self.common_params_button.setToolTip("打开通用参数设置")
        self.common_params_button.clicked.connect(self.open_common_parameters)
        self.summary_open_project_btn = QtWidgets.QPushButton("项目设置")
        self.summary_open_project_btn.setObjectName("BrowseButton")
        self.summary_open_project_btn.setToolTip("修改项目名、体系、数据源等")
        self.summary_open_project_btn.clicked.connect(self._open_project_settings)
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("ProjectStatus")
        self.select_folder_button.setObjectName("BrowseButton")

        # ---- 紧凑数据源控制带 ----
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
        self.summary_project_label.setObjectName("ReadoutValue")
        self.summary_system_label.setObjectName("ReadoutValue")
        self.summary_data_label.setObjectName("ReadoutValue")
        self.summary_project_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_system_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_data_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_layout.addWidget(self.summary_project_label)
        summary_layout.addWidget(self.summary_system_label)
        summary_layout.addWidget(self.summary_data_label)
        data_layout.addWidget(self.summary_bar)

        folder_row = QtWidgets.QHBoxLayout()
        folder_row.setSpacing(6)
        folder_label = QtWidgets.QLabel("PIE数据")
        folder_label.setObjectName("ReadoutLabel")
        folder_row.addWidget(folder_label)
        folder_row.addWidget(self.use_multi_folders)
        folder_row.addWidget(self.folder_edit, stretch=1)
        folder_row.addWidget(self.select_folder_button)
        folder_row.addWidget(self.analyze_button)
        folder_row.addWidget(self.export_button)
        folder_row.addWidget(self.export_plot_button)
        folder_row.addWidget(self.common_params_button)
        folder_row.addWidget(self.summary_open_project_btn)
        folder_row.addWidget(self.status_label)
        data_layout.addLayout(folder_row)

        # 多文件夹列表区域 (只在多文件夹模式显示)
        self.multi_folder_section = QtWidgets.QGroupBox("多段PIE文件夹 (按能量顺序添加)")
        multi_layout = QtWidgets.QVBoxLayout(self.multi_folder_section)
        multi_layout.setContentsMargins(5, 5, 5, 5)
        multi_layout.addWidget(self.folder_list)
        folder_btn_row = QtWidgets.QHBoxLayout()
        folder_btn_row.addWidget(self.add_folder_button)
        folder_btn_row.addWidget(self.remove_folder_button)
        folder_btn_row.addWidget(self.clear_folders_button)
        folder_btn_row.addWidget(QtWidgets.QLabel("合并:"))
        folder_btn_row.addWidget(self.merge_method_combo)
        folder_btn_row.addStretch()
        multi_layout.addLayout(folder_btn_row)
        self.multi_folder_section.setVisible(False)
        data_layout.addWidget(self.multi_folder_section)

        layout.addWidget(source_panel)

        # ── 曲线列表区拟合控制 ──
        self.fit_all_button = QtWidgets.QPushButton("拟合全部")
        self.fit_all_button.setObjectName("BrowseButton")
        self.fit_all_button.setToolTip("一键拟合所有PIE曲线（使用全局锁定物种配置）")
        self.fit_all_button.clicked.connect(self.fit_all_curves)
        self.fit_all_button.setFixedHeight(28)
        self.fit_all_button.setEnabled(False)

        self._locked_species: list[str] = []
        self._result_panel_expanded: bool = False  # 记录用户展开/收起状态，切换m/z时保留
        self._skip_save_config: bool = False  # 批量拟合完成后跳过一次save，防止覆盖fit配置

        # 初始化UI状态
        self.toggle_multi_folder_mode(0)

        # ---- 主工作区 ----
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("SidePanel")
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(360)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        self.summary_label = QtWidgets.QLabel("未生成PIE曲线")
        self.summary_label.setObjectName("ProjectStatus")
        list_header = QtWidgets.QHBoxLayout()
        list_header.setSpacing(6)
        list_title = QtWidgets.QLabel("m/z曲线")
        list_title.setObjectName("ReadoutLabel")
        list_header.addWidget(list_title)
        list_header.addWidget(self.summary_label, stretch=1)
        left_layout.addLayout(list_header)
        self.mz_filter_edit = QtWidgets.QLineEdit()
        self.mz_filter_edit.setObjectName("CurveSearch")
        self.mz_filter_edit.setPlaceholderText("搜索 m/z / 物种")
        self.mz_filter_edit.textChanged.connect(self.populate_mz_list)
        left_layout.addWidget(self.mz_filter_edit)
        self.mz_list = QtWidgets.QListWidget()
        self.mz_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.mz_list.currentItemChanged.connect(self.on_mz_selected)
        self.mz_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.mz_list.customContextMenuRequested.connect(self._on_mz_list_context_menu)
        left_layout.addWidget(self.mz_list, stretch=1)
        mz_hint = QtWidgets.QLabel("按住 Cmd/Ctrl 可多选")
        mz_hint.setObjectName("HintLabel")
        left_layout.addWidget(mz_hint)

        # 主拟合按钮：智能感知多选
        self.fit_button = QtWidgets.QPushButton("拟合当前")
        self.fit_button.setObjectName("PrimaryToolbarButton")
        self.fit_button.setToolTip(
            "拟合当前选中的 m/z 曲线（使用右侧面板配置）\n"
            "多选时将对每条曲线应用当前面板配置批量拟合"
        )
        self.fit_button.clicked.connect(self.fit_current_curve)
        self.fit_button.setEnabled(False)
        left_layout.addWidget(self.fit_button)

        # 次级按钮行：拟合全部 + 更多操作
        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(self.fit_all_button)

        # "更多操作"菜单
        self.more_actions_btn = QtWidgets.QPushButton("更多操作")
        self.more_actions_btn.setObjectName("BrowseButton")
        self.more_actions_btn.setToolTip("批量重拟合、穷举优选、清除拟合等高级操作")
        self.more_actions_menu = QtWidgets.QMenu(self)
        self.more_actions_menu.addAction(
            "批量重拟合（已保存配置）"
        ).triggered.connect(self.refit_selected_curves)
        self.more_actions_menu.addAction("穷举优选").triggered.connect(self._exhaustive_best_fit)
        self.more_actions_menu.addAction("清除拟合").triggered.connect(self.clear_all_fits)
        self.more_actions_menu.addSeparator()
        self.more_actions_menu.addAction("重新加载PICS截面数据库").triggered.connect(
            lambda: self.load_database(show_message=True)
        )
        self.more_actions_btn.setMenu(self.more_actions_menu)
        action_row.addWidget(self.more_actions_btn)
        left_layout.addLayout(action_row)
        splitter.addWidget(left_panel)

        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        plot_container = QtWidgets.QWidget()
        plot_container.setObjectName("PlotPanel")
        plot_container.setMinimumHeight(280)
        plot_container_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_container_layout.setContentsMargins(8, 8, 8, 8)
        plot_container_layout.setSpacing(6)

        # Plot stack: empty state + actual plot
        self._plot_stack = QtWidgets.QStackedLayout()

        # Empty state
        self._empty_state = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(self._empty_state)
        empty_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_msg = QtWidgets.QLabel('尚未生成PIE曲线\n\n选择包含PIE数据的文件夹后，点击"生成曲线"')
        empty_msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_msg.setObjectName("ProjectHint")
        empty_msg.setWordWrap(True)
        empty_layout.addWidget(empty_msg)
        self._plot_stack.addWidget(self._empty_state)

        self.plot_widget = StaticCurvePlot("Photon Energy (eV)", "Normalized Intensity", min_height=280)
        self._plot_stack.addWidget(self.plot_widget)
        self._plot_stack.setCurrentIndex(0)  # Show empty state initially

        plot_container_layout.addLayout(self._plot_stack, stretch=1)

        # 拟合统计条属于图表区域，不作为 splitter 的独立面板，避免挤占下方功能区。
        stats_bar = QtWidgets.QFrame()
        stats_bar.setObjectName("StatsBar")
        stats_bar.setFixedHeight(36)  # Adjusted from 32px to 36px for better visual proportion
        stats_bar_layout = QtWidgets.QHBoxLayout(stats_bar)
        stats_bar_layout.setContentsMargins(10, 4, 10, 4)  # Adjusted from (10, 2, 10, 2) for 36px height
        stats_bar_layout.setSpacing(12)
        stats_title = QtWidgets.QLabel("拟合统计")
        stats_title.setObjectName("StatsTitle")
        self.fit_stats_label = QtWidgets.QLabel("已拟合: <b>0</b> / 0 条曲线&nbsp;&nbsp;&nbsp;平均R\u00b2: N/A")
        self.fit_stats_label.setObjectName("HintLabel")
        self.fit_stats_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        stats_bar_layout.addWidget(stats_title)
        stats_bar_layout.addWidget(self.fit_stats_label, stretch=1)

        # 添加"结果详情 ›"轻量链接到统计条右侧
        self.show_result_detail_btn = QtWidgets.QPushButton("详细数据 ›")
        self.show_result_detail_btn.setObjectName("ResultDetailLink")
        self.show_result_detail_btn.setToolTip("展开/收起详细数据面板（曲线数据、残差等）")
        self.show_result_detail_btn.setFixedHeight(20)
        self.show_result_detail_btn.setStyleSheet("""
            #ResultDetailLink {
                border: none;
                background-color: transparent;
                color: #0369a1;
                padding: 2px 4px;
                text-decoration: underline;
                font-size: 12px;
            }
            #ResultDetailLink:hover {
                color: #0284c7;
            }
        """)
        self.show_result_detail_btn.setVisible(False)  # 初始隐藏，拟合完成后显示
        stats_bar_layout.addWidget(self.show_result_detail_btn)

        plot_container_layout.addWidget(stats_bar)
        self.right_splitter.addWidget(plot_container)

        # ---- 创建新的widget实例（Phase 4: UI分离） ----
        self.fitting_control_widget = FittingControlWidget(self)
        self.result_display_widget = ResultDisplayWidget(self)

        # ---- 配置中间splitter的2层布局 ----
        # right_splitter 现在用作中间区域：plot_container + result_display
        self.right_splitter.addWidget(self.result_display_widget)

        # ---- 中间区域尺寸分配 ----
        self.right_splitter.setStretchFactor(0, 1)   # 图表：可伸缩
        self.right_splitter.setStretchFactor(1, 0)   # 结果详情：可收起

        # ---- 中间区域可收起性 ----
        self.right_splitter.setCollapsible(0, False)  # 图表：不可收起
        self.right_splitter.setCollapsible(1, True)   # 结果详情：可收起

        # ---- 配置拟合控制widget的约束 ----
        self.fitting_control_widget.setMinimumWidth(400)
        self.fitting_control_widget.setMaximumWidth(460)

        # ---- 连接FittingControlWidget信号 ----
        self.fitting_control_widget.locked_candidate_added.connect(self._on_locked_species_changed)
        self.fitting_control_widget.locked_candidate_removed.connect(self._on_locked_species_changed)
        self.fitting_control_widget.species_config_changed.connect(self._on_fitting_config_changed)
        self.fitting_control_widget.candidates_zeroed.connect(self._on_fitting_config_changed)
        self.fitting_control_widget.candidate_lock_toggled.connect(self._on_locked_species_changed)
        self.fitting_control_widget.confirmation_requested.connect(self._confirm_identification)
        self.fitting_control_widget.export_requested.connect(self.export_pie_results)
        self.fitting_control_widget.pics_import_requested.connect(self._goto_pics_import)

        # ---- 连接ResultDisplayWidget信号 ----
        # （结果面板现仅作详细数据查看，操作按钮已移入右侧面板）

        # ---- 连接"查看结果详情"按钮 ----
        self.show_result_detail_btn.clicked.connect(lambda _: self._show_result_detail_popup())

        # ---- 重构主splitter为3层水平布局 ----
        # 原始：splitter 有2个child (left_panel, right_splitter)
        # 改为：splitter 有3个child (left_panel, middle_splitter, fitting_control_widget)
        # 将right_splitter改名为middle_splitter以体现新用途
        middle_splitter = self.right_splitter
        splitter.addWidget(middle_splitter)
        splitter.addWidget(self.fitting_control_widget)

        # ---- 主splitter尺寸分配 ----
        splitter.setStretchFactor(0, 0)   # 左侧：固定宽度
        splitter.setStretchFactor(1, 1)   # 中间：可伸缩
        splitter.setStretchFactor(2, 0)   # 右侧：固定宽度

        # ---- 主splitter可收起性 ----
        splitter.setCollapsible(0, False)  # 左侧：不可收起
        splitter.setCollapsible(1, False)  # 中间：不可收起
        splitter.setCollapsible(2, False)  # 右侧：不可收起

        # ---- 初始尺寸 ----
        # 左: 310px (m/z列表)，中: 自动伸缩(图表区)，右: 430px (拟合配置+结果)
        splitter.setSizes([310, 1000, 430])
        layout.addWidget(splitter, stretch=1)
        self._update_action_state()

        # ---- 属性别名（保持向后兼容） ----
        # 结果显示widget中的组件
        self.curve_table = self.result_display_widget.curve_table
        self.fit_table = self.result_display_widget.fit_table

        # 拟合控制widget中的组件（为兼容性创建引用）
        self.species_table = self.fitting_control_widget.species_table
        # 向后兼容性别名
        self.candidate_table = self.species_table

    def _open_project_settings(self):
        """跳转到项目管理页面。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("project")

    def _goto_pics_import(self):
        """跳转到 PICS 导入页面（处理 pics_import_requested 信号）。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("pics_import")

    def _update_result_detail_button(self):
        """根据结果状态更新'结果详情 ›'按钮的可见性和文本，并保留用户展开状态"""
        if self.current_mz is None or self.current_mz not in self.all_fit_results:
            self.show_result_detail_btn.setVisible(False)
            # 无结果时收起面板（不改变记录的展开状态）
            if self.result_display_widget.isVisible():
                self._apply_result_panel_visibility(False, update_state=False)
            return

        result = self.all_fit_results[self.current_mz]
        current_per_mz_hash = self._get_per_mz_config_hash(self.current_mz)
        current_global_hash = self.global_solver_config.get("config_hash", "")
        status = self._derive_result_status(result, current_per_mz_hash, current_global_hash)

        # 根据状态设置按钮文本和可见性
        if status == "COMPLETED":
            self.show_result_detail_btn.setVisible(True)
            if self._result_panel_expanded:
                self._apply_result_panel_visibility(True, update_state=False)
                self.show_result_detail_btn.setText("收起详细数据")
            else:
                self._apply_result_panel_visibility(False, update_state=False)
                self.show_result_detail_btn.setText("详细数据 ›")
        elif status == "OBSOLETE":
            self.show_result_detail_btn.setVisible(True)
            if self._result_panel_expanded:
                self._apply_result_panel_visibility(True, update_state=False)
                self.show_result_detail_btn.setText("收起详细数据")
            else:
                self._apply_result_panel_visibility(False, update_state=False)
                self.show_result_detail_btn.setText("[警告] 查看过期数据")
        elif status == "FAILED":
            self.show_result_detail_btn.setVisible(True)
            self._apply_result_panel_visibility(False, update_state=False)
            self.show_result_detail_btn.setText("查看失败数据")
        else:  # UNFITTED
            self.show_result_detail_btn.setVisible(False)
            self._apply_result_panel_visibility(False, update_state=False)

    def _show_result_detail_popup(self):
        """弹出子窗口显示详细数据（曲线数据、拟合明细）"""
        popup = QtWidgets.QDialog(self)
        popup.setWindowTitle(f"详细数据 - m/z {self.current_mz}")
        popup.resize(900, 600)
        popup.setWindowFlags(
            QtCore.Qt.WindowType.Dialog |
            QtCore.Qt.WindowType.WindowCloseButtonHint |
            QtCore.Qt.WindowType.WindowMaximizeButtonHint
        )
        layout = QtWidgets.QVBoxLayout(popup)
        layout.setContentsMargins(8, 8, 8, 8)

        # 将 result_display_widget 的当前内容复制到 popup 中
        # 使用 tab widget 显示曲线数据和拟合明细
        tab = QtWidgets.QTabWidget()

        # 曲线数据 tab
        curve_tab = QtWidgets.QWidget()
        curve_layout = QtWidgets.QVBoxLayout(curve_tab)
        curve_table_copy = QtWidgets.QTableWidget()
        curve_table_copy.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        curve_table_copy.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        # 复制曲线数据
        src = self.curve_table
        curve_table_copy.setRowCount(src.rowCount())
        curve_table_copy.setColumnCount(src.columnCount())
        headers = [src.horizontalHeaderItem(i).text() if src.horizontalHeaderItem(i) else "" for i in range(src.columnCount())]
        curve_table_copy.setHorizontalHeaderLabels(headers)
        for r in range(src.rowCount()):
            for c in range(src.columnCount()):
                item = src.item(r, c)
                if item:
                    curve_table_copy.setItem(r, c, QtWidgets.QTableWidgetItem(item.text()))
        curve_layout.addWidget(curve_table_copy)
        tab.addTab(curve_tab, "曲线数据")

        # 拟合明细 tab
        fit_tab = QtWidgets.QWidget()
        fit_layout = QtWidgets.QVBoxLayout(fit_tab)
        fit_table_copy = QtWidgets.QTableWidget()
        fit_table_copy.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        fit_table_copy.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        # 复制拟合数据
        src = self.fit_table
        fit_table_copy.setRowCount(src.rowCount())
        fit_table_copy.setColumnCount(src.columnCount())
        headers = [src.horizontalHeaderItem(i).text() if src.horizontalHeaderItem(i) else "" for i in range(src.columnCount())]
        fit_table_copy.setHorizontalHeaderLabels(headers)
        for r in range(src.rowCount()):
            for c in range(src.columnCount()):
                item = src.item(r, c)
                if item:
                    fit_table_copy.setItem(r, c, QtWidgets.QTableWidgetItem(item.text()))
        fit_layout.addWidget(fit_table_copy)
        tab.addTab(fit_tab, "拟合明细")

        layout.addWidget(tab)

        # 关闭按钮
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(popup.accept)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        popup.exec()

    def _toggle_result_display(self, visible: bool | None = None) -> None:
        """展开/收起结果详情面板（保留供内部使用）"""
        if visible is None:
            visible = not self.result_display_widget.isVisible()
        self._apply_result_panel_visibility(visible, update_state=True)

    def _apply_result_panel_visibility(self, visible: bool, update_state: bool = True) -> None:
        """实际执行展开/收起，可控制是否更新 _result_panel_expanded 记录"""
        if visible:
            self.result_display_widget.setVisible(True)
            heights = self.right_splitter.sizes()
            total = sum(heights)
            self.right_splitter.setSizes([int(total * 0.65), int(total * 0.35)])
            self.show_result_detail_btn.setText("收起详细数据")
        else:
            self.result_display_widget.setVisible(False)
            heights = self.right_splitter.sizes()
            total = sum(heights)
            self.right_splitter.setSizes([total, 0])
            self.show_result_detail_btn.setText("详细数据 ›")
        if update_state:
            self._result_panel_expanded = visible

    def _on_locked_species_changed(self):
        """锁定候选物种改变时标记dirty并更新状态"""
        self.mark_pie_config_changed()
        # 触发实时拟合预览
        self._rebuild_manual_fit()

    def _on_fitting_config_changed(self):
        """拟合配置改变时标记dirty、更新结果状态并重新拟合预览"""
        self.mark_pie_config_changed()
        # 如果有当前的拟合结果，标记为OBSOLETE
        if self.current_mz and self.all_fit_results.get(self.current_mz):
            current_result = self.all_fit_results[self.current_mz]
            if current_result.get("_status") == "COMPLETED":
                current_result["_status"] = "OBSOLETE"
                self.result_display_widget.set_result_status("OBSOLETE")
        # 触发实时拟合预览
        self._rebuild_manual_fit()

    def _update_species_table_coefficients(self, fit_model: dict):
        """将拟合结果中的系数更新到右侧候选物种表格"""
        if not fit_model:
            return

        species_results = fit_model.get("species", [])
        if not species_results:
            return

        # 构建物种名到系数的映射
        coeff_map = {item.get("species", ""): float(item.get("coefficient", 0.0)) for item in species_results}

        # 遍历表格并更新系数
        self.fitting_control_widget._updating = True
        try:
            for row in range(self.species_table.rowCount()):
                # 获取物种名（从 Col 1 的 widget 中）
                species_widget = self.species_table.cellWidget(row, 1)
                if species_widget:
                    name_label = species_widget.findChild(QtWidgets.QLabel)
                    if name_label:
                        species_name = name_label.text()
                        # 如果这个物种在拟合结果中，更新系数
                        if species_name in coeff_map:
                            coeff_widget = self.species_table.cellWidget(row, 2)
                            if isinstance(coeff_widget, QtWidgets.QDoubleSpinBox):
                                coeff_widget.setValue(coeff_map[species_name])
                                # 也更新数据结构
                                if row < len(self.fitting_control_widget._unified_species_data):
                                    self.fitting_control_widget._unified_species_data[row]["coefficient"] = coeff_map[species_name]
        finally:
            self.fitting_control_widget._updating = False

    def load_database(self, show_message: bool = True):
        path = self.database_edit.text().strip() if hasattr(self, "database_edit") else ""
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择PICS截面数据库",
                "",
                "SQLite Files (*.sqlite *.sqlite3 *.db);;All Files (*)",
            )
            if not path:
                return
            if hasattr(self, "database_edit"):
                self.database_edit.setText(path)
        try:
            self.database, _ = load_species_database(path)
            self.status_label.setText(f"已加载PICS截面数据库: {len(self.database)} 个物种")
            if show_message:
                QtWidgets.QMessageBox.information(self, "完成", f"已加载 {len(self.database)} 个物种")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "错误", str(exc))

    def _auto_load_database(self):
        """自动加载内置PICS截面数据库"""
        try:
            db_path = species_database_path()
            if db_path.exists():
                self.database, _ = load_species_database(str(db_path))
        except Exception:
            pass

    # ---- Phase 2: 配置版本管理和过期状态追踪 ----

    @staticmethod
    def _normalize_float(value: float) -> float:
        """规范化浮点数，处理 NaN 和 Infinity"""
        if isinstance(value, (int, float)):
            if np.isnan(value):
                return 0.0
            elif np.isinf(value):
                return float('inf') if value > 0 else float('-inf')
            return round(float(value), 10)
        return 0.0

    @staticmethod
    def _build_per_mz_config_payload(config: dict) -> dict:
        """
        从 per_mz_config 提取纯业务配置（不包括哈希字段）进行规范化。
        确保顺序一致，以产生稳定的哈希。
        """
        selected_species = config.get("selected_species", [])
        # 按 species ID 排序以保证顺序一致
        sorted_species = sorted(
            [{"id": int(s.get("id", 0))} for s in selected_species],
            key=lambda x: x["id"]
        )

        coefficients = config.get("coefficients", {})
        # 按 key 排序，规范化浮点数
        normalized_coefficients = {
            str(k): PIESpeciesFitDialog._normalize_float(v)
            for k, v in sorted(coefficients.items())
        }

        locked_ids = sorted([int(x) for x in config.get("locked_ids", [])])

        return {
            "selected_species_ids": [s["id"] for s in sorted_species],
            "coefficients": normalized_coefficients,
            "locked_ids": locked_ids,
        }

    @staticmethod
    def _compute_config_hash(config_payload: dict) -> str:
        """
        计算配置的规范化哈希。
        使用 JSON 规范化和 SHA-256 确保跨进程稳定性。
        """
        json_str = json.dumps(
            config_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _update_global_config_hash(self) -> None:
        """更新全局配置哈希。当前无真实全局求解参数，仅作接口预留。"""
        payload = {
            "solver_type": "nonnegative_lsq",  # 当前固定为非负最小二乘
        }
        self.global_solver_config["config_hash"] = self._compute_config_hash(payload)

    def _get_per_mz_config_hash(self, mz: int) -> str:
        """获取指定 m/z 的当前配置哈希。"""
        if mz not in self.per_mz_config:
            # 无历史配置，使用默认值
            default_config = {
                "selected_species": [],
                "mode": "auto",
                "coefficients": {},
                "locked_ids": [],
            }
            payload = self._build_per_mz_config_payload(default_config)
        else:
            config = self.per_mz_config[mz]
            payload = self._build_per_mz_config_payload(config)
        return self._compute_config_hash(payload)

    def _get_current_per_mz_config_hash(self, mz: int | None = None) -> str:
        """获取指定 m/z 的当前候选面板配置哈希（用于检测变化）。"""
        if mz is None:
            mz = self.current_mz
        if mz is None:
            return ""

        panel_state = self._get_candidate_panel_state()
        # 只哈希候选物种、系数（来自当前 UI）
        config = {
            "selected_species": panel_state.get("selected_species", []),
            "coefficients": panel_state.get("coefficients", {}),
            "locked_ids": panel_state.get("locked_ids", []),
        }
        payload = self._build_per_mz_config_payload(config)
        return self._compute_config_hash(payload)

    @staticmethod
    def _derive_result_status(
        result: dict | None,
        current_per_mz_hash: str,
        current_global_hash: str,
    ) -> str:
        """
        根据哈希比较推导结果状态，而不是存储字符串状态。

        返回值：
        - "UNFITTED": 无结果
        - "COMPLETED": 拟合完成且配置未改
        - "OBSOLETE": 拟合结果存在但配置已改
        - "FAILED": 拟合失败
        """
        if result is None:
            return "UNFITTED"

        if not result.get("success"):
            return "FAILED"

        # 检查配置是否改变
        fit_per_mz_hash = result.get("fit_config_hash", "")
        fit_global_hash = result.get("global_config_hash", "")

        # 如果当前哈希与拟合时的哈希相同，结果仍有效
        if fit_per_mz_hash == current_per_mz_hash and fit_global_hash == current_global_hash:
            return "COMPLETED"

        # 否则标记为过期
        return "OBSOLETE"

    @staticmethod
    def _get_status_display(status_code: str) -> tuple[str, str]:
        """
        获取状态码对应的显示文本。
        返回: (显示文本, 预留字段)
        """
        status_map = {
            "UNFITTED": ("待拟合", ""),
            "COMPLETED": ("已拟合", ""),
            "OBSOLETE": ("结果已过期", ""),
            "FAILED": ("拟合失败", ""),
        }
        return status_map.get(status_code, ("待拟合", ""))

    # ---- Per-m/z 配置管理 ----

    def _save_current_mz_config(self) -> None:
        """保存当前 m/z 的候选物种配置到 per_mz_config"""
        if self.current_mz is None:
            return
        config = self._get_candidate_panel_state()
        self.per_mz_config[self.current_mz] = {
            "selected_species": config["selected_species"],
            "coefficients": config["coefficients"],
            "locked_ids": config["locked_ids"],
        }

    def _restore_mz_config(self, mz: int) -> None:
        """从 per_mz_config 恢复指定 m/z 的配置到 UI表格"""
        if mz not in self.per_mz_config:
            # 如果没有历史配置，初始化默认配置（全选）
            self.fitting_control_widget._set_all_rows_checked(True)
            return

        config = self.per_mz_config[mz]
        selected_species = config.get("selected_species", [])
        coefficients = config.get("coefficients", {})
        locked_ids = config.get("locked_ids", [])

        # 还原物种选择（直接更新数据结构，然后刷新表格）
        selected_species_ids = {int(s.get("id", -1)) for s in selected_species}
        locked_ids_set = set(locked_ids)
        unified_data = self.fitting_control_widget._unified_species_data
        for species in unified_data:
            species_id = int(species.get("id", -1))
            species["is_enabled"] = species_id in selected_species_ids
            species["coefficient"] = coefficients.get(species_id, 0.0)
            species["is_locked"] = species_id in locked_ids_set

        # 同步锁定列表
        self.fitting_control_widget._locked_species = [
            s.get("species") for s in unified_data if s.get("is_locked")
        ]

        # 刷新表格显示
        self.fitting_control_widget._refresh_table_from_data()

    # ---- 候选物种面板方法 ----

    def _populate_candidate_table(self, mz: int):
        """根据选中的m/z填充统一的拟合物种表格"""
        filtered_db = self.get_filtered_database()
        locked_species = self.fitting_control_widget.get_locked_species()
        self.fitting_control_widget.populate_unified_species_table(mz, filtered_db, locked_species)

    def _get_candidate_panel_state(self) -> dict:
        """从统一的物种表格中获取当前配置状态"""
        selected_species = []
        locked_ids = []
        coefficients = {}

        # 从统一的数据结构中提取
        unified_data = self.fitting_control_widget._unified_species_data

        for row, species in enumerate(unified_data):
            # Col 0: 启用状态 checkbox
            enable_widget = self.species_table.cellWidget(row, 0)
            enable_chk = enable_widget.findChild(QtWidgets.QCheckBox) if enable_widget else None
            if enable_chk and enable_chk.isChecked():
                selected_species.append(species)

                species_id = int(species.get("id", row + 1))

                # Col 2: 系数（QDoubleSpinBox）
                coeff_widget = self.species_table.cellWidget(row, 2)
                if isinstance(coeff_widget, QtWidgets.QDoubleSpinBox):
                    coefficients[species_id] = coeff_widget.value()

                # 锁定状态：从数据结构中读取 is_locked 字段
                if species.get("is_locked", False):
                    locked_ids.append(species_id)

        return {
            "selected_species": selected_species,
            "coefficients": coefficients,
            "locked_ids": locked_ids,
        }

    def _rebuild_manual_fit(self):
        """使用候选面板状态重建拟合曲线(NNLS客户端预览)"""
        if self.current_mz is None or self.current_mz not in self.curves:
            return
        if self.fitting_control_widget._candidate_updating:
            return

        panel_state = self._get_candidate_panel_state()
        selected = panel_state["selected_species"]

        if not selected:
            if self.current_fit is not None:
                self.current_fit = None
                self.refresh_current_plot()
            return

        curve = self.curves[self.current_mz]
        energies = np.array(curve.get("energies", []), dtype=float)
        intensities = np.array(curve.get("intensities", []), dtype=float)

        if len(energies) == 0 or len(intensities) == 0:
            return

        point_count = min(energies.size, intensities.size)
        energies = energies[:point_count]
        intensities = intensities[:point_count]

        design_columns = []
        for species in selected:
            pic_energies = np.asarray(species.get("energies", []), dtype=float)
            pic_sections = np.asarray(species.get("cross_sections", []), dtype=float)
            pic_count = min(pic_energies.size, pic_sections.size)
            pic_energies = pic_energies[:pic_count]
            pic_sections = pic_sections[:pic_count]
            pic_valid = np.isfinite(pic_energies) & np.isfinite(pic_sections)
            pic_energies = pic_energies[pic_valid]
            pic_sections = pic_sections[pic_valid]
            if pic_energies.size < 2:
                design_columns.append(np.zeros(energies.size))
                continue
            order = np.argsort(pic_energies)
            basis = np.interp(energies, pic_energies[order], pic_sections[order], left=0.0, right=0.0)
            basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
            design_columns.append(basis)

        if not design_columns:
            return

        design = np.column_stack(design_columns)

        # 检查用户是否手动编辑了系数
        user_coefficients = panel_state["coefficients"]
        species_ids = [int(s.get("id", idx + 1)) for idx, s in enumerate(selected)]
        has_manual_coefficients = any(user_coefficients.get(sid, 0.0) > 0.0 for sid in species_ids)

        if has_manual_coefficients:
            # 用户手动编辑了系数，直接使用用户的值
            coeffs = np.array([
                user_coefficients.get(species_id, 0.0)
                for species_id in species_ids
            ], dtype=float)
        else:
            # 没有手动编辑，使用 NNLS 自动优化
            try:
                from scipy.optimize import nnls
                coeffs, _ = nnls(design, intensities)
            except Exception:
                return

        fitted = design @ coeffs
        ss_res = np.sum((intensities - fitted) ** 2)
        ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        species_results = []
        for idx, (species, c) in enumerate(zip(selected, coeffs)):
            component = design_columns[idx] * c
            species_results.append({
                "species": species.get("species", ""),
                "coefficient": float(c),
                "ie": species.get("ionization_energy"),
                "component_intensities": component.tolist(),
                "contribution_percent": float(c) / sum(coeffs) * 100 if sum(coeffs) > 0 else 0,
                "r_squared": r_squared,
            })

        manual_model = {
            "energies": energies.tolist(),
            "experimental": intensities.tolist(),
            "fitted": fitted.tolist(),
            "total_fit": fitted.tolist(),
            "residual": (intensities - fitted).tolist(),
            "species": species_results,
            "r_squared": r_squared,
            "candidate_count": len(selected),
        }

        self.current_fit = manual_model
        curve = self.curves[self.current_mz]
        self.update_plot(curve, manual_model)

        # 更新结果显示widget的曲线数据，但不改变状态标记
        self.result_display_widget.update_curve_data({
            "energies": energies.tolist(),
            "experimental": intensities.tolist(),
            "total_fit": fitted.tolist(),
            "residual": (intensities - fitted).tolist(),
        })
        self.result_display_widget.update_fit_table(manual_model)

    def get_filtered_database(self) -> list:
        """获取根据全局元素设置筛选后的数据库"""
        selected_elements = set(self.normalization_settings.selected_elements)
        if not selected_elements:
            return self.database
        return filter_species_by_elements(self.database, selected_elements)

    def fit_all_curves(self):
        """一键拟合所有曲线"""
        # 防止并发拟合
        if self._busy:
            return
        if not self.curves:
            QtWidgets.QMessageBox.warning(self, "提示", "请先生成PIE曲线")
            return

        self.set_busy(True, "正在一键拟合所有曲线...")
        self.worker = WorkerThread(
            lambda: self.fit_curves_sync(list(self.curves.keys())),
            self,
        )
        self.worker.finished_with_result.connect(self.on_fit_all_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def fit_curves_sync(self, mz_list: list) -> dict:
        """拟合指定的质量数曲线"""
        filtered_db = self.get_filtered_database()
        locked_species = self.fitting_control_widget.get_locked_species()

        # Phase 2: 获取全局配置哈希（对所有 m/z 相同）
        global_hash = self.global_solver_config.get("config_hash", "")

        results = {}
        for mz in mz_list:
            # Phase 2: 为这个 m/z 捕获启动时的配置哈希
            per_mz_hash = self._get_per_mz_config_hash(mz)

            curve = self.curves.get(mz)
            if not curve:
                continue
            fit_result = self._fit_curve(mz, curve, filtered_db, locked_species)

            # Phase 2: 添加版本快照到结果
            if fit_result.get('success'):
                fit_result['fit_config_hash'] = per_mz_hash
                fit_result['global_config_hash'] = global_hash
                fit_result['fit_timestamp'] = time.time()

            results[mz] = fit_result
        return results

    def _fit_curve(self, mz: int, curve: dict, database: list, locked_species: list) -> dict:
        """拟合单条曲线"""
        energies = np.array(curve.get('energies', []))
        intensities = np.array(curve.get('intensities', []))

        if len(energies) == 0 or len(intensities) == 0:
            return {'success': False, 'error': '无数据'}

        fit_model = identify_species_for_mz_with_curve(
            database, mz, energies, intensities, locked_species=locked_species
        )

        if not fit_model or not fit_model.get('species'):
            return {'success': False, 'error': '无匹配物种', 'model': None}

        return {
            'success': True,
            'model': fit_model,
            'species': fit_model.get('species', [])[:3],
            'r_squared': fit_model.get('r_squared', 0.0)
        }

    def on_fit_all_complete(self, results: dict):
        """一键拟合完成"""
        # 保存所有拟合结果
        self.all_fit_results = results

        fitted_count = sum(1 for r in results.values() if r.get('success'))
        total_count = len(self.curves)

        all_r_squared = [r.get('r_squared', 0.0) for r in results.values() if r.get('success')]
        avg_r_squared = sum(all_r_squared) / len(all_r_squared) if all_r_squared else 0.0

        self._update_fit_stats(fitted_count, total_count, avg_r_squared)

        if fitted_count > 0:
            first_mz, first_result = next(
                (mz, result) for mz, result in results.items() if result.get('success') and result.get('model')
            )
            self.current_mz = first_mz

            # 为所有成功拟合的 m/z 保存配置，使哈希匹配、状态显示为 COMPLETED
            for mz, result in results.items():
                if result.get('success') and result.get('model'):
                    fit_species = result['model'].get('species', [])
                    self.per_mz_config[mz] = {
                        "selected_species": fit_species,
                        "coefficients": {int(s.get("id", idx + 1)): float(s.get("coefficient", 0.0))
                                         for idx, s in enumerate(fit_species)},
                        "locked_ids": [],
                    }
                    # 重新用刚保存的配置计算哈希，确保 _derive_result_status 判断为 COMPLETED
                    new_hash = self._get_per_mz_config_hash(mz)
                    self.all_fit_results[mz]['fit_config_hash'] = new_hash

            if first_result.get('success') and first_result.get('model'):
                self.current_fit = first_result['model']

                results_list = first_result['model'].get('species', [])
                fit_df = pd.DataFrame(
                    [
                        {
                            "物种名称": item["species"],
                            "电离能(eV)": "" if item.get("ie") is None else round(float(item["ie"]), 4),
                            "匹配系数": round(float(item["coefficient"]), 6),
                            "贡献(%)": round(float(item["contribution_percent"]), 2),
                            "R²": round(float(item["r_squared"]), 5),
                        }
                        for item in results_list
                    ],
                    columns=["物种名称", "电离能(eV)", "匹配系数", "贡献(%)", "R²"],
                )
                self.set_dataframe(self.fit_table, fit_df)

                if first_mz in self.curves:
                    self.update_plot(self.curves[first_mz], first_result['model'])

            self.refresh_current_plot()
            # 设置标志位，避免 setCurrentRow 触发 on_mz_selected 时覆盖刚保存的 fit 配置
            self._skip_save_config = True
            self.mz_list.setCurrentRow(0)

        self.populate_mz_list()
        self._update_action_state()

        QtWidgets.QMessageBox.information(
            self, "完成",
            f"拟合完成！\n已拟合: {fitted_count} / {total_count} 条曲线"
        )

    def _on_mz_list_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        fit_all_action = menu.addAction("拟合全部曲线")
        action = menu.exec(self.mz_list.mapToGlobal(pos))
        if action == fit_all_action:
            self.fit_all_curves()

    def refit_selected_curves(self):
        """拟合选中的曲线，使用各自保存的候选物种配置"""
        # 防止并发拟合
        if self._busy:
            return
        selected_items = self.mz_list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择要拟合的质量数")
            return

        mz_list = []
        for item in selected_items:
            mz = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if mz is None:
                continue
            try:
                mz_list.append(int(mz))
            except (TypeError, ValueError):
                continue

        if not mz_list:
            return

        self.set_busy(True, f"正在拟合 {len(mz_list)} 条曲线...")
        self.worker = WorkerThread(
            lambda: self.fit_selected_curves_sync(mz_list),
            self,
        )
        self.worker.finished_with_result.connect(self.on_refit_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def fit_selected_curves_sync(self, mz_list: list) -> dict:
        """使用 per-m/z 配置拟合选中的曲线"""
        from bl03u_masstool.core.pie_analysis import fit_species_combination_with_curve

        # Phase 2: 获取全局配置哈希（对所有 m/z 相同）
        global_hash = self.global_solver_config.get("config_hash", "")

        results = {}
        for mz in mz_list:
            # Phase 2: 为这个 m/z 捕获启动时的配置哈希
            per_mz_hash = self._get_per_mz_config_hash(mz)

            curve = self.curves.get(mz)
            if not curve:
                continue

            energies = np.array(curve.get('energies', []))
            intensities = np.array(curve.get('intensities', []))

            if len(energies) == 0 or len(intensities) == 0:
                results[mz] = {'success': False, 'error': '无数据'}
                continue

            # 获取该 m/z 保存的候选配置
            if mz in self.per_mz_config:
                config = self.per_mz_config[mz]
                selected_species = config.get("selected_species", [])
                coefficients = config.get("coefficients", {})
                locked_ids = config.get("locked_ids", [])
            else:
                # 如果没有保存的配置，使用全局自动识别
                filtered_db = self.get_filtered_database()
                selected_species = [item for item in filtered_db if item.get("mz") == mz]
                coefficients = {}
                locked_ids = []

            if not selected_species:
                results[mz] = {'success': False, 'error': '无匹配物种'}
                continue

            # 检查是否有手动设置的系数
            species_ids = [int(s.get("id", idx + 1)) for idx, s in enumerate(selected_species)]
            has_manual_coefficients = any(coefficients.get(sid, 0.0) > 0.0 for sid in species_ids)

            if has_manual_coefficients:
                # 使用保存的系数
                fit_model = fit_species_combination_with_curve(
                    selected_species,
                    energies,
                    intensities,
                    coefficient_mode="manual", coefficients=coefficients,
                )
            else:
                # 使用 NNLS 自动优化
                fit_model = fit_species_combination_with_curve(
                    selected_species,
                    energies,
                    intensities,
                )

            fit_result_species = fit_model.get('species', [])
            # Phase 2: 保存版本快照
            results[mz] = {
                'success': bool(fit_result_species),
                'model': fit_model,
                'species': fit_result_species[:3],
                'r_squared': fit_model.get('r_squared', 0.0),
                # Phase 2 新增：版本快照
                'fit_config_hash': per_mz_hash,
                'global_config_hash': global_hash,
                'fit_timestamp': time.time(),
            }

        return results

    def on_refit_complete(self, results: dict):
        """重新拟合完成"""
        # 更新保存的拟合结果
        for mz, result in results.items():
            if result.get('success'):
                self.all_fit_results[mz] = result

        # Phase 3 Step 3: Mark results dirty when refit completes
        self.pie_state_dirty = True

        fitted_count = sum(1 for r in self.all_fit_results.values() if r.get('success'))
        total_count = len(self.curves)

        # 更新统计信息
        all_r_squared = [r.get('r_squared', 0.0) for r in self.all_fit_results.values() if r.get('success')]
        avg_r_squared = sum(all_r_squared) / len(all_r_squared) if all_r_squared else 0.0
        self._update_fit_stats(fitted_count, total_count, avg_r_squared)

        # 为所有成功拟合的 m/z 保存配置（使哈希匹配）
        for mz, result in results.items():
            if result.get('success') and result.get('model'):
                fit_species = result['model'].get('species', [])
                self.per_mz_config[mz] = {
                    "selected_species": fit_species,
                    "coefficients": {int(s.get("id", idx + 1)): float(s.get("coefficient", 0.0))
                                     for idx, s in enumerate(fit_species)},
                    "locked_ids": [],
                }
                new_hash = self._get_per_mz_config_hash(mz)
                self.all_fit_results[mz]['fit_config_hash'] = new_hash

        if fitted_count > 0:
            for mz, result in results.items():
                if result.get('success') and result.get('model'):
                    self.current_mz = mz
                    self.current_fit = result['model']

                    results_list = result['model'].get('species', [])
                    fit_df = pd.DataFrame(
                        [
                            {
                                "物种名称": item["species"],
                                "电离能(eV)": "" if item.get("ie") is None else round(float(item["ie"]), 4),
                                "匹配系数": round(float(item["coefficient"]), 6),
                                "贡献(%)": round(float(item["contribution_percent"]), 2),
                                "R²": round(float(item["r_squared"]), 5),
                            }
                            for item in results_list
                        ],
                        columns=["物种名称", "电离能(eV)", "匹配系数", "贡献(%)", "R²"],
                    )
                    self.set_dataframe(self.fit_table, fit_df)

                    if mz in self.curves:
                        self.update_plot(self.curves[mz], result['model'])
                    break

        self.refresh_current_plot()
        self.populate_mz_list()
        self._update_action_state()

        QtWidgets.QMessageBox.information(
            self, "完成",
            f"重新拟合完成！\n已拟合: {fitted_count} / {total_count} 条曲线"
        )

    def clear_all_fits(self):
        """清除所有拟合"""
        self.current_fit = None
        self.all_fit_results = {}
        # Phase 3 Step 3: Mark results dirty when clearing fits
        self.pie_state_dirty = True
        self.fit_table.setRowCount(0)
        self.result_display_widget.clear_data()
        self._update_fit_stats(0, len(self.curves), None)
        self.refresh_current_plot()
        self._update_action_state()

    def export_pie_results(self):
        """导出PIE物种鉴定结果到Excel"""
        from bl03u_masstool.core.export_pie_results import export_pie_results_to_excel

        # 检查是否有拟合结果
        if not hasattr(self, 'all_fit_results') or not self.all_fit_results:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "没有找到拟合结果，请先进行拟合操作。"
            )
            return

        # 生成默认文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"PIE_identification_results_{timestamp}.xlsx"

        # 打开文件保存对话框
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出PIE鉴定结果",
            default_filename,
            "Excel Files (*.xlsx);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # 调用导出函数
            result = export_pie_results_to_excel(self, file_path)

            if result['success']:
                record_project_artifact(
                    self,
                    "pie_identification_result_file",
                    result.get("file_path", file_path),
                    message="PIE鉴定结果已登记到项目管理",
                )
                QtWidgets.QMessageBox.information(
                    self,
                    "导出成功",
                    f"{result['message']}\n\n已登记到项目管理。"
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "导出失败",
                    result['message']
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"导出时发生错误：{str(e)}"
            )

    # ── Force species management (delegated to FittingControlWidget) ───────

    def _confirm_identification(self):
        """确认当前 m/z 的鉴定结果，标记为 CONFIRMED 状态"""
        if self.current_mz is None:
            return
        result = self.all_fit_results.get(self.current_mz)
        if not result or not result.get("success"):
            return

        # 标记为已确认
        result["_confirmed"] = True
        result["_confirmed_timestamp"] = time.time()

        # 更新右侧面板确认状态
        self.fitting_control_widget.set_fit_confirmed(True)
        # 同步中间详细数据面板状态
        self.result_display_widget.set_result_status("CONFIRMED")

        # 更新m/z列表显示（确认标记）
        self.populate_mz_list()
        self.pie_state_dirty = True

        species_list = result.get("model", {}).get("species", [])
        names = ", ".join(s.get("species", "?") for s in species_list[:3])
        r2 = result.get("r_squared", 0.0)
        self.status_label.setText(f"m/z {self.current_mz} 已确认: {names}  R²={r2:.4f}")

    # ── Force species management (delegated to FittingControlWidget) ───────

    def add_force_species(self, species_name: str):
        """向后兼容：锁定候选物种（推荐使用 lock_candidate）"""
        self.fitting_control_widget.lock_candidate(species_name)
        self._on_locked_species_changed()

    def get_force_species(self) -> list:
        """向后兼容：获取锁定候选物种列表（推荐使用 get_locked_species）"""
        return self.fitting_control_widget.get_locked_species()

    # ── Fit stats helper ─────────────────────────────────────────────────

    def _update_fit_stats(self, fitted_count: int, total_count: int, avg_r_squared: float | None = None):
        """Update fit stats label with colored R² indicator."""
        total_text = f"<b>{fitted_count}</b> / {total_count} 条曲线"
        if avg_r_squared is not None:
            if avg_r_squared >= 0.8:
                r2_color = "#16a34a"  # green
            elif avg_r_squared >= 0.5:
                r2_color = "#d97706"  # amber
            else:
                r2_color = "#dc2626"  # red
            r2_text = f"<span style='color:{r2_color};font-weight:bold;'>{avg_r_squared:.4f}</span>"
        else:
            r2_text = "N/A"
        self.fit_stats_label.setText(
            f"已拟合: {total_text}&nbsp;&nbsp;&nbsp;平均R\u00b2: {r2_text}"
        )

    def select_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择PIE扫描文件夹")
        if folder:
            self.folder_edit.setText(folder)

    def toggle_multi_folder_mode(self, checked: bool):
        """切换多文件夹模式"""
        busy = getattr(self, "_busy", False)
        self.folder_edit.setEnabled(not checked and not busy)
        self.select_folder_button.setEnabled(not checked and not busy)
        self.add_folder_button.setEnabled(checked and not busy)
        self.remove_folder_button.setEnabled(checked and not busy)
        self.clear_folders_button.setEnabled(checked and not busy)
        self.merge_method_combo.setEnabled(checked and not busy)
        self.multi_folder_section.setVisible(checked)

    def add_folder(self):
        """添加文件夹到列表"""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择PIE扫描文件夹")
        if folder:
            self.pie_folders.append(folder)
            self.folder_list.addItem(Path(folder).name + f" ({folder})")

    def remove_folder(self):
        """移除选中的文件夹"""
        current_row = self.folder_list.currentRow()
        if current_row >= 0:
            self.folder_list.takeItem(current_row)
            self.pie_folders.pop(current_row)

    def clear_folders(self):
        """清空文件夹列表"""
        self.folder_list.clear()
        self.pie_folders.clear()

    def set_project_settings(self, ps: ProjectSettings) -> None:
        """Apply ProjectSettings defaults to summary bar and folder controls.

        Also loads per-m/z configurations from the new project's state.
        Clears previous project's in-memory state to prevent data leakage.
        """
        # Phase 3 Step 2: Clear previous project's memory state before loading new one
        # This prevents m/z configs from project A appearing in project B
        self.per_mz_config = {}  # Clear configs
        self.all_fit_results = {}  # Clear fit results (Phase 3 Step 3)
        self.current_mz = None  # Clear current selection
        # Note: global_solver_config is app-wide, so don't clear it here
        # Just reset hash after project loads below

        self.project_settings = ps
        # Phase 3 Step 2: Store project directory for state persistence
        self.project_dir = str(project_root(ps))
        project_name = ps.project_name or "---"
        system = ps.system or "---"
        pie_path = ps.pie_scan_folder or self.folder_edit.text()
        self.summary_project_label.setText(f"项目: {project_name}")
        self.summary_system_label.setText(f"体系: {system}")
        self.summary_data_label.setText(f"PIE: {pie_path}")
        self.use_multi_folders.setChecked(ps.pie_multi_folder_mode)
        idx = self.merge_method_combo.findData(ps.pie_merge_method)
        if idx >= 0:
            self.merge_method_combo.setCurrentIndex(idx)
        if ps.pie_scan_folder:
            self.folder_edit.setText(ps.pie_scan_folder)
        # Phase 3 Step 2: Load per-m/z configurations from project state
        # (After clearing previous project's state)
        self._load_per_mz_configs()
        self._update_action_state()

    def _load_per_mz_configs(self) -> None:
        """Load per-m/z configurations from persistent state (Phase 3 Step 2).

        Loads configurations from .bl03u_pie_state/configs.json if it exists.
        Suppresses errors and only warns if loading fails - does not block project open.
        Recalculates config hashes after restoration to ensure consistency.
        """
        if not self.project_dir or not self.database or not self.curves:
            # Cannot load without project context or data
            return

        try:
            manager = PieStateManager(self.project_dir)
            result = manager.load_state(self.curves, self.database, self.calibration)

            if not result.get('success'):
                # Load failure only warns, doesn't block
                error_msg = result.get('error', 'Unknown error loading per-m/z configurations')
                print(f"[WARNING] PIE configurations: {error_msg}")
                return

            # Show any warnings (e.g., missing state file for first time)
            for warning in result.get('warnings', []):
                print(f"[INFO] {warning}")

            # Restore per-m/z configurations
            saved_configs = result.get('configs', {})
            for mz_str, config in saved_configs.items():
                try:
                    mz_int = int(mz_str)
                    if mz_int in self.curves:
                        self.per_mz_config[mz_int] = config
                except (ValueError, KeyError):
                    # Skip invalid m/z entries
                    pass

            # Phase 2: Recalculate config hashes after restoration to ensure consistency
            # This ensures that restored configs have up-to-date hash values
            for mz_int in self.per_mz_config:
                hash_val = self._get_current_per_mz_config_hash(mz_int)
                if hash_val:
                    self.per_mz_config[mz_int]['config_hash'] = hash_val

            self._update_global_config_hash()

        except Exception as e:
            # Graceful failure - only warn, don't block
            print(f"[WARNING] Failed to load per-m/z configurations: {e}")

    def save_per_mz_configs(self) -> None:
        """Save per-m/z configurations to persistent state (Phase 3 Step 2).

        Saves configurations to .bl03u_pie_state/configs.json.
        Called explicitly (not on every change) to avoid frequent disk writes.

        CRITICAL: Preserves existing results and arrays to avoid data loss.
        Loads current state, updates configs, and saves complete state atomically.

        Should be called when:
        - User clicks "Save Project"
        - Project is closing
        - On demand via UI action
        """
        if not self.project_dir or not self.database or not self.curves:
            # Cannot save without project context or data
            return

        try:
            manager = PieStateManager(self.project_dir)

            # Phase 3 Step 2 + Step 3 coordination:
            # Load current state to preserve results and arrays
            current_state = manager.load_state(self.curves, self.database, self.calibration)

            # Extract existing results (will be empty if this is first save)
            all_fit_results = {}
            if current_state.get('success'):
                for mz_str, result in current_state.get('results', {}).items():
                    try:
                        mz_int = int(mz_str)
                        all_fit_results[mz_int] = result
                    except (ValueError, KeyError):
                        pass  # Skip invalid m/z entries

            # Save complete state (configs + results + arrays)
            # This ensures atomic transaction - either all-or-nothing
            success, error = manager.save_state(
                self.curves,
                self.database,
                self.calibration,
                self.per_mz_config,
                all_fit_results,  # Preserve existing results (from Step 3)
                self.global_solver_config.get('config_hash', '')
            )

            if not success:
                print(f"[WARNING] Failed to save per-m/z configurations: {error}")
                # Keep dirty flag on failure
            else:
                print(f"✓ Per-m/z configurations saved")
                # Clear dirty flag on success
                self.pie_state_dirty = False

        except Exception as e:
            print(f"[WARNING] Error saving per-m/z configurations: {e}")
            # Keep dirty flag on exception

    def persist_pie_project_state(self) -> Tuple[bool, str | None]:
        """Public interface for PIE project state persistence.

        Phase 3 Step 3: Saves complete PIE project state atomically:
        - Configuration (per-m/z solver parameters)
        - Results (fitting metrics, species contributions, arrays)
        - Arrays (energies, fitted curves, residuals, components)
        - Manifest (fingerprints, metadata)

        Called by project save handlers after ProjectSettings.save().

        Returns:
            (success, error_message)
        """
        if not self.project_dir or not self.database or not self.curves:
            # No state to save (project not yet fully initialized)
            return (True, None)

        try:
            manager = PieStateManager(self.project_dir)

            # Load current state to preserve results and arrays
            current_state = manager.load_state(self.curves, self.database, self.calibration)

            # Extract existing results (will be empty if this is first save)
            all_fit_results = {}
            if current_state.get('success'):
                for mz_str, result in current_state.get('results', {}).items():
                    try:
                        mz_int = int(mz_str)
                        all_fit_results[mz_int] = result
                    except (ValueError, KeyError):
                        pass

            # Save complete state (configs + results + arrays)
            success, error = manager.save_state(
                self.curves,
                self.database,
                self.calibration,
                self.per_mz_config,
                all_fit_results,
                self.global_solver_config.get('config_hash', '')
            )

            if success:
                self.pie_state_dirty = False
            else:
                self.pie_state_dirty = True

            return (success, error)

        except Exception as e:
            self.pie_state_dirty = True
            return (False, str(e))

    def persist_per_mz_config_state(self) -> Tuple[bool, str | None]:
        """Backward compatibility alias for persist_pie_project_state()."""
        return self.persist_pie_project_state()

    def mark_pie_config_changed(self) -> None:
        """Mark PIE configuration as changed (dirty).

        Called when user modifies per-m/z configuration:
        - Candidate species selection
        - Locked coefficients
        - Forced species
        - Other per-m/z constraints
        """
        self.pie_state_dirty = True

    def open_common_parameters(self):
        dialog = CommonParametersDialog(self.normalization_settings, self.calibration, self)
        dialog.exec()
        self.calibration = load_calibration_config()

    def run_analysis(self):
        use_multi = self.use_multi_folders.isChecked()

        if use_multi:
            if not self.pie_folders:
                QtWidgets.QMessageBox.warning(self, "提示", "请先添加至少一个PIE扫描文件夹")
                return
            folders = self.pie_folders
            merge_method = self.merge_method_combo.currentData()
            message = f"正在合并{len(folders)}段PIE数据..."
        else:
            folder = self.folder_edit.text().strip()
            if not folder:
                QtWidgets.QMessageBox.warning(self, "提示", "请先选择PIE扫描文件夹")
                return
            folders = [folder]
            merge_method = None
            message = "正在生成PIE曲线..."

        ps = self.project_settings or ProjectSettings()
        energy_decimals = ps.pie_energy_decimals
        recursive = ps.pie_recursive
        prefer_gaussian = ps.pie_prefer_gaussian
        manual_peak_path = ps.manual_peak_file or None
        peak_config = load_peak_detection_config()
        threshold_end = peak_config.threshold_end
        min_intensity = peak_config.min_intensity
        settings = self.normalization_settings
        photon_mode = settings.pie_photon_mode
        photon_normalize = photon_mode != "off"
        photon_reference_mode = "none" if photon_mode == "off" else photon_mode
        mass_discrimination = settings.mass_discrimination
        light_source = settings.light_source
        self.set_busy(True, message)
        self.worker = WorkerThread(
            lambda: self.run_pie_analysis_sync(
                folders,
                merge_method=merge_method,
                recursive=recursive,
                energy_decimals=energy_decimals,
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
                manual_peak_path=manual_peak_path,
                photon_normalize=photon_normalize,
                photon_reference_mode=photon_reference_mode,
                mass_discrimination=mass_discrimination,
                light_source=light_source,
                vote_threshold=peak_config.vote_threshold,
                min_intensity_for_single_vote=peak_config.min_intensity_for_single_vote,
                mz_tolerance=peak_config.mz_tolerance,
            ),
            self,
        )
        self.worker.finished_with_result.connect(self.on_analysis_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def set_busy(self, busy: bool, message: str) -> None:
        self._busy = busy
        self.status_label.setText(message)
        self.analyze_button.setEnabled(not busy)
        self.common_params_button.setEnabled(not busy)
        self.use_multi_folders.setEnabled(not busy)
        self.summary_open_project_btn.setEnabled(not busy)
        self.toggle_multi_folder_mode(self.use_multi_folders.isChecked())
        self._update_action_state()

    def _update_action_state(self) -> None:
        busy = getattr(self, "_busy", False)
        has_curves = bool(self.curves)
        has_fit_records = bool(self.all_fit_results)
        has_successful_fits = any(result.get("success") for result in self.all_fit_results.values())
        self.export_button.setEnabled(has_curves and not busy)
        self.export_plot_button.setEnabled(has_curves and not busy)
        self.fit_button.setEnabled(has_curves and not busy)
        self.fit_all_button.setEnabled(has_curves and not busy)
        self.more_actions_btn.setEnabled(not busy)
        # 更新菜单项的启用状态
        for action in self.more_actions_menu.actions():
            if action.text() == "清除拟合":
                action.setEnabled(has_fit_records and not busy)
            elif action.text() in ("穷举优选", "批量重拟合（已保存配置）"):
                action.setEnabled(has_curves and not busy)

    def run_pie_analysis_sync(
        self,
        folders: list[str],
        *,
        merge_method: str | None = None,
        recursive: bool,
        energy_decimals: int,
        algorithm: str,
        threshold_end: float,
        min_intensity: float,
        detection_min_idx: int,
        nearby_peak_window: int,
        duplicate_window: int,
        weak_tail_early_window: int,
        weak_tail_late_window: int,
        weak_tail_ratio: float,
        gaussian_window_max: int,
        gaussian_boundary_scale: float,
        boundary_padding: int,
        prominence_ratio: float,
        smoothing_window: int,
        smoothing_poly_order: int,
        baseline_window: int,
        baseline_percentile: float,
        min_peak_width: int,
        max_peak_width: int,
        prefer_gaussian: bool,
        manual_peak_path: str | None = None,
        photon_normalize: bool = True,
        photon_reference_mode: str = "first",
        mass_discrimination: float = 1.0,
        light_source: str = "io",
        vote_threshold: float = 0.667,
        min_intensity_for_single_vote: float = 5.0,
        mz_tolerance: float = 0.2,
    ) -> tuple[pd.DataFrame, dict[int, dict]]:

        if merge_method is not None and len(folders) > 1:
            analysis_df = analyze_multiple_pie_folders(
                folders,
                calibration=self.calibration,
                recursive=recursive,
                energy_decimals=energy_decimals,
                algorithm=algorithm,
                threshold_end=threshold_end,
                min_intensity=min_intensity,
                detection_min_idx=detection_min_idx,
                nearby_peak_window=nearby_peak_window,
                duplicate_window=duplicate_window,
                weak_tail_early_window=weak_tail_early_window,
                weak_tail_late_window=weak_tail_late_window,
                weak_tail_ratio=weak_tail_ratio,
                gaussian_window_max=gaussian_window_max,
                gaussian_boundary_scale=gaussian_boundary_scale,
                boundary_padding=boundary_padding,
                prominence_ratio=prominence_ratio,
                smoothing_window=smoothing_window,
                smoothing_poly_order=smoothing_poly_order,
                baseline_window=baseline_window,
                baseline_percentile=baseline_percentile,
                min_peak_width=min_peak_width,
                max_peak_width=max_peak_width,
                prefer_gaussian=prefer_gaussian,
                manual_peak_path=manual_peak_path,
                photon_normalize=photon_normalize,
                photon_reference_mode=photon_reference_mode,
                mass_discrimination=mass_discrimination,
                light_source=light_source,
                merge_method=merge_method,
                vote_threshold=vote_threshold,
                min_intensity_for_single_vote=min_intensity_for_single_vote,
                mz_tolerance=mz_tolerance,
            )
        else:
            analysis_df = analyze_pie_folder(
                folders[0],
                calibration=self.calibration,
                recursive=recursive,
                energy_decimals=energy_decimals,
                algorithm=algorithm,
                threshold_end=threshold_end,
                min_intensity=min_intensity,
                detection_min_idx=detection_min_idx,
                nearby_peak_window=nearby_peak_window,
                duplicate_window=duplicate_window,
                weak_tail_early_window=weak_tail_early_window,
                weak_tail_late_window=weak_tail_late_window,
                weak_tail_ratio=weak_tail_ratio,
                gaussian_window_max=gaussian_window_max,
                gaussian_boundary_scale=gaussian_boundary_scale,
                boundary_padding=boundary_padding,
                prominence_ratio=prominence_ratio,
                smoothing_window=smoothing_window,
                smoothing_poly_order=smoothing_poly_order,
                baseline_window=baseline_window,
                baseline_percentile=baseline_percentile,
                min_peak_width=min_peak_width,
                max_peak_width=max_peak_width,
                prefer_gaussian=prefer_gaussian,
                manual_peak_path=manual_peak_path,
                photon_normalize=photon_normalize,
                photon_reference_mode=photon_reference_mode,
                mass_discrimination=mass_discrimination,
                light_source=light_source,
                vote_threshold=vote_threshold,
                min_intensity_for_single_vote=min_intensity_for_single_vote,
                mz_tolerance=mz_tolerance,
            )
        return analysis_df, build_pie_curves(analysis_df)

    def on_analysis_complete(self, result: object) -> None:
        self.analysis_df, self.curves = result
        self.current_fit = None
        self.all_fit_results = {}  # 重置拟合结果
        self.populate_mz_list()
        # Switch from empty state to plot display
        if hasattr(self, "_plot_stack"):
            self._plot_stack.setCurrentIndex(1)
        energy_count = self.analysis_df["energy"].nunique() if not self.analysis_df.empty else 0
        self.summary_label.setText(f"{len(self.curves)} 条m/z曲线 | {energy_count} 个能量点")
        self._update_fit_stats(0, 0, None)
        self._update_action_state()
        if self.curves:
            self.mz_list.setCurrentRow(0)
            # 若数据库已加载，自动拟合全部曲线
            if self.database:
                QtCore.QTimer.singleShot(100, self.fit_all_curves)
            else:
                QtWidgets.QMessageBox.information(self, "完成", f"生成 {len(self.curves)} 条PIE曲线")
        else:
            QtWidgets.QMessageBox.information(self, "完成", f"生成 {len(self.curves)} 条PIE曲线")

    def on_analysis_failed(self, message: str) -> None:
        self.status_label.setText(f"失败: {message}")
        self._update_action_state()
        QtWidgets.QMessageBox.critical(self, "错误", message)

    def populate_mz_list(self):
        current_mz = self.current_mz
        self.mz_list.clear()
        for mz in sorted(self.curves):
            curve = self.curves[mz]
            if not self._pie_curve_matches_filter(mz, curve):
                continue
            label = curve.get("species") or ""
            suffix = f" {label}" if label and label != "Unknown" else ""

            # Phase 2: 使用哈希推导结果状态
            result = self.all_fit_results.get(mz)
            current_per_mz_hash = self._get_per_mz_config_hash(mz)
            current_global_hash = self.global_solver_config.get("config_hash", "")
            fit_state_code = self._derive_result_status(result, current_per_mz_hash, current_global_hash)

            # 状态文字和颜色配置
            is_confirmed = result.get("_confirmed", False) if result else False
            if is_confirmed:
                r2 = result.get("r_squared", 0.0) if result else 0.0
                status_str = f"✓ R²={r2:.3f}"
                fg_color = QtGui.QColor("#166534")   # 深绿
                bg_color = QtGui.QColor("#bbf7d0")   # 更深绿背景（区分于普通已拟合）
            elif fit_state_code == "COMPLETED":
                r2 = result.get("r_squared", 0.0) if result else 0.0
                r2_str = f"{r2:.3f}"
                status_str = f"R²={r2_str}"
                if r2 >= 0.8:
                    fg_color = QtGui.QColor("#166534")   # 深绿
                    bg_color = QtGui.QColor("#dcfce7")   # 浅绿背景
                elif r2 >= 0.5:
                    fg_color = QtGui.QColor("#92400e")   # 深琥珀
                    bg_color = QtGui.QColor("#fef9c3")   # 浅黄背景
                else:
                    fg_color = QtGui.QColor("#991b1b")   # 深红
                    bg_color = QtGui.QColor("#fee2e2")   # 浅红背景
            elif fit_state_code == "OBSOLETE":
                r2 = result.get("r_squared", 0.0) if result else 0.0
                status_str = f"R²={r2:.3f} (过期)"
                fg_color = QtGui.QColor("#6b7280")       # 灰色
                bg_color = QtGui.QColor("#f3f4f6")       # 浅灰背景
            elif fit_state_code == "FAILED":
                status_str = "失败"
                fg_color = QtGui.QColor("#991b1b")
                bg_color = QtGui.QColor("#fee2e2")
            else:  # UNFITTED
                status_str = "待拟合"
                fg_color = QtGui.QColor("#374151")       # 深灰文字
                bg_color = None                          # 无背景色

            item_text = f"m/z {mz}{suffix}  [{status_str}]  ({len(curve['energies'])}点)"
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, mz)
            item.setForeground(QtGui.QBrush(fg_color))
            if bg_color:
                item.setBackground(QtGui.QBrush(bg_color))
            item.setToolTip(f"m/z {mz}{suffix} | {status_str} | {len(curve['energies'])} 个能量点")
            self.mz_list.addItem(item)
            if current_mz == mz:
                self.mz_list.setCurrentItem(item)
        if self.mz_list.count() == 0:
            self.on_mz_selected(None)

    def _pie_curve_matches_filter(self, mz: int, curve: dict) -> bool:
        query = self.mz_filter_edit.text().strip().lower() if hasattr(self, "mz_filter_edit") else ""
        if not query:
            return True

        # Phase 2: 使用哈希推导结果状态以匹配过滤
        result = self.all_fit_results.get(mz)
        current_per_mz_hash = self._get_per_mz_config_hash(mz)
        current_global_hash = self.global_solver_config.get("config_hash", "")
        fit_state_code = self._derive_result_status(result, current_per_mz_hash, current_global_hash)
        fit_state_text, _ = self._get_status_display(fit_state_code)

        haystack = " ".join(
            str(value)
            for value in (
                mz,
                curve.get("species", ""),
                fit_state_text,
            )
        ).lower()
        return query in haystack

    def on_mz_selected(self, current, previous=None):
        # 保存前一个 m/z 的候选物种配置（除非被标记为跳过）
        if self.current_mz is not None and not self._skip_save_config:
            self._save_current_mz_config()
        self._skip_save_config = False

        if current is None:
            self.current_mz = None
            self.current_fit = None
            for table in (self.curve_table, self.fit_table):
                table.clear()
                table.setRowCount(0)
                table.setColumnCount(0)
            self.fitting_control_widget.clear_ui()
            self.result_display_widget.clear_data()
            if self.plot_widget is not None:
                self.plot_widget.clear_plot(title="未选择 PIE 曲线")
            return
        self.current_mz = int(current.data(QtCore.Qt.ItemDataRole.UserRole))

        # 检查是否有保存的拟合结果
        if self.current_mz in self.all_fit_results:
            fit_result = self.all_fit_results[self.current_mz]

            # Phase 2: 计算结果状态
            current_per_mz_hash = self._get_per_mz_config_hash(self.current_mz)
            current_global_hash = self.global_solver_config.get("config_hash", "")
            fit_state_code = self._derive_result_status(fit_result, current_per_mz_hash, current_global_hash)
            fit_state_text, fit_state_emoji = self._get_status_display(fit_state_code)

            if fit_result.get('success') and fit_result.get('model'):
                self.current_fit = fit_result['model']

                # 更新右侧结果摘要区
                confirmed = fit_result.get('_confirmed', False)
                display_status = "CONFIRMED" if confirmed else fit_state_code
                self.fitting_control_widget.show_fit_result(fit_result['model'], display_status)
                if confirmed:
                    self.fitting_control_widget.set_fit_confirmed(True)

                # 更新中间详细数据面板
                self.result_display_widget.update_fit_table(fit_result['model'])
                self.result_display_widget.set_result_status(display_status)
                self._update_result_detail_button()

                # Phase 2: 在状态栏显示结果有效性
                if fit_state_code == "OBSOLETE":
                    status_msg = f"{fit_state_emoji} {fit_state_text} - 配置已修改，请重新拟合"
                else:
                    status_msg = f"{fit_state_emoji} {fit_state_text}"
                self.set_busy(False, status_msg)
            else:
                self.current_fit = None
                self.fitting_control_widget.clear_fit_result()
                self.result_display_widget.clear_data()
                self.result_display_widget.set_result_status("FAILED")
                self._update_result_detail_button()

                # Phase 2: 显示失败状态
                if not fit_result.get('success'):
                    status_msg = f"{fit_state_emoji} {fit_state_text}"
                    self.set_busy(False, status_msg)
        else:
            self.current_fit = None
            self.fitting_control_widget.clear_fit_result()
            self.result_display_widget.clear_data()
            self.result_display_widget.set_result_status("UNFITTED")
            self._update_result_detail_button()

        # 显示曲线数据
        curve = self.curves[self.current_mz]
        rows = curve["rows"].copy()
        energies = np.round(rows["energy"].astype(float), 4).to_numpy()
        point_columns = [f"{energy:g} eV" for energy in energies]
        curve_df = pd.DataFrame(
            [
                ["原始积分", *np.round(rows["raw_area"].astype(float), 4).to_numpy()],
                ["IO归一化", *np.round(rows["photon_normalized_intensity"].astype(float), 4).to_numpy()],
                ["最终强度", *np.round(rows["normalized_intensity"].astype(float), 4).to_numpy()],
            ],
            columns=["数据项", *point_columns],
        )
        self.result_display_widget.update_curve_data({
            "energies": energies,
            "experimental": rows["normalized_intensity"].astype(float).to_numpy(),
            "total_fit": self.current_fit.get("total_fit", []) if self.current_fit else [],
            "residual": self.current_fit.get("residual", []) if self.current_fit else [],
        })
        self.set_dataframe(self.curve_table, curve_df)

        # 填充候选物种面板
        self._populate_candidate_table(self.current_mz)

        # 恢复该 m/z 的候选物种配置
        self._restore_mz_config(self.current_mz)

        # 若有拟合结果，将系数同步到表格（_restore_mz_config 可能因 ID 不匹配而失败）
        if self.current_fit:
            self._update_species_table_coefficients(self.current_fit)

        # 更新图表
        self.update_plot(curve, self.current_fit)

    def _on_pie_plot_mouse_move(self, pos):
        """Retained for older signal wiring; Matplotlib trend plots are intentionally low-interaction."""
        return

    def refresh_current_plot(self):
        if self.current_mz is not None and self.current_mz in self.curves:
            self.update_plot(self.curves[self.current_mz], self.current_fit)

    def update_plot(self, curve: dict, fit_model: dict | None = None):
        if self.plot_widget is None:
            return
        x_values = np.asarray(curve["energies"], dtype=float)
        y_values = np.asarray(curve["intensities"], dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid]
        y_values = y_values[valid]

        # Simplified title, R² moved to indicator box
        r_squared_text = ""
        if fit_model is not None and fit_model.get("r_squared") is not None:
            r_squared_text = f"R² = {fit_model.get('r_squared', 0.0):.4f}"
        title = f"m/z {curve['mz']} PIE物种识别拟合"
        self.plot_widget.clear_plot(title=title, xlabel="光子能量 (eV)", ylabel="相对强度")
        if x_values.size == 0:
            self.plot_widget.show_empty("无有效数据", title=f"m/z {curve['mz']} PIE")
            return

        # Experimental data: blue scatter with thin connecting line
        exp_x, exp_y = self.plot_widget.plot_series(
            x_values,
            y_values,
            color="#2563eb",
            linewidth=1.2,
            marker="o",
            markersize=6.5,
            label="实验数据",
        )
        x_ranges = [exp_x]
        y_ranges = [exp_y]

        if fit_model is not None:
            fit_x = np.asarray(fit_model.get("energies", []), dtype=float)
            fit_y = np.asarray(fit_model.get("fitted", []), dtype=float)
            fit_valid = np.isfinite(fit_x) & np.isfinite(fit_y)
            fit_x = fit_x[fit_valid]
            fit_y = fit_y[fit_valid]
            if fit_x.size:
                x_ranges.append(fit_x)
                y_ranges.append(fit_y)
                # Total fit: thick solid line to highlight
                self.plot_widget.plot_series(
                    fit_x,
                    fit_y,
                    color="#f97316",
                    linewidth=2.8,
                    marker=None,
                    linestyle="-",
                    label=f"总拟合  R²={fit_model.get('r_squared', 0.0):.4f}" if r_squared_text else "总拟合",
                )
                species = fit_model.get("species", [])
                colors = ["#16a34a", "#9333ea", "#dc2626", "#0891b2", "#ca8a04", "#be123c", "#7c3aed", "#0369a1"]
                for idx, component in enumerate(species):
                    component_y = np.asarray(component.get("component_intensities", []), dtype=float)
                    component_count = min(fit_x.size, component_y.size)
                    if component_count == 0:
                        continue
                    component_x = fit_x[:component_count]
                    component_y = component_y[:component_count]
                    component_valid = np.isfinite(component_x) & np.isfinite(component_y)
                    component_x = component_x[component_valid]
                    component_y = component_y[component_valid]
                    if component_x.size == 0:
                        continue
                    x_ranges.append(component_x)
                    y_ranges.append(component_y)
                    # Component curves: thin dashed lines with low opacity
                    self.plot_widget.plot_series(
                        component_x,
                        component_y,
                        color=colors[idx % len(colors)],
                        linewidth=1.1,
                        marker=None,
                        linestyle="--",
                        alpha=0.58,
                        label=str(component.get("species", ""))[:24],
                    )

        # Apply data limits with 5% margin
        self.plot_widget.apply_data_limits(x_ranges, y_ranges, x_pad_min=0.2, y_pad_min=0.05)

        # Legend inside plot area, upper-left corner (avoids high-energy curve data)
        # R² is embedded in the "总拟合" legend label to avoid overlapping curves
        self.plot_widget.finish(legend=True, legend_loc="upper left")

    def fit_current_curve(self):
        # 防止并发拟合
        if self._busy:
            return
        if not self.database:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载PICS截面数据库")
            return

        # 检测多选：若选中多条，询问用户是否批量拟合
        selected_items = self.mz_list.selectedItems()
        if len(selected_items) > 1:
            mz_list = []
            for item in selected_items:
                mz = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if mz is not None:
                    try:
                        mz_list.append(int(mz))
                    except (TypeError, ValueError):
                        pass
            if mz_list:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "批量拟合",
                    f"检测到已选中 {len(mz_list)} 条曲线。\n"
                    f"将对每条曲线应用当前右侧面板配置进行批量拟合。\n\n"
                    f"是否继续？",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    self._fit_multiple_with_panel_state(mz_list)
                return

        # 单选：原有逻辑
        if self.current_mz is None or self.current_mz not in self.curves:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择一条m/z曲线")
            return
        self._fit_single_curve_with_panel_state(self.current_mz)

    def _fit_single_curve_with_panel_state(self, mz: int):
        """用当前右侧面板配置拟合单条曲线，更新图表和结果摘要。"""
        from bl03u_masstool.core.pie_analysis import fit_species_combination_with_curve

        if mz not in self.curves:
            return

        fit_per_mz_hash = self._get_current_per_mz_config_hash(mz)
        fit_global_hash = self.global_solver_config.get("config_hash", "")

        self.set_busy(True, f"拟合中...")
        try:
            curve = self.curves[mz]
            panel_state = self._get_candidate_panel_state()
            selected = panel_state["selected_species"]

            if not selected:
                filtered_db = self.get_filtered_database()
                selected = [item for item in filtered_db if item.get("mz") == mz]

            # 检查用户是否手动编辑了系数
            user_coefficients = panel_state["coefficients"]
            species_ids = [int(s.get("id", idx + 1)) for idx, s in enumerate(selected)]
            has_manual_coefficients = any(user_coefficients.get(sid, 0.0) > 0.0 for sid in species_ids)

            if has_manual_coefficients:
                # 用户手动编辑了系数，使用用户的值
                fit_model = fit_species_combination_with_curve(
                    selected, curve["energies"], curve["intensities"],
                    coefficient_mode="manual", coefficients=user_coefficients,
                )
            else:
                # 没有手动编辑，使用 NNLS 自动优化
                fit_model = fit_species_combination_with_curve(
                    selected, curve["energies"], curve["intensities"],
                )

            results = fit_model.get("species", [])
            self.current_fit = fit_model
            self.current_mz = mz

            self.all_fit_results[mz] = {
                "success": bool(results),
                "model": fit_model,
                "species": results[:3],
                "r_squared": fit_model.get("r_squared", 0.0),
                "fit_config_hash": fit_per_mz_hash,
                "global_config_hash": fit_global_hash,
                "fit_timestamp": time.time(),
            }
            self.pie_state_dirty = True

            fit_df = pd.DataFrame(
                [
                    {
                        "物种名称": item["species"],
                        "电离能(eV)": "" if item.get("ie") is None else round(float(item["ie"]), 4),
                        "匹配系数": round(float(item["coefficient"]), 6),
                        "贡献(%)": round(float(item["contribution_percent"]), 2),
                        "R²": round(float(item["r_squared"]), 5),
                    }
                    for item in results
                ],
                columns=["物种名称", "电离能(eV)", "匹配系数", "贡献(%)", "R²"],
            )
            self.set_dataframe(self.fit_table, fit_df)
            self.update_plot(curve, fit_model)

            # 将拟合结果的系数更新回右侧候选物种表格
            self._update_species_table_coefficients(fit_model)

            self.fitting_control_widget.show_fit_result(fit_model, "COMPLETED")
            self.result_display_widget.set_result_status("COMPLETED")
            self._update_result_detail_button()

            self.status_label.setText(
                f"m/z {mz}: PICS数据库候选 {fit_model.get('candidate_count', 0)} 个，"
                f"命中 {len(results)} 个，R²={fit_model.get('r_squared', 0.0):.4f}"
            )
            fitted_count = sum(1 for r in self.all_fit_results.values() if r.get("success"))
            total_count = len(self.curves)
            all_r_squared = [r.get("r_squared", 0.0) for r in self.all_fit_results.values() if r.get("success")]
            avg_r_squared = sum(all_r_squared) / len(all_r_squared) if all_r_squared else 0.0
            self._update_fit_stats(fitted_count, total_count, avg_r_squared if all_r_squared else None)
            self.populate_mz_list()
            self._update_action_state()
            if not results:
                QtWidgets.QMessageBox.information(self, "结果", "当前m/z没有匹配到可拟合的物种")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "错误", str(exc))
        finally:
            self.set_busy(False, "就绪")

    def _fit_multiple_with_panel_state(self, mz_list: list[int]):
        """用当前右侧面板配置批量拟合多条曲线（多选场景）。"""
        if self._busy:
            return
        self.set_busy(True, f"正在批量拟合 {len(mz_list)} 条曲线...")
        self.worker = WorkerThread(
            lambda: self._fit_multiple_sync(mz_list),
            self,
        )
        self.worker.finished_with_result.connect(self._on_multi_panel_fit_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def _fit_multiple_sync(self, mz_list: list[int]) -> dict:
        """工作线程：用当前面板配置批量拟合。"""
        from bl03u_masstool.core.pie_analysis import fit_species_combination_with_curve

        global_hash = self.global_solver_config.get("config_hash", "")
        panel_state = self._get_candidate_panel_state()

        results = {}
        for mz in mz_list:
            curve = self.curves.get(mz)
            if not curve:
                continue
            per_mz_hash = self._get_per_mz_config_hash(mz)
            selected = panel_state["selected_species"]
            if not selected:
                filtered_db = self.get_filtered_database()
                selected = [item for item in filtered_db if item.get("mz") == mz]

            try:
                # 检查用户是否手动编辑了系数
                user_coefficients = panel_state["coefficients"]
                species_ids = [int(s.get("id", idx + 1)) for idx, s in enumerate(selected)]
                has_manual_coefficients = any(user_coefficients.get(sid, 0.0) > 0.0 for sid in species_ids)

                if has_manual_coefficients:
                    # 用户手动编辑了系数，使用用户的值
                    fit_model = fit_species_combination_with_curve(
                        selected, curve["energies"], curve["intensities"],
                        coefficient_mode="manual", coefficients=user_coefficients,
                    )
                else:
                    # 没有手动编辑，使用 NNLS 自动优化
                    fit_model = fit_species_combination_with_curve(
                        selected, curve["energies"], curve["intensities"],
                    )

                fit_result_species = fit_model.get("species", [])
                results[mz] = {
                    "success": bool(fit_result_species),
                    "model": fit_model,
                    "species": fit_result_species[:3],
                    "r_squared": fit_model.get("r_squared", 0.0),
                    "fit_config_hash": per_mz_hash,
                    "global_config_hash": global_hash,
                    "fit_timestamp": time.time(),
                }
            except Exception:
                results[mz] = {"success": False, "error": "拟合失败"}
        return results

    def _on_multi_panel_fit_complete(self, results: dict):
        """批量面板拟合完成回调。"""
        for mz, result in results.items():
            if result.get("success"):
                self.all_fit_results[mz] = result
        self.pie_state_dirty = True

        fitted_count = sum(1 for r in self.all_fit_results.values() if r.get("success"))
        total_count = len(self.curves)
        all_r_squared = [r.get("r_squared", 0.0) for r in self.all_fit_results.values() if r.get("success")]
        avg_r_squared = sum(all_r_squared) / len(all_r_squared) if all_r_squared else 0.0
        self._update_fit_stats(fitted_count, total_count, avg_r_squared if all_r_squared else None)

        # 为所有成功拟合的 m/z 保存配置（使哈希匹配）
        for mz, result in results.items():
            if result.get("success") and result.get("model"):
                fit_species = result["model"].get("species", [])
                self.per_mz_config[mz] = {
                    "selected_species": fit_species,
                    "coefficients": {int(s.get("id", idx + 1)): float(s.get("coefficient", 0.0))
                                     for idx, s in enumerate(fit_species)},
                    "locked_ids": [],
                }

        # 显示第一条成功结果
        for mz, result in results.items():
            if result.get("success") and result.get("model"):
                self.current_mz = mz
                self.current_fit = result["model"]
                self._update_species_table_coefficients(result["model"])
                self.fitting_control_widget.show_fit_result(result["model"], "COMPLETED")
                self.result_display_widget.set_result_status("COMPLETED")
                self._update_result_detail_button()
                if mz in self.curves:
                    self.update_plot(self.curves[mz], result["model"])
                break

        self.populate_mz_list()
        self._update_action_state()
        new_count = sum(1 for r in results.values() if r.get("success"))
        QtWidgets.QMessageBox.information(
            self, "完成", f"批量拟合完成！\n本次拟合: {new_count} / {len(results)} 条曲线"
        )

    def _exhaustive_best_fit(self):
        """穷举候选物种所有组合，按R²排序，选出最优组合。"""
        from itertools import combinations

        from bl03u_masstool.core.pie_analysis import fit_species_combination_with_curve

        if self.current_mz is None or self.current_mz not in self.curves:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择一条m/z曲线")
            return
        curve = self.curves[self.current_mz]
        panel_state = self._get_candidate_panel_state()
        candidates = panel_state.get("selected_species", [])
        if not candidates:
            filtered = self.get_filtered_database()
            candidates = [item for item in filtered if item.get("mz") == self.current_mz]
        if len(candidates) < 2:
            QtWidgets.QMessageBox.information(self, "提示", "至少需要2个候选物种才能进行组合穷举")
            return

        n = len(candidates)
        max_combos = 2**n - 1
        if max_combos > 2047:
            reply = QtWidgets.QMessageBox.question(
                self, "确认",
                f"候选物种 {n} 个, 共 {max_combos} 种组合, 计算可能较慢。是否继续?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self.set_busy(True, f"穷举 {n} 个候选物种的 {max_combos} 种组合...")
        self.worker = WorkerThread(
            lambda: _run_exhaustive_fit(
                candidates, curve["energies"], curve["intensities"]
            ),
            self,
        )
        self.worker.finished_with_result.connect(self._on_exhaustive_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def _on_exhaustive_complete(self, ranked: list[dict]):
        if not ranked:
            QtWidgets.QMessageBox.information(self, "结果", "穷举拟合未产生有效结果")
            return
        best = ranked[0]
        self.current_fit = best["model"]
        self.all_fit_results[self.current_mz] = {
            "success": True,
            "model": best["model"],
            "species": best["model"].get("species", [])[:3],
            "r_squared": best["r_squared"],
        }
        # Phase 3 Step 3: Mark results dirty when exhaustive fitting completes
        self.pie_state_dirty = True
        curve = self.curves[self.current_mz]
        self.update_plot(curve, best["model"])
        # 更新拟合结果表: 排名 / 物种组合 / 物种数 / R² / RMSE
        fit_df = pd.DataFrame([
            {
                "排名": i + 1,
                "物种组合": ", ".join(item["species_names"]),
                "物种数": item["species_count"],
                "R²": round(item["r_squared"], 5),
                "RMSE": round(item["rmse"], 5),
            }
            for i, item in enumerate(ranked)
        ])
        self.set_dataframe(self.fit_table, fit_df)
        self.fit_table.itemSelectionChanged.connect(self._on_combination_selected)
        self._exhaustive_ranked = ranked
        names = ", ".join(best["species_names"])
        self.status_label.setText(
            f"m/z {self.current_mz}: 最优组合 [{names}] R²={best['r_squared']:.4f} "
            f"(共 {len(ranked)} 种组合)"
        )
        fitted_count = sum(1 for r in self.all_fit_results.values() if r.get("success"))
        all_r_squared = [r.get("r_squared", 0.0) for r in self.all_fit_results.values() if r.get("success")]
        avg_r_squared = sum(all_r_squared) / len(all_r_squared) if all_r_squared else None
        self._update_fit_stats(fitted_count, len(self.curves), avg_r_squared)
        self.populate_mz_list()
        self._update_action_state()

    def _on_combination_selected(self):
        row = self.fit_table.currentRow()
        if row < 0 or not hasattr(self, "_exhaustive_ranked"):
            return
        item = self._exhaustive_ranked[row]
        self.current_fit = item["model"]
        if self.current_mz is not None and self.current_mz in self.curves:
            self.update_plot(self.curves[self.current_mz], item["model"])

    def export_curve_data(self):
        if self.analysis_df.empty:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可导出的PIE曲线数据")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出PIE曲线数据",
            str(ensure_output_dir("exports", "pie") / "pie_curves.xlsx"),
            "Excel Files (*.xlsx);;CSV Files (*.csv)",
        )
        if not path:
            return
        if path.endswith(".xlsx"):
            self.analysis_df.to_excel(path, index=False)
        else:
            self.analysis_df.to_csv(path, index=False, encoding="utf-8-sig")

    def export_plot(self):
        if self.plot_widget is None or self.plot_widget.figure is None:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可导出的图表")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出PIE曲线图",
            str(ensure_output_dir("exports", "pie") / "pie_curve_plot.png"),
            "PNG Images (*.png);;PDF Files (*.pdf)",
        )
        if not path:
            return
        if self.plot_widget.save_plot(path):
            QtWidgets.QMessageBox.information(self, "成功", f"曲线图已导出：{path}")
        else:
            QtWidgets.QMessageBox.warning(self, "错误", "导出图表失败")
