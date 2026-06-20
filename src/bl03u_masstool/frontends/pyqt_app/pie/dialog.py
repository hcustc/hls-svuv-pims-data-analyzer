from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets

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
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread
from bl03u_masstool.frontends.pyqt_app.project_artifacts import record_project_artifact

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin, FlowLayout
from bl03u_masstool.frontends.pyqt_app.common.static_plot import StaticCurvePlot
from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersDialog

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
        self.fit_all_button.setToolTip("一键拟合所有PIE曲线")
        self.fit_all_button.clicked.connect(self.fit_all_curves)
        self.fit_all_button.setFixedHeight(28)
        self.fit_all_button.setEnabled(False)

        self.refit_selected_button = QtWidgets.QPushButton("重拟合选中")
        self.refit_selected_button.setObjectName("BrowseButton")
        self.refit_selected_button.setToolTip("重新拟合左侧选中的 m/z 曲线")
        self.refit_selected_button.clicked.connect(self.refit_selected_curves)
        self.refit_selected_button.setEnabled(False)

        self.clear_fits_button = QtWidgets.QPushButton("清除拟合")
        self.clear_fits_button.setObjectName("WarningButton")
        self.clear_fits_button.setToolTip("清除当前页面所有拟合结果")
        self.clear_fits_button.clicked.connect(self.clear_all_fits)
        self.clear_fits_button.setEnabled(False)

        self.export_pie_button = QtWidgets.QPushButton("导出鉴定结果")
        self.export_pie_button.setObjectName("ExportButton")
        self.export_pie_button.setToolTip("导出PIE物种鉴定结果")
        self.export_pie_button.clicked.connect(self.export_pie_results)
        self.export_pie_button.setEnabled(False)

        self._force_species: list[str] = []

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
        mz_hint = QtWidgets.QLabel("按住 Cmd/Ctrl 可多选，右键快捷菜单")
        mz_hint.setObjectName("HintLabel")
        left_layout.addWidget(mz_hint)
        self.fit_button = QtWidgets.QPushButton("拟合当前")
        self.fit_button.setObjectName("PrimaryToolbarButton")
        self.fit_button.setToolTip("拟合当前选中的m/z曲线")
        self.fit_button.clicked.connect(self.fit_current_curve)
        self.fit_button.setEnabled(False)
        left_layout.addWidget(self.fit_button)
        left_layout.addWidget(self.fit_all_button)
        refit_row = QtWidgets.QHBoxLayout()
        refit_row.setSpacing(6)
        refit_row.addWidget(self.refit_selected_button)
        refit_row.addWidget(self.clear_fits_button)
        left_layout.addLayout(refit_row)
        self.exhaustive_button = QtWidgets.QPushButton("穷举优选")
        self.exhaustive_button.setObjectName("WarningButton")
        self.exhaustive_button.setToolTip("穷举候选物种所有组合，按R²排序选出最优")
        self.exhaustive_button.clicked.connect(self._exhaustive_best_fit)
        self.exhaustive_button.setEnabled(False)
        left_layout.addWidget(self.exhaustive_button)
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
        empty_icon = QtWidgets.QLabel("📈")
        empty_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_icon.setStyleSheet("font-size: 48px;")
        empty_msg = QtWidgets.QLabel('尚未生成PIE曲线\n\n选择包含PIE数据的文件夹后，点击"生成曲线"')
        empty_msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_msg.setObjectName("ProjectHint")
        empty_msg.setWordWrap(True)
        empty_layout.addWidget(empty_icon)
        empty_layout.addWidget(empty_msg)
        self._plot_stack.addWidget(self._empty_state)

        self.plot_widget = StaticCurvePlot("Photon Energy (eV)", "Normalized Intensity", min_height=280)
        self._plot_stack.addWidget(self.plot_widget)
        self._plot_stack.setCurrentIndex(0)  # Show empty state initially

        plot_container_layout.addLayout(self._plot_stack, stretch=1)

        # 拟合统计条属于图表区域，不作为 splitter 的独立面板，避免挤占下方功能区。
        stats_bar = QtWidgets.QFrame()
        stats_bar.setObjectName("StatsBar")
        stats_bar.setFixedHeight(32)
        stats_bar_layout = QtWidgets.QHBoxLayout(stats_bar)
        stats_bar_layout.setContentsMargins(10, 2, 10, 2)
        stats_bar_layout.setSpacing(12)
        stats_title = QtWidgets.QLabel("拟合统计")
        stats_title.setObjectName("StatsTitle")
        self.fit_stats_label = QtWidgets.QLabel("已拟合: <b>0</b> / 0 条曲线&nbsp;&nbsp;&nbsp;平均R\u00b2: N/A")
        self.fit_stats_label.setObjectName("HintLabel")
        self.fit_stats_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        stats_bar_layout.addWidget(stats_title)
        stats_bar_layout.addWidget(self.fit_stats_label, stretch=1)
        plot_container_layout.addWidget(stats_bar)
        self.right_splitter.addWidget(plot_container)

        self.curve_table = QtWidgets.QTableWidget()
        self.fit_table = QtWidgets.QTableWidget()
        for table in (self.curve_table, self.fit_table):
            table.setWordWrap(False)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.detail_tabs = QtWidgets.QTabWidget()
        self.detail_tabs.setMinimumHeight(170)
        self.detail_tabs.addTab(self.curve_table, "曲线数据")
        self.detail_tabs.addTab(self.fit_table, "PICS拟合")

        # ---- 强制物种面板 ----
        force_panel = QtWidgets.QWidget()
        force_panel_layout = QtWidgets.QVBoxLayout(force_panel)
        force_panel_layout.setContentsMargins(0, 4, 0, 0)
        force_panel_layout.setSpacing(6)

        force_input_row = QtWidgets.QHBoxLayout()
        force_input_row.setSpacing(6)
        self.force_input = QtWidgets.QLineEdit()
        self.force_input.setPlaceholderText("输入物种名称后回车添加...")
        self.force_input.returnPressed.connect(self._add_force_from_input)
        force_input_row.addWidget(self.force_input, stretch=1)
        add_force_btn = QtWidgets.QPushButton("添加")
        add_force_btn.setObjectName("BrowseButton")
        add_force_btn.setFixedHeight(28)
        add_force_btn.clicked.connect(self._add_force_from_input)
        force_input_row.addWidget(add_force_btn)
        force_panel_layout.addLayout(force_input_row)

        tag_container = QtWidgets.QWidget()
        tag_container.setObjectName("TagCloud")
        self.force_tag_layout = FlowLayout(tag_container, margin=0, spacing=4)
        tag_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        force_panel_layout.addWidget(tag_container, stretch=1)
        self.detail_tabs.addTab(force_panel, "强制物种")

        # ---- 候选物种面板 ----
        candidate_panel = QtWidgets.QWidget()
        candidate_layout = QtWidgets.QVBoxLayout(candidate_panel)
        candidate_layout.setContentsMargins(0, 4, 0, 0)
        candidate_layout.setSpacing(4)

        self.candidate_table = QtWidgets.QTableWidget()
        self.candidate_table.setColumnCount(6)
        self.candidate_table.setHorizontalHeaderLabels(["选择", "物种", "m/z", "IE(eV)", "系数", "锁定"])
        self.candidate_table.setWordWrap(False)
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)

        # 候选控制栏
        candidate_controls = QtWidgets.QHBoxLayout()
        candidate_controls.setSpacing(6)
        candidate_controls.addWidget(QtWidgets.QLabel("系数模式:"))
        self.coefficient_mode_combo = QtWidgets.QComboBox()
        self.coefficient_mode_combo.addItem("自动拟合", "auto")
        self.coefficient_mode_combo.addItem("锁定已选", "locked_fit")
        self.coefficient_mode_combo.addItem("手动系数", "manual")
        self.coefficient_mode_combo.currentIndexChanged.connect(self._on_coefficient_mode_changed)
        candidate_controls.addWidget(self.coefficient_mode_combo)
        self.candidate_select_all_btn = QtWidgets.QPushButton("全选")
        self.candidate_select_all_btn.clicked.connect(lambda: self._set_all_candidates_checked(True))
        self.candidate_clear_btn = QtWidgets.QPushButton("清空")
        self.candidate_clear_btn.clicked.connect(lambda: self._set_all_candidates_checked(False))
        self.candidate_import_coeff_btn = QtWidgets.QPushButton("导入系数")
        self.candidate_import_coeff_btn.setToolTip("从当前拟合结果导入候选物种系数")
        self.candidate_import_coeff_btn.clicked.connect(self._import_coefficients_from_fit)
        self.candidate_zero_coeff_btn = QtWidgets.QPushButton("清零")
        self.candidate_zero_coeff_btn.setToolTip("将所有候选物种系数清零")
        self.candidate_zero_coeff_btn.clicked.connect(self._zero_all_coefficients)
        self.candidate_apply_btn = QtWidgets.QPushButton("应用")
        self.candidate_apply_btn.setObjectName("PrimaryToolbarButton")
        self.candidate_apply_btn.setToolTip("应用当前候选物种设置并拟合")
        self.candidate_apply_btn.clicked.connect(self.fit_current_curve)
        for btn in (self.candidate_select_all_btn, self.candidate_clear_btn,
                     self.candidate_import_coeff_btn, self.candidate_zero_coeff_btn):
            btn.setObjectName("BrowseButton")
        candidate_controls.addWidget(self.candidate_select_all_btn)
        candidate_controls.addWidget(self.candidate_clear_btn)
        candidate_controls.addWidget(self.candidate_import_coeff_btn)
        candidate_controls.addWidget(self.candidate_zero_coeff_btn)
        candidate_controls.addWidget(self.candidate_apply_btn)
        candidate_controls.addStretch()
        candidate_layout.addLayout(candidate_controls)
        candidate_layout.addWidget(self.candidate_table, stretch=1)

        self.detail_tabs.addTab(candidate_panel, "候选物种")

        self._candidate_data: list[dict] = []
        self._candidate_updating = False

        self.detail_container = QtWidgets.QWidget()
        self.detail_container.setObjectName("ResultPanel")
        detail_container_layout = QtWidgets.QVBoxLayout(self.detail_container)
        detail_container_layout.setContentsMargins(0, 0, 0, 0)
        detail_container_layout.setSpacing(4)
        detail_header = QtWidgets.QHBoxLayout()
        detail_header.setContentsMargins(0, 0, 0, 0)
        detail_header.setSpacing(6)
        detail_title = QtWidgets.QLabel("结果与拟合控制")
        detail_title.setObjectName("ReadoutLabel")
        self.export_pie_button.setFixedHeight(26)
        detail_header.addWidget(detail_title)
        detail_header.addStretch()

        self.pie_toggle_table_button = QtWidgets.QPushButton("展开表格")
        self.pie_toggle_table_button.setCheckable(True)
        self.pie_toggle_table_button.setToolTip("点击显示/隐藏结果与拟合控制表格")
        self.pie_toggle_table_button.setFixedWidth(86)
        self.pie_toggle_table_button.setFixedHeight(26)
        self.pie_toggle_table_button.setObjectName("BrowseButton")
        self.pie_toggle_table_button.clicked.connect(self._toggle_pie_table_visibility)
        detail_header.addWidget(self.pie_toggle_table_button)

        detail_header.addWidget(self.export_pie_button)
        detail_container_layout.addLayout(detail_header)
        detail_container_layout.addWidget(self.detail_tabs, stretch=1)

        self.right_splitter.addWidget(self.detail_container)
        self.right_splitter.setStretchFactor(0, 3)
        self.right_splitter.setStretchFactor(1, 1)
        self.right_splitter.setCollapsible(0, False)
        self.right_splitter.setCollapsible(1, False)
        self.pie_table_visible = False
        self._apply_detail_panel_state(expanded=False)
        QtCore.QTimer.singleShot(0, lambda: self._resize_detail_panel(expanded=False))

        splitter.addWidget(self.right_splitter)
        splitter.setSizes([260, 1020])
        layout.addWidget(splitter, stretch=1)
        self._update_action_state()

    def _open_project_settings(self):
        """跳转到项目管理页面。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("project")

    def _toggle_pie_table_visibility(self, checked: bool | None = None) -> None:
        """Toggle the visibility of the detail tabs and adjust splitter."""
        expanded = (not self.pie_table_visible) if checked is None else bool(checked)
        self._apply_detail_panel_state(expanded=expanded)

    def _apply_detail_panel_state(self, *, expanded: bool) -> None:
        self.pie_table_visible = expanded
        self.pie_toggle_table_button.setChecked(expanded)
        self.detail_tabs.setVisible(self.pie_table_visible)
        self.pie_toggle_table_button.setText("收起表格" if expanded else "展开表格")
        if expanded:
            self.detail_container.setMinimumHeight(220)
            self.detail_container.setMaximumHeight(16777215)
            self.detail_container.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        else:
            self.detail_container.setMinimumHeight(34)
            self.detail_container.setMaximumHeight(34)
            self.detail_container.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        self._resize_detail_panel(expanded=expanded)

    def _resize_detail_panel(self, *, expanded: bool) -> None:
        if expanded:
            total_height = max(1, self.right_splitter.height())
            detail_height = min(max(240, int(total_height * 0.34)), 360)
            self.right_splitter.setSizes([max(360, total_height - detail_height), detail_height])
        else:
            self.right_splitter.setSizes([10000, 34])

    def load_database(self, show_message: bool = True):
        path = self.database_edit.text().strip() if hasattr(self, "database_edit") else ""
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择物种数据库",
                "",
                "SQLite Files (*.sqlite *.sqlite3 *.db);;All Files (*)",
            )
            if not path:
                return
            if hasattr(self, "database_edit"):
                self.database_edit.setText(path)
        try:
            self.database, _ = load_species_database(path)
            self.status_label.setText(f"已加载PICS库: {len(self.database)} 个物种")
            if show_message:
                QtWidgets.QMessageBox.information(self, "完成", f"已加载 {len(self.database)} 个物种")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "错误", str(exc))

    def _auto_load_database(self):
        """自动加载内置PICS数据库"""
        try:
            db_path = species_database_path()
            if db_path.exists():
                self.database, _ = load_species_database(str(db_path))
        except Exception:
            pass

    # ---- 候选物种面板方法 ----

    def _populate_candidate_table(self, mz: int):
        """根据选中的m/z填充候选物种表格"""
        self._candidate_updating = True
        try:
            filtered_db = self.get_filtered_database()
            candidates = [item for item in filtered_db if item.get("mz") == mz]
            self._candidate_data = candidates
            self.candidate_table.setRowCount(len(candidates))
            for row, species in enumerate(candidates):
                # 选择 checkbox
                check_widget = QtWidgets.QWidget()
                check_layout = QtWidgets.QHBoxLayout(check_widget)
                check_layout.setContentsMargins(0, 0, 0, 0)
                check_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                chk = QtWidgets.QCheckBox()
                chk.setChecked(True)
                chk.stateChanged.connect(lambda state, r=row: self._on_candidate_changed(r))
                check_layout.addWidget(chk)
                self.candidate_table.setCellWidget(row, 0, check_widget)

                # 物种名称
                name_item = QtWidgets.QTableWidgetItem(str(species.get("species", "")))
                name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.candidate_table.setItem(row, 1, name_item)

                # m/z
                mz_item = QtWidgets.QTableWidgetItem(str(species.get("mz", "")))
                mz_item.setFlags(mz_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.candidate_table.setItem(row, 2, mz_item)

                # IE
                ie = species.get("ionization_energy")
                ie_text = f"{ie:.4f}" if ie is not None else ""
                ie_item = QtWidgets.QTableWidgetItem(ie_text)
                ie_item.setFlags(ie_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.candidate_table.setItem(row, 3, ie_item)

                # 系数 spinbox
                coeff_spin = QtWidgets.QDoubleSpinBox()
                coeff_spin.setRange(0, 1e6)
                coeff_spin.setDecimals(6)
                coeff_spin.setValue(0.0)
                coeff_spin.setEnabled(False)
                coeff_spin.valueChanged.connect(lambda val, r=row: self._on_candidate_changed(r))
                self.candidate_table.setCellWidget(row, 4, coeff_spin)

                # 锁定 checkbox
                lock_widget = QtWidgets.QWidget()
                lock_layout = QtWidgets.QHBoxLayout(lock_widget)
                lock_layout.setContentsMargins(0, 0, 0, 0)
                lock_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                lock_chk = QtWidgets.QCheckBox()
                lock_chk.setEnabled(False)
                lock_chk.stateChanged.connect(lambda state, r=row: self._on_candidate_changed(r))
                lock_layout.addWidget(lock_chk)
                self.candidate_table.setCellWidget(row, 5, lock_widget)

            self.candidate_table.resizeColumnsToContents()
            self.candidate_table.horizontalHeader().setStretchLastSection(True)
        finally:
            self._candidate_updating = False

    def _on_candidate_changed(self, row: int):
        """候选表格变化时实时刷新拟合预览"""
        if self._candidate_updating:
            return
        self._rebuild_manual_fit()

    def _on_coefficient_mode_changed(self, index: int):
        """系数模式切换"""
        mode = self.coefficient_mode_combo.currentData()
        for row in range(self.candidate_table.rowCount()):
            coeff_spin = self.candidate_table.cellWidget(row, 4)
            if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                coeff_spin.setEnabled(mode != "auto")
            lock_widget = self.candidate_table.cellWidget(row, 5)
            if lock_widget:
                lock_chk = lock_widget.findChild(QtWidgets.QCheckBox)
                if lock_chk:
                    lock_chk.setEnabled(mode == "locked_fit")
                    if mode != "locked_fit":
                        lock_chk.setChecked(False)
        self._rebuild_manual_fit()

    def _set_all_candidates_checked(self, checked: bool):
        """全选或清空候选物种"""
        self._candidate_updating = True
        try:
            for row in range(self.candidate_table.rowCount()):
                check_widget = self.candidate_table.cellWidget(row, 0)
                if check_widget:
                    chk = check_widget.findChild(QtWidgets.QCheckBox)
                    if chk:
                        chk.setChecked(checked)
        finally:
            self._candidate_updating = False
        self._rebuild_manual_fit()

    def _import_coefficients_from_fit(self):
        """从当前拟合结果导入系数到候选面板"""
        if self.current_fit is None:
            return
        species_list = self.current_fit.get("species", [])
        if not species_list:
            return
        self._candidate_updating = True
        try:
            coeff_by_name = {sp.get("species"): float(sp.get("coefficient", 0)) for sp in species_list}
            for row in range(self.candidate_table.rowCount()):
                name_item = self.candidate_table.item(row, 1)
                if name_item and name_item.text() in coeff_by_name:
                    coeff_spin = self.candidate_table.cellWidget(row, 4)
                    if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                        coeff_spin.setValue(coeff_by_name[name_item.text()])
        finally:
            self._candidate_updating = False
        self._rebuild_manual_fit()

    def _zero_all_coefficients(self):
        """将所有候选物种系数清零"""
        self._candidate_updating = True
        try:
            for row in range(self.candidate_table.rowCount()):
                coeff_spin = self.candidate_table.cellWidget(row, 4)
                if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                    coeff_spin.setValue(0.0)
        finally:
            self._candidate_updating = False
        self._rebuild_manual_fit()

    def _get_candidate_panel_state(self) -> dict:
        """获取候选面板当前状态"""
        mode = self.coefficient_mode_combo.currentData()
        selected_species = []
        locked_ids = []
        coefficients = {}

        for row in range(self.candidate_table.rowCount()):
            check_widget = self.candidate_table.cellWidget(row, 0)
            chk = check_widget.findChild(QtWidgets.QCheckBox) if check_widget else None
            if chk and chk.isChecked() and row < len(self._candidate_data):
                species = self._candidate_data[row]
                species_id = int(species.get("id", row + 1))
                selected_species.append(species)

                coeff_spin = self.candidate_table.cellWidget(row, 4)
                if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                    coefficients[species_id] = coeff_spin.value()

                lock_widget = self.candidate_table.cellWidget(row, 5)
                if lock_widget:
                    lock_chk = lock_widget.findChild(QtWidgets.QCheckBox)
                    if lock_chk and lock_chk.isChecked():
                        locked_ids.append(species_id)

        return {
            "mode": mode,
            "selected_species": selected_species,
            "coefficients": coefficients,
            "locked_ids": locked_ids,
        }

    def _rebuild_manual_fit(self):
        """使用候选面板状态重建拟合曲线(NNLS客户端预览)"""
        if self.current_mz is None or self.current_mz not in self.curves:
            return
        if self._candidate_updating:
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

        mode = panel_state["mode"]
        species_ids = [int(s.get("id", idx + 1)) for idx, s in enumerate(selected)]
        if mode == "manual":
            # 使用用户设置的系数，不做NNLS拟合
            coeffs = np.array([
                panel_state["coefficients"].get(species_id, 0.0)
                for species_id in species_ids
            ], dtype=float)
        elif mode == "locked_fit":
            try:
                from scipy.optimize import nnls
            except Exception:
                return
            coeffs = np.zeros(len(selected), dtype=float)
            locked_ids = set(panel_state["locked_ids"])
            locked_indices = [idx for idx, species_id in enumerate(species_ids) if species_id in locked_ids]
            free_indices = [idx for idx, species_id in enumerate(species_ids) if species_id not in locked_ids]
            for idx in locked_indices:
                coeffs[idx] = panel_state["coefficients"].get(species_ids[idx], 0.0)
            residual_target = intensities - design[:, locked_indices] @ coeffs[locked_indices] if locked_indices else intensities
            if free_indices:
                coeffs[free_indices], _ = nnls(design[:, free_indices], residual_target)
        else:
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
            "species": species_results,
            "r_squared": r_squared,
            "candidate_count": len(selected),
            "coefficient_mode": panel_state["mode"],
        }

        self.current_fit = manual_model
        curve = self.curves[self.current_mz]
        self.update_plot(curve, manual_model)

    def get_filtered_database(self) -> list:
        """获取根据全局元素设置筛选后的数据库"""
        selected_elements = set(self.normalization_settings.selected_elements)
        if not selected_elements:
            return self.database
        return filter_species_by_elements(self.database, selected_elements)

    def fit_all_curves(self):
        """一键拟合所有曲线"""
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
        force_species = self.get_force_species()
        results = {}
        for mz in mz_list:
            curve = self.curves.get(mz)
            if not curve:
                continue
            fit_result = self._fit_curve(mz, curve, filtered_db, force_species)
            results[mz] = fit_result
        return results

    def _fit_curve(self, mz: int, curve: dict, database: list, force_species: list) -> dict:
        """拟合单条曲线"""
        energies = np.array(curve.get('energies', []))
        intensities = np.array(curve.get('intensities', []))

        if len(energies) == 0 or len(intensities) == 0:
            return {'success': False, 'error': '无数据'}

        fit_model = identify_species_for_mz_with_curve(
            database, mz, energies, intensities, forced_species=force_species
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
        total_count = len(results)

        all_r_squared = [r.get('r_squared', 0.0) for r in results.values() if r.get('success')]
        avg_r_squared = sum(all_r_squared) / len(all_r_squared) if all_r_squared else 0.0

        self._update_fit_stats(fitted_count, total_count, avg_r_squared)

        if fitted_count > 0:
            first_mz, first_result = next(
                (mz, result) for mz, result in results.items() if result.get('success') and result.get('model')
            )
            self.current_mz = first_mz

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
                self.detail_tabs.setCurrentWidget(self.fit_table)

                if first_mz in self.curves:
                    self.update_plot(self.curves[first_mz], first_result['model'])

            self.refresh_current_plot()
            self.mz_list.setCurrentRow(0)

        self.populate_mz_list()
        self._update_action_state()

        QtWidgets.QMessageBox.information(
            self, "完成",
            f"拟合完成！\n已拟合: {fitted_count} / {total_count} 条曲线"
        )

    def _on_mz_list_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        fit_action = menu.addAction("拟合当前曲线")
        fit_all_action = menu.addAction("拟合全部曲线")
        action = menu.exec(self.mz_list.mapToGlobal(pos))
        if action == fit_action:
            self.fit_current_curve()
        elif action == fit_all_action:
            self.fit_all_curves()

    def refit_selected_curves(self):
        """重新拟合选中的曲线"""
        selected_items = self.mz_list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择要重新拟合的质量数")
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

        self.set_busy(True, f"正在重新拟合 {len(mz_list)} 条曲线...")
        self.worker = WorkerThread(
            lambda: self.fit_curves_sync(mz_list),
            self,
        )
        self.worker.finished_with_result.connect(self.on_refit_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def on_refit_complete(self, results: dict):
        """重新拟合完成"""
        # 更新保存的拟合结果
        for mz, result in results.items():
            if result.get('success'):
                self.all_fit_results[mz] = result

        fitted_count = sum(1 for r in self.all_fit_results.values() if r.get('success'))
        total_count = len(self.curves)

        # 更新统计信息
        all_r_squared = [r.get('r_squared', 0.0) for r in self.all_fit_results.values() if r.get('success')]
        avg_r_squared = sum(all_r_squared) / len(all_r_squared) if all_r_squared else 0.0
        self._update_fit_stats(fitted_count, total_count, avg_r_squared)

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
                    self.detail_tabs.setCurrentWidget(self.fit_table)

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
        self.fit_table.setRowCount(0)
        self._update_fit_stats(0, 0, None)
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

    # ── Force species tag cloud ──────────────────────────────────────────

    def _add_force_from_input(self):
        """Add species from the input field to the force-fit tag cloud."""
        name = self.force_input.text().strip()
        if name and name not in self.get_force_species():
            self._force_species = list(self.get_force_species()) + [name]
            self._rebuild_force_tags()
        self.force_input.clear()

    def add_force_species(self, species_name: str):
        """Add a force-fit species (called externally, e.g. from m/z list)."""
        current = list(self.get_force_species())
        if species_name not in current:
            current.append(species_name)
        self._force_species = current
        self._rebuild_force_tags()

    def _remove_force_species_name(self, name: str):
        """Remove a specific species from the force-fit list by name."""
        current = list(self.get_force_species())
        if name in current:
            current.remove(name)
        self._force_species = current
        self._rebuild_force_tags()

    def remove_force_species(self):
        """Remove selected force species (kept for backward compat)."""
        current = list(self.get_force_species())
        if current:
            self._force_species = current[:-1]
            self._rebuild_force_tags()

    def get_force_species(self) -> list:
        """Return current force-fit species list."""
        return list(getattr(self, '_force_species', []))

    def _rebuild_force_tags(self):
        """Rebuild the tag chips in the flow layout from _force_species."""
        # Clear existing tag widgets
        while self.force_tag_layout.count():
            item = self.force_tag_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for name in self.get_force_species():
            tag = QtWidgets.QFrame()
            tag.setObjectName("ForceTag")
            tag.setFixedHeight(24)
            tag_layout = QtWidgets.QHBoxLayout(tag)
            tag_layout.setContentsMargins(6, 1, 2, 1)
            tag_layout.setSpacing(2)

            label = QtWidgets.QLabel(name)
            label.setObjectName("ForceTagText")
            tag_layout.addWidget(label)

            close_btn = QtWidgets.QPushButton("\u2715")
            close_btn.setObjectName("TagCloseButton")
            close_btn.setFixedSize(16, 16)
            close_btn.clicked.connect(lambda checked=False, n=name: self._remove_force_species_name(n))
            tag_layout.addWidget(close_btn)

            self.force_tag_layout.addWidget(tag)

        # Force re-layout of the tag container
        container = self.force_tag_layout.parent()
        if container is not None:
            container.updateGeometry()

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
        """Apply ProjectSettings defaults to summary bar and folder controls."""
        self.project_settings = ps
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
        self._update_action_state()

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
        self.force_input.setEnabled(not busy)
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
        self.refit_selected_button.setEnabled(has_curves and not busy)
        self.exhaustive_button.setEnabled(has_curves and not busy)
        self.clear_fits_button.setEnabled(has_fit_records and not busy)
        self.export_pie_button.setEnabled(has_successful_fits and not busy)
        if hasattr(self, "candidate_apply_btn"):
            self.candidate_apply_btn.setEnabled(has_curves and not busy)

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
            fit_state = "已拟合" if mz in self.all_fit_results and self.all_fit_results[mz].get("success") else "待拟合"
            item = QtWidgets.QListWidgetItem(f"{mz}{suffix}  ({len(curve['energies'])}点)  {fit_state}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, mz)
            item.setToolTip(f"m/z {mz} | {fit_state}")
            self.mz_list.addItem(item)
            if current_mz == mz:
                self.mz_list.setCurrentItem(item)
        if self.mz_list.count() == 0:
            self.on_mz_selected(None)

    def _pie_curve_matches_filter(self, mz: int, curve: dict) -> bool:
        query = self.mz_filter_edit.text().strip().lower() if hasattr(self, "mz_filter_edit") else ""
        if not query:
            return True
        haystack = " ".join(
            str(value)
            for value in (
                mz,
                curve.get("species", ""),
                "已拟合" if mz in self.all_fit_results else "待拟合",
            )
        ).lower()
        return query in haystack

    def on_mz_selected(self, current, previous=None):
        if current is None:
            self.current_mz = None
            self.current_fit = None
            for table in (self.curve_table, self.fit_table):
                table.clear()
                table.setRowCount(0)
                table.setColumnCount(0)
            self.candidate_table.clearContents()
            self.candidate_table.setRowCount(0)
            if self.plot_widget is not None:
                self.plot_widget.clear_plot(title="未选择 PIE 曲线")
            return
        self.current_mz = int(current.data(QtCore.Qt.ItemDataRole.UserRole))

        # 检查是否有保存的拟合结果
        if self.current_mz in self.all_fit_results:
            fit_result = self.all_fit_results[self.current_mz]
            if fit_result.get('success') and fit_result.get('model'):
                self.current_fit = fit_result['model']

                # 显示拟合结果表格
                results_list = fit_result['model'].get('species', [])
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
                self.detail_tabs.setCurrentWidget(self.fit_table)
            else:
                self.current_fit = None
                self.fit_table.clear()
                self.fit_table.setRowCount(0)
                self.fit_table.setColumnCount(0)
        else:
            self.current_fit = None
            self.fit_table.clear()
            self.fit_table.setRowCount(0)
            self.fit_table.setColumnCount(0)

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
        self.set_dataframe(self.curve_table, curve_df)

        # 填充候选物种面板
        self._populate_candidate_table(self.current_mz)

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
        title = f"m/z {curve['mz']} PIE–PICS 拟合"
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
                    label="总拟合",
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

        # Add R² indicator box in top-right corner if available
        if self.plot_widget.axes is not None and r_squared_text:
            self.plot_widget.axes.text(
                0.98, 0.97, r_squared_text,
                transform=self.plot_widget.axes.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.85),
            )

        # Legend outside plot area, ordered: experimental data → total fit → components
        self.plot_widget.finish(legend=True, legend_loc="center left", legend_bbox_to_anchor=(1.02, 0.5))

    def fit_current_curve(self):
        if not self.database:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载物种数据库")
            return
        if self.current_mz is None or self.current_mz not in self.curves:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择一条m/z曲线")
            return
        try:
            from bl03u_masstool.core.pie_analysis import fit_species_combination_with_curve

            curve = self.curves[self.current_mz]
            panel_state = self._get_candidate_panel_state()
            mode = panel_state["mode"]
            selected = panel_state["selected_species"]

            if not selected:
                # 未选择任何候选时自动使用全部匹配物种
                filtered_db = self.get_filtered_database()
                selected = [item for item in filtered_db if item.get("mz") == self.current_mz]
                mode = "fit"

            if mode == "auto" or mode == "fit":
                fit_model = fit_species_combination_with_curve(
                    selected,
                    curve["energies"],
                    curve["intensities"],
                    coefficient_mode="fit",
                )
            elif mode == "manual":
                fit_model = fit_species_combination_with_curve(
                    selected,
                    curve["energies"],
                    curve["intensities"],
                    coefficient_mode="manual",
                    coefficients=panel_state["coefficients"],
                )
            elif mode == "locked_fit":
                fit_model = fit_species_combination_with_curve(
                    selected,
                    curve["energies"],
                    curve["intensities"],
                    coefficient_mode="locked_fit",
                    coefficients=panel_state["coefficients"],
                    locked_species_ids=panel_state["locked_ids"],
                )
            else:
                fit_model = fit_species_combination_with_curve(
                    selected, curve["energies"], curve["intensities"]
                )

            results = fit_model.get("species", [])
            self.current_fit = fit_model
            self.all_fit_results[self.current_mz] = {
                "success": bool(results),
                "model": fit_model,
                "species": results[:3],
                "r_squared": fit_model.get("r_squared", 0.0),
            }
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
            self.detail_tabs.setCurrentWidget(self.fit_table)
            self.update_plot(curve, fit_model)
            self.status_label.setText(
                f"m/z {self.current_mz}: PICS候选 {fit_model.get('candidate_count', 0)} 个，"
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
        curve = self.curves[self.current_mz]
        self.update_plot(curve, best["model"])
        # 更新PICS拟合表: 排名 / 物种组合 / 物种数 / R² / RMSE
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
        self.detail_tabs.setCurrentWidget(self.fit_table)
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
