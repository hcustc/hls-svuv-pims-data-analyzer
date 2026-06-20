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
    _interpolate_cross_section,
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
    select_calc_energy,
    separate_coexisting_species_signals,
)
from bl03u_masstool.frontends.pyqt_app.theme import get_plot_theme
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - only used when optional plotting is unavailable
    pg = None

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin

class MoleFractionDialog(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, calibration: Calibration, normalization_settings, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.normalization_settings = normalization_settings
        self.settings = load_mole_fraction_settings()
        self.database: list[dict] = []
        self.mz_index: dict[int, list[int]] = {}
        self.expansion_coefficients: dict[float, float] = {}
        self.parent_mf_results: dict[float, float] = {}
        self.parent_mf_by_energy: dict[float, dict[float, float]] = {}
        self.parent_signal_by_energy: dict[float, dict[float, float]] = {}
        self.parent_config_by_energy: dict[float, dict] = {}
        self.energy_parent_config: dict[float, dict] = {}
        self.product_mf_results: dict[str, dict[float, float]] = {}
        self.isomeric_results: dict[int, dict[str, dict[float, float]]] = {}
        self.temperature_scan_data: dict[float, dict] = {}
        self.pie_species_data: list[dict] = []
        self.available_energies: list[float] = []
        self.all_species_mf: dict[tuple, dict[float, float]] = {}
        self.kr_data: dict[float, float] = dict(self.settings.kr_data)
        self.peak_ranges: dict[int, tuple[int, int]] = {}
        self._init_ui()
        self._auto_load_database()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 摘要栏
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
        self.summary_open_project_btn = QtWidgets.QPushButton("打开项目设置")
        self.summary_open_project_btn.setObjectName("WorkflowButton")
        self.summary_open_project_btn.clicked.connect(self._open_project_settings)
        summary_layout.addWidget(self.summary_open_project_btn)
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("ProjectStatus")
        summary_layout.addWidget(self.status_label)
        summary_layout.addStretch()
        layout.addWidget(self.summary_bar)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_data_tab(), "1. 数据加载")
        self.tabs.addTab(self._create_params_tab(), "2. 参数设置")
        self.tabs.addTab(self._create_energy_parent_tab(), "2.5 多能量母体配置")
        self.tabs.addTab(self._create_parent_tab(), "3. 母体摩尔分数")
        self.tabs.addTab(self._create_auto_mf_tab(), "4. 自动计算摩尔分数")
        self.tabs.addTab(self._create_results_tab(), "5. 结果汇总")
        layout.addWidget(self.tabs)

    def _auto_load_database(self):
        try:
            db_path = species_database_path()
            if db_path.exists():
                self.database, self.mz_index = load_species_database(str(db_path))
                self.lbl_db_status.setText(f"已加载 {len(self.database)} 个物种")
                self.lbl_db_status.setStyleSheet("color: #6495ed;")
                if hasattr(self, "combo_parent_species"):
                    self._update_parent_species_list()
                if hasattr(self, "energy_parent_table"):
                    self._refresh_energy_parent_table()
        except Exception:
            pass

    def set_project_settings(self, ps: ProjectSettings) -> None:
        """Apply ProjectSettings defaults to MoleFractionDialog controls."""
        self.project_settings = ps
        if hasattr(self, "spin_md_exponent"):
            self.spin_md_exponent.setValue(ps.mf_mass_disc_exponent)
        if hasattr(self, "spin_parent_mz"):
            self.spin_parent_mz.setValue(ps.mf_parent_mz)
        if ps.mf_kr_data:
            self.kr_data = dict(ps.mf_kr_data)
        # 更新摘要栏
        if hasattr(self, "summary_project_label"):
            project_name = ps.project_name or "---"
            system = ps.system or "---"
            self.summary_project_label.setText(f"项目: {project_name}")
            self.summary_system_label.setText(f"体系: {system}")
            data_path = (ps.temperature_scan_folder or ps.pie_scan_folder or "---")
            self.summary_data_label.setText(f"数据源: {data_path}")

    def _open_project_settings(self):
        """跳转到项目管理页面。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("project")

    def set_busy(self, busy: bool, message: str) -> None:
        """Disable UI during long-running computation."""
        self.status_label.setText(message)
        self.tabs.setDisabled(busy)

    def _create_data_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        source_bar = QtWidgets.QWidget()
        source_layout = QtWidgets.QHBoxLayout(source_bar)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(8)
        source_layout.addWidget(QtWidgets.QLabel("物种数据库:"))
        self.lbl_db_status = QtWidgets.QLabel("未加载数据库")
        self.lbl_db_status.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        btn_load_db = QtWidgets.QPushButton("加载物种数据库")
        btn_load_db.setToolTip("加载PICS物种数据库，用于获取物种的电离能、分子式等信息")
        btn_load_db.clicked.connect(self._load_database)
        source_layout.addWidget(btn_load_db)
        source_layout.addWidget(self.lbl_db_status)
        source_layout.addStretch()
        source_bar.setMaximumHeight(42)
        source_bar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        layout.addWidget(source_bar)

        self.data_load_tabs = QtWidgets.QTabWidget()

        ts_panel = QtWidgets.QWidget()
        ts_layout = QtWidgets.QVBoxLayout(ts_panel)
        ts_layout.setContentsMargins(8, 8, 8, 8)
        ts_layout.setSpacing(8)
        ts_control_panel = QtWidgets.QWidget()
        ts_control_panel.setObjectName("ControlBar")
        ts_control_panel_layout = QtWidgets.QVBoxLayout(ts_control_panel)
        ts_control_panel_layout.setContentsMargins(10, 8, 10, 8)
        ts_control_panel_layout.setSpacing(8)

        file_peak_layout = QtWidgets.QHBoxLayout()
        file_peak_layout.setSpacing(8)
        file_peak_layout.addWidget(QtWidgets.QLabel("文件:"))
        btn_add_folder = QtWidgets.QPushButton("添加能量文件夹")
        btn_add_folder.setToolTip("添加包含温度扫描txt文件的能量文件夹")
        btn_add_folder.clicked.connect(self._add_energy_folder)
        file_peak_layout.addWidget(btn_add_folder)
        btn_clear = QtWidgets.QPushButton("清空数据")
        btn_clear.setToolTip("清空所有已加载的温度扫描数据")
        btn_clear.clicked.connect(self._clear_temperature_scan)
        file_peak_layout.addWidget(btn_clear)
        self.lbl_ts_folder = QtWidgets.QLabel("未选择文件夹")
        self.lbl_ts_folder.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        self.lbl_ts_folder.setMinimumWidth(120)
        self.lbl_ts_folder.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        file_peak_layout.addWidget(self.lbl_ts_folder, 1)

        file_peak_layout.addSpacing(16)
        file_peak_layout.addWidget(QtWidgets.QLabel("卡峰:"))
        self.btn_import_peaks = QtWidgets.QPushButton("导入卡峰范围")
        self.btn_import_peaks.setToolTip("导入包含质量数、起始通道、结束通道的CSV或Excel文件，并按该范围手动积分")
        self.btn_import_peaks.clicked.connect(self._import_peak_ranges)
        file_peak_layout.addWidget(self.btn_import_peaks)
        self.btn_auto_find_peaks = QtWidgets.QPushButton("恢复自动寻峰")
        self.btn_auto_find_peaks.setToolTip("清除手动卡峰范围，并对已加载温度扫描数据重新自动寻峰")
        self.btn_auto_find_peaks.clicked.connect(self._auto_find_peaks)
        file_peak_layout.addWidget(self.btn_auto_find_peaks)
        self.lbl_peak_status = QtWidgets.QLabel("未设置（将自动寻峰）")
        self.lbl_peak_status.setStyleSheet("color: #6495ed;")
        self.lbl_peak_status.setMinimumWidth(150)
        self.lbl_peak_status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        file_peak_layout.addWidget(self.lbl_peak_status, 1)
        ts_control_panel_layout.addLayout(file_peak_layout)

        energy_layout = QtWidgets.QHBoxLayout()
        energy_layout.setSpacing(8)
        energy_layout.addWidget(QtWidgets.QLabel("能量:"))
        self.combo_energy_select = QtWidgets.QComboBox()
        self.combo_energy_select.addItem("全部能量")
        self.combo_energy_select.setMinimumWidth(120)
        energy_layout.addWidget(self.combo_energy_select)
        self.lbl_energy_count = QtWidgets.QLabel("")
        self.lbl_energy_count.setStyleSheet("color: #6495ed;")
        energy_layout.addWidget(self.lbl_energy_count)
        self.btn_remove_energy = QtWidgets.QPushButton("移除选中能量")
        self.btn_remove_energy.clicked.connect(self._remove_selected_energy)
        self.btn_remove_energy.setEnabled(False)
        self.combo_energy_select.currentTextChanged.connect(self._on_energy_select)
        energy_layout.addWidget(self.btn_remove_energy)
        energy_layout.addStretch()
        ts_control_panel_layout.addLayout(energy_layout)
        ts_layout.addWidget(ts_control_panel)

        self.ts_data_table = QtWidgets.QTableWidget()
        self.ts_data_table.setColumnCount(6)
        self.ts_data_table.setHorizontalHeaderLabels(["能量(eV)", "温度(°C)", "文件名", "IO(nA)", "重复次数", "检测到的质量数"])
        self.ts_data_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.ts_data_table.setMinimumHeight(140)
        ts_layout.addWidget(self.ts_data_table, 1)
        self.data_load_tabs.addTab(ts_panel, "温度扫描数据")

        pie_panel = QtWidgets.QWidget()
        pie_layout = QtWidgets.QVBoxLayout(pie_panel)
        pie_layout.setContentsMargins(8, 8, 8, 8)
        pie_layout.setSpacing(8)
        pie_control_panel = QtWidgets.QWidget()
        pie_control_panel.setObjectName("ControlBar")
        pie_control_layout = QtWidgets.QHBoxLayout(pie_control_panel)
        pie_control_layout.setContentsMargins(10, 8, 10, 8)
        pie_control_layout.setSpacing(8)
        btn_load_pie = QtWidgets.QPushButton("加载PIE鉴定结果")
        btn_load_pie.clicked.connect(self._load_pie_results)
        pie_control_layout.addWidget(btn_load_pie)
        self.lbl_pie_status = QtWidgets.QLabel("未加载")
        self.lbl_pie_status.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        pie_control_layout.addWidget(self.lbl_pie_status)
        pie_control_layout.addStretch()
        pie_layout.addWidget(pie_control_panel)

        self.pie_species_table = QtWidgets.QTableWidget()
        self.pie_species_table.setColumnCount(5)
        self.pie_species_table.setHorizontalHeaderLabels(["质量数", "物种名称", "电离能(eV)", "贡献比例(%)", "R²"])
        self.pie_species_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.pie_species_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.pie_species_table.setMinimumHeight(150)
        pie_layout.addWidget(self.pie_species_table, 1)
        self.data_load_tabs.addTab(pie_panel, "PIE鉴定结果")

        layout.addWidget(self.data_load_tabs, 1)
        return widget

    def _create_params_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        calib_group = QtWidgets.QGroupBox("质量数校准（来自主程序）")
        calib_layout = QtWidgets.QGridLayout(calib_group)
        calib_layout.setContentsMargins(10, 8, 10, 8)
        calib_layout.setHorizontalSpacing(8)
        calib_layout.setVerticalSpacing(4)
        calib_layout.addWidget(QtWidgets.QLabel("校准状态:"), 0, 0)
        self.lbl_calib_status = QtWidgets.QLabel("已同步主程序校准")
        self.lbl_calib_status.setStyleSheet("color: #4ecdc4;")
        calib_layout.addWidget(self.lbl_calib_status, 0, 1)
        formula_label = QtWidgets.QLabel("m/z = A*x² + B*x + C")
        calib_layout.addWidget(formula_label, 0, 2, 1, 4)
        calib_layout.addWidget(QtWidgets.QLabel("A:"), 1, 0)
        self.txt_calib_A = QtWidgets.QLineEdit(f"{self.calibration.a:.10f}")
        self.txt_calib_A.setFixedWidth(150)
        self.txt_calib_A.setReadOnly(True)
        calib_layout.addWidget(self.txt_calib_A, 1, 1)
        calib_layout.addWidget(QtWidgets.QLabel("B:"), 1, 2)
        self.txt_calib_B = QtWidgets.QLineEdit(f"{self.calibration.b:.6f}")
        self.txt_calib_B.setFixedWidth(130)
        self.txt_calib_B.setReadOnly(True)
        calib_layout.addWidget(self.txt_calib_B, 1, 3)
        calib_layout.addWidget(QtWidgets.QLabel("C:"), 1, 4)
        self.txt_calib_C = QtWidgets.QLineEdit(f"{self.calibration.c:.4f}")
        self.txt_calib_C.setFixedWidth(130)
        self.txt_calib_C.setReadOnly(True)
        calib_layout.addWidget(self.txt_calib_C, 1, 5)
        calib_layout.setColumnStretch(6, 1)
        calib_group.setMaximumHeight(92)
        calib_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        layout.addWidget(calib_group)

        self.params_preview_tabs = QtWidgets.QTabWidget()

        md_panel = QtWidgets.QWidget()
        md_layout = QtWidgets.QVBoxLayout(md_panel)
        md_layout.setContentsMargins(8, 8, 8, 8)
        md_layout.setSpacing(8)
        md_control_panel = QtWidgets.QWidget()
        md_control_panel.setObjectName("ControlBar")
        md_controls = QtWidgets.QHBoxLayout(md_control_panel)
        md_controls.setContentsMargins(10, 8, 10, 8)
        md_controls.setSpacing(8)
        md_controls.addWidget(QtWidgets.QLabel("实验条件:"))
        self.combo_md_preset = QtWidgets.QComboBox()
        self.combo_md_preset.setToolTip("选择实验条件预设值，自动设置质量歧视指数n")
        for name in MASS_DISCRIMINATION_PRESETS:
            self.combo_md_preset.addItem(name)
        self.combo_md_preset.currentTextChanged.connect(self._on_md_preset_changed)
        md_controls.addWidget(self.combo_md_preset)
        md_controls.addWidget(QtWidgets.QLabel("指数 n:"))
        self.spin_md_exponent = QtWidgets.QDoubleSpinBox()
        self.spin_md_exponent.setToolTip("质量歧视因子公式 D_i = (MW/30)^n 中的指数n，n值取决于离子源类型和质量分析器特性")
        self.spin_md_exponent.setDecimals(5)
        self.spin_md_exponent.setValue(self.settings.mass_disc_exponent)
        self.spin_md_exponent.setSingleStep(0.001)
        self.spin_md_exponent.valueChanged.connect(self._on_md_exponent_changed)
        md_controls.addWidget(self.spin_md_exponent)
        md_formula = QtWidgets.QLabel("D_i = (MW/30)^n")
        md_controls.addWidget(md_formula)
        md_controls.addStretch()
        btn_calc_md = QtWidgets.QPushButton("预览质量歧视因子")
        btn_calc_md.clicked.connect(self._preview_mass_discrimination)
        md_controls.addWidget(btn_calc_md)
        md_layout.addWidget(md_control_panel)
        self.md_preview_table = QtWidgets.QTableWidget()
        self.md_preview_table.setColumnCount(3)
        self.md_preview_table.setHorizontalHeaderLabels(["物种", "分子量", "D_i"])
        self.md_preview_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.md_preview_table.setMinimumHeight(150)
        self.md_preview_table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        md_layout.addWidget(self.md_preview_table, 1)
        self.params_preview_tabs.addTab(md_panel, "质量歧视因子")

        ec_panel = QtWidgets.QWidget()
        ec_layout = QtWidgets.QVBoxLayout(ec_panel)
        ec_layout.setContentsMargins(8, 8, 8, 8)
        ec_layout.setSpacing(8)
        ec_control_panel = QtWidgets.QWidget()
        ec_control_panel.setObjectName("ControlBar")
        ec_top = QtWidgets.QGridLayout()
        ec_top.setContentsMargins(10, 8, 10, 8)
        ec_top.setHorizontalSpacing(8)
        ec_top.setVerticalSpacing(6)
        ec_top.addWidget(QtWidgets.QLabel("Kr数据来源:"), 0, 0)
        self.combo_kr_source = QtWidgets.QComboBox()
        self.combo_kr_source.setToolTip("选择Kr膨胀系数λ(T)的数据来源：内置默认数据/自定义文件/从温度扫描数据提取")
        self.combo_kr_source.addItems(["使用默认数据", "从文件加载", "从高能量温度扫描提取"])
        self.combo_kr_source.currentTextChanged.connect(self._on_kr_source_changed)
        ec_top.addWidget(self.combo_kr_source, 0, 1)
        ec_top.addWidget(QtWidgets.QLabel("Kr质量数 m/z:"), 0, 2)
        self.spin_kr_mz = QtWidgets.QSpinBox()
        self.spin_kr_mz.setToolTip("Kr同位素峰质量数，默认使用 m/z 84")
        self.spin_kr_mz.setRange(1, 200)
        self.spin_kr_mz.setValue(84)
        ec_top.addWidget(self.spin_kr_mz, 0, 3)
        self.btn_load_kr = QtWidgets.QPushButton("加载Kr数据")
        self.btn_load_kr.clicked.connect(self._load_kr_data)
        self.btn_load_kr.setEnabled(False)
        self.btn_extract_kr = QtWidgets.QPushButton("从温度扫描提取Kr")
        self.btn_extract_kr.clicked.connect(self._extract_kr_from_scan)
        self.btn_extract_kr.setEnabled(False)
        ec_top.addWidget(self.btn_load_kr, 0, 4)
        ec_top.addWidget(self.btn_extract_kr, 0, 5)
        btn_calc_lambda = QtWidgets.QPushButton("计算膨胀系数")
        btn_calc_lambda.clicked.connect(self._calc_expansion_coefficients)
        ec_top.addWidget(btn_calc_lambda, 0, 6)
        ec_top.setColumnStretch(7, 1)
        ec_control_panel.setLayout(ec_top)
        ec_layout.addWidget(ec_control_panel)

        self.kr_table = QtWidgets.QTableWidget()
        self.kr_table.setColumnCount(4)
        self.kr_table.setHorizontalHeaderLabels(["温度(°C)", "文件名", "Kr信号积分", "λ(T)"])
        self.kr_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.kr_table.setMinimumHeight(180)
        self.kr_table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        ec_layout.addWidget(self.kr_table, 1)
        self.params_preview_tabs.addTab(ec_panel, "膨胀系数")
        layout.addWidget(self.params_preview_tabs, 1)

        self._fill_kr_table()
        return widget

    def _create_energy_parent_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        top_layout = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("刷新可用能量")
        btn_refresh.setToolTip("根据已加载温度扫描数据刷新每个能量的母体配置行")
        btn_refresh.clicked.connect(self._refresh_energy_parent_table)
        top_layout.addWidget(btn_refresh)

        btn_reset = QtWidgets.QPushButton("重置为默认母体")
        btn_reset.clicked.connect(self._reset_energy_parent_config)
        top_layout.addWidget(btn_reset)

        btn_apply = QtWidgets.QPushButton("应用配置")
        btn_apply.clicked.connect(self._apply_energy_parent_config)
        top_layout.addWidget(btn_apply)

        self.lbl_energy_parent_status = QtWidgets.QLabel("未加载温度扫描能量")
        self.lbl_energy_parent_status.setStyleSheet("color: #6495ed;")
        top_layout.addWidget(self.lbl_energy_parent_status, 1)
        layout.addLayout(top_layout)

        self.energy_parent_table = QtWidgets.QTableWidget()
        self.energy_parent_table.setColumnCount(4)
        self.energy_parent_table.setHorizontalHeaderLabels(["能量(eV)", "参考母体 m/z", "参考母体物种", "状态"])
        self.energy_parent_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.energy_parent_table, 1)

        return widget

    def _create_parent_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        cfg_group = QtWidgets.QGroupBox("母体参数设置")
        cfg_layout = QtWidgets.QGridLayout(cfg_group)
        cfg_layout.addWidget(QtWidgets.QLabel("母体质量数 m/z:"), 0, 0)
        self.spin_parent_mz = QtWidgets.QSpinBox()
        self.spin_parent_mz.setToolTip("母体物种（反应物）的质量数")
        self.spin_parent_mz.setRange(1, 500)
        self.spin_parent_mz.setValue(self.settings.parent_mz)
        self.spin_parent_mz.valueChanged.connect(self._on_parent_mz_changed)
        cfg_layout.addWidget(self.spin_parent_mz, 0, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("母体物种:"), 1, 0)
        self.combo_parent_species = QtWidgets.QComboBox()
        self.combo_parent_species.setToolTip("在当前母体质量数下选择具体参考母体物种")
        self.combo_parent_species.currentIndexChanged.connect(self._on_parent_species_changed)
        cfg_layout.addWidget(self.combo_parent_species, 1, 1)
        self.lbl_parent_info = QtWidgets.QLabel("")
        self.lbl_parent_info.setStyleSheet("color: #6495ed;")
        cfg_layout.addWidget(self.lbl_parent_info, 1, 2, 1, 2)

        cfg_layout.addWidget(QtWidgets.QLabel("参考温度 T₀ (°C):"), 2, 0)
        self.spin_parent_t0 = QtWidgets.QSpinBox()
        self.spin_parent_t0.setToolTip("选定一个参考温度点，用于计算母体摩尔分数的基准")
        self.spin_parent_t0.setRange(0, 2000)
        self.spin_parent_t0.setValue(int(self.settings.reference_temperature or 550))
        cfg_layout.addWidget(self.spin_parent_t0, 2, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("初始摩尔分数 X(T₀):"), 3, 0)
        self.spin_parent_mf0 = QtWidgets.QDoubleSpinBox()
        self.spin_parent_mf0.setToolTip("母体物种在参考温度T₀处的摩尔分数（已知或假设值）")
        self.spin_parent_mf0.setRange(0.0, 1.0)
        self.spin_parent_mf0.setDecimals(6)
        self.spin_parent_mf0.setValue(self.settings.parent_initial_mf)
        self.spin_parent_mf0.setSingleStep(0.0001)
        cfg_layout.addWidget(self.spin_parent_mf0, 3, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("光子能量 E (eV):"), 4, 0)
        self.spin_parent_energy = QtWidgets.QDoubleSpinBox()
        self.spin_parent_energy.setToolTip("实验使用的VUV光子能量 (eV)，用于查找物种在此能量下的光电离截面")
        self.spin_parent_energy.setRange(0.0, 30.0)
        self.spin_parent_energy.setDecimals(2)
        self.spin_parent_energy.setValue(self.settings.photon_energy)
        self.spin_parent_energy.setSingleStep(0.5)
        cfg_layout.addWidget(self.spin_parent_energy, 4, 1)
        cfg_layout.setColumnStretch(2, 1)
        layout.addWidget(cfg_group)

        btn_calc_parent = QtWidgets.QPushButton("开始计算")
        btn_calc_parent.clicked.connect(self._calc_parent_mole_fraction)
        layout.addWidget(btn_calc_parent)

        self.parent_result_table = QtWidgets.QTableWidget()
        self.parent_result_table.setColumnCount(4)
        self.parent_result_table.setHorizontalHeaderLabels(["温度(°C)", "信号 S(T,E)", "λ(T)", "摩尔分数 X(T)"])
        self.parent_result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.parent_result_table, 1)

        self._update_parent_species_list()
        return widget

    def _available_parent_mz_values(self) -> list[int]:
        mz_values = {int(self.spin_parent_mz.value())} if hasattr(self, "spin_parent_mz") else set()

        for energy_data in self.temperature_scan_data.values():
            for info in energy_data.values():
                for peak in info.get("peaks_info", []) or []:
                    mz = peak.get("mz_rounded")
                    if mz:
                        mz_values.add(int(mz))

        for species in self.pie_species_data:
            mz = species.get("mz")
            if mz:
                mz_values.add(int(mz))

        for config in self.energy_parent_config.values():
            mz = config.get("mz")
            if mz:
                mz_values.add(int(mz))

        return sorted(mz_values)

    def _species_options_for_mz(self, mz: int) -> list[tuple[str, float | None]]:
        seen: set[str] = set()
        options: list[tuple[str, float | None]] = []

        for species in self.pie_species_data:
            if species.get("mz") != mz:
                continue
            name = species.get("species") or species.get("name") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            options.append((name, species.get("ie")))

        for species in self.database:
            if species.get("mz") != mz:
                continue
            name = species.get("species") or species.get("name") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            options.append((name, species.get("ie")))

        options.sort(key=lambda item: (item[1] if item[1] is not None else float("inf"), item[0]))
        return options

    def _populate_parent_species_combo(self, combo: QtWidgets.QComboBox, mz: int, include_auto: bool = True) -> None:
        combo.blockSignals(True)
        combo.clear()
        if include_auto:
            combo.addItem("自动检测", None)
        for species_name, ie in self._species_options_for_mz(mz):
            ie_text = f" (IE={ie:.3f} eV)" if ie is not None else ""
            combo.addItem(f"{species_name}{ie_text}", species_name)
        combo.blockSignals(False)

    def _set_combo_current_data(self, combo: QtWidgets.QComboBox, value) -> None:
        for idx in range(combo.count()):
            if combo.itemData(idx) == value:
                combo.setCurrentIndex(idx)
                return
        if value is None:
            combo.setCurrentIndex(0)

    def _selected_parent_species_name(self) -> str | None:
        if not hasattr(self, "combo_parent_species"):
            return None
        value = self.combo_parent_species.currentData()
        return str(value) if value else None

    def _parent_result_label(self, mz: int, species_name: str | None, energy: float | None = None) -> str:
        base_name = species_name or "母体"
        if energy is None:
            return f"{base_name}(母体,m/z={mz})" if species_name else f"母体(m/z={mz})"
        return f"{base_name}(母体,m/z={mz},E={energy:.2f}eV)"

    def _on_parent_mz_changed(self):
        self.settings.parent_mz = self.spin_parent_mz.value()
        self._update_parent_species_list()
        self._refresh_energy_parent_table()

    def _on_parent_species_changed(self):
        selected = self._selected_parent_species_name()
        mz = self.spin_parent_mz.value()
        options = self._species_options_for_mz(mz)
        if selected:
            ie = next((item_ie for name, item_ie in options if name == selected), None)
            ie_text = f"，IE={ie:.3f} eV" if ie is not None else ""
            self.lbl_parent_info.setText(f"当前母体: {selected}{ie_text}")
        elif options:
            self.lbl_parent_info.setText(f"自动检测：m/z {mz} 下有 {len(options)} 个候选物种")
        else:
            self.lbl_parent_info.setText("当前 m/z 未匹配到候选物种")
        self._refresh_energy_parent_table()

    def _update_parent_species_list(self):
        if not hasattr(self, "combo_parent_species"):
            return
        previous = self._selected_parent_species_name()
        mz = self.spin_parent_mz.value()
        self._populate_parent_species_combo(self.combo_parent_species, mz, include_auto=True)
        self._set_combo_current_data(self.combo_parent_species, previous)
        self._on_parent_species_changed()

    def _refresh_energy_parent_table(self):
        if not hasattr(self, "energy_parent_table"):
            return

        self.energy_parent_table.setRowCount(0)
        if not self.available_energies:
            self.lbl_energy_parent_status.setText("未加载温度扫描能量")
            return

        mz_values = self._available_parent_mz_values()
        default_mz = self.spin_parent_mz.value() if hasattr(self, "spin_parent_mz") else self.settings.parent_mz
        default_species = self._selected_parent_species_name()

        for energy in sorted(self.available_energies):
            row = self.energy_parent_table.rowCount()
            self.energy_parent_table.insertRow(row)

            energy_item = QtWidgets.QTableWidgetItem(f"{energy:.2f}")
            energy_item.setFlags(energy_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.energy_parent_table.setItem(row, 0, energy_item)

            cfg = self.energy_parent_config.get(energy, {})
            mz = int(cfg.get("mz") or default_mz)

            mz_combo = QtWidgets.QComboBox()
            for value in mz_values:
                mz_combo.addItem(str(value), value)
            self._set_combo_current_data(mz_combo, mz)
            self.energy_parent_table.setCellWidget(row, 1, mz_combo)

            species_combo = QtWidgets.QComboBox()
            self._populate_parent_species_combo(species_combo, mz, include_auto=True)
            self._set_combo_current_data(species_combo, cfg.get("species_name", default_species))
            self.energy_parent_table.setCellWidget(row, 2, species_combo)

            status_item = QtWidgets.QTableWidgetItem("已配置" if cfg else "默认")
            status_item.setFlags(status_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.energy_parent_table.setItem(row, 3, status_item)

            mz_combo.currentIndexChanged.connect(
                lambda _, r=row, e=energy: self._on_mz_changed_for_parent_config(r, e)
            )
            species_combo.currentIndexChanged.connect(
                lambda _, e=energy, mz_cb=mz_combo, species_cb=species_combo: self._on_species_changed_for_parent_config(
                    e, mz_cb, species_cb
                )
            )

        self.lbl_energy_parent_status.setText(f"已加载 {len(self.available_energies)} 个能量配置行")

    def _on_mz_changed_for_parent_config(self, row: int, energy: float):
        mz_combo = self.energy_parent_table.cellWidget(row, 1)
        species_combo = self.energy_parent_table.cellWidget(row, 2)
        if mz_combo is None or species_combo is None:
            return
        mz = mz_combo.currentData()
        if mz is None:
            return
        previous_species = species_combo.currentData()
        self._populate_parent_species_combo(species_combo, int(mz), include_auto=True)
        self._set_combo_current_data(species_combo, previous_species)
        self._on_species_changed_for_parent_config(energy, mz_combo, species_combo)

    def _on_species_changed_for_parent_config(
        self,
        energy: float,
        mz_combo: QtWidgets.QComboBox,
        species_combo: QtWidgets.QComboBox,
    ):
        mz = mz_combo.currentData()
        if mz is None:
            return
        species_name = species_combo.currentData()
        self.energy_parent_config[energy] = {
            "mz": int(mz),
            "species_name": str(species_name) if species_name else None,
        }
        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}

    def _reset_energy_parent_config(self):
        self.energy_parent_config = {}
        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}
        self._refresh_energy_parent_table()
        self.lbl_energy_parent_status.setText("已重置为默认母体")

    def _apply_energy_parent_config(self):
        if hasattr(self, "energy_parent_table"):
            for row in range(self.energy_parent_table.rowCount()):
                energy_item = self.energy_parent_table.item(row, 0)
                mz_combo = self.energy_parent_table.cellWidget(row, 1)
                species_combo = self.energy_parent_table.cellWidget(row, 2)
                if not energy_item or not mz_combo or not species_combo:
                    continue
                energy = float(energy_item.text())
                mz = mz_combo.currentData()
                if mz is None:
                    continue
                species_name = species_combo.currentData()
                self.energy_parent_config[energy] = {
                    "mz": int(mz),
                    "species_name": str(species_name) if species_name else None,
                }

        config_count = len(self.energy_parent_config)
        if self.parent_mf_results:
            self._recalculate_parent_mf_by_energy()
        self._refresh_results_view()
        self.lbl_energy_parent_status.setText(f"已应用 {config_count} 个能量的母体配置")
        QtWidgets.QMessageBox.information(self, "成功", f"已应用 {config_count} 个能量的母体配置")

    def _get_parent_config_for_energy(self, energy: float) -> dict:
        if energy in self.energy_parent_config:
            cfg = self.energy_parent_config[energy]
            if cfg.get("mz"):
                return {
                    "mz": int(cfg["mz"]),
                    "species_name": cfg.get("species_name"),
                }
        return {
            "mz": int(self.spin_parent_mz.value()),
            "species_name": self._selected_parent_species_name(),
        }

    def _create_auto_mf_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        ctrl_bar = QtWidgets.QWidget()
        ctrl_bar.setObjectName("ControlBar")
        ctrl_layout = QtWidgets.QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(10, 8, 10, 8)
        ctrl_layout.setSpacing(8)
        btn_calc_auto = QtWidgets.QPushButton("开始计算")
        btn_calc_auto.clicked.connect(self._calculate_auto_mf)
        ctrl_layout.addWidget(btn_calc_auto)
        ctrl_layout.addWidget(QtWidgets.QLabel("曲线显示:"))
        self.combo_auto_plot_scope = QtWidgets.QComboBox()
        self.combo_auto_plot_scope.addItem("仅产物曲线", "products")
        self.combo_auto_plot_scope.addItem("全部曲线", "all")
        self.combo_auto_plot_scope.addItem("仅母体参考", "parents")
        self.combo_auto_plot_scope.addItem("选中结果", "selected")
        self.combo_auto_plot_scope.currentIndexChanged.connect(self._plot_all_auto_mf)
        ctrl_layout.addWidget(self.combo_auto_plot_scope)
        self.lbl_auto_status = QtWidgets.QLabel("未计算")
        self.lbl_auto_status.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        ctrl_layout.addWidget(self.lbl_auto_status, 1)
        layout.addWidget(ctrl_bar)

        auto_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        auto_splitter.setChildrenCollapsible(False)

        result_group = QtWidgets.QGroupBox("计算结果")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        result_layout.setContentsMargins(10, 8, 10, 10)
        self.auto_mf_table = QtWidgets.QTableWidget()
        self.auto_mf_table.setColumnCount(6)
        self.auto_mf_table.setHorizontalHeaderLabels(["类型", "质量数", "物种名称", "电离能(eV)", "光子能量(eV)", "状态"])
        self.auto_mf_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.auto_mf_table.itemSelectionChanged.connect(self._on_auto_mf_selection_changed)
        self.auto_mf_table.setMinimumHeight(170)
        result_layout.addWidget(self.auto_mf_table, 1)
        auto_splitter.addWidget(result_group)

        detail_tabs = QtWidgets.QTabWidget()
        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(8, 8, 8, 8)
        if pg is not None:
            self.auto_mf_plot_widget = pg.PlotWidget()
            self.auto_mf_plot_widget.setBackground("#ffffff")
            self.auto_mf_plot_widget.setMinimumHeight(250)
            self.auto_mf_plot_widget.setLabel("bottom", "温度", units="°C")
            self.auto_mf_plot_widget.setLabel("left", "摩尔分数")
            self.auto_mf_plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.auto_mf_plot_widget.addLegend()
            plot_layout.addWidget(self.auto_mf_plot_widget)
        else:
            self.auto_mf_plot_widget = None
            placeholder = QtWidgets.QLabel("pyqtgraph 未安装，无法显示绘图")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plot_layout.addWidget(placeholder)
        detail_tabs.addTab(plot_panel, "摩尔分数-温度曲线")

        warning_panel = QtWidgets.QWidget()
        warning_layout = QtWidgets.QVBoxLayout(warning_panel)
        warning_layout.setContentsMargins(8, 8, 8, 8)
        self.txt_warnings = QtWidgets.QTextEdit()
        self.txt_warnings.setReadOnly(True)
        warning_layout.addWidget(self.txt_warnings)
        detail_tabs.addTab(warning_panel, "警告信息")

        auto_splitter.addWidget(detail_tabs)
        auto_splitter.setStretchFactor(0, 1)
        auto_splitter.setStretchFactor(1, 2)
        auto_splitter.setSizes([240, 430])
        layout.addWidget(auto_splitter, 1)

        return widget

    def _create_results_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("导出结果 (Excel/CSV)")
        btn_export.clicked.connect(self._export_results)
        btn_layout.addWidget(btn_export)
        btn_refresh_results = QtWidgets.QPushButton("刷新结果")
        btn_refresh_results.clicked.connect(self._refresh_results_view)
        btn_layout.addWidget(btn_refresh_results)
        self.lbl_results_status = QtWidgets.QLabel("暂无结果")
        self.lbl_results_status.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        btn_layout.addWidget(self.lbl_results_status, 1)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        results_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        results_splitter.setChildrenCollapsible(False)

        table_group = QtWidgets.QGroupBox("结果表格")
        table_layout = QtWidgets.QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 8, 10, 10)
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.results_table.setWordWrap(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.results_table.setMinimumHeight(180)
        table_layout.addWidget(self.results_table, 1)
        results_splitter.addWidget(table_group)

        plot_group = QtWidgets.QGroupBox("摩尔分数-温度曲线")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(10, 8, 10, 10)
        plot_body = QtWidgets.QWidget()
        plot_body_layout = QtWidgets.QHBoxLayout(plot_body)
        plot_body_layout.setContentsMargins(0, 0, 0, 0)
        plot_body_layout.setSpacing(8)
        if pg is not None:
            self.mf_plot_widget = pg.PlotWidget()
            self.mf_plot_widget.setBackground("#ffffff")
            self.mf_plot_widget.setMinimumHeight(300)
            self.mf_plot_widget.setLabel("bottom", "温度", units="°C")
            self.mf_plot_widget.setLabel("left", "摩尔分数")
            self.mf_plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_body_layout.addWidget(self.mf_plot_widget, 1)
        else:
            self.mf_plot_widget = None
            placeholder = QtWidgets.QLabel("pyqtgraph 未安装，无法显示绘图")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plot_body_layout.addWidget(placeholder, 1)
        series_group = QtWidgets.QGroupBox("曲线列表")
        series_layout = QtWidgets.QVBoxLayout(series_group)
        series_layout.setContentsMargins(8, 8, 8, 8)
        self.mf_series_list = QtWidgets.QListWidget()
        self.mf_series_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.mf_series_list.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.mf_series_list.setMinimumWidth(240)
        self.mf_series_list.setMaximumWidth(420)
        series_layout.addWidget(self.mf_series_list)
        plot_body_layout.addWidget(series_group)
        plot_layout.addWidget(plot_body, 1)
        results_splitter.addWidget(plot_group)
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 2)
        results_splitter.setSizes([240, 380])
        layout.addWidget(results_splitter, 1)

        return widget

    def _apply_mz_calibration(self, raw_index):
        return self.calibration.a * raw_index ** 2 + self.calibration.b * raw_index + self.calibration.c

    def _import_peak_ranges(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择卡峰范围文件",
            "",
            "Excel Files (*.xlsx *.xls);;CSV/TXT Files (*.csv *.txt);;All Files (*)",
        )
        if not file_path:
            return

        try:
            path_lower = file_path.lower()
            if path_lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path, encoding="utf-8-sig", sep=None, engine="python")

            column_mapping = self._detect_peak_range_columns(df)
            if len(column_mapping) < 3:
                missing = []
                if "mz" not in column_mapping:
                    missing.append("质量数（mz/mass/m/z）")
                if "start" not in column_mapping:
                    missing.append("起始通道（start/left/起始）")
                if "end" not in column_mapping:
                    missing.append("结束通道（end/right/结束）")
                QtWidgets.QMessageBox.warning(
                    self,
                    "提示",
                    "无法识别所有必需列。\n"
                    "请确保文件包含质量数、起始通道、结束通道三列。\n\n"
                    f"当前列: {', '.join(str(c) for c in df.columns)}\n"
                    f"无法识别: {', '.join(missing)}",
                )
                return

            peak_ranges: dict[int, tuple[int, int]] = {}
            for _, row in df.iterrows():
                mz_val = row.get(column_mapping["mz"])
                start_val = row.get(column_mapping["start"])
                end_val = row.get(column_mapping["end"])
                if pd.isna(mz_val) or pd.isna(start_val) or pd.isna(end_val):
                    continue

                mz = int(round(float(mz_val)))
                start = int(round(float(start_val)))
                end = int(round(float(end_val)))
                if start > end:
                    start, end = end, start
                peak_ranges[mz] = (start, end)

            if not peak_ranges:
                QtWidgets.QMessageBox.warning(self, "提示", "未读取到有效卡峰范围")
                return

            self.peak_ranges = peak_ranges
            peak_count = self._recompute_peak_info() if self.temperature_scan_data else 0
            suffix = f"，已重新积分 {peak_count} 个峰" if self.temperature_scan_data else ""
            self.lbl_peak_status.setText(f"已导入 {len(self.peak_ranges)} 个卡峰范围{suffix}")
            self.lbl_peak_status.setStyleSheet("color: #4ecdc4;")
            QtWidgets.QMessageBox.information(
                self,
                "成功",
                f"导入了 {len(self.peak_ranges)} 个卡峰范围{suffix}",
            )
        except Exception as e:
            import traceback

            QtWidgets.QMessageBox.critical(self, "错误", f"导入卡峰范围失败: {e}\n{traceback.format_exc()}")

    def _detect_peak_range_columns(self, df: pd.DataFrame) -> dict[str, str]:
        def find_column(candidates: list[str], keywords: list[str]) -> str | None:
            candidate_lowers = {c.lower() for c in candidates}
            for col in df.columns:
                col_text = str(col).strip()
                col_lower = col_text.lower()
                if col_text in candidates or col_lower in candidate_lowers:
                    return col
            for col in df.columns:
                col_text = str(col).strip()
                col_lower = col_text.lower()
                if any(keyword in col_text or keyword.lower() in col_lower for keyword in keywords):
                    return col
            return None

        mapping: dict[str, str] = {}
        mz_col = find_column(
            ["质量数", "mz", "mass", "m/z", "m_z", "mass_number", "M/Z", "质量数(m/z)", "质量数 (m/z)"],
            ["质量数", "m/z", "mass", "mz"],
        )
        start_col = find_column(
            ["起始通道", "起始", "start", "start_idx", "left", "left_idx", "begin", "左边界", "左边界索引"],
            ["起始", "start", "left", "begin", "左边界"],
        )
        end_col = find_column(
            ["结束通道", "结束", "end", "end_idx", "right", "right_idx", "finish", "右边界", "右边界索引"],
            ["结束", "end", "right", "finish", "右边界"],
        )
        if mz_col is not None:
            mapping["mz"] = mz_col
        if start_col is not None:
            mapping["start"] = start_col
        if end_col is not None:
            mapping["end"] = end_col
        return mapping

    def _auto_find_peaks(self):
        self.peak_ranges = {}
        if not self.temperature_scan_data:
            self.lbl_peak_status.setText("未设置（将自动寻峰）")
            self.lbl_peak_status.setStyleSheet("color: #6495ed;")
            QtWidgets.QMessageBox.information(self, "提示", "已清除手动卡峰范围，后续加载数据将自动寻峰")
            return

        try:
            self.lbl_peak_status.setText("正在自动寻峰...")
            self.lbl_peak_status.setStyleSheet("color: #f39c12;")
            peak_count = self._recompute_peak_info()
            self.lbl_peak_status.setText(f"自动寻峰完成，检测到 {peak_count} 个峰")
            self.lbl_peak_status.setStyleSheet("color: #4ecdc4;")
            QtWidgets.QMessageBox.information(self, "成功", f"自动寻峰完成，检测到 {peak_count} 个峰")
        except Exception as e:
            import traceback

            QtWidgets.QMessageBox.critical(self, "错误", f"自动寻峰失败: {e}\n{traceback.format_exc()}")
            self.lbl_peak_status.setText("自动寻峰失败")
            self.lbl_peak_status.setStyleSheet("color: #e74c3c;")

    def _recompute_peak_info(self) -> int:
        peak_ranges = self.peak_ranges or None
        total_peaks = 0
        for energy, temp_data in self.temperature_scan_data.items():
            if not temp_data:
                continue

            best_peak_count = -1
            best_peaks_info = []
            for temp in sorted(temp_data.keys()):
                t_data = temp_data[temp].get("avg_data", [])
                t_peaks = self._detect_and_integrate_peaks(t_data, peak_ranges)
                t_count = len(t_peaks)
                if t_count > best_peak_count:
                    best_peak_count = t_count
                    best_peaks_info = t_peaks

            for info in temp_data.values():
                info["peaks_info"] = best_peaks_info
                info.pop("matched_species", None)
                total_peaks += len(best_peaks_info)

        self._update_ts_table_with_species()
        self._refresh_ts_table()
        return total_peaks

    def _load_database(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择物种数据库文件", "",
            "SQLite Files (*.sqlite *.db);;Pickle Files (*.pkl);;Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return
        try:
            self.database, self.mz_index = load_species_database(file_path)
            self.lbl_db_status.setText(f"已加载 {len(self.database)} 个物种")
            self.lbl_db_status.setStyleSheet("color: #6495ed;")
            self._update_parent_species_list()
            self._refresh_energy_parent_table()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载数据库失败: {e}")

    def _load_pie_results(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择PIE鉴定结果文件", "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path, encoding="utf-8-sig")

            required_cols = ["质量数", "物种名称"]
            for col in required_cols:
                if col not in df.columns:
                    QtWidgets.QMessageBox.warning(self, "提示", f"文件中缺少必需列: {col}\n当前列: {', '.join(df.columns.tolist())}")
                    return

            self.pie_species_data = []
            self.pie_species_table.setRowCount(0)

            for _, row in df.iterrows():
                mz = int(row["质量数"]) if pd.notna(row["质量数"]) else None
                species_name = str(row["物种名称"]) if pd.notna(row["物种名称"]) else ""
                ie = float(row["电离能(eV)"]) if "电离能(eV)" in df.columns and pd.notna(row.get("电离能(eV)")) else None
                contribution = float(row["贡献比例(%)"]) if "贡献比例(%)" in df.columns and pd.notna(row.get("贡献比例(%)")) else None
                r_squared = float(row["R²"]) if "R²" in df.columns and pd.notna(row.get("R²")) else None

                if mz is None or not species_name:
                    continue

                self.pie_species_data.append({
                    "mz": mz,
                    "species": species_name,
                    "ie": ie,
                    "contribution": contribution,
                    "r_squared": r_squared,
                })

                table_row = self.pie_species_table.rowCount()
                self.pie_species_table.insertRow(table_row)
                self.pie_species_table.setItem(table_row, 0, QtWidgets.QTableWidgetItem(str(mz)))
                self.pie_species_table.setItem(table_row, 1, QtWidgets.QTableWidgetItem(species_name))
                self.pie_species_table.setItem(table_row, 2, QtWidgets.QTableWidgetItem(f"{ie:.2f}" if ie is not None else "N/A"))
                self.pie_species_table.setItem(table_row, 3, QtWidgets.QTableWidgetItem(f"{contribution:.1f}" if contribution is not None else "N/A"))
                self.pie_species_table.setItem(table_row, 4, QtWidgets.QTableWidgetItem(f"{r_squared:.4f}" if r_squared is not None else "N/A"))

            unique_mz = len(set(d["mz"] for d in self.pie_species_data))
            unique_species = len(set(d["species"] for d in self.pie_species_data))
            self.lbl_pie_status.setText(f"已加载 {unique_mz} 个质量数, {unique_species} 个物种")
            self.lbl_pie_status.setStyleSheet("color: #6495ed;")

            self._update_ts_table_with_species()
            self._update_parent_species_list()
            self._refresh_energy_parent_table()

            QtWidgets.QMessageBox.information(self, "成功", f"加载了 {len(self.pie_species_data)} 条鉴定结果\n{unique_mz} 个质量数, {unique_species} 个物种")
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self, "错误", f"加载PIE鉴定结果失败: {e}\n{traceback.format_exc()}")

    def _update_ts_table_with_species(self):
        if not self.pie_species_data:
            return
        species_by_mz: dict[int, list[str]] = {}
        for d in self.pie_species_data:
            mz = d["mz"]
            if mz not in species_by_mz:
                species_by_mz[mz] = []
            species_by_mz[mz].append(d["species"])

        for energy in self.available_energies:
            if energy not in self.temperature_scan_data:
                continue
            for temp, info in self.temperature_scan_data[energy].items():
                peaks_info = info.get("peaks_info", [])
                matched_species = []
                for peak in peaks_info:
                    if not peak.get("overlapped", False):
                        mz = peak["mz_rounded"]
                        if mz in species_by_mz:
                            matched_species.extend(species_by_mz[mz])
                info["matched_species"] = matched_species

    def _add_energy_folder(self):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setWindowTitle("选择温度扫描文件夹（可多选）")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        list_view = dlg.findChild(QtWidgets.QListView)
        if list_view:
            list_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        tree_view = dlg.findChild(QtWidgets.QTreeView)
        if tree_view:
            tree_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        if not dlg.exec():
            return
        folders = dlg.selectedFiles()
        if not folders:
            return

        import re
        added_count = 0
        skipped_count = 0

        for folder in folders:
            try:
                energy = None
                folder_name = Path(folder).name
                energy_match = re.search(r"([\d.]+)\s*[eE][vV]", folder_name)
                if energy_match:
                    energy = float(energy_match.group(1))
                if energy is None:
                    QtWidgets.QMessageBox.warning(self, "提示", f"无法从文件夹名 '{folder_name}' 中识别能量值，请确保文件夹名包含能量信息（如 8.0eV）")
                    skipped_count += 1
                    continue
                if energy in self.available_energies:
                    if QtWidgets.QMessageBox.question(self, "确认", f"能量 {energy:.2f} eV 已存在，是否覆盖？") != QtWidgets.QMessageBox.StandardButton.Yes:
                        skipped_count += 1
                        continue
                files = [f for f in Path(folder).iterdir() if f.suffix == ".txt"]
                if not files:
                    QtWidgets.QMessageBox.warning(self, "提示", f"文件夹 '{folder_name}' 中没有txt文件")
                    skipped_count += 1
                    continue

                raw_data: dict[float, list] = {energy: []}

                for file_path in sorted(files):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    temp = None
                    for line in lines[:15]:
                        temp_match = re.search(r"Temperature[:\s]*([\d.]+)\s*C", line, re.IGNORECASE)
                        if not temp_match:
                            temp_match = re.search(r"Temp[:\s]*([\d.]+)", line, re.IGNORECASE)
                        if not temp_match:
                            temp_match = re.search(r"(\d+)\s*°?C", line)
                        if temp_match:
                            temp = int(float(temp_match.group(1)))
                            break
                    if temp is None:
                        temp_match = re.search(r"_(\d+)C?_", file_path.name)
                        if not temp_match:
                            temp_match = re.search(r"-(\d+)C?\.", file_path.name)
                        if not temp_match:
                            temp_match = re.search(r"(\d{2,3})C", file_path.name, re.IGNORECASE)
                        if not temp_match:
                            temp_match = re.search(r"(\d{2,3})_", file_path.name)
                        if temp_match:
                            temp = int(temp_match.group(1))
                        else:
                            try:
                                temp = int(re.search(r"(\d+)", file_path.name).group(1))
                            except Exception:
                                continue

                    io = None
                    for line in lines[:10]:
                        io_match = re.search(r"IO[:\s]*([\d.]+)", line, re.IGNORECASE)
                        if io_match:
                            io = float(io_match.group(1))
                            break
                    if io is None:
                        io = 100.0

                    data = []
                    start_line = min(10, len(lines))
                    for line in lines[start_line:]:
                        try:
                            data.append(float(line.strip()))
                        except ValueError:
                            pass
                    if len(data) < 100:
                        continue

                    raw_data[energy].append({
                        "file_path": str(file_path),
                        "filename": file_path.name,
                        "io": io,
                        "data": data,
                        "temp": temp,
                    })

                if energy not in self.temperature_scan_data:
                    self.temperature_scan_data[energy] = {}

                temp_groups: dict[int, list] = {}
                for entry in raw_data[energy]:
                    t = entry["temp"]
                    if t not in temp_groups:
                        temp_groups[t] = []
                    temp_groups[t].append(entry)

                for temp, repeats in temp_groups.items():
                    n = len(repeats)
                    max_len = max(len(r["data"]) for r in repeats)
                    avg_data = np.zeros(max_len)
                    count_arr = np.zeros(max_len)
                    for r in repeats:
                        d = np.array(r["data"])
                        avg_data[: len(d)] += d
                        count_arr[: len(d)] += 1
                    count_arr[count_arr == 0] = 1
                    avg_data = avg_data / count_arr
                    avg_io = float(np.mean([r["io"] for r in repeats]))
                    filenames = ", ".join(r["filename"] for r in repeats)

                    self.temperature_scan_data[energy][temp] = {
                        "repeats": repeats,
                        "avg_data": avg_data.tolist(),
                        "avg_io": avg_io,
                        "filenames": filenames,
                        "repeat_count": n,
                        "peaks_info": None,
                    }

                temps_list = sorted(temp_groups.keys())
                if temps_list:
                    best_temp = None
                    best_peak_count = -1
                    best_peaks_info = None
                    peak_ranges = self.peak_ranges or None
                    for t in temps_list:
                        t_data = self.temperature_scan_data[energy][t]["avg_data"]
                        t_peaks = self._detect_and_integrate_peaks(t_data, peak_ranges)
                        t_count = len(t_peaks)
                        if t_count > best_peak_count:
                            best_peak_count = t_count
                            best_temp = t
                            best_peaks_info = t_peaks
                    peaks_info = best_peaks_info if best_peaks_info is not None else []

                    for temp in self.temperature_scan_data[energy]:
                        self.temperature_scan_data[energy][temp]["peaks_info"] = peaks_info

                added_count += 1

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", f"添加能量文件夹失败: {e}")

        self.available_energies = sorted(self.temperature_scan_data.keys())
        self.combo_energy_select.clear()
        self.combo_energy_select.addItem("全部能量")
        for e in self.available_energies:
            self.combo_energy_select.addItem(f"{e:.2f} eV")
        self.lbl_energy_count.setText(f"共 {len(self.available_energies)} 个能量点")
        self._update_ts_table_with_species()
        self._refresh_ts_table()
        self._refresh_energy_parent_table()
        self.lbl_ts_folder.setText(f"已加载 {len(self.available_energies)} 个能量点")
        self.lbl_ts_folder.setStyleSheet("color: #6495ed;")

        if added_count > 0:
            QtWidgets.QMessageBox.information(self, "成功", f"成功添加 {added_count} 个能量点的数据")
        if skipped_count > 0:
            QtWidgets.QMessageBox.information(self, "提示", f"跳过了 {skipped_count} 个文件夹")

    def _clear_temperature_scan(self):
        if not self.temperature_scan_data:
            QtWidgets.QMessageBox.warning(self, "提示", "没有数据可清空")
            return
        if QtWidgets.QMessageBox.question(self, "确认", "确定要清空所有温度扫描数据吗？") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.temperature_scan_data = {}
        self.available_energies = []
        self.combo_energy_select.clear()
        self.combo_energy_select.addItem("全部能量")
        self.lbl_energy_count.setText("")
        self.ts_data_table.setRowCount(0)
        self.lbl_ts_folder.setText("未选择文件夹")
        self.lbl_ts_folder.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}
        self._refresh_energy_parent_table()
        QtWidgets.QMessageBox.information(self, "成功", "已清空所有温度扫描数据")

    def _remove_selected_energy(self):
        current_text = self.combo_energy_select.currentText()
        if current_text == "全部能量":
            QtWidgets.QMessageBox.warning(self, "提示", "请选择一个具体的能量")
            return
        try:
            energy = float(current_text.replace(" eV", ""))
            if energy not in self.temperature_scan_data:
                QtWidgets.QMessageBox.warning(self, "提示", "该能量数据不存在")
                return
            if QtWidgets.QMessageBox.question(self, "确认", f"确定要移除能量 {energy:.2f} eV 的数据吗？") != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            del self.temperature_scan_data[energy]
            self.available_energies = sorted(self.temperature_scan_data.keys())
            self.combo_energy_select.clear()
            self.combo_energy_select.addItem("全部能量")
            for e in self.available_energies:
                self.combo_energy_select.addItem(f"{e:.2f} eV")
            self.lbl_energy_count.setText(f"共 {len(self.available_energies)} 个能量点")
            self._refresh_ts_table()
            self._refresh_energy_parent_table()
            self.btn_remove_energy.setEnabled(False)
            QtWidgets.QMessageBox.information(self, "成功", f"已移除能量 {energy:.2f} eV 的数据")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "提示", "无法解析能量值")

    def _on_energy_select(self, text):
        self.btn_remove_energy.setEnabled(text != "全部能量")

    def _refresh_ts_table(self):
        self.ts_data_table.setRowCount(0)
        for energy in self.available_energies:
            if energy not in self.temperature_scan_data:
                continue
            for temp in sorted(self.temperature_scan_data[energy].keys()):
                info = self.temperature_scan_data[energy][temp]
                peaks_info = info.get("peaks_info", [])
                peak_count = len([p for p in peaks_info if not p.get("overlapped", False)])
                overlapped_count = len([p for p in peaks_info if p.get("overlapped", False)])
                peak_info_text = f"{peak_count} 个峰"
                if overlapped_count > 0:
                    peak_info_text += f" ({overlapped_count} 个重合)"

                matched = info.get("matched_species", [])
                if matched:
                    peak_info_text += f" | {', '.join(matched[:5])}"
                    if len(matched) > 5:
                        peak_info_text += f" 等{len(matched)}个"

                row = self.ts_data_table.rowCount()
                self.ts_data_table.insertRow(row)
                self.ts_data_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{energy:.2f}"))
                self.ts_data_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{temp}"))
                self.ts_data_table.setItem(row, 2, QtWidgets.QTableWidgetItem(info["filenames"]))
                self.ts_data_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{info['avg_io']:.2f}"))
                self.ts_data_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{info['repeat_count']}"))
                self.ts_data_table.setItem(row, 5, QtWidgets.QTableWidgetItem(peak_info_text))

    def _detect_and_integrate_peaks(self, data, peak_ranges=None):
        peaks_info = []
        data_np = np.array(data)
        if len(data_np) == 0 or np.max(data_np) == 0:
            return []

        if peak_ranges:
            n = len(data_np)
            for mz, bounds in sorted(peak_ranges.items()):
                start, end = bounds
                start = max(0, int(start))
                end = min(n - 1, int(end))
                if start > end:
                    continue

                segment = data_np[start : end + 1]
                if len(segment) == 0:
                    continue

                peak_idx = start + int(np.argmax(segment))
                peak_integral = float(np.sum(segment))
                mz_est = self._apply_mz_calibration(peak_idx + 1)
                peaks_info.append({
                    "index": peak_idx,
                    "mz_raw": mz_est,
                    "mz_rounded": int(round(float(mz))),
                    "left_idx": start,
                    "right_idx": end,
                    "integral": peak_integral,
                    "overlapped": False,
                })

            peaks_info.sort(key=lambda x: x["mz_rounded"])
            return peaks_info

        detection_start = max(3000, 0)
        if detection_start >= len(data_np) - 1:
            return []

        n = len(data_np)
        local_max_indices = []
        for i in range(detection_start, n - 1):
            if data_np[i] > 3 and data_np[i] >= data_np[i - 1] and data_np[i] >= data_np[i + 1]:
                local_max_indices.append(i)

        local_max_indices.sort(key=lambda x: -data_np[x])

        used_indices: set[int] = set()
        final_max_indices: list[int] = []

        for max_idx in local_max_indices:
            if max_idx in used_indices:
                continue

            peak_val = data_np[max_idx]

            has_higher_nearby = False
            start_search = max(detection_start, max_idx - 30)
            end_search = min(n, max_idx + 31)

            for j in range(start_search, end_search):
                if j != max_idx and j in final_max_indices and data_np[j] > peak_val:
                    has_higher_nearby = True
                    break

            if has_higher_nearby:
                continue

            weak_tail_range = 90 if max_idx <= 15000 else 50
            is_weak_tail = False
            for prev_idx in final_max_indices:
                if max_idx > prev_idx and max_idx - prev_idx <= weak_tail_range:
                    if peak_val * 5 <= data_np[prev_idx]:
                        is_weak_tail = True
                        break

            if is_weak_tail:
                continue

            final_max_indices.append(max_idx)
            used_indices.add(max_idx)

            for j in range(max_idx - 10, max_idx + 11):
                if detection_start <= j < n:
                    used_indices.add(j)

        final_max_indices.sort()

        for pi in final_max_indices:
            mz_est = self._apply_mz_calibration(pi + 1)
            mz_rounded = round(mz_est)

            left_idx, right_idx = self._find_peak_bounds(pi, data)

            if right_idx - left_idx > 100:
                continue

            peak_integral = sum(data[left_idx : right_idx + 1])

            peaks_info.append({
                "index": pi,
                "mz_raw": mz_est,
                "mz_rounded": mz_rounded,
                "left_idx": left_idx,
                "right_idx": right_idx,
                "integral": peak_integral,
                "overlapped": False,
            })

        peaks_info.sort(key=lambda x: x["mz_rounded"])

        mz_groups: dict[int, list] = {}
        for peak in peaks_info:
            mz = peak["mz_rounded"]
            if mz not in mz_groups:
                mz_groups[mz] = []
            mz_groups[mz].append(peak)

        for mz, peaks in mz_groups.items():
            if len(peaks) > 1:
                peaks.sort(key=lambda x: -x["integral"])
                for peak in peaks[1:]:
                    peak["overlapped"] = True

        return peaks_info

    def _find_peak_bounds(self, peak_idx, data, min_threshold=0.1):
        max_val = data[peak_idx]
        threshold = max_val * min_threshold

        left_idx = peak_idx
        while left_idx > 0:
            left_idx -= 1
            if data[left_idx] <= threshold:
                left_idx = max(0, left_idx - 3)
                break

        right_idx = peak_idx
        while right_idx < len(data) - 1:
            right_idx += 1
            if data[right_idx] <= threshold:
                right_idx = min(len(data) - 1, right_idx + 3)
                break

        return left_idx, right_idx

    def _get_signal_from_scan_data(self, mz, energy=None):
        if not self.temperature_scan_data:
            return {}
        result: dict[float, float] = {}

        if energy is not None and energy in self.temperature_scan_data:
            energies_to_check = [energy]
        else:
            energies_to_check = self.available_energies

        target_peak = None

        for e in energies_to_check:
            if e not in self.temperature_scan_data:
                continue
            temps = list(self.temperature_scan_data[e].keys())
            if not temps:
                continue
            best_temp = None
            best_count = -1
            for t in temps:
                t_peaks = self.temperature_scan_data[e][t].get("peaks_info", [])
                if len(t_peaks) > best_count:
                    best_count = len(t_peaks)
                    best_temp = t
            if best_temp is None:
                continue
            peaks_info = self.temperature_scan_data[e][best_temp].get("peaks_info", [])

            for peak in peaks_info:
                if peak["mz_rounded"] == mz and not peak.get("overlapped", False):
                    target_peak = peak
                    break
            if target_peak is not None:
                break

        if target_peak is None:
            return result

        for e in energies_to_check:
            if e not in self.temperature_scan_data:
                continue
            for temp, info in self.temperature_scan_data[e].items():
                if temp in result:
                    continue
                data = info["avg_data"]
                io = info.get("avg_io", 100.0)

                left_idx = target_peak["left_idx"]
                right_idx = target_peak["right_idx"]
                peak_integral = sum(data[left_idx : right_idx + 1])
                signal_val = peak_integral / io if io > 0 else peak_integral
                result[temp] = signal_val

        return result

    def _on_md_preset_changed(self, text):
        if text in MASS_DISCRIMINATION_PRESETS:
            self.spin_md_exponent.setValue(MASS_DISCRIMINATION_PRESETS[text])

    def _on_md_exponent_changed(self, val):
        self.settings.mass_disc_exponent = val

    def _preview_mass_discrimination(self):
        if not self.database:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载物种数据库")
            return
        seen: set[tuple] = set()
        species_list = []
        for spec in self.database:
            key = (spec.get("species", spec.get("name", "")), spec["mz"])
            if key not in seen:
                seen.add(key)
                species_list.append(spec)
        self.md_preview_table.setRowCount(0)
        for spec in species_list[:30]:
            row = self.md_preview_table.rowCount()
            self.md_preview_table.insertRow(row)
            name = spec.get("species", spec.get("name", ""))
            self.md_preview_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.md_preview_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(spec["mz"])))
            D_i = calc_mass_discrimination(spec["mz"], self.spin_md_exponent.value())
            self.md_preview_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{D_i:.6f}"))

    def _on_kr_source_changed(self, text):
        self.btn_load_kr.setEnabled(text == "从文件加载")
        self.btn_extract_kr.setEnabled(text == "从高能量温度扫描提取")
        if text == "使用默认数据":
            self.kr_data = dict(self.settings.kr_data)
            self._fill_kr_table()

    def _fill_kr_table(self):
        self.kr_table.setRowCount(0)
        for temp in sorted(self.kr_data.keys()):
            row = self.kr_table.rowCount()
            self.kr_table.insertRow(row)
            self.kr_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
            val = self.kr_data[temp]
            if isinstance(val, dict):
                self.kr_table.setItem(row, 1, QtWidgets.QTableWidgetItem(val.get("filename", "")))
                self.kr_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{val.get('signal', 0):.6f}"))
            else:
                self.kr_table.setItem(row, 1, QtWidgets.QTableWidgetItem("-"))
                self.kr_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{val:.6f}"))
            self.kr_table.setItem(row, 3, QtWidgets.QTableWidgetItem("-"))

    def _load_kr_data(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载Kr数据", "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            self.kr_data = {}
            for _, row in df.iterrows():
                temp = float(row.iloc[0])
                signal = float(row.iloc[1])
                self.kr_data[int(temp)] = signal
            self._fill_kr_table()
            QtWidgets.QMessageBox.information(self, "成功", f"加载了 {len(self.kr_data)} 个Kr数据点")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载Kr数据失败: {e}")

    def _extract_kr_from_scan(self):
        if not self.temperature_scan_data:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在数据加载选项卡中加载温度扫描数据")
            return
        self.set_busy(True, "正在从温度扫描提取Kr信号...")
        try:
            kr_mz = self.spin_kr_mz.value()
            self.kr_data = {}
            target_peak = None
            target_energy = None

            energies_sorted = sorted(self.available_energies, reverse=True)

            for energy in energies_sorted:
                if energy not in self.temperature_scan_data:
                    continue
                temps = list(self.temperature_scan_data[energy].keys())
                if not temps:
                    continue
                best_temp = None
                best_count = -1
                for t in temps:
                    t_peaks = self.temperature_scan_data[energy][t].get("peaks_info", [])
                    if len(t_peaks) > best_count:
                        best_count = len(t_peaks)
                        best_temp = t
                if best_temp is None:
                    continue
                peaks_info = self.temperature_scan_data[energy][best_temp].get("peaks_info", [])

                for peak in peaks_info:
                    if peak["mz_rounded"] == kr_mz and not peak.get("overlapped", False):
                        target_peak = peak
                        target_energy = energy
                        break
                if target_peak is not None:
                    break

            if target_peak is None:
                QtWidgets.QMessageBox.warning(self, "提示", f"在峰最多的温度点未找到质量数 {kr_mz} 的峰")
                return

            for energy in energies_sorted:
                if energy not in self.temperature_scan_data:
                    continue
                for temp, info in self.temperature_scan_data[energy].items():
                    if temp in self.kr_data:
                        continue
                    data = info["avg_data"]
                    io = info.get("avg_io", 100.0)

                    left_idx = target_peak["left_idx"]
                    right_idx = target_peak["right_idx"]
                    peak_integral = sum(data[left_idx : right_idx + 1])
                    kr_signal = peak_integral / io if io > 0 else peak_integral
                    self.kr_data[temp] = {"filename": info.get("filenames", ""), "signal": kr_signal}

            self._fill_kr_table()
            QtWidgets.QMessageBox.information(
                self, "成功",
                f"从 {len(self.kr_data)} 个温度扫描文件中提取了Kr信号\n"
                f"质量数: {kr_mz}, 积分范围: {target_peak['left_idx']} - {target_peak['right_idx']}\n"
                f"参考能量: {target_energy:.2f} eV"
            )
        finally:
            self.set_busy(False, "就绪")

    def _calc_expansion_coefficients(self):
        self.set_busy(True, "正在计算膨胀系数...")
        try:
            kr_signal_data: dict[float, float] = {}
            for temp, data in self.kr_data.items():
                if isinstance(data, dict):
                    kr_signal_data[temp] = data.get("signal", 0)
                else:
                    kr_signal_data[temp] = data
            self.expansion_coefficients = calc_expansion_coefficients(kr_signal_data)
            for row in range(self.kr_table.rowCount()):
                temp_item = self.kr_table.item(row, 0)
                if temp_item:
                    temp = float(temp_item.text())
                    if temp in self.expansion_coefficients:
                        self.kr_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{self.expansion_coefficients[temp]:.6f}"))
            QtWidgets.QMessageBox.information(self, "成功", f"计算了 {len(self.expansion_coefficients)} 个温度点的膨胀系数")
        finally:
            self.set_busy(False, "就绪")

    def _species_record_matches(self, record: dict, mz: int, species_name: str | None = None) -> bool:
        if record.get("mz") != mz:
            return False
        if not species_name:
            return True
        return species_name in {record.get("species"), record.get("name")}

    def _find_species_record(self, mz: int, species_name: str | None = None) -> dict | None:
        for record in self.database:
            if self._species_record_matches(record, mz, species_name):
                return record
        if species_name:
            for record in self.database:
                if self._species_record_matches(record, mz):
                    return record
        return None

    def _recalculate_parent_mf_by_energy(self) -> None:
        if not self.available_energies:
            self.parent_mf_by_energy = {}
            self.parent_signal_by_energy = {}
            self.parent_config_by_energy = {}
            return

        T0 = float(self.spin_parent_t0.value())
        X0 = self.spin_parent_mf0.value()
        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}

        for energy in sorted(self.available_energies):
            cfg = self._get_parent_config_for_energy(energy)
            mz = int(cfg["mz"])
            signal_data = self._get_signal_from_scan_data(mz, energy)
            if not signal_data:
                continue
            mf = calc_parent_mole_fraction(
                signal_data,
                reference_temperature=T0,
                parent_initial_mf=X0,
                expansion_coefficients=self.expansion_coefficients,
            )
            if not mf:
                continue
            self.parent_mf_by_energy[energy] = mf
            self.parent_signal_by_energy[energy] = signal_data
            self.parent_config_by_energy[energy] = cfg

    def _reference_parent_for_energy(self, energy: float) -> tuple[int, float, float, dict[float, float], float, str | None] | None:
        if not self.parent_mf_by_energy:
            self._recalculate_parent_mf_by_energy()

        T0 = float(self.spin_parent_t0.value())
        if energy in self.parent_mf_by_energy and energy in self.parent_signal_by_energy:
            cfg = self.parent_config_by_energy.get(energy, self._get_parent_config_for_energy(energy))
            mf = self.parent_mf_by_energy[energy]
            return (
                int(cfg["mz"]),
                float(cfg["mz"]),
                energy,
                self.parent_signal_by_energy[energy],
                mf.get(T0, mf.get(max(mf.keys()), 0.0)),
                cfg.get("species_name"),
            )

        if self.parent_mf_results:
            parent_energy = self.spin_parent_energy.value()
            parent_mz = self.spin_parent_mz.value()
            parent_signal = self._get_signal_from_scan_data(parent_mz, parent_energy)
            return (
                parent_mz,
                float(parent_mz),
                parent_energy,
                parent_signal,
                self.parent_mf_results.get(T0, self.parent_mf_results.get(max(self.parent_mf_results.keys()), 0.0)),
                self._selected_parent_species_name(),
            )

        return None

    def _calc_parent_mole_fraction(self):
        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return
        self.set_busy(True, "正在计算母体摩尔分数...")
        try:
            mz = self.spin_parent_mz.value()
            parent_species_name = self._selected_parent_species_name()
            T0 = self.spin_parent_t0.value()
            X0 = self.spin_parent_mf0.value()
            energy = self.spin_parent_energy.value()

            if energy not in self.temperature_scan_data:
                QtWidgets.QMessageBox.warning(
                    self, "提示",
                    f"未找到能量 {energy:.2f} eV 的温度扫描数据\n"
                    f"已加载的能量: {', '.join(f'{e:.2f} eV' for e in self.available_energies)}"
                )
                return

            signal_data = self._get_signal_from_scan_data(mz, energy)
            if not signal_data:
                QtWidgets.QMessageBox.warning(self, "提示", f"在能量 {energy:.2f} eV 下未找到质量数 {mz} 的峰")
                return

            self.parent_mf_results = calc_parent_mole_fraction(
                signal_data,
                reference_temperature=float(T0),
                parent_initial_mf=X0,
                expansion_coefficients=self.expansion_coefficients,
            )

            self.parent_result_table.setRowCount(0)
            for temp in sorted(self.parent_mf_results.keys()):
                row = self.parent_result_table.rowCount()
                self.parent_result_table.insertRow(row)
                self.parent_result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
                self.parent_result_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{signal_data.get(temp, 0):.6f}"))
                self.parent_result_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{get_expansion_coefficient(temp, self.expansion_coefficients):.6f}"))
                self.parent_result_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{self.parent_mf_results[temp]:.6f}"))

            self._recalculate_parent_mf_by_energy()
            if energy not in self.parent_mf_by_energy:
                self.parent_mf_by_energy[energy] = self.parent_mf_results
                self.parent_signal_by_energy[energy] = signal_data
                self.parent_config_by_energy[energy] = {
                    "mz": mz,
                    "species_name": parent_species_name,
                }
            self._refresh_results_view()

            extra_msg = f"\n已准备 {len(self.parent_mf_by_energy)} 个能量的参考母体缓存"
            QtWidgets.QMessageBox.information(self, "成功", f"计算了 {len(self.parent_mf_results)} 个温度点的母体摩尔分数{extra_msg}")
        finally:
            self.set_busy(False, "就绪")

    def _calculate_auto_mf(self):
        if not self.pie_species_data:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载PIE鉴定结果")
            return

        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return

        if not self.parent_mf_results:
            QtWidgets.QMessageBox.warning(self, "提示", "请先计算母体摩尔分数")
            return

        self.set_busy(True, "正在自动计算所有物种的摩尔分数...")
        try:
            self.txt_warnings.clear()
            warnings: list[str] = []

            species_by_mz: dict[int, list[dict]] = {}
            for d in self.pie_species_data:
                mz = d["mz"]
                if mz not in species_by_mz:
                    species_by_mz[mz] = []
                species_by_mz[mz].append(d)

            all_energies = sorted(self.available_energies)

            self.all_species_mf = {}
            self._recalculate_parent_mf_by_energy()

            parent_mz = self.spin_parent_mz.value()
            parent_mw = float(parent_mz)
            parent_energy = self.spin_parent_energy.value()
            parent_signal = self._get_signal_from_scan_data(parent_mz, parent_energy)
            parent_mf_at_tm = self.parent_mf_results.get(float(self.spin_parent_t0.value()), 0)
            parent_species_name = self._selected_parent_species_name()

            parent_mz_values = set()
            for energy, mf in self.parent_mf_by_energy.items():
                cfg = self.parent_config_by_energy.get(energy, self._get_parent_config_for_energy(energy))
                cfg_mz = int(cfg["mz"])
                parent_mz_values.add(cfg_mz)
                label = cfg.get("species_name") or "母体"
                self.all_species_mf[(cfg_mz, label, energy)] = mf

            if not parent_mz_values:
                parent_mz_values.add(parent_mz)
                self.all_species_mf[(parent_mz, parent_species_name or "母体", parent_energy)] = self.parent_mf_results

            for mz, species_list in species_by_mz.items():
                if mz in parent_mz_values:
                    continue

                if len(species_list) == 1:
                    species = species_list[0]
                    ie = species.get("ie", 0) or 0
                    usable_energies = [e for e in all_energies if e >= ie]
                    calc_energies = usable_energies[:2]

                    if not calc_energies:
                        warnings.append(f"质量数 {mz} 物种 {species['species']}: 没有高于电离能({ie:.2f} eV)的能量数据")
                        continue

                    for calc_energy in calc_energies:
                        ref = self._reference_parent_for_energy(calc_energy)
                        if ref is None:
                            warnings.append(f"质量数 {mz} 物种 {species['species']}: 缺少 {calc_energy:.2f} eV 的参考母体")
                            continue
                        ref_mz, ref_mw, ref_energy, ref_signal, ref_mf_at_tm, ref_species_name = ref
                        signal_data = self._get_signal_from_scan_data(mz, calc_energy)
                        if not signal_data:
                            continue
                        mf = self._calc_product_mf_auto(
                            mz, species, ref_mz, ref_mw,
                            ref_energy, ref_signal, ref_mf_at_tm,
                            signal_data, calc_energy=calc_energy,
                            ref_species_name=ref_species_name,
                        )
                        if mf:
                            self.all_species_mf[(mz, species["species"], calc_energy)] = mf
                else:
                    species_list_sorted = sorted(
                        species_list,
                        key=lambda x: x.get("ie") if x.get("ie") is not None else float("inf"),
                    )
                    result = self._calc_multi_species_mf_auto(
                        mz, species_list_sorted, all_energies,
                        parent_mz, parent_mw, parent_energy,
                        parent_signal, parent_mf_at_tm,
                        ref_provider=self._reference_parent_for_energy,
                    )
                    if result:
                        for key, mf in result.items():
                            if key != "warning":
                                self.all_species_mf[key] = mf
                        if "warning" in result:
                            warnings.append(result["warning"])

            self._update_auto_mf_table()

            if warnings:
                self.txt_warnings.setText("\n".join(warnings))

            parent_count, product_count = self._auto_result_counts()
            self.lbl_auto_status.setText(
                f"已计算 {len(self.all_species_mf)} 条结果（产物 {product_count}，母体参考 {parent_count}）"
            )
            self.lbl_auto_status.setStyleSheet("color: #6495ed;")

            self._plot_all_auto_mf()
            self._refresh_results_view()

            QtWidgets.QMessageBox.information(self, "完成", f"自动计算完成！\n共计算 {len(self.all_species_mf)} 条摩尔分数结果")
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self, "错误", f"计算摩尔分数时出错: {e}\n\n{traceback.format_exc()}")
        finally:
            self.set_busy(False, "就绪")

    def _calc_product_mf_auto(self, mz, species, ref_mz, ref_mw,
                               ref_energy, ref_signal_data, ref_mf_at_tm,
                               signal_data, calc_energy=None,
                               ref_species_name=None):
        ie = species.get("ie", 0) or 0
        species_name = species["species"]

        if calc_energy is None:
            usable_energies = [energy for energy in self.available_energies if energy >= ie]
            best_energy = select_calc_energy(ie, usable_energies) if usable_energies else None
        else:
            best_energy = calc_energy

        if best_energy is None:
            return None

        if not signal_data:
            return None

        species_obj = self._find_species_record(mz, species_name)

        ref_obj = self._find_species_record(ref_mz, ref_species_name)

        if species_obj is None or ref_obj is None:
            return None

        energies_arr = species_obj.get("energies")
        cross_sections_arr = species_obj.get("cross_sections")
        if energies_arr is not None and cross_sections_arr is not None:
            sigma_i = _interpolate_cross_section(energies_arr, cross_sections_arr, best_energy)
        else:
            sigma_i = 0.0

        ref_species_name = ref_species_name or species.get("ref_species") or species.get("ref_name")
        if ref_species_name:
            ref_obj = self._find_species_record(ref_mz, ref_species_name) or ref_obj

        ref_energies = ref_obj.get("energies")
        ref_cross = ref_obj.get("cross_sections")
        if ref_energies is not None and ref_cross is not None:
            sigma_A = _interpolate_cross_section(ref_energies, ref_cross, ref_energy)
        else:
            sigma_A = 0.0

        if sigma_i == 0 or sigma_A == 0:
            return None

        D_i = calc_mass_discrimination(mz, self.settings.mass_disc_exponent)
        D_A = calc_mass_discrimination(ref_mw, self.settings.mass_disc_exponent)

        T_M = float(self.spin_parent_t0.value())
        if T_M not in ref_signal_data:
            T_M = max(ref_signal_data.keys()) if ref_signal_data else None
        if T_M is None:
            return None

        S_A_TM = ref_signal_data.get(T_M, 0)
        if S_A_TM == 0:
            return None

        lambda_TM = get_expansion_coefficient(T_M, self.expansion_coefficients)

        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            lambda_T = get_expansion_coefficient(T, self.expansion_coefficients)
            X_i = (
                ref_mf_at_tm
                * (S_i / S_A_TM)
                * (sigma_A / sigma_i)
                * (D_A / D_i)
                * (lambda_TM / lambda_T)
            )
            results[T] = max(0.0, X_i)

        return results

    def _calc_multi_species_mf_auto(self, mz, species_list, energies,
                                     ref_mz, ref_mw, ref_energy,
                                     ref_signal_data, ref_mf_at_tm,
                                     ref_provider=None):
        result: dict = {}
        warnings: list[str] = []

        if not energies:
            warnings.append("没有可用能量数据")
            result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
            return result

        sorted_energies = sorted(energies)
        energy_boundaries = [0] + sorted_energies

        species_groups: dict[int, list[dict]] = {}
        for s in species_list:
            ie = s.get("ie") if s.get("ie") is not None else float("inf")
            group_idx = None
            for i in range(len(energy_boundaries) - 1):
                lower = energy_boundaries[i]
                upper = energy_boundaries[i + 1]
                if lower < ie <= upper:
                    group_idx = i
                    break
            if group_idx is None:
                group_idx = len(energy_boundaries) - 1

            if group_idx not in species_groups:
                species_groups[group_idx] = []
            species_groups[group_idx].append(s)

        resolvable_species: list[dict] = []
        for group_idx, group in species_groups.items():
            if len(group) > 1:
                group.sort(key=lambda x: x.get("contribution", 0) or 0, reverse=True)
                best = group[0]
                resolvable_species.append(best)
                ignored_names = [s["species"] for s in group[1:]]
                upper_bound = (
                    f"{energy_boundaries[group_idx + 1]:.1f}"
                    if group_idx + 1 < len(energy_boundaries)
                    else "∞"
                )
                warnings.append(
                    f"电离能区间 ({energy_boundaries[group_idx]:.1f}-{upper_bound} eV) "
                    f"内存在 {len(group)} 个物种 ({', '.join(s['species'] for s in group)})，"
                    f"无法通过能量扫描分离，仅计算匹配系数最高的 {best['species']}，"
                    f"忽略了 {', '.join(ignored_names)}"
                )
            else:
                resolvable_species.append(group[0])

        resolvable_species.sort(key=lambda x: x.get("ie") if x.get("ie") is not None else float("inf"))

        if len(resolvable_species) <= 1:
            for species in resolvable_species:
                ie = species.get("ie") if species.get("ie") is not None else float("inf")
                if ie == float("inf"):
                    continue
                usable_energies = [e for e in sorted_energies if e >= ie]
                if not usable_energies:
                    continue
                calc_e = select_calc_energy(float(ie), usable_energies)
                if ref_provider is not None:
                    ref = ref_provider(calc_e)
                    if ref is not None:
                        ref_mz, ref_mw, ref_energy, ref_signal_data, ref_mf_at_tm, ref_species_name = ref
                    else:
                        continue
                else:
                    ref_species_name = None
                signal_data = self._get_signal_from_scan_data(mz, calc_e)
                if not signal_data:
                    continue
                mf = self._calc_product_mf_from_signal(
                    mz, species, calc_e, signal_data,
                    ref_mz, ref_mw, ref_energy,
                    ref_signal_data, ref_mf_at_tm,
                    ref_species_name=ref_species_name,
                )
                if mf:
                    result[(mz, species["species"], calc_e)] = mf
            if warnings:
                result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
            return result

        try:
            energy_scan_data: dict[float, dict[float, float]] = {}
            for energy in sorted_energies:
                signal_data = self._get_signal_from_scan_data(mz, energy)
                if signal_data:
                    energy_scan_data[energy] = signal_data

            if not energy_scan_data:
                warnings.append("没有可用的温度扫描信号")
                result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
                return result

            separated, pure_energies = separate_coexisting_species_signals(
                mz=mz,
                species_at_mz=[
                    {"species": s["species"], "ie": s.get("ie")}
                    for s in resolvable_species
                ],
                energy_scan_data=energy_scan_data,
                database=self.database,
                mz_index=self.mz_index,
            )

            for species in resolvable_species:
                species_name = species["species"]
                if species_name not in separated:
                    continue
                calc_e = pure_energies.get(species_name)
                if calc_e is None:
                    ie = species.get("ie") if species.get("ie") is not None else float("inf")
                    usable_energies = [e for e in sorted_energies if e >= ie]
                    if not usable_energies:
                        continue
                    calc_e = select_calc_energy(float(ie), usable_energies)
                if ref_provider is not None:
                    ref = ref_provider(calc_e)
                    if ref is not None:
                        ref_mz, ref_mw, ref_energy, ref_signal_data, ref_mf_at_tm, ref_species_name = ref
                    else:
                        continue
                else:
                    ref_species_name = None
                mf = self._calc_product_mf_from_signal(
                    mz, species, calc_e, separated[species_name],
                    ref_mz, ref_mw, ref_energy,
                    ref_signal_data, ref_mf_at_tm,
                    ref_species_name=ref_species_name,
                )
                if mf:
                    result[(mz, species_name, calc_e)] = mf
        except Exception as exc:
            warnings.append(f"多物种信号分离失败，已回退到单能量计算 ({exc})")
            for species in resolvable_species:
                ie = species.get("ie") if species.get("ie") is not None else float("inf")
                if ie == float("inf"):
                    continue
                usable_energies = [e for e in sorted_energies if e >= ie]
                if not usable_energies:
                    continue
                calc_e = select_calc_energy(float(ie), usable_energies)
                if ref_provider is not None:
                    ref = ref_provider(calc_e)
                    if ref is not None:
                        ref_mz, ref_mw, ref_energy, ref_signal_data, ref_mf_at_tm, ref_species_name = ref
                    else:
                        continue
                else:
                    ref_species_name = None
                signal_data = self._get_signal_from_scan_data(mz, calc_e)
                if not signal_data:
                    continue
                mf = self._calc_product_mf_from_signal(
                    mz, species, calc_e, signal_data,
                    ref_mz, ref_mw, ref_energy,
                    ref_signal_data, ref_mf_at_tm,
                    ref_species_name=ref_species_name,
                )
                if mf:
                    result[(mz, species["species"], calc_e)] = mf

        unresolved_count = len(species_list) - len(
            set(k[1] for k in result if k != "warning")
        )
        ambiguous_count = sum(len(g) - 1 for g in species_groups.values() if len(g) > 1)
        if unresolved_count > ambiguous_count:
            warnings.append("部分可分辨物种未能计算摩尔分数")

        if warnings:
            result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"

        return result

    def _calc_product_mf_from_signal(self, mz, species, energy, signal_data,
                                      ref_mz, ref_mw, ref_energy,
                                      ref_signal_data, ref_mf_at_tm,
                                      ref_species_name=None):
        species_name = species["species"]

        species_obj = self._find_species_record(mz, species_name)
        ref_obj = self._find_species_record(ref_mz, ref_species_name)

        if species_obj is None or ref_obj is None:
            return None

        energies_arr = species_obj.get("energies")
        cross_sections_arr = species_obj.get("cross_sections")
        if energies_arr is not None and cross_sections_arr is not None:
            sigma_i = _interpolate_cross_section(energies_arr, cross_sections_arr, energy)
        else:
            sigma_i = 0.0

        ref_energies = ref_obj.get("energies")
        ref_cross = ref_obj.get("cross_sections")
        if ref_energies is not None and ref_cross is not None:
            sigma_A = _interpolate_cross_section(ref_energies, ref_cross, ref_energy)
        else:
            sigma_A = 0.0

        if sigma_i == 0 or sigma_A == 0:
            return None

        D_i = calc_mass_discrimination(mz, self.settings.mass_disc_exponent)
        D_A = calc_mass_discrimination(ref_mw, self.settings.mass_disc_exponent)

        T_M = float(self.spin_parent_t0.value())
        if T_M not in ref_signal_data:
            T_M = max(ref_signal_data.keys()) if ref_signal_data else None
        if T_M is None:
            return None

        S_A_TM = ref_signal_data.get(T_M, 0)
        if S_A_TM == 0:
            return None

        lambda_TM = get_expansion_coefficient(T_M, self.expansion_coefficients)

        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            if S_i <= 0:
                results[T] = 0.0
                continue
            lambda_T = get_expansion_coefficient(T, self.expansion_coefficients)
            X_i = (
                ref_mf_at_tm
                * (S_i / S_A_TM)
                * (sigma_A / sigma_i)
                * (D_A / D_i)
                * (lambda_TM / lambda_T)
            )
            results[T] = max(0.0, X_i)

        return results

    def _is_parent_auto_result_key(self, key: tuple[int, str, float]) -> bool:
        mz, species, energy = key
        cfg = self.parent_config_by_energy.get(energy)
        if cfg:
            cfg_label = cfg.get("species_name") or "母体"
            return int(cfg.get("mz", -1)) == mz and cfg_label == species

        parent_energy = self.spin_parent_energy.value() if hasattr(self, "spin_parent_energy") else None
        parent_mz = self.spin_parent_mz.value() if hasattr(self, "spin_parent_mz") else None
        parent_label = self._selected_parent_species_name() or "母体"
        return parent_energy == energy and parent_mz == mz and species == parent_label

    def _auto_result_counts(self) -> tuple[int, int]:
        parent_count = sum(1 for key in self.all_species_mf if self._is_parent_auto_result_key(key))
        return parent_count, max(0, len(self.all_species_mf) - parent_count)

    def _selected_auto_mf_key(self) -> tuple[int, str, float] | None:
        selected = self.auto_mf_table.selectedItems()
        if not selected:
            return None
        row = selected[0].row()
        mz_item = self.auto_mf_table.item(row, 1)
        species_item = self.auto_mf_table.item(row, 2)
        energy_item = self.auto_mf_table.item(row, 4)
        if not mz_item or not species_item or not energy_item:
            return None
        try:
            return (int(mz_item.text()), species_item.text(), float(energy_item.text()))
        except ValueError:
            return None

    def _auto_plot_keys_for_scope(self) -> list[tuple[int, str, float]]:
        scope = "products"
        if hasattr(self, "combo_auto_plot_scope"):
            scope = self.combo_auto_plot_scope.currentData() or "products"

        keys = list(self.all_species_mf.keys())
        if scope == "selected":
            selected = self._selected_auto_mf_key()
            return [selected] if selected in self.all_species_mf else []
        if scope == "parents":
            return [key for key in keys if self._is_parent_auto_result_key(key)]
        if scope == "products":
            product_keys = [key for key in keys if not self._is_parent_auto_result_key(key)]
            return product_keys or keys
        return keys

    def _update_auto_mf_table(self):
        self.auto_mf_table.setRowCount(0)
        sorted_items = sorted(
            self.all_species_mf.items(),
            key=lambda item: (
                self._is_parent_auto_result_key(item[0]),
                item[0][0],
                item[0][1],
                item[0][2],
            ),
        )
        for (mz, species, energy), mf in sorted_items:
            row = self.auto_mf_table.rowCount()
            self.auto_mf_table.insertRow(row)
            result_type = "母体参考" if self._is_parent_auto_result_key((mz, species, energy)) else "产物"
            self.auto_mf_table.setItem(row, 0, QtWidgets.QTableWidgetItem(result_type))
            self.auto_mf_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(mz)))
            self.auto_mf_table.setItem(row, 2, QtWidgets.QTableWidgetItem(species))

            ie = None
            for d in self.pie_species_data:
                if d["mz"] == mz and d["species"] == species:
                    ie = d.get("ie")
                    break
            self.auto_mf_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{ie:.2f}" if ie else "N/A"))
            self.auto_mf_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{energy:.2f}"))
            self.auto_mf_table.setItem(row, 5, QtWidgets.QTableWidgetItem("已计算"))

    def _on_auto_mf_selection_changed(self):
        key = self._selected_auto_mf_key()
        if key in self.all_species_mf:
            if hasattr(self, "combo_auto_plot_scope"):
                self.combo_auto_plot_scope.blockSignals(True)
                self._set_combo_current_data(self.combo_auto_plot_scope, "selected")
                self.combo_auto_plot_scope.blockSignals(False)
            self._plot_single_auto_mf(key)

    def _plot_all_auto_mf(self):
        if pg is None or self.auto_mf_plot_widget is None:
            return
        if not self.all_species_mf:
            return

        self.auto_mf_plot_widget.clear()
        self.auto_mf_plot_widget.addLegend()
        self.auto_mf_plot_widget.setTitle("")

        plot_keys = self._auto_plot_keys_for_scope()
        if not plot_keys:
            self.auto_mf_plot_widget.setTitle("没有可显示的曲线")
            return

        colors = [
            (100, 200, 255), (255, 150, 100), (100, 255, 150),
            (255, 100, 100), (255, 255, 100), (200, 150, 255),
            (255, 180, 220), (150, 255, 255), (255, 200, 150),
            (180, 255, 180),
        ]

        color_idx = 0
        species_energies: dict[str, list] = {}
        for mz, species, energy in plot_keys:
            if species not in species_energies:
                species_energies[species] = []
            species_energies[species].append((mz, energy))

        all_temps: list[float] = []
        all_values: list[float] = []
        for species, entries in species_energies.items():
            entries.sort(key=lambda x: x[1])
            for mz, energy in entries:
                key = (mz, species, energy)
                mf = self.all_species_mf[key]
                if not mf:
                    continue

                color = colors[color_idx % len(colors)]
                color_idx += 1

                sorted_temps = sorted(mf.keys())
                temps_arr = np.array(sorted_temps)
                mfs_arr = np.array([mf[t] for t in sorted_temps])
                all_temps.extend(float(t) for t in sorted_temps)
                all_values.extend(float(v) for v in mfs_arr)

                label = f"{species} ({energy:.1f}eV)"
                self.auto_mf_plot_widget.plot(
                    temps_arr, mfs_arr,
                    pen=pg.mkPen(color, width=2),
                    symbol="o", symbolSize=5, symbolBrush=color,
                    name=label,
                )
        if all_temps:
            self.auto_mf_plot_widget.setXRange(min(all_temps) - 50, max(all_temps) + 50)
        if all_values:
            min_mf = min(all_values)
            max_mf = max(all_values)
            padding = (max_mf - min_mf) * 0.1 if max_mf > min_mf else max(max_mf * 0.1, 1e-9)
            self.auto_mf_plot_widget.setYRange(max(0, min_mf - padding), max_mf + padding)

    def _plot_single_auto_mf(self, key):
        if pg is None or self.auto_mf_plot_widget is None:
            return
        if key not in self.all_species_mf:
            return
        mf = self.all_species_mf[key]
        if not mf:
            return

        mz, species, energy = key

        self.auto_mf_plot_widget.clear()
        self.auto_mf_plot_widget.addLegend()

        sorted_temps = sorted(mf.keys())
        temps_arr = np.array(sorted_temps)
        mfs_arr = np.array([mf[t] for t in sorted_temps])

        color = (100, 200, 255)
        label = f"m/z={mz} {species} @ {energy:.2f} eV"
        self.auto_mf_plot_widget.plot(
            temps_arr, mfs_arr,
            pen=pg.mkPen(color, width=3),
            symbol="o", symbolSize=6, symbolBrush=color,
            name=label,
        )

    def _collect_all_results(self) -> dict[str, dict[float, float]]:
        return {series["label"]: series["values"] for series in self._collect_result_series()}

    def _collect_result_series(self) -> list[dict[str, object]]:
        series: list[dict[str, object]] = []
        seen_labels: set[str] = set()

        def add_series(
            result_type: str,
            label: str,
            values: dict[float, float],
            mz: int | None = None,
            species: str | None = None,
            energy: float | None = None,
        ) -> None:
            if not values or label in seen_labels:
                return
            seen_labels.add(label)
            series.append(
                {
                    "type": result_type,
                    "label": label,
                    "values": values,
                    "mz": mz,
                    "species": species or label,
                    "energy": energy,
                }
            )

        if self.parent_mf_by_energy:
            for energy, mf in sorted(self.parent_mf_by_energy.items()):
                cfg = self.parent_config_by_energy.get(energy, self._get_parent_config_for_energy(energy))
                mz = int(cfg["mz"])
                species_name = cfg.get("species_name") or "母体"
                add_series(
                    "母体",
                    self._parent_result_label(mz, cfg.get("species_name"), energy),
                    mf,
                    mz=mz,
                    species=species_name,
                    energy=energy,
                )
        elif self.parent_mf_results:
            mz = self.spin_parent_mz.value()
            species_name = self._selected_parent_species_name() or "母体"
            add_series(
                "母体",
                self._parent_result_label(mz, self._selected_parent_species_name()),
                self.parent_mf_results,
                mz=mz,
                species=species_name,
            )

        for label, values in self.product_mf_results.items():
            add_series("产物", label, values, species=label)

        for mz, species_data in self.isomeric_results.items():
            for species_name, mf_data in species_data.items():
                add_series(
                    "同分异构",
                    f"{species_name}(m/z={mz})",
                    mf_data,
                    mz=mz,
                    species=species_name,
                )

        if hasattr(self, "all_species_mf") and self.all_species_mf:
            for (mz, species, energy), mf in self.all_species_mf.items():
                name = f"{species}(m/z={mz},E={energy:.2f}eV)"
                result_type = "母体" if self._is_parent_auto_result_key((mz, species, energy)) else "产物"
                add_series(result_type, name, mf, mz=mz, species=species, energy=energy)

        return series

    def _refresh_results_view(self):
        self._update_results_table()
        self._plot_results_mf()

    def _update_results_table(self):
        result_series = self._collect_result_series()
        if not result_series:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            if hasattr(self, "lbl_results_status"):
                self.lbl_results_status.setText("暂无结果")
            return

        all_temps = sorted(set().union(*(series["values"].keys() for series in result_series)))
        self.results_table.setRowCount(sum(len(series["values"]) for series in result_series))
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["类型", "m/z", "物种/结果", "光子能量(eV)", "温度(°C)", "摩尔分数"])
        self.results_table.verticalHeader().setVisible(False)
        row = 0
        for series in result_series:
            values = series["values"]
            for temp in sorted(values.keys()):
                row_values = [
                    str(series["type"]),
                    "" if series["mz"] is None else str(series["mz"]),
                    str(series["species"]),
                    "" if series["energy"] is None else f"{float(series['energy']):.2f}",
                    f"{float(temp):.1f}",
                    f"{float(values[temp]):.6f}",
                ]
                for col, value in enumerate(row_values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setToolTip(str(series["label"]))
                    self.results_table.setItem(row, col, item)
                row += 1

        header = self.results_table.horizontalHeader()
        for col in range(self.results_table.columnCount()):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col, width in enumerate((70, 70, 280, 110, 90, 120)):
            self.results_table.setColumnWidth(col, width)
        if hasattr(self, "lbl_results_status"):
            self.lbl_results_status.setText(f"已显示 {len(result_series)} 条结果，{len(all_temps)} 个温度点")

    def _plot_results_mf(self):
        if pg is None or self.mf_plot_widget is None:
            return

        result_series = self._collect_result_series()
        if hasattr(self, "mf_series_list"):
            self.mf_series_list.clear()
        if not result_series:
            self.mf_plot_widget.clear()
            return

        self.mf_plot_widget.clear()

        colors = [
            (255, 100, 100), (100, 180, 255), (255, 200, 50),
            (180, 100, 255), (100, 255, 180), (255, 140, 50),
            (200, 255, 100), (255, 100, 200), (100, 220, 220),
            (220, 180, 140), (140, 100, 220), (220, 220, 100),
        ]

        all_temps = []
        all_mfs = []
        for idx, series in enumerate(result_series):
            color = colors[idx % len(colors)]
            name = str(series["label"])
            res = series["values"]
            sorted_temps = sorted(res.keys())
            all_temps.extend(sorted_temps)
            temps_arr = np.array(sorted_temps)
            mfs_arr = np.array([res[t] for t in sorted_temps])
            all_mfs.extend(mfs_arr.tolist())
            self.mf_plot_widget.plot(
                temps_arr, mfs_arr,
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=4, symbolBrush=color,
            )
            if hasattr(self, "mf_series_list"):
                pixmap = QtGui.QPixmap(12, 12)
                pixmap.fill(QtGui.QColor(*color))
                item = QtWidgets.QListWidgetItem(QtGui.QIcon(pixmap), name)
                item.setToolTip(name)
                self.mf_series_list.addItem(item)

        if all_temps:
            self.mf_plot_widget.setXRange(min(all_temps) - 50, max(all_temps) + 50)
        if all_mfs:
            min_mf = min(all_mfs)
            max_mf = max(all_mfs)
            padding = (max_mf - min_mf) * 0.1 if max_mf > min_mf else 0.1
            self.mf_plot_widget.setYRange(max(0, min_mf - padding), max_mf + padding)

    def _export_results(self):
        all_results = self._collect_all_results()
        if not all_results:
            QtWidgets.QMessageBox.warning(self, "提示", "没有计算结果可导出")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出摩尔分数结果", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not file_path:
            return
        try:
            all_temps = sorted(set().union(*(r.keys() for r in all_results.values())))
            data = {"温度(°C)": all_temps}
            for name, res in all_results.items():
                data[name] = [res.get(t, 0) for t in all_temps]
            df = pd.DataFrame(data)
            if file_path.endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
            QtWidgets.QMessageBox.information(self, "成功", "摩尔分数结果导出成功！")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def set_temperature_scan_df(self, df: pd.DataFrame):
        self._temperature_scan_df = df

    def load_species_database(self, path: str | Path | None = None):
        if path is None:
            path = species_database_path()
        try:
            self.database, self.mz_index = load_species_database(path)
            self.lbl_db_status.setText(f"已加载 {len(self.database)} 个物种")
            self.lbl_db_status.setStyleSheet("color: #6495ed;")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载数据库失败: {e}")
