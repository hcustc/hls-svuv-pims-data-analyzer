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
        self.product_mf_results: dict[str, dict[float, float]] = {}
        self.isomeric_results: dict[int, dict[str, dict[float, float]]] = {}
        self.temperature_scan_data: dict[float, dict] = {}
        self.pie_species_data: list[dict] = []
        self.available_energies: list[float] = []
        self.all_species_mf: dict[tuple, dict[float, float]] = {}
        self.kr_data: dict[float, float] = dict(self.settings.kr_data)
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
        summary_layout.addStretch()
        layout.addWidget(self.summary_bar)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_data_tab(), "1. 数据加载")
        self.tabs.addTab(self._create_params_tab(), "2. 参数设置")
        self.tabs.addTab(self._create_parent_tab(), "3. 母体摩尔分数")
        self.tabs.addTab(self._create_auto_mf_tab(), "4. 自动计算摩尔分数")
        self.tabs.addTab(self._create_results_tab(), "5. 结果汇总")
        layout.addWidget(self.tabs)

        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("ProjectStatus")
        layout.addWidget(self.status_label)

    def _auto_load_database(self):
        try:
            db_path = species_database_path()
            if db_path.exists():
                self.database, self.mz_index = load_species_database(str(db_path))
                self.lbl_db_status.setText(f"已加载 {len(self.database)} 个物种")
                self.lbl_db_status.setStyleSheet("color: #6495ed;")
        except Exception:
            pass

    def set_project_settings(self, ps: ProjectSettings) -> None:
        """Apply ProjectSettings defaults to MoleFractionDialog controls."""
        self.project_settings = ps
        if hasattr(self, "spin_md_exponent"):
            self.spin_md_exponent.setValue(ps.mf_mass_disc_exponent)
        if hasattr(self, "spin_kr_mz"):
            self.spin_kr_mz.setValue(ps.mf_parent_mz)
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

        db_group = QtWidgets.QGroupBox("物种数据库")
        db_layout = QtWidgets.QHBoxLayout(db_group)
        self.lbl_db_status = QtWidgets.QLabel("未加载数据库")
        self.lbl_db_status.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        btn_load_db = QtWidgets.QPushButton("加载物种数据库")
        btn_load_db.setToolTip("加载PICS物种数据库，用于获取物种的电离能、分子式等信息")
        btn_load_db.clicked.connect(self._load_database)
        db_layout.addWidget(btn_load_db)
        db_layout.addWidget(self.lbl_db_status, 1)
        layout.addWidget(db_group)

        ts_group = QtWidgets.QGroupBox("温度扫描数据（支持多能量）")
        ts_layout = QtWidgets.QVBoxLayout(ts_group)
        ts_top = QtWidgets.QHBoxLayout()
        btn_add_folder = QtWidgets.QPushButton("添加能量文件夹")
        btn_add_folder.setToolTip("添加包含温度扫描txt文件的能量文件夹")
        btn_add_folder.clicked.connect(self._add_energy_folder)
        ts_top.addWidget(btn_add_folder)
        btn_clear = QtWidgets.QPushButton("清空数据")
        btn_clear.setToolTip("清空所有已加载的温度扫描数据")
        btn_clear.clicked.connect(self._clear_temperature_scan)
        ts_top.addWidget(btn_clear)
        self.lbl_ts_folder = QtWidgets.QLabel("未选择文件夹")
        self.lbl_ts_folder.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        ts_top.addWidget(self.lbl_ts_folder, 1)
        ts_layout.addLayout(ts_top)

        energy_layout = QtWidgets.QHBoxLayout()
        energy_layout.addWidget(QtWidgets.QLabel("已加载能量 (eV):"))
        self.combo_energy_select = QtWidgets.QComboBox()
        self.combo_energy_select.addItem("全部能量")
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
        ts_layout.addLayout(energy_layout)

        self.ts_data_table = QtWidgets.QTableWidget()
        self.ts_data_table.setColumnCount(6)
        self.ts_data_table.setHorizontalHeaderLabels(["能量(eV)", "温度(°C)", "文件名", "IO(nA)", "重复次数", "检测到的质量数"])
        self.ts_data_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        ts_layout.addWidget(self.ts_data_table)
        layout.addWidget(ts_group, 1)

        pie_group = QtWidgets.QGroupBox("PIE鉴定结果")
        pie_layout = QtWidgets.QVBoxLayout(pie_group)
        pie_top = QtWidgets.QHBoxLayout()
        btn_load_pie = QtWidgets.QPushButton("加载PIE鉴定结果")
        btn_load_pie.clicked.connect(self._load_pie_results)
        pie_top.addWidget(btn_load_pie)
        self.lbl_pie_status = QtWidgets.QLabel("未加载")
        self.lbl_pie_status.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        pie_top.addWidget(self.lbl_pie_status, 1)
        pie_layout.addLayout(pie_top)

        self.pie_species_table = QtWidgets.QTableWidget()
        self.pie_species_table.setColumnCount(5)
        self.pie_species_table.setHorizontalHeaderLabels(["质量数", "物种名称", "电离能(eV)", "贡献比例(%)", "R²"])
        self.pie_species_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.pie_species_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        pie_layout.addWidget(self.pie_species_table)
        layout.addWidget(pie_group, 1)
        return widget

    def _create_params_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        calib_group = QtWidgets.QGroupBox("质量数校准（来自主程序）")
        calib_layout = QtWidgets.QGridLayout(calib_group)
        calib_layout.addWidget(QtWidgets.QLabel("校准状态:"), 0, 0)
        self.lbl_calib_status = QtWidgets.QLabel("已同步主程序校准")
        self.lbl_calib_status.setStyleSheet("color: #4ecdc4;")
        calib_layout.addWidget(self.lbl_calib_status, 0, 1)
        calib_layout.addWidget(QtWidgets.QLabel("A:"), 1, 0)
        self.txt_calib_A = QtWidgets.QLineEdit(f"{self.calibration.a:.10f}")
        self.txt_calib_A.setFixedWidth(200)
        self.txt_calib_A.setReadOnly(True)
        calib_layout.addWidget(self.txt_calib_A, 1, 1)
        calib_layout.addWidget(QtWidgets.QLabel("B:"), 2, 0)
        self.txt_calib_B = QtWidgets.QLineEdit(f"{self.calibration.b:.6f}")
        self.txt_calib_B.setFixedWidth(200)
        self.txt_calib_B.setReadOnly(True)
        calib_layout.addWidget(self.txt_calib_B, 2, 1)
        calib_layout.addWidget(QtWidgets.QLabel("C:"), 3, 0)
        self.txt_calib_C = QtWidgets.QLineEdit(f"{self.calibration.c:.4f}")
        self.txt_calib_C.setFixedWidth(200)
        self.txt_calib_C.setReadOnly(True)
        calib_layout.addWidget(self.txt_calib_C, 3, 1)
        calib_layout.addWidget(QtWidgets.QLabel("公式: m/z = A*x² + B*x + C"), 4, 0, 1, 2)
        layout.addWidget(calib_group)

        md_group = QtWidgets.QGroupBox("质量歧视因子 D_i")
        md_layout = QtWidgets.QGridLayout(md_group)
        md_layout.addWidget(QtWidgets.QLabel("实验条件:"), 0, 0)
        self.combo_md_preset = QtWidgets.QComboBox()
        self.combo_md_preset.setToolTip("选择实验条件预设值，自动设置质量歧视指数n")
        for name in MASS_DISCRIMINATION_PRESETS:
            self.combo_md_preset.addItem(name)
        self.combo_md_preset.currentTextChanged.connect(self._on_md_preset_changed)
        md_layout.addWidget(self.combo_md_preset, 0, 1)
        md_layout.addWidget(QtWidgets.QLabel("指数 n:"), 1, 0)
        self.spin_md_exponent = QtWidgets.QDoubleSpinBox()
        self.spin_md_exponent.setToolTip("质量歧视因子公式 D_i = (MW/30)^n 中的指数n，n值取决于离子源类型和质量分析器特性")
        self.spin_md_exponent.setDecimals(5)
        self.spin_md_exponent.setValue(self.settings.mass_disc_exponent)
        self.spin_md_exponent.setSingleStep(0.001)
        self.spin_md_exponent.valueChanged.connect(self._on_md_exponent_changed)
        md_layout.addWidget(self.spin_md_exponent, 1, 1)
        md_layout.addWidget(QtWidgets.QLabel("公式: D_i = (MW/30)^n"), 2, 0, 1, 2)
        self.md_preview_table = QtWidgets.QTableWidget()
        self.md_preview_table.setColumnCount(3)
        self.md_preview_table.setHorizontalHeaderLabels(["物种", "分子量", "D_i"])
        self.md_preview_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.md_preview_table.setMaximumHeight(200)
        md_layout.addWidget(self.md_preview_table, 3, 0, 1, 2)
        btn_calc_md = QtWidgets.QPushButton("预览质量歧视因子")
        btn_calc_md.clicked.connect(self._preview_mass_discrimination)
        md_layout.addWidget(btn_calc_md, 4, 0, 1, 2)
        layout.addWidget(md_group)

        ec_group = QtWidgets.QGroupBox("膨胀系数 λ(T)")
        ec_layout = QtWidgets.QVBoxLayout(ec_group)
        ec_top = QtWidgets.QHBoxLayout()
        ec_top.addWidget(QtWidgets.QLabel("Kr数据来源:"))
        self.combo_kr_source = QtWidgets.QComboBox()
        self.combo_kr_source.setToolTip("选择Kr膨胀系数λ(T)的数据来源：内置默认数据/自定义文件/从温度扫描数据提取")
        self.combo_kr_source.addItems(["使用默认数据", "从文件加载", "从高能量温度扫描提取"])
        self.combo_kr_source.currentTextChanged.connect(self._on_kr_source_changed)
        ec_top.addWidget(self.combo_kr_source)
        self.btn_load_kr = QtWidgets.QPushButton("加载Kr数据")
        self.btn_load_kr.clicked.connect(self._load_kr_data)
        self.btn_load_kr.setEnabled(False)
        self.btn_extract_kr = QtWidgets.QPushButton("从温度扫描提取Kr")
        self.btn_extract_kr.clicked.connect(self._extract_kr_from_scan)
        self.btn_extract_kr.setEnabled(False)
        ec_top.addWidget(self.btn_load_kr)
        ec_top.addWidget(self.btn_extract_kr)
        ec_top.addStretch()
        ec_layout.addLayout(ec_top)

        kr_params_layout = QtWidgets.QHBoxLayout()
        kr_params_layout.addWidget(QtWidgets.QLabel("Kr质量数 m/z:"))
        self.spin_kr_mz = QtWidgets.QSpinBox()
        self.spin_kr_mz.setRange(1, 200)
        self.spin_kr_mz.setValue(84)
        kr_params_layout.addWidget(self.spin_kr_mz)
        kr_params_layout.addStretch()
        ec_layout.addLayout(kr_params_layout)

        self.kr_table = QtWidgets.QTableWidget()
        self.kr_table.setColumnCount(4)
        self.kr_table.setHorizontalHeaderLabels(["温度(°C)", "文件名", "Kr信号积分", "λ(T)"])
        self.kr_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        ec_layout.addWidget(self.kr_table)
        btn_calc_lambda = QtWidgets.QPushButton("计算膨胀系数")
        btn_calc_lambda.clicked.connect(self._calc_expansion_coefficients)
        ec_layout.addWidget(btn_calc_lambda)
        layout.addWidget(ec_group, 1)

        self._fill_kr_table()
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
        cfg_layout.addWidget(self.spin_parent_mz, 0, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("参考温度 T₀ (°C):"), 1, 0)
        self.spin_parent_t0 = QtWidgets.QSpinBox()
        self.spin_parent_t0.setToolTip("选定一个参考温度点，用于计算母体摩尔分数的基准")
        self.spin_parent_t0.setRange(0, 2000)
        self.spin_parent_t0.setValue(int(self.settings.reference_temperature or 550))
        cfg_layout.addWidget(self.spin_parent_t0, 1, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("初始摩尔分数 X(T₀):"), 2, 0)
        self.spin_parent_mf0 = QtWidgets.QDoubleSpinBox()
        self.spin_parent_mf0.setToolTip("母体物种在参考温度T₀处的摩尔分数（已知或假设值）")
        self.spin_parent_mf0.setRange(0.0, 1.0)
        self.spin_parent_mf0.setDecimals(6)
        self.spin_parent_mf0.setValue(self.settings.parent_initial_mf)
        self.spin_parent_mf0.setSingleStep(0.0001)
        cfg_layout.addWidget(self.spin_parent_mf0, 2, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("光子能量 E (eV):"), 3, 0)
        self.spin_parent_energy = QtWidgets.QDoubleSpinBox()
        self.spin_parent_energy.setToolTip("实验使用的VUV光子能量 (eV)，用于查找物种在此能量下的光电离截面")
        self.spin_parent_energy.setRange(0.0, 30.0)
        self.spin_parent_energy.setDecimals(2)
        self.spin_parent_energy.setValue(self.settings.photon_energy)
        self.spin_parent_energy.setSingleStep(0.5)
        cfg_layout.addWidget(self.spin_parent_energy, 3, 1)
        layout.addWidget(cfg_group)

        btn_calc_parent = QtWidgets.QPushButton("开始计算")
        btn_calc_parent.clicked.connect(self._calc_parent_mole_fraction)
        layout.addWidget(btn_calc_parent)

        self.parent_result_table = QtWidgets.QTableWidget()
        self.parent_result_table.setColumnCount(4)
        self.parent_result_table.setHorizontalHeaderLabels(["温度(°C)", "信号 S(T,E)", "λ(T)", "摩尔分数 X(T)"])
        self.parent_result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.parent_result_table, 1)

        return widget

    def _create_auto_mf_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        ctrl_group = QtWidgets.QGroupBox("自动计算控制")
        ctrl_layout = QtWidgets.QHBoxLayout(ctrl_group)
        btn_calc_auto = QtWidgets.QPushButton("开始计算")
        btn_calc_auto.clicked.connect(self._calculate_auto_mf)
        ctrl_layout.addWidget(btn_calc_auto)
        self.lbl_auto_status = QtWidgets.QLabel("未计算")
        self.lbl_auto_status.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        ctrl_layout.addWidget(self.lbl_auto_status, 1)
        layout.addWidget(ctrl_group)

        result_group = QtWidgets.QGroupBox("计算结果")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        self.auto_mf_table = QtWidgets.QTableWidget()
        self.auto_mf_table.setColumnCount(5)
        self.auto_mf_table.setHorizontalHeaderLabels(["质量数", "物种名称", "电离能(eV)", "光子能量(eV)", "状态"])
        self.auto_mf_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.auto_mf_table.itemSelectionChanged.connect(self._on_auto_mf_selection_changed)
        result_layout.addWidget(self.auto_mf_table)
        layout.addWidget(result_group, 1)

        plot_group = QtWidgets.QGroupBox("摩尔分数-温度曲线")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
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
        layout.addWidget(plot_group, 2)

        warning_group = QtWidgets.QGroupBox("警告信息")
        warning_layout = QtWidgets.QVBoxLayout(warning_group)
        self.txt_warnings = QtWidgets.QTextEdit()
        self.txt_warnings.setReadOnly(True)
        self.txt_warnings.setMaximumHeight(80)
        self.txt_warnings.setStyleSheet("background-color: rgba(50, 50, 50, 0.8); color: #ff9999;")
        warning_layout.addWidget(self.txt_warnings)
        layout.addWidget(warning_group)

        return widget

    def _create_results_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        self.results_table = QtWidgets.QTableWidget()
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.results_table, 1)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("导出结果 (Excel/CSV)")
        btn_export.clicked.connect(self._export_results)
        btn_layout.addWidget(btn_export)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        plot_group = QtWidgets.QGroupBox("摩尔分数-温度曲线")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        if pg is not None:
            self.mf_plot_widget = pg.PlotWidget()
            self.mf_plot_widget.setBackground("#ffffff")
            self.mf_plot_widget.setMinimumHeight(300)
            self.mf_plot_widget.setLabel("bottom", "温度", units="°C")
            self.mf_plot_widget.setLabel("left", "摩尔分数")
            self.mf_plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.mf_plot_widget.addLegend()
            plot_layout.addWidget(self.mf_plot_widget)
        else:
            self.mf_plot_widget = None
            placeholder = QtWidgets.QLabel("pyqtgraph 未安装，无法显示绘图")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plot_layout.addWidget(placeholder)
        layout.addWidget(plot_group, 2)

        return widget

    def _apply_mz_calibration(self, raw_index):
        return self.calibration.a * raw_index ** 2 + self.calibration.b * raw_index + self.calibration.c

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
                if matched_species:
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
                    for t in temps_list:
                        t_data = self.temperature_scan_data[energy][t]["avg_data"]
                        t_peaks = self._detect_and_integrate_peaks(t_data)
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
        self._refresh_ts_table()
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

    def _detect_and_integrate_peaks(self, data):
        peaks_info = []
        data_np = np.array(data)
        if len(data_np) == 0 or np.max(data_np) == 0:
            return []

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

    def _calc_parent_mole_fraction(self):
        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return
        self.set_busy(True, "正在计算母体摩尔分数...")
        try:
            mz = self.spin_parent_mz.value()
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

            QtWidgets.QMessageBox.information(self, "成功", f"计算了 {len(self.parent_mf_results)} 个温度点的母体摩尔分数")
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

            parent_mz = self.spin_parent_mz.value()
            parent_mw = float(parent_mz)
            parent_energy = self.spin_parent_energy.value()
            parent_signal = self._get_signal_from_scan_data(parent_mz, parent_energy)
            parent_mf_at_tm = self.parent_mf_results.get(float(self.spin_parent_t0.value()), 0)

            for mz, species_list in species_by_mz.items():
                if mz == parent_mz:
                    self.all_species_mf[(mz, "母体", parent_energy)] = self.parent_mf_results
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
                        signal_data = self._get_signal_from_scan_data(mz, calc_energy)
                        if not signal_data:
                            continue
                        mf = self._calc_product_mf_auto(
                            mz, species, parent_mz, parent_mw,
                            parent_energy, parent_signal, parent_mf_at_tm,
                            signal_data, calc_energy=calc_energy,
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

            self.lbl_auto_status.setText(f"已计算 {len(self.all_species_mf)} 条结果")
            self.lbl_auto_status.setStyleSheet("color: #6495ed;")

            self._plot_all_auto_mf()

            QtWidgets.QMessageBox.information(self, "完成", f"自动计算完成！\n共计算 {len(self.all_species_mf)} 条摩尔分数结果")
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self, "错误", f"计算摩尔分数时出错: {e}\n\n{traceback.format_exc()}")
        finally:
            self.set_busy(False, "就绪")

    def _calc_product_mf_auto(self, mz, species, ref_mz, ref_mw,
                               ref_energy, ref_signal_data, ref_mf_at_tm,
                               signal_data, calc_energy=None):
        ie = species.get("ie", 0) or 0
        species_name = species["species"]

        if calc_energy is None:
            best_energy = None
            for energy in sorted(self.available_energies, reverse=True):
                if energy >= ie:
                    best_energy = energy
                    break
        else:
            best_energy = calc_energy

        if best_energy is None:
            return None

        if not signal_data:
            return None

        species_obj = None
        for s in self.database:
            if s.get("mz") == mz and s.get("species") == species_name:
                species_obj = s
                break
        if species_obj is None:
            for s in self.database:
                if s.get("mz") == mz:
                    species_obj = s
                    break

        ref_obj = None
        for s in self.database:
            if s.get("mz") == ref_mz:
                ref_obj = s
                break

        if species_obj is None or ref_obj is None:
            return None

        energies_arr = species_obj.get("energies")
        cross_sections_arr = species_obj.get("cross_sections")
        if energies_arr is not None and cross_sections_arr is not None:
            sigma_i = float(np.interp(best_energy, energies_arr, cross_sections_arr, left=0, right=0))
        else:
            sigma_i = 0.0

        ref_energies = ref_obj.get("energies")
        ref_cross = ref_obj.get("cross_sections")
        if ref_energies is not None and ref_cross is not None:
            sigma_A = float(np.interp(ref_energy, ref_energies, ref_cross, left=0, right=0))
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
            X_i = ref_mf_at_tm * (S_i / S_A_TM) * (sigma_A / sigma_i) * (D_A / D_i) * (lambda_TM / lambda_T)
            results[T] = max(0.0, X_i)

        return results

    def _calc_multi_species_mf_auto(self, mz, species_list, energies,
                                     ref_mz, ref_mw, ref_energy,
                                     ref_signal_data, ref_mf_at_tm):
        result: dict = {}
        warnings: list[str] = []

        if not energies:
            warnings.append(f"质量数 {mz}: 没有可用能量数据")
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
                upper_bound = f"{energy_boundaries[group_idx + 1]:.1f}" if group_idx + 1 < len(energy_boundaries) else "∞"
                warnings.append(
                    f"质量数 {mz}: 电离能区间 ({energy_boundaries[group_idx]:.1f}-{upper_bound} eV) "
                    f"内存在 {len(group)} 个物种 ({', '.join(s['species'] for s in group)})，"
                    f"无法通过能量扫描分离，仅计算匹配系数最高的 {best['species']}，"
                    f"忽略了 {', '.join(ignored_names)}"
                )
            else:
                resolvable_species.append(group[0])

        resolvable_species.sort(key=lambda x: x.get("ie") if x.get("ie") is not None else float("inf"))

        mf_results: dict[tuple, dict[float, float]] = {}

        for species in resolvable_species:
            ie = species.get("ie") if species.get("ie") is not None else float("inf")
            if ie == float("inf"):
                continue

            usable_energies = [e for e in sorted_energies if e >= ie]
            calc_energies = usable_energies[:2]

            if not calc_energies:
                continue

            for calc_e in calc_energies:
                signal_data = self._get_signal_from_scan_data(mz, calc_e)
                if not signal_data:
                    continue

                net_signal = dict(signal_data)

                for prev_key, prev_mf in mf_results.items():
                    prev_name = prev_key[1]
                    prev_energy = prev_key[2]
                    prev_obj = None
                    for s in self.database:
                        if s.get("mz") == mz and s.get("species") == prev_name:
                            prev_obj = s
                            break
                    if prev_obj is None:
                        continue

                    prev_ie = prev_obj.get("ie", 0) or 0
                    if prev_ie > calc_e:
                        continue

                    prev_energies = prev_obj.get("energies")
                    prev_cross = prev_obj.get("cross_sections")
                    if prev_energies is not None and prev_cross is not None:
                        sigma_prev_at_calc = float(np.interp(calc_e, prev_energies, prev_cross, left=0, right=0))
                        sigma_prev_at_prev = float(np.interp(prev_energy, prev_energies, prev_cross, left=0, right=0))
                    else:
                        sigma_prev_at_calc = 0.0
                        sigma_prev_at_prev = 0.0

                    if sigma_prev_at_prev > 0 and sigma_prev_at_calc > 0:
                        ratio = sigma_prev_at_calc / sigma_prev_at_prev
                        for T in net_signal:
                            if T in prev_mf:
                                contribution = prev_mf[T] * ratio
                                net_signal[T] = max(0, net_signal.get(T, 0) - contribution)

                mf = self._calc_product_mf_from_signal(
                    mz, species, calc_e, net_signal,
                    ref_mz, ref_mw, ref_energy,
                    ref_signal_data, ref_mf_at_tm,
                )
                if mf:
                    mf_results[(mz, species["species"], calc_e)] = mf

        if len(species_list) > len(set(k[1] for k in mf_results.keys())) + sum(len(g) - 1 for g in species_groups.values() if len(g) > 1):
            warnings.append(f"质量数 {mz}: 部分可分辨物种未能计算摩尔分数")

        result.update(mf_results)
        if warnings:
            result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"

        return result

    def _calc_product_mf_from_signal(self, mz, species, energy, signal_data,
                                      ref_mz, ref_mw, ref_energy,
                                      ref_signal_data, ref_mf_at_tm):
        species_name = species["species"]

        species_obj = None
        for s in self.database:
            if s.get("mz") == mz and s.get("species") == species_name:
                species_obj = s
                break
        if species_obj is None:
            for s in self.database:
                if s.get("mz") == mz:
                    species_obj = s
                    break

        ref_obj = None
        for s in self.database:
            if s.get("mz") == ref_mz:
                ref_obj = s
                break

        if species_obj is None or ref_obj is None:
            return None

        energies_arr = species_obj.get("energies")
        cross_sections_arr = species_obj.get("cross_sections")
        if energies_arr is not None and cross_sections_arr is not None:
            sigma_i = float(np.interp(energy, energies_arr, cross_sections_arr, left=0, right=0))
        else:
            sigma_i = 0.0

        ref_energies = ref_obj.get("energies")
        ref_cross = ref_obj.get("cross_sections")
        if ref_energies is not None and ref_cross is not None:
            sigma_A = float(np.interp(ref_energy, ref_energies, ref_cross, left=0, right=0))
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
            X_i = ref_mf_at_tm * (S_i / S_A_TM) * (sigma_A / sigma_i) * (D_A / D_i) * (lambda_TM / lambda_T)
            results[T] = max(0.0, X_i)

        return results

    def _update_auto_mf_table(self):
        self.auto_mf_table.setRowCount(0)
        for (mz, species, energy), mf in self.all_species_mf.items():
            row = self.auto_mf_table.rowCount()
            self.auto_mf_table.insertRow(row)
            self.auto_mf_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(mz)))
            self.auto_mf_table.setItem(row, 1, QtWidgets.QTableWidgetItem(species))

            ie = None
            for d in self.pie_species_data:
                if d["mz"] == mz and d["species"] == species:
                    ie = d.get("ie")
                    break
            self.auto_mf_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{ie:.2f}" if ie else "N/A"))
            self.auto_mf_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{energy:.2f}"))
            self.auto_mf_table.setItem(row, 4, QtWidgets.QTableWidgetItem("已计算"))

    def _on_auto_mf_selection_changed(self):
        selected = self.auto_mf_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        mz_item = self.auto_mf_table.item(row, 0)
        species_item = self.auto_mf_table.item(row, 1)
        energy_item = self.auto_mf_table.item(row, 3)
        if not mz_item or not species_item or not energy_item:
            return
        mz = int(mz_item.text())
        species = species_item.text()
        energy = float(energy_item.text())
        key = (mz, species, energy)
        if key in self.all_species_mf:
            self._plot_single_auto_mf(key)

    def _plot_all_auto_mf(self):
        if pg is None or self.auto_mf_plot_widget is None:
            return
        if not self.all_species_mf:
            return

        self.auto_mf_plot_widget.clear()
        self.auto_mf_plot_widget.addLegend()

        colors = [
            (100, 200, 255), (255, 150, 100), (100, 255, 150),
            (255, 100, 100), (255, 255, 100), (200, 150, 255),
            (255, 180, 220), (150, 255, 255), (255, 200, 150),
            (180, 255, 180),
        ]

        color_idx = 0
        species_energies: dict[str, list] = {}
        for (mz, species, energy) in self.all_species_mf.keys():
            if species not in species_energies:
                species_energies[species] = []
            species_energies[species].append((mz, energy))

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

                label = f"{species} ({energy:.1f}eV)"
                self.auto_mf_plot_widget.plot(
                    temps_arr, mfs_arr,
                    pen=pg.mkPen(color, width=2),
                    symbol="o", symbolSize=5, symbolBrush=color,
                    name=label,
                )

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

    def _update_results_table(self):
        all_results: dict[str, dict[float, float]] = {}
        if self.parent_mf_results:
            all_results[f"母体(m/z={self.settings.parent_mz})"] = self.parent_mf_results
        all_results.update(self.product_mf_results)
        for mz, species_data in self.isomeric_results.items():
            for species_name, mf_data in species_data.items():
                all_results[f"{species_name}(m/z={mz})"] = mf_data
        if hasattr(self, "all_species_mf") and self.all_species_mf:
            for (mz, species, energy), mf in self.all_species_mf.items():
                name = f"{species}(m/z={mz},E={energy:.2f}eV)"
                all_results[name] = mf
        if not all_results:
            return
        all_temps = sorted(set().union(*(r.keys() for r in all_results.values())))
        species_names = list(all_results.keys())
        self.results_table.setRowCount(len(all_temps))
        self.results_table.setColumnCount(len(species_names) + 1)
        self.results_table.setHorizontalHeaderLabels(["温度(°C)"] + species_names)
        for i, temp in enumerate(all_temps):
            self.results_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
            for j, name in enumerate(species_names):
                val = all_results[name].get(temp, 0)
                self.results_table.setItem(i, j + 1, QtWidgets.QTableWidgetItem(f"{val:.6f}"))

    def _plot_results_mf(self):
        if pg is None or self.mf_plot_widget is None:
            return

        all_results: dict[str, dict[float, float]] = {}
        if self.parent_mf_results:
            all_results[f"母体(m/z={self.settings.parent_mz})"] = self.parent_mf_results
        all_results.update(self.product_mf_results)
        for mz, species_data in self.isomeric_results.items():
            for species_name, mf_data in species_data.items():
                all_results[f"{species_name}(m/z={mz})"] = mf_data
        if hasattr(self, "all_species_mf") and self.all_species_mf:
            for (mz, species, energy), mf in self.all_species_mf.items():
                name = f"{species}(m/z={mz},E={energy:.2f}eV)"
                all_results[name] = mf
        if not all_results:
            return

        self.mf_plot_widget.clear()
        self.mf_plot_widget.addLegend()

        colors = [
            (255, 100, 100), (100, 180, 255), (255, 200, 50),
            (180, 100, 255), (100, 255, 180), (255, 140, 50),
            (200, 255, 100), (255, 100, 200), (100, 220, 220),
            (220, 180, 140), (140, 100, 220), (220, 220, 100),
        ]

        for idx, (name, res) in enumerate(all_results.items()):
            color = colors[idx % len(colors)]
            sorted_temps = sorted(res.keys())
            temps_arr = np.array(sorted_temps)
            mfs_arr = np.array([res[t] for t in sorted_temps])
            self.mf_plot_widget.plot(
                temps_arr, mfs_arr,
                pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=4, symbolBrush=color,
                name=name,
            )

    def _export_results(self):
        all_results: dict[str, dict[float, float]] = {}
        if self.parent_mf_results:
            all_results[f"母体(m/z={self.settings.parent_mz})"] = self.parent_mf_results
        all_results.update(self.product_mf_results)
        for mz, species_data in self.isomeric_results.items():
            for species_name, mf_data in species_data.items():
                all_results[f"{species_name}(m/z={mz})"] = mf_data
        if hasattr(self, "all_species_mf") and self.all_species_mf:
            for (mz, species, energy), mf in self.all_species_mf.items():
                name = f"{species}(m/z={mz},E={energy:.2f}eV)"
                all_results[name] = mf
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
