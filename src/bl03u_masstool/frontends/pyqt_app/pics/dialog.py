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
    resolve_species_database_path,
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

class PICSCalculatorDialog(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, calibration: Calibration, normalization_settings, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.normalization_settings = normalization_settings
        self.settings = load_mole_fraction_settings()

        self.new_species_name = ""
        self.new_species_formula = ""
        self.new_species_mz = 0
        self.no_mf = 0.0
        self.new_species_mf = 0.0
        self.no_cross_sections: dict[float, float] = {}
        self.mass_disc_exponent = 0.77897

        self.new_species_signal: dict[float, dict[float, float]] = {}
        self.no_signal: dict[float, dict[float, float]] = {}
        self.expansion_coefficients: dict[float, float] = {}

        self.pics_results: dict[float, float] = {}

        self.database: list[dict] = []
        self.mz_index: dict[int, list[int]] = {}
        self._loaded_database_path = ""
        self.project_settings: ProjectSettings | None = None
        self.data_worker: WorkerThread | None = None
        self._load_database()

        self._init_ui()
        self._load_no_cross_sections()

    def set_project_settings(
        self,
        ps: ProjectSettings,
        *,
        activate_project_scope: bool = True,
    ) -> None:
        """Apply ProjectSettings defaults to PICSCalculatorDialog controls."""
        self.project_settings = ps if activate_project_scope else None
        if activate_project_scope:
            self.calibration = ps.to_calibration()
            self.normalization_settings = ps.to_normalization_settings()
            self.settings = ps.to_mole_fraction_settings()
        else:
            self.settings = load_mole_fraction_settings()
        self._load_database()
        if hasattr(self, "txt_folder_path"):
            self.txt_folder_path.setText(ps.pie_scan_folder if activate_project_scope else "")
        if hasattr(self, "spin_no_mz"):
            self.spin_no_mz.setValue(ps.pics_no_mz)
        if hasattr(self, "txt_no_formula"):
            self.txt_no_formula.setText(ps.pics_no_formula)
        if hasattr(self, "double_no_mf"):
            self.double_no_mf.setValue(ps.pics_no_mf)
        if hasattr(self, "double_new_mf"):
            self.double_new_mf.setValue(ps.pics_new_species_mf)
        if hasattr(self, "spin_md_exponent"):
            exponent = (
                ps.mf_mass_disc_exponent
                if activate_project_scope
                else self.settings.mass_disc_exponent
            )
            self.spin_md_exponent.setValue(exponent)
        # 更新摘要栏
        if hasattr(self, "summary_project_label"):
            project_name = ps.project_name or "---"
            system = ps.system or "---"
            self.summary_project_label.setText(f"项目: {project_name}")
            self.summary_system_label.setText(f"体系: {system}")
            data_path = ps.pie_scan_folder or "---"
            self.summary_data_label.setText(f"数据源: {data_path}")
        if hasattr(self, "no_cs_table"):
            self._load_no_cross_sections()

    def _open_project_settings(self):
        """跳转到项目管理页面。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("project")

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

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
        summary_layout.addStretch(1)
        self.summary_import_button = QtWidgets.QPushButton("导入 PICS 数据")
        self.summary_import_button.setObjectName("BrowseButton")
        self.summary_import_button.setToolTip("打开 PICS 导入页面，向本地截面数据库添加外部数据")
        self.summary_import_button.clicked.connect(self._open_pics_import)
        summary_layout.addWidget(self.summary_import_button)
        self.summary_open_project_btn = QtWidgets.QPushButton("项目管理")
        self.summary_open_project_btn.setObjectName("BrowseButton")
        self.summary_open_project_btn.clicked.connect(self._open_project_settings)
        summary_layout.addWidget(self.summary_open_project_btn)
        layout.addWidget(self.summary_bar)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_species_tab(), "1. 物种与参考")
        self.tabs.addTab(self._create_data_tab(), "2. PIE 信号")
        self.tabs.addTab(self._create_params_tab(), "3. 计算设置")
        self.tabs.addTab(self._create_results_tab(), "4. 结果")
        self.tabs.currentChanged.connect(self._refresh_workflow_status)
        layout.addWidget(self.tabs, 1)

        status_bar = QtWidgets.QWidget(self)
        status_bar.setObjectName("ProjectActionBar")
        status_layout = QtWidgets.QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 6, 10, 6)
        self.workflow_stage_label = QtWidgets.QLabel("步骤 1/4 · 填写物种与参考信息")
        self.workflow_stage_label.setObjectName("ReadoutValue")
        status_layout.addWidget(self.workflow_stage_label)
        self.status_label = QtWidgets.QLabel("请填写待计算物种信息")
        self.status_label.setObjectName("InlineStatusLabel")
        status_layout.addWidget(self.status_label, 1)
        layout.addWidget(status_bar)

    def set_busy(self, busy: bool, message: str) -> None:
        """Disable UI during long-running computation."""
        self._set_status(message, "busy" if busy else "")
        self.tabs.setDisabled(busy)

    def _set_status(self, message: str, status: str = "") -> None:
        self.status_label.setText(message)
        self.status_label.setProperty("status", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _open_pics_import(self) -> None:
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("pics_import")

    def _refresh_workflow_status(self, index: int) -> None:
        labels = (
            "步骤 1/4 · 填写物种与参考信息",
            "步骤 2/4 · 加载并检查 PIE 信号",
            "步骤 3/4 · 确认计算设置",
            "步骤 4/4 · 计算并导出结果",
        )
        if 0 <= index < len(labels):
            self.workflow_stage_label.setText(labels[index])

    def _create_species_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        self.species_tabs = QtWidgets.QTabWidget()
        self.species_tabs.addTab(self._create_species_basics_tab(), "基础参数")
        self.species_tabs.addTab(self._create_reference_pics_tab(), "NO截面库")
        layout.addWidget(self.species_tabs)

        return widget

    def _create_species_basics_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)

        new_group = QtWidgets.QGroupBox("新物种信息")
        new_layout = QtWidgets.QFormLayout(new_group)

        self.txt_new_name = QtWidgets.QLineEdit()
        self.txt_new_name.setPlaceholderText("输入新物种名称")
        self.txt_new_name.setToolTip("待计算PICS的新物种名称")
        new_layout.addRow("物种名称:", self.txt_new_name)

        self.txt_new_formula = QtWidgets.QLineEdit()
        self.txt_new_formula.setPlaceholderText("输入分子式")
        self.txt_new_formula.setToolTip("新物种的分子式，用于从数据库查询光电离截面")
        new_layout.addRow("分子式:", self.txt_new_formula)

        self.spin_new_mz = QtWidgets.QSpinBox()
        self.spin_new_mz.setToolTip("新物种的质量数，用于从PIE数据中提取该m/z的信号")
        self.spin_new_mz.setRange(1, 500)
        new_layout.addRow("质量数 m/z:", self.spin_new_mz)

        self.double_new_ie = QtWidgets.QDoubleSpinBox()
        self.double_new_ie.setToolTip("新物种的电离能（eV），留 0 表示未知")
        self.double_new_ie.setRange(0.0, 100.0)
        self.double_new_ie.setDecimals(3)
        self.double_new_ie.setValue(0.0)
        self.double_new_ie.setSpecialValueText("未知")
        new_layout.addRow("电离能 (eV):", self.double_new_ie)

        top_row.addWidget(new_group, 2)

        no_group = QtWidgets.QGroupBox("参考物种 NO")
        no_layout = QtWidgets.QFormLayout(no_group)

        self.txt_no_formula = QtWidgets.QLineEdit("NO")
        self.txt_no_formula.setReadOnly(True)
        no_layout.addRow("分子式:", self.txt_no_formula)

        self.spin_no_mz = QtWidgets.QSpinBox()
        self.spin_no_mz.setToolTip("参考物种NO的质量数，作为PICS计算中信号比的分母")
        self.spin_no_mz.setRange(1, 100)
        self.spin_no_mz.setValue(30)
        self.spin_no_mz.setEnabled(False)
        no_layout.addRow("质量数 m/z:", self.spin_no_mz)

        top_row.addWidget(no_group, 1)
        layout.addLayout(top_row)

        mf_group = QtWidgets.QGroupBox("摩尔分数 (输入量比例)")
        mf_layout = QtWidgets.QFormLayout(mf_group)

        self.double_new_mf = QtWidgets.QDoubleSpinBox()
        self.double_new_mf.setToolTip("新物种在反应器中的输入摩尔分数（已知或假设值）")
        self.double_new_mf.setRange(0.00001, 1.0)
        self.double_new_mf.setValue(0.002)
        self.double_new_mf.setDecimals(6)
        mf_layout.addRow("新物种摩尔分数:", self.double_new_mf)

        self.double_no_mf = QtWidgets.QDoubleSpinBox()
        self.double_no_mf.setToolTip("参考物种NO在反应器中的输入摩尔分数（已知量）")
        self.double_no_mf.setRange(0.00001, 1.0)
        self.double_no_mf.setValue(0.01)
        self.double_no_mf.setDecimals(6)
        mf_layout.addRow("NO摩尔分数:", self.double_no_mf)

        layout.addWidget(mf_group)

        self.species_continue_button = QtWidgets.QPushButton("保存物种信息并继续")
        self.species_continue_button.setObjectName("PrimaryButton")
        self.species_continue_button.setToolTip("检查当前输入并前往 PIE 信号加载")
        self.species_continue_button.clicked.connect(self._confirm_species_and_continue)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.species_continue_button)
        layout.addLayout(action_row)
        layout.addStretch()

        return widget

    def _create_reference_pics_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        no_cs_group = QtWidgets.QGroupBox("NO光电离截面 (从数据库获取)")
        no_cs_layout = QtWidgets.QVBoxLayout(no_cs_group)

        self.no_cs_table = QtWidgets.QTableWidget()
        self.no_cs_table.setColumnCount(2)
        self.no_cs_table.setHorizontalHeaderLabels(["光子能量(eV)", "光电离截面(Mb)"])
        self.no_cs_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        no_cs_layout.addWidget(self.no_cs_table)

        self.lbl_no_cs_status = QtWidgets.QLabel("正在读取数据库…")
        self.lbl_no_cs_status.setObjectName("InlineStatusLabel")
        no_cs_layout.addWidget(self.lbl_no_cs_status)

        self.btn_load_no_cs = QtWidgets.QPushButton("刷新 NO 截面数据")
        self.btn_load_no_cs.setObjectName("BrowseButton")
        self.btn_load_no_cs.setToolTip("重新读取本地 PICS 数据库中的 NO 截面")
        self.btn_load_no_cs.clicked.connect(self.refresh_database)
        no_cs_layout.addWidget(self.btn_load_no_cs)

        layout.addWidget(no_cs_group)

        return widget

    def _create_data_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        folder_group = QtWidgets.QGroupBox("PIE数据文件夹")
        folder_layout = QtWidgets.QVBoxLayout(folder_group)

        folder_top = QtWidgets.QHBoxLayout()
        self.txt_folder_path = QtWidgets.QLineEdit()
        self.txt_folder_path.setReadOnly(True)
        self.txt_folder_path.setPlaceholderText("选择包含 PIE 光谱文件的目录")
        folder_top.addWidget(self.txt_folder_path)
        self.btn_browse = QtWidgets.QPushButton("选择目录")
        self.btn_browse.setObjectName("BrowseButton")
        self.btn_browse.clicked.connect(self._browse_pie_folder)
        folder_top.addWidget(self.btn_browse)
        self.btn_load_data = QtWidgets.QPushButton("加载并提取信号")
        self.btn_load_data.setObjectName("PrimaryButton")
        self.btn_load_data.clicked.connect(self._load_pie_data)
        folder_top.addWidget(self.btn_load_data)
        folder_layout.addLayout(folder_top)

        self.lbl_folder_status = QtWidgets.QLabel("尚未加载 PIE 数据")
        self.lbl_folder_status.setObjectName("InlineStatusLabel")
        folder_layout.addWidget(self.lbl_folder_status)

        layout.addWidget(folder_group)

        signal_group = QtWidgets.QGroupBox("提取的信号数据")
        signal_layout = QtWidgets.QVBoxLayout(signal_group)

        self.signal_table = QtWidgets.QTableWidget()
        self.signal_table.setColumnCount(5)
        self.signal_table.setHorizontalHeaderLabels(["光子能量(eV)", f"{self.new_species_name}信号", "NO信号", "光电流(nA)", "温度(°C)"])
        self.signal_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.signal_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.signal_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.signal_table.setAlternatingRowColors(True)
        signal_layout.addWidget(self.signal_table)

        layout.addWidget(signal_group)

        return widget

    def _create_params_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        md_group = QtWidgets.QGroupBox("质量响应校正设置")
        md_layout = QtWidgets.QFormLayout(md_group)

        self.combo_md_preset = QtWidgets.QComboBox()
        for name in MASS_DISCRIMINATION_PRESETS:
            self.combo_md_preset.addItem(name)
        self.combo_md_preset.currentTextChanged.connect(self._on_md_preset_changed)
        md_layout.addRow("实验条件预设:", self.combo_md_preset)

        self.spin_md_exponent = QtWidgets.QDoubleSpinBox()
        self.spin_md_exponent.setRange(0.0, 2.0)
        self.spin_md_exponent.setDecimals(5)
        self.spin_md_exponent.setValue(self.settings.mass_disc_exponent)
        self.spin_md_exponent.setSingleStep(0.001)
        md_layout.addRow("质量响应指数 n:", self.spin_md_exponent)

        layout.addWidget(md_group)

        io_group = QtWidgets.QGroupBox("光强校正设置")
        io_layout = QtWidgets.QVBoxLayout(io_group)

        self.chk_enable_io_correction = QtWidgets.QCheckBox("启用光强校正（用光电流归一化信号）")
        self.chk_enable_io_correction.setChecked(True)
        io_layout.addWidget(self.chk_enable_io_correction)

        io_note = QtWidgets.QLabel("光谱包含 IO 时，物种与 NO 信号将同时除以 IO；信号比本身保持不变。")
        io_note.setObjectName("HintLabel")
        io_note.setWordWrap(True)
        io_layout.addWidget(io_note)

        layout.addWidget(io_group)

        note_label = QtWidgets.QLabel("PIE 数据按相同温度条件处理，因此这里不应用温度膨胀系数。")
        note_label.setObjectName("HintLabel")
        layout.addWidget(note_label)
        layout.addStretch(1)

        return widget

    def _create_results_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        result_actions = QtWidgets.QHBoxLayout()
        self.calc_btn = QtWidgets.QPushButton("计算 PICS")
        self.calc_btn.setObjectName("PrimaryButton")
        self.calc_btn.clicked.connect(self._calculate_pics)
        result_actions.addWidget(self.calc_btn)
        result_actions.addStretch(1)
        layout.addLayout(result_actions)

        self.result_table = QtWidgets.QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["光子能量(eV)", "温度(°C)", "PICS (Mb)", "误差 (Mb)"])
        self.result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.result_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setAlternatingRowColors(True)
        layout.addWidget(self.result_table)

        stats_group = QtWidgets.QGroupBox("统计结果")
        stats_layout = QtWidgets.QFormLayout(stats_group)

        self.lbl_avg_pics = QtWidgets.QLabel("-")
        stats_layout.addRow("平均PICS:", self.lbl_avg_pics)

        self.lbl_min_pics = QtWidgets.QLabel("-")
        stats_layout.addRow("最小PICS:", self.lbl_min_pics)

        self.lbl_max_pics = QtWidgets.QLabel("-")
        stats_layout.addRow("最大PICS:", self.lbl_max_pics)

        layout.addWidget(stats_group)

        export_group = QtWidgets.QGroupBox("导出结果")
        export_layout = QtWidgets.QHBoxLayout(export_group)

        self.btn_export_csv = QtWidgets.QPushButton("导出 CSV")
        self.btn_export_csv.setObjectName("ExportButton")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._export_results_csv)
        export_layout.addWidget(self.btn_export_csv)

        self.btn_export_database = QtWidgets.QPushButton("保存到 PICS 数据库")
        self.btn_export_database.setObjectName("PrimaryButton")
        self.btn_export_database.setEnabled(False)
        self.btn_export_database.clicked.connect(self._export_to_database)
        export_layout.addWidget(self.btn_export_database)

        export_layout.addStretch()
        layout.addWidget(export_group)

        return widget

    def _update_parameters(self) -> bool:
        self.new_species_name = self.txt_new_name.text().strip()
        self.new_species_formula = self.txt_new_formula.text().strip()
        if not self.new_species_name:
            self.tabs.setCurrentIndex(0)
            self.species_tabs.setCurrentIndex(0)
            self.txt_new_name.setFocus()
            self._set_status("请填写待计算物种名称", "error")
            return False
        if not self.new_species_formula:
            self.tabs.setCurrentIndex(0)
            self.species_tabs.setCurrentIndex(0)
            self.txt_new_formula.setFocus()
            self._set_status("请填写待计算物种分子式", "error")
            return False
        self.new_species_mz = self.spin_new_mz.value()
        self.new_species_mf = self.double_new_mf.value()
        self.no_mf = self.double_no_mf.value()

        self.signal_table.setHorizontalHeaderLabels([
            "光子能量(eV)",
            f"{self.new_species_name}信号" if self.new_species_name else "新物种信号",
            "NO信号",
            "光电流(nA)",
            "温度(°C)"
        ])
        return True

    def _confirm_species_and_continue(self) -> None:
        if self._update_parameters():
            self._set_status(f"已保存 {self.new_species_name}（m/z {self.new_species_mz}）的计算信息", "success")
            self.tabs.setCurrentIndex(1)

    def _browse_pie_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择PIE数据文件夹")
        if folder:
            self.txt_folder_path.setText(folder)

    def _load_pie_data(self):
        folder_path = self.txt_folder_path.text().strip()
        if not folder_path:
            self._set_label_status(self.lbl_folder_status, "请先选择 PIE 数据文件夹", "error")
            self._set_status("PIE 数据目录尚未选择", "error")
            return
        if not self._update_parameters():
            return

        species_mz = self.new_species_mz
        no_mz = self.spin_no_mz.value()
        try:
            a, b, c = self._calibration_coefficients()
        except ValueError as exc:
            self._set_label_status(self.lbl_folder_status, str(exc), "error")
            self._set_status(str(exc), "error")
            return

        self.btn_browse.setDisabled(True)
        self.btn_load_data.setDisabled(True)
        self._set_label_status(self.lbl_folder_status, "正在读取光谱并提取信号…", "busy")
        self._set_status("正在后台加载 PIE 数据", "busy")

        def _load() -> dict:
            from bl03u_masstool.core.spectrum_io import read_spectrum
            from bl03u_masstool.core.pie_analysis import extract_photon_energy

            folder = Path(folder_path)
            files = sorted(folder.glob("*.txt"))
            if not files:
                raise ValueError("所选目录中未找到 .txt 光谱文件")

            rows = []
            skipped = []
            for filepath in files:
                try:
                    spectrum = read_spectrum(str(filepath), header_lines=10, trim_start=0)
                    if len(spectrum.y) == 0:
                        skipped.append(filepath.name)
                        continue

                    energy = extract_photon_energy(spectrum.metadata_lines, str(filepath), fallback=0.0)
                    temp = self._extract_temperature(spectrum.metadata_lines)
                    io_current = self._extract_io_current(spectrum.metadata_lines)

                    if energy <= 0:
                        skipped.append(filepath.name)
                        continue

                    mz_values = tof_to_mz(spectrum.x, a, b, c)
                    rounded_mz = np.round(mz_values)

                    species_indices = np.where(rounded_mz == species_mz)[0]
                    no_indices = np.where(rounded_mz == no_mz)[0]

                    species_signal = float(np.max(spectrum.y[species_indices])) if len(species_indices) else 0.0
                    no_signal = float(np.max(spectrum.y[no_indices])) if len(no_indices) else 0.0

                    if species_signal > 0 or no_signal > 0:
                        rows.append((float(energy), species_signal, no_signal, io_current, float(temp)))
                    else:
                        skipped.append(filepath.name)
                except Exception:
                    skipped.append(filepath.name)
            rows.sort(key=lambda row: row[0])
            return {"rows": rows, "file_count": len(files), "skipped": skipped}

        self.data_worker = WorkerThread(_load, self)
        self.data_worker.finished_with_result.connect(self._show_loaded_pie_data)
        self.data_worker.failed.connect(self._show_pie_load_error)
        self.data_worker.start()

    @staticmethod
    def _set_label_status(label: QtWidgets.QLabel, message: str, status: str = "") -> None:
        label.setText(message)
        label.setProperty("status", status)
        label.style().unpolish(label)
        label.style().polish(label)

    def _show_loaded_pie_data(self, result: object) -> None:
        values = dict(result if isinstance(result, dict) else {})
        rows = list(values.get("rows", []))
        skipped = list(values.get("skipped", []))
        self.signal_table.setRowCount(len(rows))
        for row_index, (energy, species_signal, no_signal, io_current, temp) in enumerate(rows):
            display = (
                f"{energy:.2f}",
                f"{species_signal:.2f}",
                f"{no_signal:.2f}",
                f"{io_current:.6g}" if io_current is not None else "N/A",
                f"{temp:.1f}",
            )
            for column, value in enumerate(display):
                self.signal_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(value))
        self.btn_browse.setEnabled(True)
        self.btn_load_data.setEnabled(True)
        if not rows:
            self._set_label_status(self.lbl_folder_status, "未提取到目标物种或 NO 的有效信号", "error")
            self._set_status("请检查 m/z、标定参数和数据目录", "error")
            return
        detail = f"已提取 {len(rows)}/{values.get('file_count', len(rows))} 个光谱"
        if skipped:
            detail += f"，跳过 {len(skipped)} 个"
        self._set_label_status(self.lbl_folder_status, detail, "success")
        self._set_status("PIE 信号已加载，请确认计算设置", "success")
        self.tabs.setCurrentIndex(2)

    def _show_pie_load_error(self, message: str) -> None:
        self.btn_browse.setEnabled(True)
        self.btn_load_data.setEnabled(True)
        self._set_label_status(self.lbl_folder_status, f"加载失败：{message}", "error")
        self._set_status("PIE 数据加载失败", "error")

    def _extract_temperature(self, metadata_lines):
        for line in metadata_lines:
            line = line.strip().lower()
            if 'temperature' in line:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value = parts[1].strip()
                        if 'c' in value:
                            value = value.replace('c', '').strip()
                        return float(value)
                except (TypeError, ValueError):
                    pass
        return 200.0

    def _calibration_coefficients(self) -> tuple[float, float, float]:
        """Return the configured calibration without substituting hidden defaults."""
        a = float(self.calibration.a)
        b = float(self.calibration.b)
        c = float(self.calibration.c)
        if abs(a) < 1e-20 and abs(b) < 1e-20:
            raise ValueError("定标参数无效：二次项 a 和一次项 b 不能同时为 0")
        return a, b, c

    def _extract_io_current(self, metadata_lines):
        for line in metadata_lines:
            line = line.strip().lower()
            if 'io:' in line:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value = parts[1].strip()
                        if 'na' in value.lower():
                            value = value.lower().replace('na', '').strip()
                        return float(value)
                except (TypeError, ValueError):
                    pass
        return None

    def _on_md_preset_changed(self, name):
        if name in MASS_DISCRIMINATION_PRESETS:
            self.spin_md_exponent.setValue(MASS_DISCRIMINATION_PRESETS[name])

    def _load_database(self):
        self.database = []
        self.mz_index = {}
        self._loaded_database_path = ""
        try:
            configured_path = (
                self.project_settings.pics_database_path
                if self.project_settings is not None
                else None
            )
            db_path = resolve_species_database_path(configured_path)
            if db_path.exists():
                self.database, self.mz_index = load_species_database(str(db_path))
                self._loaded_database_path = str(db_path)
        except Exception:
            pass

    def refresh_database(self) -> None:
        """Reload the local PICS database after an import or manual refresh."""
        self._load_database()
        self._load_no_cross_sections()

    def _load_no_cross_sections(self):
        self.no_cs_table.setRowCount(0)
        self.no_cross_sections = {}

        temp_data = {}

        no_mz = self.spin_no_mz.value()
        indices = self.mz_index.get(no_mz, [])

        for idx in indices:
            spec = self.database[idx]
            species_name = str(spec.get("species") or "").lower()
            formula = str(spec.get("formula") or "").lower()
            if species_name == "no" or species_name == "nitric oxide" or formula == "no":
                energies = spec.get("energies", [])
                cross_sections = spec.get("cross_sections", [])

                if isinstance(energies, np.ndarray):
                    energies = energies.tolist()
                if isinstance(cross_sections, np.ndarray):
                    cross_sections = cross_sections.tolist()

                for energy, cs in zip(energies, cross_sections):
                    energy_key = round(energy, 2)
                    if energy_key not in temp_data:
                        temp_data[energy_key] = []
                    temp_data[energy_key].append(cs)

        for energy_key in sorted(temp_data.keys()):
            values = temp_data[energy_key]
            avg_cs = sum(values) / len(values)
            self.no_cross_sections[energy_key] = avg_cs

            row_count = self.no_cs_table.rowCount()
            self.no_cs_table.insertRow(row_count)
            self.no_cs_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{energy_key:.2f}"))
            self.no_cs_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(f"{avg_cs:.4f}"))

        if self.no_cross_sections:
            self._set_label_status(
                self.lbl_no_cs_status,
                f"已加载 {len(self.no_cross_sections)} 个能量点的 NO 光电离截面（重复能量已取平均）",
                "success",
            )
        else:
            self._set_label_status(
                self.lbl_no_cs_status,
                "数据库中未找到 NO 截面；请先通过 PICS 导入页面添加参考数据",
                "error",
            )

    def _get_no_cross_section_at_energy(self, energy: float) -> float:
        if not self.no_cross_sections:
            return 5.0

        if energy in self.no_cross_sections:
            return self.no_cross_sections[energy]

        energies = sorted(self.no_cross_sections.keys())
        if not energies:
            return 5.0

        if energy <= energies[0]:
            return self.no_cross_sections[energies[0]]
        if energy >= energies[-1]:
            return self.no_cross_sections[energies[-1]]

        for i in range(len(energies) - 1):
            if energies[i] <= energy <= energies[i + 1]:
                frac = (energy - energies[i]) / (energies[i + 1] - energies[i])
                return self.no_cross_sections[energies[i]] * (1 - frac) + self.no_cross_sections[energies[i + 1]] * frac

        return 5.0

    def _calculate_pics(self):
        if not self._update_parameters():
            return
        if self.signal_table.rowCount() == 0:
            self.tabs.setCurrentIndex(1)
            self._set_status("请先加载 PIE 数据并检查信号表", "error")
            return
        if not self.no_cross_sections:
            self.tabs.setCurrentIndex(0)
            self.species_tabs.setCurrentIndex(1)
            self._set_status("缺少 NO 参考截面，无法进行 PICS 计算", "error")
            return

        self.set_busy(True, "正在计算 PICS…")
        self.result_table.setRowCount(0)
        self.pics_results = {}
        self.btn_export_csv.setEnabled(False)
        self.btn_export_database.setEnabled(False)

        species_mz = self.spin_new_mz.value()
        no_mz = self.spin_no_mz.value()
        species_mf = self.double_new_mf.value()
        no_mf = self.double_no_mf.value()
        mass_disc_exp = self.spin_md_exponent.value()

        enable_io_correction = self.chk_enable_io_correction.isChecked() if hasattr(self, 'chk_enable_io_correction') else True

        results = []

        for row in range(self.signal_table.rowCount()):
            try:
                energy = float(self.signal_table.item(row, 0).text()) if self.signal_table.item(row, 0) else 0.0
                species_signal = float(self.signal_table.item(row, 1).text()) if self.signal_table.item(row, 1) else 0.0
                no_signal = float(self.signal_table.item(row, 2).text()) if self.signal_table.item(row, 2) else 0.0
                io_current_text = self.signal_table.item(row, 3).text() if self.signal_table.item(row, 3) else ""
                temp = float(self.signal_table.item(row, 4).text()) if self.signal_table.item(row, 4) else 200.0

                if energy <= 0 or species_signal <= 0 or no_signal <= 0:
                    continue

                io_current = None
                if io_current_text and io_current_text != "N/A":
                    try:
                        io_current = float(io_current_text)
                    except (TypeError, ValueError):
                        io_current = None

                corrected_species = species_signal
                corrected_no = no_signal

                if enable_io_correction and io_current is not None and io_current > 0:
                    corrected_species = species_signal / io_current
                    corrected_no = no_signal / io_current

                sigma_no = self._get_no_cross_section_at_energy(energy)

                pics = calc_pics_single_energy(
                    corrected_species,
                    corrected_no,
                    species_mf,
                    no_mf,
                    species_mz,
                    no_mz,
                    sigma_no,
                    mass_disc_exp,
                )
                if pics <= 0:
                    continue

                results.append((energy, temp, pics))

                row_count = self.result_table.rowCount()
                self.result_table.insertRow(row_count)
                self.result_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(f"{energy:.2f}"))
                self.result_table.setItem(row_count, 1, QtWidgets.QTableWidgetItem(f"{temp:.1f}"))
                self.result_table.setItem(row_count, 2, QtWidgets.QTableWidgetItem(f"{pics:.4f}"))
                self.result_table.setItem(row_count, 3, QtWidgets.QTableWidgetItem("-"))

                self.pics_results[(energy, temp)] = pics

            except (ValueError, AttributeError, TypeError):
                continue

        if not results:
            self.set_busy(False, "没有可计算的数据点；请确认物种和 NO 信号均大于 0")
            self._set_status("没有可计算的数据点；请确认物种和 NO 信号均大于 0", "error")
            return

        pics_values = [r[2] for r in results]

        self.lbl_avg_pics.setText(f"{np.mean(pics_values):.4f} Mb")
        self.lbl_min_pics.setText(f"{min(pics_values):.4f} Mb")
        self.lbl_max_pics.setText(f"{max(pics_values):.4f} Mb")

        status = "已启用光强校正" if enable_io_correction else "未启用光强校正"
        self.set_busy(False, "")
        self.btn_export_csv.setEnabled(True)
        self.btn_export_database.setEnabled(True)
        self._set_status(f"计算完成：{len(results)} 个数据点，{status}", "success")
        self.tabs.setCurrentIndex(3)

    def _export_results_csv(self):
        if not self.pics_results:
            self._set_status("当前没有可导出的 PICS 结果", "error")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出CSV文件", f"pics_results_{self.new_species_name}.csv", "CSV Files (*.csv)"
        )

        if file_path:
            rows = []
            for (energy, temp), sigma in self.pics_results.items():
                rows.append({
                    "物种名称": self.new_species_name,
                    "分子式": self.new_species_formula,
                    "质量数": self.new_species_mz,
                    "光子能量(eV)": energy,
                    "温度(°C)": temp,
                    "PICS(Mb)": sigma,
                })

            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            self._set_status(f"结果已导出到 {file_path}", "success")

    def _export_to_database(self):
        if not self.pics_results:
            self._set_status("当前没有可保存的 PICS 结果", "error")
            return

        db_path = resolve_species_database_path(
            self.project_settings.pics_database_path
            if self.project_settings is not None
            else None
        )
        if not db_path.exists():
            self._set_status("PICS 数据库文件不存在", "error")
            return

        ie_value = self.double_new_ie.value()
        ie: float | None = None if ie_value == 0.0 else ie_value

        try:
            from bl03u_masstool.core.pics_import import write_pics_records

            points_by_energy: dict[float, list[float]] = {}
            for (energy, _temp), sigma in self.pics_results.items():
                points_by_energy.setdefault(float(energy), []).append(float(sigma))
            energies = sorted(points_by_energy.keys())
            cross_sections = [float(np.mean(points_by_energy[e])) for e in energies]

            record = {
                "mz": self.new_species_mz,
                "species": self.new_species_name,
                "ie": ie,
                "energies": energies,
                "cross_sections": cross_sections,
            }
            result = write_pics_records([record], db_path, mode="upsert")
            self.refresh_database()
            self._set_status(
                f"已保存到数据库：写入 {result['inserted_species']} 个物种，"
                f"更新 {result['replaced_species']} 条记录，{result['inserted_points']} 个数据点",
                "success",
            )
        except Exception as e:
            self._set_status(f"写入数据库失败：{e}", "error")
