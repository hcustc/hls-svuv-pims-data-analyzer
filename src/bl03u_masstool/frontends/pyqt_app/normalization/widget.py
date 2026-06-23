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

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin, combo_set_data


class CommonParametersWidget(QtWidgets.QWidget, DataFrameTableMixin):
    settings_saved = QtCore.pyqtSignal()

    def __init__(self, settings: NormalizationSettings, calibration: Calibration, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.calibration = calibration
        self.project_settings: ProjectSettings | None = None
        self.worker: WorkerThread | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_basic_tab(), "基础参数")
        tabs.addTab(self._build_mole_fraction_tab(), "摩尔分数")
        root.addWidget(tabs, 1)

        # 底部工具栏
        bottom_bar = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom_bar)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(8)
        self.save_button = QtWidgets.QPushButton("💾  保存通用参数")
        self.save_button.setFixedHeight(30)
        self.save_button.clicked.connect(self.save_settings)
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("ProjectStatus")
        bl.addWidget(self.save_button)
        bl.addWidget(self.status_label, 1)
        root.addWidget(bottom_bar)

        self.load_from_settings()

    # ── Tab 1: 基础参数 ──────────────────────────────────────────
    def _build_basic_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 归一化参数
        norm_group = QtWidgets.QGroupBox("归一化参数")
        norm_layout = QtWidgets.QGridLayout(norm_group)
        norm_layout.setHorizontalSpacing(10)
        norm_layout.setVerticalSpacing(8)

        self.light_source_combo = QtWidgets.QComboBox()
        self.light_source_combo.addItem("IO 光电流", "io")
        self.light_source_combo.addItem("Beam Current 储存环束流", "beam_current")
        self.temperature_photon_check = QtWidgets.QCheckBox("温度扫描光强归一化")
        self.temperature_kr_check = QtWidgets.QCheckBox("温度扫描使用Kr膨胀校正")
        self.pie_photon_mode_combo = QtWidgets.QComboBox()
        self.pie_photon_mode_combo.addItem("归一化到首个光强", "first")
        self.pie_photon_mode_combo.addItem("直接除以光强", "none")
        self.pie_photon_mode_combo.addItem("不做光强归一化", "off")
        self.mass_discrimination_edit = QtWidgets.QDoubleSpinBox()
        self.mass_discrimination_edit.setRange(0.000001, 1_000_000)
        self.mass_discrimination_edit.setDecimals(6)
        self.mass_discrimination_edit.setToolTip("温度扫描/PIE分析使用的质量歧视因子 D")

        norm_layout.addWidget(QtWidgets.QLabel("光强来源:"), 0, 0)
        norm_layout.addWidget(self.light_source_combo, 0, 1)
        norm_layout.addWidget(self.temperature_photon_check, 0, 2)
        norm_layout.addWidget(self.temperature_kr_check, 0, 3)
        norm_layout.addWidget(QtWidgets.QLabel("PIE光强归一化:"), 1, 0)
        norm_layout.addWidget(self.pie_photon_mode_combo, 1, 1)
        norm_layout.addWidget(QtWidgets.QLabel("质量歧视因子 D:"), 1, 2)
        norm_layout.addWidget(self.mass_discrimination_edit, 1, 3)
        norm_layout.setColumnStretch(1, 1)
        norm_layout.setColumnStretch(3, 1)
        layout.addWidget(norm_group)

        # 定标参数
        cal_group = QtWidgets.QGroupBox("质量数定标  m/z = A·x² + B·x + C")
        cal_layout = QtWidgets.QGridLayout(cal_group)
        cal_layout.setHorizontalSpacing(10)
        cal_layout.setVerticalSpacing(8)

        self.calibration_a_edit = QtWidgets.QDoubleSpinBox()
        self.calibration_b_edit = QtWidgets.QDoubleSpinBox()
        self.calibration_c_edit = QtWidgets.QDoubleSpinBox()
        for edit in (self.calibration_a_edit, self.calibration_b_edit, self.calibration_c_edit):
            edit.setRange(-1_000_000, 1_000_000)
            edit.setDecimals(12)
            edit.setSingleStep(0.000000001)

        cal_layout.addWidget(QtWidgets.QLabel("A:"), 0, 0)
        cal_layout.addWidget(self.calibration_a_edit, 0, 1)
        cal_layout.addWidget(QtWidgets.QLabel("B:"), 0, 2)
        cal_layout.addWidget(self.calibration_b_edit, 0, 3)
        cal_layout.addWidget(QtWidgets.QLabel("C:"), 0, 4)
        cal_layout.addWidget(self.calibration_c_edit, 0, 5)
        for col in (1, 3, 5):
            cal_layout.setColumnStretch(col, 1)
        layout.addWidget(cal_group)

        # 元素筛选
        elements_group = QtWidgets.QGroupBox("元素筛选（影响PIE物种匹配范围）")
        elements_layout = QtWidgets.QHBoxLayout(elements_group)
        elements_layout.setSpacing(4)
        self.element_checks = {}
        for elem in COMMON_ELEMENTS:
            chk = QtWidgets.QCheckBox(elem)
            chk.setChecked(True)
            self.element_checks[elem] = chk
            elements_layout.addWidget(chk)
        self.select_all_elements_btn = QtWidgets.QPushButton("全选")
        self.select_all_elements_btn.setFixedWidth(50)
        self.select_all_elements_btn.clicked.connect(lambda: self._set_all_elements(True))
        self.clear_elements_btn = QtWidgets.QPushButton("清空")
        self.clear_elements_btn.setFixedWidth(50)
        self.clear_elements_btn.clicked.connect(lambda: self._set_all_elements(False))
        elements_layout.addWidget(self.select_all_elements_btn)
        elements_layout.addWidget(self.clear_elements_btn)
        elements_layout.addStretch()
        layout.addWidget(elements_group)

        layout.addStretch()
        return widget

    # ── Tab 2: 摩尔分数 ──────────────────────────────────────────
    def _build_mole_fraction_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 质量歧视因子
        md_group = QtWidgets.QGroupBox("质量歧视因子  D_i = (MW / 30)^n")
        md_layout = QtWidgets.QVBoxLayout(md_group)
        md_layout.setSpacing(8)

        md_ctrl = QtWidgets.QHBoxLayout()
        md_ctrl.setSpacing(8)
        self.mf_md_preset_combo = QtWidgets.QComboBox()
        self.mf_md_preset_combo.setToolTip("选择实验条件预设，自动填入对应指数 n")
        for name in MASS_DISCRIMINATION_PRESETS:
            self.mf_md_preset_combo.addItem(name)
        self.mf_md_preset_combo.currentTextChanged.connect(self._on_mf_md_preset_changed)
        self.mf_mass_disc_exponent_edit = QtWidgets.QDoubleSpinBox()
        self.mf_mass_disc_exponent_edit.setToolTip("公式中的指数 n，决定质量歧视强度")
        self.mf_mass_disc_exponent_edit.setRange(0.0, 10.0)
        self.mf_mass_disc_exponent_edit.setDecimals(5)
        self.mf_mass_disc_exponent_edit.setSingleStep(0.001)
        self.mf_mass_disc_exponent_edit.setValue(0.77897)
        self.mf_md_preview_btn = QtWidgets.QPushButton("预览 D_i 列表")
        self.mf_md_preview_btn.setCheckable(True)
        self.mf_md_preview_btn.clicked.connect(self._toggle_md_preview)

        md_ctrl.addWidget(QtWidgets.QLabel("实验条件预设:"))
        md_ctrl.addWidget(self.mf_md_preset_combo, 2)
        md_ctrl.addWidget(QtWidgets.QLabel("指数 n:"))
        md_ctrl.addWidget(self.mf_mass_disc_exponent_edit, 1)
        md_ctrl.addWidget(self.mf_md_preview_btn)
        md_layout.addLayout(md_ctrl)

        self.mf_md_preview_table = QtWidgets.QTableWidget()
        self.mf_md_preview_table.setColumnCount(3)
        self.mf_md_preview_table.setHorizontalHeaderLabels(["物种", "分子量 (MW)", "D_i"])
        self.mf_md_preview_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.mf_md_preview_table.setAlternatingRowColors(True)
        self.mf_md_preview_table.setVisible(False)
        md_layout.addWidget(self.mf_md_preview_table, 1)
        layout.addWidget(md_group, 1)

        # Kr膨胀系数
        kr_group = QtWidgets.QGroupBox("Kr膨胀系数 λ(T)")
        kr_layout = QtWidgets.QVBoxLayout(kr_group)
        kr_layout.setSpacing(8)

        kr_form = QtWidgets.QGridLayout()
        kr_form.setHorizontalSpacing(8)
        kr_form.setVerticalSpacing(6)
        self.kr_folder_edit = QtWidgets.QLineEdit()
        self.kr_folder_edit.setPlaceholderText("选择Kr温度扫描文件夹")
        self.kr_folder_button = QtWidgets.QPushButton("浏览…")
        self.kr_folder_button.setFixedWidth(72)
        self.kr_folder_button.clicked.connect(self.select_kr_folder)

        # 卡峰模式选择
        self.kr_peak_mode_auto_radio = QtWidgets.QRadioButton("自动卡峰")
        self.kr_peak_mode_manual_radio = QtWidgets.QRadioButton("手动卡峰文件")
        self.kr_peak_mode_auto_radio.setChecked(True)
        self.kr_peak_mode_auto_radio.toggled.connect(self.on_kr_peak_mode_changed)

        self.kr_peak_file_edit = QtWidgets.QLineEdit()
        self.kr_peak_file_edit.setPlaceholderText("选择Kr手动卡峰文件 (*.csv, *.yaml, *.yml)")
        self.kr_peak_file_edit.setEnabled(False)
        self.kr_peak_file_button = QtWidgets.QPushButton("选择…")
        self.kr_peak_file_button.setFixedWidth(72)
        self.kr_peak_file_button.setEnabled(False)
        self.kr_peak_file_button.clicked.connect(self.select_kr_peak_file)
        self.compute_kr_button = QtWidgets.QPushButton("▶  计算 Kr 膨胀系数")
        self.compute_kr_button.clicked.connect(self.compute_kr_factors)

        kr_form.addWidget(QtWidgets.QLabel("扫描文件夹:"), 0, 0)
        kr_form.addWidget(self.kr_folder_edit, 0, 1)
        kr_form.addWidget(self.kr_folder_button, 0, 2)

        # 卡峰模式行
        kr_form.addWidget(QtWidgets.QLabel("卡峰模式:"), 1, 0)
        peak_mode_layout = QtWidgets.QHBoxLayout()
        peak_mode_layout.addWidget(self.kr_peak_mode_auto_radio)
        peak_mode_layout.addWidget(self.kr_peak_mode_manual_radio)
        peak_mode_layout.addStretch()
        kr_form.addLayout(peak_mode_layout, 1, 1, 1, 2)

        kr_form.addWidget(QtWidgets.QLabel("卡峰文件:"), 2, 0)
        kr_form.addWidget(self.kr_peak_file_edit, 2, 1)
        kr_form.addWidget(self.kr_peak_file_button, 2, 2)
        kr_form.addWidget(self.compute_kr_button, 3, 1)
        kr_form.setColumnStretch(1, 1)
        kr_layout.addLayout(kr_form)

        self.factor_table = QtWidgets.QTableWidget()
        self.factor_table.setColumnCount(3)
        self.factor_table.setHorizontalHeaderLabels(["温度 (°C)", "Kr 信号积分", "λ(T)"])
        self.factor_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.factor_table.setAlternatingRowColors(True)
        kr_layout.addWidget(self.factor_table, 1)
        layout.addWidget(kr_group, 1)

        return widget

    def load_from_settings(self) -> None:
        # 当有 ProjectSettings 时，优先从 ProjectSettings 加载项目特定参数
        # 这确保项目的参数设置在项目关闭/重新打开时被保留
        if self.project_settings:
            combo_set_data(self.light_source_combo, self.project_settings.light_source)
            self.temperature_photon_check.setChecked(self.project_settings.temperature_photon_normalize)
            self.temperature_kr_check.setChecked(self.project_settings.temperature_kr_correct)
            self.mass_discrimination_edit.setValue(self.project_settings.mass_discrimination)
            combo_set_data(self.pie_photon_mode_combo, self.project_settings.pie_photon_mode)
            self.kr_folder_edit.setText(self.project_settings.kr_calibration_folder)
            self.kr_peak_file_edit.setText(self.project_settings.kr_calibration_peak_file)
            # 根据是否有卡峰文件来设置模式
            use_manual_peak = bool(self.project_settings.kr_calibration_peak_file.strip())
            self.kr_peak_mode_manual_radio.setChecked(use_manual_peak)
            self.kr_peak_mode_auto_radio.setChecked(not use_manual_peak)
        else:
            # 无项目时，从本地 NormalizationSettings 加载
            combo_set_data(self.light_source_combo, self.settings.light_source)
            self.temperature_photon_check.setChecked(self.settings.temperature_photon_normalize)
            self.temperature_kr_check.setChecked(self.settings.temperature_kr_correct)
            self.mass_discrimination_edit.setValue(self.settings.mass_discrimination)
            combo_set_data(self.pie_photon_mode_combo, self.settings.pie_photon_mode)
            self.kr_folder_edit.setText(self.settings.kr_calibration_folder)
            self.kr_peak_file_edit.setText(self.settings.kr_calibration_peak_file)
            # 根据是否有卡峰文件来设置模式
            use_manual_peak = bool(self.settings.kr_calibration_peak_file.strip())
            self.kr_peak_mode_manual_radio.setChecked(use_manual_peak)
            self.kr_peak_mode_auto_radio.setChecked(not use_manual_peak)

        # 更新卡峰文件控件的启用状态
        self.on_kr_peak_mode_changed()

        # 加载校准参数（全局配置）
        calibration = load_calibration_config()
        self.calibration_a_edit.setValue(calibration.a)
        self.calibration_b_edit.setValue(calibration.b)
        self.calibration_c_edit.setValue(calibration.c)

        # 加载摩尔分数参数
        if self.project_settings:
            # 恢复预设选择
            if self.project_settings.mf_md_preset:
                combo_set_data(self.mf_md_preset_combo, self.project_settings.mf_md_preset)
            self.mf_mass_disc_exponent_edit.setValue(self.project_settings.mf_mass_disc_exponent)
        self.refresh_factor_table()
        # 加载元素筛选
        selected = self.settings.selected_elements
        for elem, chk in self.element_checks.items():
            chk.setChecked(not selected or elem in selected)

    def set_project_settings(self, project_settings: ProjectSettings) -> None:
        """Set the project settings reference for parameter synchronization."""
        self.project_settings = project_settings
        if project_settings:
            # 恢复预设选择
            if project_settings.mf_md_preset:
                combo_set_data(self.mf_md_preset_combo, project_settings.mf_md_preset)
            self.mf_mass_disc_exponent_edit.setValue(project_settings.mf_mass_disc_exponent)

    def apply_to_settings(self) -> None:
        self.settings.light_source = self.light_source_combo.currentData()
        self.settings.temperature_photon_normalize = self.temperature_photon_check.isChecked()
        self.settings.temperature_kr_correct = self.temperature_kr_check.isChecked()
        self.settings.mass_discrimination = self.mass_discrimination_edit.value()
        self.settings.pie_photon_mode = self.pie_photon_mode_combo.currentData()
        self.settings.kr_calibration_folder = self.kr_folder_edit.text().strip()
        # 根据卡峰模式决定是否保存卡峰文件路径
        if self.kr_peak_mode_manual_radio.isChecked():
            self.settings.kr_calibration_peak_file = self.kr_peak_file_edit.text().strip()
        else:
            self.settings.kr_calibration_peak_file = ""
        self.calibration = Calibration(
            a=self.calibration_a_edit.value(),
            b=self.calibration_b_edit.value(),
            c=self.calibration_c_edit.value(),
        )
        # 同步所有项目特定参数到 ProjectSettings
        if self.project_settings:
            self.project_settings.light_source = self.light_source_combo.currentData()
            self.project_settings.temperature_photon_normalize = self.temperature_photon_check.isChecked()
            self.project_settings.temperature_kr_correct = self.temperature_kr_check.isChecked()
            self.project_settings.mass_discrimination = self.mass_discrimination_edit.value()
            self.project_settings.pie_photon_mode = self.pie_photon_mode_combo.currentData()
            self.project_settings.kr_calibration_folder = self.kr_folder_edit.text().strip()
            # 根据卡峰模式决定是否保存卡峰文件路径
            if self.kr_peak_mode_manual_radio.isChecked():
                self.project_settings.kr_calibration_peak_file = self.kr_peak_file_edit.text().strip()
            else:
                self.project_settings.kr_calibration_peak_file = ""
            self.project_settings.mf_md_preset = self.mf_md_preset_combo.currentText()
            self.project_settings.mf_mass_disc_exponent = self.mf_mass_disc_exponent_edit.value()
        self.settings.selected_elements = [
            elem for elem, chk in self.element_checks.items() if chk.isChecked()
        ]

    def _set_all_elements(self, checked: bool):
        for chk in self.element_checks.values():
            chk.setChecked(checked)

    def on_kr_peak_mode_changed(self):
        """卡峰模式切换时，启用/禁用手动卡峰文件选择"""
        use_manual = self.kr_peak_mode_manual_radio.isChecked()
        self.kr_peak_file_edit.setEnabled(use_manual)
        self.kr_peak_file_button.setEnabled(use_manual)
        # 自动模式时清空卡峰文件路径
        if not use_manual:
            self.kr_peak_file_edit.clear()

    def select_kr_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择Kr定标文件夹")
        if folder:
            self.kr_folder_edit.setText(folder)

    def select_kr_peak_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择Kr手动卡峰文件",
            "",
            "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;All Files (*)",
        )
        if path:
            self.kr_peak_file_edit.setText(path)

    def save_settings(self):
        self.apply_to_settings()
        normalization_path = save_normalization_settings(self.settings)
        calibration_path = save_calibration_config(self.calibration)
        # 保存ProjectSettings
        project_path = None
        if self.project_settings:
            from bl03u_masstool.core.project_settings import ProjectSettingsManager, save_project_settings
            ps = ProjectSettingsManager().get()
            # 同步所有项目参数到 ProjectSettings
            ps.light_source = self.project_settings.light_source
            ps.temperature_photon_normalize = self.project_settings.temperature_photon_normalize
            ps.temperature_kr_correct = self.project_settings.temperature_kr_correct
            ps.mass_discrimination = self.project_settings.mass_discrimination
            ps.pie_photon_mode = self.project_settings.pie_photon_mode
            ps.kr_calibration_folder = self.project_settings.kr_calibration_folder
            ps.kr_calibration_peak_file = self.project_settings.kr_calibration_peak_file
            ps.mf_md_preset = self.project_settings.mf_md_preset
            ps.mf_mass_disc_exponent = self.project_settings.mf_mass_disc_exponent
            project_path = save_project_settings(ps)
        self.status_label.setText(f"已保存: {normalization_path}；{calibration_path}")
        self.settings_saved.emit()

    def compute_kr_factors(self):
        self.apply_to_settings()
        folder = self.settings.kr_calibration_folder
        if not folder:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择Kr定标文件夹")
            return

        # 验证文件夹存在且包含光谱文件
        from pathlib import Path
        folder_path = Path(folder)
        if not folder_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "错误",
                f"指定的文件夹不存在：\n{folder}"
            )
            return

        spectrum_files = list(folder_path.glob("*.txt")) + list(folder_path.glob("*.asc"))
        if not spectrum_files:
            QtWidgets.QMessageBox.warning(
                self,
                "错误",
                f"文件夹中没有找到光谱文件 (*.txt, *.asc)：\n{folder}"
            )
            return

        # 使用默认peak detection配置
        peak_config = load_peak_detection_config()
        self.set_busy(True, "正在计算Kr膨胀系数...")
        self.worker = WorkerThread(
            lambda: compute_kr_expansion_factors(
                folder,
                calibration=self.calibration,
                manual_peak_path=self.settings.kr_calibration_peak_file or None,
                light_source=self.settings.light_source,
                threshold_end=peak_config.threshold_end,
                min_intensity=peak_config.min_intensity,
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
        self.worker.finished_with_result.connect(self.on_kr_factors_ready)
        self.worker.failed.connect(self.on_kr_factors_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def set_busy(self, busy: bool, message: str) -> None:
        self.status_label.setText(message)
        for widget in (
            self.compute_kr_button,
            self.save_button,
            self.kr_folder_button,
            self.kr_peak_file_button,
            self.kr_peak_mode_auto_radio,
            self.kr_peak_mode_manual_radio,
        ):
            widget.setDisabled(busy)

    def on_kr_factors_ready(self, result: object) -> None:
        df = result
        from bl03u_masstool.core.mole_fraction import parse_expansion_factors_from_result

        # 使用新的辅助函数处理膨胀系数
        self.settings.expansion_factors = parse_expansion_factors_from_result(df)

        # 保存kr_signal用于展示
        self._kr_signal_data = {
            float(row["temperature"]): float(row["kr_signal"])
            for _, row in df.iterrows()
        }

        # 显示计算结果
        total_factors = len(self.settings.expansion_factors)
        msg = f"已计算 {total_factors} 个温度点的Kr膨胀系数"

        self.refresh_factor_table()
        save_normalization_settings(self.settings)
        self.status_label.setText(msg)


    def on_kr_factors_failed(self, message: str) -> None:
        # 提供更详细的错误诊断
        error_msg = message
        if "all Kr calibration signals are zero" in error_msg:
            error_msg += (
                "\n\n📋 可能的原因：\n"
                f"  1. 光强来源设置错误（当前：{self.settings.light_source}）\n"
                f"  2. Kr定标文件夹路径错误（当前：{self.settings.kr_calibration_folder}）\n"
                f"  3. 光谱数据质量差或没有Kr信号\n\n"
                "💡 尝试：\n"
                "  • 检查\"光强来源\"设置是否与光谱文件匹配\n"
                "  • 确认文件夹路径指向包含光谱文件的目录\n"
                "  • 使用包含足够Kr信号的能量点"
            )
        QtWidgets.QMessageBox.critical(self, "错误", error_msg)

    def refresh_factor_table(self) -> None:
        kr_signal = getattr(self, "_kr_signal_data", {})
        self.factor_table.setRowCount(0)

        # 检测膨胀系数格式（单能量或多能量）
        if not self.settings.expansion_factors:
            return

        first_key = next(iter(self.settings.expansion_factors.keys()), None)
        if first_key is None:
            return

        is_multi_energy = isinstance(self.settings.expansion_factors[first_key], dict)

        if is_multi_energy:
            # 多能量格式：{能量: {温度: 膨胀系数}}
            for energy in sorted(self.settings.expansion_factors.keys()):
                temp_factors = self.settings.expansion_factors[energy]
                for temp in sorted(temp_factors.keys()):
                    row = self.factor_table.rowCount()
                    self.factor_table.insertRow(row)
                    self.factor_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
                    signal = kr_signal.get(temp, None)
                    sig_text = f"{signal:.4f}" if signal is not None else "-"
                    self.factor_table.setItem(row, 1, QtWidgets.QTableWidgetItem(sig_text))
                    lam = temp_factors[temp]
                    # 显示能量信息
                    energy_text = f"{lam:.6f} @{energy:.2f}eV"
                    self.factor_table.setItem(row, 2, QtWidgets.QTableWidgetItem(energy_text))
        else:
            # 单能量格式：{温度: 膨胀系数}
            for temp in sorted(self.settings.expansion_factors.keys()):
                row = self.factor_table.rowCount()
                self.factor_table.insertRow(row)
                self.factor_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
                signal = kr_signal.get(temp, None)
                sig_text = f"{signal:.4f}" if signal is not None else "-"
                self.factor_table.setItem(row, 1, QtWidgets.QTableWidgetItem(sig_text))
                lam = self.settings.expansion_factors[temp]
                self.factor_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{lam:.6f}"))


    def _toggle_md_preview(self, checked: bool) -> None:
        if checked:
            self._preview_mf_mass_discrimination()
        else:
            self.mf_md_preview_table.setVisible(False)

    def _on_mf_md_preset_changed(self, text: str) -> None:
        if text in MASS_DISCRIMINATION_PRESETS:
            self.mf_mass_disc_exponent_edit.setValue(MASS_DISCRIMINATION_PRESETS[text])

    def _preview_mf_mass_discrimination(self) -> None:
        from bl03u_masstool.core.mole_fraction import calc_mass_discrimination
        from bl03u_masstool.core.pie_analysis import load_species_database
        from bl03u_masstool.core.config import species_database_path
        db_path = species_database_path()
        if not db_path.exists():
            QtWidgets.QMessageBox.warning(self, "提示", "物种数据库未找到，请先在PIE鉴定中导入PICS")
            return
        database, _ = load_species_database(str(db_path))
        seen: set[tuple] = set()
        species_list = []
        for spec in database:
            key = (spec.get("species", spec.get("name", "")), spec["mz"])
            if key not in seen:
                seen.add(key)
                species_list.append(spec)
        n = self.mf_mass_disc_exponent_edit.value()
        self.mf_md_preview_table.setRowCount(0)
        for spec in species_list[:30]:
            row = self.mf_md_preview_table.rowCount()
            self.mf_md_preview_table.insertRow(row)
            name = spec.get("species", spec.get("name", ""))
            self.mf_md_preview_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.mf_md_preview_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(spec["mz"])))
            D_i = calc_mass_discrimination(spec["mz"], n)
            self.mf_md_preview_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{D_i:.6f}"))
        self.mf_md_preview_table.setVisible(True)


class PeakDetectionWidget(QtWidgets.QWidget):
    settings_saved = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peak_detection = load_peak_detection_config()
        self.project_settings: ProjectSettings | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 单一面板 - 寻峰参数
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        peak_group = QtWidgets.QGroupBox("自动寻峰参数")
        peak_layout = QtWidgets.QGridLayout(peak_group)
        peak_layout.setHorizontalSpacing(8)
        peak_layout.setVerticalSpacing(8)

        self.peak_algorithm_combo = QtWidgets.QComboBox()
        self.peak_algorithm_combo.addItem("Ensemble 融合检测（推荐）", "ensemble")
        self.peak_algorithm_combo.addItem("Prominence", "prominence")
        self.peak_algorithm_combo.addItem("传统局部极大", "legacy")
        self.peak_algorithm_combo.addItem("CWT 小波", "cwt")
        self.peak_algorithm_combo.setToolTip("主工作台自动寻峰使用的算法")
        self.peak_detection_min_idx_edit = QtWidgets.QSpinBox()
        self.peak_detection_min_idx_edit.setRange(0, 10_000_000)
        self.peak_detection_min_idx_edit.setToolTip("自动寻峰从该数据点之后开始，避免文件头或低TOF噪声参与寻峰")
        self.peak_threshold_end_edit = QtWidgets.QDoubleSpinBox()
        self.peak_threshold_end_edit.setRange(0, 1_000_000)
        self.peak_threshold_end_edit.setDecimals(3)
        self.peak_threshold_end_edit.setToolTip("向峰两侧扩展边界时，低于该强度即认为到达峰结束")
        self.peak_min_intensity_edit = QtWidgets.QDoubleSpinBox()
        self.peak_min_intensity_edit.setRange(0, 1_000_000)
        self.peak_min_intensity_edit.setDecimals(3)
        self.peak_min_intensity_edit.setToolTip("低于该强度的局部极大值不会作为候选峰")
        self.peak_nearby_window_edit = QtWidgets.QSpinBox()
        self.peak_nearby_window_edit.setRange(0, 100_000)
        self.peak_nearby_window_edit.setToolTip("该半宽范围内若已有更高峰，则当前候选峰会被抑制")
        self.peak_duplicate_window_edit = QtWidgets.QSpinBox()
        self.peak_duplicate_window_edit.setRange(0, 100_000)
        self.peak_duplicate_window_edit.setToolTip("接受一个峰后，该半宽范围内的候选点会被视为同一峰")
        self.peak_boundary_padding_edit = QtWidgets.QSpinBox()
        self.peak_boundary_padding_edit.setRange(0, 100_000)
        self.peak_boundary_padding_edit.setToolTip("最终卡峰边界向左右额外扩展的数据点数")
        self.peak_weak_tail_ratio_edit = QtWidgets.QDoubleSpinBox()
        self.peak_weak_tail_ratio_edit.setRange(0.001, 1_000_000)
        self.peak_weak_tail_ratio_edit.setDecimals(3)
        self.peak_weak_tail_ratio_edit.setToolTip("弱肩峰过滤倍率；值越大，越容易保留主峰后的弱峰")
        self.peak_gaussian_window_max_edit = QtWidgets.QSpinBox()
        self.peak_gaussian_window_max_edit.setRange(3, 100_000)
        self.peak_gaussian_window_max_edit.setToolTip("自动高斯拟合时允许使用的最大半窗口")
        self.peak_gaussian_boundary_scale_edit = QtWidgets.QDoubleSpinBox()
        self.peak_gaussian_boundary_scale_edit.setRange(0.1, 100)
        self.peak_gaussian_boundary_scale_edit.setDecimals(3)
        self.peak_gaussian_boundary_scale_edit.setToolTip("高斯拟合成功后，以该倍数 FWHM 重新估计卡峰边界")
        self.peak_prominence_ratio_edit = QtWidgets.QDoubleSpinBox()
        self.peak_prominence_ratio_edit.setRange(0, 1)
        self.peak_prominence_ratio_edit.setDecimals(6)
        self.peak_prominence_ratio_edit.setSingleStep(0.001)
        self.peak_prominence_ratio_edit.setToolTip("Prominence阈值占基线校正后最高峰的比例；越大越保守")
        self.peak_smoothing_window_edit = QtWidgets.QSpinBox()
        self.peak_smoothing_window_edit.setRange(1, 100_001)
        self.peak_smoothing_window_edit.setToolTip("Savitzky-Golay平滑窗口，偶数会自动调为奇数")
        self.peak_baseline_window_edit = QtWidgets.QSpinBox()
        self.peak_baseline_window_edit.setRange(3, 1_000_001)
        self.peak_baseline_window_edit.setToolTip("滚动百分位基线窗口，通常应大于峰宽")
        self.peak_baseline_percentile_edit = QtWidgets.QDoubleSpinBox()
        self.peak_baseline_percentile_edit.setRange(0, 100)
        self.peak_baseline_percentile_edit.setDecimals(2)
        self.peak_baseline_percentile_edit.setToolTip("滚动基线使用的百分位数")
        self.peak_min_peak_width_edit = QtWidgets.QSpinBox()
        self.peak_min_peak_width_edit.setRange(1, 1_000_000)
        self.peak_min_peak_width_edit.setToolTip("Prominence/CWT允许的最小峰宽")
        self.peak_max_peak_width_edit = QtWidgets.QSpinBox()
        self.peak_max_peak_width_edit.setRange(1, 1_000_000)
        self.peak_max_peak_width_edit.setToolTip("Prominence/CWT允许的最大峰宽")

        peak_layout.addWidget(QtWidgets.QLabel("寻峰算法"), 0, 0)
        peak_layout.addWidget(self.peak_algorithm_combo, 0, 1)
        peak_layout.addWidget(QtWidgets.QLabel("寻峰起始索引"), 0, 2)
        peak_layout.addWidget(self.peak_detection_min_idx_edit, 0, 3)
        peak_layout.addWidget(QtWidgets.QLabel("最小峰强度"), 0, 4)
        peak_layout.addWidget(self.peak_min_intensity_edit, 0, 5)
        peak_layout.addWidget(QtWidgets.QLabel("峰结束阈值"), 1, 0)
        peak_layout.addWidget(self.peak_threshold_end_edit, 1, 1)
        peak_layout.addWidget(QtWidgets.QLabel("Prominence比例"), 1, 2)
        peak_layout.addWidget(self.peak_prominence_ratio_edit, 1, 3)
        peak_layout.addWidget(QtWidgets.QLabel("平滑窗口"), 1, 4)
        peak_layout.addWidget(self.peak_smoothing_window_edit, 1, 5)
        peak_layout.addWidget(QtWidgets.QLabel("邻峰抑制半宽"), 2, 0)
        peak_layout.addWidget(self.peak_nearby_window_edit, 2, 1)
        peak_layout.addWidget(QtWidgets.QLabel("重复峰排除半宽"), 2, 2)
        peak_layout.addWidget(self.peak_duplicate_window_edit, 2, 3)
        peak_layout.addWidget(QtWidgets.QLabel("边界外扩点数"), 2, 4)
        peak_layout.addWidget(self.peak_boundary_padding_edit, 2, 5)
        peak_layout.addWidget(QtWidgets.QLabel("弱肩峰过滤倍率"), 3, 0)
        peak_layout.addWidget(self.peak_weak_tail_ratio_edit, 3, 1)
        peak_layout.addWidget(QtWidgets.QLabel("高斯窗口上限"), 3, 2)
        peak_layout.addWidget(self.peak_gaussian_window_max_edit, 3, 3)
        peak_layout.addWidget(QtWidgets.QLabel("高斯边界倍数"), 3, 4)
        peak_layout.addWidget(self.peak_gaussian_boundary_scale_edit, 3, 5)
        peak_layout.addWidget(QtWidgets.QLabel("基线窗口"), 4, 0)
        peak_layout.addWidget(self.peak_baseline_window_edit, 4, 1)
        peak_layout.addWidget(QtWidgets.QLabel("基线百分位"), 4, 2)
        peak_layout.addWidget(self.peak_baseline_percentile_edit, 4, 3)
        peak_layout.addWidget(QtWidgets.QLabel("峰宽范围"), 4, 4)
        width_grp = QtWidgets.QWidget()
        width_lay = QtWidgets.QHBoxLayout(width_grp)
        width_lay.setContentsMargins(0, 0, 0, 0)
        width_lay.setSpacing(4)
        width_lay.addWidget(self.peak_min_peak_width_edit)
        width_lay.addWidget(QtWidgets.QLabel("~"))
        width_lay.addWidget(self.peak_max_peak_width_edit)
        peak_layout.addWidget(width_grp, 4, 5)
        for col in (1, 3, 5):
            peak_layout.setColumnStretch(col, 1)
        layout.addWidget(peak_group)
        layout.addStretch()

        root.addWidget(widget, 1)

        # 底部工具栏
        bottom_bar = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom_bar)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(8)
        self.save_button = QtWidgets.QPushButton("💾  保存寻峰参数")
        self.save_button.setFixedHeight(30)
        self.save_button.clicked.connect(self.save_settings)
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("ProjectStatus")
        bl.addWidget(self.save_button)
        bl.addWidget(self.status_label, 1)
        root.addWidget(bottom_bar)

        self.load_from_settings()

    def load_from_settings(self) -> None:
        # 优先从ProjectSettings读取，否则从本地配置读取
        if self.project_settings:
            config = self.project_settings.to_peak_detection_config()
        else:
            config = self.peak_detection

        combo_set_data(self.peak_algorithm_combo, config.algorithm)
        self.peak_detection_min_idx_edit.setValue(config.detection_min_idx)
        self.peak_threshold_end_edit.setValue(config.threshold_end)
        self.peak_min_intensity_edit.setValue(config.min_intensity)
        self.peak_nearby_window_edit.setValue(config.nearby_peak_window)
        self.peak_duplicate_window_edit.setValue(config.duplicate_window)
        self.peak_boundary_padding_edit.setValue(config.boundary_padding)
        self.peak_weak_tail_ratio_edit.setValue(config.weak_tail_ratio)
        self.peak_gaussian_window_max_edit.setValue(config.gaussian_window_max)
        self.peak_gaussian_boundary_scale_edit.setValue(config.gaussian_boundary_scale)
        self.peak_prominence_ratio_edit.setValue(config.prominence_ratio)
        self.peak_smoothing_window_edit.setValue(config.smoothing_window)
        self.peak_baseline_window_edit.setValue(config.baseline_window)
        self.peak_baseline_percentile_edit.setValue(config.baseline_percentile)
        self.peak_min_peak_width_edit.setValue(config.min_peak_width)
        self.peak_max_peak_width_edit.setValue(config.max_peak_width)

    def set_project_settings(self, project_settings: ProjectSettings) -> None:
        """Set the project settings reference for parameter synchronization."""
        self.project_settings = project_settings
        # 立即加载ProjectSettings中的参数
        if project_settings:
            self.load_from_settings()

    def apply_to_settings(self) -> None:
        self.peak_detection = PeakDetectionConfig(
            algorithm=str(self.peak_algorithm_combo.currentData()),
            detection_min_idx=self.peak_detection_min_idx_edit.value(),
            threshold_end=self.peak_threshold_end_edit.value(),
            min_intensity=self.peak_min_intensity_edit.value(),
            nearby_peak_window=self.peak_nearby_window_edit.value(),
            duplicate_window=self.peak_duplicate_window_edit.value(),
            weak_tail_early_window=self.peak_detection.weak_tail_early_window,
            weak_tail_late_window=self.peak_detection.weak_tail_late_window,
            weak_tail_ratio=self.peak_weak_tail_ratio_edit.value(),
            gaussian_window_max=self.peak_gaussian_window_max_edit.value(),
            gaussian_boundary_scale=self.peak_gaussian_boundary_scale_edit.value(),
            boundary_padding=self.peak_boundary_padding_edit.value(),
            prominence_ratio=self.peak_prominence_ratio_edit.value(),
            smoothing_window=self.peak_smoothing_window_edit.value(),
            smoothing_poly_order=self.peak_detection.smoothing_poly_order,
            baseline_window=self.peak_baseline_window_edit.value(),
            baseline_percentile=self.peak_baseline_percentile_edit.value(),
            min_peak_width=self.peak_min_peak_width_edit.value(),
            max_peak_width=max(self.peak_min_peak_width_edit.value(), self.peak_max_peak_width_edit.value()),
        )
        # 保存到ProjectSettings
        if self.project_settings:
            self.project_settings.peak_algorithm = self.peak_detection.algorithm
            self.project_settings.detection_min_idx = self.peak_detection.detection_min_idx
            self.project_settings.threshold_end = self.peak_detection.threshold_end
            self.project_settings.min_intensity = self.peak_detection.min_intensity
            self.project_settings.nearby_peak_window = self.peak_detection.nearby_peak_window
            self.project_settings.duplicate_window = self.peak_detection.duplicate_window
            self.project_settings.weak_tail_ratio = self.peak_detection.weak_tail_ratio
            self.project_settings.gaussian_window_max = self.peak_detection.gaussian_window_max
            self.project_settings.gaussian_boundary_scale = self.peak_detection.gaussian_boundary_scale
            self.project_settings.boundary_padding = self.peak_detection.boundary_padding
            self.project_settings.prominence_ratio = self.peak_detection.prominence_ratio
            self.project_settings.smoothing_window = self.peak_detection.smoothing_window
            self.project_settings.baseline_window = self.peak_detection.baseline_window
            self.project_settings.baseline_percentile = self.peak_detection.baseline_percentile
            self.project_settings.min_peak_width = self.peak_detection.min_peak_width
            self.project_settings.max_peak_width = self.peak_detection.max_peak_width

    def save_settings(self):
        self.apply_to_settings()
        peak_detection_path = save_peak_detection_config(self.peak_detection)
        # 也保存到ProjectSettings
        if self.project_settings:
            from bl03u_masstool.core.project_settings import ProjectSettingsManager, save_project_settings
            ps = ProjectSettingsManager().get()
            ps.peak_algorithm = self.project_settings.peak_algorithm
            ps.detection_min_idx = self.project_settings.detection_min_idx
            ps.threshold_end = self.project_settings.threshold_end
            ps.min_intensity = self.project_settings.min_intensity
            ps.nearby_peak_window = self.project_settings.nearby_peak_window
            ps.duplicate_window = self.project_settings.duplicate_window
            ps.weak_tail_ratio = self.project_settings.weak_tail_ratio
            ps.gaussian_window_max = self.project_settings.gaussian_window_max
            ps.gaussian_boundary_scale = self.project_settings.gaussian_boundary_scale
            ps.boundary_padding = self.project_settings.boundary_padding
            ps.prominence_ratio = self.project_settings.prominence_ratio
            ps.smoothing_window = self.project_settings.smoothing_window
            ps.baseline_window = self.project_settings.baseline_window
            ps.baseline_percentile = self.project_settings.baseline_percentile
            ps.min_peak_width = self.project_settings.min_peak_width
            ps.max_peak_width = self.project_settings.max_peak_width
            save_project_settings(ps)
        self.status_label.setText(f"已保存: {peak_detection_path}")
        self.settings_saved.emit()


class CommonParametersDialog(QtWidgets.QDialog):
    def __init__(self, settings: NormalizationSettings, calibration: Calibration, parent=None):
        super().__init__(parent)
        self.setWindowTitle("通用参数设置")
        self.resize(1280, 760)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.settings_widget = CommonParametersWidget(settings, calibration, self)
        layout.addWidget(self.settings_widget)


# Legacy alias for backward compatibility with core_tools/dialog.py
NormalizationSettingsWidget = CommonParametersWidget
