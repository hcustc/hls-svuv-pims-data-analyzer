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

class NormalizationSettingsWidget(QtWidgets.QWidget, DataFrameTableMixin):
    settings_saved = QtCore.pyqtSignal()

    def __init__(self, settings: NormalizationSettings, calibration: Calibration, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.calibration = calibration
        self.peak_detection = load_peak_detection_config()
        self.worker: WorkerThread | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.light_source_combo = QtWidgets.QComboBox()
        self.light_source_combo.addItem("IO 光电流", "io")
        self.light_source_combo.addItem("Beam Current 储存环束流", "beam_current")

        self.temperature_photon_check = QtWidgets.QCheckBox("温度扫描光强归一化")
        self.temperature_kr_check = QtWidgets.QCheckBox("温度扫描使用Kr膨胀校正")
        self.mass_discrimination_edit = QtWidgets.QDoubleSpinBox()
        self.mass_discrimination_edit.setRange(0.000001, 1_000_000)
        self.mass_discrimination_edit.setDecimals(6)

        self.pie_photon_mode_combo = QtWidgets.QComboBox()
        self.pie_photon_mode_combo.addItem("归一化到首个光强", "first")
        self.pie_photon_mode_combo.addItem("直接除以光强", "none")
        self.pie_photon_mode_combo.addItem("不做光强归一化", "off")

        self.calibration_a_edit = QtWidgets.QDoubleSpinBox()
        self.calibration_b_edit = QtWidgets.QDoubleSpinBox()
        self.calibration_c_edit = QtWidgets.QDoubleSpinBox()
        for edit in (self.calibration_a_edit, self.calibration_b_edit, self.calibration_c_edit):
            edit.setRange(-1_000_000, 1_000_000)
            edit.setDecimals(12)
            edit.setSingleStep(0.000000001)

        self.kr_folder_edit = QtWidgets.QLineEdit()
        self.kr_folder_edit.setPlaceholderText("选择用于计算Kr膨胀系数的温度扫描文件夹")
        self.kr_folder_button = QtWidgets.QPushButton("浏览...")
        self.kr_folder_button.clicked.connect(self.select_kr_folder)
        self.kr_peak_file_edit = QtWidgets.QLineEdit()
        self.kr_peak_file_edit.setPlaceholderText("可选: Kr手动卡峰文件")
        self.kr_peak_file_button = QtWidgets.QPushButton("选择卡峰")
        self.kr_peak_file_button.clicked.connect(self.select_kr_peak_file)
        self.compute_kr_button = QtWidgets.QPushButton("计算Kr膨胀系数")
        self.compute_kr_button.clicked.connect(self.compute_kr_factors)
        self.save_button = QtWidgets.QPushButton("保存通用参数")
        self.save_button.clicked.connect(self.save_settings)
        self.status_label = QtWidgets.QLabel("就绪")

        form.addWidget(QtWidgets.QLabel("光强来源"), 0, 0)
        form.addWidget(self.light_source_combo, 0, 1)
        form.addWidget(self.temperature_photon_check, 0, 2)
        form.addWidget(self.temperature_kr_check, 0, 3)
        form.addWidget(QtWidgets.QLabel("质量歧视因子D"), 1, 0)
        form.addWidget(self.mass_discrimination_edit, 1, 1)
        form.addWidget(QtWidgets.QLabel("PIE光强归一化"), 1, 2)
        form.addWidget(self.pie_photon_mode_combo, 1, 3)
        form.addWidget(QtWidgets.QLabel("定标 A"), 2, 0)
        form.addWidget(self.calibration_a_edit, 2, 1)
        form.addWidget(QtWidgets.QLabel("定标 B"), 2, 2)
        form.addWidget(self.calibration_b_edit, 2, 3)
        form.addWidget(QtWidgets.QLabel("定标 C"), 2, 4)
        form.addWidget(self.calibration_c_edit, 2, 5)
        form.addWidget(QtWidgets.QLabel("Kr定标文件夹"), 3, 0)
        form.addWidget(self.kr_folder_edit, 3, 1, 1, 4)
        form.addWidget(self.kr_folder_button, 3, 5)
        form.addWidget(QtWidgets.QLabel("Kr定标卡峰"), 4, 0)
        form.addWidget(self.kr_peak_file_edit, 4, 1, 1, 4)
        form.addWidget(self.kr_peak_file_button, 4, 5)
        form.addWidget(self.compute_kr_button, 5, 1)
        form.addWidget(self.save_button, 5, 2)
        form.addWidget(self.status_label, 6, 0, 1, 6)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        layout.addLayout(form)

        peak_group = QtWidgets.QGroupBox("自动寻峰参数")
        peak_layout = QtWidgets.QGridLayout(peak_group)
        peak_layout.setHorizontalSpacing(8)
        peak_layout.setVerticalSpacing(8)

        self.peak_algorithm_combo = QtWidgets.QComboBox()
        self.peak_algorithm_combo.addItem("Ensemble融合检测（推荐）", "ensemble")
        self.peak_algorithm_combo.addItem("Prominence", "prominence")
        self.peak_algorithm_combo.addItem("传统局部极大", "legacy")
        self.peak_algorithm_combo.addItem("CWT小波", "cwt")
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
        width_group = QtWidgets.QWidget()
        width_layout = QtWidgets.QHBoxLayout(width_group)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.setSpacing(4)
        width_layout.addWidget(self.peak_min_peak_width_edit)
        width_layout.addWidget(self.peak_max_peak_width_edit)
        peak_layout.addWidget(width_group, 4, 5)
        for column in (1, 3, 5):
            peak_layout.setColumnStretch(column, 1)
        layout.addWidget(peak_group)

        # 元素筛选 (全局参数)
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

        self.factor_table = QtWidgets.QTableWidget()
        layout.addWidget(self.factor_table, stretch=1)
        self.load_from_settings()

    def load_from_settings(self) -> None:
        combo_set_data(self.light_source_combo, self.settings.light_source)
        self.temperature_photon_check.setChecked(self.settings.temperature_photon_normalize)
        self.temperature_kr_check.setChecked(self.settings.temperature_kr_correct)
        self.mass_discrimination_edit.setValue(self.settings.mass_discrimination)
        combo_set_data(self.pie_photon_mode_combo, self.settings.pie_photon_mode)
        calibration = load_calibration_config()
        self.calibration_a_edit.setValue(calibration.a)
        self.calibration_b_edit.setValue(calibration.b)
        self.calibration_c_edit.setValue(calibration.c)
        self.kr_folder_edit.setText(self.settings.kr_calibration_folder)
        self.kr_peak_file_edit.setText(self.settings.kr_calibration_peak_file)
        self.load_peak_detection_controls()
        self.refresh_factor_table()
        # 加载元素筛选
        selected = self.settings.selected_elements
        for elem, chk in self.element_checks.items():
            chk.setChecked(not selected or elem in selected)

    def load_peak_detection_controls(self) -> None:
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

    def apply_to_settings(self) -> None:
        self.settings.light_source = self.light_source_combo.currentData()
        self.settings.temperature_photon_normalize = self.temperature_photon_check.isChecked()
        self.settings.temperature_kr_correct = self.temperature_kr_check.isChecked()
        self.settings.mass_discrimination = self.mass_discrimination_edit.value()
        self.settings.pie_photon_mode = self.pie_photon_mode_combo.currentData()
        self.settings.kr_calibration_folder = self.kr_folder_edit.text().strip()
        self.settings.kr_calibration_peak_file = self.kr_peak_file_edit.text().strip()
        self.calibration = Calibration(
            a=self.calibration_a_edit.value(),
            b=self.calibration_b_edit.value(),
            c=self.calibration_c_edit.value(),
        )
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
        self.settings.selected_elements = [
            elem for elem, chk in self.element_checks.items() if chk.isChecked()
        ]

    def _set_all_elements(self, checked: bool):
        for chk in self.element_checks.values():
            chk.setChecked(checked)

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
        peak_detection_path = save_peak_detection_config(self.peak_detection)
        self.status_label.setText(f"已保存: {normalization_path}；{calibration_path}；{peak_detection_path}")
        self.settings_saved.emit()

    def compute_kr_factors(self):
        self.apply_to_settings()
        folder = self.settings.kr_calibration_folder
        if not folder:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择Kr定标文件夹")
            return
        peak_config = self.peak_detection
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
        ):
            widget.setDisabled(busy)

    def on_kr_factors_ready(self, result: object) -> None:
        df = result
        self.settings.expansion_factors = {
            float(row["temperature"]): float(row["expansion_lambda"])
            for _, row in df.iterrows()
        }
        self.refresh_factor_table()
        save_normalization_settings(self.settings)
        self.status_label.setText(f"已计算 {len(self.settings.expansion_factors)} 个温度点的Kr膨胀系数")

    def on_kr_factors_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "错误", message)

    def refresh_factor_table(self) -> None:
        rows = [
            {"temperature": temperature, "expansion_lambda": value}
            for temperature, value in sorted(self.settings.expansion_factors.items())
        ]
        self.set_dataframe(self.factor_table, pd.DataFrame(rows, columns=["temperature", "expansion_lambda"]))


class CommonParametersDialog(QtWidgets.QDialog):
    def __init__(self, settings: NormalizationSettings, calibration: Calibration, parent=None):
        super().__init__(parent)
        self.setWindowTitle("通用参数设置")
        self.resize(1280, 760)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.settings_widget = NormalizationSettingsWidget(settings, calibration, self)
        layout.addWidget(self.settings_widget)
