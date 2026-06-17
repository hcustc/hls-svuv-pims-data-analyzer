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
from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersDialog

class TemperatureScanDialog(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, calibration: Calibration, normalization_settings: NormalizationSettings | None = None, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.normalization_settings = normalization_settings or NormalizationSettings()
        self.peak_detection = load_peak_detection_config()
        self.result_df = pd.DataFrame()
        self.curves: dict[int, dict] = {}
        self.current_mz: int | None = None
        self.worker: WorkerThread | None = None
        self.setWindowTitle("温度扫描分析")
        self.resize(1180, 760)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(8)
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("选择包含温度扫描 txt 文件的文件夹")
        self.browse_button = QtWidgets.QPushButton("浏览...")
        self.browse_button.setToolTip("选择包含温度扫描txt文件的文件夹")
        self.browse_button.clicked.connect(self.select_folder)
        self.run_button = QtWidgets.QPushButton("开始分析")
        self.run_button.setToolTip("开始分析温度扫描数据，生成温度-信号曲线")
        self.run_button.clicked.connect(self.run_analysis)
        self.export_button = QtWidgets.QPushButton("导出")
        self.export_button.setToolTip("导出温度扫描分析结果为CSV文件")
        self.export_button.clicked.connect(self.export_result)
        self.common_params_button = QtWidgets.QPushButton("通用参数")
        self.common_params_button.setToolTip("打开通用参数设置（光强归一化、Kr定标、寻峰参数等）")
        self.common_params_button.clicked.connect(self.open_common_parameters)
        self.peak_source_combo = QtWidgets.QComboBox()
        self.peak_source_combo.setToolTip("自动寻峰：程序自动检测峰位；手动卡峰：使用预先标定的峰文件")
        self.peak_source_combo.addItem("自动寻峰", "auto")
        self.peak_source_combo.addItem("手动卡峰", "manual")
        self.peak_file_edit = QtWidgets.QLineEdit()
        self.peak_file_edit.setPlaceholderText("可选: yaml/csv/xlsx 手动卡峰文件")
        self.select_peak_file_button = QtWidgets.QPushButton("选择卡峰")
        self.select_peak_file_button.setToolTip("选择手动卡峰文件（支持yaml/csv/xlsx格式）")
        self.select_peak_file_button.clicked.connect(self.select_peak_file)

        self.threshold_end_edit = QtWidgets.QDoubleSpinBox()
        self.threshold_end_edit.setRange(0, 1_000_000)
        self.threshold_end_edit.setDecimals(3)
        self.threshold_end_edit.setValue(self.peak_detection.threshold_end)
        self.min_intensity_edit = QtWidgets.QDoubleSpinBox()
        self.min_intensity_edit.setRange(0, 1_000_000)
        self.min_intensity_edit.setDecimals(3)
        self.min_intensity_edit.setValue(self.peak_detection.min_intensity)
        self.reference_mode_combo = QtWidgets.QComboBox()
        self.reference_mode_combo.setToolTip("累加谱寻峰：所有温度累加后统一寻峰；最高温谱寻峰：用最高温度谱独立寻峰")
        self.reference_mode_combo.addItem("累加谱寻峰", "sum")
        self.reference_mode_combo.addItem("最高温谱寻峰", "max_temperature")
        self.gaussian_check = QtWidgets.QCheckBox("高斯积分")
        self.gaussian_check.setToolTip("使用高斯峰面积而非简单峰值强度作为信号量，更准确反映积分强度")
        self.gaussian_check.setChecked(True)
        self.status_label = QtWidgets.QLabel("就绪")

        controls_layout.addWidget(QtWidgets.QLabel("温度扫描文件夹"), 0, 0)
        controls_layout.addWidget(self.folder_edit, 0, 1, 1, 4)
        controls_layout.addWidget(self.browse_button, 0, 5)
        controls_layout.addWidget(self.run_button, 0, 6)
        controls_layout.addWidget(self.export_button, 0, 7)
        controls_layout.addWidget(self.common_params_button, 0, 8)
        controls_layout.addWidget(QtWidgets.QLabel("参考峰来源"), 1, 0)
        controls_layout.addWidget(self.reference_mode_combo, 1, 1)
        controls_layout.addWidget(self.gaussian_check, 1, 2)
        controls_layout.addWidget(QtWidgets.QLabel("自动寻峰与归一化参数在“通用参数”页管理"), 1, 3, 1, 4)
        controls_layout.addWidget(QtWidgets.QLabel("卡峰来源"), 2, 0)
        controls_layout.addWidget(self.peak_source_combo, 2, 1)
        controls_layout.addWidget(self.peak_file_edit, 2, 2, 1, 4)
        controls_layout.addWidget(self.select_peak_file_button, 2, 6)
        controls_layout.addWidget(self.status_label, 3, 0, 1, 9)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(4, 1)
        layout.addWidget(controls)

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

        # 内联状态提示
        self.inline_status_bar = QtWidgets.QWidget()
        inline_layout = QtWidgets.QHBoxLayout(self.inline_status_bar)
        inline_layout.setContentsMargins(0, 0, 0, 0)
        inline_layout.setSpacing(6)
        self.inline_status_icon = QtWidgets.QLabel("")
        self.inline_status_text = QtWidgets.QLabel("就绪")
        self.inline_retry_button = QtWidgets.QPushButton("重试")
        self.inline_retry_button.setMaximumWidth(60)
        self.inline_retry_button.hide()
        self.inline_action_hint = QtWidgets.QLabel('请选择温度扫描文件夹，点击"开始分析"')
        self.inline_action_hint.setStyleSheet("color: #6b7280;")
        inline_layout.addWidget(self.inline_status_icon)
        inline_layout.addWidget(self.inline_status_text, stretch=1)
        inline_layout.addWidget(self.inline_retry_button)
        inline_layout.addStretch(2)
        layout.addWidget(self.inline_status_bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        self.summary_label = QtWidgets.QLabel("未生成温度曲线")
        left_layout.addWidget(self.summary_label)
        filter_row = QtWidgets.QWidget()
        filter_layout = QtWidgets.QHBoxLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(6)
        filter_layout.addWidget(QtWidgets.QLabel("显示"))
        self.curve_display_combo = QtWidgets.QComboBox()
        self.curve_display_combo.addItem("按类别分组", "grouped")
        self.curve_display_combo.addItem("按m/z排序", "mz")
        self.curve_display_combo.currentIndexChanged.connect(self.populate_mz_list)
        filter_layout.addWidget(self.curve_display_combo, stretch=1)
        filter_layout.addWidget(QtWidgets.QLabel("筛选"))
        self.curve_group_combo = QtWidgets.QComboBox()
        self.curve_group_combo.addItem("全部", "all")
        for key in ("formation", "consumption", "intermediate", "unclassified"):
            self.curve_group_combo.addItem(TEMPERATURE_CURVE_CLASS_LABELS[key], key)
        self.curve_group_combo.currentIndexChanged.connect(self.populate_mz_list)
        filter_layout.addWidget(self.curve_group_combo, stretch=1)
        left_layout.addWidget(filter_row)
        self.group_summary_label = QtWidgets.QLabel("")
        self.group_summary_label.setWordWrap(True)
        left_layout.addWidget(self.group_summary_label)
        self.mz_list = QtWidgets.QTreeWidget()
        self.mz_list.setHeaderHidden(True)
        self.mz_list.setRootIsDecorated(True)
        self.mz_list.setUniformRowHeights(True)
        self.mz_list.currentItemChanged.connect(self.on_mz_selected)
        left_layout.addWidget(self.mz_list, stretch=1)
        splitter.addWidget(left_panel)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        if pg is not None:
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground("#ffffff")
            self.plot_widget.setLabel("bottom", "Temperature (C)")
            self.plot_widget.setLabel("left", "Normalized Area")
            self.plot_widget.getAxis("bottom").enableAutoSIPrefix(False)
            self.plot_widget.showGrid(x=True, y=True)
            right_splitter.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            right_splitter.addWidget(QtWidgets.QLabel("未安装 pyqtgraph，无法显示温度扫描曲线图"))

        self.curve_table = QtWidgets.QTableWidget()
        self.table = QtWidgets.QTableWidget()
        for table in (self.curve_table, self.table):
            table.setWordWrap(False)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.detail_tabs = QtWidgets.QTabWidget()
        self.detail_tabs.addTab(self.curve_table, "当前曲线")
        self.detail_tabs.addTab(self.table, "全部积分结果")
        right_splitter.addWidget(self.detail_tabs)
        right_splitter.setSizes([500, 220])
        splitter.addWidget(right_splitter)
        splitter.setSizes([260, 920])
        layout.addWidget(splitter, stretch=1)

    def select_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择温度扫描文件夹")
        if folder:
            self.folder_edit.setText(folder)

    def select_peak_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择手动卡峰文件",
            "",
            "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;All Files (*)",
        )
        if path:
            self.peak_file_edit.setText(path)
            self.peak_source_combo.setCurrentIndex(1)

    def set_project_settings(self, ps: ProjectSettings) -> None:
        """Apply ProjectSettings defaults to TemperatureScanDialog inline controls."""
        self.project_settings = ps
        if ps.temperature_scan_folder:
            self.folder_edit.setText(ps.temperature_scan_folder)
        idx = self.reference_mode_combo.findData(ps.temp_reference_mode)
        if idx >= 0:
            self.reference_mode_combo.setCurrentIndex(idx)
        self.gaussian_check.setChecked(ps.temp_prefer_gaussian)
        if hasattr(self, "spin_kr_mz"):
            self.spin_kr_mz.setValue(ps.temp_kr_mz)
        # 更新摘要栏
        if hasattr(self, "summary_project_label"):
            project_name = ps.project_name or "---"
            system = ps.system or "---"
            self.summary_project_label.setText(f"项目: {project_name}")
            self.summary_system_label.setText(f"体系: {system}")
            data_path = ps.temperature_scan_folder or "---"
            self.summary_data_label.setText(f"数据源: {data_path}")

    def _open_project_settings(self):
        """跳转到项目管理页面。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("project")

    def _show_inline_error(self, msg: str, retry_callback=None):
        self.inline_status_icon.setText("\u26a0\ufe0f")
        self.inline_status_text.setText(msg)
        self.inline_status_text.setStyleSheet("color: #dc2626;")
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
        self.inline_status_text.setStyleSheet("color: #16a34a;")
        self.inline_action_hint.hide()
        self.inline_retry_button.hide()
        QtCore.QTimer.singleShot(5000, self._clear_inline_status)

    def _show_inline_empty(self, msg: str = ""):
        self.inline_status_icon.setText("")
        self.inline_status_text.setText("就绪")
        self.inline_status_text.setStyleSheet("")
        if msg:
            self.inline_action_hint.setText(msg)
            self.inline_action_hint.show()
        else:
            self.inline_action_hint.hide()
        self.inline_retry_button.hide()

    def _clear_inline_status(self):
        self.inline_status_icon.setText("")
        self.inline_status_text.setText("就绪")
        self.inline_status_text.setStyleSheet("")
        self.inline_action_hint.hide()
        self.inline_retry_button.hide()

    def open_common_parameters(self):
        dialog = CommonParametersDialog(self.normalization_settings, self.calibration, self)
        dialog.exec()
        self.calibration = load_calibration_config()

    def run_analysis(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            self._show_inline_error("请先选择温度扫描文件夹", lambda: self.select_folder())
            return
        peak_config = load_peak_detection_config()
        threshold_end = peak_config.threshold_end
        min_intensity = peak_config.min_intensity
        reference_mode = self.reference_mode_combo.currentData()
        prefer_gaussian = self.gaussian_check.isChecked()
        manual_peak_path = self.peak_file_edit.text().strip() if self.peak_source_combo.currentData() == "manual" else None
        if self.peak_source_combo.currentData() == "manual" and not manual_peak_path:
            self._show_inline_error("请选择手动卡峰文件", lambda: self.select_peak_file())
            return
        settings = self.normalization_settings
        photon_normalize = settings.temperature_photon_normalize
        kr_correct = settings.temperature_kr_correct
        kr_mz = self.spin_kr_mz.value() if hasattr(self, "spin_kr_mz") else 84
        mass_discrimination = settings.mass_discrimination
        light_source = settings.light_source
        expansion_factors = settings.expansion_factors if kr_correct else None
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
            ),
            self,
        )
        self.worker.finished_with_result.connect(self.on_analysis_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def compute_kr_expansion(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择文件夹")
            return
        peak_config = load_peak_detection_config()
        threshold_end = peak_config.threshold_end
        min_intensity = peak_config.min_intensity
        reference_mode = self.reference_mode_combo.currentData()
        prefer_gaussian = self.gaussian_check.isChecked()
        manual_peak_path = self.peak_file_edit.text().strip() if self.peak_source_combo.currentData() == "manual" else None
        if self.peak_source_combo.currentData() == "manual" and not manual_peak_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择手动卡峰文件")
            return
        settings = self.normalization_settings
        light_source = settings.light_source
        kr_mz = self.spin_kr_mz.value() if hasattr(self, "spin_kr_mz") else 84
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
        expansion_factors = {float(t): float(lam) for t, lam in zip(result["temperature"], result["expansion_lambda"])}
        settings.expansion_factors = expansion_factors
        settings.temperature_kr_correct = True
        save_normalization_settings(settings)
        QtWidgets.QMessageBox.information(
            self, "完成",
            f"成功计算 Kr 膨胀系数！\n参考温度: {result['reference_temperature'].iloc[0]:.1f}°C\n共 {len(result)} 个温度点\n\n已启用 Kr 校正，将自动重新分析温度扫描数据",
        )
        if self.worker is not None:
            try:
                self.worker.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
        self.run_analysis()


    def set_busy(self, busy: bool, message: str) -> None:
        self.status_label.setText(message)
        self.run_button.setDisabled(busy)
        self.browse_button.setDisabled(busy)
        self.export_button.setDisabled(busy)
        self.common_params_button.setDisabled(busy)
        self.reference_mode_combo.setDisabled(busy)
        self.gaussian_check.setDisabled(busy)
        self.peak_source_combo.setDisabled(busy)
        self.peak_file_edit.setDisabled(busy)
        self.select_peak_file_button.setDisabled(busy)

    def on_analysis_complete(self, result: object) -> None:
        self.result_df = result
        self.curves = build_temperature_curves(self.result_df)
        self.set_dataframe(self.table, self.result_df)
        self.populate_mz_list()
        temperature_count = self.result_df["temperature"].nunique() if not self.result_df.empty else 0
        self.summary_label.setText(f"{len(self.curves)} 条m/z曲线 | {temperature_count} 个温度点")
        self.update_group_summary()
        self._show_inline_success(f"已完成分析，生成 {len(self.result_df)} 行温度扫描结果")

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
            ]
            if not items:
                continue
            parent = QtWidgets.QTreeWidgetItem([f"{TEMPERATURE_CURVE_CLASS_LABELS[group_key]} ({len(items)}条)"])
            parent.setData(0, QtCore.Qt.ItemDataRole.UserRole, None)
            parent.setFlags(parent.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.mz_list.addTopLevelItem(parent)
            for mz, curve in items:
                child = QtWidgets.QTreeWidgetItem([self.curve_tree_label(mz, curve, include_group=False)])
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, mz)
                parent.addChild(child)
            parent.setExpanded(True)

    def populate_mz_list_flat(self, selected_group: str):
        for mz in sorted(self.curves):
            curve = self.curves[mz]
            if selected_group != "all" and curve.get("curve_class") != selected_group:
                continue
            item = QtWidgets.QTreeWidgetItem([self.curve_tree_label(mz, curve, include_group=True)])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, mz)
            self.mz_list.addTopLevelItem(item)

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
            if self.plot_widget is not None:
                self.plot_widget.clear()
            return
        mz_value = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if mz_value is None:
            if current.childCount() > 0:
                self.mz_list.setCurrentItem(current.child(0))
            return
        self.current_mz = int(mz_value)
        curve = self.curves[self.current_mz]
        rows = curve["rows"].copy()
        curve_df = pd.DataFrame(
            {
                "分类": [curve.get("curve_class_label", "")] * len(rows),
                "温度(C)": np.round(rows["temperature"].astype(float), 4),
                "原始积分": np.round(rows["raw_area"].astype(float), 4),
                "IO归一化": np.round(rows["photon_normalized_area"].astype(float), 4),
                "λ(T)": np.round(rows["expansion_lambda"].astype(float), 6),
                "最终强度": np.round(rows["area"].astype(float), 4),
            }
        )
        self.set_dataframe(self.curve_table, curve_df)
        self.update_plot(curve)

    def update_plot(self, curve: dict):
        if self.plot_widget is None:
            return
        x_values = np.asarray(curve["temperatures"], dtype=float)
        y_values = np.asarray(curve["areas"], dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid]
        y_values = y_values[valid]
        self.plot_widget.clear()
        self.plot_widget.setLabel("bottom", "Temperature (C)")
        self.plot_widget.setLabel("left", "Normalized Area")
        self.plot_widget.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        if x_values.size == 0:
            self.plot_widget.setTitle(f"m/z {curve['mz']} 温度曲线 - 无有效数据")
            return
        self.plot_widget.plot(
            x_values,
            y_values,
            pen=pg.mkPen("#2563eb", width=2),
            symbol="o",
            symbolBrush="#2563eb",
            symbolPen="#1e3a8a",
            symbolSize=8,
        )
        self.plot_widget.setTitle(f"m/z {curve['mz']} 温度扫描 - {curve.get('curve_class_label', '')}")
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        x_pad = max(5.0, (x_max - x_min) * 0.08)
        y_pad = max(1.0, (y_max - y_min) * 0.12)
        if x_min == x_max:
            x_min -= 5.0
            x_max += 5.0
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        self.plot_widget.setXRange(x_min - x_pad, x_max + x_pad, padding=0)
        self.plot_widget.setYRange(max(0.0, y_min - y_pad), y_max + y_pad, padding=0)

    def run_analysis_sync(
        self,
        folder: str,
        threshold_end: float,
        min_intensity: float,
        *,
        prefer_gaussian: bool = True,
        reference_mode: str = "sum",
        manual_peak_path: str | None = None,
        photon_normalize: bool = False,
        kr_correct: bool = False,
        mass_discrimination: float = 1.0,
        light_source: str = "io",
        expansion_factors: dict[float, float] | None = None,
    ) -> pd.DataFrame:
        peak_config = load_peak_detection_config()
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
            reference_mode=reference_mode,
            manual_peak_path=manual_peak_path,
            photon_normalize=photon_normalize,
            kr_correct=kr_correct,
            mass_discrimination=mass_discrimination,
            light_source=light_source,
            expansion_factors=expansion_factors,
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
