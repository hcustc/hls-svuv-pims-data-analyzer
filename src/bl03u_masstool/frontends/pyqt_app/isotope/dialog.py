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
    ISOTOPES,
    MONOISOTOPIC_MASSES,
    calculate_isotope_distribution,
    formula_monoisotopic_mass,
    formula_nominal_mass,
    generate_formula_candidates,
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

class IsotopeAbundanceDialog(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.candidate_worker: WorkerThread | None = None
        self.project_settings: ProjectSettings | None = None
        self.setWindowTitle("分子式与质量分析")
        self.resize(920, 680)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        mass_tab = QtWidgets.QWidget()
        mass_layout = QtWidgets.QVBoxLayout(mass_tab)
        mass_layout.setContentsMargins(8, 8, 8, 8)
        mass_layout.setSpacing(10)

        formula_group = QtWidgets.QGroupBox("输入分子式")
        formula_layout = QtWidgets.QGridLayout(formula_group)
        formula_layout.setHorizontalSpacing(8)
        formula_layout.setVerticalSpacing(8)
        self.formula_edit = QtWidgets.QLineEdit()
        self.formula_edit.setPlaceholderText("输入分子式；支持括号、同位素标记和结晶水写法")
        self.formula_edit.setAccessibleName("待分析分子式")
        self.formula_edit.returnPressed.connect(self.calculate)
        self.min_percent_edit = QtWidgets.QDoubleSpinBox()
        self.min_percent_edit.setRange(0.0, 100.0)
        self.min_percent_edit.setDecimals(4)
        self.min_percent_edit.setSingleStep(0.01)
        self.min_percent_edit.setValue(0.01)
        self.min_percent_edit.setSuffix(" %")
        self.min_percent_edit.setToolTip("低于该相对丰度的同位素峰不显示")
        self.calculate_button = QtWidgets.QPushButton("开始分析")
        self.calculate_button.setObjectName("PrimaryButton")
        self.calculate_button.setToolTip("计算名义质量、单同位素精确质量和同位素分布")
        self.calculate_button.clicked.connect(self.calculate)
        self.clear_formula_button = QtWidgets.QPushButton("清空")
        self.clear_formula_button.setObjectName("BrowseButton")
        self.clear_formula_button.clicked.connect(self.clear_formula_results)
        formula_layout.addWidget(QtWidgets.QLabel("分子式"), 0, 0)
        formula_layout.addWidget(self.formula_edit, 0, 1)
        formula_layout.addWidget(self.calculate_button, 0, 2)
        formula_layout.addWidget(self.clear_formula_button, 0, 3)
        formula_layout.addWidget(QtWidgets.QLabel("同位素峰最小丰度"), 1, 0)
        formula_layout.addWidget(self.min_percent_edit, 1, 1)
        threshold_hint = QtWidgets.QLabel("仅影响同位素峰列表，不影响质量计算")
        threshold_hint.setObjectName("HintLabel")
        formula_layout.addWidget(threshold_hint, 1, 2, 1, 2)
        formula_layout.setColumnStretch(1, 1)
        mass_layout.addWidget(formula_group)

        summary_frame = QtWidgets.QWidget(mass_tab)
        summary_frame.setObjectName("ProjectActionBar")
        summary_layout = QtWidgets.QGridLayout(summary_frame)
        summary_layout.setContentsMargins(10, 7, 10, 7)
        summary_layout.setHorizontalSpacing(18)
        summary_layout.setVerticalSpacing(4)

        self.composition_label = QtWidgets.QLabel("元素组成：—")
        self.composition_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.mass_label = QtWidgets.QLabel("名义质量：—    单同位素精确质量：—")
        self.mass_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.isotope_peak_count_label = QtWidgets.QLabel("显示峰数：0")
        self.formula_status_label = QtWidgets.QLabel("输入分子式后开始分析")
        self.formula_status_label.setObjectName("InlineStatusLabel")
        summary_layout.addWidget(self.composition_label, 0, 0)
        summary_layout.addWidget(self.mass_label, 0, 1)
        summary_layout.addWidget(self.isotope_peak_count_label, 0, 2)
        summary_layout.addWidget(self.formula_status_label, 1, 0, 1, 3)
        summary_layout.setColumnStretch(1, 1)
        mass_layout.addWidget(summary_frame)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["同位素质量 (Da)", "绝对丰度", "相对丰度 (%)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        mass_layout.addWidget(self.table, stretch=1)
        self.tabs.addTab(mass_tab, "分子式 → 质量与同位素")

        candidate_tab = QtWidgets.QWidget()
        candidate_layout = QtWidgets.QVBoxLayout(candidate_tab)
        candidate_layout.setContentsMargins(8, 8, 8, 8)
        candidate_layout.setSpacing(10)

        search_group = QtWidgets.QGroupBox("输入质量并设置搜索范围")
        search_layout = QtWidgets.QGridLayout(search_group)
        search_layout.setHorizontalSpacing(8)
        search_layout.setVerticalSpacing(8)
        self.mass_mode_combo = QtWidgets.QComboBox()
        self.mass_mode_combo.addItem("名义质量数（整数）", "nominal")
        self.mass_mode_combo.addItem("单同位素精确质量", "monoisotopic")
        self.mass_mode_combo.setToolTip("名义质量适用于整数质量数；精确质量适用于高分辨质谱")
        self.target_mass_edit = QtWidgets.QDoubleSpinBox()
        self.target_mass_edit.setRange(0.000001, 100_000)
        self.target_mass_edit.setDecimals(0)
        self.target_mass_edit.setValue(128)
        self.target_mass_edit.setSingleStep(1)
        self.target_mass_edit.setSuffix(" Da")
        self.tolerance_edit = QtWidgets.QDoubleSpinBox()
        self.tolerance_edit.setRange(0.0, 1_000_000)
        self.tolerance_edit.setDecimals(0)
        self.tolerance_edit.setValue(0)
        self.tolerance_unit_combo = QtWidgets.QComboBox()
        self.tolerance_unit_combo.addItem("Da", "Da")
        self.tolerance_unit_combo.addItem("ppm", "ppm")
        self.tolerance_unit_combo.setEnabled(False)
        element_selector = QtWidgets.QWidget(search_group)
        element_selector.setObjectName("ModeSegment")
        element_selector.setToolTip("勾选允许参与组合的元素；各元素最大原子数会根据目标质量自动限制")
        element_selector_layout = QtWidgets.QHBoxLayout(element_selector)
        element_selector_layout.setContentsMargins(4, 2, 4, 2)
        element_selector_layout.setSpacing(6)
        self.element_checks: dict[str, QtWidgets.QCheckBox] = {}
        for element in COMMON_ELEMENTS:
            check = QtWidgets.QCheckBox(element, element_selector)
            check.setChecked(element in {"C", "H", "N", "O"})
            check.setToolTip(f"在候选分子式中允许元素 {element}")
            self.element_checks[element] = check
            element_selector_layout.addWidget(check)
        element_selector_layout.addStretch(1)
        self.element_source_label = QtWidgets.QLabel("当前页面选择")
        self.element_source_label.setObjectName("HintLabel")
        self.max_results_edit = QtWidgets.QSpinBox()
        self.max_results_edit.setRange(1, 10_000)
        self.max_results_edit.setValue(200)
        self.chemical_rules_check = QtWidgets.QCheckBox("过滤明显不合理的分子式")
        self.chemical_rules_check.setChecked(True)
        self.chemical_rules_check.setToolTip("使用基础不饱和度规则过滤氢数过高等明显不合理的候选；特殊体系可关闭")
        self.generate_button = QtWidgets.QPushButton("搜索候选分子式")
        self.generate_button.setObjectName("PrimaryButton")
        self.generate_button.clicked.connect(self.generate_candidates)
        self.clear_candidates_button = QtWidgets.QPushButton("清空结果")
        self.clear_candidates_button.setObjectName("BrowseButton")
        self.clear_candidates_button.clicked.connect(self.clear_candidate_results)
        self.mass_mode_combo.currentIndexChanged.connect(self._on_mass_mode_changed)

        search_layout.addWidget(QtWidgets.QLabel("质量类型"), 0, 0)
        search_layout.addWidget(self.mass_mode_combo, 0, 1)
        search_layout.addWidget(QtWidgets.QLabel("目标质量"), 0, 2)
        search_layout.addWidget(self.target_mass_edit, 0, 3)
        search_layout.addWidget(QtWidgets.QLabel("允许误差"), 0, 4)
        search_layout.addWidget(self.tolerance_edit, 0, 5)
        search_layout.addWidget(self.tolerance_unit_combo, 0, 6)
        search_layout.addWidget(QtWidgets.QLabel("元素类型"), 1, 0)
        search_layout.addWidget(element_selector, 1, 1, 1, 6)
        search_layout.addWidget(QtWidgets.QLabel("最多结果"), 2, 0)
        search_layout.addWidget(self.max_results_edit, 2, 1)
        search_layout.addWidget(self.element_source_label, 2, 2, 1, 2)
        search_layout.addWidget(self.chemical_rules_check, 2, 4, 1, 3)
        search_layout.addWidget(self.generate_button, 3, 5)
        search_layout.addWidget(self.clear_candidates_button, 3, 6)
        search_layout.setColumnStretch(3, 1)
        search_layout.setColumnStretch(4, 1)
        candidate_layout.addWidget(search_group)

        self.candidate_status_label = QtWidgets.QLabel("输入质量后搜索；双击候选可查看其质量与同位素分布")
        self.candidate_status_label.setObjectName("InlineStatusLabel")
        candidate_layout.addWidget(self.candidate_status_label)

        self.candidate_formula_table = QtWidgets.QTableWidget()
        self.candidate_formula_table.setColumnCount(7)
        self.candidate_formula_table.setHorizontalHeaderLabels(
            ["分子式", "名义质量", "单同位素质量", "误差 (Da)", "误差 (ppm)", "DBE", "元素组成"]
        )
        self.candidate_formula_table.setAlternatingRowColors(True)
        self.candidate_formula_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_formula_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.candidate_formula_table.itemDoubleClicked.connect(self.use_candidate_formula)
        candidate_layout.addWidget(self.candidate_formula_table, stretch=1)
        self.tabs.addTab(candidate_tab, "质量数 → 候选分子式")

    @staticmethod
    def _set_status(label: QtWidgets.QLabel, text: str, status: str = "") -> None:
        label.setText(text)
        label.setProperty("status", status)
        label.style().unpolish(label)
        label.style().polish(label)

    def clear_formula_results(self) -> None:
        self.formula_edit.clear()
        self.composition_label.setText("元素组成：—")
        self.mass_label.setText("名义质量：—    单同位素精确质量：—")
        self.isotope_peak_count_label.setText("显示峰数：0")
        self.table.setRowCount(0)
        self._set_status(self.formula_status_label, "输入分子式后开始分析")
        self.formula_edit.setFocus()

    def clear_candidate_results(self) -> None:
        self.candidate_formula_table.setRowCount(0)
        self._set_status(
            self.candidate_status_label,
            "输入质量后搜索；双击候选可查看其质量与同位素分布",
        )

    def _on_mass_mode_changed(self, _index: int) -> None:
        mode = str(self.mass_mode_combo.currentData())
        exact_mode = mode == "monoisotopic"
        if exact_mode:
            self.target_mass_edit.setDecimals(6)
            self.target_mass_edit.setSingleStep(0.001)
            self.tolerance_edit.setDecimals(3)
            if self.tolerance_edit.value() == 0:
                self.tolerance_edit.setValue(5.0)
            self.tolerance_unit_combo.setCurrentIndex(1)
            self.tolerance_unit_combo.setEnabled(True)
        else:
            self.target_mass_edit.setDecimals(0)
            self.target_mass_edit.setSingleStep(1)
            self.target_mass_edit.setValue(round(self.target_mass_edit.value()))
            self.tolerance_edit.setDecimals(0)
            self.tolerance_edit.setSingleStep(1)
            self.tolerance_edit.setValue(0)
            self.tolerance_unit_combo.setCurrentIndex(0)
            self.tolerance_unit_combo.setEnabled(False)

    def set_project_settings(self, project_settings: ProjectSettings | None) -> None:
        self.project_settings = project_settings
        selected = [
            element
            for element in (project_settings.selected_elements if project_settings else [])
            if element in self.element_checks
        ]
        if not selected:
            selected = ["C", "H", "N", "O"]
            source_text = "当前页面默认元素"
        else:
            source_text = "来自项目管理：" + "、".join(selected)
        for element, check in self.element_checks.items():
            check.setChecked(element in selected)
        self.element_source_label.setText(source_text)
        self.element_source_label.setToolTip(source_text)

    def _selected_elements(self) -> list[str]:
        return [element for element, check in self.element_checks.items() if check.isChecked()]

    def _candidate_element_ranges(self, target_mass: float, tolerance: float, tolerance_unit: str) -> dict[str, tuple[int, int]]:
        selected = self._selected_elements()
        if not selected:
            raise ValueError("请至少选择一种参与搜索的元素")
        tolerance_da = tolerance if tolerance_unit.lower() == "da" else target_mass * tolerance / 1_000_000
        upper_mass = max(target_mass + tolerance_da, target_mass)
        caps = {"H": 60, "C": 30, "N": 20, "O": 20, "F": 20, "Cl": 10, "Br": 6, "I": 4, "S": 10, "P": 10, "Si": 10}
        nominal_mode = str(self.mass_mode_combo.currentData()) == "nominal"
        ranges: dict[str, tuple[int, int]] = {}
        for element in selected:
            atom_mass = (
                float(max(ISOTOPES[element], key=lambda item: item[1])[0])
                if nominal_mode
                else MONOISOTOPIC_MASSES[element]
            )
            maximum = min(caps.get(element, 10), max(0, int(upper_mass // atom_mass)))
            ranges[element] = (0, maximum)
        return ranges

    def calculate(self):
        formula = self.formula_edit.text().strip()
        if not formula:
            self._set_status(self.formula_status_label, "请输入分子式", "error")
            self.formula_edit.setFocus()
            return
        try:
            composition = parse_formula(formula)
            nominal_mass = formula_nominal_mass(formula)
            exact_mass = formula_monoisotopic_mass(formula)
            rows = calculate_isotope_distribution(formula, min_percent=self.min_percent_edit.value())
        except Exception as exc:
            self._set_status(self.formula_status_label, f"无法解析分子式：{exc}", "error")
            return

        self.composition_label.setText("元素组成：" + "、".join(f"{key} × {value}" for key, value in composition.items()))
        self.mass_label.setText(f"名义质量：{nominal_mass}    单同位素精确质量：{exact_mass:.8f} Da")
        self.isotope_peak_count_label.setText(f"显示峰数：{len(rows)}")
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            self.table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(f"{row['mass']:.4f}"))
            self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{row['abundance']:.8g}"))
            self.table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(f"{row['percent']:.4f}"))
        self._set_status(self.formula_status_label, f"已完成 {formula} 的质量与同位素分析", "success")

    def generate_candidates(self):
        target_mass = self.target_mass_edit.value()
        tolerance = self.tolerance_edit.value()
        tolerance_unit = str(self.tolerance_unit_combo.currentData())
        mass_mode = str(self.mass_mode_combo.currentData())
        max_results = self.max_results_edit.value()
        apply_chemical_rules = self.chemical_rules_check.isChecked()
        try:
            ranges = self._candidate_element_ranges(target_mass, tolerance, tolerance_unit)
        except Exception as exc:
            self._set_status(self.candidate_status_label, str(exc), "error")
            return
        self.set_candidate_busy(True)
        mode_label = "名义质量数" if mass_mode == "nominal" else "单同位素精确质量"
        self._set_status(self.candidate_status_label, f"正在按{mode_label}搜索候选分子式…", "busy")
        self.candidate_worker = WorkerThread(
            lambda: generate_formula_candidates(
                target_mass,
                tolerance=tolerance,
                tolerance_unit=tolerance_unit,
                element_ranges=ranges,
                max_results=max_results,
                mass_mode=mass_mode,
                apply_chemical_rules=apply_chemical_rules,
            ),
            self,
        )
        self.candidate_worker.finished_with_result.connect(self.show_formula_candidates)
        self.candidate_worker.failed.connect(self.show_candidate_error)
        self.candidate_worker.finished.connect(lambda: self.set_candidate_busy(False))
        self.candidate_worker.start()

    def set_candidate_busy(self, busy: bool) -> None:
        for widget in (
            self.mass_mode_combo,
            self.target_mass_edit,
            self.tolerance_edit,
            self.tolerance_unit_combo,
            self.max_results_edit,
            self.chemical_rules_check,
            self.generate_button,
        ):
            widget.setDisabled(busy)
        for check in self.element_checks.values():
            check.setDisabled(busy)
        self.generate_button.setText("正在搜索…" if busy else "搜索候选分子式")

    def show_candidate_error(self, message: str) -> None:
        self._set_status(self.candidate_status_label, f"候选搜索失败：{message}", "error")

    def show_formula_candidates(self, candidates: object) -> None:
        candidate_list = list(candidates if isinstance(candidates, list) else [])

        rows = [
            {
                "分子式": item["formula"],
                "名义质量": item["nominal_mass"],
                "单同位素质量": f"{item['monoisotopic_mass']:.8f}",
                "误差(Da)": f"{item['error_da']:.8f}",
                "误差(ppm)": f"{item['error_ppm']:.3f}",
                "DBE": "—" if item.get("dbe") is None else f"{item['dbe']:.1f}",
                "元素组成": ", ".join(f"{key}{value}" for key, value in item["composition"].items()),
            }
            for item in candidate_list
        ]
        self.set_dataframe(
            self.candidate_formula_table,
            pd.DataFrame(
                rows,
                columns=["分子式", "名义质量", "单同位素质量", "误差(Da)", "误差(ppm)", "DBE", "元素组成"],
            ),
        )
        header = self.candidate_formula_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeMode.Stretch)
        if candidate_list:
            self._set_status(
                self.candidate_status_label,
                f"找到 {len(candidate_list)} 个候选；结果按质量误差排序，双击可查看详细分析",
                "success",
            )
        else:
            self._set_status(
                self.candidate_status_label,
                "当前质量、误差和元素范围下没有候选，请放宽误差或调整元素范围",
                "error",
            )

    def use_candidate_formula(self, item: QtWidgets.QTableWidgetItem | None = None) -> None:
        row = self.candidate_formula_table.currentRow()
        if row < 0:
            return
        formula_item = self.candidate_formula_table.item(row, 0)
        if formula_item is None:
            return
        self.formula_edit.setText(formula_item.text())
        self.tabs.setCurrentIndex(0)
        self.calculate()
