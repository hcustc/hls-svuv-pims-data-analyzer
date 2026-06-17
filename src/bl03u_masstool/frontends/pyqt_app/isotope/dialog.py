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

class IsotopeAbundanceDialog(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.candidate_worker: WorkerThread | None = None
        self.setWindowTitle("分子式与同位素工具")
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

        formula_group = QtWidgets.QGroupBox("分子式质量与同位素分布")
        formula_layout = QtWidgets.QGridLayout(formula_group)
        formula_layout.setHorizontalSpacing(8)
        formula_layout.setVerticalSpacing(8)
        self.formula_edit = QtWidgets.QLineEdit()
        self.formula_edit.setPlaceholderText("输入分子式，例如 C6H6、CF3COOH、H2O")
        self.formula_edit.returnPressed.connect(self.calculate)
        self.min_percent_edit = QtWidgets.QDoubleSpinBox()
        self.min_percent_edit.setRange(0.0, 100.0)
        self.min_percent_edit.setDecimals(4)
        self.min_percent_edit.setSingleStep(0.01)
        self.min_percent_edit.setValue(0.01)
        self.calculate_button = QtWidgets.QPushButton("计算质量/同位素")
        self.calculate_button.clicked.connect(self.calculate)
        formula_layout.addWidget(QtWidgets.QLabel("分子式"), 0, 0)
        formula_layout.addWidget(self.formula_edit, 0, 1)
        formula_layout.addWidget(QtWidgets.QLabel("最小丰度(%)"), 0, 2)
        formula_layout.addWidget(self.min_percent_edit, 0, 3)
        formula_layout.addWidget(self.calculate_button, 0, 4)
        formula_layout.setColumnStretch(1, 1)
        mass_layout.addWidget(formula_group)

        self.composition_label = QtWidgets.QLabel("")
        self.composition_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.mass_label = QtWidgets.QLabel("")
        self.mass_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        mass_layout.addWidget(self.composition_label)
        mass_layout.addWidget(self.mass_label)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["质量数 (m/z)", "丰度", "丰度占比 (%)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        mass_layout.addWidget(self.table, stretch=1)
        self.tabs.addTab(mass_tab, "质量/同位素")

        candidate_tab = QtWidgets.QWidget()
        candidate_layout = QtWidgets.QVBoxLayout(candidate_tab)
        candidate_layout.setContentsMargins(8, 8, 8, 8)
        candidate_layout.setSpacing(10)

        search_group = QtWidgets.QGroupBox("按质量生成候选分子式")
        search_layout = QtWidgets.QGridLayout(search_group)
        search_layout.setHorizontalSpacing(8)
        search_layout.setVerticalSpacing(8)
        self.target_mass_edit = QtWidgets.QDoubleSpinBox()
        self.target_mass_edit.setRange(0.000001, 100_000)
        self.target_mass_edit.setDecimals(8)
        self.target_mass_edit.setValue(78.04695019)
        self.target_mass_edit.setSingleStep(0.01)
        self.tolerance_edit = QtWidgets.QDoubleSpinBox()
        self.tolerance_edit.setRange(0.0, 1_000_000)
        self.tolerance_edit.setDecimals(6)
        self.tolerance_edit.setValue(0.01)
        self.tolerance_unit_combo = QtWidgets.QComboBox()
        self.tolerance_unit_combo.addItem("Da", "Da")
        self.tolerance_unit_combo.addItem("ppm", "ppm")
        self.element_ranges_edit = QtWidgets.QLineEdit()
        self.element_ranges_edit.setPlaceholderText("C:0-20,H:0-60,O:0-10,N:0-6,F:0-20")
        self.element_ranges_edit.setText("C:0-20,H:0-60,O:0-10,N:0-6,F:0-20,Cl:0-6,Br:0-4,S:0-4")
        self.max_results_edit = QtWidgets.QSpinBox()
        self.max_results_edit.setRange(1, 10_000)
        self.max_results_edit.setValue(200)
        self.generate_button = QtWidgets.QPushButton("生成候选")
        self.generate_button.clicked.connect(self.generate_candidates)

        search_layout.addWidget(QtWidgets.QLabel("目标质量"), 0, 0)
        search_layout.addWidget(self.target_mass_edit, 0, 1)
        search_layout.addWidget(QtWidgets.QLabel("误差"), 0, 2)
        search_layout.addWidget(self.tolerance_edit, 0, 3)
        search_layout.addWidget(self.tolerance_unit_combo, 0, 4)
        search_layout.addWidget(QtWidgets.QLabel("最多结果"), 0, 5)
        search_layout.addWidget(self.max_results_edit, 0, 6)
        search_layout.addWidget(QtWidgets.QLabel("元素范围"), 1, 0)
        search_layout.addWidget(self.element_ranges_edit, 1, 1, 1, 5)
        search_layout.addWidget(self.generate_button, 1, 6)
        search_layout.setColumnStretch(1, 1)
        candidate_layout.addWidget(search_group)

        self.candidate_formula_table = QtWidgets.QTableWidget()
        self.candidate_formula_table.setAlternatingRowColors(True)
        self.candidate_formula_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_formula_table.itemDoubleClicked.connect(self.use_candidate_formula)
        candidate_layout.addWidget(self.candidate_formula_table, stretch=1)
        self.tabs.addTab(candidate_tab, "候选分子式")

    def calculate(self):
        formula = self.formula_edit.text().strip()
        if not formula:
            QtWidgets.QMessageBox.warning(self, "提示", "请输入分子式")
            return
        try:
            composition = parse_formula(formula)
            nominal_mass = formula_nominal_mass(formula)
            exact_mass = formula_monoisotopic_mass(formula)
            rows = calculate_isotope_distribution(formula, min_percent=self.min_percent_edit.value())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "错误", str(exc))
            return

        self.composition_label.setText("组成: " + ", ".join(f"{key}{value}" for key, value in composition.items()))
        self.mass_label.setText(f"名义质量: {nominal_mass}; 单同位素精确质量: {exact_mass:.8f}")
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            self.table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(f"{row['mass']:.4f}"))
            self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{row['abundance']:.8g}"))
            self.table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(f"{row['percent']:.4f}"))

    def generate_candidates(self):
        try:
            ranges = parse_element_count_ranges(self.element_ranges_edit.text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "错误", str(exc))
            return

        target_mass = self.target_mass_edit.value()
        tolerance = self.tolerance_edit.value()
        tolerance_unit = str(self.tolerance_unit_combo.currentData())
        max_results = self.max_results_edit.value()
        self.set_candidate_busy(True)
        self.candidate_worker = WorkerThread(
            lambda: generate_formula_candidates(
                target_mass,
                tolerance=tolerance,
                tolerance_unit=tolerance_unit,
                element_ranges=ranges,
                max_results=max_results,
            ),
            self,
        )
        self.candidate_worker.finished_with_result.connect(self.show_formula_candidates)
        self.candidate_worker.failed.connect(lambda message: QtWidgets.QMessageBox.warning(self, "错误", message))
        self.candidate_worker.finished.connect(lambda: self.set_candidate_busy(False))
        self.candidate_worker.start()

    def set_candidate_busy(self, busy: bool) -> None:
        for widget in (
            self.target_mass_edit,
            self.tolerance_edit,
            self.tolerance_unit_combo,
            self.element_ranges_edit,
            self.max_results_edit,
            self.generate_button,
        ):
            widget.setDisabled(busy)

    def show_formula_candidates(self, candidates: object) -> None:
        candidate_list = list(candidates if isinstance(candidates, list) else [])

        rows = [
            {
                "分子式": item["formula"],
                "名义质量": item["nominal_mass"],
                "单同位素质量": f"{item['monoisotopic_mass']:.8f}",
                "误差(Da)": f"{item['error_da']:.8f}",
                "误差(ppm)": f"{item['error_ppm']:.3f}",
                "元素组成": ", ".join(f"{key}{value}" for key, value in item["composition"].items()),
            }
            for item in candidate_list
        ]
        self.set_dataframe(
            self.candidate_formula_table,
            pd.DataFrame(rows, columns=["分子式", "名义质量", "单同位素质量", "误差(Da)", "误差(ppm)", "元素组成"]),
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
