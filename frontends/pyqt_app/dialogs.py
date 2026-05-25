from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets

from core.calibration import Calibration
from core.config import (
    PeakDetectionConfig,
    load_calibration_config,
    load_peak_detection_config,
    save_calibration_config,
    save_peak_detection_config,
    species_database_path,
)
from core.isotope import (
    calculate_isotope_distribution,
    formula_monoisotopic_mass,
    formula_nominal_mass,
    generate_formula_candidates,
    parse_element_count_ranges,
    parse_formula,
)
from core.nist_webbook import default_nist_webbook_client
from core.output_paths import ensure_output_dir
from core.pie_analysis import analyze_pie_folder, build_pie_curves, identify_species_for_mz_with_curve, load_species_database
from core.normalization import NormalizationSettings, load_normalization_settings, save_normalization_settings
from core.temperature_scan import (
    TEMPERATURE_CURVE_CLASS_LABELS,
    analyze_temperature_folder,
    build_temperature_curves,
    compute_kr_expansion_factors,
)
from core.mole_fraction import (
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
from frontends.pyqt_app.workers import WorkerThread

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - only used when optional plotting is unavailable
    pg = None


class DataFrameTableMixin:
    def set_dataframe(self, table: QtWidgets.QTableWidget, df: pd.DataFrame) -> None:
        table.setUpdatesEnabled(False)
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, value in enumerate(row):
                table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem("" if pd.isna(value) else str(value)))
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.resizeColumnsToContents()
        table.setUpdatesEnabled(True)


def combo_set_data(combo: QtWidgets.QComboBox, value: object) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            combo.setCurrentIndex(index)
            return


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


class IonizationEnergyLookupWidget(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: WorkerThread | None = None
        self.current_compounds = []
        self.setWindowTitle("电离能查询")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        input_group = QtWidgets.QGroupBox("输入物种")
        input_layout = QtWidgets.QGridLayout(input_group)
        input_layout.setHorizontalSpacing(8)
        input_layout.setVerticalSpacing(8)

        self.query_edit = QtWidgets.QLineEdit()
        self.query_edit.setPlaceholderText("输入分子式、名称、CAS号或NIST ID，例如 C6H6、Benzene、71-43-2、C71432")
        self.query_edit.returnPressed.connect(self.query_webbook)
        self.search_type_combo = QtWidgets.QComboBox()
        self.search_type_combo.addItem("自动", "auto")
        self.search_type_combo.addItem("分子式", "formula")
        self.search_type_combo.addItem("名称", "name")
        self.search_type_combo.addItem("CAS/NIST ID", "id")
        self.query_button = QtWidgets.QPushButton("查询")
        self.query_button.clicked.connect(self.query_webbook)
        self.status_label = QtWidgets.QLabel("数据源: 本地物种库优先；未命中则查询 NIST Chemistry WebBook / Gas phase ion energetics")
        self.status_label.setWordWrap(True)

        input_layout.addWidget(QtWidgets.QLabel("物种"), 0, 0)
        input_layout.addWidget(self.query_edit, 0, 1)
        input_layout.addWidget(self.search_type_combo, 0, 2)
        input_layout.addWidget(self.query_button, 0, 3)
        input_layout.addWidget(self.status_label, 1, 0, 1, 4)
        input_layout.setColumnStretch(1, 1)
        layout.addWidget(input_group)

        result_group = QtWidgets.QGroupBox("查询结果")
        result_layout = QtWidgets.QGridLayout(result_group)
        result_layout.setHorizontalSpacing(8)
        result_layout.setVerticalSpacing(8)

        self.ie_value_label = QtWidgets.QLabel("--")
        self.ie_value_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.compound_label = QtWidgets.QLabel("--")
        self.compound_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.formula_label = QtWidgets.QLabel("--")
        self.formula_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_label = QtWidgets.QLabel("--")
        self.source_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.url_label = QtWidgets.QLabel("--")
        self.url_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse | QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.url_label.setOpenExternalLinks(True)
        self.message_label = QtWidgets.QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        result_layout.addWidget(QtWidgets.QLabel("WebBook IE"), 0, 0)
        result_layout.addWidget(self.ie_value_label, 0, 1)
        result_layout.addWidget(QtWidgets.QLabel("WebBook条目"), 0, 2)
        result_layout.addWidget(self.compound_label, 0, 3)
        result_layout.addWidget(QtWidgets.QLabel("分子式"), 1, 0)
        result_layout.addWidget(self.formula_label, 1, 1)
        result_layout.addWidget(QtWidgets.QLabel("来源"), 1, 2)
        result_layout.addWidget(self.source_label, 1, 3)
        result_layout.addWidget(QtWidgets.QLabel("链接"), 2, 0)
        result_layout.addWidget(self.url_label, 2, 1, 1, 3)
        result_layout.addWidget(self.message_label, 3, 0, 1, 4)
        result_layout.setColumnStretch(1, 1)
        result_layout.setColumnStretch(3, 2)
        layout.addWidget(result_group)

        tables_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        tables_splitter.setChildrenCollapsible(False)

        candidate_group = QtWidgets.QGroupBox("本地/WebBook候选条目")
        candidate_layout = QtWidgets.QVBoxLayout(candidate_group)
        candidate_layout.setContentsMargins(8, 8, 8, 8)
        self.candidate_table = QtWidgets.QTableWidget()
        self.candidate_table.setWordWrap(False)
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.itemSelectionChanged.connect(self.on_candidate_selection_changed)
        candidate_layout.addWidget(self.candidate_table)
        tables_splitter.addWidget(candidate_group)

        determination_group = QtWidgets.QGroupBox("IE测定记录")
        determination_layout = QtWidgets.QVBoxLayout(determination_group)
        determination_layout.setContentsMargins(8, 8, 8, 8)
        self.determination_table = QtWidgets.QTableWidget()
        self.determination_table.setWordWrap(False)
        self.determination_table.setAlternatingRowColors(True)
        self.determination_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        determination_layout.addWidget(self.determination_table)
        tables_splitter.addWidget(determination_group)
        tables_splitter.setStretchFactor(0, 1)
        tables_splitter.setStretchFactor(1, 1)
        layout.addWidget(tables_splitter, stretch=2)

        prediction_group = QtWidgets.QGroupBox("IE预测模型输出")
        prediction_layout = QtWidgets.QVBoxLayout(prediction_group)
        prediction_layout.setContentsMargins(8, 8, 8, 8)
        self.model_prediction_table = QtWidgets.QTableWidget()
        self.model_prediction_table.setWordWrap(False)
        self.model_prediction_table.setAlternatingRowColors(True)
        self.model_prediction_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        prediction_layout.addWidget(self.model_prediction_table)
        layout.addWidget(prediction_group, stretch=1)
        self.set_model_prediction_rows([])

    def set_busy(self, busy: bool, message: str) -> None:
        self.query_button.setDisabled(busy)
        self.query_edit.setDisabled(busy)
        self.search_type_combo.setDisabled(busy)
        self.status_label.setText(message)

    def query_webbook(self):
        query = self.query_edit.text().strip()
        if not query:
            QtWidgets.QMessageBox.warning(self, "提示", "请输入名称、分子式、CAS号或NIST ID")
            return
        search_type = str(self.search_type_combo.currentData())
        self.set_busy(True, "正在查询本地物种库 / NIST WebBook...")
        self.worker = WorkerThread(
            lambda: default_nist_webbook_client().query_ionization_energy(query, search_type=search_type),
            self,
        )
        self.worker.finished_with_result.connect(self.on_query_complete)
        self.worker.failed.connect(self.on_query_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪: 本地物种库优先，未命中则查询 NIST WebBook"))
        self.worker.start()

    def on_query_complete(self, result: object) -> None:
        selected = getattr(result, "selected_compound", None)
        self.current_compounds = list(getattr(result, "compounds", ()) or ())
        self.show_selected_compound(selected, fallback_url=str(getattr(result, "requested_url", "--")))

        message = str(getattr(result, "message", ""))
        if getattr(result, "lost", False):
            message += " WebBook提示结果过多，可能未完整返回。"
        self.message_label.setText(message)

        candidate_rows = []
        for compound in getattr(result, "compounds", ()) or ():
            ie = compound.best_ie
            candidate_rows.append(
                {
                    "NIST ID": compound.nist_id or "",
                    "名称": compound.name or "",
                    "分子式": compound.formula or "",
                    "IE(eV)": "" if ie is None else f"{ie.value:.4f}",
                    "不确定度": "" if ie is None or ie.uncertainty is None else f"{ie.uncertainty:g}",
                    "来源": "" if ie is None else ie.source,
                    "方法": "" if ie is None else (ie.method or ""),
                    "备注": compound.message or ("" if ie is None else (ie.comment or "")),
                    "URL": compound.ion_energetics_url or compound.url or "",
                }
            )
        self.set_dataframe(
            self.candidate_table,
            pd.DataFrame(
                candidate_rows,
                columns=["NIST ID", "名称", "分子式", "IE(eV)", "不确定度", "来源", "方法", "备注", "URL"],
            ),
        )
        if self.current_compounds:
            self.candidate_table.selectRow(0)

    def on_query_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "错误", message)

    def set_model_prediction_rows(self, rows: list[dict]) -> None:
        self.set_dataframe(
            self.model_prediction_table,
            pd.DataFrame(
                rows,
                columns=["模型", "预测IE(eV)", "不确定度", "适用域", "输入", "备注"],
            ),
        )

    def on_candidate_selection_changed(self) -> None:
        row = self.candidate_table.currentRow()
        if row < 0 or row >= len(self.current_compounds):
            return
        self.show_selected_compound(self.current_compounds[row])

    def show_selected_compound(self, compound: object | None, *, fallback_url: str = "--") -> None:
        best_ie = None if compound is None else getattr(compound, "best_ie", None)
        self.ie_value_label.setText("--" if best_ie is None else best_ie.formatted_value())
        if compound is None:
            self.compound_label.setText("--")
            self.formula_label.setText("--")
            self.source_label.setText("--")
            self.url_label.setText(fallback_url)
        else:
            nist_id = getattr(compound, "nist_id", None) or "--"
            name = getattr(compound, "name", None) or "--"
            self.compound_label.setText(f"{name} ({nist_id})")
            self.formula_label.setText(getattr(compound, "formula", None) or "--")
            source = "--" if best_ie is None else f"{best_ie.source}; {best_ie.method or 'method N/A'}"
            self.source_label.setText(source)
            url = getattr(compound, "ion_energetics_url", None) or getattr(compound, "url", None)
            self.url_label.setText(f'<a href="{url}">{url}</a>' if url else "--")

        determination_rows = []
        if compound is not None:
            for item in getattr(compound, "determinations", ()) or ():
                determination_rows.append(
                    {
                        "IE(eV)": f"{item.value:.4f}",
                        "不确定度": "" if item.uncertainty is None else f"{item.uncertainty:g}",
                        "方法": item.method or "",
                        "文献": item.reference or "",
                        "备注": item.comment or "",
                    }
                )
        self.set_dataframe(
            self.determination_table,
            pd.DataFrame(determination_rows, columns=["IE(eV)", "不确定度", "方法", "文献", "备注"]),
        )


class NormalizationSettingsWidget(QtWidgets.QWidget, DataFrameTableMixin):
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
        self.kr_folder_button = QtWidgets.QPushButton("选择文件夹")
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
        self.peak_algorithm_combo.addItem("Prominence（推荐）", "prominence")
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
        self.browse_button = QtWidgets.QPushButton("选择文件夹")
        self.browse_button.clicked.connect(self.select_folder)
        self.run_button = QtWidgets.QPushButton("开始分析")
        self.run_button.clicked.connect(self.run_analysis)
        self.export_button = QtWidgets.QPushButton("导出")
        self.export_button.clicked.connect(self.export_result)
        self.common_params_button = QtWidgets.QPushButton("通用参数")
        self.common_params_button.clicked.connect(self.open_common_parameters)
        self.peak_source_combo = QtWidgets.QComboBox()
        self.peak_source_combo.addItem("自动寻峰", "auto")
        self.peak_source_combo.addItem("手动卡峰", "manual")
        self.peak_file_edit = QtWidgets.QLineEdit()
        self.peak_file_edit.setPlaceholderText("可选: yaml/csv/xlsx 手动卡峰文件")
        self.select_peak_file_button = QtWidgets.QPushButton("选择卡峰")
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
        self.reference_mode_combo.addItem("累加谱寻峰", "sum")
        self.reference_mode_combo.addItem("最高温谱寻峰", "max_temperature")
        self.gaussian_check = QtWidgets.QCheckBox("高斯积分")
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

    def open_common_parameters(self):
        dialog = CommonParametersDialog(self.normalization_settings, self.calibration, self)
        dialog.exec()
        self.calibration = load_calibration_config()

    def run_analysis(self):
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
        photon_normalize = settings.temperature_photon_normalize
        kr_correct = settings.temperature_kr_correct
        mass_discrimination = settings.mass_discrimination
        light_source = settings.light_source
        expansion_factors = settings.expansion_factors if kr_correct else None
        self.set_busy(True, "正在分析温度扫描数据...")
        self.worker = WorkerThread(
            lambda: analyze_temperature_folder(
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
            ),
            self,
        )
        self.worker.finished_with_result.connect(self.on_analysis_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

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
        QtWidgets.QMessageBox.information(self, "完成", f"生成 {len(self.result_df)} 行温度扫描结果")

    def on_analysis_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "错误", message)

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


class PIESpeciesFitDialog(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, calibration: Calibration, normalization_settings: NormalizationSettings | None = None, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.normalization_settings = normalization_settings or NormalizationSettings()
        self.peak_detection = load_peak_detection_config()
        self.database: list[dict] = []
        self.analysis_df = pd.DataFrame()
        self.curves: dict[int, dict] = {}
        self.current_mz: int | None = None
        self.current_fit: dict | None = None
        self.worker: WorkerThread | None = None
        self.setWindowTitle("PIE物种拟合")
        self.resize(1280, 780)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(8)
        default_db = species_database_path()
        self.database_edit = QtWidgets.QLineEdit(str(default_db) if default_db.exists() else "")
        self.load_button = QtWidgets.QPushButton("加载数据库")
        self.load_button.clicked.connect(self.load_database)
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("选择包含PIE扫描质谱文件的文件夹")
        self.select_folder_button = QtWidgets.QPushButton("选择文件夹")
        self.select_folder_button.clicked.connect(self.select_folder)
        self.analyze_button = QtWidgets.QPushButton("生成PIE曲线")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.export_button = QtWidgets.QPushButton("导出曲线数据")
        self.export_button.clicked.connect(self.export_curve_data)
        self.common_params_button = QtWidgets.QPushButton("通用参数")
        self.common_params_button.clicked.connect(self.open_common_parameters)
        self.peak_source_combo = QtWidgets.QComboBox()
        self.peak_source_combo.addItem("自动寻峰", "auto")
        self.peak_source_combo.addItem("手动卡峰", "manual")
        self.peak_file_edit = QtWidgets.QLineEdit()
        self.peak_file_edit.setPlaceholderText("可选: yaml/csv/xlsx 手动卡峰文件")
        self.select_peak_file_button = QtWidgets.QPushButton("选择卡峰")
        self.select_peak_file_button.clicked.connect(self.select_peak_file)

        self.threshold_end_edit = QtWidgets.QDoubleSpinBox()
        self.threshold_end_edit.setRange(0, 1_000_000)
        self.threshold_end_edit.setDecimals(3)
        self.threshold_end_edit.setValue(self.peak_detection.threshold_end)
        self.min_intensity_edit = QtWidgets.QDoubleSpinBox()
        self.min_intensity_edit.setRange(0, 1_000_000)
        self.min_intensity_edit.setDecimals(3)
        self.min_intensity_edit.setValue(self.peak_detection.min_intensity)
        self.energy_decimals_edit = QtWidgets.QSpinBox()
        self.energy_decimals_edit.setRange(0, 6)
        self.energy_decimals_edit.setValue(1)
        self.recursive_check = QtWidgets.QCheckBox("递归")
        self.recursive_check.setChecked(True)
        self.gaussian_check = QtWidgets.QCheckBox("高斯积分")
        self.gaussian_check.setChecked(True)
        self.status_label = QtWidgets.QLabel("就绪")

        controls_layout.addWidget(QtWidgets.QLabel("PICS库"), 0, 0)
        controls_layout.addWidget(self.database_edit, 0, 1, 1, 6)
        controls_layout.addWidget(self.load_button, 0, 7)
        controls_layout.addWidget(QtWidgets.QLabel("PIE文件夹"), 1, 0)
        controls_layout.addWidget(self.folder_edit, 1, 1, 1, 4)
        controls_layout.addWidget(self.select_folder_button, 1, 5)
        controls_layout.addWidget(self.analyze_button, 1, 6)
        controls_layout.addWidget(self.export_button, 1, 7)
        controls_layout.addWidget(self.common_params_button, 1, 8)
        controls_layout.addWidget(QtWidgets.QLabel("能量分组小数"), 2, 0)
        controls_layout.addWidget(self.energy_decimals_edit, 2, 1)
        controls_layout.addWidget(self.recursive_check, 2, 2)
        controls_layout.addWidget(self.gaussian_check, 2, 3)
        controls_layout.addWidget(QtWidgets.QLabel("自动寻峰与归一化参数在“通用参数”页管理"), 2, 4, 1, 4)
        controls_layout.addWidget(QtWidgets.QLabel("卡峰来源"), 3, 0)
        controls_layout.addWidget(self.peak_source_combo, 3, 1)
        controls_layout.addWidget(self.peak_file_edit, 3, 2, 1, 4)
        controls_layout.addWidget(self.select_peak_file_button, 3, 6)
        controls_layout.addWidget(self.status_label, 4, 0, 1, 9)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(4, 1)
        layout.addWidget(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        self.summary_label = QtWidgets.QLabel("未生成PIE曲线")
        left_layout.addWidget(self.summary_label)
        self.mz_list = QtWidgets.QListWidget()
        self.mz_list.currentItemChanged.connect(self.on_mz_selected)
        left_layout.addWidget(self.mz_list, stretch=1)
        self.fit_button = QtWidgets.QPushButton("拟合当前曲线")
        self.fit_button.clicked.connect(self.fit_current_curve)
        left_layout.addWidget(self.fit_button)
        self.show_fit_check = QtWidgets.QCheckBox("显示PICS拟合曲线")
        self.show_fit_check.setChecked(True)
        self.show_fit_check.stateChanged.connect(lambda _: self.refresh_current_plot())
        self.show_components_check = QtWidgets.QCheckBox("显示前三个组分")
        self.show_components_check.stateChanged.connect(lambda _: self.refresh_current_plot())
        left_layout.addWidget(self.show_fit_check)
        left_layout.addWidget(self.show_components_check)
        splitter.addWidget(left_panel)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        if pg is not None:
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground("#ffffff")
            self.plot_widget.setLabel("bottom", "Photon Energy", units="eV")
            self.plot_widget.setLabel("left", "Normalized Intensity")
            self.plot_widget.showGrid(x=True, y=True)
            right_splitter.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            right_splitter.addWidget(QtWidgets.QLabel("未安装 pyqtgraph，无法显示PIE曲线图"))

        self.curve_table = QtWidgets.QTableWidget()
        self.fit_table = QtWidgets.QTableWidget()
        for table in (self.curve_table, self.fit_table):
            table.setWordWrap(False)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.detail_tabs = QtWidgets.QTabWidget()
        self.detail_tabs.addTab(self.curve_table, "曲线数据")
        self.detail_tabs.addTab(self.fit_table, "PICS拟合")
        right_splitter.addWidget(self.detail_tabs)
        right_splitter.setSizes([520, 210])

        splitter.addWidget(right_splitter)
        splitter.setSizes([260, 1020])
        layout.addWidget(splitter, stretch=1)

        if self.database_edit.text():
            self.load_database(show_message=False)

    def load_database(self, show_message: bool = True):
        path = self.database_edit.text().strip()
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择物种数据库",
                "",
                "SQLite Files (*.sqlite *.sqlite3 *.db);;All Files (*)",
            )
            if not path:
                return
            self.database_edit.setText(path)
        try:
            self.database, _ = load_species_database(path)
            self.status_label.setText(f"已加载PICS库: {len(self.database)} 个物种")
            if show_message:
                QtWidgets.QMessageBox.information(self, "完成", f"已加载 {len(self.database)} 个物种")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "错误", str(exc))

    def select_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择PIE扫描文件夹")
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

    def open_common_parameters(self):
        dialog = CommonParametersDialog(self.normalization_settings, self.calibration, self)
        dialog.exec()
        self.calibration = load_calibration_config()

    def run_analysis(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择PIE扫描文件夹")
            return
        energy_decimals = self.energy_decimals_edit.value()
        peak_config = load_peak_detection_config()
        threshold_end = peak_config.threshold_end
        min_intensity = peak_config.min_intensity
        recursive = self.recursive_check.isChecked()
        prefer_gaussian = self.gaussian_check.isChecked()
        manual_peak_path = self.peak_file_edit.text().strip() if self.peak_source_combo.currentData() == "manual" else None
        if self.peak_source_combo.currentData() == "manual" and not manual_peak_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择手动卡峰文件")
            return
        settings = self.normalization_settings
        photon_mode = settings.pie_photon_mode
        photon_normalize = photon_mode != "off"
        photon_reference_mode = "none" if photon_mode == "off" else photon_mode
        mass_discrimination = settings.mass_discrimination
        light_source = settings.light_source
        self.set_busy(True, "正在生成PIE曲线...")
        self.worker = WorkerThread(
            lambda: self.run_pie_analysis_sync(
                folder,
                recursive=recursive,
                energy_decimals=energy_decimals,
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
                manual_peak_path=manual_peak_path,
                photon_normalize=photon_normalize,
                photon_reference_mode=photon_reference_mode,
                mass_discrimination=mass_discrimination,
                light_source=light_source,
            ),
            self,
        )
        self.worker.finished_with_result.connect(self.on_analysis_complete)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def set_busy(self, busy: bool, message: str) -> None:
        self.status_label.setText(message)
        self.analyze_button.setDisabled(busy)
        self.select_folder_button.setDisabled(busy)
        self.export_button.setDisabled(busy)
        self.common_params_button.setDisabled(busy)
        self.fit_button.setDisabled(busy)
        self.load_button.setDisabled(busy)
        self.peak_source_combo.setDisabled(busy)
        self.peak_file_edit.setDisabled(busy)
        self.select_peak_file_button.setDisabled(busy)

    def run_pie_analysis_sync(
        self,
        folder: str,
        *,
        recursive: bool,
        energy_decimals: int,
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
        prefer_gaussian: bool,
        manual_peak_path: str | None = None,
        photon_normalize: bool = True,
        photon_reference_mode: str = "first",
        mass_discrimination: float = 1.0,
        light_source: str = "io",
    ) -> tuple[pd.DataFrame, dict[int, dict]]:
        analysis_df = analyze_pie_folder(
            folder,
            calibration=self.calibration,
            recursive=recursive,
            energy_decimals=energy_decimals,
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
            prefer_gaussian=prefer_gaussian,
            manual_peak_path=manual_peak_path,
            photon_normalize=photon_normalize,
            photon_reference_mode=photon_reference_mode,
            mass_discrimination=mass_discrimination,
            light_source=light_source,
        )
        return analysis_df, build_pie_curves(analysis_df)

    def on_analysis_complete(self, result: object) -> None:
        self.analysis_df, self.curves = result
        self.current_fit = None
        self.populate_mz_list()
        energy_count = self.analysis_df["energy"].nunique() if not self.analysis_df.empty else 0
        self.summary_label.setText(f"{len(self.curves)} 条m/z曲线 | {energy_count} 个能量点")
        if self.curves:
            self.mz_list.setCurrentRow(0)
        QtWidgets.QMessageBox.information(self, "完成", f"生成 {len(self.curves)} 条PIE曲线")

    def on_analysis_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "错误", message)

    def populate_mz_list(self):
        self.mz_list.clear()
        for mz in sorted(self.curves):
            curve = self.curves[mz]
            label = curve.get("species") or ""
            suffix = f" {label}" if label and label != "Unknown" else ""
            item = QtWidgets.QListWidgetItem(f"{mz}{suffix}  ({len(curve['energies'])}点)")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, mz)
            self.mz_list.addItem(item)

    def on_mz_selected(self, current, previous=None):
        if current is None:
            self.current_mz = None
            return
        self.current_mz = int(current.data(QtCore.Qt.ItemDataRole.UserRole))
        self.current_fit = None
        curve = self.curves[self.current_mz]
        rows = curve["rows"].copy()
        curve_df = pd.DataFrame(
            {
                "光子能量(eV)": np.round(rows["energy"].astype(float), 4),
                "原始积分": np.round(rows["raw_area"].astype(float), 4),
                "IO归一化": np.round(rows["photon_normalized_intensity"].astype(float), 4),
                "最终强度": np.round(rows["normalized_intensity"].astype(float), 4),
            }
        )
        self.set_dataframe(self.curve_table, curve_df)
        self.fit_table.clear()
        self.fit_table.setRowCount(0)
        self.fit_table.setColumnCount(0)
        self.update_plot(curve, None)

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
        self.plot_widget.clear()
        plot_item = self.plot_widget.getPlotItem()
        if plot_item.legend is None:
            plot_item.addLegend(offset=(-10, 10))
        else:
            plot_item.legend.clear()
        self.plot_widget.setLabel("bottom", "Photon Energy", units="eV")
        self.plot_widget.setLabel("left", "Normalized Intensity")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        if x_values.size == 0:
            self.plot_widget.setTitle(f"m/z {curve['mz']} PIE - 无有效数据")
            return
        x_ranges = [x_values]
        y_ranges = [y_values]
        self.plot_widget.plot(
            x_values,
            y_values,
            pen=pg.mkPen("#2563eb", width=2),
            symbol="o",
            symbolBrush="#2563eb",
            symbolPen="#1e3a8a",
            symbolSize=8,
            name="实验PIE",
        )

        if fit_model is not None and self.show_fit_check.isChecked():
            fit_x = np.asarray(fit_model.get("energies", []), dtype=float)
            fit_y = np.asarray(fit_model.get("fitted", []), dtype=float)
            fit_valid = np.isfinite(fit_x) & np.isfinite(fit_y)
            fit_x = fit_x[fit_valid]
            fit_y = fit_y[fit_valid]
            if fit_x.size:
                x_ranges.append(fit_x)
                y_ranges.append(fit_y)
                self.plot_widget.plot(
                    fit_x,
                    fit_y,
                    pen=pg.mkPen("#f97316", width=2.5),
                    name="PICS总拟合",
                )
                if self.show_components_check.isChecked():
                    colors = ["#16a34a", "#9333ea", "#dc2626"]
                    for idx, component in enumerate(fit_model.get("species", [])[:3]):
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
                        self.plot_widget.plot(
                            component_x,
                            component_y,
                            pen=pg.mkPen(colors[idx % len(colors)], width=1.5, style=QtCore.Qt.PenStyle.DashLine),
                            name=str(component.get("species", ""))[:24],
                        )

        title = f"m/z {curve['mz']} PIE"
        if fit_model is not None and fit_model.get("fitted"):
            title += f" | PICS R²={fit_model.get('r_squared', 0.0):.4f}"
        self.plot_widget.setTitle(title)
        all_x = np.concatenate(x_ranges)
        all_y = np.concatenate(y_ranges)
        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))
        y_min = float(np.min(all_y))
        y_max = float(np.max(all_y))
        x_pad = max(0.1, (x_max - x_min) * 0.08)
        y_pad = max(1.0, (y_max - y_min) * 0.12)
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        self.plot_widget.setXRange(x_min - x_pad, x_max + x_pad, padding=0)
        self.plot_widget.setYRange(max(0.0, y_min - y_pad), y_max + y_pad, padding=0)

    def fit_current_curve(self):
        if not self.database:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载物种数据库")
            return
        if self.current_mz is None or self.current_mz not in self.curves:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择一条m/z曲线")
            return
        try:
            curve = self.curves[self.current_mz]
            fit_model = identify_species_for_mz_with_curve(
                self.database,
                self.current_mz,
                curve["energies"],
                curve["intensities"],
            )
            self.current_fit = fit_model
            results = fit_model.get("species", [])
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
            if not results:
                QtWidgets.QMessageBox.information(self, "结果", "当前m/z没有匹配到可拟合的物种")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "错误", str(exc))

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
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_data_tab(), "1. 数据加载")
        self.tabs.addTab(self._create_params_tab(), "2. 参数设置")
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
        except Exception:
            pass

    def _create_data_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        db_group = QtWidgets.QGroupBox("物种数据库")
        db_layout = QtWidgets.QHBoxLayout(db_group)
        self.lbl_db_status = QtWidgets.QLabel("未加载数据库")
        self.lbl_db_status.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        btn_load_db = QtWidgets.QPushButton("加载物种数据库")
        btn_load_db.clicked.connect(self._load_database)
        db_layout.addWidget(btn_load_db)
        db_layout.addWidget(self.lbl_db_status, 1)
        layout.addWidget(db_group)

        ts_group = QtWidgets.QGroupBox("温度扫描数据（支持多能量）")
        ts_layout = QtWidgets.QVBoxLayout(ts_group)
        ts_top = QtWidgets.QHBoxLayout()
        btn_add_folder = QtWidgets.QPushButton("添加能量文件夹")
        btn_add_folder.clicked.connect(self._add_energy_folder)
        ts_top.addWidget(btn_add_folder)
        btn_clear = QtWidgets.QPushButton("清空数据")
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
        for name in MASS_DISCRIMINATION_PRESETS:
            self.combo_md_preset.addItem(name)
        self.combo_md_preset.currentTextChanged.connect(self._on_md_preset_changed)
        md_layout.addWidget(self.combo_md_preset, 0, 1)
        md_layout.addWidget(QtWidgets.QLabel("指数 n:"), 1, 0)
        self.spin_md_exponent = QtWidgets.QDoubleSpinBox()
        self.spin_md_exponent.setRange(0.0, 2.0)
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
        self.spin_parent_mz.setRange(1, 500)
        self.spin_parent_mz.setValue(self.settings.parent_mz)
        cfg_layout.addWidget(self.spin_parent_mz, 0, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("参考温度 T₀ (°C):"), 1, 0)
        self.spin_parent_t0 = QtWidgets.QSpinBox()
        self.spin_parent_t0.setRange(0, 2000)
        self.spin_parent_t0.setValue(int(self.settings.reference_temperature or 550))
        cfg_layout.addWidget(self.spin_parent_t0, 1, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("初始摩尔分数 X(T₀):"), 2, 0)
        self.spin_parent_mf0 = QtWidgets.QDoubleSpinBox()
        self.spin_parent_mf0.setRange(0.0, 1.0)
        self.spin_parent_mf0.setDecimals(6)
        self.spin_parent_mf0.setValue(self.settings.parent_initial_mf)
        self.spin_parent_mf0.setSingleStep(0.0001)
        cfg_layout.addWidget(self.spin_parent_mf0, 2, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("光子能量 E (eV):"), 3, 0)
        self.spin_parent_energy = QtWidgets.QDoubleSpinBox()
        self.spin_parent_energy.setRange(0.0, 30.0)
        self.spin_parent_energy.setDecimals(2)
        self.spin_parent_energy.setValue(self.settings.photon_energy)
        self.spin_parent_energy.setSingleStep(0.5)
        cfg_layout.addWidget(self.spin_parent_energy, 3, 1)
        layout.addWidget(cfg_group)

        btn_calc_parent = QtWidgets.QPushButton("计算母体摩尔分数")
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
        btn_calc_auto = QtWidgets.QPushButton("自动计算所有物种摩尔分数")
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
            self.auto_mf_plot_widget.setBackground("#141928")
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
            self.mf_plot_widget.setBackground("#141928")
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

    def _calc_expansion_coefficients(self):
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

    def _calc_parent_mole_fraction(self):
        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return
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


class CoreToolsDialog(QtWidgets.QDialog):
    def __init__(self, calibration: Calibration, parent=None, initial_tab: str = "normalization"):
        super().__init__(parent)
        self.setWindowTitle("BL03U核心处理工具")
        self.resize(1280, 800)
        self.normalization_settings = load_normalization_settings()
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        tab_indexes = {
            "normalization": self.tabs.addTab(NormalizationSettingsWidget(self.normalization_settings, calibration, self), "通用参数"),
            "temperature": self.tabs.addTab(TemperatureScanDialog(calibration, self.normalization_settings, self), "温度扫描"),
            "pie": self.tabs.addTab(PIESpeciesFitDialog(calibration, self.normalization_settings, self), "PIE物种拟合"),
            "mole_fraction": self.tabs.addTab(MoleFractionDialog(calibration, self.normalization_settings, self), "摩尔分数"),
            "ionization": self.tabs.addTab(IonizationEnergyLookupWidget(self), "电离能查询"),
            "isotope": self.tabs.addTab(IsotopeAbundanceDialog(self), "分子/同位素"),
        }
        self.tabs.setCurrentIndex(tab_indexes.get(initial_tab, 0))
        layout.addWidget(self.tabs)
