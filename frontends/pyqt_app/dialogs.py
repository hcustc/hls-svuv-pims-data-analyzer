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
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_params_tab(), "1. 参数设置")
        self.tabs.addTab(self._create_parent_tab(), "2. 母体摩尔分数")
        self.tabs.addTab(self._create_isomeric_tab(), "3. 同分异构体分离")
        self.tabs.addTab(self._create_product_tab(), "4. 产物摩尔分数")
        self.tabs.addTab(self._create_results_tab(), "5. 结果汇总")
        layout.addWidget(self.tabs)

    def _create_params_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

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
        self.combo_kr_source.addItems(["使用默认数据", "从文件加载"])
        ec_top.addWidget(self.combo_kr_source)
        self.btn_load_kr = QtWidgets.QPushButton("加载Kr数据")
        self.btn_load_kr.clicked.connect(self._load_kr_data)
        self.btn_load_kr.setEnabled(False)
        self.combo_kr_source.currentTextChanged.connect(
            lambda t: self.btn_load_kr.setEnabled(t == "从文件加载")
        )
        ec_top.addWidget(self.btn_load_kr)
        ec_top.addStretch()
        ec_layout.addLayout(ec_top)

        self.kr_table = QtWidgets.QTableWidget()
        self.kr_table.setColumnCount(4)
        self.kr_table.setHorizontalHeaderLabels(["温度(°C)", "Kr信号", "λ(T)", ""])
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

    def _create_isomeric_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        isom_group = QtWidgets.QGroupBox("同分异构体分离设置")
        isom_layout = QtWidgets.QGridLayout(isom_group)
        isom_layout.addWidget(QtWidgets.QLabel("目标质量数 m/z:"), 0, 0)
        self.spin_isom_mz = QtWidgets.QSpinBox()
        self.spin_isom_mz.setRange(1, 500)
        self.spin_isom_mz.setValue(80)
        isom_layout.addWidget(self.spin_isom_mz, 0, 1)
        btn_find_species = QtWidgets.QPushButton("查找同质量数物种")
        btn_find_species.clicked.connect(self._find_isomeric_species)
        isom_layout.addWidget(btn_find_species, 0, 2)
        isom_layout.addWidget(QtWidgets.QLabel("低能量 E₁ (eV):"), 1, 0)
        self.spin_isom_e_low = QtWidgets.QDoubleSpinBox()
        self.spin_isom_e_low.setRange(0.0, 30.0)
        self.spin_isom_e_low.setDecimals(2)
        self.spin_isom_e_low.setValue(8.5)
        self.spin_isom_e_low.setSingleStep(0.5)
        isom_layout.addWidget(self.spin_isom_e_low, 1, 1)
        isom_layout.addWidget(QtWidgets.QLabel("高能量 E₂ (eV):"), 1, 2)
        self.spin_isom_e_high = QtWidgets.QDoubleSpinBox()
        self.spin_isom_e_high.setRange(0.0, 30.0)
        self.spin_isom_e_high.setDecimals(2)
        self.spin_isom_e_high.setValue(10.0)
        self.spin_isom_e_high.setSingleStep(0.5)
        isom_layout.addWidget(self.spin_isom_e_high, 1, 3)
        layout.addWidget(isom_group)

        self.isom_species_table = QtWidgets.QTableWidget()
        self.isom_species_table.setColumnCount(5)
        self.isom_species_table.setHorizontalHeaderLabels(["物种名称", "电离能(eV)", "σ(E₁)", "σ(E₂)", "选择"])
        self.isom_species_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.isom_species_table)

        btn_calc_isom = QtWidgets.QPushButton("计算同分异构体分离")
        btn_calc_isom.clicked.connect(self._calc_isomeric_separation)
        layout.addWidget(btn_calc_isom)

        self.isom_result_table = QtWidgets.QTableWidget()
        self.isom_result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.isom_result_table, 1)

        return widget

    def _create_product_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        ref_group = QtWidgets.QGroupBox("参考物种设置（物种A，摩尔分数已知）")
        ref_layout = QtWidgets.QGridLayout(ref_group)
        ref_layout.addWidget(QtWidgets.QLabel("参考物种 m/z:"), 0, 0)
        self.spin_ref_mz = QtWidgets.QSpinBox()
        self.spin_ref_mz.setRange(1, 500)
        self.spin_ref_mz.setValue(self.settings.reference_species_mz or self.settings.parent_mz)
        ref_layout.addWidget(self.spin_ref_mz, 0, 1)
        ref_layout.addWidget(QtWidgets.QLabel("参考物种名称:"), 0, 2)
        self.combo_ref_species = QtWidgets.QComboBox()
        ref_layout.addWidget(self.combo_ref_species, 0, 3)
        btn_refresh_ref = QtWidgets.QPushButton("刷新")
        btn_refresh_ref.clicked.connect(self._refresh_ref_species)
        ref_layout.addWidget(btn_refresh_ref, 0, 4)
        ref_layout.addWidget(QtWidgets.QLabel("T_M (°C):"), 1, 0)
        self.spin_ref_tm = QtWidgets.QSpinBox()
        self.spin_ref_tm.setRange(0, 2000)
        self.spin_ref_tm.setValue(int(self.settings.reference_species_tm or 900))
        ref_layout.addWidget(self.spin_ref_tm, 1, 1)
        ref_layout.addWidget(QtWidgets.QLabel("X_A(T_M):"), 1, 2)
        self.spin_ref_mf_tm = QtWidgets.QDoubleSpinBox()
        self.spin_ref_mf_tm.setRange(0.0, 1.0)
        self.spin_ref_mf_tm.setDecimals(6)
        self.spin_ref_mf_tm.setValue(self.settings.reference_species_mf_at_tm)
        self.spin_ref_mf_tm.setSingleStep(0.0001)
        ref_layout.addWidget(self.spin_ref_mf_tm, 1, 3)
        ref_layout.addWidget(QtWidgets.QLabel("光子能量 E (eV):"), 2, 0)
        self.spin_product_energy = QtWidgets.QDoubleSpinBox()
        self.spin_product_energy.setRange(0.0, 30.0)
        self.spin_product_energy.setDecimals(2)
        self.spin_product_energy.setValue(self.settings.photon_energy)
        self.spin_product_energy.setSingleStep(0.5)
        ref_layout.addWidget(self.spin_product_energy, 2, 1)
        layout.addWidget(ref_group)

        prod_group = QtWidgets.QGroupBox("待计算产物物种")
        prod_layout = QtWidgets.QVBoxLayout(prod_group)
        prod_top = QtWidgets.QHBoxLayout()
        prod_top.addWidget(QtWidgets.QLabel("产物 m/z:"))
        self.spin_product_mz = QtWidgets.QSpinBox()
        self.spin_product_mz.setRange(1, 500)
        self.spin_product_mz.setValue(78)
        prod_top.addWidget(self.spin_product_mz)
        prod_top.addWidget(QtWidgets.QLabel("产物名称:"))
        self.combo_product_species = QtWidgets.QComboBox()
        prod_top.addWidget(self.combo_product_species)
        btn_refresh_prod = QtWidgets.QPushButton("刷新")
        btn_refresh_prod.clicked.connect(self._refresh_product_species)
        prod_top.addWidget(btn_refresh_prod)
        prod_top.addWidget(QtWidgets.QLabel("分子量:"))
        self.spin_product_mw = QtWidgets.QDoubleSpinBox()
        self.spin_product_mw.setRange(1, 1000)
        self.spin_product_mw.setValue(78)
        prod_top.addWidget(self.spin_product_mw)
        prod_top.addStretch()
        prod_layout.addLayout(prod_top)
        btn_add_product = QtWidgets.QPushButton("添加到计算列表")
        btn_add_product.clicked.connect(self._add_product_species)
        prod_layout.addWidget(btn_add_product)
        self.product_list = QtWidgets.QListWidget()
        prod_layout.addWidget(self.product_list)
        btn_remove_product = QtWidgets.QPushButton("移除选中")
        btn_remove_product.clicked.connect(self._remove_product_species)
        prod_layout.addWidget(btn_remove_product)
        layout.addWidget(prod_group, 1)

        btn_calc_product = QtWidgets.QPushButton("计算产物摩尔分数")
        btn_calc_product.clicked.connect(self._calc_product_mole_fractions)
        layout.addWidget(btn_calc_product)

        return widget

    def _create_results_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        self.results_table = QtWidgets.QTableWidget()
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.results_table, 1)

        btn_export = QtWidgets.QPushButton("导出结果 (Excel/CSV)")
        btn_export.clicked.connect(self._export_results)
        layout.addWidget(btn_export)

        return widget

    def _on_md_preset_changed(self, text):
        if text in MASS_DISCRIMINATION_PRESETS:
            self.spin_md_exponent.setValue(MASS_DISCRIMINATION_PRESETS[text])

    def _on_md_exponent_changed(self, val):
        self.settings.mass_disc_exponent = val

    def _preview_mass_discrimination(self):
        if not self.database:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载物种数据库")
            return
        seen = set()
        species_list = []
        for spec in self.database:
            key = (spec["name"], spec["mz"])
            if key not in seen:
                seen.add(key)
                species_list.append(spec)
        self.md_preview_table.setRowCount(0)
        for spec in species_list[:30]:
            row = self.md_preview_table.rowCount()
            self.md_preview_table.insertRow(row)
            self.md_preview_table.setItem(row, 0, QtWidgets.QTableWidgetItem(spec["name"]))
            self.md_preview_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(spec["mz"])))
            D_i = calc_mass_discrimination(spec["mz"], self.spin_md_exponent.value())
            self.md_preview_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{D_i:.6f}"))

    def _fill_kr_table(self):
        self.kr_table.setRowCount(0)
        for temp in sorted(self.settings.kr_data.keys()):
            row = self.kr_table.rowCount()
            self.kr_table.insertRow(row)
            self.kr_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
            self.kr_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{self.settings.kr_data[temp]:.6f}"))
            self.kr_table.setItem(row, 2, QtWidgets.QTableWidgetItem("-"))
            self.kr_table.setItem(row, 3, QtWidgets.QTableWidgetItem(""))

    def _load_kr_data(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载Kr数据", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            self.settings.kr_data = {}
            for _, row in df.iterrows():
                temp = float(row.iloc[0])
                signal = float(row.iloc[1])
                self.settings.kr_data[temp] = signal
            self._fill_kr_table()
            QtWidgets.QMessageBox.information(self, "成功", f"加载了 {len(self.settings.kr_data)} 个Kr数据点")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载Kr数据失败: {e}")

    def _calc_expansion_coefficients(self):
        self.expansion_coefficients = calc_expansion_coefficients(self.settings.kr_data)
        for row in range(self.kr_table.rowCount()):
            temp_item = self.kr_table.item(row, 0)
            if temp_item:
                temp = float(temp_item.text())
                if temp in self.expansion_coefficients:
                    self.kr_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{self.expansion_coefficients[temp]:.6f}"))
        QtWidgets.QMessageBox.information(self, "成功", f"计算了 {len(self.expansion_coefficients)} 个温度点的膨胀系数")

    def _refresh_ref_species(self):
        mz = self.spin_ref_mz.value()
        self.combo_ref_species.clear()
        for spec in self.database:
            if spec["mz"] == mz:
                self.combo_ref_species.addItem(spec["name"])

    def _refresh_product_species(self):
        mz = self.spin_product_mz.value()
        self.combo_product_species.clear()
        for spec in self.database:
            if spec["mz"] == mz:
                self.combo_product_species.addItem(spec["name"])

    def _add_product_species(self):
        mz = self.spin_product_mz.value()
        name = self.combo_product_species.currentText() or f"m/z={mz}"
        mw = self.spin_product_mw.value()
        text = f"m/z={mz}, name={name}, MW={mw:.0f}"
        for i in range(self.product_list.count()):
            if self.product_list.item(i).text() == text:
                QtWidgets.QMessageBox.warning(self, "提示", "该物种已在列表中")
                return
        self.product_list.addItem(text)

    def _remove_product_species(self):
        current = self.product_list.currentRow()
        if current >= 0:
            self.product_list.takeItem(current)

    def _calc_parent_mole_fraction(self):
        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return
        mz = self.spin_parent_mz.value()
        T0 = self.spin_parent_t0.value()
        X0 = self.spin_parent_mf0.value()

        scan_df = getattr(self, "_temperature_scan_df", None)
        if scan_df is None or scan_df.empty:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在温度扫描选项卡中加载数据")
            return

        curves = build_temperature_curves(scan_df)
        signal_data = extract_signal_from_temperature_curves(curves, mz)
        if not signal_data:
            QtWidgets.QMessageBox.warning(self, "提示", f"未找到 m/z={mz} 的温度扫描数据")
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

    def _find_isomeric_species(self):
        mz = self.spin_isom_mz.value()
        e_low = self.spin_isom_e_low.value()
        e_high = self.spin_isom_e_high.value()
        self.isom_species_table.setRowCount(0)
        for spec in self.database:
            if spec["mz"] != mz:
                continue
            row = self.isom_species_table.rowCount()
            self.isom_species_table.insertRow(row)
            self.isom_species_table.setItem(row, 0, QtWidgets.QTableWidgetItem(spec["name"]))
            ie = spec.get("ionization_energy")
            self.isom_species_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{ie:.2f}" if ie else "N/A"))
            energies = spec.get("energies")
            cross_sections = spec.get("cross_sections")
            if energies is not None and cross_sections is not None:
                sigma_low = float(np.interp(e_low, energies, cross_sections, left=0, right=0))
                sigma_high = float(np.interp(e_high, energies, cross_sections, left=0, right=0))
            else:
                sigma_low = 0.0
                sigma_high = 0.0
            self.isom_species_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{sigma_low:.4f}"))
            self.isom_species_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{sigma_high:.4f}"))
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            self.isom_species_table.setCellWidget(row, 4, checkbox)

    def _calc_isomeric_separation(self):
        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return
        mz = self.spin_isom_mz.value()
        e_low = self.spin_isom_e_low.value()
        e_high = self.spin_isom_e_high.value()

        scan_df = getattr(self, "_temperature_scan_df", None)
        if scan_df is None or scan_df.empty:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在温度扫描选项卡中加载数据")
            return

        curves = build_temperature_curves(scan_df)
        signal_low = extract_signal_from_temperature_curves(curves, mz)
        signal_high = extract_signal_from_temperature_curves(curves, mz)

        species_list = []
        for row in range(self.isom_species_table.rowCount()):
            checkbox = self.isom_species_table.cellWidget(row, 4)
            if checkbox and checkbox.isChecked():
                spec_name = self.isom_species_table.item(row, 0).text()
                sigma_low = float(self.isom_species_table.item(row, 2).text())
                sigma_high = float(self.isom_species_table.item(row, 3).text())
                species_list.append({"name": spec_name, "sigma_low": sigma_low, "sigma_high": sigma_high})

        if len(species_list) < 2:
            QtWidgets.QMessageBox.warning(self, "提示", "请至少选择2个物种进行分离")
            return

        isom_result = calc_isomeric_separation(
            signal_low, signal_high, species_list,
            expansion_coefficients=self.expansion_coefficients,
        )
        self.isomeric_results[mz] = isom_result

        self.isom_result_table.setRowCount(0)
        common_temps = sorted(set().union(*(r.keys() for r in isom_result.values()))) if isom_result else []
        self.isom_result_table.setColumnCount(2 + len(species_list))
        headers = ["温度(°C)", "总信号"] + [s["name"] for s in species_list]
        self.isom_result_table.setHorizontalHeaderLabels(headers)
        for temp in common_temps:
            row = self.isom_result_table.rowCount()
            self.isom_result_table.insertRow(row)
            self.isom_result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
            self.isom_result_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{signal_high.get(temp, 0):.6f}"))
            for j, spec in enumerate(species_list):
                contrib = isom_result.get(spec["name"], {}).get(temp, 0)
                self.isom_result_table.setItem(row, 2 + j, QtWidgets.QTableWidgetItem(f"{contrib:.6f}"))

        QtWidgets.QMessageBox.information(self, "成功", f"完成同分异构体分离，共 {len(common_temps)} 个温度点")

    def _calc_product_mole_fractions(self):
        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return
        if self.product_list.count() == 0:
            QtWidgets.QMessageBox.warning(self, "提示", "请先添加产物物种")
            return

        ref_mz = self.spin_ref_mz.value()
        ref_name = self.combo_ref_species.currentText()
        ref_mw = float(ref_mz)
        T_M = self.spin_ref_tm.value()
        X_A_TM = self.spin_ref_mf_tm.value()
        energy = self.spin_product_energy.value()

        scan_df = getattr(self, "_temperature_scan_df", None)
        if scan_df is None or scan_df.empty:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在温度扫描选项卡中加载数据")
            return

        curves = build_temperature_curves(scan_df)
        ref_signal_data = extract_signal_from_temperature_curves(curves, ref_mz)
        if not ref_signal_data:
            QtWidgets.QMessageBox.warning(self, "提示", "未找到参考物种的温度扫描数据")
            return

        self.product_mf_results = {}
        for i in range(self.product_list.count()):
            item_text = self.product_list.item(i).text()
            parts = {}
            for part in item_text.split(","):
                k, v = part.strip().split("=")
                parts[k.strip()] = v.strip()
            prod_mz = int(parts.get("m/z", 0))
            prod_name = parts.get("name", "")
            prod_mw = float(parts.get("MW", prod_mz))

            prod_signal_data = extract_signal_from_temperature_curves(curves, prod_mz)
            if not prod_signal_data:
                continue

            results = calc_product_mole_fraction(
                prod_signal_data,
                species_mw=prod_mw,
                species_mz=prod_mz,
                species_name=prod_name,
                ref_mw=ref_mw,
                ref_mz=ref_mz,
                ref_species_name=ref_name,
                ref_signal_data=ref_signal_data,
                ref_mf_at_tm=X_A_TM,
                energy=energy,
                reference_species_tm=float(T_M),
                mass_disc_exponent=self.settings.mass_disc_exponent,
                expansion_coefficients=self.expansion_coefficients,
                database=self.database,
                mz_index=self.mz_index,
            )
            self.product_mf_results[prod_name] = results

        self._update_results_table()
        QtWidgets.QMessageBox.information(self, "成功", f"计算了 {len(self.product_mf_results)} 个产物物种的摩尔分数")

    def _update_results_table(self):
        all_results: dict[str, dict[float, float]] = {}
        if self.parent_mf_results:
            all_results[f"母体(m/z={self.settings.parent_mz})"] = self.parent_mf_results
        all_results.update(self.product_mf_results)
        for mz, species_data in self.isomeric_results.items():
            for species_name, mf_data in species_data.items():
                all_results[f"{species_name}(m/z={mz})"] = mf_data
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

    def _export_results(self):
        all_results: dict[str, dict[float, float]] = {}
        if self.parent_mf_results:
            all_results[f"母体(m/z={self.settings.parent_mz})"] = self.parent_mf_results
        all_results.update(self.product_mf_results)
        for mz, species_data in self.isomeric_results.items():
            for species_name, mf_data in species_data.items():
                all_results[f"{species_name}(m/z={mz})"] = mf_data
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
            self._refresh_ref_species()
            self._refresh_product_species()
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
