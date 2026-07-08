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
        self.query_edit.setPlaceholderText("输入分子式、名称、CAS号或NIST ID")
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
