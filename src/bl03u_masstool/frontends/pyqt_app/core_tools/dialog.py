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

from bl03u_masstool.frontends.pyqt_app.isotope.dialog import IsotopeAbundanceDialog
from bl03u_masstool.frontends.pyqt_app.mole_fraction.dialog import MoleFractionDialog
from bl03u_masstool.frontends.pyqt_app.nist.widget import IonizationEnergyLookupWidget
from bl03u_masstool.frontends.pyqt_app.normalization.widget import NormalizationSettingsWidget
from bl03u_masstool.frontends.pyqt_app.pics.dialog import PICSCalculatorDialog
from bl03u_masstool.frontends.pyqt_app.pics.import_widget import PICSImportWidget
from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog
from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

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
            "pics": self.tabs.addTab(PICSCalculatorDialog(calibration, self.normalization_settings, self), "PICS计算"),
            "pics_import": self.tabs.addTab(PICSImportWidget(self), "PICS导入"),
            "ionization": self.tabs.addTab(IonizationEnergyLookupWidget(self), "电离能查询"),
            "isotope": self.tabs.addTab(IsotopeAbundanceDialog(self), "分子式与质量分析"),
        }
        self.tabs.setCurrentIndex(tab_indexes.get(initial_tab, 0))
        layout.addWidget(self.tabs)
