import sqlite3

import numpy as np
import pandas as pd
import pytest

from bl03u_masstool.core.calibration import Calibration, fit_quadratic_calibration
from bl03u_masstool.core.cwt_peak_detection import CwtPeakDetectionConfig, detect_peaks_cwt
from bl03u_masstool.core.config import load_calibration_config, load_calibration_points, species_database_path
from bl03u_masstool.core.integration import integrate_peak, load_peak_config
from bl03u_masstool.core.isotope import (
    calculate_isotope_distribution,
    formula_monoisotopic_mass,
    formula_nominal_mass,
    generate_formula_candidates,
    parse_element_count_ranges,
    parse_formula,
)
from bl03u_masstool.core.nist_webbook import (
    NistWebBookClient,
    infer_nist_search_type,
    parse_ionization_energy_determinations,
    parse_ionization_energy_summary,
    pick_evaluated_ie,
)
from bl03u_masstool.core.normalization import extract_light_intensity
from bl03u_masstool.core import peak_detection as peak_detection_module
from bl03u_masstool.core.peak_detection import GaussianFit, Peak, detect_peaks_ensemble, detect_peaks_in_range, detect_peaks_prominence
from bl03u_masstool.core.peak_ranges import load_peak_ranges
from bl03u_masstool.core.pie_analysis import (
    analyze_pie_folder,
    build_pie_curves,
    fit_species_combination_with_curve,
    load_species_database,
    save_species_database_sqlite,
)
from bl03u_masstool.core.temperature_scan import (
    analyze_temperature_folder,
    build_temperature_curves,
    classify_temperature_curve,
    compute_kr_expansion_factors,
)


pytestmark = pytest.mark.real_data


def test_real_temperature_fixture_builds_grouped_curves(bl03u_real_data_dir):
    folder = bl03u_real_data_dir / "顺-13-二甲基环己烷 30 torr" / "温度扫描" / "11.5eV"
    if not folder.is_dir():
        pytest.skip(f"real temperature fixture not found: {folder}")

    df = analyze_temperature_folder(
        folder,
        calibration=Calibration(),
        detection_min_idx=0,
        threshold_end=2,
        min_intensity=3,
        prefer_gaussian=False,
        reference_mode="sum",
    )
    curves = build_temperature_curves(df)

    assert len(df) > 0
    assert len(curves) > 0


def test_real_pie_fixture_builds_expected_energy_grid(bl03u_real_data_dir):
    folder = bl03u_real_data_dir / "C6F11O2H" / "PIE_Scan" / "400"
    if not folder.is_dir():
        pytest.skip(f"real PIE fixture not found: {folder}")

    df = analyze_pie_folder(
        folder,
        calibration=Calibration(),
        recursive=False,
        detection_min_idx=0,
        threshold_end=2,
        min_intensity=3,
        prefer_gaussian=False,
    )
    curves = build_pie_curves(df)

    assert sorted(df["energy"].unique().tolist()) == [
        11.0,
        11.5,
        12.0,
        12.5,
        13.0,
        13.5,
        14.0,
        14.5,
    ]
    assert len(curves) >= 15
    assert curves[18]["intensities"] == [0.0, 0.0, 0.0, 0.0, 207.5, 239.5, 171.0, 93.0]
    assert curves[84]["intensities"] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 19.0, 368.0]
