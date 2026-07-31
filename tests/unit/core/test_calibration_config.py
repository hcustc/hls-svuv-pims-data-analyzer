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


def test_calibration_fit_roundtrip():
    points = [(1, 2), (2, 5), (3, 10)]
    calibration = fit_quadratic_calibration(points)
    assert round(calibration.tof_to_mz(2), 6) == 5


def test_calibration_fit_requires_three_distinct_tof_values():
    with pytest.raises(ValueError, match="three distinct TOF"):
        fit_quadratic_calibration([(100, 10), (100, 11), (100, 12)])


def test_calibration_fit_rejects_non_increasing_mass_axis():
    with pytest.raises(ValueError, match="must increase"):
        fit_quadratic_calibration([(1, 3), (2, 2), (3, 1)])


def test_inverse_calibration_uses_the_increasing_quadratic_branch():
    calibration = Calibration(a=1.0, b=-10.0, c=25.0)

    assert calibration.mz_to_tof(calibration.tof_to_mz(6.0)) == pytest.approx(6.0)


def test_yaml_config_loads_project_defaults():
    calibration = load_calibration_config()
    assert calibration.a > 0
    assert calibration.b > 0
    assert calibration.c > 0
    assert len(load_calibration_points()) >= 3
    assert species_database_path().name == "species_database.sqlite"
    peak_config = load_peak_config()
    assert "H2O" in peak_config


def test_species_database_quality_after_cleaning():
    with sqlite3.connect(species_database_path()) as conn:
        missing_ie = conn.execute(
            "SELECT id, mz, name FROM species WHERE ionization_energy IS NULL ORDER BY id"
        ).fetchall()
        negative_cross_sections = conn.execute(
            "SELECT COUNT(*) FROM pic_cross_sections WHERE cross_section < 0"
        ).fetchone()[0]
        duplicate_species = conn.execute(
            """
            SELECT mz, name, COUNT(*)
            FROM species
            GROUP BY mz, name
            HAVING COUNT(*) > 1
            """
        ).fetchall()

    assert missing_ie == []
    assert negative_cross_sections == 0
    assert duplicate_species == []
