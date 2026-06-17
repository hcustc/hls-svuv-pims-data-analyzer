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


def test_species_sqlite_roundtrip(tmp_path):
    database = [
        {
            "mz": 18,
            "species": "Water",
            "ie": 12.6,
            "energies": [11.0, 12.0, 13.0],
            "cross_sections": [0.0, 1.2, 2.4],
        }
    ]
    sqlite_path = tmp_path / "species.sqlite"
    save_species_database_sqlite(database, sqlite_path)
    loaded, index = load_species_database(sqlite_path)
    assert len(loaded) == 1
    assert index == {18: [0]}
    assert loaded[0]["species"] == "Water"
    assert [round(value, 6) for value in loaded[0]["cross_sections"].tolist()] == [0.0, 1.2, 2.4]


def test_pics_fit_returns_fitted_curve():
    species = [
        {
            "mz": 18,
            "species": "Water",
            "ie": 12.6,
            "energies": [11.0, 12.0, 13.0],
            "cross_sections": [0.0, 1.0, 2.0],
        }
    ]
    model = fit_species_combination_with_curve(species, [11.0, 12.0, 13.0], [0.0, 2.0, 4.0])
    assert model["candidate_count"] == 1
    assert model["species"][0]["species"] == "Water"
    assert round(model["species"][0]["coefficient"], 6) == 2.0
    assert [round(value, 6) for value in model["fitted"]] == [0.0, 2.0, 4.0]
    assert round(model["r_squared"], 6) == 1.0


def test_pics_fit_supports_manual_and_locked_coefficients():
    species = [
        {
            "id": 1,
            "mz": 18,
            "species": "A",
            "ie": None,
            "energies": [11.0, 12.0, 13.0],
            "cross_sections": [1.0, 2.0, 3.0],
        },
        {
            "id": 2,
            "mz": 18,
            "species": "B",
            "ie": None,
            "energies": [11.0, 12.0, 13.0],
            "cross_sections": [0.0, 1.0, 0.0],
        },
    ]
    manual = fit_species_combination_with_curve(
        species,
        [11.0, 12.0, 13.0],
        [2.0, 5.0, 6.0],
        coefficient_mode="manual",
        coefficients={1: 2.0, 2: 1.0},
    )
    assert manual["coefficient_mode"] == "manual"
    assert [round(value, 6) for value in manual["fitted"]] == [2.0, 5.0, 6.0]
    assert manual["species"][0]["coefficients_by_id"][1] == 2.0

    locked = fit_species_combination_with_curve(
        species,
        [11.0, 12.0, 13.0],
        [2.0, 5.0, 6.0],
        coefficient_mode="locked_fit",
        coefficients={1: 2.0},
        locked_species_ids=[1],
    )
    assert locked["coefficient_mode"] == "locked_fit"
    assert locked["locked_species_ids"] == [1]
    assert [round(value, 6) for value in locked["fitted"]] == [2.0, 5.0, 6.0]
