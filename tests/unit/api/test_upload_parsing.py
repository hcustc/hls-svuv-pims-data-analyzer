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


def test_api_parses_uploaded_pie_curve_tables():
    import pytest

    pytest.importorskip("fastapi")
    from bl03u_masstool.api.app import _parse_uploaded_pie_curves

    long_table = b"mz,energy,intensity\n15,11.0,1.2\n15,11.5,2.4\n29,11.0,0.4\n"
    long_df = _parse_uploaded_pie_curves(long_table, "curve.csv")
    long_curves = build_pie_curves(long_df)
    assert sorted(long_curves) == [15, 29]
    assert long_curves[15]["intensities"] == [1.2, 2.4]

    wide_table = b"energy,15,29\n11.0,1.0,0.2\n11.5,2.0,0.5\n"
    wide_df = _parse_uploaded_pie_curves(wide_table, "curve.csv")
    wide_curves = build_pie_curves(wide_df)
    assert sorted(wide_curves) == [15, 29]
    assert wide_curves[29]["energies"] == [11.0, 11.5]
