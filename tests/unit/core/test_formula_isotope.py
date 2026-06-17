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


def test_isotope_distribution_contains_main_peak():
    assert parse_formula("H2O") == {"H": 2, "O": 1}
    rows = calculate_isotope_distribution("H2O")
    assert rows
    assert rows[0]["percent"] > 99


def test_formula_mass_calculation():
    assert formula_nominal_mass("C6H6") == 78
    assert round(formula_monoisotopic_mass("H2O"), 6) == 18.010565


def test_formula_parser_supports_groups_hydrates_charges_and_isotope_labels():
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert parse_formula("CuSO4.5H2O") == {"Cu": 1, "S": 1, "O": 9, "H": 10}
    assert parse_formula("NH4+") == {"N": 1, "H": 4}
    assert parse_formula("[13C]2H4") == {"C": 2, "H": 4}
    assert formula_nominal_mass("[13C]2H4") == 30
    assert round(formula_monoisotopic_mass("[13C]H4"), 6) == 17.034655


def test_formula_parser_rejects_unparsed_trailing_text():
    with pytest.raises(ValueError):
        parse_formula("C6H6foo")


def test_generate_formula_candidates_from_mass_and_element_ranges():
    ranges = parse_element_count_ranges("C:0-2,H:0-8,O:0-2")
    candidates = generate_formula_candidates(
        18.010565,
        tolerance=20,
        tolerance_unit="ppm",
        element_ranges=ranges,
        max_results=20,
    )
    assert candidates
    assert candidates[0]["formula"] == "H2O"
    assert abs(candidates[0]["error_ppm"]) < 20


NIST_IE_HTML = """
<html><body>
<h2 id="Ion-Energetics">Gas phase ion energetics data</h2>
<table class="data">
  <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
  <tr>
    <td>IE (evaluated)</td><td>9.24378 ± 0.00007</td><td>eV</td><td>PE</td>
    <td><a href="/cgi/cbook.cgi?ID=C71432&amp;Units=SI&amp;Mask=20#ref-1">Lias, 1988</a></td>
    <td>evaluated value</td>
  </tr>
</table>
<h3>Ionization energy determinations</h3>
<table class="data">
  <tr><th>IE (eV)</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
  <tr>
    <td>9.245 ± 0.003</td><td><a href="/chemistry/">PI</a></td>
    <td><a href="#ref-2">Smith, 1970</a></td><td>threshold</td>
  </tr>
</table>
</body></html>
"""


NIST_NON_IE_SUMMARY_HTML = """
<html><body>
<h2 id="Ion-Energetics">Gas phase ion energetics data</h2>
<table class="data">
  <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
  <tr>
    <td>Proton affinity (review)</td><td>884</td><td>kJ/mol</td><td>N/A</td>
    <td>Hunter and Lias, 1998</td><td>HL</td>
  </tr>
</table>
<h3>Ionization energy determinations</h3>
<table class="data">
  <tr><th>IE (eV)</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
  <tr><td>8.320 ± 0.04</td><td>PE</td><td>Butcher, Costa, et al., 1987</td><td>LBLHLM</td></tr>
</table>
</body></html>
"""


def _compound_html(nist_id: str, name: str, formula: str, ie: float) -> str:
    return f"""
    <html><body>
    <h1 id="Top">{name}</h1>
    <ul>
      <li>Formula: {formula}</li>
      <li>CAS Registry Number: 64-17-5</li>
      <li>Other data available:
        <ul>
          <li><a href="/cgi/cbook.cgi?ID={nist_id}&amp;Units=SI&amp;Mask=20#Ion-Energetics">Gas phase ion energetics data</a></li>
        </ul>
      </li>
    </ul>
    <h2 id="Ion-Energetics">Gas phase ion energetics data</h2>
    <table class="data">
      <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
      <tr><td>IE (evaluated)</td><td>{ie}</td><td>eV</td><td>PE</td><td>NIST</td><td></td></tr>
    </table>
    </body></html>
    """
