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


def test_nist_webbook_ie_parsers_pick_evaluated_value():
    summary = parse_ionization_energy_summary(NIST_IE_HTML)
    assert len(summary) == 1
    evaluated = pick_evaluated_ie(summary)
    assert evaluated is not None
    assert evaluated.source == "evaluated"
    assert evaluated.value == 9.24378
    assert evaluated.uncertainty == 0.00007
    assert evaluated.units == "eV"

    determinations = parse_ionization_energy_determinations(NIST_IE_HTML)
    assert len(determinations) == 1
    assert determinations[0].value == 9.245
    assert determinations[0].method == "PI"


def test_nist_webbook_does_not_treat_non_ie_summary_as_ie():
    summary = parse_ionization_energy_summary(NIST_NON_IE_SUMMARY_HTML)
    assert len(summary) == 1
    assert pick_evaluated_ie(summary) is None

    class FakeNistClient(NistWebBookClient):
        def _fetch_html(self, url: str):
            return NIST_NON_IE_SUMMARY_HTML, url

    compound = FakeNistClient()._compound_from_html(
        """
        <html><body>
        <h1 id="Top">Phenyl radical</h1>
        <ul>
          <li>Formula: C6H5</li>
          <li>Other data available:
            <ul>
              <li><a href="/cgi/cbook.cgi?ID=C2396012&amp;Units=SI&amp;Mask=20#Ion-Energetics">Gas phase ion energetics data</a></li>
            </ul>
          </li>
        </ul>
        </body></html>
        """,
        "https://webbook.nist.gov/cgi/cbook.cgi?ID=C2396012&Units=SI",
        fallback_id="C2396012",
    )
    assert compound.evaluated_ie is None
    assert compound.best_ie is not None
    assert compound.best_ie.value == 8.32
    assert compound.best_ie.units == "eV"


def test_nist_webbook_client_returns_formula_isomer_candidates():
    class FakeNistClient(NistWebBookClient):
        def _fetch_html(self, url: str):
            if "Formula=C2H6O" in url:
                return (
                    """
                    <html><body><ol>
                      <li><a href="/cgi/cbook.cgi?ID=C64175&amp;Units=SI">Ethanol</a></li>
                      <li><a href="/cgi/cbook.cgi?ID=C115106&amp;Units=SI">Dimethyl ether</a></li>
                    </ol></body></html>
                    """,
                    url,
                )
            if "ID=C64175" in url:
                return _compound_html("C64175", "Ethanol", "C2H6O", 10.48), url
            if "ID=C115106" in url:
                return _compound_html("C115106", "Dimethyl ether", "C2H6O", 10.025), url
            raise AssertionError(f"unexpected URL: {url}")

    result = FakeNistClient(local_first=False).query_ionization_energy("C2H6O", search_type="formula")
    assert result.selected_compound is None
    assert len(result.compounds) == 2
    assert [compound.name for compound in result.compounds] == ["Ethanol", "Dimethyl ether"]
    assert [compound.best_ie.value for compound in result.compounds] == [10.48, 10.025]


def test_nist_webbook_search_type_inference():
    assert infer_nist_search_type("C6H6") == "formula"
    assert infer_nist_search_type("71-43-2") == "id"
    assert infer_nist_search_type("C71432") == "id"
    assert infer_nist_search_type("Benzene") == "name"


def test_nist_webbook_client_uses_local_database_before_network(tmp_path):
    db_path = tmp_path / "species.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE species (id INTEGER PRIMARY KEY, mz INTEGER NOT NULL, name TEXT NOT NULL, ionization_energy REAL)"
        )
        conn.execute(
            "INSERT INTO species (id, mz, name, ionization_energy) VALUES (1, 78, 'Benzene', 9.24378)"
        )

    class NoNetworkClient(NistWebBookClient):
        def _fetch_html(self, url: str):
            raise AssertionError("local database hit should not call NIST WebBook")

    result = NoNetworkClient(local_db_path=db_path).query_ionization_energy("Benzene", search_type="name")
    assert result.search_type == "local_database"
    assert result.selected_compound.name == "Benzene"
    assert result.best_ie.value == 9.24378
    assert result.best_ie.source == "local_database"


def test_nist_webbook_client_does_not_fuzzy_match_formula_against_local_names(tmp_path):
    db_path = tmp_path / "species.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE species (id INTEGER PRIMARY KEY, mz INTEGER NOT NULL, name TEXT NOT NULL, ionization_energy REAL)"
        )
        conn.execute(
            "INSERT INTO species (id, mz, name, ionization_energy) VALUES (1, 28, 'Carbon monoxide', 14.014)"
        )

    class FakeNistClient(NistWebBookClient):
        def _fetch_html(self, url: str):
            if "Formula=CO" in url:
                return (
                    """
                    <html><body>
                    <h1 id="Top">Carbon monoxide</h1>
                    <ul>
                      <li>Formula: CO</li>
                      <li>Other data available:
                        <ul>
                          <li><a href="/cgi/cbook.cgi?ID=C630080&amp;Units=SI&amp;Mask=20#Ion-Energetics">Gas phase ion energetics data</a></li>
                        </ul>
                      </li>
                    </ul>
                    <h2 id="Ion-Energetics">Gas phase ion energetics data</h2>
                    <table class="data">
                      <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
                      <tr><td>IE (evaluated)</td><td>14.014</td><td>eV</td><td>PE</td><td>NIST</td><td></td></tr>
                    </table>
                    </body></html>
                    """,
                    url,
                )
            raise AssertionError(f"unexpected URL: {url}")

    result = FakeNistClient(local_db_path=db_path).query_ionization_energy("CO", search_type="formula")
    assert result.search_type == "formula"
    assert [compound.name for compound in result.compounds[:2]] == ["Carbon monoxide", "Carbon monoxide"]
    assert result.compounds[0].best_ie.source == "local_database"
    assert result.compounds[1].best_ie.source == "evaluated"


def test_nist_webbook_formula_query_includes_local_mz_candidates(tmp_path):
    db_path = tmp_path / "species.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE species (id INTEGER PRIMARY KEY, mz INTEGER NOT NULL, name TEXT NOT NULL, ionization_energy REAL)"
        )
        conn.execute(
            "INSERT INTO species (id, mz, name, ionization_energy) VALUES (16, 16, 'Atomic oxygen', 13.61806)"
        )
        conn.execute(
            "INSERT INTO species (id, mz, name, ionization_energy) VALUES (19, 16, 'Methane', 12.61)"
        )

    class FakeNistClient(NistWebBookClient):
        def _fetch_html(self, url: str):
            if "Formula=CH4" in url:
                return (
                    """
                    <html><body><ol>
                      <li><a href="/cgi/cbook.cgi?ID=C74828&amp;Units=SI">Methane</a></li>
                      <li><a href="/cgi/cbook.cgi?ID=C558203&amp;Units=SI">(2H4)methane</a></li>
                    </ol></body></html>
                    """,
                    url,
                )
            if "ID=C74828" in url:
                return _compound_html("C74828", "Methane", "CH4", 12.61), url
            if "ID=C558203" in url:
                return _compound_html("C558203", "(2H4)methane", "C2H4", 12.658), url
            raise AssertionError(f"unexpected URL: {url}")

    result = FakeNistClient(local_db_path=db_path).query_ionization_energy("CH4", search_type="formula")
    assert result.compounds[0].name == "Methane"
    assert result.compounds[0].best_ie.source == "local_database"
    assert result.compounds[1].name == "Atomic oxygen"
    assert result.compounds[1].best_ie.source == "local_mz_candidate"
    assert result.compounds[2].name == "Methane"
    assert result.compounds[2].best_ie.source == "evaluated"
