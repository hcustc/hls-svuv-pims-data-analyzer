import sqlite3

import numpy as np
import pandas as pd
import pytest

from core.calibration import Calibration, fit_quadratic_calibration
from core.cwt_peak_detection import CwtPeakDetectionConfig, detect_peaks_cwt
from core.config import load_calibration_config, load_calibration_points, species_database_path
from core.integration import integrate_peak, load_peak_config
from core.isotope import (
    calculate_isotope_distribution,
    formula_monoisotopic_mass,
    formula_nominal_mass,
    generate_formula_candidates,
    parse_element_count_ranges,
    parse_formula,
)
from core.nist_webbook import (
    NistWebBookClient,
    infer_nist_search_type,
    parse_ionization_energy_determinations,
    parse_ionization_energy_summary,
    pick_evaluated_ie,
)
from core.normalization import extract_light_intensity
from core.peak_detection import GaussianFit, Peak, detect_peaks_in_range, detect_peaks_prominence
from core.peak_ranges import load_peak_ranges
from core.pie_analysis import (
    analyze_pie_folder,
    build_pie_curves,
    fit_species_combination_with_curve,
    load_species_database,
    save_species_database_sqlite,
)
from core.temperature_scan import (
    analyze_temperature_folder,
    build_temperature_curves,
    classify_temperature_curve,
    compute_kr_expansion_factors,
)


def test_calibration_fit_roundtrip():
    points = [(1, 2), (2, 5), (3, 10)]
    calibration = fit_quadratic_calibration(points)
    assert round(calibration.tof_to_mz(2), 6) == 5


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


def test_peak_detection_detects_simple_peak():
    y = [0.0] * 20 + [1.0, 4.0, 10.0, 4.0, 1.0] + [0.0] * 20
    peaks = detect_peaks_in_range(
        y,
        calibration=Calibration(a=0, b=1, c=0),
        detection_min_idx=0,
        min_intensity=3,
        threshold_end=0.5,
    )
    assert len(peaks) == 1
    assert peaks[0].mz == peaks[0].time


def test_prominence_peak_detection_handles_baseline_and_noise():
    y = [5.0] * 20 + [5.5, 8.0, 18.0, 8.0, 5.5] + [5.0] * 20
    peaks = detect_peaks_prominence(
        y,
        calibration=Calibration(a=0, b=1, c=0),
        detection_min_idx=0,
        min_intensity=3,
        threshold_end=0.5,
        baseline_window=11,
        smoothing_window=3,
        prominence_ratio=0.05,
        min_peak_width=1,
        max_peak_width=10,
    )
    assert len(peaks) == 1
    assert abs(peaks[0].time - 22) < 0.75
    assert peaks[0].mz == peaks[0].time


def test_cwt_peak_detection_handles_multiscale_peaks():
    x = np.arange(120, dtype=float)
    y = (
        3.0
        + 0.02 * x
        + 25.0 * np.exp(-0.5 * ((x - 40) / 3) ** 2)
        + 16.0 * np.exp(-0.5 * ((x - 82) / 6) ** 2)
    )
    config = CwtPeakDetectionConfig(
        window_size=5,
        poly_order=2,
        prominence_ratio=0.08,
        min_peak_distance=20,
        min_peak_width=2,
        max_peak_width=30,
        wavelet_widths=tuple(range(1, 18)),
        baseline_window_factor=8,
    )

    peaks = detect_peaks_cwt(y, calibration=Calibration(a=0, b=1, c=0), config=config)

    assert len(peaks) == 2
    assert abs(peaks[0].time - 40) < 1
    assert abs(peaks[1].time - 82) < 1
    assert peaks[0].left_bound < peaks[0].time < peaks[0].right_bound
    assert peaks[1].left_bound < peaks[1].time < peaks[1].right_bound


def test_extract_light_intensity_supports_io_and_beam_current():
    metadata = ["IO:52.262221 nA", "Beam Current:395.172mA"]
    assert extract_light_intensity(metadata, "io") == 52.262221
    assert extract_light_intensity(metadata, "beam_current") == 395.172


def test_integrate_peak_refits_gaussian_on_current_spectrum():
    peak = Peak(
        index=22,
        time=22.0,
        mz=22.0,
        intensity=100.0,
        fwhm=6.0,
        left_bound=20,
        right_bound=24,
        gaussian_params=GaussianFit(amplitude=1000.0, mean=22.0, std_dev=2.0, fwhm=4.7, baseline=0.0),
    )
    assert integrate_peak([0.0] * 50, peak, prefer_gaussian=True) == 0.0


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


def test_build_temperature_curves_groups_by_rounded_mz():
    df = pd.DataFrame(
        [
            {"temperature": 800.0, "file": "a.txt", "reference_temperature": 900.0, "mz": 17.99, "area": 10.0},
            {"temperature": 825.0, "file": "b.txt", "reference_temperature": 900.0, "mz": 18.02, "area": 12.0},
            {"temperature": 825.0, "file": "b.txt", "reference_temperature": 900.0, "mz": 18.01, "area": 3.0},
        ]
    )
    curves = build_temperature_curves(df)
    assert list(curves) == [18]
    assert curves[18]["temperatures"] == [800.0, 825.0]
    assert curves[18]["areas"] == [10.0, 15.0]


def test_temperature_curve_classification_trends():
    temperatures = [400, 500, 600, 700, 800]
    assert classify_temperature_curve(temperatures, [0, 1, 3, 6, 10])["curve_class"] == "formation"
    assert classify_temperature_curve(temperatures, [10, 6, 3, 1, 0])["curve_class"] == "consumption"
    assert classify_temperature_curve(temperatures, [0, 2, 10, 2, 0])["curve_class"] == "intermediate"
    assert classify_temperature_curve(temperatures, [3, 3.1, 3, 3.1, 3])["curve_class"] == "unclassified"


def test_load_manual_peak_ranges_from_yaml(tmp_path):
    path = tmp_path / "peaks.yaml"
    path.write_text(
        """
peak_integration:
  peaks:
    - mz: 18
      formula: H2O
      peak: 22
      start: 20
      end: 24
""",
        encoding="utf-8",
    )
    ranges = load_peak_ranges(path, calibration=Calibration(a=0, b=1, c=0))
    assert len(ranges) == 1
    assert ranges[0].label == "H2O"
    assert ranges[0].peak_index == 22
    assert ranges[0].left_bound == 20
    assert ranges[0].right_bound == 24


def test_temperature_scan_uses_manual_peak_file_and_io_normalization(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")

    def write_spectrum(name: str, temperature: float, io: float, scale: float):
        y = [0.0] * 50
        y[21] = 1.0 * scale
        y[22] = 10.0 * scale
        y[23] = 1.0 * scale
        header = [
            "Energy:13.0 eV",
            f"IO:{io} nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            f"Temperature:{temperature} C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("a.txt", 400.0, 10.0, 1.0)
    write_spectrum("b.txt", 500.0, 20.0, 2.0)

    df = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        photon_normalize=True,
        prefer_gaussian=False,
    )
    curves = build_temperature_curves(df)
    assert set(curves) == {22}
    assert [round(value, 6) for value in curves[22]["areas"]] == [0.9, 0.9]


def test_temperature_scan_can_normalize_by_beam_current(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")

    def write_spectrum(name: str, beam_current: float, scale: float):
        y = [0.0] * 50
        y[21] = 1.0 * scale
        y[22] = 10.0 * scale
        y[23] = 1.0 * scale
        header = [
            "Energy:13.0 eV",
            "IO:1 nA",
            f"Beam Current:{beam_current}mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            "Temperature:400 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("a.txt", 100.0, 1.0)
    df = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        photon_normalize=True,
        prefer_gaussian=False,
        light_source="beam_current",
    )
    assert round(float(df.iloc[0]["area"]), 6) == 0.09


def test_compute_kr_expansion_factors_from_high_energy_folder(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n84,22,21,23\n", encoding="utf-8")

    def write_spectrum(name: str, temperature: float, scale: float):
        y = [0.0] * 50
        y[21] = 1.0 * scale
        y[22] = 10.0 * scale
        y[23] = 1.0 * scale
        header = [
            "Energy:14.7 eV",
            "IO:10 nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            f"Temperature:{temperature} C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("low.txt", 400.0, 1.0)
    write_spectrum("high.txt", 800.0, 2.0)
    factors = compute_kr_expansion_factors(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        kr_mz=84,
        manual_peak_path=peak_file,
        light_source="io",
        prefer_gaussian=False,
    )
    assert factors["temperature"].tolist() == [400.0, 800.0]
    assert [round(value, 6) for value in factors["expansion_lambda"].tolist()] == [1.0, 2.0]


def test_kr_expansion_allows_user_selected_low_energy_folder(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n84,22,21,23\n", encoding="utf-8")
    y = [0.0] * 50
    y[21] = 1.0
    y[22] = 10.0
    y[23] = 1.0
    header = [
        "Energy:11.5 eV",
        "IO:10 nA",
        "Beam Current:1mA",
        "Undulator Offset:0mm",
        "Time:1 s",
        "Burner Position:0 mm",
        "Temperature:400 C",
        "DIFF PRESSURE:1Pa",
        "ION PRESSURE:1Pa",
        "TOF PRESSURE:1Pa",
    ]
    (tmp_path / "low-energy.txt").write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")
    factors = compute_kr_expansion_factors(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        kr_mz=84,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
    )
    assert factors["temperature"].tolist() == [400.0]
    assert factors["expansion_lambda"].tolist() == [1.0]


def test_temperature_scan_sum_reference_finds_peaks_across_temperatures(tmp_path):
    def write_spectrum(name: str, temperature: float, peak_index: int):
        y = [0.0] * 50
        y[peak_index - 1] = 1.0
        y[peak_index] = 10.0
        y[peak_index + 1] = 1.0
        header = [
            "Energy:13.0 eV",
            "IO:10 nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            f"Temperature:{temperature} C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("low.txt", 100.0, 20)
    write_spectrum("high.txt", 200.0, 45)

    max_df = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        threshold_end=0.5,
        min_intensity=3,
        prefer_gaussian=False,
        reference_mode="max_temperature",
        detection_min_idx=0,
    )
    sum_df = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        threshold_end=0.5,
        min_intensity=3,
        prefer_gaussian=False,
        reference_mode="sum",
        detection_min_idx=0,
    )

    assert set(build_temperature_curves(max_df)) == {45}
    assert set(build_temperature_curves(sum_df)) == {20, 45}


def test_real_temperature_fixture_builds_grouped_curves():
    df = analyze_temperature_folder(
        "tests/fixtures/C6F11O2H/Temp_Scan/12.5eV",
        calibration=Calibration(),
        detection_min_idx=0,
        threshold_end=2,
        min_intensity=3,
        prefer_gaussian=False,
        reference_mode="sum",
    )
    curves = build_temperature_curves(df)

    assert sorted(df["temperature"].unique().tolist()) == [
        400.0,
        700.0,
        750.0,
        800.0,
        825.0,
        850.0,
        875.0,
        900.0,
        925.0,
        950.0,
        975.0,
    ]
    assert len(curves) >= 60
    assert curves[31]["curve_class"] == "formation"
    assert curves[69]["curve_class"] == "formation"


def test_analyze_pie_folder_builds_selectable_curves(tmp_path):
    def write_spectrum(folder_name, energy, io, scale):
        folder = tmp_path / folder_name
        folder.mkdir()
        y = [0.0] * 20 + [1.0 * scale, 5.0 * scale, 12.0 * scale, 5.0 * scale, 1.0 * scale] + [0.0] * 20
        header = [
            f"Energy:{energy} eV",
            f"IO:{io} nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (folder / "spectrum.txt").write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("11.0eV", 11.0, 10.0, 1.0)
    write_spectrum("12.0eV", 12.0, 20.0, 2.0)

    df = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=True,
        detection_min_idx=0,
        threshold_end=0.5,
        min_intensity=3,
        prefer_gaussian=False,
    )
    curves = build_pie_curves(df)
    assert 22 in curves
    assert curves[22]["energies"] == [11.0, 12.0]


def test_real_pie_fixture_builds_expected_energy_grid():
    df = analyze_pie_folder(
        "tests/fixtures/C6F11O2H/PIE_Scan/400",
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
        12.9,
        13.0,
        13.5,
        14.0,
        14.5,
    ]
    assert len(curves) >= 15
    assert curves[18]["intensities"] == [0.0, 0.0, 0.0, 0.0, 751.0, 207.5, 239.5, 171.0, 93.0]
    assert curves[84]["intensities"] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 19.0, 368.0]


def test_pie_analysis_uses_manual_peak_file_and_direct_io_normalization(tmp_path):
    peak_file = tmp_path / "peaks.yaml"
    peak_file.write_text(
        """
peaks:
  - mz: 22
    peak: 22
    start: 21
    end: 23
""",
        encoding="utf-8",
    )

    def write_spectrum(folder_name, energy, io, scale):
        folder = tmp_path / folder_name
        folder.mkdir()
        y = [0.0] * 50
        y[21] = 1.0 * scale
        y[22] = 10.0 * scale
        y[23] = 1.0 * scale
        header = [
            f"Energy:{energy} eV",
            f"IO:{io} nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (folder / "spectrum.txt").write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("11.0eV", 11.0, 10.0, 1.0)
    write_spectrum("12.0eV", 12.0, 20.0, 2.0)
    df = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=True,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=True,
        photon_reference_mode="none",
    )
    curves = build_pie_curves(df)
    assert set(curves) == {22}
    assert [round(value, 6) for value in curves[22]["intensities"]] == [0.9, 0.9]


def test_analyze_pie_folder_uses_asc_files_by_default(tmp_path):
    for energy, scale in [(11.0, 1.0), (12.0, 2.0)]:
        y = [0.0] * 20 + [1.0 * scale, 5.0 * scale, 12.0 * scale, 5.0 * scale, 1.0 * scale] + [0.0] * 20
        (tmp_path / f"{energy:.1f}eV-test.asc").write_text("\n".join(str(value) for value in y), encoding="utf-8")

    df = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=True,
        detection_min_idx=0,
        threshold_end=0.5,
        min_intensity=3,
        prefer_gaussian=False,
    )
    curves = build_pie_curves(df)
    assert 22 in curves
    assert curves[22]["energies"] == [11.0, 12.0]


def test_api_parses_uploaded_pie_curve_tables():
    import pytest

    pytest.importorskip("fastapi")
    from api.server import _parse_uploaded_pie_curves

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
