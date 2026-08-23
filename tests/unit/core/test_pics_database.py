import csv
import sqlite3

import numpy as np
import pandas as pd
import pytest

from bl03u_masstool.core.calibration import Calibration, fit_quadratic_calibration
from bl03u_masstool.core.cwt_peak_detection import CwtPeakDetectionConfig, detect_peaks_cwt
from bl03u_masstool.core.config import load_calibration_config, load_calibration_points, species_database_path
from bl03u_masstool.core.db_migration import SCHEMA_VERSION, ensure_database_up_to_date, get_user_version
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
    has_fittable_pics_curve,
    load_species_database,
    save_species_database_sqlite,
)
from bl03u_masstool.core.species_seed import build_species_database_from_seed, default_species_seed_path
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
    assert loaded[0]["ie"] == 12.6
    assert loaded[0]["ionization_energy"] == 12.6
    assert loaded[0]["ie_source"] == "PICS数据库"
    assert [round(value, 6) for value in loaded[0]["cross_sections"].tolist()] == [0.0, 1.2, 2.4]


def test_default_species_seed_rebuilds_complete_sqlite_database(tmp_path):
    seed_path = default_species_seed_path()
    with seed_path.open("r", encoding="utf-8", newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    expected_species = len({int(row["species_id"]) for row in seed_rows})

    sqlite_path = build_species_database_from_seed(tmp_path / "species_database.sqlite")

    with sqlite3.connect(sqlite_path) as conn:
        species_count = conn.execute("SELECT COUNT(*) FROM species").fetchone()[0]
        point_count = conn.execute("SELECT COUNT(*) FROM pic_cross_sections").fetchone()[0]
        null_ie_count = conn.execute("SELECT COUNT(*) FROM species WHERE ionization_energy IS NULL").fetchone()[0]
        mz_range = conn.execute("SELECT MIN(mz), MAX(mz) FROM species").fetchone()
        mz_142_metadata = conn.execute(
            """
            SELECT name, formula, smiles
            FROM species
            WHERE name IN ('n-Decane', '1-Methylnaphthalene', '2-Methylnaphthalene')
            ORDER BY id
            """
        ).fetchall()
        methyl_radical_metadata = conn.execute(
            "SELECT formula, smiles FROM species WHERE name = 'Methyl radical'"
        ).fetchone()
        db_version = get_user_version(conn)

    assert species_count == expected_species
    assert point_count == len(seed_rows)
    assert null_ie_count == 0
    assert mz_range == (1, 720)
    assert mz_142_metadata == [
        ("n-Decane", "C10H22", "CCCCCCCCCC"),
        ("1-Methylnaphthalene", "C11H10", "Cc1cccc2ccccc12"),
        ("2-Methylnaphthalene", "C11H10", "Cc1ccc2ccccc2c1"),
    ]
    assert methyl_radical_metadata == ("CH3", "[CH3]")
    assert db_version == SCHEMA_VERSION


def test_ensure_database_up_to_date_builds_missing_database(tmp_path):
    db_path = tmp_path / "species_database.sqlite"
    assert not db_path.exists()
    ensure_database_up_to_date(db_path)
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        assert get_user_version(conn) == SCHEMA_VERSION


def test_ensure_database_up_to_date_is_noop_on_current_version(tmp_path):
    db_path = tmp_path / "species_database.sqlite"
    build_species_database_from_seed(db_path)
    mtime_before = db_path.stat().st_mtime
    ensure_database_up_to_date(db_path)
    assert db_path.stat().st_mtime == mtime_before


def test_ensure_database_up_to_date_migrates_old_version(tmp_path):
    """A legacy database is stamped and receives maintained structure metadata."""
    db_path = tmp_path / "species_database.sqlite"
    build_species_database_from_seed(db_path)
    # Simulate the version-1 metadata that used a library label as a formula
    # and did not include a structure identifier.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE species
            SET formula = 'A2-1-CH3', smiles = NULL
            WHERE name = '1-Methylnaphthalene'
            """
        )
        conn.execute(
            "UPDATE species SET smiles = NULL WHERE name = 'Methyl radical'"
        )
        conn.execute("PRAGMA user_version = 1")
    ensure_database_up_to_date(db_path)
    with sqlite3.connect(db_path) as conn:
        assert get_user_version(conn) == SCHEMA_VERSION
        metadata = conn.execute(
            "SELECT formula, smiles FROM species WHERE name = '1-Methylnaphthalene'"
        ).fetchone()
        methyl_metadata = conn.execute(
            "SELECT formula, smiles FROM species WHERE name = 'Methyl radical'"
        ).fetchone()
    assert metadata == ("C11H10", "Cc1cccc2ccccc12")
    assert methyl_metadata == ("CH3", "[CH3]")


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


def test_pics_fit_excludes_species_without_curve_support_in_scan_range():
    species = [
        {
            "id": 1,
            "mz": 112,
            "species": "Reference species",
            "ie": 9.5,
            "energies": [9.0, 10.0, 11.0],
            "cross_sections": [0.0, 1.0, 2.0],
        },
        {
            "id": 463,
            "mz": 112,
            "species": "Monochlorobenzene",
            "formula": "C6H5Cl",
            "ie": 9.07,
            "energies": [10.5, 13.6183751836558, 13.6183751836558],
            "cross_sections": [33.0, 33.0, 22.2],
        },
    ]

    model = fit_species_combination_with_curve(
        species,
        [9.0, 10.0, 11.0],
        [0.0, 2.0, 4.0],
    )

    assert model["candidate_count"] == 1
    assert [item["species"] for item in model["species"]] == ["Reference species"]
    assert [round(value, 6) for value in model["fitted"]] == [0.0, 2.0, 4.0]
    assert has_fittable_pics_curve(species[0]) is True
    assert has_fittable_pics_curve(species[1], [9.0, 10.0, 11.0]) is False

    ie_only_model = fit_species_combination_with_curve(
        [species[1]],
        [9.0, 10.0, 11.0],
        [0.0, 2.0, 4.0],
    )
    assert ie_only_model["candidate_count"] == 0
    assert ie_only_model["species"] == []
    assert ie_only_model["fitted"] == []


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
