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


def test_load_manual_peak_ranges_uses_mz_as_peak_identity(tmp_path):
    path = tmp_path / "peaks.csv"
    path.write_text(
        "\ufeffSpecies,飞行时间,质量数 (m/z),强度,左边界,右边界\n"
        "Kr,14251.12,84.11,853,14240,14261\n",
        encoding="utf-8-sig",
    )

    ranges = load_peak_ranges(path, calibration=Calibration(a=0, b=999, c=0))

    assert len(ranges) == 1
    assert ranges[0].label == "Kr"
    assert ranges[0].mz == pytest.approx(84.11)
    assert ranges[0].peak_index == 14251
    assert ranges[0].left_bound == 14240
    assert ranges[0].right_bound == 14261


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


def test_kr_expansion_reads_first_level_energy_subfolders(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n84,22,21,23\n", encoding="utf-8")
    energy_dir = tmp_path / "14.6-14.8eV"
    energy_dir.mkdir()

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
        (energy_dir / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

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
