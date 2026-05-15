import pandas as pd

from core.calibration import Calibration, fit_quadratic_calibration
from core.config import load_calibration_config, load_calibration_points, species_database_path
from core.integration import integrate_peak, load_peak_config
from core.isotope import calculate_isotope_distribution, parse_formula
from core.normalization import extract_light_intensity
from core.peak_detection import GaussianFit, Peak, detect_peaks_in_range
from core.peak_ranges import load_peak_ranges
from core.pie_analysis import (
    analyze_pie_folder,
    build_pie_curves,
    fit_species_combination_with_curve,
    load_species_database,
    save_species_database_sqlite,
)
from core.temperature_scan import analyze_temperature_folder, build_temperature_curves, compute_kr_expansion_factors


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


def test_isotope_distribution_contains_main_peak():
    assert parse_formula("H2O") == {"H": 2, "O": 1}
    rows = calculate_isotope_distribution("H2O")
    assert rows
    assert rows[0]["percent"] > 99


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
