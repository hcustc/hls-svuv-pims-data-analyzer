import sqlite3

import numpy as np
import pandas as pd
import pytest

from bl03u_masstool.core.calibration import Calibration, fit_quadratic_calibration
from bl03u_masstool.core.cwt_peak_detection import CwtPeakDetectionConfig, detect_peaks_cwt
from bl03u_masstool.core.config import load_calibration_config, load_calibration_points, species_database_path
from bl03u_masstool.core.integration import (
    integrate_peak,
    integrate_peak_with_method,
    integrate_peaks_with_method,
    load_peak_config,
)
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
    _select_kr_reference_peak_rows,
    analyze_temperature_folder,
    annotate_temperature_curve_groups,
    build_temperature_curves,
    classify_temperature_curve,
    compute_kr_expansion_factors,
    identify_product_energy_intervals,
)


def test_kr_reference_uses_one_precise_peak_in_nominal_mass_collision():
    rows = pd.DataFrame(
        [
            {
                "temperature": 650.0,
                "mz": 83.587,
                "mz_rounded": 84,
                "temperature_peak_track": 0,
                "photon_normalized_area": 1000.0,
            },
            {
                "temperature": 750.0,
                "mz": 83.587,
                "mz_rounded": 84,
                "temperature_peak_track": 0,
                "photon_normalized_area": 1000.0,
            },
            {
                "temperature": 650.0,
                "mz": 84.023,
                "mz_rounded": 84,
                "temperature_peak_track": 1,
                "photon_normalized_area": 10.0,
            },
            {
                "temperature": 750.0,
                "mz": 84.023,
                "mz_rounded": 84,
                "temperature_peak_track": 1,
                "photon_normalized_area": 20.0,
            },
        ]
    )

    selected = _select_kr_reference_peak_rows(rows, kr_mz=84)

    assert selected["temperature_peak_track"].tolist() == [1, 1]
    assert selected["photon_normalized_area"].tolist() == [10.0, 20.0]


def test_build_temperature_curves_aggregates_replicates_of_the_same_peak():
    df = pd.DataFrame(
        [
            {"temperature": 800.0, "file": "a.txt", "reference_temperature": 900.0, "mz": 17.99, "area": 10.0, "replicate_mode": "sum", "left_bound": 100, "right_bound": 110},
            {"temperature": 825.0, "file": "b.txt", "reference_temperature": 900.0, "mz": 18.02, "area": 12.0, "replicate_mode": "sum", "left_bound": 100, "right_bound": 110},
            {"temperature": 825.0, "file": "c.txt", "reference_temperature": 900.0, "mz": 18.01, "area": 3.0, "replicate_mode": "sum", "left_bound": 100, "right_bound": 110},
        ]
    )
    curves = build_temperature_curves(df)
    curve_key, curve = next(iter(curves.items()))
    assert curve_key == pytest.approx(18.0025)
    assert curve["mz"] == pytest.approx(curve_key)
    assert curve["mz_rounded"] == 18
    assert curve["temperatures"] == [800.0, 825.0]
    assert curve["areas"] == [10.0, 15.0]


def test_temperature_curves_keep_close_peaks_with_the_same_nominal_mz_separate():
    rows = []
    for temperature, low_mz, high_mz, low_area, high_area in (
        (400.0, 228.020, 228.100, 1.0, 10.0),
        (500.0, 228.022, 228.098, 2.0, 20.0),
        (600.0, 228.021, 228.099, 3.0, 30.0),
    ):
        rows.extend(
            [
                {
                    "temperature": temperature,
                    "file": f"{temperature:.0f}.txt",
                    "reference_temperature": 600.0,
                    "mz": low_mz,
                    "area": low_area,
                    "replicate_mode": "off",
                    "left_bound": 100,
                    "right_bound": 110,
                },
                {
                    "temperature": temperature,
                    "file": f"{temperature:.0f}.txt",
                    "reference_temperature": 600.0,
                    "mz": high_mz,
                    "area": high_area,
                    "replicate_mode": "off",
                    "left_bound": 112,
                    "right_bound": 122,
                },
            ]
        )
    result = pd.DataFrame(rows)

    curves = build_temperature_curves(result)

    assert len(curves) == 2
    assert all(curve["mz"] == pytest.approx(key) for key, curve in curves.items())
    assert all(curve["mz_rounded"] == 228 for curve in curves.values())
    assert all(curve["has_nominal_collision"] for curve in curves.values())
    ordered = sorted(curves.values(), key=lambda curve: curve["mz_exact_mean"])
    assert ordered[0]["mz_exact_mean"] == pytest.approx(228.021)
    assert ordered[0]["temperatures"] == [400.0, 500.0, 600.0]
    assert ordered[0]["areas"] == [1.0, 2.0, 3.0]
    assert ordered[1]["mz_exact_mean"] == pytest.approx(228.099)
    assert ordered[1]["temperatures"] == [400.0, 500.0, 600.0]
    assert ordered[1]["areas"] == [10.0, 20.0, 30.0]

    annotated = annotate_temperature_curve_groups(result, curves=curves)
    assert annotated["temperature_curve_key"].nunique() == 2
    assert annotated.groupby("temperature_curve_key").size().tolist() == [3, 3]
    assert set(annotated["temperature_peak_track"]) == {0, 1}


@pytest.mark.parametrize("method", ["sum_counts", "baseline"])
def test_batch_peak_integration_matches_individual_integration_exactly(method):
    y = np.asarray([0.0, 1.0, 4.0, 2.0, 0.0, 3.0, 7.0, 1.0, 0.0])
    peaks = [
        Peak(index=2, time=2.0, mz=2.0, intensity=4.0, fwhm=1.0, left_bound=1, right_bound=3),
        Peak(index=6, time=6.0, mz=6.0, intensity=7.0, fwhm=1.0, left_bound=5, right_bound=7),
    ]

    individual = [
        integrate_peak_with_method(
            y,
            peak,
            prefer_gaussian=False,
            integration_method=method,
        )
        for peak in peaks
    ]
    batched = integrate_peaks_with_method(
        y,
        peaks,
        prefer_gaussian=False,
        integration_method=method,
    )

    assert batched == individual


def test_explicit_integration_method_wins_over_gaussian_preference():
    y = np.asarray([5.0, 10.0, 5.0])
    peak = Peak(index=1, time=1.0, mz=1.0, intensity=10.0, fwhm=1.0, left_bound=0, right_bound=2)

    individual = integrate_peak_with_method(
        y, peak, prefer_gaussian=True, integration_method="baseline"
    )
    batched = integrate_peaks_with_method(
        y, [peak], prefer_gaussian=True, integration_method="baseline"
    )

    assert individual == (pytest.approx(5.0), "baseline")
    assert batched == [individual]


def test_temperature_filename_replicates_can_average_or_sum(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")

    def write_spectrum(name: str, energy: float, scale: float):
        y = [0.0] * 50
        y[22] = scale
        header = [
            f"Energy:{energy} eV",
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
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("C24110904-0000.txt", 14.601, 10.0)
    write_spectrum("C24110904-0001.txt", 14.699, 20.0)

    mean_df = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        replicate_mode="mean",
    )
    mean_curves = build_temperature_curves(mean_df)
    assert mean_df.iloc[0]["replicate_grouping"] == "filename"
    assert mean_df.iloc[0]["replicate_warning"] == ""
    assert mean_df.iloc[0]["file_count"] == 2
    assert mean_df.iloc[0]["integration_method"] == "sum_counts"
    assert mean_curves[22]["areas"] == [15.0]
    assert mean_curves[22]["rows"].iloc[0]["integration_method"] == "sum_counts"

    sum_df = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        replicate_mode="sum",
    )
    sum_curves = build_temperature_curves(sum_df)
    assert sum_curves[22]["areas"] == [30.0]


def test_temperature_analysis_can_cache_the_exact_annotated_curves(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")
    y = [0.0] * 50
    y[22] = 10.0
    header = [
        "Energy:12.0 eV",
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
    (tmp_path / "400C.txt").write_text(
        "\n".join(header + [str(value) for value in y]),
        encoding="utf-8",
    )

    result = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        cache_curves=True,
    )
    cached = result.attrs["_bl03u_temperature_curves"]
    rebuilt = build_temperature_curves(result)

    assert cached.keys() == rebuilt.keys()
    for mz in cached:
        assert cached[mz]["temperatures"] == rebuilt[mz]["temperatures"]
        assert cached[mz]["areas"] == rebuilt[mz]["areas"]
        assert cached[mz]["curve_class"] == rebuilt[mz]["curve_class"]
        pd.testing.assert_frame_equal(cached[mz]["rows"], rebuilt[mz]["rows"])


def test_temperature_root_with_energy_subfolders_is_discovered(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")

    root = tmp_path / "temperature_root"
    low = root / "8eV"
    high = root / "9.5eV"
    low.mkdir(parents=True)
    high.mkdir(parents=True)

    def write_spectrum(path, *, energy: float, temperature: float, scale: float):
        y = [0.0] * 50
        y[22] = scale
        header = [
            f"Energy:{energy} eV",
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
        path.write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum(low / "low_650.txt", energy=8.0, temperature=650.0, scale=10.0)
    write_spectrum(high / "high_750.txt", energy=9.5, temperature=750.0, scale=20.0)

    df = analyze_temperature_folder(
        root,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
    )

    assert sorted(df["photon_energy"].unique()) == [8.0, 9.5]
    assert sorted(df["temperature"].unique()) == [650.0, 750.0]


def test_temperature_can_use_baseline_corrected_integration(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")
    y = [0.0] * 50
    y[21] = 5.0
    y[22] = 10.0
    y[23] = 5.0
    header = [
        "Energy:12.0 eV",
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
    (tmp_path / "400C.txt").write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    summed = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        integration_method="sum_counts",
        photon_normalize=False,
    )
    baseline = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        prefer_gaussian=True,
        integration_method="baseline",
        photon_normalize=False,
    )

    assert summed.iloc[0]["raw_area"] == 20.0
    assert summed.iloc[0]["integration_method"] == "sum_counts"
    assert baseline.iloc[0]["raw_area"] == 5.0
    assert baseline.iloc[0]["integration_method"] == "baseline"


def test_temperature_replicates_are_not_merged_when_disabled(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")

    def write_spectrum(name: str, scale: float):
        y = [0.0] * 50
        y[22] = scale
        header = [
            "Energy:14.6 eV",
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
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("C24110904-0000.txt", 10.0)
    write_spectrum("C24110904-0001.txt", 20.0)

    df = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        replicate_mode="off",
    )
    curves = build_temperature_curves(df)
    assert list(df["replicate_mode"].unique()) == ["off"]
    assert curves[22]["temperatures"] == [400.0, 400.0]
    assert curves[22]["areas"] == [10.0, 20.0]


def test_temperature_fallback_grouping_warns_when_filename_replicates_are_absent(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")

    def write_spectrum(name: str, scale: float):
        y = [0.0] * 50
        y[22] = scale
        header = [
            "Energy:14.6 eV",
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
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("sample-a.txt", 10.0)
    write_spectrum("sample-b.txt", 20.0)

    df = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        replicate_mode="mean",
    )
    curves = build_temperature_curves(df)
    assert df.iloc[0]["replicate_grouping"] == "temperature"
    assert "退回按温度分组" in df.iloc[0]["replicate_warning"]
    assert curves[22]["areas"] == [15.0]


def test_temperature_curve_classification_trends():
    temperatures = [400, 500, 600, 700, 800]
    assert classify_temperature_curve(temperatures, [0, 1, 3, 6, 10])["curve_class"] == "formation"
    assert classify_temperature_curve(temperatures, [10, 6, 3, 1, 0])["curve_class"] == "consumption"
    assert classify_temperature_curve(temperatures, [0, 2, 10, 2, 0])["curve_class"] == "intermediate"
    assert classify_temperature_curve(temperatures, [3, 3.1, 3, 3.1, 3])["curve_class"] == "unclassified"


def test_identify_product_energy_intervals_uses_product_like_curve_classes():
    result = identify_product_energy_intervals(
        [
            {"energy": 8.0, "curve_class": "unclassified"},
            {"energy": 8.7, "curve_class": "formation"},
            {"energy": 9.0, "curve_class": "intermediate"},
            {"energy": 9.5, "curve_class": "consumption"},
            {"energy": 11.0, "curve_class": "formation"},
            {"energy": 11.5, "curve_class": "formation"},
        ],
        max_gap=0.8,
    )

    assert result["product_energies"] == [8.7, 9.0, 11.0, 11.5]
    assert result["intervals"] == [
        {"start": 8.7, "end": 9.0, "energies": [8.7, 9.0]},
        {"start": 11.0, "end": 11.5, "energies": [11.0, 11.5]},
    ]
    assert result["rows"]["is_product_like"].tolist() == [False, True, True, False, True, True]


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
    assert [round(value, 6) for value in curves[22]["areas"]] == [1.2, 1.2]
    assert set(df["integration_method"]) == {"sum_counts"}


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
    assert round(float(df.iloc[0]["area"]), 6) == 0.12


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


def test_kr_expansion_groups_jittered_energies_before_lambda_calculation(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n84,22,21,23\n", encoding="utf-8")

    def write_spectrum(name: str, energy: float, temperature: float, scale: float):
        y = [0.0] * 50
        y[21] = 1.0 * scale
        y[22] = 10.0 * scale
        y[23] = 1.0 * scale
        header = [
            f"Energy:{energy} eV",
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

    write_spectrum("e146-low.txt", 14.6092, 400.0, 1.0)
    write_spectrum("e146-high.txt", 14.6104, 800.0, 2.0)
    write_spectrum("e147-low.txt", 14.7092, 400.0, 3.0)
    write_spectrum("e147-high.txt", 14.7101, 800.0, 6.0)

    factors = compute_kr_expansion_factors(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        kr_mz=84,
        manual_peak_path=peak_file,
        light_source="io",
        prefer_gaussian=False,
    )

    assert factors["photon_energy"].nunique() == 2
    by_energy_temp = factors.pivot_table(
        index="temperature",
        columns="photon_energy",
        values="expansion_lambda",
        aggfunc="first",
    )
    assert by_energy_temp.loc[400.0].tolist() == [1.0, 1.0]
    assert by_energy_temp.loc[800.0].tolist() == [2.0, 2.0]


def test_temperature_scan_maps_kr_expansion_by_energy_and_temperature(tmp_path):
    peak_file = tmp_path / "peaks.csv"
    peak_file.write_text("mz,peak,start,end\n84,84,83,85\n", encoding="utf-8")

    def write_spectrum(name: str, energy: float, temperature: float, scale: float):
        y = [0.0] * 100
        y[83] = 1.0 * scale
        y[84] = 10.0 * scale
        y[85] = 1.0 * scale
        header = [
            f"Energy:{energy} eV",
            "IO:1 nA",
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

    write_spectrum("e146-low.txt", 14.6, 400.0, 1.0)
    write_spectrum("e146-high.txt", 14.6, 800.0, 2.0)
    write_spectrum("e147-low.txt", 14.7, 400.0, 1.0)
    write_spectrum("e147-high.txt", 14.7, 800.0, 4.0)

    result = analyze_temperature_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        manual_peak_path=peak_file,
        photon_normalize=True,
        prefer_gaussian=False,
        kr_correct=True,
        expansion_factors={
            14.6: {400.0: 1.0, 800.0: 2.0},
            14.7: {400.0: 1.0, 800.0: 4.0},
        },
    )

    by_energy_temp = result.pivot_table(
        index="temperature",
        columns="photon_energy",
        values=["expansion_lambda", "area"],
        aggfunc="first",
    )
    assert by_energy_temp[("expansion_lambda", 14.6)].loc[800.0] == pytest.approx(2.0)
    assert by_energy_temp[("expansion_lambda", 14.7)].loc[800.0] == pytest.approx(4.0)
    assert by_energy_temp[("area", 14.6)].loc[800.0] == pytest.approx(12.0)
    assert by_energy_temp[("area", 14.7)].loc[800.0] == pytest.approx(12.0)


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
