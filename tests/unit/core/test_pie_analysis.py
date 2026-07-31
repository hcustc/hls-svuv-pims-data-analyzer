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
    analyze_multiple_pie_folders,
    analyze_pie_folder,
    build_pie_curves,
    build_pie_ratio_curve,
    discover_pie_segment_folders,
    fit_species_combination_with_curve,
    inspect_pie_source_segments,
    load_species_database,
    merge_pie_segments,
    save_species_database_sqlite,
)
from bl03u_masstool.core.temperature_scan import (
    analyze_temperature_folder,
    build_temperature_curves,
    classify_temperature_curve,
    compute_kr_expansion_factors,
)


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
    curve_key, curve = next(iter(curves.items()))
    assert curve_key == pytest.approx(22.0)
    assert curve["mz"] == pytest.approx(curve_key)
    assert curve["mz_rounded"] == 22
    assert curve["energies"] == [11.0, 12.0]


def test_build_pie_curves_keeps_precise_peaks_separate_within_nominal_mz():
    analysis_df = pd.DataFrame(
        {
            "energy": [7.0, 8.0, 7.0, 8.0],
            "mz": [227.587157, 227.587157, 228.023117, 228.023117],
            "mz_rounded": [228, 228, 228, 228],
            "normalized_intensity": [0.0, 0.2, 1.0, 10.0],
            "raw_area": [0.0, 2.0, 10.0, 100.0],
            "photon_normalized_intensity": [0.0, 0.2, 1.0, 10.0],
            "integration_method": ["sum_counts"] * 4,
            "species": [""] * 4,
            "file_count": [1] * 4,
            "io": [1.0] * 4,
            "replicate_mode": ["off"] * 4,
            "left_bound": [24076, 24076, 24091, 24091],
            "right_bound": [24086, 24086, 24118, 24118],
        }
    )

    curves = build_pie_curves(analysis_df)

    assert set(curves) == {227.587157, 228.023117}
    assert curves[227.587157]["mz"] == pytest.approx(227.587157)
    assert curves[227.587157]["mz_rounded"] == 228
    assert curves[227.587157]["energies"] == [7.0, 8.0]
    assert curves[227.587157]["intensities"] == [0.0, 0.2]
    assert curves[228.023117]["energies"] == [7.0, 8.0]
    assert curves[228.023117]["intensities"] == [1.0, 10.0]
    assert all(curve["has_nominal_collision"] for curve in curves.values())


def test_build_pie_curves_uses_precise_key_for_single_peak():
    analysis_df = pd.DataFrame(
        {
            "energy": [7.0, 8.0],
            "mz": [228.023117, 228.023117],
            "mz_rounded": [228, 228],
            "normalized_intensity": [1.0, 10.0],
            "replicate_mode": ["off", "off"],
        }
    )

    curves = build_pie_curves(analysis_df)

    assert set(curves) == {228.023117}
    assert curves[228.023117]["mz"] == pytest.approx(228.023117)
    assert curves[228.023117]["mz_rounded"] == 228
    assert curves[228.023117]["mz_exact_mean"] == pytest.approx(228.023117)
    assert curves[228.023117]["has_nominal_collision"] is False


def test_build_pie_curves_tracks_one_peak_across_small_mz_drift():
    analysis_df = pd.DataFrame(
        {
            "energy": [9.0, 10.0, 11.0],
            "mz": [28.010, 28.015, 28.020],
            "mz_rounded": [28, 28, 28],
            "normalized_intensity": [1.0, 2.0, 3.0],
            "replicate_mode": ["off", "off", "off"],
        }
    )

    curves = build_pie_curves(analysis_df)

    assert set(curves) == {28.015}
    assert curves[28.015]["energies"] == [9.0, 10.0, 11.0]
    assert curves[28.015]["mz"] == pytest.approx(28.015)
    assert curves[28.015]["mz_rounded"] == 28


def test_discover_pie_segment_folders_detects_temporary_multi_segment_parent(tmp_path):
    low_folder = tmp_path / "PIE_low"
    high_folder = tmp_path / "PIE_high"
    low_folder.mkdir()
    high_folder.mkdir()
    for folder, energies in ((low_folder, (8.0, 9.0)), (high_folder, (9.0, 10.0))):
        for energy in energies:
            (folder / f"{energy:.1f}eV.txt").write_text("spectrum", encoding="utf-8")

    assert discover_pie_segment_folders(tmp_path) == [high_folder, low_folder]

    summaries = inspect_pie_source_segments(tmp_path)
    assert [summary.folder for summary in summaries] == [low_folder, high_folder]
    assert [summary.file_count for summary in summaries] == [2, 2]
    assert (summaries[0].min_energy, summaries[0].max_energy) == (8.0, 9.0)
    assert (summaries[1].min_energy, summaries[1].max_energy) == (9.0, 10.0)


@pytest.mark.parametrize("suffix", ["eV", ""])
def test_discover_pie_segment_folders_keeps_energy_subdirectories_as_one_source(tmp_path, suffix):
    for energy in (8.0, 9.0):
        energy_folder = tmp_path / f"{energy:.1f}{suffix}"
        energy_folder.mkdir()
        (energy_folder / "spectrum-a.txt").write_text("spectrum", encoding="utf-8")
        (energy_folder / "spectrum-b.txt").write_text("spectrum", encoding="utf-8")

    assert discover_pie_segment_folders(tmp_path) == [tmp_path]


def test_discover_pie_segment_folders_ignores_empty_candidates(tmp_path):
    populated = tmp_path / "PIE_populated"
    empty = tmp_path / "PIE_empty"
    populated.mkdir()
    empty.mkdir()
    (populated / "9.0eV.txt").write_text("spectrum", encoding="utf-8")

    assert discover_pie_segment_folders(tmp_path) == [tmp_path]


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
    assert [round(value, 6) for value in curves[22]["intensities"]] == [1.2, 1.2]
    assert set(df["integration_method"]) == {"sum_counts"}


def test_pie_groups_average_replicates_and_subtract_same_energy_blank(tmp_path):
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

    def write_spectrum(name, scale):
        y = [0.0] * 50
        y[22] = scale
        header = [
            "Energy:12.0 eV",
            "IO:10 nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("12.0eV-sample-a.txt", 10.0)
    write_spectrum("12.0eV-sample-b.txt", 20.0)
    write_spectrum("12.0eV-blank.txt", 5.0)

    df = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        replicate_mode="mean",
    )

    assert len(df) == 1
    assert df.iloc[0]["file_count"] == 2
    assert df.iloc[0]["blank_file_count"] == 1
    assert bool(df.iloc[0]["background_subtracted"]) is True
    assert df.iloc[0]["raw_area"] == 10.0
    assert df.iloc[0]["signal_to_noise"] == pytest.approx(
        10.0 / np.sqrt(12.5)
    )
    assert df.iloc[0]["signal_to_noise_method"] == "poisson_peak_window"
    assert df.iloc[0]["integration_method"] == "sum_counts"


def test_pie_background_snr_propagates_unequal_exposures(tmp_path):
    peak_file = tmp_path / "peaks.yaml"
    peak_file.write_text(
        "peaks:\n  - mz: 22\n    peak: 22\n    start: 22\n    end: 22\n",
        encoding="utf-8",
    )

    def write_spectrum(name, acquisition_time_s, count):
        y = [0.0] * 50
        y[22] = count
        header = [
            "Energy:12.0 eV",
            "IO:10 nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            f"Time:{acquisition_time_s} s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text(
            "\n".join(header + [str(value) for value in y]),
            encoding="utf-8",
        )

    write_spectrum("12.0eV-sample.txt", 60, 1200)
    write_spectrum("12.0eV-blank.txt", 30, 300)

    row = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=True,
        photon_reference_mode="none",
        normalize_by_time=True,
    ).iloc[0]

    sample_rate = 1200.0 / (60.0 * 10.0)
    blank_rate = 300.0 / (30.0 * 10.0)
    variance = (
        1200.0 / (60.0 * 10.0) ** 2
        + 300.0 / (30.0 * 10.0) ** 2
    )
    assert row["merged_intensity"] == pytest.approx(sample_rate - blank_rate)
    assert row["signal_to_noise"] == pytest.approx(
        (sample_rate - blank_rate) / np.sqrt(variance)
    )


def test_pie_records_sum_counts_fallback_when_gaussian_fit_fails(tmp_path):
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
    y = [0.0] * 50
    y[22] = 10.0
    header = [
        "Energy:12.0 eV",
        "IO:10 nA",
        "Beam Current:1mA",
        "Undulator Offset:0mm",
        "Time:1 s",
        "Burner Position:0 mm",
        "Temperature:300 C",
        "DIFF PRESSURE:1Pa",
        "ION PRESSURE:1Pa",
        "TOF PRESSURE:1Pa",
    ]
    (tmp_path / "spectrum.txt").write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    df = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=True,
        photon_normalize=False,
    )

    assert df.iloc[0]["raw_area"] == 10.0
    assert df.iloc[0]["integration_method"] == "sum_counts"


def test_pie_can_use_baseline_corrected_integration(tmp_path):
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
        "Temperature:300 C",
        "DIFF PRESSURE:1Pa",
        "ION PRESSURE:1Pa",
        "TOF PRESSURE:1Pa",
    ]
    (tmp_path / "spectrum.txt").write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    summed = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        integration_method="sum_counts",
        photon_normalize=False,
    )
    baseline = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        integration_method="baseline",
        photon_normalize=False,
    )

    assert summed.iloc[0]["raw_area"] == 20.0
    assert summed.iloc[0]["integration_method"] == "sum_counts"
    assert baseline.iloc[0]["raw_area"] == 5.0
    assert baseline.iloc[0]["integration_method"] == "baseline"


def test_explicit_pie_integration_method_wins_over_gaussian_preference(tmp_path):
    peak_file = tmp_path / "peaks.yaml"
    peak_file.write_text(
        "peaks:\n  - mz: 22\n    peak: 22\n    start: 21\n    end: 23\n",
        encoding="utf-8",
    )
    y = [0.0] * 50
    y[21:24] = [5.0, 10.0, 5.0]
    header = [
        "Energy:12.0 eV", "IO:10 nA", "Beam Current:1mA",
        "Undulator Offset:0mm", "Time:1 s", "Burner Position:0 mm",
        "Temperature:300 C", "DIFF PRESSURE:1Pa", "ION PRESSURE:1Pa",
        "TOF PRESSURE:1Pa",
    ]
    (tmp_path / "spectrum.txt").write_text(
        "\n".join(header + [str(value) for value in y]), encoding="utf-8"
    )

    result = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=True,
        integration_method="baseline",
        photon_normalize=False,
    )

    assert result.iloc[0]["integration_method"] == "baseline"
    assert result.iloc[0]["raw_area"] == pytest.approx(5.0)


def test_analyze_multiple_pie_folders_merges_energy_segments(tmp_path):
    peak_file = tmp_path / "peaks.yaml"
    peak_file.write_text(
        "peaks:\n  - mz: 22\n    peak: 22\n    start: 21\n    end: 23\n",
        encoding="utf-8",
    )
    low = tmp_path / "low"
    high = tmp_path / "high"
    low.mkdir()
    high.mkdir()

    def write_spectrum(folder, energy, intensity):
        y = [0.0] * 50
        y[22] = intensity
        header = [
            f"Energy:{energy} eV", "IO:10 nA", "Beam Current:1mA",
            "Undulator Offset:0mm", "Time:1 s", "Burner Position:0 mm",
            "Temperature:300 C", "DIFF PRESSURE:1Pa", "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (folder / f"{energy:.1f}eV.txt").write_text(
            "\n".join(header + [str(value) for value in y]), encoding="utf-8"
        )

    write_spectrum(low, 9.0, 10.0)
    write_spectrum(low, 10.0, 20.0)
    write_spectrum(high, 10.0, 2.0)
    write_spectrum(high, 11.0, 3.0)

    shared_options = {
        "calibration": Calibration(a=0, b=1, c=0),
        "recursive": False,
        "manual_peak_path": peak_file,
        "prefer_gaussian": False,
        "integration_method": "sum_counts",
        "photon_normalize": False,
    }
    single_result = analyze_pie_folder(low, **shared_options)
    single_via_multi = analyze_multiple_pie_folders(
        [low], merge_method="low_energy_dominant", **shared_options
    )
    common_columns = list(single_result.columns)
    pd.testing.assert_frame_equal(
        single_result.reset_index(drop=True),
        single_via_multi[common_columns].reset_index(drop=True),
        check_dtype=False,
    )

    with pytest.raises(ValueError, match="共享缩放QC失败"):
        analyze_multiple_pie_folders(
            [low, high],
            merge_method="low_energy_dominant",
            **shared_options,
        )


def test_pie_filename_replicates_ignore_small_energy_drift(tmp_path):
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

    def write_spectrum(name, energy, scale):
        y = [0.0] * 50
        y[22] = scale
        header = [
            f"Energy:{energy} eV",
            "IO:10 nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("C24110904-0000.txt", 14.601, 10.0)
    write_spectrum("C24110904-0001.txt", 14.699, 20.0)

    df = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        energy_decimals=1,
        replicate_mode="mean",
    )

    assert len(df) == 1
    assert df.iloc[0]["file_count"] == 2
    assert df.iloc[0]["replicate_grouping"] == "filename"
    assert df.iloc[0]["replicate_warning"] == ""
    assert df.iloc[0]["raw_area"] == 15.0


def test_pie_filename_sequence_across_energy_scan_is_not_replicate(tmp_path):
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

    def write_spectrum(name, energy, scale):
        y = [0.0] * 50
        y[22] = scale
        header = [
            f"Energy:{energy} eV",
            "IO:10 nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("C24110901-0000.txt", 8.00, 10.0)
    write_spectrum("C24110901-0001.txt", 8.05, 20.0)
    write_spectrum("C24110901-0002.txt", 8.10, 30.0)
    write_spectrum("C24110901-0003.txt", 8.15, 40.0)

    df = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        energy_decimals=1,
        replicate_mode="off",
    )

    curves = build_pie_curves(df)
    assert curves[22]["energies"] == [8.0, 8.05, 8.1, 8.15]
    assert curves[22]["intensities"] == [10.0, 20.0, 30.0, 40.0]
    assert set(df["replicate_grouping"]) == {"energy"}


def test_pie_fallback_grouping_warns_when_filename_replicates_are_absent(tmp_path):
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

    def write_spectrum(name, scale):
        y = [0.0] * 50
        y[22] = scale
        header = [
            "Energy:12.0 eV",
            "IO:10 nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            "Time:1 s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")

    write_spectrum("sample-a.txt", 10.0)
    write_spectrum("sample-b.txt", 20.0)

    df = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        photon_normalize=False,
        replicate_mode="sum",
    )

    assert len(df) == 1
    assert df.iloc[0]["replicate_grouping"] == "energy"
    assert "退回按能量分组" in df.iloc[0]["replicate_warning"]
    assert df.iloc[0]["raw_area"] == 30.0


def test_pie_blank_subtraction_rejects_mismatched_x_axis(tmp_path):
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

    header = [
        "Energy:12.0 eV",
        "IO:10 nA",
        "Beam Current:1mA",
        "Undulator Offset:0mm",
        "Time:1 s",
        "Burner Position:0 mm",
        "Temperature:300 C",
        "DIFF PRESSURE:1Pa",
        "ION PRESSURE:1Pa",
        "TOF PRESSURE:1Pa",
    ]
    sample_rows = [f"{index} {10.0 if index == 22 else 0.0}" for index in range(1, 51)]
    blank_rows = [f"{index + 0.5} {5.0 if index == 22 else 0.0}" for index in range(1, 51)]
    (tmp_path / "12.0eV-sample.txt").write_text("\n".join(header + sample_rows), encoding="utf-8")
    (tmp_path / "12.0eV-blank.txt").write_text("\n".join(header + blank_rows), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched x axes"):
        analyze_pie_folder(
            tmp_path,
            calibration=Calibration(a=0, b=1, c=0),
            recursive=False,
            manual_peak_path=peak_file,
            prefer_gaussian=False,
            photon_normalize=False,
        )


def test_pie_skips_files_without_parseable_energy(tmp_path, caplog):
    y = [0.0] * 20 + [1.0, 5.0, 12.0, 5.0, 1.0] + [0.0] * 20
    (tmp_path / "sample_without_energy.asc").write_text("\n".join(str(value) for value in y), encoding="utf-8")

    with caplog.at_level("WARNING"):
        df = analyze_pie_folder(
            tmp_path,
            calibration=Calibration(a=0, b=1, c=0),
            recursive=False,
            detection_min_idx=0,
            threshold_end=0.5,
            min_intensity=3,
            prefer_gaussian=False,
        )

    assert df.empty
    assert "no photon energy" in caplog.text


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
        photon_normalize=False,
        normalize_by_time=False,
    )
    curves = build_pie_curves(df)
    curve_key, curve = next(iter(curves.items()))
    assert curve_key == pytest.approx(22.0)
    assert curve["mz_rounded"] == 22
    assert curve["energies"] == [11.0, 12.0]


def test_merge_pie_segments_scales_high_energy_segment_from_overlap_ratio():
    def segment(energies, intensities):
        return pd.DataFrame(
            {
                "energy": energies,
                "mz_rounded": [30] * len(energies),
                "normalized_intensity": intensities,
                "raw_area": intensities,
                "photon_normalized_intensity": intensities,
                "integration_method": ["sum_counts"] * len(energies),
                "mz": [30.0] * len(energies),
                "species": [""] * len(energies),
                "file_count": [1] * len(energies),
                "io": [1.0] * len(energies),
                "light_source": ["io"] * len(energies),
                "left_bound": [29] * len(energies),
                "right_bound": [31] * len(energies),
            }
        )

    low_energy = segment([8.0, 9.0, 10.0], [10.0, 20.0, 30.0])
    high_energy = segment([9.0, 10.0, 11.0, 12.0], [2.0, 3.0, 4.0, 5.0])

    merged = merge_pie_segments(
        [low_energy, high_energy],
        merge_method="low_energy_dominant",
        min_overlap_energies=2,
        min_scale_channels=1,
        min_scale_snr=0,
    ).set_index("energy")

    assert merged.loc[9.0, "normalized_intensity"] == pytest.approx(20.0)
    assert merged.loc[10.0, "normalized_intensity"] == pytest.approx(30.0)
    assert merged.loc[11.0, "normalized_intensity"] == pytest.approx(40.0)
    assert merged.loc[12.0, "normalized_intensity"] == pytest.approx(50.0)


def test_merge_pie_segments_preserves_multiple_precise_peaks_per_nominal_mass():
    def segment(energies, peak_rows):
        rows = []
        for exact_mz, left_bound, right_bound, intensities in peak_rows:
            for energy, intensity in zip(energies, intensities):
                rows.append(
                    {
                        "energy": energy,
                        "mz_rounded": 228,
                        "normalized_intensity": intensity,
                        "raw_area": intensity,
                        "photon_normalized_intensity": intensity,
                        "integration_method": "sum_counts",
                        "mz": exact_mz,
                        "species": "",
                        "file_count": 1,
                        "io": 1.0,
                        "light_source": "io",
                        "left_bound": left_bound,
                        "right_bound": right_bound,
                    }
                )
        return pd.DataFrame(rows)

    low_energy = segment(
        [8.0, 9.0, 10.0],
        [
            (227.58715739409433, 400, 405, [1.0, 2.0, 3.0]),
            (228.02311680380544, 410, 416, [10.0, 20.0, 30.0]),
        ],
    )
    high_energy = segment(
        [9.0, 10.0, 11.0],
        [
            (227.58715739409433, 400, 405, [0.2, 0.3, 0.4]),
            (228.02311680380544, 410, 416, [2.0, 3.0, 4.0]),
        ],
    )

    merged = merge_pie_segments(
        [low_energy, high_energy],
        merge_method="low_energy_dominant",
        min_overlap_energies=2,
        min_scale_channels=2,
        min_scale_snr=0,
    )
    curves = build_pie_curves(merged)

    assert len(curves) == 2
    assert sorted(curves) == pytest.approx(
        [227.58715739409433, 228.02311680380544]
    )
    assert all(curve["mz"] == pytest.approx(key) for key, curve in curves.items())
    assert all(curve["mz_rounded"] == 228 for curve in curves.values())
    assert all(curve["energies"] == [8.0, 9.0, 10.0, 11.0] for curve in curves.values())


@pytest.mark.parametrize(
    ("merge_method", "expected_overlap"),
    [("mean", 11.0), ("first_segment_dominant", 2.0)],
)
def test_merge_pie_segments_supports_remaining_merge_modes(merge_method, expected_overlap):
    def segment(energies, intensities):
        return pd.DataFrame(
            {
                "energy": energies,
                "mz_rounded": [30] * len(energies),
                "normalized_intensity": intensities,
                "raw_area": intensities,
                "photon_normalized_intensity": intensities,
                "integration_method": ["sum_counts"] * len(energies),
                "mz": [30.0] * len(energies),
                "species": [""] * len(energies),
                "file_count": [1] * len(energies),
                "io": [1.0] * len(energies),
                "light_source": ["io"] * len(energies),
                "left_bound": [29] * len(energies),
                "right_bound": [31] * len(energies),
            }
        )

    first = segment([9.0, 10.0, 11.0], [2.0, 3.0, 4.0])
    lower_energy = segment([8.0, 9.0, 10.0], [10.0, 20.0, 30.0])

    merged = merge_pie_segments(
        [first, lower_energy],
        merge_method=merge_method,
        min_overlap_energies=2,
        min_scale_channels=1,
        min_scale_snr=0,
    ).set_index("energy")

    assert merged.loc[9.0, "normalized_intensity"] == pytest.approx(expected_overlap)


def test_pie_time_and_io_normalization_equalizes_60_and_90_second_scans(
    tmp_path,
):
    peak_file = tmp_path / "peaks.yaml"
    peak_file.write_text(
        "peaks:\n  - mz: 22\n    peak: 22\n    start: 22\n    end: 22\n",
        encoding="utf-8",
    )

    def write_spectrum(name, energy, acquisition_time_s, io, count):
        y = [0.0] * 50
        y[22] = count
        header = [
            f"Energy:{energy} eV",
            f"IO:{io} nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            f"Time:{acquisition_time_s} s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text(
            "\n".join(header + [str(value) for value in y]),
            encoding="utf-8",
        )

    write_spectrum("60s.txt", 8.0, 60, 10, 1200)
    write_spectrum("90s.txt", 9.0, 90, 20, 3600)

    result = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        integration_method="sum_counts",
        photon_normalize=True,
        photon_reference_mode="none",
        normalize_by_time=True,
    ).sort_values("energy")

    assert result["acquisition_time_s"].tolist() == [60.0, 90.0]
    assert result["raw_area"].tolist() == pytest.approx([1200.0, 3600.0])
    assert result["count_rate"].tolist() == pytest.approx([20.0, 40.0])
    assert result["io_time_normalized_intensity"].tolist() == pytest.approx(
        [2.0, 2.0]
    )
    assert result["merged_intensity"].tolist() == pytest.approx([2.0, 2.0])


def test_pie_sum_replicates_normalize_by_total_exposure(tmp_path):
    peak_file = tmp_path / "peaks.yaml"
    peak_file.write_text(
        "peaks:\n  - mz: 22\n    peak: 22\n    start: 22\n    end: 22\n",
        encoding="utf-8",
    )

    def write_spectrum(name, acquisition_time_s, io, count):
        y = [0.0] * 50
        y[22] = count
        header = [
            "Energy:8.0 eV",
            f"IO:{io} nA",
            "Beam Current:1mA",
            "Undulator Offset:0mm",
            f"Time:{acquisition_time_s} s",
            "Burner Position:0 mm",
            "Temperature:300 C",
            "DIFF PRESSURE:1Pa",
            "ION PRESSURE:1Pa",
            "TOF PRESSURE:1Pa",
        ]
        (tmp_path / name).write_text(
            "\n".join(header + [str(value) for value in y]),
            encoding="utf-8",
        )

    write_spectrum("repeat-a.txt", 60, 10, 1200)
    write_spectrum("repeat-b.txt", 90, 20, 3600)

    result = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        prefer_gaussian=False,
        integration_method="sum_counts",
        photon_normalize=True,
        photon_reference_mode="none",
        normalize_by_time=True,
        replicate_mode="sum",
    )

    row = result.iloc[0]
    assert row["file_count"] == 2
    assert row["raw_area"] == pytest.approx(4800.0)
    assert row["count_rate"] == pytest.approx(4800.0 / 150.0)
    assert row["io_normalized_intensity"] == pytest.approx(4800.0 / 30.0)
    assert row["io_time_normalized_intensity"] == pytest.approx(2.0)
    assert row["merged_intensity"] == pytest.approx(2.0)


def test_pie_missing_time_blocks_unless_explicitly_disabled(tmp_path):
    peak_file = tmp_path / "peaks.yaml"
    peak_file.write_text(
        "peaks:\n  - mz: 2\n    peak: 2\n    start: 2\n    end: 2\n",
        encoding="utf-8",
    )
    header = [
        "Energy:8 eV",
        "IO:10 nA",
        "Beam Current:1mA",
        "Undulator Offset:0mm",
        "Burner Position:0 mm",
        "Temperature:300 C",
        "DIFF PRESSURE:1Pa",
        "ION PRESSURE:1Pa",
        "TOF PRESSURE:1Pa",
        "unused:0",
    ]
    (tmp_path / "missing-time.txt").write_text(
        "\n".join(header + ["0", "0", "100", "0"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="缺少有效扫描时间"):
        analyze_pie_folder(
            tmp_path,
            calibration=Calibration(a=0, b=1, c=0),
            recursive=False,
            manual_peak_path=peak_file,
            photon_reference_mode="none",
        )

    result = analyze_pie_folder(
        tmp_path,
        calibration=Calibration(a=0, b=1, c=0),
        recursive=False,
        manual_peak_path=peak_file,
        normalize_by_time=False,
        photon_reference_mode="none",
    )
    assert result.iloc[0]["raw_area"] == pytest.approx(100.0)
    assert not bool(result.iloc[0]["time_normalized"])


def test_shared_segment_scale_preserves_isotope_ratios_and_raw_area():
    def build_segment(energies, factor):
        rows = []
        for mz, abundance in ((140, 1.0), (142, 0.32), (144, 0.16)):
            for energy in energies:
                physical = (1000.0 + 100.0 * (energy - 9.0)) * abundance
                observed = physical / factor
                rows.append(
                    {
                        "energy": energy,
                        "mz_rounded": mz,
                        "normalized_intensity": observed,
                        "merged_intensity": observed,
                        "raw_area": observed,
                        "count_rate": observed,
                        "io_time_normalized_intensity": observed,
                        "photon_normalized_intensity": observed,
                        "signal_to_noise": 20.0,
                        "integration_method": "sum_counts",
                        "source_spectra": [
                            f"/segment-{factor:g}/{energy:g}eV.txt"
                        ],
                        "mz": float(mz),
                        "species": "",
                        "file_count": 1,
                        "io": 1.0,
                        "light_source": "io",
                        "left_bound": mz - 1,
                        "right_bound": mz + 1,
                    }
                )
        return pd.DataFrame(rows)

    low = build_segment([9.0, 10.0, 11.0], 1.0)
    high = build_segment([9.0, 10.0, 11.0, 12.0], 5.0)
    high_raw_140_at_12 = float(
        high[(high["mz_rounded"] == 140) & (high["energy"] == 12.0)][
            "raw_area"
        ].iloc[0]
    )

    merged = merge_pie_segments([low, high])
    high_only = merged[merged["energy"] == 12.0].set_index("mz_rounded")

    assert high_only["shared_scale_factor"].unique() == pytest.approx([5.0])
    assert high_only.loc[140, "merged_intensity"] == pytest.approx(1300.0)
    assert high_only.loc[142, "merged_intensity"] / high_only.loc[
        140, "merged_intensity"
    ] == pytest.approx(0.32)
    assert high_only.loc[140, "raw_area"] == pytest.approx(high_raw_140_at_12)
    assert high_only.loc[140, "io_time_normalized_intensity"] == pytest.approx(
        high_raw_140_at_12
    )
    overlap_140 = merged[
        (merged["energy"] == 10.0) & (merged["mz_rounded"] == 140)
    ].iloc[0]
    assert overlap_140["source_spectra"] == [
        "/segment-1/10eV.txt",
        "/segment-5/10eV.txt",
    ]
    diagnostics = merged.attrs["segment_scaling_diagnostics"]
    assert diagnostics[1]["overlap_energy_count"] == 3
    assert diagnostics[1]["channel_count"] == 3


def test_shared_segment_scale_refuses_insufficient_overlap():
    columns = {
        "mz_rounded": [140, 142, 144],
        "normalized_intensity": [1000.0, 320.0, 160.0],
        "raw_area": [1000.0, 320.0, 160.0],
        "photon_normalized_intensity": [1000.0, 320.0, 160.0],
        "signal_to_noise": [20.0, 20.0, 20.0],
        "mz": [140.0, 142.0, 144.0],
        "species": ["", "", ""],
        "file_count": [1, 1, 1],
        "io": [1.0, 1.0, 1.0],
        "light_source": ["io", "io", "io"],
        "left_bound": [1, 3, 5],
        "right_bound": [2, 4, 6],
    }
    first = pd.DataFrame({"energy": [9.0, 9.0, 9.0], **columns})
    second = pd.DataFrame({"energy": [10.0, 10.0, 10.0], **columns})

    with pytest.raises(ValueError, match="未使用静默1.0因子"):
        merge_pie_segments([first, second])


def test_ratio_curve_masks_low_snr_points_and_keeps_valid_count():
    numerator = pd.DataFrame(
        {
            "energy": [8.0, 9.0, 10.0],
            "merged_intensity": [32.0, 64.0, 96.0],
            "signal_to_noise": [12.0, 2.0, 15.0],
        }
    )
    denominator = pd.DataFrame(
        {
            "energy": [8.0, 9.0, 10.0],
            "merged_intensity": [100.0, 200.0, 300.0],
            "signal_to_noise": [12.0, 15.0, 15.0],
        }
    )

    ratio = build_pie_ratio_curve(
        numerator,
        denominator,
        min_snr=10.0,
    )

    assert ratio["valid"].tolist() == [True, False, True]
    assert ratio.loc[ratio["valid"], "ratio"].tolist() == pytest.approx(
        [0.32, 0.32]
    )
    assert ratio.loc[1, "mask_reason"] == "低SNR"
