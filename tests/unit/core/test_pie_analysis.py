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
    assert df.iloc[0]["integration_method"] == "sum_counts"


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
    )
    curves = build_pie_curves(df)
    assert 22 in curves
    assert curves[22]["energies"] == [11.0, 12.0]
