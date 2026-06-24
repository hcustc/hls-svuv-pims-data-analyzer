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


def test_integrate_peak_falls_back_when_gaussian_area_is_zero():
    y = [0.0] * 100
    y[42] = 46.0
    y[43] = 30.0
    y[44] = 8.0
    y[45] = 2.0
    peak = Peak(
        index=50,
        time=50.0,
        mz=84.0,
        intensity=0.0,
        fwhm=21.0,
        left_bound=40,
        right_bound=61,
        is_auto=False,
    )

    assert integrate_peak(y, peak, prefer_gaussian=True) == pytest.approx(
        integrate_peak(y, peak, prefer_gaussian=False)
    )
    assert integrate_peak(y, peak, prefer_gaussian=True) > 0


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


def test_ensemble_votes_count_distinct_algorithms(monkeypatch):
    duplicate_peaks = [
        Peak(index=22, time=22.0, mz=22.0, intensity=10.0, fwhm=2.0, left_bound=21, right_bound=23),
        Peak(index=23, time=23.0, mz=22.1, intensity=9.0, fwhm=2.0, left_bound=22, right_bound=24),
    ]

    monkeypatch.setattr(peak_detection_module, "detect_peaks_in_range", lambda *args, **kwargs: duplicate_peaks)
    monkeypatch.setattr(peak_detection_module, "detect_peaks_prominence", lambda *args, **kwargs: [])

    peaks = detect_peaks_ensemble(
        [1.0] * 50,
        calibration=Calibration(a=0, b=1, c=0),
        detection_min_idx=0,
        use_legacy=True,
        use_prominence=True,
        use_cwt=False,
        vote_threshold=1.0,
        min_intensity_for_single_vote=100.0,
        mz_tolerance=0.2,
    )

    assert peaks == []


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
