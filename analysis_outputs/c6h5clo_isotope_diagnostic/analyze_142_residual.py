from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


OUTPUT_DIR = Path(__file__).resolve().parent
BASE_ANALYSIS = OUTPUT_DIR / "run_analysis.py"

MASS_C7H5_35CLO = 140.00289246272
MASS_C7H5_37CLO = 141.99994238272
MASS_C6H3_35CLO2 = 141.98215701783
MASS_C6H3_37CLO2 = 143.97920693783
MASS_C7H7_35CLO = 142.01854252718
CHLORINE_37_TO_35 = 0.32

BLUE = "#2563EB"
ORANGE = "#D97706"
PINK = "#DB2777"
INK = "#172033"
GRID = "#D9DEE8"
GREY = "#6B7280"


def load_base_analysis():
    spec = importlib.util.spec_from_file_location("c6h5clo_base_analysis", BASE_ANALYSIS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_ANALYSIS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def peak_profile(
    y: np.ndarray,
    predicted_center: float,
    *,
    half_width: int = 16,
    side_inner: int = 19,
    side_outer: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    center = int(round(predicted_center))
    x = np.arange(center - half_width, center + half_width + 1)
    left_x = np.arange(center - side_outer, center - side_inner)
    right_x = np.arange(center + side_inner + 1, center + side_outer + 1)
    left_level = float(np.median(y[left_x]))
    right_level = float(np.median(y[right_x]))
    baseline = np.interp(
        x,
        [float(np.mean(left_x)), float(np.mean(right_x))],
        [left_level, right_level],
    )
    return x, y[x] - baseline


def fit_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    expected_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(y, 0.0)
    initial_center = float(x[int(np.argmax(positive))])
    initial_sigma = 3.0
    bounds = (
        [0.0, expected_center - 10.0, 0.5],
        [np.inf, expected_center + 10.0, 15.0],
    )
    return curve_fit(
        gaussian,
        x,
        y,
        p0=[float(np.max(positive)), initial_center, initial_sigma],
        bounds=bounds,
        maxfev=20_000,
    )


def locally_corrected_mass(calibration, observed_tof: float, tof_offset: float) -> float:
    return float(calibration.tof_to_mz(observed_tof - tof_offset))


def build_residual_profile(
    aggregate: np.ndarray,
    calibration,
    isotope_ratio: float,
) -> dict:
    predicted_140 = calibration.mz_to_tof(MASS_C7H5_35CLO)
    predicted_144 = calibration.mz_to_tof(143.99780708225)

    x140, y140 = peak_profile(aggregate, predicted_140)
    fit140, cov140 = fit_gaussian(x140, y140, predicted_140)
    x144, y144 = peak_profile(aggregate, predicted_144)
    fit144, cov144 = fit_gaussian(x144, y144, predicted_144)

    offset140 = float(fit140[1] - predicted_140)
    offset144 = float(fit144[1] - predicted_144)
    fraction = (
        MASS_C7H5_37CLO - MASS_C7H5_35CLO
    ) / (143.99780708225 - MASS_C7H5_35CLO)
    local_offset = offset140 + fraction * (offset144 - offset140)

    predicted_isotope = calibration.mz_to_tof(MASS_C7H5_37CLO) + local_offset
    predicted_candidate = calibration.mz_to_tof(MASS_C6H3_35CLO2) + local_offset
    profile_center = 0.5 * (predicted_isotope + predicted_candidate)
    x142, y142 = peak_profile(aggregate, profile_center)

    isotope_template = isotope_ratio * np.interp(
        x142 - (predicted_isotope - float(fit140[1])),
        x140,
        y140,
        left=0.0,
        right=0.0,
    )
    residual = y142 - isotope_template
    residual_fit, residual_cov = fit_gaussian(x142, residual, predicted_isotope)

    bin_mass_width = float(
        calibration.tof_to_mz(float(residual_fit[1]) + 0.5)
        - calibration.tof_to_mz(float(residual_fit[1]) - 0.5)
    )
    residual_fwhm_bins = float(2.354820045 * residual_fit[2])
    residual_fwhm_da = residual_fwhm_bins * bin_mass_width
    corrected_mass = locally_corrected_mass(
        calibration,
        float(residual_fit[1]),
        local_offset,
    )

    return {
        "x140": x140,
        "y140": y140,
        "fit140": fit140,
        "cov140": cov140,
        "x142": x142,
        "observed142": y142,
        "isotope_template": isotope_template,
        "residual142": residual,
        "residual_fit": residual_fit,
        "residual_cov": residual_cov,
        "local_offset": local_offset,
        "predicted_isotope_tof": predicted_isotope,
        "predicted_candidate_tof": predicted_candidate,
        "corrected_residual_mass": corrected_mass,
        "residual_fwhm_bins": residual_fwhm_bins,
        "residual_fwhm_da": residual_fwhm_da,
        "observed_area": float(np.trapz(y142, x142)),
        "isotope_area": float(np.trapz(isotope_template, x142)),
        "residual_area": float(np.trapz(residual, x142)),
    }


def bootstrap_centers(
    rates: np.ndarray,
    calibration,
    *,
    iterations: int = 500,
) -> np.ndarray:
    rng = np.random.default_rng(20260727)
    centers: list[float] = []
    for _ in range(iterations):
        sample = rates[rng.integers(0, len(rates), len(rates))].mean(axis=0)
        try:
            result = build_residual_profile(
                sample,
                calibration,
                CHLORINE_37_TO_35,
            )
        except (RuntimeError, ValueError):
            continue
        centers.append(float(result["corrected_residual_mass"]))
    return np.asarray(centers, dtype=float)


def build_pie_residual(spectra: list[dict], calibration, base) -> pd.DataFrame:
    high_energy = [
        item["y"] / item["io_nA"] / item["time_s"]
        for item in spectra
        if item["segment"] == "PIE" and item["energy"] >= 9.5
    ]
    aggregate = np.mean(high_energy, axis=0)

    centers: dict[int, int] = {}
    for channel, mass in (
        (140, MASS_C7H5_35CLO),
        (142, MASS_C7H5_37CLO),
        (144, 143.99780708225),
        (146, 146.00039392317),
    ):
        predicted = int(round(calibration.mz_to_tof(mass)))
        left = predicted - 8
        centers[channel] = left + int(np.argmax(aggregate[left : predicted + 9]))

    rows: list[dict] = []
    for spectrum in spectra:
        if spectrum["segment"] != "PIE":
            continue
        intensities: dict[int, float] = {}
        for channel, center in centers.items():
            area, _, snr = base.robust_baseline_area(
                spectrum["y"],
                center,
                half_width=10,
            )
            intensities[channel] = area / spectrum["io_nA"] / spectrum["time_s"]
            if channel == 142:
                snr142 = snr
        residual = (
            intensities[142] - CHLORINE_37_TO_35 * intensities[140]
        )
        expected_m_plus_2_at_144 = CHLORINE_37_TO_35 * residual
        i144_after_candidate_isotope = (
            intensities[144] - expected_m_plus_2_at_144
        )
        expected_c6h5_37clo2_at_146 = (
            CHLORINE_37_TO_35 * i144_after_candidate_isotope
        )
        i146_after_chain = (
            intensities[146] - expected_c6h5_37clo2_at_146
        )
        rows.append(
            {
                "energy_eV": float(spectrum["energy"]),
                "file": spectrum["file"],
                "i140": intensities[140],
                "i142_observed": intensities[142],
                "i142_c7h5_37clo": CHLORINE_37_TO_35 * intensities[140],
                "i142_residual": residual,
                "residual_fraction_of_142": (
                    residual / intensities[142]
                    if intensities[142] > 0
                    else np.nan
                ),
                "i144_observed": intensities[144],
                "expected_c6h3_37clo2_at_144": expected_m_plus_2_at_144,
                "expected_fraction_of_144": (
                    expected_m_plus_2_at_144 / intensities[144]
                    if intensities[144] > 0
                    else np.nan
                ),
                "i144_after_candidate_isotope": i144_after_candidate_isotope,
                "i146_observed": intensities[146],
                "expected_c6h5_37clo2_at_146": (
                    expected_c6h5_37clo2_at_146
                ),
                "i146_after_isotope_chain": i146_after_chain,
                "i146_after_chain_fraction": (
                    i146_after_chain / intensities[146]
                    if intensities[146] > 0
                    else np.nan
                ),
                "snr142": snr142,
            }
        )
    return pd.DataFrame(rows).sort_values("energy_eV").reset_index(drop=True)


def make_figure(profile: dict, pie: pd.DataFrame, calibration) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    x142 = profile["x142"]
    mass_axis = calibration.tof_to_mz(x142 - profile["local_offset"])
    axes[0].plot(
        mass_axis,
        profile["observed142"],
        color=BLUE,
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        label="Observed m/z 142 profile",
    )
    axes[0].plot(
        mass_axis,
        profile["isotope_template"],
        color=ORANGE,
        linewidth=2.0,
        linestyle="--",
        label=r"0.32 × shifted m/z 140 ($^{37}$Cl template)",
    )
    axes[0].plot(
        mass_axis,
        profile["residual142"],
        color=PINK,
        linewidth=2.2,
        label="Residual",
    )
    axes[0].axvline(
        MASS_C6H3_35CLO2,
        color=INK,
        linestyle=":",
        linewidth=1.5,
        label=r"$\mathrm{C_6H_3{}^{35}ClO_2}$ exact mass",
    )
    axes[0].axvline(
        MASS_C7H5_37CLO,
        color=GREY,
        linestyle="-.",
        linewidth=1.4,
        label=r"$\mathrm{C_7H_5{}^{37}ClO}$ exact mass",
    )
    axes[0].set_title(
        "m/z 142 isotope-template subtraction\n"
        "Exact-mass lines are relative references; systematic offsets are expected"
    )
    axes[0].set_xlabel("Locally corrected m/z")
    axes[0].set_ylabel("Counts / nA / s per TOF bin")
    axes[0].grid(True, color=GRID, linewidth=0.7, alpha=0.65)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")

    high = pie[pie["energy_eV"] >= 9.0]
    axes[1].plot(
        high["energy_eV"],
        high["i142_observed"],
        color=BLUE,
        linewidth=1.8,
        marker="o",
        markersize=3.3,
        label="Observed 142",
    )
    axes[1].plot(
        high["energy_eV"],
        high["i142_c7h5_37clo"],
        color=ORANGE,
        linewidth=1.7,
        linestyle="--",
        label=r"Calculated $\mathrm{C_7H_5{}^{37}ClO}$",
    )
    axes[1].plot(
        high["energy_eV"],
        high["i142_residual"],
        color=PINK,
        linewidth=2.0,
        marker="s",
        markersize=3.0,
        label="Residual 142",
    )
    axes[1].set_title("m/z 142 PIE before and after subtraction")
    axes[1].set_xlabel("Photon energy / eV")
    axes[1].set_ylabel("Counts / nA / s")
    axes[1].grid(True, color=GRID, linewidth=0.7, alpha=0.65)
    axes[1].legend(fontsize=8.5, frameon=False, loc="upper left")

    figure.suptitle(
        "Residual analysis for m/z 142\n"
        "Profile: mean of PIE scans at 9.5–11.0 eV; "
        "PIE panel: 9.0–11.0 eV",
        fontsize=13,
        color=INK,
    )
    figure.tight_layout()
    output = OUTPUT_DIR / "residual_142_diagnostic.png"
    figure.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    base = load_base_analysis()
    spectra = base.read_raw_spectra()
    calibration = base.load_calibration()
    high_spectra = [
        item
        for item in spectra
        if item["segment"] == "PIE" and item["energy"] >= 9.5
    ]
    rates = np.asarray(
        [item["y"] / item["io_nA"] / item["time_s"] for item in high_spectra]
    )
    aggregate = rates.mean(axis=0)

    profile = build_residual_profile(
        aggregate,
        calibration,
        CHLORINE_37_TO_35,
    )
    bootstrap_masses = bootstrap_centers(rates, calibration)
    pie = build_pie_residual(spectra, calibration, base)

    profile_table = pd.DataFrame(
        {
            "tof_bin": profile["x142"],
            "locally_corrected_mz": calibration.tof_to_mz(
                profile["x142"] - profile["local_offset"]
            ),
            "observed_142": profile["observed142"],
            "c7h5_37clo_template": profile["isotope_template"],
            "residual_142": profile["residual142"],
        }
    )
    profile_table.to_csv(OUTPUT_DIR / "residual_142_profile.csv", index=False)
    pie.to_csv(OUTPUT_DIR / "residual_142_pie.csv", index=False)

    high_pie = pie[pie["energy_eV"] >= 9.0]
    last = pie.iloc[-1]
    bootstrap_low, bootstrap_high = np.quantile(bootstrap_masses, [0.025, 0.975])
    residual_fraction = profile["residual_area"] / profile["observed_area"]
    mass_separation = MASS_C7H5_37CLO - MASS_C6H3_35CLO2
    required_resolution = MASS_C6H3_35CLO2 / mass_separation
    observed_resolution = (
        profile["corrected_residual_mass"] / profile["residual_fwhm_da"]
    )

    summary = {
        "model": {
            "subtraction": "I142_residual = I142 - 0.32 * shifted I140",
            "chlorine_37_to_35_ratio": CHLORINE_37_TO_35,
            "profile_energy_range_eV": [9.5, 11.0],
            "profile_scan_count": len(high_spectra),
        },
        "exact_masses": {
            "C7H5_35ClO": MASS_C7H5_35CLO,
            "C7H5_37ClO": MASS_C7H5_37CLO,
            "C6H3_35ClO2": MASS_C6H3_35CLO2,
            "C6H3_37ClO2": MASS_C6H3_37CLO2,
            "C7H7_35ClO_comparison": MASS_C7H7_35CLO,
        },
        "residual": {
            "observed_142_profile_area": profile["observed_area"],
            "subtracted_isotope_profile_area": profile["isotope_area"],
            "residual_profile_area": profile["residual_area"],
            "residual_fraction_of_142_profile": residual_fraction,
            "residual_fraction_9_11_median": float(
                high_pie["residual_fraction_of_142"].median()
            ),
            "residual_fraction_at_11eV": float(
                last["residual_fraction_of_142"]
            ),
            "corrected_residual_centroid_mz": profile[
                "corrected_residual_mass"
            ],
            "bootstrap_centroid_95pct": [
                float(bootstrap_low),
                float(bootstrap_high),
            ],
            "fwhm_bins": profile["residual_fwhm_bins"],
            "fwhm_da": profile["residual_fwhm_da"],
        },
        "formula_discrimination": {
            "C7H5_37ClO_minus_C6H3_35ClO2_da": mass_separation,
            "required_resolving_power": required_resolution,
            "observed_resolving_power_from_residual_fwhm": observed_resolution,
            "centroid_minus_C6H3_35ClO2_da": (
                profile["corrected_residual_mass"] - MASS_C6H3_35CLO2
            ),
            "centroid_minus_C7H5_37ClO_da": (
                profile["corrected_residual_mass"] - MASS_C7H5_37CLO
            ),
            "centroid_minus_C7H7_35ClO_da": (
                profile["corrected_residual_mass"] - MASS_C7H7_35CLO
            ),
            "centroid_is_not_an_assignment_test": True,
            "interpretation": (
                "Expected systematic mass offsets make the absolute centroid "
                "differences non-diagnostic for excluding C6H3ClO2."
            ),
        },
        "paired_isotope_check_at_144": {
            "expected_C6H3_37ClO2_fraction_of_144_9_11_median": float(
                high_pie["expected_fraction_of_144"].median()
            ),
            "expected_C6H3_37ClO2_fraction_of_144_at_11eV": float(
                last["expected_fraction_of_144"]
            ),
            "note": (
                "This component is only expected if the residual is "
                "C6H3ClO2; it is unresolved from C6H5-35ClO2 at m/z 144."
            ),
        },
        "isotope_chain_closure": {
            "model": (
                "140 -> 142 residual candidate -> 144 candidate-37Cl; "
                "remaining 144 C6H5-35ClO2 -> 146 C6H5-37ClO2"
            ),
            "i146_after_chain_fraction_at_11eV": float(
                last["i146_after_chain_fraction"]
            ),
            "i146_after_chain_positive_points_9_11": int(
                (high_pie["i146_after_isotope_chain"] > 0).sum()
            ),
            "n_points_9_11": int(len(high_pie)),
            "interpretation": (
                "A small 146 residual at 11 eV is compatible with an added "
                "C8H2O3 component, but negative low-energy residuals show that "
                "the chain model and response corrections are not yet exact."
            ),
        },
        "limitations": [
            (
                "Systematic mass offsets are expected, so absolute centroid "
                "differences are not used to accept or reject formulas."
            ),
            "The residual peak FWHM is much wider than the 17.8 mDa formula separation.",
            "The 140 peak is assumed to be pure C7H5-35ClO.",
            "The 0.32 isotope ratio is assumed to apply without mass-response bias.",
        ],
    }

    sensitivity: list[dict] = []
    for isotope_ratio in (0.30, 0.32, 0.34):
        result = build_residual_profile(aggregate, calibration, isotope_ratio)
        sensitivity.append(
            {
                "isotope_ratio": isotope_ratio,
                "residual_fraction": (
                    result["residual_area"] / result["observed_area"]
                ),
                "corrected_residual_centroid_mz": result[
                    "corrected_residual_mass"
                ],
                "fwhm_da": result["residual_fwhm_da"],
            }
        )
    pd.DataFrame(sensitivity).to_csv(
        OUTPUT_DIR / "residual_142_sensitivity.csv",
        index=False,
    )
    summary["sensitivity"] = sensitivity

    figure_path = make_figure(profile, pie, calibration)
    summary["figure"] = str(figure_path)
    (OUTPUT_DIR / "residual_142_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
