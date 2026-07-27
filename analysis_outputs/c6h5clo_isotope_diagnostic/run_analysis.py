from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
import yaml


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(
    "/Users/huangchen/Downloads/Project_C6H5ClO/raw_data/pie_scan/PIE-总"
)
PROJECT_CONFIG = Path(
    "/Users/huangchen/Downloads/Project_C6H5ClO/config/project.yaml"
)
PEAK_FILE = Path(
    "/Users/huangchen/Downloads/Project_C6H5ClO/analysis/spectrum/manual_peaks/"
    "peak_sets/peak-20260727T043641202752Z-b15932e2.csv"
)
CACHE_DIR = Path(
    "/Users/huangchen/Downloads/Project_C6H5ClO/analysis/pie/cache/"
    "1f20afdfad77d7b85c3c1986"
)

TARGET_MZ = {
    99: 99.00,
    101: 101.02,
    127: 127.05,
    128: 128.00,
    129: 129.00,
    130: 130.00,
    142: 142.06,
    144: 144.02,
    146: 146.02,
    148: 148.01,
}
KNOWN_CHLORINE_PAIRS = [(127, 129), (99, 101), (128, 130)]
TARGET_RATIO_PAIRS = [(146, 144), (148, 146)]
EXPECTED_BASE_TO_PLUS2 = 3.1
EXPECTED_PLUS2_TO_BASE = 0.32

BLUE = "#2563EB"
ORANGE = "#D97706"
PINK = "#DB2777"
OLIVE = "#6B7A16"
INK = "#172033"
GRID = "#D9DEE8"


@dataclass(frozen=True)
class Calibration:
    a: float
    b: float
    c: float

    def tof_to_mz(self, tof: float | np.ndarray) -> float | np.ndarray:
        tof_array = np.asarray(tof, dtype=float)
        result = self.c + self.b * tof_array + self.a * tof_array**2
        return float(result) if np.ndim(result) == 0 else result

    def mz_to_tof(self, mz: float) -> float:
        discriminant = self.b**2 - 4.0 * self.a * (self.c - mz)
        if discriminant < 0:
            raise ValueError(f"m/z {mz} is outside the calibration domain")
        return float((-self.b + np.sqrt(discriminant)) / (2.0 * self.a))


def _first_number(text: str) -> float:
    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        text,
    )
    if not match:
        raise ValueError(f"no numeric value in {text!r}")
    return float(match.group(0))


def _header_value(lines: list[str], prefix: str) -> float:
    prefix_lower = prefix.lower()
    for line in lines[:10]:
        if line.lower().startswith(prefix_lower):
            return _first_number(line.split(":", 1)[1])
    raise ValueError(f"missing {prefix!r} header")


def read_raw_spectra() -> list[dict]:
    spectra: list[dict] = []
    for segment in ("PIE-2", "PIE"):
        for path in sorted((DATA_ROOT / segment).glob("*.txt")):
            lines = path.read_text(encoding="utf-8").splitlines()
            y = np.asarray([float(value) for value in lines[10:]], dtype=float)
            spectra.append(
                {
                    "segment": segment,
                    "path": path,
                    "file": path.name,
                    "energy": _header_value(lines, "Energy:"),
                    "io_nA": _header_value(lines, "IO:"),
                    "beam_current_mA": _header_value(lines, "Beam Current:"),
                    "time_s": _header_value(lines, "Time:"),
                    "temperature_C": _header_value(lines, "Temperature:"),
                    "n_bins": int(y.size),
                    "y": y,
                }
            )
    if not spectra:
        raise FileNotFoundError(f"no PIE spectra found under {DATA_ROOT}")
    return spectra


def load_calibration() -> Calibration:
    payload = yaml.safe_load(PROJECT_CONFIG.read_text(encoding="utf-8"))
    coefficients = payload["calibration"]
    return Calibration(
        a=float(coefficients["a"]),
        b=float(coefficients["b"]),
        c=float(coefficients["c"]),
    )


def load_manual_target_rows() -> pd.DataFrame:
    peaks = pd.read_csv(PEAK_FILE, encoding="utf-8-sig")
    mz_column = "质量数 (m/z)"
    center_column = "飞行时间"
    left_column = "左边界"
    right_column = "右边界"
    rows: list[dict] = []
    for nominal, exact_mz in TARGET_MZ.items():
        index = (pd.to_numeric(peaks[mz_column]) - exact_mz).abs().idxmin()
        row = peaks.loc[index]
        rows.append(
            {
                "channel": nominal,
                "label_mz": float(row[mz_column]),
                "old_center_tof": int(round(float(row[center_column]))),
                "old_left_tof": int(round(float(row[left_column]))),
                "old_right_tof": int(round(float(row[right_column]))),
            }
        )
    return pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)


def robust_baseline_area(
    y: np.ndarray,
    center: int,
    *,
    half_width: int = 10,
    side_inner: int = 18,
    side_outer: int = 30,
) -> tuple[float, float, float]:
    peak_x = np.arange(center - half_width, center + half_width + 1)
    peak_y = y[peak_x]
    left_x = np.arange(center - side_outer, center - side_inner)
    right_x = np.arange(center + side_inner + 1, center + side_outer + 1)
    left_level = float(np.median(y[left_x]))
    right_level = float(np.median(y[right_x]))
    side_mid_left = float(np.mean(left_x))
    side_mid_right = float(np.mean(right_x))
    fraction = (peak_x - side_mid_left) / (side_mid_right - side_mid_left)
    baseline = left_level + fraction * (right_level - left_level)
    area = float(np.sum(peak_y - baseline))
    baseline_sum = float(np.sum(np.maximum(baseline, 0.0)))
    variance = max(float(np.sum(np.maximum(peak_y, 0.0))) + baseline_sum, 1.0)
    snr = area / np.sqrt(variance)
    return area, float(np.mean(baseline)), float(snr)


def build_alignment(
    spectra: list[dict],
    calibration: Calibration,
    manual_rows: pd.DataFrame,
) -> pd.DataFrame:
    high_energy = [
        item["y"] / item["io_nA"] / item["time_s"]
        for item in spectra
        if item["segment"] == "PIE" and item["energy"] >= 9.5
    ]
    aggregate = np.mean(high_energy, axis=0)
    rows: list[dict] = []
    for item in manual_rows.to_dict("records"):
        expected_center = calibration.mz_to_tof(float(item["label_mz"]))
        expected_rounded = int(round(expected_center))
        search_left = expected_rounded - 7
        search_right = expected_rounded + 8
        observed_center = search_left + int(
            np.argmax(aggregate[search_left:search_right])
        )
        shift = observed_center - int(item["old_center_tof"])
        observed_mz = float(calibration.tof_to_mz(observed_center))
        rows.append(
            {
                **item,
                "calibrated_center_tof": expected_center,
                "observed_center_tof": observed_center,
                "observed_mz": observed_mz,
                "tof_offset_bins": shift,
                "shifted_left_tof": int(item["old_left_tof"]) + shift,
                "shifted_right_tof": int(item["old_right_tof"]) + shift,
                "old_window_contains_peak": bool(
                    int(item["old_left_tof"])
                    <= observed_center
                    <= int(item["old_right_tof"])
                ),
                "aggregate_peak_rate": float(aggregate[observed_center]),
            }
        )
    return pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)


def integrate_all_spectra(
    spectra: list[dict],
    alignment: pd.DataFrame,
    *,
    half_width: int = 10,
) -> pd.DataFrame:
    alignment_by_channel = alignment.set_index("channel").to_dict("index")
    rows: list[dict] = []
    for spectrum in spectra:
        y = spectrum["y"]
        for channel, target in alignment_by_channel.items():
            old_left = int(target["old_left_tof"])
            old_right = int(target["old_right_tof"])
            shifted_left = int(target["shifted_left_tof"])
            shifted_right = int(target["shifted_right_tof"])
            center = int(target["observed_center_tof"])
            corrected_area, baseline, snr = robust_baseline_area(
                y,
                center,
                half_width=half_width,
            )
            old_sum = float(np.sum(y[old_left : old_right + 1]))
            shifted_sum = float(np.sum(y[shifted_left : shifted_right + 1]))
            io_value = float(spectrum["io_nA"])
            time_value = float(spectrum["time_s"])
            rows.append(
                {
                    "segment": spectrum["segment"],
                    "file": spectrum["file"],
                    "energy_eV": float(spectrum["energy"]),
                    "energy_round_2": round(float(spectrum["energy"]), 2),
                    "io_nA": io_value,
                    "time_s": time_value,
                    "channel": int(channel),
                    "label_mz": float(target["label_mz"]),
                    "observed_mz": float(target["observed_mz"]),
                    "old_sum_counts": old_sum,
                    "shifted_sum_counts": shifted_sum,
                    "corrected_area_counts": corrected_area,
                    "local_baseline_counts_per_bin": baseline,
                    "area_snr": snr,
                    "ui_old_intensity_counts_per_nA": old_sum / io_value,
                    "corrected_intensity_counts_per_nA": corrected_area / io_value,
                    "corrected_rate_counts_per_nA_s": (
                        corrected_area / io_value / time_value
                    ),
                    "half_width_bins": half_width,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["energy_eV", "segment", "channel"],
        kind="stable",
    )


def build_scan_metadata(spectra: list[dict]) -> pd.DataFrame:
    rows = [
        {
            key: value
            for key, value in item.items()
            if key not in {"y", "path"}
        }
        for item in spectra
    ]
    return pd.DataFrame(rows).sort_values(
        ["energy", "segment", "file"],
        kind="stable",
    )


def verify_cached_curves() -> tuple[pd.DataFrame, dict]:
    results = pd.read_csv(CACHE_DIR / "results.csv")
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    target_rows = results[
        results["mz"].round(2).isin([round(value, 2) for value in TARGET_MZ.values()])
    ].copy()
    max_rows = (
        target_rows.groupby(["mz_rounded", "mz"], as_index=False)
        .agg(
            max_plotted_intensity=("normalized_intensity", "max"),
            max_raw_sum_counts=("raw_area", "max"),
            n_points=("energy", "size"),
            min_energy_eV=("energy", "min"),
            max_energy_eV=("energy", "max"),
        )
        .sort_values("mz_rounded")
    )
    grouped = target_rows.groupby("energy", as_index=False).agg(
        io_variants=("io", "nunique"),
        file_count_variants=("file_count", "nunique"),
        light_source_variants=("light_source", "nunique"),
    )
    reproducibility_error = float(
        np.max(
            np.abs(
                target_rows["normalized_intensity"].to_numpy()
                - (
                    target_rows["raw_area"].to_numpy()
                    / target_rows["io"].to_numpy()
                )
            )
        )
    )
    checks = {
        "cache_id": CACHE_DIR.name,
        "target_row_count": int(len(target_rows)),
        "same_io_for_all_targets_at_each_spectrum": bool(
            (grouped["io_variants"] == 1).all()
        ),
        "same_file_count_for_all_targets_at_each_spectrum": bool(
            (grouped["file_count_variants"] == 1).all()
        ),
        "same_light_source_for_all_targets_at_each_spectrum": bool(
            (grouped["light_source_variants"] == 1).all()
        ),
        "max_abs_error_normalized_equals_raw_over_io": reproducibility_error,
        "analysis_parameters": manifest["analysis_provenance"][
            "analysis_parameters"
        ],
    }
    return max_rows, checks


def ratio_series(
    intensities: pd.DataFrame,
    numerator: int,
    denominator: int,
    *,
    value_column: str,
) -> pd.DataFrame:
    selected = intensities[
        intensities["channel"].isin([numerator, denominator])
    ].pivot(
        index=[
            "segment",
            "file",
            "energy_eV",
            "energy_round_2",
            "io_nA",
            "time_s",
        ],
        columns="channel",
        values=value_column,
    )
    selected = selected.reset_index()
    selected["numerator_channel"] = numerator
    selected["denominator_channel"] = denominator
    selected["ratio"] = selected[numerator] / selected[denominator]
    selected.loc[
        ~np.isfinite(selected["ratio"]) | (selected[denominator] <= 0),
        "ratio",
    ] = np.nan
    return selected


def _nearest_energy_row(
    frame: pd.DataFrame,
    energy: float,
    *,
    segment: str = "PIE",
) -> pd.Series:
    subset = frame[frame["segment"] == segment]
    return subset.iloc[(subset["energy_eV"] - energy).abs().argmin()]


def build_ratio_summaries(
    intensities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_rows: list[dict] = []
    ratio_rows: list[dict] = []
    all_pairs = [
        *[(a, b, EXPECTED_BASE_TO_PLUS2, "known chlorine pair") for a, b in KNOWN_CHLORINE_PAIRS],
        (146, 144, EXPECTED_PLUS2_TO_BASE, "I146/I144 test"),
        (148, 146, EXPECTED_PLUS2_TO_BASE, "I148/I146 test"),
    ]
    for numerator, denominator, expected, purpose in all_pairs:
        corrected = ratio_series(
            intensities,
            numerator,
            denominator,
            value_column="corrected_area_counts",
        )
        old = ratio_series(
            intensities,
            numerator,
            denominator,
            value_column="old_sum_counts",
        )
        corrected = corrected.rename(columns={"ratio": "corrected_ratio"})
        old = old[
            ["segment", "file", "energy_eV", "ratio"]
        ].rename(columns={"ratio": "old_window_ratio"})
        merged = corrected.merge(
            old,
            on=["segment", "file", "energy_eV"],
            how="left",
        )
        merged["pair"] = f"{numerator}/{denominator}"
        merged["expected_ratio"] = expected
        merged["purpose"] = purpose
        ratio_rows.extend(merged.to_dict("records"))

        fixed_corrected = _nearest_energy_row(corrected, 11.0)
        fixed_old = _nearest_energy_row(old, 11.0)
        high = corrected[
            (corrected["segment"] == "PIE")
            & (corrected["energy_eV"] >= 9.0)
            & np.isfinite(corrected["corrected_ratio"])
        ]
        fixed_rows.append(
            {
                "pair": f"{numerator}/{denominator}",
                "purpose": purpose,
                "expected_ratio": expected,
                "energy_eV": float(fixed_corrected["energy_eV"]),
                "old_window_ratio": float(fixed_old["old_window_ratio"]),
                "corrected_ratio": float(fixed_corrected["corrected_ratio"]),
                "corrected_ratio_minus_expected": float(
                    fixed_corrected["corrected_ratio"] - expected
                ),
                "high_energy_n": int(len(high)),
                "high_energy_median": float(high["corrected_ratio"].median()),
                "high_energy_q25": float(high["corrected_ratio"].quantile(0.25)),
                "high_energy_q75": float(high["corrected_ratio"].quantile(0.75)),
            }
        )
    ratio_frame = pd.DataFrame(ratio_rows)
    fixed_frame = pd.DataFrame(fixed_rows)

    sensitivity_rows: list[dict] = []
    for half_width in (6, 8, 10, 12):
        recalculated = integrate_all_spectra(
            _SPECTRA_CACHE,
            _ALIGNMENT_CACHE,
            half_width=half_width,
        )
        for numerator, denominator, expected, purpose in all_pairs:
            values = ratio_series(
                recalculated,
                numerator,
                denominator,
                value_column="corrected_area_counts",
            )
            fixed = _nearest_energy_row(values, 11.0)
            sensitivity_rows.append(
                {
                    "half_width_bins": half_width,
                    "pair": f"{numerator}/{denominator}",
                    "purpose": purpose,
                    "expected_ratio": expected,
                    "ratio_at_11eV": float(fixed["ratio"]),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    return ratio_frame, fixed_frame, sensitivity


def build_overlap_summary(intensities: pd.DataFrame) -> pd.DataFrame:
    selected_channels = [128, 130, 142, 144, 146]
    overlap = intensities[
        intensities["channel"].isin(selected_channels)
        & intensities["energy_round_2"].isin([8.00, 8.05, 8.10])
    ]
    rows: list[dict] = []
    for channel in selected_channels:
        for energy in (8.00, 8.05, 8.10):
            values = overlap[
                (overlap["channel"] == channel)
                & (overlap["energy_round_2"] == energy)
            ].set_index("segment")
            if not {"PIE-2", "PIE"}.issubset(values.index):
                continue
            low = values.loc["PIE-2"]
            high = values.loc["PIE"]
            io_only_ratio = (
                low["corrected_intensity_counts_per_nA"]
                / high["corrected_intensity_counts_per_nA"]
            )
            io_time_ratio = (
                low["corrected_rate_counts_per_nA_s"]
                / high["corrected_rate_counts_per_nA_s"]
            )
            rows.append(
                {
                    "energy_eV": energy,
                    "channel": channel,
                    "PIE2_over_PIE_after_IO_only": float(io_only_ratio),
                    "PIE2_over_PIE_after_IO_and_time": float(io_time_ratio),
                }
            )
    return pd.DataFrame(rows)


def build_shape_tests(
    intensities: pd.DataFrame,
    ratio_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for numerator, denominator, expected in (
        (146, 144, EXPECTED_PLUS2_TO_BASE),
        (148, 146, EXPECTED_PLUS2_TO_BASE),
    ):
        ratio = ratio_frame[
            (ratio_frame["pair"] == f"{numerator}/{denominator}")
            & (ratio_frame["segment"] == "PIE")
            & (ratio_frame["energy_eV"] >= 9.0)
        ].dropna(subset=["corrected_ratio"])
        regression = linregress(ratio["energy_eV"], ratio["corrected_ratio"])
        pivot = (
            intensities[
                intensities["channel"].isin([numerator, denominator])
                & (intensities["segment"] == "PIE")
                & (intensities["energy_eV"] >= 9.0)
            ]
            .pivot(index="energy_eV", columns="channel", values="corrected_rate_counts_per_nA_s")
            .dropna()
        )
        valid = pivot[(pivot[numerator] > 0) & (pivot[denominator] > 0)]
        correlation, correlation_p = pearsonr(
            valid[denominator],
            valid[numerator],
        )
        slope_zero = float(
            np.dot(valid[denominator], valid[numerator])
            / np.dot(valid[denominator], valid[denominator])
        )
        residual = valid[numerator] - slope_zero * valid[denominator]
        ss_res = float(np.sum(residual**2))
        ss_tot_zero = float(np.sum(valid[numerator] ** 2))
        r2_zero = 1.0 - ss_res / ss_tot_zero if ss_tot_zero else np.nan
        rows.append(
            {
                "pair": f"{numerator}/{denominator}",
                "expected_ratio": expected,
                "n_energy_points": int(len(valid)),
                "pearson_r_of_PIE_intensities": float(correlation),
                "pearson_p": float(correlation_p),
                "best_scale_through_origin": slope_zero,
                "r2_through_origin": r2_zero,
                "ratio_vs_energy_slope_per_eV": float(regression.slope),
                "ratio_vs_energy_p": float(regression.pvalue),
                "ratio_median": float(ratio["corrected_ratio"].median()),
                "ratio_q25": float(ratio["corrected_ratio"].quantile(0.25)),
                "ratio_q75": float(ratio["corrected_ratio"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors=INK, labelsize=9)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)


def plot_scan_conditions(scan_metadata: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    colors = {"PIE-2": ORANGE, "PIE": BLUE}
    markers = {"PIE-2": "s", "PIE": "o"}
    for segment in ("PIE-2", "PIE"):
        selected = scan_metadata[scan_metadata["segment"] == segment]
        axes[0].plot(
            selected["energy"],
            selected["io_nA"],
            color=colors[segment],
            marker=markers[segment],
            markersize=3.5,
            linewidth=1.3,
            label=f"{segment} ({int(selected['time_s'].iloc[0])} s/point)",
        )
        axes[1].plot(
            selected["energy"],
            selected["time_s"],
            color=colors[segment],
            marker=markers[segment],
            markersize=3.5,
            linewidth=1.3,
        )
    axes[0].set_title("Photon-flux monitor and acquisition time by energy")
    axes[0].set_ylabel("IO / nA")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_ylabel("Acquisition time / s")
    axes[1].set_xlabel("Photon energy / eV")
    for axis in axes:
        _style_axis(axis)
    figure.suptitle(
        "All mass channels share each spectrum's metadata; scan protocol changes at the segment boundary",
        fontsize=10,
        color=INK,
        y=0.99,
    )
    figure.tight_layout()
    path = OUTPUT_DIR / "scan_conditions.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_fixed_energy_spectrum(
    spectra: list[dict],
    alignment: pd.DataFrame,
    calibration: Calibration,
) -> Path:
    spectrum = min(
        (item for item in spectra if item["segment"] == "PIE"),
        key=lambda item: abs(item["energy"] - 11.0),
    )
    y = spectrum["y"]
    tof = np.arange(y.size, dtype=float)
    mz = np.asarray(calibration.tof_to_mz(tof), dtype=float)
    panels = [
        ((98.5, 101.5), [99, 101]),
        ((126.5, 130.5), [127, 128, 129, 130]),
        ((141.5, 148.5), [142, 144, 146, 148]),
    ]
    alignment_by_channel = alignment.set_index("channel")
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 10.5))
    for axis, (xlim, channels) in zip(axes, panels):
        mask = (mz >= xlim[0]) & (mz <= xlim[1])
        axis.plot(
            mz[mask],
            np.maximum(y[mask], 0.5),
            color=INK,
            linewidth=1.0,
            label="raw spectrum",
        )
        for channel in channels:
            target = alignment_by_channel.loc[channel]
            old_left_mz = calibration.tof_to_mz(target["old_left_tof"])
            old_right_mz = calibration.tof_to_mz(target["old_right_tof"])
            corrected_left_mz = calibration.tof_to_mz(
                target["observed_center_tof"] - 10
            )
            corrected_right_mz = calibration.tof_to_mz(
                target["observed_center_tof"] + 10
            )
            axis.axvspan(
                old_left_mz,
                old_right_mz,
                color=ORANGE,
                alpha=0.13,
                linewidth=0,
            )
            axis.axvspan(
                corrected_left_mz,
                corrected_right_mz,
                color=BLUE,
                alpha=0.12,
                linewidth=0,
            )
            axis.axvline(
                target["observed_mz"],
                color=BLUE,
                linestyle=(0, (3, 3)),
                linewidth=0.9,
            )
            peak_height = max(
                float(y[int(target["observed_center_tof"])]),
                1.0,
            )
            axis.annotate(
                f"{channel}",
                xy=(target["observed_mz"], peak_height),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=BLUE,
                fontsize=8,
            )
        axis.set_xlim(*xlim)
        axis.set_yscale("log")
        axis.set_ylabel("Raw counts (log; zeros at 0.5)")
        _style_axis(axis)
    axes[-1].set_xlabel("m/z using the active project calibration")
    axes[0].set_title(
        f"Raw high-resolution spectrum at {spectrum['energy']:.4f} eV "
        f"({spectrum['time_s']:.0f} s, IO={spectrum['io_nA']:.2f} nA)"
    )
    axes[0].plot([], [], color=ORANGE, linewidth=8, alpha=0.25, label="stale window")
    axes[0].plot([], [], color=BLUE, linewidth=8, alpha=0.22, label="recentered ±10 bins")
    axes[0].legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    figure.tight_layout()
    path = OUTPUT_DIR / "fixed_energy_high_resolution_spectrum.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_known_isotope_ratios(ratio_frame: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)
    pair_colors = {
        "127/129": BLUE,
        "99/101": ORANGE,
        "128/130": PINK,
    }
    for axis, pair in zip(axes, pair_colors):
        selected = ratio_frame[
            (ratio_frame["pair"] == pair)
            & (ratio_frame["segment"] == "PIE")
            & (ratio_frame["energy_eV"] >= 9.0)
        ]
        axis.plot(
            selected["energy_eV"],
            selected["corrected_ratio"],
            color=pair_colors[pair],
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            label="recentered raw-spectrum integration",
        )
        axis.axhline(
            EXPECTED_BASE_TO_PLUS2,
            color=INK,
            linewidth=1.0,
            linestyle=(0, (5, 4)),
            label="expected 3.1",
        )
        axis.set_ylabel(f"I{pair.replace('/', ' / I')}")
        axis.set_title(f"Known chlorine-isotope diagnostic: {pair}")
        axis.set_ylim(bottom=0)
        _style_axis(axis)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].set_xlabel("Photon energy / eV")
    figure.suptitle(
        "High-signal region (≥9.0 eV); a valid singly chlorinated pair should stay near 3.1",
        fontsize=10,
        color=INK,
        y=0.995,
    )
    figure.tight_layout()
    path = OUTPUT_DIR / "known_isotope_ratios.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_target_relationships(
    intensities: pd.DataFrame,
    ratio_frame: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0))
    colors = {144: ORANGE, 146: BLUE, 148: PINK}
    high_segment = intensities[
        (intensities["segment"] == "PIE")
        & intensities["channel"].isin(colors)
    ]
    for channel, color in colors.items():
        selected = high_segment[high_segment["channel"] == channel]
        axes[0].plot(
            selected["energy_eV"],
            selected["corrected_rate_counts_per_nA_s"],
            color=color,
            marker="o",
            markersize=3.4,
            linewidth=1.2,
            label=f"m/z {channel}",
        )
    axes[0].set_title("Recentered PIE intensities on one common normalization")
    axes[0].set_ylabel("Baseline-corrected counts / (nA·s)")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    for pair, color in (("146/144", BLUE), ("148/146", PINK)):
        selected = ratio_frame[
            (ratio_frame["pair"] == pair)
            & (ratio_frame["segment"] == "PIE")
            & (ratio_frame["energy_eV"] >= 9.0)
        ]
        axes[1].plot(
            selected["energy_eV"],
            selected["corrected_ratio"],
            color=color,
            marker="o",
            markersize=3.4,
            linewidth=1.2,
            label=pair,
        )
    axes[1].axhline(
        EXPECTED_PLUS2_TO_BASE,
        color=INK,
        linewidth=1.0,
        linestyle=(0, (5, 4)),
        label="expected 0.32",
    )
    axes[1].set_title("Energy dependence of the chlorine-isotope tests")
    axes[1].set_ylabel("Intensity ratio")
    axes[1].set_xlabel("Photon energy / eV")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlim(9.0, 11.02)
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout()
    path = OUTPUT_DIR / "target_relationships.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def build_summary(
    scan_metadata: pd.DataFrame,
    alignment: pd.DataFrame,
    cached_checks: dict,
    fixed_ratios: pd.DataFrame,
    overlap: pd.DataFrame,
    shape_tests: pd.DataFrame,
) -> dict:
    scan_summary = (
        scan_metadata.groupby("segment")
        .agg(
            files=("file", "size"),
            min_energy_eV=("energy", "min"),
            max_energy_eV=("energy", "max"),
            acquisition_time_s=("time_s", "first"),
            acquisition_time_variants=("time_s", "nunique"),
            min_io_nA=("io_nA", "min"),
            max_io_nA=("io_nA", "max"),
            min_bins=("n_bins", "min"),
            max_bins=("n_bins", "max"),
        )
        .reset_index()
    )
    overlap_io_only = overlap["PIE2_over_PIE_after_IO_only"].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    overlap_io_time = overlap["PIE2_over_PIE_after_IO_and_time"].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    return {
        "sources": {
            "raw_data_root": str(DATA_ROOT),
            "project_config": str(PROJECT_CONFIG),
            "manual_peak_file": str(PEAK_FILE),
            "pie_cache": str(CACHE_DIR),
        },
        "scan_summary": scan_summary.to_dict("records"),
        "cached_curve_checks": cached_checks,
        "window_alignment": {
            "all_target_peaks_inside_old_windows": bool(
                alignment["old_window_contains_peak"].all()
            ),
            "target_peaks_missed_by_old_windows": int(
                (~alignment["old_window_contains_peak"]).sum()
            ),
            "target_count": int(len(alignment)),
            "median_tof_offset_bins": float(
                alignment["tof_offset_bins"].median()
            ),
            "min_tof_offset_bins": int(alignment["tof_offset_bins"].min()),
            "max_tof_offset_bins": int(alignment["tof_offset_bins"].max()),
        },
        "fixed_energy_ratios": fixed_ratios.to_dict("records"),
        "shape_tests": shape_tests.to_dict("records"),
        "segment_overlap": {
            "n_comparisons": int(len(overlap)),
            "median_PIE2_over_PIE_after_IO_only": float(
                overlap_io_only.median()
            ),
            "median_PIE2_over_PIE_after_IO_and_time": float(
                overlap_io_time.median()
            ),
        },
    }


_SPECTRA_CACHE: list[dict] = []
_ALIGNMENT_CACHE = pd.DataFrame()


def main() -> dict:
    global _SPECTRA_CACHE, _ALIGNMENT_CACHE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spectra = read_raw_spectra()
    calibration = load_calibration()
    manual_rows = load_manual_target_rows()
    alignment = build_alignment(spectra, calibration, manual_rows)
    _SPECTRA_CACHE = spectra
    _ALIGNMENT_CACHE = alignment
    intensities = integrate_all_spectra(spectra, alignment, half_width=10)
    scan_metadata = build_scan_metadata(spectra)
    cached_maxima, cached_checks = verify_cached_curves()
    ratio_frame, fixed_ratios, sensitivity = build_ratio_summaries(intensities)
    overlap = build_overlap_summary(intensities)
    shape_tests = build_shape_tests(intensities, ratio_frame)

    scan_metadata.to_csv(OUTPUT_DIR / "scan_metadata.csv", index=False)
    alignment.to_csv(OUTPUT_DIR / "window_alignment.csv", index=False)
    intensities.to_csv(OUTPUT_DIR / "intensities_corrected.csv", index=False)
    cached_maxima.to_csv(OUTPUT_DIR / "cached_curve_maxima.csv", index=False)
    ratio_frame.to_csv(OUTPUT_DIR / "ratio_by_energy.csv", index=False)
    fixed_ratios.to_csv(OUTPUT_DIR / "fixed_energy_ratio_summary.csv", index=False)
    sensitivity.to_csv(OUTPUT_DIR / "integration_window_sensitivity.csv", index=False)
    overlap.to_csv(OUTPUT_DIR / "segment_overlap_summary.csv", index=False)
    shape_tests.to_csv(OUTPUT_DIR / "shape_tests.csv", index=False)

    figures = {
        "scan_conditions": str(plot_scan_conditions(scan_metadata)),
        "fixed_energy_spectrum": str(
            plot_fixed_energy_spectrum(spectra, alignment, calibration)
        ),
        "known_isotope_ratios": str(plot_known_isotope_ratios(ratio_frame)),
        "target_relationships": str(
            plot_target_relationships(intensities, ratio_frame)
        ),
    }
    summary = build_summary(
        scan_metadata,
        alignment,
        cached_checks,
        fixed_ratios,
        overlap,
        shape_tests,
    )
    summary["figures"] = figures
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, ensure_ascii=False, indent=2))
