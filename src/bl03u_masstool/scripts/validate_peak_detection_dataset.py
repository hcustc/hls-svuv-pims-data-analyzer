"""Build a cumulative BL03U reference spectrum and validate peak detectors.

The annotation table is treated as the union of peaks visible across all
spectra. Matching uses annotated TOF intervals, not m/z, so calibration drift
does not get counted as a peak-detection error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bl03u_masstool.core.config import load_calibration_config, load_peak_detection_config
from bl03u_masstool.core.peak_detection import Peak, detect_peaks_by_algorithm
from bl03u_masstool.core.spectrum_io import list_spectrum_files, read_spectrum
from bl03u_masstool.core.temperature_scan import extract_photon_energy, extract_temperature


ALGORITHMS = ("legacy", "prominence", "cwt", "ensemble")
ANNOTATION_ALIASES = {
    "mz": ("mz", "m/z", "质量数 (m/z)", "质量数"),
    "tof": ("tof", "peak_index", "飞行时间"),
    "left_bound": ("left_bound", "左边界"),
    "right_bound": ("right_bound", "右边界"),
    "reference_intensity": ("reference_intensity", "intensity", "强度"),
}


def _resolve_column(frame: pd.DataFrame, canonical: str, required: bool = True) -> str | None:
    for candidate in ANNOTATION_ALIASES[canonical]:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(f"annotation table has no {canonical!r} column; columns={list(frame.columns)!r}")
    return None


def load_annotations(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    result = pd.DataFrame(
        {
            "mz": pd.to_numeric(source[_resolve_column(source, "mz")], errors="raise"),
            "tof": pd.to_numeric(source[_resolve_column(source, "tof")], errors="raise"),
            "left_bound": pd.to_numeric(source[_resolve_column(source, "left_bound")], errors="raise"),
            "right_bound": pd.to_numeric(source[_resolve_column(source, "right_bound")], errors="raise"),
        }
    )
    intensity_column = _resolve_column(source, "reference_intensity", required=False)
    if intensity_column is not None:
        result["reference_intensity"] = pd.to_numeric(source[intensity_column], errors="coerce")
    if result[["mz", "tof", "left_bound", "right_bound"]].isna().any().any():
        raise ValueError("annotation table contains missing core values")
    return result.sort_values("tof", kind="stable").reset_index(drop=True)


def load_spectra(folder: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    paths = list_spectrum_files(folder, suffixes=(".txt", ".asc", ".888"))
    if not paths:
        raise ValueError(f"no spectrum files found in {folder}")
    records: list[dict[str, Any]] = []
    lengths: set[int] = set()
    for index, path in enumerate(paths):
        spectrum = read_spectrum(path, header_lines=10 if path.suffix.lower() == ".txt" else None, trim_start=0)
        if not len(spectrum.y):
            raise ValueError(f"empty spectrum: {path}")
        lengths.add(len(spectrum.y))
        records.append(
            {
                "file": path.name,
                "path": str(path),
                "temperature": extract_temperature(spectrum.metadata_lines, float(index)),
                "energy": extract_photon_energy(spectrum.metadata_lines, None),
                "y": np.asarray(spectrum.y, dtype=float),
            }
        )
    if len(lengths) != 1:
        raise ValueError(f"spectra have inconsistent point counts: {sorted(lengths)}")
    matrix = np.vstack([record["y"] for record in records])
    return records, matrix


def normalized_truth_intervals(annotations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    left = annotations["left_bound"].to_numpy(dtype=float)
    right = annotations["right_bound"].to_numpy(dtype=float)
    center = annotations["tof"].to_numpy(dtype=float)
    normalized_left = np.minimum(left, np.floor(center))
    normalized_right = np.maximum(right, np.ceil(center))
    changed = int(np.count_nonzero((normalized_left != left) | (normalized_right != right)))
    return normalized_left, normalized_right, changed


def match_peaks(
    peaks: Iterable[Peak],
    annotations: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    detected = list(peaks)
    left, right, _ = normalized_truth_intervals(annotations)
    centers = annotations["tof"].to_numpy(dtype=float)
    candidates: list[tuple[float, int, int]] = []
    for detected_index, peak in enumerate(detected):
        possible = np.flatnonzero((peak.time >= left) & (peak.time <= right))
        for truth_index in possible:
            candidates.append((abs(float(peak.time) - float(centers[truth_index])), int(truth_index), detected_index))
    candidates.sort()
    used_truth: set[int] = set()
    used_detected: set[int] = set()
    matches: list[dict[str, Any]] = []
    for error, truth_index, detected_index in candidates:
        if truth_index in used_truth or detected_index in used_detected:
            continue
        used_truth.add(truth_index)
        used_detected.add(detected_index)
        peak = detected[detected_index]
        matches.append(
            {
                "truth_index": truth_index,
                "detected_index": detected_index,
                "center_error_samples": float(error),
                "detected_time": float(peak.time),
                "detected_mz": float(peak.mz),
                "detected_intensity": float(peak.intensity),
                "detected_left_bound": float(peak.left_bound + 1.0),
                "detected_right_bound": float(peak.right_bound + 1.0),
            }
        )
    missed = sorted(set(range(len(annotations))) - used_truth)
    extras = sorted(set(range(len(detected))) - used_detected)
    return matches, missed, extras


def evaluate(peaks: list[Peak], annotations: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], list[int], list[int]]:
    matches, missed, extras = match_peaks(peaks, annotations)
    truth_count = len(annotations)
    detected_count = len(peaks)
    true_positive = len(matches)
    recall = true_positive / truth_count if truth_count else 0.0
    precision = true_positive / detected_count if detected_count else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    errors = np.asarray([item["center_error_samples"] for item in matches], dtype=float)
    summary = {
        "truth_count": truth_count,
        "detected_count": detected_count,
        "true_positive": true_positive,
        "missed_count": len(missed),
        "extra_count": len(extras),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "median_center_error_samples": float(np.median(errors)) if errors.size else np.nan,
        "p95_center_error_samples": float(np.percentile(errors, 95)) if errors.size else np.nan,
    }
    return summary, matches, missed, extras


def detect(profile: np.ndarray, algorithm: str, calibration: Any, peak_kwargs: dict[str, Any]) -> tuple[list[Peak], float]:
    kwargs = dict(peak_kwargs)
    kwargs["algorithm"] = algorithm
    started = time.perf_counter()
    peaks = detect_peaks_by_algorithm(profile, calibration=calibration, time_offset=1.0, **kwargs)
    return peaks, time.perf_counter() - started


def cluster_extra_detections(rows: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: item["detected_time"])
    clusters: list[list[dict[str, Any]]] = []
    for row in ordered:
        if not clusters or row["detected_time"] - clusters[-1][0]["detected_time"] > tolerance:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    result: list[dict[str, Any]] = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        temperatures = sorted({float(item["temperature"]) for item in cluster})
        result.append(
            {
                "cluster_id": cluster_id,
                "center_tof": float(np.median([item["detected_time"] for item in cluster])),
                "detection_count": len(cluster),
                "spectrum_count": len({item["file"] for item in cluster}),
                "temperatures": ";".join(f"{value:g}" for value in temperatures),
            }
        )
    return result


def calibration_diagnostics(annotations: pd.DataFrame, calibration: Any) -> dict[str, Any]:
    tof = annotations["tof"].to_numpy(dtype=float)
    mz = annotations["mz"].to_numpy(dtype=float)
    configured = np.asarray(calibration.tof_to_mz(tof), dtype=float)
    fitted = np.polyfit(tof, mz, deg=2)
    predicted = np.polyval(fitted, tof)
    residual = predicted - mz
    configured_residual = configured - mz
    return {
        "configured": {
            "a": float(calibration.a),
            "b": float(calibration.b),
            "c": float(calibration.c),
            "mae_mz": float(np.mean(np.abs(configured_residual))),
            "max_abs_error_mz": float(np.max(np.abs(configured_residual))),
            "median_signed_error_mz": float(np.median(configured_residual)),
        },
        "fitted_from_annotations": {
            "a": float(fitted[0]),
            "b": float(fitted[1]),
            "c": float(fitted[2]),
            "mae_mz": float(np.mean(np.abs(residual))),
            "max_abs_error_mz": float(np.max(np.abs(residual))),
        },
    }


def bounds_overlap_count(annotations: pd.DataFrame) -> int:
    ordered = annotations.sort_values("left_bound").reset_index(drop=True)
    count = 0
    active_right = -np.inf
    for row in ordered.itertuples(index=False):
        if float(row.left_bound) <= active_right:
            count += 1
        active_right = max(active_right, float(row.right_bound))
    return count


def reference_intensity_match_rate(annotations: pd.DataFrame, summed: np.ndarray) -> float | None:
    if "reference_intensity" not in annotations:
        return None
    matched = 0
    usable = 0
    for row in annotations.itertuples(index=False):
        left = max(0, int(round(row.left_bound - 1.0)))
        right = min(len(summed) - 1, int(round(row.right_bound - 1.0)))
        if right < left or not np.isfinite(row.reference_intensity):
            continue
        usable += 1
        if np.isclose(float(row.reference_intensity), float(np.max(summed[left : right + 1])), rtol=0, atol=1e-8):
            matched += 1
    return matched / usable if usable else None


def format_metric_table(frame: pd.DataFrame) -> str:
    lines = [
        "| 算法 | 检出 | 命中 | 漏检 | 额外候选 | 召回率 | 精确率 | F1 | 耗时(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.algorithm} | {row.detected_count} | {row.true_positive} | {row.missed_count} | "
            f"{row.extra_count} | {row.recall:.3f} | {row.precision:.3f} | {row.f1:.3f} | "
            f"{row.runtime_seconds:.3f} |"
        )
    return "\n".join(lines)


def write_report(
    output_path: Path,
    records: list[dict[str, Any]],
    annotations: pd.DataFrame,
    profile_summary: pd.DataFrame,
    recurrence: pd.DataFrame,
    extra_clusters: pd.DataFrame,
    diagnostics: dict[str, Any],
    normalized_interval_count: int,
    intensity_match_rate: float | None,
) -> None:
    summed_metrics = profile_summary[profile_summary["profile"] == "sum"].copy()
    single_union = (
        recurrence.groupby("algorithm", as_index=False)
        .agg(
            union_peaks_seen=("detected_spectrum_count", lambda values: int((values > 0).sum())),
            median_spectra_per_peak=("detected_spectrum_count", "median"),
        )
    )
    single_union["union_recall"] = single_union["union_peaks_seen"] / len(annotations)
    extra_union = (
        extra_clusters.groupby("algorithm").size().to_dict()
        if not extra_clusters.empty
        else {}
    )
    lines = [
        "# 11.0 eV 温度扫描寻峰验证",
        "",
        "## 数据与判定口径",
        "",
        f"- 光谱数：{len(records)}；温度：{min(item['temperature'] for item in records):g}–"
        f"{max(item['temperature'] for item in records):g} °C；每张谱点数：{len(records[0]['y'])}。",
        f"- 人工并集峰：{len(annotations)}；峰区间重叠：{bounds_overlap_count(annotations)} 处；"
        f"中心超出原区间并仅在评估时扩展：{normalized_interval_count} 条。",
        "- 数组第 0 点对应文件坐标 TOF=1；检测使用 `time_offset=1.0`。",
        "- 主指标在 12 张谱的逐点累计谱上计算；检测峰中心落入人工区间才算命中，并执行一对一分配。",
        "- 单温谱的人工真值是所有温度的并集，因此单谱只报告“并集覆盖”，不把未出现峰直接称为漏检。",
    ]
    if intensity_match_rate is not None:
        lines.append(f"- 人工表“强度”与累计谱各区间最大值完全一致的比例：{intensity_match_rate:.1%}。")
    lines.extend(
        [
            "",
            "## 累计谱结果（当前项目参数）",
            "",
            format_metric_table(summed_metrics),
            "",
            "## 跨温度并集覆盖",
            "",
            "| 算法 | 至少被一张单谱检出的人工峰 | 并集覆盖率 | 每个人工峰检出谱数中位数 | 未匹配候选簇 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in single_union.itertuples(index=False):
        lines.append(
            f"| {row.algorithm} | {row.union_peaks_seen} | {row.union_recall:.3f} | "
            f"{row.median_spectra_per_peak:.1f} | {extra_union.get(row.algorithm, 0)} |"
        )
    configured = diagnostics["configured"]
    fitted = diagnostics["fitted_from_annotations"]
    sum_by_algorithm = summed_metrics.set_index("algorithm")
    mean_by_algorithm = profile_summary[profile_summary["profile"] == "mean"].set_index("algorithm")
    ensemble_sum = sum_by_algorithm.loc["ensemble"]
    ensemble_mean = mean_by_algorithm.loc["ensemble"]
    lines.extend(
        [
            "",
            "## 标定诊断",
            "",
            f"- 项目标定对人工 m/z 的 MAE：{configured['mae_mz']:.6f} m/z；"
            f"最大绝对误差：{configured['max_abs_error_mz']:.6f} m/z；"
            f"中位有符号误差：{configured['median_signed_error_mz']:+.6f} m/z。",
            f"- 由 299 个标注点拟合的二次系数：`a={fitted['a']:.15g}`、"
            f"`b={fitted['b']:.15g}`、`c={fitted['c']:.15g}`；"
            f"拟合 MAE={fitted['mae_mz']:.6f} m/z。",
            "- 因此本报告用 TOF 区间评价寻峰，不用 m/z 容差混入标定误差；拟合系数仅作诊断，未改写项目标定。",
            "",
            "## 结论与使用建议",
            "",
            f"- 若固定采用 12 张谱的累计强度，当前 ensemble 的召回率为 {ensemble_sum.recall:.2%}，"
            f"适合作为“候选峰全集生成器”；但 {int(ensemble_sum.extra_count)} 个未匹配候选仍需人工确认，"
            "不宜直接当成最终峰表。",
            f"- 同一信号除以 12 后，ensemble 召回率从 {ensemble_sum.recall:.2%} 降至 "
            f"{ensemble_mean.recall:.2%}。当前绝对强度阈值对累计张数和归一化方式敏感，"
            "尚不足以作为跨数据集、跨采集时长的稳健默认算法。",
            f"- 当前 CWT 在累计谱上仅命中 {int(sum_by_algorithm.loc['cwt'].true_positive)}/{len(annotations)}，"
            "不应在现参数下承担补漏；建议改为真正的跨尺度脊线追踪，或先从 ensemble 中停用。",
            "- 推荐工作流：先对原始谱做噪声/基线归一化；用累计谱生成并集候选；再回到各温度谱做区间积分；"
            "按“多算法同意、跨温度重复出现、局部 SNR、峰形完整性”给候选分级，最后只让人工复核低置信和冲突峰。",
            "- 下一轮优化应以本数据的 299 峰为固定验证集，同时报告召回率、未匹配候选数和强度缩放稳定性，"
            "避免只优化单一召回率。",
            "",
            "## 输出说明",
            "",
            "- `summed_spectrum.csv`：12 张谱逐点求和与均值。",
            "- `peak_ranges_core.csv`：人工峰表的四个核心字段，原始文件未改写。",
            "- `peak_integrals_by_temperature.csv` / `peak_integrals_summary.csv`：各人工区间积分及跨温度汇总。",
            "- `profile_validation_summary.csv`：累计谱和均值谱的四算法指标。",
            "- `profile_missing_annotations.csv`：各算法未命中的人工峰。",
            "- `per_spectrum_validation.csv`：每张温度谱相对完整并集的覆盖情况。",
            "- `annotation_detection_recurrence.csv`：每个人工峰在多少张单谱中被检出。",
            "- `unmatched_detection_clusters.csv`：跨温度合并后的未匹配候选簇。",
            "- `calibration_diagnostics.yaml`：项目标定与标注拟合标定的误差。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(dataset_dir: Path, spectra_subdir: str, output_subdir: str) -> Path:
    spectra_dir = dataset_dir / spectra_subdir
    annotations_path = dataset_dir / "peak_ranges.csv"
    output_dir = dataset_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(annotations_path)
    records, matrix = load_spectra(spectra_dir)
    summed = matrix.sum(axis=0)
    mean = matrix.mean(axis=0)
    tof_axis = np.arange(1, matrix.shape[1] + 1, dtype=int)
    pd.DataFrame({"tof": tof_axis, "intensity_sum": summed, "intensity_mean": mean}).to_csv(
        output_dir / "summed_spectrum.csv", index=False
    )
    annotations[["mz", "tof", "left_bound", "right_bound"]].to_csv(
        output_dir / "peak_ranges_core.csv", index=False
    )

    integral_rows: list[dict[str, Any]] = []
    for truth_index, row in annotations.iterrows():
        left = max(0, int(round(float(row.left_bound) - 1.0)))
        right = min(matrix.shape[1] - 1, int(round(float(row.right_bound) - 1.0)))
        for spectrum_index, record in enumerate(records):
            region = matrix[spectrum_index, left : right + 1]
            apex_local = int(np.argmax(region))
            integral_rows.append(
                {
                    "truth_index": truth_index,
                    "mz": float(row.mz),
                    "tof": float(row.tof),
                    "left_bound": float(row.left_bound),
                    "right_bound": float(row.right_bound),
                    "file": record["file"],
                    "temperature": record["temperature"],
                    "energy": record["energy"],
                    "area": float(np.sum(region)),
                    "apex_intensity": float(region[apex_local]),
                    "apex_tof": float(left + apex_local + 1),
                }
            )
    integral_frame = pd.DataFrame(integral_rows)
    integral_frame.to_csv(output_dir / "peak_integrals_by_temperature.csv", index=False)
    integral_summary = (
        integral_frame.groupby(["truth_index", "mz", "tof", "left_bound", "right_bound"], as_index=False)
        .agg(
            total_area=("area", "sum"),
            mean_area=("area", "mean"),
            max_area=("area", "max"),
            max_apex_intensity=("apex_intensity", "max"),
        )
    )
    maximum_temperature = (
        integral_frame.loc[integral_frame.groupby("truth_index")["area"].idxmax(), ["truth_index", "temperature"]]
        .rename(columns={"temperature": "max_area_temperature"})
    )
    integral_summary.merge(maximum_temperature, on="truth_index", how="left").to_csv(
        output_dir / "peak_integrals_summary.csv", index=False
    )

    calibration = load_calibration_config(PROJECT_ROOT / "config" / "calibration.yaml")
    peak_config = load_peak_detection_config(PROJECT_ROOT / "config" / "peak_detection.yaml")
    peak_kwargs = asdict(peak_config)
    peak_kwargs.pop("algorithm", None)
    diagnostics = calibration_diagnostics(annotations, calibration)
    (output_dir / "calibration_diagnostics.yaml").write_text(
        yaml.safe_dump(diagnostics, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    profile_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for profile_name, profile in (("sum", summed), ("mean", mean)):
        for algorithm in ALGORITHMS:
            peaks, runtime = detect(profile, algorithm, calibration, peak_kwargs)
            metrics, matches, missed, extras = evaluate(peaks, annotations)
            profile_rows.append(
                {"profile": profile_name, "algorithm": algorithm, **metrics, "runtime_seconds": runtime}
            )
            for truth_index in missed:
                truth = annotations.iloc[truth_index]
                missing_rows.append(
                    {
                        "profile": profile_name,
                        "algorithm": algorithm,
                        "truth_index": truth_index,
                        "mz": truth.mz,
                        "tof": truth.tof,
                        "left_bound": truth.left_bound,
                        "right_bound": truth.right_bound,
                        "reference_intensity": truth.get("reference_intensity", np.nan),
                    }
                )
            matched_by_detected = {item["detected_index"]: item for item in matches}
            for detected_index, peak in enumerate(peaks):
                match = matched_by_detected.get(detected_index)
                detail_rows.append(
                    {
                        "profile": profile_name,
                        "algorithm": algorithm,
                        "detected_index": detected_index,
                        "detected_time": peak.time,
                        "detected_mz": peak.mz,
                        "detected_intensity": peak.intensity,
                        "truth_index": match["truth_index"] if match else np.nan,
                        "center_error_samples": match["center_error_samples"] if match else np.nan,
                        "matched": match is not None,
                    }
                )
    profile_summary = pd.DataFrame(profile_rows)
    profile_summary.to_csv(output_dir / "profile_validation_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "profile_detection_details.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(output_dir / "profile_missing_annotations.csv", index=False)

    per_spectrum_rows: list[dict[str, Any]] = []
    recurrence_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        detected_by_truth: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(annotations))}
        algorithm_extras: list[dict[str, Any]] = []
        for spectrum_index, record in enumerate(records):
            peaks, runtime = detect(matrix[spectrum_index], algorithm, calibration, peak_kwargs)
            metrics, matches, _, extras = evaluate(peaks, annotations)
            per_spectrum_rows.append(
                {
                    "algorithm": algorithm,
                    "file": record["file"],
                    "temperature": record["temperature"],
                    "energy": record["energy"],
                    "detected_count": metrics["detected_count"],
                    "matched_to_union": metrics["true_positive"],
                    "unmatched_count": metrics["extra_count"],
                    "coverage_of_full_union": metrics["recall"],
                    "runtime_seconds": runtime,
                }
            )
            for match in matches:
                detected_by_truth[match["truth_index"]].append(
                    {
                        "file": record["file"],
                        "temperature": record["temperature"],
                        "detected_time": match["detected_time"],
                    }
                )
            for detected_index in extras:
                peak = peaks[detected_index]
                algorithm_extras.append(
                    {
                        "algorithm": algorithm,
                        "file": record["file"],
                        "temperature": record["temperature"],
                        "detected_time": float(peak.time),
                        "detected_mz": float(peak.mz),
                        "detected_intensity": float(peak.intensity),
                    }
                )
        for truth_index, detections in detected_by_truth.items():
            truth = annotations.iloc[truth_index]
            recurrence_rows.append(
                {
                    "algorithm": algorithm,
                    "truth_index": truth_index,
                    "mz": truth.mz,
                    "tof": truth.tof,
                    "detected_spectrum_count": len(detections),
                    "temperatures": ";".join(f"{item['temperature']:g}" for item in detections),
                }
            )
        for cluster in cluster_extra_detections(algorithm_extras, tolerance=float(peak_config.duplicate_window)):
            unmatched_rows.append({"algorithm": algorithm, **cluster})

    per_spectrum = pd.DataFrame(per_spectrum_rows)
    recurrence = pd.DataFrame(recurrence_rows)
    extra_clusters = pd.DataFrame(unmatched_rows)
    per_spectrum.to_csv(output_dir / "per_spectrum_validation.csv", index=False)
    recurrence.to_csv(output_dir / "annotation_detection_recurrence.csv", index=False)
    extra_clusters.to_csv(output_dir / "unmatched_detection_clusters.csv", index=False)

    normalized_interval_count = normalized_truth_intervals(annotations)[2]
    intensity_match_rate = reference_intensity_match_rate(annotations, summed)
    quality = {
        "spectrum_count": len(records),
        "spectrum_point_count": int(matrix.shape[1]),
        "temperatures": [float(item["temperature"]) for item in records],
        "energies": [float(item["energy"]) for item in records],
        "annotation_count": len(annotations),
        "duplicate_tof_count": int(annotations["tof"].duplicated().sum()),
        "duplicate_mz_count": int(annotations["mz"].duplicated().sum()),
        "overlapping_interval_count": bounds_overlap_count(annotations),
        "normalized_interval_count_for_evaluation": normalized_interval_count,
        "reference_intensity_match_rate": intensity_match_rate,
        "time_offset": 1.0,
        "peak_detection_config": asdict(peak_config),
    }
    (output_dir / "dataset_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir / "validation_report.md",
        records,
        annotations,
        profile_summary,
        recurrence,
        extra_clusters,
        diagnostics,
        normalized_interval_count,
        intensity_match_rate,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="folder containing peak_ranges.csv and the spectra subfolder")
    parser.add_argument("--spectra-subdir", default="11.0eV")
    parser.add_argument("--output-subdir", default="processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run(args.dataset_dir.resolve(), args.spectra_subdir, args.output_subdir)
    print(output_dir)


if __name__ == "__main__":
    main()
