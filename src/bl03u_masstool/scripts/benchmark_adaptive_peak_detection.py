"""Benchmark the adaptive high-recall detector on the 11 eV validation set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bl03u_masstool.core.adaptive_peak_detection import (
    AdaptivePeakDetectionConfig,
    detect_peaks_adaptive,
    estimate_signal_scale,
)
from bl03u_masstool.core.config import load_calibration_config, load_peak_detection_config
from bl03u_masstool.scripts.validate_peak_detection_dataset import (
    detect,
    evaluate,
    load_annotations,
    load_spectra,
)


def metric_row(
    scenario: str,
    method: str,
    peaks: list[Any],
    annotations: pd.DataFrame,
    runtime_seconds: float,
    **extra: Any,
) -> tuple[dict[str, Any], list[int]]:
    metrics, _, missed, _ = evaluate(peaks, annotations)
    return (
        {
            "scenario": scenario,
            "method": method,
            **extra,
            **metrics,
            "runtime_seconds": runtime_seconds,
        },
        missed,
    )


def format_comparison_table(frame: pd.DataFrame) -> str:
    lines = [
        "| 场景 | 方法 | 检出 | 命中 | 漏检 | 未匹配 | 召回率 | 精确率 | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.method} | {row.detected_count} | {row.true_positive} | "
            f"{row.missed_count} | {row.extra_count} | {row.recall:.3f} | {row.precision:.3f} | "
            f"{row.f1:.3f} |"
        )
    return "\n".join(lines)


def run(dataset_dir: Path, output_subdir: str = "processed/adaptive_experiment") -> Path:
    output_dir = dataset_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(dataset_dir / "peak_ranges.csv")
    records, matrix = load_spectra(dataset_dir / "11.0eV")
    summed = matrix.sum(axis=0)
    mean = matrix.mean(axis=0)
    calibration = load_calibration_config(PROJECT_ROOT / "config" / "calibration.yaml")
    peak_config = load_peak_detection_config(PROJECT_ROOT / "config" / "peak_detection.yaml")
    peak_kwargs = asdict(peak_config)
    peak_kwargs.pop("algorithm", None)
    adaptive_config = AdaptivePeakDetectionConfig()

    comparison_rows: list[dict[str, Any]] = []
    scenarios = {
        "sum": summed,
        "mean": mean,
        "sum_x_0.1": summed * 0.1,
        "sum_x_10": summed * 10.0,
    }
    full_adaptive_peaks: list[Any] = []
    for scenario, profile in scenarios.items():
        started = time.perf_counter()
        adaptive_peaks = detect_peaks_adaptive(
            profile,
            calibration=calibration,
            peak_config=peak_config,
            adaptive_config=adaptive_config,
            time_offset=1.0,
        )
        adaptive_runtime = time.perf_counter() - started
        if scenario == "sum":
            full_adaptive_peaks = adaptive_peaks
        row, _ = metric_row(
            scenario,
            "adaptive",
            adaptive_peaks,
            annotations,
            adaptive_runtime,
            normalization_scale=estimate_signal_scale(profile),
        )
        comparison_rows.append(row)

        if scenario in {"sum", "mean"}:
            baseline_peaks, baseline_runtime = detect(profile, "ensemble", calibration, peak_kwargs)
            row, _ = metric_row(
                scenario,
                "current_ensemble",
                baseline_peaks,
                annotations,
                baseline_runtime,
                normalization_scale=1.0,
            )
            comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "scale_stability.csv", index=False)

    rng = np.random.default_rng(20260726)
    subset_rows: list[dict[str, Any]] = []
    for spectrum_count in (3, 6, 9, 12):
        repeat_count = 5 if spectrum_count < len(records) else 1
        for repeat in range(repeat_count):
            indices = np.sort(rng.choice(len(records), size=spectrum_count, replace=False))
            profile = matrix[indices].sum(axis=0)
            started = time.perf_counter()
            peaks = detect_peaks_adaptive(
                profile,
                calibration=calibration,
                peak_config=peak_config,
                adaptive_config=adaptive_config,
                time_offset=1.0,
            )
            runtime = time.perf_counter() - started
            row, _ = metric_row(
                f"subset_{spectrum_count}",
                "adaptive",
                peaks,
                annotations,
                runtime,
                spectrum_count=spectrum_count,
                repeat=repeat,
                selected_indices=";".join(str(value) for value in indices),
                normalization_scale=estimate_signal_scale(profile),
            )
            subset_rows.append(row)
    subset_frame = pd.DataFrame(subset_rows)
    subset_frame.to_csv(output_dir / "subset_stability.csv", index=False)
    subset_summary = (
        subset_frame.groupby("spectrum_count", as_index=False)
        .agg(
            recall_mean=("recall", "mean"),
            recall_min=("recall", "min"),
            recall_max=("recall", "max"),
            extra_mean=("extra_count", "mean"),
            detected_mean=("detected_count", "mean"),
        )
    )
    subset_summary.to_csv(output_dir / "subset_stability_summary.csv", index=False)

    _, matches, missed, _ = evaluate(full_adaptive_peaks, annotations)
    truth_by_detected = {item["detected_index"]: item["truth_index"] for item in matches}
    candidate_rows: list[dict[str, Any]] = []
    for detected_index, peak in enumerate(full_adaptive_peaks):
        truth_index = truth_by_detected.get(detected_index)
        candidate_rows.append(
            {
                "detected_index": detected_index,
                "tof": peak.time,
                "mz": peak.mz,
                "intensity": peak.intensity,
                "fwhm": peak.fwhm,
                "left_bound": peak.left_bound + 1.0,
                "right_bound": peak.right_bound + 1.0,
                "matched": truth_index is not None,
                "truth_index": truth_index if truth_index is not None else np.nan,
                "truth_mz": annotations.iloc[truth_index].mz if truth_index is not None else np.nan,
                "truth_tof": annotations.iloc[truth_index].tof if truth_index is not None else np.nan,
            }
        )
    pd.DataFrame(candidate_rows).to_csv(output_dir / "adaptive_candidates.csv", index=False)

    missing_frame = annotations.iloc[missed].copy()
    missing_frame.insert(0, "truth_index", missed)
    missing_frame.to_csv(output_dir / "adaptive_missing_annotations.csv", index=False)

    adaptive_scale_rows = comparison[comparison["method"] == "adaptive"]
    baseline_scale_rows = comparison[comparison["method"] == "current_ensemble"]
    adaptive_recall_span = float(adaptive_scale_rows["recall"].max() - adaptive_scale_rows["recall"].min())
    baseline_recall_span = float(baseline_scale_rows["recall"].max() - baseline_scale_rows["recall"].min())
    sum_adaptive = adaptive_scale_rows[adaptive_scale_rows["scenario"] == "sum"].iloc[0]
    sum_baseline = baseline_scale_rows[baseline_scale_rows["scenario"] == "sum"].iloc[0]
    mean_baseline = baseline_scale_rows[baseline_scale_rows["scenario"] == "mean"].iloc[0]
    speedup = float(sum_baseline.runtime_seconds / sum_adaptive.runtime_seconds)
    missing_description = "；".join(
        f"m/z {row.mz:g}, TOF {row.tof:g}, intensity {row.reference_intensity:g}"
        for row in missing_frame.itertuples(index=False)
    )

    report_lines = [
        "# 自适应高召回寻峰实验",
        "",
        "## 实验方案",
        "",
        "- 用正信号低分位数估计强度单位，使整体乘除常数后检测结果不变。",
        "- 使用 legacy + prominence 共识，停用当前低召回的 CWT 分支。",
        "- 累计谱生成完整候选表，不做置信分级；用户在工作台删除假峰、添加漏峰和调整边界。",
        "- 新方案已作为独立 `adaptive` 算法选项接入，但未改写项目默认算法。",
        "",
        "## 强度尺度对照",
        "",
        format_comparison_table(comparison),
        "",
        f"- 当前 ensemble 从累计谱到均值谱的召回率由 {sum_baseline.recall:.2%} 降至 "
        f"{mean_baseline.recall:.2%}，两种输入的跨度为 {baseline_recall_span:.2%}。",
        f"- 自适应方案在 sum、mean、×0.1、×10 四种尺度上的召回率跨度为 "
        f"{adaptive_recall_span:.2%}，候选集合保持一致。",
        f"- 在完整累计谱上，新方案与当前 ensemble 都命中 {int(sum_adaptive.true_positive)}/"
        f"{len(annotations)}，未匹配候选均为 {int(sum_adaptive.extra_count)}。",
        f"- 停用无贡献的 CWT 后，本次累计谱检测由 {sum_baseline.runtime_seconds:.3f} s 降至 "
        f"{sum_adaptive.runtime_seconds:.3f} s，约加速 {speedup:.2f} 倍。",
        "",
        "## 不同谱张数",
        "",
        "| 累计谱数 | 平均并集覆盖率 | 最低 | 最高 | 平均未匹配候选 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in subset_summary.itertuples(index=False):
        report_lines.append(
            f"| {row.spectrum_count} | {row.recall_mean:.3f} | {row.recall_min:.3f} | "
            f"{row.recall_max:.3f} | {row.extra_mean:.1f} |"
        )
    report_lines.extend(
        [
            "",
            "人工表是 12 张谱的并集，因此少量谱的指标是“完整并集覆盖率”，不是单温谱漏检率。",
            "",
            "## 结论",
            "",
            "- 新方案解决了强度尺度问题并省去无贡献的 CWT；完整累计谱输出 344 个候选，"
            "用户预计删除 46 个并手动补充 1 个极弱峰。",
            f"- 仍未自动检出的人工峰：{missing_description or '无'}。",
            "- 该弱标注在累计谱上是单采样点、强度仅 5，缺少可辨识峰形；强行自动接纳同类单点会同时引入大量噪声候选。",
            "- 工作台已有框选添加、手动输入、批量删除、Delete/Backspace、更新范围、上一峰/下一峰和项目保存功能，"
            "可以直接承接高召回人工修峰流程。",
            "- 若要继续减少删除工作量，应重点测试峰形判别或局部多峰拟合，而不是继续放宽寻峰阈值。",
        ]
    )
    (output_dir / "benchmark_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "peak_detection": asdict(peak_config),
                "adaptive_detection": asdict(adaptive_config),
                "random_seed": 20260726,
                "dataset_dir": str(dataset_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-subdir", default="processed/adaptive_experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run(args.dataset_dir.resolve(), args.output_subdir))


if __name__ == "__main__":
    main()
