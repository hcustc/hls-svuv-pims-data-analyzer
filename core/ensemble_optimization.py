"""Ensemble peak detection parameter optimization.

Bayesian optimization for the ensemble peak detection framework.
Balances recall vs false positive rate through parameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

from .bo_peak_optimization import (
    ParameterSpec,
    PeakOptimizationResult,
    run_peak_parameter_calibration,
    _normalise_parameters,
    ParameterValue,
)
from .calibration import Calibration
from .peak_detection import detect_peaks_ensemble
from .peak_benchmark import (
    SyntheticPeakCase,
    evaluate_synthetic_peak_benchmark,
    _match_peaks,
)


# 融合搜索空间：优化投票参数 + 三个算法的关键参数
ENSEMBLE_SEARCH_SPACE: tuple[ParameterSpec, ...] = (
    # 融合参数
    ParameterSpec("vote_threshold", 0.334, 1.0, "float"),  # 1/3 到 3/3
    ParameterSpec("min_intensity_for_single_vote", 1.0, 20.0, "float"),
    ParameterSpec("mz_tolerance", 0.1, 0.5, "float"),

    # Legacy 算法参数（recall-focused）
    ParameterSpec("legacy_threshold_end", 0.5, 3.0, "float"),
    ParameterSpec("legacy_min_intensity", 0.5, 3.0, "float"),
    ParameterSpec("legacy_weak_tail_ratio", 1.5, 8.0, "float"),
    ParameterSpec("legacy_weak_tail_early_window", 60, 180, "int"),
    ParameterSpec("legacy_weak_tail_late_window", 30, 120, "int"),

    # Prominence 算法参数（recall-focused）
    ParameterSpec("prominence_ratio", 0.0005, 0.01, "log_float"),
    ParameterSpec("prominence_min_intensity", 0.5, 5.0, "float"),
    ParameterSpec("baseline_percentile", 1.0, 15.0, "float"),
    ParameterSpec("smoothing_window", 5, 15, "int"),

    # CWT 算法参数（recall-focused）
    ParameterSpec("cwt_snr_threshold", 0.003, 0.05, "log_float"),
    ParameterSpec("cwt_wavelet_max_width", 30, 90, "int"),
    ParameterSpec("cwt_min_peak_width", 1, 10, "int"),
)

# 快速基准搜索空间（维度更少，用于快速探索）
ENSEMBLE_QUICK_SEARCH_SPACE: tuple[ParameterSpec, ...] = (
    # 融合参数
    ParameterSpec("vote_threshold", 0.334, 1.0, "float"),
    ParameterSpec("min_intensity_for_single_vote", 1.0, 20.0, "float"),
    ParameterSpec("mz_tolerance", 0.15, 0.4, "float"),

    # Legacy 关键参数
    ParameterSpec("legacy_min_intensity", 0.5, 3.0, "float"),
    ParameterSpec("legacy_weak_tail_ratio", 1.5, 8.0, "float"),

    # Prominence 关键参数
    ParameterSpec("prominence_ratio", 0.001, 0.01, "log_float"),
    ParameterSpec("baseline_percentile", 2.0, 10.0, "float"),

    # CWT 关键参数
    ParameterSpec("cwt_snr_threshold", 0.005, 0.03, "log_float"),
    ParameterSpec("cwt_wavelet_max_width", 40, 80, "int"),
)


def _evaluate_ensemble_synthetic(
    cases: list[SyntheticPeakCase],
    parameters: dict[str, ParameterValue],
    *,
    calibration: Calibration,
) -> tuple[float, dict[str, Any]]:
    """评估融合方案在合成峰上的性能。

    目标：最大化召回率，同时惩罚假阳性
    评分 = recall_score - fp_penalty
    """
    # 映射参数到 detect_peaks_ensemble 签名
    ensemble_params = {
        "vote_threshold": float(parameters.get("vote_threshold", 0.667)),
        "min_intensity_for_single_vote": float(parameters.get("min_intensity_for_single_vote", 5.0)),
        "mz_tolerance": float(parameters.get("mz_tolerance", 0.2)),
        "threshold_end": float(parameters.get("legacy_threshold_end", 1.0)),
        "min_intensity": float(parameters.get("legacy_min_intensity", 1.0)),
        "weak_tail_ratio": float(parameters.get("legacy_weak_tail_ratio", 3.0)),
        "weak_tail_early_window": int(parameters.get("legacy_weak_tail_early_window", 120)),
        "weak_tail_late_window": int(parameters.get("legacy_weak_tail_late_window", 80)),
        "prominence_ratio": float(parameters.get("prominence_ratio", 0.002)),
        "baseline_percentile": float(parameters.get("baseline_percentile", 5.0)),
        "smoothing_window": int(parameters.get("smoothing_window", 7)),
        "cwt_snr_threshold": float(parameters.get("cwt_snr_threshold", 0.01)),
        "cwt_wavelet_max_width": int(parameters.get("cwt_wavelet_max_width", 60)),
        "min_peak_width": int(parameters.get("cwt_min_peak_width", 1)),
    }

    # 评估所有合成峰用例
    case_results = []
    total_tp = 0
    total_fp = 0
    total_truth = 0

    for case in cases:
        # 直接使用融合检测
        peaks = detect_peaks_ensemble(
            case.y,
            calibration=calibration,
            **ensemble_params,
        )

        # 匹配检出的峰和注入的峰
        matches, missed, false_peaks = _match_peaks(
            peaks,
            case.injected_peaks,
            mz_tolerance=0.35,
        )

        result = {
            "case": case.name,
            "description": case.description,
            "detected_count": len(peaks),
            "true_positive_count": len(matches),
            "false_positive_count": len(false_peaks),
            "missed_count": len(missed),
            "truth_count": len(case.injected_peaks),
        }

        # 计算 recall/precision
        if len(case.injected_peaks) > 0:
            result["recall"] = len(matches) / len(case.injected_peaks)
            result["precision"] = len(matches) / len(peaks) if len(peaks) > 0 else 0.0
        else:
            result["recall"] = 1.0
            result["precision"] = 1.0 if len(false_peaks) == 0 else 0.0

        case_results.append(result)
        total_tp += len(matches)
        total_fp += len(false_peaks)
        total_truth += len(case.injected_peaks)

    # 计算评分
    if total_truth == 0:
        # 空白背景：最小化假阳性
        recall = 1.0
        fp_penalty = 0.1 * total_fp
        objective_score = 100.0 - fp_penalty
    else:
        # 非空白：优先召回，次要惩罚假阳性
        recall = total_tp / total_truth
        fp_penalty = 0.05 * total_fp  # 轻微惩罚（recall 优先）
        objective_score = 100.0 * recall - fp_penalty

    metrics = {
        "score": float(objective_score),
        "objective": "ensemble_recall_with_fp_penalty",
        "total_recall": float(recall),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_truth": int(total_truth),
        "fp_penalty": float(fp_penalty),
        "case_count": len(case_results),
        "case_results": case_results,
    }

    return float(objective_score), metrics


def optimize_ensemble_parameters(
    *,
    cases: list[SyntheticPeakCase],
    calibration: Calibration,
    base_parameters: dict[str, ParameterValue] | None = None,
    n_trials: int = 30,
    initial_random: int = 8,
    random_state: int = 42,
    method: str = "bayesian",
    quick_mode: bool = False,
) -> PeakOptimizationResult:
    """优化融合峰检测的参数。

    Args:
        cases: 合成峰用例列表
        calibration: 质量标度校准
        base_parameters: 初始参数（可选）
        n_trials: 总试验次数
        initial_random: 随机搜索试验次数
        random_state: 随机种子
        method: 搜索方法 ('bayesian', 'random', 'grid', 'adaptive')
        quick_mode: 使用快速搜索空间（更少维度）

    Returns:
        优化结果（最优参数 + 试验历史）
    """
    search_space = list(ENSEMBLE_QUICK_SEARCH_SPACE if quick_mode else ENSEMBLE_SEARCH_SPACE)

    # 默认基础参数
    if base_parameters is None:
        base_parameters = {
            "vote_threshold": 0.667,
            "min_intensity_for_single_vote": 5.0,
            "mz_tolerance": 0.2,
            "legacy_threshold_end": 1.0,
            "legacy_min_intensity": 1.0,
            "legacy_weak_tail_ratio": 3.0,
            "legacy_weak_tail_early_window": 120,
            "legacy_weak_tail_late_window": 80,
            "prominence_ratio": 0.002,
            "prominence_min_intensity": 1.0,
            "baseline_percentile": 5.0,
            "smoothing_window": 7,
            "cwt_snr_threshold": 0.01,
            "cwt_wavelet_max_width": 60,
            "cwt_min_peak_width": 1,
        }

    def evaluator(parameters: dict[str, ParameterValue]) -> tuple[float, dict[str, Any]]:
        return _evaluate_ensemble_synthetic(
            cases,
            parameters,
            calibration=calibration,
        )

    result = run_peak_parameter_calibration(
        analysis_type="ensemble_peak_detection",
        folder="synthetic_benchmark",
        base_parameters=base_parameters,
        evaluator=evaluator,
        search_space=search_space,
        method=method,
        n_trials=n_trials,
        initial_random=initial_random,
        random_state=random_state,
    )

    return result


def extract_ensemble_parameters(
    optimization_result: PeakOptimizationResult,
) -> dict[str, Any]:
    """从优化结果中提取融合方案参数。

    返回可直接传给 detect_peaks_ensemble() 的参数字典。
    """
    params = optimization_result.best_parameters

    return {
        "vote_threshold": float(params.get("vote_threshold", 0.667)),
        "min_intensity_for_single_vote": float(params.get("min_intensity_for_single_vote", 5.0)),
        "mz_tolerance": float(params.get("mz_tolerance", 0.2)),
        # Legacy
        "threshold_end": float(params.get("legacy_threshold_end", 1.0)),
        "min_intensity": float(params.get("legacy_min_intensity", 1.0)),
        "weak_tail_ratio": float(params.get("legacy_weak_tail_ratio", 3.0)),
        "weak_tail_early_window": int(params.get("legacy_weak_tail_early_window", 120)),
        "weak_tail_late_window": int(params.get("legacy_weak_tail_late_window", 80)),
        # Prominence
        "prominence_ratio": float(params.get("prominence_ratio", 0.002)),
        "baseline_percentile": float(params.get("baseline_percentile", 5.0)),
        "smoothing_window": int(params.get("smoothing_window", 7)),
        # CWT
        "cwt_snr_threshold": float(params.get("cwt_snr_threshold", 0.01)),
        "cwt_wavelet_max_width": int(params.get("cwt_wavelet_max_width", 60)),
        "min_peak_width": int(params.get("cwt_min_peak_width", 1)),
    }
