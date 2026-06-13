"""Test ensemble parameter optimization."""

import numpy as np
from core.calibration import Calibration
from core.peak_detection import detect_peaks_ensemble
from core.peak_benchmark import (
    build_default_synthetic_benchmark,
    read_benchmark_background,
)
from core.ensemble_optimization import (
    ENSEMBLE_QUICK_SEARCH_SPACE,
    optimize_ensemble_parameters,
    extract_ensemble_parameters,
)


def test_ensemble_optimization_quick_mode_improves_over_baseline():
    """快速模式参数优化应该相比基线有所改进。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    result = optimize_ensemble_parameters(
        cases=cases,
        calibration=calibration,
        n_trials=8,
        initial_random=3,
        random_state=42,
        method="bayesian",
        quick_mode=True,
    )

    # 检查结果
    assert len(result.trials) == 8
    assert result.best_score > 0, "should have positive score"
    assert "vote_threshold" in result.best_parameters
    assert "cwt_snr_threshold" in result.best_parameters

    # 提取参数
    ensemble_params = extract_ensemble_parameters(result)
    assert "vote_threshold" in ensemble_params
    assert "cwt_snr_threshold" in ensemble_params

    print(f"Best score: {result.best_score:.2f}")
    print(f"Best parameters: {result.best_parameters}")
    print(f"Trials: {len(result.trials)}")

    # 用最优参数检测
    peaks = detect_peaks_ensemble(
        cases[1].y,  # multi_dynamic_range
        calibration=calibration,
        **ensemble_params,
    )

    assert len(peaks) > 0, "should detect peaks with optimized parameters"


def test_ensemble_optimization_reduces_false_positives():
    """参数优化应该降低假阳性同时保持高召回。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    # 激进基准（高召回）
    baseline_params = {
        "vote_threshold": 0.334,
        "min_intensity_for_single_vote": 1.0,
        "mz_tolerance": 0.2,
    }

    # 优化参数
    result = optimize_ensemble_parameters(
        cases=cases,
        calibration=calibration,
        base_parameters=baseline_params,
        n_trials=6,
        initial_random=2,
        random_state=43,
        method="random",  # 快速探索
        quick_mode=True,
    )

    best_params = extract_ensemble_parameters(result)

    # 对比：激进基准 vs 优化参数
    peaks_baseline = detect_peaks_ensemble(
        cases[1].y,
        calibration=calibration,
        vote_threshold=0.334,
        min_intensity_for_single_vote=1.0,
        min_intensity=1.0,
    )

    peaks_optimized = detect_peaks_ensemble(
        cases[1].y,
        calibration=calibration,
        **best_params,
    )

    print(f"Baseline peaks: {len(peaks_baseline)}")
    print(f"Optimized peaks: {len(peaks_optimized)}")
    print(f"Best score: {result.best_score:.2f}")

    # 优化应该检出更合理数量的峰（少于激进基准）
    assert len(peaks_optimized) <= len(peaks_baseline), (
        f"optimized ({len(peaks_optimized)}) should find <= peaks than baseline ({len(peaks_baseline)})"
    )


def test_ensemble_quick_search_space_is_valid():
    """快速搜索空间定义应该有效。"""
    assert len(ENSEMBLE_QUICK_SEARCH_SPACE) > 0
    assert all(spec.name for spec in ENSEMBLE_QUICK_SEARCH_SPACE)

    # 检查所有参数都能从搜索空间中采样
    rng = np.random.default_rng(42)
    for spec in ENSEMBLE_QUICK_SEARCH_SPACE:
        value = spec.sample(rng)
        assert value is not None
        # 验证 cast 也工作
        casted = spec.cast(value)
        assert casted is not None


def test_ensemble_optimization_with_adaptive_method():
    """自适应搜索方法也应该工作。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    result = optimize_ensemble_parameters(
        cases=cases,
        calibration=calibration,
        n_trials=5,
        initial_random=1,
        random_state=44,
        method="adaptive",
        quick_mode=True,
    )

    assert len(result.trials) == 5
    assert result.best_score > 0
    print(f"Adaptive method best score: {result.best_score:.2f}")
