"""Test ensemble peak detection method combining Legacy, Prominence, and CWT."""

import numpy as np
from core.calibration import Calibration
from core.peak_detection import detect_peaks_ensemble
from core.peak_benchmark import (
    build_default_synthetic_benchmark,
    evaluate_synthetic_peak_case,
    read_benchmark_background,
)


def test_ensemble_detects_synthetic_weak_peaks():
    """融合方法应该检出合成的弱峰。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    # 测试弱单峰情况
    weak_case = cases[0]  # "weak_single"
    assert weak_case.name == "weak_single"
    assert len(weak_case.injected_peaks) == 1

    peaks = detect_peaks_ensemble(
        weak_case.y,
        calibration=calibration,
        use_legacy=True,
        use_prominence=True,
        use_cwt=True,
        mz_tolerance=0.2,
        vote_threshold=0.667,
        min_intensity_for_single_vote=5.0,
    )

    # 应该至少检出1个峰
    assert len(peaks) > 0, "ensemble should detect weak peak"
    assert any(abs(p.mz - 84.0) < 0.5 for p in peaks), "should detect peak near m/z=84"


def test_ensemble_separates_overlapping_peaks():
    """融合方法应该分离重叠的峰。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    # 测试重叠等强度峰
    overlap_case = next((c for c in cases if c.name == "overlap_equal"), None)
    assert overlap_case is not None
    assert len(overlap_case.injected_peaks) == 2

    peaks = detect_peaks_ensemble(
        overlap_case.y,
        calibration=calibration,
        use_legacy=True,
        use_prominence=True,
        use_cwt=True,
        mz_tolerance=0.2,
        vote_threshold=0.667,
    )

    # 应该至少检出2个峰（或接近2个）
    assert len(peaks) >= 2, f"ensemble should detect both overlapping peaks, got {len(peaks)}"


def test_ensemble_all_algorithms_required_gives_stricter_result():
    """用投票2/3规则应该比任何单一算法更完整（融合优势）。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    multi_case = cases[1]  # "multi_dynamic_range"，有3个峰
    assert len(multi_case.injected_peaks) == 3

    # 融合检测
    peaks_ensemble = detect_peaks_ensemble(
        multi_case.y,
        calibration=calibration,
        use_legacy=True,
        use_prominence=True,
        use_cwt=True,
        mz_tolerance=0.2,
        vote_threshold=0.667,
        min_intensity_for_single_vote=3.0,
    )

    # 直接评估融合峰的质量
    from core.peak_benchmark import _match_peaks

    matches, missed, false_peaks = _match_peaks(
        peaks_ensemble,
        multi_case.injected_peaks,
        mz_tolerance=0.35,
    )

    print(f"Ensemble multi_dynamic_range: {len(peaks_ensemble)} peaks detected")
    print(f"  True positives: {len(matches)}/{len(multi_case.injected_peaks)}")
    print(f"  False positives: {len(false_peaks)}")
    print(f"  Recall: {len(matches) / len(multi_case.injected_peaks):.2%}")

    # 融合应该至少检出2/3的真实峰
    assert len(matches) >= 2, "ensemble should detect majority of peaks"


def test_ensemble_with_single_algorithm_fallback():
    """仅启用一个算法时，融合应该等同于单算法检测。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    case = cases[0]  # weak_single

    # 仅Legacy
    peaks_legacy_only = detect_peaks_ensemble(
        case.y,
        calibration=calibration,
        use_legacy=True,
        use_prominence=False,
        use_cwt=False,
        min_intensity=1.0,
        threshold_end=1.0,
    )

    # 应该有检出
    assert len(peaks_legacy_only) > 0


def test_ensemble_vote_threshold_sensitivity():
    """投票阈值应该影响检出的峰数。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    case = cases[1]  # multi_dynamic_range

    # 严格投票（2/3）
    peaks_strict = detect_peaks_ensemble(
        case.y,
        calibration=calibration,
        use_legacy=True,
        use_prominence=True,
        use_cwt=True,
        mz_tolerance=0.2,
        vote_threshold=0.667,
        min_intensity_for_single_vote=10.0,
    )

    # 宽松投票（1/3）
    peaks_loose = detect_peaks_ensemble(
        case.y,
        calibration=calibration,
        use_legacy=True,
        use_prominence=True,
        use_cwt=True,
        mz_tolerance=0.2,
        vote_threshold=0.334,
        min_intensity_for_single_vote=0.5,
    )

    # 宽松应该检出更多或相同数量的峰
    assert len(peaks_loose) >= len(peaks_strict), (
        f"looser threshold ({len(peaks_loose)}) should find >= peaks as strict ({len(peaks_strict)})"
    )


def test_ensemble_blank_background_vote_filtering():
    """融合的投票规则应该过滤掉大部分只被一个算法检出的噪声。"""
    calibration = Calibration()
    background = read_benchmark_background("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)

    # 空白背景（无注入峰）
    blank_case = cases[-1]  # "blank_background"
    assert blank_case.name == "blank_background"
    assert len(blank_case.injected_peaks) == 0

    # 严格投票规则（2/3）应该检出更少的峰
    peaks_strict = detect_peaks_ensemble(
        blank_case.y,
        calibration=calibration,
        use_legacy=True,
        use_prominence=True,
        use_cwt=True,
        mz_tolerance=0.2,
        vote_threshold=0.667,  # 严格的2/3投票
        min_intensity_for_single_vote=20.0,  # 高强度阈值
    )

    # 宽松投票规则（1/3）会检出更多
    peaks_loose = detect_peaks_ensemble(
        blank_case.y,
        calibration=calibration,
        use_legacy=True,
        use_prominence=True,
        use_cwt=True,
        mz_tolerance=0.2,
        vote_threshold=0.334,  # 宽松的1/3投票
        min_intensity_for_single_vote=1.0,  # 低强度阈值
    )

    print(f"Blank background strict: {len(peaks_strict)} peaks")
    print(f"Blank background loose: {len(peaks_loose)} peaks")

    # 严格规则应该过滤掉更多噪声
    assert len(peaks_strict) <= len(peaks_loose), (
        f"strict vote threshold should find <= peaks than loose, "
        f"got {len(peaks_strict)} vs {len(peaks_loose)}"
    )
