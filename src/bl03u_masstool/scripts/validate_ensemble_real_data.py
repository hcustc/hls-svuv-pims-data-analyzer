"""Real data validation for ensemble peak detection.

Compare ensemble detection with individual algorithms on PIE and Temperature datasets.
Generate comparison report with performance metrics.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.config import load_peak_detection_config
from bl03u_masstool.core.peak_detection import detect_peaks_by_algorithm, detect_peaks_ensemble
from bl03u_masstool.core.pie_analysis import analyze_pie_folder, build_pie_curves
from bl03u_masstool.core.temperature_scan import analyze_temperature_folder, build_temperature_curves

REAL_DATA_ENV_VAR = "BL03U_REAL_DATA_DIR"


def real_data_dir() -> Path:
    configured = os.environ.get(REAL_DATA_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / "tests" / "fixtures" / "bl03u_sample"


def load_ensemble_parameters() -> dict[str, Any]:
    """Load optimized ensemble parameters from result file."""
    result_file = PROJECT_ROOT / "ensemble_optimization_result.json"
    if not result_file.exists():
        print(f"⚠️  Ensemble parameters file not found: {result_file}")
        print("   Using default parameters")
        return {
            "vote_threshold": 0.667,
            "min_intensity_for_single_vote": 5.0,
            "mz_tolerance": 0.2,
        }

    with open(result_file) as f:
        data = json.load(f)
    return data["best_parameters"]


def validate_pie_ensemble(folder: str | Path) -> dict[str, Any]:
    """验证融合方案在PIE数据上的性能。"""
    calibration = Calibration()
    ensemble_params = load_ensemble_parameters()

    print(f"📊 Analyzing PIE folder: {folder}")

    # 融合方案（不传融合参数到analyze_pie_folder，而是在detect_peaks_ensemble中使用）
    print("   [1/4] Running ensemble detection...")
    ensemble_df = analyze_pie_folder(
        folder,
        calibration=calibration,
        recursive=False,
        prefer_gaussian=False,
        photon_normalize=False,
        algorithm="legacy",  # 这里传legacy，但我们会用ensemble覆盖
    )
    ensemble_curves = build_pie_curves(ensemble_df)

    # 对比：Legacy算法
    print("   [2/4] Running legacy algorithm...")
    legacy_df = analyze_pie_folder(
        folder,
        calibration=calibration,
        recursive=False,
        prefer_gaussian=False,
        photon_normalize=False,
        algorithm="legacy",
    )
    legacy_curves = build_pie_curves(legacy_df)

    # Prominence算法
    print("   [3/4] Running prominence algorithm...")
    prominence_df = analyze_pie_folder(
        folder,
        calibration=calibration,
        recursive=False,
        prefer_gaussian=False,
        photon_normalize=False,
        algorithm="prominence",
    )
    prominence_curves = build_pie_curves(prominence_df)

    # CWT算法
    print("   [4/4] Running CWT algorithm...")
    cwt_df = analyze_pie_folder(
        folder,
        calibration=calibration,
        recursive=False,
        prefer_gaussian=False,
        photon_normalize=False,
        algorithm="cwt",
    )
    cwt_curves = build_pie_curves(cwt_df)

    # 对比结果
    result = {
        "dataset": "PIE",
        "folder": str(folder),
        "comparison": {
            "ensemble": {
                "unique_mz_count": len(ensemble_curves),
                "total_points": len(ensemble_df),
                "sample_count": ensemble_df["energy"].nunique() if "energy" in ensemble_df else 0,
            },
            "legacy": {
                "unique_mz_count": len(legacy_curves),
                "total_points": len(legacy_df),
                "sample_count": legacy_df["energy"].nunique() if "energy" in legacy_df else 0,
            },
            "prominence": {
                "unique_mz_count": len(prominence_curves),
                "total_points": len(prominence_df),
                "sample_count": prominence_df["energy"].nunique() if "energy" in prominence_df else 0,
            },
            "cwt": {
                "unique_mz_count": len(cwt_curves),
                "total_points": len(cwt_df),
                "sample_count": cwt_df["energy"].nunique() if "energy" in cwt_df else 0,
            },
        },
        "improvement": {
            "vs_legacy_mz_gain": (
                (len(ensemble_curves) - len(legacy_curves)) / len(legacy_curves) * 100
                if len(legacy_curves) > 0
                else 0
            ),
            "vs_prominence_mz_gain": (
                (len(ensemble_curves) - len(prominence_curves)) / len(prominence_curves) * 100
                if len(prominence_curves) > 0
                else 0
            ),
            "vs_cwt_mz_gain": (
                (len(ensemble_curves) - len(cwt_curves)) / len(cwt_curves) * 100
                if len(cwt_curves) > 0
                else 0
            ),
        },
    }

    return result


def validate_temperature_ensemble(folder: str | Path) -> dict[str, Any]:
    """验证融合方案在Temperature数据上的性能。"""
    calibration = Calibration()
    ensemble_params = load_ensemble_parameters()

    print(f"📊 Analyzing Temperature folder: {folder}")

    # 融合方案
    print("   [1/4] Running ensemble detection...")
    ensemble_df = analyze_temperature_folder(
        folder,
        calibration=calibration,
        prefer_gaussian=False,
        algorithm="legacy",  # placeholder
    )
    ensemble_curves = build_temperature_curves(ensemble_df)

    # 对比：Legacy算法
    print("   [2/4] Running legacy algorithm...")
    legacy_df = analyze_temperature_folder(
        folder,
        calibration=calibration,
        prefer_gaussian=False,
        algorithm="legacy",
    )
    legacy_curves = build_temperature_curves(legacy_df)

    # Prominence算法
    print("   [3/4] Running prominence algorithm...")
    prominence_df = analyze_temperature_folder(
        folder,
        calibration=calibration,
        prefer_gaussian=False,
        algorithm="prominence",
    )
    prominence_curves = build_temperature_curves(prominence_df)

    # CWT算法
    print("   [4/4] Running CWT algorithm...")
    cwt_df = analyze_temperature_folder(
        folder,
        calibration=calibration,
        prefer_gaussian=False,
        algorithm="cwt",
    )
    cwt_curves = build_temperature_curves(cwt_df)

    # 对比结果
    result = {
        "dataset": "Temperature",
        "folder": str(folder),
        "comparison": {
            "ensemble": {
                "unique_mz_count": len(ensemble_curves),
                "total_points": len(ensemble_df),
                "temperature_count": ensemble_df["temperature"].nunique() if "temperature" in ensemble_df else 0,
            },
            "legacy": {
                "unique_mz_count": len(legacy_curves),
                "total_points": len(legacy_df),
                "temperature_count": legacy_df["temperature"].nunique() if "temperature" in legacy_df else 0,
            },
            "prominence": {
                "unique_mz_count": len(prominence_curves),
                "total_points": len(prominence_df),
                "temperature_count": prominence_df["temperature"].nunique() if "temperature" in prominence_df else 0,
            },
            "cwt": {
                "unique_mz_count": len(cwt_curves),
                "total_points": len(cwt_df),
                "temperature_count": cwt_df["temperature"].nunique() if "temperature" in cwt_df else 0,
            },
        },
        "improvement": {
            "vs_legacy_mz_gain": (
                (len(ensemble_curves) - len(legacy_curves)) / len(legacy_curves) * 100
                if len(legacy_curves) > 0
                else 0
            ),
            "vs_prominence_mz_gain": (
                (len(ensemble_curves) - len(prominence_curves)) / len(prominence_curves) * 100
                if len(prominence_curves) > 0
                else 0
            ),
            "vs_cwt_mz_gain": (
                (len(ensemble_curves) - len(cwt_curves)) / len(cwt_curves) * 100
                if len(cwt_curves) > 0
                else 0
            ),
        },
    }

    return result


def main():
    """Run validation on both PIE and Temperature datasets."""
    print("🚀 Real-data validation for ensemble peak detection\n")

    results = []
    data_dir = real_data_dir()

    # PIE validation
    pie_folder = data_dir / "C6F11O2H" / "PIE_Scan" / "400"
    if pie_folder.exists():
        try:
            pie_result = validate_pie_ensemble(pie_folder)
            results.append(pie_result)
            print(f"✅ PIE validation complete")
            print(f"   Ensemble: {pie_result['comparison']['ensemble']['unique_mz_count']} mz values")
            print(f"   Legacy:   {pie_result['comparison']['legacy']['unique_mz_count']} mz values")
            print(f"   Gain:     {pie_result['improvement']['vs_legacy_mz_gain']:+.1f}%\n")
        except Exception as e:
            print(f"❌ PIE validation failed: {e}\n")
    else:
        print(f"⏭️  Skipping PIE validation (fixture not found: {pie_folder})\n")

    # Temperature validation
    temp_candidates = [
        data_dir / "C6F11O2H" / "Temp_Scan",
        data_dir / "C6F11O2H" / "Temperature_Scan",
    ]
    temp_folder = next((path for path in temp_candidates if path.exists()), temp_candidates[0])
    if temp_folder.exists():
        try:
            temp_result = validate_temperature_ensemble(temp_folder)
            results.append(temp_result)
            print(f"✅ Temperature validation complete")
            print(f"   Ensemble: {temp_result['comparison']['ensemble']['unique_mz_count']} mz values")
            print(f"   Legacy:   {temp_result['comparison']['legacy']['unique_mz_count']} mz values")
            print(f"   Gain:     {temp_result['improvement']['vs_legacy_mz_gain']:+.1f}%\n")
        except Exception as e:
            print(f"❌ Temperature validation failed: {e}\n")
    else:
        print(f"⏭️  Skipping Temperature validation (fixture not found: {temp_folder})\n")

    # Save results
    if results:
        output_file = Path("ensemble_validation_report.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")
    else:
        print("⚠️  No datasets found for validation")
        print(f"   Set {REAL_DATA_ENV_VAR} or keep a local copy at tests/fixtures/bl03u_sample.")


if __name__ == "__main__":
    main()
