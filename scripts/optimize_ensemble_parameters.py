#!/usr/bin/env python
"""Generate optimized ensemble peak detection parameters.

This script runs Bayesian parameter optimization on synthetic peaks
and outputs the best parameters for production use.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.calibration import Calibration
from core.peak_benchmark import build_default_synthetic_benchmark, read_benchmark_background
from core.ensemble_optimization import (
    optimize_ensemble_parameters,
    extract_ensemble_parameters,
)


def main():
    """Run optimization and save results."""
    print("🚀 Starting ensemble peak detection parameter optimization...")

    # Load calibration and synthetic benchmark
    calibration = Calibration()
    background_path = Path("tests/fixtures/C6F11O2H/PIE_Scan/400/12.9eV-60s-400C-空白谱.asc")

    if not background_path.exists():
        print(f"❌ Background file not found: {background_path}")
        sys.exit(1)

    print(f"📊 Loading background from {background_path}")
    background = read_benchmark_background(background_path)

    print("🧪 Building synthetic peak cases...")
    cases = build_default_synthetic_benchmark(background, calibration=calibration)
    print(f"   Created {len(cases)} benchmark cases")

    # Run optimization
    print("\n🔍 Running Bayesian parameter optimization...")
    print("   Trials: 20 | Initial random: 6 | Quick mode: ON")

    result = optimize_ensemble_parameters(
        cases=cases,
        calibration=calibration,
        n_trials=20,
        initial_random=6,
        random_state=42,
        method="bayesian",
        quick_mode=True,
    )

    print(f"\n✅ Optimization complete!")
    print(f"   Best score: {result.best_score:.2f}")
    print(f"   Total trials: {len(result.trials)}")

    # Extract and display parameters
    best_params = extract_ensemble_parameters(result)

    print("\n📋 Best parameters found:")
    print(f"   vote_threshold: {best_params['vote_threshold']:.4f}")
    print(f"   min_intensity_for_single_vote: {best_params['min_intensity_for_single_vote']:.2f}")
    print(f"   mz_tolerance: {best_params['mz_tolerance']:.4f}")
    print(f"   legacy_min_intensity: {best_params['min_intensity']:.2f}")
    print(f"   legacy_weak_tail_ratio: {best_params['weak_tail_ratio']:.2f}")
    print(f"   prominence_ratio: {best_params['prominence_ratio']:.6f}")
    print(f"   cwt_snr_threshold: {best_params['cwt_snr_threshold']:.6f}")

    # Save results
    output_file = Path("ensemble_optimization_result.json")
    output_data = {
        "best_score": float(result.best_score),
        "best_parameters": best_params,
        "all_trials": [
            {
                "trial": t.trial,
                "score": float(t.score),
                "strategy": t.strategy,
                "parameters": {k: float(v) if isinstance(v, (int, float)) else v
                              for k, v in t.parameters.items()},
            }
            for t in result.trials
        ],
        "search_space": [
            {"name": spec.name, "low": float(spec.low), "high": float(spec.high), "kind": spec.kind}
            for spec in result.search_space
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")

    # Trial summary
    print(f"\n📈 Trial history:")
    print(f"   {'Trial':<6} {'Strategy':<15} {'Score':<8}")
    print(f"   {'-'*30}")
    for trial in result.trials[:10]:  # Show first 10
        print(f"   {trial.trial:<6} {trial.strategy:<15} {trial.score:<8.2f}")
    if len(result.trials) > 10:
        print(f"   ... ({len(result.trials) - 10} more trials)")

    # Metrics summary
    best_metrics = result.best_metrics
    if "case_results" in best_metrics:
        print(f"\n🎯 Performance on benchmark cases:")
        for case_result in best_metrics["case_results"]:
            print(f"   {case_result['case']:<20} "
                  f"Recall: {case_result.get('recall', 0):.1%} "
                  f"TP: {case_result['true_positive_count']}/{case_result['truth_count']} "
                  f"FP: {case_result['false_positive_count']}")

    print("\n✨ Ready for production use!")
    print(f"   Use the parameters from {output_file}")


if __name__ == "__main__":
    main()
