from __future__ import annotations

import pandas as pd

from bl03u_masstool.core.analysis_artifacts import (
    build_analysis_manifest,
    build_pie_evidence_objects,
    build_summary_report,
    build_temperature_evidence_objects,
    score_pie_fit,
)


def test_build_manifest_records_data_summary():
    df = pd.DataFrame(
        {
            "energy": [11.0, 12.0],
            "file_count": [1, 1],
            "mz_rounded": [18, 18],
        }
    )
    curves = {18: {"mz": 18, "energies": [11.0, 12.0], "intensities": [1.0, 2.0]}}

    manifest = build_analysis_manifest(
        analysis_type="pie",
        input_path="input",
        parameters={"photon_mode": "off"},
        analysis_df=df,
        curves=curves,
    )

    assert manifest["analysis_type"] == "pie"
    assert manifest["data_summary"]["row_count"] == 2
    assert manifest["data_summary"]["curve_count"] == 1
    assert manifest["data_summary"]["energy_count"] == 2
    assert manifest["data_summary"]["mz_values"] == [18]


def test_build_manifest_file_count_uses_named_axis():
    df = pd.DataFrame(
        {
            "mz_rounded": [18, 18],
            "energy": [11.0, 12.0],
            "file_count": [1, 1],
        }
    )
    manifest = build_analysis_manifest(
        analysis_type="pie",
        input_path="input",
        analysis_df=df,
    )
    assert manifest["data_summary"]["file_count"] == 2


def test_pie_evidence_scores_fit_quality():
    curves = {18: {"mz": 18, "energies": [11.0, 12.0, 13.0], "intensities": [1.0, 2.0, 3.0]}}
    fits = {
        18: {
            "r_squared": 0.98,
            "candidate_count": 1,
            "species": [{"species": "Water", "contribution_percent": 100.0}],
        }
    }

    evidence = build_pie_evidence_objects(curves, fits)

    assert evidence["18"]["confidence_level"] == "high"
    assert evidence["18"]["candidate_count"] == 1
    assert score_pie_fit(fits[18], point_count=3)["warnings"] == []


def test_pie_evidence_filters_non_finite_curve_values():
    curves = {
        18: {
            "mz": 18,
            "energies": [11.0, float("nan"), float("inf"), 12.0],
            "intensities": [1.0, float("-inf"), 2.0],
        }
    }
    evidence = build_pie_evidence_objects(curves)
    assert evidence["18"]["curve"]["energies"] == [11.0, 12.0]
    assert evidence["18"]["curve"]["intensities"] == [1.0, 2.0]


def test_temperature_evidence_reports_curve_class():
    curves = {
        22: {
            "mz": 22,
            "temperatures": [400.0, 500.0, 600.0],
            "areas": [1.0, 2.0, 1.0],
            "curve_class": "intermediate",
            "curve_class_label": "中间体(先升后降低)",
            "curve_class_reason": "中间温度信号最高且两端明显降低",
        }
    }

    evidence = build_temperature_evidence_objects(curves)
    report = build_summary_report(
        {"analysis_type": "temperature", "input_path": "input", "data_summary": {"row_count": 3, "curve_count": 1}},
        evidence,
    )

    assert evidence["22"]["curve_class"] == "intermediate"
    assert "中间体" in report
