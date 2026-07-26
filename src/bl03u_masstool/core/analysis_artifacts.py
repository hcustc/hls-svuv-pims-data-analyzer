from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import sys
from typing import Any

import pandas as pd


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _series_values(values: Any) -> list[float]:
    parsed_values: list[float] = []
    for value in values:
        parsed = _json_float(value)
        if parsed is not None:
            parsed_values.append(parsed)
    return parsed_values


def _curve_payload(curve: dict[str, Any]) -> dict[str, Any]:
    exact_mz = float(curve.get("mz_exact_mean", curve.get("mz", 0)))
    nominal_mz = int(
        round(float(curve.get("mz_rounded", curve.get("mz", exact_mz))))
    )
    return {
        "mz": exact_mz,
        "mz_rounded": nominal_mz,
        "mz_exact_mean": exact_mz,
        "species": curve.get("species", ""),
        "energies": _series_values(curve.get("energies", [])),
        "intensities": _series_values(curve.get("intensities", [])),
        "temperatures": _series_values(curve.get("temperatures", [])),
        "areas": _series_values(curve.get("areas", [])),
        "curve_class": curve.get("curve_class"),
        "curve_class_label": curve.get("curve_class_label"),
        "curve_class_reason": curve.get("curve_class_reason"),
    }


def build_analysis_manifest(
    *,
    analysis_type: str,
    input_path: str | Path,
    parameters: dict[str, Any] | None = None,
    outputs: dict[str, str | Path] | None = None,
    analysis_df: pd.DataFrame | None = None,
    curves: dict[int | float, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row_count = 0 if analysis_df is None else int(len(analysis_df))
    curve_count = 0 if curves is None else int(len(curves))
    manifest: dict[str, Any] = {
        "analysis_type": analysis_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "parameters": parameters or {},
        "outputs": {key: str(value) for key, value in (outputs or {}).items() if value},
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "data_summary": {
            "row_count": row_count,
            "curve_count": curve_count,
        },
    }
    if analysis_df is not None and not analysis_df.empty:
        if "energy" in analysis_df:
            manifest["data_summary"]["energy_count"] = int(analysis_df["energy"].nunique())
        if "temperature" in analysis_df:
            manifest["data_summary"]["temperature_count"] = int(analysis_df["temperature"].nunique())
        if "file_count" in analysis_df:
            if "energy" in analysis_df:
                file_count = analysis_df.groupby("energy")["file_count"].first().sum()
            elif "temperature" in analysis_df:
                file_count = analysis_df.groupby("temperature")["file_count"].first().sum()
            else:
                file_count = analysis_df["file_count"].sum()
            manifest["data_summary"]["file_count"] = int(file_count)
    if curves:
        manifest["data_summary"]["mz_values"] = sorted(curves, key=float)
    return manifest


def score_pie_fit(fit: dict[str, Any] | None, *, point_count: int = 0) -> dict[str, Any]:
    warnings: list[str] = []
    if not fit:
        return {"confidence_level": "unfitted", "warnings": ["未进行 PICS 拟合"]}

    r_squared = _json_float(fit.get("r_squared"))
    candidate_count = int(fit.get("candidate_count") or len(fit.get("species", [])))
    if point_count < 3:
        warnings.append("有效能量点少于 3 个")
    if candidate_count == 0:
        warnings.append("无候选物种")
    if r_squared is None:
        warnings.append("缺少 R² 指标")
    elif r_squared < 0.8:
        warnings.append("PICS 拟合 R² 较低")
    elif r_squared < 0.95:
        warnings.append("PICS 拟合质量中等")

    contributions = [
        float(item.get("contribution_percent", 0.0))
        for item in fit.get("species", [])
        if item.get("contribution_percent") is not None
    ]
    if contributions and max(contributions) < 60:
        warnings.append("候选物种贡献较分散")

    if warnings:
        level = "low" if any("较低" in item or "少于" in item or "无候选" in item for item in warnings) else "medium"
    else:
        level = "high"
    return {"confidence_level": level, "warnings": warnings}


def build_pie_evidence_objects(
    curves: dict[int | float, dict[str, Any]],
    fits: dict[int | float, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fits = fits or {}
    objects: dict[str, Any] = {}
    for curve_key, curve in sorted(curves.items(), key=lambda item: float(item[0])):
        payload = _curve_payload(curve)
        fit = fits.get(curve_key)
        score = score_pie_fit(fit, point_count=len(payload["energies"]))
        exact_mz = float(curve.get("mz_exact_mean", curve_key))
        nominal_mz = int(
            round(float(curve.get("mz_rounded", curve.get("mz", exact_mz))))
        )
        objects[str(curve_key)] = {
            "analysis_type": "pie",
            "curve_key": curve_key,
            "mz": exact_mz,
            "mz_rounded": nominal_mz,
            "mz_exact_mean": exact_mz,
            "curve": {
                "energies": payload["energies"],
                "intensities": payload["intensities"],
                "point_count": len(payload["energies"]),
            },
            "candidate_count": int(fit.get("candidate_count") or len(fit.get("species", []))) if fit else 0,
            "fit": fit,
            "confidence_level": score["confidence_level"],
            "warnings": score["warnings"],
        }
    return objects


def build_temperature_evidence_objects(
    curves: dict[int | float, dict[str, Any]],
) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for curve_key, curve in sorted(
        curves.items(),
        key=lambda item: float(item[0]),
    ):
        payload = _curve_payload(curve)
        warnings = []
        if len(payload["temperatures"]) < 3:
            warnings.append("有效温度点少于 3 个")
        if payload.get("curve_class") == "unclassified":
            warnings.append("温度响应类型暂未区分")
        exact_mz = float(curve.get("mz_exact_mean", curve_key))
        nominal_mz = int(
            round(
                float(
                    curve.get(
                        "mz_rounded",
                        curve.get("mz", exact_mz),
                    )
                )
            )
        )
        objects[str(curve_key)] = {
            "analysis_type": "temperature",
            "curve_key": curve_key,
            "mz": exact_mz,
            "mz_rounded": nominal_mz,
            "mz_exact_mean": exact_mz,
            "curve": {
                "temperatures": payload["temperatures"],
                "areas": payload["areas"],
                "point_count": len(payload["temperatures"]),
            },
            "curve_class": payload.get("curve_class"),
            "curve_class_label": payload.get("curve_class_label"),
            "curve_class_reason": payload.get("curve_class_reason"),
            "confidence_level": "low" if warnings else "medium",
            "warnings": warnings,
        }
    return objects


def build_summary_report(
    manifest: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    summary = manifest.get("data_summary", {})
    lines = [
        f"# {manifest.get('analysis_type', 'analysis')} analysis report",
        "",
        "## 数据与参数概览",
        f"- 输入路径：`{manifest.get('input_path', '')}`",
        f"- 数据行数：{summary.get('row_count', 0)}",
        f"- 曲线数量：{summary.get('curve_count', 0)}",
    ]
    if "energy_count" in summary:
        lines.append(f"- 光子能量点数：{summary['energy_count']}")
    if "temperature_count" in summary:
        lines.append(f"- 温度点数：{summary['temperature_count']}")
    lines.extend(["", "## m/z 级证据摘要", "", "| m/z | 类型/拟合 | 置信度 | 警告 |", "| --- | --- | --- | --- |"])
    for mz, item in sorted(evidence.items(), key=lambda pair: float(pair[0])):
        if item.get("analysis_type") == "temperature":
            descriptor = item.get("curve_class_label") or item.get("curve_class") or "未分类"
        else:
            fit = item.get("fit") or {}
            r_squared = fit.get("r_squared")
            descriptor = "未拟合" if r_squared is None else f"R²={float(r_squared):.3f}"
        warnings = "；".join(item.get("warnings", [])) or "无"
        lines.append(f"| {mz} | {descriptor} | {item.get('confidence_level', '')} | {warnings} |")
    outputs = manifest.get("outputs", {})
    if outputs:
        lines.extend(["", "## 输出文件", ""])
        for key, value in outputs.items():
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def write_artifact_bundle(
    *,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    manifest_json: str | Path | None = None,
    evidence_json: str | Path | None = None,
    report_md: str | Path | None = None,
) -> dict[str, Path]:
    import json

    written: dict[str, Path] = {}
    if manifest_json:
        path = Path(manifest_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written["manifest_json"] = path
    if evidence_json:
        path = Path(evidence_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written["evidence_json"] = path
    if report_md:
        path = Path(report_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_summary_report(manifest, evidence), encoding="utf-8")
        written["report_md"] = path
    return written
