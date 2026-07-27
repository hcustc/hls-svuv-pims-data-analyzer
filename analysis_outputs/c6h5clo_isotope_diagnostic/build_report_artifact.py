from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_analysis import (
    DATA_ROOT,
    PEAK_FILE,
    PROJECT_CONFIG,
    CACHE_DIR,
    load_calibration,
    read_raw_spectra,
)


OUTPUT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = OUTPUT_DIR / "report_artifact.json"


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict("records")


def source(
    source_id: str,
    label: str,
    path: Path,
    description: str,
    *,
    filters: list[str],
    metric_definitions: list[str],
    tables_used: list[str],
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": str(path),
        "query": {
            "description": description,
            "engine": "Python",
            "language": "python",
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "filters": filters,
            "metric_definitions": metric_definitions,
            "tables_used": tables_used,
        },
    }


def inline_source(base: dict, sql: str, description: str) -> dict:
    value = deepcopy(base)
    value["query"]["sql"] = sql
    value["query"]["description"] = description
    return value


def build_artifact() -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = json.loads(
        (OUTPUT_DIR / "analysis_summary.json").read_text(encoding="utf-8")
    )
    alignment = pd.read_csv(OUTPUT_DIR / "window_alignment.csv")
    fixed = pd.read_csv(OUTPUT_DIR / "fixed_energy_ratio_summary.csv")
    ratios = pd.read_csv(OUTPUT_DIR / "ratio_by_energy.csv")
    shape = pd.read_csv(OUTPUT_DIR / "shape_tests.csv")
    scan_metadata = pd.read_csv(OUTPUT_DIR / "scan_metadata.csv")
    sensitivity = pd.read_csv(OUTPUT_DIR / "integration_window_sensitivity.csv")

    scan_summary = (
        scan_metadata.groupby("segment", as_index=False)
        .agg(
            files=("file", "size"),
            min_energy_eV=("energy", "min"),
            max_energy_eV=("energy", "max"),
            acquisition_time_s=("time_s", "first"),
            min_io_nA=("io_nA", "min"),
            max_io_nA=("io_nA", "max"),
            spectrum_bins=("n_bins", "first"),
        )
        .sort_values("min_energy_eV")
    )

    known_ratios = ratios[
        ratios["pair"].isin(["127/129", "99/101", "128/130"])
        & (ratios["segment"] == "PIE")
        & (ratios["energy_eV"] >= 9.0)
    ][
        [
            "energy_eV",
            "pair",
            "corrected_ratio",
            "old_window_ratio",
            "expected_ratio",
            "io_nA",
            "time_s",
        ]
    ].sort_values(["energy_eV", "pair"])
    target_ratios = ratios[
        ratios["pair"].isin(["146/144", "148/146"])
        & (ratios["segment"] == "PIE")
        & (ratios["energy_eV"] >= 9.0)
    ][
        [
            "energy_eV",
            "pair",
            "corrected_ratio",
            "old_window_ratio",
            "expected_ratio",
            "io_nA",
            "time_s",
        ]
    ].sort_values(["energy_eV", "pair"])

    spectra = read_raw_spectra()
    calibration = load_calibration()
    fixed_spectrum = min(
        (item for item in spectra if item["segment"] == "PIE"),
        key=lambda item: abs(item["energy"] - 11.0),
    )
    tof = np.arange(fixed_spectrum["y"].size, dtype=float)
    mz = np.asarray(calibration.tof_to_mz(tof), dtype=float)
    highres_mask = (mz >= 141.5) & (mz <= 148.5)
    highres = pd.DataFrame(
        {
            "tof_bin": tof[highres_mask].astype(int),
            "mz": mz[highres_mask],
            "raw_counts": fixed_spectrum["y"][highres_mask],
            "log10_counts": np.log10(
                np.maximum(fixed_spectrum["y"][highres_mask], 0.5)
            ),
            "energy_eV": float(fixed_spectrum["energy"]),
            "io_nA": float(fixed_spectrum["io_nA"]),
            "time_s": float(fixed_spectrum["time_s"]),
        }
    )
    highres_path = OUTPUT_DIR / "highres_11ev.csv"
    highres.to_csv(highres_path, index=False)

    source_analysis = source(
        "src_analysis",
        "Reproducible isotope diagnostic outputs",
        OUTPUT_DIR / "analysis_summary.json",
        "Recenter target peaks with the active project calibration, integrate matched windows with local baseline subtraction, and calculate isotope ratios.",
        filters=[
            "PIE segment only for 9.0–11.0 eV trend tests",
            "11.0 eV nearest acquired spectrum for fixed-energy checks",
            "±10 TOF-bin integration with ±6/8/10/12 sensitivity analysis",
        ],
        metric_definitions=[
            "Corrected intensity = local-baseline-subtracted peak area / IO / acquisition time.",
            "Known-pair diagnostic = lower-mass corrected peak area / higher-mass corrected peak area; expected ≈3.1.",
            "I146/I144 and I148/I146 expected ≈0.32 for pure singly chlorinated isotope relationships.",
        ],
        tables_used=[
            str(DATA_ROOT),
            str(PROJECT_CONFIG),
            str(PEAK_FILE),
            str(CACHE_DIR / "results.csv"),
        ],
    )
    source_raw = source(
        "src_raw_11ev",
        "Raw BL03U spectrum at 10.9998 eV",
        Path(fixed_spectrum["path"]),
        "Read the 10-line BL03U header and plot the unmodified detector counts against the active project TOF-to-m/z calibration.",
        filters=["m/z 141.5–148.5", "no smoothing", "no intensity normalization"],
        metric_definitions=[
            "raw_counts = detector counts in the source text file.",
            "log10_counts = log10(max(raw_counts, 0.5)) for visual dynamic range only.",
        ],
        tables_used=[str(fixed_spectrum["path"]), str(PROJECT_CONFIG)],
    )
    source_cache = source(
        "src_cache",
        "Existing PIE cache and provenance",
        CACHE_DIR / "results.csv",
        "Verify that the plotted cache uses one IO value and one file count for every target channel at each acquired spectrum, and that normalized_intensity equals raw_area / IO.",
        filters=[
            "active peak set peak-20260727T043641202752Z-b15932e2",
            "target channels 99, 101, 127, 128, 129, 130, 142, 144, 146, 148",
        ],
        metric_definitions=[
            "cached normalized_intensity = raw_area / IO because mass_discrimination = 1.0.",
        ],
        tables_used=[str(CACHE_DIR / "results.csv"), str(CACHE_DIR / "manifest.json")],
    )
    source_analysis["query"]["sql"] = (
        f"SELECT * FROM read_json_auto('{OUTPUT_DIR / 'analysis_summary.json'}')"
    )
    source_raw["query"]["sql"] = (
        f"SELECT * FROM read_csv_auto('{highres_path}') "
        "WHERE mz BETWEEN 141.5 AND 148.5 ORDER BY mz"
    )
    source_cache["query"]["sql"] = (
        f"SELECT * FROM read_csv_auto('{CACHE_DIR / 'results.csv'}')"
    )
    known_ratio_source = inline_source(
        source_analysis,
        (
            f"SELECT energy_eV, pair, corrected_ratio, old_window_ratio, "
            f"expected_ratio, io_nA, time_s FROM read_csv_auto("
            f"'{OUTPUT_DIR / 'ratio_by_energy.csv'}') "
            "WHERE segment = 'PIE' AND energy_eV >= 9.0 "
            "AND pair IN ('127/129','99/101','128/130') "
            "ORDER BY energy_eV, pair"
        ),
        "Select the high-signal corrected known-pair ratios used in the chart.",
    )
    target_ratio_source = inline_source(
        source_analysis,
        (
            f"SELECT energy_eV, pair, corrected_ratio, old_window_ratio, "
            f"expected_ratio, io_nA, time_s FROM read_csv_auto("
            f"'{OUTPUT_DIR / 'ratio_by_energy.csv'}') "
            "WHERE segment = 'PIE' AND energy_eV >= 9.0 "
            "AND pair IN ('146/144','148/146') "
            "ORDER BY energy_eV, pair"
        ),
        "Select the high-signal corrected target ratios used in the chart.",
    )
    alignment_source = inline_source(
        source_analysis,
        f"SELECT * FROM read_csv_auto('{OUTPUT_DIR / 'window_alignment.csv'}') ORDER BY channel",
        "Read the audited old and recentered target-window positions.",
    )
    fixed_ratio_source = inline_source(
        source_analysis,
        f"SELECT * FROM read_csv_auto('{OUTPUT_DIR / 'fixed_energy_ratio_summary.csv'}') ORDER BY pair",
        "Read fixed-energy and high-energy isotope-ratio summaries.",
    )
    shape_source = inline_source(
        source_analysis,
        f"SELECT * FROM read_csv_auto('{OUTPUT_DIR / 'shape_tests.csv'}') ORDER BY pair",
        "Read the PIE correlation and ratio-vs-energy regression diagnostics.",
    )
    scan_source = inline_source(
        source_cache,
        (
            f"SELECT segment, count(*) AS files, min(energy) AS min_energy_eV, "
            f"max(energy) AS max_energy_eV, first(time_s) AS acquisition_time_s, "
            f"min(io_nA) AS min_io_nA, max(io_nA) AS max_io_nA, "
            f"first(n_bins) AS spectrum_bins FROM read_csv_auto("
            f"'{OUTPUT_DIR / 'scan_metadata.csv'}') GROUP BY segment "
            "ORDER BY min_energy_eV"
        ),
        "Summarize file counts, energy coverage, IO range, acquisition time, and spectrum length by scan segment.",
    )

    sensitivity_ranges = (
        sensitivity.groupby("pair")["ratio_at_11eV"]
        .agg(["min", "max"])
        .round(3)
        .to_dict("index")
    )
    fixed_lookup = fixed.set_index("pair")
    shape_lookup = shape.set_index("pair")
    window_summary = summary["window_alignment"]
    overlap_summary = summary["segment_overlap"]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "PIE 同位素与处理流程诊断",
        "description": "BL03U C6H5ClO 原始 PIE 扫描的峰窗、同位素比与段间归一化诊断。",
        "generatedAt": generated_at,
        "sources": [source_analysis, source_raw, source_cache],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# PIE 同位素与处理流程诊断",
            },
            {
                "id": "technical-summary",
                "type": "markdown",
                "sourceId": "src_analysis",
                "body": (
                    "## 技术摘要\n\n"
                    f"- **采集口径一致，但积分对象失配。** 142、144、146 来自同一批 84 张原始谱；"
                    f"缓存逐谱共享同一 IO、文件数和单位。可是 10/10 个目标峰都不在旧积分窗内，"
                    f"TOF 偏移为 {window_summary['min_tof_offset_bins']}–"
                    f"{window_summary['max_tof_offset_bins']} bin（中位数 "
                    f"{window_summary['median_tof_offset_bins']:.0f}）。\n"
                    f"- **已知同位素对没有恢复 3.1:1。** 11 eV 重定位后，"
                    f"127/129={fixed_lookup.loc['127/129','corrected_ratio']:.3f}、"
                    f"99/101={fixed_lookup.loc['99/101','corrected_ratio']:.3f}、"
                    f"128/130={fixed_lookup.loc['128/130','corrected_ratio']:.3f}。\n"
                    f"- **146 不能仅凭当前曲线认作新的单氯母体。** 11 eV 时 "
                    f"I146/I144={fixed_lookup.loc['146/144','corrected_ratio']:.3f}，"
                    f"I148/I146={fixed_lookup.loc['148/146','corrected_ratio']:.3f}；"
                    "二者都偏离 0.32，并随光子能量显著变化。\n"
                    f"- **两段扫描不能直接拼接。** 低能段为 90 s/点，高能段为 60 s/点；"
                    f"在 8.00–8.10 eV 重叠区，仅除 IO 后的低能段/高能段中位数仍为 "
                    f"{overlap_summary['median_PIE2_over_PIE_after_IO_only']:.3f}。"
                ),
            },
            {
                "id": "same-acquisition",
                "type": "markdown",
                "sourceId": "src_cache",
                "body": (
                    "## 142、144、146 的光通量与纵轴单位相同\n\n"
                    "**每一个能量点，这三个通道都取自同一张原始 TOF 谱，因而共享 IO、采集时间和扫描文件数。** "
                    "缓存中的纵轴计算严格等于峰面积除以 IO，质量歧视因子为 1.0；三张截图的不同 y 轴上限来自各图独立自动缩放，"
                    "不是把每条曲线分别归一化到最大值。需要注意的是，整个 7–11 eV 范围跨越两套采集时长，"
                    "当前缓存没有除以 Time，也没有消除重叠区显示出的额外段间增益差。"
                ),
            },
            {"id": "scan-table-block", "type": "table", "tableId": "scan-summary"},
            {
                "id": "alignment-result",
                "type": "markdown",
                "sourceId": "src_analysis",
                "body": (
                    "## 旧手动峰窗整体落在目标峰左侧\n\n"
                    "**活动峰表保存的 m/z 标签仍然正确，但 TOF 中心没有按当前项目标定重算。** "
                    "例如 142.06 的旧中心为 18791，而当前标定预测 18839、聚合原始谱峰顶为 18838；"
                    "144.02、146.02、148.01 也分别错开约 45–50 bin。"
                    "因此当前三张 PIE 曲线主要积分了邻峰、峰尾或本底，不能用来做物种归属。"
                ),
            },
            {"id": "alignment-table-block", "type": "table", "tableId": "alignment"},
            {
                "id": "fixed-spectrum-result",
                "type": "markdown",
                "sourceId": "src_raw_11ev",
                "body": (
                    "## 11 eV 原始高分辨谱确认 142/144/146/148 都存在独立峰\n\n"
                    "下图直接显示 10.9998 eV 的未平滑原始计数，并仅对纵轴取 log10 以容纳动态范围。"
                    "峰顶位于约 m/z 142.04、144.04、146.00、147.95；这一固定能量切片证明峰位本身清楚，"
                    "问题出在跨数据集复用的 TOF 积分窗，而不是 PIE 曲线绘图。"
                ),
            },
            {"id": "highres-chart-block", "type": "chart", "chartId": "highres-11ev"},
            {
                "id": "known-ratio-result",
                "type": "markdown",
                "sourceId": "src_analysis",
                "body": (
                    "## 已知同位素对未通过 3.1:1 流程诊断\n\n"
                    f"在 9–11 eV 高信号区，127/129 的中位数为 "
                    f"{fixed_lookup.loc['127/129','high_energy_median']:.3f}，"
                    f"99/101 为 {fixed_lookup.loc['99/101','high_energy_median']:.3f}，"
                    f"128/130 为 {fixed_lookup.loc['128/130','high_energy_median']:.3f}；"
                    "均不接近 3.1。固定能量谱中 m/z 129 又处在强 m/z 128 的同位素/邻峰复杂区，"
                    "所以失败更像是未分辨叠加或响应偏差，而不是一个简单的统一比例系数。"
                ),
            },
            {"id": "known-chart-block", "type": "chart", "chartId": "known-ratios"},
            {"id": "fixed-ratio-table-block", "type": "table", "tableId": "fixed-ratios"},
            {
                "id": "target-ratio-result",
                "type": "markdown",
                "sourceId": "src_analysis",
                "body": (
                    "## 146/144 与 148/146 的比值随能量显著变化\n\n"
                    f"9–11 eV 内，I146/I144 的最佳过原点比例为 "
                    f"{shape_lookup.loc['146/144','best_scale_through_origin']:.3f}，"
                    f"比值随能量的斜率为 "
                    f"{shape_lookup.loc['146/144','ratio_vs_energy_slope_per_eV']:.3f}/eV "
                    f"(p={shape_lookup.loc['146/144','ratio_vs_energy_p']:.1e})；"
                    f"I148/I146 的最佳比例为 "
                    f"{shape_lookup.loc['148/146','best_scale_through_origin']:.3f}，"
                    f"斜率为 {shape_lookup.loc['148/146','ratio_vs_energy_slope_per_eV']:.3f}/eV "
                    f"(p={shape_lookup.loc['148/146','ratio_vs_energy_p']:.1e})。"
                    "两组 PIE 强度本身高度相关，是因为都随能量上升；但常数比例检验不成立。"
                ),
            },
            {"id": "target-chart-block", "type": "chart", "chartId": "target-ratios"},
            {"id": "shape-table-block", "type": "table", "tableId": "shape-tests"},
            {
                "id": "scope-definitions",
                "type": "markdown",
                "sourceId": "src_analysis",
                "body": (
                    "## 范围、数据与指标定义\n\n"
                    "数据包含 PIE-2 的 23 张谱（6.9998–8.1000 eV，90 s/点）与 PIE 的 61 张谱"
                    "（7.9998–10.9998 eV，60 s/点），每张谱均有 29,984 个 TOF bin。"
                    "“校正强度”定义为局部线性侧带本底扣除后的峰面积，再除以 IO 与采集时间；"
                    "同一固定能量下的同位素比在除法中抵消这两个共同因子。"
                ),
            },
            {
                "id": "methods-robustness",
                "type": "markdown",
                "sourceId": "src_analysis",
                "body": (
                    "## 方法与稳健性\n\n"
                    "目标中心先由项目二次标定求得，再在 9.5–11 eV 聚合谱的 ±7 bin 内定位峰顶。"
                    "主分析使用 ±10 bin 对称积分窗，并用两侧 18–30 bin 的中位数线性插值作为本底。"
                    "把半窗宽改成 6、8、10、12 bin 后，11 eV 的 I146/I144 为 "
                    f"{sensitivity_ranges['146/144']['min']:.3f}–"
                    f"{sensitivity_ranges['146/144']['max']:.3f}，"
                    "I148/I146 为 "
                    f"{sensitivity_ranges['148/146']['min']:.3f}–"
                    f"{sensitivity_ranges['148/146']['max']:.3f}；"
                    "结论对积分宽度不敏感。"
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 限制与不确定性\n\n"
                    "- 目录中没有单独标记的 blank/background 谱，因此这里无法验证化学本底扣除。\n"
                    "- 高相关的单调 PIE 曲线不等价于同位素对应；常数比例和已知对照更有诊断力。\n"
                    "- 本分析确认了窗口错位和比例失败，但未把 m/z 129 等复杂峰拆成具体分子式贡献；"
                    "“多物种/同量异位素叠加”是由原始谱形和能量依赖推断出的最可能解释，而不是唯一解释。\n"
                    "- 两段扫描重叠区只含 3 个能量点，足以否定直接拼接，但不足以建立高精度能量依赖校正函数。"
                ),
            },
            {
                "id": "next-steps",
                "type": "markdown",
                "sourceId": "src_analysis",
                "body": (
                    "## 建议的下一步\n\n"
                    "1. 用 PIE 扫描自身的参考谱重新生成 peak set，或在每次校准变更后由 m/z 反算 TOF 并重定位窗口。\n"
                    "2. 在 PIE 流程中把 `raw_area / IO` 改为 `raw_area / IO / Time`，并把采集时间写入曲线元数据。\n"
                    "3. 用 8.00–8.10 eV 重叠区对两段做显式比例校准；至少用 128、130、142、144 多个高信噪比通道检查比例是否一致。\n"
                    "4. 把 127/129、99/101、128/130 的 3.1:1 恢复设为处理流程的自动化验收测试；未通过时阻止物种拟合/归属。\n"
                    "5. 在这些检查通过前，将 146 标记为“多物种候选/待确认”，不要以 148 的当前曲线作为单氯母体证据。"
                ),
            },
            {
                "id": "further-questions",
                "type": "markdown",
                "body": (
                    "## 进一步问题\n\n"
                    "- 手动峰表是在何时、基于哪一批原始谱和哪组标定系数生成的？\n"
                    "- 8 eV 两段重叠扫描之间是否更换了样品流量、透镜电压、MCP 增益或触发/累加设置？\n"
                    "- 是否有同步 blank、Kr 或其他单氯标准可用于拆分化学本底与质量依赖响应？"
                ),
            },
        ],
        "charts": [
            {
                "id": "highres-11ev",
                "title": "Raw high-resolution spectrum at 10.9998 eV",
                "description": "m/z 141.5–148.5; y-axis is log10 of raw counts for display.",
                "type": "line",
                "intent": "trend",
                "question": "Where are the 142, 144, 146, and 148 peaks in one fixed-energy raw spectrum?",
                "rationale": "A line over calibrated m/z preserves the high-resolution peak positions and relative raw-count scale.",
                "dataset": "highres_11ev",
                "encodings": {
                    "x": {"field": "mz", "type": "quantitative", "title": "m/z"},
                    "y": {
                        "field": "log10_counts",
                        "type": "quantitative",
                        "title": "log10 raw counts",
                    },
                },
                "palette": {"kind": "sequential", "name": "blue"},
                "legend": {"position": "bottom"},
                "source": source_raw,
            },
            {
                "id": "known-ratios",
                "title": "Known chlorine-isotope ratios vs photon energy",
                "description": "9.0–11.0 eV; a valid base/+2 pair should remain near 3.1.",
                "type": "line",
                "intent": "trend",
                "question": "Do the known 127/129, 99/101, and 128/130 pairs recover the expected 3.1 ratio?",
                "rationale": "A grouped line chart shows both absolute deviation from 3.1 and energy-dependent instability.",
                "dataset": "known_ratio_energy",
                "encodings": {
                    "x": {
                        "field": "energy_eV",
                        "type": "quantitative",
                        "title": "Photon energy / eV",
                    },
                    "y": {
                        "field": "corrected_ratio",
                        "type": "quantitative",
                        "title": "Lower-mass / higher-mass intensity",
                    },
                    "color": {"field": "pair", "type": "nominal", "title": "Pair"},
                },
                "palette": {"kind": "categorical", "name": "diagnostic"},
                "legend": {"position": "bottom", "title": "Pair"},
                "labels": {"values": "none"},
                "referenceLines": [
                    {
                        "axis": "y",
                        "value": 3.1,
                        "label": "Expected 3.1",
                        "lineStyle": "dashed",
                        "color": "neutral",
                    }
                ],
                "source": known_ratio_source,
            },
            {
                "id": "target-ratios",
                "title": "I146/I144 and I148/I146 vs photon energy",
                "description": "9.0–11.0 eV; pure singly chlorinated isotope relationships should remain near 0.32.",
                "type": "line",
                "intent": "trend",
                "question": "Are the 144/146 and 146/148 relationships constant at the expected 0.32 scale?",
                "rationale": "A grouped line chart directly tests constancy and exposes systematic energy dependence.",
                "dataset": "target_ratio_energy",
                "encodings": {
                    "x": {
                        "field": "energy_eV",
                        "type": "quantitative",
                        "title": "Photon energy / eV",
                    },
                    "y": {
                        "field": "corrected_ratio",
                        "type": "quantitative",
                        "title": "Intensity ratio",
                    },
                    "color": {"field": "pair", "type": "nominal", "title": "Ratio"},
                },
                "palette": {"kind": "categorical", "name": "diagnostic"},
                "legend": {"position": "bottom", "title": "Ratio"},
                "labels": {"values": "none"},
                "referenceLines": [
                    {
                        "axis": "y",
                        "value": 0.32,
                        "label": "Expected 0.32",
                        "lineStyle": "dashed",
                        "color": "neutral",
                    }
                ],
                "source": target_ratio_source,
            },
        ],
        "tables": [
            {
                "id": "scan-summary",
                "title": "Scan segments and acquisition conditions",
                "description": "Both segments contain one spectrum per energy, but use different acquisition times.",
                "dataset": "scan_summary",
                "columns": [
                    {"field": "segment", "label": "Segment", "type": "text"},
                    {"field": "files", "label": "Files", "type": "number"},
                    {"field": "min_energy_eV", "label": "Min eV", "type": "number"},
                    {"field": "max_energy_eV", "label": "Max eV", "type": "number"},
                    {"field": "acquisition_time_s", "label": "Time / s", "type": "number"},
                    {"field": "min_io_nA", "label": "Min IO / nA", "type": "number"},
                    {"field": "max_io_nA", "label": "Max IO / nA", "type": "number"},
                    {"field": "spectrum_bins", "label": "TOF bins", "type": "number"},
                ],
                "defaultSort": {"field": "min_energy_eV", "direction": "asc"},
                "source": scan_source,
            },
            {
                "id": "alignment",
                "title": "Target-window alignment audit",
                "description": "All ten old windows miss the current-calibration peak center.",
                "dataset": "alignment",
                "columns": [
                    {"field": "channel", "label": "Channel", "type": "number"},
                    {"field": "label_mz", "label": "Stored m/z", "type": "number"},
                    {"field": "old_center_tof", "label": "Old center", "type": "number"},
                    {
                        "field": "calibrated_center_tof",
                        "label": "Calibrated center",
                        "type": "number",
                    },
                    {
                        "field": "observed_center_tof",
                        "label": "Observed peak",
                        "type": "number",
                    },
                    {"field": "tof_offset_bins", "label": "Offset / bin", "type": "number"},
                    {
                        "field": "old_window_contains_peak",
                        "label": "Old window contains peak",
                        "type": "boolean",
                    },
                ],
                "defaultSort": {"field": "channel", "direction": "asc"},
                "source": alignment_source,
            },
            {
                "id": "fixed-ratios",
                "title": "Fixed-energy isotope-ratio checks",
                "description": "Nearest acquired point to 11.0 eV; corrected values use recentered, baseline-subtracted windows.",
                "dataset": "fixed_ratios",
                "columns": [
                    {"field": "pair", "label": "Pair", "type": "text"},
                    {"field": "expected_ratio", "label": "Expected", "type": "number"},
                    {"field": "old_window_ratio", "label": "Old window", "type": "number"},
                    {"field": "corrected_ratio", "label": "Corrected", "type": "number"},
                    {
                        "field": "high_energy_median",
                        "label": "9–11 eV median",
                        "type": "number",
                    },
                    {"field": "high_energy_q25", "label": "Q25", "type": "number"},
                    {"field": "high_energy_q75", "label": "Q75", "type": "number"},
                ],
                "defaultSort": {"field": "pair", "direction": "asc"},
                "source": fixed_ratio_source,
            },
            {
                "id": "shape-tests",
                "title": "PIE shape and ratio-constancy tests",
                "description": "High correlation alone does not satisfy a constant isotope ratio.",
                "dataset": "shape_tests",
                "columns": [
                    {"field": "pair", "label": "Pair", "type": "text"},
                    {
                        "field": "pearson_r_of_PIE_intensities",
                        "label": "Pearson r",
                        "type": "number",
                    },
                    {
                        "field": "best_scale_through_origin",
                        "label": "Best constant scale",
                        "type": "number",
                    },
                    {
                        "field": "r2_through_origin",
                        "label": "R² through origin",
                        "type": "number",
                    },
                    {
                        "field": "ratio_vs_energy_slope_per_eV",
                        "label": "Ratio slope / eV",
                        "type": "number",
                    },
                    {
                        "field": "ratio_vs_energy_p",
                        "label": "Slope p-value",
                        "type": "number",
                    },
                ],
                "defaultSort": {"field": "pair", "direction": "asc"},
                "source": shape_source,
            },
        ],
    }

    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": {
            "scan_summary": records(scan_summary),
            "alignment": records(alignment),
            "fixed_ratios": records(fixed),
            "shape_tests": records(shape),
            "known_ratio_energy": records(known_ratios),
            "target_ratio_energy": records(target_ratios),
            "highres_11ev": records(highres),
        },
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
    }
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact


if __name__ == "__main__":
    artifact = build_artifact()
    print(ARTIFACT_PATH)
    print(
        {
            dataset: len(rows)
            for dataset, rows in artifact["snapshot"]["datasets"].items()
        }
    )
