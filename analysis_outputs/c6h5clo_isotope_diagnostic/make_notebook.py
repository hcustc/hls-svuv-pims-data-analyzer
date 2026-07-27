from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"
NOTEBOOK_PATH = OUTPUT_DIR / "c6h5clo_isotope_diagnostic.ipynb"


def ratio(summary: dict, pair: str) -> dict:
    return next(
        row for row in summary["fixed_energy_ratios"] if row["pair"] == pair
    )


def build_notebook() -> Path:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    pair_127 = ratio(summary, "127/129")
    pair_99 = ratio(summary, "99/101")
    pair_128 = ratio(summary, "128/130")
    pair_146 = ratio(summary, "146/144")
    pair_148 = ratio(summary, "148/146")
    alignment = summary["window_alignment"]
    overlap = summary["segment_overlap"]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# C6H5ClO PIE 同位素与处理流程诊断\n\n"
            "基于原始 PIE 高分辨质谱、项目标定、活动手动峰表和已生成 PIE 缓存。"
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"- **当前 142/144/146 图使用相同的原始谱、IO 和纵轴单位，但旧积分窗已经失配。**"
            f" 10/10 个目标峰均不在旧窗口内；TOF 偏移为 "
            f"{alignment['min_tof_offset_bins']}–{alignment['max_tof_offset_bins']} bin"
            f"（中位数 {alignment['median_tof_offset_bins']:.0f}）。\n"
            "- **已知氯同位素对未恢复 3.1:1。** 11 eV 重定位后，"
            f"127/129={pair_127['corrected_ratio']:.3f}、"
            f"99/101={pair_99['corrected_ratio']:.3f}、"
            f"128/130={pair_128['corrected_ratio']:.3f}。\n"
            "- **144/146 不能视为纯同位素对。** 11 eV 时 "
            f"I146/I144={pair_146['corrected_ratio']:.3f}，"
            f"9–11 eV 中位数为 {pair_146['high_energy_median']:.3f}，且随能量显著变化。\n"
            "- **148 不支持“146 为新的单氯母体”的充分证据。** 11 eV 时 "
            f"I148/I146={pair_148['corrected_ratio']:.3f}，"
            f"9–11 eV 中位数仅 {pair_148['high_energy_median']:.3f}，明显低于 0.32。\n"
            "- **两段扫描不能直接拼接。** 低能段为 90 s/点，高能段为 60 s/点；"
            f"在 8.00–8.10 eV 重叠区，低能段/高能段在仅除 IO 后的中位数仍只有 "
            f"{overlap['median_PIE2_over_PIE_after_IO_only']:.3f}，说明还存在额外段间增益/源强差。"
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### Key Assumptions\n\n"
            "- 氯天然丰度诊断采用 \\(^{35}\\mathrm{Cl}/^{37}\\mathrm{Cl}\\approx3.1\\)，"
            "等价于 \\(I_{M+2}/I_M\\approx0.32\\)。\n"
            "- 同一原始谱内的质量比不受 IO 或采集时间的共同乘法因子影响。\n"
            "- 目标峰位按项目当前二次 TOF→m/z 标定计算，并在 ±7 bin 内用 9.5–11 eV"
            " 聚合谱定位峰顶；随后用对称 ±10 bin、线性侧带本底进行积分。\n"
            "- 使用 ±6、±8、±10、±12 bin 做积分窗敏感性检查。"
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "from pathlib import Path\n"
            "os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/bl03u_mpl')\n"
            "os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/bl03u_cache')\n"
            "from run_analysis import main\n\n"
            "summary = main()\n"
            "OUTPUT_DIR = Path.cwd()\n"
            "print('analysis complete:', OUTPUT_DIR)"
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            "import pandas as pd\n\n"
            "scan_metadata = pd.read_csv('scan_metadata.csv')\n"
            "alignment = pd.read_csv('window_alignment.csv')\n"
            "fixed_ratios = pd.read_csv('fixed_energy_ratio_summary.csv')\n"
            "shape_tests = pd.read_csv('shape_tests.csv')\n"
            "sensitivity = pd.read_csv('integration_window_sensitivity.csv')\n\n"
            "scan_metadata.groupby('segment').agg(\n"
            "    files=('file', 'size'),\n"
            "    energy_min=('energy', 'min'),\n"
            "    energy_max=('energy', 'max'),\n"
            "    time_s=('time_s', 'first'),\n"
            "    io_min=('io_nA', 'min'),\n"
            "    io_max=('io_nA', 'max'),\n"
            ")"
        ),
        nbf.v4.new_code_cell(
            "alignment[[\n"
            "    'channel', 'label_mz', 'old_center_tof', 'calibrated_center_tof',\n"
            "    'observed_center_tof', 'tof_offset_bins', 'old_window_contains_peak'\n"
            "]]"
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n\n"
            "display(Image(filename='fixed_energy_high_resolution_spectrum.png'))"
        ),
        nbf.v4.new_code_cell(
            "display(fixed_ratios)\n"
            "display(shape_tests)\n"
            "display(sensitivity.pivot(index='pair', columns='half_width_bins', values='ratio_at_11eV'))"
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename='scan_conditions.png'))\n"
            "display(Image(filename='known_isotope_ratios.png'))\n"
            "display(Image(filename='target_relationships.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. 在解释 142/144/146/148 的 PIE 形状之前，必须先用 PIE 数据自身重建或重定位峰窗；"
            "当前手动峰表不能继续直接使用。\n"
            "2. 每个能量点的强度应至少除以 IO 和采集时间；两段扫描还需要用 8.00–8.10 eV"
            " 重叠区估计额外段间比例因子，且应按高信噪比通道分别验证。\n"
            "3. 已知同位素对仍远离 3.1:1，说明窗口修复后仍有未分辨同量异位素/多物种叠加、"
            "邻峰尾部或质量依赖响应问题；在这些诊断通过前，不宜对 146 赋予新母体身份。\n"
            "4. 148 与 146 的 PIE 总体相关，但最佳比例约 0.18 而非 0.32，且比值随能量增加；"
            "“共同上升”不足以证明同位素对应关系。"
        ),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build_notebook())
