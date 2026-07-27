from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .peak_detection import detect_peaks_by_algorithm
from .peak_sets import PeakSet, create_peak_set, verify_peak_set
from .project_settings import ProjectSettings
from .spectrum_io import list_spectrum_files, sum_spectra


ProgressCallback = Callable[[int, str], None]


def _report(callback: ProgressCallback | None, value: int, message: str) -> None:
    if callback is not None:
        callback(value, message)


def _folder_fingerprint(folder: Path, files: list[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_size = 0
    latest_mtime_ns = 0
    for path in files:
        stat = path.stat()
        relative = path.relative_to(folder).as_posix()
        total_size += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return {
        "exists": True,
        "kind": "folder",
        "file_count": len(files),
        "total_size": total_size,
        "latest_mtime_ns": latest_mtime_ns,
        "listing_sha256": digest.hexdigest(),
    }


def generate_project_peak_set(
    project_settings: ProjectSettings,
    *,
    progress_callback: ProgressCallback | None = None,
) -> PeakSet:
    """Generate and register a project-owned peak set from its accumulated spectrum.

    This function deliberately stops after creating and verifying the immutable
    peak-set record. Activating it in ``ProjectSettings`` remains a UI-thread
    transaction so a failed project-config save cannot start curve analysis.
    """
    project_dir = Path(project_settings.output_dir).expanduser().resolve()
    source_folder = Path(project_settings.sum_spectrum_folder).expanduser()
    if not source_folder.is_absolute():
        source_folder = project_dir / source_folder
    source_folder = source_folder.resolve()
    if not source_folder.is_dir():
        raise FileNotFoundError(
            "项目累计谱目录不存在。请先在质谱工作台登记累计谱，"
            "或在项目管理中导入卡峰文件。"
        )

    files = list_spectrum_files(source_folder, suffixes=(".txt",))
    if not files:
        raise ValueError(
            "项目累计谱目录中没有 .txt 质谱文件。请检查项目数据源后重试。"
        )

    _report(progress_callback, 10, f"正在读取 {len(files)} 个累计谱文件…")
    spectrum = sum_spectra(source_folder, suffixes=(".txt",), trim_start=4000)
    if len(spectrum.y) == 0:
        raise ValueError("项目累计谱没有可用于寻峰的数值数据。")

    time_offset = float(spectrum.x[0]) if len(spectrum.x) else 0.0
    peak_config = project_settings.to_peak_detection_config()
    _report(progress_callback, 55, "正在使用项目参数自动寻峰…")
    peaks = detect_peaks_by_algorithm(
        spectrum.y,
        calibration=project_settings.to_calibration(),
        start_idx=0,
        end_idx=len(spectrum.y),
        time_offset=time_offset,
        **peak_config.to_peak_kwargs(),
    )
    if not peaks:
        raise ValueError(
            "项目累计谱未检出有效峰，未创建卡峰文件。"
            "请检查累计谱、定标和项目寻峰参数。"
        )

    rows = [
        {
            "label": peak.species or "Unknown",
            "peak_index": int(round(float(peak.time))),
            "mz": float(peak.mz),
            "left_bound": int(round(float(peak.left_bound) + time_offset)),
            "right_bound": int(round(float(peak.right_bound) + time_offset)),
        }
        for peak in sorted(peaks, key=lambda item: (float(item.mz), float(item.time)))
    ]
    frame = pd.DataFrame(
        rows,
        columns=["label", "peak_index", "mz", "left_bound", "right_bound"],
    )
    csv_content = frame.to_csv(index=False).encode("utf-8-sig")
    source = {
        "scope": "project",
        "mode": "sum",
        "path": str(source_folder),
        "x_axis_mode": "mz",
        "time_offset": time_offset,
        "calibration": {
            "a": float(project_settings.cal_a),
            "b": float(project_settings.cal_b),
            "c": float(project_settings.cal_c),
        },
        "fingerprint": _folder_fingerprint(source_folder, files),
    }
    manifest = {
        "version": 2,
        "source": source,
        "created_from": {
            "tool": "project_analysis_preflight",
            "state": "automatic",
        },
        "peak_detection": peak_config.to_peak_kwargs(),
    }
    _report(progress_callback, 85, "正在写入并登记项目卡峰集…")
    record = create_peak_set(
        project_dir,
        content=csv_content,
        extension=".csv",
        label=(
            f"自动生成 · {len(rows)} 个峰 · "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        ),
        origin="auto_generated",
        manifest=manifest,
        metadata={
            "peak_count": len(rows),
            "source_mode": "sum",
            "source_path": str(source_folder),
        },
    )
    verify_peak_set(project_dir, record)
    _report(progress_callback, 100, "项目卡峰集已生成并校验")
    return record
