from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from .calibration import Calibration
from .peak_detection import Peak


@dataclass(frozen=True)
class PeakRange:
    mz: float
    peak_index: int
    left_bound: int
    right_bound: int
    label: str = "Manual"


COLUMN_ALIASES = {
    "mz": {"mz", "m/z", "mass", "mass_number", "质量数", "质量数 (m/z)", "质量数(m/z)", "质荷比"},
    "peak_index": {
        "peak",
        "peak_index",
        "index",
        "center",
        "center_idx",
        "tof",
        "time",
        "飞行时间",
        "峰位",
        "中心",
        "中心点",
    },
    "left_bound": {"start", "left", "left_bound", "start_idx", "起点", "左边界"},
    "right_bound": {"end", "right", "right_bound", "end_idx", "终点", "右边界"},
    "label": {"formula", "species", "name", "label", "物种", "名称", "分子式"},
}


def load_peak_ranges(path: str | Path, *, calibration: Calibration = Calibration()) -> list[PeakRange]:
    """Load manually curated peak ranges from YAML, CSV, or Excel."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        records = _load_yaml_records(path)
    elif suffix == ".csv":
        records = pd.read_csv(path).to_dict("records")
    elif suffix in {".xlsx", ".xls"}:
        records = pd.read_excel(path).to_dict("records")
    else:
        raise ValueError("manual peak file must be YAML, CSV, or Excel")
    return peak_ranges_from_records(records, calibration=calibration)


def peak_ranges_to_peaks(ranges: list[PeakRange]) -> list[Peak]:
    return [
        Peak(
            index=item.peak_index,
            time=float(item.peak_index),
            mz=float(item.mz),
            intensity=0.0,
            fwhm=float(item.right_bound - item.left_bound),
            left_bound=int(item.left_bound),
            right_bound=int(item.right_bound),
            is_auto=False,
            gaussian_params=None,
            species=item.label,
        )
        for item in ranges
    ]


def peak_ranges_from_records(records: list[dict[str, Any]], *, calibration: Calibration = Calibration()) -> list[PeakRange]:
    ranges: list[PeakRange] = []
    for row_index, raw_record in enumerate(records, start=1):
        record = _normalize_record(raw_record)
        left = _required_int(record, "left_bound", row_index)
        right = _required_int(record, "right_bound", row_index)
        if left > right:
            raise ValueError(f"manual peak row {row_index}: left bound is greater than right bound")
        peak_index = _optional_int(record, "peak_index")
        if peak_index is None:
            peak_index = int(round((left + right) / 2))
        mz = _optional_float(record, "mz")
        if mz is None:
            mz = float(calibration.tof_to_mz(peak_index))
        label = str(record.get("label") or "Manual")
        ranges.append(
            PeakRange(
                mz=float(mz),
                peak_index=int(peak_index),
                left_bound=int(left),
                right_bound=int(right),
                label=label,
            )
        )
    return ranges


def _load_yaml_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("peak_integration"), dict):
            data = data["peak_integration"]
        if isinstance(data.get("peaks"), list):
            return data["peaks"]
    raise ValueError("YAML manual peak file must contain a list or a 'peaks' list")


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            continue
        compact_key = _normalize_column_key(key)
        for canonical, aliases in COLUMN_ALIASES.items():
            if compact_key in {_normalize_column_key(alias) for alias in aliases}:
                normalized[canonical] = value
                break
    return normalized


def _normalize_column_key(key: Any) -> str:
    text = str(key).strip().lower().lstrip("\ufeff")
    text = text.replace("（", "(").replace("）", ")").replace("／", "/")
    return re.sub(r"\s+", "", text)


def _optional_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(record: dict[str, Any], key: str) -> int | None:
    value = _optional_float(record, key)
    return None if value is None else int(round(value))


def _required_int(record: dict[str, Any], key: str, row_index: int) -> int:
    value = _optional_int(record, key)
    if value is None:
        raise ValueError(f"manual peak row {row_index}: missing '{key}'")
    return value
