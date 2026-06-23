from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import project_path, writable_project_path
from .spectrum_io import extract_first_number


DEFAULT_NORMALIZATION_CONFIG = Path("config/normalization.yaml")


@dataclass
class NormalizationSettings:
    light_source: str = "io"
    temperature_photon_normalize: bool = True
    temperature_kr_correct: bool = False
    pie_photon_mode: str = "first"
    mass_discrimination: float = 1.0
    kr_calibration_folder: str = ""
    kr_calibration_peak_file: str = ""
    expansion_factors: dict[float, float] = field(default_factory=dict)
    selected_elements: list[str] = field(default_factory=list)


def extract_light_intensity(metadata_lines: list[str], source: str = "io", fallback: float = 1.0) -> float:
    source = source.lower()
    if source not in {"io", "beam_current"}:
        raise ValueError("light source must be 'io' or 'beam_current'")

    prefixes = ("io:", "io=") if source == "io" else ("beamcurrent:", "beamcurrent=")
    for line in metadata_lines:
        compact = line.replace(" ", "").lower()
        if compact.startswith(prefixes):
            value = extract_first_number(line)
            if value is not None and value > 0:
                return value
    return fallback


def load_normalization_settings(path: str | Path = DEFAULT_NORMALIZATION_CONFIG) -> NormalizationSettings:
    config_path = project_path(path)
    if not config_path.exists():
        return NormalizationSettings()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"normalization config root must be a mapping: {config_path}")
    settings_data = data.get("normalization", data)
    if not isinstance(settings_data, dict):
        raise ValueError("normalization config must be a mapping")
    if "expansion_factors" in settings_data:
        settings_data = settings_data.copy()
        raw_factors = settings_data.get("expansion_factors") or {}

        # 处理两种格式：单能量 {温度: 值} 或多能量 {能量: {温度: 值}}
        parsed_factors = {}
        for key, value in raw_factors.items():
            float_key = float(key)
            if isinstance(value, dict):
                # 多能量格式
                parsed_factors[float_key] = {float(k): float(v) for k, v in value.items()}
            else:
                # 单能量格式
                parsed_factors[float_key] = float(value)

        settings_data["expansion_factors"] = parsed_factors
    return NormalizationSettings(**{key: value for key, value in settings_data.items() if key in NormalizationSettings.__dataclass_fields__})



def save_normalization_settings(
    settings: NormalizationSettings,
    path: str | Path = DEFAULT_NORMALIZATION_CONFIG,
) -> Path:
    config_path = writable_project_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = asdict(settings)

    # 处理膨胀系数的序列化（支持单能量和多能量格式）
    serialized_factors = {}
    for key, value in settings.expansion_factors.items():
        float_key = float(key)
        if isinstance(value, dict):
            # 嵌套字典：{能量: {温度: 值}} 或 {温度: {温度: 值}}（边界情况）
            serialized_factors[float_key] = {float(k): float(v) for k, v in value.items()}
        else:
            # 简单值：{温度: 值} 或其他标量值
            try:
                serialized_factors[float_key] = float(value)
            except (TypeError, ValueError):
                # 如果无法转换为float，保持原样
                serialized_factors[float_key] = value

    data["expansion_factors"] = serialized_factors
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"normalization": data}, handle, allow_unicode=True, sort_keys=False)
    return config_path
