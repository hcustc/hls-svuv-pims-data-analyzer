from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

from .calibration import Calibration
from .runtime_paths import ensure_user_copy, resource_path, runtime_read_path, writable_path


PROJECT_ROOT = resource_path(".")
CONFIG_ROOT = Path("config")
DEFAULT_APP_CONFIG = CONFIG_ROOT / "app.yaml"


@dataclass
class PeakDetectionConfig:
    algorithm: str = "ensemble"
    detection_min_idx: int = 3000
    threshold_end: float = 2.0
    min_intensity: float = 3.0
    nearby_peak_window: int = 30
    duplicate_window: int = 20
    weak_tail_early_window: int = 90
    weak_tail_late_window: int = 50
    weak_tail_ratio: float = 5.0
    gaussian_window_max: int = 30
    gaussian_boundary_scale: float = 1.5
    boundary_padding: int = 2
    prominence_ratio: float = 0.005
    smoothing_window: int = 5
    smoothing_poly_order: int = 2
    baseline_window: int = 301
    baseline_percentile: float = 5.0
    min_peak_width: int = 1
    max_peak_width: int = 80
    # CWT parameters
    cwt_snr_threshold: float = 0.02
    cwt_wavelet_max_width: int = 30
    weak_tail_cutoff_idx: int = 15000
    # Ensemble-specific parameters
    vote_threshold: float = 0.667
    min_intensity_for_single_vote: float = 5.0
    mz_tolerance: float = 0.2

    def to_peak_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict for detect_peaks_by_algorithm()."""
        return asdict(self)


def project_path(path: str | Path) -> Path:
    return runtime_read_path(path)


def writable_project_path(path: str | Path) -> Path:
    return writable_path(path)


def readable_config_path(path: str | Path) -> Path:
    return runtime_read_path(path)


def writable_config_path(path: str | Path) -> Path:
    return writable_path(path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with readable_config_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_app_config(path: str | Path = DEFAULT_APP_CONFIG) -> dict[str, Any]:
    return load_yaml(path)


def app_path(section: str, key: str, fallback: str) -> Path:
    config = load_app_config()
    value = config.get(section, {}).get(key, fallback)
    return project_path(value)


def species_database_path() -> Path:
    app = load_app_config()
    value = app.get("database", {}).get("pics", "database/species_database.sqlite")
    path = ensure_user_copy(value)
    from .db_migration import ensure_database_up_to_date

    ensure_database_up_to_date(path)
    return path


def resolve_species_database_path(configured_path: str | Path | None = None) -> Path:
    """Return an existing project PICS database, or the maintained default database."""
    value = str(configured_path or "").strip()
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate
    return species_database_path()


def load_calibration_config(path: str | Path | None = None) -> Calibration:
    app = load_app_config()
    config_path = path or app.get("config", {}).get("calibration", "config/calibration.yaml")
    data = load_yaml(config_path)
    coefficients = data.get("calibration", {}).get("coefficients", {})
    return Calibration(
        a=float(coefficients["a"]),
        b=float(coefficients["b"]),
        c=float(coefficients["c"]),
    )


def save_calibration_config(calibration: Calibration, path: str | Path | None = None) -> Path:
    app = load_app_config()
    config_path = writable_config_path(path or app.get("config", {}).get("calibration", "config/calibration.yaml"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "calibration": {
            "equation": "mz = c + b * tof + a * tof^2",
            "coefficients": {
                "a": float(calibration.a),
                "b": float(calibration.b),
                "c": float(calibration.c),
            },
        }
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return config_path


def load_calibration_points(path: str | Path | None = None) -> list[tuple[float, float]]:
    app = load_app_config()
    config_path = path or app.get("config", {}).get("calibration_points", "config/calibration_points.yaml")
    path_to_load = readable_config_path(config_path)
    if not path_to_load.exists():
        return []
    data = load_yaml(config_path)
    points = data.get("calibration_points", [])
    return [(float(item["tof"]), float(item["mz"])) for item in points]


def save_calibration_points(points: list[tuple[float, float]], path: str | Path | None = None) -> Path:
    app = load_app_config()
    config_path = writable_config_path(path or app.get("config", {}).get("calibration_points", "config/calibration_points.yaml"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "calibration_points": [
            {"tof": float(tof), "mz": float(mz)}
            for tof, mz in points
        ]
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return config_path


def load_peak_detection_config(path: str | Path | None = None) -> PeakDetectionConfig:
    app = load_app_config()
    path_to_load = readable_config_path(path or app.get("config", {}).get("peak_detection", "config/peak_detection.yaml"))
    if not path_to_load.exists():
        return PeakDetectionConfig()
    data = load_yaml(path_to_load)
    values = data.get("peak_detection", {})
    defaults = asdict(PeakDetectionConfig())
    defaults.update({key: values[key] for key in defaults if key in values})
    return PeakDetectionConfig(
        algorithm=str(defaults["algorithm"]),
        detection_min_idx=int(defaults["detection_min_idx"]),
        threshold_end=float(defaults["threshold_end"]),
        min_intensity=float(defaults["min_intensity"]),
        nearby_peak_window=int(defaults["nearby_peak_window"]),
        duplicate_window=int(defaults["duplicate_window"]),
        weak_tail_early_window=int(defaults["weak_tail_early_window"]),
        weak_tail_late_window=int(defaults["weak_tail_late_window"]),
        weak_tail_ratio=float(defaults["weak_tail_ratio"]),
        gaussian_window_max=int(defaults["gaussian_window_max"]),
        gaussian_boundary_scale=float(defaults["gaussian_boundary_scale"]),
        boundary_padding=int(defaults["boundary_padding"]),
        prominence_ratio=float(defaults["prominence_ratio"]),
        smoothing_window=int(defaults["smoothing_window"]),
        smoothing_poly_order=int(defaults["smoothing_poly_order"]),
        baseline_window=int(defaults["baseline_window"]),
        baseline_percentile=float(defaults["baseline_percentile"]),
        min_peak_width=int(defaults["min_peak_width"]),
        max_peak_width=int(defaults["max_peak_width"]),
        cwt_snr_threshold=float(defaults["cwt_snr_threshold"]),
        cwt_wavelet_max_width=int(defaults["cwt_wavelet_max_width"]),
        weak_tail_cutoff_idx=int(defaults["weak_tail_cutoff_idx"]),
        vote_threshold=float(defaults["vote_threshold"]),
        min_intensity_for_single_vote=float(defaults["min_intensity_for_single_vote"]),
        mz_tolerance=float(defaults["mz_tolerance"]),
    )


def save_peak_detection_config(config: PeakDetectionConfig, path: str | Path | None = None) -> Path:
    app = load_app_config()
    config_path = writable_config_path(path or app.get("config", {}).get("peak_detection", "config/peak_detection.yaml"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"peak_detection": asdict(config)}, handle, allow_unicode=True, sort_keys=False)
    return config_path


def load_peak_integration_config(path: str | Path | None = None) -> list[dict[str, Any]]:
    app = load_app_config()
    config_path = path or app.get("config", {}).get("peak_integration", "config/peak_integration.yaml")
    data = load_yaml(config_path)
    return list(data.get("peak_integration", {}).get("peaks", []))
