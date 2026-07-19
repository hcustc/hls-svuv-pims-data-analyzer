from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from .calibration import Calibration
from .config import PeakDetectionConfig, writable_config_path, readable_config_path, load_peak_detection_config
from .elements import COMMON_ELEMENTS
from .normalization import NormalizationSettings, load_normalization_settings
from .mole_fraction import MoleFractionSettings, load_mole_fraction_settings

DEFAULT_PROJECT_CONFIG = Path("config/project.yaml")
logger = logging.getLogger(__name__)

_PROJECT_PATH_FIELDS: tuple[str, ...] = (
    "single_spectrum_file",
    "sum_spectrum_folder",
    "temperature_scan_folder",
    "pie_scan_folder",
    "pics_database_path",
    "manual_peak_file",
    "temperature_scan_result_file",
    "pie_identification_result_file",
    "mole_fraction_result_file",
    "kr_calibration_folder",
    "kr_calibration_peak_file",
)


def _optional_float_dict(value):
    if not value:
        return {}
    result = {}
    for key, item in value.items():
        float_key = float(key)
        if isinstance(item, dict):
            result[float_key] = {
                float(nested_key): float(nested_value)
                for nested_key, nested_value in item.items()
            }
        else:
            result[float_key] = float(item)
    return result


@dataclass
class ProjectSettings:
    """Unified project settings replacing scattered config files."""

    # === Project Identity ===
    project_name: str = ""
    system: str = ""
    description: str = ""
    output_dir: str = "output"

    # === Data Source Paths ===
    single_spectrum_file: str = ""
    sum_spectrum_folder: str = ""
    temperature_scan_folder: str = ""
    pie_scan_folder: str = ""
    pie_scan_folders: list[str] = field(default_factory=list)
    pics_database_path: str = ""
    manual_peak_file: str = ""

    # === Analysis Artifact Paths ===
    temperature_scan_result_file: str = ""
    pie_identification_result_file: str = ""
    mole_fraction_result_file: str = ""

    # === Calibration ===
    cal_a: float = 3.66334e-7
    cal_b: float = 0.000637719
    cal_c: float = 0.272489072
    calibration_points: list[dict[str, float]] = field(default_factory=list)

    # === Normalization ===
    light_source: str = "io"
    temperature_photon_normalize: bool = True
    temperature_kr_correct: bool = False
    pie_photon_mode: str = "none"
    mass_discrimination: float = 1.0
    kr_calibration_folder: str = ""
    kr_calibration_peak_file: str = ""
    kr_mz: int = 84
    expansion_factors: dict[float, float] = field(default_factory=dict)
    selected_elements: list[str] = field(default_factory=lambda: list(COMMON_ELEMENTS))

    # === Peak Detection ===
    peak_algorithm: str = "ensemble"
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
    cwt_snr_threshold: float = 0.02
    cwt_wavelet_max_width: int = 30
    weak_tail_cutoff_idx: int = 15000
    vote_threshold: float = 0.667
    min_intensity_for_single_vote: float = 5.0
    mz_tolerance: float = 0.2

    # === PIE Defaults ===
    pie_energy_decimals: int = 1
    pie_recursive: bool = True
    pie_prefer_gaussian: bool = False
    pie_integration_method: str = "sum_counts"
    pie_multi_folder_mode: bool = False
    pie_merge_method: str = "low_energy_dominant"
    pie_replicate_mode: str = "off"

    # === Temperature Scan Defaults ===
    temp_peak_source: str = "auto"  # "auto" or "manual"
    temp_reference_mode: str = "sum"  # Only used when temp_peak_source == "auto"
    temp_prefer_gaussian: bool = False
    temp_integration_method: str = "sum_counts"
    temp_kr_mz: int = 84
    temp_curve_class_change_threshold: float = 0.25  # 相对变化阈值：判断"生成"或"消耗"需要达到最大值的多少百分比
    temp_curve_class_peak_fraction: float = 0.65  # 端点峰值比例：端点信号需达到最大值的多少百分比才判定为"生成"或"消耗"
    temp_replicate_mode: str = "off"

    # === PICS Defaults ===
    pics_no_mz: int = 30
    pics_no_formula: str = "NO"
    pics_no_mf: float = 0.01
    pics_new_species_mf: float = 0.002

    # === Mole Fraction Defaults ===
    mf_mass_disc_exponent: float = 0.77897
    mf_md_preset: str = "光电离"  # 实验条件预设名称
    mf_parent_mz: int = 0
    mf_parent_initial_mf: float = 0.002
    mf_reference_temperature: float | None = None
    mf_reference_species_mz: int | None = None
    mf_reference_species_tm: float | None = None
    mf_reference_species_mf_at_tm: float = 0.001
    mf_photon_energy: float = 10.0
    mf_kr_data: dict[float, float] = field(default_factory=dict)

    # === Converters ===

    def effective_pie_scan_folders(self) -> list[str]:
        """Return unique configured PIE segment folders with legacy fallback."""
        if isinstance(self.pie_scan_folders, str):
            configured = [self.pie_scan_folders]
        else:
            configured = list(self.pie_scan_folders or [])
        if not configured and self.pie_scan_folder:
            configured = [self.pie_scan_folder]

        folders: list[str] = []
        seen: set[str] = set()
        for value in configured:
            path = str(value or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            folders.append(path)
        return folders

    def to_calibration(self) -> Calibration:
        return Calibration(a=self.cal_a, b=self.cal_b, c=self.cal_c)

    def to_peak_detection_config(self) -> PeakDetectionConfig:
        return PeakDetectionConfig(
            algorithm=self.peak_algorithm,
            detection_min_idx=self.detection_min_idx,
            threshold_end=self.threshold_end,
            min_intensity=self.min_intensity,
            nearby_peak_window=self.nearby_peak_window,
            duplicate_window=self.duplicate_window,
            weak_tail_early_window=self.weak_tail_early_window,
            weak_tail_late_window=self.weak_tail_late_window,
            weak_tail_ratio=self.weak_tail_ratio,
            gaussian_window_max=self.gaussian_window_max,
            gaussian_boundary_scale=self.gaussian_boundary_scale,
            boundary_padding=self.boundary_padding,
            prominence_ratio=self.prominence_ratio,
            smoothing_window=self.smoothing_window,
            smoothing_poly_order=self.smoothing_poly_order,
            baseline_window=self.baseline_window,
            baseline_percentile=self.baseline_percentile,
            min_peak_width=self.min_peak_width,
            max_peak_width=self.max_peak_width,
            cwt_snr_threshold=self.cwt_snr_threshold,
            cwt_wavelet_max_width=self.cwt_wavelet_max_width,
            weak_tail_cutoff_idx=self.weak_tail_cutoff_idx,
            vote_threshold=self.vote_threshold,
            min_intensity_for_single_vote=self.min_intensity_for_single_vote,
            mz_tolerance=self.mz_tolerance,
        )

    def to_normalization_settings(self) -> NormalizationSettings:
        return NormalizationSettings(
            light_source=self.light_source,
            temperature_photon_normalize=self.temperature_photon_normalize,
            temperature_kr_correct=self.temperature_kr_correct,
            pie_photon_mode=self.pie_photon_mode,
            mass_discrimination=self.mass_discrimination,
            kr_calibration_folder=self.kr_calibration_folder,
            kr_calibration_peak_file=self.kr_calibration_peak_file,
            kr_mz=self.kr_mz,
            expansion_factors=dict(self.expansion_factors),
            selected_elements=list(self.selected_elements),
        )

    def to_mole_fraction_settings(self) -> MoleFractionSettings:
        from .mole_fraction import MoleFractionSettings
        return MoleFractionSettings(
            mass_disc_exponent=self.mf_mass_disc_exponent,
            parent_mz=self.mf_parent_mz,
            parent_initial_mf=self.mf_parent_initial_mf,
            reference_temperature=self.mf_reference_temperature,
            reference_species_mz=self.mf_reference_species_mz,
            reference_species_tm=self.mf_reference_species_tm,
            reference_species_mf_at_tm=self.mf_reference_species_mf_at_tm,
            photon_energy=self.mf_photon_energy,
            expansion_factors=dict(self.expansion_factors),
            kr_data=dict(self.mf_kr_data),
        )


def _nested_to_flat(data: dict) -> dict:
    """Flatten project.yaml into ProjectSettings fields.

    The on-disk YAML uses user-facing names (``project.name``,
    ``calibration.a``, ``peak_detection.algorithm``), while the dataclass uses
    namespaced field names to avoid collisions. Keep this mapping explicit so a
    saved project can be loaded back without silently dropping values.

    Supports dual-scope configuration:
    - Top-level sections: project, data_sources, analysis_artifacts, calibration (flat)
    - general_parameters scope: normalization (flat)
    - function_defaults scope: peak_detection, pie, temperature_scan, pics, mole_fraction (nested)

    Backward compatible: Also reads function_params from top level if not under scopes.
    """
    flat: dict[str, Any] = {}

    section_mappings: dict[str, dict[str, str]] = {
        "project": {
            "name": "project_name",
            "project_name": "project_name",
            "system": "system",
            "description": "description",
            "output_dir": "output_dir",
        },
        "data_sources": {
            "single_spectrum_file": "single_spectrum_file",
            "sum_spectrum_folder": "sum_spectrum_folder",
            "temperature_scan_folder": "temperature_scan_folder",
            "pie_scan_folder": "pie_scan_folder",
            "pie_scan_folders": "pie_scan_folders",
            "pics_database_path": "pics_database_path",
            "manual_peak_file": "manual_peak_file",
        },
        "analysis_artifacts": {
            "temperature_scan_result_file": "temperature_scan_result_file",
            "pie_identification_result_file": "pie_identification_result_file",
            "mole_fraction_result_file": "mole_fraction_result_file",
        },
        "calibration": {
            "a": "cal_a",
            "b": "cal_b",
            "c": "cal_c",
            "cal_a": "cal_a",
            "cal_b": "cal_b",
            "cal_c": "cal_c",
            "points": "calibration_points",
            "calibration_points": "calibration_points",
        },
    }

    # Load top-level sections (project, data_sources, analysis_artifacts, calibration)
    for section, mapping in section_mappings.items():
        section_data = data.get(section, {})
        if not isinstance(section_data, dict):
            continue
        for yaml_key, field_name in mapping.items():
            if yaml_key in section_data:
                flat[field_name] = section_data[yaml_key]

    # Load legacy artifacts section
    legacy_artifacts = data.get("artifacts", {})
    if isinstance(legacy_artifacts, dict):
        for yaml_key, field_name in section_mappings["analysis_artifacts"].items():
            if yaml_key in legacy_artifacts and field_name not in flat:
                flat[field_name] = legacy_artifacts[yaml_key]

    # Load from general_parameters scope (flat structure)
    gp = data.get("general_parameters", {})
    if isinstance(gp, dict):
        norm = gp.get("normalization", {})
        if isinstance(norm, dict):
            norm_mappings = {
                "light_source": "light_source",
                "temperature_photon_normalize": "temperature_photon_normalize",
                "temperature_kr_correct": "temperature_kr_correct",
                "pie_photon_mode": "pie_photon_mode",
                "mass_discrimination": "mass_discrimination",
                "kr_calibration_folder": "kr_calibration_folder",
                "kr_calibration_peak_file": "kr_calibration_peak_file",
                "kr_mz": "kr_mz",
                "expansion_factors": "expansion_factors",
                "selected_elements": "selected_elements",
            }
            for yaml_key, field_name in norm_mappings.items():
                if yaml_key in norm:
                    flat[field_name] = norm[yaml_key]

    # Fallback: Load normalization from top-level if not under general_parameters (backward compat)
    if "normalization" in data and "general_parameters" not in data:
        norm = data.get("normalization", {})
        if isinstance(norm, dict):
            norm_mappings = {
                "light_source": "light_source",
                "temperature_photon_normalize": "temperature_photon_normalize",
                "temperature_kr_correct": "temperature_kr_correct",
                "pie_photon_mode": "pie_photon_mode",
                "mass_discrimination": "mass_discrimination",
                "kr_calibration_folder": "kr_calibration_folder",
                "kr_calibration_peak_file": "kr_calibration_peak_file",
                "kr_mz": "kr_mz",
                "expansion_factors": "expansion_factors",
                "selected_elements": "selected_elements",
            }
            for yaml_key, field_name in norm_mappings.items():
                if yaml_key in norm and field_name not in flat:
                    flat[field_name] = norm[yaml_key]

    # Load from function_defaults scope (nested structure)
    fd = data.get("function_defaults", {})

    # Fallback: If no function_defaults scope, try function_params at top level
    if not isinstance(fd, dict) or not fd:
        legacy_fd = data.get("function_params", {})
        if isinstance(legacy_fd, dict):
            fd = legacy_fd

    if isinstance(fd, dict):
        pie = fd.get("pie", {})
        if isinstance(pie, dict):
            for k, kk in [("pie_energy_decimals", "energy_decimals"),
                          ("pie_recursive", "recursive"),
                          ("pie_prefer_gaussian", "prefer_gaussian"),
                          ("pie_integration_method", "integration_method"),
                          ("pie_multi_folder_mode", "multi_folder_mode"),
                          ("pie_merge_method", "merge_method"),
                          ("pie_replicate_mode", "replicate_mode")]:
                if kk in pie:
                    flat[k] = pie[kk]
            if "integration_method" not in pie and pie.get("prefer_gaussian") is True:
                flat["pie_integration_method"] = "gaussian"

        temp = fd.get("temperature_scan", {})
        if isinstance(temp, dict):
            for k, kk in [("temp_peak_source", "peak_source"),
                          ("temp_reference_mode", "reference_mode"),
                          ("temp_prefer_gaussian", "prefer_gaussian"),
                          ("temp_integration_method", "integration_method"),
                          ("temp_kr_mz", "kr_mz"),
                          ("temp_curve_class_change_threshold", "curve_class_change_threshold"),
                          ("temp_curve_class_peak_fraction", "curve_class_peak_fraction"),
                          ("temp_replicate_mode", "replicate_mode")]:
                if kk in temp:
                    flat[k] = temp[kk]
            if "integration_method" not in temp and temp.get("prefer_gaussian") is True:
                flat["temp_integration_method"] = "gaussian"

        pics = fd.get("pics", {})
        if isinstance(pics, dict):
            for k, kk in [("pics_no_mz", "no_mz"), ("pics_no_formula", "no_formula"),
                          ("pics_no_mf", "no_mf"), ("pics_new_species_mf", "new_species_mf")]:
                if kk in pics:
                    flat[k] = pics[kk]

        mf = fd.get("mole_fraction", {})
        if isinstance(mf, dict):
            for k, kk in [("mf_md_preset", "md_preset"),
                          ("mf_mass_disc_exponent", "mass_disc_exponent"),
                          ("mf_parent_mz", "parent_mz"),
                          ("mf_parent_initial_mf", "parent_initial_mf"),
                          ("mf_reference_temperature", "reference_temperature"),
                          ("mf_reference_species_mz", "reference_species_mz"),
                          ("mf_reference_species_tm", "reference_species_tm"),
                          ("mf_reference_species_mf_at_tm", "reference_species_mf_at_tm"),
                          ("mf_photon_energy", "photon_energy"),
                          ("mf_kr_data", "kr_data")]:
                if kk in mf:
                    flat[k] = mf[kk]

        # Load peak_detection from function_defaults scope
        peak_det = fd.get("peak_detection", {})
        if isinstance(peak_det, dict):
            peak_mappings = {
                "algorithm": "peak_algorithm",
                "peak_algorithm": "peak_algorithm",
                "detection_min_idx": "detection_min_idx",
                "threshold_end": "threshold_end",
                "min_intensity": "min_intensity",
                "nearby_peak_window": "nearby_peak_window",
                "duplicate_window": "duplicate_window",
                "weak_tail_early_window": "weak_tail_early_window",
                "weak_tail_late_window": "weak_tail_late_window",
                "weak_tail_ratio": "weak_tail_ratio",
                "gaussian_window_max": "gaussian_window_max",
                "gaussian_boundary_scale": "gaussian_boundary_scale",
                "boundary_padding": "boundary_padding",
                "prominence_ratio": "prominence_ratio",
                "smoothing_window": "smoothing_window",
                "smoothing_poly_order": "smoothing_poly_order",
                "baseline_window": "baseline_window",
                "baseline_percentile": "baseline_percentile",
                "min_peak_width": "min_peak_width",
                "max_peak_width": "max_peak_width",
                "cwt_snr_threshold": "cwt_snr_threshold",
                "cwt_wavelet_max_width": "cwt_wavelet_max_width",
                "weak_tail_cutoff_idx": "weak_tail_cutoff_idx",
                "vote_threshold": "vote_threshold",
                "min_intensity_for_single_vote": "min_intensity_for_single_vote",
                "mz_tolerance": "mz_tolerance",
            }
            for yaml_key, field_name in peak_mappings.items():
                if yaml_key in peak_det:
                    flat[field_name] = peak_det[yaml_key]

    # Fallback: Load peak_detection from top level if not in function_defaults (backward compat)
    if "peak_detection" in data and ("function_defaults" not in data):
        peak_det = data.get("peak_detection", {})
        if isinstance(peak_det, dict):
            peak_mappings = {
                "algorithm": "peak_algorithm",
                "peak_algorithm": "peak_algorithm",
                "detection_min_idx": "detection_min_idx",
                "threshold_end": "threshold_end",
                "min_intensity": "min_intensity",
                "nearby_peak_window": "nearby_peak_window",
                "duplicate_window": "duplicate_window",
                "weak_tail_early_window": "weak_tail_early_window",
                "weak_tail_late_window": "weak_tail_late_window",
                "weak_tail_ratio": "weak_tail_ratio",
                "gaussian_window_max": "gaussian_window_max",
                "gaussian_boundary_scale": "gaussian_boundary_scale",
                "boundary_padding": "boundary_padding",
                "prominence_ratio": "prominence_ratio",
                "smoothing_window": "smoothing_window",
                "smoothing_poly_order": "smoothing_poly_order",
                "baseline_window": "baseline_window",
                "baseline_percentile": "baseline_percentile",
                "min_peak_width": "min_peak_width",
                "max_peak_width": "max_peak_width",
                "cwt_snr_threshold": "cwt_snr_threshold",
                "cwt_wavelet_max_width": "cwt_wavelet_max_width",
                "weak_tail_cutoff_idx": "weak_tail_cutoff_idx",
                "vote_threshold": "vote_threshold",
                "min_intensity_for_single_vote": "min_intensity_for_single_vote",
                "mz_tolerance": "mz_tolerance",
            }
            for yaml_key, field_name in peak_mappings.items():
                if yaml_key in peak_det and field_name not in flat:
                    flat[field_name] = peak_det[yaml_key]

    # Older project files stored the only Kr mass under the temperature
    # function defaults.  The common-parameter ``kr_mz`` is now canonical.
    if "kr_mz" not in flat and "temp_kr_mz" in flat:
        flat["kr_mz"] = flat["temp_kr_mz"]

    for key in ("expansion_factors", "mf_kr_data"):
        if key in flat and isinstance(flat[key], dict):
            flat[key] = _optional_float_dict(flat[key])

    return flat

def _flat_to_nested(settings: ProjectSettings) -> dict:
    """Convert flat ProjectSettings to nested project.yaml structure.

    New dual-scope structure:
    - general_parameters.normalization: general parameters (flat)
    - function_defaults: function-specific parameters (nested)

    Backward compatible: Still supports reading from top-level sections for old projects.
    """
    d = asdict(settings)

    return {
        "project": {
            "name": d["project_name"],
            "system": d["system"],
            "description": d["description"],
            "output_dir": d["output_dir"],
        },
        "data_sources": {
            "single_spectrum_file": d["single_spectrum_file"],
            "sum_spectrum_folder": d["sum_spectrum_folder"],
            "temperature_scan_folder": d["temperature_scan_folder"],
            "pie_scan_folder": d["pie_scan_folder"],
            "pie_scan_folders": d["pie_scan_folders"],
            "pics_database_path": d["pics_database_path"],
            "manual_peak_file": d["manual_peak_file"],
        },
        "analysis_artifacts": {
            "temperature_scan_result_file": d["temperature_scan_result_file"],
            "pie_identification_result_file": d["pie_identification_result_file"],
            "mole_fraction_result_file": d["mole_fraction_result_file"],
        },
        "calibration": {
            "a": d["cal_a"],
            "b": d["cal_b"],
            "c": d["cal_c"],
            "points": d.get("calibration_points", []),
        },
        "general_parameters": {
            "normalization": {
                "light_source": d["light_source"],
                "temperature_photon_normalize": d["temperature_photon_normalize"],
                "temperature_kr_correct": d["temperature_kr_correct"],
                "pie_photon_mode": d["pie_photon_mode"],
                "kr_calibration_folder": d["kr_calibration_folder"],
                "kr_calibration_peak_file": d["kr_calibration_peak_file"],
                "kr_mz": d["kr_mz"],
                "expansion_factors": _optional_float_dict(d["expansion_factors"]),
                "selected_elements": d["selected_elements"],
            },
        },
        "function_defaults": {
            "peak_detection": {
                "algorithm": d["peak_algorithm"],
                "detection_min_idx": d["detection_min_idx"],
                "threshold_end": d["threshold_end"],
                "min_intensity": d["min_intensity"],
                "nearby_peak_window": d["nearby_peak_window"],
                "duplicate_window": d["duplicate_window"],
                "weak_tail_early_window": d["weak_tail_early_window"],
                "weak_tail_late_window": d["weak_tail_late_window"],
                "weak_tail_ratio": d["weak_tail_ratio"],
                "gaussian_window_max": d["gaussian_window_max"],
                "gaussian_boundary_scale": d["gaussian_boundary_scale"],
                "boundary_padding": d["boundary_padding"],
                "prominence_ratio": d["prominence_ratio"],
                "smoothing_window": d["smoothing_window"],
                "smoothing_poly_order": d["smoothing_poly_order"],
                "baseline_window": d["baseline_window"],
                "baseline_percentile": d["baseline_percentile"],
                "min_peak_width": d["min_peak_width"],
                "max_peak_width": d["max_peak_width"],
                "cwt_snr_threshold": d["cwt_snr_threshold"],
                "cwt_wavelet_max_width": d["cwt_wavelet_max_width"],
                "weak_tail_cutoff_idx": d["weak_tail_cutoff_idx"],
                "vote_threshold": d["vote_threshold"],
                "min_intensity_for_single_vote": d["min_intensity_for_single_vote"],
                "mz_tolerance": d["mz_tolerance"],
            },
            "pie": {
                "energy_decimals": d["pie_energy_decimals"],
                "recursive": d["pie_recursive"],
                "integration_method": d["pie_integration_method"],
                "multi_folder_mode": d["pie_multi_folder_mode"],
                "merge_method": d["pie_merge_method"],
                "replicate_mode": d["pie_replicate_mode"],
            },
            "temperature_scan": {
                "peak_source": d["temp_peak_source"],
                "reference_mode": d["temp_reference_mode"],
                "integration_method": d["temp_integration_method"],
                "kr_mz": d["temp_kr_mz"],
                "curve_class_change_threshold": d["temp_curve_class_change_threshold"],
                "curve_class_peak_fraction": d["temp_curve_class_peak_fraction"],
                "replicate_mode": d["temp_replicate_mode"],
            },
            "pics": {
                "no_mz": d["pics_no_mz"],
                "no_formula": d["pics_no_formula"],
                "no_mf": d["pics_no_mf"],
                "new_species_mf": d["pics_new_species_mf"],
            },
            "mole_fraction": {
                "md_preset": d["mf_md_preset"],
                "mass_disc_exponent": d["mf_mass_disc_exponent"],
                "parent_mz": d["mf_parent_mz"],
                "parent_initial_mf": d["mf_parent_initial_mf"],
                "reference_temperature": d["mf_reference_temperature"],
                "reference_species_mz": d["mf_reference_species_mz"],
                "reference_species_tm": d["mf_reference_species_tm"],
                "reference_species_mf_at_tm": d["mf_reference_species_mf_at_tm"],
                "photon_energy": d["mf_photon_energy"],
                "kr_data": _optional_float_dict(d["mf_kr_data"]),
            },
        },
    }


def _absolute_pure_path(value: str) -> PureWindowsPath | PurePosixPath | None:
    """Parse an absolute path without assuming the current operating system."""
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute():
        return windows_path
    posix_path = PurePosixPath(value)
    if posix_path.is_absolute():
        return posix_path
    return None


def _relative_to_root(value: str, root: str | Path) -> PureWindowsPath | PurePosixPath | None:
    value_path = _absolute_pure_path(value)
    root_path = _absolute_pure_path(str(root))
    if value_path is None or root_path is None:
        return None
    if isinstance(value_path, PureWindowsPath) != isinstance(root_path, PureWindowsPath):
        return None
    try:
        return value_path.relative_to(root_path)
    except ValueError:
        return None


def _portable_relative_parts(value: str) -> tuple[str, ...]:
    """Return path components for a relative path written by either OS."""
    if "\\" in value:
        return PureWindowsPath(value).parts
    return PurePosixPath(value).parts


def _resolve_portable_path(
    value: str,
    project_root: Path,
    *,
    stored_project_root: str,
) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    absolute_path = _absolute_pure_path(raw)
    if absolute_path is None:
        parts = _portable_relative_parts(raw)
        return str(project_root.joinpath(*parts).resolve())

    legacy_relative = _relative_to_root(raw, stored_project_root)
    if legacy_relative is not None:
        return str(project_root.joinpath(*legacy_relative.parts).resolve())
    return raw


def _resolve_project_scoped_paths(settings: ProjectSettings, project_root: str | Path) -> ProjectSettings:
    """Resolve portable paths for runtime and rebase legacy project-local paths."""
    root = Path(project_root).expanduser().resolve()
    stored_project_root = str(settings.output_dir or "").strip()
    pie_folders = (
        [settings.pie_scan_folders]
        if isinstance(settings.pie_scan_folders, str)
        else list(settings.pie_scan_folders or [])
    )
    resolved = replace(settings, pie_scan_folders=pie_folders)
    resolved.output_dir = str(root)

    for field_name in _PROJECT_PATH_FIELDS:
        setattr(
            resolved,
            field_name,
            _resolve_portable_path(
                getattr(settings, field_name),
                root,
                stored_project_root=stored_project_root,
            ),
        )
    resolved.pie_scan_folders = [
        _resolve_portable_path(value, root, stored_project_root=stored_project_root)
        for value in pie_folders
        if str(value or "").strip()
    ]
    return resolved


def _portable_project_path(
    value: str,
    project_root: Path,
    *,
    runtime_project_root: str = "",
) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    value_absolute = _absolute_pure_path(raw)
    if value_absolute is None:
        return PurePosixPath(*_portable_relative_parts(raw)).as_posix()

    relative = _relative_to_root(raw, project_root)
    if relative is None and runtime_project_root:
        relative = _relative_to_root(raw, runtime_project_root)
    if relative is None:
        root_absolute = _absolute_pure_path(str(project_root))
        if root_absolute is None or (
            isinstance(value_absolute, PureWindowsPath)
            != isinstance(root_absolute, PureWindowsPath)
        ):
            return raw
        # Resolve local aliases such as /var -> /private/var on macOS.
        try:
            relative = PurePosixPath(
                *Path(raw).expanduser().resolve().relative_to(project_root.resolve()).parts
            )
        except (OSError, ValueError):
            return raw
    return relative.as_posix() or "."


def _project_settings_for_storage(
    settings: ProjectSettings,
    project_root: str | Path,
) -> ProjectSettings:
    """Create a copy whose project-local paths are portable POSIX relatives."""
    root = Path(project_root).expanduser().resolve()
    pie_folders = (
        [settings.pie_scan_folders]
        if isinstance(settings.pie_scan_folders, str)
        else list(settings.pie_scan_folders or [])
    )
    portable = replace(settings, pie_scan_folders=pie_folders)
    runtime_project_root = str(settings.output_dir or "").strip()
    portable.output_dir = "."
    for field_name in _PROJECT_PATH_FIELDS:
        setattr(
            portable,
            field_name,
            _portable_project_path(
                getattr(settings, field_name),
                root,
                runtime_project_root=runtime_project_root,
            ),
        )
    portable.pie_scan_folders = [
        _portable_project_path(value, root, runtime_project_root=runtime_project_root)
        for value in pie_folders
        if str(value or "").strip()
    ]
    return portable


def portable_project_settings_dict(
    settings: ProjectSettings,
    project_root: str | Path,
) -> dict[str, Any]:
    """Return a flat, portable settings mapping for project metadata exports."""
    return asdict(_project_settings_for_storage(settings, project_root))


def _inferred_project_root(path: str | Path, config_path: Path) -> Path | None:
    """Infer ``<root>`` only for explicit ``<root>/config/project.yaml`` paths."""
    raw_path = Path(path)
    if raw_path == DEFAULT_PROJECT_CONFIG:
        return None
    if len(raw_path.parts) >= 3 and raw_path.parts[-2:] == ("config", "project.yaml"):
        return config_path.parent.parent
    return None


def load_project_settings(
    path: str | Path = DEFAULT_PROJECT_CONFIG,
    *,
    project_root: str | Path | None = None,
) -> ProjectSettings:
    config_path = readable_config_path(path)
    effective_root = (
        Path(project_root).expanduser().resolve()
        if project_root
        else _inferred_project_root(path, config_path)
    )
    if not config_path.exists():
        logger.warning("Project settings file not found; using defaults: %s", config_path)
        defaults = ProjectSettings()
        if effective_root is not None:
            defaults.output_dir = str(effective_root)
        return defaults
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"project config root must be a mapping: {config_path}")
    flat = _nested_to_flat(data)
    settings = ProjectSettings(**{k: v for k, v in flat.items() if k in ProjectSettings.__dataclass_fields__})
    if effective_root is not None:
        settings = _resolve_project_scoped_paths(settings, effective_root)
    return settings


def save_project_settings(
    settings: ProjectSettings,
    path: str | Path = DEFAULT_PROJECT_CONFIG,
    *,
    project_root: str | Path | None = None,
) -> Path:
    config_path = writable_config_path(path)
    effective_root = (
        Path(project_root).expanduser().resolve()
        if project_root
        else _inferred_project_root(path, config_path)
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    stored_settings = (
        _project_settings_for_storage(settings, effective_root)
        if effective_root is not None
        else settings
    )
    nested = _flat_to_nested(stored_settings)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(nested, handle, allow_unicode=True, sort_keys=False)
    return config_path


def migrate_from_legacy_configs() -> ProjectSettings:
    """One-time migration: merge all legacy YAML configs into ProjectSettings."""
    s = ProjectSettings()

    try:
        from .config import load_calibration_config, load_peak_detection_config
        cal = load_calibration_config()
        s.cal_a = cal.a
        s.cal_b = cal.b
        s.cal_c = cal.c
    except Exception:
        pass

    try:
        ns = load_normalization_settings()
        s.light_source = ns.light_source
        s.temperature_photon_normalize = ns.temperature_photon_normalize
        s.temperature_kr_correct = ns.temperature_kr_correct
        s.pie_photon_mode = ns.pie_photon_mode
        s.kr_calibration_folder = ns.kr_calibration_folder
        s.kr_calibration_peak_file = ns.kr_calibration_peak_file
        s.kr_mz = ns.kr_mz
        s.expansion_factors = dict(ns.expansion_factors)
        if ns.selected_elements:
            s.selected_elements = list(ns.selected_elements)
    except Exception:
        pass

    try:
        pd = load_peak_detection_config()
        s.peak_algorithm = pd.algorithm
        s.detection_min_idx = pd.detection_min_idx
        s.threshold_end = pd.threshold_end
        s.min_intensity = pd.min_intensity
        s.nearby_peak_window = pd.nearby_peak_window
        s.duplicate_window = pd.duplicate_window
        s.weak_tail_early_window = pd.weak_tail_early_window
        s.weak_tail_late_window = pd.weak_tail_late_window
        s.weak_tail_ratio = pd.weak_tail_ratio
        s.gaussian_window_max = pd.gaussian_window_max
        s.gaussian_boundary_scale = pd.gaussian_boundary_scale
        s.boundary_padding = pd.boundary_padding
        s.prominence_ratio = pd.prominence_ratio
        s.smoothing_window = pd.smoothing_window
        s.smoothing_poly_order = pd.smoothing_poly_order
        s.baseline_window = pd.baseline_window
        s.baseline_percentile = pd.baseline_percentile
        s.min_peak_width = pd.min_peak_width
        s.max_peak_width = pd.max_peak_width
    except Exception:
        pass

    try:
        mf = load_mole_fraction_settings()
        s.mf_mass_disc_exponent = mf.mass_disc_exponent
        s.mf_parent_mz = mf.parent_mz
        s.mf_parent_initial_mf = mf.parent_initial_mf
        s.mf_reference_temperature = mf.reference_temperature
        s.mf_reference_species_mz = mf.reference_species_mz
        s.mf_reference_species_tm = mf.reference_species_tm
        s.mf_reference_species_mf_at_tm = mf.reference_species_mf_at_tm
        s.mf_photon_energy = mf.photon_energy
        s.mf_kr_data = dict(mf.kr_data) if mf.kr_data else {}
    except Exception:
        pass

    return s


class ProjectSettingsManager:
    """Singleton holding the active ProjectSettings instance.

    Supports both global config (config/project.yaml) and per-project configs (project_root/config/project.yaml).
    """

    _instance: ProjectSettingsManager | None = None
    _settings: ProjectSettings | None = None
    _project_config_path: Path | None = None  # 项目特定配置路径

    def __new__(cls) -> ProjectSettingsManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_project_path(self, project_root: Path | str) -> None:
        """设置项目级配置路径。当打开项目时调用此方法。

        Args:
            project_root: 项目根目录路径
        """
        project_root = Path(project_root).expanduser().resolve()
        self._project_config_path = project_root / "config" / "project.yaml"
        # 重新加载设置以使用新的项目路径
        self.reload()

    def get_project_config_path(self) -> Path:
        """获取当前使用的项目配置路径。"""
        if self._project_config_path:
            return self._project_config_path
        return DEFAULT_PROJECT_CONFIG

    def has_project_path(self) -> bool:
        """Return True when settings are scoped to an opened project folder."""
        return self._project_config_path is not None

    def _project_root(self) -> Path | None:
        if self._project_config_path is None:
            return None
        return self._project_config_path.parent.parent

    def get(self) -> ProjectSettings:
        if self._settings is None:
            config_path = self.get_project_config_path()
            project_root = self._project_root()
            self._settings = load_project_settings(config_path, project_root=project_root)
            if self._is_fresh(config_path):
                try:
                    self._settings = migrate_from_legacy_configs()
                    if project_root is not None:
                        self._settings.output_dir = str(project_root.resolve())
                    save_project_settings(self._settings, config_path, project_root=project_root)
                except Exception:
                    pass
        return self._settings

    def save(self) -> Path:
        config_path = self.get_project_config_path()
        return save_project_settings(self.get(), config_path, project_root=self._project_root())

    def reload(self) -> ProjectSettings:
        config_path = self.get_project_config_path()
        self._settings = load_project_settings(config_path, project_root=self._project_root())
        return self._settings

    def set(self, settings: ProjectSettings) -> None:
        self._settings = settings

    def _is_fresh(self, config_path: Path = None) -> bool:
        """检查项目配置是否为新建（不存在）。"""
        if config_path is None:
            config_path = self.get_project_config_path()
        try:
            return not readable_config_path(config_path).exists()
        except Exception:
            return True

    def clear_project_path(self) -> None:
        """清除项目路径，恢复为使用全局配置。"""
        self._project_config_path = None
        self._settings = None
