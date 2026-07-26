from __future__ import annotations

import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import project_path, writable_project_path
from .isotope import formula_nominal_mass
from .pie_analysis import load_species_database
from .temperature_scan import build_temperature_curves


logger = logging.getLogger(__name__)

DEFAULT_MOLE_FRACTION_CONFIG = Path("config/mole_fraction.yaml")

MASS_DISCRIMINATION_PRESETS = {
    "760 Torr / 80μm": 0.77897,
    "5 Torr / 400μm (Catalysis)": 0.36267,
    "30 Torr (Catalysis)": 0.75148,
    "30 Torr / 350μm (Catalysis)": 0.75148,
    "150 Torr (Combustion)": 0.76155,
    "150 Torr / 150μm (Chemistry)": 0.76155,
}


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
class MoleFractionSettings:
    mass_disc_exponent: float = 0.77897
    parent_mz: float = 0.0
    parent_initial_mf: float = 0.002
    reference_temperature: float | None = None
    reference_species_mz: float | None = None
    reference_species_tm: float | None = None
    reference_species_mf_at_tm: float = 0.001
    photon_energy: float = 10.0
    expansion_factors: dict[float, float] = field(default_factory=dict)
    kr_data: dict[float, float] = field(default_factory=dict)


def load_mole_fraction_settings(
    path: str | Path = DEFAULT_MOLE_FRACTION_CONFIG,
) -> MoleFractionSettings:
    config_path = project_path(path)
    if not config_path.exists():
        return MoleFractionSettings()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"mole fraction config root must be a mapping: {config_path}")
    settings_data = data.get("mole_fraction", data)
    if not isinstance(settings_data, dict):
        raise ValueError("mole fraction config must be a mapping")
    if "expansion_factors" in settings_data:
        settings_data = settings_data.copy()
        settings_data["expansion_factors"] = _optional_float_dict(settings_data.get("expansion_factors"))
    if "kr_data" in settings_data:
        settings_data = settings_data.copy()
        settings_data["kr_data"] = _optional_float_dict(settings_data.get("kr_data"))
    return MoleFractionSettings(
        **{
            key: value
            for key, value in settings_data.items()
            if key in MoleFractionSettings.__dataclass_fields__
        }
    )


def save_mole_fraction_settings(
    settings: MoleFractionSettings,
    path: str | Path = DEFAULT_MOLE_FRACTION_CONFIG,
) -> Path:
    config_path = writable_project_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = asdict(settings)
    data["expansion_factors"] = _optional_float_dict(settings.expansion_factors)
    data["kr_data"] = _optional_float_dict(settings.kr_data)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"mole_fraction": data}, handle, allow_unicode=True, sort_keys=False
        )
    return config_path


def calc_mass_discrimination(molecular_weight: float, exponent: float = 0.77897) -> float:
    return float((molecular_weight / 30.0) ** exponent)


def calc_expansion_coefficients(
    kr_data: dict[float, float],
) -> dict[float, float]:
    if not kr_data:
        return {}
    ref_temp = min(kr_data.keys())
    ref_signal = kr_data[ref_temp]
    if ref_signal == 0:
        return {}
    return {temp: signal / ref_signal for temp, signal in kr_data.items()}


def get_expansion_coefficient(
    temperature: float,
    expansion_coefficients: dict[float, float],
) -> float:
    if not expansion_coefficients:
        return 1.0

    # 检测是否为多能量格式 {energy: {temp: coeff}}，自动退化为第一个可用能量
    first_val = next(iter(expansion_coefficients.values()), None)
    if isinstance(first_val, dict):
        return get_expansion_coefficient(temperature, first_val)

    if temperature in expansion_coefficients:
        return expansion_coefficients[temperature]
    temps = sorted(expansion_coefficients.keys())
    if len(temps) < 2:
        return 1.0
    coeffs = [expansion_coefficients[t] for t in temps]
    return float(np.interp(temperature, temps, coeffs))


def parse_expansion_factors_from_result(result_df) -> dict:
    """从 compute_kr_expansion_factors 的 DataFrame 结果转换成可存储的格式。

    支持两种格式：
    - 单能量：{温度: 膨胀系数}
    - 多能量：{能量: {温度: 膨胀系数}}
    """
    if result_df.empty:
        return {}

    # 检查是否包含多个能量（多能量）
    if "photon_energy" in result_df.columns and result_df["photon_energy"].nunique(dropna=True) > 1:
        # 多能量格式
        result = {}
        for energy in result_df["photon_energy"].unique():
            energy_data = result_df[result_df["photon_energy"] == energy]
            energy_factors = {
                float(row["temperature"]): float(row["expansion_lambda"])
                for _, row in energy_data.iterrows()
            }
            result[float(energy)] = energy_factors
        return result
    else:
        # 单能量格式（向后兼容）
        return {
            float(row["temperature"]): float(row["expansion_lambda"])
            for _, row in result_df.iterrows()
        }


def get_expansion_coefficient_for_energy(
    temperature: float,
    energy: float | None,
    expansion_factors: dict,
) -> float:
    """获取特定温度和能量的膨胀系数。

    支持以下格式：
    - 单能量格式：{温度: 膨胀系数}
    - 多能量格式：{能量: {温度: 膨胀系数}}

    Args:
        temperature: 目标温度
        energy: 目标能量（如果为None，使用单能量逻辑）
        expansion_factors: 膨胀系数字典
    """
    if not expansion_factors:
        return 1.0

    # 检测格式
    first_key = next(iter(expansion_factors.keys()), None)
    if first_key is None:
        return 1.0

    # 判断是多能量还是单能量
    is_multi_energy = isinstance(expansion_factors[first_key], dict)

    if is_multi_energy:
        # 多能量格式：选择最接近的能量
        if energy is None:
            # 如果没有指定能量，使用第一个能量的数据
            energy_factors = next(iter(expansion_factors.values()))
        else:
            # 找最接近的能量
            energies = list(expansion_factors.keys())
            closest_energy = min(energies, key=lambda e: abs(e - energy))
            energy_factors = expansion_factors[closest_energy]

        return get_expansion_coefficient(temperature, energy_factors)
    else:
        # 单能量格式
        return get_expansion_coefficient(temperature, expansion_factors)



def select_calc_energy(
    ie: float, usable_energies: list[float], threshold: float = 0.05
) -> float:
    if not usable_energies:
        raise ValueError("no usable energies")
    sorted_energies = sorted(usable_energies)
    closest_energy = min(sorted_energies, key=lambda e: abs(e - ie))
    energy_diff = abs(closest_energy - ie)
    if energy_diff < threshold:
        idx = sorted_energies.index(closest_energy)
        if idx + 1 < len(sorted_energies):
            return sorted_energies[idx + 1]
    return closest_energy


def _get_cross_section_from_db(
    database: list[dict],
    mz_index: dict[int, list[int]],
    species_name: str,
    mz: int,
    energy: float,
) -> float:
    indices = mz_index.get(mz, [])
    for idx in indices:
        spec = database[idx]
        if spec.get("species") == species_name or spec.get("name") == species_name:
            energies = spec.get("energies")
            cross_sections = spec.get("cross_sections")
            if energies is not None and cross_sections is not None:
                return _interpolate_cross_section(energies, cross_sections, energy)
    if indices:
        spec = database[indices[0]]
        energies = spec.get("energies")
        cross_sections = spec.get("cross_sections")
        if energies is not None and cross_sections is not None:
            return _interpolate_cross_section(energies, cross_sections, energy)
    return 0.0


def _interpolate_cross_section(
    energies: np.ndarray,
    cross_sections: np.ndarray,
    energy: float,
) -> float:
    energy_values = np.asarray(energies, dtype=float)
    cross_section_values = np.asarray(cross_sections, dtype=float)
    valid = np.isfinite(energy_values) & np.isfinite(cross_section_values)
    energy_values = energy_values[valid]
    cross_section_values = cross_section_values[valid]
    if energy_values.size == 0:
        return 0.0

    order = np.argsort(energy_values)
    energy_values = energy_values[order]
    cross_section_values = cross_section_values[order]

    min_energy = float(energy_values[0])
    max_energy = float(energy_values[-1])
    if min_energy <= energy <= max_energy:
        return float(np.interp(energy, energy_values, cross_section_values))
    if energy > max_energy:
        return float(
            np.interp(
                energy,
                energy_values,
                cross_section_values,
                left=0.0,
                right=float(cross_section_values[-1]),
            )
        )
    if energy_values.size >= 2:
        e1, e2 = energy_values[:2]
        c1, c2 = cross_section_values[:2]
        if e2 > e1:
            slope = (c2 - c1) / (e2 - e1)
            extrapolated = c1 + slope * (energy - e1)
            return float(max(0.0, min(extrapolated, c1)))
    return 0.0


def _get_species_from_database(
    database: list[dict],
    mz_index: dict[int, list[int]],
    species_name: str,
    mz: int | float,
) -> dict | None:
    nominal_mz = int(round(float(mz)))
    for idx in mz_index.get(nominal_mz, []):
        spec = database[idx]
        if spec.get("species") == species_name or spec.get("name") == species_name:
            return spec
    return None


def separate_coexisting_species_signals(
    mz: int,
    species_at_mz: list[dict],
    energy_scan_data: dict[float, dict[float, float]],
    database: list[dict],
    mz_index: dict[int, list[int]],
    target_energy: float | None = None,
) -> tuple[dict[str, dict[float, float]], dict[str, float]]:
    if not species_at_mz or not energy_scan_data:
        return {}, {}

    species_with_ie = [s for s in species_at_mz if s.get("ie") is not None]
    species_with_ie.sort(key=lambda s: s.get("ie", float("inf")))
    if not species_with_ie:
        return {}, {}

    available_energies = sorted(float(e) for e in energy_scan_data)
    pure_signals: dict[str, dict[float, float]] = {}
    pure_energies: dict[str, float] = {}

    for idx, species in enumerate(species_with_ie):
        species_name = str(species["species"])
        species_ie = float(species.get("ie") or 0.0)
        next_ie = (
            float(species_with_ie[idx + 1].get("ie", float("inf")))
            if idx + 1 < len(species_with_ie)
            else float("inf")
        )

        candidates = [e for e in available_energies if species_ie <= e < next_ie]
        if not candidates:
            candidates = [e for e in available_energies if e >= species_ie]
        if not candidates:
            logger.info(
                "物种分离: %s (IE=%.3f eV, m/z=%s) 没有可用能量",
                species_name,
                species_ie,
                mz,
            )
            continue

        pure_energy = select_calc_energy(species_ie, candidates)
        pure_energies[species_name] = pure_energy
        net_signal = {
            float(temp): float(signal)
            for temp, signal in energy_scan_data[pure_energy].items()
        }

        for prev_name, prev_pure_signal in pure_signals.items():
            prev_pure_energy = pure_energies[prev_name]
            prev_db = _get_species_from_database(database, mz_index, prev_name, mz)
            if prev_db is None:
                logger.warning("物种分离: 数据库中未找到 %s", prev_name)
                continue
            prev_energies = prev_db.get("energies")
            prev_cross = prev_db.get("cross_sections")
            if prev_energies is None or prev_cross is None:
                logger.warning("物种分离: %s 没有光电离截面数据", prev_name)
                continue

            sigma_prev_at_prev = _interpolate_cross_section(
                prev_energies, prev_cross, prev_pure_energy
            )
            sigma_prev_at_current = _interpolate_cross_section(
                prev_energies, prev_cross, pure_energy
            )
            if sigma_prev_at_prev <= 0 or sigma_prev_at_current <= 0:
                continue

            ratio = sigma_prev_at_current / sigma_prev_at_prev
            fallback_prev = (
                sum(prev_pure_signal.values()) / len(prev_pure_signal)
                if prev_pure_signal
                else 0.0
            )
            for temp in list(net_signal):
                prev_signal = prev_pure_signal.get(temp, fallback_prev)
                net_signal[temp] = max(0.0, net_signal[temp] - prev_signal * ratio)

        pure_signals[species_name] = net_signal

    if target_energy is None:
        return pure_signals, pure_energies

    if target_energy not in energy_scan_data:
        return pure_signals, pure_energies

    target_result: dict[str, dict[float, float]] = {}
    target_temps = sorted(float(t) for t in energy_scan_data[target_energy])
    for species in species_with_ie:
        species_name = str(species["species"])
        species_ie = float(species.get("ie") or 0.0)
        if species_name not in pure_signals:
            continue
        if target_energy < species_ie:
            target_result[species_name] = {temp: 0.0 for temp in target_temps}
            continue

        species_db = _get_species_from_database(database, mz_index, species_name, mz)
        if species_db is None:
            logger.warning("物种分离: 数据库中未找到 %s", species_name)
            continue
        species_energies = species_db.get("energies")
        species_cross = species_db.get("cross_sections")
        if species_energies is None or species_cross is None:
            logger.warning("物种分离: %s 没有光电离截面数据", species_name)
            continue

        pure_energy = pure_energies[species_name]
        sigma_at_pure = _interpolate_cross_section(
            species_energies, species_cross, pure_energy
        )
        sigma_at_target = _interpolate_cross_section(
            species_energies, species_cross, target_energy
        )
        if sigma_at_pure <= 0:
            continue
        ratio = sigma_at_target / sigma_at_pure
        target_result[species_name] = {
            temp: pure_signals[species_name].get(temp, 0.0) * ratio
            for temp in target_temps
        }

    return target_result, pure_energies


def calc_parent_mole_fraction(
    signal_data: dict[float, float],
    *,
    reference_temperature: float | None = None,
    parent_initial_mf: float = 0.002,
    expansion_coefficients: dict[float, float] | None = None,
) -> dict[float, float]:
    if not signal_data:
        return {}
    if expansion_coefficients is None:
        expansion_coefficients = {}
    if reference_temperature is None:
        reference_temperature = min(signal_data.keys())
    T0 = reference_temperature
    X_T0 = parent_initial_mf
    S_T0 = signal_data.get(T0, 0)
    lambda_T0 = get_expansion_coefficient(T0, expansion_coefficients)
    if S_T0 == 0:
        return {}
    results: dict[float, float] = {}
    for T, S_T in signal_data.items():
        lambda_T = get_expansion_coefficient(T, expansion_coefficients)
        X_T = X_T0 * (S_T / S_T0) * (lambda_T0 / lambda_T)
        results[T] = max(0.0, float(X_T))
    return results


def calc_product_mole_fraction(
    signal_data: dict[float, float],
    *,
    species_mw: float,
    species_mz: int,
    species_name: str,
    ref_mw: float,
    ref_mz: int,
    ref_species_name: str,
    ref_mf_at_tm: float,
    ref_signal_data: dict[float, float] | None = None,
    energy: float = 10.0,
    reference_temperature: float | None = None,
    mass_disc_exponent: float = 0.77897,
    expansion_coefficients: dict[float, float] | None = None,
    database: list[dict] | None = None,
    mz_index: dict[int, list[int]] | None = None,
) -> dict[float, float]:
    if not signal_data:
        return {}
    if expansion_coefficients is None:
        expansion_coefficients = {}
    if database is None:
        database = []
    if mz_index is None:
        mz_index = {}

    D_i = calc_mass_discrimination(species_mw, mass_disc_exponent)
    D_A = calc_mass_discrimination(ref_mw, mass_disc_exponent)

    sigma_i = _get_cross_section_from_db(database, mz_index, species_name, species_mz, energy)
    sigma_A = _get_cross_section_from_db(database, mz_index, ref_species_name, ref_mz, energy)

    if sigma_i == 0:
        return {}

    T_M = reference_temperature
    if T_M is None:
        T_M = max(signal_data.keys())
    if T_M is None:
        return {}
    lambda_TM = get_expansion_coefficient(T_M, expansion_coefficients)

    if ref_signal_data is not None and ref_signal_data:
        S_A_TM = ref_signal_data.get(T_M, 0)
        if S_A_TM == 0:
            return {}
        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            lambda_T = get_expansion_coefficient(T, expansion_coefficients)
            X_i = ref_mf_at_tm * (S_i / S_A_TM) * (sigma_A / sigma_i) * (D_A / D_i) * (lambda_TM / lambda_T)
            results[T] = max(0.0, float(X_i))
        return results

    results = {}
    for T, S_i in signal_data.items():
        lambda_T = get_expansion_coefficient(T, expansion_coefficients)
        X_i = ref_mf_at_tm * (S_i / sigma_i) * (sigma_A / 1.0) * (D_A / D_i) * (lambda_TM / lambda_T)
        results[T] = max(0.0, float(X_i))
    return results


def calc_isomeric_separation(
    signal_low: dict[float, float],
    signal_high: dict[float, float],
    species_list: list[dict],
    *,
    expansion_coefficients: dict[float, float] | None = None,
) -> dict[str, dict[float, float]]:
    if expansion_coefficients is None:
        expansion_coefficients = {}
    common_temps = sorted(set(signal_low.keys()) & set(signal_high.keys()))
    if not common_temps:
        return {}

    results: dict[str, dict[float, float]] = {spec["name"]: {} for spec in species_list}

    for temp in common_temps:
        S_high = signal_high[temp]
        total_sigma_high = sum(spec.get("sigma_high", 0) for spec in species_list)
        total_sigma_low = sum(spec.get("sigma_low", 0) for spec in species_list)

        if total_sigma_low > 0 and total_sigma_high > 0:
            for spec in species_list:
                ratio = spec["sigma_low"] / total_sigma_low
                contrib = S_high * ratio * (spec["sigma_high"] / spec["sigma_low"]) if spec["sigma_low"] > 0 else 0
                results[spec["name"]][temp] = float(contrib)
        else:
            per_species = S_high / len(species_list)
            for spec in species_list:
                results[spec["name"]][temp] = float(per_species)

    return results


def extract_signal_from_temperature_curves(
    temperature_curves: dict[int | float, dict],
    mz: int | float,
    *,
    energy: float | None = None,
) -> dict[float, float]:
    if energy is not None:
        energy_curve = temperature_curves.get(energy)
        if isinstance(energy_curve, dict) and "temperatures" not in energy_curve:
            return extract_signal_from_temperature_curves(energy_curve, mz)

    requested_mz = float(mz)
    nominal_mz = int(round(requested_mz))
    candidates: list[tuple[float, dict]] = []
    for curve_key, candidate in temperature_curves.items():
        if not isinstance(candidate, dict) or "temperatures" not in candidate:
            continue
        candidate_nominal_mz = int(
            round(
                float(
                    candidate.get(
                        "mz_rounded",
                        candidate.get("mz", curve_key),
                    )
                )
            )
        )
        if candidate_nominal_mz != nominal_mz:
            continue
        candidate_exact_mz = float(
            candidate.get(
                "curve_key",
                candidate.get("mz_exact_mean", candidate.get("mz", curve_key)),
            )
        )
        candidates.append((candidate_exact_mz, candidate))

    curve = None
    if requested_mz.is_integer():
        # A nominal mass is no longer a unique signal when multiple resolved
        # peaks occupy it. Never select or sum one of them silently.
        if len(candidates) == 1:
            curve = candidates[0][1]
        elif len(candidates) > 1:
            return {}
    elif candidates:
        candidates.sort(key=lambda item: (abs(item[0] - requested_mz), item[0]))
        best_distance = abs(candidates[0][0] - requested_mz)
        if best_distance <= 0.25:
            if (
                len(candidates) == 1
                or abs(
                    best_distance
                    - abs(candidates[1][0] - requested_mz)
                )
                > 1e-12
            ):
                curve = candidates[0][1]
    if curve is None:
        return {}
    if energy is not None and isinstance(curve, dict):
        by_energy = curve.get("by_energy") or curve.get("energy_curves")
        if isinstance(by_energy, dict):
            nested_curve = by_energy.get(energy)
            if nested_curve is not None:
                return extract_signal_from_temperature_curves({mz: nested_curve}, mz)
    temps = curve.get("temperatures", [])
    areas = curve.get("areas", [])
    if len(temps) != len(areas):
        return {}
    return {float(t): float(a) for t, a in zip(temps, areas)}


def compute_all_mole_fractions(
    temperature_scan_df: pd.DataFrame,
    *,
    settings: MoleFractionSettings,
    database: list[dict] | None = None,
    mz_index: dict[int, list[int]] | None = None,
    product_species: list[dict] | None = None,
) -> pd.DataFrame:
    if database is None:
        database = []
    if mz_index is None:
        mz_index = {}
    if product_species is None:
        product_species = []

    expansion_coefficients = calc_expansion_coefficients(settings.kr_data)

    curves = build_temperature_curves(temperature_scan_df)

    parent_signal = extract_signal_from_temperature_curves(
        curves, settings.parent_mz
    )
    parent_mf = calc_parent_mole_fraction(
        parent_signal,
        reference_temperature=settings.reference_temperature,
        parent_initial_mf=settings.parent_initial_mf,
        expansion_coefficients=expansion_coefficients,
    )

    all_results: dict[str, dict[float, float]] = {}
    if parent_mf:
        all_results[f"parent_mz{settings.parent_mz}"] = parent_mf

    ref_signal = extract_signal_from_temperature_curves(
        curves, settings.reference_species_mz or settings.parent_mz
    )

    for prod in product_species:
        prod_mz = int(prod.get("mz", 0))
        prod_signal_mz = float(
            prod.get(
                "curve_key",
                prod.get("mz_exact_mean", prod.get("mz_exact", prod_mz)),
            )
        )
        prod_name = str(prod.get("species") or prod.get("name") or f"m/z={prod_mz}")
        prod_mw = prod.get("mw")
        if prod_mw is None:
            formula = prod.get("formula")
            if formula:
                try:
                    prod_mw = float(formula_nominal_mass(str(formula)))
                except ValueError:
                    prod_mw = float(prod_mz)
            else:
                prod_mw = float(prod_mz)
        prod_mw = float(prod_mw)

        prod_signal = extract_signal_from_temperature_curves(curves, prod_signal_mz)

        prod_mf = calc_product_mole_fraction(
            prod_signal,
            species_mw=prod_mw,
            species_mz=prod_mz,
            species_name=prod_name,
            ref_mw=float(settings.reference_species_mz or settings.parent_mz),
            ref_mz=settings.reference_species_mz or settings.parent_mz,
            ref_species_name=str(prod.get("ref_species") or prod.get("ref_name") or ""),
            ref_mf_at_tm=settings.reference_species_mf_at_tm,
            ref_signal_data=ref_signal,
            energy=settings.photon_energy,
            reference_temperature=settings.reference_temperature,
            mass_disc_exponent=settings.mass_disc_exponent,
            expansion_coefficients=expansion_coefficients,
            database=database,
            mz_index=mz_index,
        )
        if prod_mf:
            all_results[prod_name] = prod_mf

    if not all_results:
        return pd.DataFrame()

    all_temps: set[float] = set()
    for res in all_results.values():
        all_temps.update(res.keys())
    all_temps_sorted = sorted(all_temps)

    data: dict[str, list] = {"temperature": all_temps_sorted}
    for name, res in all_results.items():
        data[name] = [res.get(t, 0.0) for t in all_temps_sorted]

    return pd.DataFrame(data)


class MoleFractionCalculator:
    def __init__(self) -> None:
        self.database: list[dict] = []
        self.mz_index: dict[int, list[int]] = {}
        self.expansion_coefficients: dict[float, float] = {}
        self.mass_disc_exponent: float = 0.77897
        self.parent_mz: float | None = None
        self.parent_initial_mf: float = 0.002
        self.reference_temperature: float | None = None
        self.reference_species_mz: float | None = None
        self.reference_species_tm: float | None = None
        self.identified_species: dict[str, dict] = {}
        self.calibrated_mf: dict[str, dict[float, float]] = {}
        self.parent_mf_results: dict[float, float] = {}

    def load_species_database(self, file_path: str | Path) -> int:
        file_path = str(file_path)
        if file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self.database = data["database"]
            self.mz_index = data["mz_index"]
        elif file_path.endswith((".xlsx", ".xls")):
            self._parse_excel_database(file_path)
        elif file_path.endswith(".csv"):
            self._parse_csv_database(file_path)
        elif file_path.endswith((".sqlite", ".sqlite3", ".db")):
            self.database, self.mz_index = load_species_database(file_path)
        return len(self.database)

    def _parse_excel_database(self, file_path: str | Path) -> None:
        df = pd.read_excel(file_path, header=None)
        self._parse_dataframe_database(df)

    def _parse_csv_database(self, file_path: str | Path) -> None:
        df = pd.read_csv(file_path, header=None)
        self._parse_dataframe_database(df)

    def _parse_dataframe_database(self, df: pd.DataFrame) -> None:
        self.database = []
        self.mz_index = {}
        n_rows, n_cols = df.shape
        i = 0
        while i < n_rows - 1:
            try:
                col5_val = df.iloc[i, 5] if pd.notna(df.iloc[i, 5]) else ""
                if str(col5_val).strip() == "Energy(eV)":
                    header_row = i
                    data_row = i + 1
                    mz_raw = df.iloc[header_row, 0]
                    if pd.isna(mz_raw):
                        i += 1
                        continue
                    try:
                        mz = int(round(float(str(mz_raw).strip())))
                    except ValueError:
                        i += 1
                        continue
                    species_symbol = (
                        str(df.iloc[header_row, 2]).strip()
                        if pd.notna(df.iloc[header_row, 2])
                        else ""
                    )
                    ie_raw = df.iloc[header_row, 4]
                    ie = None
                    if pd.notna(ie_raw):
                        try:
                            ie = float(str(ie_raw).strip())
                        except ValueError:
                            pass
                    energy_values: list[float] = []
                    for j in range(6, n_cols):
                        energy_raw = df.iloc[header_row, j]
                        if pd.isna(energy_raw):
                            continue
                        try:
                            energy_values.append(float(str(energy_raw).strip()))
                        except ValueError:
                            continue
                    if not energy_values:
                        i += 1
                        continue
                    species_name = (
                        str(df.iloc[data_row, 2]).strip()
                        if pd.notna(df.iloc[data_row, 2])
                        else ""
                    )
                    if not species_name or species_name == "nan":
                        species_name = species_symbol
                    smiles = ""
                    if n_cols > 3 and pd.notna(df.iloc[header_row, 3]):
                        smiles = str(df.iloc[header_row, 3]).strip()
                    if smiles == "nan":
                        smiles = ""
                    cross_sections: list[float] = []
                    for j in range(6, n_cols):
                        if j - 6 >= len(energy_values):
                            break
                        cs_raw = df.iloc[data_row, j]
                        if pd.isna(cs_raw):
                            cross_sections.append(0.0)
                        else:
                            try:
                                cross_sections.append(float(str(cs_raw).strip()))
                            except ValueError:
                                cross_sections.append(0.0)
                    if len(cross_sections) > 0:
                        idx = len(self.database)
                        self.database.append(
                            {
                                "mz": mz,
                                "species": species_name,
                                "ie": ie,
                                "smiles": smiles,
                                "energies": np.array(energy_values, dtype=np.float32),
                                "cross_sections": np.array(
                                    cross_sections, dtype=np.float32
                                ),
                            }
                        )
                        if mz not in self.mz_index:
                            self.mz_index[mz] = []
                        self.mz_index[mz].append(idx)
                    i += 2
                else:
                    i += 1
            except Exception:
                i += 1
                continue

    def get_species_by_mz(self, mz: int | float) -> list[dict]:
        nominal_mz = int(round(float(mz)))
        return [
            spec
            for spec in self.database
            if int(round(float(spec["mz"]))) == nominal_mz
        ]

    def get_cross_section_at_energy(self, species: dict, energy: float) -> float:
        return _interpolate_cross_section(
            species["energies"], species["cross_sections"], energy
        )

    @staticmethod
    def calc_mass_discrimination(
        molecular_weight: float, exponent: float = 0.77897
    ) -> float:
        return float((molecular_weight / 30.0) ** exponent)

    def calc_expansion_coefficients(
        self, kr_data: dict[float, float]
    ) -> dict[float, float]:
        if not kr_data:
            return {}
        ref_temp = min(kr_data.keys())
        ref_signal = kr_data[ref_temp]
        self.expansion_coefficients = {}
        if ref_signal == 0:
            return {}
        for temp, signal in kr_data.items():
            self.expansion_coefficients[temp] = signal / ref_signal
        return self.expansion_coefficients

    def get_expansion_coefficient(self, temperature: float) -> float:
        if temperature in self.expansion_coefficients:
            return self.expansion_coefficients[temperature]
        temps = sorted(self.expansion_coefficients.keys())
        if len(temps) < 2:
            return 1.0
        coeffs = [self.expansion_coefficients[t] for t in temps]
        return float(np.interp(temperature, temps, coeffs))

    def calc_parent_mole_fraction(
        self,
        signal_data: dict[float, float],
        energy: float | None = None,
    ) -> dict[float, float]:
        if not signal_data:
            return {}

        if self.reference_temperature is None:
            self.reference_temperature = min(signal_data.keys())

        T0 = self.reference_temperature

        if T0 not in signal_data:
            return {}

        X_T0 = self.parent_initial_mf
        S_T0 = signal_data.get(T0, 0)
        lambda_T0 = self.get_expansion_coefficient(T0)

        if S_T0 == 0:
            return {}

        results: dict[float, float] = {}
        for T, S_T in signal_data.items():
            lambda_T = self.get_expansion_coefficient(T)
            X_T = X_T0 * (S_T / S_T0) * (lambda_T0 / lambda_T)
            results[T] = max(0.0, float(X_T))

        self.parent_mf_results = results
        return results

    def calc_parent_mf_by_reference(
        self,
        signal_data: dict[float, float],
        species_mw: float,
        species_name: str,
        ref_mw: float,
        ref_species_name: str,
        ref_signal_data: dict[float, float],
        ref_mf_at_tm: float,
        calc_energy: float,
        ref_energy: float,
    ) -> dict[float, float] | None:
        if not signal_data or not ref_signal_data or ref_mf_at_tm == 0:
            return None

        species_obj = _get_species_from_database(
            self.database, self.mz_index, species_name, int(round(species_mw))
        )
        ref_obj = _get_species_from_database(
            self.database, self.mz_index, ref_species_name, int(round(ref_mw))
        )
        if species_obj is None:
            species_list = self.get_species_by_mz(int(round(species_mw)))
            species_obj = species_list[0] if species_list else None
        if ref_obj is None:
            ref_species_list = self.get_species_by_mz(int(round(ref_mw)))
            ref_obj = ref_species_list[0] if ref_species_list else None
        if species_obj is None or ref_obj is None:
            return None

        sigma_i = self.get_cross_section_at_energy(species_obj, calc_energy)
        sigma_ref = self.get_cross_section_at_energy(ref_obj, ref_energy)
        D_i = self.calc_mass_discrimination(species_mw, self.mass_disc_exponent)
        D_ref = self.calc_mass_discrimination(ref_mw, self.mass_disc_exponent)

        T_M = self.reference_temperature
        if T_M is None:
            T_M = max(ref_signal_data.keys()) if ref_signal_data else None
        if T_M is None:
            return None

        lambda_TM = self.get_expansion_coefficient(T_M)
        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            S_ref_T = ref_signal_data.get(T)
            if not S_ref_T:
                continue
            lambda_T = self.get_expansion_coefficient(T)
            cross_section_ratio = sigma_ref / sigma_i if sigma_i > 0 and sigma_ref > 0 else 1.0
            mass_disc_ratio = D_ref / D_i if D_i > 0 and D_ref > 0 else 1.0
            X_i = (
                ref_mf_at_tm
                * (S_i / S_ref_T)
                * cross_section_ratio
                * mass_disc_ratio
                * (lambda_TM / lambda_T)
            )
            results[T] = max(0.0, float(X_i))

        return results

    def calc_product_mf_with_ref_signal(
        self,
        species_mw: float,
        species_mz: int,
        species_name: str,
        signal_data: dict[float, float],
        ref_mw: float,
        ref_mz: int,
        ref_species_name: str,
        ref_signal_data: dict[float, float],
        ref_mf_at_tm: float,
        energy: float,
    ) -> dict[float, float]:
        D_i = self.calc_mass_discrimination(species_mw, self.mass_disc_exponent)
        D_A = self.calc_mass_discrimination(ref_mw, self.mass_disc_exponent)

        species_list = self.get_species_by_mz(species_mz)
        ref_species_list = self.get_species_by_mz(ref_mz)

        species_obj = None
        for s in species_list:
            if s["species"] == species_name:
                species_obj = s
                break
        if species_obj is None and species_list:
            species_obj = species_list[0]

        ref_obj = None
        for s in ref_species_list:
            if s["species"] == ref_species_name:
                ref_obj = s
                break
        if ref_obj is None and ref_species_list:
            ref_obj = ref_species_list[0]

        if species_obj is None or ref_obj is None:
            return {}

        sigma_i = self.get_cross_section_at_energy(species_obj, energy)
        sigma_A = self.get_cross_section_at_energy(ref_obj, energy)

        if sigma_i == 0:
            return {}

        T_M = self.reference_temperature
        if T_M is None:
            T_M = max(ref_signal_data.keys())
        S_A_TM = ref_signal_data.get(T_M, 0)
        if S_A_TM == 0:
            return {}
        lambda_TM = self.get_expansion_coefficient(T_M)

        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            lambda_T = self.get_expansion_coefficient(T)
            X_i = (
                ref_mf_at_tm
                * (S_i / S_A_TM)
                * (sigma_A / sigma_i)
                * (D_A / D_i)
                * (lambda_TM / lambda_T)
            )
            results[T] = max(0.0, float(X_i))
        return results

    def calibrate_species_mf(
        self, species_name: str, mf_data: dict[float, float]
    ) -> None:
        self.calibrated_mf[species_name] = mf_data

    def get_calibrated_mf(self, species_name: str) -> dict[float, float]:
        return self.calibrated_mf.get(species_name, {})

    def get_species_with_ionization_energy(
        self,
        mz: int,
        min_energy: float | None = None,
        max_energy: float | None = None,
    ) -> list[dict]:
        species_list = self.get_species_by_mz(mz)
        if min_energy is None and max_energy is None:
            return species_list
        filtered: list[dict] = []
        for spec in species_list:
            ie = spec.get("ie")
            if ie is None:
                continue
            if min_energy is not None and ie < min_energy:
                continue
            if max_energy is not None and ie > max_energy:
                continue
            filtered.append(spec)
        return filtered

    def calc_product_mf_auto(
        self,
        mz: int | float,
        species: dict,
        ref_mz: int | float,
        ref_mw: float,
        ref_energy: float,
        ref_signal_data: dict[float, float],
        ref_mf_at_tm: float,
        signal_data: dict[float, float],
        calc_energy: float | None = None,
        available_energies: list[float] | None = None,
        ref_species_name: str | None = None,
    ) -> dict[float, float] | None:
        ie = species.get("ie", 0) or 0
        species_name = species["species"]

        if calc_energy is None:
            usable_energies = [
                energy for energy in (available_energies or []) if energy >= ie
            ]
            best_energy = select_calc_energy(ie, usable_energies) if usable_energies else None
        else:
            best_energy = calc_energy

        if best_energy is None:
            return None

        if not signal_data:
            return None

        species_obj = _get_species_from_database(
            self.database, self.mz_index, species_name, mz
        )
        species_list = self.get_species_by_mz(mz)
        if species_obj is None and species_list:
            species_obj = species_list[0]

        ref_species_name = (
            ref_species_name
            or species.get("ref_species")
            or species.get("ref_name")
            or ""
        )
        ref_obj = None
        if ref_species_name:
            ref_obj = _get_species_from_database(
                self.database, self.mz_index, str(ref_species_name), ref_mz
            )
        ref_species_list = self.get_species_by_mz(ref_mz)
        if ref_obj is None and ref_species_list:
            ref_obj = ref_species_list[0]

        if species_obj is None or ref_obj is None:
            return None

        sigma_i = self.get_cross_section_at_energy(species_obj, best_energy)
        sigma_A = self.get_cross_section_at_energy(ref_obj, ref_energy)

        if sigma_i == 0 or sigma_A == 0:
            return None

        D_i = self.calc_mass_discrimination(
            float(species.get("mw") or species.get("mz") or round(float(mz))),
            self.mass_disc_exponent,
        )
        D_A = self.calc_mass_discrimination(ref_mw, self.mass_disc_exponent)

        T_M = self.reference_temperature
        if T_M is None:
            T_M = max(ref_signal_data.keys()) if ref_signal_data else None
        if T_M is None:
            return None

        S_A_TM = ref_signal_data.get(T_M, 0)
        if S_A_TM == 0:
            return None

        lambda_TM = self.get_expansion_coefficient(T_M)

        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            lambda_T = self.get_expansion_coefficient(T)
            X_i = (
                ref_mf_at_tm
                * (S_i / S_A_TM)
                * (sigma_A / sigma_i)
                * (D_A / D_i)
                * (lambda_TM / lambda_T)
            )
            results[T] = max(0.0, float(X_i))

        return results

    def calc_multi_species_mf_auto(
        self,
        mz: int | float,
        species_list: list[dict],
        energies: list[float],
        ref_mz: int | float,
        ref_mw: float,
        ref_energy: float,
        ref_signal_data: dict[float, float],
        ref_mf_at_tm: float,
        temperature_curves: dict[int | float, dict],
    ) -> dict:
        result: dict = {}
        warnings: list[str] = []

        if not energies:
            warnings.append(f"质量数 {mz}: 没有可用能量数据")
            result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
            return result

        sorted_energies = sorted(energies)
        energy_boundaries = [0] + sorted_energies

        species_groups: dict[int, list[dict]] = {}
        for s in species_list:
            ie = s.get("ie") if s.get("ie") is not None else float("inf")
            group_idx = None
            for i in range(len(energy_boundaries) - 1):
                lower = energy_boundaries[i]
                upper = energy_boundaries[i + 1]
                if lower < ie <= upper:
                    group_idx = i
                    break
            if group_idx is None:
                group_idx = len(energy_boundaries) - 1

            if group_idx not in species_groups:
                species_groups[group_idx] = []
            species_groups[group_idx].append(s)

        resolvable_species: list[dict] = []
        for group_idx, group in species_groups.items():
            if len(group) > 1:
                group.sort(
                    key=lambda x: x.get("contribution", 0) or 0, reverse=True
                )
                best = group[0]
                resolvable_species.append(best)
                ignored_names = [s["species"] for s in group[1:]]
                upper_bound = (
                    f"{energy_boundaries[group_idx + 1]:.1f}"
                    if group_idx + 1 < len(energy_boundaries)
                    else "∞"
                )
                warnings.append(
                    f"质量数 {mz}: 电离能区间 ({energy_boundaries[group_idx]:.1f}-{upper_bound} eV) "
                    f"内存在 {len(group)} 个物种 ({', '.join(s['species'] for s in group)})，"
                    f"无法通过能量扫描分离，仅计算匹配系数最高的 {best['species']}，"
                    f"忽略了 {', '.join(ignored_names)}"
                )
            else:
                resolvable_species.append(group[0])

        resolvable_species.sort(
            key=lambda x: x.get("ie") if x.get("ie") is not None else float("inf")
        )

        if len(resolvable_species) <= 1:
            for species in resolvable_species:
                ie = species.get("ie") if species.get("ie") is not None else float("inf")
                if ie == float("inf"):
                    continue
                usable_energies = [e for e in sorted_energies if e >= ie]
                if not usable_energies:
                    continue
                calc_e = select_calc_energy(float(ie), usable_energies)
                signal_data = extract_signal_from_temperature_curves(
                    temperature_curves, mz, energy=calc_e
                )
                if not signal_data:
                    continue
                mf = self.calc_product_mf_from_signal(
                    mz,
                    species,
                    calc_e,
                    signal_data,
                    ref_mz,
                    ref_mw,
                    ref_energy,
                    ref_signal_data,
                    ref_mf_at_tm,
                )
                if mf:
                    result[(mz, species["species"], calc_e)] = mf
            if warnings:
                result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
            return result

        try:
            energy_scan_data: dict[float, dict[float, float]] = {}
            for energy in sorted_energies:
                signal_data = extract_signal_from_temperature_curves(
                    temperature_curves, mz, energy=energy
                )
                if signal_data:
                    energy_scan_data[energy] = signal_data
            if not energy_scan_data:
                warnings.append(f"质量数 {mz}: 没有可用的温度扫描信号")
                result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
                return result

            species_at_mz = [
                {"species": species["species"], "ie": species.get("ie")}
                for species in resolvable_species
            ]
            separated, pure_energies = separate_coexisting_species_signals(
                mz=int(round(float(mz))),
                species_at_mz=species_at_mz,
                energy_scan_data=energy_scan_data,
                database=self.database,
                mz_index=self.mz_index,
            )

            for species in resolvable_species:
                species_name = species["species"]
                if species_name not in separated:
                    continue
                calc_e = pure_energies.get(species_name)
                if calc_e is None:
                    ie = species.get("ie") if species.get("ie") is not None else float("inf")
                    usable_energies = [e for e in sorted_energies if e >= ie]
                    if not usable_energies:
                        continue
                    calc_e = select_calc_energy(float(ie), usable_energies)
                mf = self.calc_product_mf_from_signal(
                    mz,
                    species,
                    calc_e,
                    separated[species_name],
                    ref_mz,
                    ref_mw,
                    ref_energy,
                    ref_signal_data,
                    ref_mf_at_tm,
                )
                if mf:
                    result[(mz, species_name, calc_e)] = mf
        except Exception as exc:
            logger.warning("质量数 %s 多物种信号分离失败: %s", mz, exc)
            for species in resolvable_species:
                ie = species.get("ie") if species.get("ie") is not None else float("inf")
                if ie == float("inf"):
                    continue
                usable_energies = [e for e in sorted_energies if e >= ie]
                if not usable_energies:
                    continue
                calc_e = select_calc_energy(float(ie), usable_energies)
                signal_data = extract_signal_from_temperature_curves(
                    temperature_curves, mz, energy=calc_e
                )
                if not signal_data:
                    continue
                mf = self.calc_product_mf_from_signal(
                    mz,
                    species,
                    calc_e,
                    signal_data,
                    ref_mz,
                    ref_mw,
                    ref_energy,
                    ref_signal_data,
                    ref_mf_at_tm,
                )
                if mf:
                    result[(mz, species["species"], calc_e)] = mf

        if len(species_list) > len(
            set(k[1] for k in result if k != "warning")
        ) + sum(len(g) - 1 for g in species_groups.values() if len(g) > 1):
            warnings.append(f"质量数 {mz}: 部分可分辨物种未能计算摩尔分数")

        if warnings:
            result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"

        return result

    def calc_product_mf_from_signal(
        self,
        mz: int | float,
        species: dict,
        energy: float,
        signal_data: dict[float, float],
        ref_mz: int | float,
        ref_mw: float,
        ref_energy: float,
        ref_signal_data: dict[float, float],
        ref_mf_at_tm: float,
        ref_species_name: str | None = None,
    ) -> dict[float, float] | None:
        species_name = species["species"]

        species_obj = _get_species_from_database(
            self.database, self.mz_index, species_name, mz
        )
        species_list = self.get_species_by_mz(mz)
        if species_obj is None and species_list:
            species_obj = species_list[0]

        ref_obj = None
        ref_species_name = (
            ref_species_name
            or species.get("ref_species")
            or species.get("ref_name")
            or ""
        )
        if ref_species_name:
            ref_obj = _get_species_from_database(
                self.database, self.mz_index, str(ref_species_name), ref_mz
            )
        ref_species_list = self.get_species_by_mz(ref_mz)
        if ref_obj is None and ref_species_list:
            ref_obj = ref_species_list[0]

        if species_obj is None or ref_obj is None:
            return None

        sigma_i = self.get_cross_section_at_energy(species_obj, energy)
        sigma_A = self.get_cross_section_at_energy(ref_obj, ref_energy)

        if sigma_i == 0 or sigma_A == 0:
            return None

        D_i = self.calc_mass_discrimination(
            float(species.get("mw") or species.get("mz") or round(float(mz))),
            self.mass_disc_exponent,
        )
        D_A = self.calc_mass_discrimination(ref_mw, self.mass_disc_exponent)

        T_M = self.reference_temperature
        if T_M is None:
            T_M = max(ref_signal_data.keys()) if ref_signal_data else None
        if T_M is None:
            return None

        S_A_TM = ref_signal_data.get(T_M, 0)
        if S_A_TM == 0:
            return None

        lambda_TM = self.get_expansion_coefficient(T_M)

        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            if S_i <= 0:
                results[T] = 0.0
                continue
            lambda_T = self.get_expansion_coefficient(T)
            X_i = (
                ref_mf_at_tm
                * (S_i / S_A_TM)
                * (sigma_A / sigma_i)
                * (D_A / D_i)
                * (lambda_TM / lambda_T)
            )
            results[T] = max(0.0, float(X_i))

        return results

    def calc_auto_mole_fractions(
        self,
        pie_species_data: list[dict],
        temperature_curves: dict[int | float, dict],
        parent_mf_results: dict[float, float] | None = None,
        available_energies: list[float] | None = None,
        parent_energy: float = 10.0,
    ) -> tuple[dict[tuple[float, str, float], dict[float, float]], list[str]]:
        warnings: list[str] = []

        if not pie_species_data:
            return {}, ["没有PIE鉴定结果"]

        if not self.expansion_coefficients:
            return {}, ["请先计算膨胀系数"]

        if parent_mf_results is None:
            parent_mf_results = self.parent_mf_results
        if not parent_mf_results:
            return {}, ["请先计算母体摩尔分数"]

        species_by_mz: dict[float, list[dict]] = {}
        for d in pie_species_data:
            mz = float(
                d.get(
                    "curve_key",
                    d.get("mz_exact_mean", d.get("mz_exact", d["mz"])),
                )
            )
            if mz not in species_by_mz:
                species_by_mz[mz] = []
            species_by_mz[mz].append(d)

        all_energies = sorted(available_energies) if available_energies else [parent_energy]

        all_species_mf: dict[tuple[float, str, float], dict[float, float]] = {}

        parent_mz = self.parent_mz
        if parent_mz is None:
            return {}, ["请先设置母体质量数"]

        parent_mw = float(round(parent_mz))
        parent_signal = extract_signal_from_temperature_curves(
            temperature_curves, parent_mz
        )
        reference_temperature = self.reference_temperature
        if reference_temperature not in parent_mf_results:
            common_temperatures = sorted(set(parent_signal) & set(parent_mf_results))
            if common_temperatures:
                reference_temperature = common_temperatures[-1]
            elif parent_mf_results:
                reference_temperature = sorted(parent_mf_results)[-1]
        if self.reference_temperature != reference_temperature:
            self.reference_temperature = reference_temperature
        parent_mf_at_tm = parent_mf_results.get(reference_temperature, 0)

        parent_species_mz = float(parent_mz)
        parent_candidates = sorted(
            (
                abs(float(candidate_mz) - float(parent_mz)),
                float(candidate_mz),
            )
            for candidate_mz in species_by_mz
            if int(round(float(candidate_mz))) == int(round(float(parent_mz)))
        )
        if float(parent_mz).is_integer():
            if len(parent_candidates) == 1:
                parent_species_mz = parent_candidates[0][1]
        elif parent_candidates and parent_candidates[0][0] <= 0.25:
            if (
                len(parent_candidates) == 1
                or abs(parent_candidates[0][0] - parent_candidates[1][0]) > 1e-12
            ):
                parent_species_mz = parent_candidates[0][1]

        for mz, sp_list in species_by_mz.items():
            if mz == parent_species_mz:
                all_species_mf[(mz, "母体", parent_energy)] = parent_mf_results
                continue

            if len(sp_list) == 1:
                species = sp_list[0]
                ie = species.get("ie", 0) or 0
                usable_energies = [e for e in all_energies if e >= ie]
                calc_energies = usable_energies[:2]

                if not calc_energies:
                    warnings.append(
                        f"质量数 {mz} 物种 {species['species']}: "
                        f"没有高于电离能({ie:.2f} eV)的能量数据"
                    )
                    continue

                signal_data = extract_signal_from_temperature_curves(
                    temperature_curves, mz
                )
                for energy in calc_energies:
                    mf = self.calc_product_mf_auto(
                        mz,
                        species,
                        parent_mz,
                        parent_mw,
                        parent_energy,
                        parent_signal,
                        parent_mf_at_tm,
                        signal_data=signal_data,
                        calc_energy=energy,
                    )
                    if mf:
                        all_species_mf[(mz, species["species"], energy)] = mf
            else:
                sp_list_sorted = sorted(
                    sp_list,
                    key=lambda x: x.get("ie")
                    if x.get("ie") is not None
                    else float("inf"),
                )
                result = self.calc_multi_species_mf_auto(
                    mz,
                    sp_list_sorted,
                    all_energies,
                    parent_mz,
                    parent_mw,
                    parent_energy,
                    parent_signal,
                    parent_mf_at_tm,
                    temperature_curves,
                )
                if result:
                    for key, mf in result.items():
                        if key != "warning":
                            all_species_mf[key] = mf
                    if "warning" in result:
                        warnings.append(result["warning"])

        return all_species_mf, warnings


def calc_auto_mole_fractions(
    calculator: MoleFractionCalculator,
    pie_species_data: list[dict],
    temperature_curves: dict[int | float, dict],
    parent_mf_results: dict[float, float],
    available_energies: list[float] | None = None,
    parent_energy: float = 10.0,
) -> tuple[dict[tuple[float, str, float], dict[float, float]], list[str]]:
    return calculator.calc_auto_mole_fractions(
        pie_species_data,
        temperature_curves,
        parent_mf_results,
        available_energies,
        parent_energy,
    )
