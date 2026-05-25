from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import CONFIG_ROOT, project_path
from .isotope import formula_nominal_mass
from .pie_analysis import load_species_database
from .temperature_scan import build_temperature_curves


DEFAULT_MOLE_FRACTION_CONFIG = CONFIG_ROOT / "mole_fraction.yaml"

MASS_DISCRIMINATION_PRESETS = {
    "760 Torr / 80μm": 0.77897,
    "5 Torr / 400μm (Catalysis)": 0.36267,
    "30 Torr (Catalysis)": 0.75148,
    "30 Torr / 350μm (Catalysis)": 0.75148,
    "150 Torr (Combustion)": 0.76155,
    "150 Torr / 150μm (Chemistry)": 0.76155,
}


@dataclass
class MoleFractionSettings:
    mass_disc_exponent: float = 0.77897
    parent_mz: int = 128
    parent_initial_mf: float = 0.002
    reference_temperature: float | None = None
    reference_species_mz: int | None = None
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
        settings_data["expansion_factors"] = {
            float(key): float(value)
            for key, value in (settings_data.get("expansion_factors") or {}).items()
        }
    if "kr_data" in settings_data:
        settings_data = settings_data.copy()
        settings_data["kr_data"] = {
            float(key): float(value)
            for key, value in (settings_data.get("kr_data") or {}).items()
        }
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
    config_path = project_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = asdict(settings)
    data["expansion_factors"] = {
        float(key): float(value) for key, value in settings.expansion_factors.items()
    }
    data["kr_data"] = {
        float(key): float(value) for key, value in settings.kr_data.items()
    }
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
    if temperature in expansion_coefficients:
        return expansion_coefficients[temperature]
    temps = sorted(expansion_coefficients.keys())
    if len(temps) < 2:
        return 1.0
    coeffs = [expansion_coefficients[t] for t in temps]
    return float(np.interp(temperature, temps, coeffs))


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
        if spec["name"] == species_name:
            energies = spec.get("energies")
            cross_sections = spec.get("cross_sections")
            if energies is not None and cross_sections is not None:
                return float(
                    np.interp(energy, energies, cross_sections, left=0, right=0)
                )
    if indices:
        spec = database[indices[0]]
        energies = spec.get("energies")
        cross_sections = spec.get("cross_sections")
        if energies is not None and cross_sections is not None:
            return float(
                np.interp(energy, energies, cross_sections, left=0, right=0)
            )
    return 0.0


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
    reference_species_tm: float | None = None,
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

    T_M = reference_species_tm
    if T_M is None:
        T_M = max(signal_data.keys())
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
    temperature_curves: dict[int, dict],
    mz: int,
    *,
    energy: float | None = None,
) -> dict[float, float]:
    curve = temperature_curves.get(mz)
    if curve is None:
        return {}
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
        prod_mz = prod.get("mz", 0)
        prod_name = prod.get("name", f"m/z={prod_mz}")
        prod_mw = prod.get("mw", float(prod_mz))

        prod_signal = extract_signal_from_temperature_curves(curves, prod_mz)

        prod_mf = calc_product_mole_fraction(
            prod_signal,
            species_mw=prod_mw,
            species_mz=prod_mz,
            species_name=prod_name,
            ref_mw=float(settings.reference_species_mz or settings.parent_mz),
            ref_mz=settings.reference_species_mz or settings.parent_mz,
            ref_species_name=prod.get("ref_name", ""),
            ref_mf_at_tm=settings.reference_species_mf_at_tm,
            ref_signal_data=ref_signal,
            energy=settings.photon_energy,
            reference_species_tm=settings.reference_species_tm,
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
