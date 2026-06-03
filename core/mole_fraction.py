from __future__ import annotations

import pickle
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
        if spec.get("species") == species_name or spec.get("name") == species_name:
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
        self.parent_mz: int | None = None
        self.parent_initial_mf: float = 0.002
        self.reference_temperature: float | None = None
        self.reference_species_mz: int | None = None
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

    def get_species_by_mz(self, mz: int) -> list[dict]:
        return [spec for spec in self.database if spec["mz"] == mz]

    def get_cross_section_at_energy(self, species: dict, energy: float) -> float:
        return float(
            np.interp(
                energy, species["energies"], species["cross_sections"], left=0, right=0
            )
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
            available_temps = sorted(signal_data.keys())
            closest_temp = min(available_temps, key=lambda t: abs(t - T0))
            self.reference_temperature = closest_temp
            T0 = closest_temp

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
        mz: int,
        species: dict,
        ref_mz: int,
        ref_mw: float,
        ref_energy: float,
        ref_signal_data: dict[float, float],
        ref_mf_at_tm: float,
        signal_data: dict[float, float],
        calc_energy: float | None = None,
        available_energies: list[float] | None = None,
    ) -> dict[float, float] | None:
        ie = species.get("ie", 0) or 0
        species_name = species["species"]

        if calc_energy is None:
            best_energy = None
            if available_energies:
                for energy in sorted(available_energies, reverse=True):
                    if energy >= ie:
                        best_energy = energy
                        break
        else:
            best_energy = calc_energy

        if best_energy is None:
            return None

        if not signal_data:
            return None

        species_obj = None
        for s in self.get_species_by_mz(mz):
            if s["species"] == species_name:
                species_obj = s
                break
        if species_obj is None and self.get_species_by_mz(mz):
            species_obj = self.get_species_by_mz(mz)[0]

        ref_obj = None
        for s in self.get_species_by_mz(ref_mz):
            ref_obj = s
            break
        if ref_obj is None and self.get_species_by_mz(ref_mz):
            ref_obj = self.get_species_by_mz(ref_mz)[0]

        if species_obj is None or ref_obj is None:
            return None

        sigma_i = self.get_cross_section_at_energy(species_obj, best_energy)
        sigma_A = self.get_cross_section_at_energy(ref_obj, ref_energy)

        if sigma_i == 0 or sigma_A == 0:
            return None

        D_i = self.calc_mass_discrimination(mz, self.mass_disc_exponent)
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
        mz: int,
        species_list: list[dict],
        energies: list[float],
        ref_mz: int,
        ref_mw: float,
        ref_energy: float,
        ref_signal_data: dict[float, float],
        ref_mf_at_tm: float,
        temperature_curves: dict[int, dict],
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

        mf_results: dict[tuple, dict[float, float]] = {}

        for species in resolvable_species:
            ie = species.get("ie") if species.get("ie") is not None else float("inf")
            if ie == float("inf"):
                continue

            usable_energies = [e for e in sorted_energies if e >= ie]
            calc_energies = usable_energies[:2]

            if not calc_energies:
                continue

            for calc_e in calc_energies:
                signal_data = extract_signal_from_temperature_curves(
                    temperature_curves, mz
                )
                if not signal_data:
                    continue

                net_signal = dict(signal_data)

                for prev_key, prev_mf in mf_results.items():
                    prev_name = prev_key[1]
                    prev_energy = prev_key[2]
                    prev_obj = None
                    for s in self.get_species_by_mz(mz):
                        if s["species"] == prev_name:
                            prev_obj = s
                            break
                    if prev_obj is None:
                        continue

                    prev_ie = prev_obj.get("ie", 0) or 0
                    if prev_ie > calc_e:
                        continue

                    sigma_prev_at_calc = self.get_cross_section_at_energy(
                        prev_obj, calc_e
                    )
                    sigma_prev_at_prev = self.get_cross_section_at_energy(
                        prev_obj, prev_energy
                    )

                    if sigma_prev_at_prev > 0 and sigma_prev_at_calc > 0:
                        ratio = sigma_prev_at_calc / sigma_prev_at_prev
                        for T in net_signal:
                            if T in prev_mf:
                                contribution = prev_mf[T] * ratio
                                net_signal[T] = max(
                                    0, net_signal.get(T, 0) - contribution
                                )

                mf = self.calc_product_mf_from_signal(
                    mz,
                    species,
                    calc_e,
                    net_signal,
                    ref_mz,
                    ref_mw,
                    ref_energy,
                    ref_signal_data,
                    ref_mf_at_tm,
                )
                if mf:
                    mf_results[(mz, species["species"], calc_e)] = mf

        if len(species_list) > len(
            set(k[1] for k in mf_results.keys())
        ) + sum(len(g) - 1 for g in species_groups.values() if len(g) > 1):
            warnings.append(f"质量数 {mz}: 部分可分辨物种未能计算摩尔分数")

        result.update(mf_results)
        if warnings:
            result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"

        return result

    def calc_product_mf_from_signal(
        self,
        mz: int,
        species: dict,
        energy: float,
        signal_data: dict[float, float],
        ref_mz: int,
        ref_mw: float,
        ref_energy: float,
        ref_signal_data: dict[float, float],
        ref_mf_at_tm: float,
    ) -> dict[float, float] | None:
        species_name = species["species"]

        species_obj = None
        for s in self.get_species_by_mz(mz):
            if s["species"] == species_name:
                species_obj = s
                break
        if species_obj is None and self.get_species_by_mz(mz):
            species_obj = self.get_species_by_mz(mz)[0]

        ref_obj = None
        for s in self.get_species_by_mz(ref_mz):
            ref_obj = s
            break
        if ref_obj is None and self.get_species_by_mz(ref_mz):
            ref_obj = self.get_species_by_mz(ref_mz)[0]

        if species_obj is None or ref_obj is None:
            return None

        sigma_i = self.get_cross_section_at_energy(species_obj, energy)
        sigma_A = self.get_cross_section_at_energy(ref_obj, ref_energy)

        if sigma_i == 0 or sigma_A == 0:
            return None

        D_i = self.calc_mass_discrimination(mz, self.mass_disc_exponent)
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
        temperature_curves: dict[int, dict],
        parent_mf_results: dict[float, float] | None = None,
        available_energies: list[float] | None = None,
        parent_energy: float = 10.0,
    ) -> tuple[dict[tuple[int, str, float], dict[float, float]], list[str]]:
        warnings: list[str] = []

        if not pie_species_data:
            return {}, ["没有PIE鉴定结果"]

        if not self.expansion_coefficients:
            return {}, ["请先计算膨胀系数"]

        if parent_mf_results is None:
            parent_mf_results = self.parent_mf_results
        if not parent_mf_results:
            return {}, ["请先计算母体摩尔分数"]

        species_by_mz: dict[int, list[dict]] = {}
        for d in pie_species_data:
            mz = d["mz"]
            if mz not in species_by_mz:
                species_by_mz[mz] = []
            species_by_mz[mz].append(d)

        all_energies = sorted(available_energies) if available_energies else [parent_energy]

        all_species_mf: dict[tuple[int, str, float], dict[float, float]] = {}

        parent_mz = self.parent_mz
        if parent_mz is None:
            return {}, ["请先设置母体质量数"]

        parent_mw = float(parent_mz)
        parent_signal = extract_signal_from_temperature_curves(
            temperature_curves, parent_mz
        )
        parent_mf_at_tm = parent_mf_results.get(self.reference_temperature, 0)

        for mz, sp_list in species_by_mz.items():
            if mz == parent_mz:
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
    temperature_curves: dict[int, dict],
    parent_mf_results: dict[float, float],
    available_energies: list[float] | None = None,
    parent_energy: float = 10.0,
) -> tuple[dict[tuple[int, str, float], dict[float, float]], list[str]]:
    return calculator.calc_auto_mole_fractions(
        pie_species_data,
        temperature_curves,
        parent_mf_results,
        available_energies,
        parent_energy,
    )
