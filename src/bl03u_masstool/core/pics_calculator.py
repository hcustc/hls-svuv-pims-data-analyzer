from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
PICSResultKey = float | tuple[float, float]


@dataclass
class PICSResult:
    energy: float
    cross_section: float
    cross_section_error: float = 0.0


@dataclass
class PICSInputData:
    new_species_name: str = ""
    new_species_formula: str = ""
    new_species_mz: float = 0.0
    
    no_mz: int = 30
    no_formula: str = "NO"
    
    new_species_mf: float = 0.0
    no_mf: float = 0.0
    
    new_species_signal: dict[float, float] = field(default_factory=dict)
    no_signal: dict[float, float] = field(default_factory=dict)
    
    mass_disc_exponent: float = 0.77897
    no_cross_section: float = 5.0
    
    expansion_coefficients: dict[float, float] = field(default_factory=dict)


def calc_mass_discrimination(molecular_weight: float, exponent: float = 0.77897) -> float:
    return float((molecular_weight / 30.0) ** exponent)


def extract_peak_height_at_mz(
    mz_values,
    intensities,
    target_mz: float,
    *,
    half_window_da: float = 0.08,
) -> float:
    """Extract the local apex nearest an exact m/z without using a 1 Da bin."""
    mz_array = np.asarray(mz_values, dtype=float)
    intensity_array = np.asarray(intensities, dtype=float)
    if mz_array.shape != intensity_array.shape:
        raise ValueError("m/z and intensity arrays must have the same shape")
    if mz_array.size == 0:
        return 0.0

    finite = np.isfinite(mz_array) & np.isfinite(intensity_array)
    if not np.any(finite):
        return 0.0
    mz_array = mz_array[finite]
    intensity_array = intensity_array[finite]
    order = np.argsort(mz_array, kind="stable")
    mz_array = mz_array[order]
    intensity_array = intensity_array[order]

    target = float(target_mz)
    half_window = float(half_window_da)
    if not np.isfinite(target) or half_window <= 0:
        raise ValueError("target m/z must be finite and half_window_da must be positive")

    local_indices = np.flatnonzero(np.abs(mz_array - target) <= half_window)
    if local_indices.size == 0:
        nearest = int(np.argmin(np.abs(mz_array - target)))
        return float(max(0.0, intensity_array[nearest]))

    local_apices: list[int] = []
    for index in local_indices:
        left = intensity_array[index - 1] if index > 0 else -np.inf
        right = (
            intensity_array[index + 1]
            if index + 1 < intensity_array.size
            else -np.inf
        )
        value = intensity_array[index]
        if value >= left and value >= right and (value > left or value > right):
            local_apices.append(int(index))

    candidate_indices = local_apices or [int(index) for index in local_indices]
    selected = min(
        candidate_indices,
        key=lambda index: (
            abs(float(mz_array[index]) - target),
            -float(intensity_array[index]),
        ),
    )
    return float(max(0.0, intensity_array[selected]))


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


def calc_pics_single_energy(
    new_species_signal: float,
    no_signal: float,
    new_species_mf: float,
    no_mf: float,
    new_species_mz: float,
    no_mz: int = 30,
    no_cross_section: float = 5.0,
    mass_disc_exponent: float = 0.77897,
) -> float:
    if new_species_signal <= 0 or no_signal <= 0:
        return 0.0
    if new_species_mf <= 0 or no_mf <= 0:
        return 0.0
    
    D_new = calc_mass_discrimination(float(new_species_mz), mass_disc_exponent)
    D_no = calc_mass_discrimination(float(no_mz), mass_disc_exponent)
    
    if D_new == 0:
        return 0.0
    
    sigma_new = no_cross_section * \
                (new_species_signal / no_signal) * \
                (no_mf / new_species_mf) * \
                (D_no / D_new)
    
    return max(0.0, float(sigma_new))


def calc_pics_with_temperature(
    input_data: PICSInputData,
) -> dict[PICSResultKey, PICSResult]:
    results: dict[PICSResultKey, PICSResult] = {}

    if input_data.new_species_mf <= 0 or input_data.no_mf <= 0:
        return results
    
    new_signal = input_data.new_species_signal
    no_signal = input_data.no_signal
    
    if isinstance(list(new_signal.values())[0], dict) if new_signal else False:
        all_energies = set(new_signal.keys()) & set(no_signal.keys())
        
        for energy in all_energies:
            new_signals = new_signal.get(energy, {})
            no_signals = no_signal.get(energy, {})
            
            common_temps = set(new_signals.keys()) & set(no_signals.keys())
            
            for temp in common_temps:
                ns = new_signals.get(temp, 0)
                nos = no_signals.get(temp, 0)
                
                if ns <= 0 or nos <= 0:
                    continue
                
                lambda_new = get_expansion_coefficient(temp, input_data.expansion_coefficients)
                lambda_no = lambda_new
                
                D_new = calc_mass_discrimination(float(input_data.new_species_mz), input_data.mass_disc_exponent)
                D_no = calc_mass_discrimination(float(input_data.no_mz), input_data.mass_disc_exponent)
                
                if D_new == 0 or lambda_new == 0:
                    continue
                
                sigma_new = input_data.no_cross_section * \
                            (ns / nos) * \
                            (input_data.no_mf / input_data.new_species_mf) * \
                            (D_no / D_new) * \
                            (lambda_no / lambda_new)
                
                results[(energy, temp)] = PICSResult(
                    energy=energy,
                    cross_section=max(0.0, float(sigma_new)),
                    cross_section_error=0.0
                )
    else:
        common_temps = set(new_signal.keys()) & set(no_signal.keys())
        
        for temp in common_temps:
            ns = new_signal.get(temp, 0)
            nos = no_signal.get(temp, 0)
            
            if ns <= 0 or nos <= 0:
                continue
            
            lambda_new = get_expansion_coefficient(temp, input_data.expansion_coefficients)
            lambda_no = lambda_new
            
            D_new = calc_mass_discrimination(float(input_data.new_species_mz), input_data.mass_disc_exponent)
            D_no = calc_mass_discrimination(float(input_data.no_mz), input_data.mass_disc_exponent)
            
            if D_new == 0 or lambda_new == 0:
                continue
            
            sigma_new = input_data.no_cross_section * \
                        (ns / nos) * \
                        (input_data.no_mf / input_data.new_species_mf) * \
                        (D_no / D_new) * \
                        (lambda_no / lambda_new)
            
            results[temp] = PICSResult(
                energy=temp,
                cross_section=max(0.0, float(sigma_new)),
                cross_section_error=0.0
            )
    
    return results


def calc_pics_multi_energy(
    new_species_signal_by_energy: dict[float, dict[float, float]],
    no_signal_by_energy: dict[float, dict[float, float]],
    input_data: PICSInputData,
) -> dict[float, PICSResult]:
    results: dict[float, PICSResult] = {}

    if input_data.new_species_mf <= 0 or input_data.no_mf <= 0:
        return results
    
    all_energies = set(new_species_signal_by_energy.keys()) & set(no_signal_by_energy.keys())
    
    for energy in all_energies:
        new_signal_dict = new_species_signal_by_energy.get(energy, {})
        no_signal_dict = no_signal_by_energy.get(energy, {})
        
        common_temps = set(new_signal_dict.keys()) & set(no_signal_dict.keys())
        
        if not common_temps:
            continue
        
        sigma_values = []
        for temp in common_temps:
            new_signal = new_signal_dict.get(temp, 0)
            no_signal = no_signal_dict.get(temp, 0)
            
            if new_signal <= 0 or no_signal <= 0:
                continue
            
            lambda_new = get_expansion_coefficient(temp, input_data.expansion_coefficients)
            lambda_no = get_expansion_coefficient(temp, input_data.expansion_coefficients)
            
            D_new = calc_mass_discrimination(float(input_data.new_species_mz), input_data.mass_disc_exponent)
            D_no = calc_mass_discrimination(float(input_data.no_mz), input_data.mass_disc_exponent)
            
            if D_new == 0 or lambda_new == 0:
                continue
            
            sigma_new = input_data.no_cross_section * \
                        (new_signal / no_signal) * \
                        (input_data.no_mf / input_data.new_species_mf) * \
                        (D_no / D_new) * \
                        (lambda_no / lambda_new)
            
            sigma_values.append(max(0.0, float(sigma_new)))
        
        if sigma_values:
            avg_sigma = float(np.mean(sigma_values))
            std_sigma = float(np.std(sigma_values)) if len(sigma_values) > 1 else 0.0
            
            results[energy] = PICSResult(
                energy=energy,
                cross_section=avg_sigma,
                cross_section_error=std_sigma
            )
    
    return results


class PICSCalculator:
    def __init__(self):
        self.input_data = PICSInputData()
        self.results: dict[PICSResultKey, PICSResult] = {}
    
    def set_input_data(self, data: PICSInputData) -> None:
        self.input_data = data
    
    def calculate(self) -> dict[PICSResultKey, PICSResult]:
        self.results = calc_pics_with_temperature(self.input_data)
        return self.results
    
    def calculate_multi_energy(
        self,
        new_species_signal_by_energy: dict[float, dict[float, float]],
        no_signal_by_energy: dict[float, dict[float, float]],
    ) -> dict[float, PICSResult]:
        self.results = calc_pics_multi_energy(
            new_species_signal_by_energy,
            no_signal_by_energy,
            self.input_data
        )
        return self.results
    
    def get_results(self) -> dict[PICSResultKey, PICSResult]:
        return self.results
    
    def get_average_cross_section(self) -> float:
        if not self.results:
            return 0.0
        values = [r.cross_section for r in self.results.values()]
        return float(np.mean(values))


# Backwards-compatible alias for older callers.
PICS_Calculator = PICSCalculator
