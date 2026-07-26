from __future__ import annotations

import numpy as np
import pytest

from bl03u_masstool.core.pics_calculator import (
    PICSCalculator,
    PICSInputData,
    calc_pics_multi_energy,
    calc_pics_single_energy,
    calc_pics_with_temperature,
    extract_peak_height_at_mz,
)


def test_exact_mz_peak_height_does_not_mix_a_nominal_mass_collision():
    mz_values = np.array([227.56, 227.587, 227.61, 228.0, 228.023, 228.05])
    intensities = np.array([0.0, 2.0, 0.0, 0.0, 100.0, 0.0])

    assert extract_peak_height_at_mz(
        mz_values,
        intensities,
        227.587,
        half_window_da=0.03,
    ) == pytest.approx(2.0)
    assert extract_peak_height_at_mz(
        mz_values,
        intensities,
        228.023,
        half_window_da=0.03,
    ) == pytest.approx(100.0)


def test_exact_mz_peak_height_chooses_nearest_apex_not_strongest_neighbour():
    mz_values = np.array([227.96, 227.98, 228.0, 228.04, 228.06])
    intensities = np.array([0.0, 2.0, 0.0, 100.0, 0.0])

    assert extract_peak_height_at_mz(
        mz_values,
        intensities,
        227.98,
        half_window_da=0.08,
    ) == pytest.approx(2.0)


def test_calc_pics_single_energy_basic():
    assert calc_pics_single_energy(
        new_species_signal=20.0,
        no_signal=10.0,
        new_species_mf=0.5,
        no_mf=0.25,
        new_species_mz=30,
        no_mz=30,
        no_cross_section=5.0,
        mass_disc_exponent=1.0,
    ) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "new_species_signal,no_signal,new_species_mf,no_mf",
    [
        (0.0, 10.0, 0.5, 0.25),
        (20.0, 0.0, 0.5, 0.25),
        (20.0, 10.0, 0.0, 0.25),
        (20.0, 10.0, 0.5, 0.0),
    ],
)
def test_calc_pics_single_energy_invalid_inputs_return_zero(
    new_species_signal, no_signal, new_species_mf, no_mf
):
    assert calc_pics_single_energy(
        new_species_signal,
        no_signal,
        new_species_mf,
        no_mf,
        new_species_mz=30,
    ) == 0.0


def test_calc_pics_with_temperature_returns_empty_for_missing_mole_fraction():
    data = PICSInputData(
        new_species_mz=30,
        new_species_signal={100.0: 20.0},
        no_signal={100.0: 10.0},
        new_species_mf=0.0,
        no_mf=0.25,
    )
    assert calc_pics_with_temperature(data) == {}


def test_calc_pics_with_temperature_single_point():
    data = PICSInputData(
        new_species_mz=30,
        no_mz=30,
        new_species_signal={100.0: 20.0},
        no_signal={100.0: 10.0},
        new_species_mf=0.5,
        no_mf=0.25,
        no_cross_section=5.0,
        mass_disc_exponent=1.0,
    )
    results = calc_pics_with_temperature(data)
    assert results[100.0].cross_section == pytest.approx(5.0)


def test_calc_pics_with_temperature_multi_energy_tuple_keys():
    data = PICSInputData(
        new_species_mz=30,
        no_mz=30,
        new_species_signal={11.0: {200.0: 20.0}},
        no_signal={11.0: {200.0: 10.0}},
        new_species_mf=0.5,
        no_mf=0.25,
        no_cross_section=5.0,
        mass_disc_exponent=1.0,
    )
    results = calc_pics_with_temperature(data)
    assert results[(11.0, 200.0)].cross_section == pytest.approx(5.0)


def test_calc_pics_multi_energy_averages_common_temperatures():
    data = PICSInputData(
        new_species_mz=30,
        no_mz=30,
        new_species_mf=0.5,
        no_mf=0.25,
        no_cross_section=5.0,
        mass_disc_exponent=1.0,
    )
    results = calc_pics_multi_energy(
        {11.0: {200.0: 20.0, 300.0: 40.0}},
        {11.0: {200.0: 10.0, 300.0: 10.0}},
        data,
    )
    assert results[11.0].cross_section == pytest.approx(7.5)
    assert results[11.0].cross_section_error > 0


def test_pics_calculator_class_uses_core_calculation():
    calculator = PICSCalculator()
    calculator.set_input_data(
        PICSInputData(
            new_species_mz=30,
            no_mz=30,
            new_species_signal={100.0: 20.0},
            no_signal={100.0: 10.0},
            new_species_mf=0.5,
            no_mf=0.25,
        )
    )
    assert calculator.calculate()[100.0].cross_section == pytest.approx(5.0)
    assert calculator.get_average_cross_section() == pytest.approx(5.0)
