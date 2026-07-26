from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bl03u_masstool.core.mole_fraction import (
    MASS_DISCRIMINATION_PRESETS,
    MoleFractionCalculator,
    MoleFractionSettings,
    _interpolate_cross_section,
    calc_auto_mole_fractions,
    calc_expansion_coefficients,
    calc_isomeric_separation,
    calc_mass_discrimination,
    calc_parent_mole_fraction,
    calc_product_mole_fraction,
    compute_all_mole_fractions,
    extract_signal_from_temperature_curves,
    get_expansion_coefficient,
    get_expansion_coefficient_for_energy,
    load_mole_fraction_settings,
    separate_coexisting_species_signals,
)


def test_temperature_signal_requires_exact_key_when_nominal_mass_is_ambiguous():
    curves = {
        228.021: {
            "mz": 228.021,
            "mz_rounded": 228,
            "mz_exact_mean": 228.021,
            "temperatures": [400.0, 500.0],
            "areas": [1.0, 2.0],
        },
        228.099: {
            "mz": 228.099,
            "mz_rounded": 228,
            "mz_exact_mean": 228.099,
            "temperatures": [400.0, 500.0],
            "areas": [10.0, 20.0],
        },
    }

    assert extract_signal_from_temperature_curves(curves, 228) == {}
    assert extract_signal_from_temperature_curves(curves, 228.021) == {
        400.0: 1.0,
        500.0: 2.0,
    }
    assert extract_signal_from_temperature_curves(curves, 228.023) == {
        400.0: 1.0,
        500.0: 2.0,
    }


def test_auto_mole_fraction_matches_parent_across_small_exact_mz_drift():
    calculator = MoleFractionCalculator()
    calculator.expansion_coefficients = {400.0: 1.0, 500.0: 1.0}
    calculator.parent_mz = 228.021

    results, warnings = calc_auto_mole_fractions(
        calculator,
        pie_species_data=[
            {
                "mz": 228,
                "curve_key": 228.023,
                "species": "Parent",
                "ie": 10.0,
            }
        ],
        temperature_curves={
            228.021: {
                "mz": 228.021,
                "mz_rounded": 228,
                "curve_key": 228.021,
                "temperatures": [400.0, 500.0],
                "areas": [100.0, 80.0],
            }
        },
        parent_mf_results={400.0: 0.01, 500.0: 0.008},
        available_energies=[12.0],
        parent_energy=12.0,
    )

    assert warnings == []
    assert results[(228.023, "母体", 12.0)] == {
        400.0: 0.01,
        500.0: 0.008,
    }


class TestCalcMassDiscrimination:
    def test_default_exponent(self):
        result = calc_mass_discrimination(30.0)
        assert result == pytest.approx(1.0)

    def test_custom_exponent(self):
        result = calc_mass_discrimination(60.0, exponent=0.5)
        assert result == pytest.approx((60.0 / 30.0) ** 0.5)

    def test_presets_are_valid(self):
        for name, exponent in MASS_DISCRIMINATION_PRESETS.items():
            assert 0 < exponent < 2
            assert calc_mass_discrimination(30.0, exponent) == pytest.approx(1.0)


class TestCalcExpansionCoefficients:
    def test_empty_input(self):
        assert calc_expansion_coefficients({}) == {}

    def test_single_point(self):
        result = calc_expansion_coefficients({100.0: 50.0})
        assert result == {100.0: 1.0}

    def test_multiple_points(self):
        kr_data = {100.0: 100.0, 200.0: 50.0, 300.0: 25.0}
        result = calc_expansion_coefficients(kr_data)
        assert result[100.0] == pytest.approx(1.0)
        assert result[200.0] == pytest.approx(0.5)
        assert result[300.0] == pytest.approx(0.25)

    def test_zero_ref_signal(self):
        result = calc_expansion_coefficients({100.0: 0.0, 200.0: 50.0})
        assert result == {}


class TestGetExpansionCoefficient:
    def test_exact_match(self):
        coeffs = {100.0: 1.0, 200.0: 0.5}
        assert get_expansion_coefficient(100.0, coeffs) == pytest.approx(1.0)

    def test_interpolation(self):
        coeffs = {100.0: 1.0, 200.0: 0.5}
        result = get_expansion_coefficient(150.0, coeffs)
        assert result == pytest.approx(0.75)

    def test_empty_coefficients(self):
        assert get_expansion_coefficient(100.0, {}) == 1.0

    def test_single_coefficient(self):
        assert get_expansion_coefficient(100.0, {200.0: 0.5}) == 1.0

    def test_multi_energy_uses_closest_energy_group(self):
        coeffs = {
            14.6: {650.0: 1.0, 750.0: 2.0},
            14.8: {650.0: 10.0, 750.0: 20.0},
        }

        assert get_expansion_coefficient_for_energy(750.0, 14.79, coeffs) == pytest.approx(20.0)


class TestCalcParentMoleFraction:
    def test_basic_calculation(self):
        signal_data = {100.0: 1000.0, 200.0: 500.0, 300.0: 250.0}
        result = calc_parent_mole_fraction(
            signal_data,
            reference_temperature=100.0,
            parent_initial_mf=0.01,
        )
        assert result[100.0] == pytest.approx(0.01)
        assert result[200.0] == pytest.approx(0.005)
        assert result[300.0] == pytest.approx(0.0025)

    def test_with_expansion_coefficients(self):
        signal_data = {100.0: 1000.0, 200.0: 500.0}
        expansion = {100.0: 1.0, 200.0: 2.0}
        result = calc_parent_mole_fraction(
            signal_data,
            reference_temperature=100.0,
            parent_initial_mf=0.01,
            expansion_coefficients=expansion,
        )
        assert result[100.0] == pytest.approx(0.01)
        assert result[200.0] == pytest.approx(0.0025)

    def test_empty_signal(self):
        assert calc_parent_mole_fraction({}) == {}

    def test_zero_ref_signal(self):
        result = calc_parent_mole_fraction({100.0: 0.0, 200.0: 500.0})
        assert result == {}

    def test_missing_reference_temperature_returns_empty(self):
        result = calc_parent_mole_fraction(
            {650.0: 1000.0, 750.0: 500.0},
            reference_temperature=550.0,
            parent_initial_mf=0.02,
        )

        assert result == {}


class TestCalcIsomericSeparation:
    def test_basic_separation(self):
        signal_low = {100.0: 100.0}
        signal_high = {100.0: 200.0}
        species_list = [
            {"name": "A", "sigma_low": 10.0, "sigma_high": 20.0},
            {"name": "B", "sigma_low": 10.0, "sigma_high": 10.0},
        ]
        result = calc_isomeric_separation(signal_low, signal_high, species_list)
        assert "A" in result
        assert "B" in result
        assert 100.0 in result["A"]
        assert 100.0 in result["B"]

    def test_no_common_temps(self):
        signal_low = {100.0: 100.0}
        signal_high = {200.0: 200.0}
        species_list = [{"name": "A", "sigma_low": 10.0, "sigma_high": 20.0}]
        result = calc_isomeric_separation(signal_low, signal_high, species_list)
        assert all(len(v) == 0 for v in result.values())


class TestMoleFractionSettings:
    def test_default_values(self):
        settings = MoleFractionSettings()
        assert settings.mass_disc_exponent == 0.77897
        assert settings.parent_mz == 0
        assert settings.parent_initial_mf == 0.002

    def test_load_save_roundtrip(self, tmp_path):
        settings = MoleFractionSettings(
            mass_disc_exponent=0.5,
            parent_mz=100,
            kr_data={100.0: 50.0, 200.0: 25.0},
        )
        from bl03u_masstool.core.mole_fraction import save_mole_fraction_settings

        path = save_mole_fraction_settings(settings, tmp_path / "test_mf.yaml")
        loaded = load_mole_fraction_settings(path)
        assert loaded.mass_disc_exponent == pytest.approx(0.5)
        assert loaded.parent_mz == 100
        assert loaded.kr_data[100.0] == pytest.approx(50.0)


def test_compute_all_mole_fractions_accepts_pie_species_field_and_formula_mass():
    temperature_scan_df = pd.DataFrame(
        [
            {"temperature": 100.0, "file": "t100.txt", "reference_temperature": 100.0, "mz": 30.0, "area": 1000.0},
            {"temperature": 200.0, "file": "t200.txt", "reference_temperature": 100.0, "mz": 30.0, "area": 800.0},
            {"temperature": 100.0, "file": "t100.txt", "reference_temperature": 100.0, "mz": 18.0, "area": 50.0},
            {"temperature": 200.0, "file": "t200.txt", "reference_temperature": 100.0, "mz": 18.0, "area": 80.0},
        ]
    )
    database = [
        {"mz": 18, "species": "Water", "energies": np.array([12.0]), "cross_sections": np.array([2.0])},
        {"mz": 30, "species": "NO", "energies": np.array([12.0]), "cross_sections": np.array([4.0])},
    ]
    settings = MoleFractionSettings(
        mass_disc_exponent=0.0,
        parent_mz=30,
        parent_initial_mf=0.01,
        reference_temperature=100.0,
        reference_species_mz=30,
        reference_species_mf_at_tm=0.01,
        photon_energy=12.0,
        kr_data={100.0: 1.0, 200.0: 1.0},
    )

    result = compute_all_mole_fractions(
        temperature_scan_df,
        settings=settings,
        database=database,
        mz_index={18: [0], 30: [1]},
        product_species=[{"mz": 18, "species": "Water", "formula": "H2O"}],
    )

    assert result["Water"].tolist() == pytest.approx([0.001, 0.0016])


def test_product_mole_fraction_uses_max_signal_temperature_when_reference_unset():
    database = [
        {"mz": 18, "species": "Water", "energies": np.array([12.0]), "cross_sections": np.array([2.0])},
        {"mz": 30, "species": "NO", "energies": np.array([12.0]), "cross_sections": np.array([4.0])},
    ]

    result = calc_product_mole_fraction(
        {650.0: 50.0, 750.0: 80.0},
        species_mw=18.0,
        species_mz=18,
        species_name="Water",
        ref_mw=30.0,
        ref_mz=30,
        ref_species_name="NO",
        ref_mf_at_tm=0.01,
        ref_signal_data={650.0: 1000.0, 750.0: 800.0},
        energy=12.0,
        mass_disc_exponent=0.0,
        database=database,
        mz_index={18: [0], 30: [1]},
    )

    assert result[650.0] == pytest.approx(0.00125)
    assert result[750.0] == pytest.approx(0.002)


def test_product_mole_fraction_returns_empty_when_reference_temperature_missing():
    database = [
        {"mz": 18, "species": "Water", "energies": np.array([12.0]), "cross_sections": np.array([2.0])},
        {"mz": 30, "species": "NO", "energies": np.array([12.0]), "cross_sections": np.array([4.0])},
    ]

    result = calc_product_mole_fraction(
        {650.0: 50.0, 750.0: 80.0},
        species_mw=18.0,
        species_mz=18,
        species_name="Water",
        ref_mw=30.0,
        ref_mz=30,
        ref_species_name="NO",
        ref_mf_at_tm=0.01,
        ref_signal_data={650.0: 1000.0, 750.0: 800.0},
        energy=12.0,
        reference_temperature=550.0,
        mass_disc_exponent=0.0,
        database=database,
        mz_index={18: [0], 30: [1]},
    )

    assert result == {}


def test_auto_mole_fractions_uses_parent_result_temperature_when_reference_is_unset():
    calculator = MoleFractionCalculator()
    calculator.database = [
        {"mz": 18, "species": "Water", "energies": np.array([12.0]), "cross_sections": np.array([2.0])},
        {"mz": 30, "species": "NO", "energies": np.array([12.0]), "cross_sections": np.array([4.0])},
    ]
    calculator.mz_index = {18: [0], 30: [1]}
    calculator.expansion_coefficients = {100.0: 1.0, 200.0: 1.0}
    calculator.mass_disc_exponent = 0.0
    calculator.parent_mz = 30

    results, warnings = calc_auto_mole_fractions(
        calculator,
        pie_species_data=[{"mz": 18, "species": "Water", "ie": 11.0}],
        temperature_curves={
            30: {"temperatures": [100.0, 200.0], "areas": [1000.0, 800.0]},
            18: {"temperatures": [100.0, 200.0], "areas": [50.0, 80.0]},
        },
        parent_mf_results={100.0: 0.01, 200.0: 0.008},
        available_energies=[12.0],
        parent_energy=12.0,
    )

    assert warnings == []
    assert results[(18, "Water", 12.0)][100.0] == pytest.approx(0.001)
    assert results[(18, "Water", 12.0)][200.0] == pytest.approx(0.0016)


def test_cross_section_interpolation_extrapolates_low_energy_edge():
    energies = np.array([9.0, 10.0])
    cross_sections = np.array([1.0, 3.0])

    assert _interpolate_cross_section(energies, cross_sections, 9.5) == pytest.approx(2.0)
    assert _interpolate_cross_section(energies, cross_sections, 8.8) == pytest.approx(0.6)
    assert _interpolate_cross_section(energies, cross_sections, 8.0) == pytest.approx(0.0)
    assert _interpolate_cross_section(energies, cross_sections, 11.0) == pytest.approx(3.0)


def test_separate_coexisting_species_signals_uses_energy_resolved_subtraction():
    temperatures = [550.0, 600.0, 650.0]
    naphthalene_85 = np.array([0.10, 0.15, 0.22])
    chlorophenol_90 = np.array([2.00, 1.90, 1.80])
    database = [
        {
            "species": "Naphthalene",
            "mz": 128,
            "ie": 8.14,
            "energies": np.array([8.5, 9.0, 10.0]),
            "cross_sections": np.array([2.0, 3.5, 4.8]),
        },
        {
            "species": "o-Chlorophenol",
            "mz": 128,
            "ie": 8.90,
            "energies": np.array([9.0, 10.0]),
            "cross_sections": np.array([1.5, 4.5]),
        },
    ]
    mz_index = {128: [0, 1]}
    energy_scan_data = {
        8.5: dict(zip(temperatures, naphthalene_85)),
        9.0: dict(zip(temperatures, naphthalene_85 * (3.5 / 2.0) + chlorophenol_90)),
        10.0: dict(
            zip(
                temperatures,
                naphthalene_85 * (4.8 / 2.0)
                + chlorophenol_90 * (4.5 / 1.5),
            )
        ),
    }

    separated, pure_energies = separate_coexisting_species_signals(
        mz=128,
        species_at_mz=[
            {"species": "Naphthalene", "ie": 8.14},
            {"species": "o-Chlorophenol", "ie": 8.90},
        ],
        energy_scan_data=energy_scan_data,
        database=database,
        mz_index=mz_index,
        target_energy=10.0,
    )

    assert pure_energies == {"Naphthalene": 8.5, "o-Chlorophenol": 9.0}
    assert list(separated["Naphthalene"].values()) == pytest.approx(
        (naphthalene_85 * (4.8 / 2.0)).tolist()
    )
    assert list(separated["o-Chlorophenol"].values()) == pytest.approx(
        (chlorophenol_90 * (4.5 / 1.5)).tolist()
    )
