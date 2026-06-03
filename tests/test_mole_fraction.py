from __future__ import annotations

import numpy as np
import pytest

from core.mole_fraction import (
    MASS_DISCRIMINATION_PRESETS,
    MoleFractionSettings,
    calc_expansion_coefficients,
    calc_isomeric_separation,
    calc_mass_discrimination,
    calc_parent_mole_fraction,
    calc_product_mole_fraction,
    get_expansion_coefficient,
    load_mole_fraction_settings,
)


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
        assert settings.parent_mz == 128
        assert settings.parent_initial_mf == 0.002

    def test_load_save_roundtrip(self, tmp_path):
        settings = MoleFractionSettings(
            mass_disc_exponent=0.5,
            parent_mz=100,
            kr_data={100.0: 50.0, 200.0: 25.0},
        )
        from core.mole_fraction import save_mole_fraction_settings

        path = save_mole_fraction_settings(settings, tmp_path / "test_mf.yaml")
        loaded = load_mole_fraction_settings(path)
        assert loaded.mass_disc_exponent == pytest.approx(0.5)
        assert loaded.parent_mz == 100
        assert loaded.kr_data[100.0] == pytest.approx(50.0)
