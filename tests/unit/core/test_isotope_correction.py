from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bl03u_masstool.core.isotope_correction import (
    IsotopeHypothesis,
    apply_isotope_correction,
    infer_curve_columns,
    nominal_isotope_pattern,
    prepare_curve_matrix,
)


def test_nominal_isotope_pattern_matches_c11h8cl2o_ratios():
    pattern = nominal_isotope_pattern("C11H8Cl2O", max_shift=4)

    assert pattern[0] == pytest.approx(1.0)
    assert pattern[1] == pytest.approx(0.1202740, rel=1e-6)
    assert pattern[2] == pytest.approx(0.6485599, rel=1e-6)
    assert pattern[4] == pytest.approx(0.1079229, rel=1e-6)


def test_infer_and_prepare_temperature_curve_matrix_with_replicate_mean():
    data = pd.DataFrame(
        {
            "temperature": [300, 300, 300, 300, 400, 400],
            "mz": [225.99, 226.02, 227.98, 228.01, 225.99, 227.98],
            "area": [100.0, 120.0, 60.0, 70.0, 200.0, 120.0],
        }
    )

    inferred = infer_curve_columns(data, "temperature")
    assert inferred["axis_column"] == "temperature"
    assert inferred["mass_column"] == "mz"
    assert inferred["preferred_signal_column"] == "area"
    assert inferred["signal_columns"] == ["area"]

    matrix = prepare_curve_matrix(
        data,
        axis_column="temperature",
        mass_column="mz",
        intensity_column="area",
        aggregation="mean",
    )

    assert list(matrix.columns) == [226, 228]
    assert matrix.loc[300, 226] == pytest.approx(110.0)
    assert matrix.loc[300, 228] == pytest.approx(65.0)
    assert matrix.loc[400, 226] == pytest.approx(200.0)


def test_infer_curve_columns_excludes_sqlite_identity_and_geometry_columns():
    data = pd.DataFrame(
        {
            "temperature": [300.0],
            "mz_rounded": [226],
            "mz": [226.01],
            "channel_id": [1],
            "peak_track": [0],
            "left_bound": [100.0],
            "right_bound": [120.0],
            "photon_energy": [11.0],
            "area": [136.0],
            "normalized_area": [1.0],
            "raw_area": [140.0],
        }
    )

    inferred = infer_curve_columns(data, "temperature")

    assert inferred["signal_columns"] == [
        "area",
        "normalized_area",
        "raw_area",
    ]


def test_manual_fraction_calculates_contribution_and_residual_curves():
    matrix = pd.DataFrame(
        {
            226: [100.0, 200.0],
            228: [80.0, 150.0],
            230: [20.0, 30.0],
        },
        index=pd.Index([300.0, 400.0], name="temperature"),
    )
    result = apply_isotope_correction(
        matrix,
        [IsotopeHypothesis("C11H8Cl2O", 226, fraction=0.5)],
        mode="manual_fraction",
    )

    first_rows = result.curve_table[result.curve_table["temperature"] == 300.0].set_index("mz")
    assert first_rows.loc[226, "isotope_contribution"] == pytest.approx(0.0)
    assert first_rows.loc[226, "residual"] == pytest.approx(100.0)
    assert first_rows.loc[228, "isotope_contribution"] == pytest.approx(50.0 * 0.6485599237)
    assert first_rows.loc[230, "residual"] == pytest.approx(20.0 - 50.0 * 0.1079228948)
    assert result.source_table.iloc[0]["source_amplitude"] == pytest.approx(50.0)
    assert not (
        (result.component_table["mz"] == 226)
        & (result.component_table["formula"] == "C11H8Cl2O")
    ).any()
    assert result.pattern_rank == 1
    assert result.pattern_table.columns.is_unique


def test_max_compatible_mode_keeps_residuals_nonnegative():
    matrix = pd.DataFrame(
        {226: [136.0], 228: [80.0], 230: [27.0]},
        index=pd.Index([11.0], name="energy"),
    )
    result = apply_isotope_correction(
        matrix,
        [IsotopeHypothesis("C11H8Cl2O", 226)],
        mode="max_compatible",
    )

    source = result.source_table.iloc[0]["source_amplitude"]
    assert source == pytest.approx(80.0 / 0.6485599236631806)
    assert (result.curve_table["residual"] >= -1e-10).all()
    residual_230 = result.curve_table.loc[result.curve_table["mz"] == 230, "residual"].iloc[0]
    assert residual_230 == pytest.approx(13.68768872045)
    residual_percent_230 = result.curve_table.loc[
        result.curve_table["mz"] == 230,
        "residual_percent",
    ].iloc[0]
    assert residual_percent_230 == pytest.approx(50.6951434091)


def test_max_compatible_mode_solves_multiple_parent_hypotheses_together():
    first_pattern = nominal_isotope_pattern("C11H8Cl2O", max_shift=4)
    second_pattern = nominal_isotope_pattern("C18H12", max_shift=2)
    first_amplitude = 100.0
    second_amplitude = 20.0
    matrix = pd.DataFrame(
        {
            226: [first_amplitude],
            228: [first_amplitude * first_pattern[2] + second_amplitude],
            230: [
                first_amplitude * first_pattern[4]
                + second_amplitude * second_pattern[2]
            ],
        },
        index=pd.Index([11.0], name="energy"),
    )

    result = apply_isotope_correction(
        matrix,
        [
            IsotopeHypothesis("C11H8Cl2O", 226),
            IsotopeHypothesis("C18H12", 228),
        ],
        mode="max_compatible",
    )

    amplitudes = result.source_table.set_index("formula")["source_amplitude"]
    assert amplitudes["C11H8Cl2O"] == pytest.approx(first_amplitude)
    assert amplitudes["C18H12"] == pytest.approx(second_amplitude)
    residuals = result.curve_table.set_index("mz")["residual"]
    assert residuals[226] == pytest.approx(first_amplitude)
    assert residuals[228] == pytest.approx(second_amplitude)
    assert residuals[230] == pytest.approx(0.0, abs=1e-8)
    assert result.pattern_rank == 2


def test_correction_requires_formula_parent_channel_and_complete_values():
    matrix = pd.DataFrame(
        {226: [100.0], 228: [np.nan]},
        index=pd.Index([300.0], name="temperature"),
    )
    with pytest.raises(ValueError, match="缺失值"):
        apply_isotope_correction(
            matrix,
            [IsotopeHypothesis("C11H8Cl2O", 226)],
        )

    complete = matrix.fillna(10.0)
    with pytest.raises(ValueError, match="不在所选质量范围"):
        apply_isotope_correction(
            complete,
            [IsotopeHypothesis("C18H14", 230)],
        )

    with pytest.raises(ValueError, match="0 到 1"):
        IsotopeHypothesis("C11H8Cl2O", 226, fraction=1.01)

    with pytest.raises(ValueError, match="名义质量"):
        IsotopeHypothesis("C18H12", 230)

    with pytest.raises(ValueError, match="总和不能超过"):
        apply_isotope_correction(
            complete,
            [
                IsotopeHypothesis("C11H8Cl2O", 226, fraction=0.6),
                IsotopeHypothesis("C18H10", 226, fraction=0.5),
            ],
        )
