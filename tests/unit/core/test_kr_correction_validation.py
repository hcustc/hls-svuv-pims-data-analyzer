from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bl03u_masstool.core.temperature_scan import (
    KrCorrectionError,
    _apply_temperature_normalization,
    _build_kr_expansion_factor_table,
    analyze_temperature_folder,
    compute_kr_expansion_factors,
)


def _signal_rows(kr_signals=(10.0, 20.0)):
    return pd.DataFrame([
        {
            "temperature": temperature, "photon_energy": 12.0,
            "file": f"{temperature:.0f}.txt", "mz": mz, "mz_rounded": mz,
            "raw_area": value, "photon_normalized_area": value,
        }
        for temperature, kr, target in zip((400.0, 500.0), kr_signals, (12.0, 24.0))
        for mz, value in ((84, kr), (22, target))
    ])


def _correct(rows, **kwargs):
    return _apply_temperature_normalization(
        rows, kr_correct=True, kr_mz=84, mass_discrimination=1.0, **kwargs,
    )


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_kr_correction_rejects_invalid_references_without_mutating_input(value, all_invalid):
    rows = _signal_rows((value if all_invalid else 10.0, value))
    original = rows.copy(deep=True)

    with pytest.raises(ValueError) as error:
        _correct(rows)

    message = str(error.value)
    assert "84" in message and "Kr" in message
    assert str(400 if all_invalid else 500) in message
    pd.testing.assert_frame_equal(rows, original)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("nested", [False, True])
def test_kr_correction_rejects_invalid_supplied_factors(value, nested):
    rows = _signal_rows()
    original = rows.copy(deep=True)
    factors = {400.0: 1.0, 500.0: value}
    if nested:
        factors = {12.0: factors}

    with pytest.raises(ValueError) as error:
        _correct(rows, expansion_factors=factors)

    message = str(error.value)
    assert "84" in message and "500" in message
    assert "factor" in message.lower()
    pd.testing.assert_frame_equal(rows, original)


def test_kr_correction_does_not_replace_missing_reference_with_unit_factor():
    rows = _signal_rows().query("not (temperature == 500 and mz == 84)")
    with pytest.raises(ValueError, match="500"):
        _correct(rows)


@pytest.mark.parametrize("other_energy", [False, True])
def test_kr_correction_rejects_empty_supplied_energy_table(other_energy):
    factors = {12.0: {}}
    if other_energy:
        factors[14.0] = {400.0: 1.0, 500.0: 2.0}
    with pytest.raises(ValueError, match="400"):
        _correct(_signal_rows(), expansion_factors=factors)


def test_kr_correction_does_not_hide_nan_reference_in_replicate_sum():
    rows = _signal_rows()
    invalid_replicate = rows.iloc[[0]].copy()
    invalid_replicate["photon_normalized_area"] = np.nan
    rows = pd.concat([rows, invalid_replicate], ignore_index=True)
    with pytest.raises(ValueError, match="400"):
        _correct(rows)


@pytest.mark.parametrize("overflow", ["factor", "corrected_signal"])
def test_kr_correction_rejects_nonfinite_calculation(overflow):
    if overflow == "factor":
        rows, factors = _signal_rows((1e-308, 1e308)), None
    else:
        rows, factors = _signal_rows(), {400.0: 1.0, 500.0: 1e-320}
    with pytest.raises(ValueError, match="500"):
        _correct(rows, expansion_factors=factors)


@pytest.mark.parametrize("supplied", [False, True])
def test_positive_kr_correction_preserves_results_and_source_signals(supplied):
    # Independent calibration factors do not require an in-scan Kr signal.
    rows = _signal_rows((10.0, 0.0)) if supplied else _signal_rows()
    original = rows.copy(deep=True)
    result = _correct(rows, expansion_factors={400: 1, 500: 2} if supplied else None)

    target = result[result["mz"] == 22]
    assert target["raw_area"].tolist() == [12.0, 24.0]
    assert target["photon_normalized_area"].tolist() == [12.0, 24.0]
    assert target["expansion_lambda"].tolist() == [1.0, 2.0]
    assert target["area"].tolist() == [12.0, 12.0]
    pd.testing.assert_frame_equal(rows, original)


def test_disabled_kr_correction_keeps_target_signal_when_kr_is_zero():
    rows = _signal_rows((10.0, 0.0))
    result = _apply_temperature_normalization(
        rows, kr_correct=False, kr_mz=84, mass_discrimination=1.0,
    )
    assert result.loc[result["mz"] == 22, "area"].tolist() == [12.0, 24.0]


def test_valid_kr_reference_preserves_measured_zero_target_signal():
    rows = _signal_rows()
    rows.loc[(rows["temperature"] == 500) & (rows["mz"] == 22),
             ["raw_area", "photon_normalized_area"]] = 0.0
    result = _correct(rows)
    assert result.loc[result["mz"] == 22, "area"].tolist() == [12.0, 0.0]


@pytest.mark.parametrize(
    "temperatures", [[], [None, np.nan], ["missing", "invalid"]],
    ids=["empty", "missing-temperatures", "non-numeric-temperatures"],
)
def test_calibration_without_temperature_rows_reports_precondition(temperatures):
    rows = pd.DataFrame({
        "temperature": temperatures,
        "photon_normalized_area": [10.0] * len(temperatures),
    })
    original = rows.copy(deep=True)

    with pytest.raises(KrCorrectionError, match="Kr m/z 86: no calibration rows with numeric temperatures"):
        _build_kr_expansion_factor_table(rows, kr_mz=86)

    pd.testing.assert_frame_equal(rows, original)


def test_highest_energy_calibration_validates_only_selected_reference_signals():
    low = _signal_rows((0.0, 0.0)).query("mz == 84")
    high = _signal_rows().query("mz == 84").copy()
    high["photon_energy"] = 14.0
    rows = pd.concat([low, high], ignore_index=True)
    original = rows.copy(deep=True)

    result = _build_kr_expansion_factor_table(rows, use_highest_energy=True)

    assert result["photon_energy"].tolist() == [14.0, 14.0]
    assert result["expansion_lambda"].tolist() == [1.0, 2.0]
    pd.testing.assert_frame_equal(rows, original)
    with pytest.raises(ValueError, match="12"):
        _build_kr_expansion_factor_table(rows)


@pytest.mark.parametrize("entry", [analyze_temperature_folder, compute_kr_expansion_factors])
def test_public_temperature_paths_reject_zero_kr_reference(tmp_path, entry):
    scan = tmp_path / "scan"
    scan.mkdir()
    peaks = tmp_path / "peaks.csv"
    peaks.write_text("mz,peak,start,end\n22,22,21,23\n84,84,83,85\n", encoding="utf-8")
    for temperature, kr, target in ((400, 10, 12), (500, 0, 24)):
        header = [f"Temperature:{temperature} C", "Energy:12 eV", "IO:1 nA"] + ["Metadata"] * 7
        y = [0.0] * 100
        y[22], y[84] = target, kr
        (scan / f"{temperature}.txt").write_text(
            "\n".join(header + [str(value) for value in y]), encoding="utf-8",
        )

    kwargs = {"kr_correct": True} if entry is analyze_temperature_folder else {}
    with pytest.raises(ValueError) as error:
        entry(scan, manual_peak_path=peaks, integration_method="sum_counts", **kwargs)
    assert "500" in str(error.value) and "84" in str(error.value)
