from __future__ import annotations

import numpy as np
import pytest

from bl03u_masstool.core.temperature_scan import (
    analyze_temperature_folder,
    extract_photon_energy,
    extract_temperature,
)


def _write_scan(tmp_path, metadata):
    scan = tmp_path / "scan"
    scan.mkdir()
    y = [0.0] * 50
    y[21:24] = [1.0, 10.0, 1.0]
    spectrum = scan / "sample.txt"
    spectrum.write_text(
        "\n".join(metadata + ["Metadata"] * (10 - len(metadata)) + [str(v) for v in y]),
        encoding="utf-8",
    )
    peaks = tmp_path / "peaks.csv"
    peaks.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")
    return scan, peaks


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Temperature", None), ("Temperature", "NaN C"),
        ("Temperature", "inf C"), ("Temperature", "1e309 C"),
        ("Temperature", "unavailable 400 C"), ("Temperature", "1.2.3 C"),
        ("Energy", None), ("Energy", "0 eV"), ("Energy", "-12 eV"),
        ("Energy", "NaN eV"), ("Energy", "inf eV"),
        ("Energy", "1e309 eV"), ("Energy", "unavailable 12 eV"),
    ],
)
def test_temperature_analysis_rejects_invalid_axis_metadata(tmp_path, field, value):
    metadata = {"Temperature": "400 C", "Energy": "12 eV", "IO": "10 nA"}
    metadata[field] = value
    header = [f"{key}:{value}" for key, value in metadata.items() if value is not None]
    # Unrelated numeric headers and the filename must not supply an axis value.
    header += ["Temperature Offset:25 C", "Energy Offset:3 eV", "Time:10 s"]
    scan, peaks = _write_scan(tmp_path, header)

    with pytest.raises(ValueError) as error:
        analyze_temperature_folder(scan, manual_peak_path=peaks, photon_normalize=False)

    assert "sample.txt" in str(error.value)
    assert field.lower() in str(error.value).lower()


@pytest.mark.parametrize(("source", "field", "other"), [
    ("io", "IO", "Beam Current"), ("beam_current", "Beam Current", "IO"),
])
@pytest.mark.parametrize("value", [None, "0", "-1", "NaN", "inf", "1e309", "unavailable 10"])
def test_temperature_normalization_requires_selected_intensity(tmp_path, source, field, other, value):
    header = ["Temperature:400 C", "Energy:12 eV", f"{other}:20 nA"]
    if value is not None:
        header.append(f"{field}:{value} nA")
    scan, peaks = _write_scan(tmp_path, header)

    with pytest.raises(ValueError) as error:
        analyze_temperature_folder(
            scan, manual_peak_path=peaks, photon_normalize=True, light_source=source,
        )

    assert "sample.txt" in str(error.value)
    assert field.lower() in str(error.value).lower()


@pytest.mark.parametrize("source", ["io", "beam_current"])
@pytest.mark.parametrize("intensity", [None, "0", "NaN"])
def test_temperature_analysis_without_normalization_keeps_missing_intensity_missing(tmp_path, source, intensity):
    header = ["Temperature:400 C", "Energy:12 eV"]
    if intensity is not None:
        field = "IO" if source == "io" else "Beam Current"
        header.append(f"{field}:{intensity} nA")
    scan, peaks = _write_scan(tmp_path, header)

    result = analyze_temperature_folder(
        scan, manual_peak_path=peaks, integration_method="sum_counts",
        photon_normalize=False, light_source=source,
    )

    assert result["temperature"].tolist() == [400.0]
    assert result["photon_energy"].tolist() == [12.0]
    assert result["io"].isna().all()
    assert result["area"].tolist() == [12.0]


@pytest.mark.parametrize(("source", "intensity_header", "expected"), [
    ("io", "IO:10nA", 1.2),
    ("beam_current", "Beam Current=2mA", 6.0),
    ("io", "io = 1e1 nA", 1.2),
])
def test_temperature_analysis_preserves_explicit_metadata_and_normalized_values(
    tmp_path, source, intensity_header, expected,
):
    scan, peaks = _write_scan(tmp_path, [
        "Temperature = -2.5e1 C", "Energy:1.2e1 eV", intensity_header,
    ])

    result = analyze_temperature_folder(
        scan, manual_peak_path=peaks, integration_method="sum_counts",
        photon_normalize=True, light_source=source,
    )

    assert result["temperature"].tolist() == [-25.0]
    assert result["photon_energy"].tolist() == [12.0]
    assert result["raw_area"].tolist() == [12.0]
    assert result["area"].tolist() == pytest.approx([expected])
    assert np.isfinite(result["io"]).all()


@pytest.mark.parametrize(("header", "temperature"), [
    ("Temperature:400 C", 400.0), ("Temperature:0", 0.0),
    ("temperature -25°C", -25.0), ("Temperature=4e2 ℃", 400.0),
])
def test_temperature_analysis_accepts_supported_axis_formats(tmp_path, header, temperature):
    scan, peaks = _write_scan(tmp_path, [
        "Temperature Offset:25 C", "Energy Offset:3 eV", header, "Photon Energy=12eV",
    ])

    result = analyze_temperature_folder(
        scan, manual_peak_path=peaks, integration_method="sum_counts",
    )

    assert result["temperature"].tolist() == [temperature]
    assert result["photon_energy"].tolist() == [12.0]
    assert result["area"].tolist() == [12.0]


@pytest.mark.parametrize(("extractor", "field", "fallback"), [
    (extract_temperature, "Temperature", 400.0), (extract_photon_energy, "Energy", 12.0),
])
def test_axis_helper_fallback_requires_explicit_valid_value(extractor, field, fallback):
    unrelated = ["Time:10 s", f"{field} Offset:25"]
    assert extractor(unrelated, fallback=fallback) == fallback
    with pytest.raises(ValueError, match=field):
        extractor(unrelated)
    with pytest.raises(ValueError, match=field):
        extractor([f"{field}:NaN"], fallback=fallback)
    with pytest.raises(ValueError, match=field):
        extractor(unrelated, fallback=float("nan"))
