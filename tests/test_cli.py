from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "bl03u_cli.py"


def _write_spectrum(path: Path, *, energy: float, temperature: float, io: float, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = [0.0] * 50
    y[21] = 1.0 * scale
    y[22] = 10.0 * scale
    y[23] = 1.0 * scale
    header = [
        f"Energy:{energy} eV",
        f"IO:{io} nA",
        "Beam Current:1mA",
        "Undulator Offset:0mm",
        "Time:1 s",
        "Burner Position:0 mm",
        f"Temperature:{temperature} C",
        "DIFF PRESSURE:1Pa",
        "ION PRESSURE:1Pa",
        "TOF PRESSURE:1Pa",
    ]
    path.write_text("\n".join(header + [str(value) for value in y]), encoding="utf-8")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def test_cli_formula_outputs_json():
    result = _run_cli("formula", "Ca(OH)2")
    payload = json.loads(result.stdout)

    assert payload["composition"] == {"Ca": 1, "O": 2, "H": 2}
    assert payload["nominal_mass"] == 74


def test_cli_pie_writes_csv_and_curve_json(tmp_path):
    _write_spectrum(tmp_path / "11.0eV" / "spectrum.txt", energy=11.0, temperature=300, io=10, scale=1)
    _write_spectrum(tmp_path / "12.0eV" / "spectrum.txt", energy=12.0, temperature=300, io=10, scale=2)
    peaks = tmp_path / "peaks.csv"
    peaks.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")
    output = tmp_path / "pie.csv"
    curves_json = tmp_path / "pie_curves.json"

    _run_cli(
        "pie",
        str(tmp_path),
        "--manual-peak-path",
        str(peaks),
        "--no-gaussian",
        "--photon-mode",
        "off",
        "--output",
        str(output),
        "--curves-json",
        str(curves_json),
    )

    df = pd.read_csv(output)
    curves = json.loads(curves_json.read_text(encoding="utf-8"))
    assert df["energy"].tolist() == [11.0, 12.0]
    assert "22" in curves
    assert curves["22"]["intensities"] == [9.0, 18.0]


def test_cli_temperature_writes_csv_and_curve_json(tmp_path):
    _write_spectrum(tmp_path / "400.txt", energy=12.0, temperature=400, io=10, scale=1)
    _write_spectrum(tmp_path / "500.txt", energy=12.0, temperature=500, io=10, scale=2)
    peaks = tmp_path / "peaks.csv"
    peaks.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")
    output = tmp_path / "temperature.csv"
    curves_json = tmp_path / "temperature_curves.json"

    _run_cli(
        "temperature",
        str(tmp_path),
        "--manual-peak-path",
        str(peaks),
        "--no-gaussian",
        "--no-photon-normalize",
        "--output",
        str(output),
        "--curves-json",
        str(curves_json),
    )

    df = pd.read_csv(output)
    curves = json.loads(curves_json.read_text(encoding="utf-8"))
    assert df["temperature"].tolist() == [400.0, 500.0]
    assert "22" in curves
    assert curves["22"]["areas"] == [9.0, 18.0]
