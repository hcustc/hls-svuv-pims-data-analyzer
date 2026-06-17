from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLI_MODULE = "bl03u_masstool.scripts.bl03u_cli"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src_root = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_root + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src_root
    return env


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
        [sys.executable, "-m", CLI_MODULE, *args],
        cwd=PROJECT_ROOT,
        env=_cli_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _run_cli_allow_failure(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", CLI_MODULE, *args],
        cwd=PROJECT_ROOT,
        env=_cli_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
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
    manifest_json = tmp_path / "pie_manifest.json"
    evidence_json = tmp_path / "pie_evidence.json"
    report_md = tmp_path / "pie_report.md"

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
        "--manifest-json",
        str(manifest_json),
        "--evidence-json",
        str(evidence_json),
        "--report-md",
        str(report_md),
    )

    df = pd.read_csv(output)
    curves = json.loads(curves_json.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_json.read_text(encoding="utf-8"))
    report = report_md.read_text(encoding="utf-8")
    assert df["energy"].tolist() == [11.0, 12.0]
    assert "22" in curves
    assert curves["22"]["intensities"] == [9.0, 18.0]
    assert manifest["analysis_type"] == "pie"
    assert manifest["data_summary"]["curve_count"] == 1
    assert evidence["22"]["confidence_level"] == "unfitted"
    assert "m/z 级证据摘要" in report


def test_cli_pie_fit_pics_requires_database(tmp_path):
    result = _run_cli_allow_failure("pie", str(tmp_path), "--fit-pics")
    assert result.returncode == 2
    assert "error: --fit-pics requires --database" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_temperature_writes_csv_and_curve_json(tmp_path):
    _write_spectrum(tmp_path / "400.txt", energy=12.0, temperature=400, io=10, scale=1)
    _write_spectrum(tmp_path / "500.txt", energy=12.0, temperature=500, io=10, scale=2)
    peaks = tmp_path / "peaks.csv"
    peaks.write_text("mz,peak,start,end\n22,22,21,23\n", encoding="utf-8")
    output = tmp_path / "temperature.csv"
    curves_json = tmp_path / "temperature_curves.json"
    manifest_json = tmp_path / "temperature_manifest.json"
    evidence_json = tmp_path / "temperature_evidence.json"
    report_md = tmp_path / "temperature_report.md"

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
        "--manifest-json",
        str(manifest_json),
        "--evidence-json",
        str(evidence_json),
        "--report-md",
        str(report_md),
    )

    df = pd.read_csv(output)
    curves = json.loads(curves_json.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_json.read_text(encoding="utf-8"))
    report = report_md.read_text(encoding="utf-8")
    assert df["temperature"].tolist() == [400.0, 500.0]
    assert "22" in curves
    assert curves["22"]["areas"] == [9.0, 18.0]
    assert manifest["analysis_type"] == "temperature"
    assert evidence["22"]["curve"]["point_count"] == 2
    assert "有效温度点少于 3 个" in report
